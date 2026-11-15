//! How many pairs run at once, from the memory each needs and what the machine
//! has, and runs them under that bound.

use super::overlap::NuisanceOpts;
use super::tissue::Tissue;

pub use crate::util::memory::check_memory_budget;
pub(super) use crate::util::memory::memory_budget;
use crate::util::memory::{available_memory, budget_var, half_available};

/// [`memory_budget`] now, of what is available ([`available_memory`]). A
/// variable that does not parse is refused at the start
/// ([`check_memory_budget`]); one that reaches here anyway is warned about
/// once and counts as unset.
pub(super) fn pair_memory_budget() -> usize {
    static SET: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let set = SET.get_or_init(|| {
        let var = budget_var()
            .map_err(|e| log::warn!("{e}; using half the available memory"))
            .ok()??;
        memory_budget(Some(&var), None).ok()
    });
    set.unwrap_or_else(|| half_available(available_memory()))
}

/// Bytes of `rows` rows of a propensity fit ([`super::regression::logistic_base`]) on
/// `p` confounders: those and the pair's own columns.
fn propensity_bytes(rows: usize, p: usize) -> usize {
    rows * (p + 8) * std::mem::size_of::<super::regression::LogisticRow>()
}

/// Propensities a threshold keeps for its bordered refits: the main and
/// each adjustment step's ([`super::fits::ThresholdFits`]).
const PROPENSITIES_KEPT: usize = 1 + super::tissue::STEP_NAMES.len();

/// Base propensities a ligand's share keeps ([`super::fits::LigandShare`]):
/// the main fit at each threshold of the grid (τ = 0, the curve quantiles and
/// the headline) and each adjustment step's at the headline.
const SHARED_BASES: usize =
    2 + super::options::CURVE_QUANTILES.len() + super::tissue::STEP_NAMES.len();

/// Memory held beside the data: `vectors` receiver-length vectors, a
/// propensity's rows ([`super::regression::logistic_base`]) for every fold,
/// since its folds are fitted side by side, and `kept` propensities, each
/// with its propensity vector and per fold its Hessian factor, weights and
/// rows.
fn held(vectors: usize, kept: usize, tissue: &Tissue, nuisance: &NuisanceOpts) -> usize {
    let (n, p, k) = (tissue.n(), tissue.design.p(), tissue.cf.k);
    let rows = n.min(nuisance.max_propensity_rows);
    let propensity = (n + k * (p * p + 2 * rows)) * 8;
    n * 8 * vectors + k * propensity_bytes(rows, p) + kept * propensity
}

/// Memory one live pair (or benchmark scenario) fitting its own
/// propensities holds ([`held`]).
pub(super) fn pair_memory(vectors: usize, tissue: &Tissue, nuisance: &NuisanceOpts) -> usize {
    held(vectors, PROPENSITIES_KEPT, tissue, nuisance)
}

/// Memory one live ligand holds: its live pair, whose propensities are its
/// ligand's shared bases, and those bases ([`held`]).
pub(super) fn ligand_memory(vectors: usize, tissue: &Tissue, nuisance: &NuisanceOpts) -> usize {
    held(vectors, SHARED_BASES, tissue, nuisance)
}

/// A propensity fit's rows, in bytes, that stay in a core's share of the
/// cache; past this, fits side by side compete for memory.
const FITS_IN_CACHE: usize = 4 << 20;

/// Pairs (or benchmark scenarios) to run at once, each holding `per_pair`
/// bytes: what the memory budget allows, and on a tissue whose fits outgrow
/// the cache a quarter of the threads, the rest serving each pair's parallel
/// steps ([`run_bounded`]). Fits slow each other down past that: more of them
/// at once take longer in all.
pub(super) fn pair_workers(per_pair: usize, tissue: &Tissue, nuisance: &NuisanceOpts) -> usize {
    let rows = tissue.n().min(nuisance.max_propensity_rows);
    let fit_bytes = propensity_bytes(rows, tissue.design.p());
    workers_within(
        pair_memory_budget(),
        per_pair,
        rayon::current_num_threads(),
        fit_bytes,
    )
}

/// [`pair_workers`] on a `budget` of bytes and `threads`, each pair holding
/// `per_pair` bytes and fitting propensities of `fit_bytes`: at least one.
pub(super) fn workers_within(
    budget: usize,
    per_pair: usize,
    threads: usize,
    fit_bytes: usize,
) -> usize {
    let by_cache = if fit_bytes < FITS_IN_CACHE {
        threads
    } else {
        (threads / 4).max(1)
    };
    (budget / per_pair.max(1)).min(by_cache).max(1)
}

/// `one` over every item, at most `workers` at a time, in item order. Each
/// worker takes the next item as soon as it is free, so one slow item holds
/// up only its own worker (no barrier between batches). The threads are
/// shared out: each worker's item runs its parallel steps on a rayon pool of
/// its own, so no item waits inside another item's work, which a shared pool
/// would have it steal and finish first.
pub(super) fn run_bounded<T: Sync, R: Send>(
    items: &[T],
    workers: usize,
    one: impl Fn(&T) -> Vec<R> + Sync,
) -> Vec<R> {
    let workers = workers.min(items.len());
    if workers <= 1 {
        return items.iter().flat_map(&one).collect();
    }
    let threads = (rayon::current_num_threads() / workers).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut done: Vec<(usize, Vec<R>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let work = || {
                        let mut out = Vec::new();
                        loop {
                            let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(item) = items.get(k) else { break };
                            out.push((k, one(item)));
                        }
                        out
                    };
                    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                        Ok(pool) => pool.install(work),
                        Err(_) => work(),
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("pair worker panicked"))
            .collect()
    });
    done.sort_unstable_by_key(|(k, _)| *k);
    done.into_iter().flat_map(|(_, r)| r).collect()
}
