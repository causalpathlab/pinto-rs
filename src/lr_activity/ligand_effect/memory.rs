//! How many pairs run at once, from the memory each needs and what the machine
//! has, and runs them under that bound.

use super::overlap::NuisanceOpts;
use super::tissue::Tissue;

/// Memory allowed for pairs estimated at once when the machine does not say
/// what it has, and the least allowed when it does.
const MIN_PAIR_MEMORY_BUDGET: usize = 2 << 30;

/// Memory allowed for pairs estimated at once: half of what is available now
/// ([`available_memory`]), the rest left to the tissue and the machine's
/// other work.
fn pair_memory_budget() -> usize {
    available_memory().map_or(MIN_PAIR_MEMORY_BUDGET, |a| {
        (a / 2).max(MIN_PAIR_MEMORY_BUDGET)
    })
}

/// Bytes this process could still allocate: the system's available memory, or
/// less when a cgroup (a batch job, a container) limits it. `None` where the
/// system does not say.
fn available_memory() -> Option<usize> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available = system.available_memory();
    let left = system
        .cgroup_limits()
        .map_or(available, |group| group.free_memory.min(available));
    (left > 0).then_some(left as usize)
}

/// Bytes of `rows` rows of a propensity fit ([`super::regression::logistic_base`]) on
/// `p` confounders: those and the pair's own columns.
fn propensity_bytes(rows: usize, p: usize) -> usize {
    rows * (p + 8) * 8
}

/// Memory one live pair (or benchmark scenario) holds: `vectors` receiver-length
/// vectors and the propensity's rows ([`super::regression::logistic_base`])
/// for every fold, since its folds are fitted side by side.
pub(super) fn pair_memory(vectors: usize, tissue: &Tissue, nuisance: &NuisanceOpts) -> usize {
    let (n, p) = (tissue.n(), tissue.design.p());
    let rows = n.min(nuisance.max_propensity_rows);
    n * 8 * vectors + tissue.cf.k * propensity_bytes(rows, p)
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
    let threads = rayon::current_num_threads();
    let rows = tissue.n().min(nuisance.max_propensity_rows);
    let fit_bytes = propensity_bytes(rows, tissue.design.p());
    let by_cache = if fit_bytes < FITS_IN_CACHE {
        threads
    } else {
        (threads / 4).max(1)
    };
    (pair_memory_budget() / per_pair.max(1))
        .min(by_cache)
        .max(1)
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
