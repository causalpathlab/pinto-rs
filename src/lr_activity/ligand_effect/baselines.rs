//! Co-expression baselines for `pinto lra-simulate`: the ligand–receptor
//! statistics in common use, none of which adjusts for a confounder, each
//! asked the same question as the estimator ("is the pair interacting?")
//! and tested at the same 5% level, one-sided (more co-expression than
//! chance). They see what the estimator sees: the detected (possibly
//! shifted) ligand and the receptor, as depth-normalized `log1p(x / sf)`
//! (detection for the co-expression log odds).
//!
//! - **Cell-type label permutation** (CellPhoneDB v2, Efremova et al. 2020;
//!   CellChat, Jin et al. 2021). Cell types are the identity bins coarsened
//!   to their three leading bits (bit `k` of a bin is the sign of the `k`-th
//!   singular direction of the identity projection, so `bin & 7` splits the
//!   receivers along the three strongest axes: at most 8 types). The score
//!   of sender type A and receiver type B is CellChat's law-of-mass-action
//!   product, mean ligand in A × mean receptor in B (CellPhoneDB's mean of
//!   the two means would call a pair whenever either partner alone is
//!   type-specific). As in CellPhoneDB, a type pair is tested only when the
//!   ligand is detected in at least [`MIN_DETECTED`] of A and the receptor
//!   in as many of B. Null: cell-type labels permuted across cells
//!   ([`PERMUTATIONS`] times). The pair's p is the smallest type-pair p,
//!   Bonferroni-corrected over the tested type pairs.
//! - **Bivariate Moran's I** (SpatialDM, Li et al. 2023): the global
//!   bivariate Moran's I of ligand in senders and receptor in receivers over
//!   the row-standardized contact graph, `I = (n / S0) Σ_ij w_ij L̃_i R̃_j /
//!   √(Σ L̃² Σ R̃²)`. Null: ligand permuted across cells within each section.
//!   SpatialDM weights pairs by a Gaussian kernel of distance; here the
//!   contact graph is the kernel.
//! - **Contact co-expression score** (NICHES, Raredon et al. 2022; COMMOT,
//!   Cang et al. 2023, without its optimal transport): `Σ_contacts L_i R_j`
//!   over both orientations of every contact. Null: ligand permuted across
//!   cells within each section, sharing the permutations of Moran's I.
//! - The co-expression log odds ratio `pinto lra` reports
//!   ([`super::estimate::coexpression`], Jeffreys), with its normal p.
//!
//! Permutation p-values are `(1 + #{null ≥ observed}) / (1 + permutations)`.
//! Permutations run in seeded chunks, a round of chunks in parallel, so a
//! run is reproducible whatever the thread count.
//!
//! **Cost.** The permutations stop early once the verdict at 5% is settled
//! (curtailed sampling, the sequential idea of Besag & Clifford 1991,
//! Biometrika 78:301): after each round, when every statistic already has
//! so many exceedances that even none in the remaining permutations would
//! leave its p (Bonferroni-scaled for the cell-type test) at or above 5%,
//! the rest are skipped and p is read from the permutations run. Rejection
//! is therefore exactly what all [`PERMUTATIONS`] would give, on the same
//! permutation stream, so the calibration is unchanged; only a
//! non-rejecting p is estimated from fewer permutations. Subsampling the
//! rows instead would change the statistic and save nothing on a tissue of
//! a few thousand receivers.

use super::estimate::coexpression;
use super::inference::upper_normal_p;
use super::tissue::Tissue;
use crate::util::common::*;
use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Permutations of every permutation baseline. Bonferroni over up to 64
/// type pairs needs `64 / (1 + P) < 0.05`, so `P ≥ 1280`.
pub const PERMUTATIONS: usize = 2000;

/// Permutations per chunk, each chunk with its own seed.
const CHUNK: usize = 100;

/// Chunks per round of the early stop ([`permuted`]), run in parallel.
const ROUND: usize = 5;

/// Leading identity bits kept as cell types (at most `2^3 = 8` types).
pub const TYPE_BITS: usize = 3;

/// Least detection share of the ligand in the sender type and of the
/// receptor in the receiver type for a type pair to be tested
/// (CellPhoneDB's default threshold).
pub const MIN_DETECTED: f64 = 0.1;

/// The level every method is tested at.
pub const ALPHA: f64 = 0.05;

/// One run's baselines: each statistic and its one-sided p (NaN where the
/// statistic is undefined, which never rejects).
#[derive(Clone, Copy, Debug)]
pub struct Baselines {
    /// The largest tested type-pair score.
    pub celltype_score: f64,
    /// Bonferroni-corrected smallest type-pair p.
    pub celltype_p: f64,
    pub celltype_tested: usize,
    pub moran_i: f64,
    pub moran_p: f64,
    pub contact_score: f64,
    pub contact_p: f64,
    pub coexpr_log_or: f64,
    pub coexpr_se: f64,
    pub coexpr_p: f64,
}

/// Reads one baseline's p from a run.
pub type PValue = fn(&Baselines) -> f64;

/// The baselines in the summary's order: column name and p.
pub const METHODS: [(&str, PValue); 4] = [
    ("celltype", |b| b.celltype_p),
    ("moran", |b| b.moran_p),
    ("contact", |b| b.contact_p),
    ("coexpr", |b| b.coexpr_p),
];

pub fn rejects(p: f64) -> bool {
    p < ALPHA
}

/// Cell type of each row: the identity bin's [`TYPE_BITS`] leading bits.
pub fn cell_types(bin: &[usize]) -> Vec<u8> {
    bin.iter()
        .map(|&b| (b & ((1 << TYPE_BITS) - 1)) as u8)
        .collect()
}

/// `null ≥ observed`, forgiving the rounding of a sum taken in another
/// order: a tie (common with sparse counts) counts against the observed.
fn at_least(null: f64, observed: f64) -> bool {
    null >= observed - 1e-9 * observed.abs() - 1e-12
}

fn perm_p(exceed: usize, perms: usize) -> f64 {
    (1 + exceed) as f64 / (1 + perms) as f64
}

/// Run the permutation chunks, a round of [`ROUND`] at a time in parallel:
/// `f(rng, k)` handles `k` permutations and returns its exceedance counts,
/// which are summed. After each round, `settled(counts)` says whether the
/// verdict can no longer change; then the rest are skipped. Returns the
/// counts and the permutations run.
fn permuted<F>(
    seed: u64,
    len: usize,
    settled: impl Fn(&[usize]) -> bool,
    f: F,
) -> (Vec<usize>, usize)
where
    F: Fn(&mut SmallRng, usize) -> Vec<usize> + Sync,
{
    let chunks = PERMUTATIONS.div_ceil(CHUNK);
    let (mut counts, mut done) = (vec![0usize; len], 0usize);
    for round in (0..chunks).collect::<Vec<_>>().chunks(ROUND) {
        let part = round
            .par_iter()
            .map(|&c| {
                let k = CHUNK.min(PERMUTATIONS - c * CHUNK);
                (
                    f(&mut SmallRng::seed_from_u64(mix_seed(seed, c as u64)), k),
                    k,
                )
            })
            .collect::<Vec<_>>();
        for (hits, k) in part {
            for (x, y) in counts.iter_mut().zip(hits) {
                *x += y;
            }
            done += k;
        }
        if done < PERMUTATIONS && settled(&counts) {
            break;
        }
    }
    (counts, done)
}

/// Whether `exceed` exceedances leave a p, scaled by `m` (Bonferroni), at
/// or above [`ALPHA`] whatever the permutations not yet run give.
fn cannot_reject(exceed: usize, m: usize) -> bool {
    perm_p(exceed, PERMUTATIONS) * m as f64 >= ALPHA
}

/// Every baseline on one pair's ligand and receptor counts.
pub fn baselines(
    tissue: &Tissue,
    ligand_counts: &[f64],
    receptor_counts: &[f64],
    seed: u64,
) -> Baselines {
    let n = tissue.n();
    let sf = &tissue.size_factor;
    let xl: Vec<f64> = (0..n).map(|i| (ligand_counts[i] / sf[i]).ln_1p()).collect();
    let xr: Vec<f64> = (0..n)
        .map(|i| (receptor_counts[i] / sf[i]).ln_1p())
        .collect();
    let (celltype_score, celltype_p, celltype_tested) =
        celltype(tissue, &xl, &xr, mix_seed(seed, 1));
    let (moran_i, moran_p, contact_score, contact_p) = spatial(tissue, &xl, &xr, mix_seed(seed, 2));
    let all = vec![true; n];
    let (coexpr_log_or, coexpr_se) = coexpression(tissue, ligand_counts, receptor_counts, &all);
    Baselines {
        celltype_score,
        celltype_p,
        celltype_tested,
        moran_i,
        moran_p,
        contact_score,
        contact_p,
        coexpr_log_or,
        coexpr_se,
        coexpr_p: upper_normal_p(coexpr_log_or / coexpr_se),
    }
}

// * Cell-type label permutation

/// Per-type sums of ligand and receptor under labels `ty`.
fn type_sums(ty: &[u8], xl: &[f64], xr: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut sl, mut sr) = (vec![0.0; k], vec![0.0; k]);
    for i in 0..ty.len() {
        sl[ty[i] as usize] += xl[i];
        sr[ty[i] as usize] += xr[i];
    }
    (sl, sr)
}

/// (largest tested score, Bonferroni p, type pairs tested).
fn celltype(tissue: &Tissue, xl: &[f64], xr: &[f64], seed: u64) -> (f64, f64, usize) {
    let ty = cell_types(&tissue.identity_bin);
    let k = 1 << TYPE_BITS;
    let mut cells = vec![0usize; k];
    let (mut dl, mut dr) = (vec![0usize; k], vec![0usize; k]);
    for i in 0..ty.len() {
        let t = ty[i] as usize;
        cells[t] += 1;
        dl[t] += (xl[i] > 0.0) as usize;
        dr[t] += (xr[i] > 0.0) as usize;
    }
    let share = |d: usize, t: usize| cells[t] > 0 && d as f64 >= MIN_DETECTED * cells[t] as f64;
    let tested: Vec<(usize, usize)> = (0..k)
        .flat_map(|a| (0..k).map(move |b| (a, b)))
        .filter(|&(a, b)| share(dl[a], a) && share(dr[b], b))
        .collect();
    if tested.is_empty() {
        return (f64::NAN, f64::NAN, 0);
    }
    let scores = |sl: &[f64], sr: &[f64]| -> Vec<f64> {
        tested
            .iter()
            .map(|&(a, b)| sl[a] / cells[a] as f64 * sr[b] / cells[b] as f64)
            .collect()
    };
    let (sl, sr) = type_sums(&ty, xl, xr, k);
    let obs = scores(&sl, &sr);
    let m = tested.len();
    let settled = |c: &[usize]| c.iter().all(|&e| cannot_reject(e, m));
    let (exceed, perms) = permuted(seed, tested.len(), settled, |rng, perms| {
        let mut lab = ty.clone();
        let mut hit = vec![0usize; tested.len()];
        for _ in 0..perms {
            lab.shuffle(rng);
            let (sl, sr) = type_sums(&lab, xl, xr, k);
            for (h, (s, o)) in hit.iter_mut().zip(scores(&sl, &sr).iter().zip(&obs)) {
                *h += at_least(*s, *o) as usize;
            }
        }
        hit
    });
    let p_min = exceed
        .iter()
        .map(|&e| perm_p(e, perms))
        .fold(f64::INFINITY, f64::min);
    let best = obs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (best, (p_min * m as f64).min(1.0), m)
}

// * Contact statistics

/// (Moran's I, its p, contact score, its p), sharing one set of ligand
/// permutations within sections.
fn spatial(tissue: &Tissue, xl: &[f64], xr: &[f64], seed: u64) -> (f64, f64, f64, f64) {
    let n = tissue.n();
    let nan = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n.max(1) as f64;
    let (ml, mr) = (mean(xl), mean(xr));
    let ssl: f64 = xl.iter().map(|x| (x - ml).powi(2)).sum();
    let ssr: f64 = xr.iter().map(|x| (x - mr).powi(2)).sum();
    if ssl <= 0.0 || ssr <= 0.0 {
        return nan;
    }
    // Row-standardized lag of the centred receptor, and the receptor
    // summed over contacts: each statistic is a dot product with ligand.
    let rc: Vec<f64> = xr.iter().map(|x| x - mr).collect();
    let lag_sum = tissue.contacts.sum(&rc);
    let s0 = (0..n).filter(|&i| tissue.contacts.degree(i) > 0).count() as f64;
    let lag: Vec<f64> = (0..n)
        .map(|i| match tissue.contacts.degree(i) {
            0 => 0.0,
            d => lag_sum[i] / d as f64,
        })
        .collect();
    let around = tissue.contacts.sum(xr);
    let moran_scale = n as f64 / s0.max(1.0) / (ssl * ssr).sqrt();
    // Centring ligand is a shift by its mean, unchanged by permutation.
    let lag_total: f64 = lag.iter().sum();
    let moran = |dot: f64| moran_scale * (dot - ml * lag_total);
    let dot = |v: &[f64]| xl.iter().zip(v).map(|(x, y)| x * y).sum::<f64>();
    let (i_obs, c_obs) = (moran(dot(&lag)), dot(&around));
    let sections = tissue.section.iter().max().map_or(1, |&s| s as usize + 1);
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); sections];
    for i in 0..n {
        rows[tissue.section[i] as usize].push(i);
    }
    let settled = |c: &[usize]| c.iter().all(|&e| cannot_reject(e, 1));
    let (exceed, perms) = permuted(seed, 2, settled, |rng, perms| {
        let mut vals: Vec<Vec<f64>> = rows
            .iter()
            .map(|r| r.iter().map(|&i| xl[i]).collect())
            .collect();
        let mut hit = vec![0usize; 2];
        for _ in 0..perms {
            let (mut a, mut b) = (0.0, 0.0);
            for (r, v) in rows.iter().zip(vals.iter_mut()) {
                v.shuffle(rng);
                for (&i, &x) in r.iter().zip(v.iter()) {
                    a += x * lag[i];
                    b += x * around[i];
                }
            }
            hit[0] += at_least(moran(a), i_obs) as usize;
            hit[1] += at_least(b, c_obs) as usize;
        }
        hit
    });
    (
        i_obs,
        perm_p(exceed[0], perms),
        c_obs,
        perm_p(exceed[1], perms),
    )
}
