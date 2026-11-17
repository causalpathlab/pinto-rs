//! Soft feature inclusion: how likely each feature is to carry spatial
//! signal, as a probability `π_g` the training loop samples features by.
//!
//! cage gives every trained feature its own adjacency task with the same
//! positive budget, so on data where most features are noise the noise
//! tasks swamp the signal (a hard HVG cut helps, but needs a threshold and
//! drops features by rank). Here each feature instead gets
//!
//! 1. **Moran's I** of its log-normalised expression over the spatial graph,
//!    centred within each section and pooled over sections. The units are
//!    cage's finest PB super-cells (their pooled counts, and the spatial
//!    edges between them), or the cells themselves under cage-cell, which
//!    builds no pseudobulks;
//! 2. a **delete-one-block jackknife** standard error. The blocks are coarse
//!    super-cells of the PB tree (merged to about `--inclusion-blocks` per
//!    section, siblings together), or a grid of geometric tiles over
//!    each section's plane on cells. Deleting a block removes its units,
//!    every edge touching them and their share of their section's mean (an
//!    exact deletion: the left-out units inform nothing). Neighbouring spots
//!    are correlated, so the textbook variance is too small; the jackknife's
//!    is not, and a feature whose structure sits in one patch gets a wide
//!    one. Blocks lie within sections, never whole sections, so a domain only
//!    one section holds is not penalised;
//! 3. a **two-group empirical-Bayes** model on `t = (I − E₀[I]) / se`: the null
//!    component is a normal fitted to the bulk (median, MAD), its share `π₀`
//!    is estimated from the bulk too, the mixture density by a binned Gaussian
//!    kernel estimate, and `π_g = 1 − lfdr_g` (Efron's local false discovery
//!    rate), one-sided (only positive autocorrelation is signal) and
//!    non-decreasing in `t`. No threshold or top-N is chosen. A feature
//!    with no jackknife spread (no variation, or no edges) is left out of
//!    the model and gets `π = 0`.
//!
//! Each epoch then visits feature `g` with probability `π_g` (a seeded
//! Bernoulli mask), so the expected objective is `Σ_g π_g L_g`. A feature
//! the model calls null (`π_g = 0`, at or below the bulk) is never drawn.
//!
//! References: P. A. P. Moran, "Notes on continuous stochastic phenomena",
//! Biometrika 37 (1950); H. R. Künsch, "The jackknife and the bootstrap for
//! general stationary observations", Ann. Statist. 17 (1989), for deleting
//! blocks rather than single units of dependent data; B. Efron, R.
//! Tibshirani, J. D. Storey and V. Tusher, "Empirical Bayes analysis of a
//! microarray experiment", JASA 96 (2001), for the two-group model and the
//! local fdr; B. Efron, "Large-scale simultaneous hypothesis testing: the
//! choice of a null hypothesis", JASA 99 (2004), for the null fitted to the
//! bulk; B. W. Silverman, "Density Estimation for Statistics and Data
//! Analysis" (1986), for the kernel bandwidth.

use crate::cell_activity_graph_embedding::pb_frame::majority_label;
use crate::link_community::profiles::coarsen_cell_expression_dense;
use crate::util::ablation::{dense_ids, group};
use crate::util::common::{jackknife_se, log_normalise, quantile_sorted, upper_median, Mat};
use crate::util::feature_axis::FeatureAxis;
use crate::util::graph_coarsen::fold_edges_to_super;
use crate::util::stage1_divisor::Stage1Divisor;
use data_beans::sparse_io_vector::SparseIoVec;
use log::info;
use rayon::prelude::*;

/// Each feature's Moran's I, jackknife standard error, t statistic and
/// inclusion probability.
#[derive(Debug, Clone)]
pub struct FeatureInclusion {
    pub moran: Vec<f64>,
    pub se: Vec<f64>,
    pub t: Vec<f64>,
    pub pi: Vec<f32>,
}

/// One section reduced to what the jackknife reads: the section's centred
/// Moran sums per feature, and per block what deleting it changes.
/// Per-feature arrays of blocks are row-major, `blocks × features`.
#[derive(Debug, Clone)]
pub struct SectionJack {
    /// Units and edges of the section.
    n: f64,
    w: f64,
    /// `Σ_edges (x_i − μ)(x_j − μ)` and `Σ_units (x_i − μ)²` per feature.
    num: Vec<f64>,
    den: Vec<f64>,
    /// Per block: its units, and the edges touching it, which deleting it
    /// removes.
    nb: Vec<f64>,
    mb: Vec<f64>,
    /// Per block: the change in `num` and `den` when it is deleted (the
    /// section re-centred without it).
    dnum: Vec<f64>,
    dden: Vec<f64>,
}

/// Centred Moran sums from raw ones: units `n`, `Σ x`, `Σ x²`, edges `w`,
/// `Σ x_i x_j`, `Σ (x_i + x_j)`.
fn centred(n: f64, sx: f64, sxx: f64, w: f64, exy: f64, ex: f64) -> (f64, f64) {
    let mu = if n > 0.0 { sx / n } else { 0.0 };
    (
        exy - mu * ex + w * mu * mu,
        sxx - 2.0 * mu * sx + n * mu * mu,
    )
}

/// Each column of `pooled` (features × units, raw counts) as a sparse,
/// log-normalised profile (see [`log_normalise`]).
pub fn normalised_profiles(pooled: &Mat) -> Vec<Vec<(u32, f32)>> {
    let mut profiles = raw_profiles(pooled);
    profiles.iter_mut().for_each(|p| log_normalise(p));
    profiles
}

/// Each cell's tile in a `tiles × tiles` grid over the bounding box of
/// `xy` (one section).
pub fn tile_of(xy: &[(f32, f32)], tiles: usize) -> Vec<usize> {
    let tiles = tiles.max(1);
    let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for &(x, y) in xy {
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    let cut = |v: f32, lo: f32, hi: f32| {
        let w = (hi - lo).max(f32::EPSILON);
        (((v - lo) / w * tiles as f32) as usize).min(tiles - 1)
    };
    xy.iter()
        .map(|&(x, y)| cut(x, x0, x1) * tiles + cut(y, y0, y1))
        .collect()
}

/// One section's [`SectionJack`] from its units' log-normalised profiles
/// (`(feature, value)` lists sorted by feature), each unit's block (any
/// labels; made dense here, so a label with no unit is no block), and the
/// section's edges as local unit pairs. Memory is `blocks × features` per
/// array, with at most one accumulator of edge products per worker thread.
pub fn section_jack<P: AsRef<[(u32, f32)]> + Sync>(
    n_features: usize,
    profiles: &[P],
    block: &[usize],
    edges: &[(usize, usize)],
) -> SectionJack {
    let block = dense_blocks(block);
    jack_from_sums(raw_sums(n_features, profiles, &block, edges))
}

/// Block labels made dense, so a label with no unit is no block.
fn dense_blocks(block: &[usize]) -> Vec<usize> {
    dense_ids(block.iter())
        .into_iter()
        .map(|b| b as usize)
        .collect()
}

/// A section's raw Moran sums, per block (`nb`, `sx`, `sxx`) and per row of
/// edges (`m`, `ex`, `exy`: the edges touching each block, then all of
/// them), each per-feature array `rows × features` row-major.
struct RawSums {
    g: usize,
    n_blocks: usize,
    nb: Vec<f64>,
    m: Vec<f64>,
    sx: Vec<f64>,
    sxx: Vec<f64>,
    ex: Vec<f64>,
    exy: Vec<f64>,
}

/// [`RawSums`] of `profiles` over dense `block`s and the section's `edges`.
fn raw_sums<P: AsRef<[(u32, f32)]> + Sync>(
    n_features: usize,
    profiles: &[P],
    block: &[usize],
    edges: &[(usize, usize)],
) -> RawSums {
    let g = n_features;
    let n_blocks = block.iter().max().map_or(0, |&b| b + 1);
    // Per-row sums run over the edges touching each block, and in the last
    // row over every edge. An edge touches its ends' blocks, so a unit's
    // edges all touch its own block and the last row (its degree counts
    // them), and an edge between blocks also the other end's: count those
    // per unit and block.
    let rows = n_blocks + 1;
    let mut m = vec![0f64; rows];
    let mut degree = vec![0f64; profiles.len()];
    let mut across: Vec<(usize, usize)> = Vec::new();
    for &(i, j) in edges {
        let (a, b) = (block[i], block[j]);
        degree[i] += 1.0;
        degree[j] += 1.0;
        m[n_blocks] += 1.0;
        m[a] += 1.0;
        if a != b {
            m[b] += 1.0;
            across.push((i, b));
            across.push((j, a));
        }
    }
    across.sort_unstable();
    // raw sums per block, and per row the linear edge term `Σ (x_i + x_j)`,
    // each unit's value times its edges touching the row
    let mut nb = vec![0f64; n_blocks];
    let (mut sx, mut sxx) = (vec![0f64; n_blocks * g], vec![0f64; n_blocks * g]);
    let mut ex = vec![0f64; rows * g];
    let mut others = across.chunk_by(|x, y| x.0 == y.0).peekable();
    let mut touch: Vec<(usize, f64)> = Vec::new();
    for (i, p) in profiles.iter().enumerate() {
        let a = block[i];
        nb[a] += 1.0;
        touch.clear();
        touch.extend([(n_blocks, degree[i]), (a, degree[i])]);
        if let Some(run) = others.next_if(|run| run[0].0 == i) {
            touch.extend(
                run.chunk_by(|x, y| x.1 == y.1)
                    .map(|r| (r[0].1, r.len() as f64)),
            );
        }
        for &(f, v) in p.as_ref() {
            let (f, v) = (f as usize, f64::from(v));
            sx[a * g + f] += v;
            sxx[a * g + f] += v * v;
            for &(r, c) in &touch {
                ex[r * g + f] += c * v;
            }
        }
    }
    // `Σ x_i x_j` per row over the edges touching it
    let zero = || vec![0f64; rows * g];
    let per_thread = edges.len().div_ceil(rayon::current_num_threads()).max(1);
    let exy = edges
        .par_iter()
        .with_min_len(per_thread)
        .fold(zero, |mut exy, &(i, j)| {
            let (a, b) = (block[i], block[j]);
            let touch = [n_blocks, a, b];
            let touch = &touch[..if a == b { 2 } else { 3 }];
            let (pa, pb) = (profiles[i].as_ref(), profiles[j].as_ref());
            let (mut p, mut q) = (0, 0);
            while p < pa.len() && q < pb.len() {
                match pa[p].0.cmp(&pb[q].0) {
                    std::cmp::Ordering::Less => p += 1,
                    std::cmp::Ordering::Greater => q += 1,
                    std::cmp::Ordering::Equal => {
                        let xy = f64::from(pa[p].1) * f64::from(pb[q].1);
                        for &r in touch {
                            exy[r * g + pa[p].0 as usize] += xy;
                        }
                        p += 1;
                        q += 1;
                    }
                }
            }
            exy
        })
        .reduce(zero, |mut x, y| {
            x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            x
        });
    RawSums {
        g,
        n_blocks,
        nb,
        m,
        sx,
        sxx,
        ex,
        exy,
    }
}

/// The section's centred sums and each block's deletion from its raw sums.
fn jack_from_sums(raw: RawSums) -> SectionJack {
    let RawSums {
        g,
        n_blocks,
        nb,
        mut m,
        sx,
        sxx,
        ex,
        exy,
    } = raw;
    // the section's totals, then each block's deletion, feature innermost
    let mut tsx = vec![0f64; g];
    let mut tsxx = vec![0f64; g];
    for k in 0..n_blocks {
        for f in 0..g {
            tsx[f] += sx[k * g + f];
            tsxx[f] += sxx[k * g + f];
        }
    }
    let (n, w) = (nb.iter().sum::<f64>(), m[n_blocks]);
    let (texy, tex) = (&exy[n_blocks * g..], &ex[n_blocks * g..]);
    let (num, den): (Vec<f64>, Vec<f64>) = (0..g)
        .map(|f| centred(n, tsx[f], tsxx[f], w, texy[f], tex[f]))
        .unzip();
    let (mut dnum, mut dden) = (vec![0f64; n_blocks * g], vec![0f64; n_blocks * g]);
    for k in 0..n_blocks {
        for f in 0..g {
            let x = k * g + f;
            let (nu, de) = centred(
                n - nb[k],
                tsx[f] - sx[x],
                tsxx[f] - sxx[x],
                w - m[k],
                texy[f] - exy[x],
                tex[f] - ex[x],
            );
            dnum[x] = nu - num[f];
            dden[x] = de - den[f];
        }
    }
    m.truncate(n_blocks);
    SectionJack {
        n,
        w,
        num,
        den,
        nb,
        mb: m,
        dnum,
        dden,
    }
}

/// Moran's I pooled over sections (each centred on its own mean), its
/// delete-one-block jackknife standard error, and `t = (I + 1/(N − 1)) / se`
/// (0 where a feature does not vary). Deleting a block changes only its own
/// section.
pub fn jackknife_moran(sections: &[SectionJack]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let g = sections.first().map_or(0, |s| s.num.len());
    let n: f64 = sections.iter().map(|s| s.n).sum();
    let w: f64 = sections.iter().map(|s| s.w).sum();
    let moran =
        |n: f64, w: f64, num: f64, den: f64| (den > 0.0 && w > 0.0).then(|| (n / w) * num / den);
    let per: Vec<(f64, f64, f64)> = (0..g)
        .into_par_iter()
        .map(|f| {
            let num: f64 = sections.iter().map(|s| s.num[f]).sum();
            let den: f64 = sections.iter().map(|s| s.den[f]).sum();
            let Some(i_all) = moran(n, w, num, den) else {
                return (0.0, 0.0, 0.0);
            };
            let loo: Vec<f64> = sections
                .iter()
                .flat_map(|s| {
                    (0..s.nb.len()).filter_map(move |k| {
                        let x = k * g + f;
                        moran(n - s.nb[k], w - s.mb[k], num + s.dnum[x], den + s.dden[x])
                    })
                })
                .collect();
            // NaN (fewer than two replicates) counts as no spread
            let se = jackknife_se(&loo).max(0.0);
            let e0 = -1.0 / (n - 1.0).max(1.0);
            let t = if se > 0.0 { (i_all - e0) / se } else { 0.0 };
            (i_all, se, t)
        })
        .collect();
    (
        per.iter().map(|p| p.0).collect(),
        per.iter().map(|p| p.1).collect(),
        per.iter().map(|p| p.2).collect(),
    )
}

/// This epoch's features: each of `features` kept with probability `pi[g]`,
/// in a shuffled order. Never empty: if no feature is kept, one is drawn with
/// probability proportional to `pi` (uniformly when every `pi` is 0), so an
/// epoch always trains.
///
/// Every draw comes from feature `g`'s own stream for `(seed, epoch)`, and the
/// order is by a key from that stream, so whether a feature is kept, and the
/// relative order of the kept ones, do not depend on which other features are
/// listed. The fallback pick, made only when nothing is kept, does.
pub fn sample_included(features: &[usize], pi: &[f32], seed: u64, epoch: usize) -> Vec<usize> {
    use legume_numeric::matrix::rand_util::{mix_seed, stream_seed};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    let epoch_seed = stream_seed(seed, "feature-inclusion", epoch, 0);
    // Feature g's stream: its keep draw, then its order key, then (read
    // only when nothing is kept) its fallback key.
    let stream = |g: usize| SmallRng::seed_from_u64(mix_seed(epoch_seed, g as u64));
    let mut kept: Vec<(f64, usize)> = features
        .iter()
        .filter_map(|&g| {
            let mut rng = stream(g);
            let keep = rng.random::<f32>() < pi[g];
            let order = rng.random::<f64>();
            keep.then_some((order, g))
        })
        .collect();
    if kept.is_empty() {
        // The key ln(u)/w: the largest is a draw in proportion to the weights
        // (P. S. Efraimidis and P. G. Spirakis, "Weighted random sampling with
        // a reservoir", Inf. Process. Lett. 97 (2006)); with no weight
        // anywhere, ln(u) alone is a uniform draw.
        let fallback_key = |g: usize| {
            let mut rng = stream(g);
            let _keep: f32 = rng.random();
            let _order: f64 = rng.random();
            let ln_u = rng.random::<f64>().ln();
            let w = f64::from(pi[g]);
            if w > 0.0 {
                (true, ln_u / w)
            } else {
                (false, ln_u)
            }
        };
        return features
            .iter()
            .map(|&g| (fallback_key(g), g))
            .max_by(|(x, _), (y, _)| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)))
            .map(|(_, g)| g)
            .into_iter()
            .collect();
    }
    kept.sort_by(|x, y| x.0.total_cmp(&y.0));
    kept.into_iter().map(|(_, g)| g).collect()
}

/// Features an epoch visits in expectation: every one of `features`, or
/// `Σ π` over them under inclusion, rounded and at least one (an epoch is
/// never empty, see [`sample_included`]).
pub fn expected_visits(features: &[usize], pi: Option<&[f32]>) -> usize {
    match pi {
        Some(pi) => (features.iter().map(|&g| pi[g]).sum::<f32>().round() as usize).max(1),
        None => features.len(),
    }
}

fn std_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// `π_g = 1 − lfdr_g` from the two-group model on `t`, with the null share
/// `π₀`. See the module docs.
pub fn two_group_inclusion(t: &[f64]) -> (Vec<f32>, f64) {
    let n = t.len();
    if n < 3 {
        return (vec![1.0; n], 0.0);
    }
    let mut s = t.to_vec();
    s.sort_by(f64::total_cmp);
    let at = |q: f64| quantile_sorted(&s, q).unwrap_or(0.0);
    let med = at(0.5);
    let mut dev: Vec<f64> = t.iter().map(|v| (v - med).abs()).collect();
    let mad = 1.4826 * upper_median(&mut dev).unwrap_or(0.0);
    if mad <= 0.0 {
        return (vec![1.0; n], 0.0);
    }
    // the bulk within one null sd holds 68.27% of the null
    let bulk = t.iter().filter(|&&v| (v - med).abs() < mad).count() as f64 / n as f64;
    let pi0 = (bulk / 0.682_689_492).min(1.0);
    // mixture density: Gaussian kernel (Silverman's bandwidth) on a grid
    let mean = t.iter().sum::<f64>() / n as f64;
    let sd = (t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let (q1, q3) = (at(0.25), at(0.75));
    let iqr = (q3 - q1) / 1.34;
    let spread = if iqr > 0.0 { sd.min(iqr) } else { sd }.max(f64::EPSILON);
    let h = 0.9 * spread * (n as f64).powf(-0.2);
    // The grid spans the data, but no further than 40 null sds from the
    // centre, where the null density has vanished (a t beyond it is
    // signal whatever the mixture density, and is binned at the edge); its
    // step is at most h/4 unless that takes more than GRID_MAX points, and
    // the kernel is normalised on the grid, so the estimate stays a density
    // however coarse the step.
    const GRID_MAX: usize = 1 << 16;
    let lo = s[0].max(med - 40.0 * mad) - 4.0 * h;
    let hi = s[n - 1].min(med + 40.0 * mad) + 4.0 * h;
    let grid = (((hi - lo) / (h / 4.0)).ceil() as usize + 1).clamp(2, GRID_MAX);
    let step = (hi - lo) / (grid - 1) as f64;
    let bin = |v: f64| (((v - lo) / step).round().max(0.0) as usize).min(grid - 1);
    let mut hist = vec![0f64; grid];
    for &v in t {
        hist[bin(v)] += 1.0;
    }
    let reach = ((4.0 * h) / step).ceil() as usize;
    let shape: Vec<f64> = (0..=reach)
        .map(|d| std_normal_pdf(d as f64 * step / h))
        .collect();
    let mass = step * (2.0 * shape.iter().sum::<f64>() - shape[0]);
    let kernel: Vec<f64> = shape.iter().map(|k| k / (n as f64 * mass)).collect();
    let dens: Vec<f64> = (0..grid)
        .into_par_iter()
        .map(|x| {
            let a = x.saturating_sub(reach);
            let b = (x + reach).min(grid - 1);
            (a..=b).map(|y| hist[y] * kernel[x.abs_diff(y)]).sum()
        })
        .collect();
    let f_at = |v: f64| dens[bin(v)];
    let raw: Vec<f64> = t
        .iter()
        .map(|&v| {
            if v <= med {
                return 0.0; // one-sided: only positive autocorrelation is signal
            }
            let f0 = std_normal_pdf((v - med) / mad) / mad;
            let lfdr = (pi0 * f0 / f_at(v).max(f64::MIN_POSITIVE)).clamp(0.0, 1.0);
            1.0 - lfdr
        })
        .collect();
    // non-decreasing in t: a feature never outranks a more autocorrelated one
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
    let mut pi = vec![0f32; n];
    let mut run = 0f64;
    for &g in &order {
        run = run.max(raw[g]);
        pi[g] = run as f32;
    }
    (pi, pi0)
}

/// A section's units under a stage-1 divisor (`--offset`), for Moran's I
/// on Pearson residuals `r_ug = (y_ug − μ_ug) / σ_{g,b(u)}`: each unit's raw
/// counts `y_u` (sorted by feature), its stage-1 mixture `w_u = Σ_{i∈u} n_i θ_i`
/// (`[units × K]` row-major) and its stage-1 batch `b(u)`, so
/// `μ_ug = w_uᵀ p_{b(u)}(g)` (exact for a unit within one stage-1 batch, the
/// unit's majority batch otherwise).
pub struct PearsonUnits<'a> {
    /// Each unit's raw counts, scaled by `1/σ` in place when scored.
    pub counts: Vec<Vec<(u32, f32)>>,
    pub w: &'a [f32],
    pub batch: &'a [usize],
    pub stage1: &'a Stage1Divisor,
}

/// `1 / σ_{gb}`, or 0 where the scale is 0 (no expectation in the batch).
#[inline]
fn inv_sigma(s1: &Stage1Divisor, g: usize, b: usize) -> f64 {
    let s = f64::from(s1.sigma(g, b));
    if s > 0.0 {
        1.0 / s
    } else {
        0.0
    }
}

#[inline]
fn dot64(a: &[f64], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, &y)| x * f64::from(y)).sum()
}

/// `pᵀ M q` for `K × K` row-major `m`.
#[inline]
fn quad(m: &[f64], p: &[f32], q: &[f32]) -> f64 {
    let k = p.len();
    let mut total = 0.0;
    for (a, &pa) in p.iter().enumerate() {
        if pa != 0.0 {
            total += f64::from(pa) * dot64(&m[a * k..(a + 1) * k], q);
        }
    }
    total
}

/// What a section's units add up to beyond their nonzero counts: the
/// nonzero cross terms per feature, and the `K`- and `K × K` accumulators the
/// zeros' closed forms read, dense over block or row and the section's own
/// stage-1 batches (`nb`, local ids; usually one).
struct PearsonAcc {
    k: usize,
    nb: usize,
    /// `Σ_{u∈block} a_ug c_ug`, `blocks × g`.
    sac: Vec<f64>,
    /// `Σ_{edges touching row} (a_i c_j + c_i a_j)`, `rows × g`.
    cross: Vec<f64>,
    /// `Σ_{u∈block, b(u)=b} w_u`, `[blocks × nb × K]`, and `Σ w_u w_uᵀ`,
    /// `[blocks × nb × K × K]`.
    sw: Vec<f64>,
    sww: Vec<f64>,
    /// `Σ_u t_{u,row} w_u` (`t` its edges touching the row), `[rows × nb × K]`.
    st: Vec<f64>,
    /// `Σ_{edges touching row} w_i w_jᵀ`, halved from each end,
    /// `[rows × nb × nb × K × K]`.
    cc: Vec<f64>,
}

impl PearsonAcc {
    fn new(k: usize, nb: usize, n_blocks: usize, g: usize) -> Self {
        let rows = n_blocks + 1;
        Self {
            k,
            nb,
            sac: vec![0.0; n_blocks * g],
            cross: vec![0.0; rows * g],
            sw: vec![0.0; n_blocks * nb * k],
            sww: vec![0.0; n_blocks * nb * k * k],
            st: vec![0.0; rows * nb * k],
            cc: vec![0.0; rows * nb * nb * k * k],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (x, y) in [
            (&mut self.sac, &other.sac),
            (&mut self.cross, &other.cross),
            (&mut self.sw, &other.sw),
            (&mut self.sww, &other.sww),
            (&mut self.st, &other.st),
            (&mut self.cc, &other.cc),
        ] {
            x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
        }
        self
    }

    fn sw_at(&self, blk: usize, b: usize) -> usize {
        (blk * self.nb + b) * self.k
    }

    fn sww_at(&self, blk: usize, b: usize) -> usize {
        (blk * self.nb + b) * self.k * self.k
    }

    fn cc_at(&self, row: usize, bi: usize, bj: usize) -> usize {
        ((row * self.nb + bi) * self.nb + bj) * self.k * self.k
    }
}

/// `out += scale · x yᵀ` over `K × K` row-major `out`.
#[inline]
fn add_outer(out: &mut [f64], x: &[f64], y: &[f64], scale: f64) {
    let k = x.len();
    for (r, &xr) in x.iter().enumerate() {
        if xr != 0.0 {
            let xr = xr * scale;
            out[r * k..(r + 1) * k]
                .iter_mut()
                .zip(y)
                .for_each(|(o, &yc)| *o += xr * yc);
        }
    }
}

/// One unit's neighbour sums, reused across units: per key (which rows of
/// edges it feeds, the neighbour's local batch), the summed mixtures.
#[derive(Default)]
struct NeighbourScratch {
    /// `(None, b)`: every edge (rows `all` and the unit's own block);
    /// `(Some(B), b)`: the edges to block `B` only.
    keys: Vec<(Option<usize>, usize)>,
    /// `keys.len() × K`.
    sums: Vec<f64>,
    /// `(row, edges of the unit touching it)`.
    touch: Vec<(usize, f64)>,
    w: Vec<f64>,
}

impl NeighbourScratch {
    fn clear(&mut self) {
        self.keys.clear();
        self.sums.clear();
        self.touch.clear();
    }

    fn add(&mut self, key: (Option<usize>, usize), w: &[f32]) {
        let k = w.len();
        let at = match self.keys.iter().position(|&x| x == key) {
            Some(at) => at,
            None => {
                self.keys.push(key);
                self.sums.resize(self.keys.len() * k, 0.0);
                self.keys.len() - 1
            }
        };
        self.sums[at * k..(at + 1) * k]
            .iter_mut()
            .zip(w)
            .for_each(|(s, &x)| *s += f64::from(x));
    }

    fn touch(&mut self, row: usize) {
        match self.touch.iter_mut().find(|t| t.0 == row) {
            Some(t) => t.1 += 1.0,
            None => self.touch.push((row, 1.0)),
        }
    }
}

/// One section's [`SectionJack`] of Pearson residuals under the stage-1
/// divisor (see [`PearsonUnits`]): the same sums as [`section_jack`], each
/// split into its nonzero terms (streamed over the counts `a = y / σ`) and
/// the expectation's, which factorise through the mixtures:
/// `Σ_u c_ug = p_b(g)ᵀ Σ_u w_u / σ`, `Σ_u c_ug² = p_b(g)ᵀ (Σ_u w_u w_uᵀ) p_b(g) / σ²`,
/// and `Σ_e c_ig c_jg = p_{b_i}(g)ᵀ (Σ_e w_i w_jᵀ) p_{b_j}(g) / (σ_i σ_j)`,
/// per block and per row of edges as [`section_jack`] keeps them. Nothing
/// is dense over units × features: a unit costs its nonzeros times its few
/// neighbour sums, `K` each, and `K²` per neighbour sum.
pub fn section_jack_pearson(
    n_features: usize,
    units: PearsonUnits<'_>,
    block: &[usize],
    edges: &[(usize, usize)],
) -> SectionJack {
    let g = n_features;
    let s1 = units.stage1;
    let k = s1.k();
    let n_units = units.counts.len();
    let block = dense_blocks(block);
    let n_blocks = block.iter().max().map_or(0, |&b| b + 1);
    let rows = n_blocks + 1;
    let w = |u: usize| &units.w[u * k..(u + 1) * k];
    // the section's own batches, dense
    let mut batches: Vec<usize> = units.batch.to_vec();
    batches.sort_unstable();
    batches.dedup();
    let nb = batches.len().max(1);
    let local: Vec<usize> = units
        .batch
        .iter()
        .map(|b| batches.binary_search(b).expect("a unit's batch"))
        .collect();

    // a = y / σ, the nonzero part of r, in place of the counts
    let mut a = units.counts;
    a.par_iter_mut().zip(units.batch).for_each(|(row, &b)| {
        for (f, y) in row.iter_mut() {
            *y = (f64::from(*y) * inv_sigma(s1, *f as usize, b)) as f32;
        }
    });
    let mut raw = raw_sums(g, &a, &block, edges);

    // each unit's neighbours
    let mut start = vec![0usize; n_units + 1];
    for &(i, j) in edges {
        start[i + 1] += 1;
        start[j + 1] += 1;
    }
    for u in 0..n_units {
        start[u + 1] += start[u];
    }
    let mut fill = start.clone();
    let mut nbr = vec![0usize; start[n_units]];
    for &(i, j) in edges {
        nbr[fill[i]] = j;
        fill[i] += 1;
        nbr[fill[j]] = i;
        fill[j] += 1;
    }

    let per_thread = n_units.div_ceil(rayon::current_num_threads()).max(1);
    let acc = (0..n_units)
        .into_par_iter()
        .with_min_len(per_thread)
        .fold(
            || {
                (
                    PearsonAcc::new(k, nb, n_blocks, g),
                    NeighbourScratch::default(),
                )
            },
            |(mut acc, mut sc), u| {
                let (lb, bu, blk) = (local[u], units.batch[u], block[u]);
                sc.clear();
                sc.w.clear();
                sc.w.extend(w(u).iter().map(|&x| f64::from(x)));
                let at = acc.sw_at(blk, lb);
                acc.sw[at..at + k]
                    .iter_mut()
                    .zip(&sc.w)
                    .for_each(|(s, x)| *s += x);
                let at = acc.sww_at(blk, lb);
                add_outer(&mut acc.sww[at..at + k * k], &sc.w, &sc.w, 1.0);
                // neighbour mixtures: those of every edge (rows `all` and the
                // unit's own block) and per other block, by neighbour batch
                for &v in &nbr[start[u]..start[u + 1]] {
                    let blv = block[v];
                    sc.add((None, local[v]), w(v));
                    sc.touch(n_blocks);
                    sc.touch(blk);
                    if blv != blk {
                        sc.add((Some(blv), local[v]), w(v));
                        sc.touch(blv);
                    }
                }
                for &(row, t) in &sc.touch {
                    let at = (row * nb + lb) * k;
                    acc.st[at..at + k]
                        .iter_mut()
                        .zip(&sc.w)
                        .for_each(|(s, x)| *s += t * x);
                }
                // the rows of edges a neighbour sum feeds: every edge's
                // (`all` and the unit's own block), or one other block's
                let rows_of = |kind: Option<usize>| -> ([usize; 2], usize) {
                    match kind {
                        None => ([n_blocks, blk], 2),
                        Some(other) => ([other, other], 1),
                    }
                };
                for (key, &(kind, lv)) in sc.keys.iter().enumerate() {
                    let vsum = &sc.sums[key * k..(key + 1) * k];
                    let (rows_at, n_at) = rows_of(kind);
                    for &row in &rows_at[..n_at] {
                        let at = acc.cc_at(row, lb, lv);
                        add_outer(&mut acc.cc[at..at + k * k], &sc.w, vsum, 0.5);
                    }
                }
                // the nonzero cross terms
                for &(f, av) in &a[u] {
                    let (f, av) = (f as usize, f64::from(av));
                    let c = inv_sigma(s1, f, bu) * dot64(&sc.w, s1.profile(bu, f));
                    acc.sac[blk * g + f] += av * c;
                    for (key, &(kind, lv)) in sc.keys.iter().enumerate() {
                        let bv = batches[lv];
                        let vsum = &sc.sums[key * k..(key + 1) * k];
                        let x = av * inv_sigma(s1, f, bv) * dot64(vsum, s1.profile(bv, f));
                        let (rows_at, n_at) = rows_of(kind);
                        for &row in &rows_at[..n_at] {
                            acc.cross[row * g + f] += x;
                        }
                    }
                }
                (acc, sc)
            },
        )
        .map(|(acc, _)| acc)
        .reduce(|| PearsonAcc::new(k, nb, n_blocks, g), PearsonAcc::merge);

    // the expectation's closed forms, feature by feature
    for f in 0..g {
        for blk in 0..n_blocks {
            for (lb, &b) in batches.iter().enumerate() {
                let (inv, p) = (inv_sigma(s1, f, b), s1.profile(b, f));
                if inv == 0.0 {
                    continue;
                }
                let at = acc.sw_at(blk, lb);
                raw.sx[blk * g + f] -= inv * dot64(&acc.sw[at..at + k], p);
                let at = acc.sww_at(blk, lb);
                raw.sxx[blk * g + f] += inv * inv * quad(&acc.sww[at..at + k * k], p, p);
            }
            raw.sxx[blk * g + f] -= 2.0 * acc.sac[blk * g + f];
        }
        for row in 0..rows {
            for (lb, &b) in batches.iter().enumerate() {
                let inv = inv_sigma(s1, f, b);
                if inv == 0.0 {
                    continue;
                }
                let at = (row * nb + lb) * k;
                raw.ex[row * g + f] -= inv * dot64(&acc.st[at..at + k], s1.profile(b, f));
                for (lv, &bv) in batches.iter().enumerate() {
                    let scale = inv * inv_sigma(s1, f, bv);
                    if scale != 0.0 {
                        let at = acc.cc_at(row, lb, lv);
                        raw.exy[row * g + f] += scale
                            * quad(&acc.cc[at..at + k * k], s1.profile(b, f), s1.profile(bv, f));
                    }
                }
            }
            raw.exy[row * g + f] -= acc.cross[row * g + f];
        }
    }
    jack_from_sums(raw)
}

/// One section's edges as pairs of its own unit indices.
type LocalEdges = Vec<(usize, usize)>;

/// A section's unit profiles, and each unit's jackknife block.
type SectionInputs<P> = (Vec<P>, Vec<usize>);

/// The PB frame's super edges, and each cell's batch.
pub type FrameEdges<'a> = (&'a [(u32, u32)], &'a [u32]);

/// The units of each section, and the edges within each as local unit
/// pairs (edges across sections are left out).
fn split_sections(section: &[u32], edges: &[(usize, usize)]) -> (Vec<Vec<usize>>, Vec<LocalEdges>) {
    let units_of = group(section);
    let mut local = vec![0usize; section.len()];
    for units in &units_of {
        for (l, &u) in units.iter().enumerate() {
            local[u] = l;
        }
    }
    let mut edges_of = vec![Vec::new(); units_of.len()];
    for &(i, j) in edges {
        if section[i] == section[j] {
            edges_of[section[i] as usize].push((local[i], local[j]));
        }
    }
    (units_of, edges_of)
}

/// [`FeatureInclusion`] of units in sections, the spatial `edges` between
/// them (global unit pairs; pairs across sections are left out): Moran's I
/// per section, pooled, jackknifed over blocks, then the two-group model.
/// `inputs` gives a section's units' profiles and blocks from its units
/// (ascending global ids); sections run one at a time, so memory holds one
/// section's profiles (the edge pass inside is parallel). `units` names
/// them in the log.
pub(super) fn score_sections<P: AsRef<[(u32, f32)]> + Sync>(
    n_features: usize,
    section: &[u32],
    edges: &[(usize, usize)],
    units: &str,
    mut inputs: impl FnMut(&[usize]) -> anyhow::Result<SectionInputs<P>>,
) -> anyhow::Result<FeatureInclusion> {
    score_sections_with(n_features, section, edges, units, |members, es| {
        let (profiles, block) = inputs(members)?;
        Ok(section_jack(n_features, &profiles, &block, es))
    })
}

/// [`score_sections`] with each section's [`SectionJack`] made by `jack`
/// from its units (ascending global ids) and its edges as local unit pairs.
fn score_sections_with(
    n_features: usize,
    section: &[u32],
    edges: &[(usize, usize)],
    units: &str,
    mut jack: impl FnMut(&[usize], &[(usize, usize)]) -> anyhow::Result<SectionJack>,
) -> anyhow::Result<FeatureInclusion> {
    let (units_of, edges_of) = split_sections(section, edges);
    let sections = units_of
        .iter()
        .zip(&edges_of)
        .map(|(members, es)| jack(members, es))
        .collect::<anyhow::Result<Vec<SectionJack>>>()?;
    Ok(finish(n_features, &sections, units))
}

/// Jackknife blocks from a tree of nested labels per unit (`levels`, coarse
/// to fine, the units themselves taken as the finest): about `per_section`
/// blocks in every section, never spanning two. In each section the
/// coarsest level holding at least `per_section` labels there is taken,
/// and its labels, ordered by their parent so siblings stay together, are
/// merged into `per_section` runs of about equal units.
pub fn tree_blocks(levels: &[Vec<usize>], section: &[u32], per_section: usize) -> Vec<usize> {
    let per_section = per_section.max(1);
    let units: Vec<usize> = (0..section.len()).collect();
    let tree: Vec<&[usize]> = levels
        .iter()
        .map(Vec::as_slice)
        .chain([units.as_slice()])
        .collect();
    let mut block = vec![0usize; section.len()];
    for (s, members) in group(section).iter().enumerate() {
        let n_labels = |labels: &[usize]| {
            let mut l: Vec<usize> = members.iter().map(|&u| labels[u]).collect();
            l.sort_unstable();
            l.dedup();
            l.len()
        };
        let at = (0..tree.len())
            .find(|&l| n_labels(tree[l]) >= per_section)
            .unwrap_or(tree.len() - 1);
        let parent = |u: usize| at.checked_sub(1).map_or(0, |p| tree[p][u]);
        // each label's units, in order of (parent, label)
        let mut size: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
        for &u in members {
            *size.entry((parent(u), tree[at][u])).or_default() += 1;
        }
        let mut run_of = std::collections::HashMap::new();
        let mut before = 0;
        for ((_, label), c) in size {
            run_of.insert(
                label,
                s * per_section + before * per_section / members.len(),
            );
            before += c;
        }
        for &u in members {
            block[u] = run_of[&tree[at][u]];
        }
    }
    block
}

/// Jackknife, two-group model and a log line, over `sections`.
fn finish(n_features: usize, sections: &[SectionJack], units: &str) -> FeatureInclusion {
    let n_blocks: usize = sections.iter().map(|s| s.nb.len()).sum();
    let n_edges: f64 = sections.iter().map(|s| s.w).sum();
    let (moran, se, t) = jackknife_moran(sections);
    // only features with spread enter the two-group model; one with none
    // (no variation, no edges) has t = 0 by default, not by evidence, and a
    // majority of them would make the MAD zero
    let spread: Vec<usize> = (0..n_features).filter(|&f| se[f] > 0.0).collect();
    let t_spread: Vec<f64> = spread.iter().map(|&f| t[f]).collect();
    let (pi_spread, pi0) = two_group_inclusion(&t_spread);
    let mut pi = vec![0f32; n_features];
    for (&f, &p) in spread.iter().zip(&pi_spread) {
        pi[f] = p;
    }
    let total: f32 = pi.iter().sum();
    let kept = pi.iter().filter(|&&p| p > 0.0).count();
    info!(
        "feature inclusion: Moran's I over {n_edges} edges between {units}, jackknife over \
         {n_blocks} blocks; null share {pi0:.2}; {kept} of {n_features} features have π > 0, \
         {total:.0} expected per epoch"
    );
    if total < 1.0 {
        log::warn!(
            "feature inclusion: no feature shows spatial signal (expected {total:.2} per epoch); \
             each epoch trains at least one"
        );
    }
    FeatureInclusion { moran, se, t, pi }
}

/// The spatial graph's cell edges folded onto the finest PBs: distinct
/// pairs of different PBs, from cell edges within one section. `frame` is
/// the PB frame's super edges and each cell's batch, given when the spatial
/// graph is the frame's own graph: the frame folded the same edges, keeping
/// those within one batch, so where that keeps the same edges its super
/// edges are reused rather than folded again.
pub fn pb_spatial_edges(
    cell_to_pb: &[usize],
    section: &[u32],
    spatial_edges: &[(usize, usize)],
    frame: Option<FrameEdges<'_>>,
) -> Vec<(usize, usize)> {
    let same_section: Vec<bool> = spatial_edges
        .iter()
        .map(|&(i, j)| section[i] == section[j])
        .collect();
    if let Some((super_edges, batch)) = frame {
        let same = spatial_edges
            .iter()
            .zip(&same_section)
            .all(|(&(i, j), &s)| (batch[i] == batch[j]) == s);
        if same {
            return super_edges
                .iter()
                .map(|&(a, b)| (a as usize, b as usize))
                .collect();
        }
    }
    fold_edges_to_super(cell_to_pb, spatial_edges, Some(&same_section)).0
}

/// The finest PBs' inputs to [`pb_inclusion`].
pub struct PbUnits<'a> {
    /// The cells' counts, which are summed per PB here: raw, undivided.
    pub data: &'a SparseIoVec,
    /// The number of finest PBs.
    pub n_pb: usize,
    /// Each cell's finest PB.
    pub cell_to_pb: &'a [usize],
    /// Each PB's parent at every level coarser than the finest, coarse first.
    pub parents: &'a [Vec<usize>],
    /// The spatial edges between PBs (see [`pb_spatial_edges`]).
    pub edges: &'a [(usize, usize)],
}

/// [`FeatureInclusion`] over the finest PB super-cells: their pooled
/// counts, the spatial edges between them, each PB's majority section, and
/// coarse super-cells of the PB tree as jackknife blocks (about
/// `per_section` per section). Under a stage-1 divisor (`stage1`), on the
/// PBs' Pearson residuals instead ([`section_jack_pearson`]).
pub fn pb_inclusion(
    axis: &FeatureAxis,
    pb: &PbUnits<'_>,
    section: &[u32],
    per_section: usize,
    stage1: Option<&Stage1Divisor>,
) -> anyhow::Result<FeatureInclusion> {
    let n_pb = pb.n_pb;
    let n_sections = section.iter().map(|&s| s as usize + 1).max().unwrap_or(0);
    let (pb_section, _) = majority_label(pb.cell_to_pb, n_pb, section, n_sections);
    let block = tree_blocks(pb.parents, &pb_section, per_section);
    if let Some(s1) = stage1 {
        pearson_ready(axis)?;
        let counts = coarsen_cell_expression_dense(pb.data, pb.cell_to_pb, n_pb, None)?;
        let raw = raw_profiles(&counts);
        drop(counts);
        let k = s1.k();
        let mut w = vec![0f32; n_pb * k];
        for (cell, &u) in pb.cell_to_pb.iter().enumerate() {
            s1.add_mixture(cell, &mut w[u * k..(u + 1) * k]);
        }
        let cell_batch: Vec<u32> = (0..s1.n_cells()).map(|i| s1.batch_of(i) as u32).collect();
        let (pb_batch, _) = majority_label(pb.cell_to_pb, n_pb, &cell_batch, s1.n_batches());
        let pb_batch: Vec<usize> = pb_batch.iter().map(|&b| b as usize).collect();
        return score_sections_with(
            axis.n_features(),
            &pb_section,
            pb.edges,
            "units",
            |units, es| {
                let counts: Vec<Vec<(u32, f32)>> = units.iter().map(|&u| raw[u].clone()).collect();
                let wl: Vec<f32> = units
                    .iter()
                    .flat_map(|&u| w[u * k..(u + 1) * k].iter().copied())
                    .collect();
                let batch: Vec<usize> = units.iter().map(|&u| pb_batch[u]).collect();
                let bl: Vec<usize> = units.iter().map(|&u| block[u]).collect();
                let pu = PearsonUnits {
                    counts,
                    w: &wl,
                    batch: &batch,
                    stage1: s1,
                };
                Ok(section_jack_pearson(axis.n_features(), pu, &bl, es))
            },
        );
    }
    let profiles = {
        let counts = coarsen_cell_expression_dense(pb.data, pb.cell_to_pb, n_pb, None)?;
        let pooled = axis.pool_rows_opt(&counts);
        normalised_profiles(pooled.as_ref().unwrap_or(&counts))
    };
    let inputs = |units: &[usize]| {
        Ok((
            units
                .iter()
                .map(|&u| profiles[u].as_slice())
                .collect::<Vec<_>>(),
            units.iter().map(|&u| block[u]).collect(),
        ))
    };
    score_sections(axis.n_features(), &pb_section, pb.edges, "units", inputs)
}

/// Each column of `counts` (rows × units) as a sparse profile of its raw
/// counts, sorted by row.
fn raw_profiles(counts: &Mat) -> Vec<Vec<(u32, f32)>> {
    counts
        .column_iter()
        .map(|col| {
            col.iter()
                .enumerate()
                .filter(|&(_, &v)| v > 0.0)
                .map(|(f, &v)| (f as u32, v))
                .collect()
        })
        .collect()
}

/// Pearson residuals are taken on the matrix rows, one feature per row: a
/// feature pooled over splice tracks would need its tracks' expectations and
/// scales pooled too, which the stage-1 tables do not carry.
fn pearson_ready(axis: &FeatureAxis) -> anyhow::Result<()> {
    anyhow::ensure!(
        !axis.is_channelized(),
        "--feature-inclusion moran under --offset reads Pearson residuals per row, and \
         this matrix pools splice tracks into features: use --feature-inclusion all"
    );
    Ok(())
}

/// Cells read per block by [`cell_inclusion`] in a fit.
pub const CELL_COLUMN_BLOCK: usize = 4096;

/// [`FeatureInclusion`] over cells (cage-cell, which builds no pseudobulks):
/// each cell's log-normalised counts, the spatial graph `edges` (global cell
/// pairs; pairs across sections are left out), and a grid of geometric
/// tiles over each section's plane (the first two non-batch columns of
/// `coordinates`, named) as jackknife blocks, about `per_section` per
/// section. One section at a time, its contiguous runs of cells read
/// `column_block` at once in parallel and reduced to its sums before the
/// next, so memory holds one section's profiles. Under a stage-1 divisor
/// (`stage1`), on the cells' Pearson residuals ([`section_jack_pearson`]).
pub fn cell_inclusion(
    (data, stage1): (&SparseIoVec, Option<&Stage1Divisor>),
    axis: &FeatureAxis,
    (coordinates, names): (&Mat, &[Box<str>]),
    section: &[u32],
    edges: &[(usize, usize)],
    per_section: usize,
    column_block: usize,
) -> anyhow::Result<FeatureInclusion> {
    if stage1.is_some() {
        pearson_ready(axis)?;
    }
    let (cx, cy) = crate::util::ablation::plane_columns(names).map_err(|_| {
        anyhow::anyhow!(
            "--feature-inclusion moran on cells tiles each section's plane, which needs \
             two coordinate columns; found {names:?}"
        )
    })?;
    let tiles = (per_section as f64).sqrt().ceil().max(1.0) as usize;
    let column_block = column_block.max(1);
    let inputs = |cells: &[usize]| -> anyhow::Result<SectionInputs<Vec<(u32, f32)>>> {
        // runs of consecutive cells, cut to `column_block`
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        for run in cells.chunk_by(|a, b| b == &(a + 1)) {
            let (lo, hi) = (run[0], run[run.len() - 1] + 1);
            intervals.extend(
                (lo..hi)
                    .step_by(column_block)
                    .map(|a| (a, (a + column_block).min(hi))),
            );
        }
        let blocks = intervals
            .par_iter()
            .map(|&(lo, hi)| {
                let csc = data.read_columns_csc(lo..hi)?;
                // CSC row indices ascend, and pooling sorts by feature, so
                // every profile is sorted by feature
                Ok((0..csc.ncols())
                    .map(|j| {
                        let col = csc.col(j);
                        let obs: Vec<(u32, f32)> = col
                            .row_indices()
                            .iter()
                            .zip(col.values())
                            .map(|(&r, &v)| (r as u32, v))
                            .collect();
                        if stage1.is_some() {
                            return obs;
                        }
                        let mut p = axis.pool_profile(obs);
                        log_normalise(&mut p);
                        p
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let plane: Vec<(f32, f32)> = cells
            .iter()
            .map(|&i| (coordinates[(i, cx)], coordinates[(i, cy)]))
            .collect();
        Ok((
            blocks.into_iter().flatten().collect(),
            tile_of(&plane, tiles),
        ))
    };
    let Some(s1) = stage1 else {
        return score_sections(axis.n_features(), section, edges, "cells", inputs);
    };
    let k = s1.k();
    score_sections_with(axis.n_features(), section, edges, "cells", |cells, es| {
        let (counts, tiles) = inputs(cells)?;
        let mut w = vec![0f32; cells.len() * k];
        for (j, &i) in cells.iter().enumerate() {
            s1.add_mixture(i, &mut w[j * k..(j + 1) * k]);
        }
        let batch: Vec<usize> = cells.iter().map(|&i| s1.batch_of(i)).collect();
        let pu = PearsonUnits {
            counts,
            w: &w,
            batch: &batch,
            stage1: s1,
        };
        Ok(section_jack_pearson(axis.n_features(), pu, &tiles, es))
    })
}
