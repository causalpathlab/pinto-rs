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
use crate::util::ablation::{dense_ids, group};
use crate::util::common::{jackknife_se, quantile_sorted, upper_median, Mat, LIBRARY_SCALE};
use crate::util::feature_axis::FeatureAxis;
use crate::util::graph_coarsen::fold_edges_to_super;
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

/// `ln(1 + s x / Σ x)` in place, `s` the [`LIBRARY_SCALE`]; an empty unit
/// stays empty.
pub fn log_normalise(p: &mut [(u32, f32)]) {
    let lib: f32 = p.iter().map(|&(_, v)| v).sum();
    if lib > 0.0 {
        p.iter_mut()
            .for_each(|(_, v)| *v = (LIBRARY_SCALE * *v / lib).ln_1p());
    }
}

/// Each column of `pooled` (features × units, raw counts) as a sparse,
/// log-normalised profile (see [`log_normalise`]).
pub fn normalised_profiles(pooled: &Mat) -> Vec<Vec<(u32, f32)>> {
    pooled
        .column_iter()
        .map(|col| {
            let mut p: Vec<(u32, f32)> = col
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v > 0.0)
                .map(|(f, &v)| (f as u32, v))
                .collect();
            log_normalise(&mut p);
            p
        })
        .collect()
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
    let g = n_features;
    let block: Vec<usize> = dense_ids(block.iter())
        .into_iter()
        .map(|b| b as usize)
        .collect();
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
/// in order. Never empty: if no feature is kept, one is drawn with
/// probability proportional to `pi` (uniformly when every `pi` is 0), so an
/// epoch always trains.
pub fn sample_included(features: &mut Vec<usize>, pi: &[f32], rng: &mut impl rand::Rng) {
    use rand::RngExt;
    use rand_distr::weighted::WeightedIndex;
    use rand_distr::Distribution;
    if features.is_empty() {
        return;
    }
    let all = features.clone();
    features.retain(|&g| rng.random::<f32>() < pi[g]);
    if features.is_empty() {
        let pick = match WeightedIndex::new(all.iter().map(|&g| pi[g])) {
            Ok(by_pi) => by_pi.sample(rng),
            Err(_) => rng.random_range(0..all.len()),
        };
        features.push(all[pick]);
    }
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
    let (units_of, edges_of) = split_sections(section, edges);
    let sections = units_of
        .iter()
        .zip(&edges_of)
        .map(|(members, es)| {
            let (profiles, block) = inputs(members)?;
            Ok(section_jack(n_features, &profiles, &block, es))
        })
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
    /// Pooled raw counts on the matrix ROW axis, `rows × n_pb`.
    pub counts: &'a Mat,
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
/// `per_section` per section).
pub fn pb_inclusion(
    axis: &FeatureAxis,
    pb: &PbUnits<'_>,
    section: &[u32],
    per_section: usize,
) -> anyhow::Result<FeatureInclusion> {
    let n_pb = pb.counts.ncols();
    let pooled = axis.pool_rows_opt(pb.counts);
    let profiles = normalised_profiles(pooled.as_ref().unwrap_or(pb.counts));
    drop(pooled);
    let n_sections = section.iter().map(|&s| s as usize + 1).max().unwrap_or(0);
    let (pb_section, _) = majority_label(pb.cell_to_pb, n_pb, section, n_sections);
    let block = tree_blocks(pb.parents, &pb_section, per_section);
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

/// Cells read per block by [`cell_inclusion`] in a fit.
pub const CELL_COLUMN_BLOCK: usize = 4096;

/// [`FeatureInclusion`] over cells (cage-cell, which builds no pseudobulks):
/// each cell's log-normalised counts, the spatial graph `edges` (global cell
/// pairs; pairs across sections are left out), and a grid of geometric
/// tiles over each section's plane (the first two non-batch columns of
/// `coordinates`, named) as jackknife blocks, about `per_section` per
/// section. One section at a time, its contiguous runs of cells read
/// `column_block` at once in parallel and reduced to its sums before the
/// next, so memory holds one section's profiles.
pub fn cell_inclusion(
    data: &SparseIoVec,
    axis: &FeatureAxis,
    (coordinates, names): (&Mat, &[Box<str>]),
    section: &[u32],
    edges: &[(usize, usize)],
    per_section: usize,
    column_block: usize,
) -> anyhow::Result<FeatureInclusion> {
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
    score_sections(axis.n_features(), section, edges, "cells", inputs)
}
