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
//!    super-cells of the PB tree (the coarsest level giving every section
//!    about `--inclusion-blocks` of them), or a grid of geometric tiles over
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
//!    non-decreasing in `t`. No threshold or top-N is chosen.
//!
//! Each epoch then visits feature `g` with probability `π_g` (a seeded
//! Bernoulli mask), so the expected objective is `Σ_g π_g L_g`. A feature
//! the model calls null (`π_g = 0`, at or below the bulk) is never drawn.

use crate::util::ablation::{dense_ids, group};
use crate::util::common::{jackknife_se, Mat};
use crate::util::feature_axis::FeatureAxis;
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
    n_features: usize,
    /// Units and edges of the section.
    n: f64,
    w: f64,
    /// `Σ_edges (x_i − μ)(x_j − μ)` and `Σ_units (x_i − μ)²` per feature.
    num: Vec<f64>,
    den: Vec<f64>,
    /// Per block: the change in units and edges, and in `num` and `den`,
    /// when it is deleted (the section re-centred without it).
    dn: Vec<f64>,
    dw: Vec<f64>,
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

/// `ln(1 + 1e4 x / Σ x)` in place; an empty unit stays empty.
pub fn log_normalise(p: &mut [(u32, f32)]) {
    let lib: f32 = p.iter().map(|&(_, v)| v).sum();
    if lib > 0.0 {
        p.iter_mut()
            .for_each(|(_, v)| *v = (1e4 * *v / lib).ln_1p());
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
/// section's edges as local unit pairs.
pub fn section_jack<P: AsRef<[(u32, f32)]>>(
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
    // raw sums per block
    let mut nb = vec![0f64; n_blocks];
    let (mut sx, mut sxx) = (vec![0f64; n_blocks * g], vec![0f64; n_blocks * g]);
    for (i, p) in profiles.iter().enumerate() {
        let o = block[i] * g;
        nb[block[i]] += 1.0;
        for &(f, v) in p.as_ref() {
            let v = f64::from(v);
            sx[o + f as usize] += v;
            sxx[o + f as usize] += v * v;
        }
    }
    // raw sums per pair of blocks holding an edge, one pair id per edge
    let key = |&(i, j): &(usize, usize)| (block[i].min(block[j]), block[i].max(block[j]));
    let pair: Vec<u32> = dense_ids(edges.iter().map(key));
    let n_pairs = pair.iter().max().map_or(0, |&q| q as usize + 1);
    let mut pairs_of = vec![Vec::new(); n_blocks];
    let mut seen = vec![false; n_pairs];
    for (e, &q) in edges.iter().zip(&pair) {
        if !std::mem::replace(&mut seen[q as usize], true) {
            let (a, b) = key(e);
            pairs_of[a].push(q as usize);
            if b != a {
                pairs_of[b].push(q as usize);
            }
        }
    }
    let mut m = vec![0f64; n_pairs];
    let (mut exy, mut ex) = (vec![0f64; n_pairs * g], vec![0f64; n_pairs * g]);
    for (&(i, j), &q) in edges.iter().zip(&pair) {
        let o = q as usize * g;
        m[q as usize] += 1.0;
        let (a, b) = (profiles[i].as_ref(), profiles[j].as_ref());
        for &(f, v) in a.iter().chain(b) {
            ex[o + f as usize] += f64::from(v);
        }
        let (mut p, mut r) = (0, 0);
        while p < a.len() && r < b.len() {
            match a[p].0.cmp(&b[r].0) {
                std::cmp::Ordering::Less => p += 1,
                std::cmp::Ordering::Greater => r += 1,
                std::cmp::Ordering::Equal => {
                    exy[o + a[p].0 as usize] += f64::from(a[p].1) * f64::from(b[r].1);
                    p += 1;
                    r += 1;
                }
            }
        }
    }
    // the section's totals, then each block's deletion, feature innermost
    let row_sum = |v: &[f64], rows: usize| {
        let mut t = vec![0f64; g];
        for k in 0..rows {
            for (a, &x) in t.iter_mut().zip(&v[k * g..(k + 1) * g]) {
                *a += x;
            }
        }
        t
    };
    let (n, w) = (nb.iter().sum::<f64>(), m.iter().sum::<f64>());
    let (tsx, tsxx) = (row_sum(&sx, n_blocks), row_sum(&sxx, n_blocks));
    let (texy, tex) = (row_sum(&exy, n_pairs), row_sum(&ex, n_pairs));
    let (num, den): (Vec<f64>, Vec<f64>) = (0..g)
        .map(|f| centred(n, tsx[f], tsxx[f], w, texy[f], tex[f]))
        .unzip();
    let (mut dn, mut dw) = (vec![0f64; n_blocks], vec![0f64; n_blocks]);
    let (mut dnum, mut dden) = (vec![0f64; n_blocks * g], vec![0f64; n_blocks * g]);
    for k in 0..n_blocks {
        let wk: f64 = pairs_of[k].iter().map(|&q| m[q]).sum();
        dn[k] = -nb[k];
        dw[k] = -wk;
        for f in 0..g {
            let (mut pxy, mut px) = (0f64, 0f64);
            for &q in &pairs_of[k] {
                pxy += exy[q * g + f];
                px += ex[q * g + f];
            }
            let (nu, de) = centred(
                n - nb[k],
                tsx[f] - sx[k * g + f],
                tsxx[f] - sxx[k * g + f],
                w - wk,
                texy[f] - pxy,
                tex[f] - px,
            );
            dnum[k * g + f] = nu - num[f];
            dden[k * g + f] = de - den[f];
        }
    }
    SectionJack {
        n_features,
        n,
        w,
        num,
        den,
        dn,
        dw,
        dnum,
        dden,
    }
}

/// Moran's I pooled over sections (each centred on its own mean), its
/// delete-one-block jackknife standard error, and `t = (I + 1/(N − 1)) / se`
/// (0 where a feature does not vary). Deleting a block changes only its own
/// section.
pub fn jackknife_moran(sections: &[SectionJack]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let g = sections.first().map_or(0, |s| s.n_features);
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
                    (0..s.dn.len()).filter_map(move |k| {
                        let x = k * g + f;
                        moran(n + s.dn[k], w + s.dw[k], num + s.dnum[x], den + s.dden[x])
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
    if features.is_empty() {
        return;
    }
    let all = features.clone();
    features.retain(|&g| rng.random::<f32>() < pi[g]);
    if features.is_empty() {
        let total: f32 = all.iter().map(|&g| pi[g]).sum();
        let pick = if total > 0.0 {
            let mut u = rng.random::<f32>() * total;
            *all.iter()
                .find(|&&g| {
                    u -= pi[g];
                    u <= 0.0
                })
                .unwrap_or(&all[all.len() - 1])
        } else {
            all[rng.random_range(0..all.len())]
        };
        features.push(pick);
    }
}

fn std_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// `π_g = 1 − lfdr_g` from the two-group model on `t`, with the null share
/// `π₀`. See the module docs.
pub fn two_group_inclusion(t: &[f64]) -> (Vec<f32>, f64) {
    let n = t.len();
    if n < 3 {
        return (vec![1.0; n], 0.0);
    }
    let mut s = t.to_vec();
    let med = median(&mut s);
    let mut dev: Vec<f64> = t.iter().map(|v| (v - med).abs()).collect();
    let mad = 1.4826 * median(&mut dev);
    if mad <= 0.0 {
        return (vec![1.0; n], 0.0);
    }
    // the bulk within one null sd holds 68.27% of the null
    let bulk = t.iter().filter(|&&v| (v - med).abs() < mad).count() as f64 / n as f64;
    let pi0 = (bulk / 0.682_689_492).min(1.0);
    // mixture density: Gaussian kernel (Silverman's bandwidth) on a grid
    let mean = t.iter().sum::<f64>() / n as f64;
    let sd = (t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let (q1, q3) = (s[n / 4], s[(3 * n) / 4]);
    let iqr = (q3 - q1) / 1.34;
    let spread = if iqr > 0.0 { sd.min(iqr) } else { sd }.max(f64::EPSILON);
    let h = 0.9 * spread * (n as f64).powf(-0.2);
    let (lo, hi) = (s[0] - 4.0 * h, s[n - 1] + 4.0 * h);
    const GRID: usize = 2048;
    let step = (hi - lo) / (GRID - 1) as f64;
    let mut hist = vec![0f64; GRID];
    for &v in t {
        hist[(((v - lo) / step).round() as usize).min(GRID - 1)] += 1.0;
    }
    let reach = ((4.0 * h) / step).ceil() as usize;
    let kernel: Vec<f64> = (0..=reach)
        .map(|d| std_normal_pdf(d as f64 * step / h) / (n as f64 * h))
        .collect();
    let dens: Vec<f64> = (0..GRID)
        .into_par_iter()
        .map(|x| {
            let a = x.saturating_sub(reach);
            let b = (x + reach).min(GRID - 1);
            (a..=b).map(|y| hist[y] * kernel[x.abs_diff(y)]).sum()
        })
        .collect();
    let f_at = |v: f64| dens[(((v - lo) / step).round() as usize).min(GRID - 1)];
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

/// Units (cells, or the finest PB super-cells) with their log-normalised
/// profiles, section, jackknife block and the spatial edges between them:
/// what [`inclusion_of`] scores. A block lies within one section.
#[derive(Debug, Clone)]
pub struct UnitGraph {
    pub profiles: Vec<Vec<(u32, f32)>>,
    pub section: Vec<u32>,
    pub block: Vec<usize>,
    pub edges: Vec<(usize, usize)>,
}

/// One section's edges as pairs of its own unit indices.
type LocalEdges = Vec<(usize, usize)>;

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

/// Jackknife blocks from a tree of nested labels per unit (`levels`, coarse
/// to fine): the coarsest level that gives every section at least
/// `per_section` blocks, a block being one label within one section (the
/// last level when none does). Dense ids.
pub fn tree_blocks(levels: &[Vec<usize>], section: &[u32], per_section: usize) -> Vec<usize> {
    let n_sections = section.iter().map(|&s| s as usize + 1).max().unwrap_or(0);
    let mut last: Vec<usize> = (0..section.len()).collect();
    for labels in levels {
        let ids = dense_ids(section.iter().zip(labels));
        let mut first_of_id = vec![true; ids.iter().map(|&b| b as usize + 1).max().unwrap_or(0)];
        let mut count = vec![0usize; n_sections];
        for (&b, &s) in ids.iter().zip(section) {
            if std::mem::replace(&mut first_of_id[b as usize], false) {
                count[s as usize] += 1;
            }
        }
        last = ids.into_iter().map(|b| b as usize).collect();
        if count.iter().all(|&c| c >= per_section) {
            break;
        }
    }
    last
}

/// [`FeatureInclusion`] of the units of `g`: Moran's I per section, pooled,
/// jackknifed over `g`'s blocks, then the two-group model.
pub fn inclusion_of(n_features: usize, g: &UnitGraph) -> FeatureInclusion {
    let (units_of, edges_of) = split_sections(&g.section, &g.edges);
    let sections: Vec<SectionJack> = units_of
        .par_iter()
        .zip(&edges_of)
        .map(|(units, es)| {
            let profiles: Vec<&[(u32, f32)]> =
                units.iter().map(|&u| g.profiles[u].as_slice()).collect();
            let block: Vec<usize> = units.iter().map(|&u| g.block[u]).collect();
            section_jack(n_features, &profiles, &block, es)
        })
        .collect();
    finish(n_features, &sections, "units")
}

/// Jackknife, two-group model and a log line, over `sections`.
fn finish(n_features: usize, sections: &[SectionJack], units: &str) -> FeatureInclusion {
    let n_blocks: usize = sections.iter().map(|s| s.dn.len()).sum();
    let n_edges: f64 = sections.iter().map(|s| s.w).sum();
    let (moran, se, t) = jackknife_moran(sections);
    let (pi, pi0) = two_group_inclusion(&t);
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

/// Each PB's section: the most common section of its cells (ties to the
/// lowest).
fn majority_section(cell_to_pb: &[usize], n_pb: usize, section: &[u32]) -> Vec<u32> {
    let n_sections = section.iter().map(|&s| s as usize + 1).max().unwrap_or(0);
    let mut count = vec![0usize; n_pb * n_sections];
    for (&p, &s) in cell_to_pb.iter().zip(section) {
        count[p * n_sections + s as usize] += 1;
    }
    (0..n_pb)
        .map(|p| {
            let row = &count[p * n_sections..(p + 1) * n_sections];
            let best = row
                .iter()
                .enumerate()
                .fold(0, |b, (s, &c)| if c > row[b] { s } else { b });
            best as u32
        })
        .collect()
}

/// The finest PBs' inputs to [`pb_inclusion`].
pub struct PbUnits<'a> {
    /// Pooled raw counts on the matrix ROW axis, `rows × n_pb`.
    pub counts: &'a Mat,
    /// Each cell's finest PB.
    pub cell_to_pb: &'a [usize],
    /// Each PB's parent at every level coarser than the finest, coarse first.
    pub parents: &'a [Vec<usize>],
}

/// [`FeatureInclusion`] over the finest PB super-cells: their pooled
/// counts, the spatial graph's edges mapped onto them (distinct pairs of
/// different PBs), each PB's majority section, and coarse super-cells of
/// the PB tree as jackknife blocks (about `per_section` per section).
pub fn pb_inclusion(
    axis: &FeatureAxis,
    pb: &PbUnits<'_>,
    section: &[u32],
    spatial_edges: &[(usize, usize)],
    per_section: usize,
) -> FeatureInclusion {
    let n_pb = pb.counts.ncols();
    let pooled = axis.pool_rows_opt(pb.counts);
    let profiles = normalised_profiles(pooled.as_ref().unwrap_or(pb.counts));
    drop(pooled);
    let pb_section = majority_section(pb.cell_to_pb, n_pb, section);
    let mut edges: Vec<(usize, usize)> = spatial_edges
        .iter()
        .map(|&(i, j)| (pb.cell_to_pb[i], pb.cell_to_pb[j]))
        .filter(|&(a, b)| a != b)
        .map(|(a, b)| (a.min(b), a.max(b)))
        .collect();
    edges.sort_unstable();
    edges.dedup();
    let block = tree_blocks(pb.parents, &pb_section, per_section);
    let g = UnitGraph {
        profiles,
        section: pb_section,
        block,
        edges,
    };
    inclusion_of(axis.n_features(), &g)
}

/// [`FeatureInclusion`] over cells (cage-cell, which builds no pseudobulks):
/// each cell's log-normalised counts, the spatial graph `edges` (global cell
/// pairs; pairs across sections are left out), and a grid of geometric
/// tiles over each section's plane (`xy`, cells × ≥2) as jackknife blocks,
/// about `per_section` per section. Sections are read in parallel, each
/// reduced to its sums before its memory is released.
pub fn cell_inclusion(
    data: &SparseIoVec,
    axis: &FeatureAxis,
    xy: &Mat,
    section: &[u32],
    edges: &[(usize, usize)],
    per_section: usize,
) -> anyhow::Result<FeatureInclusion> {
    let tiles = (per_section as f64).sqrt().ceil().max(1.0) as usize;
    let n_features = axis.n_features();
    let (cells_of, edges_of) = split_sections(section, edges);
    let sections: Vec<SectionJack> = cells_of
        .par_iter()
        .zip(&edges_of)
        .map(|(cells, es)| -> anyhow::Result<SectionJack> {
            let csc = data.read_columns_csc(cells.iter().copied())?;
            let profiles: Vec<Vec<(u32, f32)>> = csc
                .col_iter()
                .map(|col| {
                    let obs: Vec<(u32, f32)> = col
                        .row_indices()
                        .iter()
                        .zip(col.values())
                        .map(|(&r, &v)| (r as u32, v))
                        .collect();
                    let mut p = axis.pool_profile(obs);
                    p.sort_by_key(|&(f, _)| f);
                    log_normalise(&mut p);
                    p
                })
                .collect();
            drop(csc);
            let plane: Vec<(f32, f32)> = cells.iter().map(|&i| (xy[(i, 0)], xy[(i, 1)])).collect();
            Ok(section_jack(
                n_features,
                &profiles,
                &tile_of(&plane, tiles),
                es,
            ))
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(finish(n_features, &sections, "cells"))
}
