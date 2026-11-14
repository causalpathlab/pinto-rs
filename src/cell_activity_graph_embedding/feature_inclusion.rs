//! Soft feature inclusion: how likely each feature is to carry spatial
//! signal, as a probability `π_g` the training loop samples features by.
//!
//! cage gives every trained feature its own adjacency task with the same
//! positive budget, so on data where most features are noise the noise
//! tasks swamp the signal (a hard HVG cut helps, but needs a threshold and
//! drops features outright). Here each feature instead gets
//!
//! 1. **Moran's I** of its log-normalised expression over the spatial graph,
//!    centred within each section and pooled over sections;
//! 2. a **delete-one-tile jackknife** standard error: each section's plane is
//!    cut into `tiles × tiles` tiles and the pooled I is recomputed without
//!    each tile: its cells, every edge touching them, and their share of the
//!    section's mean (an exact deletion, so the left-out cells inform nothing). Neighbouring
//!    spots are correlated, so the textbook variance is too small; the
//!    jackknife's is not, and a feature whose structure sits in one patch
//!    gets a wide one. Blocks are tiles within sections, never whole
//!    sections, so a domain only one section holds is not penalised;
//! 3. a **two-group empirical-Bayes** model on `t = (I − E₀[I]) / se`: the null
//!    component is a normal fitted to the bulk (median, MAD), its share `π₀`
//!    is estimated from the bulk too, the mixture density by a binned Gaussian
//!    kernel estimate, and `π_g = 1 − lfdr_g` (Efron's local false discovery
//!    rate), one-sided (only positive autocorrelation is signal) and
//!    non-decreasing in `t`. No threshold or top-N is chosen.
//!
//! Each epoch then visits feature `g` with probability `π_g` (a seeded
//! Bernoulli mask), so the expected objective is `Σ_g π_g L_g`: features are
//! weighted, not cut, and the likely-noise ones mostly cost nothing.

use crate::util::common::Mat;
use crate::util::feature_axis::FeatureAxis;
use data_beans::sparse_io_vector::SparseIoVec;
use log::info;
use rayon::prelude::*;

/// What [`moran_inclusion`] returns, per feature.
#[derive(Debug, Clone)]
pub struct FeatureInclusion {
    pub moran: Vec<f64>,
    pub se: Vec<f64>,
    pub t: Vec<f64>,
    pub pi: Vec<f32>,
}

/// One section's raw sums, by block, from which Moran's I of the section
/// with any one block deleted follows in closed form: the block's units,
/// every edge touching them, and their share of the section mean all go.
/// Per-feature arrays are row-major, `rows × features`.
#[derive(Debug, Clone)]
pub struct SectionSums {
    pub n_features: usize,
    /// Per block: units, and `Σ x`, `Σ x²` over its units.
    pub n: Vec<f64>,
    pub sx: Vec<f64>,
    pub sxx: Vec<f64>,
    /// Per pair of blocks `(a ≤ b)` holding an edge: its edges, and
    /// `Σ x_i x_j`, `Σ (x_i + x_j)` over them.
    pub pairs: Vec<(usize, usize)>,
    /// The pairs touching each block.
    pub pairs_of: Vec<Vec<usize>>,
    pub m: Vec<f64>,
    pub exy: Vec<f64>,
    pub ex: Vec<f64>,
}

/// A section's raw sums for one feature: units, `Σ x`, `Σ x²`, edges,
/// `Σ x_i x_j`, `Σ (x_i + x_j)`.
#[derive(Clone, Copy, Default)]
struct Raw {
    n: f64,
    sx: f64,
    sxx: f64,
    w: f64,
    exy: f64,
    ex: f64,
}

/// The centred Moran sums of a [`Raw`]: `Σ_edges (x_i − μ)(x_j − μ)`,
/// `Σ_units (x_i − μ)²`, units and edges.
struct Centred {
    num: f64,
    den: f64,
    n: f64,
    w: f64,
}

impl Raw {
    fn centred(&self) -> Centred {
        let mu = if self.n > 0.0 { self.sx / self.n } else { 0.0 };
        Centred {
            num: self.exy - mu * self.ex + self.w * mu * mu,
            den: self.sxx - 2.0 * mu * self.sx + self.n * mu * mu,
            n: self.n,
            w: self.w,
        }
    }
}

impl SectionSums {
    pub fn n_blocks(&self) -> usize {
        self.n.len()
    }

    fn total(&self, f: usize) -> Raw {
        let g = self.n_features;
        let mut r = Raw::default();
        for k in 0..self.n_blocks() {
            r.n += self.n[k];
            r.sx += self.sx[k * g + f];
            r.sxx += self.sxx[k * g + f];
        }
        for q in 0..self.pairs.len() {
            r.w += self.m[q];
            r.exy += self.exy[q * g + f];
            r.ex += self.ex[q * g + f];
        }
        r
    }

    /// `total` with block `k`, and every pair touching it, taken out.
    fn without(&self, f: usize, total: &Raw, k: usize) -> Raw {
        let g = self.n_features;
        let mut r = *total;
        r.n -= self.n[k];
        r.sx -= self.sx[k * g + f];
        r.sxx -= self.sxx[k * g + f];
        for &q in &self.pairs_of[k] {
            r.w -= self.m[q];
            r.exy -= self.exy[q * g + f];
            r.ex -= self.ex[q * g + f];
        }
        r
    }
}

/// Each cell's tile in a `tiles × tiles` grid over the bounding box of
/// `xy` (cells × 2, one section).
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

/// One section's [`SectionSums`] from its units' log-normalised profiles
/// (`(feature, value)` lists sorted by feature), each unit's block (dense
/// within the section), and the section's edges as local unit pairs.
pub fn section_sums<P: AsRef<[(u32, f32)]>>(
    n_features: usize,
    profiles: &[P],
    tile: &[usize],
    edges: &[(usize, usize)],
) -> SectionSums {
    let g = n_features;
    let n_tiles = tile.iter().max().map_or(0, |&b| b + 1);
    let (mut n, mut sx, mut sxx) = (
        vec![0f64; n_tiles],
        vec![0f64; n_tiles * g],
        vec![0f64; n_tiles * g],
    );
    for (i, p) in profiles.iter().enumerate() {
        let o = tile[i] * g;
        n[tile[i]] += 1.0;
        for &(f, v) in p.as_ref() {
            let v = f64::from(v);
            sx[o + f as usize] += v;
            sxx[o + f as usize] += v * v;
        }
    }
    let mut pair_of: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
    for &(i, j) in edges {
        let (a, b) = (tile[i].min(tile[j]), tile[i].max(tile[j]));
        let next = pair_of.len();
        pair_of.entry((a, b)).or_insert(next);
    }
    let n_pairs = pair_of.len();
    let (mut m, mut exy, mut ex) = (
        vec![0f64; n_pairs],
        vec![0f64; n_pairs * g],
        vec![0f64; n_pairs * g],
    );
    for &(i, j) in edges {
        let q = pair_of[&(tile[i].min(tile[j]), tile[i].max(tile[j]))];
        let o = q * g;
        m[q] += 1.0;
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
    let mut pairs = vec![(0, 0); n_pairs];
    let mut pairs_of = vec![Vec::new(); n_tiles];
    for (&(a, b), &q) in &pair_of {
        pairs[q] = (a, b);
        pairs_of[a].push(q);
        if b != a {
            pairs_of[b].push(q);
        }
    }
    SectionSums {
        n_features,
        n,
        sx,
        sxx,
        pairs,
        pairs_of,
        m,
        exy,
        ex,
    }
}

/// Moran's I pooled over sections (each centred on its own mean), its
/// delete-one-tile jackknife standard error, and `t = (I + 1/(N − 1)) / se`
/// (0 where a feature does not vary). Deleting a tile removes its cells,
/// every edge touching them and their share of their section's mean, and
/// leaves the other sections as they are.
pub fn jackknife_moran(sections: &[SectionSums]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let g = sections.first().map_or(0, |s| s.n_features);
    let per: Vec<(f64, f64, f64)> = (0..g)
        .into_par_iter()
        .map(|f| {
            let totals: Vec<Raw> = sections.iter().map(|s| s.total(f)).collect();
            let full: Vec<Centred> = totals.iter().map(Raw::centred).collect();
            let sum = |c: &[&Centred]| {
                c.iter().fold((0f64, 0f64, 0f64, 0f64), |a, c| {
                    (a.0 + c.num, a.1 + c.den, a.2 + c.n, a.3 + c.w)
                })
            };
            let (num, den, n, w) = sum(&full.iter().collect::<Vec<_>>());
            if den <= 0.0 || w <= 0.0 {
                return (0.0, 0.0, 0.0);
            }
            let i_all = (n / w) * num / den;
            let mut loo = Vec::new();
            for (s, sec) in sections.iter().enumerate() {
                for k in 0..sec.n_blocks() {
                    let r = sec.without(f, &totals[s], k).centred();
                    // the other sections as they are, this one without block k
                    let (dn, dd, dc, dw) = (
                        r.num - full[s].num,
                        r.den - full[s].den,
                        r.n - full[s].n,
                        r.w - full[s].w,
                    );
                    let (num_k, den_k, n_k, w_k) = (num + dn, den + dd, n + dc, w + dw);
                    if den_k > 0.0 && w_k > 0.0 {
                        loo.push((n_k / w_k) * num_k / den_k);
                    }
                }
            }
            let m = loo.len() as f64;
            if m < 2.0 {
                return (i_all, 0.0, 0.0);
            }
            let mean = loo.iter().sum::<f64>() / m;
            let se = ((m - 1.0) / m * loo.iter().map(|v| (v - mean).powi(2)).sum::<f64>()).sqrt();
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
/// in order.
/// Never empty: if no feature is kept, one is drawn with probability
/// proportional to `pi` (uniformly when every `pi` is 0), so an epoch always
/// trains.
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
    pub section: Vec<usize>,
    pub block: Vec<usize>,
    pub edges: Vec<(usize, usize)>,
}

/// Each column of `pooled` (features × units, raw counts) as a sparse
/// profile, `ln(1 + 1e4 x / Σ x)`; an empty unit stays empty.
pub fn normalised_profiles(pooled: &Mat) -> Vec<Vec<(u32, f32)>> {
    pooled
        .column_iter()
        .map(|col| {
            let lib: f32 = col.iter().sum();
            col.iter()
                .enumerate()
                .filter(|&(_, &v)| v > 0.0 && lib > 0.0)
                .map(|(f, &v)| (f as u32, (1e4 * v / lib).ln_1p()))
                .collect()
        })
        .collect()
}

/// Jackknife blocks from a tree of nested labels per unit (`levels`, coarse
/// to fine): the coarsest level that gives every section at least
/// `per_section` blocks, a block being one label within one section (the
/// finest level when none does). Dense ids.
pub fn tree_blocks(levels: &[Vec<usize>], section: &[usize], per_section: usize) -> Vec<usize> {
    let n_sections = section.iter().max().map_or(0, |&s| s + 1);
    let dense = |labels: &[usize]| {
        let mut id: std::collections::HashMap<(usize, usize), usize> = Default::default();
        let mut count = vec![0usize; n_sections];
        let blocks: Vec<usize> = labels
            .iter()
            .zip(section)
            .map(|(&l, &s)| {
                let next = id.len();
                *id.entry((s, l)).or_insert_with(|| {
                    count[s] += 1;
                    next
                })
            })
            .collect();
        (blocks, count)
    };
    let mut last = (0..section.len()).collect();
    for labels in levels {
        let (blocks, count) = dense(labels);
        if count.iter().all(|&c| c >= per_section) {
            return blocks;
        }
        last = blocks;
    }
    last
}

/// [`FeatureInclusion`] of the units of `g`: Moran's I per section, pooled,
/// jackknifed over `g`'s blocks, then the two-group model.
pub fn inclusion_of(n_features: usize, g: &UnitGraph) -> FeatureInclusion {
    let n_sections = g.section.iter().max().map_or(0, |&s| s + 1);
    let mut units_of = vec![Vec::new(); n_sections];
    let mut local = vec![0usize; g.section.len()];
    for (u, &s) in g.section.iter().enumerate() {
        local[u] = units_of[s].len();
        units_of[s].push(u);
    }
    let mut edges_of = vec![Vec::new(); n_sections];
    for &(i, j) in &g.edges {
        if g.section[i] == g.section[j] {
            edges_of[g.section[i]].push((local[i], local[j]));
        }
    }
    let sections: Vec<SectionSums> = units_of
        .par_iter()
        .zip(&edges_of)
        .map(|(units, es)| {
            let mut id: std::collections::HashMap<usize, usize> = Default::default();
            let block: Vec<usize> = units
                .iter()
                .map(|&u| {
                    let next = id.len();
                    *id.entry(g.block[u]).or_insert(next)
                })
                .collect();
            let profiles: Vec<&[(u32, f32)]> =
                units.iter().map(|&u| g.profiles[u].as_slice()).collect();
            section_sums(n_features, &profiles, &block, es)
        })
        .collect();
    finish(n_features, &sections, "units")
}

/// Jackknife, two-group model and a log line, over `sections`.
fn finish(n_features: usize, sections: &[SectionSums], units: &str) -> FeatureInclusion {
    let n_blocks: usize = sections.iter().map(SectionSums::n_blocks).sum();
    let n_edges: f64 = sections.iter().flat_map(|s| s.m.iter()).sum();
    let (moran, se, t) = jackknife_moran(sections);
    let (pi, pi0) = two_group_inclusion(&t);
    let total: f32 = pi.iter().sum();
    info!(
        "feature inclusion: Moran's I over {n_edges} edges between {units}, jackknife over \
         {n_blocks} blocks; null share {pi0:.2}, expected {total:.0} of {n_features} features \
         per epoch"
    );
    if total < 1.0 {
        log::warn!(
            "feature inclusion: no feature shows spatial signal (expected {total:.2} per epoch); \
             each epoch trains at least one"
        );
    }
    FeatureInclusion { moran, se, t, pi }
}

/// [`FeatureInclusion`] over the finest PB super-cells of `frame`: their
/// pooled counts, the super edges between them, and coarse super-cells of
/// the PB tree as jackknife blocks (about `per_section` per section).
pub fn pb_inclusion(
    data: &SparseIoVec,
    axis: &FeatureAxis,
    frame: &crate::cell_activity_graph_embedding::pb_frame::PbFrame,
    per_section: usize,
) -> anyhow::Result<FeatureInclusion> {
    let rows = crate::link_community::profiles::coarsen_cell_expression_dense(
        data,
        &frame.cell_to_finest_pb,
        frame.n_pb,
    )?;
    let pooled = axis.pool_rows_opt(&rows).unwrap_or(rows);
    let section: Vec<usize> = frame.pb_exp_batch.iter().map(|&b| b as usize).collect();
    let block = tree_blocks(&frame.pb_parent_maps, &section, per_section);
    let g = UnitGraph {
        profiles: normalised_profiles(&pooled),
        section,
        block,
        edges: frame
            .super_edges
            .iter()
            .map(|&(a, b)| (a as usize, b as usize))
            .collect(),
    };
    Ok(inclusion_of(axis.n_features(), &g))
}

/// [`FeatureInclusion`] over cells (cage-cell, which builds no pseudobulks):
/// each cell's log-normalised counts, the spatial graph `edges` (global cell
/// pairs; pairs across sections are left out), and a grid of geometric
/// tiles over each section's plane (`xy`, cells × ≥2) as jackknife blocks,
/// about `per_section` per section. Reads one section at a time.
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
    let n_sections = section.iter().map(|&s| s as usize + 1).max().unwrap_or(0);
    let mut cells_of = vec![Vec::new(); n_sections];
    let mut local = vec![0usize; section.len()];
    for (i, &s) in section.iter().enumerate() {
        local[i] = cells_of[s as usize].len();
        cells_of[s as usize].push(i);
    }
    let mut edges_of = vec![Vec::new(); n_sections];
    for &(i, j) in edges {
        if section[i] == section[j] {
            edges_of[section[i] as usize].push((local[i], local[j]));
        }
    }
    let sections: Vec<SectionSums> = cells_of
        .par_iter()
        .zip(&edges_of)
        .map(|(cells, es)| -> anyhow::Result<SectionSums> {
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
                    let lib: f32 = p.iter().map(|&(_, v)| v).sum();
                    if lib > 0.0 {
                        p.iter_mut()
                            .for_each(|(_, v)| *v = (1e4 * *v / lib).ln_1p());
                    }
                    p
                })
                .collect();
            let plane: Vec<(f32, f32)> = cells.iter().map(|&i| (xy[(i, 0)], xy[(i, 1)])).collect();
            Ok(section_sums(
                n_features,
                &profiles,
                &tile_of(&plane, tiles),
                es,
            ))
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(finish(n_features, &sections, "cells"))
}
