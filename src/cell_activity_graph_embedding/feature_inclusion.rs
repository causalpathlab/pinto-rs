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

/// One section's raw sums, by tile, from which Moran's I of the section
/// with any one tile deleted follows in closed form: the tile's cells, every
/// edge touching them, and their share of the section mean all go.
/// Per-feature arrays are row-major, `rows × features`.
#[derive(Debug, Clone)]
pub struct SectionSums {
    pub n_features: usize,
    /// Per tile: cells, and `Σ x`, `Σ x²` over its cells.
    pub n: Vec<f64>,
    pub sx: Vec<f64>,
    pub sxx: Vec<f64>,
    /// Per pair of tiles `(a ≤ b)` holding an edge: its edges, and
    /// `Σ x_i x_j`, `Σ (x_i + x_j)` over them.
    pub pairs: Vec<(usize, usize)>,
    pub m: Vec<f64>,
    pub exy: Vec<f64>,
    pub ex: Vec<f64>,
}

/// A section's centred Moran sums for one feature: `Σ_edges (x_i − μ)(x_j − μ)`,
/// `Σ_cells (x_i − μ)²`, cells and edges, with tile `drop` (if any) deleted.
struct Centred {
    num: f64,
    den: f64,
    n: f64,
    w: f64,
}

impl SectionSums {
    pub fn n_tiles(&self) -> usize {
        self.n.len()
    }

    fn centred(&self, f: usize, drop: Option<usize>) -> Centred {
        let g = self.n_features;
        let keep_tile = |k: usize| drop != Some(k);
        let keep_pair = |&(a, b): &(usize, usize)| keep_tile(a) && keep_tile(b);
        let (mut n, mut sx, mut sxx) = (0f64, 0f64, 0f64);
        for k in (0..self.n_tiles()).filter(|&k| keep_tile(k)) {
            n += self.n[k];
            sx += self.sx[k * g + f];
            sxx += self.sxx[k * g + f];
        }
        let (mut w, mut exy, mut ex) = (0f64, 0f64, 0f64);
        for (q, _) in self.pairs.iter().enumerate().filter(|(_, p)| keep_pair(p)) {
            w += self.m[q];
            exy += self.exy[q * g + f];
            ex += self.ex[q * g + f];
        }
        let mu = if n > 0.0 { sx / n } else { 0.0 };
        Centred {
            num: exy - mu * ex + w * mu * mu,
            den: sxx - 2.0 * mu * sx + n * mu * mu,
            n,
            w,
        }
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

/// One section's [`SectionSums`] from its cells' log-normalised profiles
/// (`(feature, value)` lists sorted by feature), each cell's tile, and the
/// section's edges as local cell pairs.
pub fn section_sums(
    n_features: usize,
    profiles: &[Vec<(u32, f32)>],
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
        for &(f, v) in p {
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
        let (a, b) = (&profiles[i], &profiles[j]);
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
    for (&key, &q) in &pair_of {
        pairs[q] = key;
    }
    SectionSums {
        n_features,
        n,
        sx,
        sxx,
        pairs,
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
            let full: Vec<Centred> = sections.iter().map(|s| s.centred(f, None)).collect();
            let total = |skip: Option<(usize, &Centred)>| {
                let (mut num, mut den, mut n, mut w) = (0f64, 0f64, 0f64, 0f64);
                for (s, c) in full.iter().enumerate() {
                    let c = match skip {
                        Some((t, r)) if t == s => r,
                        _ => c,
                    };
                    num += c.num;
                    den += c.den;
                    n += c.n;
                    w += c.w;
                }
                (num, den, n, w)
            };
            let (num, den, n, w) = total(None);
            if den <= 0.0 || w <= 0.0 {
                return (0.0, 0.0, 0.0);
            }
            let i_all = (n / w) * num / den;
            let mut loo = Vec::new();
            for (s, sec) in sections.iter().enumerate() {
                for k in 0..sec.n_tiles() {
                    let r = sec.centred(f, Some(k));
                    let (num, den, n, w) = total(Some((s, &r)));
                    if den > 0.0 && w > 0.0 {
                        loo.push((n / w) * num / den);
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
pub fn sample_included(features: &mut Vec<usize>, pi: &[f32], rng: &mut impl rand::Rng) {
    use rand::RngExt;
    features.retain(|&g| rng.random::<f32>() < pi[g]);
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

/// [`FeatureInclusion`] of every feature of `data` over the spatial graph
/// `edges` (global cell pairs; pairs across sections are left out), with
/// `section` each cell's section and `xy` (cells × ≥2) its plane coordinates.
/// Reads one section at a time.
pub fn moran_inclusion(
    data: &SparseIoVec,
    axis: &FeatureAxis,
    xy: &Mat,
    section: &[u32],
    edges: &[(usize, usize)],
    tiles: usize,
) -> anyhow::Result<FeatureInclusion> {
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
            let tile = tile_of(&plane, tiles);
            Ok(section_sums(n_features, &profiles, &tile, es))
        })
        .collect::<anyhow::Result<_>>()?;
    let n_blocks: usize = sections.iter().map(SectionSums::n_tiles).sum();
    let (moran, se, t) = jackknife_moran(&sections);
    let (pi, pi0) = two_group_inclusion(&t);
    let total: f32 = pi.iter().sum();
    info!(
        "feature inclusion: Moran's I over {} edges, jackknife over {n_blocks} tiles; \
         null share {pi0:.2}, expected {total:.0} of {n_features} features per epoch",
        edges.len()
    );
    Ok(FeatureInclusion { moran, se, t, pi })
}
