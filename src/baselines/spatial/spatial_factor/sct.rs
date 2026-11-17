//! SCTransform, version 1 (Hafemeister and Satija 2019, "Normalization and
//! variance stabilization of single-cell RNA-seq data using regularized
//! negative binomial regression", Genome Biology 20:296,
//! doi:10.1186/s13059-019-1874-1), written from the paper's Methods:
//!
//! 1. Per gene, `log E(x) = beta0 + beta1 log10 m` (`m` a cell's total
//!    count over all genes), fitted as the paper's first method: a Poisson
//!    GLM for the betas, then the negative binomial's `theta` by maximum
//!    likelihood at those means (variance `mu + mu^2 / theta`). The fits
//!    run on 2000 genes drawn with probability `1 / d(log10 gmean)`, `d`
//!    the density of the genes' log10 geometric means (every gene when
//!    there are at most 2000).
//! 2. Each parameter (`beta0`, `beta1` and `log10 theta`) regularized
//!    against the log10 geometric mean by kernel regression, a normal
//!    kernel at three times `bw.SJ`'s bandwidth (R's `ksmooth`: the
//!    kernel's quartiles at a quarter of the bandwidth either side).
//! 3. Pearson residuals `(x - mu) / sqrt(mu + mu^2 / theta)` at the
//!    regularized parameters, clipped at `+-sqrt(N)`, `N` the cells.
//!
//! The geometric mean is the paper's, `exp(mean(log(x + 1))) - 1`. The
//! density of step 1 is a Gaussian kernel density at Silverman's bandwidth
//! (R's `density` default), on a grid of 512 points; the draw is seeded.
//! `theta` is smoothed on the log scale, as the paper plots it; the paper
//! does not say which. Its maximum likelihood is searched on `[1e-6,
//! 1e7]`, the upper end a Poisson gene.

use super::bandwidth::{sheather_jones, silverman};
use super::kit::grid_then_brent;
use nalgebra_sparse::CscMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Genes the per-gene models are fitted on.
pub const FIT_GENES: usize = 2000;
/// The bandwidth adjustment factor on `bw.SJ`.
pub const BAF: f64 = 3.0;
const LOG_THETA: (f64, f64) = (-13.815_510_557_964_274, 16.118_095_650_958_32);
const DENSITY_GRID: usize = 512;
/// `qnorm(0.75)`: `ksmooth`'s normal kernel has its quartiles at `+-0.25`
/// bandwidths.
const Q75: f64 = 0.674_489_750_196_081_7;

/// One gene's counts: its nonzero entries, cells and counts.
#[derive(Debug, Clone, Copy)]
pub struct GeneCounts<'a> {
    cells: &'a [u32],
    counts: &'a [f32],
}

impl GeneCounts<'_> {
    /// The `(cell, count)` entries.
    pub fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.cells
            .iter()
            .zip(self.counts)
            .map(|(&c, &v)| (c as usize, v as f64))
    }
}

/// Every gene's nonzero counts, by gene, held at their exact size: a cell
/// index (4 bytes) and a count (4) a nonzero.
#[derive(Debug, Clone)]
pub struct ByGene {
    ptr: Vec<usize>,
    cells: Vec<u32>,
    counts: Vec<f32>,
}

impl ByGene {
    /// The nonzeros of `counts` (genes × cells), counted per gene first.
    pub fn from_csc(counts: &CscMatrix<f32>) -> Self {
        let m = counts.nrows();
        let mut ptr = vec![0usize; m + 1];
        for (&g, _) in counts
            .row_indices()
            .iter()
            .zip(counts.values())
            .filter(|e| *e.1 != 0.0)
        {
            ptr[g + 1] += 1;
        }
        for g in 0..m {
            ptr[g + 1] += ptr[g];
        }
        let mut next = ptr.clone();
        let mut cells = vec![0u32; ptr[m]];
        let mut values = vec![0f32; ptr[m]];
        for (c, col) in counts.col_iter().enumerate() {
            for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                if v != 0.0 {
                    cells[next[g]] = c as u32;
                    values[next[g]] = v;
                    next[g] += 1;
                }
            }
        }
        ByGene {
            ptr,
            cells,
            counts: values,
        }
    }

    pub fn n_genes(&self) -> usize {
        self.ptr.len() - 1
    }

    pub fn gene(&self, g: usize) -> GeneCounts<'_> {
        let r = self.ptr[g]..self.ptr[g + 1];
        GeneCounts {
            cells: &self.cells[r.clone()],
            counts: &self.counts[r],
        }
    }
}

/// The cells' `log10` libraries, and the distinct finite ones with their
/// numbers of cells: a cell's mean under a gene model depends on its
/// library alone, so a sum over the cells of a function of the mean is a
/// sum over the distinct libraries.
#[derive(Debug, Clone)]
pub struct Depths {
    /// `log10 m` of each cell (`-inf` for a cell with no counts).
    pub t: Vec<f64>,
    /// The distinct finite `t` and how many cells have each.
    levels: Vec<(f64, f64)>,
}

impl Depths {
    pub fn new(lib: &[f64]) -> Self {
        let t: Vec<f64> = lib.iter().map(|&m| m.log10()).collect();
        let mut sorted: Vec<f64> = t.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(f64::total_cmp);
        let mut levels: Vec<(f64, f64)> = Vec::new();
        for v in sorted {
            match levels.last_mut() {
                Some((u, k)) if *u == v => *k += 1.0,
                _ => levels.push((v, 1.0)),
            }
        }
        Depths { t, levels }
    }

    /// The means `exp(beta0 + beta1 t)` of each cell (0 for a cell with
    /// no counts) and of each distinct library with its number of cells.
    pub fn means(&self, beta0: f64, beta1: f64) -> (Vec<f64>, Vec<(f64, f64)>) {
        let mean = |t: f64| (beta0 + beta1 * t).exp();
        let mu = self
            .t
            .iter()
            .map(|&t| if t.is_finite() { mean(t) } else { 0.0 })
            .collect();
        let levels = self.levels.iter().map(|&(t, k)| (mean(t), k)).collect();
        (mu, levels)
    }
}

/// A gene's model: `log mu = beta0 + beta1 log10 m`, dispersion `theta`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct GeneModel {
    pub beta0: f64,
    pub beta1: f64,
    pub theta: f64,
}

/// `exp(mean(log(x + 1))) - 1` of a gene over `n` cells.
pub fn geometric_mean(x: GeneCounts, n: usize) -> f64 {
    (x.iter().map(|(_, v)| v.ln_1p()).sum::<f64>() / n as f64).exp_m1()
}

/// The Poisson GLM `log mu = beta0 + beta1 t` of `x` on cells with
/// `t = log10 m` (a cell with no counts, `t = -inf`, has `mu = 0` and is
/// left out), by Newton's method with step halving.
pub fn poisson_fit(x: GeneCounts, t: &[f64]) -> (f64, f64) {
    let (sx, sxt) = x
        .iter()
        .fold((0.0, 0.0), |(a, b), (c, v)| (a + v, b + v * t[c]));
    let live: Vec<f64> = t.iter().copied().filter(|v| v.is_finite()).collect();
    let total_m: f64 = live.iter().map(|&ti| 10f64.powf(ti)).sum();
    let mut beta = [(sx / total_m).max(1e-300).ln(), std::f64::consts::LN_10];
    let loglik = |b: &[f64; 2]| -> f64 {
        let mu: f64 = live.iter().map(|&ti| (b[0] + b[1] * ti).exp()).sum();
        b[0] * sx + b[1] * sxt - mu
    };
    let mut ll = loglik(&beta);
    for _ in 0..100 {
        let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
        for &ti in &live {
            let mu = (beta[0] + beta[1] * ti).exp();
            s0 += mu;
            s1 += mu * ti;
            s2 += mu * ti * ti;
        }
        let g = [sx - s0, sxt - s1];
        let det = s0 * s2 - s1 * s1;
        if det <= 0.0 {
            break;
        }
        let step = [(s2 * g[0] - s1 * g[1]) / det, (s0 * g[1] - s1 * g[0]) / det];
        let mut h = 1.0;
        loop {
            let cand = [beta[0] + h * step[0], beta[1] + h * step[1]];
            let lc = loglik(&cand);
            if lc >= ll || h < 1e-10 {
                if lc >= ll {
                    beta = cand;
                    ll = lc;
                }
                break;
            }
            h *= 0.5;
        }
        if (h * step[0]).abs().max((h * step[1]).abs()) < 1e-10 {
            break;
        }
    }
    (beta[0], beta[1])
}

/// The negative binomial log-likelihood in `theta` at means `mu` (all
/// cells), up to terms free of `theta`: `sum_nz [lnG(x + theta) -
/// lnG(theta) - x ln(theta + mu)] + sum_all [theta ln theta - theta
/// ln(theta + mu)]`, the second sum over `levels`, the distinct means with
/// their numbers of cells (a cell of mean 0 adds nothing to it).
pub fn nb_loglik(x: GeneCounts, mu: &[f64], levels: &[(f64, f64)], theta: f64) -> f64 {
    let lg = |v: f64| special::Gamma::ln_gamma(v).0;
    let nz: f64 = x
        .iter()
        .map(|(c, v)| lg(v + theta) - lg(theta) - v * (theta + mu[c]).ln())
        .sum();
    let all: f64 = levels
        .iter()
        .map(|&(m, k)| k * (theta * theta.ln() - theta * (theta + m).ln()))
        .sum();
    nz + all
}

/// `theta` maximizing [`nb_loglik`] at means `mu` (`levels` as there), on
/// `log theta` by a grid of every decade then Brent.
pub fn theta_ml(x: GeneCounts, mu: &[f64], levels: &[(f64, f64)]) -> f64 {
    let grid: Vec<f64> = (0..=13)
        .map(|i| LOG_THETA.0 + (LOG_THETA.1 - LOG_THETA.0) * i as f64 / 13.0)
        .collect();
    let (lt, _, _) = grid_then_brent(|lt| -nb_loglik(x, mu, levels, lt.exp()), &grid, 1e-8, 120);
    lt.exp()
}

/// The paper's first fitting method for one gene: the Poisson GLM, then
/// `theta` at its means.
pub fn fit_gene(x: GeneCounts, depths: &Depths) -> GeneModel {
    let (beta0, beta1) = poisson_fit(x, &depths.t);
    let (mu, levels) = depths.means(beta0, beta1);
    GeneModel {
        beta0,
        beta1,
        theta: theta_ml(x, &mu, &levels),
    }
}

/// A Gaussian kernel density of `x` at Silverman's bandwidth, evaluated at
/// each `x` by linear interpolation on a [`DENSITY_GRID`]-point grid
/// spanning `x` with three bandwidths either side.
fn density_at(x: &[f64]) -> Vec<f64> {
    let bw = silverman(x).max(1e-6);
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min) - 3.0 * bw;
    let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 3.0 * bw;
    let step = (hi - lo) / (DENSITY_GRID - 1) as f64;
    let dens: Vec<f64> = (0..DENSITY_GRID)
        .into_par_iter()
        .map(|i| {
            let at = lo + step * i as f64;
            x.iter()
                .map(|&v| (-0.5 * ((at - v) / bw).powi(2)).exp())
                .sum::<f64>()
        })
        .collect();
    x.iter()
        .map(|&v| {
            let f = ((v - lo) / step).clamp(0.0, (DENSITY_GRID - 1) as f64);
            let i = (f.floor() as usize).min(DENSITY_GRID - 2);
            let w = f - i as f64;
            (1.0 - w) * dens[i] + w * dens[i + 1]
        })
        .collect()
}

/// [`FIT_GENES`] of `candidates` (all when fewer), drawn without
/// replacement with probability proportional to `1 / d(log10 gmean)`
/// (Efraimidis and Spirakis 2006, Information Processing Letters 97:181,
/// doi:10.1016/j.ipl.2005.11.003), seeded.
pub fn sample_genes(log_gmean: &[f64], seed: u64) -> Vec<usize> {
    let k = log_gmean.len();
    if k <= FIT_GENES {
        return (0..k).collect();
    }
    let d = density_at(log_gmean);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut keys: Vec<(f64, usize)> = (0..k)
        .map(|i| {
            let u: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
            // key u^(1/w), w = 1 / d: log key = d log u
            (d[i] * u.ln(), i)
        })
        .collect();
    keys.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<usize> = keys[..FIT_GENES].iter().map(|e| e.1).collect();
    out.sort_unstable();
    out
}

/// The per-gene models regularized: each of `beta0`, `beta1` and `log10
/// theta` of the fitted genes (at log10 geometric means `fit_x`) smoothed
/// by kernel regression at `at` (the log10 geometric means of the genes
/// wanted), a normal kernel at [`BAF`] times `bw.SJ` of `fit_x`.
pub fn regularize(fit_x: &[f64], fits: &[GeneModel], at: &[f64]) -> (Vec<GeneModel>, f64) {
    let bw = BAF * sheather_jones(fit_x).max(1e-6);
    let sd = 0.25 * bw / Q75;
    let models = at
        .par_iter()
        .map(|&x0| {
            let (mut w, mut b0, mut b1, mut lt) = (0.0, 0.0, 0.0, 0.0);
            for (x, f) in fit_x.iter().zip(fits) {
                let k = (-0.5 * ((x - x0) / sd).powi(2)).exp();
                w += k;
                b0 += k * f.beta0;
                b1 += k * f.beta1;
                lt += k * f.theta.log10();
            }
            if w <= 0.0 {
                // beyond every fitted gene: the nearest one's
                let near = fit_x
                    .iter()
                    .enumerate()
                    .min_by(|a, b| (a.1 - x0).abs().total_cmp(&(b.1 - x0).abs()))
                    .map(|e| e.0)
                    .expect("fitted genes");
                return fits[near];
            }
            GeneModel {
                beta0: b0 / w,
                beta1: b1 / w,
                theta: 10f64.powf(lt / w),
            }
        })
        .collect();
    (models, bw)
}

/// Gene `x`'s Pearson residuals over the cells at model `g` (`t = log10
/// m`, a cell with no counts left at 0), clipped at `+-sqrt(N)`.
pub fn residuals(x: GeneCounts, t: &[f64], g: &GeneModel) -> Vec<f64> {
    let n = t.len();
    let clip = (n as f64).sqrt();
    let mut z: Vec<f64> = t
        .iter()
        .map(|&ti| {
            if !ti.is_finite() {
                return 0.0;
            }
            let mu = (g.beta0 + g.beta1 * ti).exp();
            let sd = (mu + mu * mu / g.theta).sqrt();
            if sd > 0.0 {
                (-mu / sd).max(-clip)
            } else {
                0.0
            }
        })
        .collect();
    for (c, v) in x.iter() {
        let ti = t[c];
        if !ti.is_finite() {
            continue;
        }
        let mu = (g.beta0 + g.beta1 * ti).exp();
        let sd = (mu + mu * mu / g.theta).sqrt();
        z[c] = ((v - mu) / sd).clamp(-clip, clip);
    }
    z
}

/// What a run records of the transform.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SctReport {
    pub genes_fitted: usize,
    pub bandwidth: f64,
}

/// SCTransform's regularized models for the genes `wanted` of `genes`
/// (every gene's nonzero counts), cells' libraries `depths`: the models of
/// `wanted` in order, and the report.
pub fn sctransform(
    genes: &ByGene,
    depths: &Depths,
    wanted: &[usize],
    seed: u64,
) -> (Vec<GeneModel>, SctReport) {
    let n = depths.t.len();
    let gmean: Vec<f64> = (0..genes.n_genes())
        .into_par_iter()
        .map(|g| geometric_mean(genes.gene(g), n))
        .collect();
    let candidates: Vec<usize> = (0..genes.n_genes()).filter(|&g| gmean[g] > 0.0).collect();
    let log_gm: Vec<f64> = candidates.iter().map(|&g| gmean[g].log10()).collect();
    let picked: Vec<usize> = sample_genes(&log_gm, seed)
        .into_iter()
        .map(|i| candidates[i])
        .collect();
    let fits: Vec<GeneModel> = picked
        .par_iter()
        .map(|&g| fit_gene(genes.gene(g), depths))
        .collect();
    let fit_x: Vec<f64> = picked.iter().map(|&g| gmean[g].log10()).collect();
    let at: Vec<f64> = wanted
        .iter()
        .map(|&g| gmean[g].max(1e-300).log10())
        .collect();
    let (models, bandwidth) = regularize(&fit_x, &fits, &at);
    (
        models,
        SctReport {
            genes_fitted: picked.len(),
            bandwidth,
        },
    )
}
