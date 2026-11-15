//! SpatialPCA as the paper formulates it (`--reference`): the Gaussian
//! kernel `K(s, s') = exp(-||s - s'||^2 / gamma)` of Eq. (2), one block per
//! section (the multi-sample extension), `gamma` by the paper's rule of
//! thumb, `tau` by Brent on the profile SN(7), the Woodbury forms SN(11) and
//! the low-rank approximation SN(12), SN(13) with the paper's rank rule.
//!
//! The kernel is never stored whole: each block's eigenpairs are taken
//! densely when the block has at most [`DENSE_KERNEL_MAX`] cells (as the
//! paper's small-data path), else by subspace iteration on products formed
//! in tiles; `K V` products are tiled the same way.

use crate::spatial_baselines::spatial_factor::kit::{
    grid_then_brent, sym_eig_desc, top_eigs, Dense, EigOpts,
};
use rayon::prelude::*;

/// Cells up to which the paper takes every eigenpair of `K` and picks the
/// rank by the 90% rule; above it, `r = 20` for the `tau` search and
/// `ceil(0.1 n)` for the PCs (Supplementary Note 1, after SN(13)).
pub const DENSE_KERNEL_MAX: usize = 5000;
/// Share of `tr K` the small-data rank explains.
const RANK_SHARE: f64 = 0.9;
/// The large-data rank of the `tau` search.
const LARGE_RANK: usize = 20;
/// Rows of a kernel tile.
const TILE: usize = 2048;
/// Points of the log-`tau` grid before Brent, on [1e-4, 1e4].
const TAU_GRID: usize = 41;
const BRENT_TOL: f64 = 1e-8;
const BRENT_MAX_EVAL: usize = 100;

/// Each section's coordinates, every axis centred and scaled to unit
/// standard deviation (an axis with none left centred).
pub fn standardised(xy: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = xy.len();
    if n == 0 {
        return Vec::new();
    }
    let dims = xy[0].len();
    let mut out = xy.to_vec();
    for j in 0..dims {
        let mean = xy.iter().map(|p| p[j]).sum::<f64>() / n as f64;
        let var = xy.iter().map(|p| (p[j] - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
        let sd = if var > 0.0 { var.sqrt() } else { 1.0 };
        out.iter_mut().for_each(|p| p[j] = (p[j] - mean) / sd);
    }
    out
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    // R's default (type 7)
    let h = (sorted.len() as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Silverman's rule of thumb (Silverman 1986, "Density Estimation for
/// Statistics and Data Analysis", Chapman and Hall, eq. 3.31):
/// `0.9 min(sd, IQR / 1.34) n^(-1/5)`.
pub fn silverman(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let iqr = quantile(&s, 0.75) - quantile(&s, 0.25);
    0.9 * sd.min(iqr / 1.34) * n.powf(-0.2)
}

/// Distinct values of `x` with their counts.
fn tallied(x: &[f64]) -> Vec<(f64, f64)> {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let mut out: Vec<(f64, f64)> = Vec::new();
    for v in s {
        match out.last_mut() {
            Some((u, c)) if *u == v => *c += 1.0,
            _ => out.push((v, 1.0)),
        }
    }
    out
}

/// `sum_i sum_j phi^(r)((x_i - x_j) / a)` over every ordered pair (the
/// diagonal included), `r` 4 or 6, from the tallied values.
fn pair_sum(t: &[(f64, f64)], a: f64, r: u32) -> f64 {
    let phi = |u: f64| (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let deriv = |u: f64| {
        let u2 = u * u;
        match r {
            4 => (u2 * u2 - 6.0 * u2 + 3.0) * phi(u),
            _ => (u2 * u2 * u2 - 15.0 * u2 * u2 + 45.0 * u2 - 15.0) * phi(u),
        }
    };
    t.par_iter()
        .enumerate()
        .map(|(i, &(xi, ci))| {
            let mut s = ci * ci * deriv(0.0);
            for &(xj, cj) in &t[i + 1..] {
                s += 2.0 * ci * cj * deriv((xi - xj) / a);
            }
            s
        })
        .sum()
}

/// Sheather and Jones' plug-in bandwidth, the solve-the-equation variant
/// (Sheather and Jones 1991, JRSS B 53:683,
/// doi:10.1111/j.2517-6161.1991.tb01857.x, section 3): `h = [R(K) / (n
/// S(alpha2(h)))]^(1/5)` with the Gaussian kernel, `alpha2(h) = 1.357
/// [S(a) / T(b)]^(1/7) h^(5/7)`, `a = 0.920 lambda n^(-1/7)`, `b = 0.912
/// lambda n^(-1/9)`, `lambda` the interquartile range, and the density
/// functionals `S`, `T` summed exactly over all pairs. 0 when the
/// interquartile range is 0 (as Silverman's rule then gives); Silverman's
/// rule when the functionals are not positive.
pub fn sheather_jones(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let lambda = quantile(&s, 0.75) - quantile(&s, 0.25);
    if lambda <= 0.0 || n < 2.0 {
        return 0.0;
    }
    let t = tallied(x);
    let nn = n * (n - 1.0);
    let s_hat = |a: f64| pair_sum(&t, a, 4) / (nn * a.powi(5));
    let t_hat = |b: f64| -pair_sum(&t, b, 6) / (nn * b.powi(7));
    let a = 0.920 * lambda * n.powf(-1.0 / 7.0);
    let b = 0.912 * lambda * n.powf(-1.0 / 9.0);
    let (sa, tb) = (s_hat(a), t_hat(b));
    if !(sa > 0.0 && tb > 0.0) {
        return silverman(x);
    }
    let rk = 1.0 / (2.0 * std::f64::consts::PI.sqrt());
    let ratio = 1.357 * (sa / tb).powf(1.0 / 7.0);
    // g(log h) = log h - log of the right-hand side, solved by the
    // Illinois false position on a bracket around Silverman's value
    let g = |lh: f64| {
        let h = lh.exp();
        let sv = s_hat(ratio * h.powf(5.0 / 7.0));
        if sv <= 0.0 {
            return f64::INFINITY;
        }
        lh - 0.2 * (rk / (n * sv)).ln()
    };
    let h0 = 1.06 * lambda / 1.34 * n.powf(-0.2);
    let (mut lo, mut hi) = (h0.ln() - 3.0, h0.ln() + 3.0);
    let (mut glo, mut ghi) = (g(lo), g(hi));
    let mut tries = 0;
    while glo.signum() == ghi.signum() && tries < 10 {
        lo -= 3.0;
        hi += 3.0;
        glo = g(lo);
        ghi = g(hi);
        tries += 1;
    }
    if !(glo.is_finite() && ghi.is_finite()) || glo.signum() == ghi.signum() {
        return silverman(x);
    }
    let mut side = 0;
    for _ in 0..100 {
        let mid = (lo * ghi - hi * glo) / (ghi - glo);
        let gm = g(mid);
        if gm.abs() < 1e-10 || (hi - lo).abs() < 1e-10 {
            return mid.exp();
        }
        if gm.signum() == glo.signum() {
            lo = mid;
            glo = gm;
            if side == -1 {
                ghi /= 2.0;
            }
            side = -1;
        } else {
            hi = mid;
            ghi = gm;
            if side == 1 {
                glo /= 2.0;
            }
            side = 1;
        }
    }
    (0.5 * (lo + hi)).exp()
}

/// The paper's bandwidth on a section of `n` cells: per gene, Silverman's
/// rule above [`DENSE_KERNEL_MAX`] cells and Sheather-Jones at or below,
/// on the gene's standardised expression; `gamma` the median over genes.
pub fn bandwidth(rows: &[Vec<f64>]) -> f64 {
    let mut h: Vec<f64> = rows
        .iter()
        .map(|x| {
            if x.len() > DENSE_KERNEL_MAX {
                silverman(x)
            } else {
                sheather_jones(x)
            }
        })
        .collect();
    h.sort_by(f64::total_cmp);
    let k = h.len();
    match k {
        0 => 0.0,
        _ if k % 2 == 1 => h[k / 2],
        _ => 0.5 * (h[k / 2 - 1] + h[k / 2]),
    }
}

/// A section's Gaussian kernel, its cells' standardised coordinates and
/// bandwidth.
pub struct KernelBlock {
    pub cells: Vec<usize>,
    pub xy: Vec<Vec<f64>>,
    pub gamma: f64,
}

impl KernelBlock {
    fn entry(&self, a: usize, b: usize) -> f64 {
        let d2: f64 = self.xy[a]
            .iter()
            .zip(&self.xy[b])
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        (-d2 / self.gamma).exp()
    }

    /// `K_s V` for `V` (section cells × k), in row tiles, no `n × n`
    /// matrix held.
    pub fn apply(&self, v: &Dense) -> Dense {
        let n = self.cells.len();
        let k = v.ncols();
        let tiles: Vec<Dense> = (0..n.div_ceil(TILE))
            .into_par_iter()
            .map(|t| {
                let rows = t * TILE..((t + 1) * TILE).min(n);
                let mut out = Dense::zeros(rows.len(), k);
                for (r, a) in rows.clone().enumerate() {
                    for b in 0..n {
                        let e = self.entry(a, b);
                        for j in 0..k {
                            out[(r, j)] += e * v[(b, j)];
                        }
                    }
                }
                out
            })
            .collect();
        let mut out = Dense::zeros(n, k);
        for (t, tile) in tiles.into_iter().enumerate() {
            out.rows_mut(t * TILE, tile.nrows()).copy_from(&tile);
        }
        out
    }

    fn dense(&self) -> Dense {
        let n = self.cells.len();
        Dense::from_fn(n, n, |a, b| self.entry(a, b))
    }

    /// The block's top `r` eigenpairs (all of them densely at or below
    /// [`DENSE_KERNEL_MAX`] cells).
    fn eigs(&self, r: usize, seed: u64) -> (Vec<f64>, Dense) {
        let n = self.cells.len();
        if n <= DENSE_KERNEL_MAX {
            let (vals, vecs) = sym_eig_desc(&self.dense());
            let r = r.min(n);
            return (vals[..r].to_vec(), vecs.columns(0, r).into_owned());
        }
        let r = r.min(n);
        let o = EigOpts {
            d: r,
            block: (r + 10).min(n),
            seed,
            tol: 1e-10,
            max_iter: 500,
        };
        let (vals, vecs, _) = top_eigs(|v| self.apply(v), n, None, &o);
        (vals, vecs)
    }
}

/// The kernel's eigenpairs over all blocks, sorted by eigenvalue: each
/// eigenvector as a column over all `n` cells, zero off its block.
pub struct Eigen {
    pub values: Vec<f64>,
    pub vectors: Dense,
}

/// The ranks the paper's rule gives on `n` cells for the union of block
/// eigenvalues `values` (largest first): `(r_est, r_z)`.
pub fn ranks(values: &[f64], n: usize, set: Option<usize>) -> (usize, usize) {
    if let Some(r) = set {
        let r = r.clamp(1, values.len());
        return (r, r);
    }
    if n > DENSE_KERNEL_MAX {
        let rz = (n as f64 * 0.1).ceil() as usize;
        return (LARGE_RANK.min(values.len()), rz.min(values.len()));
    }
    let total: f64 = values.iter().map(|v| v.max(0.0)).sum();
    let mut acc = 0.0;
    for (j, v) in values.iter().enumerate() {
        acc += v.max(0.0);
        if acc >= RANK_SHARE * total {
            return (j + 1, j + 1);
        }
    }
    (values.len(), values.len())
}

/// Eigenpairs of the block-diagonal kernel: each block's top `want`, then
/// merged.
pub fn kernel_eigen(blocks: &[KernelBlock], n: usize, want: usize, seed: u64) -> Eigen {
    let per: Vec<(Vec<f64>, Dense)> = blocks
        .iter()
        .enumerate()
        .map(|(s, b)| b.eigs(want, seed ^ s as u64))
        .collect();
    let mut all: Vec<(f64, usize, usize)> = per
        .iter()
        .enumerate()
        .flat_map(|(s, (vals, _))| vals.iter().enumerate().map(move |(j, &v)| (v, s, j)))
        .collect();
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut vectors = Dense::zeros(n, all.len());
    for (c, &(_, s, j)) in all.iter().enumerate() {
        for (a, &cell) in blocks[s].cells.iter().enumerate() {
            vectors[(cell, c)] = per[s].1[(a, j)];
        }
    }
    Eigen {
        values: all.iter().map(|e| e.0).collect(),
        vectors,
    }
}

/// `K V` on all `n` cells, block by block.
pub fn kernel_apply(blocks: &[KernelBlock], v: &Dense) -> Dense {
    let mut out = Dense::zeros(v.nrows(), v.ncols());
    for b in blocks {
        let local = Dense::from_fn(b.cells.len(), v.ncols(), |a, j| v[(b.cells[a], j)]);
        let kv = b.apply(&local);
        for (a, &c) in b.cells.iter().enumerate() {
            for j in 0..v.ncols() {
                out[(c, j)] = kv[(a, j)];
            }
        }
    }
    out
}

/// What the profile in `tau` reads, precomputed once: `Y Y^T`, `Y U`,
/// `U^T 1` and the kernel eigenvalues, all at the search's rank.
pub struct TauProfile {
    pub yy: Dense,
    pub ytil: Dense,
    pub u1: Vec<f64>,
    pub delta: Vec<f64>,
    pub m: usize,
    pub n: usize,
    pub d: usize,
}

/// One evaluation of the profile in `tau`.
#[derive(Debug, Clone)]
pub struct TauEval {
    pub tau: f64,
    pub s: f64,
    pub ell: f64,
    pub w: Dense,
    pub eigvals: Vec<f64>,
}

impl TauProfile {
    /// `Y` (genes × cells, centred per gene) and the kernel's top `r`
    /// eigenpairs.
    pub fn new(y: &Dense, eig: &Eigen, r: usize, d: usize) -> Self {
        let u = eig.vectors.columns(0, r);
        let ytil = y * u;
        let u1 = (0..r).map(|j| u.column(j).sum()).collect();
        TauProfile {
            yy: y * y.transpose(),
            ytil,
            u1,
            delta: eig.values[..r].iter().map(|v| v.max(0.0)).collect(),
            m: y.nrows(),
            n: y.ncols(),
            d: d.min(y.nrows()),
        }
    }

    /// SN(6) to SN(8) at `tau`, by SN(11) and SN(12) as printed:
    /// `F = I - U (tau D + I)^{-1} U^T`, `G = Y F Y^T + g g^T / q` with `g =
    /// Y F 1` and `q = n - 1^T F 1`, `S = tr(Y Y^T) - sum_{l<=d} lambda_l(G)`,
    /// and `ell = -(m (n - 1) / 2) log S - (d / 2) log|tau K + I| - (d / 2)
    /// log|1^T (tau K + I)^{-1} 1|`, the determinants on the kept
    /// eigenpairs (`|1^T U (I + tau D)^{-1} U^T 1|`, the printed form).
    pub fn eval(&self, tau: f64) -> TauEval {
        let f: Vec<f64> = self.delta.iter().map(|&d| 1.0 / (tau * d + 1.0)).collect();
        let r = f.len();
        let mut scaled = self.ytil.clone();
        for (j, &fj) in f.iter().enumerate() {
            scaled.column_mut(j).scale_mut(fj);
        }
        let mut g = &self.yy - &scaled * self.ytil.transpose();
        let fu: Vec<f64> = (0..r).map(|j| f[j] * self.u1[j]).collect();
        let q: f64 = (0..r).map(|j| self.u1[j] * fu[j]).sum();
        if q > 1e-300 {
            let gv = &self.ytil * nalgebra::DVector::from_vec(fu);
            g += &gv * gv.transpose() / q;
        }
        let (vals, vecs) = if self.m <= 1500 {
            sym_eig_desc(&g)
        } else {
            let o = EigOpts {
                d: self.d,
                block: self.d + 10,
                seed: 0,
                tol: 1e-12,
                max_iter: 500,
            };
            let (v, w, _) = top_eigs(|v| &g * v, self.m, None, &o);
            (v, w)
        };
        let top: f64 = vals[..self.d].iter().sum();
        let s = (self.yy.trace() - top).max(1e-300);
        let logdet: f64 = self.delta.iter().map(|&d| (tau * d).ln_1p()).sum();
        let xdet: f64 = (0..r)
            .map(|j| self.u1[j] * self.u1[j] / (1.0 + tau * self.delta[j]))
            .sum();
        let k = self.m as f64 * (self.n as f64 - 1.0) / 2.0;
        let d = self.d as f64;
        let ell = -k * s.ln() - d / 2.0 * logdet - d / 2.0 * xdet.max(1e-300).ln();
        TauEval {
            tau,
            s,
            ell,
            w: vecs.columns(0, self.d).into_owned(),
            eigvals: vals[..self.d].to_vec(),
        }
    }

    /// SN(9): `tau` by the 41-point log grid on `[1e-4, 1e4]`, then Brent on
    /// `log tau` in the best point's bracket. The maximum and the
    /// evaluations used.
    pub fn fit(&self) -> (TauEval, usize) {
        let grid: Vec<f64> = (0..TAU_GRID)
            .map(|i| (-4.0 + 8.0 * i as f64 / (TAU_GRID - 1) as f64) * std::f64::consts::LN_10)
            .collect();
        let (lt, _, evals) = grid_then_brent(
            |lt| -self.eval(lt.exp()).ell,
            &grid,
            BRENT_TOL,
            BRENT_MAX_EVAL + TAU_GRID,
        );
        if (lt - grid[0]).abs() < 1e-6 || (lt - grid[TAU_GRID - 1]).abs() < 1e-6 {
            log::warn!(
                "spatialpca --reference: tau {:.3e} is on the search's bound",
                lt.exp()
            );
        }
        (self.eval(lt.exp()), evals + 1)
    }
}

/// SN(13): `Z = tau W^T Y M K - tau W^T Y M U (tau^{-1} D^{-1} + U^T M U)^{-1}
/// U^T M K` with the top `r` eigenpairs, `K` applied in tiles; with `Y`
/// centred, `Y M = Y`. Cells × d.
pub fn pcs(y: &Dense, w: &Dense, tau: f64, eig: &Eigen, r: usize, blocks: &[KernelBlock]) -> Dense {
    let n = y.ncols();
    let a1t = y.transpose() * w; // n × d, (W^T Y)^T
    let ka = kernel_apply(blocks, &a1t);
    let u = eig.vectors.columns(0, r).into_owned();
    let delta: Vec<f64> = eig.values[..r].iter().map(|v| v.max(1e-12)).collect();
    let u1 = u.row_sum().transpose();
    let ones = Dense::from_element(n, 1, 1.0);
    let k1 = kernel_apply(blocks, &ones);
    // K M U = U D - (K 1)(1^T U) / n
    let mut kmu = u.clone();
    for (j, &dj) in delta.iter().enumerate() {
        kmu.column_mut(j).scale_mut(dj);
    }
    kmu -= &k1 * u1.transpose() / n as f64;
    let mut mid = Dense::identity(r, r) - &u1 * u1.transpose() / n as f64;
    for j in 0..r {
        mid[(j, j)] += 1.0 / (tau * delta[j]);
    }
    let yz = a1t.transpose() * &u; // d × r
    let inner = mid
        .lu()
        .solve(&yz.transpose())
        .expect("tau^{-1} D^{-1} + U^T M U is positive definite"); // r × d
    (ka - kmu * inner) * tau
}

/// The kernel blocks of the sections `sections` (cells per section) at
/// coordinates `xy` (cells × dims), each with its bandwidth from the
/// genes' rows of `y` (genes × cells) over its cells, or `gamma` when set.
/// Errors when a rule's median is 0.
pub fn blocks(
    y: &Dense,
    xy: &[Vec<f64>],
    sections: &[Vec<usize>],
    gamma: Option<f64>,
) -> anyhow::Result<Vec<KernelBlock>> {
    sections
        .iter()
        .filter(|c| !c.is_empty())
        .map(|cells| {
            let local: Vec<Vec<f64>> = cells.iter().map(|&c| xy[c].clone()).collect();
            let gamma = match gamma {
                Some(g) => g,
                None => {
                    let per: Vec<Vec<f64>> = (0..y.nrows())
                        .map(|g| cells.iter().map(|&c| y[(g, c)]).collect())
                        .collect();
                    bandwidth(&per)
                }
            };
            anyhow::ensure!(
                gamma > 0.0,
                "spatialpca --reference: the bandwidth rule gives 0 on a section of {} cells; \
                 set --bandwidth",
                cells.len()
            );
            Ok(KernelBlock {
                cells: cells.clone(),
                xy: standardised(&local),
                gamma,
            })
        })
        .collect()
}
