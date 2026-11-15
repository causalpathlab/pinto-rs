//! MEFISTO's factor prior on the sparse graph (the default): `z_k ~ N(0,
//! Q_k^{-1})`, `Q_k = c_k (I + rho_k L)`, `c_k = cbar(rho_k)`, so the mean
//! prior variance is one as `diag(Sigma_k) = 1` in SI (24), replacing the
//! squared-exponential kernel; sections independent a priori, sharing
//! `rho_k`. `rho_k = 0` is MEFISTO's unstructured factor (`l_k = 0`).
//!
//! With the likelihood's precision constant within a section, `A_k^{-1} =
//! S_k + Q_k` is `(s_g + c) I + c rho L_g` on section `g`, so every trace
//! and log-determinant is a spectral sum over `L_g` and the mean a
//! conjugate-gradient solve.

use super::super::spatial_factor::kit::grid_then_brent;
use super::super::spatial_factor::prior::{cg_solve, Laplacian, Spectrum};
use super::vb::FactorPrior;
use nalgebra::DMatrix;

/// The range of the `rho` search, over the mean degree.
const RHO_RANGE: (f64, f64) = (1e-2, 1e3);
const BRENT_TOL: f64 = 1e-6;
const BRENT_MAX_EVAL: usize = 60;

/// The graph prior of every factor.
pub struct GraphPrior<'a> {
    pub lap: &'a Laplacian,
    pub spec: &'a Spectrum,
    pub cg_tol: f64,
    /// Points of the log grid of the `rho` search.
    pub n_grid: usize,
    pub rho: Vec<f64>,
    pub c: Vec<f64>,
    /// The current `q(z_k)`'s precision: `(s_g, c, rho)` of `A_k^{-1}`.
    a: Vec<(Vec<f64>, f64, f64)>,
}

impl<'a> GraphPrior<'a> {
    /// `k` factors, each starting at `rho`.
    pub fn new(
        lap: &'a Laplacian,
        spec: &'a Spectrum,
        k: usize,
        rho: f64,
        cg_tol: f64,
        n_grid: usize,
    ) -> Self {
        let n = lap.n();
        let c = spec.cbar(rho, n).0;
        GraphPrior {
            lap,
            spec,
            cg_tol,
            n_grid,
            rho: vec![rho; k],
            c: vec![c; k],
            a: vec![(vec![0.0; lap.sections.len()], c, rho); k],
        }
    }

    /// The starting `rho`: the middle of the search's log range.
    pub fn start_rho(lap: &Laplacian) -> f64 {
        (RHO_RANGE.0 * RHO_RANGE.1).sqrt() / lap.mean_degree().max(1e-12)
    }

    /// `(tr A, tr L A, log det A)` of the current `q(z_k)`.
    fn traces(&self, k: usize) -> (f64, f64, f64) {
        let (s, c, rho) = &self.a[k];
        let mut t0 = 0.0;
        let mut t1 = 0.0;
        let mut ld = 0.0;
        for (g, &sg) in s.iter().enumerate() {
            let shift = sg + c;
            let beta = c * rho;
            t0 += self.spec.sum(g, |q| 1.0 / (shift + beta * q));
            t1 += self.spec.sum(g, |q| q / (shift + beta * q));
            ld -= self.spec.sum(g, |q| (shift + beta * q).ln());
        }
        (t0, t1, ld)
    }

    /// SI (40) with `Sigma^{-1} = Q(rho)`, `q` fixed: `0.5 [N log c(rho) +
    /// log det(I + rho L)] - 0.5 c(rho) [T0 + rho T1] - 0.5 c(rho) [mu^T mu +
    /// rho mu^T L mu]`.
    fn objective(&self, rho: f64, t0: f64, t1: f64, mm: f64, mlm: f64) -> f64 {
        let n = self.lap.n() as f64;
        let c = self.spec.cbar(rho, self.lap.n()).0;
        let ld = self.spec.total(|q| (rho * q).ln_1p());
        0.5 * (n * c.ln() + ld) - 0.5 * c * (t0 + rho * t1) - 0.5 * c * (mm + rho * mlm)
    }

    /// The smoothness the sparse prior reports in place of `1 - zeta_k`:
    /// the prior correlation between graph neighbours, `[tr(W M) / (2 sum
    /// w)] / [tr(M) / N]`, `M = (I + rho L)^{-1}`; `tr(D M)` by Hutchinson's
    /// estimator with `probes` solves, the rest by the spectrum.
    pub fn neighbour_correlation(&self, k: usize, probes: usize, seed: u64) -> f64 {
        let rho = self.rho[k];
        if rho == 0.0 {
            return 0.0;
        }
        let n = self.lap.n();
        let v = super::super::spatial_factor::kit::gaussian(n, probes.max(1), seed).map(|x| {
            if x >= 0.0 {
                1.0
            } else {
                -1.0
            }
        });
        let shift = vec![1.0; self.lap.sections.len()];
        let (mv, _) = cg_solve(self.lap, &shift, rho, &v, None, 1e-8);
        let mut tdm = 0.0;
        for j in 0..v.ncols() {
            for i in 0..n {
                tdm += v[(i, j)] * self.lap.degree[i] * mv[(i, j)];
            }
        }
        tdm /= v.ncols() as f64;
        let tlm = self.spec.total(|q| q / (1.0 + rho * q));
        let tm = self.spec.total(|q| 1.0 / (1.0 + rho * q));
        let w = self.lap.total_weight();
        if w == 0.0 {
            return 0.0;
        }
        ((tdm - tlm) / (2.0 * w)) / (tm / n as f64)
    }
}

impl FactorPrior for GraphPrior<'_> {
    fn update(&mut self, k: usize, s: &[f64], b: &[f64], mu: &mut [f64]) -> Vec<f64> {
        let (c, rho) = (self.c[k], self.rho[k]);
        let shift: Vec<f64> = s.iter().map(|&sg| sg + c).collect();
        let rhs = DMatrix::from_column_slice(b.len(), 1, b);
        let x0 = DMatrix::from_column_slice(mu.len(), 1, mu);
        let (x, _) = cg_solve(self.lap, &shift, c * rho, &rhs, Some(&x0), self.cg_tol);
        mu.copy_from_slice(x.as_slice());
        self.a[k] = (s.to_vec(), c, rho);
        shift
            .iter()
            .enumerate()
            .map(|(g, &sh)| self.spec.sum(g, |q| 1.0 / (sh + c * rho * q)))
            .collect()
    }

    fn kl_term(&self, k: usize, mu: &[f64]) -> f64 {
        let (t0, t1, ld_a) = self.traces(k);
        let (c, rho) = (self.c[k], self.rho[k]);
        let n = self.lap.n() as f64;
        let ld_q = n * c.ln() + self.spec.total(|q| (rho * q).ln_1p());
        let mm: f64 = mu.iter().map(|v| v * v).sum();
        let mlm = self.lap.quad(mu);
        0.5 * ld_q - 0.5 * c * (t0 + rho * t1) - 0.5 * c * (mm + rho * mlm) + 0.5 * ld_a + n / 2.0
    }

    fn hyper(&mut self, k: usize, mu: &mut [f64]) -> Option<Vec<f64>> {
        let (t0, t1, _) = self.traces(k);
        let mm: f64 = mu.iter().map(|v| v * v).sum();
        let mlm = self.lap.quad(mu);
        let dbar = self.lap.mean_degree().max(1e-12);
        let (lo, hi) = ((RHO_RANGE.0 / dbar).ln(), (RHO_RANGE.1 / dbar).ln());
        let p = self.n_grid.max(2);
        let grid: Vec<f64> = (0..p)
            .map(|i| lo + (hi - lo) * i as f64 / (p - 1) as f64)
            .collect();
        let (lr, f, _) = grid_then_brent(
            |lr| -self.objective(lr.exp(), t0, t1, mm, mlm),
            &grid,
            BRENT_TOL,
            BRENT_MAX_EVAL + p,
        );
        let at_zero = -self.objective(0.0, t0, t1, mm, mlm);
        let rho = if at_zero < f { 0.0 } else { lr.exp() };
        self.rho[k] = rho;
        self.c[k] = self.spec.cbar(rho, self.lap.n()).0;
        None
    }

    fn report(&self) -> serde_json::Value {
        serde_json::json!({ "rho": self.rho, "cbar": self.c })
    }
}
