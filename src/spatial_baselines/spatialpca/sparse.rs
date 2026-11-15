//! SpatialPCA's profile likelihood on the sparse prior (the default).
//!
//! The paper's model (Eq. 1, 2): `Y = (X B)^T + W Z + E`, `X = 1`, `W^T W =
//! I`, `E ~ N(0, sigma0^2)`, each spatial PC `Z_l ~ N(0, sigma0^2 tau K)`.
//! Here `K` is the sparse prior's `R(rho) = (I + rho L)^{-1} / cbar(rho)`.
//! With `Y` centred per gene and `L 1 = 0`, Supplementary Note 1's
//! profile (SN(5) to SN(10), `B` integrated out, `sigma0^2` and `W`
//! profiled) reduces exactly to
//!
//! - `t = tau / cbar(rho)`, `C = I + (I + rho L) / t`;
//! - `G = Y C^{-1} Y^T`, `W` its top `d` eigenvectors (SN(8)),
//!   `S = tr(Y Y^T) - sum_l lambda_l(G)`, `sigma0^2 = S / (m (n - 1))`
//!   (SN(6));
//! - `ell = -(m (n - 1) / 2) log S - (d / 2) [sum_i log((1 + t + rho
//!   mu_i) / (1 + rho mu_i)) - log(1 + t)]` (SN(7)), `mu_i` the
//!   eigenvalues of `L`;
//! - `Z_l = C^{-1} Y^T w_l` (SN(10)).
//!
//! `C^{-1}` is applied by conjugate gradients, `G`'s top eigenpairs by
//! subspace iteration on its products, the log-determinant by the prior's
//! spectrum; `(tau, rho)` maximise `ell` by a projected L-BFGS with its
//! analytic gradient (the envelope theorem: `dS = sum_l z_l^T dC z_l`).

use crate::spatial_baselines::spatial_factor::kit::{lbfgs_box, top_eigs, Dense, EigOpts};
use crate::spatial_baselines::spatial_factor::prep::DenseRows;
use crate::spatial_baselines::spatial_factor::prior::{cg_solve, Laplacian, Spectrum};
use std::cell::RefCell;

/// The box on `(log tau, log rho)`.
pub const LOG_TAU: (f64, f64) = (-9.210_340_371_976_184, 9.210_340_371_976_184);
pub const LOG_RHO: (f64, f64) = (-6.907_755_278_982_137, 9.210_340_371_976_184);

/// Extra vectors of the subspace iteration's block beyond the `d` wanted.
const EXTRA_BLOCK: usize = 10;
const EIG_MAX_ITER: usize = 300;

/// One evaluation of the profile at `(tau, rho)`.
#[derive(Debug, Clone)]
pub struct Eval {
    pub tau: f64,
    pub rho: f64,
    /// `tau / cbar(rho)`.
    pub t: f64,
    /// `S`, the residual sum of squares the profile's `sigma0^2` is from.
    pub s: f64,
    /// The bracketed log-determinant term of `ell`.
    pub h: f64,
    pub ell: f64,
    /// `d ell / d (log tau, log rho)`.
    pub grad: [f64; 2],
    /// Loadings, genes × d (from [`Profile::eval`] with the loadings fixed,
    /// empty: they are the profile's own; [`Profile::fit`] returns them).
    pub w: Dense,
    /// Spatial PCs, cells × d.
    pub z: Dense,
    /// `w_l^T G w_l`, the top eigenvalues of `G` (or, with the loadings
    /// fixed, what they explain).
    pub explained: Vec<f64>,
}

impl Eval {
    /// `sigma0^2 = S / (m (n - 1))`.
    pub fn sigma2(&self, m: usize, n: usize) -> f64 {
        self.s / (m as f64 * (n as f64 - 1.0))
    }
}

/// The profile likelihood of one fit: `d` spatial PCs of `op` on the prior
/// of `lap`, profiled over `W` or with `W` fixed (the `-pb` unravel).
pub struct Profile<'a> {
    pub op: &'a DenseRows,
    pub lap: &'a Laplacian,
    pub spec: &'a Spectrum,
    pub d: usize,
    pub cg_tol: f64,
    pub seed: u64,
    /// Fixed loadings and `Y^T W`.
    fixed: Option<(Dense, Dense)>,
    trace: f64,
    warm: RefCell<Option<Dense>>,
}

impl<'a> Profile<'a> {
    pub fn new(
        op: &'a DenseRows,
        lap: &'a Laplacian,
        spec: &'a Spectrum,
        d: usize,
        cg_tol: f64,
        seed: u64,
    ) -> Self {
        let trace = op.frob2(&lap.section, lap.sections.len());
        Profile {
            op,
            lap,
            spec,
            d: d.min(op.m),
            cg_tol,
            seed,
            fixed: None,
            trace,
            warm: RefCell::new(None),
        }
    }

    /// The same profile with the loadings `w` (genes × d) held fixed.
    pub fn with_loadings(mut self, w: Dense) -> Self {
        let u = self.op.apply_t(&w);
        self.d = w.ncols();
        self.fixed = Some((w, u));
        self
    }

    fn solve(&self, shift: &[f64], beta: f64, rhs: &Dense) -> Dense {
        cg_solve(self.lap, shift, beta, rhs, None, self.cg_tol).0
    }

    pub fn eval(&self, log_tau: f64, log_rho: f64) -> Eval {
        let (m, n) = (self.op.m, self.op.n);
        let (tau, rho) = (log_tau.exp(), log_rho.exp());
        let (cbar, dlog_cbar) = self.spec.cbar(rho, n);
        let t = tau / cbar;
        let shift = vec![1.0 + 1.0 / t; self.lap.sections.len()];
        let beta = rho / t;
        let profiled = match &self.fixed {
            Some(_) => None,
            None => {
                let block = (self.d + EXTRA_BLOCK).min(m);
                let o = EigOpts {
                    d: self.d,
                    block,
                    seed: self.seed,
                    tol: (10.0 * self.cg_tol).max(1e-10),
                    max_iter: EIG_MAX_ITER,
                };
                let warm = self.warm.borrow().clone();
                let (_, w, last) = top_eigs(
                    |v| {
                        self.op
                            .apply(&self.solve(&shift, beta, &self.op.apply_t(v)))
                    },
                    m,
                    warm.as_ref(),
                    &o,
                );
                *self.warm.borrow_mut() = Some(last);
                let u = self.op.apply_t(&w);
                Some((w, u))
            }
        };
        let u = match (&self.fixed, &profiled) {
            (Some((_, u)), _) | (None, Some((_, u))) => u,
            (None, None) => unreachable!("loadings fixed or profiled"),
        };
        let z = self.solve(&shift, beta, u);
        let d = self.d as f64;
        let explained: Vec<f64> = (0..self.d).map(|l| u.column(l).dot(&z.column(l))).collect();
        let s = (self.trace - explained.iter().sum::<f64>()).max(1e-300);
        let zz: f64 = z.norm_squared();
        let zlz: f64 = (0..self.d)
            .map(|l| self.lap.quad(z.column(l).as_slice()))
            .sum();
        let sp = self.spec;
        let h = sp.total(|q| ((1.0 + t + rho * q) / (1.0 + rho * q)).ln()) - t.ln_1p();
        let k = m as f64 * (n as f64 - 1.0) / 2.0;
        let ell = -k * s.ln() - d / 2.0 * h;
        let ds_dt = -(zz + rho * zlz) / (t * t);
        let ds_drho = zlz / t;
        let dh_dt = sp.total(|q| 1.0 / (1.0 + t + rho * q)) - 1.0 / (1.0 + t);
        let dh_drho = sp.total(|q| q / (1.0 + t + rho * q) - q / (1.0 + rho * q));
        let dl_dt = -k * ds_dt / s - d / 2.0 * dh_dt;
        let dl_drho = -k * ds_drho / s - d / 2.0 * dh_drho;
        let dt_dlog_rho = -t * dlog_cbar;
        Eval {
            tau,
            rho,
            t,
            s,
            h,
            ell,
            grad: [dl_dt * t, dl_drho * rho + dl_dt * dt_dlog_rho],
            w: profiled.map_or_else(|| Dense::zeros(0, 0), |(w, _)| w),
            z,
            explained,
        }
    }

    /// Maximise `ell` over the box [`LOG_TAU`] × [`LOG_RHO`]: the best of
    /// the 3 × 3 grid `tau, rho in {0.1, 1, 10}`, then the projected
    /// L-BFGS from it, at most `max_evals` evaluations. The fit at the
    /// maximum (its loadings' signs fixed so each column's largest entry is
    /// positive) and the evaluations used.
    pub fn fit(&self, max_evals: usize) -> (Eval, usize) {
        let start = [0.1f64, 1.0, 10.0];
        let mut best = (f64::NEG_INFINITY, [0.0, 0.0]);
        for &tau in &start {
            for &rho in &start {
                let e = self.eval(tau.ln(), rho.ln());
                if e.ell > best.0 {
                    best = (e.ell, [tau.ln(), rho.ln()]);
                }
            }
        }
        let (r, mut e) = lbfgs_box(
            |x| {
                let e = self.eval(x[0], x[1]);
                (-e.ell, vec![-e.grad[0], -e.grad[1]], e)
            },
            &best.1,
            &[LOG_TAU.0, LOG_RHO.0],
            &[LOG_TAU.1, LOG_RHO.1],
            max_evals,
            1e-6,
            1e-10,
        );
        match &self.fixed {
            Some((w, _)) => e.w = w.clone(),
            None => fix_signs(&mut e.w, &mut e.z),
        }
        let on_bound = (r.x[0] - LOG_TAU.0).abs() < 1e-9
            || (r.x[0] - LOG_TAU.1).abs() < 1e-9
            || (r.x[1] - LOG_RHO.0).abs() < 1e-9
            || (r.x[1] - LOG_RHO.1).abs() < 1e-9;
        if on_bound {
            log::warn!(
                "spatialpca: the maximum (tau {:.3e}, rho {:.3e}) is on the search box",
                e.tau,
                e.rho
            );
        }
        (e, 9 + r.evals)
    }
}

/// Each loading column's sign flipped, with its PC, so its largest-magnitude
/// entry is positive (the paper's rotation `R = I`, sign by convention).
pub fn fix_signs(w: &mut Dense, z: &mut Dense) {
    for l in 0..w.ncols() {
        let at = w.column(l).iamax();
        if w[(at, l)] < 0.0 {
            w.column_mut(l).neg_mut();
            z.column_mut(l).neg_mut();
        }
    }
}
