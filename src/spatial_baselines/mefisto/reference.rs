//! MEFISTO's own factor prior (`--reference`): each factor a Gaussian
//! process over the coordinates and the groups (SI (9)-(13), (24)),
//!
//! `Sigma_k[n, n'] = (1 - zeta_k) exp(-||c_n - c_n'||^2 / (2 l_k^2))
//! K^G_k[g(n), g(n')] + zeta_k [n = n']`,
//!
//! `K^G_k` the correlation matrix of `x_k x_k^T + sigma_k^2 I` (rank
//! `R`), on `M` inducing cells (SI section 4): `p(z_k | u_k)` the prior's
//! conditional, `q(u_k) = N(nu_k, B_k)` (SI (79)-(83)), so `E[z] = Sigma_ZU
//! Sigma_UU^{-1} nu` and `Var(z_n)` the conditional variance plus `B`'s
//! share. With every cell an inducing point (`N <= --n-inducing`) this is
//! the full GP of SI section 3 exactly (`Sigma_ZU Sigma_UU^{-1} = I`, no
//! conditional variance). `Sigma_ZU` is formed in row blocks, never whole
//! (no `N × N` matrix).
//!
//! The unstructured factor (`l_k = 0`, or `zeta_k = 1`) is the paper's
//! ordinary MOFA factor: `z_n ~ N(0, 1)` independently on every cell, its
//! `q(z_n)` per cell, not through the inducing cells (whose projection
//! would give the others no signal at all).
//!
//! Hyperparameters (SI section 3.2.2, (84)): per factor, the lengthscale on
//! a grid from half the smallest to twice the largest pairwise distance,
//! plus 0; at each, `zeta`, `sigma`, `x` maximise the factor's part of the
//! ELBO with `q(u)` fixed by a box-constrained quasi-Newton search, from
//! their current values. With fewer inducing cells than cells that part is
//! the sparse-GP bound (SI section 4; Titsias 2009, AISTATS 5:567): the
//! likelihood's expected terms move with the hyperparameters through
//! `Sigma_ZU Sigma_UU^{-1}` and the conditional variance, as well as the
//! prior's `-KL(q(u) || p(u))`; every cell inducing, only the latter moves.
//! With one group (one section) only `zeta` is free and `Sigma_UU = V
//! ((1 - zeta) Lambda + zeta I) V^T` in each lengthscale's eigenbasis
//! (built once per lengthscale and kept), so the search is a scalar one on
//! `zeta` (a log grid then Brent) at `O(M^2)` an evaluation; the general
//! quasi-Newton search serves several groups.
//! The best structured candidate of the grid (its search starts from the
//! current hyperparameters, so it is no worse than them with `q(u)`
//! fixed) is then given its optimal `q(u)` (one update) and compared with
//! the unstructured factor at its optimal `q(z)`, both scored with `q` at
//! its optimum; the better is kept, so the step never lowers the ELBO and
//! a factor left unstructured can take a lengthscale again. Then the
//! identifiability rule (`l = 0` or `1 - zeta = 0` gives the unstructured
//! factor).

use super::super::spatial_factor::kit::{grid_then_brent, lbfgs_box, sym_eig_desc, Dense};
use super::vb::FactorPrior;
use nalgebra::{Cholesky, DVector, Dyn};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::OnceLock;

const ROW_BLOCK: usize = 4096;
const JITTER: f64 = 1e-8;
const ZETA_MIN: f64 = 1e-4;
const SIGMA_MIN: f64 = 1e-4;
const HYPER_MAX_EVALS: usize = 100;
const HYPER_PGTOL: f64 = 1e-5;
/// Points of the one-group search's log grid on `zeta`, before Brent.
const ZETA_GRID: usize = 21;
const ZETA_BRENT_TOL: f64 = 1e-8;
const ZETA_MAX_EVAL: usize = 100;

/// One factor's kernel hyperparameters.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hyper {
    pub lengthscale: f64,
    pub zeta: f64,
    /// Group factors, groups × rank, row-major.
    pub x: Vec<f64>,
    pub sigma: f64,
}

impl Hyper {
    /// The unstructured factor, `x` of length `len`.
    fn unstructured(len: usize) -> Self {
        Hyper {
            lengthscale: 0.0,
            zeta: 1.0,
            x: vec![0.0; len],
            sigma: 1.0,
        }
    }

    /// `l = 0` or `1 - zeta = 0` (the identifiability rule).
    pub fn is_unstructured(&self) -> bool {
        self.lengthscale == 0.0 || 1.0 - self.zeta <= 1e-12
    }
}

/// A factor's variational posterior.
#[derive(Debug, Clone)]
enum Posterior {
    /// `q(u) = N(nu, B)` and `log det B`.
    Inducing {
        nu: DVector<f64>,
        b: Dense,
        ld_b: f64,
    },
    /// The unstructured factor: `q(z_n) = N(E[z_n], var[g(n)])`.
    Cells { var: Vec<f64> },
}

/// The likelihood's sums a lengthscale's search reads, for fewer inducing
/// cells than cells, `e_n` the squared-exponential row of cell `n` over
/// the inducing cells: per group `h`, `s_h sum_{n in h} e_n e_n^T` and
/// `sum_{n in h} b_n e_n`; `b` and `s` at the inducing cells; `sum_g s_g
/// N_g`.
struct LikSums {
    q: Vec<Dense>,
    r: Vec<DVector<f64>>,
    b_u: DVector<f64>,
    s_u: Vec<f64>,
    s_total: f64,
}

/// What a search at one lengthscale holds fixed: `q(u)`, the inducing
/// cells' squared-exponential kernel, and the likelihood's terms (a
/// constant with every cell inducing).
struct Search<'a> {
    nu: &'a DVector<f64>,
    b: &'a Dense,
    ld_b: f64,
    e_uu: Dense,
    lik: Result<LikSums, f64>,
}

/// One lengthscale's squared-exponential kernel on the inducing cells in
/// its eigenbasis, `E_UU = V diag(lambda) V^T`, and with fewer inducing
/// cells than cells `V^T E_UZ E_ZU V` (both fixed by the lengthscale).
struct Basis {
    l: f64,
    lambda: Vec<f64>,
    v: Dense,
    q: Option<Dense>,
}

/// A one-group search at one lengthscale in the kernel's eigenbasis
/// ([`GpPrior::fast_search`]): with one group `K^G = 1` and `Sigma_UU =
/// V ((1 - zeta) Lambda + zeta I) V^T`, so the objective of
/// [`GpPrior::objective`] in `zeta` costs `O(M^2)` from `nu`, `B`, `P`'s
/// parts and `c`'s parts rotated once into the basis.
struct FastSearch<'a> {
    lambda: &'a [f64],
    /// `V^T nu`, `V^T B V`, `log det B`.
    nu: DVector<f64>,
    b: Dense,
    ld_b: f64,
    lik: Result<FastLik<'a>, f64>,
}

/// The likelihood's parts in the basis: `s V^T E_UZ E_ZU V` (as `s` and
/// the basis' `q`), `V^T E_UZ b`, `V^T b_U`, and `sum_n s_n`.
struct FastLik<'a> {
    q: &'a Dense,
    s: f64,
    c1: DVector<f64>,
    b_u: DVector<f64>,
    s_total: f64,
}

impl FastSearch<'_> {
    /// [`GpPrior::objective`]'s value at `zeta`, one group: `P = (1 -
    /// zeta)^2 s E_UZ E_ZU + 2 zeta (1 - zeta) s E_UU + zeta^2 s I`, `c =
    /// (1 - zeta) E_UZ b + zeta b_U`, every term a sum over the basis.
    fn eval(&self, zeta: f64) -> f64 {
        let m = self.lambda.len();
        let d: Vec<f64> = self
            .lambda
            .iter()
            .map(|&l| (1.0 - zeta) * l + zeta)
            .collect();
        let mut f = 0.5 * self.ld_b + 0.5 * m as f64;
        for i in 0..m {
            f -= 0.5 * (d[i].ln() + (self.b[(i, i)] + self.nu[i] * self.nu[i]) / d[i]);
        }
        match &self.lik {
            Err(v) => f + v,
            Ok(l) => {
                let a: Vec<f64> = (0..m).map(|i| self.nu[i] / d[i]).collect();
                let w2 = (1.0 - zeta).powi(2) * l.s;
                // P's diagonal part
                let pd: Vec<f64> = (0..m)
                    .map(|i| l.s * zeta * (2.0 * (1.0 - zeta) * self.lambda[i] + zeta))
                    .collect();
                let (mut apa, mut trxp, mut trxbxp, mut ca) = (0.0, 0.0, 0.0, 0.0);
                for i in 0..m {
                    ca += ((1.0 - zeta) * l.c1[i] + zeta * l.b_u[i]) * a[i];
                    let pii = w2 * l.q[(i, i)] + pd[i];
                    trxp += pii / d[i];
                    apa += pd[i] * a[i] * a[i];
                    trxbxp += self.b[(i, i)] * pd[i] / (d[i] * d[i]);
                    let (mut row_a, mut row_b) = (0.0, 0.0);
                    for j in 0..m {
                        let q = l.q[(j, i)];
                        row_a += q * a[j];
                        row_b += self.b[(i, j)] * q / d[j];
                    }
                    apa += w2 * a[i] * row_a;
                    trxbxp += w2 * row_b / d[i];
                }
                f + ca - 0.5 * apa - 0.5 * l.s_total + 0.5 * trxp - 0.5 * trxbxp
            }
        }
    }
}

/// The GP prior of every factor.
pub struct GpPrior {
    xy: Vec<Vec<f64>>,
    group: Vec<usize>,
    n_groups: usize,
    /// Cells of each group.
    members: Vec<Vec<usize>>,
    rank: usize,
    /// The inducing cells.
    pub inducing: Vec<usize>,
    /// The lengthscale grid (0 first).
    pub grid: Vec<f64>,
    pub hyper: Vec<Hyper>,
    q: Vec<Posterior>,
    /// One group's lengthscale bases, by grid position, built at first use.
    bases: Vec<OnceLock<Basis>>,
    /// One group's search in the eigenbasis (else the general search);
    /// tests turn it off to compare.
    pub fast: bool,
}

fn d2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// The group correlation `K^G` of `x x^T + sigma^2 I`.
fn group_kernel(x: &[f64], sigma: f64, g: usize, r: usize) -> Dense {
    let kt = group_cov(x, sigma, g, r);
    Dense::from_fn(g, g, |a, b| kt[(a, b)] / (kt[(a, a)] * kt[(b, b)]).sqrt())
}

/// `x x^T + sigma^2 I`.
fn group_cov(x: &[f64], sigma: f64, g: usize, r: usize) -> Dense {
    Dense::from_fn(g, g, |a, b| {
        let mut v: f64 = (0..r).map(|j| x[a * r + j] * x[b * r + j]).sum();
        if a == b {
            v += sigma * sigma;
        }
        v
    })
}

/// `sum_ij a_ij b_ij`.
fn inner(a: &Dense, b: &Dense) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// The sizes of a [`GpPrior`].
#[derive(Debug, Clone, Copy)]
pub struct GpSizes {
    /// Factors.
    pub k: usize,
    /// Inducing cells at most.
    pub n_inducing: usize,
    /// Points of the lengthscale grid, 0 aside.
    pub n_grid: usize,
    /// Rank of the group kernel.
    pub rank: usize,
    pub seed: u64,
}

impl GpPrior {
    /// The prior on cells at `xy` in groups `group`, with at most
    /// `n_inducing` inducing cells (placed as [`inducing_cells`]), `k`
    /// factors, a lengthscale grid of `n_grid` points and group rank
    /// `rank`.
    pub fn new(xy: Vec<Vec<f64>>, group: Vec<usize>, n_groups: usize, z: GpSizes) -> Self {
        let GpSizes {
            k,
            n_inducing,
            n_grid,
            rank,
            seed,
        } = z;
        let inducing = inducing_cells(&xy, &group, n_groups, n_inducing, seed);
        let (lo, hi) = distance_range(&xy);
        let p = n_grid.max(1);
        let mut grid = vec![0.0];
        grid.extend((0..p).map(|i| {
            if p == 1 {
                0.5 * lo
            } else {
                0.5 * lo + (2.0 * hi - 0.5 * lo) * i as f64 / (p - 1) as f64
            }
        }));
        // the start: the median grid value, zeta 0.5, x 1/sqrt(G), sigma 0.5
        let start = Hyper {
            lengthscale: grid[1 + p / 2 - usize::from(p.is_multiple_of(2) && p > 1)],
            zeta: 0.5,
            x: vec![1.0 / (n_groups as f64).sqrt(); n_groups * rank],
            sigma: 0.5,
        };
        let m = inducing.len();
        let mut members = vec![Vec::new(); n_groups];
        for (c, &g) in group.iter().enumerate() {
            members[g].push(c);
        }
        GpPrior {
            xy,
            group,
            n_groups,
            members,
            rank,
            inducing,
            bases: (0..grid.len()).map(|_| OnceLock::new()).collect(),
            grid,
            fast: true,
            hyper: vec![start; k],
            q: vec![
                Posterior::Inducing {
                    nu: DVector::zeros(m),
                    b: Dense::identity(m, m),
                    ld_b: 0.0,
                };
                k
            ],
        }
    }

    fn every_cell_inducing(&self) -> bool {
        self.inducing.len() == self.xy.len()
    }

    fn entry(&self, h: &Hyper, kg: &Dense, a: usize, b: usize) -> f64 {
        let same = if a == b { h.zeta } else { 0.0 };
        let e = (-d2(&self.xy[a], &self.xy[b]) / (2.0 * h.lengthscale * h.lengthscale)).exp();
        (1.0 - h.zeta) * e * kg[(self.group[a], self.group[b])] + same
    }

    fn kg(&self, h: &Hyper) -> Dense {
        group_kernel(&h.x, h.sigma, self.n_groups, self.rank)
    }

    /// `Sigma_UU`.
    fn suu(&self, h: &Hyper) -> Dense {
        let kg = self.kg(h);
        let u = &self.inducing;
        Dense::from_fn(u.len(), u.len(), |a, b| self.entry(h, &kg, u[a], u[b]))
    }

    /// Rows `rows` of `Sigma_ZU`.
    fn szu(&self, h: &Hyper, kg: &Dense, rows: std::ops::Range<usize>) -> Dense {
        let u = &self.inducing;
        let start = rows.start;
        Dense::from_fn(rows.len(), u.len(), |r, b| {
            self.entry(h, kg, start + r, u[b])
        })
    }

    /// The squared-exponential rows of `cells` over the inducing cells at
    /// lengthscale `l`.
    fn exp_rows(&self, l: f64, cells: &[usize]) -> Dense {
        let u = &self.inducing;
        Dense::from_fn(cells.len(), u.len(), |r, b| {
            (-d2(&self.xy[cells[r]], &self.xy[u[b]]) / (2.0 * l * l)).exp()
        })
    }

    fn chol(m: Dense) -> Cholesky<f64, Dyn> {
        let n = m.nrows();
        let mut jitter = 0.0;
        loop {
            let a = &m + Dense::identity(n, n) * jitter;
            if let Some(c) = a.cholesky() {
                return c;
            }
            jitter = if jitter == 0.0 { JITTER } else { jitter * 10.0 };
        }
    }

    fn logdet(c: &Cholesky<f64, Dyn>) -> f64 {
        2.0 * c.l().diagonal().iter().map(|v| v.ln()).sum::<f64>()
    }

    /// The unstructured factor's `q` at its optimum given `s` and `b`:
    /// `E[z_n] = b_n / (1 + s_g)` into `mu`, `var_g = 1 / (1 + s_g)`; the
    /// variances and their per-group sums.
    fn cells_update(&self, s: &[f64], b: &[f64], mu: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
        let var: Vec<f64> = s.iter().map(|&sg| 1.0 / (1.0 + sg)).collect();
        for (c, m) in mu.iter_mut().enumerate() {
            *m = b[c] * var[self.group[c]];
        }
        let tr = (0..self.n_groups)
            .map(|g| self.members[g].len() as f64 * var[g])
            .collect();
        (var, tr)
    }

    /// Factor `k`'s part of the ELBO at its current `q` (means `mu`,
    /// per-group variance sums `tr`) given the likelihood's `s` and `b`:
    /// `b . E[z] - sum_n s_g E[z_n^2] / 2 + E[log p(z)] - E[log q(z)]`.
    fn factor_value(&self, k: usize, s: &[f64], b: &[f64], mu: &[f64], tr: &[f64]) -> f64 {
        let lin: f64 = (0..mu.len())
            .map(|c| b[c] * mu[c] - 0.5 * s[self.group[c]] * mu[c] * mu[c])
            .sum();
        let var: f64 = tr.iter().zip(s).map(|(t, sg)| 0.5 * sg * t).sum();
        lin - var + self.kl_term(k, mu)
    }

    /// The unstructured factor's part of the ELBO at its optimum:
    /// `sum_n [b_n^2 / (1 + s_g) - log(1 + s_g)] / 2`.
    fn cells_optimum(&self, s: &[f64], b: &[f64]) -> f64 {
        let lin: f64 = (0..b.len())
            .map(|c| b[c] * b[c] / (1.0 + s[self.group[c]]))
            .sum();
        let ld: f64 = (0..self.n_groups)
            .map(|g| self.members[g].len() as f64 * s[g].ln_1p())
            .sum();
        0.5 * (lin - ld)
    }

    /// Each cell's `E[z]` and the per-group sums of `Var(z)` from `q(u)`:
    /// `E[z_n] = k_n^T Sigma_UU^{-1} nu`, `Var(z_n) = 1 - k_n^T
    /// Sigma_UU^{-1} k_n + k_n^T Sigma_UU^{-1} B Sigma_UU^{-1} k_n`.
    fn moments(&self, k: usize, mu: &mut [f64]) -> Vec<f64> {
        let h = &self.hyper[k];
        let Posterior::Inducing { nu, b, .. } = &self.q[k] else {
            unreachable!("moments of a factor on the inducing cells")
        };
        let kg = self.kg(h);
        let cu = Self::chol(self.suu(h));
        let alpha = cu.solve(nu);
        let mid = cu.solve(&cu.solve(b).transpose());
        let n = self.xy.len();
        let mut tr = vec![0.0; self.n_groups];
        for start in (0..n).step_by(ROW_BLOCK) {
            let rows = start..(start + ROW_BLOCK).min(n);
            let kz = self.szu(h, &kg, rows.clone());
            let m = &kz * &alpha;
            let si_k = cu.solve(&kz.transpose());
            let mid_k = &mid * kz.transpose();
            for (r, c) in rows.enumerate() {
                mu[c] = m[r];
                let v = 1.0 - kz.row(r).dot(&si_k.column(r).transpose())
                    + kz.row(r).dot(&mid_k.column(r).transpose());
                tr[self.group[c]] += v.max(0.0);
            }
        }
        tr
    }

    /// Lengthscale `l`'s [`Basis`].
    fn basis_at(&self, l: f64) -> Basis {
        let (lambda, v) = sym_eig_desc(&self.exp_rows(l, &self.inducing));
        let q = (!self.every_cell_inducing()).then(|| {
            let m = self.inducing.len();
            let cells: Vec<usize> = (0..self.xy.len()).collect();
            let mut q = Dense::zeros(m, m);
            for chunk in cells.chunks(ROW_BLOCK) {
                let w = self.exp_rows(l, chunk) * &v;
                q.gemm_tr(1.0, &w, &w, 1.0);
            }
            q
        });
        Basis { l, lambda, v, q }
    }

    /// The fixed part of a one-group search at `basis`' lengthscale with
    /// `q(u) = (nu, B)`, given the likelihood's `s` and `b`.
    fn fast_search<'a>(
        &self,
        basis: &'a Basis,
        s: &[f64],
        b: &[f64],
        nu: &DVector<f64>,
        bq: &Dense,
        ld_b: f64,
    ) -> FastSearch<'a> {
        let v = &basis.v;
        let lik = match &basis.q {
            None => Err((0..nu.len())
                .map(|c| b[c] * nu[c] - 0.5 * s[0] * (nu[c] * nu[c] + bq[(c, c)]))
                .sum()),
            Some(q) => {
                let m = self.inducing.len();
                let cells: Vec<usize> = (0..self.xy.len()).collect();
                let mut r = DVector::zeros(m);
                for chunk in cells.chunks(ROW_BLOCK) {
                    let bc = DVector::from_iterator(chunk.len(), chunk.iter().map(|&c| b[c]));
                    r.gemv_tr(1.0, &self.exp_rows(basis.l, chunk), &bc, 1.0);
                }
                let b_u = DVector::from_iterator(m, self.inducing.iter().map(|&c| b[c]));
                Ok(FastLik {
                    q,
                    s: s[0],
                    c1: v.tr_mul(&r),
                    b_u: v.tr_mul(&b_u),
                    s_total: s[0] * self.xy.len() as f64,
                })
            }
        };
        FastSearch {
            lambda: &basis.lambda,
            nu: v.tr_mul(nu),
            b: v.tr_mul(&(bq * v)),
            ld_b,
            lik,
        }
    }

    /// `zeta` maximising a one-group search's objective: a log grid on
    /// `[ZETA_MIN, 1]` with the start `zeta0` added, then Brent in the
    /// best point's bracket. The `zeta` and its value.
    fn fast_zeta(x: &FastSearch, zeta0: f64) -> (f64, f64) {
        let (lo, hi) = (ZETA_MIN.ln(), 0.0);
        let mut grid: Vec<f64> = (0..ZETA_GRID)
            .map(|i| lo + (hi - lo) * i as f64 / (ZETA_GRID - 1) as f64)
            .collect();
        grid.push(zeta0.clamp(ZETA_MIN, 1.0).ln());
        grid.sort_by(f64::total_cmp);
        grid.dedup();
        let (lz, f, _) = grid_then_brent(
            |lz| -x.eval(lz.exp().min(1.0)),
            &grid,
            ZETA_BRENT_TOL,
            ZETA_MAX_EVAL + grid.len(),
        );
        (lz.exp().min(1.0), -f)
    }

    /// The fixed part of a search at lengthscale `l > 0` with `q(u) = (nu,
    /// B)`, given the likelihood's `s` and `b`.
    fn search<'a>(
        &self,
        l: f64,
        s: &[f64],
        b: &[f64],
        nu: &'a DVector<f64>,
        bq: &'a Dense,
        ld_b: f64,
    ) -> Search<'a> {
        let e_uu = self.exp_rows(l, &self.inducing);
        let lik = if self.every_cell_inducing() {
            // E[z] = nu, Var(z_n) = B_nn whatever the hyperparameters
            let v: f64 = (0..nu.len())
                .map(|c| {
                    let sc = s[self.group[c]];
                    b[c] * nu[c] - 0.5 * sc * (nu[c] * nu[c] + bq[(c, c)])
                })
                .sum();
            Err(v)
        } else {
            let m = self.inducing.len();
            let mut q = vec![Dense::zeros(m, m); self.n_groups];
            let mut r = vec![DVector::zeros(m); self.n_groups];
            for (g, cells) in self.members.iter().enumerate() {
                for chunk in cells.chunks(ROW_BLOCK) {
                    let e = self.exp_rows(l, chunk);
                    q[g].gemm_tr(1.0, &e, &e, 1.0);
                    let bc = DVector::from_iterator(chunk.len(), chunk.iter().map(|&c| b[c]));
                    r[g].gemv_tr(1.0, &e, &bc, 1.0);
                }
                q[g] *= s[g];
            }
            let s_total = (0..self.n_groups)
                .map(|g| s[g] * self.members[g].len() as f64)
                .sum();
            Ok(LikSums {
                q,
                r,
                b_u: DVector::from_iterator(m, self.inducing.iter().map(|&c| b[c])),
                s_u: self.inducing.iter().map(|&c| s[self.group[c]]).collect(),
                s_total,
            })
        };
        Search {
            nu,
            b: bq,
            ld_b,
            e_uu,
            lik,
        }
    }

    /// The factor's part of the ELBO at hyperparameters `h` (lengthscale
    /// `> 0`, the search's) with `q(u)` fixed, and its gradient in `(zeta,
    /// sigma, x)`. With `X = Sigma_UU^{-1}`, `alpha = X nu`, `P = Sigma_UZ
    /// S Sigma_ZU`, `c = Sigma_UZ b`:
    ///
    /// `c^T alpha - alpha^T P alpha / 2 - sum_n s_n / 2 + tr(X P) / 2 -
    /// tr(X B X P) / 2` (the likelihood's expected terms, from `E[z] =
    /// Sigma_ZU alpha` and `Var(z_n)`) `- log det Sigma_UU / 2 - tr(X B) /
    /// 2 - nu^T alpha / 2 + log det B / 2 + M / 2` (`-KL(q(u) || p(u))`).
    /// `P` and `c` from the search's sums: `Sigma_ZU = (1 - zeta) E o K^G
    /// + zeta I_ZU`.
    fn objective(&self, x: &Search, h: &Hyper, grad: bool) -> (f64, Vec<f64>) {
        let u = &self.inducing;
        let m = u.len();
        let (g, r) = (self.n_groups, self.rank);
        let zeta = h.zeta;
        let kg = self.kg(h);
        let gu: Vec<usize> = u.iter().map(|&c| self.group[c]).collect();
        let ek = Dense::from_fn(m, m, |a, b| x.e_uu[(a, b)] * kg[(gu[a], gu[b])]);
        let sigma = &ek * (1.0 - zeta) + Dense::identity(m, m) * zeta;
        let c = Self::chol(sigma);
        let xi = c.inverse();
        let alpha = &xi * x.nu;
        let xb = &xi * x.b;
        let mut f = -0.5 * Self::logdet(&c) - 0.5 * xb.trace() - 0.5 * x.nu.dot(&alpha)
            + 0.5 * x.ld_b
            + 0.5 * m as f64;
        // P = (1 - zeta)^2 P2 + zeta (1 - zeta) P1 + zeta^2 P0, c = (1 -
        // zeta) c1 + zeta b_U
        let lik = match &x.lik {
            Err(v) => {
                f += v;
                None
            }
            Ok(ls) => {
                let kh = |hh: usize| -> Vec<f64> { gu.iter().map(|&gb| kg[(hh, gb)]).collect() };
                let mut p2 = Dense::zeros(m, m);
                let mut c1 = DVector::zeros(m);
                for hh in 0..g {
                    let k = kh(hh);
                    for a in 0..m {
                        c1[a] += k[a] * ls.r[hh][a];
                        for b in 0..m {
                            p2[(a, b)] += k[a] * k[b] * ls.q[hh][(a, b)];
                        }
                    }
                }
                let p1 = Dense::from_fn(m, m, |a, b| (ls.s_u[a] + ls.s_u[b]) * ek[(a, b)]);
                let mut p = &p2 * (1.0 - zeta).powi(2) + &p1 * (zeta * (1.0 - zeta));
                for a in 0..m {
                    p[(a, a)] += zeta * zeta * ls.s_u[a];
                }
                let cv = &c1 * (1.0 - zeta) + &ls.b_u * zeta;
                let xp = &xi * &p;
                let pa = &p * &alpha;
                f += cv.dot(&alpha) - 0.5 * alpha.dot(&pa) - 0.5 * ls.s_total + 0.5 * xp.trace()
                    - 0.5 * inner(&xb, &xp.transpose());
                Some((ls, p2, p1, cv, c1, xp, pa))
            }
        };
        if !grad {
            return (f, Vec::new());
        }
        // dF = <G_S, dSigma_UU> + <G_P, dP> + alpha . dc
        let xbx = &xb * &xi;
        let aat = &alpha * alpha.transpose();
        let mut gs = (&xbx + &aat - &xi) * 0.5;
        let mut hmat = Dense::zeros(g, g);
        let mut dzeta = 0.0;
        if let Some((ls, p2, p1, cv, c1, xp, pa)) = &lik {
            let xr = &xi * (cv - pa);
            let xpx = xp * &xi;
            let w = &xbx * xp.transpose();
            gs += (&xr * alpha.transpose() + &alpha * xr.transpose()) * -0.5;
            gs += (&w + w.transpose() - xpx) * 0.5;
            let gp = (&xi - &xbx - &aat) * 0.5;
            let mut dp = p2 * (-2.0 * (1.0 - zeta)) + p1 * (1.0 - 2.0 * zeta);
            for a in 0..m {
                dp[(a, a)] += 2.0 * zeta * ls.s_u[a];
            }
            dzeta += inner(&gp, &dp) + alpha.dot(&(&ls.b_u - c1));
            for a in 0..m {
                for b in 0..m {
                    let (ga, gb) = (gu[a], gu[b]);
                    hmat[(ga, gb)] +=
                        zeta * (1.0 - zeta) * gp[(a, b)] * (ls.s_u[a] + ls.s_u[b]) * x.e_uu[(a, b)];
                }
            }
            // through P2 and c1, each group's sums: d/dK^G[hh, y]
            for hh in 0..g {
                let mut t = Dense::zeros(g, g);
                let mut cr = vec![0.0; g];
                for a in 0..m {
                    cr[gu[a]] += alpha[a] * ls.r[hh][a];
                    for b in 0..m {
                        t[(gu[a], gu[b])] += gp[(a, b)] * ls.q[hh][(a, b)];
                    }
                }
                for y in 0..g {
                    let via_p2: f64 = (0..g).map(|yy| t[(y, yy)] * kg[(hh, yy)]).sum();
                    hmat[(hh, y)] += 2.0 * (1.0 - zeta).powi(2) * via_p2 + (1.0 - zeta) * cr[y];
                }
            }
        }
        dzeta += inner(&gs, &(Dense::identity(m, m) - &ek));
        for a in 0..m {
            for b in 0..m {
                hmat[(gu[a], gu[b])] += (1.0 - zeta) * gs[(a, b)] * x.e_uu[(a, b)];
            }
        }
        let hmat = (&hmat + hmat.transpose()) * 0.5;
        // through the correlation normalisation of K~ = x x^T + sigma^2 I
        let kt = group_cov(&h.x, h.sigma, g, r);
        let mut gt = Dense::zeros(g, g);
        for a in 0..g {
            for bb in 0..g {
                if a != bb {
                    gt[(a, bb)] = hmat[(a, bb)] / (kt[(a, a)] * kt[(bb, bb)]).sqrt();
                }
            }
            let mut diag = 0.0;
            for bb in 0..g {
                if bb != a {
                    diag -= hmat[(a, bb)] * kg[(a, bb)];
                }
            }
            gt[(a, a)] = diag / kt[(a, a)];
        }
        let sym = &gt + gt.transpose();
        let mut out = vec![dzeta, 2.0 * h.sigma * gt.trace()];
        for a in 0..g {
            for j in 0..r {
                out.push((0..g).map(|bb| sym[(a, bb)] * h.x[bb * r + j]).sum());
            }
        }
        (f, out)
    }

    /// The parameters a lengthscale's search moves: `zeta` (and with more
    /// than one group, `sigma` and `x`), as a vector with its box.
    fn free(&self, h: &Hyper) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut x = vec![h.zeta];
        let mut lo = vec![ZETA_MIN];
        let mut hi = vec![1.0];
        if self.n_groups > 1 {
            x.push(h.sigma);
            lo.push(SIGMA_MIN);
            hi.push(1.0);
            for &v in &h.x {
                x.push(v);
                lo.push(-1.0);
                hi.push(1.0);
            }
        }
        (x, lo, hi)
    }

    fn with(&self, h: &Hyper, v: &[f64]) -> Hyper {
        let mut out = h.clone();
        out.zeta = v[0];
        if self.n_groups > 1 {
            out.sigma = v[1];
            out.x.copy_from_slice(&v[2..]);
        }
        out
    }

    /// The group kernel of factor `k` and its sharedness (the mean absolute
    /// off-diagonal entry).
    pub fn group_report(&self, k: usize) -> (Dense, f64) {
        let kg = self.kg(&self.hyper[k]);
        let g = self.n_groups;
        let off: f64 = (0..g)
            .flat_map(|a| (0..g).filter(move |&b| b != a).map(move |b| (a, b)))
            .map(|(a, b)| kg[(a, b)].abs())
            .sum();
        let pairs = (g * g.saturating_sub(1)).max(1) as f64;
        (kg, off / pairs)
    }

    /// The factor's part of the ELBO with `q(u)` fixed at `h` (for tests).
    #[cfg(test)]
    pub fn factor_term(&self, k: usize, s: &[f64], b: &[f64], h: &Hyper) -> f64 {
        let Posterior::Inducing { nu, b: bq, ld_b } = &self.q[k] else {
            return self.cells_optimum(s, b);
        };
        let x = self.search(h.lengthscale, s, b, nu, bq, *ld_b);
        self.objective(&x, h, false).0
    }

    /// [`Self::factor_term`] by the one-group search in the eigenbasis (for
    /// tests).
    #[cfg(test)]
    pub fn fast_factor_term(&self, k: usize, s: &[f64], b: &[f64], h: &Hyper) -> f64 {
        let Posterior::Inducing { nu, b: bq, ld_b } = &self.q[k] else {
            unreachable!("a factor on the inducing cells")
        };
        let basis = self.basis_at(h.lengthscale);
        self.fast_search(&basis, s, b, nu, bq, *ld_b).eval(h.zeta)
    }

    /// Factor `k` at hyperparameters `h` with `q(u)` kept: its moments into
    /// `mu`, the per-group variance sums returned (for tests).
    #[cfg(test)]
    pub fn set_hyper(&mut self, k: usize, h: Hyper, mu: &mut [f64]) -> Vec<f64> {
        self.hyper[k] = h;
        self.moments(k, mu)
    }

    /// The gradient of [`Self::factor_term`] in `(zeta, sigma, x)` (for
    /// tests).
    #[cfg(test)]
    pub fn factor_grad(&self, k: usize, s: &[f64], b: &[f64], h: &Hyper) -> Vec<f64> {
        let Posterior::Inducing { nu, b: bq, ld_b } = &self.q[k] else {
            unreachable!("a factor on the inducing cells")
        };
        let x = self.search(h.lengthscale, s, b, nu, bq, *ld_b);
        self.objective(&x, h, true).1
    }
}

impl FactorPrior for GpPrior {
    fn update(&mut self, k: usize, s: &[f64], b: &[f64], mu: &mut [f64]) -> Vec<f64> {
        let h = self.hyper[k].clone();
        if h.is_unstructured() {
            let (var, tr) = self.cells_update(s, b, mu);
            self.q[k] = Posterior::Cells { var };
            return tr;
        }
        let kg = self.kg(&h);
        let suu = self.suu(&h);
        let m = suu.nrows();
        let n = self.xy.len();
        let mut p = Dense::zeros(m, m);
        let mut cvec = DVector::zeros(m);
        for start in (0..n).step_by(ROW_BLOCK) {
            let rows = start..(start + ROW_BLOCK).min(n);
            let kz = self.szu(&h, &kg, rows.clone());
            let mut sk = kz.clone();
            for (r, c) in rows.clone().enumerate() {
                sk.row_mut(r).scale_mut(s[self.group[c]]);
            }
            p += kz.transpose() * &sk;
            let bb = DVector::from_iterator(rows.len(), rows.map(|c| b[c]));
            cvec += kz.transpose() * bb;
        }
        let a = Self::chol(&p + &suu);
        let tmp = a.solve(&cvec);
        let nu = &suu * &tmp;
        let bmat = &suu * a.solve(&suu);
        let bmat = (&bmat + bmat.transpose()) * 0.5;
        let ld_b = Self::logdet(&Self::chol(bmat.clone()));
        self.q[k] = Posterior::Inducing { nu, b: bmat, ld_b };
        self.moments(k, mu)
    }

    fn kl_term(&self, k: usize, mu: &[f64]) -> f64 {
        match &self.q[k] {
            Posterior::Inducing { nu, b, ld_b } => {
                let c = Self::chol(self.suu(&self.hyper[k]));
                let si_b = c.solve(b);
                let m = nu.len() as f64;
                -0.5 * Self::logdet(&c) - 0.5 * si_b.trace() - 0.5 * nu.dot(&c.solve(nu))
                    + 0.5 * ld_b
                    + m / 2.0
            }
            Posterior::Cells { var } => {
                let mm: f64 = mu.iter().map(|v| v * v).sum();
                let per: f64 = (0..self.n_groups)
                    .map(|g| {
                        self.members[g].len() as f64 * (0.5 * var[g].ln() - 0.5 * var[g] + 0.5)
                    })
                    .sum();
                per - 0.5 * mm
            }
        }
    }

    fn hyper(&mut self, k: usize, s: &[f64], b: &[f64], mu: &mut [f64]) -> Option<Vec<f64>> {
        let cur = self.hyper[k].clone();
        // the q(u) the structured candidates hold fixed: the current one, or
        // the unstructured factor's at the inducing cells
        let (nu, bq, ld_b) = match &self.q[k] {
            Posterior::Inducing { nu, b, ld_b } => (nu.clone(), b.clone(), *ld_b),
            Posterior::Cells { var } => {
                let v: Vec<f64> = self.inducing.iter().map(|&c| var[self.group[c]]).collect();
                (
                    DVector::from_iterator(v.len(), self.inducing.iter().map(|&c| mu[c])),
                    Dense::from_diagonal(&DVector::from_vec(v.clone())),
                    v.iter().map(|x| x.ln()).sum(),
                )
            }
        };
        // each lengthscale's search on its own thread
        let this = &*self;
        let lengths: Vec<usize> = (0..this.grid.len())
            .filter(|&i| this.grid[i] > 0.0)
            .collect();
        let found: Vec<(f64, Hyper)> = lengths
            .par_iter()
            .map(|&i| {
                let l = this.grid[i];
                let h0 = if cur.is_unstructured() {
                    Hyper {
                        lengthscale: l,
                        zeta: 0.5,
                        sigma: 0.5,
                        x: vec![1.0 / (this.n_groups as f64).sqrt(); cur.x.len()],
                    }
                } else {
                    Hyper {
                        lengthscale: l,
                        ..cur.clone()
                    }
                };
                if this.n_groups == 1 && this.fast {
                    let basis = this.bases[i].get_or_init(|| this.basis_at(l));
                    let x = this.fast_search(basis, s, b, &nu, &bq, ld_b);
                    let (zeta, f) = Self::fast_zeta(&x, h0.zeta);
                    return (f, Hyper { zeta, ..h0 });
                }
                let x = this.search(l, s, b, &nu, &bq, ld_b);
                let (x0, lo, hi) = this.free(&h0);
                let r = lbfgs_box(
                    |v| {
                        let (f, g) = this.objective(&x, &this.with(&h0, v), true);
                        (-f, g.iter().take(v.len()).map(|x| -x).collect())
                    },
                    &x0,
                    &lo,
                    &hi,
                    HYPER_MAX_EVALS,
                    HYPER_PGTOL,
                    0.0,
                );
                let cand = this.with(&h0, &r.x);
                (this.objective(&x, &cand, false).0, cand)
            })
            .collect();
        // the best structured candidate with q(u) held fixed, then given
        // its optimal q(u), against the unstructured factor at its own
        // optimum: both sides scored with their q at the optimum
        let winner = found
            .into_iter()
            .filter(|(_, h)| !h.is_unstructured())
            .max_by(|a, b| a.0.total_cmp(&b.0));
        let unstructured = self.cells_optimum(s, b);
        if let Some((_, h)) = winner {
            let before = (self.hyper[k].clone(), self.q[k].clone());
            self.hyper[k] = h;
            let mut m = mu.to_vec();
            let tr = self.update(k, s, b, &mut m);
            if self.factor_value(k, s, b, &m, &tr) > unstructured {
                mu.copy_from_slice(&m);
                return Some(tr);
            }
            (self.hyper[k], self.q[k]) = before;
        }
        let (var, tr) = self.cells_update(s, b, mu);
        self.hyper[k] = Hyper::unstructured(cur.x.len());
        self.q[k] = Posterior::Cells { var };
        Some(tr)
    }

    fn hyper_values(&self, k: usize) -> Vec<f64> {
        let h = &self.hyper[k];
        let mut v = vec![h.lengthscale.ln(), h.zeta.ln(), h.sigma.ln()];
        v.extend(&h.x);
        v
    }

    fn report(&self) -> serde_json::Value {
        let per: Vec<serde_json::Value> = (0..self.hyper.len())
            .map(|k| {
                let h = &self.hyper[k];
                let (kg, shared) = self.group_report(k);
                serde_json::json!({
                    "lengthscale": h.lengthscale,
                    "zeta": h.zeta,
                    "smoothness": 1.0 - h.zeta,
                    "group_kernel": (0..kg.nrows()).map(|a| (0..kg.ncols()).map(|b| kg[(a, b)]).collect::<Vec<_>>()).collect::<Vec<_>>(),
                    "sharedness": shared,
                })
            })
            .collect();
        serde_json::json!({
            "factors": per,
            "n_inducing": self.inducing.len(),
            "lengthscale_grid": self.grid,
        })
    }
}

/// The smallest positive and the largest pairwise distances over all cells
/// pooled, exactly: the smallest among each cell's nearest, the largest
/// over the convex hull in 2-D (over all pairs otherwise).
pub fn distance_range(xy: &[Vec<f64>]) -> (f64, f64) {
    let n = xy.len();
    if n < 2 {
        return (1.0, 1.0);
    }
    let m = crate::util::common::Mat::from_fn(n, xy[0].len(), |i, j| xy[i][j] as f32);
    let (_, dist) = legume_numeric::matrix::knn::all_pairs::knn_rows_l2(&m, 8.min(n - 1));
    let lo = dist
        .iter()
        .flat_map(|d| d.iter().copied())
        .filter(|&d| d > 0.0)
        .fold(f32::INFINITY, f32::min) as f64;
    let pts: Vec<&Vec<f64>> = if xy[0].len() == 2 {
        hull(xy)
    } else {
        xy.iter().collect()
    };
    let mut hi: f64 = 0.0;
    for a in 0..pts.len() {
        for b in a + 1..pts.len() {
            hi = hi.max(d2(pts[a], pts[b]).sqrt());
        }
    }
    (if lo.is_finite() { lo } else { 1.0 }, hi.max(1e-12))
}

/// The convex hull of 2-D points (Andrew's monotone chain).
fn hull(xy: &[Vec<f64>]) -> Vec<&Vec<f64>> {
    let mut p: Vec<&Vec<f64>> = xy.iter().collect();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup_by(|a, b| a[0] == b[0] && a[1] == b[1]);
    if p.len() < 3 {
        return p;
    }
    let cross = |o: &Vec<f64>, a: &Vec<f64>, b: &Vec<f64>| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut lower: Vec<&Vec<f64>> = Vec::new();
    for &q in &p {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], q) <= 0.0 {
            lower.pop();
        }
        lower.push(q);
    }
    let mut upper: Vec<&Vec<f64>> = Vec::new();
    for &q in p.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], q) <= 0.0 {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The inducing cells (SI section 4.3, "a subset of the training inputs in
/// a regular grid ... ties shuffled randomly"; the 2-D reading is pinto's):
/// every cell when there are at most `m` of them; else per section a share
/// of `m` proportional to its cells, each section's bounding box laid with
/// a lattice of about that many nodes, each node taking its nearest cell,
/// duplicates dropped, a shortfall filled by the cells farthest from those
/// chosen, and a surplus cut in a seeded shuffled order.
pub fn inducing_cells(
    xy: &[Vec<f64>],
    group: &[usize],
    n_groups: usize,
    m: usize,
    seed: u64,
) -> Vec<usize> {
    let n = xy.len();
    if n <= m {
        return (0..n).collect();
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(m);
    let members: Vec<Vec<usize>> = (0..n_groups)
        .map(|g| (0..n).filter(|&c| group[c] == g).collect())
        .collect();
    for (g, cells) in members.iter().enumerate() {
        if cells.is_empty() {
            continue;
        }
        let share = if g + 1 == n_groups {
            m - out.len()
        } else {
            ((m as f64 * cells.len() as f64 / n as f64).round() as usize).min(m - out.len())
        }
        .min(cells.len());
        if share == 0 {
            continue;
        }
        let dims = xy[cells[0]].len();
        let lo: Vec<f64> = (0..dims)
            .map(|j| {
                cells
                    .iter()
                    .map(|&c| xy[c][j])
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let hi: Vec<f64> = (0..dims)
            .map(|j| {
                cells
                    .iter()
                    .map(|&c| xy[c][j])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        let (w, h) = (
            (hi[0] - lo[0]).max(1e-12),
            (hi.get(1).copied().unwrap_or(0.0) - lo.get(1).copied().unwrap_or(0.0)).max(1e-12),
        );
        let aspect = w / h;
        let nx = ((share as f64 * aspect).sqrt().ceil() as usize).max(1);
        let ny = ((share as f64 / aspect).sqrt().ceil() as usize).max(1);
        let mut chosen: Vec<usize> = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                let px = lo[0] + w * (i as f64 + 0.5) / nx as f64;
                let py = lo.get(1).copied().unwrap_or(0.0) + h * (j as f64 + 0.5) / ny as f64;
                let near = cells
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        let da = (xy[a][0] - px).powi(2)
                            + xy[a].get(1).map_or(0.0, |y| (y - py).powi(2));
                        let db = (xy[b][0] - px).powi(2)
                            + xy[b].get(1).map_or(0.0, |y| (y - py).powi(2));
                        da.total_cmp(&db)
                    })
                    .expect("a section with cells");
                chosen.push(near);
            }
        }
        chosen.sort_unstable();
        chosen.dedup();
        chosen.shuffle(&mut rng);
        chosen.truncate(share);
        while chosen.len() < share {
            let far = cells
                .iter()
                .copied()
                .filter(|c| !chosen.contains(c))
                .max_by(|&a, &b| {
                    let da = chosen
                        .iter()
                        .map(|&c| d2(&xy[a], &xy[c]))
                        .fold(f64::INFINITY, f64::min);
                    let db = chosen
                        .iter()
                        .map(|&c| d2(&xy[b], &xy[c]))
                        .fold(f64::INFINITY, f64::min);
                    da.total_cmp(&db)
                })
                .expect("cells left to choose");
            chosen.push(far);
        }
        out.extend(chosen);
    }
    out.sort_unstable();
    out
}
