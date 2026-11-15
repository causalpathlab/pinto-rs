//! MEFISTO's own factor prior (`--reference`): each factor a Gaussian
//! process over the coordinates and the groups (SI (9)-(13), (24)),
//!
//! `Sigma_k[n, n'] = (1 - zeta_k) exp(-||c_n - c_n'||^2 / (2 l_k^2))
//! K^G_k[g(n), g(n')] + zeta_k [n = n']`,
//!
//! `K^G_k` the correlation matrix of `x_k x_k^T + sigma_k^2 I` (rank
//! `R`), on `M` inducing cells (SI section 4): `p(z_k | u_k)` the prior's
//! conditional, `q(u_k) = N(nu_k, B_k)` (SI (79)-(83)). With every cell an
//! inducing point (`N <= --n-inducing`) this is the full GP of SI section
//! 3 exactly (`Sigma_ZU Sigma_UU^{-1} = I`, no conditional variance).
//! `Sigma_ZU` is formed in row blocks, never whole (no `N × N` matrix).
//!
//! Hyperparameters (SI section 3.2.2, (84)): per factor, the lengthscale on
//! a grid from half the smallest to twice the largest pairwise distance,
//! plus 0; at each, `zeta`, `sigma`, `x` maximise the inducing terms by a
//! box-constrained quasi-Newton search, from their current values; the
//! best kept, then the identifiability rule (`l = 0` or `1 - zeta = 0`
//! gives the unstructured prior).

use super::super::spatial_factor::kit::{lbfgs_box, Dense};
use super::vb::FactorPrior;
use nalgebra::{Cholesky, DVector, Dyn};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const ROW_BLOCK: usize = 4096;
const JITTER: f64 = 1e-8;
const ZETA_MIN: f64 = 1e-4;
const SIGMA_MIN: f64 = 1e-4;
const HYPER_MAX_EVALS: usize = 100;
const HYPER_PGTOL: f64 = 1e-5;

/// One factor's kernel hyperparameters.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hyper {
    pub lengthscale: f64,
    pub zeta: f64,
    /// Group factors, groups × rank, row-major.
    pub x: Vec<f64>,
    pub sigma: f64,
}

/// The GP prior of every factor.
pub struct GpPrior {
    xy: Vec<Vec<f64>>,
    group: Vec<usize>,
    n_groups: usize,
    rank: usize,
    /// The inducing cells.
    pub inducing: Vec<usize>,
    /// The lengthscale grid (0 first).
    pub grid: Vec<f64>,
    pub hyper: Vec<Hyper>,
    /// `q(u_k)`: mean, covariance, log-determinant of the covariance.
    q: Vec<(DVector<f64>, Dense, f64)>,
}

fn d2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// The group correlation `K^G` of `x x^T + sigma^2 I`.
fn group_kernel(x: &[f64], sigma: f64, g: usize, r: usize) -> Dense {
    let kt = Dense::from_fn(g, g, |a, b| {
        let mut v: f64 = (0..r).map(|j| x[a * r + j] * x[b * r + j]).sum();
        if a == b {
            v += sigma * sigma;
        }
        v
    });
    Dense::from_fn(g, g, |a, b| kt[(a, b)] / (kt[(a, a)] * kt[(b, b)]).sqrt())
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
        GpPrior {
            xy,
            group,
            n_groups,
            rank,
            inducing,
            grid,
            hyper: vec![start; k],
            q: vec![(DVector::zeros(m), Dense::identity(m, m), 0.0); k],
        }
    }

    fn entry(&self, h: &Hyper, kg: &Dense, a: usize, b: usize) -> f64 {
        let same = if a == b { h.zeta } else { 0.0 };
        if h.lengthscale == 0.0 {
            return if a == b { 1.0 } else { 0.0 };
        }
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

    /// Each cell's `E[z]` and the per-group sums of `Var(z)` from `q(u)`:
    /// `E[z_n] = k_n^T Sigma_UU^{-1} nu`, `Var(z_n) = 1 - k_n^T
    /// Sigma_UU^{-1} k_n + k_n^T Sigma_UU^{-1} B Sigma_UU^{-1} k_n`.
    fn moments(&self, k: usize, mu: &mut [f64]) -> Vec<f64> {
        let h = &self.hyper[k];
        let (nu, b, _) = &self.q[k];
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

    /// SI (84): `-0.5 log det Sigma_UU - 0.5 tr(Sigma_UU^{-1} B) - 0.5 nu^T
    /// Sigma_UU^{-1} nu` (the hyperparameters' part), and its gradient in
    /// `(zeta, sigma, x)` at lengthscale `h.lengthscale > 0`.
    fn hyper_objective(&self, k: usize, h: &Hyper, grad: bool) -> (f64, Vec<f64>) {
        let (nu, b, _) = &self.q[k];
        let s = self.suu(h);
        let c = Self::chol(s);
        let si = c.inverse();
        let sinu = &si * nu;
        let f = -0.5 * Self::logdet(&c) - 0.5 * (&si * b).trace() - 0.5 * nu.dot(&sinu);
        if !grad {
            return (f, Vec::new());
        }
        // dF = 0.5 tr(Gamma dSigma), Gamma = S^{-1} (B + nu nu^T) S^{-1} - S^{-1}
        let nnt = nu * nu.transpose();
        let gm = &si * (b + nnt) * &si - &si;
        let u = &self.inducing;
        let (g, r) = (self.n_groups, self.rank);
        let mut dzeta = 0.0;
        let mut hmat = Dense::zeros(g, g);
        let kg = self.kg(h);
        for a in 0..u.len() {
            for bb in 0..u.len() {
                let e = (-d2(&self.xy[u[a]], &self.xy[u[bb]])
                    / (2.0 * h.lengthscale * h.lengthscale))
                    .exp();
                let (ga, gb) = (self.group[u[a]], self.group[u[bb]]);
                let gab = gm[(a, bb)];
                dzeta -= gab * e * kg[(ga, gb)];
                if a == bb {
                    dzeta += gab;
                }
                hmat[(ga, gb)] += 0.5 * (1.0 - h.zeta) * gab * e;
            }
        }
        dzeta *= 0.5;
        // through the correlation normalisation of K~ = x x^T + sigma^2 I
        let kt = Dense::from_fn(g, g, |a, bb| {
            let mut v: f64 = (0..r).map(|j| h.x[a * r + j] * h.x[bb * r + j]).sum();
            if a == bb {
                v += h.sigma * h.sigma;
            }
            v
        });
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
}

impl FactorPrior for GpPrior {
    fn update(&mut self, k: usize, s: &[f64], b: &[f64], mu: &mut [f64]) -> Vec<f64> {
        let h = self.hyper[k].clone();
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
        self.q[k] = (nu, bmat, ld_b);
        self.moments(k, mu)
    }

    fn kl_term(&self, k: usize, _mu: &[f64]) -> f64 {
        let (nu, b, ld_b) = &self.q[k];
        let c = Self::chol(self.suu(&self.hyper[k]));
        let si_b = c.solve(b);
        let m = nu.len() as f64;
        -0.5 * Self::logdet(&c) - 0.5 * si_b.trace() - 0.5 * nu.dot(&c.solve(nu))
            + 0.5 * ld_b
            + m / 2.0
    }

    fn hyper(&mut self, k: usize, mu: &mut [f64]) -> Option<Vec<f64>> {
        let cur = self.hyper[k].clone();
        let mut best: Option<(f64, Hyper)> = None;
        for &l in &self.grid.clone() {
            let cand = if l == 0.0 {
                Hyper {
                    lengthscale: 0.0,
                    zeta: 1.0,
                    x: vec![0.0; cur.x.len()],
                    sigma: 1.0,
                }
            } else {
                let h0 = Hyper {
                    lengthscale: l,
                    ..cur.clone()
                };
                let h0 = if h0.zeta >= 1.0 {
                    Hyper {
                        zeta: 0.5,
                        sigma: 0.5,
                        x: vec![1.0 / (self.n_groups as f64).sqrt(); cur.x.len()],
                        ..h0
                    }
                } else {
                    h0
                };
                let (x0, lo, hi) = self.free(&h0);
                let r = lbfgs_box(
                    |v| {
                        let (f, g) = self.hyper_objective(k, &self.with(&h0, v), true);
                        (-f, g.iter().take(v.len()).map(|x| -x).collect())
                    },
                    &x0,
                    &lo,
                    &hi,
                    HYPER_MAX_EVALS,
                    HYPER_PGTOL,
                    0.0,
                );
                self.with(&h0, &r.x)
            };
            let f = self.hyper_objective(k, &cand, false).0;
            if best.as_ref().is_none_or(|b| f > b.0) {
                best = Some((f, cand));
            }
        }
        let (_, mut h) = best.expect("a grid of at least one lengthscale");
        // the identifiability rule
        if h.lengthscale == 0.0 || 1.0 - h.zeta <= 1e-12 {
            h = Hyper {
                lengthscale: 0.0,
                zeta: 1.0,
                x: vec![0.0; h.x.len()],
                sigma: 1.0,
            };
        }
        self.hyper[k] = h;
        Some(self.moments(k, mu))
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
