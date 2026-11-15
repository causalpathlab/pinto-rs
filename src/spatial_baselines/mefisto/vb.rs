//! MEFISTO's variational Bayes (Supplementary Information sections 2 to
//! 4; the multi-group updates of MOFA+, Argelaguet et al. 2020, Genome
//! Biology 21:111, doi:10.1186/s13059-020-02015-1), one view, the sections
//! as groups, over any factor prior ([`FactorPrior`]): the same weight,
//! ARD, sparsity and noise updates and ELBO whether the factors carry the
//! paper's Gaussian process or the sparse graph prior.
//!
//! Model (SI (14)-(24)): `y_c[n, d] ~ N(sum_k w[d,k] z[n,k], 1 /
//! tau[d, g(n)])`, spike-and-slab weights `w = w^ s`, `w^ ~ N(0, 1 /
//! alpha_k)`, `s ~ Bernoulli(theta_k)`, `theta_k ~ Beta(1, 1)`, `alpha_k,
//! tau[d, g] ~ Gamma(1e-3, 1e-3)`. The variational family is SI (29); each
//! update below is the coordinate ascent step of SI (28), written for the
//! per-group noise of SI (23) (identical to SI (51)/(53) with one group).

use super::super::spatial_factor::kit::{top_eigs, Dense, EigOpts};
use super::super::spatial_factor::prep::ResidualOp;

const A0: f64 = 1e-3;
const B0: f64 = 1e-3;
const A0_THETA: f64 = 1.0;
const B0_THETA: f64 = 1.0;
const LN_2PI: f64 = 1.837_877_066_409_345_5;

fn digamma(x: f64) -> f64 {
    special::Gamma::digamma(x)
}

fn ln_gamma(x: f64) -> f64 {
    special::Gamma::ln_gamma(x).0
}

/// A factor prior: the factor update, its ELBO terms and its
/// hyperparameter step, for factor `k`.
pub trait FactorPrior {
    /// `q(z_k)` given the likelihood's precision `s[g]` in each section and
    /// the linear term `b` (SI (33)-(34)): its mean into `mu` and its
    /// per-section variance sums returned.
    fn update(&mut self, k: usize, s: &[f64], b: &[f64], mu: &mut [f64]) -> Vec<f64>;
    /// `E[log p(z_k)] - E[log q(z_k)]` at the current `q` and prior.
    fn kl_term(&self, k: usize, mu: &[f64]) -> f64;
    /// The hyperparameter step (SI (39)-(40)), given the likelihood's `s`
    /// and `b` at the current state, as [`FactorPrior::update`] takes them:
    /// it never lowers the factor's part of the ELBO. The factor's mean
    /// and per-section variance sums afterwards, when the step moves them.
    fn hyper(&mut self, k: usize, s: &[f64], b: &[f64], mu: &mut [f64]) -> Option<Vec<f64>>;
    /// Factor `k`'s hyperparameters on the scale [`fit`]'s convergence
    /// compares them (logarithms of the positive ones; `-inf` for a zero).
    fn hyper_values(&self, k: usize) -> Vec<f64>;
    /// What a run records of the prior's state.
    fn report(&self) -> serde_json::Value;
}

/// The data the updates read: the centred log expression, per-gene and
/// per-section sums of squares, and the sections.
pub struct VbData<'a> {
    pub op: &'a ResidualOp,
    pub section: &'a [usize],
    pub n_sections: usize,
    /// `sum_{n in g} y_c[n, d]^2`, genes × sections.
    pub yy: Dense,
    pub sizes: Vec<f64>,
}

impl<'a> VbData<'a> {
    pub fn new(op: &'a ResidualOp, section: &'a [usize], n_sections: usize) -> Self {
        let mut sizes = vec![0f64; n_sections];
        for &s in section {
            sizes[s] += 1.0;
        }
        VbData {
            yy: op.section_sumsq(section, n_sections),
            op,
            section,
            n_sections,
            sizes,
        }
    }

    /// `Y_g mu` for each section, genes × K each, and the per-section Gram
    /// `mu_g^T mu_g`.
    pub fn data_terms(&self, mu: &Dense) -> (Vec<Dense>, Vec<Dense>) {
        let (n, k) = mu.shape();
        let g = self.n_sections;
        let per = self.op.apply_by_section(mu, self.section, g);
        let mut gram = vec![Dense::zeros(k, k); g];
        for c in 0..n {
            let s = self.section[c];
            for a in 0..k {
                for b in 0..k {
                    gram[s][(a, b)] += mu[(c, a)] * mu[(c, b)];
                }
            }
        }
        (per, gram)
    }
}

/// The variational state.
#[derive(Debug, Clone)]
pub struct State {
    /// `E[z]`, cells × K.
    pub mu: Dense,
    /// `sum_{n in g} Var(z[n, k])`, K × sections.
    pub tr: Dense,
    /// Slab probability, slab mean and variance of each weight, genes × K.
    pub gamma: Dense,
    pub mu_w: Dense,
    pub s2_w: Dense,
    /// `q(alpha_k) = Gamma(a, b)`, `q(theta_k) = Beta(a, b)`.
    pub alpha: Vec<(f64, f64)>,
    pub theta: Vec<(f64, f64)>,
    /// `q(tau[d, g]) = Gamma(a, b)`, genes × sections each.
    pub tau_a: Dense,
    pub tau_b: Dense,
}

impl State {
    /// The paper is silent on the start; pinto's: `E[z]` from a seeded
    /// truncated SVD of `Y_c` (each factor scaled to unit variance), no
    /// variance, no weights, `E[alpha] = 1`, `E[theta] = 0.5`, `E[tau] = 1 /
    /// variance` of each gene in each section.
    pub fn init(data: &VbData<'_>, k: usize, seed: u64) -> Self {
        let op = data.op;
        let m = op.m;
        let o = EigOpts {
            d: k,
            block: (k + 10).min(m),
            seed,
            tol: 1e-8,
            max_iter: 200,
        };
        let (_, u, _) = top_eigs(|v| op.apply(&op.apply_t(v)), m, None, &o);
        let mut mu = op.apply_t(&u);
        let n = op.n;
        for j in 0..mu.ncols() {
            let col = mu.column(j);
            let mean = col.mean();
            let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            let sd = if sd > 0.0 { sd } else { 1.0 };
            mu.column_mut(j).iter_mut().for_each(|v| *v /= sd);
        }
        Self::start(data, mu)
    }

    /// The state with factors `mu`, no variance, no weights, `E[alpha] =
    /// 1`, `E[theta] = 0.5`, `E[tau] = 1 / variance` of each gene in each
    /// section.
    pub fn start(data: &VbData<'_>, mu: Dense) -> Self {
        let m = data.op.m;
        let g = data.n_sections;
        let tau_a = Dense::from_fn(m, g, |_, s| A0 + data.sizes[s] / 2.0);
        let tau_b = Dense::from_fn(m, g, |d, s| {
            let var = data.yy[(d, s)] / data.sizes[s].max(1.0);
            tau_a[(d, s)] * var.max(1e-6)
        });
        let kk = mu.ncols();
        State {
            mu,
            tr: Dense::zeros(kk, g),
            gamma: Dense::zeros(m, kk),
            mu_w: Dense::zeros(m, kk),
            s2_w: Dense::zeros(m, kk),
            alpha: vec![(1.0, 1.0); kk],
            theta: vec![(1.0, 1.0); kk],
            tau_a,
            tau_b,
        }
    }

    pub fn k(&self) -> usize {
        self.mu.ncols()
    }

    /// `E[w]`, genes × K.
    pub fn w_mean(&self) -> Dense {
        self.gamma.component_mul(&self.mu_w)
    }

    /// `E[w^2]`.
    fn w2(&self) -> Dense {
        let sq = self.mu_w.component_mul(&self.mu_w) + &self.s2_w;
        self.gamma.component_mul(&sq)
    }

    fn tau_mean(&self) -> Dense {
        self.tau_a.component_div(&self.tau_b)
    }
}

/// Settings of [`fit`].
#[derive(Debug, Clone, Copy)]
pub struct VbOpts {
    pub tol: f64,
    pub max_iter: usize,
    /// The hyperparameter step runs at iteration `t` when `t >= gp_start`
    /// and `(t - gp_start) % gp_every == 0`.
    pub gp_start: usize,
    pub gp_every: usize,
    /// The weights, ARD and sparsity held fixed (the `-pb` unravel).
    pub frozen_weights: bool,
    /// The hyperparameters are fitted once a step moves none of them by
    /// more than this (on [`FactorPrior::hyper_values`]' scale).
    pub hyper_tol: f64,
}

/// How a fit ended.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Trace {
    pub elbo: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Noise sums `E_q[sum_{n in g} (y - sum_k w z)^2]`, genes × sections.
fn residual_sums(data: &VbData<'_>, st: &State, ym: &[Dense], gram: &[Dense]) -> Dense {
    let (m, k) = (st.gamma.nrows(), st.k());
    let wm = st.w_mean();
    let w2 = st.w2();
    Dense::from_fn(m, data.n_sections, |d, g| {
        let mut e = data.yy[(d, g)];
        for a in 0..k {
            e -= 2.0 * wm[(d, a)] * ym[g][(d, a)];
            e += w2[(d, a)] * (gram[g][(a, a)] + st.tr[(a, g)]);
            for b in 0..k {
                if a != b {
                    e += wm[(d, a)] * wm[(d, b)] * gram[g][(a, b)];
                }
            }
        }
        e.max(0.0)
    })
}

/// SI (50)-(53): the weights of factor `k`, the others' as they stand.
fn update_weights(st: &mut State, k: usize, ym: &[Dense], gram: &[Dense]) {
    let (m, kk) = (st.gamma.nrows(), st.k());
    let tau = st.tau_mean();
    let (aa, ab) = st.alpha[k];
    let alpha = aa / ab;
    let (ta, tb) = st.theta[k];
    let elt = digamma(ta) - digamma(ta + tb);
    let el1t = digamma(tb) - digamma(ta + tb);
    let wm = st.w_mean();
    for d in 0..m {
        let mut t = alpha;
        let mut b = 0.0;
        for g in 0..ym.len() {
            t += tau[(d, g)] * (gram[g][(k, k)] + st.tr[(k, g)]);
            let mut cross = 0.0;
            for j in 0..kk {
                if j != k {
                    cross += wm[(d, j)] * gram[g][(k, j)];
                }
            }
            b += tau[(d, g)] * (ym[g][(d, k)] - cross);
        }
        let lambda = elt - el1t + 0.5 * alpha.ln() - 0.5 * t.ln() + 0.5 * b * b / t;
        st.mu_w[(d, k)] = b / t;
        st.s2_w[(d, k)] = 1.0 / t;
        st.gamma[(d, k)] = 1.0 / (1.0 + (-lambda).exp());
    }
}

/// SI (54)-(57): ARD and sparsity of every factor.
fn update_ard_theta(st: &mut State) {
    let m = st.gamma.nrows() as f64;
    for k in 0..st.k() {
        let alpha = st.alpha[k].0 / st.alpha[k].1;
        let ss: f64 = (0..st.gamma.nrows())
            .map(|d| {
                let g = st.gamma[(d, k)];
                g * (st.mu_w[(d, k)].powi(2) + st.s2_w[(d, k)]) + (1.0 - g) / alpha
            })
            .sum();
        st.alpha[k] = (A0 + m / 2.0, B0 + 0.5 * ss);
        let sg: f64 = st.gamma.column(k).sum();
        st.theta[k] = (A0_THETA + sg, B0_THETA + m - sg);
    }
}

/// SI (58)-(59), per gene and section.
fn update_noise(data: &VbData<'_>, st: &mut State, ym: &[Dense], gram: &[Dense]) {
    let e = residual_sums(data, st, ym, gram);
    for d in 0..e.nrows() {
        for g in 0..e.ncols() {
            st.tau_a[(d, g)] = A0 + data.sizes[g] / 2.0;
            st.tau_b[(d, g)] = B0 + 0.5 * e[(d, g)];
        }
    }
}

/// What a factor's update reads of the weights and the noise, fixed while
/// the factors move: `E[tau]`, `E[w]`, `E[w^2]`, and `Y_c^T (tau_g o
/// E[w])` at each cell's own section `g`.
pub struct FactorTerms {
    tau: Dense,
    wm: Dense,
    w2: Dense,
    ytu: Dense,
}

impl FactorTerms {
    pub fn new(data: &VbData<'_>, st: &State) -> Self {
        let (m, kk) = (st.gamma.nrows(), st.k());
        let tau = st.tau_mean();
        let wm = st.w_mean();
        let u: Vec<Dense> = (0..data.n_sections)
            .map(|sec| Dense::from_fn(m, kk, |d, k| tau[(d, sec)] * wm[(d, k)]))
            .collect();
        let ytu = data.op.apply_t_by_section(&u, data.section);
        FactorTerms {
            tau,
            w2: st.w2(),
            wm,
            ytu,
        }
    }

    /// Factor `k`'s likelihood precision per section and linear term per
    /// cell (SI (33)-(34)), the other factors at their means `mu`.
    pub fn of(&self, data: &VbData<'_>, k: usize, mu: &Dense) -> (Vec<f64>, Vec<f64>) {
        let (m, kk) = (self.wm.nrows(), self.wm.ncols());
        let (tau, wm) = (&self.tau, &self.wm);
        let s: Vec<f64> = (0..data.n_sections)
            .map(|sec| (0..m).map(|d| self.w2[(d, k)] * tau[(d, sec)]).sum())
            .collect();
        // sum_d tau_dg w_dk w_dl, per section and other factor
        let cross: Vec<Vec<f64>> = (0..data.n_sections)
            .map(|sec| {
                (0..kk)
                    .map(|l| {
                        if l == k {
                            0.0
                        } else {
                            (0..m)
                                .map(|d| tau[(d, sec)] * wm[(d, k)] * wm[(d, l)])
                                .sum()
                        }
                    })
                    .collect()
            })
            .collect();
        let b: Vec<f64> = (0..mu.nrows())
            .map(|c| {
                let sec = data.section[c];
                let mut v = self.ytu[(c, k)];
                for l in 0..kk {
                    v -= cross[sec][l] * mu[(c, l)];
                }
                v
            })
            .collect();
        (s, b)
    }
}

/// SI (33)-(34): every factor in turn, through its prior.
fn update_factors(
    data: &VbData<'_>,
    st: &mut State,
    prior: &mut dyn FactorPrior,
    terms: &FactorTerms,
) {
    for k in 0..st.k() {
        let (s, b) = terms.of(data, k, &st.mu);
        let mut mu_k: Vec<f64> = st.mu.column(k).iter().copied().collect();
        let tr = prior.update(k, &s, &b, &mut mu_k);
        set_factor(st, k, &mu_k, tr);
    }
}

fn set_factor(st: &mut State, k: usize, mu: &[f64], tr: Vec<f64>) {
    st.mu.column_mut(k).copy_from_slice(mu);
    for (sec, v) in tr.into_iter().enumerate() {
        st.tr[(k, sec)] = v;
    }
}

/// The hyperparameter step of every factor in turn, each given the
/// likelihood at the current state; the largest move of any
/// hyperparameter.
fn update_hyper(
    data: &VbData<'_>,
    st: &mut State,
    prior: &mut dyn FactorPrior,
    terms: &FactorTerms,
) -> f64 {
    let mut moved = 0f64;
    for k in 0..st.k() {
        let before = prior.hyper_values(k);
        let (s, b) = terms.of(data, k, &st.mu);
        let mut mu: Vec<f64> = st.mu.column(k).iter().copied().collect();
        if let Some(tr) = prior.hyper(k, &s, &b, &mut mu) {
            set_factor(st, k, &mu, tr);
        }
        for (a, b) in before.iter().zip(prior.hyper_values(k)) {
            // the same value, an infinity included, has not moved
            let d = if *a == b { 0.0 } else { (a - b).abs() };
            moved = moved.max(if d.is_nan() { f64::INFINITY } else { d });
        }
    }
    moved
}

fn gamma_kl(a0: f64, b0: f64, a: f64, b: f64) -> f64 {
    // E_q[log p] - E_q[log q] for Gamma(a0, b0) prior, Gamma(a, b) posterior
    let el = digamma(a) - b.ln();
    let mean = a / b;
    (a0 * b0.ln() - ln_gamma(a0) + (a0 - 1.0) * el - b0 * mean)
        - (a * b.ln() - ln_gamma(a) + (a - 1.0) * el - b * mean)
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// SI (60)-(68): the evidence lower bound, the factor terms from the prior.
pub fn elbo(
    data: &VbData<'_>,
    st: &State,
    prior: &dyn FactorPrior,
    ym: &[Dense],
    gram: &[Dense],
) -> f64 {
    let (m, kk, g) = (st.gamma.nrows(), st.k(), data.n_sections);
    let e = residual_sums(data, st, ym, gram);
    let mut total = 0.0;
    for d in 0..m {
        for s in 0..g {
            let (a, b) = (st.tau_a[(d, s)], st.tau_b[(d, s)]);
            let el = digamma(a) - b.ln();
            total +=
                -0.5 * data.sizes[s] * LN_2PI + 0.5 * data.sizes[s] * el - 0.5 * a / b * e[(d, s)];
            total += gamma_kl(A0, B0, a, b);
        }
    }
    for k in 0..kk {
        let (aa, ab) = st.alpha[k];
        let alpha = aa / ab;
        let ela = digamma(aa) - ab.ln();
        let (ta, tb) = st.theta[k];
        let elt = digamma(ta) - digamma(ta + tb);
        let el1t = digamma(tb) - digamma(ta + tb);
        for d in 0..m {
            let gm = st.gamma[(d, k)];
            let s2 = st.s2_w[(d, k)];
            let what2 = gm * (st.mu_w[(d, k)].powi(2) + s2) + (1.0 - gm) / alpha;
            total += -0.5 * LN_2PI + 0.5 * ela - 0.5 * alpha * what2;
            total += gm * elt + (1.0 - gm) * el1t;
            let h = |p: f64| if p > 0.0 { -p * p.ln() } else { 0.0 };
            total += h(gm) + h(1.0 - gm);
            if gm > 0.0 {
                total += gm * 0.5 * (LN_2PI + 1.0 + s2.max(1e-300).ln());
            }
            total += (1.0 - gm) * 0.5 * (LN_2PI + 1.0 - alpha.ln());
        }
        total += gamma_kl(A0, B0, aa, ab);
        total += (A0_THETA - 1.0) * elt + (B0_THETA - 1.0) * el1t
            - ln_beta(A0_THETA, B0_THETA)
            - ((ta - 1.0) * elt + (tb - 1.0) * el1t - ln_beta(ta, tb));
        let mu: Vec<f64> = st.mu.column(k).iter().copied().collect();
        total += prior.kl_term(k, &mu);
    }
    total
}

/// Coordinate ascent: per iteration the weights (each factor in turn),
/// ARD, sparsity (unless frozen), the noise, the factors, then the prior's
/// hyperparameter step on its schedule (from `gp_start`, every
/// `gp_every`), then the ELBO. The fit has converged at an iteration whose
/// ELBO changed by less than `tol` (relative) and, when the
/// hyperparameters are fitted at all (`gp_start <= max_iter`), whose step
/// moved none of them by more than `hyper_tol`: the ELBO settles long
/// before the factors and their hyperparameters stop drifting, so the
/// ELBO alone would stop with them unfitted. Else the fit stops at
/// `max_iter` iterations. (The paper is silent on the rule; this is
/// pinto's.)
pub fn fit(data: &VbData<'_>, st: &mut State, prior: &mut dyn FactorPrior, o: &VbOpts) -> Trace {
    let mut trace = Vec::new();
    let mut converged = false;
    let (mut ym, mut gram) = data.data_terms(&st.mu);
    let fits_hyper = o.gp_start <= o.max_iter;
    let mut t = 0;
    while t < o.max_iter {
        t += 1;
        if !o.frozen_weights {
            for k in 0..st.k() {
                update_weights(st, k, &ym, &gram);
            }
            update_ard_theta(st);
        }
        update_noise(data, st, &ym, &gram);
        let terms = FactorTerms::new(data, st);
        update_factors(data, st, prior, &terms);
        let scheduled = t >= o.gp_start && (t - o.gp_start).is_multiple_of(o.gp_every.max(1));
        let moved = scheduled.then(|| update_hyper(data, st, prior, &terms));
        (ym, gram) = data.data_terms(&st.mu);
        let e = elbo(data, st, prior, &ym, &gram);
        let change = trace
            .last()
            .map(|&prev: &f64| ((e - prev) / prev.abs().max(1e-300)).abs());
        trace.push(e);
        let settled = change.is_some_and(|c| c < o.tol);
        if settled && (!fits_hyper || moved.is_some_and(|m| m < o.hyper_tol)) {
            converged = true;
            break;
        }
    }
    Trace {
        elbo: trace,
        iterations: t,
        converged,
    }
}
