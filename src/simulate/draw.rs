//! Counts from expected counts, through a low-rank Gaussian copula.
//!
//! ```text
//! z[g,i] = √ρ f[k(g), i] + √(1 − ρ) ε[g,i]          f, ε ~ N(0, 1)
//! y[g,i] = F⁻¹(Φ(z[g,i]); μ[g,i], dispersion[g])     F negative binomial
//! ```
//!
//! A Gaussian copula joins per-gene marginals (Sklar 1959, "Fonctions de
//! répartition à n dimensions et leurs marges", Publ. Inst. Statist. Univ.
//! Paris 8:229-231), as scDesign3 does for single-cell and spatial data
//! (Song et al. 2023, doi:10.1038/s41587-023-01772-1). Each gene loads on
//! one of `copula_rank` factors at `√ρ`, `ρ = copula_strength`, programme
//! genes on their state's factor, so co-expression follows the planted
//! modules; this is the low-rank-plus-ridge form of data-beans-sim's
//! reference copula, with the loadings planted rather than fitted. `f` and
//! `ε` come from keyed streams (`rng.rs`), so a spot gets the same uniform
//! in every scenario and a run with `nuisance_off` is its exact
//! counterfactual.
//!
//! The NB quantile is computed here, not through data-beans, so a shared
//! convention cannot cancel between the simulator and the methods it tests.

use super::params::SimParams;
use super::rng::{rng, Stream};
use rand_distr::{Distribution, StandardNormal};
use special::{Beta, Error, Gamma};

/// Standard normal CDF.
pub fn phi(z: f64) -> f64 {
    0.5 * (-z / std::f64::consts::SQRT_2).compl_error()
}

/// A negative binomial of mean `μ` and variance `μ + δμ²`, or Poisson when
/// `δ = 0`.
struct Marginal {
    mu: f64,
    /// `1/δ`, infinite for Poisson.
    r: f64,
}

impl Marginal {
    fn log_p0(&self) -> f64 {
        if self.r.is_finite() {
            self.r * (self.r / (self.r + self.mu)).ln()
        } else {
            -self.mu
        }
    }

    /// `P(Y = k + 1) / P(Y = k)`.
    fn ratio(&self, k: f64) -> f64 {
        if self.r.is_finite() {
            (k + self.r) / (k + 1.0) * self.mu / (self.r + self.mu)
        } else {
            self.mu / (k + 1.0)
        }
    }

    /// `P(Y ≤ k)`: the regularised incomplete beta `I_{r/(r+μ)}(r, k + 1)`,
    /// or `1 − P(k + 1, μ)` with the regularised lower incomplete gamma.
    fn cdf(&self, k: u32) -> f64 {
        let k1 = k as f64 + 1.0;
        if self.r.is_finite() {
            (self.r / (self.r + self.mu)).inc_beta(self.r, k1, self.r.ln_beta(k1))
        } else {
            1.0 - self.mu.inc_gamma(k1)
        }
    }
}

/// Smallest `y` with `P(Y ≤ y) ≥ u`. Sums the mass upward from 0 by its
/// ratio recurrence while the mass at 0 is representable; past that (means
/// of several hundred) it bisects on the exact CDF.
pub fn nb_quantile(u: f64, mu: f64, dispersion: f64) -> u32 {
    if mu <= 0.0 || u <= 0.0 {
        return 0;
    }
    let m = Marginal {
        mu,
        r: if dispersion > 0.0 {
            1.0 / dispersion
        } else {
            f64::INFINITY
        },
    };
    // far past any quantile short of 1 − 1e-15
    let cap = (mu * (1.0 + 50.0 * dispersion) + 50.0 * mu.sqrt() + 1000.0) as u32;
    let log_p0 = m.log_p0();
    if log_p0 > -600.0 {
        let (mut pmf, mut k) = (log_p0.exp(), 0u32);
        let mut cdf = pmf;
        while cdf < u && k < cap {
            pmf *= m.ratio(k as f64);
            k += 1;
            cdf += pmf;
        }
        return k;
    }
    let (mut lo, mut hi) = (0u32, cap);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if m.cdf(mid) >= u {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// The copula's factors over one section's spots.
pub struct Latent {
    /// `factor[i * rank + k]`
    factor: Vec<f64>,
    rank: usize,
    n: usize,
    load: f64,
    unique: f64,
    seed: u64,
    section: usize,
}

impl Latent {
    pub fn new(p: &SimParams, s: usize, n: usize) -> Self {
        let mut r = rng(p.seed, Stream::Factor, &[s]);
        Self {
            factor: (0..n * p.copula_rank)
                .map(|_| StandardNormal.sample(&mut r))
                .collect(),
            rank: p.copula_rank,
            n,
            load: p.copula_strength.sqrt(),
            unique: (1.0 - p.copula_strength).sqrt(),
            seed: p.seed,
            section: s,
        }
    }

    /// The latent normal of gene `g`, loading on `factor`, over the spots.
    pub fn gene(&self, g: usize, factor: usize) -> Vec<f64> {
        let mut r = rng(self.seed, Stream::Unique, &[self.section, g]);
        (0..self.n)
            .map(|i| {
                let e: f64 = StandardNormal.sample(&mut r);
                self.load * self.factor[i * self.rank + factor] + self.unique * e
            })
            .collect()
    }
}
