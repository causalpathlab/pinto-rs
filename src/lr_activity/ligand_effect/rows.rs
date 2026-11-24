//! One row of the `lra` tables, in memory: a pair and section with its estimates.

use super::baselines::Baselines;
use super::effects::ScopeEffects;
use super::inference::Estimate;
use super::jackknife::Jackknife;
use super::tissue::ALL_SECTIONS;

/// A statistic's local fdr ([`crate::util::local_fdr`]) under the null
/// fitted to the bulk of its family (Efron's empirical null) and under the
/// theoretical `N(0, 1)`; NaN in a family too small to fit
/// ([`super::report::MIN_STATISTICS_FOR_LOCAL_FDR`]).
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct LocalFdr {
    pub empirical_null: f64,
    pub theoretical_null: f64,
}

impl LocalFdr {
    pub const NAN: Self = Self {
        empirical_null: f64::NAN,
        theoretical_null: f64::NAN,
    };
}

/// One output row: a pair in one section, or pooled.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct EffectRow {
    pub section: Box<str>,
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub n_targets: usize,
    pub n_removed: usize,
    pub n_receivers: usize,
    pub n_competent: usize,
    pub n_exposed: usize,
    /// The exposure threshold τ the pair used.
    pub threshold: f64,
    pub coexpr_log_or: f64,
    pub coexpr_se: f64,
    /// The correlation of the response and `log1p` contact ligand over
    /// competent receivers ([`super::association::marginal_correlation`]), combined
    /// over weightings.
    pub marginal_cor: Estimate,
    /// The estimands; the step and no-spill estimates are NaN on
    /// per-section rows.
    pub effects: ScopeEffects,
    /// Share of the variance of the exposure explained by the receiver's
    /// dominant-neighbour identity bin ([`super::dose::eta_squared`]).
    pub exposure_eta2: f64,
    /// Distinct identity bins among the competent and the incompetent
    /// receivers: competence is a function of the bin, so these say how
    /// many cell types each side of the contrast is made of.
    pub n_competent_bins: usize,
    pub n_incompetent_bins: usize,
    /// The threshold–response curve (pooled row only; empty elsewhere).
    pub curve: Vec<CurvePoint>,
    /// The co-expression baselines over the whole tissue
    /// (`--marginal-baselines`, pooled row only; `None` elsewhere).
    pub marginal: Option<Baselines>,
    /// The delete-one-tile jackknife's SE ratios, by which `effects` was
    /// widened (region rows near significance only; NaN elsewhere).
    pub jackknife: Jackknife,
    // From here on filled in by `report`, once every pair is in.
    /// Two-sided p of the headline (the competent effect) against the t reference with
    /// clusters − 1 degrees of freedom ([`super::inference::Estimate::p`]).
    pub effect_dr_p: f64,
    /// One-sided (positive) normal p of the co-expression log odds ratio,
    /// and its local fdr among the section's pairs, signal above the null only
    /// ([`super::report::report`]).
    pub coexpr_p: f64,
    pub coexpr_lfdr: LocalFdr,
    /// Pairs in the section sharing this pair's ligand.
    pub n_pairs_same_ligand: usize,
    /// Why a pair is not estimable, or that its SE was not widened by the
    /// tile jackknife.
    pub warnings: String,
}

impl EffectRow {
    /// A row holding what [`super::report::report_row`] reads, the rest
    /// blank: the estimands, co-expression and the exposure's identity share.
    /// `lra simulate` reports its runs through this.
    pub fn pending(
        effects: ScopeEffects,
        (coexpr_log_or, coexpr_se): (f64, f64),
        exposure_eta2: f64,
    ) -> Self {
        Self {
            section: ALL_SECTIONS.into(),
            ligand: "".into(),
            receptor: "".into(),
            n_targets: 0,
            n_removed: 0,
            n_receivers: 0,
            n_competent: 0,
            n_exposed: 0,
            threshold: f64::NAN,
            coexpr_log_or,
            coexpr_se,
            marginal_cor: Estimate::nan(0, 0),
            effects,
            exposure_eta2,
            n_competent_bins: 0,
            n_incompetent_bins: 0,
            curve: Vec::new(),
            marginal: None,
            jackknife: Jackknife::NONE,
            effect_dr_p: f64::NAN,
            coexpr_p: f64::NAN,
            coexpr_lfdr: LocalFdr::NAN,
            n_pairs_same_ligand: 0,
            warnings: String::new(),
        }
    }
}

/// One point of the threshold–response curve: the competent effect (pooled) at exposure
/// threshold `tau`, `quantile` being the share of positive exposures at or
/// below it; the pointwise 95% interval and the simultaneous band.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug)]
pub struct CurvePoint {
    pub tau: f64,
    pub quantile: f64,
    pub effect: Estimate,
    pub lo: f64,
    pub hi: f64,
    pub band_lo: f64,
    pub band_hi: f64,
}

impl CurvePoint {
    /// The point with its pointwise interval and a band of half-width
    /// `band_crit` × SE, never narrower than the pointwise interval.
    pub fn new(tau: f64, quantile: f64, effect: Estimate, band_crit: f64) -> Self {
        let c = effect.crit();
        let b = if band_crit.is_finite() {
            band_crit.max(c)
        } else {
            c
        };
        let (t, se) = (effect.theta, effect.se);
        let (lo, hi) = effect.interval();
        Self {
            tau,
            quantile,
            effect,
            lo,
            hi,
            band_lo: t - b * se,
            band_hi: t + b * se,
        }
    }
}

/// An estimate read two ways: on the log scale it is estimated on (the
/// signed mean `log1p` program), as a mean (θ) with SD = SE and a 95%
/// interval from the t reference (the mean and SD are the estimate's own
/// `theta` and `se`); and as a % change of the program,
/// `100 (e^θ − 1)`, with the delta-method SD `100 e^θ SE` and the log-scale
/// interval's endpoints mapped through the same function (which keeps
/// them above −100% and exact under the monotone map).
#[derive(Clone, Copy, Debug)]
pub struct Moments {
    pub lo: f64,
    pub hi: f64,
    pub pct: f64,
    pub pct_sd: f64,
    pub pct_lo: f64,
    pub pct_hi: f64,
}

impl Moments {
    pub fn of(e: &Estimate) -> Self {
        let (lo, hi) = e.interval();
        let pct = |x: f64| 100.0 * x.exp_m1();
        Self {
            lo,
            hi,
            pct: pct(e.theta),
            pct_sd: 100.0 * e.theta.exp() * e.se,
            pct_lo: pct(lo),
            pct_hi: pct(hi),
        }
    }
}
