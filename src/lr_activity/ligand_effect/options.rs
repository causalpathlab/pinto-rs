//! What a contact-effect estimate is asked to do: its options and their defaults.

use super::inference::Estimate;
use super::overlap::NuisanceOpts;

/// Overlap weight below which a receiver's exposure counts as nearly
/// determined by its confounders.
pub const NEAR_DETERMINED_H: f64 = 0.01;

/// Competence thresholds averaged over (share of an identity bin detecting
/// the receptor).
pub const COMPETENCE_GRID: [f64; 3] = [0.05, 0.1, 0.2];

/// Prior SD of an effect at one exposure threshold, on the response's scale.
pub const ABF_PRIOR_SD: f64 = 0.002;

/// Background programmes per pair, and how many of the genes nearest in expression
/// each background gene is drawn from.
pub const BACKGROUND_DRAWS: usize = 20;

pub(super) const BACKGROUND_NEIGHBOURS: usize = 20;

/// Quantiles of the positive exposures the curve visits beside τ = 0.
pub const CURVE_QUANTILES: [f64; 3] = [0.25, 0.5, 0.75];

/// Multiplier draws for the simultaneous band.
pub const BAND_DRAWS: usize = 2000;

/// Genes in the background pool at most ([`super::background::background_pool_genes`]).
pub const BACKGROUND_POOL: usize = 2000;

#[derive(Clone, Copy, Debug)]
pub struct EffectOpts {
    /// Exposure threshold τ; `None` picks it per pair
    /// ([`super::dose::choose_threshold`]).
    pub threshold: Option<f64>,
    /// Shares of an identity bin detecting the receptor for its receivers
    /// to count as competent, averaged over ([`COMPETENCE_GRID`]); the
    /// benchmark plants on one.
    pub competences: &'static [f64],
    /// Read the pair beyond its headline threshold: the threshold–response
    /// curve, and the average over those thresholds
    /// ([`super::thresholds::over_thresholds`]).
    pub curve: bool,
    pub nuisance: NuisanceOpts,
    /// Fewest receivers in either arm (exposed, unexposed) of a subgroup
    /// for its effect to be reported.
    pub min_arm: usize,
    /// Smallest effective sample size of a subgroup's overlap weights.
    pub min_ess: f64,
    /// Smallest mean overlap weight `e(1 − e)` (at most 0.25). The effective
    /// sample size cannot see weights that are all tiny alike; this can.
    pub min_overlap: f64,
    /// Prior SD of an effect at one exposure threshold, for the average over
    /// thresholds ([`super::averaging::abf_average`]).
    pub abf_prior_sd: f64,
    /// Background programmes per pair ([`super::background::BackgroundPool::draw`]).
    pub background_draws: usize,
    pub seed: u64,
    /// Run even the fits whose every subset leaves an arm thin
    /// ([`super::fits::Reads`]), which the guard then reports as not estimable all the
    /// same; for tests that check skipping them changes nothing.
    pub fit_thin: bool,
}

impl Default for EffectOpts {
    fn default() -> Self {
        Self {
            threshold: None,
            competences: &[0.1],
            curve: true,
            nuisance: NuisanceOpts::default(),
            min_arm: 20,
            min_ess: 30.0,
            min_overlap: 0.01,
            abf_prior_sd: ABF_PRIOR_SD,
            background_draws: BACKGROUND_DRAWS,
            seed: 42,
            fit_thin: false,
        }
    }
}

impl EffectOpts {
    /// The middle competence threshold, the one counts, diagnostics and
    /// the benchmark read.
    pub fn competence(&self) -> f64 {
        self.competences[self.competences.len() / 2]
    }

    /// NaN out a subgroup estimate whose arms or overlap are too thin for
    /// its standard error to mean anything: a handful of exposed receivers,
    /// or exposure nearly universal, leaves the influence-function variance
    /// resting on a few points. So does a subgroup inside one variance
    /// cluster: its clustered SE is a rounding residue with no degrees of
    /// freedom; it comes back with `df` 0 to say so ([`ONE_CLUSTER_DF`]).
    pub fn guard(&self, (e, sums): (Estimate, Vec<f64>)) -> (Estimate, Vec<f64>) {
        let thin = self.thin_arm(e.n, e.n_treated)
            // NaN fails both comparisons, so a NaN estimate counts as thin.
            || e.ess.is_nan()
            || e.ess < self.min_ess
            || e.mean_h.is_nan()
            || e.mean_h < self.min_overlap;
        let one_cluster = !thin && e.theta.is_finite() && (e.df.is_nan() || e.df <= 0.0);
        if thin || one_cluster {
            let nan = Estimate::nan(e.n, e.n_treated);
            let df = if one_cluster { ONE_CLUSTER_DF } else { nan.df };
            (Estimate { df, ..nan }, vec![0.0; sums.len()])
        } else {
            (e, sums)
        }
    }

    /// Whether `n` receivers, `treated` of them exposed, leave an arm
    /// under `min_arm` ([`thin_arm`]).
    pub fn thin_arm(&self, n: usize, treated: usize) -> bool {
        thin_arm(self.min_arm, n, treated)
    }
}

/// Whether `n` receivers, `treated`
/// of them exposed, leave either arm under `min_arm`: the arm guard
/// ([`EffectOpts::guard`]), the fits skipped as hopeless ([`super::fits::Reads`]) and
/// the verdict note all read this one rule.
pub fn thin_arm(min_arm: usize, n: usize, treated: usize) -> bool {
    treated < min_arm || n.saturating_sub(treated) < min_arm
}

/// The `df` of a not-estimable estimate whose rows lie in fewer than two
/// variance clusters, so the verdict can say why.
pub const ONE_CLUSTER_DF: f64 = 0.0;
