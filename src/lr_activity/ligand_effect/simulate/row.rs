//! One simulated run's result, and the methods compared across runs.

use super::scenario::Scenario;
use crate::lr_activity::ligand_effect::baselines::{self, Baselines};
use crate::lr_activity::ligand_effect::inference::Estimate;
use crate::lr_activity::ligand_effect::verdict::Verdict;

#[derive(Clone, Debug)]
pub struct SimRow {
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub scenario: Scenario,
    /// What the estimate targets: the realized effect contrasted between the
    /// arms the estimator sees, under its arm weights, at each threshold, and
    /// averaged over thresholds by the estimate's own weights.
    pub truth: f64,
    /// Mean effect among the truly exposed competent receivers, without the
    /// estimator. Below `truth`, it shows what misclassified exposure costs;
    /// above it, what leaning to high thresholds adds.
    pub truth_exposed: f64,
    pub estimate: Estimate,
    pub marginal_cor: Estimate,
    pub crude: Estimate,
    pub ipw: Estimate,
    /// The outcome regression alone, a point estimate: it has no interval of
    /// its own.
    pub outcome: f64,
    pub incompetent: Estimate,
    /// The competent-receiver estimate without the neighbour-response and
    /// bleed covariates; fitted only under spillover, NaN otherwise.
    pub no_spill: Estimate,
    /// The competent-receiver estimate under the partial adjustments:
    /// section only, plus the spatial smooth, plus cell identity.
    pub steps: [Estimate; 3],
    /// The competent effect minus the receptor-negative one.
    pub contrast: Estimate,
    /// The verdict `lra` would give this run alone, its q-values being its
    /// p-values: more lenient than a run with many pairs to correct over.
    pub verdict: Verdict,
    /// Share of competent receivers whose seen exposure is not their true
    /// exposure.
    pub misclassified: f64,
    /// The co-expression baselines on the seen ligand and the receptor.
    pub baselines: Baselines,
    /// Wall time of the whole run and of its baselines, in seconds.
    pub seconds: f64,
    pub baseline_seconds: f64,
}

impl SimRow {
    pub fn estimable(&self) -> bool {
        self.estimate.theta.is_finite() && self.estimate.se.is_finite()
    }
    /// The estimate rejects at 5%; a run without one does not.
    pub fn rejected(&self) -> bool {
        self.estimable() && self.estimate.rejects()
    }
    /// Whether each method of [`COMPARED`] rejects at 5%.
    pub fn rejections(&self) -> [bool; COMPARED.len()] {
        let b = |k: usize| baselines::rejects((baselines::METHODS[k].1)(&self.baselines));
        [
            self.marginal_cor.rejects(),
            self.crude.rejects(),
            self.ipw.rejects(),
            b(0),
            b(1),
            b(2),
            b(3),
            self.rejected(),
        ]
    }
}

/// The methods `lra-simulate` compares at 5%: the marginal correlation, the
/// crude contrast, overlap weights alone, the co-expression baselines
/// ([`super::super::baselines::METHODS`]) and the estimator.
pub const COMPARED: [&str; 8] = [
    "marginal", "crude", "ipw", "celltype", "moran", "contact", "coexpr", "estimate",
];
