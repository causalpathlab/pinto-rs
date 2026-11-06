//! Semi-synthetic benchmark: can the estimator recover a known effect on
//! this tissue, and does the crude contrast get fooled?
//!
//! The tissue stays real: coordinates, contacts, the confounder matrix,
//! identity bins, depth, receptor competence, and the per-bin levels of the
//! real ligand and of every real programme gene. What the truth needs is
//! simulated: who truly makes ligand, and how each programme gene responds.
//!
//! 1. **True and detected ligand.** Each cell has a latent producing state
//!    `P_i ~ Bernoulli(p_i)` and a true rate `λ_i = P_i r_b`; its detected
//!    count is `Poisson(sf_i λ_i)`. Calibration, per identity bin `b`, from
//!    the real ligand: a zero-inflated Poisson fitted by moments, matching
//!    the bin's mean `m_b` of count / size factor and its detection share
//!    `d_b`: `p_b r_b = m_b` and `p_b · mean_i(1 − e^{−sf_i r_b}) = d_b`.
//!    `p_b` is thus the detection-corrected share of producers and `r_b`
//!    the mean among them. A bin detected at least as often as Poisson
//!    with every cell producing gets `p_b = 1`, `r_b = m_b`. The rate is
//!    floored at [`R_FLOOR`] expected counts at median depth: in a bin of
//!    scattered single counts mean and detection cannot tell few producers
//!    from many weak ones, and without the floor ambient counts would make
//!    every cell a weak producer. A hidden field moves production:
//!    `logit p_i = logit p_b + α f(Z_i)`.
//!
//!    The **true exposure** is the dose version
//!    `T* = 1{Σ_contacts λ_j > τ*}`, with `τ*` from the estimator's
//!    threshold rule applied to the latent dose (so `τ* = 0`, "some contact
//!    produces", unless that leaves an arm thin). Receivers respond to `T*`;
//!    the estimator only sees detected counts, so dropout misclassifies its
//!    exposure and attenuates its estimate (column `misclassified`).
//! 2. **Response from the real programme.** Each programme gene `g`, with
//!    signed weight `w_g` (sign `s_g`), is drawn per receiver as
//!    `Poisson(sf · μ_{b,g} · e^{s_g β f(Z)} · e^{s_g (θ_b C T* + θ_ring,b C T*_ring)})`,
//!    with `μ_{b,g}` the real gene's mean count / size factor in the bin and
//!    `C` whether the receiver responds. The response is the signed
//!    weighted programme the estimator uses,
//!    `Σ_g w_g log1p(x_g / sf) / Σ_g |w_g|`. Potential outcomes are drawn
//!    from one shared uniform per receiver and gene, so their difference is
//!    the effect alone.
//! 3. **Hidden confounder** `Z`, standardized, never shown to the estimator
//!    ([`ConfounderKind`]): random Fourier features (the adjustment's own
//!    basis: a best case), random Gaussian bumps at random centres, or a
//!    real gene's expression smoothed over contacts (a field from neither
//!    the adjustment's basis nor its identity, which leaves the pair genes
//!    out). It enters through a shape ([`ConfoundShape`]): `Z` or `Z² − 1`.
//!    It acts alike in every cell type: a field that differs by cell type is
//!    not simulated, so these results do not speak to it (future work).
//!    Composition confounding comes free: `p_b`, `r_b` and `μ_{b,g}` follow identity,
//!    and neighbours' identities are correlated.
//! 4. **Truths**, two of them. `truth` is the sample average effect of
//!    `T*` on the response over competent receivers,
//!    `mean (y(T* = 1) − y(T* = 0))` at the realized ring exposure, defined
//!    without the estimator: no weights, no detected exposure. It is taken
//!    on the receivers, the scale the
//!    response lives on, as measured (with spillover's bleed, below).
//!    `truth_weighted` is the same contrast under the
//!    estimator's own overlap weights `h`, the population its weighting
//!    targets; the two part where the effect is heterogeneous and the
//!    weights move away from the competent receivers at large.
//! 5. **Heterogeneous effect**: `θ_b = θ u_b`, `u_b` uniform on `[0.5, 1.5]`
//!    per identity bin.
//! 6. **Marker**: the ligand is made only by a "marker type", the top
//!    quarter of identity bins by real ligand, which alone responds and
//!    carries a programme baseline raised by [`MARKER_LIFT`]; contacts stay
//!    real, so the type's homotypic contacts make its cells the exposed
//!    ones. The receptor call the estimator uses is the real one plus the
//!    marker type. Exposure then tracks cell type: the crude contrast should
//!    reject almost always, the estimator should find no overlap in the
//!    marker type (not estimable, or near 0 under no effect), and power
//!    under an effect is low.
//!
//! 7. **Confounded receptor** (on in every scenario of the default grid,
//!    `confound_receptor`; off in [`Scenario::default`], which keeps the
//!    real receptor). The receptor is simulated too, so the hidden field
//!    that drives ligand and response also drives the receptor:
//!    `Poisson(sf · ρ_b · e^{β f(Z)} / mean e^{β f(Z)})`, with `ρ_b` the real
//!    receptor's mean count / size factor in the bin and `β` the response's
//!    confounding strength. Without it the co-expression baselines
//!    ([`super::baselines`]) would see a receptor the field never touches
//!    and could not be fooled by it. The receptor does nothing to the
//!    response: competence (who responds, and whom the estimator calls
//!    competent) is read by the estimator's own rule from the simulated
//!    receptor's detection, as `pinto lra` reads it from the real one.
//!
//! Every run also scores the co-expression baselines of
//! [`super::baselines`] on the ligand and receptor the estimator sees.
//!
//! Three departures, each off by default:
//!
//! - **Spillover** κ: segmentation hands each receiver a share κ of every
//!   contact neighbour's programme counts, and of their ligand counts in
//!   its own-ligand covariate. It changes the measurement, not the effect:
//!   the truths are the same contrasts on the measured scale, each
//!   potential count carrying the realized bleed from the neighbours'
//!   observed counts (bleed inside `log1p` compresses a contrast, and an
//!   unbled truth would charge that compression to the estimator). The
//!   neighbour-response and bleed covariates are what should absorb it,
//!   and a fit without them is reported beside.
//! - **Reach** `θ_ring`: a second effect from ring partners (`T*_ring`).
//!   At 0 the effect is contact-only and the ring estimate should sit at
//!   0; its truth is the same sample average contrast, in `T*_ring`.
//! - **Shift**: the estimator sees the detected ligand field displaced by
//!   a few contact lengths (each cell reads the cell nearest its point plus
//!   the offset, in its own section). Receivers still respond to the
//!   undisplaced field, so the seen exposure has no effect (truths 0); a
//!   contact effect should vanish while confounding at a coarser scale
//!   still fools the crude contrast.

use rayon::prelude::*;

use super::baselines::{self, baselines, Baselines};
use super::confounders::fourier_features;
use super::estimate::{
    choose_threshold, eta_squared, expected_bleed, exposed, indicator, over_thresholds, report,
    thresholds, EffectOpts, EffectRow, Exposure, Response, Rows, Sensitivity,
};
use super::exposure::competence;
use super::output::Verdict;
use super::overlap::{crude_difference, Estimate};
use super::regression::standardize;
use super::tissue::Tissue;
use super::INTERNALS;
use crate::util::common::HashMap;
use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson, StandardNormal};
use special::Error as _;

/// Floor on a producer's true rate, in expected counts at median depth.
pub const R_FLOOR: f64 = 0.5;

/// Log-rate lift of the programme's baseline in the marker type (signed
/// per gene, about a doubling).
pub const MARKER_LIFT: f64 = 0.7;

/// Two-sided 5% normal critical value, for the crude contrast's naive
/// interval; the estimates use their t reference ([`Estimate::crit`]).
pub const Z_CRIT: f64 = super::overlap::Z_975;

// * Scenarios

/// The hidden confounder's spatial form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConfounderKind {
    /// Random Fourier features, the adjustment smooth's own basis
    Fourier,
    /// Random Gaussian bumps of width `scale` contact lengths, random signs
    Bumps,
    /// A real gene outside the benchmarked pairs, smoothed over contacts
    RealGene,
}

impl ConfounderKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fourier => "fourier",
            Self::Bumps => "bumps",
            Self::RealGene => "realgene",
        }
    }
}

impl std::str::FromStr for ConfounderKind {
    type Err = anyhow::Error;

    /// A kind by its [`label`](Self::label).
    fn from_str(s: &str) -> anyhow::Result<Self> {
        [Self::Fourier, Self::Bumps, Self::RealGene]
            .into_iter()
            .find(|k| k.label() == s)
            .ok_or_else(|| anyhow::anyhow!("unknown confounder {s:?}"))
    }
}

/// How the hidden field enters ligand production and response.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConfoundShape {
    /// Z
    Linear,
    /// Z² − 1
    Quadratic,
}

impl ConfoundShape {
    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
        }
    }
}

impl std::str::FromStr for ConfoundShape {
    type Err = anyhow::Error;

    /// A shape by its [`label`](Self::label).
    fn from_str(s: &str) -> anyhow::Result<Self> {
        [Self::Linear, Self::Quadratic]
            .into_iter()
            .find(|k| k.label() == s)
            .ok_or_else(|| anyhow::anyhow!("unknown confound shape {s:?}"))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Scenario {
    /// θ: log-rate change of the programme in exposed responding receivers.
    pub effect: f64,
    /// θ_ring: log-rate change in responding receivers whose ring partners
    /// make ligand; 0 keeps the effect at contact range.
    pub ring_effect: f64,
    /// α: hidden field's push on ligand production (log odds).
    pub confound_ligand: f64,
    /// β: hidden field's push on the response.
    pub confound_response: f64,
    /// Hidden field's length scale, in contact lengths.
    pub scale: f64,
    pub confounder: ConfounderKind,
    pub shape: ConfoundShape,
    /// θ varies by identity bin, `θ u_b` with `u_b ~ U[0.5, 1.5]`.
    pub heterogeneous: bool,
    /// Ligand, response and programme baseline confined to a marker type.
    pub marker: bool,
    /// κ: share of each contact neighbour's counts measured in the receiver.
    pub spillover: f64,
    /// Displacement of the ligand field the estimator sees, in contact
    /// lengths; 0 for none.
    pub shift: f64,
    /// Simulate the receptor, driven by the hidden field at strength β.
    pub confound_receptor: bool,
    /// How much of the field's push on the response (β) reaches the background programme
    /// programme: 1 the same push, 0 none.
    pub background_confounding: f64,
    /// The planted effect grows with the true dose (a Hill curve) rather than
    /// being a step at any exposure.
    pub dose_response: bool,
    /// Where the dose response is half way, as a quantile of the positive true
    /// doses, and how sharply it rises (the Hill order).
    pub dose_midpoint: f64,
    pub dose_order: i32,
    /// γ: each cell has a hidden trait that raises its ligand and, by γ per unit
    /// of its contacts' mean trait, every gene of its competent contacts, background programmes
    /// included. It acts at contact range on every programme, so neither the
    /// receptor-negative receivers nor distant cells can show it; 0 for none.
    pub generic_shift: f64,
    pub rep: usize,
}

impl Default for Scenario {
    /// No effect, no confounding, no departure. The confounder kind only
    /// matters once `α` or `β` is set: Fourier, the original field, unless
    /// a scenario names another (the grids name theirs).
    fn default() -> Self {
        Self {
            effect: 0.0,
            ring_effect: 0.0,
            confound_ligand: 0.0,
            confound_response: 0.0,
            scale: 32.0,
            confounder: ConfounderKind::Fourier,
            shape: ConfoundShape::Linear,
            heterogeneous: false,
            marker: false,
            spillover: 0.0,
            shift: 0.0,
            confound_receptor: false,
            background_confounding: 1.0,
            dose_response: false,
            dose_midpoint: 0.5,
            dose_order: 2,
            generic_shift: 0.0,
            rep: 0,
        }
    }
}

/// The choices a benchmark grid is built from: the `scenarios` section of
/// a study. Fields left out of the JSON keep their defaults; unknown ones
/// are an error, so a misspelt field does not silently fall back.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GridOpts {
    #[serde(rename = "_doc", skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    /// Planted effects θ, as log-rate changes of the programme; each runs
    /// beside θ = 0.
    pub effect_sizes: Vec<f64>,
    /// Replicates of every scenario (set from the study's `reps`).
    #[serde(skip)]
    pub reps: usize,
    pub confounders: Vec<ConfounderKind>,
    /// Confounder length scales, in contact lengths; the first is the one
    /// the nonlinear shapes and the shifted field use.
    pub scales: Vec<f64>,
    pub shapes: Vec<ConfoundShape>,
    /// Segmentation spillover shares κ, each its own unconfounded scenario.
    pub spillover: Vec<f64>,
    /// Displacements of the ligand field the estimator sees, in contact
    /// lengths, each with the first confounder at the first scale.
    pub shift: Vec<f64>,
    /// Ligand, response and programme confined to one cell type.
    pub marker: bool,
    /// An effect varying by identity bin (planted effects only).
    pub heterogeneous: bool,
    /// Planted effects that also reach ring range (θ_ring), each its own
    /// unconfounded scenario.
    pub ring_effect: Vec<f64>,
    /// Simulate the receptor in every scenario, driven by the hidden field
    /// at the response's strength; false keeps the real receptor.
    pub confound_receptor: bool,
    /// How much of the field's push on the response reaches the background programme
    /// programme: 1 the same push, 0 none.
    pub background_confounding: f64,
    /// Planted effects grow with the true dose.
    pub dose_response: bool,
    /// Where it is half way, as a quantile of the positive true doses.
    pub dose_midpoint: f64,
    /// How sharply it rises (the Hill order).
    pub dose_order: i32,
    /// Each a scenario with a generic contact-range shift of this size.
    pub generic_shift: Vec<f64>,
}

impl Default for GridOpts {
    fn default() -> Self {
        Self {
            doc: None,
            effect_sizes: vec![0.3],
            reps: 5,
            confounders: vec![ConfounderKind::Bumps, ConfounderKind::RealGene],
            scales: vec![32.0, 4.0],
            shapes: vec![ConfoundShape::Linear, ConfoundShape::Quadratic],
            spillover: vec![0.1],
            shift: vec![4.0],
            marker: true,
            heterogeneous: true,
            ring_effect: vec![0.3],
            confound_receptor: true,
            background_confounding: 1.0,
            dose_response: false,
            dose_midpoint: 0.5,
            dose_order: 2,
            generic_shift: Vec::new(),
        }
    }
}

/// The grid: no effect and each planted one, each with
///
/// - no confounding;
/// - per confounder kind: linear confounding at every scale, same-sign and
///   opposite-sign (`α = 1, β = ±1`), and each nonlinear shape at the
///   first scale, same-sign;
/// - a heterogeneous effect (planted effects only);
/// - unconfounded segmentation spillover at each κ, and an effect that also
///   reaches ring range at each θ_ring;
/// - the first confounder at the first scale, seen shifted by each shift;
/// - the marker type, unconfounded.
pub fn default_grid(g: &GridOpts) -> Vec<Scenario> {
    let mut out = Vec::new();
    let first_scale = g.scales.first().copied().unwrap_or(32.0);
    let thetas = std::iter::once(0.0).chain(g.effect_sizes.iter().copied().filter(|&t| t != 0.0));
    for theta in thetas {
        for rep in 0..g.reps {
            let plain = Scenario {
                effect: theta,
                confound_receptor: g.confound_receptor,
                background_confounding: g.background_confounding,
                dose_response: g.dose_response && theta != 0.0,
                dose_midpoint: g.dose_midpoint,
                dose_order: g.dose_order,
                rep,
                ..Scenario::default()
            };
            out.push(plain);
            for &confounder in &g.confounders {
                let conf = Scenario {
                    confounder,
                    confound_ligand: 1.0,
                    confound_response: 1.0,
                    ..plain
                };
                for (k, &scale) in g.scales.iter().enumerate() {
                    for &shape in &g.shapes {
                        if shape == ConfoundShape::Linear {
                            out.push(Scenario { scale, ..conf });
                            out.push(Scenario {
                                scale,
                                confound_response: -1.0,
                                ..conf
                            });
                        } else if k == 0 {
                            out.push(Scenario {
                                scale,
                                shape,
                                ..conf
                            });
                        }
                    }
                }
            }
            if theta != 0.0 && g.heterogeneous {
                out.push(Scenario {
                    heterogeneous: true,
                    ..plain
                });
            }
            for &spillover in &g.spillover {
                out.push(Scenario { spillover, ..plain });
            }
            for &generic_shift in &g.generic_shift {
                out.push(Scenario {
                    generic_shift,
                    ..plain
                });
            }
            for &ring_effect in &g.ring_effect {
                out.push(Scenario {
                    ring_effect,
                    ..plain
                });
            }
            if let Some(&confounder) = g.confounders.first() {
                for &shift in &g.shift {
                    out.push(Scenario {
                        confounder,
                        confound_ligand: 1.0,
                        confound_response: 1.0,
                        scale: first_scale,
                        shift,
                        ..plain
                    });
                }
            }
            if g.marker {
                out.push(Scenario {
                    marker: true,
                    ..plain
                });
            }
        }
    }
    out
}

// * Study

/// A `pinto lra-simulate` study: the scenario grid, its replicates, and how
/// many pairs (the most abundant distinct ligands) it runs on.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Study {
    #[serde(rename = "_doc", skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    /// Replicates of every scenario.
    pub reps: usize,
    /// Pairs benchmarked: the most abundant distinct ligands.
    pub pairs: usize,
    pub scenarios: GridOpts,
}

impl Default for Study {
    fn default() -> Self {
        Self {
            doc: None,
            reps: 3,
            pairs: 3,
            scenarios: GridOpts::default(),
        }
    }
}

impl Study {
    /// The grid options with the study's replicates.
    pub fn grid(&self) -> GridOpts {
        GridOpts {
            reps: self.reps,
            ..self.scenarios.clone()
        }
    }

    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let s: Study = serde_json::from_str(text)?;
        anyhow::ensure!(s.reps > 0, "reps must be at least 1");
        anyhow::ensure!(s.pairs > 0, "pairs must be at least 1");
        anyhow::ensure!(
            s.scenarios.scales.iter().all(|&x| x > 0.0),
            "scenarios.scales must be positive contact lengths"
        );
        Ok(s)
    }

    pub fn read(path: &str) -> anyhow::Result<Self> {
        let text =
            std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("--study {path}: {e}"))?;
        Self::from_json(&text).map_err(|e| anyhow::anyhow!("--study {path}: {e}"))
    }

    /// The built-in study as JSON, each section explained in its `_doc`
    /// (JSON has no comments). It parses back to [`Study::default`].
    pub fn example_json() -> String {
        let study = Study {
            doc: Some(
                "pinto lra-simulate study. Plants ligand effects of known size on this \
                 tissue, with and without hidden spatial confounders, and checks whether \
                 lra finds them (power) and stays quiet when there is nothing (type I \
                 error). reps: replicates of every scenario. pairs: how many ligand-receptor \
                 pairs to plant on (the most abundant distinct ligands). Fields left out \
                 keep these values; _doc fields are ignored."
                    .into(),
            ),
            scenarios: GridOpts {
                doc: Some(
                    "Each effect size runs beside no effect. effect_sizes: log-rate change \
                     of the response programme (0.3 is about +35%). confounders: hidden \
                     field kinds (bumps, realgene, fourier). scales: their length scales in \
                     contact lengths. shapes: how the field acts (linear, quadratic). \
                     spillover: shares of a neighbour's counts bleeding into \
                     the receiver. shift: displacement of the ligand field, in contact \
                     lengths (a null). marker: effect confined to one cell type. \
                     heterogeneous: effect varying by cell type. ring_effect: effect sizes \
                     reaching distant, non-touching cells. confound_receptor: simulate \
                     the receptor too, driven by the same hidden field (false keeps the \
                     real receptor, which the field cannot touch). background_confounding: how \
                     much of the field's push on the response reaches the background programme \
                     (1 the same push, 0 none). dose_response: planted effects grow with the \
                     true dose instead of being a step at any exposure, half way at the \
                     quantile dose_midpoint of the positive doses and rising with the Hill \
                     order dose_order. generic_shift: sizes of a contact-range shift of every gene \
                     in competent receivers, tied to the ligand by a hidden trait of the \
                     neighbours."
                        .into(),
                ),
                ..GridOpts::default()
            },
            ..Study::default()
        };
        serde_json::to_string_pretty(&study).expect("a study serializes")
    }
}

// * Results

#[derive(Clone, Debug)]
pub struct SimRow {
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub scenario: Scenario,
    /// Sample average effect of the true exposure over competent receivers
    /// on the analysed rows; independent of the estimator.
    /// Over all competent receivers: under a dose response, those with no true
    /// dose count at zero, so this sits below the effect among the exposed.
    pub truth: f64,
    /// The same contrast under the overlap weights at the headline threshold.
    pub truth_weighted: f64,
    pub estimate: Estimate,
    pub crude: f64,
    pub crude_se: f64,
    pub incompetent: Estimate,
    pub ring: Estimate,
    /// Sample average effect of the true ring exposure over competent
    /// receivers.
    pub ring_truth: f64,
    /// The competent-receiver estimate without the neighbour-response and
    /// bleed covariates; fitted only under spillover, NaN otherwise.
    pub no_spill: Estimate,
    /// The competent-receiver estimate under the partial adjustments:
    /// section only, plus the spatial smooth, plus cell identity.
    pub steps: [Estimate; 3],
    /// The competent effect minus the receptor-negative one.
    pub contrast: Estimate,
    /// The competent effect minus the ring effect.
    pub over_ring: Estimate,
    /// The competent effect on the simulated background programme.
    pub background: Estimate,
    /// The competent effect minus the background's.
    pub over_background: Estimate,
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
    pub fn crude_rejected(&self) -> bool {
        (self.crude / self.crude_se).abs() > Z_CRIT
    }
    /// Whether each method of [`COMPARED`] rejects at 5%.
    pub fn rejections(&self) -> [bool; 6] {
        let b = |k: usize| baselines::rejects((baselines::METHODS[k].1)(&self.baselines));
        [
            self.crude_rejected(),
            b(0),
            b(1),
            b(2),
            b(3),
            self.rejected(),
        ]
    }
}

/// The methods `lra-simulate` compares at 5%: the crude contrast, the
/// co-expression baselines ([`super::baselines::METHODS`]) and the estimator.
pub const COMPARED: [&str; 6] = [
    "crude", "celltype", "moran", "contact", "coexpr", "estimate",
];

// * Real ingredients

/// Real ingredients of one pair that the simulator keeps.
pub struct SimBase<'a> {
    pub ligand: &'a str,
    pub receptor: &'a str,
    /// The real ligand's counts per row: only its per-bin mean and
    /// detection are used.
    pub ligand_counts: &'a [f64],
    /// Signed weight of each programme gene.
    pub target_weights: &'a [f64],
    /// Each programme gene's mean count / size factor per identity bin,
    /// bins × genes, row-major ([`target_bin_means`]).
    pub target_bin_means: &'a [f64],
    pub competent: &'a [bool],
    /// The real receptor's counts per row (the scarcest subunit's): its
    /// per-bin mean calibrates the simulated receptor, and it is what the
    /// baselines read when the receptor is not simulated.
    pub receptor_counts: &'a [f64],
    /// Counts per row of a real gene outside the benchmarked pairs, for
    /// [`ConfounderKind::RealGene`].
    pub confounder_gene: Option<&'a [f64]>,
}

fn n_bins(tissue: &Tissue) -> usize {
    tissue
        .identity_bin
        .iter()
        .copied()
        .max()
        .map_or(0, |b| b + 1)
}

/// Mean count / size factor of each gene (non-zeros by row) per identity
/// bin: bins × genes, row-major.
pub fn target_bin_means(tissue: &Tissue, genes: &[&[(u32, f32)]]) -> Vec<f64> {
    let nb = n_bins(tissue);
    let ng = genes.len();
    let mut cells = vec![0usize; nb];
    for &b in &tissue.identity_bin {
        cells[b] += 1;
    }
    let mut out = vec![0.0; nb * ng];
    for (g, nz) in genes.iter().enumerate() {
        for &(r, x) in *nz {
            let r = r as usize;
            out[tissue.identity_bin[r] * ng + g] += x as f64 / tissue.size_factor[r];
        }
    }
    for b in 0..nb {
        for v in &mut out[b * ng..(b + 1) * ng] {
            *v /= cells[b].max(1) as f64;
        }
    }
    out
}

/// One bin's calibrated producers: share `p`, rate among producers `r`.
#[derive(Clone, Copy, Debug)]
pub struct Producers {
    pub p: f64,
    pub r: f64,
    pub cells: usize,
}

/// The zero-inflated Poisson moment fit of the module docs, per bin.
pub fn calibrate_producers(tissue: &Tissue, counts: &[f64]) -> Vec<Producers> {
    let sf = &tissue.size_factor;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_bins(tissue)];
    for (i, &b) in tissue.identity_bin.iter().enumerate() {
        members[b].push(i);
    }
    members
        .iter()
        .map(|rows| {
            let k = rows.len() as f64;
            let none = Producers {
                p: 0.0,
                r: 0.0,
                cells: rows.len(),
            };
            if rows.is_empty() {
                return none;
            }
            let m = rows.iter().map(|&i| counts[i] / sf[i]).sum::<f64>() / k;
            let d = rows.iter().filter(|&&i| counts[i] > 0.0).count() as f64 / k;
            if m <= 0.0 || d <= 0.0 {
                return none;
            }
            let detect = |r: f64| rows.iter().map(|&i| 1.0 - (-sf[i] * r).exp()).sum::<f64>() / k;
            // Predicted minus observed detection at producer rate r, with
            // the share set by the mean; decreasing in r.
            let excess = |r: f64| m / r * detect(r) - d;
            let r = if excess(m) <= 0.0 {
                m
            } else {
                let (mut lo, mut hi) = (m, 2.0 * m);
                while excess(hi) > 0.0 && hi < 1e6 * m.max(1.0) {
                    lo = hi;
                    hi *= 2.0;
                }
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if excess(mid) > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                0.5 * (lo + hi)
            };
            let r = r.max(R_FLOOR);
            Producers {
                p: (m / r).min(1.0),
                r,
                cells: rows.len(),
            }
        })
        .collect()
}

/// The marker type: the top quarter (at least one) of occupied identity
/// bins by mean ligand.
pub fn marker_bins(cal: &[Producers]) -> Vec<bool> {
    let mut occupied: Vec<usize> = (0..cal.len()).filter(|&b| cal[b].cells > 0).collect();
    occupied.sort_by(|&x, &y| (cal[y].p * cal[y].r).total_cmp(&(cal[x].p * cal[x].r)));
    let k = ((occupied.len() as f64 / 4.0).round() as usize).max(1);
    let mut out = vec![false; cal.len()];
    for &b in occupied.iter().take(k) {
        out[b] = true;
    }
    out
}

// * Hidden fields

/// Random Fourier features at `scale` contact lengths.
fn fourier_field(tissue: &Tissue, scale: f64, seed: u64) -> Vec<f64> {
    let d = 64;
    let feats = fourier_features(
        &tissue.coords,
        &tissue.section,
        &tissue.unit,
        scale,
        d,
        seed,
    );
    let mut rng = SmallRng::seed_from_u64(mix_seed(seed, 7));
    let a: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
    let z: Vec<f64> = (0..tissue.n())
        .map(|i| feats.iter().zip(&a).map(|(f, a)| f[i] * a).sum())
        .collect();
    standardize(&z)
}

/// A sum of Gaussian bumps of width `scale` contact lengths at random
/// cells, with random signs and amplitudes in `[0.5, 1.5]`: about two per
/// width² of tissue (a cell per contact length²), cut at three widths.
fn bump_field(tissue: &Tissue, scale: f64, seed: u64) -> Vec<f64> {
    let n = tissue.n();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut z = vec![0.0; n];
    for s in 0..tissue.unit.len() {
        let rows: Vec<usize> = (0..n)
            .filter(|&i| tissue.section[i] as usize == s)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let w = scale * tissue.unit[s];
        let cut = 3.0 * w;
        let key = |x: f64, y: f64| ((x / cut).floor() as i64, (y / cut).floor() as i64);
        let k = ((2.0 * rows.len() as f64 / (scale * scale)).round() as usize).max(2);
        // Bumps `[x, y, signed amplitude]` by grid square.
        let mut grid: HashMap<(i64, i64), Vec<[f64; 3]>> = HashMap::default();
        for _ in 0..k {
            let c = rows[rng.random_range(0..rows.len())];
            let (x, y) = tissue.coords[c];
            let sign = if rng.random::<bool>() { 1.0 } else { -1.0 };
            let amp = sign * (0.5 + rng.random::<f64>());
            grid.entry(key(x as f64, y as f64))
                .or_default()
                .push([x as f64, y as f64, amp]);
        }
        for &i in &rows {
            let (x, y) = (tissue.coords[i].0 as f64, tissue.coords[i].1 as f64);
            let (gx, gy) = key(x, y);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for &[bx, by, amp] in grid.get(&(gx + dx, gy + dy)).into_iter().flatten() {
                        let d2 = (x - bx).powi(2) + (y - by).powi(2);
                        if d2 < cut * cut {
                            z[i] += amp * (-d2 / (2.0 * w * w)).exp();
                        }
                    }
                }
            }
        }
    }
    standardize(&z)
}

/// A real gene's `log1p(count / sf)`, smoothed by `scale / 2` rounds of
/// averaging each row with its contacts.
fn gene_field(tissue: &Tissue, counts: &[f64], scale: f64) -> Vec<f64> {
    let n = tissue.n();
    let sf = &tissue.size_factor;
    let mut v: Vec<f64> = (0..n).map(|i| (counts[i] / sf[i]).ln_1p()).collect();
    for _ in 0..((scale / 2.0).round() as usize).max(1) {
        let s = tissue.contacts.sum(&v);
        for i in 0..n {
            v[i] = (v[i] + s[i]) / (1 + tissue.contacts.degree(i)) as f64;
        }
    }
    standardize(&v)
}

/// What every scenario of a pair shares: its calibrated producers
/// ([`calibrate_producers`]) and the real-gene field at each scale the
/// grid uses ([`ConfounderKind::RealGene`]), computed once.
pub struct SimShared<'a> {
    pub producers: Vec<Producers>,
    /// `(scale, field)` ([`gene_fields`]); a scale missing here is
    /// computed when a scenario needs it.
    pub gene_fields: &'a [(f64, Vec<f64>)],
}

/// The real-gene field at each distinct scale; empty without a gene.
pub fn gene_fields(tissue: &Tissue, gene: Option<&[f64]>, scales: &[f64]) -> Vec<(f64, Vec<f64>)> {
    let Some(g) = gene else {
        return Vec::new();
    };
    let mut out: Vec<(f64, Vec<f64>)> = Vec::new();
    for &scale in scales {
        if !out.iter().any(|(s, _)| *s == scale) {
            out.push((scale, gene_field(tissue, g, scale)));
        }
    }
    out
}

/// The scenario's hidden field through its shape: `f(Z)` per row.
fn confounding(
    tissue: &Tissue,
    base: &SimBase<'_>,
    shared: &SimShared<'_>,
    scen: &Scenario,
    seed: u64,
) -> anyhow::Result<Vec<f64>> {
    let n = tissue.n();
    if scen.confound_ligand == 0.0 && scen.confound_response == 0.0 {
        return Ok(vec![0.0; n]);
    }
    let z = match scen.confounder {
        ConfounderKind::Fourier => fourier_field(tissue, scen.scale, seed),
        ConfounderKind::Bumps => bump_field(tissue, scen.scale, seed),
        ConfounderKind::RealGene => {
            let g = base
                .confounder_gene
                .ok_or_else(|| anyhow::anyhow!("no real gene given for the confounder"))?;
            match shared.gene_fields.iter().find(|(s, _)| *s == scen.scale) {
                Some((_, z)) => z.clone(),
                None => gene_field(tissue, g, scen.scale),
            }
        }
    };
    Ok((0..n)
        .map(|i| match scen.shape {
            ConfoundShape::Linear => z[i],
            ConfoundShape::Quadratic => z[i] * z[i] - 1.0,
        })
        .collect())
}

// * Simulation

/// A background programme matched gene by gene to the real one: the same weights
/// and bin means, no planted effect, the field's push on the response scaled
/// by `share`, and the `generic` lift of each row. Counts come from their own stream, so the background changes nothing
/// else in the run.
#[allow(clippy::too_many_arguments)]
pub fn simulate_background(
    tissue: &Tissue,
    base: &SimBase<'_>,
    lifts: &[Vec<f64>; 2],
    kappa: f64,
    seed: u64,
) -> Vec<Vec<(u32, f32)>> {
    let n = tissue.n();
    let (sf, bin) = (&tissue.size_factor, &tissue.identity_bin);
    let ng = base.target_weights.len();
    let mut rng = SmallRng::seed_from_u64(seed);
    base.target_weights
        .iter()
        .enumerate()
        .map(|(g, &w)| {
            let lift = &lifts[(w < 0.0) as usize];
            let counts: Vec<f64> = (0..n)
                .map(|i| {
                    poisson(
                        &mut rng,
                        sf[i] * base.target_bin_means[bin[i] * ng + g] * lift[i],
                    )
                })
                .collect();
            if kappa > 0.0 {
                let bled = tissue.contacts.sum(&counts);
                nonzeros(counts.iter().zip(&bled).map(|(c, b)| c + kappa * b))
            } else {
                nonzeros(counts.into_iter())
            }
        })
        .collect()
}

/// What multiplies a gene's bin mean at each row, for genes of positive then
/// of negative weight: the field `f` at strength `beta` (normalized to keep the
/// mean), the marker types' lift and the shift `generic`. Real and background genes
/// are generated through this alike.
pub(super) fn sign_lifts(
    f: &[f64],
    beta: f64,
    generic: &[f64],
    marker: &[bool],
    bin: &[usize],
) -> [Vec<f64>; 2] {
    [1.0, -1.0].map(|s| {
        let norm = f.iter().map(|z| (s * beta * z).exp()).sum::<f64>() / f.len().max(1) as f64;
        (0..f.len())
            .map(|i| {
                let m = if marker[bin[i]] { s * MARKER_LIFT } else { 0.0 };
                (s * beta * f[i] + generic[i] + m).exp() / norm
            })
            .collect()
    })
}

/// The Poisson quantile at `u`: exact by summing the pmf for small rates,
/// normal with continuity correction above 30.
fn poisson_quantile(u: f64, lam: f64) -> f64 {
    if lam <= 0.0 {
        return 0.0;
    }
    if lam > 30.0 {
        let z = std::f64::consts::SQRT_2 * (2.0 * u - 1.0).clamp(-0.999_999, 0.999_999).inv_error();
        return (lam + lam.sqrt() * z + 0.5).floor().max(0.0);
    }
    let mut k = 0.0;
    let mut p = (-lam).exp();
    let mut cdf = p;
    while cdf < u && k < 1000.0 {
        k += 1.0;
        p *= lam / k;
        cdf += p;
    }
    k
}

fn poisson(rng: &mut SmallRng, rate: f64) -> f64 {
    if rate > 0.0 {
        Poisson::new(rate).map(|p| p.sample(rng)).unwrap_or(0.0)
    } else {
        0.0
    }
}

/// Non-zero entries of a count vector.
fn nonzeros(c: impl Iterator<Item = f64>) -> Vec<(u32, f32)> {
    c.enumerate()
        .filter(|(_, v)| *v > 0.0)
        .map(|(i, v)| (i as u32, v as f32))
        .collect()
}

/// Mean of `a − b` over `subset`; NaN for an empty subset.
fn sample_effect(a: &[f64], b: &[f64], subset: &[bool]) -> f64 {
    let (mut s, mut k) = (0.0, 0usize);
    for i in (0..a.len()).filter(|&i| subset[i]) {
        s += a[i] - b[i];
        k += 1;
    }
    if k == 0 {
        f64::NAN
    } else {
        s / k as f64
    }
}

/// One run of `scen` on `base`, with the pair's shared ingredients
/// computed once ([`SimShared`]); see the module doc.
pub fn simulate(
    tissue: &Tissue,
    base: &SimBase<'_>,
    shared: &SimShared<'_>,
    scen: &Scenario,
    opts: &EffectOpts,
    seed: u64,
) -> anyhow::Result<SimRow> {
    let start = std::time::Instant::now();
    let n = tissue.n();
    let sf = &tissue.size_factor;
    let bin = &tissue.identity_bin;
    let nb = n_bins(tissue);
    let ng = base.target_weights.len();
    anyhow::ensure!(ng > 0, "the programme has no genes");
    anyhow::ensure!(
        base.target_bin_means.len() == nb * ng,
        "programme bin means are {} long, expected {nb} bins × {ng} genes",
        base.target_bin_means.len()
    );
    let seed = mix_seed(seed, scen.rep as u64);
    let f = confounding(tissue, base, shared, scen, mix_seed(seed, 1))?;

    // Who truly makes ligand, and what is detected of it.
    let mut cal = shared.producers.clone();
    let marker = if scen.marker {
        marker_bins(&cal)
    } else {
        vec![false; nb]
    };
    if scen.marker {
        for (c, &m) in cal.iter_mut().zip(&marker) {
            if !m {
                c.p = 0.0;
            }
        }
    }
    // The generic shift's hidden trait, and each row's contacts' mean of it.
    let trait_z: Vec<f64> = if scen.generic_shift != 0.0 {
        let mut r = SmallRng::seed_from_u64(mix_seed(seed, 10));
        (0..n).map(|_| StandardNormal.sample(&mut r)).collect()
    } else {
        vec![0.0; n]
    };
    let trait_near = tissue.contacts.mean(&trait_z);
    let mut rng = SmallRng::seed_from_u64(mix_seed(seed, 2));
    let lam_true: Vec<f64> = (0..n)
        .map(|i| {
            let Producers { p, r, .. } = cal[bin[i]];
            let pi = if p <= 0.0 {
                0.0
            } else {
                let q = p.clamp(1e-9, 1.0 - 1e-9);
                let eta = (q / (1.0 - q)).ln()
                    + scen.confound_ligand * f[i]
                    + scen.generic_shift * trait_z[i];
                1.0 / (1.0 + (-eta).exp())
            };
            if rng.random::<f64>() < pi {
                r
            } else {
                0.0
            }
        })
        .collect();
    let ligand: Vec<f64> = (0..n)
        .map(|i| poisson(&mut rng, sf[i] * lam_true[i]))
        .collect();

    // What receivers respond to: the latent dose.
    let a_star = tissue.contacts.sum(&lam_true);
    let tau_star = choose_threshold(&a_star, opts);
    let t_star = exposed(&a_star, tau_star);
    let r_star = exposed(&tissue.ring_sum(&lam_true), tau_star);
    // How much of the planted effect each receiver's true dose reaches.
    let dose_share: Vec<f64> = if scen.dose_response {
        let mut pos: Vec<f64> = a_star.iter().copied().filter(|&v| v > 0.0).collect();
        pos.sort_by(f64::total_cmp);
        let (q, order) = (scen.dose_midpoint, scen.dose_order);
        let k = pos
            .get(((pos.len() as f64 - 1.0) * q) as usize)
            .copied()
            .unwrap_or(1.0);
        a_star
            .iter()
            .map(|&v| {
                let (x, m) = (v.powi(order), k.powi(order));
                x / (x + m).max(1e-300)
            })
            .collect()
    } else {
        vec![1.0; n]
    };

    // What the estimator sees: detected counts, possibly displaced,
    // possibly with neighbours' ligand bled into the own-ligand covariate.
    let lnorm: Vec<f64> = (0..n).map(|i| ligand[i] / sf[i]).collect();
    let shifted = scen.shift > 0.0;
    let seen = if shifted {
        shift_field(tissue, &lnorm, scen.shift, mix_seed(seed, 3))
    } else {
        lnorm
    };
    // The estimator's own exposure rule on the counts it sees.
    let seen_counts: Vec<f64> = (0..n).map(|i| seen[i] * sf[i]).collect();
    let mut exposure = Exposure::build(tissue, &seen_counts, opts);

    // The receptor the estimator and the baselines see, and competence read
    // from it by the estimator's own rule.
    let sim_receptor;
    let (receptor, competent): (&[f64], Vec<bool>) = if scen.confound_receptor {
        sim_receptor = simulate_receptor(
            tissue,
            base.receptor_counts,
            &f,
            scen.confound_response,
            mix_seed(seed, 6),
        );
        let detected: Vec<bool> = sim_receptor.iter().map(|&r| r > 0.0).collect();
        (&sim_receptor, competence(bin, &detected, opts.competence()))
    } else {
        (base.receptor_counts, base.competent.to_vec())
    };
    let clock = std::time::Instant::now();
    let coexpression_baselines = baselines(tissue, &seen_counts, receptor, mix_seed(seed, 8));
    let baseline_seconds = clock.elapsed().as_secs_f64();
    let kappa = scen.spillover;
    let own_seen = if kappa > 0.0 {
        let bled = tissue.contacts.sum(&seen_counts);
        (0..n)
            .map(|i| (seen_counts[i] + kappa * bled[i]) / sf[i])
            .collect()
    } else {
        seen
    };

    // Who responds, and whom the estimator calls competent.
    let responds: Vec<bool> = (0..n)
        .map(|i| {
            if scen.marker {
                marker[bin[i]]
            } else {
                competent[i]
            }
        })
        .collect();
    let comp: Vec<bool> = (0..n)
        .map(|i| competent[i] || (scen.marker && marker[bin[i]]))
        .collect();
    let u_bin: Vec<f64> = if scen.heterogeneous {
        let mut r = SmallRng::seed_from_u64(mix_seed(seed, 5));
        (0..nb).map(|_| 0.5 + r.random::<f64>()).collect()
    } else {
        vec![1.0; nb]
    };

    // The programme, gene by gene: observed counts (as measured, with
    // spillover) and potential counts, in T* at the realized ring
    // exposure and in the ring exposure at the realized T*, accumulated
    // into responses. Potential counts carry the realized bleed, so the
    // truths are on the measured scale.
    let total_w: f64 = base
        .target_weights
        .iter()
        .map(|w| w.abs())
        .sum::<f64>()
        .max(1e-12);
    let beta = scen.confound_response;
    let generic: Vec<f64> = (0..n)
        .map(|i| {
            if comp[i] {
                scen.generic_shift * trait_near[i]
            } else {
                0.0
            }
        })
        .collect();
    let lifts = sign_lifts(&f, beta, &generic, &marker, bin);
    let ring_on = scen.ring_effect != 0.0;
    let (mut y_seen, mut y1, mut y0) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut bleed = vec![0.0; n];
    let (mut yd1, mut yd0) = if ring_on {
        (vec![0.0; n], vec![0.0; n])
    } else {
        (Vec::new(), Vec::new())
    };
    let (mut c_obs, mut c1, mut c0) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut d1, mut d0) = if ring_on {
        (vec![0.0; n], vec![0.0; n])
    } else {
        (Vec::new(), Vec::new())
    };
    let mut seen_g = vec![0.0; n];
    let mut bin_cells = vec![0usize; nb];
    for &b in bin.iter() {
        bin_cells[b] += 1;
    }
    for (g, &w) in base.target_weights.iter().enumerate() {
        let s = if w < 0.0 { -1.0 } else { 1.0 };
        let lift = &lifts[(w < 0.0) as usize];
        for i in 0..n {
            let b = bin[i];
            let r0 = sf[i] * base.target_bin_means[b * ng + g] * lift[i];
            let u: f64 = rng.random();
            let (th, thr) = if responds[i] {
                (
                    s * scen.effect * u_bin[b] * dose_share[i],
                    s * scen.ring_effect * u_bin[b],
                )
            } else {
                (0.0, 0.0)
            };
            let q = |l: f64| poisson_quantile(u, r0 * l.exp());
            let ring_lift = if r_star[i] { thr } else { 0.0 };
            let hi = q(th + ring_lift);
            let lo = if th == 0.0 { hi } else { q(ring_lift) };
            c1[i] = hi;
            c0[i] = lo;
            c_obs[i] = if t_star[i] { hi } else { lo };
            if ring_on {
                let contact = if t_star[i] { th } else { 0.0 };
                // The count at the ring exposure r* does not have.
                let other = if thr == 0.0 {
                    c_obs[i]
                } else {
                    q(contact + if r_star[i] { 0.0 } else { thr })
                };
                (d1[i], d0[i]) = if r_star[i] {
                    (c_obs[i], other)
                } else {
                    (other, c_obs[i])
                };
            }
        }
        // Bleed from the neighbours' observed counts, shared by every
        // potential count of the receiver.
        let bled: Vec<f64> = if kappa > 0.0 {
            tissue
                .contacts
                .sum(&c_obs)
                .into_iter()
                .map(|b| kappa * b)
                .collect()
        } else {
            vec![0.0; n]
        };
        for i in 0..n {
            seen_g[i] = c_obs[i] + bled[i];
            c1[i] += bled[i];
            c0[i] += bled[i];
            y_seen[i] += w * (seen_g[i] / sf[i]).ln_1p();
            y1[i] += w * (c1[i] / sf[i]).ln_1p();
            y0[i] += w * (c0[i] / sf[i]).ln_1p();
            if ring_on {
                yd1[i] += w * ((d1[i] + bled[i]) / sf[i]).ln_1p();
                yd0[i] += w * ((d0[i] + bled[i]) / sf[i]).ln_1p();
            }
        }
        // The bleed covariate from the measured counts, as
        // [`super::estimate::programme_columns`] reads them: bin means, then expected
        // neighbour counts.
        let mut m = vec![0.0; nb];
        for i in 0..n {
            m[bin[i]] += seen_g[i] / sf[i];
        }
        for (mb, &c) in m.iter_mut().zip(&bin_cells) {
            *mb /= c.max(1) as f64;
        }
        for (o, x) in bleed
            .iter_mut()
            .zip(expected_bleed(&m, bin, &tissue.contacts, sf))
        {
            *o += w * x;
        }
    }
    drop((c_obs, c1, c0, d1, d0, seen_g));
    for v in [
        &mut y_seen,
        &mut y1,
        &mut y0,
        &mut yd1,
        &mut yd0,
        &mut bleed,
    ] {
        for x in v.iter_mut() {
            *x /= total_w;
        }
    }
    let ring_truth = if shifted || !ring_on {
        0.0
    } else {
        sample_effect(&yd1, &yd0, &comp)
    };
    drop((yd1, yd0));
    let misclassified = {
        let k = comp.iter().filter(|&&c| c).count();
        let miss = (0..n)
            .filter(|&i| comp[i] && exposure.t[i] != t_star[i])
            .count();
        miss as f64 / k.max(1) as f64
    };

    exposure.own = own_seen.iter().map(|v| v.ln_1p()).collect();
    let response = Response {
        spill: tissue.contacts.mean(&y_seen),
        y: y_seen,
        bleed,
    };
    let background_lifts = sign_lifts(
        &f,
        beta * scen.background_confounding,
        &generic,
        &marker,
        bin,
    );
    let make_backgrounds = || -> Vec<Response> {
        (0..opts.background_draws)
            .into_par_iter()
            .map(|k| {
                let seed = mix_seed(mix_seed(seed, 9), k as u64);
                let nz = simulate_background(tissue, base, &background_lifts, kappa, seed);
                let genes: Vec<(f64, &[(u32, f32)])> = base
                    .target_weights
                    .iter()
                    .zip(&nz)
                    .map(|(&w, v)| (w, v.as_slice()))
                    .collect();
                Response::build(tissue, &genes)
            })
            .collect()
    };

    // The estimands over the exposure thresholds, as `lra` reads them; a fixed
    // threshold has no others here, no curve being read.
    let levels = std::slice::from_ref(&comp);
    let taus = if opts.threshold.is_none() {
        thresholds(&exposure, levels, opts)
    } else {
        Vec::new()
    };
    let rows = Rows::cells(tissue).pooled();
    let avg = over_thresholds(
        rows,
        &exposure,
        &taus,
        std::slice::from_ref(&response),
        make_backgrounds,
        levels,
        Sensitivity {
            nospill: kappa > 0.0,
            steps: true,
        },
        opts,
    )?;
    let e = avg.scopes[0];
    // The displaced field the estimator sees has no effect.
    let (truth, truth_weighted) = if shifted {
        (0.0, 0.0)
    } else {
        let (mut num, mut den) = (0.0, 0.0);
        for i in (0..avg.h.len()).filter(|&i| comp[i]) {
            num += avg.h[i] * (y1[i] - y0[i]);
            den += avg.h[i];
        }
        let weighted = if den > 0.0 { num / den } else { f64::NAN };
        (sample_effect(&y1, &y0, &comp), weighted)
    };
    let comp_rows: Vec<u32> = (0..n as u32).filter(|&i| comp[i as usize]).collect();
    let crude = crude_difference(&response.y, &exposure.t, &comp_rows);
    let crude_se = naive_se(&response.y, &exposure.t, &comp);
    let mut judged = EffectRow::pending(
        e,
        (
            coexpression_baselines.coexpr_log_or,
            coexpression_baselines.coexpr_se,
        ),
        eta_squared(
            &tissue.neighbour_bin,
            &indicator(&exposure.leaned_on(e.abf)),
            &rows.scope_rows[0],
        ),
        (avg.background, avg.over_background),
        ng,
    );
    report(std::slice::from_mut(&mut judged), &INTERNALS.report_opts());
    Ok(SimRow {
        ligand: base.ligand.into(),
        receptor: base.receptor.into(),
        scenario: *scen,
        truth,
        truth_weighted,
        estimate: e.competent,
        crude,
        crude_se,
        incompetent: e.incompetent,
        ring: e.ring,
        ring_truth,
        no_spill: e.nospill,
        steps: e.steps,
        contrast: e.effect,
        over_ring: e.over_ring,
        background: avg.background,
        over_background: avg.over_background,
        verdict: judged.verdict,
        misclassified,
        baselines: coexpression_baselines,
        seconds: start.elapsed().as_secs_f64(),
        baseline_seconds,
    })
}

/// The receptor driven by the hidden field: `Poisson(sf · ρ_b · e^{γ f} /
/// mean e^{γ f})`, `ρ_b` the real receptor's mean count / size factor in
/// identity bin `b`.
fn simulate_receptor(tissue: &Tissue, real: &[f64], f: &[f64], gamma: f64, seed: u64) -> Vec<f64> {
    let n = tissue.n();
    let (sf, bin) = (&tissue.size_factor, &tissue.identity_bin);
    let nb = n_bins(tissue);
    let (mut rho, mut cells) = (vec![0.0; nb], vec![0usize; nb]);
    for i in 0..n {
        rho[bin[i]] += real[i] / sf[i];
        cells[bin[i]] += 1;
    }
    for (r, &c) in rho.iter_mut().zip(&cells) {
        *r /= c.max(1) as f64;
    }
    let norm = f.iter().map(|z| (gamma * z).exp()).sum::<f64>() / n.max(1) as f64;
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|i| poisson(&mut rng, sf[i] * rho[bin[i]] * (gamma * f[i]).exp() / norm))
        .collect()
}

/// `v` read at each row's point plus an offset of `shift` contact lengths
/// in a random direction: the value of the nearest row in the same
/// section. A grid of one contact length per section, searched in growing
/// square rings until no closer row can lie outside.
fn shift_field(tissue: &Tissue, v: &[f64], shift: f64, seed: u64) -> Vec<f64> {
    let n = tissue.n();
    let phi = SmallRng::seed_from_u64(seed).random::<f64>() * std::f64::consts::TAU;
    let (dx, dy) = (phi.cos(), phi.sin());
    let cell_of = |s: u32, x: f64, y: f64| {
        let g = tissue.unit[s as usize];
        (s, (x / g).floor() as i64, (y / g).floor() as i64)
    };
    let mut grid: HashMap<(u32, i64, i64), Vec<u32>> = HashMap::default();
    for i in 0..n {
        let (x, y) = tissue.coords[i];
        grid.entry(cell_of(tissue.section[i], x as f64, y as f64))
            .or_default()
            .push(i as u32);
    }
    // Far enough to cross any section from any point near it.
    let max_ring = 1 + (grid.len() as f64).sqrt() as i64 * 4;
    (0..n)
        .map(|i| {
            let s = tissue.section[i];
            let g = tissue.unit[s as usize];
            let (x, y) = tissue.coords[i];
            let (qx, qy) = (x as f64 + shift * g * dx, y as f64 + shift * g * dy);
            let (_, cx, cy) = cell_of(s, qx, qy);
            let mut best = (f64::INFINITY, i);
            for r in 0..=max_ring {
                for gx in (cx - r)..=(cx + r) {
                    for gy in (cy - r)..=(cy + r) {
                        if (gx - cx).abs() != r && (gy - cy).abs() != r {
                            continue;
                        }
                        for &j in grid.get(&(s, gx, gy)).into_iter().flatten() {
                            let (jx, jy) = tissue.coords[j as usize];
                            let d2 = (jx as f64 - qx).powi(2) + (jy as f64 - qy).powi(2);
                            if d2 < best.0 {
                                best = (d2, j as usize);
                            }
                        }
                    }
                }
                if best.0 <= (r as f64 * g).powi(2) {
                    break;
                }
            }
            v[best.1]
        })
        .collect()
}

/// The two-sample standard error a naive comparison would report.
fn naive_se(y: &[f64], t: &[bool], subset: &[bool]) -> f64 {
    let group = |arm: bool| -> (f64, f64, usize) {
        let v: Vec<f64> = (0..y.len())
            .filter(|&i| subset[i] && t[i] == arm)
            .map(|i| y[i])
            .collect();
        let n = v.len();
        if n < 2 {
            return (0.0, 0.0, n);
        }
        let m = v.iter().sum::<f64>() / n as f64;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
        (m, var, n)
    };
    let (_, v1, n1) = group(true);
    let (_, v0, n0) = group(false);
    if n1 < 2 || n0 < 2 {
        return f64::NAN;
    }
    (v1 / n1 as f64 + v0 / n0 as f64).sqrt()
}
