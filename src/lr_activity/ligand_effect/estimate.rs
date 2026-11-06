//! One ligand-receptor pair: exposure, response, and every estimand.
//!
//! | Column | Estimand |
//! |---|---|
//! | `coexpr_log_or` | log odds of ligand on one side and receptor on the other, across contacts (association) |
//! | `crude` | mean response of exposed minus unexposed competent receivers (association) |
//! | `effect_competent` | the competent effect, **the headline**: overlap-weighted effect where the receptor is expressed, averaged over exposure thresholds |
//! | `effect_log10_bf`, `tau_top`, `tau_top_weight` | the evidence for an effect at any threshold against none, and the threshold the average leans on most |
//! | `effect_incompetent` | the incompetent effect: the same where it is not; a negative control, ≈ 0 |
//! | `effect_contrast` | the contrast: competent effect − incompetent effect; a contrast between cell types, not only between receptor states (below) |
//! | `ring_effect` | the ring effect: effect of ligand on distant, non-touching cells, given the real exposure; a negative control, ≈ 0 |
//! | `effect_over_ring` | the competent effect − the ring effect, paired over variance clusters |
//! | `effect_over_background` | the competent effect − the effect on the background programme, paired over variance clusters (pooled row only) |
//! | `effect_competent_{section,spatial,identity}` | the competent effect under nested adjustment sets (pooled row only) |
//! | `effect_competent_nospill` | the competent effect without the neighbour-response and bleed covariates (pooled row only) |
//!
//! Competence is read from identity bins, so competent and incompetent
//! receivers are different cell types by construction: the contrast conflates the
//! receptor with everything else that differs between those types (their
//! baseline responsiveness, their neighbours, their exposure). So the
//! contrast alone is no finding, and the headline stays the competent effect.
//! The verdict still asks for it, conservatively: an effect that does not
//! exceed the incompetent one is not called specific to the receptor, which
//! can also turn away a real effect on two responsive cell types.
//!
//! `crude` and the sensitivity fits below (the nested adjustment sets and the
//! no-spill fit) read the headline threshold alone, while `effect_competent` is
//! averaged over thresholds, so a gap between them can come from the threshold
//! as well as the adjustment.
//!
//! The sequential adjustment refits the competent effect adjusting for (a) section only,
//! (b) + spatial smooth, (c) + cell identity (receiver and neighbour
//! identity, cell type when given); `effect_competent` is the full set,
//! adding depth. Read left to right with `crude`, it shows how much of an
//! apparent effect each confounder group explains. The pair's own columns
//! (own ligand, spillover, bleed, niche ligand, competence) are in every
//! step.
//!
//! Exposure `T = 1{Σ_contacts ligand / size factor > τ}`. Response
//! `Y = Σ_g w_g log1p(x_g / size factor) / Σ_g |w_g|` over the pair's
//! programme. Extra confounders for the pair:
//!
//! - the receiver's own ligand (bleed on the exposure side);
//! - the contact-neighbour mean response, and the neighbours' expected
//!   programme counts on the receiver's own scale ([`programme_columns`]):
//!   bleed on the response side, what a neighbour's transcripts add to the
//!   receiver's measurement. Bleed is a sum over contacts divided by the
//!   receiver's size factor, so it grows with the contact count, which also
//!   drives exposure, and shrinks with the receiver's depth; a neighbour
//!   mean sees neither. It is built from the neighbours' identity bins and
//!   depths rather than their observed counts, which carry their own
//!   response to shared senders;
//! - the mean ligand of second-order neighbours (contacts of contacts), a
//!   proxy of a niche that drives both ligand and response at a few contact
//!   lengths, finer than the spatial smooth can follow. Under contact-only
//!   signalling it is no exposure itself;
//! - competence.
//!
//! The two response-side columns enter the outcome model only, so one
//! propensity serves every fit at a threshold with the same treatment and
//! design ([`ThresholdFits`]).
//!
//! Neighbours share senders with the receiver, so their response is partly
//! downstream of the same exposure and adjusting for it can attenuate a
//! real effect; `effect_competent_nospill` drops both response-side bleed
//! columns as a sensitivity check. One fit serves the pooled row and one
//! row per section: a section's estimate is the same estimator summed over
//! that section's receivers.
//!
//! Inference across pairs ([`report`]) runs once every pair is in: see
//! there for the q-values, the identifiability diagnostics and the
//! verdict labels.
//!
//! **Choices averaged, not asked for.** Where the analysis has a choice
//! with no right answer it makes each and averages the estimates by
//! Rubin's rules ([`super::overlap::combine`]), so the spread between
//! the choices widens the interval instead of a setting picking one:
//!
//! - competence, the share of an identity bin that must detect the
//!   receptor, over [`COMPETENCE_GRID`]. Competence is a property of
//!   identity bins (cell types), not of single receivers, so the competent effect compares
//!   exposed with unexposed receivers among the cell types that carry the
//!   receptor; the grid moves the line between "carries" and "does not";
//! - the programme's weighting ([`PairData::reweighted`]).
//!
//! **The exposure threshold** is averaged too, differently. Each threshold is a
//! model of how exposure acts, with its own fits, and the estimates are averaged
//! by their posterior weights under an approximate Bayes factor
//! ([`super::overlap::abf_average`]), not by Rubin's rules: thresholds are not
//! exchangeable draws, and which one the data favour is itself of interest. Each
//! estimand has its own weights. The weights read the estimates they average, so
//! the average leans toward large z-scores; the verdict's comparison with the
//! negative controls, averaged the same way, is what guards against that.
//!
//! **Threshold–response curve.** The headline τ is chosen per pair
//! ([`choose_threshold`]). Beside it, the competent effect (pooled) at τ = 0 and at the
//! quartiles of the positive exposures ([`curve_taus`]) with pointwise
//! intervals and a simultaneous 95% band ([`sup_t_crit`]): how the effect
//! grows with dose, and whether the headline rests on one threshold.
//!
//! **Background programmes** ([`BackgroundPool`]): the pair's programme with every
//! target swapped for a random gene of similar mean expression outside every
//! programme and pair, same weights, drawn many times. They should show no
//! effect; when they do, exposure moves generic expression (activation, depth,
//! residual confounding), and the pair's own competent effect is a finding only
//! where it stands out from them ([`super::overlap::background_comparison`]).
//!
//! **Reading an effect** ([`Moments`]): θ is a difference in the signed
//! mean `log1p` programme, read as a mean with SD (= SE) and 95% interval,
//! and as a % change `100 (e^θ − 1)` with the delta-method SD
//! `100 e^θ SE` and the interval's endpoints mapped the same way.
//!
//! **By receiver cell type**: with named cell types
//! ([`super::tissue::Tissue::with_cell_type_names`]) each pair also gets
//! one row per type, `section` = `type:<name>`: the estimands among that
//! type's receivers, from the pooled fit.

use legume_numeric::matrix::rand_util::{mix_seed, name_seed};
use rayon::prelude::*;

use super::output::Verdict;
use super::overlap::{
    abf_average, abf_summary, background_comparison, combine, crude_difference, difference,
    overlap_terms_with, propensity, subset_estimate, upper_normal_p, AbfSummary, CrossFit,
    DesignView, Estimate, FoldStarts, NuisanceOpts, OverlapTerms, PairInputs,
};
use super::regression::standardize;
use super::tissue::Tissue;
use crate::lr_activity::edge_scores::jeffreys_log_odds;

/// The section label of the pooled row.
pub const ALL_SECTIONS: &str = "all";

/// Overlap weight below which a receiver's exposure counts as nearly
/// determined by its confounders.
pub const NEAR_DETERMINED_H: f64 = 0.01;

/// A programme: `(weight, non-zero counts by row)` for each gene.
pub type Targets<'a> = [(f64, &'a [(u32, f32)])];

/// A pair's raw data on the tissue's rows.
pub struct PairData<'a> {
    pub ligand: &'a str,
    pub receptor: &'a str,
    pub ligand_counts: &'a [f64],
    pub receptor_counts: &'a [f64],
    /// `(weight, non-zero counts by row)` for each programme gene.
    pub targets: &'a [(f64, &'a [(u32, f32)])],
    /// The programme reweighted (hub targets down-weighted by
    /// specificity), averaged with `targets`; `None` for `targets` alone.
    pub reweighted: Option<&'a Targets<'a>>,
    pub n_removed: usize,
}

/// Competence thresholds averaged over (share of an identity bin detecting
/// the receptor).
pub const COMPETENCE_GRID: [f64; 3] = [0.05, 0.1, 0.2];

/// Prior SD of an effect at one exposure threshold, on the response's scale.
pub const ABF_PRIOR_SD: f64 = 0.002;

/// Background programmes per pair, and how many of the genes nearest in expression
/// each background gene is drawn from.
pub const BACKGROUND_DRAWS: usize = 20;
const BACKGROUND_NEIGHBOURS: usize = 20;

/// Quantiles of the positive exposures the curve visits beside τ = 0.
pub const CURVE_QUANTILES: [f64; 3] = [0.25, 0.5, 0.75];

/// Multiplier draws for the simultaneous band.
pub const BAND_DRAWS: usize = 2000;

/// Genes in the background pool at most ([`background_pool_genes`]).
pub const BACKGROUND_POOL: usize = 2000;

#[derive(Clone, Copy, Debug)]
pub struct EffectOpts {
    /// Exposure threshold τ; `None` picks it per pair ([`choose_threshold`]).
    pub threshold: Option<f64>,
    /// Shares of an identity bin detecting the receptor for its receivers
    /// to count as competent, averaged over ([`COMPETENCE_GRID`]); the
    /// benchmark plants on one.
    pub competences: &'static [f64],
    /// Read the pair beyond its headline threshold: the threshold–response
    /// curve, and the average over those thresholds ([`over_thresholds`]).
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
    /// thresholds ([`abf_average`]).
    pub abf_prior_sd: f64,
    /// Background programmes per pair ([`BackgroundPool::draw`]).
    pub background_draws: usize,
    pub seed: u64,
    /// Run even the fits whose every subset leaves an arm thin
    /// ([`Reads`]), which the guard then reports as not estimable all the
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
/// ([`EffectOpts::guard`]), the fits skipped as hopeless ([`Reads`]) and
/// the verdict note all read this one rule.
pub fn thin_arm(min_arm: usize, n: usize, treated: usize) -> bool {
    treated < min_arm || n.saturating_sub(treated) < min_arm
}

/// The `df` of a not-estimable estimate whose rows lie in fewer than two
/// variance clusters, so the verdict can say why.
pub const ONE_CLUSTER_DF: f64 = 0.0;

/// One output row: a pair in one section, or pooled.
#[derive(Clone, Debug)]
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
    pub crude: f64,
    /// The estimands; the step and no-spill estimates are NaN on
    /// per-section rows.
    pub effects: ScopeEffects,
    /// Share of the variance of the exposure explained by the receiver's
    /// dominant-neighbour identity bin ([`eta_squared`]).
    pub exposure_eta2: f64,
    /// Distinct identity bins among the competent and the incompetent
    /// receivers: competence is a function of the bin, so these say how
    /// many cell types each side of the contrast is made of.
    pub n_competent_bins: usize,
    pub n_incompetent_bins: usize,
    /// The competent effect on the matched background programme (pooled row only; NaN without a
    /// background), and the background's gene count.
    pub background_effect: Estimate,
    pub n_background_genes: usize,
    /// The competent effect minus the background's (pooled row only; NaN without a background).
    pub effect_over_background: Estimate,
    /// The threshold–response curve (pooled row only; empty elsewhere).
    pub curve: Vec<CurvePoint>,
    // From here on filled in by `report`, once every pair is in.
    /// BH q of the background effect, two-sided, within the section.
    pub background_q: f64,
    /// Two-sided p of the headline (the competent effect) against the t reference with
    /// clusters − 1 degrees of freedom ([`Estimate::p`]), and its BH q
    /// within the section.
    pub effect_p: f64,
    pub effect_q: f64,
    /// One-sided (positive) normal p of the co-expression log odds ratio,
    /// and its BH q within the section.
    pub coexpr_p: f64,
    pub coexpr_q: f64,
    /// BH q of the controls (incompetent effect, ring effect), two-sided, within the section.
    pub effect_incompetent_q: f64,
    /// BH q of the contrast between the effects in competent and in
    /// receptor-negative receivers, among the significant effects.
    pub effect_contrast_q: f64,
    pub ring_effect_q: f64,
    /// BH q of the competent effect over the ring effect and over the background's,
    /// among the significant effects.
    pub effect_over_ring_q: f64,
    pub effect_over_background_q: f64,
    /// Pairs in the section sharing this pair's ligand.
    pub n_pairs_same_ligand: usize,
    /// BH q over ligands of the Simes p of each ligand's pairs.
    pub ligand_q: f64,
    pub verdict: Verdict,
    /// Why the verdict is what it is beyond the q-values: failed controls,
    /// confounding, why a pair is not estimable. Semicolon separated.
    pub warnings: String,
}

impl EffectRow {
    /// A row holding what [`report`] reads for its verdict, the rest blank: the
    /// estimands, co-expression, the exposure's identity share and the background.
    /// `lra-simulate` judges its runs by the rule `lra` uses through this.
    pub fn pending(
        effects: ScopeEffects,
        (coexpr_log_or, coexpr_se): (f64, f64),
        exposure_eta2: f64,
        (background_effect, effect_over_background): (Estimate, Estimate),
        n_background_genes: usize,
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
            crude: f64::NAN,
            effects,
            exposure_eta2,
            n_competent_bins: 0,
            n_incompetent_bins: 0,
            background_effect,
            n_background_genes,
            effect_over_background,
            curve: Vec::new(),
            background_q: f64::NAN,
            effect_p: f64::NAN,
            effect_q: f64::NAN,
            coexpr_p: f64::NAN,
            coexpr_q: f64::NAN,
            effect_incompetent_q: f64::NAN,
            effect_contrast_q: f64::NAN,
            ring_effect_q: f64::NAN,
            effect_over_ring_q: f64::NAN,
            effect_over_background_q: f64::NAN,
            n_pairs_same_ligand: 0,
            ligand_q: f64::NAN,
            verdict: Verdict::InsufficientData,
            warnings: String::new(),
        }
    }
}

/// One point of the threshold–response curve: the competent effect (pooled) at exposure
/// threshold `tau`, `quantile` being the share of positive exposures at or
/// below it; the pointwise 95% interval and the simultaneous band.
#[derive(Clone, Copy, Debug)]
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
        Self {
            tau,
            quantile,
            effect,
            lo: t - c * se,
            hi: t + c * se,
            band_lo: t - b * se,
            band_hi: t + b * se,
        }
    }
}

/// An estimate read two ways: on the log scale it is estimated on (the
/// signed mean `log1p` programme), as a mean (θ) with SD = SE and a 95%
/// interval from the t reference (the mean and SD are the estimate's own
/// `theta` and `se`); and as a % change of the programme,
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
        let c = e.crit();
        let (lo, hi) = (e.theta - c * e.se, e.theta + c * e.se);
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

/// The exposure side of a pair's per-row quantities: the same under every
/// weighting of its programme.
pub struct Exposure {
    /// The exposure threshold used.
    pub tau: f64,
    pub t: Vec<bool>,
    pub t_ring: Vec<bool>,
    /// `log1p` of the receiver's own ligand / size factor.
    pub own: Vec<f64>,
    /// `log1p` of the mean ligand / size factor over second-order
    /// neighbours.
    pub niche: Vec<f64>,
    /// Ligand / size factor summed over each row's contacts, which the
    /// threshold is read from.
    pub dose: Vec<f64>,
    /// The same over its distant ring. The ring holds as many partners as
    /// there are contacts, so the same τ means the same dose.
    pub ring_dose: Vec<f64>,
}

impl Exposure {
    /// The exposure side from the ligand counts the estimator sees.
    pub fn build(tissue: &Tissue, ligand_counts: &[f64], opts: &EffectOpts) -> Self {
        let (lnorm, dose) =
            super::receivers::contact_dose(&tissue.contacts, &tissue.size_factor, ligand_counts);
        let ring_dose = tissue.ring_sum(&lnorm);
        let tau = choose_threshold(&dose, opts);
        Self {
            tau,
            t: exposed(&dose, tau),
            t_ring: exposed(&ring_dose, tau),
            own: lnorm.iter().map(|v| v.ln_1p()).collect(),
            niche: niche_ligand(tissue, &lnorm),
            dose,
            ring_dose,
        }
    }

    /// The exposure at the threshold an average leaned on, else the headline's.
    pub fn leaned_on(&self, abf: AbfSummary) -> Vec<bool> {
        if abf.top_tau.is_finite() && abf.top_tau != self.tau {
            exposed(&self.dose, abf.top_tau)
        } else {
            self.t.clone()
        }
    }

    /// The same exposure read at another threshold.
    pub fn at(&self, tau: f64) -> Self {
        Self {
            tau,
            t: exposed(&self.dose, tau),
            t_ring: exposed(&self.ring_dose, tau),
            own: self.own.clone(),
            niche: self.niche.clone(),
            dose: self.dose.clone(),
            ring_dose: self.ring_dose.clone(),
        }
    }
}

/// The response side of a pair's per-row quantities: rebuilt for each
/// programme (each weighting, the background).
pub struct Response {
    pub y: Vec<f64>,
    /// The contact-neighbour mean response.
    pub spill: Vec<f64>,
    /// The neighbours' expected programme counts on the receiver's scale
    /// ([`programme_columns`]).
    pub bleed: Vec<f64>,
}

impl Response {
    pub fn build(tissue: &Tissue, targets: &Targets<'_>) -> Self {
        let (genes, weights) = genes_and_weights(targets);
        Self::build_many(tissue, &genes, &[weights])
            .pop()
            .expect("one weighting")
    }

    /// One response side per weighting of the same genes, each gene read
    /// once; a zero weight leaves the gene out of that weighting.
    pub fn build_many(
        tissue: &Tissue,
        genes: &[&[(u32, f32)]],
        weightings: &[Vec<f64>],
    ) -> Vec<Self> {
        let sf = &tissue.size_factor;
        let (ys, bleeds) = programme_columns(
            genes,
            weightings,
            &tissue.identity_bin,
            &tissue.contacts,
            sf,
        );
        ys.into_iter()
            .zip(bleeds)
            .map(|(y, bleed)| Self {
                spill: tissue.contacts.mean(&y),
                y,
                bleed,
            })
            .collect()
    }
}

/// One gene's non-zero counts by row.
type Gene<'t> = &'t [(u32, f32)];

/// A programme's genes and weights, apart.
fn genes_and_weights<'t>(targets: &Targets<'t>) -> (Vec<Gene<'t>>, Vec<f64>) {
    targets.iter().map(|&(w, nz)| (nz, w)).unzip()
}

/// `log1p` of the mean of `lnorm` (ligand / size factor) over each row's
/// second-order neighbours.
pub fn niche_ligand(tissue: &Tissue, lnorm: &[f64]) -> Vec<f64> {
    tissue
        .second_mean(lnorm)
        .into_iter()
        .map(f64::ln_1p)
        .collect()
}

/// The response and the bleed of each weighting of `genes`, in one pass
/// over the genes: each gene's `log1p` counts and expected bleed are
/// computed once and added to every weighting that gives it a non-zero
/// weight.
///
/// The response is the signed weighted mean of `log1p(x / size factor)`
/// over the programme, accumulated over non-zeros only (a zero count
/// contributes `log1p(0) = 0`).
///
/// The bleed is what contact neighbours' programme transcripts are expected to add to
/// each receiver's response if segmentation handed it all of them: per
/// gene, the neighbours' expected counts (their size factor times their
/// identity bin's mean count / size factor) summed and normalized by the
/// receiver's size factor, through the programme's signed `log1p` mean.
///
/// Expected rather than observed counts: a neighbour's observed programme
/// is partly its own response to senders it shares with the receiver, so
/// adjusting for it would absorb a real effect. The expectation keeps what
/// drives bleed and is not downstream of exposure: how many contacts,
/// their depth and identity, and the receiver's depth.
pub fn programme_columns(
    genes: &[&[(u32, f32)]],
    weightings: &[Vec<f64>],
    identity_bin: &[usize],
    contacts: &super::exposure::Contacts,
    sf: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = sf.len();
    let nb = identity_bin.iter().copied().max().map_or(0, |b| b + 1);
    let mut cells = vec![0usize; nb];
    for &b in identity_bin {
        cells[b] += 1;
    }
    let mut ys = vec![vec![0.0; n]; weightings.len()];
    let mut bleeds = vec![vec![0.0; n]; weightings.len()];
    let mut m = vec![0.0; nb];
    for (g, nz) in genes.iter().enumerate() {
        let on: Vec<usize> = (0..weightings.len())
            .filter(|&k| weightings[k][g] != 0.0)
            .collect();
        if on.is_empty() {
            continue;
        }
        m.fill(0.0);
        for &(r, v) in *nz {
            let r = r as usize;
            let x = v as f64 / sf[r];
            m[identity_bin[r]] += x;
            let l = x.ln_1p();
            for &k in &on {
                ys[k][r] += weightings[k][g] * l;
            }
        }
        for (mb, &c) in m.iter_mut().zip(&cells) {
            *mb /= c.max(1) as f64;
        }
        let b = expected_bleed(&m, identity_bin, contacts, sf);
        for &k in &on {
            let w = weightings[k][g];
            for (o, x) in bleeds[k].iter_mut().zip(&b) {
                *o += w * x;
            }
        }
    }
    for (k, w) in weightings.iter().enumerate() {
        let total: f64 = w.iter().map(|w| w.abs()).sum::<f64>().max(1e-12);
        for v in ys[k].iter_mut().chain(bleeds[k].iter_mut()) {
            *v /= total;
        }
    }
    (ys, bleeds)
}

/// One gene's `log1p(Σ_contacts sf_j m_{bin(j)} / sf_i)`, `m` being the
/// gene's mean count / size factor per identity bin.
pub fn expected_bleed(
    m: &[f64],
    identity_bin: &[usize],
    contacts: &super::exposure::Contacts,
    sf: &[f64],
) -> Vec<f64> {
    contacts
        .adj
        .par_iter()
        .enumerate()
        .map(|(i, adj)| {
            let e: f64 = adj
                .iter()
                .map(|&j| sf[j as usize] * m[identity_bin[j as usize]])
                .sum();
            (e / sf[i]).ln_1p()
        })
        .collect()
}

pub fn exposed(a: &[f64], tau: f64) -> Vec<bool> {
    a.iter().map(|&v| v > tau).collect()
}

/// The exposure threshold for one pair: the requested τ, else 0 (any
/// ligand on a neighbour) when that leaves `min_arm` receivers on each
/// side, else the median exposure. An abundant ligand reaches nearly every
/// receiver, and its contrast is then high against low exposure rather
/// than some against none.
pub fn choose_threshold(a: &[f64], opts: &EffectOpts) -> f64 {
    if let Some(t) = opts.threshold {
        return t;
    }
    let pos = a.iter().filter(|&&v| v > 0.0).count();
    if pos >= opts.min_arm && a.len() - pos >= opts.min_arm {
        return 0.0;
    }
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.total_cmp(y));
    // Strictly-above split: step below the median when ties at the median
    // would leave the exposed arm empty.
    let med = s[s.len() / 2];
    if a.iter().filter(|&&v| v > med).count() >= opts.min_arm {
        med
    } else {
        s.iter().rev().copied().find(|&v| v < med).unwrap_or(0.0)
    }
}

/// The rows an effect is estimated on: the tissue's receivers, with the rows
/// inside each scope, pooled first ([`Tissue::scopes`]).
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub design: &'a super::overlap::Design,
    pub steps: &'a [super::tissue::AdjustmentStep],
    pub cf: &'a super::overlap::CrossFit,
    pub scope_rows: &'a [Vec<u32>],
}

impl<'a> Rows<'a> {
    pub fn cells(tissue: &'a Tissue) -> Self {
        Self {
            design: &tissue.design,
            steps: &tissue.steps,
            cf: &tissue.cf,
            scope_rows: &tissue.scope_rows,
        }
    }

    /// The same rows with the pooled scope alone.
    pub fn pooled(self) -> Self {
        Self {
            scope_rows: &self.scope_rows[..1],
            ..self
        }
    }
}

/// Which fit of a pair: the main one, the main one without the
/// response-side bleed columns (spillover, bleed), or the ring control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    Main,
    NoSpill,
    Ring,
}

/// Whether the competent ones (`level`) among the rows `inside` leave both arms
/// of `t` at `min_arm`: the counts the arm guard reads of the subset
/// ([`subset_estimate`]). Every other scope is a subset of the pooled one, so
/// neither of its arms is larger.
fn arms_open(inside: &[u32], t: &[bool], level: &[bool], opts: &EffectOpts) -> bool {
    let competent = inside.iter().filter(|&&i| level[i as usize]);
    let (n, treated) = competent.fold((0, 0), |(n, x), &i| (n + 1, x + t[i as usize] as usize));
    !opts.thin_arm(n, treated)
}

/// What the guard reports of a subset it rejects for a thin arm, without
/// the fit: not estimable with the subset's counts, and zero cluster sums.
fn thin_subset(t: &[bool], rows: &[u32], cf: &CrossFit) -> (Estimate, Vec<f64>) {
    let treated = rows.iter().filter(|&&i| t[i as usize]).count();
    (
        Estimate::nan(rows.len(), treated),
        vec![0.0; cf.n_clusters()],
    )
}

/// Which of a threshold's fits serve a subset with both arms at `min_arm`
/// ([`thin_arm`]) on the pooled rows. The main fit is read for the effect in
/// competent receivers, at the middle competence threshold: a pair resting on a
/// looser threshold alone has no estimate, whatever those thresholds show. The
/// no-spill, step, curve and background fits read the same subset. The ring fit is
/// read beside it, when one of its own subsets is open.
///
/// A fit not read is skipped, propensity and all, and its estimands are what
/// the guard would make of them, not estimable with their counts
/// ([`thin_subset`]): the skip is the guard read before the fit, not a second
/// rule.
#[derive(Clone, Copy, Debug)]
pub struct Reads {
    pub competent: bool,
    pub ring: bool,
}

impl Reads {
    pub fn of(rows: Rows<'_>, x: &Exposure, levels: &[Vec<bool>], opts: &EffectOpts) -> Self {
        let pooled = &rows.scope_rows[0];
        let competent = levels
            .get(levels.len() / 2)
            .is_some_and(|c| arms_open(pooled, &x.t, c, opts));
        Self {
            competent,
            ring: competent && levels.iter().any(|c| arms_open(pooled, &x.t_ring, c, opts)),
        }
    }
}

/// The controls' propensities ([`ThresholdFits::with_controls`]); `None` for a
/// fit skipped as hopeless ([`Reads`]).
struct Controls {
    ring: Option<Vec<f64>>,
    /// Each adjustment step's, when asked for and the competent effect is read.
    steps: Option<Vec<Vec<f64>>>,
}

/// One threshold's exposure side as its fits read it, built once and shared by
/// every fit at it: the competence indicators (raw, the outcome
/// model's effect modifiers), the pair columns both models carry
/// (standardized: own ligand, niche, every competence indicator, and the
/// contact exposure the ring fit adds to them), which fits serve a subset
/// ([`Reads`]), and the cross-fitted propensities ([`propensity`]) of the
/// main fit and, with [`ThresholdFits::with_controls`], of the ring fit and each
/// adjustment step. The propensity reads no response-side column
/// ([`fit_main`]), so one serves every programme weighting, the background and
/// the no-spill fit.
pub struct ThresholdFits<'a> {
    pub exposure: &'a Exposure,
    /// Competence at each threshold, per row.
    pub levels: &'a [Vec<bool>],
    pub reads: Reads,
    /// [`EffectOpts::fit_thin`]: run the fits [`Reads`] would skip.
    force: bool,
    modifiers: Vec<Vec<f64>>,
    extra: Vec<Vec<f64>>,
    exposed: Vec<f64>,
    main: Option<Vec<f64>>,
    controls: Option<Controls>,
}

/// Where a threshold's propensities start: the previous threshold's solutions,
/// model by model ([`FoldStarts`]).
#[derive(Clone, Debug, Default)]
pub struct Starts {
    pub(super) main: FoldStarts,
    ring: FoldStarts,
}

pub(super) fn indicator(v: &[bool]) -> Vec<f64> {
    v.iter().map(|&b| b as u8 as f64).collect()
}

impl<'a> ThresholdFits<'a> {
    /// The columns and the main fit's propensity on `rows`: all the curve
    /// and the background need.
    pub fn new(
        rows: Rows<'_>,
        exposure: &'a Exposure,
        levels: &'a [Vec<bool>],
        opts: &EffectOpts,
        starts: &mut FoldStarts,
    ) -> anyhow::Result<Self> {
        let modifiers: Vec<Vec<f64>> = levels.iter().map(|c| indicator(c)).collect();
        let extra = [&exposure.own, &exposure.niche]
            .into_iter()
            .chain(&modifiers)
            .map(|c| standardize(c))
            .collect();
        let mut fits = Self {
            exposure,
            levels,
            reads: Reads::of(rows, exposure, levels, opts),
            force: opts.fit_thin,
            modifiers,
            extra,
            exposed: standardize(&indicator(&exposure.t)),
            main: None,
            controls: None,
        };
        if fits.runs(fits.reads.competent) {
            let extra = fits.extra(Fit::Main);
            let main = propensity(
                rows.design.view(),
                rows.cf,
                &exposure.t,
                &extra,
                &opts.nuisance,
                starts,
            )?;
            fits.main = Some(main);
        }
        Ok(fits)
    }

    /// Also the ring fit's propensity and, when `steps`, each adjustment
    /// step's: what [`effects_on`] needs.
    pub fn with_controls(
        mut self,
        rows: Rows<'_>,
        steps: bool,
        opts: &EffectOpts,
        starts: &mut Starts,
    ) -> anyhow::Result<Self> {
        let x = self.exposure;
        let ring = if self.runs(self.reads.ring) {
            let extra = self.extra(Fit::Ring);
            Some(timing::time(timing::MAIN, || {
                propensity(
                    rows.design.view(),
                    rows.cf,
                    &x.t_ring,
                    &extra,
                    &opts.nuisance,
                    &mut starts.ring,
                )
            })?)
        } else {
            None
        };
        let steps = if steps && self.runs(self.reads.competent) {
            let extra = self.extra(Fit::Main);
            // A step starts from the main fit, the model it nests in.
            let p = rows.design.p();
            Some(timing::time(timing::STEPS, || {
                rows.steps
                    .iter()
                    .map(|s| {
                        let view = s.design.view(rows.design);
                        let mut start = starts.main.nested(p, s.design.p());
                        propensity(view, rows.cf, &x.t, &extra, &opts.nuisance, &mut start)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })?)
        } else {
            None
        };
        self.controls = Some(Controls { ring, steps });
        Ok(self)
    }

    /// Whether a fit serving subsets that `read` says are read runs.
    fn runs(&self, read: bool) -> bool {
        read || self.force
    }

    /// The pair columns of both models under `kind`.
    fn extra(&self, kind: Fit) -> Vec<&[f64]> {
        let mut cols: Vec<&[f64]> = self.extra.iter().map(|c| c.as_slice()).collect();
        if kind == Fit::Ring {
            cols.push(&self.exposed);
        }
        cols
    }

    /// The propensity of the fit `kind` (of adjustment step `step` when
    /// given); an error when it was not fitted: the controls' before
    /// [`ThresholdFits::with_controls`], or a fit skipped as hopeless.
    fn propensity(&self, kind: Fit, step: Option<usize>) -> anyhow::Result<&[f64]> {
        let controls = || {
            self.controls
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("the controls' propensities are not fitted"))
        };
        let e = match (kind, step) {
            (Fit::Ring, _) => controls()?.ring.as_deref(),
            (_, Some(s)) => controls()?
                .steps
                .as_ref()
                .and_then(|v| v.get(s))
                .map(Vec::as_slice),
            _ => self.main.as_deref(),
        };
        e.ok_or_else(|| anyhow::anyhow!("the {kind:?} propensity (step {step:?}) is not fitted"))
    }
}

/// One cross-fitted fit of a pair on `design`, every competence threshold
/// at once, with its propensity from `fits` ([`ThresholdFits`]): the main one,
/// or adjustment step `step`'s, or the ring's; an error when that one is
/// not fitted.
///
/// The thresholds give nested indicators `C_c`. One outcome model carries
/// every `C_c` and every `T · C_c`, and one propensity model every `C_c`;
/// The competent effect at threshold `c` is then the subset estimate over the receivers
/// competent at `c`. Each single-threshold model is nested in this one, so
/// each `m_t` stays correctly specified wherever its own was; what changes
/// is one fit instead of one per competence threshold.
///
/// The pair columns: own ligand, niche, every competence indicator, and
/// under [`Fit::Ring`] the contact exposure, the ring exposure being the
/// treatment, in both models; spillover and bleed (not under
/// [`Fit::NoSpill`]) in the outcome model alone. Read off the response,
/// they would tie the propensity to one programme; without them it depends
/// on the exposure side only and is fitted once per exposure threshold for every
/// programme weighting, the background and the no-spill fit.
///
/// That is a trade-off in robustness. The estimator is doubly robust in
/// the confounders and pair columns both models carry: right when either
/// the propensity or the outcome model is. For spillover and bleed it
/// relies on the outcome model alone: the propensity does not balance
/// them, so if they confound exposure and the outcome model gets them
/// wrong, the estimate is biased.
#[allow(clippy::too_many_arguments)]
pub fn fit_main(
    design: DesignView<'_>,
    cf: &CrossFit,
    fits: &ThresholdFits<'_>,
    r: &Response,
    kind: Fit,
    step: Option<usize>,
    opts: &EffectOpts,
) -> anyhow::Result<OverlapTerms> {
    let e = fits.propensity(kind, step)?;
    let x = fits.exposure;
    let extra = fits.extra(kind);
    let outcome_only: Vec<&[f64]> = match kind {
        Fit::NoSpill => Vec::new(),
        Fit::Main | Fit::Ring => vec![&r.spill, &r.bleed],
    };
    let modifiers: Vec<&[f64]> = fits.modifiers.iter().map(|c| c.as_slice()).collect();
    let inputs = PairInputs {
        t: if kind == Fit::Ring { &x.t_ring } else { &x.t },
        y: &r.y,
        extra: &extra,
        outcome_only: &outcome_only,
        modifiers: &modifiers,
    };
    overlap_terms_with(design, cf, &inputs, e, &opts.nuisance)
}

/// Where an estimand is summed: the whole tissue, one section, or one
/// receiver cell type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    All,
    Section(u32),
    CellType(u32),
}

/// Every estimand in one scope.
#[derive(Clone, Copy, Debug)]
pub struct ScopeEffects {
    /// the contrast: competent effect − incompetent effect; not the headline (see the module doc).
    pub effect: Estimate,
    /// The competent effect, the headline.
    pub competent: Estimate,
    pub incompetent: Estimate,
    pub ring: Estimate,
    /// The competent effect minus the ring effect.
    pub over_ring: Estimate,
    /// The competent effect under the section, spatial and identity steps (pooled only; NaN
    /// elsewhere).
    pub steps: [Estimate; 3],
    /// The competent effect without the neighbour-response and bleed covariates (pooled
    /// only; NaN elsewhere).
    pub nospill: Estimate,
    /// Share of the scope's competent rows whose
    /// overlap weight is below [`NEAR_DETERMINED_H`] in the main fit; NaN
    /// without competent rows or without the main fit, which is skipped
    /// when the competent subset leaves an arm thin ([`Reads`]), on every
    /// path alike.
    pub near_determined: f64,
    /// The evidence behind the average of `competent` over thresholds
    /// ([`abf_average`]).
    pub abf: AbfSummary,
}

impl ScopeEffects {
    /// Several analyses of one scope combined estimand by estimand with
    /// `f`; the near-determined share is their mean over the finite ones.
    pub fn zip_with(parts: &[ScopeEffects], f: impl Fn(&[Estimate]) -> Estimate) -> Self {
        let pick = |g: &dyn Fn(&ScopeEffects) -> Estimate| -> Estimate {
            f(&parts.iter().map(g).collect::<Vec<_>>())
        };
        let near: Vec<f64> = parts
            .iter()
            .map(|p| p.near_determined)
            .filter(|x| x.is_finite())
            .collect();
        Self {
            effect: pick(&|e| e.effect),
            competent: pick(&|e| e.competent),
            incompetent: pick(&|e| e.incompetent),
            ring: pick(&|e| e.ring),
            over_ring: pick(&|e| e.over_ring),
            steps: [
                pick(&|e| e.steps[0]),
                pick(&|e| e.steps[1]),
                pick(&|e| e.steps[2]),
            ],
            nospill: pick(&|e| e.nospill),
            abf: AbfSummary::NONE,
            near_determined: if near.is_empty() {
                f64::NAN
            } else {
                near.iter().sum::<f64>() / near.len() as f64
            },
        }
    }
}

/// One competence threshold's estimands per scope, and the cluster sums of
/// its pooled competent effect's linearization ψ (zeros when not estimable).
pub struct LevelEffects {
    pub scopes: Vec<ScopeEffects>,
    pub pooled_sums: Vec<f64>,
}

/// [`effects_on`]'s result: per competence threshold, and the main fit's
/// overlap weights `h` per row (empty when the main fit was skipped).
pub struct Effects {
    pub levels: Vec<LevelEffects>,
    pub h: Vec<f64>,
}

/// Which sensitivity fits [`effects_on`] runs beside the main and ring
/// fits; each serves the pooled scope alone, and an estimand left out is
/// NaN, which the average then leaves out.
#[derive(Clone, Copy, Debug)]
pub struct Sensitivity {
    /// The fit without the response-side bleed covariates.
    pub nospill: bool,
    /// The nested adjustment-step fits (section, spatial, identity).
    pub steps: bool,
}

impl Sensitivity {
    pub const ALL: Self = Self {
        nospill: true,
        steps: true,
    };
    pub const NONE: Self = Self {
        nospill: false,
        steps: false,
    };
}

/// The estimands on `rows` for each competence threshold
/// ([`ThresholdFits::levels`]) and each scope of `rows`, from one fit per
/// adjustment set ([`fit_main`]) of the response `r`; `fits` carries the
/// controls' propensities ([`ThresholdFits::with_controls`]). The sensitivity
/// fits `sens` asks for serve the pooled scope (the first). A fit none of
/// whose subsets has both arms at `min_arm` is skipped ([`Reads`]); without
/// the main fit, the near-determined shares are NaN and `h` is empty.
pub fn effects_on(
    rows: Rows<'_>,
    fits: &ThresholdFits<'_>,
    r: &Response,
    sens: Sensitivity,
    opts: &EffectOpts,
) -> anyhow::Result<Effects> {
    use timing::{time, MAIN, NOSPILL, STEPS};
    let (design, cf) = (rows.design.view(), rows.cf);
    let reads = fits.reads;
    let fit = |run: bool, kind: Fit, step: Option<usize>, d: DesignView<'_>| {
        run.then(|| fit_main(d, cf, fits, r, kind, step, opts))
            .transpose()
    };
    let (main, ring) = time(MAIN, || -> anyhow::Result<_> {
        Ok((
            fit(fits.runs(reads.competent), Fit::Main, None, design)?,
            fit(fits.runs(reads.ring), Fit::Ring, None, design)?,
        ))
    })?;
    let runs_competent = fits.runs(reads.competent);
    let nospill = time(NOSPILL, || {
        fit(sens.nospill && runs_competent, Fit::NoSpill, None, design)
    })?;
    let steps = if sens.steps && runs_competent {
        time(STEPS, || {
            rows.steps
                .iter()
                .enumerate()
                .map(|(k, s)| {
                    fit_main(
                        s.design.view(rows.design),
                        cf,
                        fits,
                        r,
                        Fit::Main,
                        Some(k),
                        opts,
                    )
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })?
    } else {
        Vec::new()
    };
    let (t, t_ring) = (&fits.exposure.t, &fits.exposure.t_ring);
    // Per scope, whether the competent effect has an estimate: the middle
    // threshold ([`Reads`]).
    let middle = &fits.levels[fits.levels.len() / 2];
    let headline: Vec<bool> = rows
        .scope_rows
        .iter()
        .map(|inside| arms_open(inside, t, middle, opts))
        .collect();
    let levels = fits
        .levels
        .iter()
        .map(|competent| {
            let mut pooled_sums = Vec::new();
            let scopes = rows
                .scope_rows
                .iter()
                .enumerate()
                .map(|(k, inside)| {
                    let (comp, incomp): (Vec<u32>, Vec<u32>) =
                        inside.iter().partition(|&&i| competent[i as usize]);
                    let open = headline[k];
                    // A subset's estimate from `terms`, or the guard's verdict
                    // on it when the fit was skipped or the scope has no
                    // headline.
                    let est = |terms: Option<&OverlapTerms>, t: &[bool], sub: &[u32]| match terms {
                        Some(terms) if open => opts.guard(subset_estimate(terms, t, sub, cf)),
                        _ => thin_subset(t, sub, cf),
                    };
                    let (on_competent, sums) = est(main.as_ref(), t, &comp);
                    let (on_incompetent, incompetent_sums) = est(main.as_ref(), t, &incomp);
                    let contrast =
                        difference(&on_competent, &sums, &on_incompetent, &incompetent_sums);
                    let (on_ring, ring_sums) = est(ring.as_ref(), t_ring, &comp);
                    let over_ring = difference(&on_competent, &sums, &on_ring, &ring_sums);
                    let nan = Estimate::nan(on_competent.n, on_competent.n_treated);
                    let mut step_est = [nan; 3];
                    let mut without_spill = nan;
                    let near_determined = match &main {
                        Some(m) if open && !comp.is_empty() => {
                            let near = comp
                                .iter()
                                .filter(|&&i| m.h[i as usize] < NEAR_DETERMINED_H)
                                .count();
                            near as f64 / comp.len() as f64
                        }
                        _ => f64::NAN,
                    };
                    if k == 0 {
                        if sens.nospill {
                            without_spill = est(nospill.as_ref(), t, &comp).0;
                        }
                        if sens.steps {
                            for (j, e) in step_est.iter_mut().enumerate() {
                                *e = est(steps.get(j), t, &comp).0;
                            }
                        }
                        pooled_sums = sums;
                    }
                    ScopeEffects {
                        effect: contrast,
                        competent: on_competent,
                        incompetent: on_incompetent,
                        ring: on_ring,
                        over_ring,
                        steps: step_est,
                        nospill: without_spill,
                        near_determined,
                        abf: AbfSummary::NONE,
                    }
                })
                .collect();
            LevelEffects {
                scopes,
                pooled_sums,
            }
        })
        .collect();
    let h = match main {
        Some(m) if reads.competent => m.h,
        _ => Vec::new(),
    };
    Ok(Effects { levels, h })
}

/// The pooled competent effect of the response `r` at the middle threshold with
/// the cluster sums of its ψ, from one main fit carrying every threshold
/// ([`fit_main`]); skipped when the subset leaves an arm thin ([`Reads`]). For
/// the curve and the background.
pub fn competent_effect(
    rows: Rows<'_>,
    fits: &ThresholdFits<'_>,
    r: &Response,
    opts: &EffectOpts,
) -> anyhow::Result<(Estimate, Vec<f64>)> {
    let (design, cf) = (rows.design.view(), rows.cf);
    let t = &fits.exposure.t;
    let middle = &fits.levels[fits.levels.len() / 2];
    let comp: Vec<u32> = rows.scope_rows[0]
        .iter()
        .copied()
        .filter(|&i| middle[i as usize])
        .collect();
    if !fits.runs(fits.reads.competent) {
        return Ok(thin_subset(t, &comp, cf));
    }
    let terms = fit_main(design, cf, fits, r, Fit::Main, None, opts)?;
    Ok(opts.guard(subset_estimate(&terms, t, &comp, cf)))
}

/// Each analysis combined into one by Rubin's rules,
/// estimand by estimand, scope by scope.
pub fn combine_scopes(per: &[Vec<ScopeEffects>]) -> Vec<ScopeEffects> {
    let Some(first) = per.first() else {
        return Vec::new();
    };
    (0..first.len())
        .map(|s| {
            let parts: Vec<ScopeEffects> = per.iter().map(|d| d[s]).collect();
            ScopeEffects::zip_with(&parts, super::overlap::combine)
        })
        .collect()
}

// * Threshold–response curve

/// The curve's thresholds as `(τ, quantile)`: τ = 0 (any ligand on a
/// neighbour), the [`CURVE_QUANTILES`] of the positive exposures, and the
/// headline τ; each kept only when it leaves `min_arm` `eligible` rows on
/// either side. `quantile` is the share of positive exposures at or below
/// τ. Strictly increasing in τ.
pub fn curve_taus(a: &[f64], headline: f64, eligible: &[bool], min_arm: usize) -> Vec<(f64, f64)> {
    let mut pos: Vec<f64> = a.iter().copied().filter(|&v| v > 0.0).collect();
    if pos.is_empty() {
        return Vec::new();
    }
    pos.sort_by(f64::total_cmp);
    let at = |q: f64| pos[(q * (pos.len() - 1) as f64).round() as usize];
    let mut cand: Vec<f64> = std::iter::once(0.0)
        .chain(CURVE_QUANTILES.iter().map(|&q| at(q)))
        .chain(std::iter::once(headline))
        .filter(|t| t.is_finite())
        .collect();
    cand.sort_by(f64::total_cmp);
    cand.dedup();
    cand.into_iter()
        .filter(|&tau| {
            let (mut above, mut below) = (0usize, 0usize);
            for (&v, &e) in a.iter().zip(eligible) {
                if e {
                    if v > tau {
                        above += 1;
                    } else {
                        below += 1;
                    }
                }
            }
            above >= min_arm && below >= min_arm
        })
        .map(|tau| {
            let q = pos.partition_point(|&v| v <= tau) as f64 / pos.len() as f64;
            (tau, q)
        })
        .collect()
}

/// The sup-t critical value of a simultaneous 95% band over `K` estimates,
/// by a Gaussian multiplier bootstrap over clusters.
///
/// Each row of `stacked` is one cluster's sums of ψ for the `K` estimates
/// (one row per cluster and analysis, analyses being the competence and weighting choices
/// the estimates average over, so `ΣᵀΣ` is their summed covariance). A
/// bootstrap draw multiplies each row by an independent standard normal,
/// `T_k = Σ_r g_r S_rk / √V_kk` with `V_kk = Σ_r S_rk²`, and the critical
/// value is the 95% quantile of `max_k |T_k|` over [`BAND_DRAWS`] draws.
/// Columns with `V_kk = 0` are left out; NaN when none is left.
pub fn sup_t_crit(stacked: &[Vec<f64>], k: usize, seed: u64) -> f64 {
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut v = vec![0.0f64; k];
    for row in stacked {
        for (vk, x) in v.iter_mut().zip(row) {
            *vk += x * x;
        }
    }
    let keep: Vec<usize> = (0..k).filter(|&j| v[j] > 0.0).collect();
    if keep.is_empty() {
        return f64::NAN;
    }
    let sd: Vec<f64> = keep.iter().map(|&j| v[j].sqrt()).collect();
    let rows: Vec<&Vec<f64>> = stacked
        .iter()
        .filter(|r| keep.iter().any(|&j| r[j] != 0.0))
        .collect();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut t = vec![0.0f64; keep.len()];
    let mut sup: Vec<f64> = (0..BAND_DRAWS)
        .map(|_| {
            t.fill(0.0);
            for row in &rows {
                let g: f64 = StandardNormal.sample(&mut rng);
                for (tj, &j) in t.iter_mut().zip(&keep) {
                    *tj += g * row[j];
                }
            }
            t.iter()
                .zip(&sd)
                .map(|(x, s)| (x / s).abs())
                .fold(0.0, f64::max)
        })
        .collect();
    sup.sort_by(f64::total_cmp);
    sup[((0.95 * BAND_DRAWS as f64).ceil() as usize).min(BAND_DRAWS) - 1]
}

/// The curve from its points' combined estimates and the stacked cluster
/// sums. The bootstrap's critical value is a normal-reference one; with few
/// clusters each point's is scaled by its t reference over the normal's,
/// `c_k = c · t_{df_k} / z`, and never below the pointwise `t_{df_k}`.
pub fn build_curve(
    taus: &[(f64, f64)],
    est: &[Estimate],
    stacked: &[Vec<f64>],
    seed: u64,
) -> Vec<CurvePoint> {
    let ok: Vec<bool> = est
        .iter()
        .map(|e| e.theta.is_finite() && e.se.is_finite())
        .collect();
    let masked: Vec<Vec<f64>> = stacked
        .iter()
        .map(|r| {
            r.iter()
                .zip(&ok)
                .map(|(&x, &o)| if o { x } else { 0.0 })
                .collect()
        })
        .collect();
    let crit = sup_t_crit(&masked, est.len(), seed);
    taus.iter()
        .zip(est)
        .map(|(&(tau, q), e)| {
            let c = crit * e.crit() / super::overlap::Z_975;
            CurvePoint::new(tau, q, *e, c)
        })
        .collect()
}

// * Background programme

/// Genes for the background pool: up to [`BACKGROUND_POOL`] of the `n_genes` genes
/// not in `exclude` (every ligand, receptor and programme gene), and at
/// most a fifth of them so a small panel keeps most of its genes for the
/// identity; drawn at random with `seed`, sorted.
pub fn background_pool_genes(n_genes: usize, exclude: &[usize], seed: u64) -> Vec<usize> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let ex: std::collections::HashSet<usize> = exclude.iter().copied().collect();
    let mut cand: Vec<usize> = (0..n_genes).filter(|g| !ex.contains(g)).collect();
    let k = BACKGROUND_POOL.min(cand.len() / 5);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed ^ 0x000d_ec07);
    cand.shuffle(&mut rng);
    cand.truncate(k);
    cand.sort_unstable();
    cand
}

/// Mean of `x / size factor` over all rows.
pub fn mean_expression(nz: &[(u32, f32)], sf: &[f64]) -> f64 {
    nz.iter()
        .map(|&(r, x)| x as f64 / sf[r as usize])
        .sum::<f64>()
        / sf.len().max(1) as f64
}

/// Genes outside every pair and programme, by mean expression, from which
/// each pair's background programme is matched.
#[derive(Default)]
pub struct BackgroundPool<'a> {
    /// `(log mean expression, non-zeros)`, sorted by the first.
    genes: Vec<(f64, &'a [(u32, f32)])>,
}

impl<'a> BackgroundPool<'a> {
    /// The pool from its genes' non-zero counts on the tissue's rows; genes
    /// never detected are left out.
    pub fn new(sf: &[f64], genes: Vec<&'a [(u32, f32)]>) -> Self {
        let mut genes: Vec<(f64, &'a [(u32, f32)])> = genes
            .into_iter()
            .filter(|nz| !nz.is_empty())
            .map(|nz| (mean_expression(nz, sf).ln(), nz))
            .collect();
        genes.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self { genes }
    }

    /// Each target's weight and log mean expression, by decreasing `|w|`: the
    /// order [`BackgroundPool::draw`] matches them in.
    pub fn plan(sf: &[f64], targets: &[(f64, &[(u32, f32)])]) -> Vec<(f64, f64)> {
        let mut order: Vec<usize> = (0..targets.len()).collect();
        order.sort_by(|&a, &b| targets[b].0.abs().total_cmp(&targets[a].0.abs()));
        order
            .into_iter()
            .map(|k| {
                let (w, nz) = targets[k];
                (w, mean_expression(nz, sf).max(1e-12).ln())
            })
            .collect()
    }

    /// A background programme for a programme: a random unused gene among the
    /// [`BACKGROUND_NEIGHBOURS`] nearest in log mean expression to each target of
    /// `plan`, with the target's weight. Fewer genes than targets when the
    /// pool runs out; empty when it holds fewer than half the programme.
    pub fn draw(&self, plan: &[(f64, f64)], seed: u64) -> Vec<(f64, &'a [(u32, f32)])> {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut used = vec![false; self.genes.len()];
        let mut out = Vec::with_capacity(plan.len());
        for &(w, key) in plan {
            let at = self.genes.partition_point(|g| g.0 < key);
            // The nearest unused, walking outward from the target's place.
            let (mut lo, mut hi) = (at, at);
            let mut near: Vec<usize> = Vec::with_capacity(BACKGROUND_NEIGHBOURS);
            while near.len() < BACKGROUND_NEIGHBOURS && (lo > 0 || hi < used.len()) {
                let go_low = match (lo > 0, hi < used.len()) {
                    (true, true) => key - self.genes[lo - 1].0 <= self.genes[hi].0 - key,
                    (low, _) => low,
                };
                let g = if go_low {
                    lo -= 1;
                    lo
                } else {
                    hi += 1;
                    hi - 1
                };
                if !used[g] {
                    near.push(g);
                }
            }
            if near.is_empty() {
                break;
            }
            let j = near[rng.random_range(0..near.len())];
            used[j] = true;
            out.push((w, self.genes[j].1));
        }
        if 2 * out.len() < plan.len() || out.is_empty() {
            return Vec::new();
        }
        out
    }
}

// * One pair

/// Every estimand of a pair averaged over its exposure thresholds
/// ([`over_thresholds`]).
pub struct ThresholdAverage {
    pub scopes: Vec<ScopeEffects>,
    pub background: Estimate,
    pub over_background: Estimate,
    /// Each threshold fitted, the headline first: its first weighting's
    /// effect at the middle competence threshold and that effect's cluster
    /// sums, what a curve reads.
    pub points: Vec<(f64, Estimate, Vec<f64>)>,
    /// The main fit's overlap weights at the headline threshold under the
    /// first weighting; empty when that fit was skipped.
    pub h: Vec<f64>,
}

/// The thresholds a pair is read at beyond the headline: none without
/// [`EffectOpts::curve`], else [`curve_taus`] on the rows competent at some level.
pub fn thresholds(ex: &Exposure, levels: &[Vec<bool>], opts: &EffectOpts) -> Vec<(f64, f64)> {
    if !opts.curve {
        return Vec::new();
    }
    let eligible: Vec<bool> = (0..ex.dose.len())
        .map(|i| levels.iter().any(|c| c[i]))
        .collect();
    curve_taus(&ex.dose, ex.tau, &eligible, opts.min_arm)
}

/// Every estimand of a pair over the headline threshold and `taus`: the fits at
/// each, the estimands combined over competence thresholds and programme
/// weightings by Rubin's rules, then averaged over the exposure thresholds by
/// their posterior weights, estimand by estimand ([`abf_average`]), so each
/// control is averaged by its own evidence and not by the programme's.
///
/// A fixed [`EffectOpts::threshold`] is not averaged: its estimate is the
/// headline, and `taus` only the curve. A headline threshold with no estimate
/// ends the pair: the other thresholds are not fitted, nor are the background programmes made.
/// `sens` runs at the headline threshold alone, and the background programmes read the middle
/// competence threshold.
#[allow(clippy::too_many_arguments)]
pub fn over_thresholds(
    rows: Rows<'_>,
    ex: &Exposure,
    taus: &[(f64, f64)],
    responses: &[Response],
    make_backgrounds: impl Fn() -> Vec<Response>,
    levels: &[Vec<bool>],
    sens: Sensitivity,
    opts: &EffectOpts,
) -> anyhow::Result<ThresholdAverage> {
    let mid = levels.len() / 2;
    // The headline first; a curve threshold equal to it is read off its fits.
    let mut grid: Vec<f64> = vec![ex.tau];
    grid.extend(taus.iter().map(|t| t.0).filter(|&t| t != ex.tau));
    let mut starts = Starts::default();
    let mut at_tau: Vec<Vec<ScopeEffects>> = Vec::with_capacity(grid.len());
    let mut points = Vec::with_capacity(grid.len());
    let mut h = Vec::new();
    let mut background_responses: Vec<Response> = Vec::new();
    let mut background_at: Vec<Vec<Estimate>> = Vec::with_capacity(grid.len());
    for (g, &tau) in grid.iter().enumerate() {
        let clock = std::time::Instant::now();
        let other = (g > 0).then(|| ex.at(tau));
        let exposure = other.as_ref().unwrap_or(ex);
        let sens = if g == 0 { sens } else { Sensitivity::NONE };
        let fits = timing::time(timing::MAIN, || {
            ThresholdFits::new(rows, exposure, levels, opts, &mut starts.main)
        })?
        .with_controls(rows, sens.steps, opts, &mut starts)?;
        if g == 0 {
            if fits.runs(fits.reads.competent) {
                background_responses = timing::time(timing::FRAME, &make_backgrounds);
            } else {
                timing::all_skipped();
            }
        }
        let mut per_weighting = Vec::with_capacity(responses.len());
        for (w, r) in responses.iter().enumerate() {
            let per_c = effects_on(rows, &fits, r, sens, opts)?;
            if w == 0 {
                let head = &per_c.levels[mid];
                points.push((tau, head.scopes[0].competent, head.pooled_sums.clone()));
                if g == 0 {
                    h = per_c.h;
                }
            }
            let scopes: Vec<Vec<ScopeEffects>> =
                per_c.levels.into_iter().map(|l| l.scopes).collect();
            let mut combined = combine_scopes(&scopes);
            // The counts of the level that decides whether a pair is estimable.
            for (c, m) in combined.iter_mut().zip(&scopes[mid]) {
                c.competent.n = m.competent.n;
                c.competent.n_treated = m.competent.n_treated;
            }
            per_weighting.push(combined);
        }
        at_tau.push(combine_scopes(&per_weighting));
        if g > 0 {
            timing::add(timing::THRESHOLDS, clock);
        }
        background_at.push(
            background_responses
                .iter()
                .map(|r| {
                    timing::time(timing::BACKGROUND, || {
                        competent_effect(rows, &fits, r, opts)
                    })
                    .map(|e| e.0)
                })
                .collect::<anyhow::Result<Vec<_>>>()?,
        );
        // No estimate at the headline threshold: nothing to average or to put
        // a curve on.
        if g == 0 && !fits.reads.competent {
            break;
        }
    }
    let prior_var = opts.abf_prior_sd * opts.abf_prior_sd;
    let averaged = if opts.threshold.is_some() {
        1
    } else {
        at_tau.len()
    };
    let scopes: Vec<ScopeEffects> = (0..at_tau[0].len())
        .map(|s| {
            let parts: Vec<ScopeEffects> = at_tau[..averaged].iter().map(|a| a[s]).collect();
            let competent: Vec<Estimate> = parts.iter().map(|p| p.competent).collect();
            ScopeEffects {
                near_determined: parts[0].near_determined,
                abf: abf_summary(&competent, &grid, prior_var),
                ..ScopeEffects::zip_with(&parts, |p| abf_average(p, prior_var))
            }
        })
        .collect();
    // Each background averaged over the thresholds by its own evidence, as the effect is.
    let background_means: Vec<Estimate> = (0..background_responses.len())
        .map(|k| {
            let parts: Vec<Estimate> = background_at[..averaged].iter().map(|d| d[k]).collect();
            abf_average(&parts, prior_var)
        })
        .collect();
    let (background, over_background) =
        background_comparison(&scopes[0].competent, &background_means);
    Ok(ThresholdAverage {
        scopes,
        background,
        over_background,
        points,
        h,
    })
}

/// The curve at `taus` from the fitted `points`, in the order of its thresholds:
/// one row of stacked sums per variance cluster that carries any.
fn threshold_curve(
    n_clusters: usize,
    taus: &[(f64, f64)],
    points: &[(f64, Estimate, Vec<f64>)],
    seed: u64,
) -> Vec<CurvePoint> {
    if taus.is_empty() {
        return Vec::new();
    }
    let point = |tau: f64| points.iter().find(|p| p.0 == tau);
    let estimates: Vec<Estimate> = taus
        .iter()
        .map(|t| point(t.0).map_or_else(|| combine(&[]), |p| p.1))
        .collect();
    let mut stacked: Vec<Vec<f64>> = Vec::new();
    for c in 0..n_clusters {
        let row: Vec<f64> = taus
            .iter()
            .map(|t| point(t.0).and_then(|p| p.2.get(c).copied()).unwrap_or(0.0))
            .collect();
        if row.iter().any(|&x| x != 0.0) {
            stacked.push(row);
        }
    }
    timing::time(timing::BAND, || {
        build_curve(taus, &estimates, &stacked, seed)
    })
}

/// `b`'s weights on `a`'s genes (0 where `b` leaves a gene out), when
/// `b`'s genes are a subsequence of `a`'s; `None` otherwise.
fn aligned(a: &[(f64, &[(u32, f32)])], b: &[(f64, &[(u32, f32)])]) -> Option<Vec<f64>> {
    let mut j = 0;
    let w = a
        .iter()
        .map(|(_, nz)| match b.get(j) {
            Some(&(w, g)) if std::ptr::eq(g, *nz) => {
                j += 1;
                w
            }
            _ => 0.0,
        })
        .collect();
    (j == b.len()).then_some(w)
}

/// Rows for the pooled tissue, each section and each named receiver cell
/// type, with a background programme matched from `pool`. Verdicts and q-values
/// are left empty for [`report`], which needs every pair.
///
/// With [`PairData::reweighted`], the two weightings of the programme are
/// combined ([`over_thresholds`]); the curve and the background describe the
/// pair, not the weighting, and are read under `targets`. Counts,
/// co-expression, the crude contrast and the diagnostics read competence
/// at the middle threshold of the grid.
pub fn estimate_pair_with(
    tissue: &Tissue,
    pair: &PairData<'_>,
    pool: &BackgroundPool<'_>,
    opts: &EffectOpts,
) -> anyhow::Result<Vec<EffectRow>> {
    let n = tissue.n();
    let ex = timing::time(timing::FRAME, || {
        Exposure::build(tissue, pair.ligand_counts, opts)
    });
    let detected: Vec<bool> = pair.receptor_counts.iter().map(|&r| r > 0.0).collect();
    let comps = super::exposure::competence_grid(&tissue.identity_bin, &detected, opts.competences);
    let taus = thresholds(&ex, &comps, opts);
    let responses: Vec<Response> = timing::time(timing::FRAME, || {
        match pair.reweighted.and_then(|b| aligned(pair.targets, b)) {
            Some(wb) => {
                let (genes, wa) = genes_and_weights(pair.targets);
                Response::build_many(tissue, &genes, &[wa, wb])
            }
            None => std::iter::once(pair.targets)
                .chain(pair.reweighted)
                .map(|p| Response::build(tissue, p))
                .collect(),
        }
    });
    // Seeded by the pair, so a pair's background programmes don't depend on which others run.
    let pair_seed = name_seed(opts.seed, &format!("{}\0{}", pair.ligand, pair.receptor));
    let plan = BackgroundPool::plan(&tissue.size_factor, pair.targets);
    let backgrounds: Vec<_> = (0..opts.background_draws)
        .map(|k| pool.draw(&plan, mix_seed(pair_seed, k as u64)))
        .filter(|d| !d.is_empty())
        .collect();
    let n_background_genes = backgrounds.first().map_or(0, Vec::len);
    let eff = over_thresholds(
        Rows::cells(tissue),
        &ex,
        &taus,
        &responses,
        || {
            backgrounds
                .par_iter()
                .map(|d| Response::build(tissue, d))
                .collect()
        },
        &comps,
        Sensitivity::ALL,
        opts,
    )?;
    let mut curve = Some(threshold_curve(
        tissue.cf.n_clusters(),
        &taus,
        &eff.points,
        mix_seed(opts.seed, 0xc0_ffee),
    ));
    // The response of each weighting, for the crude contrast.
    let ys: Vec<Vec<f64>> = responses.into_iter().map(|r| r.y).collect();
    let competent = &comps[comps.len() / 2];
    let mut rows = Vec::with_capacity(tissue.scopes.len());
    let scopes = tissue.scopes.iter().zip(&tissue.scope_rows);
    for (((label, scope), inside_rows), e) in scopes.zip(eff.scopes) {
        let mut inside = vec![false; n];
        for &i in inside_rows {
            inside[i as usize] = true;
        }
        let comp: Vec<u32> = inside_rows
            .iter()
            .copied()
            .filter(|&i| competent[i as usize])
            .collect();
        let (lor, lse) = coexpression(tissue, pair.ligand_counts, pair.receptor_counts, &inside);
        let bins = |side: bool| -> usize {
            let set: std::collections::HashSet<usize> = inside_rows
                .iter()
                .map(|&i| i as usize)
                .filter(|&i| competent[i] == side)
                .map(|i| tissue.identity_bin[i])
                .collect();
            set.len()
        };
        // The crude contrast of each weighting, averaged.
        let crude = {
            let c: Vec<f64> = ys
                .iter()
                .map(|y| crude_difference(y, &ex.t, &comp))
                .collect();
            if c.len() == 2 {
                0.5 * (c[0] + c[1])
            } else {
                c[0]
            }
        };
        let pooled = *scope == Scope::All;
        // How far neighbour identity explains the exposure, at the threshold
        // the average leans on.
        let eta2 = eta_squared(
            &tissue.neighbour_bin,
            &indicator(&ex.leaned_on(e.abf)),
            inside_rows,
        );
        let (background, over_background) = if pooled {
            (eff.background, eff.over_background)
        } else {
            (Estimate::nan(0, 0), Estimate::nan(0, 0))
        };
        rows.push(EffectRow {
            section: label.clone(),
            ligand: pair.ligand.into(),
            receptor: pair.receptor.into(),
            n_targets: pair.targets.len(),
            n_removed: pair.n_removed,
            n_receivers: inside_rows.len(),
            n_competent: comp.len(),
            n_exposed: inside_rows.iter().filter(|&&i| ex.t[i as usize]).count(),
            threshold: ex.tau,
            crude,
            n_competent_bins: bins(true),
            n_incompetent_bins: bins(false),
            curve: if pooled {
                curve.take().unwrap_or_default()
            } else {
                Vec::new()
            },
            ..EffectRow::pending(
                e,
                (lor, lse),
                eta2,
                (background, over_background),
                if pooled { n_background_genes } else { 0 },
            )
        });
    }
    Ok(rows)
}

/// The co-expression log odds over the contacts inside the scope, both orientations: the Jeffreys
/// log odds ratio of ligand detected in the sender and receptor detected in
/// the receiver, and its SE.
pub fn coexpression(
    tissue: &Tissue,
    ligand_counts: &[f64],
    receptor_counts: &[f64],
    inside: &[bool],
) -> (f64, f64) {
    let (mut n, mut n11, mut nl, mut nr) = (0usize, 0usize, 0usize, 0usize);
    for (i, adj) in tissue.contacts.adj.iter().enumerate() {
        if !inside[i] {
            continue;
        }
        let l = ligand_counts[i] > 0.0;
        for &j in adj {
            let j = j as usize;
            if !inside[j] {
                continue;
            }
            // Each contact is visited from both ends: one oriented instance
            // per visit, sender i, receiver j.
            let r = receptor_counts[j] > 0.0;
            n += 1;
            nl += l as usize;
            nr += r as usize;
            n11 += (l && r) as usize;
        }
    }
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let (lor, se) = jeffreys_log_odds(n, n11, nl, nr);
    (lor as f64, se as f64)
}

// * Stage timing

/// Wall time per stage summed over calls (and threads), for the run log.
pub mod timing {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const STAGES: [&str; 9] = [
        "frame",
        "main+ring fits",
        "no-spill fit",
        "step fits",
        "other thresholds",
        "background fit",
        "band",
        "(in fits: outcome ridge",
        "propensity)",
    ];
    pub const RIDGE: usize = 7;
    pub const LOGISTIC: usize = 8;
    pub const FRAME: usize = 0;
    pub const MAIN: usize = 1;
    pub const NOSPILL: usize = 2;
    pub const STEPS: usize = 3;
    pub const THRESHOLDS: usize = 4;
    pub const BACKGROUND: usize = 5;
    pub const BAND: usize = 6;

    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    static NS: [AtomicU64; 9] = [ZERO; 9];
    static ALL_SKIPPED: AtomicU64 = AtomicU64::new(0);
    static PROPENSITY_FITS: AtomicU64 = AtomicU64::new(0);
    static HESSIANS: AtomicU64 = AtomicU64::new(0);
    static UNCONVERGED: AtomicU64 = AtomicU64::new(0);

    /// One logistic fit that computed `hessians` Hessians; `converged`
    /// false when it stopped at the step guard, not at the optimum.
    pub fn newton(hessians: usize, converged: bool) {
        PROPENSITY_FITS.fetch_add(1, Ordering::Relaxed);
        HESSIANS.fetch_add(hessians as u64, Ordering::Relaxed);
        if !converged {
            UNCONVERGED.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn time<T>(stage: usize, f: impl FnOnce() -> T) -> T {
        let t = std::time::Instant::now();
        let out = f();
        NS[stage].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    }

    pub fn add(stage: usize, since: std::time::Instant) {
        NS[stage].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// One pair none of whose fits ran: every subset left an arm thin.
    pub fn all_skipped() {
        ALL_SKIPPED.fetch_add(1, Ordering::Relaxed);
    }

    /// Thread-seconds per stage and the pairs
    /// with every fit skipped.
    pub fn summary() -> String {
        let stages: Vec<String> = STAGES
            .iter()
            .zip(&NS)
            .map(|(s, n)| format!("{s} {:.0}s", n.load(Ordering::Relaxed) as f64 * 1e-9))
            .collect();
        let fits = PROPENSITY_FITS.load(Ordering::Relaxed);
        format!(
            "{}; {} pairs with every fit skipped (thin \
             arms); {} propensity fold fits, {:.1} Hessians each, {} not converged",
            stages.join(", "),
            ALL_SKIPPED.load(Ordering::Relaxed),
            fits,
            HESSIANS.load(Ordering::Relaxed) as f64 / fits.max(1) as f64,
            UNCONVERGED.load(Ordering::Relaxed)
        )
    }
}

// * Inference across pairs

/// What [`report`] needs beyond the rows.
#[derive(Clone, Copy, Debug)]
pub struct ReportOpts {
    /// BH level for every q-value the verdict reads.
    pub fdr: f64,
    /// `exposure_eta2` at or above which a pair is "cell-type confounded".
    pub max_exposure_eta2: f64,
    /// [`EffectOpts::min_arm`], to say why a pair is not estimable.
    pub min_arm: usize,
}

impl Default for ReportOpts {
    fn default() -> Self {
        super::INTERNALS.report_opts()
    }
}

/// Benjamini–Hochberg q-values. Non-finite p are left out of the count and
/// get NaN.
pub fn bh(p: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_finite()).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let m = idx.len() as f64;
    let mut q = vec![f64::NAN; p.len()];
    let mut run = 1.0f64;
    for (r, &i) in idx.iter().enumerate().rev() {
        run = run.min(p[i] * m / (r + 1) as f64);
        q[i] = run;
    }
    q
}

/// Simes p of one family, `min_r p_(r) k / r` over its `k` finite p; NaN
/// when there are none.
pub fn simes(p: &[f64]) -> f64 {
    let mut v: Vec<f64> = p.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let k = v.len() as f64;
    v.iter()
        .enumerate()
        .map(|(r, &x)| x * k / (r + 1) as f64)
        .fold(1.0, f64::min)
}

/// η² of `x` by `group` over `rows`: between-group sum of squares over
/// the total. 1 when `x` is a function of the group, near 0 when unrelated
/// (biased up by about `(groups − 1) / n`); NaN when `x` is constant.
pub fn eta_squared(group: &[usize], x: &[f64], rows: &[u32]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    let mean = rows.iter().map(|&i| x[i as usize]).sum::<f64>() / rows.len() as f64;
    let total: f64 = rows.iter().map(|&i| (x[i as usize] - mean).powi(2)).sum();
    if total <= 1e-12 {
        return f64::NAN;
    }
    // Ordered, so the sum is the same in every run.
    let mut by: std::collections::BTreeMap<usize, (f64, usize)> = Default::default();
    for &i in rows {
        let e = by.entry(group[i as usize]).or_default();
        e.0 += x[i as usize];
        e.1 += 1;
    }
    let between: f64 = by
        .values()
        .map(|&(s, c)| c as f64 * (s / c as f64 - mean).powi(2))
        .sum();
    (between / total).min(1.0)
}

/// What one row's verdict rests on.
#[derive(Clone, Copy, Debug, Default)]
pub struct Evidence {
    /// The headline (the competent effect) has an estimate.
    pub estimable: bool,
    /// The headline's q is below the FDR level.
    pub effect: bool,
    /// The co-expression q is below the FDR level.
    pub coexpr: bool,
    /// A control beside a significant effect did not hold: the effect does not
    /// exceed the incompetent, ring or background effect, or one could not be checked.
    pub control_failed: bool,
    /// `exposure_eta2` reaches the internal guard `max_exposure_eta2`
    /// (`outputs.lra.settings` in the manifest).
    pub confounded: bool,
}

/// The verdict. In order: no headline estimate; exposure that cannot be
/// told from neighbour identity (then a significant effect is not a
/// finding); a significant effect beside a failed control; else the 2 × 2
/// of co-expression and effect.
pub fn verdict(e: Evidence) -> Verdict {
    if !e.estimable {
        Verdict::InsufficientData
    } else if e.confounded {
        Verdict::CelltypeInseparable
    } else if e.control_failed && e.effect {
        Verdict::ControlFailed
    } else {
        match (e.coexpr, e.effect) {
            (true, true) => Verdict::Supported,
            (true, false) => Verdict::ColocationOnly,
            (false, true) => Verdict::EffectWithoutCoexpression,
            (false, false) => Verdict::NoEvidence,
        }
    }
}

/// p-values, BH q-values and verdicts for every row, within each section
/// (the pooled rows, `all`, form their own family).
///
/// Pairs sharing a ligand share its exposure (the same `T`), and pairs
/// sharing senders share neighbourhoods, so their tests are positively
/// correlated. BH stays valid under positive regression dependence (PRDS;
/// Benjamini & Yekutieli 2001, Ann Stat 29:1165), which one-sided normal
/// tests with non-negative correlations satisfy; for the two-sided effect
/// test it is an approximation. `ligand_q` asks the coarser question with
/// one test per ligand, so the strongest correlation, within a ligand, stays
/// inside a family: each ligand's headline p are combined by Simes
/// (`min_r p_(r) k / r` over its `k` pairs, valid under PRDS) and BH runs
/// over ligands. `n_pairs_same_ligand` is that `k`.
///
/// The controls (receptor-negative receivers, distant cells, the background, and
/// the contrast with the first) are corrected only among the pairs
/// whose effect is significant, and are NaN for the rest: a control decides a
/// verdict only beside a significant effect, and over every pair its q would
/// rest on what unrelated pairs in the table happen to show.
pub fn report(rows: &mut [EffectRow], opts: &ReportOpts) {
    let mut by_section: std::collections::HashMap<Box<str>, Vec<usize>> = Default::default();
    for (i, r) in rows.iter().enumerate() {
        by_section.entry(r.section.clone()).or_default().push(i);
    }
    for idx in by_section.values() {
        let col = |g: &dyn Fn(&EffectRow) -> f64| -> Vec<f64> {
            idx.iter().map(|&i| g(&rows[i])).collect()
        };
        let pe = col(&|r| r.effects.competent.p());
        let pc = col(&|r| upper_normal_p(r.coexpr_log_or / r.coexpr_se));
        let qe = bh(&pe);
        let qc = bh(&pc);
        let control = |g: &dyn Fn(&EffectRow) -> f64| -> Vec<f64> {
            let p: Vec<f64> = col(g)
                .into_iter()
                .zip(&qe)
                .map(|(p, &q)| if q < opts.fdr { p } else { f64::NAN })
                .collect();
            bh(&p)
        };
        let q1 = control(&|r| r.effects.effect.p());
        let q3 = control(&|r| r.effects.incompetent.p());
        let q5 = control(&|r| r.effects.ring.p());
        let qd = control(&|r| r.background_effect.p());
        let qr = control(&|r| r.effects.over_ring.p());
        let qo = control(&|r| r.effect_over_background.p());
        let mut by_ligand: std::collections::HashMap<String, Vec<usize>> = Default::default();
        for (k, &i) in idx.iter().enumerate() {
            by_ligand
                .entry(rows[i].ligand.to_ascii_uppercase())
                .or_default()
                .push(k);
        }
        let families: Vec<&Vec<usize>> = by_ligand.values().collect();
        let simes_p: Vec<f64> = families
            .iter()
            .map(|ks| simes(&ks.iter().map(|&k| pe[k]).collect::<Vec<_>>()))
            .collect();
        let ql = bh(&simes_p);
        for (g, ks) in families.iter().enumerate() {
            for &k in ks.iter() {
                let r = &mut rows[idx[k]];
                r.ligand_q = ql[g];
                r.n_pairs_same_ligand = ks.len();
            }
        }
        for (k, &i) in idx.iter().enumerate() {
            let r = &mut rows[i];
            r.effect_p = pe[k];
            r.effect_q = qe[k];
            r.coexpr_p = pc[k];
            r.coexpr_q = qc[k];
            r.effect_contrast_q = q1[k];
            r.effect_incompetent_q = q3[k];
            r.ring_effect_q = q5[k];
            r.background_q = qd[k];
            r.effect_over_ring_q = qr[k];
            r.effect_over_background_q = qo[k];
        }
    }
    for r in rows.iter_mut() {
        judge(r, opts);
    }
}

/// Verdict and warnings of one row from its q-values and diagnostics.
fn judge(row: &mut EffectRow, opts: &ReportOpts) {
    let sig = |q: f64| q < opts.fdr;
    let head = row.effects.competent;
    let effect = sig(row.effect_q);
    // Beside a significant effect, the effect must exceed each negative
    // control, the same way, and a comparison that could not run has not held,
    // but for a background programme where none exists (no match in the pool, or a section's
    // row). A hidden field moves the controls too, so one being significant
    // does not rule the effect out: it is reported, and what decides is
    // whether the effect is larger.
    let unchecked = |q: f64| effect && q.is_nan();
    let exceeds = |q: f64, over: Estimate| sig(q) && over.theta * head.theta > 0.0;
    let beaten = |q: f64, over: Estimate| effect && !q.is_nan() && !exceeds(q, over);
    let (negative_q, ring_q, background_q) = (
        row.effect_incompetent_q,
        row.ring_effect_q,
        row.background_q,
    );
    let (over_negative_q, over_ring_q) = (row.effect_contrast_q, row.effect_over_ring_q);
    let over_background_q = row.effect_over_background_q;
    let has_background = row.n_background_genes > 0;
    let controls = [
        (
            unchecked(over_negative_q),
            "receptor-negative receivers could not be checked",
        ),
        (
            beaten(over_negative_q, row.effects.effect),
            "the effect does not exceed the one in receptor-negative receivers",
        ),
        (
            unchecked(over_ring_q),
            "distant senders could not be checked",
        ),
        (
            beaten(over_ring_q, row.effects.over_ring),
            "the effect does not exceed the one from distant senders",
        ),
        (
            has_background && unchecked(over_background_q),
            "the matched background programme could not be checked",
        ),
        (
            has_background && beaten(over_background_q, row.effect_over_background),
            "the effect does not exceed the one on a matched background programme",
        ),
    ];
    let notes = [
        (sig(negative_q), "receptor-negative receivers respond"),
        (sig(ring_q), "distant senders have an effect"),
        (
            sig(background_q),
            "a matched background programme responds too",
        ),
    ];
    let ev = Evidence {
        estimable: head.theta.is_finite() && head.se.is_finite(),
        effect,
        coexpr: sig(row.coexpr_q),
        control_failed: controls.iter().any(|c| c.0),
        confounded: row.exposure_eta2 >= opts.max_exposure_eta2,
    };
    row.verdict = verdict(ev);
    let mut w: Vec<String> = Vec::new();
    if !ev.estimable {
        w.push(if row.n_competent == 0 {
            "not estimable: no receptor-competent receivers".into()
        } else if thin_arm(opts.min_arm, head.n, head.n_treated) {
            format!(
                "not estimable: thin arm ({} exposed, {} unexposed competent receivers; \
                 internal guard min_arm {}, see outputs.lra.settings in the manifest)",
                head.n_treated,
                head.n.saturating_sub(head.n_treated),
                opts.min_arm
            )
        } else if head.df == ONE_CLUSTER_DF {
            "not estimable: the competent receivers lie in one variance cluster (spatial \
             tile), so the standard error has no degrees of freedom"
                .into()
        } else {
            "not estimable: low overlap (mean overlap weight or its effective sample size \
             under the internal guards min_overlap / min_ess, see outputs.lra.settings in the \
             manifest)"
                .into()
        });
    }
    if ev.confounded {
        w.push(format!(
            "exposure {:.0}% explained by the dominant neighbour's identity: the effect \
             cannot be told from neighbour cell type",
            100.0 * row.exposure_eta2
        ));
        if ev.coexpr {
            w.push("co-expression is likely a cell-type artefact".into());
        }
    }
    w.extend(
        controls
            .iter()
            .chain(&notes)
            .filter(|c| c.0)
            .map(|c| c.1.to_string()),
    );
    row.warnings = w.join("; ");
}
