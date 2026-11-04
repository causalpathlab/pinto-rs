//! One ligand-receptor pair: exposure, response, and every estimand.
//!
//! | Column | Estimand |
//! |---|---|
//! | `coexpr_log_or` | E0: log odds of ligand on one side and receptor on the other, across contacts (association) |
//! | `crude` | E0′: mean response of exposed minus unexposed competent receivers (association) |
//! | `effect_competent` | E2, **the headline**: overlap-weighted effect where the receptor is expressed |
//! | `effect_incompetent` | E3: the same where it is not; a negative control, ≈ 0 |
//! | `effect_contrast` | E1: E2 − E3; a contrast between cell types, not only between receptor states (below) |
//! | `ring_effect` | E5: effect of ligand on distant, non-touching cells, given the real exposure; a negative control, ≈ 0 |
//! | `effect_competent_{section,spatial,identity}` | E2 under nested adjustment sets (pooled row only) |
//! | `effect_competent_nospill` | E2 without the neighbour-response and bleed covariates (pooled row only) |
//!
//! Competence is read from identity bins, so competent and incompetent
//! receivers are different cell types by construction: E1 conflates the
//! receptor with everything else that differs between those types (their
//! baseline responsiveness, their neighbours, their exposure). It stays a
//! column, but the verdict rests on E2, with E3 as a control.
//!
//! The sequential adjustment refits E2 adjusting for (a) section only,
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
//! propensity serves every fit of a draw with the same treatment and
//! design ([`DrawFits`]).
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
//! Rubin's rules ([`super::pseudobulk::combine`]), so the spread between
//! the choices widens the interval instead of a setting picking one:
//!
//! - competence, the share of an identity bin that must detect the
//!   receptor, over [`COMPETENCE_GRID`]. Competence is a property of
//!   identity bins (cell types), not of single receivers, so E2 compares
//!   exposed with unexposed receivers among the cell types that carry the
//!   receptor; the grid moves the line between "carries" and "does not";
//! - the random pseudobulk partitions, as many as [`effects`] needs
//!   ([`super::pseudobulk::enough_draws`]);
//! - the programme's weighting ([`PairData::reweighted`]).
//!
//! **Threshold–response curve.** The headline τ is chosen per pair
//! ([`choose_threshold`]). Beside it, E2 (pooled) at τ = 0 and at the
//! quartiles of the positive exposures ([`curve_taus`]) with pointwise
//! intervals and a simultaneous 95% band ([`sup_t_crit`]): how the effect
//! grows with dose, and whether the headline rests on one threshold.
//!
//! **Decoy programme** ([`DecoyPool`]): the pair's programme with every
//! target swapped for a gene of matching mean expression outside every
//! programme and pair, same weights. E2 on the decoy should be 0; when
//! it is not, exposure moves generic expression (activation, depth,
//! residual confounding) and the pair's own E2 is no finding.
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

use super::output::Verdict;
use super::overlap::{
    crude_difference, difference, overlap_terms_with, propensity, subset_estimate, upper_normal_p,
    CrossFit, DesignView, Estimate, NuisanceOpts, OverlapTerms, PairInputs,
};
use super::pseudobulk::{self, UnitTally};
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

/// [`EffectOpts::draws`] for as many pseudobulk draws as the estimate
/// needs ([`super::pseudobulk::enough_draws`]).
pub const AUTO_DRAWS: usize = 0;

/// Quantiles of the positive exposures the curve visits beside τ = 0.
pub const CURVE_QUANTILES: [f64; 3] = [0.25, 0.5, 0.75];

/// Multiplier draws for the simultaneous band.
pub const BAND_DRAWS: usize = 2000;

/// Genes in the decoy pool at most ([`decoy_pool_genes`]).
pub const DECOY_POOL: usize = 2000;

#[derive(Clone, Copy, Debug)]
pub struct EffectOpts {
    /// Exposure threshold τ; `None` picks it per pair ([`choose_threshold`]).
    pub threshold: Option<f64>,
    /// Shares of an identity bin detecting the receptor for its receivers
    /// to count as competent, averaged over ([`COMPETENCE_GRID`]); the
    /// benchmark plants on one.
    pub competences: &'static [f64],
    /// Estimate the threshold–response curve.
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
    /// Receivers per random pseudobulk unit; 1 estimates on single cells.
    pub pseudobulk: usize,
    /// Random partitions averaged by Rubin's rules; [`AUTO_DRAWS`] runs as
    /// many as the estimate needs.
    pub draws: usize,
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
            pseudobulk: 1,
            draws: 1,
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

    /// Whether `n` analysis units, `treated` of them exposed, leave an arm
    /// under `min_arm` ([`thin_arm`]).
    pub fn thin_arm(&self, n: usize, treated: usize) -> bool {
        thin_arm(self.min_arm, n, treated)
    }
}

/// Whether `n` analysis units (receivers, or pseudobulk units), `treated`
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
    /// many cell types each side of E1 is made of.
    pub n_competent_bins: usize,
    pub n_incompetent_bins: usize,
    /// E2 on the matched decoy programme (pooled row only; NaN without a
    /// decoy), and the decoy's gene count.
    pub decoy_effect: Estimate,
    pub n_decoy_genes: usize,
    /// The threshold–response curve (pooled row only; empty elsewhere).
    pub curve: Vec<CurvePoint>,
    // From here on filled in by `report`, once every pair is in.
    /// BH q of the decoy effect, two-sided, within the section.
    pub decoy_q: f64,
    /// Two-sided p of the headline (E2) against the t reference with
    /// clusters − 1 degrees of freedom ([`Estimate::p`]), and its BH q
    /// within the section.
    pub effect_p: f64,
    pub effect_q: f64,
    /// One-sided (positive) normal p of the co-expression log odds ratio,
    /// and its BH q within the section.
    pub coexpr_p: f64,
    pub coexpr_q: f64,
    /// BH q of the controls (E3, E5), two-sided, within the section.
    pub effect_incompetent_q: f64,
    pub ring_effect_q: f64,
    /// Pairs in the section sharing this pair's ligand.
    pub n_pairs_same_ligand: usize,
    /// BH q over ligands of the Simes p of each ligand's pairs.
    pub ligand_q: f64,
    pub verdict: Verdict,
    /// Why the verdict is what it is beyond the q-values: failed controls,
    /// confounding, why a pair is not estimable. Semicolon separated.
    pub warnings: String,
}

/// One point of the threshold–response curve: E2 (pooled) at exposure
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
}

impl Exposure {
    /// The exposure side from the ligand counts the estimator sees, and
    /// each row's contact dose (ligand / size factor summed over its
    /// contacts), which the threshold is read from.
    pub fn build(tissue: &Tissue, ligand_counts: &[f64], opts: &EffectOpts) -> (Self, Vec<f64>) {
        let lnorm: Vec<f64> = ligand_counts
            .iter()
            .zip(&tissue.size_factor)
            .map(|(c, s)| c / s)
            .collect();
        let a = tissue.contacts.sum(&lnorm);
        let ring = tissue.ring_sum(&lnorm);
        let tau = choose_threshold(&a, opts);
        let x = Self {
            tau,
            t: exposed(&a, tau),
            t_ring: exposed(&ring, tau),
            own: lnorm.iter().map(|v| v.ln_1p()).collect(),
            niche: niche_ligand(tissue, &lnorm),
        };
        (x, a)
    }
}

/// The response side of a pair's per-row quantities: rebuilt for each
/// programme (each weighting, the decoy).
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
    use rayon::prelude::*;
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

/// Contact and ring exposure of each row from the ligand counts the
/// estimator sees, and the τ used. The ring holds as many partners as
/// there are contacts, so the same τ means the same dose.
pub fn exposures(
    tissue: &Tissue,
    ligand_counts: &[f64],
    opts: &EffectOpts,
) -> (Vec<bool>, Vec<bool>, f64) {
    let (x, _) = Exposure::build(tissue, ligand_counts, opts);
    (x.t, x.t_ring, x.tau)
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

/// The rows an effect is estimated on: the tissue's receivers, or one
/// random partition of them into pseudobulk units; with the rows inside
/// each scope, pooled first ([`Tissue::scopes`]).
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

    pub fn units(u: &'a super::pseudobulk::Units) -> Self {
        Self {
            design: &u.design,
            steps: &u.steps,
            cf: &u.cf,
            scope_rows: &u.scope_rows,
        }
    }

    /// The units of a draw when there is one, else the cells.
    pub fn of(units: &'a Option<super::pseudobulk::Units>, tissue: &'a Tissue) -> Self {
        units
            .as_ref()
            .map_or_else(|| Self::cells(tissue), Self::units)
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

/// The units (receivers, or pseudobulk units) of the pooled scope of `rows`
/// where `keep` holds, and how many of them `t` exposes: the counts the arm
/// guard reads of the subset ([`subset_estimate`]). Every other scope is a
/// subset of the pooled one, so neither of its arms is larger.
fn pooled_arms(rows: Rows<'_>, t: &[bool], keep: impl Fn(usize) -> bool) -> (usize, usize) {
    let (mut n, mut treated) = (0usize, 0usize);
    for &i in &rows.scope_rows[0] {
        let i = i as usize;
        if keep(i) {
            n += 1;
            treated += t[i] as usize;
        }
    }
    (n, treated)
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

/// Which of a draw's fits serve a subset with both arms at `min_arm`
/// ([`thin_arm`]) on the pooled rows: the main fit's E2 subsets (the
/// competent units at some competence threshold; also what the no-spill,
/// step, curve and decoy fits read), its E3 subsets (the incompetent ones)
/// and the ring fit's. A fit none of whose subsets is read is skipped,
/// propensity and all, and its estimands are what the guard would make of
/// them, not estimable with their counts ([`thin_subset`]): the units of an
/// arm are fixed by the strata whatever the seed ([`pseudobulk::partition`]),
/// so the skip is the guard read before the fit, not a second rule.
#[derive(Clone, Copy, Debug)]
pub struct Reads {
    pub e2: bool,
    pub e3: bool,
    pub ring: bool,
}

impl Reads {
    pub fn of(rows: Rows<'_>, x: &Exposure, levels: &[Vec<bool>], opts: &EffectOpts) -> Self {
        let open = |t: &[bool], keep: &dyn Fn(usize) -> bool| {
            let (n, treated) = pooled_arms(rows, t, keep);
            !opts.thin_arm(n, treated)
        };
        Self {
            e2: levels.iter().any(|c| open(&x.t, &|i| c[i])),
            e3: levels.iter().any(|c| open(&x.t, &|i| !c[i])),
            ring: levels.iter().any(|c| open(&x.t_ring, &|i| c[i])),
        }
    }

    /// The main fit serves some subset.
    pub fn main(&self) -> bool {
        self.e2 || self.e3
    }

    /// Some fit of the draw runs: else its estimates are the same under
    /// every seed ([`super::pseudobulk::stop_drawing`]).
    pub fn any(&self) -> bool {
        self.main() || self.ring
    }
}

/// The controls' propensities ([`DrawFits::with_controls`]); `None` for a
/// fit skipped as hopeless ([`Reads`]).
struct Controls {
    ring: Option<Vec<f64>>,
    /// Each adjustment step's, when asked for and E2 is read.
    steps: Option<Vec<Vec<f64>>>,
}

/// One draw's exposure side as its fits read it, built once and shared by
/// every fit on the draw: the competence indicators (raw, the outcome
/// model's effect modifiers), the pair columns both models carry
/// (standardized: own ligand, niche, every competence indicator, and the
/// contact exposure the ring fit adds to them), which fits serve a subset
/// ([`Reads`]), and the cross-fitted propensities ([`propensity`]) of the
/// main fit and, with [`DrawFits::with_controls`], of the ring fit and each
/// adjustment step. The propensity reads no response-side column
/// ([`fit_main`]), so one serves every programme weighting, the decoy and
/// the no-spill fit.
pub struct DrawFits<'a> {
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

fn indicator(v: &[bool]) -> Vec<f64> {
    v.iter().map(|&b| b as u8 as f64).collect()
}

impl<'a> DrawFits<'a> {
    /// The columns and the main fit's propensity on `rows`: all the curve
    /// and the decoy need.
    pub fn new(
        rows: Rows<'_>,
        exposure: &'a Exposure,
        levels: &'a [Vec<bool>],
        opts: &EffectOpts,
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
        if fits.runs(fits.reads.main()) {
            let extra = fits.extra(Fit::Main);
            let main = propensity(
                rows.design.view(),
                rows.cf,
                &exposure.t,
                &extra,
                &opts.nuisance,
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
                )
            })?)
        } else {
            None
        };
        let steps = if steps && self.runs(self.reads.e2) {
            let extra = self.extra(Fit::Main);
            Some(timing::time(timing::STEPS, || {
                rows.steps
                    .iter()
                    .map(|s| {
                        let view = s.design.view(rows.design);
                        propensity(view, rows.cf, &x.t, &extra, &opts.nuisance)
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
    /// [`DrawFits::with_controls`], or a fit skipped as hopeless.
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
/// at once, with its propensity from `fits` ([`DrawFits`]): the main one,
/// or adjustment step `step`'s, or the ring's; an error when that one is
/// not fitted.
///
/// The thresholds give nested indicators `C_c`. One outcome model carries
/// every `C_c` and every `T · C_c`, and one propensity model every `C_c`;
/// E2 at threshold `c` is then the subset estimate over the receivers
/// competent at `c`. Each single-threshold model is nested in this one, so
/// each `m_t` stays correctly specified wherever its own was; what changes
/// is one fit per draw instead of one per threshold.
///
/// The pair columns: own ligand, niche, every competence indicator, and
/// under [`Fit::Ring`] the contact exposure, the ring exposure being the
/// treatment, in both models; spillover and bleed (not under
/// [`Fit::NoSpill`]) in the outcome model alone. Read off the response,
/// they would tie the propensity to one programme; without them it depends
/// on the exposure side only and is fitted once per draw for every
/// programme weighting, the decoy and the no-spill fit.
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
    fits: &DrawFits<'_>,
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
    /// E1 = E2 − E3; not the headline (see the module doc).
    pub effect: Estimate,
    /// E2, the headline.
    pub competent: Estimate,
    pub incompetent: Estimate,
    pub ring: Estimate,
    /// E2 under the section, spatial and identity steps (pooled only; NaN
    /// elsewhere).
    pub steps: [Estimate; 3],
    /// E2 without the neighbour-response and bleed covariates (pooled
    /// only; NaN elsewhere).
    pub nospill: Estimate,
    /// Share of the scope's competent rows (units, on pseudobulk) whose
    /// overlap weight is below [`NEAR_DETERMINED_H`] in the main fit; NaN
    /// without competent rows or without the main fit, which is skipped
    /// when every E2 and E3 subset leaves an arm thin ([`Reads`]), on every
    /// path alike.
    pub near_determined: f64,
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
            steps: [
                pick(&|e| e.steps[0]),
                pick(&|e| e.steps[1]),
                pick(&|e| e.steps[2]),
            ],
            nospill: pick(&|e| e.nospill),
            near_determined: if near.is_empty() {
                f64::NAN
            } else {
                near.iter().sum::<f64>() / near.len() as f64
            },
        }
    }
}

/// One competence threshold's estimands per scope, and the cluster sums of
/// its pooled E2's linearization ψ (zeros when not estimable).
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
/// NaN, which Rubin's rules then leave out of the draws.
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

    /// The sensitivity fits run on the first [`MIN_DRAWS`](super::pseudobulk::MIN_DRAWS)
    /// draws only: they feed diagnostic columns, not the headline, and
    /// Rubin's rules over those draws are still Rubin's rules.
    pub fn on_draw(d: usize) -> Self {
        if d < super::pseudobulk::MIN_DRAWS {
            Self::ALL
        } else {
            Self::NONE
        }
    }
}

/// The estimands on `rows` for each competence threshold
/// ([`DrawFits::levels`]) and each scope of `rows`, from one fit per
/// adjustment set ([`fit_main`]) of the response `r`; `fits` carries the
/// controls' propensities ([`DrawFits::with_controls`]). The sensitivity
/// fits `sens` asks for serve the pooled scope (the first). A fit none of
/// whose subsets has both arms at `min_arm` is skipped ([`Reads`]); without
/// the main fit, the near-determined shares are NaN and `h` is empty.
pub fn effects_on(
    rows: Rows<'_>,
    fits: &DrawFits<'_>,
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
            fit(fits.runs(reads.main()), Fit::Main, None, design)?,
            fit(fits.runs(reads.ring), Fit::Ring, None, design)?,
        ))
    })?;
    let on_e2 = fits.runs(reads.e2);
    let nospill = time(NOSPILL, || {
        fit(sens.nospill && on_e2, Fit::NoSpill, None, design)
    })?;
    let steps = if sens.steps && on_e2 {
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
    // A subset's estimate from `terms`, or the guard's verdict on it when
    // the fit was skipped.
    let est = |terms: Option<&OverlapTerms>, t: &[bool], sub: &[u32]| match terms {
        Some(terms) => opts.guard(subset_estimate(terms, t, sub, cf)),
        None => thin_subset(t, sub, cf),
    };
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
                    let (e2, s2) = est(main.as_ref(), t, &comp);
                    let (e3, s3) = est(main.as_ref(), t, &incomp);
                    let e1 = difference(&e2, &s2, &e3, &s3);
                    let (e5, _) = est(ring.as_ref(), t_ring, &comp);
                    let nan = Estimate::nan(e2.n, e2.n_treated);
                    let mut step_est = [nan; 3];
                    let mut e2_nospill = nan;
                    let near_determined = match &main {
                        Some(m) if reads.main() && !comp.is_empty() => {
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
                            e2_nospill = est(nospill.as_ref(), t, &comp).0;
                        }
                        if sens.steps {
                            for (j, e) in step_est.iter_mut().enumerate() {
                                *e = est(steps.get(j), t, &comp).0;
                            }
                        }
                        pooled_sums = s2;
                    }
                    ScopeEffects {
                        effect: e1,
                        competent: e2,
                        incompetent: e3,
                        ring: e5,
                        steps: step_est,
                        nospill: e2_nospill,
                        near_determined,
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
        Some(m) if reads.main() => m.h,
        _ => Vec::new(),
    };
    Ok(Effects { levels, h })
}

/// Pooled E2 of the response `r` at competence threshold `k` with the
/// cluster sums of its ψ, from one main fit carrying every threshold
/// ([`fit_main`]), and whether the fit ran: it is skipped when the subset
/// leaves an arm thin ([`Reads`]). For the curve and the decoy.
pub fn competent_effect(
    rows: Rows<'_>,
    fits: &DrawFits<'_>,
    r: &Response,
    k: usize,
    opts: &EffectOpts,
) -> anyhow::Result<(Estimate, Vec<f64>, bool)> {
    let (design, cf) = (rows.design.view(), rows.cf);
    let t = &fits.exposure.t;
    let comp: Vec<u32> = rows.scope_rows[0]
        .iter()
        .copied()
        .filter(|&i| fits.levels[k][i as usize])
        .collect();
    let treated = comp.iter().filter(|&&i| t[i as usize]).count();
    let read = !opts.thin_arm(comp.len(), treated);
    if !fits.runs(read) {
        let (e, s) = thin_subset(t, &comp, cf);
        return Ok((e, s, false));
    }
    let terms = fit_main(design, cf, fits, r, Fit::Main, None, opts)?;
    let (e, s) = opts.guard(subset_estimate(&terms, t, &comp, cf));
    Ok((e, s, read))
}

/// Each analysis (draw or choice) combined into one by Rubin's rules,
/// estimand by estimand, scope by scope.
pub fn combine_scopes(per: &[Vec<ScopeEffects>]) -> Vec<ScopeEffects> {
    let Some(first) = per.first() else {
        return Vec::new();
    };
    (0..first.len())
        .map(|s| {
            let parts: Vec<ScopeEffects> = per.iter().map(|d| d[s]).collect();
            ScopeEffects::zip_with(&parts, super::pseudobulk::combine)
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
/// (one row per cluster and analysis, analyses being the draws and choices
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

// * Decoy programme

/// Genes for the decoy pool: up to [`DECOY_POOL`] of the `n_genes` genes
/// not in `exclude` (every ligand, receptor and programme gene), and at
/// most a fifth of them so a small panel keeps most of its genes for the
/// identity; drawn at random with `seed`, sorted.
pub fn decoy_pool_genes(n_genes: usize, exclude: &[usize], seed: u64) -> Vec<usize> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let ex: std::collections::HashSet<usize> = exclude.iter().copied().collect();
    let mut cand: Vec<usize> = (0..n_genes).filter(|g| !ex.contains(g)).collect();
    let k = DECOY_POOL.min(cand.len() / 5);
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
/// each pair's decoy programme is matched.
#[derive(Default)]
pub struct DecoyPool<'a> {
    /// `(log mean expression, non-zeros)`, sorted by the first.
    genes: Vec<(f64, &'a [(u32, f32)])>,
}

impl<'a> DecoyPool<'a> {
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

    /// The decoy of a programme: target by target in order of decreasing
    /// `|w|`, the unused pool gene nearest in log mean expression (ties to
    /// the lower), with the target's weight. Fewer genes than targets when
    /// the pool runs out; empty when it holds fewer than half the
    /// programme.
    pub fn matched(
        &self,
        sf: &[f64],
        targets: &[(f64, &[(u32, f32)])],
    ) -> Vec<(f64, &'a [(u32, f32)])> {
        let mut order: Vec<usize> = (0..targets.len()).collect();
        order.sort_by(|&a, &b| targets[b].0.abs().total_cmp(&targets[a].0.abs()));
        let mut used = vec![false; self.genes.len()];
        let mut out = Vec::with_capacity(targets.len());
        for k in order {
            let (w, nz) = targets[k];
            let key = mean_expression(nz, sf).max(1e-12).ln();
            let at = self.genes.partition_point(|g| g.0 < key);
            // Nearest unused on each side.
            let mut lo = at;
            while lo > 0 && used[lo - 1] {
                lo -= 1;
            }
            let mut hi = at;
            while hi < used.len() && used[hi] {
                hi += 1;
            }
            let left = (lo > 0).then(|| lo - 1);
            let right = (hi < used.len()).then_some(hi);
            let pick = match (left, right) {
                (Some(l), Some(r)) if key - self.genes[l].0 <= self.genes[r].0 - key => Some(l),
                (Some(_), Some(r)) => Some(r),
                (l, r) => l.or(r),
            };
            let Some(j) = pick else { break };
            used[j] = true;
            out.push((w, self.genes[j].1));
        }
        if 2 * out.len() < targets.len() || out.is_empty() {
            return Vec::new();
        }
        out
    }
}

// * One pair

/// Everything the analyses of one programme weighting give, combined.
struct Analysis {
    scopes: Vec<ScopeEffects>,
    curve: Vec<CurvePoint>,
    decoy: Estimate,
}

impl Analysis {
    /// Two weightings of the same pair (same scopes) averaged the way
    /// pseudobulk draws are: each estimand by Rubin's rules, so the spread
    /// between them widens its interval. The near-determined shares and
    /// the curve are the first's; a curve is read under the first
    /// weighting only.
    fn average(self, b: Analysis) -> Analysis {
        let mix = |x: Estimate, y: Estimate| super::pseudobulk::combine(&[x, y]);
        let scopes = self
            .scopes
            .iter()
            .zip(&b.scopes)
            .map(|(x, y)| ScopeEffects {
                near_determined: x.near_determined,
                ..ScopeEffects::zip_with(&[*x, *y], super::pseudobulk::combine)
            })
            .collect();
        Analysis {
            scopes,
            curve: self.curve,
            decoy: mix(self.decoy, b.decoy),
        }
    }
}

/// The mean θ and root mean variance of the estimable `parts`, the rest
/// NaN; NaN when none is estimable. What the stopping rule reads of a
/// draw's competence thresholds.
fn mean_estimate(parts: &[Estimate]) -> Estimate {
    let ok: Vec<&Estimate> = parts.iter().filter(|e| pseudobulk::usable(e)).collect();
    if ok.is_empty() {
        return Estimate::nan(0, 0);
    }
    let k = ok.len() as f64;
    Estimate {
        theta: ok.iter().map(|e| e.theta).sum::<f64>() / k,
        se: (ok.iter().map(|e| e.se * e.se).sum::<f64>() / k).sqrt(),
        df: ok.iter().map(|e| e.df).fold(f64::INFINITY, f64::min),
        ..Estimate::nan(0, 0)
    }
}

/// One weighting of a pair's programme: its targets and its response side
/// on the receivers.
struct Weighting<'t> {
    targets: &'t [(f64, &'t [(u32, f32)])],
    response: Response,
}

/// The decoy's programme and its response side on the receivers.
struct Decoy<'d> {
    targets: Vec<(f64, &'d [(u32, f32)])>,
    response: Response,
}

/// Every analysis of one pair, per programme weighting: for each
/// pseudobulk draw (or the cells once), the estimands per competence
/// threshold and scope from one fit per adjustment set ([`effects_on`]);
/// under the first weighting also E2 at each curve threshold and on the
/// decoy at the middle competence threshold. Draws are combined first,
/// then competence thresholds, both by Rubin's rules. The weightings share
/// each draw's partition, design and exposure side; only the response side
/// is rebuilt per weighting.
///
/// Each weighting's draws stop by [`super::pseudobulk::stop_drawing`] on its
/// pooled E2 averaged over the thresholds and on whether the draw could
/// change under another seed (its partition is random and some fit ran,
/// [`Reads::any`]); `lra-simulate` stops its headline by the same rule.
/// Curve thresholds other than the headline's get their own partitions
/// (units never mix arms), drawn with the same seeds, and stop by the same
/// rule on their own estimates and partitions, so a random curve partition
/// neither adds draws to a fixed headline nor goes without them. With the
/// tally of every partition drawn.
#[allow(clippy::too_many_arguments)]
fn effects(
    tissue: &Tissue,
    ex: &mut Exposure,
    weightings: &[Weighting<'_>],
    decoy: Option<&Decoy<'_>>,
    ligand_counts: &[f64],
    comps: &[Vec<bool>],
    a: &[f64],
    taus: &[(f64, f64)],
    opts: &EffectOpts,
) -> anyhow::Result<(Vec<Analysis>, UnitTally)> {
    use super::pseudobulk::{combine, draw, draw_limit, stop_drawing};
    use legume_numeric::matrix::rand_util::mix_seed;
    let nc = comps.len();
    let mid = nc / 2;
    let nk = taus.len();
    let nw = weightings.len();
    let head_k = taus.iter().position(|&(t, _)| t == ex.tau);
    let units_on = opts.pseudobulk > 1;
    let max_draws = draw_limit(opts);
    let random = |u: &Option<super::pseudobulk::Units>| u.as_ref().is_some_and(|u| u.random);
    let mut tally = UnitTally::default();
    // Whether any fit ran, for the run log.
    let mut fitted = false;
    // Per weighting: per competence threshold, the scope effects of each
    // draw; and the headline of each draw.
    let mut scope_draws: Vec<Vec<Vec<Vec<ScopeEffects>>>> = vec![vec![Vec::new(); nc]; nw];
    let mut headline: Vec<Vec<Estimate>> = vec![Vec::new(); nw];
    let mut done = vec![false; nw];
    let mut decoy_draws: Vec<Estimate> = Vec::new();
    let mut curve_draws: Vec<Vec<Estimate>> = vec![Vec::new(); nk];
    // The headline's curve point is read off the first weighting's draws.
    let mut curve_done: Vec<bool> = (0..nk).map(|k| Some(k) == head_k).collect();
    let mut stacked: Vec<Vec<f64>> = Vec::new();
    for d in 0..max_draws {
        if done.iter().chain(&curve_done).all(|&x| x) {
            break;
        }
        let seed = mix_seed(opts.seed, d as u64);
        // Cluster sums of ψ per curve threshold, this draw.
        let mut sums: Vec<Vec<f64>> = vec![Vec::new(); nk];
        if !done.iter().all(|&x| x) {
            let units = units_on.then(|| {
                timing::time(timing::PARTITION, || {
                    draw(tissue, ex, ligand_counts, opts.pseudobulk, seed)
                })
            });
            tally.add(units.as_ref());
            let rows = Rows::of(&units, tissue);
            let unit_levels: Vec<Vec<bool>>;
            let (exposure, levels): (&Exposure, &[Vec<bool>]) = match &units {
                Some(u) => {
                    unit_levels = comps.iter().map(|c| u.of_lead(c)).collect();
                    (&u.exposure, &unit_levels)
                }
                None => (&*ex, comps),
            };
            let sens = Sensitivity::on_draw(d);
            let fits = timing::time(timing::MAIN, || DrawFits::new(rows, exposure, levels, opts))?
                .with_controls(rows, sens.steps, opts)?;
            fitted |= fits.reads.any();
            let varies = random(&units) && fits.reads.any();
            // The response side of a programme on this draw's rows.
            let on_rows = |r: &Response, targets: &[(f64, &[(u32, f32)])]| {
                units.as_ref().map(|u| u.response_of(r, targets))
            };
            for (w, wt) in weightings.iter().enumerate() {
                if done[w] {
                    continue;
                }
                let own = on_rows(&wt.response, wt.targets);
                let r = own.as_ref().unwrap_or(&wt.response);
                let per_c = effects_on(rows, &fits, r, sens, opts)?;
                if w == 0 {
                    if let Some(dc) = decoy {
                        let own = on_rows(&dc.response, &dc.targets);
                        let r = own.as_ref().unwrap_or(&dc.response);
                        let (e, _, _) = timing::time(timing::DECOY, || {
                            competent_effect(rows, &fits, r, mid, opts)
                        })?;
                        decoy_draws.push(e);
                    }
                    if let Some(k) = head_k {
                        curve_draws[k].push(per_c.levels[mid].scopes[0].competent);
                        sums[k] = per_c.levels[mid].pooled_sums.clone();
                    }
                }
                let pooled: Vec<Estimate> =
                    per_c.levels.iter().map(|l| l.scopes[0].competent).collect();
                headline[w].push(mean_estimate(&pooled));
                for (ci, level) in per_c.levels.into_iter().enumerate() {
                    scope_draws[w][ci].push(level.scopes);
                }
                if stop_drawing(opts, &headline[w], varies) {
                    done[w] = true;
                }
            }
        }
        let w0 = &weightings[0];
        for (k, &(tau, _)) in taus.iter().enumerate() {
            if curve_done[k] {
                continue;
            }
            let curve_timer = std::time::Instant::now();
            let t0 = std::mem::replace(&mut ex.t, exposed(a, tau));
            let units_k = units_on.then(|| draw(tissue, ex, ligand_counts, opts.pseudobulk, seed));
            tally.add(units_k.as_ref());
            let (e, s, ran) = match &units_k {
                Some(u) => {
                    let rows = Rows::units(u);
                    let lv: Vec<Vec<bool>> = comps.iter().map(|c| u.of_lead(c)).collect();
                    let r = u.response_of(&w0.response, w0.targets);
                    let fits = DrawFits::new(rows, &u.exposure, &lv, opts)?;
                    competent_effect(rows, &fits, &r, mid, opts)?
                }
                None => {
                    let rows = Rows::cells(tissue);
                    let fits = DrawFits::new(rows, ex, comps, opts)?;
                    competent_effect(rows, &fits, &w0.response, mid, opts)?
                }
            };
            fitted |= ran;
            curve_draws[k].push(e);
            sums[k] = s;
            ex.t = t0;
            if stop_drawing(opts, &curve_draws[k], random(&units_k) && ran) {
                curve_done[k] = true;
            }
            timing::add(timing::CURVE, curve_timer);
        }
        for c in 0..tissue.cf.n_clusters() {
            let row: Vec<f64> = sums
                .iter()
                .map(|s| s.get(c).copied().unwrap_or(0.0))
                .collect();
            if row.iter().any(|&x| x != 0.0) {
                stacked.push(row);
            }
        }
    }
    if !fitted {
        timing::all_skipped();
    }
    let analyses = (0..nw)
        .map(|w| {
            timing::draws(headline[w].len());
            let per_c: Vec<Vec<ScopeEffects>> =
                scope_draws[w].iter().map(|d| combine_scopes(d)).collect();
            let (curve, decoy) = if w == 0 {
                let curve_est: Vec<Estimate> = curve_draws.iter().map(|d| combine(d)).collect();
                let curve = if nk == 0 {
                    Vec::new()
                } else {
                    timing::time(timing::BAND, || {
                        build_curve(taus, &curve_est, &stacked, mix_seed(opts.seed, 0xc0_ffee))
                    })
                };
                (curve, combine(&decoy_draws))
            } else {
                (Vec::new(), combine(&[]))
            };
            Analysis {
                scopes: combine_scopes(&per_c),
                curve,
                decoy,
            }
        })
        .collect();
    Ok((analyses, tally))
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
/// type, with a decoy programme matched from `pool`. Verdicts and q-values
/// are left empty for [`report`], which needs every pair.
///
/// With [`PairData::reweighted`], the two weightings of the programme are
/// averaged ([`Analysis::average`]); the curve and the decoy describe the
/// pair, not the weighting, and are read under `targets`. Counts,
/// co-expression, the crude contrast and the diagnostics read competence
/// at the middle threshold of the grid. With the tally of the pseudobulk
/// partitions drawn, for the run's mean unit size.
pub fn estimate_pair_with(
    tissue: &Tissue,
    pair: &PairData<'_>,
    pool: &DecoyPool<'_>,
    opts: &EffectOpts,
) -> anyhow::Result<(Vec<EffectRow>, UnitTally)> {
    let n = tissue.n();
    let (mut ex, a) = timing::time(timing::FRAME, || {
        Exposure::build(tissue, pair.ligand_counts, opts)
    });
    let detected: Vec<bool> = pair.receptor_counts.iter().map(|&r| r > 0.0).collect();
    let comps = super::exposure::competence_grid(&tissue.identity_bin, &detected, opts.competences);
    let taus = if opts.curve {
        let eligible: Vec<bool> = (0..n).map(|i| comps.iter().any(|c| c[i])).collect();
        curve_taus(&a, ex.tau, &eligible, opts.min_arm)
    } else {
        Vec::new()
    };
    let programmes: Vec<&Targets<'_>> = std::iter::once(pair.targets)
        .chain(pair.reweighted)
        .collect();
    let weightings = timing::time(timing::FRAME, || {
        let responses = match pair.reweighted.and_then(|b| aligned(pair.targets, b)) {
            Some(wb) => {
                let (genes, wa) = genes_and_weights(pair.targets);
                Response::build_many(tissue, &genes, &[wa, wb])
            }
            None => programmes
                .iter()
                .map(|p| Response::build(tissue, p))
                .collect(),
        };
        programmes
            .iter()
            .zip(responses)
            .map(|(&targets, response)| Weighting { targets, response })
            .collect::<Vec<_>>()
    });
    let decoy_targets = pool.matched(&tissue.size_factor, pair.targets);
    let n_decoy_genes = decoy_targets.len();
    let decoy = (!decoy_targets.is_empty()).then(|| Decoy {
        response: Response::build(tissue, &decoy_targets),
        targets: decoy_targets,
    });
    let (analyses, tally) = effects(
        tissue,
        &mut ex,
        &weightings,
        decoy.as_ref(),
        pair.ligand_counts,
        &comps,
        &a,
        &taus,
        opts,
    )?;
    let mut analyses = analyses.into_iter();
    let first = analyses.next().expect("one analysis per weighting");
    let eff = match analyses.next() {
        Some(b) => first.average(b),
        None => first,
    };
    // The response of each weighting, for the crude contrast.
    let ys: Vec<Vec<f64>> = weightings.into_iter().map(|w| w.response.y).collect();
    let competent = &comps[comps.len() / 2];
    let tf = indicator(&ex.t);
    let mut rows = Vec::with_capacity(tissue.scopes.len());
    let mut curve = Some(eff.curve);
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
            coexpr_log_or: lor,
            coexpr_se: lse,
            crude,
            effects: e,
            exposure_eta2: eta_squared(&tissue.neighbour_bin, &tf, inside_rows),
            n_competent_bins: bins(true),
            n_incompetent_bins: bins(false),
            decoy_effect: if pooled {
                eff.decoy
            } else {
                Estimate::nan(0, 0)
            },
            n_decoy_genes: if pooled { n_decoy_genes } else { 0 },
            curve: if pooled {
                curve.take().unwrap_or_default()
            } else {
                Vec::new()
            },
            decoy_q: f64::NAN,
            effect_p: f64::NAN,
            effect_q: f64::NAN,
            coexpr_p: f64::NAN,
            coexpr_q: f64::NAN,
            effect_incompetent_q: f64::NAN,
            ring_effect_q: f64::NAN,
            n_pairs_same_ligand: 0,
            ligand_q: f64::NAN,
            verdict: Verdict::InsufficientData,
            warnings: String::new(),
        });
    }
    Ok((rows, tally))
}

/// E0 over the contacts inside the scope, both orientations: the Jeffreys
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

    pub const STAGES: [&str; 10] = [
        "frame",
        "partition",
        "main+ring fits",
        "no-spill fit",
        "step fits",
        "curve fits",
        "decoy fit",
        "band",
        "(in fits: outcome ridge",
        "propensity)",
    ];
    pub const RIDGE: usize = 8;
    pub const LOGISTIC: usize = 9;
    pub const FRAME: usize = 0;
    pub const PARTITION: usize = 1;
    pub const MAIN: usize = 2;
    pub const NOSPILL: usize = 3;
    pub const STEPS: usize = 4;
    pub const CURVE: usize = 5;
    pub const DECOY: usize = 6;
    pub const BAND: usize = 7;

    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    static NS: [AtomicU64; 10] = [ZERO; 10];
    static DRAWS: AtomicU64 = AtomicU64::new(0);
    static ANALYSES: AtomicU64 = AtomicU64::new(0);
    static ALL_SKIPPED: AtomicU64 = AtomicU64::new(0);

    pub fn time<T>(stage: usize, f: impl FnOnce() -> T) -> T {
        let t = std::time::Instant::now();
        let out = f();
        NS[stage].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    }

    pub fn add(stage: usize, since: std::time::Instant) {
        NS[stage].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// One analysis (pair × weighting) that used `draws` draws.
    pub fn draws(draws: usize) {
        DRAWS.fetch_add(draws as u64, Ordering::Relaxed);
        ANALYSES.fetch_add(1, Ordering::Relaxed);
    }

    /// One pair none of whose fits ran: every subset left an arm thin.
    pub fn all_skipped() {
        ALL_SKIPPED.fetch_add(1, Ordering::Relaxed);
    }

    /// Thread-seconds per stage, the mean draws per analysis and the pairs
    /// with every fit skipped.
    pub fn summary() -> String {
        let stages: Vec<String> = STAGES
            .iter()
            .zip(&NS)
            .map(|(s, n)| format!("{s} {:.0}s", n.load(Ordering::Relaxed) as f64 * 1e-9))
            .collect();
        let a = ANALYSES.load(Ordering::Relaxed).max(1) as f64;
        format!(
            "{}; {:.1} pseudobulk draws per analysis; {} pairs with every fit skipped (thin arms)",
            stages.join(", "),
            DRAWS.load(Ordering::Relaxed) as f64 / a,
            ALL_SKIPPED.load(Ordering::Relaxed)
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
    /// The estimator's points are pseudobulk units ([`EffectOpts::pseudobulk`]
    /// above 1), not receivers: what the thin-arm note counts.
    pub units: bool,
}

impl Default for ReportOpts {
    fn default() -> Self {
        Self {
            fdr: 0.05,
            max_exposure_eta2: 0.8,
            min_arm: 20,
            units: false,
        }
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
    /// The headline (E2) has an estimate.
    pub estimable: bool,
    /// The headline's q is below the FDR level.
    pub effect: bool,
    /// The co-expression q is below the FDR level.
    pub coexpr: bool,
    /// E3, the ring or the decoy has a q below the FDR level.
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
        let q3 = bh(&col(&|r| r.effects.incompetent.p()));
        let q5 = bh(&col(&|r| r.effects.ring.p()));
        let qd = bh(&col(&|r| r.decoy_effect.p()));
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
            r.effect_incompetent_q = q3[k];
            r.ring_effect_q = q5[k];
            r.decoy_q = qd[k];
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
    let failed_e3 = sig(row.effect_incompetent_q);
    let failed_ring = sig(row.ring_effect_q);
    let failed_decoy = sig(row.decoy_q);
    let ev = Evidence {
        estimable: head.theta.is_finite() && head.se.is_finite(),
        effect: sig(row.effect_q),
        coexpr: sig(row.coexpr_q),
        control_failed: failed_e3 || failed_ring || failed_decoy,
        confounded: row.exposure_eta2 >= opts.max_exposure_eta2,
    };
    row.verdict = verdict(ev);
    let mut w: Vec<String> = Vec::new();
    if !ev.estimable {
        w.push(if row.n_competent == 0 {
            "not estimable: no receptor-competent receivers".into()
        } else if thin_arm(opts.min_arm, head.n, head.n_treated) {
            let (points, guards) = if opts.units {
                ("pseudobulk units", "pseudobulk and min_arm")
            } else {
                ("receivers", "min_arm")
            };
            format!(
                "not estimable: thin arm ({} exposed, {} unexposed competent {points}; \
                 internal guards {guards} {}, see outputs.lra.settings in the manifest)",
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
    if failed_e3 {
        w.push("receptor-negative receivers respond".into());
    }
    if failed_ring {
        w.push("distant senders have an effect".into());
    }
    if failed_decoy {
        w.push("a matched decoy programme responds too".into());
    }
    row.warnings = w.join("; ");
}
