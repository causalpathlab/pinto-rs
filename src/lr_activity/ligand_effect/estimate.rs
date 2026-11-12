//! One ligand-receptor pair: exposure, response, and every estimand.
//!
//! | Column | Estimand |
//! |---|---|
//! | `coexpr_log_or` | log odds of ligand on one side and receptor on the other, across contacts (association) |
//! | `marginal_cor` | Pearson correlation of the response and `log1p` contact ligand over competent receivers, on its own scale (association) |
//! | `unadjusted_diff` | the unadjusted difference: mean response of the exposed minus mean response of the unexposed, over competent receivers, every receiver weighted 1 and nothing adjusted for (association) |
//! | `effect_ipw` | the competent effect by overlap weights alone, no outcome model |
//! | `effect_outcome` | the competent effect by the outcome regression alone, averaged under the overlap weights, no residual correction; a point estimate, with no SE or interval |
//! | `effect_competent` | the competent effect, **the headline**: both combined, doubly robust, overlap-weighted effect where the receptor is expressed, averaged over exposure thresholds |
//! | `effect_log10_bf`, `tau_top`, `tau_top_weight` | the evidence for an effect at any threshold against none, and the threshold the average leans on most |
//! | `effect_incompetent` | the incompetent effect: the same where it is not; a negative control, ≈ 0 |
//! | `effect_contrast` | the contrast: competent effect − incompetent effect; a contrast between cell types, not only between receptor states (below) |
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
//! `unadjusted_diff`, `effect_ipw` and `effect_outcome` are averaged over thresholds as
//! the headline is, each by its own evidence, and read for comparison: they
//! enter no verdict. The sensitivity fits below (the nested adjustment sets
//! and the no-spill fit) read the headline threshold alone, so a gap between
//! them and `effect_competent` can come from the threshold as well as the
//! adjustment.
//!
//! The sequential adjustment refits the competent effect adjusting for (a) section only,
//! (b) + spatial smooth, (c) + cell identity (receiver and neighbour
//! identity, cell type when given); `effect_competent` is the full set,
//! adding depth. Read left to right with `unadjusted_diff`, it shows how much of an
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
//!   programme counts on the receiver's own scale
//!   ([`super::response::programme_columns`]): bleed on the response side, what
//!   a neighbour's transcripts add to the receiver's measurement. Bleed is a
//!   sum over contacts divided by the receiver's size factor, so it grows with
//!   the contact count, which also drives exposure, and shrinks with the
//!   receiver's depth; a neighbour mean sees neither. It is built from the
//!   neighbours' identity bins and depths rather than their observed counts,
//!   which carry their own response to shared senders;
//! - the mean ligand of second-order neighbours (contacts of contacts), a
//!   proxy of a niche that drives both ligand and response at a few contact
//!   lengths, finer than the spatial smooth can follow. Under contact-only
//!   signalling it is no exposure itself;
//! - competence.
//!
//! The two response-side columns enter both models; the propensity of each
//! response borders the one exposure-side fit shared at a threshold
//! ([`super::fits::ThresholdFits`]) rather than refitting it.
//!
//! Neighbours share senders with the receiver, so their response is partly
//! downstream of the same exposure and adjusting for it can attenuate a
//! real effect; `effect_competent_nospill` drops both response-side bleed
//! columns as a sensitivity check. One fit serves the pooled row and one
//! row per section: a section's estimate is the same estimator summed over
//! that section's receivers.
//!
//! Inference across pairs ([`super::report::report`]) runs once every pair is in: see
//! there for the q-values, the identifiability diagnostics and the
//! verdict labels.
//!
//! **Choices averaged, not asked for.** Where the analysis has a choice
//! with no right answer it makes each and averages the estimates by
//! Rubin's rules ([`super::averaging::combine`]), so the spread between
//! the choices widens the interval instead of a setting picking one:
//!
//! - competence, the share of an identity bin that must detect the receptor,
//!   over [`super::options::COMPETENCE_GRID`]. Competence is a property of
//!   identity bins (cell types), not of single receivers, so the competent
//!   effect compares exposed with unexposed receivers among the cell types that
//!   carry the receptor; the grid moves the line between "carries" and "does
//!   not";
//! - the programme's weighting ([`PairData::reweighted`]).
//!
//! **The exposure threshold** is averaged too, differently. Each threshold is a
//! model of how exposure acts, with its own fits, and the estimates are averaged
//! by their posterior weights under an approximate Bayes factor
//! ([`super::averaging::abf_average`]), not by Rubin's rules: thresholds are not
//! exchangeable draws, and which one the data favour is itself of interest. Each
//! estimand has its own weights. The weights read the estimates they average, so
//! the average leans toward large z-scores; the verdict's comparison with the
//! negative control, averaged the same way, is what guards against that.
//!
//! **Threshold–response curve.** The headline τ is chosen per pair
//! ([`super::dose::choose_threshold`]). Beside it, the competent effect
//! (pooled) at τ = 0 and at the quartiles of the positive exposures
//! ([`super::curve::curve_taus`]) with pointwise intervals and a simultaneous
//! 95% band ([`super::curve::sup_t_crit`]): how the effect grows with dose, and
//! whether the headline rests on one threshold.
//!
//! **Reading an effect** ([`super::rows::Moments`]): θ is a difference in the signed
//! mean `log1p` programme, read as a mean with SD (= SE) and 95% interval,
//! and as a % change `100 (e^θ − 1)` with the delta-method SD
//! `100 e^θ SE` and the interval's endpoints mapped the same way.
//!
//! **By receiver cell type**: with named cell types
//! ([`super::tissue::Tissue::with_cell_type_names`]) each pair also gets
//! one row per type, `section` = `type:<name>`: the estimands among that
//! type's receivers, from the pooled fit.

use super::association::marginal_correlation;
use super::averaging::combine;
use super::baselines::coexpression;
use super::curve::threshold_curve;
use super::dose::{eta_squared, Exposure};
use super::effects::{ScopeEffects, Sensitivity};
use super::fits::{indicator, Rows};
use super::options::EffectOpts;
use super::response::{genes_and_weights, Response, Targets};
use super::rows::{CurvePoint, EffectRow};
use super::thresholds::{aligned, over_thresholds, thresholds};
use super::timing;
use super::tissue::{Scope, Tissue};
use legume_numeric::matrix::rand_util::mix_seed;

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

/// Rows for the pooled tissue, each section and each named receiver cell
/// type. Verdicts and q-values
/// are left empty for [`super::report::report`], which needs every pair.
///
/// With [`PairData::reweighted`], the two weightings of the programme are
/// combined ([`super::thresholds::over_thresholds`]); the curve
/// describes the pair, not the weighting, and is read under `targets`.
/// Counts, co-expression, the marginal correlation and the diagnostics read
/// competence at the middle threshold of the grid.
pub fn estimate_pair(
    tissue: &Tissue,
    pair: &PairData<'_>,
    opts: &EffectOpts,
) -> anyhow::Result<Vec<EffectRow>> {
    let ex = timing::time(timing::FRAME, || {
        Exposure::build(tissue, pair.ligand_counts, opts)
    });
    let detected: Vec<bool> = pair.receptor_counts.iter().map(|&r| r > 0.0).collect();
    let comps = super::contacts::competence_grid(&tissue.identity_bin, &detected, opts.competences);
    let taus = thresholds(&ex, &comps, opts);
    let responses = timing::time(timing::FRAME, || pair_responses(tissue, pair));
    let eff = over_thresholds(
        Rows::cells(tissue),
        &ex,
        &taus,
        &responses,
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
    let log_dose: Vec<f64> = ex.dose.iter().map(|d| d.ln_1p()).collect();
    let reads = PairReads {
        tissue,
        pair,
        ex: &ex,
        responses: &responses,
        log_dose: &log_dose,
        competent: &comps[comps.len() / 2],
    };
    let mut rows = Vec::with_capacity(tissue.scopes.len());
    let scopes = tissue.scopes.iter().zip(&tissue.scope_rows);
    for (((label, scope), inside_rows), e) in scopes.zip(eff.scopes) {
        let curve = if *scope == Scope::All {
            curve.take().unwrap_or_default()
        } else {
            Vec::new()
        };
        rows.push(reads.row(label, inside_rows, e, curve));
    }
    Ok(rows)
}

/// The pair's response per weighting of its programme: one joint build when the
/// reweighted programme lies on the same genes, else one build each.
fn pair_responses(tissue: &Tissue, pair: &PairData<'_>) -> Vec<Response> {
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
}

/// What every scope's row reads of one pair.
struct PairReads<'a> {
    tissue: &'a Tissue,
    pair: &'a PairData<'a>,
    ex: &'a Exposure,
    responses: &'a [Response],
    log_dose: &'a [f64],
    /// Competence at the middle threshold of the grid.
    competent: &'a [bool],
}

impl PairReads<'_> {
    /// The row of one scope, its verdict and q-values left empty.
    fn row(
        &self,
        label: &str,
        inside_rows: &[u32],
        e: ScopeEffects,
        curve: Vec<CurvePoint>,
    ) -> EffectRow {
        let (tissue, pair, ex) = (self.tissue, self.pair, self.ex);
        let mut inside = vec![false; tissue.n()];
        for &i in inside_rows {
            inside[i as usize] = true;
        }
        let comp: Vec<u32> = inside_rows
            .iter()
            .copied()
            .filter(|&i| self.competent[i as usize])
            .collect();
        let (lor, lse) = coexpression(tissue, pair.ligand_counts, pair.receptor_counts, &inside);
        let marginal_cor = combine(
            &self
                .responses
                .iter()
                .map(|r| marginal_correlation(&r.y, self.log_dose, &comp, &tissue.cf))
                .collect::<Vec<_>>(),
        );
        // How far neighbour identity explains the exposure, at the threshold
        // the average leans on.
        let eta2 = eta_squared(
            &tissue.neighbour_bin,
            &indicator(&ex.leaned_on(e.abf)),
            inside_rows,
        );
        EffectRow {
            section: label.into(),
            ligand: pair.ligand.into(),
            receptor: pair.receptor.into(),
            n_targets: pair.targets.len(),
            n_removed: pair.n_removed,
            n_receivers: inside_rows.len(),
            n_competent: comp.len(),
            n_exposed: inside_rows.iter().filter(|&&i| ex.t[i as usize]).count(),
            threshold: ex.tau,
            marginal_cor,
            n_competent_bins: self.identity_bins(inside_rows, true),
            n_incompetent_bins: self.identity_bins(inside_rows, false),
            curve,
            ..EffectRow::pending(e, (lor, lse), eta2)
        }
    }

    /// How many identity bins the rows on this `side` of competence fall in.
    fn identity_bins(&self, inside_rows: &[u32], side: bool) -> usize {
        let set: std::collections::HashSet<usize> = inside_rows
            .iter()
            .map(|&i| i as usize)
            .filter(|&i| self.competent[i] == side)
            .map(|i| self.tissue.identity_bin[i])
            .collect();
        set.len()
    }
}
