//! One ligand-receptor pair: exposure, response, and every estimand.
//!
//! | Column | Estimand |
//! |---|---|
//! | `coexpr_log_or` | log odds of ligand on one side and receptor on the other, across contacts (association) |
//! | `marginal_cor` | Pearson correlation of the response and `log1p` contact ligand over competent receivers, on its own scale (association) |
//! | `crude` | mean response of exposed minus unexposed competent receivers, no adjustment (association) |
//! | `effect_ipw` | the competent effect by overlap weights alone, no outcome model |
//! | `effect_outcome` | the competent effect by the outcome regression alone, averaged under the overlap weights, no residual correction |
//! | `effect_competent` | the competent effect, **the headline**: both combined, doubly robust, overlap-weighted effect where the receptor is expressed, averaged over exposure thresholds |
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
//! `crude`, `effect_ipw` and `effect_outcome` are averaged over thresholds as
//! the headline is, each by its own evidence, and read for comparison: they
//! enter no verdict. The sensitivity fits below (the nested adjustment sets
//! and the no-spill fit) read the headline threshold alone, so a gap between
//! them and `effect_competent` can come from the threshold as well as the
//! adjustment.
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
//! The two response-side columns enter the outcome model only, so one
//! propensity serves every fit at a threshold with the same treatment and
//! design ([`super::fits::ThresholdFits`]).
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
//! negative controls, averaged the same way, is what guards against that.
//!
//! **Threshold–response curve.** The headline τ is chosen per pair
//! ([`super::dose::choose_threshold`]). Beside it, the competent effect
//! (pooled) at τ = 0 and at the quartiles of the positive exposures
//! ([`super::curve::curve_taus`]) with pointwise intervals and a simultaneous
//! 95% band ([`super::curve::sup_t_crit`]): how the effect grows with dose, and
//! whether the headline rests on one threshold.
//!
//! **Background programmes** ([`super::background::BackgroundPool`]): the
//! pair's programme with every target swapped for a random gene of similar mean
//! expression outside every programme and pair, same weights, drawn many times.
//! They should show no effect; when they do, exposure moves generic expression
//! (activation, depth, residual confounding), and the pair's own competent
//! effect is a finding only where it stands out from them
//! ([`super::averaging::background_comparison`]).
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

use super::averaging::combine;
use super::background::BackgroundPool;
use super::curve::threshold_curve;
use super::dose::Exposure;
use super::effects::{Scope, Sensitivity};
use super::fits::{indicator, Rows};
use super::inference::Estimate;
use super::options::EffectOpts;
use super::overlap::marginal_correlation;
use super::report::eta_squared;
use super::response::{genes_and_weights, Response, Targets};
use super::rows::EffectRow;
use super::thresholds::{aligned, over_thresholds, thresholds};
use super::tissue::Tissue;
use crate::lr_activity::edge_scores::jeffreys_log_odds;
use legume_numeric::matrix::rand_util::{mix_seed, name_seed};
use rayon::prelude::*;

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
/// type, with a background programme matched from `pool`. Verdicts and q-values
/// are left empty for [`super::report::report`], which needs every pair.
///
/// With [`PairData::reweighted`], the two weightings of the programme are
/// combined ([`super::thresholds::over_thresholds`]); the curve and the
/// background describe the pair, not the weighting, and are read under
/// `targets`. Counts, co-expression, the marginal correlation and the
/// diagnostics read competence at the middle threshold of the grid.
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
    // Seeded by the pair, so a pair's background programmes don't depend on
    // which others run.
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
    let log_dose: Vec<f64> = ex.dose.iter().map(|d| d.ln_1p()).collect();
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
        let marginal_cor = combine(
            &responses
                .iter()
                .map(|r| marginal_correlation(&r.y, &log_dose, &comp, &tissue.cf))
                .collect::<Vec<_>>(),
        );
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
            marginal_cor,
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

/// The co-expression log odds over the contacts inside the scope, both
/// orientations: the Jeffreys log odds ratio of ligand detected in the sender
/// and receptor detected in the receiver, and its SE.
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
