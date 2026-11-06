//! One pair across exposure thresholds: each estimand fitted at each, then
//! averaged by its own evidence.

use super::averaging::{abf_average, abf_summary, background_comparison};
use super::curve::curve_taus;
use super::dose::Exposure;
use super::effects::{
    combine_scopes, competent_effect, effects_on, LevelEffects, ScopeEffects, Sensitivity,
};
use super::fits::{Rows, Starts, ThresholdFits};
use super::inference::Estimate;
use super::options::EffectOpts;
use super::response::Response;
use super::timing;

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
    /// The main fit's arm weights under the first weighting, one per point of
    /// `points`; empty unless [`super::options::EffectOpts::keep_arm_weights`].
    pub arm_weights: Vec<Vec<f64>>,
}

/// The thresholds a pair is read at beyond the headline: none without
/// [`super::options::EffectOpts::curve`], else [`super::curve::curve_taus`] on
/// the rows competent at some level.
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
/// their posterior weights, estimand by estimand
/// ([`super::averaging::abf_average`]), so each control is averaged by its own
/// evidence and not by the programme's.
///
/// A fixed [`super::options::EffectOpts::threshold`] is not averaged: its
/// estimate is the headline, and `taus` only the curve. A headline threshold
/// with no estimate ends the pair: the other thresholds are not fitted, nor are
/// the background programmes made. `sens` runs at the headline threshold alone,
/// and the background programmes read the middle competence threshold.
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
    let grid = threshold_grid(ex, taus);
    let mut starts = Starts::default();
    let mut at_tau: Vec<Vec<ScopeEffects>> = Vec::with_capacity(grid.len());
    let mut points = Vec::with_capacity(grid.len());
    let mut arm_weights = Vec::new();
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
                if opts.keep_arm_weights {
                    arm_weights.push(per_c.arm_weight);
                }
            }
            per_weighting.push(combine_levels(per_c.levels, mid));
        }
        at_tau.push(combine_scopes(&per_weighting));
        if g > 0 {
            timing::add(timing::THRESHOLDS, clock);
        }
        background_at.push(background_effects(
            rows,
            &fits,
            &background_responses,
            opts,
        )?);
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
    let scopes = average_scopes(&at_tau[..averaged], &grid, prior_var);
    let background_means = average_backgrounds(
        &background_at[..averaged],
        background_responses.len(),
        prior_var,
    );
    let (background, over_background) =
        background_comparison(&scopes[0].competent, &background_means);
    Ok(ThresholdAverage {
        scopes,
        background,
        over_background,
        points,
        arm_weights,
    })
}

/// The thresholds to fit: the headline first; a curve threshold equal to it is
/// read off its fits.
fn threshold_grid(ex: &Exposure, taus: &[(f64, f64)]) -> Vec<f64> {
    let mut grid: Vec<f64> = vec![ex.tau];
    grid.extend(taus.iter().map(|t| t.0).filter(|&t| t != ex.tau));
    grid
}

/// One weighting's estimands combined over the competence thresholds, with the
/// counts of the level `mid`, the one that decides whether a pair is estimable.
fn combine_levels(levels: Vec<LevelEffects>, mid: usize) -> Vec<ScopeEffects> {
    let scopes: Vec<Vec<ScopeEffects>> = levels.into_iter().map(|l| l.scopes).collect();
    let mut combined = combine_scopes(&scopes);
    for (c, m) in combined.iter_mut().zip(&scopes[mid]) {
        c.competent.n = m.competent.n;
        c.competent.n_treated = m.competent.n_treated;
    }
    combined
}

/// The competent effect of each background programme at one threshold.
fn background_effects(
    rows: Rows<'_>,
    fits: &ThresholdFits<'_>,
    responses: &[Response],
    opts: &EffectOpts,
) -> anyhow::Result<Vec<Estimate>> {
    responses
        .iter()
        .map(|r| {
            timing::time(timing::BACKGROUND, || competent_effect(rows, fits, r, opts)).map(|e| e.0)
        })
        .collect()
}

/// Each scope's estimands averaged over the thresholds `at_tau` by their own
/// evidence; the near-determined share is the headline threshold's, and `grid`
/// names the thresholds in the evidence summary.
fn average_scopes(at_tau: &[Vec<ScopeEffects>], grid: &[f64], prior_var: f64) -> Vec<ScopeEffects> {
    (0..at_tau[0].len())
        .map(|s| {
            let parts: Vec<ScopeEffects> = at_tau.iter().map(|a| a[s]).collect();
            let competent: Vec<Estimate> = parts.iter().map(|p| p.competent).collect();
            ScopeEffects {
                near_determined: parts[0].near_determined,
                abf: abf_summary(&competent, grid, prior_var),
                ..ScopeEffects::zip_with(&parts, |p| abf_average(p, prior_var))
            }
        })
        .collect()
}

/// Each of the `n` background programmes averaged over the thresholds by its own
/// evidence, as the effect is.
fn average_backgrounds(at_tau: &[Vec<Estimate>], n: usize, prior_var: f64) -> Vec<Estimate> {
    (0..n)
        .map(|k| {
            let parts: Vec<Estimate> = at_tau.iter().map(|d| d[k]).collect();
            abf_average(&parts, prior_var)
        })
        .collect()
}

/// `b`'s weights on `a`'s genes (0 where `b` leaves a gene out), when
/// `b`'s genes are a subsequence of `a`'s; `None` otherwise.
pub(super) fn aligned(a: &[(f64, &[(u32, f32)])], b: &[(f64, &[(u32, f32)])]) -> Option<Vec<f64>> {
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
