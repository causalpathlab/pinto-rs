//! One pair across exposure thresholds: each estimand fitted at each, then
//! averaged by its own evidence.

use super::averaging::{abf_average, abf_summary, background_comparison};
use super::curve::curve_taus;
use super::dose::Exposure;
use super::effects::{combine_scopes, competent_effect, effects_on, ScopeEffects, Sensitivity};
use super::estimate::timing;
use super::fits::{Rows, Starts, ThresholdFits};
use super::inference::Estimate;
use super::options::EffectOpts;
use super::response::Response;

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
