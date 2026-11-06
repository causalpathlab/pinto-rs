//! Every estimand read from the fits at one threshold, and their combination
//! across competence and weightings.

use super::averaging::AbfSummary;
use super::crossfit::DesignView;
use super::estimate::timing;
use super::fits::{arms_open, fit_main, thin_subset, Fit, Rows, ThresholdFits};
use super::inference::Estimate;
use super::options::{EffectOpts, NEAR_DETERMINED_H};
use super::overlap::{arm_contrast, difference, outcome_estimate, subset_estimate, OverlapTerms};
use super::response::Response;

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
    /// the contrast: competent effect − incompetent effect; not the headline
    /// (see the module doc).
    pub effect: Estimate,
    /// The competent effect, the headline.
    pub competent: Estimate,
    /// The unadjusted contrast over competent receivers ([`arm_contrast`]).
    pub crude: Estimate,
    /// The competent effect by overlap weights alone ([`arm_contrast`]).
    pub ipw: Estimate,
    /// The competent effect by the outcome regression alone ([`outcome_estimate`]).
    pub outcome: Estimate,
    pub incompetent: Estimate,
    pub ring: Estimate,
    /// The competent effect minus the ring effect.
    pub over_ring: Estimate,
    /// The competent effect under the section, spatial and identity steps
    /// (pooled only; NaN elsewhere).
    pub steps: [Estimate; 3],
    /// The competent effect without the neighbour-response and bleed covariates (pooled
    /// only; NaN elsewhere).
    pub nospill: Estimate,
    /// Share of the scope's competent rows whose
    /// overlap weight is below [`super::options::NEAR_DETERMINED_H`] in the main fit; NaN
    /// without competent rows or without the main fit, which is skipped
    /// when the competent subset leaves an arm thin ([`super::fits::Reads`]), on every
    /// path alike.
    pub near_determined: f64,
    /// The evidence behind the average of `competent` over thresholds
    /// ([`super::averaging::abf_average`]).
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
            crude: pick(&|e| e.crude),
            ipw: pick(&|e| e.ipw),
            outcome: pick(&|e| e.outcome),
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
/// ([`super::fits::ThresholdFits::levels`]) and each scope of `rows`, from one
/// fit per adjustment set ([`super::fits::fit_main`]) of the response `r`;
/// `fits` carries the controls' propensities
/// ([`super::fits::ThresholdFits::with_controls`]). The sensitivity fits `sens`
/// asks for serve the pooled scope (the first). A fit none of whose subsets has
/// both arms at `min_arm` is skipped ([`super::fits::Reads`]); without the main
/// fit, the near-determined shares are NaN and `h` is empty.
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
    // threshold ([`super::fits::Reads`]).
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
                    let crude = arm_contrast(&r.y, t, &comp, cf, |_| 1.0);
                    let (ipw, outcome) = match &main {
                        Some(m) if on_competent.theta.is_finite() => (
                            arm_contrast(&r.y, t, &comp, cf, |i| m.arm_weight[i]),
                            outcome_estimate(m, &on_competent, &comp),
                        ),
                        _ => (nan, nan),
                    };
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
                        crude,
                        ipw,
                        outcome,
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
/// ([`super::fits::fit_main`]); skipped when the subset leaves an arm thin
/// ([`super::fits::Reads`]). For the curve and the background.
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
            ScopeEffects::zip_with(&parts, super::averaging::combine)
        })
        .collect()
}
