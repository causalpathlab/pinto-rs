//! Every estimand read from the fits at one threshold, and their combination
//! across competence and weightings.

use super::association::arm_contrast;
use super::averaging::AbfSummary;
use super::crossfit::{CrossFit, DesignView};
use super::fits::{arms_open, fit_main, thin_subset, Fit, Rows, ThresholdFits};
use super::inference::Estimate;
use super::options::{EffectOpts, NEAR_DETERMINED_H};
use super::overlap::{difference, outcome_estimate, subset_estimate, OverlapTerms};
use super::response::Response;
use super::timing;

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
    /// Only `theta` is reported: the rest is borrowed for averaging, and must not
    /// be read as this estimate's own uncertainty.
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
/// weight of each row's own arm (empty when the main fit was skipped).
pub struct Effects {
    pub levels: Vec<LevelEffects>,
    pub arm_weight: Vec<f64>,
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
    let terms = ResponseFits::run(rows, fits, r, sens, opts)?;
    let (t, t_ring) = (&fits.exposure.t, &fits.exposure.t_ring);
    // Per scope, whether the competent effect has an estimate: the middle
    // threshold ([`super::fits::Reads`]).
    let middle = &fits.levels[fits.levels.len() / 2];
    let headline: Vec<bool> = rows
        .scope_rows
        .iter()
        .map(|inside| arms_open(inside, t, middle, opts))
        .collect();
    let reader = ScopeReader {
        cf: rows.cf,
        t,
        t_ring,
        y: &r.y,
        terms: &terms,
        sens,
        opts,
    };
    let levels = fits
        .levels
        .iter()
        .map(|competent| reader.level(rows.scope_rows, competent, &headline))
        .collect();
    let arm_weight = match terms.main {
        Some(m) if fits.reads.competent => m.arm_weight,
        _ => Vec::new(),
    };
    Ok(Effects { levels, arm_weight })
}

/// The per-row terms of each fit of one response at one threshold; a fit
/// that was skipped is absent.
struct ResponseFits {
    main: Option<OverlapTerms>,
    ring: Option<OverlapTerms>,
    nospill: Option<OverlapTerms>,
    steps: Vec<OverlapTerms>,
}

impl ResponseFits {
    /// The main and ring fits where their subsets are open, and the
    /// sensitivity fits `sens` asks for where the main one is.
    fn run(
        rows: Rows<'_>,
        fits: &ThresholdFits<'_>,
        r: &Response,
        sens: Sensitivity,
        opts: &EffectOpts,
    ) -> anyhow::Result<Self> {
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
        Ok(Self {
            main,
            ring,
            nospill,
            steps,
        })
    }
}

/// Reads the estimands of one response at one threshold off its fits.
struct ScopeReader<'a> {
    cf: &'a CrossFit,
    t: &'a [bool],
    t_ring: &'a [bool],
    y: &'a [f64],
    terms: &'a ResponseFits,
    sens: Sensitivity,
    opts: &'a EffectOpts,
}

impl ScopeReader<'_> {
    /// Every scope at the competence level `competent`; the pooled scope is the
    /// first, and its competent effect's cluster sums are kept.
    fn level(
        &self,
        scope_rows: &[Vec<u32>],
        competent: &[bool],
        headline: &[bool],
    ) -> LevelEffects {
        let mut pooled_sums = Vec::new();
        let scopes = scope_rows
            .iter()
            .enumerate()
            .map(|(k, inside)| {
                let (effects, sums) = self.scope(inside, competent, headline[k], k == 0);
                if k == 0 {
                    pooled_sums = sums;
                }
                effects
            })
            .collect();
        LevelEffects {
            scopes,
            pooled_sums,
        }
    }

    /// Every estimand on the rows `inside`, with the cluster sums of the
    /// competent effect; `open` says the scope has a headline, and the
    /// sensitivity estimands are read on the `pooled` scope alone.
    fn scope(
        &self,
        inside: &[u32],
        competent: &[bool],
        open: bool,
        pooled: bool,
    ) -> (ScopeEffects, Vec<f64>) {
        let (comp, incomp): (Vec<u32>, Vec<u32>) =
            inside.iter().partition(|&&i| competent[i as usize]);
        let main = self.terms.main.as_ref();
        let (on_competent, sums) = self.subset(main, open, self.t, &comp);
        let (on_incompetent, incompetent_sums) = self.subset(main, open, self.t, &incomp);
        let contrast = difference(&on_competent, &sums, &on_incompetent, &incompetent_sums);
        let (on_ring, ring_sums) = self.subset(self.terms.ring.as_ref(), open, self.t_ring, &comp);
        let over_ring = difference(&on_competent, &sums, &on_ring, &ring_sums);
        let nan = Estimate::nan(on_competent.n, on_competent.n_treated);
        let crude = arm_contrast(self.y, self.t, &comp, self.cf, |_| 1.0);
        let (ipw, outcome) = match main {
            Some(m) if on_competent.theta.is_finite() => (
                arm_contrast(self.y, self.t, &comp, self.cf, |i| m.arm_weight[i]),
                outcome_estimate(m, &on_competent, &comp),
            ),
            _ => (nan, nan),
        };
        let near_determined = match main {
            Some(m) if open && !comp.is_empty() => near_determined_share(m, &comp),
            _ => f64::NAN,
        };
        let (steps, nospill) = if pooled {
            self.sensitivity(open, &comp, nan)
        } else {
            ([nan; 3], nan)
        };
        let effects = ScopeEffects {
            effect: contrast,
            competent: on_competent,
            crude,
            ipw,
            outcome,
            incompetent: on_incompetent,
            ring: on_ring,
            over_ring,
            steps,
            nospill,
            near_determined,
            abf: AbfSummary::NONE,
        };
        (effects, sums)
    }

    /// A subset's estimate from `terms`, or the guard's verdict on it when the
    /// fit was skipped or the scope has no headline.
    fn subset(
        &self,
        terms: Option<&OverlapTerms>,
        open: bool,
        t: &[bool],
        sub: &[u32],
    ) -> (Estimate, Vec<f64>) {
        match terms {
            Some(terms) if open => self.opts.guard(subset_estimate(terms, t, sub, self.cf)),
            _ => thin_subset(t, sub, self.cf),
        }
    }

    /// The step and no-spill estimands on the competent rows `comp`; an estimand
    /// not asked for is `nan`.
    fn sensitivity(&self, open: bool, comp: &[u32], nan: Estimate) -> ([Estimate; 3], Estimate) {
        let mut steps = [nan; 3];
        let mut nospill = nan;
        if self.sens.nospill {
            nospill = self
                .subset(self.terms.nospill.as_ref(), open, self.t, comp)
                .0;
        }
        if self.sens.steps {
            for (j, e) in steps.iter_mut().enumerate() {
                *e = self.subset(self.terms.steps.get(j), open, self.t, comp).0;
            }
        }
        (steps, nospill)
    }
}

/// Share of the rows `comp` whose overlap weight is below
/// [`NEAR_DETERMINED_H`]; `comp` is not empty.
fn near_determined_share(terms: &OverlapTerms, comp: &[u32]) -> f64 {
    let near = comp
        .iter()
        .filter(|&&i| terms.h[i as usize] < NEAR_DETERMINED_H)
        .count();
    near as f64 / comp.len() as f64
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
