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
use crate::util::common::HashMap;

/// Every estimand in one scope.
#[derive(Clone, Copy, Debug)]
pub struct ScopeEffects {
    /// the contrast: competent effect − incompetent effect; not the headline
    /// (see the module doc).
    pub effect: Estimate,
    /// The competent effect, the headline.
    pub competent: Estimate,
    /// The unadjusted difference: over the competent receivers `C`, the mean
    /// response of the exposed minus the mean response of the unexposed,
    ///
    /// `Σ_{i∈C, T_i=1} y_i / n_1 − Σ_{i∈C, T_i=0} y_i / n_0`,
    ///
    /// every receiver weighted 1 ([`arm_contrast`] with unit weights). Nothing
    /// is adjusted for: no cell type, space, depth or bleed covariates, no
    /// weights and no outcome model. Its SE is clustered by tile, and like the
    /// headline it is averaged over exposure thresholds. It is the answer a
    /// comparison without adjustment would give, the reference the adjusted
    /// estimates are read against, and it enters no verdict.
    pub unadj: Estimate,
    /// The competent effect by overlap weights alone ([`arm_contrast`]).
    pub ipw: Estimate,
    /// The competent effect by the outcome regression alone ([`outcome_estimate`]).
    pub outcome: Estimate,
    pub incompetent: Estimate,
    /// The competent effect in this scope minus that on every other
    /// competent receiver, from the same fit: whether the effect is the
    /// scope's own (a section's, a region's) rather than the tissue's. NaN
    /// on the pooled scope, which has no rest. Tiles never cross sections,
    /// so for a section the two parts share no cluster; a region's rows
    /// may share tiles with the rest, which the clustered variance of the
    /// difference counts.
    pub versus_rest: Estimate,
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
    /// Several analyzes of one scope combined estimand by estimand with
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
            unadj: pick(&|e| e.unadj),
            ipw: pick(&|e| e.ipw),
            outcome: pick(&|e| e.outcome),
            incompetent: pick(&|e| e.incompetent),
            versus_rest: pick(&|e| e.versus_rest),
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

/// Which sensitivity fits [`effects_on`] runs beside the main fit; each
/// serves the pooled scope alone, and an estimand left out is NaN, which the
/// average then leaves out.
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
    /// The adjustment steps alone: what the simulator reads, having no
    /// spillover for the no-spill fit to measure.
    pub const STEPS: Self = Self {
        nospill: false,
        steps: true,
    };
}

/// The estimands on `rows` for each competence threshold
/// ([`super::fits::ThresholdFits::levels`]) and each scope of `rows`, from one
/// fit per adjustment set ([`super::fits::fit_main`]) of the response `r`;
/// `fits` carries the adjustment steps' propensities
/// ([`super::fits::ThresholdFits::with_steps`]). The sensitivity fits `sens`
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
    let t = &fits.exposure.t;
    // Per scope, whether the competent effect has an estimate: the middle
    // threshold ([`super::fits::Reads`]).
    let middle = &fits.levels[fits.levels.len() / 2];
    let headline: Vec<bool> = rows
        .scope_rows
        .iter()
        .map(|inside| arms_open(inside, t, middle, opts))
        .collect();
    let reader = ScopeReader {
        deleted: rows.deleted,
        cf: rows.cf,
        t,
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
    nospill: Option<OverlapTerms>,
    steps: Vec<OverlapTerms>,
}

impl ResponseFits {
    /// The main fit where its subset is open, and the sensitivity fits
    /// `sens` asks for where it is.
    fn run(
        rows: Rows<'_>,
        fits: &ThresholdFits<'_>,
        r: &Response,
        sens: Sensitivity,
        opts: &EffectOpts,
    ) -> anyhow::Result<Self> {
        use timing::{time, MAIN, NOSPILL, STEPS};
        let (design, cf) = (rows.design.view(), rows.cf);
        let fit = |run: bool, kind: Fit, d: DesignView<'_>| {
            run.then(|| fit_main(d, cf, fits, r, kind, None, opts))
                .transpose()
        };
        let runs_competent = fits.runs_competent();
        let main = time(MAIN, || fit(runs_competent, Fit::Main, design))?;
        let nospill = time(NOSPILL, || {
            fit(sens.nospill && runs_competent, Fit::NoSpill, design)
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
            nospill,
            steps,
        })
    }
}

/// Reads the estimands of one response at one threshold off its fits.
struct ScopeReader<'a> {
    /// Per scope, the scope it is a jackknife replicate of and the tile
    /// deleted ([`Rows::deleted`]).
    deleted: &'a [Option<(usize, u32)>],
    cf: &'a CrossFit,
    t: &'a [bool],
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
        // A jackknife replicate's outcome regression has the tile out of its
        // rows, and its coefficients move by the tile's leave-out shift
        // ([`super::overlap::OutcomeInfluence`]), weighed by its scope's rows,
        // read once for all of the scope's replicates.
        let main = self.terms.main.as_ref();
        let influence = main.and_then(|m| Some((&m.h, m.influence.as_ref()?)));
        let mut weights: HashMap<usize, Vec<Vec<f64>>> = HashMap::default();
        if let Some((h, inf)) = influence {
            for &(parent, _) in self.deleted.iter().flatten() {
                weights.entry(parent).or_insert_with(|| {
                    let comp: Vec<u32> = scope_rows[parent]
                        .iter()
                        .copied()
                        .filter(|&i| competent[i as usize])
                        .collect();
                    inf.weights(h, &comp, self.cf)
                });
            }
        }
        let scopes = scope_rows
            .iter()
            .enumerate()
            .map(|(k, inside)| {
                let deleted = self.deleted.get(k).copied().flatten();
                // A replicate's contrast with the rest is not read.
                let rest = (k > 0 && deleted.is_none()).then(|| Rest {
                    rows: complement(&scope_rows[0], inside),
                    open: headline[0],
                });
                let (mut effects, sums) =
                    self.scope(inside, competent, headline[k], k == 0, rest.as_ref());
                if k == 0 {
                    pooled_sums = sums;
                }
                if let (Some((parent, tile)), Some((_, inf))) = (deleted, influence) {
                    let o = &mut effects.outcome;
                    let sh = o.mean_h * o.n as f64;
                    if let Some(v) = weights.get(&parent).filter(|_| sh > 1e-12) {
                        o.theta += inf.shift(v, tile) / sh;
                    }
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
        rest: Option<&Rest>,
    ) -> (ScopeEffects, Vec<f64>) {
        let (comp, incomp): (Vec<u32>, Vec<u32>) =
            inside.iter().partition(|&&i| competent[i as usize]);
        let main = self.terms.main.as_ref();
        let (on_competent, sums) = self.subset(main, open, &comp);
        let (on_incompetent, incompetent_sums) = self.subset(main, open, &incomp);
        let contrast = difference(&on_competent, &sums, &on_incompetent, &incompetent_sums);
        let nan = Estimate::nan(on_competent.n, on_competent.n_treated);
        let versus_rest = match rest {
            Some(rest) => {
                let others: Vec<u32> = rest
                    .rows
                    .iter()
                    .copied()
                    .filter(|&i| competent[i as usize])
                    .collect();
                let (on_rest, rest_sums) = self.subset(main, rest.open, &others);
                difference(&on_competent, &sums, &on_rest, &rest_sums)
            }
            None => nan,
        };
        let unadj = arm_contrast(self.y, self.t, &comp, self.cf, |_| 1.0);
        let (ipw, outcome) = match main {
            Some(m) if on_competent.theta.is_finite() => (
                arm_contrast(self.y, self.t, &comp, self.cf, |i| m.arm_weight[i]),
                outcome_estimate(m, self.t, &comp, self.cf),
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
            unadj,
            ipw,
            outcome,
            incompetent: on_incompetent,
            versus_rest,
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
        sub: &[u32],
    ) -> (Estimate, Vec<f64>) {
        match terms {
            Some(terms) if open => self
                .opts
                .guard(subset_estimate(terms, self.t, sub, self.cf)),
            _ => thin_subset(self.t, sub, self.cf),
        }
    }

    /// The step and no-spill estimands on the competent rows `comp`; an estimand
    /// not asked for is `nan`.
    fn sensitivity(&self, open: bool, comp: &[u32], nan: Estimate) -> ([Estimate; 3], Estimate) {
        let mut steps = [nan; 3];
        let mut nospill = nan;
        if self.sens.nospill {
            nospill = self.subset(self.terms.nospill.as_ref(), open, comp).0;
        }
        if self.sens.steps {
            for (j, e) in steps.iter_mut().enumerate() {
                *e = self.subset(self.terms.steps.get(j), open, comp).0;
            }
        }
        (steps, nospill)
    }
}

/// The pooled rows outside a scope, and whether the pooled scope has a
/// headline, which decides whether their estimate is read.
struct Rest {
    rows: Vec<u32>,
    open: bool,
}

/// The rows of `all` not in `inside`, both ascending.
pub(super) fn complement(all: &[u32], inside: &[u32]) -> Vec<u32> {
    let mut j = 0;
    all.iter()
        .copied()
        .filter(|&i| {
            while j < inside.len() && inside[j] < i {
                j += 1;
            }
            !(j < inside.len() && inside[j] == i)
        })
        .collect()
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
