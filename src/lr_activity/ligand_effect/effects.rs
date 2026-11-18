//! Every estimand read from the fits at one threshold, and their combination
//! across competence and weightings.

use super::association::arm_contrast;
use super::averaging::AbfSummary;
use super::crossfit::{CrossFit, DesignView};
use super::fits::{arms_open, fit_main, thin_subset, Fit, Rows, ThresholdFits};
use super::inference::jackknife_sd;
use super::inference::Estimate;
use super::jackknife::Replicates;
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
    /// Each scope's tile-jackknife replicates; empty without
    /// [`Rows::jackknife`].
    pub replicates: Vec<Replicates>,
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
        tiles: rows.jackknife,
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
        // The outcome model's tile influence, for the jackknife.
        let main_kind = if rows.jackknife.is_some() {
            Fit::Jackknifed
        } else {
            Fit::Main
        };
        let main = time(MAIN, || fit(runs_competent, main_kind, design))?;
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
    /// Per scope, the tiles its jackknife replicates delete
    /// ([`Rows::jackknife`]).
    tiles: Option<&'a [Vec<u32>]>,
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
        let mut replicates = Vec::new();
        let scopes = scope_rows
            .iter()
            .enumerate()
            .map(|(k, inside)| {
                let rest = (k > 0).then(|| Rest {
                    rows: complement(&scope_rows[0], inside),
                    open: headline[0],
                });
                let tiles = self.tiles.map(|t| t[k].as_slice());
                let (effects, sums, reps) =
                    self.scope(inside, competent, headline[k], k == 0, rest.as_ref(), tiles);
                if k == 0 {
                    pooled_sums = sums;
                }
                if let Some(r) = reps {
                    replicates.push(r);
                }
                effects
            })
            .collect();
        LevelEffects {
            scopes,
            pooled_sums,
            replicates,
        }
    }

    /// Every estimand on the rows `inside`, with the cluster sums of the
    /// competent effect and, deleting each of `tiles`, the estimators'
    /// jackknife replicates; `open` says the scope has a headline, and the
    /// sensitivity estimands are read on the `pooled` scope alone.
    fn scope(
        &self,
        inside: &[u32],
        competent: &[bool],
        open: bool,
        pooled: bool,
        rest: Option<&Rest>,
        tiles: Option<&[u32]>,
    ) -> (ScopeEffects, Vec<f64>, Option<Replicates>) {
        let (comp, incomp): (Vec<u32>, Vec<u32>) =
            inside.iter().partition(|&&i| competent[i as usize]);
        let main = self.terms.main.as_ref();
        let (on_competent, sums) = self.subset(main, open, &comp);
        let (on_incompetent, incompetent_sums) = self.subset(main, open, &incomp);
        let contrast = difference(&on_competent, &sums, &on_incompetent, &incompetent_sums);
        let nan = Estimate::nan(on_competent.n, on_competent.n_treated);
        let others: Option<Vec<u32>> = rest.map(|rest| {
            rest.rows
                .iter()
                .copied()
                .filter(|&i| competent[i as usize])
                .collect()
        });
        let versus_rest = match (rest, &others) {
            (Some(rest), Some(others)) => {
                let (on_rest, rest_sums) = self.subset(main, rest.open, others);
                difference(&on_competent, &sums, &on_rest, &rest_sums)
            }
            _ => nan,
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
        let mut effects = ScopeEffects {
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
        let reps = tiles.map(|tiles| {
            let mut r = self.replicates(&effects, &comp, tiles);
            r.0[4] = self.contrast_replicates(&effects.effect, &comp, &incomp, tiles);
            r.0[5] = match &others {
                Some(others) => self.contrast_replicates(
                    &effects.versus_rest,
                    &comp,
                    others,
                    self.tiles.map_or(&[][..], |t| &t[0]),
                ),
                None => Vec::new(),
            };
            r
        });
        if let Some(r) = &reps {
            // The outcome regression's SE at this threshold, which weighs its
            // thresholds, is its own jackknife's: with the fits taken as known
            // it leaves out the coefficients' error.
            (effects.outcome.se, effects.outcome.df) = jackknife_sd(&r.0[2]);
        }
        (effects, sums, reps)
    }

    /// [`Replicates`] of `e`, the estimates on the competent rows `comp`,
    /// each tile of `tiles` deleted: every estimator a ratio of per-tile
    /// sums, the full sum less the tile's, the fits held. The outcome
    /// regression's coefficients move by the tile's leave-out solve
    /// ([`super::overlap::OutcomeInfluence`]). NaN where the headline has no
    /// estimate or a deletion leaves a denominator empty.
    fn replicates(&self, e: &ScopeEffects, comp: &[u32], tiles: &[u32]) -> Replicates {
        let g = tiles.len();
        let main = self.terms.main.as_ref();
        let Some(m) = main.filter(|_| e.competent.theta.is_finite()) else {
            return Replicates(std::array::from_fn(|j| {
                vec![f64::NAN; if j < 4 { g } else { 0 }]
            }));
        };
        let mut at = vec![usize::MAX; self.cf.n_clusters()];
        for (j, &c) in tiles.iter().enumerate() {
            at[c as usize] = j;
        }
        // Per tile: Σh, Σnum, Σh(m₁ − m₀), and each arm's weight and
        // weighted response under the overlap weights and under weight 1.
        let mut sums = vec![[0.0f64; 11]; g];
        for &i in comp {
            let i = i as usize;
            let Some(s) = sums.get_mut(at[self.cf.cluster[i] as usize]) else {
                continue;
            };
            let (w, y) = (m.arm_weight[i], self.y[i]);
            let arm = if self.t[i] { 3 } else { 5 };
            s[0] += m.h[i];
            s[1] += m.num[i];
            s[2] += m.outcome[i];
            s[arm] += w;
            s[arm + 1] += w * y;
            s[arm + 4] += 1.0;
            s[arm + 5] += y;
        }
        let total: [f64; 11] = std::array::from_fn(|k| sums.iter().map(|s| s[k]).sum());
        let weights = m
            .influence
            .as_ref()
            .map(|inf| inf.weights(&m.h, comp, self.cf));
        let contrast = |r: &[f64; 11], a: usize| {
            let (w1, w0) = (r[a], r[a + 2]);
            if w1 > 1e-12 && w0 > 1e-12 {
                r[a + 1] / w1 - r[a + 3] / w0
            } else {
                f64::NAN
            }
        };
        let mut out: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(g));
        for (j, s) in sums.iter().enumerate() {
            let r: [f64; 11] = std::array::from_fn(|k| total[k] - s[k]);
            let shift = match (&m.influence, &weights) {
                (Some(inf), Some(w)) => inf.shift(w, tiles[j]),
                _ => 0.0,
            };
            let ratio = |x: f64| if r[0] > 1e-12 { x / r[0] } else { f64::NAN };
            out[0].push(ratio(r[1]));
            out[1].push(contrast(&r, 3));
            out[2].push(ratio(r[2] + shift));
            out[3].push(contrast(&r, 7));
        }
        Replicates(out)
    }

    /// The replicates of `e`, the augmented estimate on `a` less that on `b`
    /// ([`difference`]), each tile of `tiles` deleted; NaN without `e`.
    fn contrast_replicates(&self, e: &Estimate, a: &[u32], b: &[u32], tiles: &[u32]) -> Vec<f64> {
        let Some(m) = self.terms.main.as_ref().filter(|_| e.theta.is_finite()) else {
            return vec![f64::NAN; tiles.len()];
        };
        let ra = ratio_replicates(&m.num, &m.h, a, tiles, self.cf);
        let rb = ratio_replicates(&m.num, &m.h, b, tiles, self.cf);
        ra.iter().zip(&rb).map(|(x, y)| x - y).collect()
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

/// `Σ num / Σ h` over `rows`, each tile of `tiles` deleted; NaN where a
/// deletion leaves no weight.
fn ratio_replicates(
    num: &[f64],
    h: &[f64],
    rows: &[u32],
    tiles: &[u32],
    cf: &CrossFit,
) -> Vec<f64> {
    let mut at = vec![usize::MAX; cf.n_clusters()];
    for (j, &c) in tiles.iter().enumerate() {
        at[c as usize] = j;
    }
    let mut sums = vec![[0.0f64; 2]; tiles.len()];
    let (mut sn, mut sh) = (0.0, 0.0);
    for &i in rows {
        let i = i as usize;
        sn += num[i];
        sh += h[i];
        if let Some(s) = sums.get_mut(at[cf.cluster[i] as usize]) {
            s[0] += num[i];
            s[1] += h[i];
        }
    }
    sums.iter()
        .map(|s| {
            let w = sh - s[1];
            if w > 1e-12 {
                (sn - s[0]) / w
            } else {
                f64::NAN
            }
        })
        .collect()
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
