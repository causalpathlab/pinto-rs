//! The nuisance fits at one exposure threshold, shared by every estimand read there.

use super::crossfit::{CrossFit, DesignView};
use super::dose::Exposure;
use super::inference::Estimate;
use super::options::EffectOpts;
use super::overlap::{
    overlap_terms_with, propensity, propensity_bordered, FoldStarts, OverlapTerms, PairInputs,
    Propensity,
};
use super::regression::standardize;
use super::response::Response;
use super::timing;
use super::tissue::Tissue;

/// The rows an effect is estimated on: the tissue's receivers, with the rows
/// inside each scope, pooled first ([`Tissue::scopes`]).
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub design: &'a super::crossfit::Design,
    pub steps: &'a [super::tissue::AdjustmentStep],
    pub cf: &'a super::crossfit::CrossFit,
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
/// ([`super::overlap::subset_estimate`]). Every other scope is a subset of the
/// pooled one, so neither of its arms is larger.
pub(super) fn arms_open(inside: &[u32], t: &[bool], level: &[bool], opts: &EffectOpts) -> bool {
    let competent = inside.iter().filter(|&&i| level[i as usize]);
    let (n, treated) = competent.fold((0, 0), |(n, x), &i| (n + 1, x + t[i as usize] as usize));
    !opts.thin_arm(n, treated)
}

/// What the guard reports of a subset it rejects for a thin arm, without
/// the fit: not estimable with the subset's counts, and zero cluster sums.
pub(super) fn thin_subset(t: &[bool], rows: &[u32], cf: &CrossFit) -> (Estimate, Vec<f64>) {
    let treated = rows.iter().filter(|&&i| t[i as usize]).count();
    (
        Estimate::nan(rows.len(), treated),
        vec![0.0; cf.n_clusters()],
    )
}

/// Which of a threshold's fits serve a subset with both arms at `min_arm`
/// ([`super::options::thin_arm`]) on the pooled rows. The main fit is read for
/// the effect in competent receivers, at the middle competence threshold: a
/// pair resting on a looser threshold alone has no estimate, whatever those
/// thresholds show. The no-spill, step, curve and background fits read the same
/// subset. The ring fit is read beside it, when one of its own subsets is open.
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
    ring: Option<Propensity>,
    /// Each adjustment step's, when asked for and the competent effect is read.
    steps: Option<Vec<Propensity>>,
}

/// One threshold's exposure side as its fits read it, built once and shared by
/// every fit at it: the competence indicators (raw, the outcome
/// model's effect modifiers), the pair columns both models carry
/// (standardized: own ligand, niche, every competence indicator, and the
/// contact exposure the ring fit adds to them), which fits serve a subset
/// ([`Reads`]), and the cross-fitted propensities ([`propensity`]) of the
/// main fit and, with [`ThresholdFits::with_controls`], of the ring fit and each
/// adjustment step. These read no response-side column, so one serves every
/// programme weighting, the background and the no-spill fit; a fit that adds
/// spillover and bleed borders it ([`fit_main`]).
pub struct ThresholdFits<'a> {
    pub exposure: &'a Exposure,
    /// Competence at each threshold, per row.
    pub levels: &'a [Vec<bool>],
    pub reads: Reads,
    /// [`super::options::EffectOpts::fit_thin`]: run the fits [`Reads`] would skip.
    force: bool,
    modifiers: Vec<Vec<f64>>,
    extra: Vec<Vec<f64>>,
    exposed: Vec<f64>,
    main: Option<Propensity>,
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
    /// step's: what [`super::effects::effects_on`] needs.
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
    pub(super) fn runs(&self, read: bool) -> bool {
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
    fn propensity(&self, kind: Fit, step: Option<usize>) -> anyhow::Result<&Propensity> {
        let controls = || {
            self.controls
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("the controls' propensities are not fitted"))
        };
        let e = match (kind, step) {
            (Fit::Ring, _) => controls()?.ring.as_ref(),
            (_, Some(s)) => controls()?.steps.as_ref().and_then(|v| v.get(s)),
            _ => self.main.as_ref(),
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
/// treatment, in both models; and spillover and bleed (not under
/// [`Fit::NoSpill`]), in both models too, so the estimator is doubly robust
/// in them as well. Read off the response, they would tie the propensity to
/// one programme, so it is not refitted: the threshold's shared
/// exposure-side fit is bordered with these two columns
/// ([`propensity_bordered`]).
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
    let x = fits.exposure;
    let t = if kind == Fit::Ring { &x.t_ring } else { &x.t };
    let mut extra = fits.extra(kind);
    let modifiers: Vec<&[f64]> = fits.modifiers.iter().map(|c| c.as_slice()).collect();
    // Spillover and bleed in both models, so the propensity of this
    // response balances them.
    let added: Vec<&[f64]> = match kind {
        Fit::NoSpill => Vec::new(),
        Fit::Main | Fit::Ring => vec![&r.spill, &r.bleed],
    };
    let shared = fits.propensity(kind, step)?;
    let own;
    let e = if added.is_empty() {
        shared.e.as_slice()
    } else {
        own = propensity_bordered(design, cf, t, &extra, &added, shared, &opts.nuisance)?;
        extra.extend(added);
        own.as_slice()
    };
    let inputs = PairInputs {
        t,
        y: &r.y,
        extra: &extra,
        modifiers: &modifiers,
    };
    overlap_terms_with(design, cf, &inputs, e, &opts.nuisance)
}
