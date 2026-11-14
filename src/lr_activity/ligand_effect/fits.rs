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
use crate::util::common::HashMap;
use std::borrow::Cow;
use std::sync::{Arc, Mutex};

/// The rows an effect is estimated on: the tissue's receivers, with the rows
/// inside each scope, pooled first ([`Tissue::scopes`]).
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub design: &'a super::crossfit::Design,
    pub steps: &'a [super::tissue::AdjustmentStep],
    pub cf: &'a super::crossfit::CrossFit,
    pub scope_rows: &'a [Vec<u32>],
    /// The ligand's shared base propensities on these rows, when its pairs
    /// share them ([`LigandShare`]).
    pub share: Option<&'a LigandShare>,
}

impl<'a> Rows<'a> {
    pub fn cells(tissue: &'a Tissue) -> Self {
        Self {
            design: &tissue.design,
            steps: &tissue.steps,
            cf: &tissue.cf,
            scope_rows: &tissue.scope_rows,
            share: None,
        }
    }

    /// The same rows with the pooled scope alone.
    #[cfg(test)]
    pub fn pooled(self) -> Self {
        Self {
            scope_rows: &self.scope_rows[..1],
            ..self
        }
    }

    /// The same rows, the propensities bordered from `share`'s base fits.
    pub fn shared(self, share: &'a LigandShare) -> Self {
        Self {
            share: Some(share),
            ..self
        }
    }
}

/// Which fit of a pair: the main one, or the main one without the
/// response-side bleed columns (spillover, bleed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    Main,
    NoSpill,
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
/// thresholds show. The no-spill, step and curve fits read the same
/// subset.
///
/// A fit not read is skipped, propensity and all, and its estimands are what
/// the guard would make of them, not estimable with their counts
/// ([`thin_subset`]): the skip is the guard read before the fit, not a second
/// rule.
#[derive(Clone, Copy, Debug)]
pub struct Reads {
    pub competent: bool,
}

impl Reads {
    pub fn of(rows: Rows<'_>, x: &Exposure, levels: &[Vec<bool>], opts: &EffectOpts) -> Self {
        let pooled = &rows.scope_rows[0];
        let competent = levels
            .get(levels.len() / 2)
            .is_some_and(|c| arms_open(pooled, &x.t, c, opts));
        Self { competent }
    }
}

/// One threshold's exposure side as its fits read it, built once and shared by
/// every fit at it: the competence indicators (raw, the outcome
/// model's effect modifiers), the pair columns both models carry
/// (standardized: own ligand, niche, every competence indicator), which fits
/// serve a subset ([`Reads`]), and the cross-fitted propensities
/// ([`propensity`]) of the main fit and, with [`ThresholdFits::with_steps`],
/// of each adjustment step. These read no response-side column, so one serves every
/// programme weighting and the no-spill fit; a fit that adds
/// spillover and bleed borders it ([`fit_main`]).
pub struct ThresholdFits<'a> {
    pub exposure: &'a Exposure,
    /// Competence at each threshold, per row.
    pub levels: &'a [Vec<bool>],
    pub reads: Reads,
    /// [`super::options::EffectOpts::fit_thin`]: run the fits [`Reads`] would skip.
    force: bool,
    modifiers: Vec<Vec<f64>>,
    /// The pair columns of both models that depend on the ligand alone
    /// (own ligand, niche), standardized: what a shared base carries.
    ligand_cols: Vec<Vec<f64>>,
    /// The receptor's pair columns (every competence indicator),
    /// standardized: what a pair borders a shared base with.
    receptor_cols: Vec<Vec<f64>>,
    main: Option<PairPropensity>,
    /// Each adjustment step's propensity, when asked for and the competent
    /// effect is read.
    steps: Option<Vec<PairPropensity>>,
}

/// A base fit's key: its exposure threshold (as bits) and its adjustment
/// step, `None` for the main fit.
type BaseKey = (u64, Option<usize>);

/// One ligand's propensities on the design and its own columns (own ligand,
/// niche), per exposure threshold and adjustment step, fitted by the first
/// pair that asks and shared by every later pair of the ligand.
///
/// The exposure, own ligand and niche depend on the ligand alone, so of a
/// pair's propensity columns only the competence indicators belong to the
/// receptor. A pair borders the shared fit with them
/// ([`propensity_bordered`]), which reaches the optimum the pair's own fit
/// on every column reaches, for a fraction of its cost.
///
/// A main base starts cold, and a step's from its main base, so a base, and
/// every pair bordered on it, is the same whichever of the ligand's pairs
/// asked first: a pair estimated alone (`--only`) gets what it gets in a full
/// run. (Starting a curve threshold's base from the headline's was tried and
/// is slower: the share exposed moves with the threshold, so zero is the
/// nearer start.)
///
/// Valid for one ligand on one set of rows (design and folds) and one
/// [`EffectOpts`]: a share is made per ligand and dropped with it. A
/// ligand's pairs run one after another, so a share has one writer and
/// holds its lock through a fit.
#[derive(Default)]
pub struct LigandShare {
    fits: Mutex<HashMap<BaseKey, Arc<Propensity>>>,
    /// Base propensities fitted, reuses not counted: the map alone would not
    /// show a base fitted twice.
    #[cfg(test)]
    fitted: std::sync::atomic::AtomicUsize,
}

impl LigandShare {
    /// How many base propensities have been fitted.
    #[cfg(test)]
    pub fn fits(&self) -> usize {
        self.fitted.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The base propensity at threshold `tau` of `step` (the main fit when
    /// `None`), fitted by `fit` from its warm start on first use.
    fn base(
        &self,
        tau: f64,
        step: Option<usize>,
        fit: impl FnOnce() -> anyhow::Result<Propensity>,
    ) -> anyhow::Result<Arc<Propensity>> {
        let mut fits = self.fits.lock().expect("a share is not poisoned");
        if let Some(p) = fits.get(&(tau.to_bits(), step)) {
            return Ok(p.clone());
        }
        let p = Arc::new(fit()?);
        #[cfg(test)]
        self.fitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        fits.insert((tau.to_bits(), step), p.clone());
        Ok(p)
    }
}

/// A pair's cross-fitted propensity: fitted on every pair column, or its
/// ligand's shared base, which the pair borders with its receptor columns
/// where a propensity is read.
enum PairPropensity {
    Own(Propensity),
    Shared(Arc<Propensity>),
}

impl PairPropensity {
    /// The propensity on the pair's `ligand` and `receptor` columns and
    /// `added` after them, bordered from the fit this one holds.
    fn on<'p>(
        &'p self,
        design: DesignView<'_>,
        cf: &CrossFit,
        t: &[bool],
        (ligand, receptor): (&[&[f64]], &[&[f64]]),
        added: &[&[f64]],
    ) -> anyhow::Result<Cow<'p, [f64]>> {
        Ok(match self {
            Self::Own(p) if added.is_empty() => Cow::Borrowed(p.e.as_slice()),
            Self::Own(p) => {
                let extra: Vec<&[f64]> = ligand.iter().chain(receptor).copied().collect();
                Cow::Owned(propensity_bordered(design, cf, t, &extra, added, p)?)
            }
            Self::Shared(base) => {
                let more: Vec<&[f64]> = receptor.iter().chain(added).copied().collect();
                Cow::Owned(propensity_bordered(design, cf, t, ligand, &more, base)?)
            }
        })
    }
}

/// The pair's propensity on `design` and its `ligand` and `receptor`
/// columns: its ligand's shared base at `tau` when there is a `share`, else
/// fitted on every column from `starts`.
fn pair_propensity(
    design: DesignView<'_>,
    cf: &CrossFit,
    t: &[bool],
    (ligand, receptor): (&[&[f64]], &[&[f64]]),
    opts: &EffectOpts,
    starts: &mut FoldStarts,
    share: Option<(&LigandShare, f64, Option<usize>)>,
) -> anyhow::Result<PairPropensity> {
    let fit =
        |cols: &[&[f64]], s: &mut FoldStarts| propensity(design, cf, t, cols, &opts.nuisance, s);
    let Some((share, tau, step)) = share else {
        let extra: Vec<&[f64]> = ligand.iter().chain(receptor).copied().collect();
        return Ok(PairPropensity::Own(fit(&extra, starts)?));
    };
    let base = match step {
        // Its solutions are what the adjustment steps nest in.
        None => {
            let base = share.base(tau, None, || fit(ligand, &mut FoldStarts::default()))?;
            *starts = base.starts();
            base
        }
        // From the main base cut down to the step's design (`starts`).
        Some(_) => share.base(tau, step, || fit(ligand, starts))?,
    };
    Ok(PairPropensity::Shared(base))
}

/// Columns as slices.
fn slices(cols: &[Vec<f64>]) -> Vec<&[f64]> {
    cols.iter().map(|c| c.as_slice()).collect()
}

pub(super) fn indicator(v: &[bool]) -> Vec<f64> {
    v.iter().map(|&b| b as u8 as f64).collect()
}

impl<'a> ThresholdFits<'a> {
    /// The columns and the main fit's propensity on `rows`, bordered from
    /// the ligand's shared base fits when `rows` carries a share.
    pub fn new(
        rows: Rows<'a>,
        exposure: &'a Exposure,
        levels: &'a [Vec<bool>],
        opts: &EffectOpts,
        starts: &mut FoldStarts,
    ) -> anyhow::Result<Self> {
        let modifiers: Vec<Vec<f64>> = levels.iter().map(|c| indicator(c)).collect();
        let ligand_cols = [&exposure.own, &exposure.niche]
            .into_iter()
            .map(|c| standardize(c))
            .collect();
        let receptor_cols = modifiers.iter().map(|c| standardize(c)).collect();
        let mut fits = Self {
            exposure,
            levels,
            reads: Reads::of(rows, exposure, levels, opts),
            force: opts.fit_thin,
            modifiers,
            ligand_cols,
            receptor_cols,
            main: None,
            steps: None,
        };
        if fits.runs_competent() {
            let main = pair_propensity(
                rows.design.view(),
                rows.cf,
                &exposure.t,
                (&slices(&fits.ligand_cols), &slices(&fits.receptor_cols)),
                opts,
                starts,
                rows.share.map(|s| (s, exposure.tau, None)),
            )?;
            fits.main = Some(main);
        }
        Ok(fits)
    }

    /// Also, when `steps`, each adjustment step's propensity, started from
    /// the main fit's solutions `main`: what [`super::effects::effects_on`]
    /// needs.
    pub fn with_steps(
        mut self,
        rows: Rows<'_>,
        steps: bool,
        opts: &EffectOpts,
        main: &FoldStarts,
    ) -> anyhow::Result<Self> {
        let x = self.exposure;
        self.steps = if steps && self.runs_competent() {
            let (ligand, receptor) = (slices(&self.ligand_cols), slices(&self.receptor_cols));
            // A step starts from the main fit, the model it nests in.
            let p = rows.design.p();
            let share = rows.share;
            Some(timing::time(timing::STEPS, || {
                rows.steps
                    .iter()
                    .enumerate()
                    .map(|(k, s)| {
                        let view = s.design.view(rows.design);
                        let mut start = main.nested(p, s.design.p());
                        let share = share.map(|sh| (sh, x.tau, Some(k)));
                        let cols = (ligand.as_slice(), receptor.as_slice());
                        pair_propensity(view, rows.cf, &x.t, cols, opts, &mut start, share)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })?)
        } else {
            None
        };
        Ok(self)
    }

    /// Whether the fits serving the competent subsets run.
    pub(super) fn runs_competent(&self) -> bool {
        self.reads.competent || self.force
    }

    /// The main fit's propensity, or adjustment step `step`'s when given;
    /// an error when it was not fitted: a step's before
    /// [`ThresholdFits::with_steps`], or a fit skipped as hopeless.
    fn propensity(&self, step: Option<usize>) -> anyhow::Result<&PairPropensity> {
        let e = match step {
            Some(s) => self.steps.as_ref().and_then(|v| v.get(s)),
            None => self.main.as_ref(),
        };
        e.ok_or_else(|| anyhow::anyhow!("the propensity (step {step:?}) is not fitted"))
    }
}

/// One cross-fitted fit of a pair on `design`, every competence threshold
/// at once, with its propensity from `fits` ([`ThresholdFits`]): the main one,
/// or adjustment step `step`'s; an error when that one is not fitted.
///
/// The thresholds give nested indicators `C_c`. One outcome model carries
/// every `C_c` and every `T · C_c`, and one propensity model every `C_c`;
/// The competent effect at threshold `c` is then the subset estimate over the receivers
/// competent at `c`. Each single-threshold model is nested in this one, so
/// each `m_t` stays correctly specified wherever its own was; what changes
/// is one fit instead of one per competence threshold.
///
/// The pair columns: own ligand, niche, every competence indicator, in both
/// models; and spillover and bleed (not under
/// [`Fit::NoSpill`]), in both models too, so the estimator is doubly robust
/// in them as well. Read off the response, they would tie the propensity to
/// one programme, so it is not refitted: the threshold's shared
/// exposure-side fit is bordered with these two columns
/// ([`propensity_bordered`]).
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
    let t = &x.t;
    let (ligand, receptor) = (slices(&fits.ligand_cols), slices(&fits.receptor_cols));
    let modifiers = slices(&fits.modifiers);
    // Spillover and bleed in both models, so the propensity of this
    // response balances them.
    let added: Vec<&[f64]> = match kind {
        Fit::NoSpill => Vec::new(),
        Fit::Main => vec![&r.spill, &r.bleed],
    };
    let e = fits
        .propensity(step)?
        .on(design, cf, t, (&ligand, &receptor), &added)?;
    let extra: Vec<&[f64]> = ligand.into_iter().chain(receptor).chain(added).collect();
    let inputs = PairInputs {
        t,
        y: &r.y,
        extra: &extra,
        modifiers: &modifiers,
    };
    overlap_terms_with(design, cf, &inputs, &e, &opts.nuisance)
}
