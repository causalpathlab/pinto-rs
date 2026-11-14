mod adjustment;
mod averaging;
mod baselines;
mod benchmark;
mod clusters;
mod combining;
mod covariates;
mod interface;
mod marginals;
mod only;
mod overlap;
mod pairs;
#[cfg(feature = "view")]
mod receivers;
mod regions;
mod reporting;
mod simulate;
mod targets;

// * Shared helpers

use super::crossfit::{CrossFit, DesignView};
use super::dose::Exposure;
use super::effects::{effects_on, Effects, Sensitivity};
use super::estimate::estimate_pair;
use super::fits::{Rows, ThresholdFits};
use super::options::EffectOpts;
use super::overlap::{
    overlap_terms_with, propensity, FoldStarts, NuisanceOpts, OverlapTerms, PairInputs,
};
use super::regression::standardize;
use super::response::Response;
use super::simulate::{calibrate_producers, Scenario, SimBase, SimRow, SimShared};
use super::tissue::Tissue;

/// [`crate::lr_activity::ligand_effect::effects::effects_on`] with the
/// threshold's propensities fitted for it.
pub(super) fn fitted_effects(
    rows: Rows<'_>,
    x: &Exposure,
    r: &Response,
    levels: &[Vec<bool>],
    sens: Sensitivity,
    opts: &EffectOpts,
) -> anyhow::Result<Effects> {
    let mut starts = Default::default();
    let fits = ThresholdFits::new(rows, x, levels, opts, &mut starts)?
        .with_steps(rows, sens.steps, opts, &starts)?;
    effects_on(rows, &fits, r, sens, opts)
}

/// One benchmark run, the pair's shared ingredients computed for it alone.
pub(super) fn run_sim(
    tissue: &Tissue,
    base: &SimBase<'_>,
    scen: &Scenario,
    opts: &EffectOpts,
    seed: u64,
) -> anyhow::Result<SimRow> {
    let shared = SimShared {
        producers: calibrate_producers(tissue, base.ligand_counts),
        gene_fields: &[],
    };
    super::simulate::simulate(tissue, base, &shared, scen, opts, seed)
}

/// Cross-fit both nuisances on a (possibly nested) design and return the
/// per-row estimator pieces, `pair.extra` given on the raw scale. The
/// estimator itself fits the propensity once for every fit that shares it
/// (`propensity`, `overlap_terms_with`).
pub(super) fn overlap_terms(
    design: DesignView<'_>,
    cf: &CrossFit,
    pair: &PairInputs<'_>,
    opts: &NuisanceOpts,
) -> anyhow::Result<OverlapTerms> {
    let extra: Vec<Vec<f64>> = pair.extra.iter().map(|c| standardize(c)).collect();
    let extra: Vec<&[f64]> = extra.iter().map(|c| c.as_slice()).collect();
    let pair = PairInputs {
        extra: &extra,
        ..*pair
    };
    let e = propensity(design, cf, pair.t, &extra, opts, &mut FoldStarts::default())?.e;
    overlap_terms_with(design, cf, &pair, &e, opts)
}

/// The rows where `mask` holds, ascending.
pub(super) fn rows_of(mask: &[bool]) -> Vec<u32> {
    (0..mask.len() as u32)
        .filter(|&i| mask[i as usize])
        .collect()
}
