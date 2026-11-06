mod adjustment;
mod averaging;
mod baselines;
mod benchmark;
mod combining;
mod covariates;
mod interface;
mod only;
mod overlap;
mod pairs;
#[cfg(feature = "view")]
mod receivers;
mod reporting;
mod simulate;
mod targets;

// * Shared helpers

use super::estimate::{
    effects_on, estimate_pair_with, BackgroundPool, EffectOpts, EffectRow, Effects, Exposure,
    PairData, Response, Rows, Sensitivity, ThresholdFits,
};
use super::simulate::{calibrate_producers, Scenario, SimBase, SimRow, SimShared};
use super::tissue::Tissue;

/// One pair's rows without a background.
pub(super) fn estimate_pair(
    tissue: &Tissue,
    pair: &PairData<'_>,
    opts: &EffectOpts,
) -> anyhow::Result<Vec<EffectRow>> {
    estimate_pair_with(tissue, pair, &BackgroundPool::default(), opts)
}

/// [`effects_on`] with the threshold's propensities fitted for it.
pub(super) fn fitted_effects(
    rows: Rows<'_>,
    x: &Exposure,
    r: &Response,
    levels: &[Vec<bool>],
    sens: Sensitivity,
    opts: &EffectOpts,
) -> anyhow::Result<Effects> {
    let mut starts = crate::lr_activity::ligand_effect::estimate::Starts::default();
    let fits = ThresholdFits::new(rows, x, levels, opts, &mut starts.main)?.with_controls(
        rows,
        sens.steps,
        opts,
        &mut starts,
    )?;
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

/// The rows where `mask` holds, ascending.
pub(super) fn rows_of(mask: &[bool]) -> Vec<u32> {
    (0..mask.len() as u32)
        .filter(|&i| mask[i as usize])
        .collect()
}
