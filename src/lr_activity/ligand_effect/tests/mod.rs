mod adjustment;
mod averaging;
mod baselines;
mod benchmark;
mod covariates;
mod interface;
mod overlap;
mod pairs;
mod pseudobulk;
mod reporting;
mod simulate;
mod targets;

// * Shared helpers

use super::estimate::{estimate_pair_with, DecoyPool, EffectOpts, EffectRow, PairData};
use super::simulate::{calibrate_producers, Scenario, SimBase, SimRow, SimShared};
use super::tissue::Tissue;

/// One pair's rows without a decoy.
pub(super) fn estimate_pair(
    tissue: &Tissue,
    pair: &PairData<'_>,
    opts: &EffectOpts,
) -> anyhow::Result<Vec<EffectRow>> {
    estimate_pair_with(tissue, pair, &DecoyPool::default(), opts)
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
