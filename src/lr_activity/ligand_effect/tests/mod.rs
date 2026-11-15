mod adjustment;
mod averaging;
mod baselines;
mod benchmark;
mod clusters;
mod combining;
mod covariates;
mod interface;
mod marginal_run;
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
use crate::lr_activity::input::UNASSIGNED;

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

/// Cell coordinates and contacts.
pub(super) type Lattice = (Vec<Option<(f32, f32)>>, Vec<(usize, usize)>);

/// A `side` by `side` grid of cells 10 apart, cell `i` in column `i % side`
/// and row `i / side`, and its four-neighbour contacts `(i, i + 1)` and
/// `(i, i + side)`, leaving out those where `cut` holds.
pub(super) fn lattice_cut(side: usize, cut: impl Fn(usize, usize) -> bool) -> Lattice {
    let n = side * side;
    let coords = (0..n)
        .map(|i| Some(((i % side) as f32 * 10.0, (i / side) as f32 * 10.0)))
        .collect();
    let mut pairs = Vec::new();
    for i in 0..n {
        if i % side + 1 < side && !cut(i, i + 1) {
            pairs.push((i, i + 1));
        }
        if i / side + 1 < side && !cut(i, i + side) {
            pairs.push((i, i + side));
        }
    }
    (coords, pairs)
}

/// [`lattice_cut`] with every contact kept.
pub(super) fn lattice(side: usize) -> Lattice {
    lattice_cut(side, |_, _| false)
}

/// The contacts of `copies` [`lattice`]s numbered one after another, none
/// between them.
pub(super) fn lattice_contacts(side: usize, copies: usize) -> Vec<(usize, usize)> {
    let (_, pairs) = lattice(side);
    let per = side * side;
    (0..copies)
        .flat_map(|s| pairs.iter().map(move |&(i, j)| (i + s * per, j + s * per)))
        .collect()
}

/// Three named cell types and the unassigned.
pub(super) fn type_names() -> Vec<Box<str>> {
    ["CT1", "CT2", "CT3", UNASSIGNED].map(Into::into).to_vec()
}

/// The rows where `mask` holds, ascending.
pub(super) fn rows_of(mask: &[bool]) -> Vec<u32> {
    (0..mask.len() as u32)
        .filter(|&i| mask[i as usize])
        .collect()
}
