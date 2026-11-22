//! The state-aware batch fold (data-beans' `group_fold`), as tables pinto
//! divides by.
//!
//! δ is one factor per (feature × batch); the fold adds the part of a batch
//! effect that depends on the cell state, one factor per (feature × node ×
//! batch). Both kinds are read through the same [`BatchDivisor`] pinto
//! already divides by: a table of folds and each unit's column in it.
//!
//! * [`CellFold`] (samples kind): each cell's own fold. The baselines and
//!   anything pooled from cells (their pseudobulks) divide by it.
//! * [`PairFold`] (graph kind): each cell pair's own fold, estimated on pair
//!   units (every cell's reads split once over its pairs and a self-loop).
//!   lc, dsvd and cage divide a pair's pooled counts by it; a super edge
//!   takes the mean log fold of its pairs ([`PairFold::super_edge_folds`]).

use crate::util::batch_effects::BatchDivisor;
use crate::util::common::*;
use data_beans::alg::collapse_data::state_aware_batch_fold::{
    collapse_cell_pairs_and_estimate_state_aware_batch_fold,
    estimate_state_aware_batch_fold_over_samples, CollapsePseudobulks, SampleBatchFold,
    StateAwareBatchFold, StateAwareBatchFoldParams,
};
use data_beans::alg::collapse_data::MultilevelParams;

/// A fold per unit: `fold[:, col_of_unit[u]]` multiplies unit `u`'s rates.
pub struct FoldTable {
    /// `[rows × columns]`: one column per (node, batch), then one per batch
    /// (δ, for units without a node).
    pub fold: Mat,
    pub col_of_unit: Vec<u32>,
}

impl FoldTable {
    fn of(gf: &StateAwareBatchFold, batch_of_unit: impl Fn(usize) -> usize) -> Self {
        let nb = gf.n_batches;
        let n_node_cols = gf.n_nodes * nb;
        let mut fold = Mat::zeros(gf.log_fold.nrows(), n_node_cols + nb);
        fold.columns_mut(0, n_node_cols).copy_from(&gf.log_fold);
        fold.columns_mut(n_node_cols, nb).copy_from(&gf.log_delta);
        fold.apply(|v| *v = v.exp());
        let col_of_unit = gf
            .node_of_unit
            .iter()
            .enumerate()
            .map(|(u, node)| {
                let b = batch_of_unit(u);
                match node {
                    Some(k) => (k * nb + b) as u32,
                    None => (n_node_cols + b) as u32,
                }
            })
            .collect();
        Self { fold, col_of_unit }
    }

    /// The divisor pinto's readers take ([`BatchDivisor`]): each unit's column.
    pub fn divisor(&self) -> BatchDivisor<'_> {
        BatchDivisor {
            delta: &self.fold,
            batch_of_cell: &self.col_of_unit,
        }
    }
}

/// Each cell's own fold (samples kind), and the fit that placed the cells,
/// which the pair fold builds on.
pub struct CellFold {
    pub table: FoldTable,
    fit: SampleBatchFold,
    batch_of_cell: Vec<usize>,
}

impl CellFold {
    fn batch_of_cell(&self) -> &[usize] {
        &self.batch_of_cell
    }
}

/// Each listed pair's fold per endpoint (graph kind), in the order of the
/// pairs it was estimated on: a within-batch pair's two endpoints both take
/// the pair's own fold; a pair across batches has none, and each endpoint
/// takes its own cell's fold.
pub struct PairFold {
    /// `[rows × columns]`: the pair folds' columns, then the cell folds'.
    pub fold: Mat,
    /// Each pair's (first endpoint's, second endpoint's) column of `fold`.
    pub endpoint_cols: Vec<(u32, u32)>,
}

impl PairFold {
    /// Pair `e`'s fold for its first and its second endpoint.
    pub fn endpoint_folds(
        &self,
        e: usize,
    ) -> (
        nalgebra::DVectorView<'_, f32>,
        nalgebra::DVectorView<'_, f32>,
    ) {
        let (a, b) = self.endpoint_cols[e];
        (self.fold.column(a as usize), self.fold.column(b as usize))
    }
}

/// The fold's settings: data-beans' defaults; no user-facing knob.
fn params() -> StateAwareBatchFoldParams {
    StateAwareBatchFoldParams::default()
}

/// Each cell's fold over the collapse that gave δ: its finest groups (the
/// pseudobulks matched) and coarsest groups (the tree's leaves), δ's mean
/// (`delta`, `[rows × batches]`) and the batch-corrected projection it
/// collapsed on (`proj`, `[dims × cells]`).
pub fn estimate_cell_fold(
    data_vec: &SparseIoVec,
    batch_of_cell: &[u32],
    delta: &Mat,
    groups: &crate::util::batch_effects::CollapseGroups,
    proj: &Mat,
) -> anyhow::Result<CellFold> {
    let batch: Vec<usize> = batch_of_cell.iter().map(|&b| b as usize).collect();
    let log_delta = delta.map(|v| v.max(1e-8).ln());
    let fit = estimate_state_aware_batch_fold_over_samples(
        data_vec,
        &batch,
        &CollapsePseudobulks {
            finest: &groups.finest,
            coarsest: &groups.coarsest,
        },
        &log_delta,
        proj,
        &params(),
    )?;
    let table = FoldTable::of(&fit.fold, |c| batch[c]);
    Ok(CellFold {
        table,
        fit,
        batch_of_cell: batch,
    })
}

/// Each pair's fold (cell-pair kind), by data-beans'
/// [`collapse_cell_pairs_and_estimate_state_aware_batch_fold`]: every cell's
/// reads split once over its pairs and a self-loop, those pair units
/// collapsed for their own δ, each pair placed on the cells' tree (`cells`)
/// by its endpoints. `pairs` are cell indices into `data_vec`.
pub fn estimate_pair_batch_fold(
    data_vec: &SparseIoVec,
    pairs: &[(usize, usize)],
    batch_membership: &[Box<str>],
    cells: &CellFold,
    proj_dim: usize,
    num_levels: usize,
) -> anyhow::Result<PairFold> {
    let collapse = MultilevelParams {
        knn_pb_samples: crate::util::input::BATCH_KNN,
        sort_dim: proj_dim.min(crate::util::batch_effects::BATCH_SORT_DIM),
        num_levels,
        ..MultilevelParams::new(proj_dim)
    };
    let fold = collapse_cell_pairs_and_estimate_state_aware_batch_fold(
        data_vec,
        pairs,
        batch_membership,
        &cells.fit,
        &collapse,
        proj_dim,
        &params(),
    )?;
    let first_cell: Vec<usize> = pairs.iter().map(|&(u, _)| u).collect();
    let batch_of_cell = cells.batch_of_cell();
    let pair_table = FoldTable::of(&fold.fold, |e| batch_of_cell[first_cell[e]]);
    // the pair folds' columns, then the cells'; a pair without a node (across
    // batches) divides each endpoint by its own cell's fold
    let n_pair_cols = pair_table.fold.ncols();
    let mut table = Mat::zeros(
        pair_table.fold.nrows(),
        n_pair_cols + cells.table.fold.ncols(),
    );
    table
        .columns_mut(0, n_pair_cols)
        .copy_from(&pair_table.fold);
    table
        .columns_mut(n_pair_cols, cells.table.fold.ncols())
        .copy_from(&cells.table.fold);
    let endpoint_cols = pairs
        .iter()
        .enumerate()
        .map(|(e, &(u, v))| match fold.fold.node_of_unit[e] {
            Some(_) => (pair_table.col_of_unit[e], pair_table.col_of_unit[e]),
            None => {
                let cell = |c: usize| n_pair_cols as u32 + cells.table.col_of_unit[c];
                (cell(u), cell(v))
            }
        })
        .collect();
    Ok(PairFold {
        fold: table,
        endpoint_cols,
    })
}

/// Columns of `y`, column `j` one endpoint of pair `first + j` (`second`:
/// the pairs' second endpoints), each divided by that endpoint's fold with
/// the scale-matched division (the fold's mean over the column's nonzeros is
/// matched to theirs: `x / (d · λ)`, `λ = Σ x / Σ d`), then `log1p`.
pub fn divide_by_pair_folds_log1p(
    y: &mut nalgebra_sparse::CscMatrix<f32>,
    pair_fold: &PairFold,
    first: usize,
    second: bool,
) {
    for (j, mut col) in y.col_iter_mut().enumerate() {
        let (fu, fv) = pair_fold.endpoint_folds(first + j);
        let f = if second { fv } else { fu };
        let (rows, vals) = col.rows_and_values_mut();
        let sum_x: f32 = vals.iter().sum();
        let sum_d: f32 = rows.iter().map(|&r| f[r]).sum();
        let lambda = if sum_d > 0.0 { sum_x / sum_d } else { 0.0 };
        for (&r, v) in rows.iter().zip(vals.iter_mut()) {
            let d = f[r] * lambda;
            *v = if d > 0.0 { (*v / d).ln_1p() } else { 0.0 };
        }
    }
}
