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
use data_beans::alg::cell_pairs::split_reads_over_pair_units;
use data_beans::alg::collapse_data::group_fold::{
    estimate_pair_fold, estimate_sample_fold, GroupFold, GroupFoldParams, SampleFold,
};
use data_beans::alg::collapse_data::*;
use data_beans::alg::random_projection::*;
use data_beans::sparse_io::{create_sparse_from_triplets_owned, SparseIoBackend};
use legume_numeric::param::traits::Inference;
use std::sync::Arc;

/// A fold per unit: `fold[:, col_of_unit[u]]` multiplies unit `u`'s rates.
pub struct FoldTable {
    /// `[rows × columns]`: one column per (node, batch), then one per batch
    /// (δ, for units without a node).
    pub fold: Mat,
    pub col_of_unit: Vec<u32>,
}

impl FoldTable {
    fn of(gf: &GroupFold, batch_of_unit: impl Fn(usize) -> usize) -> Self {
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

    /// Unit `u`'s fold over the rows.
    pub fn unit_fold(&self, u: usize) -> nalgebra::DVectorView<'_, f32> {
        self.fold.column(self.col_of_unit[u] as usize)
    }
}

/// Each cell's own fold (samples kind), and the fit that placed the cells,
/// which the pair fold builds on.
pub struct CellFold {
    pub table: FoldTable,
    fit: SampleFold,
}

/// Each listed pair's own fold (graph kind), in the order of the pairs it was
/// estimated on.
pub struct PairFold {
    pub table: FoldTable,
}

impl PairFold {
    /// Each super edge's fold, `[rows × n_super]`: the geometric mean of its
    /// pairs' folds (`fine_to_super[e]`, the super edge of pair `e`; one out
    /// of range: in none). A super edge with no pair takes fold 1.
    pub fn super_edge_folds(&self, fine_to_super: &[usize], n_super: usize) -> Mat {
        let rows = self.table.fold.nrows();
        let mut log_sum = Mat::zeros(rows, n_super);
        let mut count = vec![0f32; n_super];
        for (e, &s) in fine_to_super.iter().enumerate() {
            if s < n_super {
                let f = self.table.unit_fold(e);
                for r in 0..rows {
                    log_sum[(r, s)] += f[r].ln();
                }
                count[s] += 1.0;
            }
        }
        for (s, &c) in count.iter().enumerate() {
            let scale = if c > 0.0 { 1.0 / c } else { 0.0 };
            log_sum.column_mut(s).apply(|v| *v = (*v * scale).exp());
        }
        log_sum
    }
}

/// The fold's settings: data-beans' defaults; no user-facing knob.
fn params() -> GroupFoldParams {
    GroupFoldParams::default()
}

/// Each cell's fold over the collapse that gave δ: its coarsest groups
/// (`coarse_groups`), δ's mean (`delta`, `[rows × batches]`) and the
/// batch-corrected projection it collapsed on (`proj`, `[dims × cells]`).
pub fn estimate_cell_fold(
    data_vec: &SparseIoVec,
    batch_of_cell: &[u32],
    delta: &Mat,
    coarse_groups: &[usize],
    proj: &Mat,
) -> anyhow::Result<CellFold> {
    let batch: Vec<usize> = batch_of_cell.iter().map(|&b| b as usize).collect();
    let log_delta = delta.map(|v| v.max(1e-8).ln());
    let fit = estimate_sample_fold(data_vec, &batch, coarse_groups, &log_delta, proj, &params())?;
    let table = FoldTable::of(&fit.fold, |c| batch[c]);
    Ok(CellFold { table, fit })
}

/// Each pair's fold (graph kind): the reads of every cell split once over its
/// pairs and a self-loop, those pair units collapsed for their own δ (the
/// self-loops a stratum of their own), then the fold per pair node placed by
/// the pair's endpoints in the cells' tree (`cells`).
///
/// `pairs` are cell indices into `data_vec`; a pair's batch is its first
/// endpoint's. The pair units live in a temporary file for the estimate only.
pub fn estimate_pair_batch_fold(
    data_vec: &SparseIoVec,
    pairs: &[(usize, usize)],
    batch_of_cell: &[u32],
    batch_labels: &[Box<str>],
    cells: &CellFold,
    proj_dim: usize,
    num_levels: usize,
) -> anyhow::Result<PairFold> {
    let (n_rows, n_cells, n_pairs) = (data_vec.num_rows(), data_vec.num_columns(), pairs.len());
    info!("Pair batch fold: splitting reads over {n_pairs} pairs and {n_cells} self-loops");
    let triplets = split_reads_over_pair_units(data_vec, pairs, None)?;
    let n_units = n_pairs + n_cells;
    let nnz = triplets.len();
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("pair_units.zarr.zip");
    let mut backend = create_sparse_from_triplets_owned(
        triplets,
        (n_rows, n_units, nnz),
        Some(path.to_str().expect("utf-8 temporary path")),
        Some(&SparseIoBackend::Zarr),
    )?;
    backend.register_row_names_vec(&data_vec.row_names()?);
    backend.register_column_names_vec(
        &(0..n_units)
            .map(|u| format!("unit{u}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut units = SparseIoVec::new();
    units.push(Arc::from(backend), None)?;

    let unit_batch: Vec<u32> = (0..n_units)
        .map(|u| {
            if u < n_pairs {
                batch_of_cell[pairs[u].0]
            } else {
                batch_of_cell[u - n_pairs]
            }
        })
        .collect();
    let unit_label: Vec<Box<str>> = unit_batch
        .iter()
        .map(|&b| batch_labels[b as usize].clone())
        .collect();
    let strata: Vec<usize> = (0..n_units).map(|u| usize::from(u >= n_pairs)).collect();

    let proj = units.project_columns_with_batch_correction(proj_dim, None, Some(&unit_label))?;
    let out = collapse_columns_multilevel_with_hierarchy(
        &mut units,
        &proj.proj,
        &unit_label,
        &MultilevelParams {
            knn_pb_samples: crate::util::input::BATCH_KNN,
            sort_dim: proj_dim.min(crate::util::batch_effects::BATCH_SORT_DIM),
            num_levels,
            strata: Some(strata),
            ..MultilevelParams::new(proj.proj.nrows())
        },
    )?;
    let pair_delta = out
        .levels
        .first()
        .and_then(|l| l.delta.as_ref())
        .ok_or_else(|| anyhow::anyhow!("the pair-unit collapse gave no batch effect"))?
        .posterior_mean()
        .map(|v| v.max(1e-8).ln());
    anyhow::ensure!(
        pair_delta.ncols() == batch_labels.len(),
        "pair units span {} of {} batches",
        pair_delta.ncols(),
        batch_labels.len()
    );

    let cell_batch: Vec<usize> = batch_of_cell.iter().map(|&b| b as usize).collect();
    let fold = estimate_pair_fold(
        &units,
        pairs,
        &cell_batch,
        &cells.fit,
        &pair_delta,
        &params(),
    )?;
    let table = FoldTable::of(&fold.fold, |e| cell_batch[pairs[e].0]);
    Ok(PairFold { table })
}

/// Columns of `y`, column `j` an endpoint of pair `first + j`, divided by that
/// pair's fold with the scale-matched division (the fold's mean over the
/// column's nonzeros is matched to theirs: `x / (d · λ)`,
/// `λ = Σ x / Σ d`), then `log1p`. No library-size rescale: it would cancel
/// the fold.
pub fn divide_by_pair_folds_log1p(
    y: &mut nalgebra_sparse::CscMatrix<f32>,
    table: &FoldTable,
    first: usize,
) {
    for (j, mut col) in y.col_iter_mut().enumerate() {
        let f = table.unit_fold(first + j);
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
