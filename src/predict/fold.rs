//! The state-aware fold a query is divided by, from the model's saved fold
//! ([`SavedCellFold`]): `pinto apply predict` of a cage model and
//! `pinto apply impute`.
//!
//! A query cell from a batch the model knows is placed on the node whose
//! composition, in that batch, explains its counts best (data-beans'
//! `place_by_composition`; summed with up to [`PLACEMENT_NEIGHBOURS`]
//! spatial neighbours when a graph is given), then divided by that node's
//! fold in that batch, δ included: the table the model's own cells were
//! divided by for its profiles. (cage's encoder was trained on δ-divided
//! rows, so predict feeds it δ, not this.) A cell from a batch the model never saw, or with no read on
//! the model's features, keeps its column of the δ predict divides by today
//! (the model's δ, or the query's own estimate moved into its frame).

use crate::util::batch_fold::SavedCellFold;
use crate::util::common::*;

/// Spatial neighbours a query cell's counts are summed with before it is
/// placed. On a held-out section (sim donor-confounded, lc fit on the other
/// five), placement agreed with the planted domain at ARI 0.475 alone, 0.673
/// with 4 neighbours, 0.699 with 12; 4 smooths less across domain edges.
pub(crate) const PLACEMENT_NEIGHBOURS: usize = 4;

/// What divides each query cell: `table` (the query's rows × columns,
/// linear) and each cell's column of it.
pub(crate) struct QueryFoldDivisor {
    pub table: Mat,
    pub col_of_cell: Vec<u32>,
    /// Cells divided by a placed node's fold (the rest keep δ).
    pub n_placed: usize,
}

/// The query's batches as the model's fold names them.
pub(crate) struct QueryBatches<'a> {
    /// Each cell's query batch (a column of `delta`).
    pub of_cell: &'a [u32],
    /// Each query batch's label, as the model's batches are named when it
    /// is one of them (`None`: new to the model).
    pub model_label: &'a [Option<Box<str>>],
}

/// The divisor of the query `data` (rows `rows`): a placed node's fold for
/// each cell of a batch the model knows, else the cell's column of `delta`
/// (rows × query batches, linear; the δ predict divides by without a fold).
/// `neighbours`: the query's spatial pairs, when a graph was built.
pub(crate) fn query_fold_divisor(
    saved: &SavedCellFold,
    data: &SparseIoVec,
    rows: &[Box<str>],
    delta: &Mat,
    batches: &QueryBatches<'_>,
    neighbours: Option<&[(u32, u32)]>,
) -> anyhow::Result<QueryFoldDivisor> {
    let n_cells = data.num_columns();
    let nb = saved.batches.len();
    anyhow::ensure!(
        batches.of_cell.len() == n_cells && delta.nrows() == rows.len(),
        "query fold: one batch per cell and one δ row per data row"
    );
    // each query row's feature in the fold, and each query batch's fold batch
    let feature_of: HashMap<&str, usize> = saved
        .features
        .iter()
        .enumerate()
        .map(|(g, f)| (f.as_ref(), g))
        .collect();
    let fold_row: Vec<Option<usize>> = rows
        .iter()
        .map(|r| feature_of.get(r.as_ref()).copied())
        .collect();
    let fold_batch: Vec<Option<usize>> = batches
        .model_label
        .iter()
        .map(|l| {
            l.as_ref()
                .and_then(|l| saved.batches.iter().position(|b| b == l))
        })
        .collect();

    // each cell's counts on the fold's features; a new batch's divided by its δ
    let mut counts: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_cells];
    for (lb, ub) in
        legume_numeric::matrix::utils::generate_minibatch_intervals(n_cells, rows.len(), None)
    {
        let block = data.read_columns_csc(lb..ub)?;
        for (c, col) in (lb..ub).zip(block.col_iter()) {
            let qb = batches.of_cell[c] as usize;
            let known = fold_batch[qb].is_some();
            for (&r, &x) in col.row_indices().iter().zip(col.values()) {
                if let Some(g) = fold_row[r] {
                    let x = if known {
                        x
                    } else {
                        x / delta[(r, qb)].max(1e-8)
                    };
                    counts[c].push((g, x));
                }
            }
        }
    }
    let counts = match neighbours {
        Some(pairs) => summed_with_neighbours(&counts, pairs),
        None => counts,
    };
    let batch_of_cell: Vec<Option<usize>> = batches
        .of_cell
        .iter()
        .map(|&qb| fold_batch[qb as usize])
        .collect();
    let node = data_beans::alg::collapse_data::state_aware_batch_fold::place_by_composition(
        &counts,
        &batch_of_cell,
        &saved.log_mean_composition,
        &saved.log_fold,
        nb,
    );

    // the fold's columns on the query's rows (1 where the model has no such
    // row), then δ's
    let n_fold = saved.log_fold.ncols();
    let table = Mat::from_fn(rows.len(), n_fold + delta.ncols(), |r, c| {
        if c < n_fold {
            fold_row[r].map_or(1.0, |g| saved.log_fold[(g, c)].exp())
        } else {
            delta[(r, c - n_fold)]
        }
    });
    let mut n_placed = 0;
    let col_of_cell = (0..n_cells)
        .map(|c| match (node[c], batch_of_cell[c]) {
            (Some(k), Some(b)) => {
                n_placed += 1;
                (k * nb + b) as u32
            }
            _ => (n_fold + batches.of_cell[c] as usize) as u32,
        })
        .collect();
    let mode = if neighbours.is_some() {
        format!("each with up to {PLACEMENT_NEIGHBOURS} spatial neighbours")
    } else {
        "each on its own counts".to_string()
    };
    info!(
        "Batch fold: {n_placed} of {n_cells} query cells placed on the model's {} nodes \
         ({mode}); the rest (a batch new to the model, or no read) keep δ",
        saved.log_mean_composition.ncols()
    );
    Ok(QueryFoldDivisor {
        table,
        col_of_cell,
        n_placed,
    })
}

/// Each cell's counts summed with those of up to [`PLACEMENT_NEIGHBOURS`] of
/// its neighbours in `pairs` (as they are listed).
fn summed_with_neighbours(
    counts: &[Vec<(usize, f32)>],
    pairs: &[(u32, u32)],
) -> Vec<Vec<(usize, f32)>> {
    let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); counts.len()];
    for &(u, v) in pairs {
        let (u, v) = (u as usize, v as usize);
        if nbrs[u].len() < PLACEMENT_NEIGHBOURS {
            nbrs[u].push(v);
        }
        if nbrs[v].len() < PLACEMENT_NEIGHBOURS {
            nbrs[v].push(u);
        }
    }
    (0..counts.len())
        .map(|c| {
            let mut all: Vec<(usize, f32)> = counts[c]
                .iter()
                .chain(nbrs[c].iter().flat_map(|&n| counts[n].iter()))
                .copied()
                .collect();
            all.sort_unstable_by_key(|&(g, _)| g);
            all.dedup_by(|next, kept| {
                if next.0 == kept.0 {
                    kept.1 += next.1;
                    true
                } else {
                    false
                }
            });
            all
        })
        .collect()
}

/// What divides the model's own cells (impute's reference): the fold column
/// of each fitted cell's saved node in its batch (`batch_label_of_cell`,
/// as the model named it), on rows `rows`, beside `model_delta` (rows ×
/// `model_delta_batches`, linear) for a cell the fit divided by δ alone.
/// Each cell's column; `None` when its batch is not the model's.
pub(crate) fn fitted_cells_fold_divisor(
    saved: &SavedCellFold,
    rows: &[Box<str>],
    cells: &[Box<str>],
    batch_label_of_cell: &[Box<str>],
    (model_delta, model_delta_batches): (&Mat, &[Box<str>]),
) -> (Mat, Vec<Option<usize>>) {
    let nb = saved.batches.len();
    let n_fold = saved.log_fold.ncols();
    let feature_of: HashMap<&str, usize> = saved
        .features
        .iter()
        .enumerate()
        .map(|(g, f)| (f.as_ref(), g))
        .collect();
    let table = Mat::from_fn(rows.len(), n_fold + model_delta.ncols(), |r, c| {
        if c < n_fold {
            feature_of
                .get(rows[r].as_ref())
                .map_or(1.0, |&g| saved.log_fold[(g, c)].exp())
        } else {
            model_delta[(r, c - n_fold)]
        }
    });
    let col = cells
        .iter()
        .zip(batch_label_of_cell)
        .map(|(cell, label)| {
            let node = saved.node_of_cell.get(cell).copied().flatten();
            let fold_b = saved.batches.iter().position(|b| b == label);
            match (node, fold_b) {
                (Some(k), Some(b)) => Some(k * nb + b),
                _ => model_delta_batches
                    .iter()
                    .position(|b| b == label)
                    .map(|b| n_fold + b),
            }
        })
        .collect();
    (table, col)
}
