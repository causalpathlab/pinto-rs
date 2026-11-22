//! The state-aware fold a query is divided by, from the model's saved fold
//! ([`SavedCellFold`]): `pinto apply predict` of a cage model and
//! `pinto apply impute`.
//!
//! A query cell from a batch the model knows takes the node the fit gave it
//! when the fit saw it under the same name (read from several files, as the
//! fit read them; one file names cells by barcode alone, so they are placed
//! again); otherwise it is placed on the node whose
//! composition, in that batch, explains its counts best (data-beans'
//! `place_by_composition`; summed with its up to [`PLACEMENT_NEIGHBOURS`]
//! nearest spatial neighbours in its batch when a graph is given). It is
//! then divided by that node's fold in that batch, δ included: the table
//! the model's own cells were divided by for its profiles. (cage's encoder
//! was trained on δ-divided rows, so predict feeds it δ, not this.) A cell
//! from a batch the model never saw, with no read on the model's features,
//! or kept on δ by the fit, keeps its column of the δ predict divides by
//! today (the model's δ, or the query's own estimate moved into its frame);
//! so do a placed cell's rows the fold has no feature for.

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
    /// Of those, cells the fit itself placed (by name), not placed again.
    pub n_fitted: usize,
}

/// The query's batches as the model's fold names them.
pub(crate) struct QueryBatches<'a> {
    /// Each cell's query batch (a column of `delta`).
    pub of_cell: &'a [u32],
    /// Each query batch's label, as the model's batches are named when it
    /// is one of them (`None`: new to the model).
    pub model_label: &'a [Option<Box<str>>],
}

/// The query cells: their names (a cell the fit placed keeps its node) and
/// each one's spatial neighbours to sum before placing it
/// ([`placement_neighbours`]; `None`: each on its own counts).
pub(crate) struct QueryCells<'a> {
    pub names: &'a [Box<str>],
    pub neighbours: Option<&'a [Vec<usize>]>,
}

/// Each cell's nearest neighbours to sum before it is placed: up to
/// [`PLACEMENT_NEIGHBOURS`] of its spatial pairs (`edge_kind`: none, or
/// spatial), in its own query batch, nearest first by `coordinates` (cells ×
/// dims).
pub(crate) fn placement_neighbours(
    pairs: &[(u32, u32)],
    edge_kind: Option<&[i32]>,
    coordinates: &Mat,
    batch_of_cell: &[u32],
) -> Vec<Vec<usize>> {
    let n = batch_of_cell.len();
    let dist = |u: usize, v: usize| (coordinates.row(u) - coordinates.row(v)).norm_squared();
    let mut cand: Vec<Vec<(f32, usize)>> = vec![Vec::new(); n];
    for (e, &(u, v)) in pairs.iter().enumerate() {
        let spatial = edge_kind.is_none_or(|k| k[e] == crate::util::cell_pairs::EDGE_KIND_SPATIAL);
        let (u, v) = (u as usize, v as usize);
        if spatial && u != v && batch_of_cell[u] == batch_of_cell[v] {
            let d = dist(u, v);
            cand[u].push((d, v));
            cand[v].push((d, u));
        }
    }
    cand.into_iter()
        .map(|mut c| {
            c.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            c.dedup_by_key(|x| x.1);
            c.into_iter()
                .take(PLACEMENT_NEIGHBOURS)
                .map(|(_, v)| v)
                .collect()
        })
        .collect()
}

/// The fold's columns on rows `rows` (each row's feature in the fold:
/// `fold_row`), then `delta`'s. A row the fold lacks is divided by δ of the
/// batch its column is in (`delta_col_of_fold_batch`; 1 when no δ column is
/// in that batch), so it reads as it would without a fold.
fn fold_table_with_delta(
    saved: &SavedCellFold,
    fold_row: &[Option<usize>],
    delta: &Mat,
    delta_col_of_fold_batch: &[Option<usize>],
) -> Mat {
    let nb = saved.batches.len();
    let n_fold = saved.log_fold.ncols();
    Mat::from_fn(fold_row.len(), n_fold + delta.ncols(), |r, c| {
        if c >= n_fold {
            return delta[(r, c - n_fold)];
        }
        match fold_row[r] {
            Some(g) => saved.log_fold[(g, c)].exp(),
            None => delta_col_of_fold_batch[c % nb].map_or(1.0, |q| delta[(r, q)]),
        }
    })
}

/// Each row's feature in the fold.
fn fold_rows(saved: &SavedCellFold, rows: &[Box<str>]) -> Vec<Option<usize>> {
    let feature_of: HashMap<&str, usize> = saved
        .features
        .iter()
        .enumerate()
        .map(|(g, f)| (f.as_ref(), g))
        .collect();
    rows.iter()
        .map(|r| feature_of.get(r.as_ref()).copied())
        .collect()
}

/// The divisor of the query `data` (rows `rows`): a node's fold for each
/// cell of a batch the model knows (the fit's own node for a cell it placed,
/// else one placed by composition), else the cell's column of `delta` (rows
/// × query batches, linear; the δ predict divides by without a fold). The
/// query is read block by block, each block with its cells' neighbours.
pub(crate) fn query_fold_divisor(
    saved: &SavedCellFold,
    data: &SparseIoVec,
    rows: &[Box<str>],
    delta: &Mat,
    batches: &QueryBatches<'_>,
    cells: &QueryCells<'_>,
) -> anyhow::Result<QueryFoldDivisor> {
    let n_cells = data.num_columns();
    let nb = saved.batches.len();
    anyhow::ensure!(
        batches.of_cell.len() == n_cells
            && cells.names.len() == n_cells
            && cells.neighbours.is_none_or(|n| n.len() == n_cells)
            && delta.nrows() == rows.len(),
        "query fold: one batch, name and neighbour list per cell, one δ row per data row"
    );
    let fold_row = fold_rows(saved, rows);
    let fold_batch: Vec<Option<usize>> = batches
        .model_label
        .iter()
        .map(|l| {
            l.as_ref()
                .and_then(|l| saved.batches.iter().position(|b| b == l))
        })
        .collect();
    let batch_of_cell: Vec<Option<usize>> = batches
        .of_cell
        .iter()
        .map(|&qb| fold_batch[qb as usize])
        .collect();

    // a cell the fit placed keeps the fit's node (or δ, when the fit kept δ)
    let mut node: Vec<Option<usize>> = vec![None; n_cells];
    let mut to_place: Vec<bool> = vec![false; n_cells];
    let mut n_fitted = 0;
    for c in 0..n_cells {
        if batch_of_cell[c].is_none() {
            continue;
        }
        match saved.node_of_cell.get(&cells.names[c]) {
            Some(&k) => {
                node[c] = k;
                n_fitted += usize::from(k.is_some());
            }
            None => to_place[c] = true,
        }
    }

    // the rest placed by composition, block by block, each cell on its
    // counts (a new batch's divided by its δ) summed with its neighbours'
    let cell_counts = |c: usize, col: nalgebra_sparse::csc::CscCol<'_, f32>| {
        let qb = batches.of_cell[c] as usize;
        let known = fold_batch[qb].is_some();
        col.row_indices()
            .iter()
            .zip(col.values())
            .filter_map(|(&r, &x)| {
                let g = fold_row[r]?;
                Some((
                    g,
                    if known {
                        x
                    } else {
                        x / delta[(r, qb)].max(1e-8)
                    },
                ))
            })
            .collect::<Vec<(usize, f32)>>()
    };
    for (lb, ub) in
        legume_numeric::matrix::utils::generate_minibatch_intervals(n_cells, rows.len(), None)
    {
        let placing: Vec<usize> = (lb..ub).filter(|&c| to_place[c]).collect();
        if placing.is_empty() {
            continue;
        }
        let mut read: Vec<usize> = placing.clone();
        if let Some(nbrs) = cells.neighbours {
            read.extend(placing.iter().flat_map(|&c| nbrs[c].iter().copied()));
        }
        read.sort_unstable();
        read.dedup();
        let block = data.read_columns_csc(read.iter().copied())?;
        let counts_of: HashMap<usize, Vec<(usize, f32)>> = read
            .iter()
            .zip(block.col_iter())
            .map(|(&c, col)| (c, cell_counts(c, col)))
            .collect();
        let summed: Vec<Vec<(usize, f32)>> = placing
            .iter()
            .map(|&c| {
                let nbrs = cells.neighbours.map_or(&[][..], |n| &n[c][..]);
                summed_counts(std::iter::once(c).chain(nbrs.iter().copied()), &counts_of)
            })
            .collect();
        let batch: Vec<Option<usize>> = placing.iter().map(|&c| batch_of_cell[c]).collect();
        let placed = data_beans::alg::collapse_data::state_aware_batch_fold::place_by_composition(
            &summed,
            &batch,
            &saved.log_mean_composition,
            &saved.log_fold,
            nb,
        )?;
        for (&c, k) in placing.iter().zip(placed) {
            node[c] = k;
        }
    }

    // a fold batch's δ column: the query batch the model names so
    let delta_col_of_fold_batch: Vec<Option<usize>> = (0..nb)
        .map(|b| fold_batch.iter().position(|&f| f == Some(b)))
        .collect();
    let table = fold_table_with_delta(saved, &fold_row, delta, &delta_col_of_fold_batch);
    let n_fold = saved.log_fold.ncols();
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
    let mode = if cells.neighbours.is_some() {
        format!("each with up to {PLACEMENT_NEIGHBOURS} spatial neighbours in its batch")
    } else {
        "each on its own counts".to_string()
    };
    info!(
        "Batch fold: {n_placed} of {n_cells} query cells on the model's {} nodes \
         ({n_fitted} where the fit placed them, the rest placed {mode}); \
         the others (a batch new to the model, no read, or kept on δ by the fit) keep δ",
        saved.log_mean_composition.ncols()
    );
    Ok(QueryFoldDivisor {
        table,
        col_of_cell,
        n_placed,
        n_fitted,
    })
}

/// The counts of `cells` (each `counts_of` one), summed per feature.
fn summed_counts(
    cells: impl Iterator<Item = usize>,
    counts_of: &HashMap<usize, Vec<(usize, f32)>>,
) -> Vec<(usize, f32)> {
    let mut all: Vec<(usize, f32)> = cells.flat_map(|c| counts_of[&c].iter().copied()).collect();
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
    let delta_col_of_fold_batch: Vec<Option<usize>> = saved
        .batches
        .iter()
        .map(|b| model_delta_batches.iter().position(|d| d == b))
        .collect();
    let table = fold_table_with_delta(
        saved,
        &fold_rows(saved, rows),
        model_delta,
        &delta_col_of_fold_batch,
    );
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
