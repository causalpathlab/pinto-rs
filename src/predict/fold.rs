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

use crate::util::batch_effects::index_by_name;
use crate::util::batch_fold::SavedCellFold;
use crate::util::common::*;
use rayon::prelude::*;

/// Spatial neighbours a query cell's counts are summed with before it is
/// placed. On a held-out section (sim donor-confounded, lc fit on the other
/// five), placement agreed with the planted domain at ARI 0.475 alone, 0.673
/// with 4 neighbours, 0.699 with 12; 4 smooths less across domain edges.
pub(crate) const PLACEMENT_NEIGHBOURS: usize = 4;

/// Query cells placed per block (read together with their neighbours; the
/// blocks run in parallel).
const PLACEMENT_BLOCK: usize = 10_000;

/// What divides each query cell: `table` (the query's rows × columns,
/// linear) and each cell's column of it.
pub(crate) struct QueryFoldDivisor {
    pub table: Mat,
    pub col_of_cell: Vec<u32>,
    /// Cells divided by a node's fold (the rest keep δ).
    pub n_placed: usize,
    /// Of those, cells the fit itself placed (by name), not placed again.
    pub n_fitted: usize,
    /// Neighbours each placed cell's counts were summed with (0: its own).
    pub neighbours: usize,
}

/// The query's batches as the model's fold names them.
pub(crate) struct QueryBatches<'a> {
    /// Each cell's query batch (a column of `delta`).
    pub of_cell: &'a [u32],
    /// Each query batch's label, as the model's batches are named when it
    /// is one of them (`None`: new to the model).
    pub model_label: &'a [Option<Box<str>>],
}

/// The query's pairs, what a cell's placement neighbours are drawn from
/// ([`placement_neighbours`]).
pub(crate) struct PlacementGraph<'a> {
    pub pairs: &'a [(u32, u32)],
    /// Each pair's kind (`None`: every pair spatial).
    pub edge_kind: Option<&'a [i32]>,
    /// Cells × dims.
    pub coordinates: &'a Mat,
}

/// The query cells: their names (a cell the fit placed keeps its node) and
/// the graph its neighbours are summed from before it is placed (`None`:
/// each on its own counts).
pub(crate) struct QueryCells<'a> {
    pub names: &'a [Box<str>],
    pub graph: Option<PlacementGraph<'a>>,
}

/// Each cell's nearest neighbours to sum before it is placed: up to
/// [`PLACEMENT_NEIGHBOURS`] of its spatial pairs, in its own query batch,
/// nearest first.
pub(crate) fn placement_neighbours(
    graph: &PlacementGraph<'_>,
    batch_of_cell: &[u32],
) -> Vec<Vec<usize>> {
    let n = batch_of_cell.len();
    let xy = graph.coordinates;
    let mut cand: Vec<Vec<(f32, usize)>> = vec![Vec::new(); n];
    for (e, &(u, v)) in graph.pairs.iter().enumerate() {
        let spatial = graph
            .edge_kind
            .is_none_or(|k| k[e] == crate::util::cell_pairs::EDGE_KIND_SPATIAL);
        let (u, v) = (u as usize, v as usize);
        if spatial && u != v && batch_of_cell[u] == batch_of_cell[v] {
            let d = (xy.row(u) - xy.row(v)).norm_squared();
            cand[u].push((d, v));
            cand[v].push((d, u));
        }
    }
    cand.into_par_iter()
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

/// A cell's column of [`fold_table_with_delta`]'s table: its node's fold in
/// its fold batch, else its δ column; `None` without either.
fn cell_column(
    saved: &SavedCellFold,
    node: Option<usize>,
    fold_batch: Option<usize>,
    delta_col: Option<usize>,
) -> Option<usize> {
    match (node, fold_batch) {
        (Some(k), Some(b)) => Some(k * saved.batches.len() + b),
        _ => delta_col.map(|q| saved.log_fold.ncols() + q),
    }
}

/// Each label's position in `labels`.
fn position_of(labels: &[Box<str>]) -> HashMap<&str, usize> {
    labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_ref(), i))
        .collect()
}

/// The divisor of the query `data` (rows `rows`): a node's fold for each
/// cell of a batch the model knows (the fit's own node for a cell it placed,
/// else one placed by composition), else the cell's column of `delta` (rows
/// × query batches, linear; the δ predict divides by without a fold). The
/// cells to place are read in blocks, each with its cells' neighbours.
pub(crate) fn query_fold_divisor(
    saved: &SavedCellFold,
    data: &SparseIoVec,
    rows: &[Box<str>],
    delta: &Mat,
    batches: &QueryBatches<'_>,
    cells: &QueryCells<'_>,
) -> anyhow::Result<QueryFoldDivisor> {
    let n_cells = data.num_columns();
    anyhow::ensure!(
        batches.of_cell.len() == n_cells
            && cells.names.len() == n_cells
            && delta.nrows() == rows.len(),
        "query fold: one batch and name per cell, one δ row per data row"
    );
    let fold_row = index_by_name(&saved.features, rows);
    let saved_batch = position_of(&saved.batches);
    let fold_batch: Vec<Option<usize>> = batches
        .model_label
        .iter()
        .map(|l| {
            l.as_ref()
                .and_then(|l| saved_batch.get(l.as_ref()).copied())
        })
        .collect();
    let batch_of_cell: Vec<Option<usize>> = batches
        .of_cell
        .iter()
        .map(|&qb| fold_batch[qb as usize])
        .collect();

    // a cell the fit placed keeps the fit's node (or δ, when the fit kept δ)
    let mut node: Vec<Option<usize>> = vec![None; n_cells];
    let mut to_place: Vec<usize> = Vec::new();
    for c in 0..n_cells {
        if batch_of_cell[c].is_none() {
            continue;
        }
        match saved.node_of_cell.get(&cells.names[c]) {
            Some(&k) => node[c] = k,
            None => to_place.push(c),
        }
    }
    let n_fitted = node.iter().filter(|k| k.is_some()).count();

    // the rest placed by composition, each cell on its counts (a new
    // batch's divided by its δ) summed with its neighbours'
    let neighbours = match (&cells.graph, to_place.is_empty()) {
        (Some(g), false) => Some(placement_neighbours(g, batches.of_cell)),
        _ => None,
    };
    let placer = data_beans::alg::collapse_data::state_aware_batch_fold::CompositionPlacer::new(
        &saved.log_mean_composition,
        &saved.log_fold,
        saved.batches.len(),
    )?;
    let read_counts = CountsOnFold {
        fold_row: &fold_row,
        n_features: saved.features.len(),
        delta,
        of_cell: batches.of_cell,
        fold_batch: &fold_batch,
    };
    let placed: Vec<Vec<(usize, Option<usize>)>> = to_place
        .par_chunks(PLACEMENT_BLOCK)
        .map(|placing| -> anyhow::Result<_> {
            let summed = read_counts.summed(data, placing, neighbours.as_deref())?;
            let batch: Vec<Option<usize>> = placing.iter().map(|&c| batch_of_cell[c]).collect();
            let k = placer.place(&summed, &batch)?;
            Ok(placing.iter().copied().zip(k).collect())
        })
        .collect::<anyhow::Result<_>>()?;
    for (c, k) in placed.into_iter().flatten() {
        node[c] = k;
    }

    // a fold batch's δ column: the query batch the model names so
    let delta_col_of_fold_batch: Vec<Option<usize>> = (0..saved.batches.len())
        .map(|b| fold_batch.iter().position(|&f| f == Some(b)))
        .collect();
    let table = fold_table_with_delta(saved, &fold_row, delta, &delta_col_of_fold_batch);
    let col_of_cell: Vec<u32> = (0..n_cells)
        .map(|c| {
            let qb = batches.of_cell[c] as usize;
            cell_column(saved, node[c], batch_of_cell[c], Some(qb)).expect("a δ column") as u32
        })
        .collect();
    let n_fold = saved.log_fold.ncols() as u32;
    let n_placed = col_of_cell.iter().filter(|&&c| c < n_fold).count();
    let n_neighbours = if neighbours.is_some() {
        PLACEMENT_NEIGHBOURS
    } else {
        0
    };
    info!(
        "Batch fold: {n_placed} of {n_cells} query cells on the model's {} nodes \
         ({n_fitted} where the fit placed them, the rest placed with up to {n_neighbours} \
         spatial neighbours in their batch); the others (a batch new to the model, no read, \
         or kept on δ by the fit) keep δ",
        saved.log_mean_composition.ncols()
    );
    Ok(QueryFoldDivisor {
        table,
        col_of_cell,
        n_placed,
        n_fitted,
        neighbours: n_neighbours,
    })
}

/// Reads query cells' counts on the fold's features: a cell of a batch the
/// model knows as read, one of a new batch divided by its δ (a non-positive
/// δ leaves the count, as [`crate::util::batch_effects::divide_columns`]).
struct CountsOnFold<'a> {
    fold_row: &'a [Option<usize>],
    n_features: usize,
    delta: &'a Mat,
    of_cell: &'a [u32],
    fold_batch: &'a [Option<usize>],
}

impl CountsOnFold<'_> {
    /// Each of `placing`'s counts summed with its `neighbours`' (`None`: its
    /// own alone), the cells and their neighbours read once.
    fn summed(
        &self,
        data: &SparseIoVec,
        placing: &[usize],
        neighbours: Option<&[Vec<usize>]>,
    ) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
        let no_neighbours: &[usize] = &[];
        let cells_of = |c: usize| {
            std::iter::once(c).chain(neighbours.map_or(no_neighbours, |n| &n[c]).iter().copied())
        };
        let mut read: Vec<usize> = placing.iter().flat_map(|&c| cells_of(c)).collect();
        read.sort_unstable();
        read.dedup();
        let block = data.read_columns_csc(read.iter().copied())?;
        let mut sum = vec![0f32; self.n_features];
        let mut touched: Vec<usize> = Vec::new();
        Ok(placing
            .iter()
            .map(|&c| {
                for u in cells_of(c) {
                    let j = read.binary_search(&u).expect("read");
                    let qb = self.of_cell[u] as usize;
                    let known = self.fold_batch[qb].is_some();
                    let col = block.col(j);
                    for (&r, &x) in col.row_indices().iter().zip(col.values()) {
                        let Some(g) = self.fold_row[r] else { continue };
                        let d = self.delta[(r, qb)];
                        let x = if known || d <= 0.0 { x } else { x / d };
                        if sum[g] == 0.0 {
                            touched.push(g);
                        }
                        sum[g] += x;
                    }
                }
                touched.sort_unstable();
                let out = touched.iter().map(|&g| (g, sum[g])).collect();
                for &g in &touched {
                    sum[g] = 0.0;
                }
                touched.clear();
                out
            })
            .collect())
    }
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
    let saved_batch = position_of(&saved.batches);
    let delta_batch = position_of(model_delta_batches);
    let delta_col_of_fold_batch: Vec<Option<usize>> = saved
        .batches
        .iter()
        .map(|b| delta_batch.get(b.as_ref()).copied())
        .collect();
    let table = fold_table_with_delta(
        saved,
        &index_by_name(&saved.features, rows),
        model_delta,
        &delta_col_of_fold_batch,
    );
    let col = cells
        .iter()
        .zip(batch_label_of_cell)
        .map(|(cell, label)| {
            cell_column(
                saved,
                saved.node_of_cell.get(cell).copied().flatten(),
                saved_batch.get(label.as_ref()).copied(),
                delta_batch.get(label.as_ref()).copied(),
            )
        })
        .collect();
    (table, col)
}
