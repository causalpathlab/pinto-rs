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
    estimate_state_aware_batch_fold_over_samples, CellPairFoldParams, CollapsePseudobulks,
    LinkStateDecomposition, SampleBatchFold, StateAwareBatchFold, StateAwareBatchFoldParams,
};
use data_beans::alg::collapse_data::MultilevelParams;

/// The states a cell pair's fold is estimated over (`--pair-fold-states`).
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PairStates {
    /// Learned from the pair itself: each pair divided by the fold its two
    /// cells predict, the leftover fitted over the pairs' own states.
    #[default]
    Link,
    /// The pair of its two cells' nodes on the cells' tree.
    Endpoint,
}

impl PairStates {
    fn of_data_beans(self) -> data_beans::alg::collapse_data::state_aware_batch_fold::PairStates {
        use data_beans::alg::collapse_data::state_aware_batch_fold::PairStates as D;
        match self {
            Self::Link => D::LinkStates,
            Self::Endpoint => D::EndpointStates,
        }
    }
}

/// The pair fold's options, for the methods that model pairs (lc, dsvd,
/// cage). Flatten into their args with `#[command(flatten)]`.
#[derive(clap::Args, Debug, Clone)]
pub struct PairFoldArgs {
    #[arg(
        long,
        value_enum,
        default_value_t = PairStates::Link,
        hide = true,
        help = "States the cell-pair batch fold is estimated over",
        long_help = "States the cell pairs' batch fold is estimated over.\n\
                     link (default): each pair is divided by the fold its two cells\n\
                     predict, and the leftover is fitted over the pairs' own states.\n\
                     endpoint: each pair is placed on the cells' tree by its two\n\
                     cells' nodes, and one fold is fitted per pair node."
    )]
    pub pair_fold_states: PairStates,
}

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
    /// `--pair-fold-states link`: the fold's decomposition (cell-predicted +
    /// each link state's leftover), with each pair's link state.
    pub link_states: Option<LinkStateDecomposition>,
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

/// Which state-aware folds a fit wrote beside δ.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FoldWritten {
    #[default]
    None,
    /// The cells' fold.
    Cells,
    /// The cells' fold and the cell pairs' link states.
    CellsAndLinkStates,
}

impl FoldWritten {
    /// What a fit with these folds wrote ([`write_fold_tables`]).
    pub fn of(cells: Option<&CellFold>, pairs: Option<&PairFold>) -> Self {
        match (cells, pairs.and_then(|p| p.link_states.as_ref())) {
            (None, _) => Self::None,
            (Some(_), None) => Self::Cells,
            (Some(_), Some(_)) => Self::CellsAndLinkStates,
        }
    }
}

/// What a run's feature_community profiles were fitted on.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProfilesDividedBy {
    /// Counts divided by δ (gene × batch).
    Delta,
    /// Counts divided by each cell's state-aware fold.
    CellFold,
}

/// The fold tables a fit wrote, relative to its manifest. predict and
/// impute divide by them; a run without them is read on δ.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct BatchFoldFiles {
    /// feature × columns: `fold:n{node}:{batch}`, each node's final log fold
    /// in each batch (δ included; what a cell there is divided by), then
    /// `composition:n{node}`, each node's fold-corrected log composition.
    pub cells: String,
    /// Each fitted cell's node: `n{node}`, or `delta` for a cell divided by δ alone.
    pub cell_nodes: String,
    /// feature × columns: `leftover:s{state}:{batch}`, each link state's log
    /// fold over the cell-predicted fold, then `composition:s{state}`.
    /// Absent under `--pair-fold-states endpoint`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link_states: Option<String>,
}

impl BatchFoldFiles {
    /// The files under `stem` (a path prefix, or a manifest's stem) for
    /// what was written; `None` when no fold was.
    pub fn written(stem: &str, fold: FoldWritten) -> Option<Self> {
        (fold != FoldWritten::None).then(|| Self {
            cells: format!("{stem}.batch_fold_cells.parquet"),
            cell_nodes: format!("{stem}.batch_fold_cell_nodes.parquet"),
            link_states: (fold == FoldWritten::CellsAndLinkStates)
                .then(|| format!("{stem}.batch_fold_link_states.parquet")),
        })
    }
}

/// Write a fit's folds under `prefix` ([`BatchFoldFiles`]): `features` the
/// data's rows, `cells` its columns, `batches` the batch labels the folds'
/// batch ids index. Returns what was written.
pub fn write_fold_tables(
    prefix: &str,
    features: &[Box<str>],
    cells: &[Box<str>],
    batches: &[Box<str>],
    cell_fold: Option<&CellFold>,
    pair_fold: Option<&PairFold>,
) -> anyhow::Result<FoldWritten> {
    let written = FoldWritten::of(cell_fold, pair_fold);
    let (Some(cf), Some(files)) = (cell_fold, BatchFoldFiles::written(prefix, written)) else {
        return Ok(FoldWritten::None);
    };
    write_cell_fold_tables(
        &files,
        FoldTableNames {
            features,
            cells,
            batches,
        },
        &cf.fit.fold,
        &cf.fit.node_log_mean_composition,
    )?;
    if let (Some(path), Some(ls)) = (
        &files.link_states,
        pair_fold.and_then(|p| p.link_states.as_ref()),
    ) {
        let (table, names) = with_compositions(
            &ls.leftover_log_fold,
            &ls.link_state_log_mean_composition,
            batches,
            ("leftover", "s"),
        );
        table.to_parquet_with_names(path, (Some(features), Some("feature")), Some(&names))?;
    }
    Ok(written)
}

/// `fold` (features × (units · batches)) then `composition` (features ×
/// units) side by side, with their columns' names: `{tag}:{unit}{k}:{batch}`
/// and `composition:{unit}{k}`.
fn with_compositions(
    fold: &Mat,
    composition: &Mat,
    batches: &[Box<str>],
    (tag, unit): (&str, &str),
) -> (Mat, Vec<Box<str>>) {
    let n = composition.ncols();
    let mut names: Vec<Box<str>> = (0..n)
        .flat_map(|k| {
            batches
                .iter()
                .map(move |b| format!("{tag}:{unit}{k}:{b}").into())
        })
        .collect();
    names.extend((0..n).map(|k| format!("composition:{unit}{k}").into()));
    let table = Mat::from_fn(fold.nrows(), fold.ncols() + n, |g, c| {
        if c < fold.ncols() {
            fold[(g, c)]
        } else {
            composition[(g, c - fold.ncols())]
        }
    });
    (table, names)
}

/// What a fold table names: the data's rows, its cells, and the batch labels
/// the fold's batch ids index.
pub(crate) struct FoldTableNames<'a> {
    pub features: &'a [Box<str>],
    pub cells: &'a [Box<str>],
    pub batches: &'a [Box<str>],
}

/// The cells' tables of `files`: each node's fold in each batch and its
/// composition, and each fitted cell's node (`n{k}`, or [`CELL_ON_DELTA`]).
/// Its rows, cells and batches are named by the [`FoldTableNames`].
pub(crate) fn write_cell_fold_tables(
    files: &BatchFoldFiles,
    FoldTableNames {
        features,
        cells,
        batches,
    }: FoldTableNames<'_>,
    fold: &StateAwareBatchFold,
    composition: &Mat,
) -> anyhow::Result<()> {
    use legume_numeric::matrix::parquet::{write_named_table, Column};
    anyhow::ensure!(
        fold.n_batches == batches.len(),
        "cell fold: {} batches for {} labels",
        fold.n_batches,
        batches.len()
    );
    let (table, names) = with_compositions(&fold.log_fold, composition, batches, ("fold", "n"));
    table.to_parquet_with_names(
        &files.cells,
        (Some(features), Some("feature")),
        Some(&names),
    )?;
    let node_of_cell: Vec<Box<str>> = fold
        .node_of_unit
        .iter()
        .map(|k| k.map_or_else(|| CELL_ON_DELTA.into(), |k| format!("n{k}").into()))
        .collect();
    write_named_table(
        &files.cell_nodes,
        "cell",
        cells,
        &[("node".into(), Column::Str(&node_of_cell))],
    )
}

/// How a query's cells were placed on a model's saved fold (predict,
/// impute), as its manifest records it.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchFoldPlacement {
    /// Spatial neighbours each cell's counts were summed with (0: its own).
    pub neighbours: usize,
    /// Cells divided by a placed node's fold; the rest kept δ.
    pub placed_cells: usize,
    /// Of those, cells the fit itself saw, which keep its node.
    #[serde(default)]
    pub fitted_cells: usize,
    pub cells: usize,
}

/// A fitted cell that keeps δ: its `cell_nodes` row.
pub const CELL_ON_DELTA: &str = "delta";

/// A fit's cell fold, read back from its [`BatchFoldFiles`] for predict and
/// impute.
pub struct SavedCellFold {
    pub features: Vec<Box<str>>,
    /// The batch labels the fold's columns are in, in order.
    pub batches: Vec<Box<str>>,
    /// features × (nodes · batches): each node's final log fold per batch.
    pub log_fold: Mat,
    /// features × nodes: each node's fold-corrected log composition.
    pub log_mean_composition: Mat,
    /// Each fitted cell's node; `None`: it was divided by δ alone.
    pub node_of_cell: rustc_hash::FxHashMap<Box<str>, Option<usize>>,
}

impl SavedCellFold {
    /// The cell fold the run `run` saved; `None` for a run that saved none
    /// (an older run, or one that divided by δ alone), whose readers keep δ.
    pub fn read(run: &crate::util::metadata::Run) -> anyhow::Result<Option<Self>> {
        let Some(files) = run.meta.outputs.batch_fold.as_ref() else {
            return Ok(None);
        };
        let found = |rel: &str| -> anyhow::Result<String> {
            run.recorded_output(Some(rel), rel)?
                .map(|p| p.to_string_lossy().into_owned())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{rel}: the model's manifest records its batch fold but the file is not \
                         beside it; copy the run's .batch_fold_*.parquet files with the model"
                    )
                })
        };
        Self::read_files(&found(&files.cells)?, &found(&files.cell_nodes)?).map(Some)
    }

    pub(crate) fn read_files(cells: &str, cell_nodes: &str) -> anyhow::Result<Self> {
        use legume_numeric::matrix::parquet::read_table_columns;
        let t = Mat::from_parquet(cells)?;
        let n_nodes = t
            .cols
            .iter()
            .filter(|c| c.starts_with("composition:"))
            .count();
        let n_fold = t.cols.len() - n_nodes;
        anyhow::ensure!(
            n_nodes > 0 && n_fold % n_nodes == 0,
            "{cells}: {n_fold} fold columns for {n_nodes} nodes"
        );
        let nb = n_fold / n_nodes;
        let batches: Vec<Box<str>> = t.cols[..nb]
            .iter()
            .map(|c| c.splitn(3, ':').nth(2).unwrap_or_default().into())
            .collect();
        let (strs, _) = read_table_columns(cell_nodes, &["cell", "node"], &[])?;
        let [cell, node]: [Vec<Box<str>>; 2] = strs
            .try_into()
            .map_err(|_| anyhow::anyhow!("{cell_nodes}: expected cell and node columns"))?;
        let node_of_cell = cell
            .into_iter()
            .zip(node)
            .map(|(c, n)| {
                let k = (n.as_ref() != CELL_ON_DELTA)
                    .then(|| n.strip_prefix('n').and_then(|k| k.parse::<usize>().ok()))
                    .flatten();
                (c, k)
            })
            .collect();
        Ok(Self {
            features: t.rows,
            batches,
            log_fold: t.mat.columns(0, n_fold).into_owned(),
            log_mean_composition: t.mat.columns(n_fold, n_nodes).into_owned(),
            node_of_cell,
        })
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
/// collapsed for their own δ, each pair's state found over `pair_states`
/// (both build on the cells' fit, `cells`). `pairs` are cell indices into
/// `data_vec`.
pub fn estimate_pair_batch_fold(
    data_vec: &SparseIoVec,
    pairs: &[(usize, usize)],
    batch_membership: &[Box<str>],
    cells: &CellFold,
    proj_dim: usize,
    num_levels: usize,
    pair_states: PairStates,
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
        &CellPairFoldParams {
            pair_states: pair_states.of_data_beans(),
            fold: params(),
        },
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
        link_states: fold.link_states,
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
