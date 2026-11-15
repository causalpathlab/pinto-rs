//! Batch effect estimation via hierarchical pseudobulk collapsing.
//!
//! ## How multilevel collapsing estimates batch effects
//!
//! The batch correction uses a METIS-inspired hierarchical coarsening
//! scheme that avoids expensive cell-level cross-batch comparisons:
//!
//! 1. **Partition cells into pseudobulk groups** via binary sort on
//!    random projection coordinates (at progressively finer `sort_dim`
//!    per level).
//!
//! 2. **Build pb-samples**: each (batch, group) intersection becomes
//!    a pb-sample with a centroid (mean projection vector) and
//!    aggregated feature sums. This is a small set — typically
//!    O(n_batches * n_groups) entries.
//!
//! 3. **Cross-batch KNN matching** (`--batch-knn`): for each
//!    pb-sample, find its `batch_knn` nearest neighbors among
//!    pb-samples from *other* batches using HNSW on centroids.
//!    Because we search over coarsened pb-samples (not individual
//!    cells), this is fast even with many cells.
//!
//! 4. **Counterfactual imputation**: the matched neighbors provide a
//!    "what would this group look like in another batch?" estimate.
//!    - `imputed_sum[g,s]` = weighted average of matched neighbors'
//!      per-cell feature expression, scaled by cell count
//!    - `matched_bs[b,s]` = how much of that counterfactual came from
//!      each source batch
//!
//! 5. **EM-style optimization**: one batch-free rate μ per (feature, group)
//!    with both the observed and the counterfactual side Poisson at that
//!    rate, and a per-(feature, batch) fold δ identified by the counterfactual
//!    side and pinned to geometric mean 1 over the frame batches. The
//!    per-group readouts μ_resid (own fold) and γ (source fold) are derived
//!    from μ afterwards.
//!
//! 6. **Multilevel refinement**: repeat steps 1-5 at each level with
//!    increasing `sort_dim` (coarse → fine). Coarser levels capture
//!    large-scale batch effects; finer levels refine them.
//!
//! The key insight: because KNN matching operates on pb-sample
//! centroids (not individual cells), `batch_knn` remains small
//! (default 10) regardless of dataset size. The hierarchical levels
//! ensure both coarse global corrections and fine-grained local
//! adjustments.

use crate::util::common::*;
use data_beans::alg::collapse_data::*;
use data_beans::alg::random_projection::*;
use legume_numeric::matrix::parquet::read_table_columns;
use legume_numeric::param::dmatrix_gamma::GammaMatrix;
use legume_numeric::param::io::ParamIo;
use legume_numeric::param::traits::Inference;
use rustc_hash::FxHashMap;

pub struct EstimateBatchArgs {
    pub proj_dim: usize,
    pub sort_dim: usize,
    pub block_size: Option<usize>,
    /// KNN for cross-batch pb-sample matching during hierarchical
    /// collapsing. Searches are over coarsened pb-sample centroids,
    /// not individual cells, so this stays small.
    pub batch_knn: usize,
    pub num_levels: usize,
    /// Per-cell CNV stratum from `mung clones` (`0` = mixable). Maps to
    /// [`MultilevelParams::strata`].
    pub strata: Option<Vec<usize>>,
}

/// Read a `mung clones` table and align strata to `data_vec` column order.
pub fn load_cnv_cell_strata(
    clones_path: &str,
    data_vec: &SparseIoVec,
) -> anyhow::Result<Vec<usize>> {
    let cell_to_stratum = strata_of(clones_path, &data_vec.column_names()?)?;
    let n_kept = cell_to_stratum.iter().filter(|&&s| s > 0).count();
    info!(
        "CNV strata from {clones_path}: {} / {} cells in donor-private clones",
        n_kept,
        cell_to_stratum.len()
    );
    Ok(cell_to_stratum)
}

/// Each of `cells`' stratum in the clone table at `path` (its `cell` and
/// `stratum` columns; the file is the whole contract with mung). Cells it
/// does not list go to stratum 0, with a warning.
pub(crate) fn strata_of(path: &str, cells: &[Box<str>]) -> anyhow::Result<Vec<usize>> {
    let (strs, nums) = read_table_columns(path, &["cell"], &["stratum"])?;
    let cell = strs.into_iter().next().unwrap_or_default();
    let stratum = nums.into_iter().next().unwrap_or_default();
    anyhow::ensure!(
        cell.len() == stratum.len(),
        "{path}: columns of different lengths"
    );
    let mut by_cell: FxHashMap<&str, usize> = FxHashMap::default();
    for (i, (c, &s)) in cell.iter().zip(&stratum).enumerate() {
        anyhow::ensure!(
            s >= 0.0 && s.fract() == 0.0,
            "{path}: row {i}: stratum {s} is not a count"
        );
        by_cell.insert(c, s as usize);
    }
    let mut missing = 0usize;
    let out: Vec<usize> = cells
        .iter()
        .map(|c| {
            by_cell.get(c.as_ref()).copied().unwrap_or_else(|| {
                missing += 1;
                0
            })
        })
        .collect();
    if missing > 0 {
        warn!(
            "{missing} / {} cells missing from the clone table; treating as stratum 0",
            cells.len()
        );
    }
    Ok(out)
}

/// Estimate per-feature batch effect multipliers (δ) via hierarchical
/// pseudobulk collapsing with cross-batch KNN matching.
///
/// Returns `None` if fewer than 2 batches are present.
pub fn estimate_batch(
    data_vec: &mut SparseIoVec,
    batch_membership: &[Box<str>],
    args: EstimateBatchArgs,
) -> anyhow::Result<Option<GammaMatrix>> {
    let batch_hash: HashSet<Box<str>> = batch_membership.iter().cloned().collect();
    let nbatch = batch_hash.len();
    if nbatch < 2 {
        return Ok(None);
    }

    let proj_out = data_vec.project_columns_with_batch_correction(
        args.proj_dim,
        args.block_size,
        Some(batch_membership),
    )?;

    let cell_proj_kn = proj_out.proj;
    info!(
        "Proj: {} x {} ...",
        cell_proj_kn.nrows(),
        cell_proj_kn.ncols()
    );

    let collapse_out = data_vec.collapse_columns_multilevel(
        &cell_proj_kn,
        batch_membership,
        &MultilevelParams {
            knn_pb_samples: args.batch_knn,
            sort_dim: args.sort_dim,
            num_levels: args.num_levels,
            strata: args.strata.clone(),
            ..MultilevelParams::new(cell_proj_kn.nrows())
        },
    )?;

    Ok(collapse_out.delta)
}

/// Estimate batch effects and write them to parquet.
///
/// Skips estimation when fewer than 2 batches are present.
/// Returns the posterior mean matrix `[n_features × n_batches]` when
/// multi-batch, or `None` for single-batch.
pub fn estimate_and_write_batch_effects(
    data_vec: &mut SparseIoVec,
    batch_membership: &[Box<str>],
    args: EstimateBatchArgs,
    out_prefix: &str,
) -> anyhow::Result<Option<Mat>> {
    let uniq_batches: HashSet<&Box<str>> = batch_membership.iter().collect();
    let n_batches = uniq_batches.len();
    drop(uniq_batches);

    if n_batches < 2 {
        return Ok(None);
    }

    info!("Estimating batch effects ({} batches)...", n_batches);
    let batch_effects = estimate_batch(data_vec, batch_membership, args)?;

    if let Some(batch_db) = batch_effects.as_ref() {
        let outfile = out_prefix.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let feature_names = data_vec.row_names()?;
        batch_db.to_melted_parquet(
            &outfile,
            (Some(&feature_names), Some("feature")),
            (batch_names.as_deref(), Some("batch")),
        )?;
    }

    Ok(batch_effects.map(|x| x.posterior_mean().clone()))
}

/// A run's batch effects, read back from its `{prefix}.delta.parquet`: the
/// posterior mean of δ, one row per matrix row the run read (a splice
/// channel such as `FEATURE/count/spliced` is a row of its own) and one
/// column per batch label as estimated (after `--auto-batch`), sorted.
pub struct RunDelta {
    pub rows: Vec<Box<str>>,
    pub batches: Vec<Box<str>>,
    /// `[rows × batches]`
    pub mean: Mat,
}

/// Read a melted `delta.parquet` (`feature`, `batch`, `mean`, ...); only the
/// mean is kept, the value the run itself divided by.
pub fn read_delta(path: &str) -> anyhow::Result<RunDelta> {
    let (strs, nums) = read_table_columns(path, &["feature", "batch"], &["mean"])?;
    let [feature, batch]: [Vec<Box<str>>; 2] = strs
        .try_into()
        .map_err(|_| anyhow::anyhow!("{path}: expected feature and batch columns"))?;
    let mean = nums.into_iter().next().unwrap_or_default();
    anyhow::ensure!(
        feature.len() == batch.len() && batch.len() == mean.len(),
        "{path}: columns of different lengths"
    );
    let mut batches: Vec<Box<str>> = batch.clone();
    batches.sort();
    batches.dedup();
    let col_of: HashMap<&str, usize> = batches
        .iter()
        .enumerate()
        .map(|(j, b)| (b.as_ref(), j))
        .collect();
    let mut rows: Vec<Box<str>> = Vec::new();
    let mut row_of: HashMap<Box<str>, usize> = HashMap::default();
    for f in &feature {
        if !row_of.contains_key(f) {
            row_of.insert(f.clone(), rows.len());
            rows.push(f.clone());
        }
    }
    let mut m = Mat::from_element(rows.len(), batches.len(), f32::NAN);
    for ((f, b), &v) in feature.iter().zip(&batch).zip(&mean) {
        m[(row_of[f], col_of[b.as_ref()])] = v as f32;
    }
    anyhow::ensure!(
        m.iter().all(|v| v.is_finite()),
        "{path}: not every (feature, batch) pair has a finite mean"
    );
    Ok(RunDelta {
        rows,
        batches,
        mean: m,
    })
}

impl RunDelta {
    /// Each of `rows`' row in this δ, by name; `None` for a row the run did
    /// not read.
    pub fn row_index(&self, rows: &[Box<str>]) -> Vec<Option<usize>> {
        let of: HashMap<&str, usize> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.as_ref(), i))
            .collect();
        rows.iter().map(|r| of.get(r.as_ref()).copied()).collect()
    }

    /// This δ on another matrix's `rows`, matched by name: `[rows × batches]`,
    /// with 1 (no shift) on a row the run did not read.
    pub fn on_rows(&self, rows: &[Box<str>]) -> Mat {
        let idx = self.row_index(rows);
        Mat::from_fn(rows.len(), self.batches.len(), |i, j| {
            idx[i].map_or(1.0, |r| self.mean[(r, j)])
        })
    }
}

/// The δ a fitted run left at `prefix`, when it has one: the run's manifest
/// says whether it wrote one (a stale file from an earlier fit under the
/// same prefix is not read); a run without a manifest is taken at its
/// `{prefix}.delta.parquet`.
pub fn run_delta_path(prefix: &str) -> Option<String> {
    let path = format!("{prefix}.delta.parquet");
    let manifest = crate::util::metadata::manifest_path(prefix);
    let recorded = match crate::util::metadata::PintoMetadata::read(std::path::Path::new(&manifest))
    {
        Ok(meta) => meta.outputs.batch_effects.is_some(),
        Err(_) => true,
    };
    (recorded && std::path::Path::new(&path).is_file()).then_some(path)
}

/// Each cell's batch label as the run estimated it, from its
/// `coord_pairs.parquet` (`left_batch` / `right_batch`). These are the
/// labels δ's columns are named by; the cells table keeps the labels as
/// read, before `--auto-batch`. Empty when the table has no batch columns
/// (a single-batch run).
pub fn cell_batches_from_coord_pairs(path: &str) -> anyhow::Result<HashMap<Box<str>, Box<str>>> {
    let pairs = crate::lr_activity::io::read_pair_table(path, false)?;
    let mut out: HashMap<Box<str>, Box<str>> = HashMap::default();
    for e in pairs {
        if let Some((l, r)) = e.side_batches {
            out.insert(e.left_cell, l);
            out.insert(e.right_cell, r);
        }
    }
    Ok(out)
}

/// Divide each count of `x` by its row's δ in its column's batch, a plain
/// `x / δ`: `batch_of_column(j)` is column `j`'s column of `delta`
/// (`[rows × batches]`), `None` to leave the column as it is (no batch, or
/// no use for it). A non-positive δ leaves its count as is.
///
/// lc's edge profiles and impute's reference cells divide this way; the
/// cage pair divisor does not, since it also divides by the cell's own
/// rate (a self-normalising Poisson ratio, see `PairBatchDivisor`).
pub fn divide_columns(
    x: &mut nalgebra_sparse::CscMatrix<f32>,
    delta: &Mat,
    batch_of_column: impl Fn(usize) -> Option<usize>,
) {
    for (j, mut col) in x.col_iter_mut().enumerate() {
        let Some(b) = batch_of_column(j) else {
            continue;
        };
        let (rows, vals) = col.rows_and_values_mut();
        for (&g, v) in rows.iter().zip(vals.iter_mut()) {
            let d = delta[(g, b)];
            if d > 0.0 {
                *v /= d;
            }
        }
    }
}
