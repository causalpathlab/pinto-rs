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
//! 3. **Cross-batch KNN matching** (`BATCH_KNN` neighbours): for each
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
    /// KNN for cross-batch pb-sample matching during hierarchical
    /// collapsing. Searches are over coarsened pb-sample centroids,
    /// not individual cells, so this stays small.
    pub batch_knn: usize,
    pub num_levels: usize,
    /// Per-cell CNV stratum from `mung clones` (`0` = mixable). Maps to
    /// [`MultilevelParams::strata`].
    pub strata: Option<Vec<usize>>,
}

/// The finest `sort_dim` of the estimate's levels at most: its finest
/// level splits each batch into at most `2^BATCH_SORT_DIM` groups.
pub const BATCH_SORT_DIM: usize = 10;

impl EstimateBatchArgs {
    /// The estimate at projection `proj_dim`, its finest level at
    /// [`BATCH_SORT_DIM`] (or `proj_dim`, when smaller).
    pub fn at(
        proj_dim: usize,
        batch_knn: usize,
        num_levels: usize,
        strata: Option<Vec<usize>>,
    ) -> Self {
        EstimateBatchArgs {
            proj_dim,
            sort_dim: proj_dim.min(BATCH_SORT_DIM),
            batch_knn,
            num_levels,
            strata,
        }
    }
}

/// The distinct labels of `batches`, sorted: δ's columns.
pub fn sorted_batch_labels(batches: &[Box<str>]) -> Vec<Box<str>> {
    batches
        .iter()
        .map(AsRef::as_ref)
        .collect::<std::collections::BTreeSet<&str>>()
        .into_iter()
        .map(Box::from)
        .collect()
}

/// Bytes a stored nonzero holds while δ is estimated (data-beans'
/// multilevel collapse): the preloaded counts, the cells' projection and
/// the collapse's per-cell buffers. Calibrated on srt-simulate runs of two
/// sections (500 and 2000 genes, 512 to 16200 cells): the largest measured
/// residual per nonzero, with a margin.
const DELTA_NONZERO_BYTES: usize = 52;

/// Bytes a gene holds per pseudobulk sample while δ is estimated: the
/// collapse's dense gene × sample statistics over its levels. Calibrated
/// with [`DELTA_NONZERO_BYTES`].
const DELTA_GENE_SAMPLE_BYTES: usize = 128;

/// Pseudobulk samples the estimate's finest level holds per batch at most:
/// `2^sort_dim` groups, at [`BATCH_SORT_DIM`].
pub const DELTA_SAMPLES_PER_BATCH: usize = 1 << BATCH_SORT_DIM;

/// What the δ estimate holds at once, above the process's own, for
/// `genes` genes, `nonzeros` stored nonzeros and at most `samples`
/// pseudobulk samples (each batch's cells, at most
/// [`DELTA_SAMPLES_PER_BATCH`] of them).
pub fn delta_estimate_bytes(genes: usize, nonzeros: usize, samples: usize) -> usize {
    DELTA_NONZERO_BYTES * nonzeros + DELTA_GENE_SAMPLE_BYTES * genes * samples
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

/// The clone table at `path`: each cell it lists and its stratum (its
/// `cell` and `stratum` columns; the file is the whole contract with mung),
/// refused when the columns differ in length or a stratum is not a count.
fn read_clone_table(path: &str) -> anyhow::Result<Vec<(Box<str>, usize)>> {
    let (strs, nums) = read_table_columns(path, &["cell"], &["stratum"])?;
    let cell = strs.into_iter().next().unwrap_or_default();
    let stratum = nums.into_iter().next().unwrap_or_default();
    anyhow::ensure!(
        cell.len() == stratum.len(),
        "{path}: columns of different lengths"
    );
    cell.into_iter()
        .zip(stratum)
        .enumerate()
        .map(|(i, (c, s))| {
            anyhow::ensure!(
                s >= 0.0 && s.fract() == 0.0,
                "{path}: row {i}: stratum {s} is not a count"
            );
            Ok((c, s as usize))
        })
        .collect()
}

/// The strata of the clone table at `path`: those it lists, and stratum 0
/// (the mixable cells, and those it does not list).
pub fn n_strata(path: &str) -> anyhow::Result<usize> {
    let mut strata: std::collections::BTreeSet<usize> = read_clone_table(path)?
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    strata.insert(0);
    Ok(strata.len())
}

/// Each of `cells`' stratum in the clone table at `path`
/// ([`read_clone_table`]). Cells it does not list go to stratum 0, with a
/// warning.
pub(crate) fn strata_of(path: &str, cells: &[Box<str>]) -> anyhow::Result<Vec<usize>> {
    let by_cell: FxHashMap<Box<str>, usize> = read_clone_table(path)?.into_iter().collect();
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
/// pseudobulk collapsing with cross-batch KNN matching, and the
/// batch-corrected projection it collapses on. The caller counts the
/// batches ([`estimate_and_write_batch_effects`] needs two or more).
fn estimate_batch(
    data_vec: &mut SparseIoVec,
    batch_membership: &[Box<str>],
    args: EstimateBatchArgs,
) -> anyhow::Result<(Option<GammaMatrix>, RandColProjOut)> {
    let proj_out = data_vec.project_columns_with_batch_correction(
        args.proj_dim,
        None,
        Some(batch_membership),
    )?;

    let cell_proj_kn = &proj_out.proj;
    info!(
        "Proj: {} x {} ...",
        cell_proj_kn.nrows(),
        cell_proj_kn.ncols()
    );

    let collapse_out = data_vec.collapse_columns_multilevel(
        cell_proj_kn,
        batch_membership,
        &MultilevelParams {
            knn_pb_samples: args.batch_knn,
            sort_dim: args.sort_dim,
            num_levels: args.num_levels,
            strata: args.strata.clone(),
            ..MultilevelParams::new(cell_proj_kn.nrows())
        },
    )?;

    Ok((collapse_out.delta, proj_out))
}

/// What [`estimate_and_write_batch_effects`] gives: the posterior (`None`
/// when the estimate gave no batch effect), and the batch-corrected random
/// projection (`proj_dim × cells`) the estimate collapsed on, when the
/// caller keeps it.
pub struct BatchEffects {
    pub posterior: Option<BatchPosterior>,
    pub projection: Option<RandColProjOut>,
}

/// Estimate batch effects, and (with `out_prefix`) write them to
/// `{out_prefix}.delta.parquet` and each cell's batch to
/// `{out_prefix}.delta_cells.parquet`.
///
/// `labels` are the sorted, distinct labels of `batch_membership`
/// ([`sorted_batch_labels`]); nothing is estimated with fewer than 2
/// (`None`). Returns the posterior on the data's rows, its batches in label
/// order (data-beans' batch ids), so a caller indexing cells by sorted label
/// indexes its columns; with `keep_projection`, the projection too, which
/// is otherwise freed before the writes. An estimate that gives no batch
/// effect writes nothing and has no posterior, its projection still kept.
pub fn estimate_and_write_batch_effects(
    data_vec: &mut SparseIoVec,
    batch_membership: &[Box<str>],
    labels: &[Box<str>],
    args: EstimateBatchArgs,
    out_prefix: Option<&str>,
    keep_projection: bool,
) -> anyhow::Result<Option<BatchEffects>> {
    if labels.len() < 2 {
        return Ok(None);
    }

    info!("Estimating batch effects ({} batches)...", labels.len());
    let (gamma, proj) = estimate_batch(data_vec, batch_membership, args)?;
    let projection = keep_projection.then_some(proj);
    let Some(gamma) = gamma else {
        warn!(
            "{} batches, but no batch effect was estimated, so the counts are not divided \
             by one",
            labels.len()
        );
        return Ok(Some(BatchEffects {
            posterior: None,
            projection,
        }));
    };
    let batches = data_vec
        .batch_names()
        .ok_or_else(|| anyhow::anyhow!("batch effects estimated without batch names"))?;
    let rows = data_vec.row_names()?;
    if let Some(prefix) = out_prefix {
        gamma.to_melted_parquet(
            &format!("{prefix}.delta.parquet"),
            (Some(&rows), Some("feature")),
            (Some(&batches), Some("batch")),
        )?;
        write_cell_batches(
            &delta_cells_path(prefix),
            &data_vec.column_names()?,
            batch_membership,
        )?;
    }
    Ok(Some(BatchEffects {
        posterior: Some(BatchPosterior {
            gamma,
            rows,
            batches,
        }),
        projection,
    }))
}

/// The batch effect posterior an estimate gives, on its rows and batches.
/// A fit divides by the mean alone ([`Self::into_mean`]); `predict` carries
/// the posterior in full ([`Self::into_delta`]).
pub struct BatchPosterior {
    pub gamma: GammaMatrix,
    pub rows: Vec<Box<str>>,
    pub batches: Vec<Box<str>>,
}

impl BatchPosterior {
    /// The mean, `[rows × batches]`, its other planes left alone.
    pub fn into_mean(self) -> anyhow::Result<Mat> {
        self.check_mean()?;
        Ok(self.gamma.posterior_mean().clone())
    }

    /// The posterior in full. The mean must be `[rows × batches]`; a spread
    /// plane the estimate left unallocated is filled (no spread, log 0), as
    /// the writer has it.
    pub fn into_delta(self) -> anyhow::Result<RunDelta> {
        self.check_mean()?;
        let shape = (self.rows.len(), self.batches.len());
        let g = &self.gamma;
        let plane = |m: &Mat, fill: f32| {
            if m.shape() == shape {
                m.clone()
            } else {
                Mat::from_element(shape.0, shape.1, fill)
            }
        };
        let planes = [
            g.posterior_mean().clone(),
            plane(g.posterior_sd(), FILL[1]),
            plane(g.posterior_log_mean(), FILL[2]),
            plane(g.posterior_log_sd(), FILL[3]),
        ];
        Ok(RunDelta::from_planes(self.rows, self.batches, planes))
    }

    fn check_mean(&self) -> anyhow::Result<()> {
        let shape = self.gamma.posterior_mean().shape();
        let want = (self.rows.len(), self.batches.len());
        anyhow::ensure!(
            shape == want,
            "batch effect mean is {} x {}, not {} rows x {} batches",
            shape.0,
            shape.1,
            want.0,
            want.1
        );
        Ok(())
    }
}

/// A run's batch effects, as its `{prefix}.delta.parquet` holds them: the
/// posterior of δ, one row per matrix row the run read (a splice channel
/// such as `FEATURE/count/spliced` is a row of its own) and one column per
/// batch label as estimated (after `--auto-batch`), sorted. The mean is
/// what the run divided by; the spread planes are carried so a table
/// written from this one has the fits' schema.
#[derive(Clone)]
pub struct RunDelta {
    pub rows: Vec<Box<str>>,
    pub batches: Vec<Box<str>>,
    /// `[rows × batches]` each.
    pub mean: Mat,
    pub sd: Mat,
    pub log_mean: Mat,
    pub log_sd: Mat,
}

/// The columns of a melted δ table after `feature` and `batch`, as the
/// fits write them.
const DELTA_PLANES: [&str; 4] = ["mean", "sd", "log_mean", "log_sd"];

/// Each plane's value for a row or batch a δ lacks: no shift (mean 1, no
/// spread, log 0).
const FILL: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// Read a melted `delta.parquet` (`feature`, `batch`, `mean`, and the fits'
/// `sd`, `log_mean`, `log_sd`; a table with the mean alone reads with the
/// spread planes zero).
pub fn read_delta(path: &str) -> anyhow::Result<RunDelta> {
    let (strs, nums) = read_table_columns(path, &["feature", "batch"], &DELTA_PLANES)
        .or_else(|_| read_table_columns(path, &["feature", "batch"], &DELTA_PLANES[..1]))?;
    let [feature, batch]: [Vec<Box<str>>; 2] = strs
        .try_into()
        .map_err(|_| anyhow::anyhow!("{path}: expected feature and batch columns"))?;
    anyhow::ensure!(
        nums.iter().all(|c| c.len() == feature.len()) && feature.len() == batch.len(),
        "{path}: columns of different lengths"
    );
    let batches = sorted_batch_labels(&batch);
    let col_of: HashMap<&str, usize> = batches
        .iter()
        .enumerate()
        .map(|(j, b)| (b.as_ref(), j))
        .collect();
    let mut rows: Vec<Box<str>> = Vec::new();
    let mut row_of: HashMap<&str, usize> = HashMap::default();
    for f in &feature {
        if !row_of.contains_key(f.as_ref()) {
            row_of.insert(f, rows.len());
            rows.push(f.clone());
        }
    }
    let planes: [Mat; 4] = std::array::from_fn(|k| {
        // the mean has no fill: every (feature, batch) pair must be there
        let fill = if k == 0 { f32::NAN } else { FILL[k] };
        let mut m = Mat::from_element(rows.len(), batches.len(), fill);
        if let Some(values) = nums.get(k) {
            for ((f, b), &v) in feature.iter().zip(&batch).zip(values) {
                m[(row_of[f.as_ref()], col_of[b.as_ref()])] = v as f32;
            }
        }
        m
    });
    anyhow::ensure!(
        planes[0].iter().all(|v| v.is_finite()),
        "{path}: not every (feature, batch) pair has a finite mean"
    );
    Ok(RunDelta::from_planes(rows, batches, planes))
}

impl RunDelta {
    /// A δ from its planes, in [`DELTA_PLANES`] order.
    pub fn from_planes(
        rows: Vec<Box<str>>,
        batches: Vec<Box<str>>,
        [mean, sd, log_mean, log_sd]: [Mat; 4],
    ) -> Self {
        Self {
            rows,
            batches,
            mean,
            sd,
            log_mean,
            log_sd,
        }
    }

    /// The planes, in [`DELTA_PLANES`] order.
    pub fn planes(&self) -> [&Mat; 4] {
        [&self.mean, &self.sd, &self.log_mean, &self.log_sd]
    }

    /// A δ on `rows` and `batches` whose planes are `plane(p, this δ's
    /// plane p, its fill)`.
    pub(crate) fn map_planes(
        &self,
        rows: Vec<Box<str>>,
        batches: Vec<Box<str>>,
        plane: impl Fn(usize, &Mat, f32) -> Mat,
    ) -> Self {
        let mats = self.planes();
        Self::from_planes(
            rows,
            batches,
            std::array::from_fn(|p| plane(p, mats[p], FILL[p])),
        )
    }

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

    /// The mean alone on another matrix's `rows`, matched by name; a row
    /// the run did not read is no shift.
    pub fn mean_on_rows(&self, rows: &[Box<str>]) -> Mat {
        let idx = self.row_index(rows);
        Mat::from_fn(rows.len(), self.batches.len(), |i, j| {
            idx[i].map_or(FILL[0], |r| self.mean[(r, j)])
        })
    }

    /// This δ on another matrix's `rows`, matched by name, and which of
    /// them the run read; a row it did not read is no shift (mean 1, no
    /// spread).
    pub fn on_rows(&self, rows: &[Box<str>]) -> (Self, Vec<bool>) {
        let idx = self.row_index(rows);
        let on = self.map_planes(rows.to_vec(), self.batches.clone(), |_, m, fill| {
            Mat::from_fn(rows.len(), self.batches.len(), |i, j| {
                idx[i].map_or(fill, |r| m[(r, j)])
            })
        });
        (on, idx.iter().map(Option::is_some).collect())
    }

    /// This δ's columns `batches`, by label; a label it lacks is no shift.
    pub fn on_batches(&self, batches: &[Box<str>]) -> Self {
        let idx: Vec<Option<usize>> = batches
            .iter()
            .map(|b| self.batches.iter().position(|m| m == b))
            .collect();
        self.map_planes(self.rows.clone(), batches.to_vec(), |_, m, fill| {
            Mat::from_fn(self.rows.len(), batches.len(), |i, j| {
                idx[j].map_or(fill, |c| m[(i, c)])
            })
        })
    }

    /// Write this δ melted to `path`, in the fits' schema (`feature`,
    /// `batch`, `mean`, `sd`, `log_mean`, `log_sd`), so [`read_delta`]
    /// reads it back.
    pub fn write(&self, path: &str) -> anyhow::Result<()> {
        use legume_numeric::matrix::parquet::{write_named_table, Column};
        let n = self.rows.len() * self.batches.len();
        let (mut feature, mut batch) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut planes: [Vec<f32>; 4] = Default::default();
        let mats = self.planes();
        for (b, label) in self.batches.iter().enumerate() {
            for (g, row) in self.rows.iter().enumerate() {
                feature.push(row.clone());
                batch.push(label.clone());
                for (p, m) in planes.iter_mut().zip(mats) {
                    p.push(m[(g, b)]);
                }
            }
        }
        let mut cols = vec![("batch".into(), Column::Str(&batch))];
        cols.extend(
            DELTA_PLANES
                .iter()
                .zip(&planes)
                .map(|(name, p)| ((*name).into(), Column::F32(p))),
        );
        write_named_table(path, "feature", &feature, &cols)
    }
}

/// The δ a fitted run left at `prefix`, when it has one. With the run's
/// manifest (`run`), the file it records ([`Run::recorded_output`]: never
/// from the working directory, and an error when it is nowhere); a stale
/// file from an earlier fit under the same prefix is not read when it
/// records none.
///
/// [`Run::recorded_output`]: crate::util::metadata::Run::recorded_output
pub fn run_delta_path(run: &crate::util::metadata::Run) -> anyhow::Result<Option<String>> {
    let found = run.recorded_output(run.meta.outputs.batch_effects.as_deref(), "delta.parquet")?;
    Ok(found.map(|p| p.to_string_lossy().into_owned()))
}

/// The run's δ and the path it was read from, found as
/// [`run_delta_path`] finds it; `None` when the manifest records none.
pub fn read_run_delta(
    run: &crate::util::metadata::Run,
) -> anyhow::Result<Option<(RunDelta, String)>> {
    run_delta_path(run)?
        .map(|path| Ok((read_delta(&path)?, path)))
        .transpose()
}

/// Where a run records each cell's batch, when it does: the table its
/// manifest records, found as its δ is ([`run_delta_path`]). `None` for a
/// run that records none; an error when one is recorded but nowhere.
pub fn run_delta_cells_path(run: &crate::util::metadata::Run) -> anyhow::Result<Option<String>> {
    let found = run.recorded_output(
        run.meta.outputs.delta_cells.as_deref(),
        "delta_cells.parquet",
    )?;
    Ok(found.map(|p| p.to_string_lossy().into_owned()))
}

/// Where a fit writes each cell's batch label as δ's columns name it.
pub fn delta_cells_path(prefix: &str) -> String {
    format!("{prefix}.delta_cells.parquet")
}

/// `path`: one row per cell (`cell`), with the label of the δ column its
/// counts are divided by (`batch`), after `--auto-batch`.
fn write_cell_batches(
    path: &str,
    cells: &[Box<str>],
    batch_membership: &[Box<str>],
) -> anyhow::Result<()> {
    use legume_numeric::matrix::parquet::{write_named_table, Column};
    anyhow::ensure!(
        cells.len() == batch_membership.len(),
        "{path}: {} batch labels for {} cells",
        batch_membership.len(),
        cells.len()
    );
    write_named_table(
        path,
        "cell",
        cells,
        &[("batch".into(), Column::Str(batch_membership))],
    )
}

/// Each cell's batch label as the run estimated it, from the table
/// [`delta_cells_path`] names.
pub fn read_cell_batches(path: &str) -> anyhow::Result<HashMap<Box<str>, Box<str>>> {
    let (strs, _) = read_table_columns(path, &["cell", "batch"], &[])?;
    let [cell, batch]: [Vec<Box<str>>; 2] = strs
        .try_into()
        .map_err(|_| anyhow::anyhow!("{path}: expected cell and batch columns"))?;
    anyhow::ensure!(
        cell.len() == batch.len(),
        "{path}: columns of different lengths"
    );
    Ok(cell.into_iter().zip(batch).collect())
}

/// A run's batch effect with each cell's batch: what divides the counts a
/// run reports on, such as its feature x community dictionary
/// ([`divide_columns`]), and, per endpoint, cage's pairs.
#[derive(Copy, Clone)]
pub struct BatchDivisor<'a> {
    /// `[rows × batches]` multiplicative batch effect `δ` (its mean).
    pub delta: &'a Mat,
    /// Cell → its column of `delta`.
    pub batch_of_cell: &'a [u32],
}

impl<'a> BatchDivisor<'a> {
    /// The divisor of a run with a batch effect `delta` and its cells'
    /// batches, `None` when it has neither.
    pub fn of(delta: Option<&'a Mat>, batch_of_cell: Option<&'a [u32]>) -> Option<Self> {
        delta.zip(batch_of_cell).map(|(delta, batch_of_cell)| Self {
            delta,
            batch_of_cell,
        })
    }
}

/// What divides the counts a fit reads, on their nonzeros: a batch effect
/// `δ`, over the batches its estimate registered on the data, or with each
/// cell's batch given.
#[derive(Copy, Clone)]
pub enum Divisor<'a> {
    /// `δ`, `[rows × batches]`, each cell's column the batch its estimate
    /// registered on the data (`get_batch_membership`).
    Registered(&'a Mat),
    /// `δ` with each cell's batch.
    Batch(BatchDivisor<'a>),
}

impl<'a> From<BatchDivisor<'a>> for Divisor<'a> {
    fn from(d: BatchDivisor<'a>) -> Self {
        Divisor::Batch(d)
    }
}

impl Divisor<'_> {
    /// The rows it divides, to check against the data's.
    pub fn n_rows(&self) -> usize {
        match self {
            Divisor::Registered(delta) => delta.nrows(),
            Divisor::Batch(b) => b.delta.nrows(),
        }
    }

    /// The cells it knows, when it names them (`None` for a registered δ).
    pub fn n_cells(&self) -> Option<usize> {
        match self {
            Divisor::Registered(_) => None,
            Divisor::Batch(b) => Some(b.batch_of_cell.len()),
        }
    }

    /// Columns `cells` of `data`, read into `x`, each count divided by its
    /// divisor: a plain `x / δ` ([`divide_columns`]).
    pub fn divide(
        &self,
        data: &SparseIoVec,
        x: &mut nalgebra_sparse::CscMatrix<f32>,
        cells: &[usize],
    ) {
        match self {
            Divisor::Registered(delta) => {
                let batches = data.get_batch_membership(cells.iter().copied());
                divide_columns(x, delta, |j| Some(batches[j]));
            }
            Divisor::Batch(b) => {
                divide_columns(x, b.delta, |j| Some(b.batch_of_cell[cells[j]] as usize))
            }
        }
    }

    /// One cell's values over `rows` divided by the self-normalizing
    /// Poisson ratio. A registered δ names no cell's batch, so it has none.
    pub fn ratio_column(&self, rows: &[usize], vals: &mut [f32], cell: usize) {
        match self {
            Divisor::Registered(_) => {
                unreachable!("a registered δ names no cell's batch; pairs take a per-cell divisor")
            }
            Divisor::Batch(bd) => {
                let b = bd.batch_of_cell[cell] as usize;
                adjust_by_poisson_ratio(vals, |k| bd.delta[(rows[k], b)]);
            }
        }
    }
}

/// Each cell's column of a δ over the sorted `labels`
/// ([`sorted_batch_labels`] of `batch_membership`).
pub fn batch_ids(batch_membership: &[Box<str>], labels: &[Box<str>]) -> Vec<u32> {
    batch_membership
        .iter()
        .map(|b| {
            labels
                .binary_search(b)
                .expect("a cell's batch is one of the sorted labels") as u32
        })
        .collect()
}

/// Divide each count of `x` by its row's δ in its column's batch, a plain
/// `x / δ`: `batch_of_column(j)` is column `j`'s column of `delta`
/// (`[rows × batches]`), `None` to leave the column as it is (no batch, or
/// no use for it). A non-positive δ leaves its count as is.
///
/// lc's edge profiles and impute's reference cells divide this way; the
/// cage pair divisor does not, since it also divides by the cell's own
/// rate (a self-normalizing Poisson ratio, see `PairBatchDivisor`).
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
