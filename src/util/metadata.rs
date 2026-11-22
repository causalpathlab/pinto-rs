//! JSON metadata output for pinto runs.
//!
//! Writes a `{prefix}.pinto.json` file containing:
//! - Run parameters
//! - Output file paths (relative to prefix)
//! - Data statistics (n_cells, n_edges, n_communities, etc.)
//! - Hierarchical level information

use crate::util::cli::value_name;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

mod fit_manifests;
mod lra;
mod manifest_writing;
mod run_paths;
pub use fit_manifests::*;
pub use lra::{LraFiles, LraOptions, LraSubset};
pub(crate) use manifest_writing::*;
use manifest_writing::{locked, replace};
pub use run_paths::*;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PintoMetadata {
    pub command: String,
    pub version: String,
    pub timestamp: String,
    pub prefix: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_files: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub coord_file: Option<String>,

    pub n_cells: usize,
    /// One per feature-axis entry.
    pub n_features: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_edges: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_communities: Option<usize>,

    /// A cage fit on cells (`pinto fit cage-cell`): cells and cell pairs are
    /// its trained units, with no pseudobulk table. Batch effects are
    /// estimated and removed as in cage (unless `--ignore-batch`), so a
    /// reader corrects a query as it would for cage. Absent (false) on
    /// every other run.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cell_level: bool,

    /// Which graph the pairs came from. Absent on manifests written before
    /// this was recorded, and on runs that build no graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphParams>,

    /// Present only when the input's feature axis carried
    /// `{feature}/count/{spliced,unspliced}` rows. Absent means "no channels",
    /// which is not the same as "no features identified" — see
    /// [`SpliceTrackInfo::n_delta_identified`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub splice: Option<SpliceTrackInfo>,

    /// `pinto apply predict` / `impute`: how the query's cells were placed
    /// on the model's saved fold. Absent: divided by δ alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_fold_placement: Option<crate::util::batch_fold::BatchFoldPlacement>,

    pub outputs: OutputFiles,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub levels: Option<Vec<LevelInfo>>,

    /// Present on an ablation (`--permute-coords`, `--drop-features`,
    /// `--mask-cells`): what was taken away, from which parent run. pinto's
    /// own provenance; the run's `version` above is the pinto that made it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ablation: Option<crate::util::ablation::Ablation>,

    /// Present on `pinto apply predict` of an lc / dsvd model: the held-out
    /// scoring, what the model left out, and its log-likelihoods.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holdout: Option<crate::holdout::HoldoutInfo>,

    /// How the run treated its features (`--feature-modules`, cage's
    /// `--feature-inclusion moran`, lc's `--feature-network`): fitted on
    /// modules or on the features. Absent on manifests written before this
    /// was recorded, and on runs that fit nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_modules: Option<crate::util::feature_modules::FeatureModulesInfo>,

    /// What named the batches the run's `delta.parquet` is over, so a
    /// later sample's batches can be matched to them. Absent without a
    /// δ, and on manifests written before this was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_labels: Option<BatchLabels>,

    /// With file batches ([`BatchLabels::Files`]), each data file's batch
    /// name and the file's path as given, in `data_files`' order. Absent
    /// otherwise, and on manifests written before this was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_batches: Option<Vec<FileBatch>>,

    /// Present on a spatial baseline's run (each baseline and its `-pb`
    /// form): its input layout and what it was fitted on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<BaselineRun>,

    /// The `--max-threads` an lc run had: its Gibbs split each component
    /// into up to this many blocks, so with `--seed` and the data it fixes
    /// the draws. Absent on other runs, and on manifests written before
    /// this was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_threads: Option<usize>,

    /// How the run read its rows' names (`--feature-name-mode`, resolved:
    /// never `auto`), so what reads its data back reads the rows it fitted.
    /// Absent on runs that take no such flag (lc, dsvd, the baselines: they
    /// read the names exactly), and on manifests written before this was
    /// recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_names: Option<crate::cell_activity_graph_embedding::args::FeatureNameMode>,
}

/// A spatial baseline's input layout and what it was fitted on. Each runs
/// pinto's sparse formulation; the `mode` an earlier pinto recorded is
/// read past.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BaselineRun {
    /// How the input was held, `dense` or `sparse`, for a method whose
    /// layout depends on the device (stagate: dense on a GPU).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_layout: Option<InputLayout>,
    /// `cells`, or the super-cells a `-pb` run was fitted on.
    pub input: BaselineInput,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum InputLayout {
    Dense,
    Sparse,
}

/// What a baseline was fitted on: the cells, or super-cells.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BaselineInput {
    Cells,
    Pseudobulk(PseudobulkInput),
}

/// The super-cells a `-pb` run was fitted on, and how their
/// graph was weighted.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PseudobulkInput {
    /// `-d --n-pseudobulk`, the target.
    pub n_pseudobulk: usize,
    /// Super-cells fitted on.
    pub n_super_cells: usize,
    /// Super-edges of the collapsed graph.
    pub n_super_edges: usize,
    /// What a super-edge's weight is.
    pub edge_weighting: String,
}

/// What a run's batch labels are: one per data file (named by the file),
/// the `--batch-files` labels, or `--auto-batch` components.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BatchLabels {
    /// One batch per data file, named by it
    /// ([`crate::util::input::data_file_name`]).
    #[serde(rename = "file-names")]
    Files,
    BatchFiles,
    AutoBatch,
}

impl BatchLabels {
    /// The labels a run with batch options `c` gives its cells
    /// (`--batch-files` wins over `--auto-batch`, as preprocessing has it).
    pub fn of(c: &crate::util::input::BatchArgs) -> Self {
        if c.batch_files.is_some() {
            Self::BatchFiles
        } else if c.auto_batch {
            Self::AutoBatch
        } else {
            Self::Files
        }
    }
}

/// A data file's batch, when the file is its own.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FileBatch {
    /// Its name ([`crate::util::input::data_file_name`]).
    pub name: String,
    /// The data file, as given.
    pub path: String,
}

/// Each of `files`' batch when the batches are labelled by file
/// (`labels`): see [`PintoMetadata::file_batches`].
pub fn file_batches(labels: Option<BatchLabels>, files: &[Box<str>]) -> Option<Vec<FileBatch>> {
    if labels != Some(BatchLabels::Files) {
        return None;
    }
    let names = crate::util::input::data_file_names(files).ok()?;
    Some(
        names
            .into_iter()
            .zip(files)
            .map(|(name, path)| FileBatch {
                name: name.into(),
                path: path.to_string(),
            })
            .collect(),
    )
}

/// Data files as a manifest records them: each canonical (absolute) when
/// it resolves, as given otherwise, so a later run finds them from any
/// working directory.
pub fn recorded_files(files: &[Box<str>]) -> Vec<String> {
    files
        .iter()
        .map(|f| {
            crate::util::lupin_rounds::canonical(Path::new(f.as_ref()))
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

/// The k's that decided the cell-pair graph.
///
/// `n_edges` says how many pairs a run produced; this says why. Without it the
/// only way to tell a spatial-only run from an augmented one is to open
/// `coord_pairs.parquet` and notice that `edge_kind` is missing — the column is
/// written only when a union happened. Recording the inputs makes a run
/// reproducible from its manifest alone.
///
/// To replay: `knn_base` is `-k` when the run had `coord_file`, and
/// `--knn-expr` when it did not — those are the same graph either way, since
/// without coordinates the expression graph IS the base graph. `knn_expr` is
/// the augmentation, and is meaningful only alongside a `coord_file`.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct GraphParams {
    /// Neighbours per cell in the base graph: spatial under `--coord`,
    /// expression without it.
    pub knn_base: usize,
    /// Neighbours per cell in the expression graph unioned in. `0` means none,
    /// which is the default under `--coord` and always the case without it.
    /// So `knn_expr > 0` is the test for "this run was augmented" — there is
    /// no second field saying so, because a stored copy could contradict this
    /// one.
    ///
    /// This is what was ASKED FOR, not what the union yielded: a scoped search
    /// whose every component is smaller than k contributes no edges and still
    /// reports a positive k here. Read `n_edges` against an unaugmented
    /// control to learn what the union actually added.
    pub knn_expr: usize,
    /// `--knn-expr-scope`, as its CLI spelling. Absent when no expression
    /// search ran — recording it there would assert a scoping decision the run
    /// never made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knn_expr_scope: Option<String>,
    /// `--reciprocal`: mutual-KNN instead of union-KNN. Applies to every graph
    /// the run builds and changes the edge set, so two runs differing only in
    /// this would otherwise emit identical blocks with different `n_edges`.
    pub reciprocal: bool,
    /// Expression mode's base graph (no `--coord`) was built on the
    /// batch-corrected projection: by a fit only when it estimated a δ, so
    /// its counts are divided too. A query of the model builds its graph
    /// alike. Absent: on the projection as read (spatial, one batch, no
    /// δ, `--ignore-batch` or `--auto-batch`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub base_batch_corrected: bool,
}

impl GraphParams {
    /// The graph a run built with resolved k's `knn`, its base graph on
    /// the batch-corrected projection or not (`base_batch_corrected`).
    pub fn of(knn: &crate::util::input::ResolvedKnn, base_batch_corrected: bool) -> Self {
        Self {
            base_batch_corrected,
            ..knn.into()
        }
    }
}

impl From<&crate::util::input::ResolvedKnn> for GraphParams {
    /// Built from the RESOLVED spec, not the raw flags, so the manifest records
    /// what the run did rather than what was typed. Those differ on every
    /// defaulted run, which is most of them.
    fn from(knn: &crate::util::input::ResolvedKnn) -> Self {
        Self {
            knn_base: knn.base,
            knn_expr: knn.augment,
            // Only recorded when a search actually ran. Writing it
            // unconditionally would assert a scoping decision the run never
            // made, which is the whole failure this struct exists to prevent.
            knn_expr_scope: (knn.augment > 0).then(|| value_name(&knn.scope)),
            reciprocal: knn.reciprocal,
            base_batch_corrected: false,
        }
    }
}

/// The base track for `delta` — see [`SpliceTrackInfo::delta_base`].
pub const DELTA_BASE_SPLICED: &str = "spliced";

/// What a splice-channelized input's two tracks can pin.
///
/// `n_features` on the parent counts FEATURES; a channelized matrix has two rows per
/// feature and every feature-side output is on the feature axis, so `n_rows` is the only
/// place the matrix's own shape survives.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpliceTrackInfo {
    /// Matrix rows behind the parent's `n_features`.
    pub n_rows: usize,
    /// Features carrying counts on BOTH tracks — the only ones for which a
    /// nascent-minus-mature contrast is identified at all. With no spliced
    /// counts only `beta + delta` is pinned; with no unspliced counts `delta`
    /// enters no likelihood term and would come straight from the prior.
    pub n_delta_identified: usize,
    /// Nascent share of the total library. The second half of the go/no-go: a
    /// negligible share means the contrast is nominally identified and
    /// practically empty.
    pub nascent_count_fraction: f64,
    /// Which track the deviation `delta` is measured FROM. Always `"spliced"`
    /// here: `unspliced = beta + delta`, the same convention `senna gem`'s
    /// contrast table uses.
    ///
    /// Recorded because the sign is NOT a convention every producer is
    /// guaranteed to share, so two contrast tables are comparable only after
    /// reading this field. Without it the tables look interchangeable and
    /// are not.
    pub delta_base: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct OutputFiles {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coord_pairs: Option<String>,

    /// Bare coordinate column basenames (without the `left_`/`right_`
    /// prefix) that the writer emitted alongside `coord_pairs`, in the
    /// order readers take them: `[x, y]`.
    /// `None` for a run without coordinates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coord_columns: Option<Vec<String>>,

    /// Every cell the run read, before QC: coordinates, batch, and whether it
    /// became a graph node (`{prefix}.cells.parquet`). `coord_pairs` only
    /// names cells with an edge. Absent on expression-only runs and on runs
    /// written before this table existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cells: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub propensity: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_community: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_community: Option<String>,

    /// Every feature_community the run wrote (its levels' too) was fitted
    /// on counts divided by its batch effect, `batch_effects`. Absent: on
    /// the counts as read (one batch or `--ignore-batch`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub feature_community_batch_adjusted: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub scores: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_effects: Option<String>,

    /// Each cell's batch label as δ's columns name it (after
    /// `--auto-batch`), beside `batch_effects`. Absent on runs fitted
    /// before it was written; their `coord_pairs` carry the labels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_cells: Option<String>,

    /// The state-aware batch fold the fit wrote beside δ
    /// ([`crate::util::batch_fold::BatchFoldFiles`]). Absent: it divided by
    /// δ alone, or predates the fold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_fold: Option<crate::util::batch_fold::BatchFoldFiles>,

    /// What every feature_community was fitted on: counts divided by δ, or
    /// by each cell's state-aware fold. Absent on runs before it was
    /// written: `feature_community_batch_adjusted` then means δ.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profiles_divided_by: Option<crate::util::batch_fold::ProfilesDividedBy>,

    /// Cosine dictionary-merge artifacts: the merge tree, and its consensus
    /// cut when `--merge-cut` was given. Absent when the merge did not run.
    /// Without a cut, or when the cut collapses nothing, the draft is the
    /// final partition. A merged consensus partition is published under the
    /// bare prefix
    /// (`{prefix}.{propensity,link_community,feature_community}.parquet`), so this
    /// struct only points at the auxiliary tree + cut files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dict_merge: Option<DictMergeFiles>,

    /// JSON sidecar from `pinto lra --method coexpression-test`. Optional;
    /// written only when that method is run against this prefix and emits
    /// per-significant-pair edge participation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_activity: Option<String>,

    /// Per-batch contact-association score table from
    /// `pinto lra --method coexpression-scores`. Optional; written only when
    /// that mode is run against this prefix. Descriptive phenotypes, not a
    /// test, so it is a separate slot from `lr_activity`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_scores: Option<String>,
    /// `pinto lra` contact-effect outputs and `pinto lra simulate`'s check
    /// ([`LraFiles`]).
    #[serde(default, skip_serializing_if = "LraFiles::is_empty")]
    pub lra: LraFiles,

    /// `pinto fit cage` cell embedding `[N × D]`: every cell's own placement on
    /// the feature embedding by the same map that places its pairs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_embedding: Option<String>,

    /// `pinto fit cage` per-cell bias `[N]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_bias: Option<String>,

    /// The pair encoder `pinto fit cage` fitted (safetensors), which `pinto
    /// predict` reloads so a new sample is placed by the run's own map.
    /// Absent on a run that solved every pair exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pair_encoder: Option<String>,

    /// Trained super-edge table: a row per pseudobulk (finest-level
    /// super-cell) pair, phase 1's training unit, keyed `"P-Q"`. The pair
    /// encoder is trained to reproduce it; cells and pairs are placed by
    /// the encoder afterwards.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pb_pair_embedding: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pb_bias: Option<String>,

    /// Cell -> finest-level PB id map.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_pb: Option<String>,

    /// `pinto fit cage` feature (feature) embedding `[G × D]` — same shared
    /// D-dim space as `cell_embedding`. Cosine similarity between
    /// feature rows is directly interpretable. The trained effects as-is:
    /// the selection gated the gradient, so it is already expressed here
    /// and must not be re-applied on top.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_embedding: Option<String>,

    /// `pinto fit cage` per-feature bias `[G]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_bias: Option<String>,

    /// `pinto apply predict` of an lc / dsvd model: one row per held-out unit, its
    /// held-out log-likelihood beside the pooled K = 1 profile's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predictive: Option<String>,
}

/// What the dictionary merge actually did, when it wrote its tree. `None`
/// when it was skipped. Present also when the tree was not cut, or when the
/// cut collapsed nothing; the `.draft.*` files exist only when a cut
/// collapsed communities.
#[derive(Clone, Copy, Debug)]
pub struct DictMergeSummary {
    pub min_nnz: usize,
    pub features_scored: usize,
    /// The cosine the merge tree was cut at (`--merge-cut`); `None` when
    /// it was not cut.
    pub cut_cosine: Option<f64>,
    /// Sections (crossed with batches, for a run with a batch effect)
    /// centred on a mean of their own before the merge, when more than one
    /// holds an edge.
    pub centred_sections: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DictMergeFiles {
    /// Full agglomerative merge tree (one row per merge step).
    pub merges: String,
    /// Per-fine-community consensus label produced by the cut; absent when
    /// the tree was not cut (no `--merge-cut`), so present exactly when
    /// `cut_cosine` is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cut: Option<String>,
    /// Detection cutoff the merge scored on. Chosen from the data, so a run
    /// is not reproducible from its outputs without it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_nnz: Option<usize>,
    /// How many features cleared that cutoff.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features_scored: Option<usize>,
    /// The cosine the tree was cut at (`--merge-cut`); absent when it was
    /// not cut.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cut_cosine: Option<f64>,
    /// Sections (crossed with batches, for a run with a batch effect)
    /// centred on a mean of their own before the merge; absent with a
    /// single one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centred_sections: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LevelInfo {
    pub tag: String,
    pub level_index: usize,
    pub propensity: String,
    /// Per-edge community parquet. Every subcommand that clusters cell
    /// pairs (lc / dsvd / cage) writes one; `None` is kept for
    /// runs that produce no per-edge table at all, such as a simulation's
    /// truth run, whose levels the viewer draws without edges.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_community: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_community: Option<String>,
    /// A name per community, `C{k}` → `community_names[k]`, for levels
    /// whose communities have names of their own (a simulation's planted
    /// domains). Absent: the viewer names them `C{k}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub community_names: Option<Vec<String>>,
}

impl PintoMetadata {
    /// The method that made the run, as `pinto run` names it: the command,
    /// except that a cell-level cage run is `cage-cell`. Read by the viewer.
    #[cfg(any(feature = "view", test))]
    pub fn method(&self) -> &str {
        if self.cell_level {
            "cage-cell"
        } else {
            &self.command
        }
    }

    /// Whether the run's expression-mode base graph was built on the
    /// batch-corrected projection, as its manifest records
    /// ([`GraphParams::base_batch_corrected`]).
    pub fn base_graph_batch_corrected(&self) -> bool {
        self.graph.as_ref().is_some_and(|g| g.base_batch_corrected)
    }

    /// Read a run's manifest, change it, write it back, holding the
    /// manifest's lock throughout ([`PintoMetadata::update`]). A missing
    /// manifest is left alone (at debug level); one that cannot be read or
    /// written is warned about and left as it was.
    ///
    /// Under `--atomic-outputs` the manifest is changed when this run's
    /// outputs are in place, the same way ([`crate::util::staging`]).
    pub fn backfill(path: &Path, set: impl FnOnce(&mut PintoMetadata) + Send + 'static) {
        let Some(set) = crate::util::staging::defer_update(path, set) else {
            return;
        };
        if !path.exists() {
            log::debug!("no upstream manifest at {} to update", path.display());
            return;
        }
        if let Err(e) = Self::update(path, set) {
            log::warn!("could not update {}: {e:#}", path.display());
        }
    }

    /// Read a run's manifest, set one output slot, write it back.
    ///
    /// Both `pinto lra` modes back-fill the upstream manifest this way;
    /// keeping it in one place stops the two call sites from drifting on
    /// error handling (one used to swallow write failures silently).
    pub fn backfill_output(path: &Path, set: impl FnOnce(&mut OutputFiles) + Send + 'static) {
        Self::backfill(path, |m| set(&mut m.outputs));
    }

    /// Read the manifest at `path`, change it and write it back as one step:
    /// the manifest's lock is held from the read to the write, so runs that
    /// update one manifest at once (lra and its power check on one fit, a
    /// sweep's cells) each see the others' changes and lose none.
    pub fn update(path: &Path, set: impl FnOnce(&mut PintoMetadata)) -> anyhow::Result<()> {
        locked(path, |path| {
            let mut meta = Self::read(path)?;
            set(&mut meta);
            replace(path, &meta.json()?)
        })
    }

    /// Write the manifest at `path`, under its lock and whole (a reader
    /// sees the old manifest or the new one, never part of one).
    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        let json = self.json()?;
        locked(path, |path| replace(path, &json))
    }

    /// [`PintoMetadata::write`] when no manifest is at `path`; one that is
    /// there, even one written since the caller looked, is kept. Under
    /// `--atomic-outputs` that is done when the run's outputs are in place
    /// ([`crate::util::staging`]).
    pub fn write_if_missing(&self, path: &Path) -> anyhow::Result<()> {
        if crate::util::staging::defer_write_if_missing(path, self) {
            return Ok(());
        }
        locked(path, |path| {
            if !path.exists() {
                replace(path, &self.json()?)?;
            }
            Ok(())
        })
    }

    fn json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let meta: PintoMetadata = serde_json::from_str(&json)?;
        Ok(meta)
    }

    /// The run's levels. Manifests from before `levels` name one
    /// propensity table; that is the only level, `final`.
    pub fn level_list(&self) -> Vec<LevelInfo> {
        match &self.levels {
            Some(levels) if !levels.is_empty() => levels.clone(),
            _ => self
                .outputs
                .propensity
                .iter()
                .map(|propensity| LevelInfo {
                    tag: "final".to_string(),
                    level_index: 0,
                    propensity: propensity.clone(),
                    link_community: self.outputs.link_community.clone(),
                    feature_community: self.outputs.feature_community.clone(),
                    community_names: None,
                })
                .collect(),
        }
    }
}
