//! The preprocessing's config and result types, and the small steps
//! around it: what a caller asks of [`super::preprocess_srt`] and what it
//! hands back, the δ estimate, and the graph algorithms navigate.

use crate::util::ablation::{Ablation, AblationArgs};
use crate::util::batch_effects::{
    estimate_and_write_batch_effects, load_cnv_cell_strata, BatchEffects, BatchPosterior,
    EstimateBatchArgs,
};
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use crate::util::graph_coarsen::SeedingParams;
use crate::util::input::{CollapseArgs, ResolvedKnn, SrtInputArgs};
use data_beans::aux::feature_names::FeatureNameKind;
use legume_numeric::matrix::knn_graph::{EdgeSource, KnnGraph};
///////////////////////////
// Config + result types //
///////////////////////////

pub struct SrtPreprocessConfig<'a> {
    pub common: &'a SrtInputArgs,
    /// The fit's ablation flags; `None` for a command that fits nothing
    /// (`predict`).
    pub ablation: Option<&'a AblationArgs>,
    /// Whether to estimate per-batch effects, and what to do with them.
    pub batch_effects: BatchEstimate,
    /// Whether, and how strictly, to resolve the FEATURE unit axis from the row
    /// names and fold the running statistics onto it in the same pass.
    pub feature_axis: FeatureAxisMode,
    /// Row-name canonicalization strategy. `None` falls back to
    /// `FeatureNameKind::Exact` (strict equality), matching the
    /// historical behaviour of `lc` / `svd`. `cage` passes
    /// `FeatureNameKind::Gene` or an `auto_detect`'d kind so feature
    /// symbols register as aliases of `ENSG..._SYMBOL` row names.
    pub feature_kind: Option<FeatureNameKind>,
    /// Compute the shared post-batch-correction cell projection and hand it
    /// back on [`SrtPreprocessed::cell_proj`]. `lc` and `cage` both need one
    /// for coarsening, so taking it here spares them a second full pass.
    pub cell_projection: bool,
    /// Under `--auto-batch`, checked once the graph's components have
    /// relabelled the batches, and once covariates have binned the cells,
    /// before δ is estimated over the labels: the cells in each (section,
    /// label) group and the number of labels. A fit whose memory the
    /// batches decide refuses here ([`AutoBatchCheck`]).
    pub auto_batch_memory_check: Option<&'a AutoBatchCheck<'a>>,
    /// Which state-aware batch fold to estimate beside δ, when there is one
    /// ([`crate::util::batch_fold`]).
    pub batch_fold: BatchFoldKind,
    /// A query's model's recorded covariates, which bin the query's cells
    /// for δ as the model's were ([`crate::util::covariates`]); a fit models
    /// its own under `--covariate`, and this is `None`.
    pub covariate_model: Option<&'a crate::util::covariates::CovariateModel>,
}

/// The state-aware batch fold a fit divides by, beside δ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchFoldKind {
    /// None: δ alone (queries, and fits that divide by nothing).
    None,
    /// Each cell's own fold: the baselines, on cells and on their
    /// pseudobulks (sums of divided cells).
    Cells,
    /// Each cell's fold and each cell pair's own fold, estimated on pair
    /// units over the given states (`--pair-fold-states`): lc, dsvd and
    /// cage, which model pairs.
    Pairs(crate::util::batch_fold::PairStates),
}

/// A check on δ's labels (`--auto-batch` components, covariate bins) before
/// δ is estimated over them ([`SrtPreprocessConfig::auto_batch_memory_check`]).
pub type AutoBatchCheck<'a> = dyn Fn(&[usize], usize) -> anyhow::Result<()> + 'a;

/// What preprocessing does about batch effects (nothing under
/// `--ignore-batch`, or with one batch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchEstimate {
    /// A fit's: estimate, and write the estimate to `{out}.delta.parquet`
    /// with each cell's batch beside it.
    Fit,
    /// Nothing: the caller divides by a fitted δ of its own, or by none.
    Skip,
    /// A query of a model with a δ (`predict`, which writes the δ it
    /// divides by itself): estimate, unwritten, unless every batch is
    /// known to the model's δ (`all_known`); then the projection is still
    /// corrected by batch, as an estimate would have it. Expression mode's
    /// base graph is built on the batch-corrected projection exactly when
    /// the model's was (`base_corrected`, its manifest's
    /// [`crate::util::metadata::GraphParams::base_batch_corrected`]).
    Query {
        all_known: bool,
        base_corrected: bool,
    },
}

/// Whether preprocessing estimates batch effects over `n_labels` distinct
/// labels, and whether, without an estimate, the batches are still
/// corrected for (two or more, every one known to a fitted model).
pub(crate) fn batch_plan(mode: BatchEstimate, ignore_batch: bool, n_labels: usize) -> (bool, bool) {
    match mode {
        _ if ignore_batch => (false, false),
        BatchEstimate::Fit
        | BatchEstimate::Query {
            all_known: false, ..
        } => (true, false),
        BatchEstimate::Skip => (false, false),
        BatchEstimate::Query {
            all_known: true, ..
        } => (false, n_labels > 1),
    }
}

/// How a caller wants the feature axis resolved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FeatureAxisMode {
    /// Stay on the matrix rows. `svd` stacks its channels there and never
    /// folds, so a feature axis would buy it nothing.
    Rows,
    /// Resolve a feature axis, and FAIL on a feature axis that only partly parses.
    /// For a consumer that pools a feature's tracks: a row whose track is unknown
    /// has no correct pooled answer.
    Strict,
    /// Resolve a feature axis, falling back to one unit per row when the axis is
    /// mixed. For a consumer that only filters and reports per feature, where one
    /// unit per row is a defined answer and aborting would reject multimodal
    /// matrices that used to work.
    Lenient,
}

/// Expression mode's base-graph projection, and whether it is corrected by
/// the cells' batch labels.
pub(super) struct BaseProjection {
    pub(super) proj: data_beans::alg::random_projection::RandColProjOut,
    pub(super) corrected: bool,
}

/// Spatial seeding's inputs (see [`seeding_inputs`]).
pub struct Seeding {
    pub plane: Mat,
    pub groups: Vec<u32>,
}

impl Seeding {
    pub fn params(&self) -> SeedingParams<'_> {
        SeedingParams {
            coordinates: &self.plane,
            groups: Some(&self.groups),
        }
    }
}

pub struct SrtPreprocessed {
    pub data_vec: SparseIoVec,
    pub coordinates: Mat,
    pub coordinate_names: Vec<Box<str>>,
    pub batch_membership: Vec<Box<str>>,
    /// Each kept cell's label for δ: its batch, made `batch|cvK` when
    /// covariates are binned ([`crate::util::covariates`]), else
    /// [`Self::batch_membership`]. δ is estimated over, divided by and
    /// corrected for these; sections and pairs keep the plain labels.
    pub delta_membership: Vec<Box<str>>,
    /// Whether covariates were binned into [`Self::delta_membership`].
    pub covariates_binned: bool,
    /// Each kept cell's section: one per data file, crossed with the
    /// --batch-files labels as read (not the --auto-batch components).
    pub section_ids: Vec<u32>,
    /// What spatial seeding grids over, for a run with coordinates and
    /// enough cells to be seeded. Under `--permute-coords` its positions are
    /// the scrambled ones the graph was built on, or real-space structure
    /// would leak into the ablation; `coordinates` stays the cells' own.
    pub seeding: Option<Seeding>,
    /// The labels of [`Self::delta_membership`], sorted and distinct: δ's
    /// columns, when there is one.
    pub batch_labels: Vec<Box<str>>,
    /// The batch-effect posterior, `[n_rows × batch_labels]`. `None` when
    /// single-batch or not estimated.
    pub batch_effects: Option<BatchPosterior>,
    /// Each cell's state-aware fold (rows of δ), with [`BatchFoldKind::Cells`]
    /// or [`BatchFoldKind::Pairs`] and a δ.
    pub cell_fold: Option<crate::util::batch_fold::CellFold>,
    /// Each pair of [`Self::graph`]'s edges' own fold, in edge order, with
    /// [`BatchFoldKind::Pairs`] and a δ.
    pub pair_fold: Option<crate::util::batch_fold::PairFold>,
    /// The fold tables written beside δ (only for [`BatchFoldKind::Pairs`]).
    pub fold_written: crate::util::batch_fold::FoldWritten,
    /// Expression mode's base graph was built on the batch-corrected
    /// projection, for the manifest's
    /// [`crate::util::metadata::GraphParams::base_batch_corrected`].
    pub base_batch_corrected: bool,
    pub graph: KnnGraph,
    /// The k's [`Self::graph`] was actually built with. Carried rather than
    /// re-derived from the args, so a manifest cannot describe a graph the run
    /// did not build.
    pub knn: ResolvedKnn,
    /// The pre-augmentation SPATIAL graph, `Some` only when `--knn-expr`
    /// added expression edges. Prefer [`topology_graph`] over reaching
    /// for this directly.
    pub spatial_graph: Option<KnnGraph>,
    /// Which graph each edge of [`Self::graph`] came from, parallel to
    /// `graph.edges`. `None` when no augmentation ran.
    pub edge_source: Option<Vec<EdgeSource>>,
    /// The projection taken here, with the SAME batch argument `lc` and
    /// `cage` would have used, so they can reuse it instead of paying for a
    /// second full pass. `None` unless the config asked for one.
    pub cell_proj: Option<data_beans::alg::random_projection::RandColProjOut>,
    /// The feature unit axis. `Some` iff the config asked for it. Identity (one
    /// feature per row) unless the rows carry splice channels.
    pub feature_axis: Option<FeatureAxis>,
    pub n_cells: usize,
    /// Matrix rows. Equal to the feature count only on a non-channelized axis.
    pub n_rows: usize,
    /// Under `--permute-coords`, the scrambled positions the graph was
    /// built on (`coordinates` stays the cells' own, for output): what a
    /// fit that reads positions itself (a spatial baseline) fits on.
    pub permuted_coordinates: Option<Mat>,
    /// What `--permute-coords`, `--drop-features` or `--mask-cells` did, for
    /// the manifest's `ablation` block. `None` on an ordinary fit.
    pub ablation: Option<Ablation>,
}

/// The batch-effect estimate under collapse options `c` over `batch_membership` (its
/// sorted labels `batch_labels`), written under `out` when given (a fit);
/// the projection it collapsed on kept with `keep_projection`.
pub(crate) fn estimate_batches(
    c: &CollapseArgs,
    out: Option<&str>,
    data_vec: &mut SparseIoVec,
    batch_membership: &[Box<str>],
    batch_labels: &[Box<str>],
    keep_projection: bool,
) -> anyhow::Result<Option<BatchEffects>> {
    let strata = match c.cnv_clones.as_deref() {
        Some(path) => Some(load_cnv_cell_strata(path, data_vec)?),
        None => None,
    };
    estimate_and_write_batch_effects(
        data_vec,
        batch_membership,
        batch_labels,
        EstimateBatchArgs::at(
            c.proj_dim,
            crate::util::input::BATCH_KNN,
            c.num_levels,
            strata,
        ),
        out,
        keep_projection,
    )
}

/// The graph neighbourhood algorithms must navigate, given the modelled graph
/// and the pre-augmentation spatial one.
///
/// The union is the edge set being MODELLED. It is the wrong graph to walk.
/// Seeding floods along adjacency, so over expression edges a seed can jump to
/// a similar cell anywhere in the tissue. Refinement asks whether removing a
/// node would disconnect its cluster, and expression edges can only ever add
/// connections, so the check weakens as they are added. How far it weakens
/// depends on `--knn-expr` and on the data; the point is that neither question
/// is being asked about adjacency any more. Both take the spatial graph.
///
/// In expression mode this hands back the expression graph, because that is
/// the only graph there is. Nothing is lost: a run without `--coord` has no
/// adjacency to preserve, so there is no distinction for this to draw.
///
/// A free function rather than a method because every caller destructures
/// [`SrtPreprocessed`] field by field, which is deliberate: adding a field
/// should force each call site to say what it wants.
pub fn topology_graph<'a>(
    graph: &'a KnnGraph,
    spatial_graph: &'a Option<KnnGraph>,
) -> &'a KnnGraph {
    spatial_graph.as_ref().unwrap_or(graph)
}
