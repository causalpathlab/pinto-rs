//! Shared SRT preprocessing pipeline.
//!
//! `lc`, `dsvd`, and `cage` all share the same opening sequence:
//! load → invariants → spatial-or-expression KNN → optional auto-batch →
//! optional batch effects → optional NB Fisher feature weights.
//! `preprocess_srt` extracts that once.
//!
//! `SrtCellPairs<'a>` borrows the `SparseIoVec` and `Mat`, so this bundle
//! returns the *owned* data + graph, and the caller builds its pairs from
//! `SrtCellPairs::with_graph`, passing the data, coordinates, graph and edge
//! source it finds here.

use crate::util::ablation::{
    cell_keep, cells_in_regions, feature_keep, induced_subgraph, is_region_file, permute_blocks,
    permute_within, plane_columns, read_cell_list, read_regions, sections, seeding_inputs,
    Ablation, AblationArgs, AblationKind, MaskMode, PermuteCoords,
};
use crate::util::batch_effects::{
    estimate_and_write_batch_effects, load_cnv_cell_strata, EstimateBatchArgs,
};
use crate::util::cell_pairs::{
    build_expression_graph, build_expression_knn, build_expression_knn_within, build_spatial_graph,
    connected_components, SrtCellPairsArgs,
};
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use crate::util::graph_coarsen::{SeedingParams, SEEDING_THRESHOLD};
use crate::util::input::{
    auto_batch_from_components, read_data_with_coordinates, read_data_without_coordinates,
    KnnExprScope, ResolvedKnn, SRTData, SrtInputArgs,
};
use crate::util::knn_graph::KnnGraph;
use data_beans::alg::gene_weighting::fisher_weights_from_stats;
use data_beans::alg::random_projection::RandProjOps;
use data_beans::alg::sparse_streaming::streaming_sparse_running_stats;
use data_beans::aux::feature_names::FeatureNameKind;
use legume_numeric::matrix::knn_graph::{DistanceMerge, EdgeSource};

///////////////////////////
// Config + result types //
///////////////////////////

pub struct SrtPreprocessConfig<'a> {
    pub common: &'a SrtInputArgs,
    /// The fit's ablation flags; `None` for a command that fits nothing
    /// (`predict`).
    pub ablation: Option<&'a AblationArgs>,
    /// Compute per-feature NB Fisher-info weights. `lc` needs them; `svd`
    /// and (v1) `cage` do not.
    pub fisher_weights: bool,
    /// Estimate per-batch effects. All current subcommands set this.
    pub batch_effects: bool,
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
    /// What spatial seeding grids over, for a run with coordinates and
    /// enough cells to be seeded. Under `--permute-coords` its positions are
    /// the scrambled ones the graph was built on, or real-space structure
    /// would leak into the ablation; `coordinates` stays the cells' own.
    pub seeding: Option<Seeding>,
    /// Posterior batch-effect mean `[n_features × n_batches]`. `None`
    /// when single-batch.
    pub batch_effects: Option<Mat>,
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
    /// Per-ROW NB Fisher-info weights. This is the right axis for the
    /// projection and for anything that reads the matrix directly.
    pub row_weights: Option<Vec<f32>>,
    /// The ROW-axis statistics [`Self::row_weights`] came from. Its `sum()` is
    /// the per-row count total, which spares a caller that needs one a second
    /// full pass over the data.
    pub row_stats: Option<legume_numeric::matrix::sparse_stat::SparseRunningStatistics<f32>>,
    /// Per-FEATURE NB Fisher-info weights, `Some` iff both `fisher_weights` and
    /// `feature_axis` were asked for.
    ///
    /// These are NOT a fold of [`Self::row_weights`]. A Fisher weight is a
    /// function of a feature's abundance and mean, and that function is not
    /// additive, so folding the weights would hand a feature a precision no
    /// measurement supports. The fold happens on the statistics instead, and it
    /// happens inside the streaming pass because `npos` and `s2` do not survive
    /// one applied afterwards.
    pub feature_weights: Option<Vec<f32>>,
    /// The feature-axis statistics [`Self::feature_weights`] came from. Carried out
    /// because the dictionary merge needs per-feature DETECTION counts, which is
    /// exactly the statistic a post-hoc fold gets wrong.
    pub feature_stats: Option<legume_numeric::matrix::sparse_stat::SparseRunningStatistics<f32>>,
    pub n_cells: usize,
    /// Matrix rows. Equal to the feature count only on a non-channelized axis.
    pub n_rows: usize,
    /// What `--permute-coords`, `--drop-features` or `--mask-cells` did, for
    /// the manifest's `ablation` block. `None` on an ordinary fit.
    pub ablation: Option<Ablation>,
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

////////////////////
// preprocess_srt //
////////////////////

/// Run the shared SRT preamble: load, build the cell-cell KNN graph,
/// optionally auto-detect batches from disconnected components, estimate
/// per-batch effects, and (optionally) compute NB Fisher feature weights.
///
/// In expression mode (no `--coord`), the cell embedding used to build
/// the graph is the random-projected count matrix with no batch
/// correction (batch effects haven't been estimated yet at this point).
pub fn preprocess_srt(cfg: SrtPreprocessConfig<'_>) -> anyhow::Result<SrtPreprocessed> {
    let c = cfg.common;

    // Argv-only, so it runs BEFORE anything is read. A typo like a
    // `--knn-expr-scope` without `--coord` needs no data to judge, and making
    // the user wait through a load, a possible zarr conversion and a full QC
    // pass to hear about it is time spent on an answer already known.
    c.validate_knn_flags()?;
    let no_ablation = AblationArgs::default();
    let ab = cfg.ablation.unwrap_or(&no_ablation);
    ab.validate(c.has_coordinates())?;

    info!("Loading data files...");
    let has_coords = c.has_coordinates();

    let kind = cfg.feature_kind.unwrap_or(FeatureNameKind::Exact);
    let SRTData {
        data: mut data_vec,
        mut coordinates,
        mut coordinate_names,
        batches: mut batch_membership,
    } = if has_coords {
        read_data_with_coordinates(c.to_read_args_with_kind(kind.clone()))?
    } else {
        info!("No coordinate files provided — using expression mode");
        read_data_without_coordinates(c.to_read_args_with_kind(kind))?
    };

    // The per-cell table written at the end names every cell read, so the
    // cells QC drops are set aside here (the kept ones are still in hand at
    // the end). Its batch column is the sample a cell was READ from, not an
    // `--auto-batch` component: components of one section share its
    // coordinate frame, and only samples overlap and need telling apart.
    let multi_sample = {
        let first = batch_membership.first();
        batch_membership.iter().any(|b| Some(b) != first)
    };
    let mut dropped: Option<DroppedCells> = None;
    let mut ablation: Option<Ablation> = None;

    // Each cell's section: one id per data file and batch label, taken while
    // the cells are still in file order (QC may later empty a section and
    // leave its id unused), and kept in step with the cells from here on. A permutation keeps cells inside their section, and
    // spatial seeding lays a grid over each section's own frame.
    let mut section_ids: Vec<u32> = sections(&data_vec.num_columns_by_data()?, &batch_membership);

    // Dropped features never reach QC, so the cells it keeps are judged on
    // the features the ablation fits.
    if let Some(spec) = ab.drop_features.as_deref() {
        let (keep, spec) = feature_keep(&data_vec.row_names()?, spec)?;
        info!(
            "Ablation: dropping {} of {} features",
            spec["n_dropped"], spec["n_rows"]
        );
        data_vec.mask_rows(&keep)?;
        ablation = Some(ab.record(AblationKind::FeatureDrop, spec, None));
    }

    // Optional shared cell QC — applied before the KNN graph is built so
    // dropped cells never become graph nodes. MAD outliers are dropped from
    // the working set; coordinates + batch labels are filtered in lockstep.
    // (Near-empty cells are kept; the gem-style near-empty output floor is
    // not applied to pinto's per-cell outputs.)
    if let Some(qc_cfg) = c.qc.to_config() {
        let report = data_beans::qc_lib::compute_qc(&data_vec, &qc_cfg, c.block_size)?;
        // Before masking: the report is indexed by the ORIGINAL cell order, so
        // the names have to be read while they still line up. Read once, for
        // the report and for the cells set aside below.
        let names = if c.qc.qc_report.is_some() || (has_coords && report.n_cells_dropped > 0) {
            Some(data_vec.column_names()?)
        } else {
            None
        };
        if let (Some(path), Some(names)) = (c.qc.qc_report.as_deref(), names.as_ref()) {
            data_beans::qc_lib::write_qc_report(path, names, &report)?;
            info!("Wrote QC report to {}", path);
        }
        let n_near_empty = report.near_empty.iter().filter(|&&e| e).count();
        info!(
            "QC: dropped {}/{} cells from the spatial graph ({} near-empty kept)",
            report.n_cells_dropped,
            report.train_keep.len(),
            n_near_empty,
        );
        if report.n_cells_dropped > 0 {
            if let (true, Some(names)) = (has_coords, names) {
                let rows: Vec<usize> = (0..report.train_keep.len())
                    .filter(|&i| !report.train_keep[i])
                    .collect();
                dropped = Some(DroppedCells {
                    names: rows.iter().map(|&i| names[i].clone()).collect(),
                    coordinates: coordinates.select_rows(rows.iter()),
                    coordinate_names: coordinate_names.clone(),
                    batches: rows.iter().map(|&i| batch_membership[i].clone()).collect(),
                });
            }
            let kept: Vec<usize> = report
                .train_keep
                .iter()
                .enumerate()
                .filter(|&(_, &k)| k)
                .map(|(i, _)| i)
                .collect();
            // Spatial mode: coordinates [n_cells × n_dims] are consumed by
            // the graph build below, so they must be filtered in lockstep
            // (error loudly on any size mismatch rather than silently skip).
            // Expression mode overwrites `coordinates` with the embedding of
            // the already-masked data_vec, so no filtering is needed there.
            if has_coords {
                anyhow::ensure!(
                    coordinates.nrows() == report.train_keep.len(),
                    "QC: coordinate rows {} != cell count {}",
                    coordinates.nrows(),
                    report.train_keep.len()
                );
                coordinates = coordinates.select_rows(kept.iter());
            }
            data_vec.mask_columns(&report.train_keep)?;
            batch_membership =
                data_beans::qc_lib::filter_by_keep(&batch_membership, &report.train_keep);
            section_ids = data_beans::qc_lib::filter_by_keep(&section_ids, &report.train_keep);
        }
    }

    // Masked cells go after QC. Under cut-edges (with coordinates) they stay
    // for the graph build and are taken out with their edges right after it.
    let mut cut_after_graph: Option<Vec<bool>> = None;
    if let Some(path) = ab.mask_cells.as_deref() {
        let names = data_vec.column_names()?;
        let (listed, regions) = if is_region_file(path) {
            let regions = read_regions(path)?;
            let (xc, yc) = plane_columns(&coordinate_names)?;
            let x: Vec<f32> = coordinates.column(xc).iter().copied().collect();
            let y: Vec<f32> = coordinates.column(yc).iter().copied().collect();
            let (cells, counts) = cells_in_regions(&names, &x, &y, &regions)?;
            info!(
                "Ablation: {} polygon(s) hold {:?} cells",
                regions.len(),
                counts
            );
            (cells, Some(regions.len()))
        } else {
            (read_cell_list(path)?, None)
        };
        let keep = cell_keep(&names, &listed)?;
        let n_masked = keep.iter().filter(|&&k| !k).count();
        let cut = has_coords && ab.mask_mode == MaskMode::CutEdges;
        info!(
            "Ablation: masking {} of {} cells ({})",
            n_masked,
            keep.len(),
            if cut {
                "their edges cut after the graph is built"
            } else {
                "left out of the graph build"
            }
        );
        ablation = Some(ab.record(
            AblationKind::CellMask,
            serde_json::json!({
                "file": path,
                "n_polygons": regions,
                "mode": if cut { "cut-edges" } else { "drop" },
                "n_masked": n_masked,
                "n_cells": keep.len(),
            }),
            None,
        ));
        if cut {
            cut_after_graph = Some(keep);
        } else {
            mask_cells(
                &keep,
                &mut data_vec,
                &mut coordinates,
                &coordinate_names,
                &mut batch_membership,
                &mut section_ids,
                has_coords.then_some(&mut dropped),
            )?;
        }
    }

    let n_rows = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    anyhow::ensure!(c.proj_dim > 0, "proj_dim must be > 0");
    // Both k's resolve here, before either graph is built. The KNN builder
    // floors k at 1 and clamps it to n_cells-1 rather than rejecting either
    // extreme, so nothing downstream would catch a 0 or an over-large k.
    let knn = c.resolve_knn(n_cells)?;
    let base_knn = knn.base;

    // The positions the graph is built on: the cells' own, or scrambled ones
    // under `--permute-coords`. The graph and spatial seeding see the
    // scrambled ones; `coordinates` stays the cells' own, for output.
    let permuted: Option<Mat> = match ab.permute_coords {
        Some(mode) => {
            let seed = ab.seed(c.seed);
            let n_sections = section_ids.iter().collect::<HashSet<_>>().len();
            let (moved, spec) = match mode {
                PermuteCoords::WithinBatch => (
                    permute_within(&coordinates, &section_ids, seed),
                    serde_json::json!({ "mode": "within-batch", "n_sections": n_sections }),
                ),
                PermuteCoords::Block => {
                    let xy = plane_columns(&coordinate_names)?;
                    let (moved, n_tiles) =
                        permute_blocks(&coordinates, xy, &section_ids, ab.permute_block, seed);
                    (
                        moved,
                        serde_json::json!({
                            "mode": "block",
                            "per_side": ab.permute_block,
                            "n_tiles": n_tiles,
                            "n_sections": n_sections,
                        }),
                    )
                }
            };
            info!("Ablation: graph built on permuted positions ({spec})");
            ablation = Some(ab.record(AblationKind::CoordPermutation, spec, Some(seed)));
            Some(moved)
        }
        _ => None,
    };

    let graph = if has_coords {
        info!("Building spatial KNN graph (k={})...", base_knn);
        build_spatial_graph(
            permuted.as_ref().unwrap_or(&coordinates),
            SrtCellPairsArgs {
                knn: base_knn,
                block_size: c.block_size,
                reciprocal: knn.reciprocal,
            },
        )?
    } else {
        info!(
            "Building expression KNN graph (k={}, proj_dim={})...",
            base_knn, c.proj_dim
        );
        let cell_proj_pre = data_vec.project_columns_with_batch_correction(
            c.proj_dim,
            c.block_size,
            None::<&[Box<str>]>,
        )?;
        let (g, embedding) = build_expression_graph(
            &cell_proj_pre.proj,
            SrtCellPairsArgs {
                knn: base_knn,
                block_size: c.block_size,
                reciprocal: knn.reciprocal,
            },
        )?;
        coordinates = embedding;
        coordinate_names = vec!["pc_1".into(), "pc_2".into()];
        g
    };

    let (graph, n_cells) = match cut_after_graph {
        Some(keep) => {
            let graph = induced_subgraph(&graph, &keep);
            mask_cells(
                &keep,
                &mut data_vec,
                &mut coordinates,
                &coordinate_names,
                &mut batch_membership,
                &mut section_ids,
                Some(&mut dropped),
            )?;
            info!(
                "Ablation: {} cells and {} edges left after the cut",
                graph.n_nodes,
                graph.edges.len()
            );
            (graph, data_vec.num_columns())
        }
        None => (graph, n_cells),
    };

    // The kept cells' read-time labels, before `--auto-batch` relabels them.
    let read_batches = (has_coords && multi_sample).then(|| batch_membership.clone());

    if c.auto_batch && c.batch_files.is_none() {
        auto_batch_from_components(&graph, &coordinates, &mut batch_membership, has_coords);
    }

    // `--ignore-batch` drops the estimate, and with it every use of one
    // (the projection's correction below, a model's own divide). The labels
    // stay: they also say which section a cell is in, for the pair table,
    // the viewer and the models' own bookkeeping.
    let batch_effects = if cfg.batch_effects && !c.ignore_batch {
        let batch_sort_dim = c.proj_dim.min(10);
        let strata = match c.cnv_clones.as_deref() {
            Some(path) => Some(load_cnv_cell_strata(path, &data_vec)?),
            None => None,
        };
        estimate_and_write_batch_effects(
            &mut data_vec,
            &batch_membership,
            EstimateBatchArgs {
                proj_dim: c.proj_dim,
                sort_dim: batch_sort_dim,
                block_size: c.block_size,
                batch_knn: c.batch_knn,
                num_levels: c.num_levels,
                strata,
            },
            &c.out,
        )?
    } else {
        None
    };

    // Augmentation happens AFTER this point, never before, for two reasons.
    //
    // `auto_batch_from_components` above must see the spatial graph. It splits
    // batches by connected component, and expression edges bridge sections
    // freely, so on a multi-section slide the union will typically collapse to
    // one component. Then it reports one batch, batch-effect estimation is
    // skipped, and every tissue core lands in one frame downstream. That
    // failure is silent, which is why the ordering here is load-bearing.
    //
    // That argument is about the UNION, and so about spatial runs. In
    // expression mode the base graph is itself an expression graph and
    // auto-batch sees it by construction — there a component is not a section
    // and was never claimed to be, which is why that path folds no fragments.
    //
    // And the expression graph should be built on a batch-corrected
    // projection, or its neighbours match batch rather than cell type.
    let batch_arg: Option<&[Box<str>]> = batch_effects
        .is_some()
        .then_some(batch_membership.as_slice());

    // `ResolvedKnn::augment` is already 0 without coordinates, so these two
    // tests stay in step by construction rather than by both remembering to
    // say `&& has_coords`. That agreement is what the `expect` below rests on.
    let augment_knn = knn.augment;

    let cell_proj = if cfg.cell_projection || augment_knn > 0 {
        Some(data_vec.project_columns_with_batch_correction(c.proj_dim, c.block_size, batch_arg)?)
    } else {
        None
    };

    let (graph, spatial_graph, edge_source) = if augment_knn == 0 {
        (graph, None, None)
    } else {
        info!(
            "Adding expression KNN edges (k={}, proj_dim={})...",
            augment_knn, c.proj_dim
        );
        let proj = cell_proj
            .as_ref()
            .expect("the projection above is taken whenever augment_knn() > 0");
        let knn_args = SrtCellPairsArgs {
            knn: augment_knn,
            block_size: c.block_size,
            reciprocal: knn.reciprocal,
        };
        let expr_graph = match knn.scope {
            KnnExprScope::Global => build_expression_knn(&proj.proj, knn_args)?,
            KnnExprScope::Within => {
                let (component_of_cell, n_components) = connected_components(&graph);
                if n_components > 1 {
                    info!(
                        "Searching expression neighbours inside each of {} spatial \
                         components, so they stay within a sample",
                        n_components
                    );
                }
                build_expression_knn_within(&proj.proj, &component_of_cell, n_components, knn_args)?
            }
        };
        let (merged, source) = graph.union_with(&expr_graph, DistanceMerge::SourceRank)?;
        let (mut n_spatial, mut n_expr, mut n_both) = (0usize, 0usize, 0usize);
        for s in source.iter() {
            match s {
                EdgeSource::Primary => n_spatial += 1,
                EdgeSource::Secondary => n_expr += 1,
                EdgeSource::Both => n_both += 1,
            }
        }
        info!(
            "{} cell pairs after augmentation: {} spatial, {} expression, {} shared",
            merged.edges.len(),
            n_spatial + n_both,
            n_expr,
            n_both
        );
        (merged, Some(graph), Some(source))
    };

    // What a ROW means, decided once, before anything keyed on a feature runs.
    let feature_axis = match cfg.feature_axis {
        FeatureAxisMode::Rows => None,
        FeatureAxisMode::Strict => Some(FeatureAxis::resolve(&data_vec.row_names()?)?),
        FeatureAxisMode::Lenient => Some(FeatureAxis::resolve_or_identity(&data_vec.row_names()?)?),
    };

    // One streaming pass, up to four consumers. `compute_nb_fisher_weights`
    // builds these same statistics and returns only the weights, dropping
    // `npos`, which is exactly the per-feature detection count the dictionary
    // merge keys on. Taking the statistics here saves a second full read.
    //
    // On a channelized axis the pass folds as it goes rather than afterwards,
    // because `npos` and `s2` do not survive a post-hoc fold: a cell detected
    // on both tracks would count twice, and the variance would lose its cross
    // term. Those are the two statistics the merge filter and the dispersion
    // trend respectively depend on.
    let mut row_weights = None;
    let mut row_stats = None;
    let mut feature_weights = None;
    let mut feature_stats = None;
    if cfg.fisher_weights {
        info!("Computing NB Fisher-info weights for inference...");
        match feature_axis.as_ref() {
            Some(axis) => {
                let (r_stats, g_stats) =
                    axis.running_stats(&data_vec, c.block_size, "NB-Fisher")?;
                let rw = fisher_weights_from_stats(&r_stats, n_cells);
                let gw = fisher_weights_from_stats(&g_stats, n_cells);
                log_weights("Row weights w_r", &rw);
                log_weights("Feature weights w_g", &gw);
                row_weights = Some(rw);
                row_stats = Some(r_stats);
                feature_weights = Some(gw);
                feature_stats = Some(g_stats);
            }
            None => {
                let stats = streaming_sparse_running_stats(&data_vec, c.block_size, "NB-Fisher")?;
                let rw = fisher_weights_from_stats(&stats, n_cells);
                log_weights("Row weights w_r", &rw);
                row_weights = Some(rw);
                row_stats = Some(stats);
            }
        }
    }

    // Each section's own plane, crossed with the final batch labels; under
    // --permute-coords, the scrambled positions the graph was built on.
    let seeding = (has_coords && n_cells > SEEDING_THRESHOLD).then(|| {
        let (plane, groups) = seeding_inputs(
            permuted.as_ref().unwrap_or(&coordinates),
            &coordinate_names,
            &section_ids,
            &batch_membership,
        );
        Seeding { plane, groups }
    });

    if has_coords {
        write_cells_table(
            &c.out,
            &data_vec.column_names()?,
            &coordinates,
            &coordinate_names,
            read_batches.as_deref(),
            dropped.as_ref(),
        )?;
    }

    Ok(SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        seeding,
        batch_effects,
        graph,
        knn,
        spatial_graph,
        edge_source,
        cell_proj,
        feature_axis,
        row_weights,
        row_stats,
        feature_weights,
        feature_stats,
        n_cells,
        n_rows,
        ablation,
    })
}

/// Take the cells `keep` rules out of the working set: the data, the
/// coordinates and the batch labels in lockstep, and (with coordinates) set
/// aside for the cells table as cells not in the graph.
fn mask_cells(
    keep: &[bool],
    data_vec: &mut SparseIoVec,
    coordinates: &mut Mat,
    coordinate_names: &[Box<str>],
    batch_membership: &mut Vec<Box<str>>,
    section_ids: &mut Vec<u32>,
    dropped: Option<&mut Option<DroppedCells>>,
) -> anyhow::Result<()> {
    // `dropped` is given exactly when the run has coordinates. Without them
    // the coordinates are a placeholder (QC does not filter it, and the
    // expression embedding replaces it after the graph), so it is left alone.
    let with_coords = dropped.is_some();
    anyhow::ensure!(
        batch_membership.len() == keep.len() && (!with_coords || coordinates.nrows() == keep.len()),
        "mask: {} coordinate rows and {} batch labels for {} cells",
        coordinates.nrows(),
        batch_membership.len(),
        keep.len()
    );
    let out: Vec<usize> = (0..keep.len()).filter(|&i| !keep[i]).collect();
    let kept: Vec<usize> = (0..keep.len()).filter(|&i| keep[i]).collect();
    if let Some(dropped) = dropped {
        let names = data_vec.column_names()?;
        let set_aside = DroppedCells {
            names: out.iter().map(|&i| names[i].clone()).collect(),
            coordinates: coordinates.select_rows(out.iter()),
            coordinate_names: coordinate_names.to_vec(),
            batches: out.iter().map(|&i| batch_membership[i].clone()).collect(),
        };
        *dropped = Some(match dropped.take() {
            None => set_aside,
            Some(qc) => qc.join(set_aside)?,
        });
    }
    if with_coords {
        *coordinates = coordinates.select_rows(kept.iter());
    }
    data_vec.mask_columns(keep)?;
    *batch_membership = data_beans::qc_lib::filter_by_keep(batch_membership, keep);
    *section_ids = data_beans::qc_lib::filter_by_keep(section_ids, keep);
    Ok(())
}

/// One log line per weight vector, so a channelized run shows both axes and a
/// reader can see at a glance whether the two disagree.
fn log_weights(label: &str, w: &[f32]) {
    info!(
        "{}: min={:.3e}, mean={:.3e}, max={:.3e} (n={})",
        label,
        w.iter().cloned().fold(f32::INFINITY, f32::min),
        w.iter().sum::<f32>() / (w.len().max(1) as f32),
        w.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        w.len(),
    );
}

/// Cells QC dropped before the graph was built, with what they were read with.
struct DroppedCells {
    names: Vec<Box<str>>,
    coordinates: Mat,
    coordinate_names: Vec<Box<str>>,
    batches: Vec<Box<str>>,
}

impl DroppedCells {
    /// QC's cells, then a mask's: both read with the same columns.
    fn join(self, more: DroppedCells) -> anyhow::Result<DroppedCells> {
        anyhow::ensure!(
            self.coordinate_names == more.coordinate_names,
            "cells set aside with different coordinate columns"
        );
        let n = self.names.len() + more.names.len();
        let d = self.coordinates.ncols();
        let coordinates = Mat::from_fn(n, d, |i, j| {
            if i < self.names.len() {
                self.coordinates[(i, j)]
            } else {
                more.coordinates[(i - self.names.len(), j)]
            }
        });
        Ok(DroppedCells {
            names: self.names.into_iter().chain(more.names).collect(),
            coordinates,
            coordinate_names: self.coordinate_names,
            batches: self.batches.into_iter().chain(more.batches).collect(),
        })
    }
}

/// `{prefix}.cells.parquet`: the kept cells as they stand, then the ones QC
/// dropped, their coordinates matched by column name (the `batch` offset
/// column is left out by the writer). `read_batches` is the kept cells'
/// read-time sample labels, `None` for a single sample.
fn write_cells_table(
    prefix: &str,
    kept: &[Box<str>],
    coordinates: &Mat,
    coordinate_names: &[Box<str>],
    read_batches: Option<&[Box<str>]>,
    dropped: Option<&DroppedCells>,
) -> anyhow::Result<()> {
    let n_kept = kept.len();
    anyhow::ensure!(
        coordinates.nrows() == n_kept,
        "cells table: {n_kept} cells but {} coordinate rows",
        coordinates.nrows()
    );
    let n_dropped = dropped.map_or(0, |d| d.names.len());
    let mut names = kept.to_vec();
    let mut batches: Vec<Box<str>> = match read_batches {
        Some(b) => {
            anyhow::ensure!(
                b.len() == n_kept,
                "cells table: {} batch labels for {n_kept} cells",
                b.len()
            );
            b.to_vec()
        }
        None => vec![Box::from(""); n_kept],
    };
    // Each dropped cell's value in every final coordinate column, by name.
    let dropped_column = |name: &str| -> Option<Vec<f32>> {
        let d = dropped?;
        let j = d.coordinate_names.iter().position(|n| n.as_ref() == name)?;
        Some(d.coordinates.column(j).iter().copied().collect())
    };
    let columns: Vec<Vec<f32>> = coordinate_names
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let mut col: Vec<f32> = coordinates.column(j).iter().copied().collect();
            col.extend(dropped_column(name).unwrap_or_else(|| vec![f32::NAN; n_dropped]));
            col
        })
        .collect();
    if let Some(d) = dropped {
        names.extend(d.names.iter().cloned());
        match read_batches {
            Some(_) => batches.extend(d.batches.iter().cloned()),
            None => batches.extend(std::iter::repeat_n(Box::from(""), n_dropped)),
        }
    }
    let all = Mat::from_fn(n_kept + n_dropped, columns.len(), |i, j| columns[j][i]);
    let in_graph: Vec<bool> = (0..n_kept + n_dropped).map(|i| i < n_kept).collect();
    crate::util::parquet_io::write_cells_table(
        prefix,
        &names,
        &all,
        coordinate_names,
        &batches,
        &in_graph,
    )
}
