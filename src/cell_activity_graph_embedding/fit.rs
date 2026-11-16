//! `pinto cage` entrypoint: activity-gated cell-graph embedding.
//!
//! Learns a per-SUPER-CELL (pseudobulk, PB) embedding `e_pb [P x D]` by
//! contrastive (NCE) prediction of PB-PB adjacency, one feature at a time. No
//! cell and no cell-cell pair is ever trained on: cells enter once, when
//! their spatial KNN edges are folded into PB super edges, and reappear only
//! in evaluation readouts (the per-pair latent, the propensity, and each
//! cell's own placement by the pair encoder).
//!
//! ```text
//! load -> spatial KNN -> batch effects                  util::srt_pipeline
//! HVG-weighted projection -> graph_coarsen_multilevel   nested super-cell levels
//! build_feature_active_fine_edges                          per-feature activity, active edges
//! training loop                                         the only SGD
//! parquet outputs + .pinto.json
//! ```
//!
//! # What the training loop optimizes
//!
//! Positives are PB-PB SUPER EDGES at the finest coarsening level:
//! cell-cell spatial KNN edges folded by endpoint labels, each carrying
//! the summed per-feature activity of the fine edges inside it. Negatives
//! are sibling PBs drawn per chain level: at level `l` a negative sits
//! in a different super-cell at `l` but the SAME parent at `l-1`, so the
//! contrast sharpens with depth. The coarsening hierarchy is live in
//! every step, one level up from where it used to be. There is one
//! trained embedding, over finest-level PBs, started at random.
//!
//! ```text
//! s(P,Q) = theta_g . (e_P (*) e_Q) + b_P + b_Q
//! theta_g = e_feat[g]
//! ```
//!
//! `e_P (*) e_Q` is the SUPER-EDGE embedding: the pair's joint
//! participation in each latent community, high only when both endpoint
//! PBs load on it.
//!
//! A per-(feature, dim) feature gate on `theta_g` — first a learned
//! spike-and-slab, then a Gibbs-sampled per-epoch mask — was carried here and
//! removed with the rest of the workspace's gate machinery.

use crate::cell_activity_graph_embedding::args::{
    CellActivityGraphEmbeddingArgs, FeatureEmbeddingMode, FeatureInclusionMode, ModulePlan,
    ALPHA_NEG, N_NEGATIVES, PAIR_BLOCK, PAIR_RIDGE, PER_FEATURE_BATCH,
};
use crate::cell_activity_graph_embedding::feature_chain_sampler::{
    build_feature_exp_batch_cache, FeatureGatedChainSampler,
};
use crate::cell_activity_graph_embedding::feature_gating::{
    build_feature_active_fine_edges, ActivityNorm,
};
use crate::cell_activity_graph_embedding::pair_projection::{
    project_pairs, CellLatent, PairBatchDivisor, PairEncoderSpec, PairLatent, PairProjectionArgs,
    PairSolver,
};
use crate::cell_activity_graph_embedding::pretrained;
use crate::link_community::profiles::{
    compute_propensity_and_feature_community_stat, PropensityReportConfig,
};
use crate::util::cell_pairs::SrtCellPairs;
use crate::util::common::*;
use crate::util::graph_coarsen::{graph_coarsen_multilevel, CoarsenConfig};
use crate::util::metadata::{create_cage_metadata, RunInputs, SpliceTrackInfo, DELTA_BASE_SPLICED};
use crate::util::score_trace::write_score_trace;
use crate::util::srt_pipeline::{
    preprocess_srt, topology_graph, BatchEstimate, FeatureAxisMode, FisherStats, Seeding,
    SrtPreprocessConfig, SrtPreprocessed,
};

use data_beans::alg::gene_weighting::save_fisher_weights;
use data_beans::alg::hvg::select_hvg_streaming;
use data_beans::alg::random_projection::RandProjOps;
use graph_embedding_util::embedding_col_names;
use graph_embedding_util::loss::{build_per_batch_unit_samplers, ChainGroupFilter};
use graph_embedding_util::model::E_FEAT_VAR_NAME;
// `Optimizer` is what puts `AdamW::new` in scope.
use legume_numeric::candle::candle_core::Tensor;
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use legume_numeric::candle::frozen_features::trainable_vars;
use legume_numeric::matrix::common_io::mkdir_parent;

mod model;
mod module_cells;
mod modules;
mod tables;
mod train;

use model::{build_model, BuiltModel, ModelInputs};
use module_cells::CellCounts;
use tables::write_trained_tables;
use train::{resolve_feature_batch_size, train, ModuleUnits, ProbeInputs, TrainInputs};

pub fn fit_cell_activity_graph_embedding(
    args: &CellActivityGraphEmbeddingArgs,
) -> anyhow::Result<()> {
    let c = &args.common;
    mkdir_parent(&c.out)?;

    // The width: the flag, else a pinned dictionary's, read from its footer
    // before any data is opened.
    let dictionary_width = match args.feature_embedding.as_deref() {
        Some(path) => Some(pretrained::dictionary_width(path)?),
        None => None,
    };
    let embedding_dim = args.resolve_embedding_dim(dictionary_width)?;
    // Chain levels, resolved and validated BEFORE any data is loaded: the
    // trained unit is the finest coarsening level, so chain entries must
    // be strictly coarser. Unset adapts to the hierarchy depth (up to
    // three coarser levels); explicit values are checked here, where a
    // mistake costs milliseconds rather than a load.
    let chain_levels: Vec<usize> = match &args.chain_levels {
        // the cells themselves (see `cell_level_hierarchy`)
        None if args.cell_level => vec![0],
        Some(levels) => {
            for &lvl in levels {
                anyhow::ensure!(
                    lvl + 1 < c.collapse.num_levels,
                    "--chain-levels entry {} must be coarser than the finest level \
                     (num_levels = {}; the finest level is the trained unit)",
                    lvl,
                    c.collapse.num_levels
                );
            }
            levels.clone()
        }
        None => {
            anyhow::ensure!(
                c.collapse.num_levels >= 2,
                "--num-levels {} leaves no level coarser than the finest; \
                 the chain needs at least 2 levels",
                c.collapse.num_levels
            );
            (0..(c.collapse.num_levels - 1).min(3)).collect()
        }
    };
    if args.cell_level {
        info!("Cell level: cells are the trained unit; negatives within each batch");
    } else {
        info!(
            "Chain levels: {:?} of {} coarsening levels (finest = trained unit)",
            chain_levels, c.collapse.num_levels
        );
    }
    anyhow::ensure!(args.epochs > 0, "epochs must be > 0");
    anyhow::ensure!(!chain_levels.is_empty(), "chain-levels must be non-empty");
    // On unless the flags turn the layer off; its count is resolved with
    // the model, against the profiles.
    let module_plan = args.module_plan()?;
    anyhow::ensure!(
        args.feature_inclusion != FeatureInclusionMode::Moran || !c.coord_files.is_empty(),
        "--feature-inclusion moran needs coordinates (-c): it scores spatial signal"
    );
    // Peek the first data file's row names so `auto` can dispatch
    // FeatureNameKind::auto_detect without paying for a full sparse
    // load up front.
    let peek_names = data_beans::convert::try_open_or_convert(&c.data_files[0])?.row_names()?;
    let feature_kind = args.feature_name_mode.resolve_kind(&peek_names);
    info!(
        "Feature-name canonicalization: {:?} (mode = {:?}, peeked {} names from {})",
        feature_kind,
        args.feature_name_mode,
        peek_names.len(),
        c.data_files[0]
    );

    /////////////////////////////////////
    // 1-3. Load + KNN + batch effects //
    /////////////////////////////////////
    // Needed before preprocessing, which decides whether to take a shared
    // projection on our behalf. HVG weighting makes that projection a
    // different object, so asking for one we cannot use would compute and
    // then hold a full `proj_dim x n_cells` matrix for the whole run.
    let hvg_enabled = args.hvg.n_hvg > 0 || args.hvg.feature_list_file.is_some();
    // cage trains on the coarsening; cage-cell builds one only to seed its
    // modules.
    let coarsens = !args.cell_level || module_plan != ModulePlan::Off;

    let SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        section_ids,
        seeding,
        batch_labels,
        batch_effects: batch_db,
        no_batch_effect,
        graph,
        knn,
        base_batch_corrected,
        spatial_graph,
        edge_source,
        cell_proj: shared_cell_proj,
        feature_axis,
        n_cells,
        n_rows,
        ablation,
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        ablation: Some(&args.ablation),
        // cage-cell too: the estimate is preprocessing (fitted on data-beans'
        // pseudobulk samples), not training; --ignore-batch turns it off
        batch_effects: BatchEstimate::Fit,
        feature_axis: FeatureAxisMode::Strict,
        cell_projection: !hvg_enabled && coarsens,
        feature_kind: Some(feature_kind.clone()),
    })?;
    // Only the mean divides; the full posterior is the file preprocessing wrote.
    let batch_db = batch_db
        .map(crate::util::batch_effects::BatchPosterior::into_mean)
        .transpose()?;

    let cell_names = data_vec.column_names()?;
    let row_names = data_vec.row_names()?;

    // What a ROW means, decided once. On a splice-channelized matrix (faba's
    // `{feature}/count/{spliced,unspliced}`) two rows are one feature, and everything
    // feature-side below folds through this; on any other matrix the axis is the
    // identity and every fold is a pass-through. The two are NOT interchangeable
    // names for one number, so `n_rows` and `n_features` stay apart from here on.
    let feature_axis = feature_axis.expect("FeatureAxisMode::Strict must yield Some");
    let n_features = feature_axis.n_features();
    let feature_names: Vec<Box<str>> = feature_axis.feature_names().to_vec();

    // Persist the NB-Fisher precision weights so downstream tools can reload
    // them, matching the senna/chickpea convention. These are per-ROW NB
    // precisions, consumed by the projection, which reads the matrix, so they
    // keep the row names.
    let FisherStats {
        row_weights: fisher_weights,
        row_stats,
        feature_weights: feature_fisher_weights,
        feature_stats: _,
    } = FisherStats::compute(&data_vec, &feature_axis)?;
    save_fisher_weights(&c.out, &fisher_weights, &row_names)?;

    // Fisher weights, on the axis each consumer actually indexes.
    //
    // `fisher_weights` is per ROW and stays that way: the projection reads
    // the matrix. The TRAINING
    // LOOP does not — it weights a per-FEATURE loss and indexes by feature id — so on
    // a channelized matrix `w[g]` would hand feature `g` an unrelated row's
    // precision, silently, because `n_features < n_rows` means it never goes out
    // of bounds.
    //
    // Both vectors come from one streaming pass, and the feature-axis one is
    // computed from statistics folded INSIDE that pass rather than from the row
    // statistics folded afterwards. That distinction is not cosmetic: `s2` of a
    // sum is not the sum of `s2`, so a post-hoc fold loses the cross term
    // between a feature's two tracks and hands the dispersion trend a variance
    // that is too small exactly where the two tracks covary most.

    let srt_cell_pairs = SrtCellPairs::with_graph(
        &data_vec,
        &coordinates,
        &graph,
        edge_source.as_deref(),
        Some(&batch_membership),
    );
    srt_cell_pairs.write_coord_pairs(&c.out, &coordinate_names)?;
    let fine_edges: Vec<(u32, u32)> = srt_cell_pairs
        .inner
        .pairs()
        .iter()
        .map(|&(i, j)| (i as u32, j as u32))
        .collect();
    let n_fine_edges = fine_edges.len();
    if feature_axis.is_channelized() {
        info!(
            "{} cells, {} features ({} channel rows), {} edges",
            n_cells, n_features, n_rows, n_fine_edges
        );
    } else {
        info!(
            "{} cells, {} features, {} edges",
            n_cells, n_features, n_fine_edges
        );
    }

    // Per-ROW count totals, streamed ONCE. Two consumers need them and both are
    // whole-matrix passes: the splice go/no-go here, and the pair projection's
    // partition at the end of the run. The counts do not change in between
    // (QC is already applied, and δ divides a pair's counts as they are read,
    // never the matrix), so a second pass would re-read every column of the
    // zarr to rebuild the same vector. These totals are undivided: they set
    // the pair projection's abundance offset, a part of the model, not the
    // written feature_community, which is fitted on divided counts.
    // Off the streaming pass; `sum()` IS the per-row total.
    let row_totals: Vec<f64> = row_stats.sum().iter().map(|&x| f64::from(x)).collect();
    let feature_totals = feature_axis.pool_totals(&row_totals);

    // The go/no-go for everything a velocity contrast would be built on, taken
    // BEFORE the expensive fit so a thin input is caught by reading the log
    // rather than by reading a model. Only a channelized input has a contrast to
    // report on at all.
    let splice_report = feature_axis
        .report_delta_identifiability(&row_totals)
        .map(|r| SpliceTrackInfo {
            n_rows,
            n_delta_identified: r.n_identified,
            nascent_count_fraction: r.nascent_fraction,
            delta_base: DELTA_BASE_SPLICED.to_string(),
        });

    /////////////////////////////////////////
    // 4. HVG-weighted proj -> coarsening //
    /////////////////////////////////////////
    // The coarsening hierarchy: cage's trained units, and under cage-cell
    // (which trains on cells) only the pseudobulks whose profiles seed
    // its feature modules.
    let coarsen = |shared_cell_proj: Option<
        data_beans::alg::random_projection::RandColProjOut,
    >|
     -> anyhow::Result<_> {
        let batch_arg: Option<&[Box<str>]> = if batch_db.is_some() {
            Some(&batch_membership)
        } else {
            None
        };
        // HVG selection WEIGHTS this projection; it does not subset the trained
        // feature axis. Non-selected features get projection weight 0, so they sit out
        // the basis the coarsening hierarchy is built from, but they stay on the
        // feature axis, still trained. The selection shapes WHERE the pseudobulks
        // land rather than which features the model may use — matching `senna bge`
        // and `senna gem`.
        let must_train = data_beans::alg::hvg::load_must_train(
            args.hvg.must_train_features.as_deref(),
            hvg_enabled,
        )?;
        let hvg_weights: Option<Vec<f32>> = if hvg_enabled {
            let hvg = select_hvg_streaming(
                &data_vec,
                (args.hvg.n_hvg > 0).then_some(args.hvg.n_hvg),
                args.hvg.feature_list_file.as_deref(),
                must_train.as_ref(),
                None,
            )?;
            // Selection ranks ROWS, and the projection it weights reads rows, so
            // both stay on the row axis. What cannot stay there is a feature: picking a
            // feature's spliced row without its unspliced one would weight half a feature
            // into the basis the coarsening hierarchy is cut from, so the weights are
            // promoted to whole features. `--n-hvg N` therefore still counts N ROWS —
            // between N/2 and N features on a channelized matrix — which is what the
            // realized count below reports.
            let mut w = hvg.row_weights(n_rows);
            let n_weighted = feature_axis.promote_row_weights(&mut w);
            info!(
                "HVG-weighted projection: {} of {} features weighted ({} of {} rows); \
             all stay on the trained axis",
                n_weighted,
                n_features,
                w.iter().filter(|&&x| x > 0.0).count(),
                n_rows
            );
            Some(w)
        } else {
            None
        };

        let mut cell_proj = match hvg_weights.as_deref() {
            // HVG weighting makes this a genuinely different projection, so the
            // shared one cannot stand in for it. Preprocessing skips taking one in
            // this case, and with `--knn-expr` off there is nothing to drop. When
            // it IS asked for, preprocessing takes an unweighted projection for the
            // expression graph, and this drops it rather than carry a second full
            // matrix through training.
            Some(w) => {
                drop(shared_cell_proj);
                data_vec.project_columns_weighted(c.collapse.proj_dim, None, batch_arg, w)?
            }
            None => {
                shared_cell_proj.expect("preprocessing takes the projection whenever HVG is off")
            }
        };
        let topology = topology_graph(&graph, &spatial_graph);
        Ok(graph_coarsen_multilevel(
            topology,
            &mut cell_proj.proj,
            srt_cell_pairs.inner.pairs(),
            CoarsenConfig {
                n_clusters: c.n_pseudobulk,
                num_levels: c.collapse.num_levels,
                seeding: seeding.as_ref().map(Seeding::params),
                modularity_veto: None,
            },
        ))
    };
    let coarsened = if coarsens {
        Some(coarsen(shared_cell_proj)?)
    } else {
        // taken only for the expression graph, if at all
        drop(shared_cell_proj);
        None
    };
    // Under cage-cell, each cell's finest pseudobulk when modules need
    // seeding; the rest of that hierarchy is dropped here.
    let (ml, module_seed_labels) = if args.cell_level {
        (
            cell_level_hierarchy(n_cells, srt_cell_pairs.inner.pairs()),
            coarsened.and_then(|mut seed| seed.all_cell_labels.pop()),
        )
    } else {
        (coarsened.expect("cage coarsens"), None)
    };

    // Batch ids + the PB TRAINING FRAME: its nesting / density / batch checks
    // are cheap, so they run before anything reads the counts again.
    let n_exp_batches = batch_labels.len().max(1);
    let batch_membership_u32 =
        crate::util::batch_effects::batch_ids(&batch_membership, &batch_labels);

    let (pb_frame, fine_to_super) = crate::cell_activity_graph_embedding::pb_frame::build_pb_frame(
        &ml,
        &graph,
        &batch_membership_u32,
        n_exp_batches,
    )?;
    let n_pb = pb_frame.n_pb;

    // Each feature's probability of spatial signal, which each epoch
    // samples features by (`--feature-inclusion moran`): on the finest PB
    // super-cells (their summed counts) and the spatial edges between
    // them, coarse super-cells of the tree as jackknife blocks; on cells
    // under cage-cell (no trained pseudobulks), geometric tiles as blocks.
    let inclusion = match args.feature_inclusion {
        FeatureInclusionMode::All => None,
        FeatureInclusionMode::Moran => {
            use crate::cell_activity_graph_embedding::feature_inclusion::{
                cell_inclusion, pb_inclusion, pb_spatial_edges, PbUnits, CELL_COLUMN_BLOCK,
            };
            let spatial_edges = &topology_graph(&graph, &spatial_graph).edges;
            let inc = if args.cell_level {
                cell_inclusion(
                    &data_vec,
                    &feature_axis,
                    (&coordinates, &coordinate_names),
                    &section_ids,
                    spatial_edges,
                    args.inclusion_blocks,
                    CELL_COLUMN_BLOCK,
                )?
            } else {
                // the frame folded `graph`'s edges, which are the
                // spatial ones when no separate spatial graph was built
                let edges = pb_spatial_edges(
                    &pb_frame.cell_to_finest_pb,
                    &section_ids,
                    spatial_edges,
                    spatial_graph
                        .is_none()
                        .then_some((&pb_frame.super_edges[..], &batch_membership_u32[..])),
                );
                let units = PbUnits {
                    data: &data_vec,
                    n_pb,
                    cell_to_pb: &pb_frame.cell_to_finest_pb,
                    parents: &pb_frame.pb_parent_maps,
                    edges: &edges,
                };
                pb_inclusion(&feature_axis, &units, &section_ids, args.inclusion_blocks)?
            };
            write_feature_inclusion(&c.out, &feature_names, &inc)?;
            Some(inc)
        }
    };

    let n_chain_levels = chain_levels.len();

    ///////////////////////////////
    // 5. The super-edge samplers //
    ///////////////////////////////
    //
    // From here on the trained unit is a FINEST-LEVEL SUPER-CELL (PB),
    // never a cell: cell-cell edges are folded into PB-PB super edges
    // once, and the chain machinery runs one level up. This step decides
    // which super edges cage may train on:
    //
    //   - cross-batch super edges are dropped outright;
    //   - a super edge survives only if BOTH endpoint PBs share a
    //     super-cell at EVERY `--chain-levels` entry (the coarser-level
    //     analog of the old within-super-cell positives);
    //   - per-PB degree over the retained super edges becomes the
    //     `ALPHA_NEG` negative weight, and per-level sibling pools are
    //     precomputed for the hard-negative draw.
    //
    // All of it is a deterministic function of the coarsening labels and
    // the edge list — no RNG here; randomness enters only at draw time.
    let pb_filter = ChainGroupFilter {
        unit_to_group_per_level: &pb_frame.pb_parent_maps,
        levels: &chain_levels,
    };
    let (samplers_per_exp_batch, sampler_stats) = build_per_batch_unit_samplers(
        &pb_frame.super_edges,
        &pb_frame.pb_exp_batch,
        n_exp_batches,
        n_pb,
        ALPHA_NEG,
        // on cells there is no coarser group for an edge to straddle;
        // cross-batch edges are dropped either way
        (!args.cell_level).then_some(pb_filter),
    );
    info!(
        "PB super-edge data: {} batches; dropped {} cross-batch, {} straddling a chain-level boundary",
        n_exp_batches, sampler_stats.cross_batch_dropped, sampler_stats.group_mismatch_dropped
    );

    let active_batch_count = samplers_per_exp_batch
        .iter()
        .filter(|s| s.is_some())
        .count();
    anyhow::ensure!(
        active_batch_count > 0,
        "no batch retained any within-batch super edges; consider --chain-levels or --reciprocal"
    );

    //////////////////////////////////////////////////
    // 6. Per-feature activities + (feature, batch) cache //
    //////////////////////////////////////////////////
    info!("Computing per-feature cell activities...");
    let fine_edge_activities = build_feature_active_fine_edges(
        &data_vec,
        &fine_edges,
        ActivityNorm::Log1p,
        &feature_axis,
    )?;
    // Fold per-feature fine-edge activity onto the PB super edges,
    // CONSUMING the fine-level lists (they are the largest resident
    // structure and nothing needs them past this line). Features whose
    // activity is entirely INSIDE super-cells cannot inform a PB-PB
    // contrast and drop out of training here.
    // `activities` indexes `fine_edges`; `fine_to_super` indexes
    // `graph.edges`. They are the same list today only because
    // `CellPairs::from_graph` aliases the graph's edges — pin that
    // invariant before folding through it.
    anyhow::ensure!(
        fine_to_super.len() == fine_edges.len(),
        "edge index spaces diverged: {} fine-to-super entries vs {} pair-list edges",
        fine_to_super.len(),
        fine_edges.len()
    );
    let super_edge_activities =
        crate::cell_activity_graph_embedding::feature_gating::fold_active_edges_to_super(
            fine_edge_activities,
            &fine_to_super,
        );
    drop(fine_to_super);
    let nonzero_features = super_edge_activities
        .feature_active_edges
        .iter()
        .filter(|v| !v.is_empty())
        .count();
    info!(
        "{}/{} features have ≥1 active super edge",
        nonzero_features, n_features
    );

    info!("Precomputing per-(feature, batch) positive distributions...");
    let cache = build_feature_exp_batch_cache(
        &super_edge_activities,
        &samplers_per_exp_batch,
        args.activity_alpha,
    );
    info!(
        "Feature-batch cache: {} active (feature, batch) pairs",
        cache.n_active_pairs()
    );
    // List of features with at least one cached (feature, batch) entry — used
    // to skip empty features in every epoch's permutation. Computed now so
    // we can free `activities` afterwards.
    // Under --feature-inclusion a feature with π = 0 is never drawn, so it
    // leaves the list (unless every π is 0, when an epoch draws uniformly).
    // Each feature's keep draw, order key and sampling stream are its own,
    // and the CUDA chunk-size probe walks `active_features`, so this changes
    // no other feature's draws.
    let active_features: Vec<usize> = (0..n_features)
        .filter(|&g| cache.entries[g].iter().any(|e| e.is_some()))
        .collect();
    let positive_pi = inclusion
        .as_ref()
        .map(|f| f.pi.as_slice())
        .filter(|pi| active_features.iter().any(|&g| pi[g] > 0.0));
    let trainable_features: Vec<usize> = match positive_pi {
        Some(pi) => active_features
            .iter()
            .copied()
            .filter(|&g| pi[g] > 0.0)
            .collect(),
        None => active_features.clone(),
    };
    info!(
        "{} trainable features (≥1 active batch{})",
        trainable_features.len(),
        if positive_pi.is_some() {
            ", π > 0"
        } else {
            ""
        }
    );

    // Features an epoch visits: all of them, or Σπ under inclusion.
    let expected_visits = crate::cell_activity_graph_embedding::feature_inclusion::expected_visits(
        &trainable_features,
        inclusion.as_ref().map(|f| f.pi.as_slice()),
    );
    let positives_per_feature_exp_batch =
        positives_per_feature_batch(args.positives_per_epoch, expected_visits, n_exp_batches);
    // --features-per-epoch caps the visits, not the split: each visited
    // feature keeps its draws and the epoch draws fewer in all.
    let features_visited = match args.features_per_epoch {
        0 => expected_visits,
        cap => cap.min(expected_visits),
    };
    info!(
        "Positive budget: {} super-edge draws per (feature, batch){}; {} features visited{} x {} batch(es) = {} per epoch",
        positives_per_feature_exp_batch,
        match args.positives_per_epoch {
            Some(total) => format!(
                " (--positives-per-epoch {total} split over {expected_visits} expected visits x {n_exp_batches} batch(es))"
            ),
            None => String::new(),
        },
        features_visited,
        match (inclusion.is_some(), args.features_per_epoch > 0) {
            (true, true) => " (Σπ under --feature-inclusion, capped by --features-per-epoch)",
            (true, false) => " (Σπ under --feature-inclusion)",
            (false, true) => " (capped by --features-per-epoch)",
            (false, false) => "",
        },
        n_exp_batches,
        positives_per_feature_exp_batch * features_visited * n_exp_batches,
    );

    // Activities can be dropped now — the cache owns everything the
    // sampler needs for v1.
    drop(super_edge_activities);

    anyhow::ensure!(
        !trainable_features.is_empty(),
        "no features have an active edge in any batch — nothing to train"
    );

    //////////////////////////
    // 7. Model + optimizer //
    //////////////////////////
    let dev = args.device.to_device(args.device_no)?;
    // Never `{:?}` the Device: candle's DeviceId is a process-global
    // creation counter (always 1 for a one-GPU run), not the CUDA
    // ordinal, and logging it made --device-no look ignored.
    info!("Using device: {} (ordinal {})", args.device, args.device_no);
    let varmap = VarMap::new();
    let lora_spec =
        (args.feature_embedding_mode == FeatureEmbeddingMode::Lora).then(|| args.lora.spec());
    let BuiltModel {
        mut model,
        frozen_feature,
        modules,
    } = build_model(ModelInputs {
        args,
        profile_labels: module_seed_labels.as_deref().unwrap_or_else(|| {
            ml.all_cell_labels
                .last()
                .expect("coarsening produced no levels")
        }),
        data_vec: &data_vec,
        feature_axis: &feature_axis,
        feature_kind: &feature_kind,
        module_plan,
        lora_spec,
        n_pb,
        embedding_dim,
        varmap: &varmap,
        dev: &dev,
    })?;
    // What the exact unit-module term pools each step: cage's pseudobulk
    // profiles, on the device as `[n_pb × G]`, or under cage-cell the drawn
    // cells' own sparse counts (the seeding pseudobulks have done their work
    // once the model is built).
    drop(module_seed_labels);
    if modules.is_none() && !args.seeding_flags.is_empty() {
        log::warn!(
            "cage-cell: no feature modules were found, so {} had no effect: they shape \
             only the HVG projection or the pseudobulks built to seed the modules",
            args.seeding_flags.join(", ")
        );
    }
    let (module_cfg, module_profiles) = modules.unzip();
    // (matched by value, so cage-cell's unread profiles are freed here)
    let module_units = match (module_profiles, args.cell_level) {
        (None, _) => None,
        (Some(_), true) => Some(ModuleUnits::Cells(CellCounts {
            data: &data_vec,
            feature_axis: &feature_axis,
        })),
        (Some(p), false) => {
            anyhow::ensure!(
                p.ncols() == n_pb,
                "module profiles over {} pseudobulks, {} trained",
                p.ncols(),
                n_pb
            );
            // column-major `[G × n_pb]` is row-major `[n_pb × G]`
            let rows: Vec<f32> = p.as_slice().to_vec();
            Some(ModuleUnits::Pseudobulks(Tensor::from_vec(
                rows,
                (p.ncols(), p.nrows()),
                &dev,
            )?))
        }
    };
    // One AdamW over the map, or two under LoRA: the shared factor `v` leaves
    // the main group for its own at the LoRA+ rate.
    let lora_v_name = legume_numeric::candle::lora::factor_names(E_FEAT_VAR_NAME).1;
    let lora_plus = lora_spec.map(|l| legume_numeric::candle::lora::LoraPlus {
        v_var: &lora_v_name,
        lr_ratio: l.lr_ratio,
        ridge: l.ridge,
    });
    let mut adams = vec![AdamW::new(
        match lora_plus {
            Some(_) => trainable_vars(&varmap, &[&lora_v_name]),
            None => varmap.all_vars(),
        },
        ParamsAdamW {
            lr: args.lr as f64,
            ..Default::default()
        },
    )?];
    if let Some(lp) = &lora_plus {
        adams.push(lp.optimizer(&varmap, args.lr)?);
    }

    //////////////////////
    // 8. Training loop //
    //////////////////////
    let pb_to_parent_per_level: Vec<&[usize]> = chain_levels
        .iter()
        .map(|&lvl| pb_frame.pb_parent_maps[lvl].as_slice())
        .collect();

    let sampler = FeatureGatedChainSampler {
        super_edges: &pb_frame.super_edges,
        samplers_per_exp_batch: &samplers_per_exp_batch,
        cache: &cache,
        unit_to_group_per_level: &pb_to_parent_per_level,
        positives_per_draw: positives_per_feature_exp_batch,
        n_negatives: N_NEGATIVES,
        exclude_partner: args.cell_level,
    };

    let feature_batch_size = resolve_feature_batch_size(ProbeInputs {
        args,
        active_features: &active_features,
        n_exp_batches,
        sampler: &sampler,
        model: &model,
        dev: &dev,
    });

    let score_trace = train(TrainInputs {
        args,
        model: &mut model,
        frozen_feature: frozen_feature.as_ref(),
        adams: &mut adams,
        lora_plus: lora_plus.as_ref(),
        sampler: &sampler,
        trainable_features: &trainable_features,
        features_visited,
        feature_batch_size,
        feature_fisher_weights: &feature_fisher_weights,
        inclusion: inclusion.as_ref().map(|f| f.pi.as_slice()),
        module_cfg: module_cfg.as_ref(),
        module_units,
        n_exp_batches,
        n_pb,
        n_features,
        n_chain_levels,
        dev: &dev,
    })?;

    ////////////////
    // 9. Outputs //
    ////////////////
    info!("Writing cage outputs...");

    let e_feature_out = write_trained_tables(
        c,
        &mut model,
        &pb_frame,
        &cell_names,
        &feature_names,
        embedding_dim,
    )?;
    // Phase 1's tables are on disk; the encoder below trains on the same
    // device, so release them first.
    drop(model);

    write_score_trace(&(c.out.to_string() + ".scores.parquet"), &score_trace)?;

    /////////////////////////////////////////////////////////
    // 10. Pair projection -> link communities -> propensity  //
    /////////////////////////////////////////////////////////
    // cage's downstream contract is the one `lc` and `dsvd` publish, and it is
    // built on the CELL PAIR, not the cell: a latent per pair, k-means over
    // those pairs for link communities, and a cell's propensity as the mix of
    // communities its incident edges carry.
    //
    // The pair latent comes from projecting each pair's POOLED counts onto the
    // frozen feature embedding (`pair_projection`) — the same phase-2 move `senna
    // bge` makes for cells, with the pair as the node. cage's own score
    // decomposes an edge as `⟨θ_g, e_u ⊙ e_v⟩`, so a Hadamard product is the
    // closed form this generalizes: the projection lets the pair's own expression
    // move it off that point, which is what puts a boundary pair between the
    // two programs it pools instead of on either endpoint. The same pass
    // places every cell (its self-pair), which is the cell embedding below.
    let pair_batch = batch_db.as_ref().map(|delta| PairBatchDivisor {
        delta,
        batch_of_cell: &batch_membership_u32,
    });
    let encoder_spec = PairEncoderSpec {
        ridge: PAIR_RIDGE,
        ..PairEncoderSpec::default()
    };
    let pair_encoder_path = format!("{}.pair_encoder.safetensors", c.out);
    let PairLatent {
        latent: pair_latent,
        bias: pair_bias,
        scores: _,
        cells: CellLatent {
            latent: cell_latent,
            bias: cell_bias,
        },
    } = project_pairs(
        &data_vec,
        &fine_edges,
        &e_feature_out,
        pair_batch,
        &PairProjectionArgs {
            solver: PairSolver::TrainEncoder {
                spec: &encoder_spec,
                dev: &dev,
                save_to: &pair_encoder_path,
            },
            seed: c.seed,
            pair_block: PAIR_BLOCK,
            eval_features: None,
            score_pairs: false,
        },
        &feature_axis,
        &feature_totals,
    )?;
    // Held-out feature evaluation, against the SAME frozen dictionary the projection above
    // just used. Placed here rather than after clustering because it tests the projection,
    // not the cut: a link community is a downstream choice, and folding it in would confuse
    // "does the embedding predict expression" with "did k-means pick a good k".
    // `β_uv` is the pair's log pooled depth and `β_c` the cell's; neither
    // leaves this function, but their spread is the cheapest check that the
    // projection saw real data.
    log_intercept_spread("Pair intercept β", pair_bias);
    log_intercept_spread("Cell intercept β", cell_bias);

    let fine_edges_usize: Vec<(usize, usize)> = fine_edges
        .iter()
        .map(|&(u, v)| (u as usize, v as usize))
        .collect();

    // `[D × E]` with every pair L2-normalized — the exact shape and
    // normalization `pinto dsvd` hands the shared routine, so edge k-means
    // clusters on composition and not on pooled depth.
    let mut pair_latent_kn = pair_latent.transpose();
    pair_latent_kn.normalize_columns_inplace();

    // One `[E × D]` copy, written out and then clustered — the shared routine
    // takes pairs as rows, so this is the same buffer both times.
    let pair_latent_nk = pair_latent_kn.transpose();
    pair_latent_nk.to_parquet_with_names(
        &(c.out.to_string() + ".latent.parquet"),
        (None, Some("cell_pair")),
        Some(&embedding_col_names(embedding_dim)),
    )?;

    // Cluster the pairs -> per-edge community -> cell propensity (incident-edge
    // fractions) + entropy + the Poisson-Gamma feature x community dictionary. One
    // routine, shared verbatim with `lc` and `dsvd`, so every subcommand's
    // propensity means the same thing.
    // Leiden by default: the pair latent has no reason to carry exactly
    // `embedding_dim` interaction regimes, so the graph decides the count and
    // `n_edge_clusters` is only a target (or nothing, when left unset). Under
    // k-means the requested count IS the count.
    let clustering = args.edge_clustering.resolve(c.seed);
    let prop_out = compute_propensity_and_feature_community_stat(
        &pair_latent_nk,
        &fine_edges_usize,
        &data_vec,
        n_cells,
        &PropensityReportConfig {
            clustering,
            // Deliberately row-keyed, and documented as such in `pinto cage
            // --help`: a channelized run lists a feature's two tracks as two rows,
            // which is the one place the nascent/mature contrast is reportable.
            // `lc` folds instead, because its dictionary merge has to index the
            // same axis its feature filter chose on.
            feature_axis: None,
            edge_kind: srt_cell_pairs.edge_kind.as_deref(),
            // batch-adjusted, with the plain divide (the pairs' is the
            // self-normalising Poisson ratio)
            divisor: pair_batch,
        },
        &c.out,
    )?;
    let n_edge_clusters = prop_out.n_clusters;

    // Cell embedding: every cell's own placement on the feature embedding by the
    // same map that placed its pairs — the annotate-consumable table (same D
    // as the feature embedding, since the latents are projections against it).
    cell_latent.to_parquet_with_names(
        &(c.out.to_string() + ".cell_embedding.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&embedding_col_names(cell_latent.ncols())),
    )?;
    info!(
        "Wrote {}.cell_embedding.parquet",
        crate::util::staging::final_str(&c.out)
    );

    // Metadata
    {
        let coord_file_str = c.coord_files_joined();
        let mut meta = create_cage_metadata(
            &RunInputs {
                prefix: &c.out,
                data_files: &c.data_files,
                coord_file: coord_file_str.as_deref(),
                coord_columns: &coordinate_names,
                n_cells,
                n_features,
                n_edges: n_fine_edges,
                graph: crate::util::metadata::GraphParams::of(&knn, base_batch_corrected),
                ablation,
                k: n_edge_clusters,
                delta: batch_db
                    .is_some()
                    .then(|| crate::util::metadata::DeltaWritten::fit(c)),
                no_batch_effect,
            },
            splice_report,
        );
        meta.cell_level = args.cell_level;
        let meta_path = std::path::PathBuf::from(crate::util::metadata::manifest_path(&c.out));
        meta.write(&meta_path)?;
        info!(
            "Wrote {}",
            crate::util::staging::final_path(&meta_path).display()
        );
    }

    info!("Done");
    Ok(())
}

/// `{out}.feature_inclusion.parquet`: each feature's Moran's I, its
/// jackknife standard error, t statistic and inclusion probability.
fn write_feature_inclusion(
    out: &str,
    features: &[Box<str>],
    inc: &crate::cell_activity_graph_embedding::feature_inclusion::FeatureInclusion,
) -> anyhow::Result<()> {
    use legume_numeric::matrix::parquet::{write_named_table, Column};
    let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let (moran, se, t) = (f32s(&inc.moran), f32s(&inc.se), f32s(&inc.t));
    write_named_table(
        &format!("{out}.feature_inclusion.parquet"),
        "feature",
        features,
        &[
            ("moran".into(), Column::F32(&moran)),
            ("jackknife_se".into(), Column::F32(&se)),
            ("t".into(), Column::F32(&t)),
            ("pi".into(), Column::F32(&inc.pi)),
        ],
    )?;
    info!(
        "Wrote {}.feature_inclusion.parquet",
        crate::util::staging::final_str(out)
    );
    Ok(())
}

/// Positives drawn per (feature, batch), one scalar for every feature
/// (see `FeatureGatedChainSampler::batch_size` for why the budget is not
/// split per feature). Unset, [`PER_FEATURE_BATCH`]; with
/// `--positives-per-epoch` `total`, `total` split evenly over
/// `expected_visits` (the trainable features, or Σπ under
/// `--feature-inclusion`) times `n_batches`, floored, and at least one.
pub(crate) fn positives_per_feature_batch(
    total: Option<usize>,
    expected_visits: usize,
    n_batches: usize,
) -> usize {
    match total {
        Some(total) => (total / (expected_visits * n_batches).max(1)).max(1),
        None => PER_FEATURE_BATCH,
    }
}

/// What `cage-cell` does with a flag of cage's that has no meaning on cells.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OnCells {
    /// Off its help: a qualifier of a refused flag, refused with it.
    Hide,
    /// Off its help, and warned of when given: it shapes only the HVG
    /// weighting of the projection that builds pseudobulks, or the
    /// pseudobulks themselves, which cage-cell builds only to seed its
    /// feature modules (no warning unless the run trains without them).
    /// (`--num-levels` and the batch
    /// matching shape the batch-effect estimate, which cage-cell makes.)
    Warn,
    /// Off its help, and refused when given: it picks from a hierarchy of
    /// pseudobulks.
    Refuse,
}

/// cage's flags that `cage-cell` hides, warns of or refuses, by argument id.
pub(crate) const CELL_LEVEL_FLAGS: [(&str, OnCells); 9] = [
    ("n_hvg", OnCells::Warn),
    ("feature_list_file", OnCells::Warn),
    ("must_train_features", OnCells::Warn),
    ("n_pseudobulk", OnCells::Warn),
    ("chain_levels", OnCells::Refuse),
    ("feature_embedding", OnCells::Refuse),
    ("feature_embedding_bias", OnCells::Hide),
    ("feature_embedding_mode", OnCells::Hide),
    ("feature_init_mode", OnCells::Hide),
];

/// `arg`, hidden when `cage-cell` has no use for it (every flag of
/// [`CELL_LEVEL_FLAGS`]).
pub(crate) fn hide_on_cells(arg: clap::Arg) -> clap::Arg {
    if CELL_LEVEL_FLAGS.iter().any(|(id, _)| arg.get_id() == *id) {
        arg.hide(true)
    } else {
        arg
    }
}

/// The flags of [`CELL_LEVEL_FLAGS`] marked `what` that `cage-cell` (`cmd`,
/// its matches `m`) was given on its command line; defaults are not the
/// user's doing.
pub(crate) fn cell_level_given(
    cmd: &clap::Command,
    m: &clap::ArgMatches,
    what: OnCells,
) -> Vec<String> {
    let ids = CELL_LEVEL_FLAGS
        .iter()
        .filter(|(_, w)| *w == what)
        .map(|(id, _)| *id);
    crate::util::cli::given_on_cli(cmd, m, ids)
}

/// Refuse the flags `cage-cell` cannot combine with and warn of those that
/// do nothing on cells, from its command line (`cmd`, its matches `m`).
/// Returns the flags given that shape only the pseudobulks seeding
/// `--feature-modules auto`: they will have done nothing if the fit finds
/// no module layer, which only the fit can say.
pub(crate) fn check_cell_level(
    cmd: &clap::Command,
    m: &clap::ArgMatches,
) -> anyhow::Result<Vec<String>> {
    let refused = cell_level_given(cmd, m, OnCells::Refuse);
    anyhow::ensure!(
        refused.is_empty(),
        "cage-cell: {} need a hierarchy of pseudobulks to pick from, which cage-cell \
         does not train on",
        refused.join(", ")
    );
    let no_ops = cell_level_no_ops(cmd, m);
    if !no_ops.is_empty() {
        log::warn!(
            "cage-cell: {} shape only the HVG projection or the pseudobulks, which \
             cage-cell builds only to seed its feature modules, off in this run; they \
             do nothing here",
            no_ops.join(", ")
        );
    }
    Ok(match cell_level_plan(m) {
        Ok(ModulePlan::Auto) => cell_level_given(cmd, m, OnCells::Warn),
        _ => Vec::new(),
    })
}

/// The [`ModulePlan`] of `cage-cell`'s matches `m`.
fn cell_level_plan(m: &clap::ArgMatches) -> anyhow::Result<ModulePlan> {
    use clap::FromArgMatches;
    CellActivityGraphEmbeddingArgs::from_arg_matches(m)?.module_plan()
}

/// The flags given to `cage-cell` (`cmd`, its matches `m`) that do nothing
/// in this run: those marked [`OnCells::Warn`], when the run's
/// [`ModulePlan`] is off. With modules (the default) they shape the
/// pseudobulks that seed them. A plan the run will refuse names none.
pub(crate) fn cell_level_no_ops(cmd: &clap::Command, m: &clap::ArgMatches) -> Vec<String> {
    match cell_level_plan(m) {
        Ok(ModulePlan::Off) => cell_level_given(cmd, m, OnCells::Warn),
        _ => Vec::new(),
    }
}

/// The "hierarchy" of a cell-level fit: every cell its own unit at both
/// levels. The trained unit is the cell; the one chain level is the cell
/// too, so the negative sampler's rejection (a candidate in the anchor's
/// group is redrawn) excludes exactly the anchor; the sampler's
/// `exclude_partner` redraws the positive partner too, so a negative is
/// any other cell of the anchor's batch (the samplers are per batch).
/// Nothing is coarsened.
pub(crate) fn cell_level_hierarchy(
    n_cells: usize,
    pairs: &[(usize, usize)],
) -> crate::util::graph_coarsen::MultiLevelCoarsenResult {
    let cells: Vec<usize> = (0..n_cells).collect();
    let (pair_to_sample, n_samples) =
        crate::util::graph_coarsen::cell_labels_to_pair_samples(&cells, pairs);
    crate::util::graph_coarsen::MultiLevelCoarsenResult {
        all_pair_to_sample: vec![pair_to_sample.clone(), pair_to_sample],
        all_num_samples: vec![n_samples; 2],
        all_cell_labels: vec![cells.clone(), cells],
    }
}

/// Min / median / max of a fitted intercept vector, as one log line.
fn log_intercept_spread(what: &str, values: Vec<f32>) {
    if !values.is_empty() {
        let q = legume_numeric::matrix::utils::quantiles(&values, &[0.0, 0.5, 1.0]);
        info!(
            "{what}: min {:.3}, median {:.3}, max {:.3}",
            q[0], q[1], q[2]
        );
    }
}
