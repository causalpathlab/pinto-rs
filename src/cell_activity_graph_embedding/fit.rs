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
//! collapse + SVD                                        pb_basis -> warm start of e_pb
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
//! trained embedding, over finest-level PBs; the collapse SVD is its
//! warm start.
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
    CellActivityGraphEmbeddingArgs, FeatureEmbeddingMode,
};
use crate::cell_activity_graph_embedding::feature_chain_sampler::{
    build_feature_exp_batch_cache, FeatureGatedChainSampler,
};
use crate::cell_activity_graph_embedding::feature_gating::build_feature_active_fine_edges;
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
use crate::util::graph_coarsen::{
    graph_coarsen_multilevel, CoarsenConfig, DcPoissonConfig, SeedingParams,
};
use crate::util::metadata::{create_cage_metadata, RunInputs, SpliceTrackInfo, DELTA_BASE_SPLICED};
use crate::util::score_trace::write_score_trace;
use crate::util::srt_pipeline::{
    preprocess_srt, topology_graph, FeatureAxisMode, SrtPreprocessConfig, SrtPreprocessed,
};

use data_beans::alg::gene_weighting::save_fisher_weights;
use data_beans::alg::hvg::select_hvg_streaming;
use data_beans::alg::random_projection::RandProjOps;
use graph_embedding_util::embedding_col_names;
use graph_embedding_util::loss::{build_per_batch_unit_samplers, ChainGroupFilter};
use graph_embedding_util::model::E_FEAT_VAR_NAME;
// `Optimizer` is what puts `AdamW::new` in scope.
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use legume_numeric::candle::frozen_features::trainable_vars;
use legume_numeric::matrix::common_io::mkdir_parent;

mod model;
mod tables;
mod train;

use model::{build_model, BuiltModel, ModelInputs};
use tables::write_trained_tables;
use train::{resolve_feature_batch_size, train, ProbeInputs, TrainInputs};

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
    // mistake costs milliseconds rather than a full collapse.
    let chain_levels: Vec<usize> = match &args.chain_levels {
        Some(levels) => {
            for &lvl in levels {
                anyhow::ensure!(
                    lvl + 1 < c.num_levels,
                    "--chain-levels entry {} must be coarser than the finest level \
                     (num_levels = {}; the finest level is the trained unit)",
                    lvl,
                    c.num_levels
                );
            }
            levels.clone()
        }
        None => {
            anyhow::ensure!(
                c.num_levels >= 2,
                "--num-levels {} leaves no level coarser than the finest; \
                 the chain needs at least 2 levels",
                c.num_levels
            );
            (0..(c.num_levels - 1).min(3)).collect()
        }
    };
    info!(
        "Chain levels: {:?} of {} coarsening levels (finest = trained unit)",
        chain_levels, c.num_levels
    );
    anyhow::ensure!(args.epochs > 0, "epochs must be > 0");
    anyhow::ensure!(!chain_levels.is_empty(), "chain-levels must be non-empty");
    // Opt-in on cage.
    let module_cfg = args.modules.resolve(None)?;
    anyhow::ensure!(
        !(module_cfg.is_some() && args.feature_embedding.is_some()),
        "--feature-modules learns the feature side through a module layer and --feature-embedding \
         installs a pre-trained one; the two parameterizations are exclusive. Drop one."
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

    let SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        batch_effects: batch_db,
        graph,
        knn,
        spatial_graph,
        edge_source,
        cell_proj: shared_cell_proj,
        feature_axis,
        row_weights: fisher_weights,
        row_stats,
        feature_weights: feature_fisher_weights,
        feature_stats: _,
        n_cells,
        n_rows,
        graph_coordinates,
        ablation,
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        ablation: Some(&args.ablation),
        fisher_weights: !args.no_fisher_weights,
        batch_effects: true,
        feature_axis: FeatureAxisMode::Strict,
        cell_projection: !hvg_enabled,
        feature_kind: Some(feature_kind.clone()),
    })?;

    let has_coords = c.has_coordinates();
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

    // Persist the NB-Fisher precision weights (when computed) so downstream
    // tools can reload them, matching the senna/chickpea convention. These are
    // per-ROW NB precisions, consumed by the projection and the Poisson
    // refinement, both of which read the matrix — so they keep the row names.
    if let Some(w) = fisher_weights.as_ref() {
        save_fisher_weights(&c.out, w, &row_names)?;
    }

    // Fisher weights, on the axis each consumer actually indexes.
    //
    // `fisher_weights` is per ROW and stays that way: the projection and the
    // degree-corrected Poisson refinement both read the matrix. The TRAINING
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
    // partition at the end of the run. The counts do not change in between —
    // batch effects and QC are both already applied — so a second pass would
    // re-read every column of the zarr to rebuild the same vector.
    // Off the streaming pass when it ran; `sum()` IS the per-row total. Only
    // `--no-fisher-weights` leaves it absent, and that is the one case worth a
    // second read.
    let row_totals: Vec<f64> = match row_stats {
        Some(st) => st.sum().iter().map(|&x| f64::from(x)).collect(),
        None => crate::link_community::profiles::compute_row_totals(&data_vec, c.block_size)?,
    };
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
            c.block_size,
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

    let cell_proj = match hvg_weights.as_deref() {
        // HVG weighting makes this a genuinely different projection, so the
        // shared one cannot stand in for it. Preprocessing skips taking one in
        // this case, and with `--knn-expr` off there is nothing to drop. When
        // it IS asked for, preprocessing takes an unweighted projection for the
        // expression graph, and this drops it rather than carry a second full
        // matrix through training.
        Some(w) => {
            drop(shared_cell_proj);
            data_vec.project_columns_weighted(c.proj_dim, c.block_size, batch_arg, w)?
        }
        None => shared_cell_proj.expect("preprocessing takes the projection whenever HVG is off"),
    };
    let topology = topology_graph(&graph, &spatial_graph);
    let ml = graph_coarsen_multilevel(
        topology,
        &mut cell_proj.proj.clone(),
        srt_cell_pairs.inner.pairs(),
        CoarsenConfig {
            n_clusters: c.n_pseudobulk,
            num_levels: c.num_levels,
            refine_iterations: c.refine_iterations,
            // The positions the graph was built on: under --permute-coords
            // the scrambled ones, so the seeding follows the graph.
            seeding: has_coords.then(|| SeedingParams {
                coordinates: graph_coordinates.as_ref().unwrap_or(&coordinates),
                batch_membership: Some(&batch_membership),
            }),
            modularity_veto: None,
            // Second-opinion refinement on RAW counts, matching `pinto lc`.
            // cage was the only caller passing `None` here, so its levels were
            // cut on cosine-of-projection alone while `lc` got a degree-corrected
            // Poisson pass over the actual counts.
            //
            // This is also the path that consumes the parent labels for its
            // sibling constraint, and those were fed the WRONG level until the
            // coarsening loop was split into cut-then-refine passes — so
            // enabling it before that fix would have constrained moves against
            // the finer level rather than the coarser one.
            //
            // Opt out with `--no-dc-poisson`; the context build reads the count
            // matrix once, then every level reuses it.
            dc_poisson: (!args.no_dc_poisson).then(|| DcPoissonConfig {
                params: data_beans::alg::dc_poisson::RefineParams {
                    num_gibbs: 10,
                    num_greedy: 5,
                    feature_weighting: data_beans::alg::dc_poisson::FeatureWeighting::FisherInfoNb,
                    seed: c.seed,
                    gibbs_stagnation: 0.005,
                    profile_source: data_beans::alg::dc_poisson::ProfileSource::Raw,
                    ..Default::default()
                },
                data: &data_vec,
                num_features: n_rows,
            }),
        },
    );

    // Batch ids + the PB TRAINING FRAME, built BEFORE the pseudobulk basis:
    // its nesting / density / batch checks are cheap, and failing them
    // here costs milliseconds instead of a full collapse.
    let batch_id_of: HashMap<Box<str>, u32> = {
        let mut uniq: Vec<Box<str>> = batch_membership.to_vec();
        uniq.sort();
        uniq.dedup();
        uniq.into_iter()
            .enumerate()
            .map(|(i, b)| (b, i as u32))
            .collect()
    };
    let n_exp_batches = batch_id_of.len().max(1);
    let batch_membership_u32: Vec<u32> = batch_membership
        .iter()
        .map(|b| *batch_id_of.get(b).expect("batch id"))
        .collect();

    let (pb_frame, fine_to_super) = crate::cell_activity_graph_embedding::pb_frame::build_pb_frame(
        &ml,
        &graph,
        &batch_membership_u32,
        n_exp_batches,
    )?;
    let n_pb = pb_frame.n_pb;

    // Collapse cells into super-cells per coarsening level and fit one shared
    // SVD over them: the WARM START of the trained PB table, and the basis the
    // `pseudobulk_cells.parquet` diagnostic exports.
    let basis = crate::cell_activity_graph_embedding::pb_basis::build_pseudobulks(
        crate::cell_activity_graph_embedding::pb_basis::PseudobulkArgs {
            data: &data_vec,
            all_cell_labels: &ml.all_cell_labels,
            graph: &graph,
            embedding_dim,
            feature_axis: &feature_axis,
        },
    )?;
    let mut e_pb_warm: Option<Mat> = None;
    // Diagnostic export: per CELL, its spatial coords, its finest-level
    // super-cell id, and that super-cell's embedding. Lets the basis be
    // inspected as a UMAP and as a spatial map before anything is built on it.
    if let Some(fine) = basis.levels.last() {
        e_pb_warm = Some(fine.e_pb_svd.clone());
        let d = fine.e_pb_svd.ncols();
        let mut out = Mat::zeros(n_cells, 3 + d);
        for i in 0..n_cells {
            let p = fine.cell_labels[i];
            out[(i, 0)] = coordinates[(i, 0)];
            out[(i, 1)] = coordinates[(i, 1)];
            out[(i, 2)] = p as f32;
            for h in 0..d {
                out[(i, 3 + h)] = fine.e_pb_svd[(p, h)];
            }
        }
        let mut cols: Vec<Box<str>> = vec![Box::from("x"), Box::from("y"), Box::from("pb")];
        cols.extend(embedding_col_names(d));
        out.to_parquet_with_names(
            &(c.out.to_string() + ".pseudobulk_cells.parquet"),
            (Some(&cell_names), Some("cell")),
            Some(&cols),
        )?;
        info!("Wrote {}.pseudobulk_cells.parquet", c.out);
    }

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
    //     `--alpha-neg` negative weight, and per-level sibling pools are
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
        args.alpha_neg,
        Some(pb_filter),
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
        c.block_size,
        args.activity_norm,
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
    let trainable_features: Vec<usize> = (0..n_features)
        .filter(|&g| cache.entries[g].iter().any(|e| e.is_some()))
        .collect();
    info!(
        "{} trainable features (≥1 active batch)",
        trainable_features.len()
    );

    // Positives drawn per (feature, batch), one scalar for every feature.
    // `--positives-per-epoch` overrides `--per-feature-batch` with an absolute
    // total. See `FeatureGatedChainSampler::batch_size` for why the budget is not
    // split per-feature.
    let positives_per_feature_exp_batch = match args.positives_per_epoch {
        Some(total) => (total / (trainable_features.len() * n_exp_batches).max(1)).max(1),
        None => args.per_feature_batch,
    };
    info!(
        "Positive budget: {} super-edge draws per (feature, batch), {} features x {} batch(es) = {} per epoch{}",
        positives_per_feature_exp_batch,
        trainable_features.len(),
        n_exp_batches,
        positives_per_feature_exp_batch * trainable_features.len() * n_exp_batches,
        if args.positives_per_epoch.is_some() {
            " (--positives-per-epoch)"
        } else {
            " (--per-feature-batch)"
        }
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
        module_profiles,
    } = build_model(ModelInputs {
        args,
        ml: &ml,
        data_vec: &data_vec,
        feature_axis: &feature_axis,
        feature_kind: &feature_kind,
        module_cfg: module_cfg.as_ref(),
        lora_spec,
        n_pb,
        embedding_dim,
        e_pb_warm: e_pb_warm.as_ref(),
        varmap: &varmap,
        dev: &dev,
    })?;
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
        n_negatives: args.n_negatives,
    };

    let feature_batch_size = resolve_feature_batch_size(ProbeInputs {
        args,
        trainable_features: &trainable_features,
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
        feature_batch_size,
        feature_fisher_weights: feature_fisher_weights.as_deref(),
        module_cfg: module_cfg.as_ref(),
        module_profiles: module_profiles.as_ref(),
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
        ridge: args.pair_ridge,
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
            pair_block: args.pair_block,
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
            block_size: c.block_size,
            // Deliberately row-keyed, and documented as such in `pinto cage
            // --help`: a channelized run lists a feature's two tracks as two rows,
            // which is the one place the nascent/mature contrast is reportable.
            // `lc` folds instead, because its dictionary merge has to index the
            // same axis its feature filter chose on.
            feature_axis: None,
            edge_kind: srt_cell_pairs.edge_kind.as_deref(),
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
    info!("Wrote {}.cell_embedding.parquet", c.out);

    // Metadata
    {
        let coord_file_str = c.coord_files_joined();
        let meta = create_cage_metadata(
            &RunInputs {
                prefix: &c.out,
                data_files: &c.data_files,
                coord_file: coord_file_str.as_deref(),
                coord_columns: &coordinate_names,
                n_cells,
                n_features,
                n_edges: n_fine_edges,
                graph: (&knn).into(),
                ablation,
                k: n_edge_clusters,
            },
            batch_db.is_some(),
            splice_report,
        );
        let meta_path = std::path::PathBuf::from(format!("{}.pinto.json", c.out));
        meta.write(&meta_path)?;
        info!("Wrote {}", meta_path.display());
    }

    info!("Done");
    Ok(())
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
