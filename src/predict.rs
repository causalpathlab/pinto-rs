//! `pinto apply predict` — apply a trained `cage` run to a new sample.
//!
//! What crosses samples is the feature side and the community dictionary; the
//! geometry is local and is rebuilt from the new sample's own coordinates:
//!
//! 1. The new data goes through the same preprocessing as `cage` — coordinates,
//!    batch labels, the cell-cell KNN graph (spatial under `-c`, expression
//!    otherwise) — with the same flags, so the pairs are what `cage` would have
//!    built on this sample.
//! 2. The frozen feature embedding `{model}.feature_embedding.parquet` is aligned
//!    to the new sample's feature axis by name. A feature the model never saw gets
//!    no dictionary row: it is dropped from the partition rather than seeded
//!    (this is inference, not a warm start — inventing a row would let a feature
//!    the model knows nothing about pull on every pair's latent).
//! 3. Every cell pair — and every cell — is placed on that dictionary by the
//!    model's own pair encoder (`{model}.pair_encoder.safetensors`, the map
//!    that placed the training run's pairs), or by the exact per-pair Poisson
//!    MAP when the model has none, giving a pair latent in the trained space.
//! 4. Each pair is assigned to the nearest trained link community by cosine
//!    against the training centroids — the mean L2-normalized pair latent of
//!    each community, recomputed here from `{model}.latent.parquet` and
//!    `{model}.link_community.parquet` so no new artifact is needed.
//! 5. A cell's propensity is its incident-edge fraction per community, exactly
//!    the definition `cage` / `lc` / `dsvd` publish, and the cell embedding is
//!    the mean of the cell's pairs' latents from step 3, as `cage` writes it
//!    (so it depends on which neighbours the cell has in this sample).
//!
//! Outputs mirror `cage`'s inference tables: `{out}.{coord_pairs, latent,
//! link_community, propensity, feature_community, cell_embedding}.parquet` and a
//! `{out}.pinto.json` manifest, so `pinto view` and `lupin annotate` read a
//! predicted sample as they would a fitted one.

use crate::cell_activity_graph_embedding::args::FeatureNameMode;
use crate::cell_activity_graph_embedding::pair_projection::{
    project_pairs, CellLatent, PairBatchDivisor, PairLatent, PairProjectionArgs, PairScore,
    PairSolver,
};
use crate::link_community::outputs::{write_partition_outputs, PartitionInputs};
use crate::util::cell_pairs::SrtCellPairs;
use crate::util::common::*;
use crate::util::device::ComputeDevice;
use crate::util::metadata::{create_cage_metadata, BatchLabels, DeltaWritten, Run, RunInputs};
use crate::util::srt_pipeline::{
    preprocess_srt, BatchEstimate, FeatureAxisMode, FisherStats, SrtPreprocessConfig,
    SrtPreprocessed,
};
use clap::Args;
use data_beans::aux::frozen_features::{load_frozen_feature_host_matching, FrozenLoadArgs};
use graph_embedding_util::embedding_col_names;
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::matrix::traits::IoOps;
use log::info;
use rayon::prelude::*;
use std::path::Path;

#[derive(Args, Debug, Clone)]
pub struct PredictArgs {
    #[command(flatten)]
    pub core: PredictCoreArgs,

    #[command(flatten)]
    pub holdout: crate::holdout::HoldoutArgs,

    #[command(flatten)]
    pub scoring: ScoringArgs,
}

/// The flags `pinto apply predict` and `pinto apply impute` share: the query, the
/// model, and how the model is applied to the query.
#[derive(Args, Debug, Clone)]
pub struct PredictCoreArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[arg(
        long,
        required = true,
        help = "Trained run prefix: cage, or lc / dsvd for held-out scoring",
        long_help = "Trained run prefix. An lc or dsvd run (by its .pinto.json) is\n\
                     scored on held-out cells against {model}.feature_community.parquet.\n\
                     A `pinto fit cage` run is applied to the new sample. Reads:\n  \
                     {model}.feature_embedding.parquet  feature × D frozen dictionary\n  \
                     {model}.latent.parquet             training pair latent (E × D)\n  \
                     {model}.link_community.parquet     training pair → community\n\
                     The last two give the community centroids the new pairs are assigned to."
    )]
    pub model: Box<str>,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "Device ordinal (for cuda/metal)",
        hide = true
    )]
    pub device_no: usize,

    #[arg(
        long,
        default_value_t = FeatureNameMode::Auto,
        value_enum,
        help = "Feature-name canonicalization for matching the model's features to the new sample",
        hide = true
    )]
    pub feature_name_mode: FeatureNameMode,

    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "FRACTION",
        help = "Refuse to predict below this share of the model's features (0 = no gate)",
        long_help = "Feature coverage is always reported.\n\
                     This turns it into a hard floor.\n\
                     \n\
                     The share is of the MODEL's features, not this sample's,\n\
                     so a whole-transcriptome query is not penalized\n\
                     for carrying features the model never had.\n\
                     \n\
                     Zero mapped features is always refused: that is a naming failure.",
        hide = true
    )]
    pub min_feature_overlap: f32,

    #[arg(
        long,
        help = "Drop features the model never saw instead of initializing them through its modules",
        long_help = "Features of this sample that the model never saw are placed through the model's\n\
                     learned modules (needs {model}.module_membership.parquet and\n\
                     {model}.module_dictionary.parquet): membership averaged over the closest\n\
                     matched features by count profile over pseudobulks of this sample, row =\n\
                     membership times the module dictionary, with their own counts on the\n\
                     partition axis and provenance in {out}.feature_embedding_init.parquet.\n\
                     This flag restores the historical drop: no row, not on the axis.",
        hide = true
    )]
    pub no_init_features: bool,
}

/// How `pinto apply predict` scores its prediction, which `pinto apply impute` does
/// not.
#[derive(Args, Debug, Clone, Default)]
pub struct ScoringArgs {
    /// Restrict the agreement correlations to these features (one name per line).
    ///
    /// Off by default because a pair-level correlation sorts the feature axis once
    /// per pair, and a sample has far more pairs than cells. Pass the same file
    /// here and to `senna predict --eval-features` and the two commands'
    /// `spearman` / `pearson_log1p` columns are the same measurement.
    #[arg(
        long,
        value_name = "FILE",
        help = "Score the agreement correlations on these features only (one name per line)"
    )]
    pub eval_features: Option<Box<str>>,

    /// Training data, read once for the per-feature totals the frozen dictionary
    /// needs.
    ///
    /// Those totals are not only the null: they become `b_g`, the per-feature log
    /// abundance that is HALF of the pair log-rate `b_g + <e_g, e_uv>`. Taken
    /// from the query, as they were before this flag existed, the prediction is
    /// anchored on the very data it is scored against and the likelihood is not
    /// held out at all -- a dictionary that had learned nothing would still
    /// score well on `b_g` alone. Pass the training half and both the rate and
    /// the null come from data the model was actually fitted on.
    ///
    /// The value is a comma-separated list of data files.
    ///
    /// Same flag, same meaning as `senna predict --null-from`; pass the same
    /// training half to both and the two commands' scores are comparable.
    #[arg(
        long,
        value_name = "FILE",
        num_args = 1..,
        value_delimiter = ',',
        help = "Training data (comma-separated files) supplying the per-feature totals (b_g) and the null; pass the train half"
    )]
    pub null_from: Option<Vec<Box<str>>>,
}

/// The model's pair encoder, `{model}.pair_encoder.safetensors`: the one map
/// that places a sample's pairs and cells, so a model saved without it cannot
/// be predicted from.
fn pair_encoder_path(model: &str) -> anyhow::Result<String> {
    let path = format!("{model}.pair_encoder.safetensors");
    anyhow::ensure!(
        Path::new(&path).is_file(),
        "{path} does not exist: the model was fitted before `pinto fit cage` saved its pair encoder, \
         or it is a cage-cell model, which has none (placing new pairs on one comes in a later \
         version); refit it with `pinto fit cage`"
    );
    Ok(path)
}

/// Per-pair community labels from `{model}.link_community.parquet`, **unfiltered**.
///
/// Deliberately not `util::parquet_io::read_link_community`: that loader
/// drops every row whose `edge_kind` is not spatial, while `{model}.latent.parquet`
/// carries one row per pair including the expression-augmented ones. Pairing the two
/// positionally after that filter either aborts on a self-consistent model — the
/// counts disagree — or, worse, builds every centroid from the wrong pairs.
fn training_communities(path: &str) -> anyhow::Result<Vec<i64>> {
    crate::util::parquet_io::read_link_community_labels(Path::new(path))
}

/// Community centroids from a training run: the mean of the member pairs' L2-normalized
/// latents, renormalized, `[K × D]`.
///
/// The renormalization is what `assign_to_centroids` needs: it compares cosines, and
/// a diffuse community's short mean must not lose every pair to a tight one.
fn training_centroids(model: &str) -> anyhow::Result<(Mat, Vec<f32>)> {
    let latent = Mat::from_parquet(&format!("{model}.latent.parquet"))?.mat;
    let communities = training_communities(&format!("{model}.link_community.parquet"))?;
    anyhow::ensure!(
        communities.len() == latent.nrows(),
        "{model}: latent has {} pairs but link_community lists {} — the two files are \
         from different runs",
        latent.nrows(),
        communities.len()
    );
    let k = communities
        .iter()
        .copied()
        .filter(|&c| c >= 0)
        .max()
        .map_or(0, |m| m as usize + 1);
    anyhow::ensure!(k > 0, "{model}: link_community has no communities");
    let d = latent.ncols();
    let mut centroids = Mat::zeros(k, d);
    let mut counts = vec![0f32; k];
    for (e, &c) in communities.iter().enumerate() {
        if c < 0 {
            continue;
        }
        let mut row = latent.row(e).into_owned();
        let n = row.norm();
        if n > 0.0 {
            row /= n;
        }
        let mut dst = centroids.row_mut(c as usize);
        dst += row;
        counts[c as usize] += 1.0;
    }
    for c in 0..k {
        let mut row = centroids.row_mut(c);
        let nrm = row.norm();
        if nrm > 0.0 {
            row /= nrm;
        }
    }
    let live = counts.iter().filter(|&&n| n > 0.0).count();
    anyhow::ensure!(
        live > 0,
        "{model}: every community centroid is empty or degenerate"
    );
    if live < k {
        // An empty community keeps an all-zero row, which scores exactly 0.0 cosine
        // against every pair and so beats any genuinely anti-correlated centroid — the
        // "empty type is a magnet" failure already fixed once in `annotate`.
        // `assign_to_centroids` refuses to assign to a zero row instead.
        log::warn!(
            "{model}: {} of {k} communities have no pairs in the training latent; \
             they cannot be predicted into",
            k - live
        );
    }
    info!("Training centroids: {live} live of {k} communities in a {d}-dim pair latent");
    Ok((centroids, counts))
}

/// Nearest live centroid by cosine, for every (already L2-normalized) pair latent.
///
/// `None` where the pair cannot be assigned: a pair with no counts on any model feature
/// projects to the origin and is equidistant from everything, and a community with no
/// training pairs has a zero centroid that would otherwise capture every pair whose
/// cosine to all live centroids is negative. Both abstain rather than silently landing
/// in community 0.
fn assign_to_centroids(latent_nk: &Mat, centroids_kd: &Mat, counts: &[f32]) -> Vec<Option<usize>> {
    (0..latent_nk.nrows())
        .into_par_iter()
        .map(|e| {
            let row = latent_nk.row(e);
            if row.norm() <= 0.0 {
                return None;
            }
            let mut best: Option<(usize, f32)> = None;
            for (k, &n) in counts.iter().enumerate() {
                if n <= 0.0 {
                    continue;
                }
                let s: f32 = row.dot(&centroids_kd.row(k));
                if !s.is_finite() {
                    continue;
                }
                if best.is_none_or(|(_, b)| s > b) {
                    best = Some((k, s));
                }
            }
            best.map(|(k, _)| k)
        })
        .collect()
}

/// Returns the per-cell propensity and the cell names — the same table
/// written to `{out}.propensity.parquet` — so a caller that consumes the
/// prediction (`pinto apply impute`) need not round-trip it through parquet.
pub fn predict(args: &PredictArgs, given: &ModelOnlyFlags) -> anyhow::Result<(Mat, Vec<Box<str>>)> {
    // A model named by its manifest is the same model: records name runs that
    // way, so a held-out scoring links to the fit it scores.
    let mut args = args.clone();
    let run = open_model(&mut args.core.model)?;
    let args = &args;
    anyhow::ensure!(
        !args.core.common.data_files.is_empty(),
        "predict: no data files given"
    );
    let model = args.core.model.as_ref();
    if matches!(run.meta.command.as_str(), "lc" | "dsvd") {
        anyhow::ensure!(
            given.cage.is_empty(),
            "{model} is an {} model: {} apply to cage models only",
            run.meta.command,
            given.cage.join(", ")
        );
        crate::holdout::predict_profiles(args, &run)
    } else {
        anyhow::ensure!(
            given.holdout.is_empty(),
            "{model} is not an lc / dsvd model: the held-out scoring flags \
             ({}) apply to those only",
            given.holdout.join(", ")
        );
        predict_cage(&args.core, &args.scoring, &run)
    }
}

/// The flags given on `pinto apply predict`'s command line that apply to one kind
/// of model only, by name.
#[derive(Debug, Default)]
pub struct ModelOnlyFlags {
    /// Cage models only.
    pub cage: Vec<String>,
    /// lc / dsvd models only: the held-out scoring flags.
    pub holdout: Vec<String>,
}

impl ModelOnlyFlags {
    /// Those of `cmd` (`pinto apply predict`, its matches `m`).
    pub fn given(cmd: &clap::Command, m: &clap::ArgMatches) -> Self {
        use crate::util::cli::{given_on_cli, group_ids};
        ModelOnlyFlags {
            cage: given_on_cli(cmd, m, ["device", "device_no", "no_init_features"]),
            holdout: given_on_cli(cmd, m, group_ids(cmd, "HoldoutArgs")),
        }
    }
}

/// The model `model` names, opened once: its manifest (by prefix, or by the
/// manifest's own path, whatever its file name), with `model` set to the
/// run's prefix. A model without a manifest, or one that cannot be read, is
/// an error.
pub(crate) fn open_model(model: &mut Box<str>) -> anyhow::Result<Run> {
    let run = Run::open(model)?;
    // a residual fit's model is not on raw counts: predict and impute both
    // come through here
    run.refuse_residual_fit()?;
    *model = run.prefix.as_str().into();
    Ok(run)
}

/// The δ of the model at prefix `model` (its manifest `run`) and its path,
/// when it has one and `--ignore-batch` is not given. A model without one
/// (one batch or `--ignore-batch`) trained its
/// encoder on undivided counts, so its query is not divided either. A
/// model whose manifest says its dictionary is batch-adjusted refuses
/// `--ignore-batch` ([`delta::batch_adjusted`]).
fn model_delta(
    model: &str,
    run: &Run,
    ignore_batch: bool,
) -> anyhow::Result<Option<(crate::util::batch_effects::RunDelta, String)>> {
    delta::batch_adjusted(model, run, ignore_batch, delta::QueryReader::Cage)?;
    if ignore_batch {
        return Ok(None);
    }
    crate::util::batch_effects::read_run_delta(run)
}

/// `pinto apply predict` of the cage model at prefix `args.model`, its manifest
/// `run`.
pub fn predict_cage(
    args: &PredictCoreArgs,
    scoring: &ScoringArgs,
    run: &Run,
) -> anyhow::Result<(Mat, Vec<Box<str>>)> {
    let c = &args.common;
    anyhow::ensure!(
        !run.meta.cell_level,
        "{}: cage-cell models have no pair encoder; placing new pairs comes in a later version",
        args.model
    );
    mkdir_parent(&c.out)?;
    anyhow::ensure!(!c.data_files.is_empty(), "predict: no data files given");

    // A query batch the model trained on is divided by the model's own δ,
    // and only a new one is estimated, on the query; a model without a δ
    // trained on undivided counts, so the query is not divided.
    let model_delta = model_delta(&args.model, run, c.batch.ignore_batch)?;
    if model_delta.is_none() && !c.batch.ignore_batch {
        info!(
            "{}: the model has no batch effects, so the query is not batch-corrected",
            args.model
        );
    }
    let naming = delta::ModelNaming::of(run);
    let all_known = match &model_delta {
        Some((d, _)) => delta::every_batch_known(c, &d.batches, &naming)?,
        None => false,
    };
    if all_known {
        info!("Every batch of this sample is the model's; none is estimated");
    }
    // The δ is written once, below: the one the query is divided by.
    // Without coordinates the query's graph is built as the model's was,
    // on the batch-corrected projection or not, as its manifest records.
    let batch_plan = match model_delta {
        Some(_) => BatchEstimate::Query {
            all_known,
            base_corrected: run.meta.base_graph_batch_corrected(),
        },
        None => BatchEstimate::Skip,
    };
    // A model whose dictionary is batch-adjusted has its query's written
    // alike; any other's stays on the counts as read.
    let model_adjusted = run.meta.outputs.feature_community_batch_adjusted;

    let peek_names = data_beans::convert::try_open_or_convert(&c.data_files[0])?.row_names()?;
    let feature_kind = args.feature_name_mode.resolve_kind(&peek_names);

    let SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        section_ids: _,
        batch_labels: query_batches,
        batch_effects: estimate,
        graph,
        knn,
        edge_source,
        feature_axis,
        n_cells,
        n_rows: _,
        cell_proj,
        base_batch_corrected,
        ..
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        ablation: None,
        batch_effects: batch_plan,
        feature_axis: FeatureAxisMode::Strict,
        cell_projection: true,
        auto_batch_memory_check: None,
        batch_fold: crate::util::srt_pipeline::BatchFoldKind::None,
        feature_kind: Some(feature_kind.clone()),
    })?;
    let feature_axis = feature_axis.expect("FeatureAxisMode::Strict must yield Some");
    let FisherStats {
        row_stats,
        feature_weights,
        ..
    } = FisherStats::compute(&data_vec, &feature_axis)?;
    let cell_names = data_vec.column_names()?;
    let feature_names: Vec<Box<str>> = feature_axis.feature_names().to_vec();
    let n_features = feature_axis.n_features();

    // The frozen feature side, on THIS sample's feature axis. Unmatched features keep a
    // zero row and a zero total, which `PairDictionary` reads as "not on the
    // partition axis" — dropped, not invented.
    // Only the model's gene and region rows name a feature: a mixed-type
    // table's terms, words and cell types may share a gene's name.
    let dictionary_path = format!("{}.feature_embedding.parquet", args.model);
    let mut n_model = 0;
    let host = load_frozen_feature_host_matching(
        FrozenLoadArgs {
            dictionary_path: &dictionary_path,
            bias_path: None,
            target_feature_names: &feature_names,
            name_kind: feature_kind.clone(),
            source_name_map: None,
        },
        |names| {
            let marks = crate::util::dictionary_rows::matchable_rows(&dictionary_path, names);
            n_model = marks.iter().filter(|&&m| m).count();
            Ok(marks)
        },
    )?;
    let n_matched = host.keep_target_indices.len();
    // The MODEL's feature count: its rows that may match, from the dictionary
    // file. NOT `e_feat.nrows()`, which is the count AFTER the intersection and
    // therefore always equals `n_matched` — a coverage built from it is
    // identically 1 and `--min-feature-overlap` can never fire. Nor every row
    // of a mixed-type table, whose terms and words no sample carries.
    // The share of the MODEL's features this sample carries — the same denominator
    // senna's identically-named flag uses. Dividing by the query's feature count
    // instead would refuse a whole-transcriptome sample containing every panel feature.
    let coverage = n_matched as f32 / n_model.max(1) as f32;
    info!(
        "Feature alignment: {n_matched} of the model's {n_model} features are present here \
         ({:.1}% coverage; this sample has {n_features} features, D = {})",
        100.0 * coverage,
        host.h
    );
    anyhow::ensure!(
        n_matched > 0,
        "predict: none of the {n_features} features match the model's dictionary — check \
         --feature-name-mode"
    );
    anyhow::ensure!(
        coverage >= args.min_feature_overlap,
        "predict: only {:.1}% of the model's features are present, below --min-feature-overlap {:.1}%",
        100.0 * coverage,
        100.0 * args.min_feature_overlap
    );
    let mut e_full = Mat::zeros(n_features, host.h);
    for (i, &g) in host.keep_target_indices.iter().enumerate() {
        e_full.row_mut(g).copy_from(&host.e_feat.row(i));
    }
    // Which features are on the partition axis: every matched feature, plus — unless
    // `--no-init-features` — the features the model never saw, placed through its
    // modules by the same loader cage trains from.
    let mut on_axis: Vec<bool> = vec![false; n_features];
    for &g in &host.keep_target_indices {
        on_axis[g] = true;
    }
    if !args.no_init_features && n_matched < n_features {
        use crate::cell_activity_graph_embedding::pretrained;
        // Profiles over pseudobulks of THIS sample: k-means on its random
        // projection, then per-feature sums per cluster.
        let build_profiles = || -> anyhow::Result<Mat> {
            let proj = cell_proj
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("predict: no cell projection to cluster on"))?;
            let cells_by_dim = proj.proj.transpose(); // [n_cells × k]
            let n_pb = graph_embedding_util::transfer::pseudobulk_count(n_cells);
            let (_, labels) = legume_numeric::matrix::principal_graph::kmeans_centroids_seeded(
                &cells_by_dim,
                n_pb,
                20,
                0,
            );
            let row_profiles = crate::link_community::profiles::coarsen_cell_expression_dense(
                &data_vec, &labels, n_pb, None,
            )?;
            Ok(feature_axis
                .pool_rows_opt(&row_profiles)
                .unwrap_or(row_profiles))
        };
        let pre = pretrained::load_pretrained_feature_embedding(pretrained::PretrainedArgs {
            dictionary_path: &dictionary_path,
            bias_path: None,
            feature_names: &feature_names,
            name_kind: feature_kind.clone(),
            feature_profiles: &build_profiles,
            membership_init: Some(graph_embedding_util::transfer::AlignKnobs {
                k: graph_embedding_util::transfer::DEFAULT_INIT_NEIGHBOURS,
                similarity_floor: graph_embedding_util::transfer::DEFAULT_SIMILARITY_FLOOR,
            }),
        })?;
        let mut n_init = 0usize;
        for (g, r) in pre.records.iter().enumerate() {
            if r.init == pretrained::InitKind::Membership {
                e_full.row_mut(g).copy_from(&pre.e_feat.row(g));
                on_axis[g] = true;
                n_init += 1;
            }
        }
        if n_init > 0 {
            pretrained::write_init_report(&c.out, &pre.records)?;
            info!(
                "{n_init} features the model never saw were initialized through its modules and \
                 join the partition axis (see {}.feature_embedding_init.parquet)",
                crate::util::staging::final_str(&c.out)
            );
        } else {
            info!(
                "no module tables beside {}.feature_embedding.parquet: features the model never \
                 saw stay off the partition axis",
                args.model
            );
        }
    }
    let mut feature_totals = match scoring.null_from.as_deref() {
        Some(files) => crate::util::profile_match::training_feature_totals(
            files,
            &feature_kind,
            &feature_names,
        )?,
        None => {
            log::warn!(
                "no --null-from: per-feature totals come from the QUERY, so the pair log-rate \
                 is anchored on the data being scored and llik is not held out. Pass the \
                 training half"
            );
            let row_totals: Vec<f64> = row_stats.sum().iter().map(|&x| f64::from(x)).collect();
            feature_axis.pool_totals(&row_totals)
        }
    };
    for (g, t) in feature_totals.iter_mut().enumerate() {
        if !on_axis[g] {
            *t = 0.0;
        }
    }

    let eval_features: Option<Vec<Box<str>>> = match scoring.eval_features.as_deref() {
        // The same reader `senna predict --eval-features` uses. The help on both
        // commands tells the user to pass ONE file to both; parsing it two ways
        // meant a two-column panel scored senna on the right features and matched
        // nothing here.
        Some(path) => Some(
            legume_numeric::matrix::common_io::read_name_list(path)
                .map_err(|e| anyhow::anyhow!("reading --eval-features {path}: {e}"))?,
        ),
        None => None,
    };

    let (centroids, centroid_counts) = training_centroids(&args.model)?;
    anyhow::ensure!(
        centroids.ncols() == host.h,
        "{}: the pair latent is {}-dim but the feature dictionary is {}-dim",
        args.model,
        centroids.ncols(),
        host.h
    );

    // Pairs on the new sample's own graph.
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
    let n_edges = fine_edges.len();
    info!("{n_cells} cells, {n_features} features, {n_edges} pairs");
    anyhow::ensure!(
        n_edges > 0,
        "predict: this sample yielded no cell pairs, so there is nothing to predict — \
         check --coord and the neighbourhood flags (-k / --knn-expr)"
    );

    let batch_of_cell = crate::util::batch_effects::batch_ids(&batch_membership, &query_batches);

    let rows = data_vec.row_names()?;
    // The estimate in full (its spread too), on this sample's rows, by
    // sorted label (as data-beans numbers batches; `on_batches` holds it).
    let estimate = estimate
        .map(|e| e.into_delta().map(|d| d.on_batches(&query_batches)))
        .transpose()?;
    let batch_effects = model_delta.as_ref().and_then(|(model, path)| {
        let matched = delta::match_batches(&query_batches, c, &model.batches, &naming);
        delta::query_delta(
            (model, path),
            &matched,
            &rows,
            &query_batches,
            estimate.as_ref(),
        )
    });
    if let Some(d) = &batch_effects {
        d.write(&format!("{}.delta.parquet", c.out))?;
    }
    // The model's saved fold, when it has one: each query cell of a batch it
    // knows divided by a placed node's fold, as its own cells were; the rest
    // by the δ above ([`fold::query_fold_divisor`]).
    let saved_fold = match model_adjusted {
        true => crate::util::batch_fold::SavedCellFold::read(run)?,
        false => None,
    };
    let fold_divisor = match (&saved_fold, &batch_effects, &model_delta) {
        (Some(saved), Some(d), Some((model, _))) => {
            let matched = delta::match_batches(&query_batches, c, &model.batches, &naming);
            let model_label: Vec<Option<Box<str>>> = matched
                .iter()
                .map(|m| m.map(|m| model.batches[m].clone()))
                .collect();
            Some(fold::query_fold_divisor(
                saved,
                &data_vec,
                &rows,
                &d.mean,
                &fold::QueryBatches {
                    of_cell: &batch_of_cell,
                    model_label: &model_label,
                },
                &fold::QueryCells {
                    names: &cell_names,
                    neighbours: Some(&fold::placement_neighbours(
                        &fine_edges,
                        srt_cell_pairs.edge_kind.as_deref(),
                        &coordinates,
                        &batch_of_cell,
                    )),
                },
            )?)
        }
        _ => None,
    };
    // The encoder was trained on δ-divided rows (the fit's `project_pairs`),
    // so it reads δ here too; the feature-community statistic, fitted on
    // cell-fold-divided counts, reads the placed fold when there is one.
    let pair_batch = batch_effects.as_ref().map(|delta| {
        crate::util::batch_effects::Divisor::Batch(PairBatchDivisor {
            delta: &delta.mean,
            batch_of_cell: &batch_of_cell,
        })
    });
    let profile_divisor = match &fold_divisor {
        Some(f) => Some(crate::util::batch_effects::Divisor::Batch(
            PairBatchDivisor {
                delta: &f.table,
                batch_of_cell: &f.col_of_cell,
            },
        )),
        None => pair_batch,
    };

    let dev = args.device.to_device(args.device_no)?;
    let encoder_path = pair_encoder_path(&args.model)?;
    let PairLatent {
        latent,
        bias: _,
        scores,
        cells: CellLatent {
            latent: cell_latent,
        },
    } = project_pairs(
        &data_vec,
        &fine_edges,
        &e_full,
        pair_batch,
        &PairProjectionArgs {
            solver: PairSolver::LoadEncoder {
                path: &encoder_path,
                dev: &dev,
            },
            seed: c.seed,
            pair_block: crate::cell_activity_graph_embedding::args::PAIR_BLOCK,
            eval_features: eval_features.clone(),
            score_pairs: true,
        },
        &feature_axis,
        &feature_totals,
    )?;

    // Composition only, as in cage: L2-normalize each pair before the cut.
    let mut latent_kn = latent.transpose();
    latent_kn.normalize_columns_inplace();
    let latent_nk = latent_kn.transpose();
    latent_nk.to_parquet_with_names(
        &(c.out.to_string() + ".latent.parquet"),
        (None, Some("cell_pair")),
        Some(&embedding_col_names(latent_nk.ncols())),
    )?;

    write_predictive(&c.out, &scores)?;

    let assigned = assign_to_centroids(&latent_nk, &centroids, &centroid_counts);
    let k = centroids.nrows();
    let n_abstain = assigned.iter().filter(|a| a.is_none()).count();
    // An unassignable pair is dropped from the partition rather than parked in a
    // community: it contributes to no cell's propensity, which is the honest reading
    // of "this pair carries no evidence".
    let (kept_edges, labels): (Vec<(usize, usize)>, Vec<usize>) = fine_edges
        .iter()
        .zip(&assigned)
        .filter_map(|(&(u, v), a)| a.map(|k| ((u as usize, v as usize), k)))
        .unzip();
    {
        let mut hist = vec![0usize; k];
        for &l in &labels {
            hist[l] += 1;
        }
        let used = hist.iter().filter(|&&n| n > 0).count();
        info!(
            "Assigned {} of {n_edges} pairs to {used} of {k} trained communities (largest {}); \
             {n_abstain} abstained",
            labels.len(),
            hist.iter().max().copied().unwrap_or(0)
        );
    }
    anyhow::ensure!(
        !labels.is_empty(),
        "predict: no pair could be assigned to a trained community — check that the \
         model and this sample share features (see the coverage line above)"
    );

    let propensity = write_partition_outputs(
        &c.out,
        &PartitionInputs {
            edges: &kept_edges,
            // The edge-kind column is per-EDGE and `kept_edges` is a subset, so it can
            // only be forwarded when nothing abstained.
            edge_kind: (n_abstain == 0)
                .then_some(srt_cell_pairs.edge_kind.as_deref())
                .flatten(),
            cell_names: &cell_names,
            data_vec: &data_vec,
            feature_weights: &feature_weights,
            axis: &feature_axis,
            // batch-adjusted when the model's is
            divisor: profile_divisor.filter(|_| model_adjusted),
        },
        &labels,
        k,
    )?
    .propensity;

    // Each cell the mean of its pairs' latents, by the same map — what cage writes.
    cell_latent.to_parquet_with_names(
        &(c.out.to_string() + ".cell_embedding.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&embedding_col_names(cell_latent.ncols())),
    )?;

    let coord_file_str = c.coord_files_joined();
    let mut meta = create_cage_metadata(
        &RunInputs {
            prefix: &c.out,
            data_files: &c.data_files,
            coord_file: coord_file_str.as_deref(),
            coord_columns: &coordinate_names,
            n_cells,
            n_features,
            n_edges,
            graph: crate::util::metadata::GraphParams::of(&knn, base_batch_corrected),
            ablation: None,
            k,
            // the δ alone: the cells' batches are its own labels, in its
            // coord_pairs
            delta: batch_effects.as_ref().map(|_| DeltaWritten {
                labels: BatchLabels::of(&c.batch),
                cells: false,
                fold: crate::util::batch_fold::FoldWritten::None,
            }),
        },
        None,
    );
    meta.command = "predict".to_string();
    meta.batch_fold_placement =
        fold_divisor
            .as_ref()
            .map(|f| crate::util::batch_fold::BatchFoldPlacement {
                neighbours: fold::PLACEMENT_NEIGHBOURS,
                placed_cells: f.n_placed,
                fitted_cells: f.n_fitted,
                cells: n_cells,
            });
    if fold_divisor.as_ref().is_some_and(|f| f.n_placed > 0) {
        meta.outputs.profiles_divided_by =
            Some(crate::util::batch_fold::ProfilesDividedBy::CellFold);
    }
    meta.feature_names = FeatureNameMode::of_kind(&feature_kind);
    meta.outputs.feature_community_batch_adjusted &= model_adjusted;
    // A predicted sample is placed by the model's encoder; it is not a model
    // and carries none of its own.
    meta.outputs.pair_encoder = None;
    let meta_path = crate::util::metadata::manifest_path(&c.out);
    meta.write(Path::new(&meta_path))?;
    info!("Wrote {}", crate::util::staging::final_str(&meta_path));
    info!("Done");
    Ok((propensity, cell_names))
}

/// The data files a model's manifest `run` records. Fits record them
/// canonical; a relative entry is resolved beside the manifest, else from
/// here, and kept as written when it resolves in neither.
pub(crate) fn model_data_files(run: &Run) -> Vec<String> {
    run.meta
        .data_files
        .iter()
        .flatten()
        .map(|f| {
            run.resolve(f)
                .map_or_else(|| f.clone(), |p| p.to_string_lossy().into_owned())
        })
        .collect()
}

/// Per-pair held-out scores, under the same column names `senna predict` uses
/// for the same quantities.
///
/// The `eval_` prefix is not decoration. senna's `predictive.parquet` also has a
/// bare `llik` / `llik_per_count`, and those are the *backend's own*
/// decoder-dependent likelihood, which must not be compared across families. The
/// `eval_` columns are the multinomial over the scored features, which is what both
/// commands agree on. Writing pinto's comparable number under the bare name
/// would put two different estimands in one column of two files that share a
/// filename — a benchmark reading `llik_per_count` from both would rank an NB
/// density against multinomial nats/count and get a plausible wrong answer.
///
/// `eval_llik_per_count` is nats per observed count, invariant to how many cells
/// were pooled into the profile, so a pair's value sits on the same axis as a
/// cell's.
fn write_predictive(out: &str, scores: &[PairScore]) -> anyhow::Result<()> {
    let n = scores.len();
    let mut pred = Mat::zeros(n, 6);
    let (mut llik, mut null, mut count) = (0f64, 0f64, 0f64);
    for (i, s) in scores.iter().enumerate() {
        pred[(i, 0)] = s.llik;
        pred[(i, 1)] = s.total;
        // NaN, not 0: a pair with no counts has no per-count score, and 0 nats
        // per count reads as a PERFECT prediction to anything averaging this.
        pred[(i, 2)] = if s.total > 0.0 {
            s.llik / s.total
        } else {
            f32::NAN
        };
        pred[(i, 3)] = s.agreement.spearman;
        pred[(i, 4)] = s.agreement.pearson_log1p;
        // The floor, per pair and in the same units — so a row is readable on its
        // own rather than only against the aggregate on the log line.
        pred[(i, 5)] = if s.total > 0.0 {
            s.null_llik / s.total
        } else {
            f32::NAN
        };
        llik += f64::from(s.llik);
        null += f64::from(s.null_llik);
        count += f64::from(s.total);
    }
    let cols: Vec<Box<str>> = vec![
        "eval_llik".into(),
        "eval_count".into(),
        "eval_llik_per_count".into(),
        "spearman".into(),
        "pearson_log1p".into(),
        "eval_null_llik_per_count".into(),
    ];
    pred.to_parquet_with_names(
        &(out.to_string() + ".predictive.parquet"),
        (None, Some("cell_pair")),
        Some(&cols),
    )?;
    if count > 0.0 {
        info!(
            "Predictive: llik/count {:.4} vs abundance null {:.4} (gain {:+.4}) over {:.0} counts",
            llik / count,
            null / count,
            (llik - null) / count,
            count
        );
    }
    info!(
        "Wrote {}.predictive.parquet",
        crate::util::staging::final_str(out)
    );
    Ok(())
}

pub(crate) mod delta;
pub(crate) mod fold;

#[cfg(test)]
mod tests;
