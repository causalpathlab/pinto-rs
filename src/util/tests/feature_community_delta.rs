//! The feature x community dictionary a run writes is batch-adjusted when
//! the run divides by a batch effect: fitted on `x / δ[g, batch(cell)]`,
//! the plain divide. Under `--ignore-batch` it is fitted on the counts as
//! read, as before.

use super::ignore_batch::{preprocess_with, two_batches};
use crate::link_community::profiles::fit_feature_community_param;
use crate::util::batch_effects::{
    batch_ids, delta_cells_path, read_cell_batches, read_delta, BatchDivisor,
};
use crate::util::common::Mat;
use crate::util::input::SrtInputArgs;
use crate::util::srt_pipeline::{
    preprocess_srt, BatchEstimate, FeatureAxisMode, SrtPreprocessConfig,
};
use clap::Parser;
use legume_numeric::matrix::traits::IoOps;
use legume_numeric::param::traits::Inference;

/// With one community per batch, the second batch's planted fold (three
/// times the first's on features 0..6) shows in the undivided dictionary
/// and is gone from the one divided by the run's estimate.
#[test]
fn dividing_by_the_estimate_removes_the_per_batch_bias() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_batches(dir.path(), 60);
    let out = dir.path().join("run").to_str().unwrap().to_string();
    let mut pre = preprocess_with(&files, &out, &[], BatchEstimate::Fit);
    let delta = pre
        .batch_effects
        .take()
        .expect("two files, two batches")
        .into_mean()
        .unwrap();
    let batch_of_cell = batch_ids(&pre.batch_membership, &pre.batch_labels);
    let mut by_batch = Mat::zeros(batch_of_cell.len(), 2);
    for (i, &b) in batch_of_cell.iter().enumerate() {
        by_batch[(i, b as usize)] = 1.0;
    }
    let ones = vec![1.0; pre.data_vec.num_rows()];
    let fit = |divisor| {
        fit_feature_community_param(&by_batch, &pre.data_vec, Some(&ones), None, divisor)
            .unwrap()
            .posterior_mean()
            .clone()
    };
    let raw = fit(None);
    let divided = fit(Some(
        BatchDivisor {
            delta: &delta,
            batch_of_cell: &batch_of_cell,
        }
        .into(),
    ));
    for g in 0..12 {
        let (r, d) = (raw[(g, 1)] / raw[(g, 0)], divided[(g, 1)] / divided[(g, 0)]);
        let planted = if g < 6 { 3.0 } else { 1.0 };
        assert!((r - planted).abs() < 0.1, "feature {g}: raw ratio {r}");
        assert!((d - 1.0).abs() < 0.1, "feature {g}: divided ratio {d}");
    }
}

/// What a fit at `prefix` (on `data`, `coords`) wrote, against the same
/// propensity refitted on the counts divided by the run's own δ and on the
/// counts as read: the largest difference to each.
fn against_refits(dir: &std::path::Path, prefix: &str, data: &str, coords: &str) -> (f32, f32) {
    let (written, _) = crate::util::parquet_io::read_feature_community(std::path::Path::new(
        &format!("{prefix}.feature_community.parquet"),
    ))
    .unwrap();
    let k = written.ncols();
    let prop = Mat::from_parquet(&format!("{prefix}.propensity.parquet")).unwrap();
    let propensity = prop.mat.columns(0, k).into_owned();

    let scratch = dir.join("refit").to_str().unwrap().to_string();
    let argv = [
        "pinto",
        data,
        "-c",
        coords,
        "--coord-column-names",
        "x,y",
        "-o",
        &scratch,
    ];
    let args = SrtInputArgs::try_parse_from(argv).unwrap();
    let pre = preprocess_srt(SrtPreprocessConfig {
        common: &args,
        ablation: None,
        batch_effects: BatchEstimate::Skip,
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: None,
        cell_projection: false,
        auto_batch_memory_check: None,
    })
    .unwrap();
    let cells = pre.data_vec.column_names().unwrap();
    assert_eq!(cells, prop.rows, "the refit reads the run's cells");
    // a fit's cells' batches are written beside its δ; a query's are its
    // labels as read
    let cells_path = delta_cells_path(prefix);
    let membership: Vec<Box<str>> = if std::path::Path::new(&cells_path).exists() {
        let batch_of = read_cell_batches(&cells_path).unwrap();
        cells.iter().map(|c| batch_of[c].clone()).collect()
    } else {
        pre.batch_membership.clone()
    };

    let fit = |divisor| {
        fit_feature_community_param(&propensity, &pre.data_vec, None, None, divisor)
            .unwrap()
            .posterior_mean()
            .clone()
    };
    let gap = |m: &Mat| (m - &written).abs().max();
    let raw = gap(&fit(None));
    let delta_path = format!("{prefix}.delta.parquet");
    if !std::path::Path::new(&delta_path).exists() {
        return (f32::NAN, raw);
    }
    let delta = read_delta(&delta_path).unwrap();
    assert_eq!(delta.rows, pre.data_vec.row_names().unwrap());
    let batch_of_cell = batch_ids(&membership, &delta.batches);
    let divided = gap(&fit(Some(
        BatchDivisor {
            delta: &delta.mean,
            batch_of_cell: &batch_of_cell,
        }
        .into(),
    )));
    (divided, raw)
}

fn adjusted_in_manifest(prefix: &str) -> bool {
    let path = crate::util::metadata::manifest_path(prefix);
    crate::util::metadata::PintoMetadata::read(std::path::Path::new(&path))
        .unwrap()
        .outputs
        .feature_community_batch_adjusted
}

/// `fit` (given its input, output and extra flags) on two sections, the
/// second folded on a third of the features: with δ the written dictionary
/// is the divided refit and not the undivided one, and the manifest says
/// so; under `--ignore-batch` it is the undivided refit.
fn check(fit: &dyn Fn(&[&str])) {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = crate::test_support::two_sections(dir.path(), 8);
    let run = |name: &str, extra: &[&str]| {
        let out = dir.path().join(name).to_str().unwrap().to_string();
        let mut argv = vec![
            data.as_str(),
            "-c",
            coords.as_str(),
            "--coord-column-names",
            "x,y",
            "-o",
            out.as_str(),
        ];
        argv.extend_from_slice(extra);
        fit(&argv);
        out
    };

    let adjusted = run("adjusted", &[]);
    let (divided, raw) = against_refits(dir.path(), &adjusted, &data, &coords);
    assert!(divided < 1e-3, "not the divided refit: {divided}");
    assert!(raw > 0.1, "the undivided refit too: {raw}");
    assert!(adjusted_in_manifest(&adjusted));

    let ignored = run("ignored", &["--ignore-batch"]);
    let (_, raw) = against_refits(dir.path(), &ignored, &data, &coords);
    assert!(raw < 1e-3, "--ignore-batch: not the undivided refit: {raw}");
    assert!(!adjusted_in_manifest(&ignored));
}

#[test]
fn lc_writes_a_batch_adjusted_dictionary() {
    check(&|input| {
        let flags = [
            "--n-communities",
            "2",
            "--num-gibbs",
            "2",
            "--num-greedy",
            "1",
            "--num-levels",
            "2",
        ];
        let argv = ["lc"].iter().chain(input).chain(&flags);
        crate::link_community::fit::fit_srt_link_community(
            &crate::link_community::args::SrtLinkCommunityArgs::try_parse_from(argv).unwrap(),
        )
        .unwrap();
    });
}

#[test]
fn dsvd_writes_a_batch_adjusted_dictionary() {
    check(&|input| {
        let flags = ["-t", "3", "--num-levels", "2"];
        let argv = ["dsvd"].iter().chain(input).chain(&flags);
        crate::svd::fit::fit_srt_delta_svd(
            &crate::svd::fit::SrtDeltaSvdArgs::try_parse_from(argv).unwrap(),
        )
        .unwrap();
    });
}

/// `pinto fit cage` on `input` (its data, output and extra flags), small.
fn fit_cage(input: &[&str]) {
    use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        cage: CellActivityGraphEmbeddingArgs,
    }
    let flags = [
        "--pair-encoder-steps",
        "8",
        "--epochs",
        "2",
        "--embedding-dim",
        "4",
        "-d",
        "24",
        "--num-levels",
        "3",
        "--n-edge-clusters",
        "2",
        "--edge-cluster-method",
        "kmeans",
    ];
    let argv = ["x"].iter().chain(input).chain(&flags);
    crate::cell_activity_graph_embedding::fit::fit_cell_activity_graph_embedding(
        &Cli::try_parse_from(argv).unwrap().cage,
    )
    .unwrap();
}

#[test]
fn cage_writes_a_batch_adjusted_dictionary() {
    check(&fit_cage);
}

/// `pinto apply predict` of the model at `model` on `input` (its data and extra
/// flags) to `{dir}/{name}`: its prefix.
fn predict(dir: &std::path::Path, model: &str, input: &[&str], name: &str) -> String {
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        predict: crate::predict::PredictArgs,
    }
    let out = dir.join(name).to_str().unwrap().to_string();
    let mut argv = vec!["x"];
    argv.extend_from_slice(input);
    argv.extend_from_slice(&["--model", model, "-o", &out]);
    crate::predict::predict(
        &Cli::try_parse_from(argv).unwrap().predict,
        &Default::default(),
    )
    .unwrap();
    out
}

/// The manifest at `prefix`, changed by `set`.
fn rewrite(prefix: &str, set: impl FnOnce(&mut crate::util::metadata::PintoMetadata)) {
    let path = crate::util::metadata::manifest_path(prefix);
    let path = std::path::Path::new(&path);
    let mut meta = crate::util::metadata::PintoMetadata::read(path).unwrap();
    set(&mut meta);
    meta.write(path).unwrap();
}

/// A cage model's query writes its dictionary as the model wrote its own:
/// divided by the δ the query is divided by when the model's is
/// batch-adjusted, and on the counts as read for an older model (its
/// manifest without the flag), whose query is still divided for the
/// encoder.
#[test]
fn a_cage_query_dictionary_is_adjusted_as_the_models() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = crate::test_support::two_sections(dir.path(), 8);
    let model = dir.path().join("model").to_str().unwrap().to_string();
    let input = [
        data.as_str(),
        "-c",
        coords.as_str(),
        "--coord-column-names",
        "x,y",
    ];
    let mut argv = input.to_vec();
    argv.extend_from_slice(&["-o", &model]);
    fit_cage(&argv);

    let adjusted = predict(dir.path(), &model, &input, "adjusted");
    let (divided, raw) = against_refits(dir.path(), &adjusted, &data, &coords);
    assert!(divided < 1e-3, "not the divided refit: {divided}");
    assert!(raw > 0.1, "the undivided refit too: {raw}");
    assert!(adjusted_in_manifest(&adjusted));

    rewrite(&model, |m| {
        m.outputs.feature_community_batch_adjusted = false
    });
    let older = predict(dir.path(), &model, &input, "older");
    assert!(
        std::path::Path::new(&format!("{older}.delta.parquet")).exists(),
        "the query is still divided for the encoder"
    );
    let (_, raw) = against_refits(dir.path(), &older, &data, &coords);
    assert!(raw < 1e-3, "an older model: not the undivided refit: {raw}");
    assert!(!adjusted_in_manifest(&older));
}

/// The cell pairs `prefix` wrote, each ordered, sorted.
fn pairs(prefix: &str) -> Vec<(Box<str>, Box<str>)> {
    let table =
        crate::lr_activity::io::read_pair_table(&format!("{prefix}.coord_pairs.parquet"), false)
            .unwrap();
    let mut p: Vec<(Box<str>, Box<str>)> = table
        .into_iter()
        .map(|e| {
            if e.left_cell <= e.right_cell {
                (e.left_cell, e.right_cell)
            } else {
                (e.right_cell, e.left_cell)
            }
        })
        .collect();
    p.sort();
    p
}

/// Without coordinates a cage fit with a δ builds its graph on the
/// batch-corrected projection and records it; its query, the training
/// data again, builds the same graph. An older model (a manifest
/// recording neither flag) has its query's graph built uncorrected, as
/// before: the graph an `--ignore-batch` query builds.
#[test]
fn a_cage_query_graph_is_built_as_the_models() {
    let dir = tempfile::tempdir().unwrap();
    let (data, _) = crate::test_support::two_sections(dir.path(), 8);
    let model = dir.path().join("model").to_str().unwrap().to_string();
    fit_cage(&[data.as_str(), "-o", &model]);
    let path = crate::util::metadata::manifest_path(&model);
    let meta = crate::util::metadata::PintoMetadata::read(std::path::Path::new(&path)).unwrap();
    assert!(
        meta.base_graph_batch_corrected(),
        "the fit records its graph"
    );
    let trained = pairs(&model);
    let query = predict(dir.path(), &model, &[data.as_str()], "query");
    assert_eq!(pairs(&query), trained);

    rewrite(&model, |m| {
        m.outputs.feature_community_batch_adjusted = false;
        m.graph.as_mut().unwrap().base_batch_corrected = false;
    });
    let older = pairs(&predict(dir.path(), &model, &[data.as_str()], "older"));
    let ignored = predict(
        dir.path(),
        &model,
        &[data.as_str(), "--ignore-batch"],
        "ignored",
    );
    assert_eq!(older, pairs(&ignored));
    assert_ne!(older, trained, "the corrected graph should differ");
}
