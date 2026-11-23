//! `--ignore-batch`: no batch effect is estimated, so none is removed. The
//! batch labels themselves stay, since they also say which section a cell
//! is in.

use crate::test_support::write_zarr;
use crate::util::input::SrtInputArgs;
use crate::util::srt_pipeline::FeatureAxisMode;
use crate::util::srt_pipeline::{
    preprocess_srt, BatchEstimate, SrtPreprocessConfig, SrtPreprocessed,
};
use clap::Parser;

/// Two data files of `cells` cells over twelve features, the second with its
/// first six features three times the first's: a batch effect to estimate.
pub(super) fn two_batches(dir: &std::path::Path, cells: usize) -> Vec<Box<str>> {
    let rows: Vec<String> = (0..12).map(|g| format!("GENE{g}")).collect();
    let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
    ["a", "b"]
        .into_iter()
        .map(|tag| {
            let mut triplets = Vec::new();
            for c in 0..cells as u64 {
                for g in 0..12u64 {
                    let base = 2.0 + ((c + 2 * g) % 5) as f32;
                    let fold = if tag == "b" && g < 6 { 3.0 } else { 1.0 };
                    triplets.push((g, c, base * fold));
                }
            }
            write_zarr(dir, tag, &rows, cells, &triplets)
        })
        .collect()
}

pub(super) fn preprocess(files: &[Box<str>], out: &str, extra: &[&str]) -> SrtPreprocessed {
    preprocess_with(files, out, extra, BatchEstimate::Fit)
}

/// [`preprocess`] doing `batch` about batch effects.
pub(super) fn preprocess_with(
    files: &[Box<str>],
    out: &str,
    extra: &[&str],
    batch: BatchEstimate,
) -> SrtPreprocessed {
    let mut argv = vec!["pinto"];
    argv.extend(files.iter().map(|f| f.as_ref()));
    argv.extend_from_slice(&["-o", out, "--no-qc", "--knn-expr", "5"]);
    argv.extend_from_slice(extra);
    let args = SrtInputArgs::try_parse_from(argv).expect("args should parse");
    preprocess_srt(SrtPreprocessConfig {
        common: &args,
        ablation: None,
        batch_effects: batch,
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: None,
        cell_projection: false,
        auto_batch_memory_check: None,
        batch_fold: crate::util::srt_pipeline::BatchFoldKind::None,
    })
    .expect("preprocessing should run")
}

#[test]
fn no_batch_effect_is_estimated_and_the_labels_stay() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_batches(dir.path(), 60);
    let out = |tag: &str| dir.path().join(tag).to_str().unwrap().to_string();

    let corrected = preprocess(&files, &out("on"), &[]);
    assert!(corrected.batch_effects.is_some(), "two files, two batches");

    let ignored = preprocess(&files, &out("off"), &["--ignore-batch"]);
    assert!(ignored.batch_effects.is_none());
    assert_eq!(ignored.batch_membership, corrected.batch_membership);
    let mut labels = ignored.batch_membership.clone();
    labels.sort();
    labels.dedup();
    assert_eq!(labels.len(), 2, "the files still label their cells");
}

/// One asks for batches the files do not give and the other for no
/// correction between any.
#[test]
fn it_cannot_be_combined_with_auto_batch() {
    let argv = [
        "pinto",
        "a.zarr",
        "-o",
        "out",
        "--ignore-batch",
        "--auto-batch",
    ];
    assert!(SrtInputArgs::try_parse_from(argv).is_err());
}

/// dsvd's per-cell projection is corrected by batch unless
/// `--ignore-batch`: preprocessing's is taken when it is that one (with a
/// δ, or under `--ignore-batch`), and dsvd's own is taken otherwise (one
/// batch); either way, the projection a fresh pass gives.
#[test]
fn dsvd_takes_preprocessings_projection_when_it_is_its_own() {
    use data_beans::alg::random_projection::RandProjOps;
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = crate::test_support::two_sections(dir.path(), 8);
    let first = data.split(',').next().unwrap().to_string();
    let first_coords = coords.split(',').next().unwrap().to_string();
    let out = dir.path().join("run").to_str().unwrap().to_string();
    let cases = [
        (&data, &coords, None, true),
        (&data, &coords, Some("--ignore-batch"), false),
        (&first, &first_coords, None, false),
    ];
    for (data, coords, extra, has_delta) in cases {
        let mut argv = vec![
            "pinto",
            data.as_str(),
            "-c",
            coords.as_str(),
            "--coord-column-names",
            "x,y",
            "-o",
            &out,
            "-p",
            "8",
        ];
        argv.extend(extra);
        let args = SrtInputArgs::try_parse_from(argv).unwrap();
        let pre = preprocess_srt(SrtPreprocessConfig {
            common: &args,
            ablation: None,
            batch_effects: BatchEstimate::Fit,
            feature_axis: FeatureAxisMode::Rows,
            feature_kind: None,
            cell_projection: true,
            auto_batch_memory_check: None,
            batch_fold: crate::util::srt_pipeline::BatchFoldKind::None,
        })
        .unwrap();
        assert_eq!(pre.batch_effects.is_some(), has_delta, "{extra:?}");
        let got = crate::svd::fit::dsvd_cell_projection(
            pre.cell_proj,
            pre.batch_effects.is_some(),
            &args,
            &pre.data_vec,
            &pre.batch_membership,
        )
        .unwrap();
        let fresh = pre
            .data_vec
            .project_columns_with_batch_correction(
                8,
                None,
                (!args.batch.ignore_batch).then_some(pre.batch_membership.as_slice()),
            )
            .unwrap();
        assert_eq!(got.proj, fresh.proj, "{extra:?}, delta {has_delta}");
        assert_eq!(got.basis, fresh.basis, "{extra:?}, delta {has_delta}");
    }
}
