//! `--ignore-batch`: no batch effect is estimated, so none is removed. The
//! batch labels themselves stay, since they also say which section a cell
//! is in.

use crate::test_support::write_zarr;
use crate::util::input::SrtInputArgs;
use crate::util::srt_pipeline::FeatureAxisMode;
use crate::util::srt_pipeline::{preprocess_srt, SrtPreprocessConfig, SrtPreprocessed};
use clap::Parser;

/// Two data files of `cells` cells over twelve features, the second with its
/// first six features three times the first's: a batch effect to estimate.
fn two_batches(dir: &std::path::Path, cells: usize) -> Vec<Box<str>> {
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

fn preprocess(files: &[Box<str>], out: &str, extra: &[&str]) -> SrtPreprocessed {
    let mut argv = vec!["pinto"];
    argv.extend(files.iter().map(|f| f.as_ref()));
    argv.extend_from_slice(&["-o", out, "--no-qc", "--knn-expr", "5"]);
    argv.extend_from_slice(extra);
    let args = SrtInputArgs::try_parse_from(argv).expect("args should parse");
    preprocess_srt(SrtPreprocessConfig {
        common: &args,
        ablation: None,
        fisher_weights: false,
        batch_effects: true,
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: None,
        cell_projection: false,
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
