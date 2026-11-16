//! A fit on a residual's excess counts estimates no batch effect: the
//! files are recognised by the `residual.json` that names them, and the
//! shared preprocessing then reads them as under `--ignore-batch`.

use super::ignore_batch::{preprocess, two_batches};
use crate::test_support::mark_residual;
use crate::util::input::SrtInputArgs;
use crate::util::metadata::NoBatchEffect;
use crate::util::residual_input::{naming_marker, residual_of_files};
use crate::util::srt_pipeline::{
    preprocess_srt, BatchEstimate, FeatureAxisMode, SrtPreprocessConfig,
};
use clap::Parser;
use std::path::Path;

/// [`two_batches`] in a new folder `name` of `dir`.
fn batches_in(dir: &Path, name: &str) -> Vec<Box<str>> {
    let folder = dir.join(name);
    std::fs::create_dir_all(&folder).unwrap();
    two_batches(&folder, 60)
}

/// [`batches_in`], marked as a residual's excess counts.
fn residual_in(dir: &Path, name: &str) -> Vec<Box<str>> {
    let files = batches_in(dir, name);
    mark_residual(&files.iter().map(|f| f.as_ref()).collect::<Vec<_>>());
    files
}

/// What preprocessing a fit of `files` with `extra` flags says.
fn fit_error(files: &[Box<str>], out: &str, extra: &[&str]) -> String {
    let mut argv = vec!["pinto"];
    argv.extend(files.iter().map(|f| f.as_ref()));
    argv.extend_from_slice(&["-o", out, "--no-qc", "--knn-expr", "5"]);
    argv.extend_from_slice(extra);
    let args = SrtInputArgs::try_parse_from(argv).expect("args should parse");
    preprocess_srt(SrtPreprocessConfig {
        common: &args,
        ablation: None,
        batch_effects: BatchEstimate::Fit,
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: None,
        cell_projection: false,
    })
    .err()
    .expect("preprocessing should refuse")
    .to_string()
}

/// A file is a residual's when the `residual.json` beside it names it: not
/// without that file, not when it names other files only, and not the raw
/// file it names.
#[test]
fn a_residual_s_file_is_the_one_its_marker_names() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    for f in ["res/S1.zarr", "res/S2.zarr", "res/T1.zarr", "plain/S1.zarr"] {
        std::fs::create_dir_all(d.join(f).parent().unwrap()).unwrap();
        std::fs::write(d.join(f), b"").unwrap();
    }
    let marker = d.join("res/residual.json");
    assert_eq!(naming_marker(&d.join("res/S1.zarr")), None, "no marker yet");

    // as `pinto residual` writes it: the raw files, and each file written
    std::fs::write(
        &marker,
        r#"{"counts": "excess",
            "raw_data_files": ["../raw/S1.zarr.zip"],
            "files": [{"data_file": "S2.h5", "residual_file": "res/S2.zarr"}]}"#,
    )
    .unwrap();
    assert_eq!(naming_marker(&d.join("res/S1.zarr")), Some(marker.clone()));
    assert_eq!(naming_marker(&d.join("res/S2.zarr")), Some(marker.clone()));
    assert_eq!(naming_marker(&d.join("res/T1.zarr")), None, "not named");
    assert_eq!(naming_marker(&d.join("plain/S1.zarr")), None, "no marker");

    // a raw file the marker names, kept in its folder, is not excess
    std::fs::write(
        &marker,
        r#"{"counts": "excess", "raw_data_files": ["S1.zarr"]}"#,
    )
    .unwrap();
    assert_eq!(naming_marker(&d.join("res/S1.zarr")), None, "the raw file");

    // a marker of other counts marks nothing
    std::fs::write(
        &marker,
        r#"{"counts": "raw", "raw_data_files": ["../raw/S1.zarr"]}"#,
    )
    .unwrap();
    assert_eq!(naming_marker(&d.join("res/S1.zarr")), None);
}

/// The rule holds for a list when one residual names all of it; a list
/// that mixes a residual's files with others, or two residuals', is
/// refused.
#[test]
fn a_list_is_one_residual_s_or_none_s() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let plain = batches_in(d, "plain");
    let res = residual_in(d, "res");
    let other = residual_in(d, "other");
    assert_eq!(residual_of_files(&plain).unwrap(), None);
    assert_eq!(
        residual_of_files(&res).unwrap(),
        Some(d.join("res/residual.json"))
    );

    for mixed in [
        vec![res[0].clone(), plain[1].clone()],
        vec![plain[0].clone(), res[1].clone()],
    ] {
        let e = residual_of_files(&mixed).unwrap_err().to_string();
        assert!(
            e.contains("holds a residual's excess counts") && e.contains("does not"),
            "{e}"
        );
        // and a fit of it goes no further
        let e = fit_error(&mixed, d.join("mixed").to_str().unwrap(), &[]);
        assert!(e.contains("holds a residual's excess counts"), "{e}");
    }
    let two = vec![res[0].clone(), other[1].clone()];
    let e = residual_of_files(&two).unwrap_err().to_string();
    assert!(e.contains("two residuals"), "{e}");
}

/// A fit's preprocessing estimates no batch effect on a residual's files,
/// with or without `--ignore-batch`, writes no δ and says why; the files
/// still label their cells. The same counts in a folder without the marker
/// are estimated on.
#[test]
fn a_fit_on_a_residual_s_files_estimates_no_batch_effect() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let out = |tag: &str| d.join(tag).to_str().unwrap().to_string();
    let delta = |tag: &str| Path::new(&format!("{}.delta.parquet", out(tag))).exists();

    let plain = batches_in(d, "plain");
    let estimated = preprocess(&plain, &out("on"), &[]);
    assert!(estimated.batch_effects.is_some(), "two files, two batches");
    assert_eq!(estimated.no_batch_effect, None);
    assert!(delta("on"));

    let res = residual_in(d, "res");
    for (tag, extra) in [("r", &[][..]), ("ri", &["--ignore-batch"][..])] {
        let pre = preprocess(&res, &out(tag), extra);
        assert!(pre.batch_effects.is_none(), "{tag}");
        assert_eq!(
            pre.no_batch_effect,
            Some(NoBatchEffect::ResidualInput),
            "{tag}"
        );
        assert!(!delta(tag), "{tag}: a δ table was written");
        assert_eq!(pre.batch_labels.len(), 2, "{tag}: the files' labels stay");
    }

    let ignored = preprocess(&plain, &out("off"), &["--ignore-batch"]);
    assert_eq!(ignored.no_batch_effect, Some(NoBatchEffect::IgnoreBatch));
}

/// The options that only set a batch-effect estimate up are refused on a
/// residual's files, each named with the reason.
#[test]
fn batch_options_are_refused_on_a_residual_s_files() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let res = residual_in(d, "res");
    let out = d.join("out").to_str().unwrap().to_string();
    let labels = d.join("labels.txt");
    std::fs::write(&labels, "a\n".repeat(60)).unwrap();
    let batch_files = format!("{0},{0}", labels.to_str().unwrap());
    let clones = d.join("x.clones.parquet");
    for (flag, extra) in [
        ("--batch-files", vec!["--batch-files", batch_files.as_str()]),
        ("--auto-batch", vec!["--auto-batch"]),
        (
            "--cnv-clones",
            vec!["--cnv-clones", clones.to_str().unwrap()],
        ),
    ] {
        let e = fit_error(&res, &out, &extra);
        assert!(
            e.starts_with(&format!("{flag} sets up a batch-effect estimate"))
                && e.contains("none is made on a residual's excess counts")
                && e.ends_with(&format!("Drop {flag}")),
            "{e}"
        );
    }
}

/// dsvd's own per-cell projection is not corrected by the files of a
/// residual either: it is the uncorrected one, as under `--ignore-batch`.
#[test]
fn dsvd_s_projection_is_uncorrected_on_a_residual_s_files() {
    use data_beans::alg::random_projection::RandProjOps;
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let res = residual_in(d, "res");
    let out = d.join("out").to_str().unwrap().to_string();
    let mut argv = vec!["pinto"];
    argv.extend(res.iter().map(|f| f.as_ref()));
    argv.extend_from_slice(&["-o", &out, "--no-qc", "--knn-expr", "5", "-p", "8"]);
    let args = SrtInputArgs::try_parse_from(argv).unwrap();
    let pre = preprocess_srt(SrtPreprocessConfig {
        common: &args,
        ablation: None,
        batch_effects: BatchEstimate::Fit,
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: None,
        cell_projection: false,
    })
    .unwrap();
    let got = crate::svd::fit::dsvd_cell_projection(
        None,
        pre.batch_effects.is_some(),
        pre.no_batch_effect.is_some(),
        &args,
        &pre.data_vec,
        &pre.batch_membership,
    )
    .unwrap();
    let project = |by_batch: bool| {
        pre.data_vec
            .project_columns_with_batch_correction(
                8,
                None,
                by_batch.then_some(pre.batch_membership.as_slice()),
            )
            .unwrap()
            .proj
    };
    assert_eq!(got.proj, project(false));
    assert_ne!(got.proj, project(true), "the batches change the projection");
}
