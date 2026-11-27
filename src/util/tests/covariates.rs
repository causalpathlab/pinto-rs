//! Per-cell covariates as a second axis of batch, for δ alone: the cells'
//! δ labels take their bins, their plain labels (sections, pairs) stay, and
//! a query is binned by the fit's recorded model as the fit's cells were.

use super::ignore_batch::preprocess_with;
use crate::test_support::write_zarr;
use crate::util::covariates::{files, CovariateModel};
use crate::util::input::SrtInputArgs;
use crate::util::srt_pipeline::{
    preprocess_srt, BatchEstimate, FeatureAxisMode, SrtPreprocessConfig,
};
use clap::Parser;

const CELLS: usize = 80;
const MITO: [&str; 12] = [
    "MT-ATP6", "MT-ATP8", "MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND2", "MT-ND3",
    "MT-ND4", "MT-ND4L", "MT-ND5",
];

/// Whether cell `c` carries the high mitochondrial share.
fn mito_high(c: usize) -> bool {
    c < CELLS / 2
}

/// Two data files of two cell types (alternate cells) crossed with a high
/// and a low mitochondrial share (the first and second half), the types far
/// apart in expression so like cells group by type, not by share.
fn two_batches(dir: &std::path::Path) -> Vec<Box<str>> {
    let mut rows: Vec<String> = MITO.iter().map(|g| (*g).to_string()).collect();
    rows.extend((0..24).map(|g| format!("GENE{g}")));
    let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
    ["a", "b"]
        .into_iter()
        .map(|tag| {
            let mut triplets = Vec::new();
            for c in 0..CELLS {
                let mito = if mito_high(c) { 12.0 } else { 2.0 };
                for g in 0..MITO.len() {
                    triplets.push((g as u64, c as u64, mito + ((c + g) % 3) as f32));
                }
                for g in 0..24 {
                    let own = (g < 12) == (c % 2 == 0);
                    let base = if own { 40.0 } else { 1.0 };
                    let fold = if tag == "b" && g % 3 == 0 { 2.0 } else { 1.0 };
                    let x = (base + ((c * 7 + g) % 5) as f32) * fold;
                    triplets.push(((MITO.len() + g) as u64, c as u64, x));
                }
            }
            write_zarr(dir, tag, &rows, CELLS, &triplets)
        })
        .collect()
}

const BINNED: [&str; 4] = ["--covariate", "mito", "--covariate-bins", "2"];

#[test]
fn delta_labels_take_the_bins_and_the_plain_labels_stay() {
    let dir = tempfile::tempdir().unwrap();
    let data = two_batches(dir.path());
    let out = dir.path().join("fit").to_str().unwrap().to_string();
    let pre = preprocess_with(&data, &out, &BINNED, BatchEstimate::Fit);

    assert!(pre.covariates_binned);
    let mut plain = pre.batch_membership.clone();
    plain.sort();
    plain.dedup();
    assert_eq!(plain, vec!["a".into(), "b".into()] as Vec<Box<str>>);
    // a cell's δ label is its batch with a bin, or its batch alone where
    // its group of like cells sits in one bin
    for (d, b) in pre.delta_membership.iter().zip(&pre.batch_membership) {
        assert!(
            d == b || d.starts_with(&format!("{b}|cv")[..]),
            "{d} of batch {b}"
        );
    }
    // within each batch, the high and low shares are in different bins
    for batch in 0..2 {
        let at = |c: usize| &pre.delta_membership[batch * CELLS + c];
        assert_eq!(at(0), at(1), "same share, same bin");
        assert_ne!(at(0), at(CELLS - 1), "the two shares, two bins");
    }
    let mut levels = pre.delta_membership.clone();
    levels.sort();
    levels.dedup();
    assert_eq!(pre.batch_labels, levels, "δ's columns: the δ labels");
    assert_eq!(
        pre.batch_effects.as_ref().map(|p| p.batches.clone()),
        Some(pre.batch_labels.clone())
    );
    for path in [files::model(&out), files::cells(&out), files::effect(&out)] {
        assert!(std::path::Path::new(&path).is_file(), "{path} written");
    }
}

#[test]
fn nothing_is_binned_without_a_delta() {
    let dir = tempfile::tempdir().unwrap();
    let data = two_batches(dir.path());
    let out = dir.path().join("fit").to_str().unwrap().to_string();
    // the default `all` bins nothing without a δ; a list asked for is refused
    let pre = preprocess_with(&data, &out, &["--ignore-batch"], BatchEstimate::Fit);
    assert!(!pre.covariates_binned);
    assert_eq!(pre.delta_membership, pre.batch_membership);
    assert!(!std::path::Path::new(&files::model(&out)).exists());
    let named = SrtInputArgs::try_parse_from(
        [
            "pinto",
            data[0].as_ref(),
            "-o",
            out.as_str(),
            "--ignore-batch",
        ]
        .into_iter()
        .chain(BINNED),
    )
    .unwrap();
    let refused = preprocess_srt(SrtPreprocessConfig {
        covariate_model: None,
        common: &named,
        ablation: None,
        batch_effects: BatchEstimate::Fit,
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: None,
        cell_projection: false,
        auto_batch_memory_check: None,
        batch_fold: crate::util::srt_pipeline::BatchFoldKind::None,
    });
    assert!(refused.is_err(), "--covariate mito with --ignore-batch");

    let none = preprocess_with(&data, &out, &["--covariate", "none"], BatchEstimate::Fit);
    assert!(!none.covariates_binned);
    assert_eq!(none.batch_labels.len(), 2);
}

#[test]
fn a_query_is_binned_as_the_fit_s_cells_were() {
    let dir = tempfile::tempdir().unwrap();
    let data = two_batches(dir.path());
    let out = dir.path().join("fit").to_str().unwrap().to_string();
    let fit = preprocess_with(&data, &out, &BINNED, BatchEstimate::Fit);
    let model = CovariateModel::read(std::path::Path::new(&files::model(&out)))
        .unwrap()
        .expect("the fit wrote its model");

    let query_out = dir.path().join("query").to_str().unwrap().to_string();
    let mut argv = vec!["pinto"];
    argv.extend(data.iter().map(|f| f.as_ref()));
    // the query's own flags say nothing: the model's bins decide
    argv.extend_from_slice(&["-o", &query_out, "--no-qc", "--knn-expr", "5"]);
    let args = SrtInputArgs::try_parse_from(argv).unwrap();
    let query = preprocess_srt(SrtPreprocessConfig {
        covariate_model: Some(&model),
        common: &args,
        ablation: None,
        batch_effects: BatchEstimate::Query {
            all_known: false,
            base_corrected: false,
        },
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: None,
        cell_projection: false,
        auto_batch_memory_check: None,
        batch_fold: crate::util::srt_pipeline::BatchFoldKind::None,
    })
    .unwrap();
    assert!(query.covariates_binned);
    // the fit's binned cells take the same bins; a cell the fit kept plain
    // (its group in one bin) is binned, as new cells form no groups
    for ((q, f), b) in query
        .delta_membership
        .iter()
        .zip(&fit.delta_membership)
        .zip(&fit.batch_membership)
    {
        if f != b {
            assert_eq!(q, f);
        } else {
            assert!(q.starts_with(&format!("{b}|cv")[..]), "{q} of batch {b}");
        }
    }
    assert_eq!(query.batch_membership, fit.batch_membership);
    assert!(
        !std::path::Path::new(&files::model(&query_out)).exists(),
        "a query records no model of its own"
    );
}
