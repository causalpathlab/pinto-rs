//! Without coordinates the base graph is an expression kNN graph. When a
//! fit estimates a δ over two or more batches it is built on the
//! batch-corrected projection (the one the estimate collapses on), so
//! neighbours match cell state rather than batch; a query's is built as
//! the model's was. Otherwise (`--ignore-batch`, `--auto-batch`, one
//! batch, an older model) it is built as before, uncorrected.

use super::ignore_batch::preprocess;
use crate::test_support::write_zarr;

/// Two data files (batches) of `cells` cells each, half of one state (high
/// on features 0..4) and half of another (high on 4..8). The second batch
/// carries a fold on features 8..12, which neither state uses.
fn two_states_two_batches(dir: &std::path::Path, cells: usize) -> Vec<Box<str>> {
    let rows: Vec<String> = (0..12).map(|g| format!("GENE{g}")).collect();
    let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
    ["a", "b"]
        .into_iter()
        .map(|tag| {
            let mut triplets = Vec::new();
            for c in 0..cells as u64 {
                let state = c % 2;
                for g in 0..12u64 {
                    let jitter = noise(c, g, tag == "b");
                    let on = (g / 4) == state;
                    let value = if g >= 8 {
                        if tag == "b" {
                            6.0 + jitter
                        } else {
                            1.0 + jitter
                        }
                    } else if on {
                        20.0 + jitter
                    } else {
                        1.0 + jitter
                    };
                    triplets.push((g, c, value));
                }
            }
            write_zarr(dir, tag, &rows, cells, &triplets)
        })
        .collect()
}

/// A deterministic draw in `[0, 4)` per (cell, feature, batch), so no two
/// cells share a profile.
fn noise(c: u64, g: u64, b: bool) -> f32 {
    let mut h = (c << 20) ^ (g << 4) ^ (b as u64);
    h = h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 32;
    (h % 4000) as f32 / 1000.0
}

/// The share of `pre.graph`'s edges between cells of different batches.
fn cross_batch_share(pre: &crate::util::srt_pipeline::SrtPreprocessed) -> f64 {
    let b = &pre.batch_membership;
    let edges = &pre.graph.edges;
    let cross = edges.iter().filter(|&&(i, j)| b[i] != b[j]).count();
    cross as f64 / edges.len() as f64
}

#[test]
fn the_expression_graph_mixes_batches_unless_batch_is_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_states_two_batches(dir.path(), 80);
    let out = |tag: &str| dir.path().join(tag).to_str().unwrap().to_string();

    let corrected = preprocess(&files, &out("on"), &[]);
    let ignored = preprocess(&files, &out("off"), &["--ignore-batch"]);
    let (on, off) = (cross_batch_share(&corrected), cross_batch_share(&ignored));
    assert!(
        off < 0.05,
        "uncorrected, neighbours should match batch: cross-batch share {off}"
    );
    assert!(
        on > 0.3,
        "corrected, neighbours should cross batches: cross-batch share {on} (uncorrected {off})"
    );
}

/// `--ignore-batch` builds the graph on the uncorrected projection, as a
/// single-batch run does: the same graph as a run that never saw the labels.
#[test]
fn under_ignore_batch_the_graph_is_the_uncorrected_one() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_states_two_batches(dir.path(), 40);
    let out = dir.path().join("off").to_str().unwrap().to_string();
    let ignored = preprocess(&files, &out, &["--ignore-batch", "-p", "8"]);
    assert_eq!(uncorrected_graph(&ignored, 8), ignored.graph.edges);
}

/// The graph `pre` was built on, rebuilt on the uncorrected projection.
fn uncorrected_graph(
    pre: &crate::util::srt_pipeline::SrtPreprocessed,
    proj_dim: usize,
) -> Vec<(usize, usize)> {
    use data_beans::alg::random_projection::RandProjOps;
    let plain = pre
        .data_vec
        .project_columns_with_batch_correction(proj_dim, None, None::<&[Box<str>]>)
        .unwrap();
    crate::util::cell_pairs::build_expression_knn(
        &plain.proj,
        crate::util::cell_pairs::SrtCellPairsArgs {
            knn: pre.knn.base,
            block_size: None,
            reciprocal: false,
        },
    )
    .unwrap()
    .edges
}

/// `--auto-batch` takes its batches from the base graph's components, so
/// the graph is built as before, on the uncorrected projection: the labels
/// it starts from (one per file) are not the batches it ends with.
#[test]
fn under_auto_batch_the_graph_is_the_uncorrected_one() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_states_two_batches(dir.path(), 40);
    let out = dir.path().join("auto").to_str().unwrap().to_string();
    let auto = preprocess(&files, &out, &["--auto-batch", "-p", "8"]);
    assert_eq!(uncorrected_graph(&auto, 8), auto.graph.edges);
}

/// The cells' projection a corrected expression-mode run hands back is
/// the base graph's, the one the δ estimate collapsed on: the same matrix
/// a fresh projection with that `-p`, block size and batch membership
/// gives.
#[test]
fn the_reused_projection_is_a_fresh_one() {
    use crate::util::input::SrtInputArgs;
    use crate::util::srt_pipeline::{
        preprocess_srt, BatchEstimate, FeatureAxisMode, SrtPreprocessConfig,
    };
    use clap::Parser;
    use data_beans::alg::random_projection::RandProjOps;
    let dir = tempfile::tempdir().unwrap();
    let files = two_states_two_batches(dir.path(), 40);
    let out = dir.path().join("on").to_str().unwrap().to_string();
    let mut argv = vec!["pinto"];
    argv.extend(files.iter().map(|f| f.as_ref()));
    argv.extend_from_slice(&["-o", &out, "--no-qc", "-p", "8", "--block-size", "7"]);
    let args = SrtInputArgs::try_parse_from(argv).unwrap();
    let pre = preprocess_srt(SrtPreprocessConfig {
        common: &args,
        ablation: None,
        fisher_weights: false,
        batch_effects: BatchEstimate::Fit,
        feature_axis: FeatureAxisMode::Rows,
        feature_kind: None,
        cell_projection: true,
    })
    .unwrap();
    assert!(pre.batch_effects.is_some(), "two files, two batches");
    let kept = pre.cell_proj.as_ref().expect("asked for");
    let fresh = pre
        .data_vec
        .project_columns_with_batch_correction(8, Some(7), Some(pre.batch_membership.as_slice()))
        .unwrap();
    assert_eq!(kept.proj, fresh.proj);
    assert_eq!(kept.basis, fresh.basis);
}

/// The graph a fit records building: on the batch-corrected projection
/// when it estimated a δ, else (`--ignore-batch`, `--auto-batch`, one
/// batch) on the uncorrected one, as before.
#[test]
fn a_fit_reports_whether_its_graph_is_corrected() {
    let dir = tempfile::tempdir().unwrap();
    let files = two_states_two_batches(dir.path(), 40);
    let out = |tag: &str| dir.path().join(tag).to_str().unwrap().to_string();
    let corrected = preprocess(&files, &out("on"), &["-p", "8"]);
    assert!(corrected.batch_effects.is_some());
    assert!(corrected.base_batch_corrected);
    assert_ne!(uncorrected_graph(&corrected, 8), corrected.graph.edges);
    for (tag, extra) in [("off", "--ignore-batch"), ("auto", "--auto-batch")] {
        let pre = preprocess(&files, &out(tag), &[extra, "-p", "8"]);
        assert!(!pre.base_batch_corrected, "{extra}");
    }
    let one = preprocess(&files[..1], &out("one"), &["-p", "8"]);
    assert!(!one.base_batch_corrected, "one batch");
    assert_eq!(uncorrected_graph(&one, 8), one.graph.edges);
}

/// A query builds its expression graph as the model's was, whatever it
/// estimates: corrected for a model whose manifest says its graph was,
/// uncorrected for any other (an older model), as before.
#[test]
fn a_query_graph_is_built_as_the_models() {
    use super::ignore_batch::preprocess_with;
    use crate::util::srt_pipeline::BatchEstimate;
    let dir = tempfile::tempdir().unwrap();
    let files = two_states_two_batches(dir.path(), 40);
    let out = |tag: &str| dir.path().join(tag).to_str().unwrap().to_string();
    let fit = preprocess(&files, &out("fit"), &["-p", "8"]);
    for all_known in [false, true] {
        let query = |base_corrected: bool, tag: &str| {
            preprocess_with(
                &files,
                &out(tag),
                &["-p", "8"],
                BatchEstimate::Query {
                    all_known,
                    base_corrected,
                },
            )
        };
        let new = query(true, "new");
        assert!(new.base_batch_corrected, "all_known {all_known}");
        assert_eq!(new.graph.edges, fit.graph.edges, "all_known {all_known}");
        let older = query(false, "older");
        assert!(!older.base_batch_corrected, "all_known {all_known}");
        assert_eq!(
            uncorrected_graph(&older, 8),
            older.graph.edges,
            "all_known {all_known}"
        );
    }
}
