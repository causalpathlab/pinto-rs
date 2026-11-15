//! Without coordinates the base graph is an expression kNN graph. With two
//! or more batches it is built on the batch-corrected projection (the one
//! the δ estimate collapses on), so neighbours match cell state rather than
//! batch; under `--ignore-batch` it is built as before, uncorrected.

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
    use data_beans::alg::random_projection::RandProjOps;
    let dir = tempfile::tempdir().unwrap();
    let files = two_states_two_batches(dir.path(), 40);
    let out = dir.path().join("off").to_str().unwrap().to_string();
    let ignored = preprocess(&files, &out, &["--ignore-batch", "-p", "8"]);
    const PROJ_DIM: usize = 8;
    let plain = ignored
        .data_vec
        .project_columns_with_batch_correction(PROJ_DIM, None, None::<&[Box<str>]>)
        .unwrap();
    let graph = crate::util::cell_pairs::build_expression_knn(
        &plain.proj,
        crate::util::cell_pairs::SrtCellPairsArgs {
            knn: ignored.knn.base,
            block_size: None,
            reciprocal: false,
        },
    )
    .unwrap();
    assert_eq!(graph.edges, ignored.graph.edges);
}
