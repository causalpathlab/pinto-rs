//! The regression Stage 0 exists to prevent, stated end to end.
//!
//! `cage` scored a splice-channelized matrix as if each channel row were its own
//! feature: the feature axis doubled, and a feature's positives were drawn from half its
//! evidence twice over rather than from the feature once. The property that says it
//! is fixed is an EQUALITY — a two-channel matrix must produce exactly the
//! activities of the single-channel matrix whose counts are its per-feature sums.
//!
//! Also pinned here: an edge's weight is the pair's pooled raw count, over the
//! features at either end, which keeps a cell's depth.

use super::feature_gating::{build_feature_active_fine_edges, FeatureActiveEdges};
use crate::test_support::{numbered, sparse_vec};
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;

const N_CELLS: usize = 6;
const EDGES: [(u32, u32); 5] = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5)];

/// `(feature, cell) -> (spliced, unspliced)`. Deliberately uneven: some cells carry
/// only one track, which is where a per-row activity and a per-feature one differ
/// most.
const COUNTS: [(usize, usize, f32, f32); 10] = [
    (0, 0, 5.0, 2.0),
    (0, 1, 3.0, 0.0),
    (0, 3, 0.0, 4.0),
    (1, 0, 10.0, 1.0),
    (1, 1, 20.0, 6.0),
    (1, 4, 15.0, 0.0),
    (1, 5, 0.0, 25.0),
    (2, 2, 7.0, 3.0),
    (2, 4, 1.0, 1.0),
    (2, 5, 2.0, 0.0),
];

fn write(
    dir: &tempfile::TempDir,
    tag: &str,
    rows: &[Box<str>],
    triplets: &[(u64, u64, f32)],
) -> SparseIoVec {
    sparse_vec(dir.path(), tag, rows, &numbered("c", N_CELLS), triplets)
}

/// One row per feature, holding `spliced + unspliced`.
fn pooled_input(dir: &tempfile::TempDir) -> SparseIoVec {
    let rows: Vec<Box<str>> = (0..3).map(|g| format!("FEATURE{g}").into()).collect();
    let triplets: Vec<(u64, u64, f32)> = COUNTS
        .iter()
        .map(|&(g, c, s, u)| (g as u64, c as u64, s + u))
        .collect();
    write(dir, "pooled", &rows, &triplets)
}

/// Two rows per feature. The row ORDER is interleaved and puts a feature's nascent row
/// before its mature one, because nothing about the grammar promises otherwise
/// and a fold that only works on sorted rows would pass a friendlier fixture.
fn channelized_input(dir: &tempfile::TempDir) -> SparseIoVec {
    let mut rows: Vec<Box<str>> = Vec::new();
    for g in 0..3 {
        rows.push(format!("FEATURE{g}/count/unspliced").into());
        rows.push(format!("FEATURE{g}/count/spliced").into());
    }
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for &(g, c, s, u) in COUNTS.iter() {
        if u > 0.0 {
            triplets.push((2 * g as u64, c as u64, u));
        }
        if s > 0.0 {
            triplets.push((2 * g as u64 + 1, c as u64, s));
        }
    }
    triplets.sort_by_key(|&(r, c, _)| (c, r));
    write(dir, "channelized", &rows, &triplets)
}

fn activities(data: &SparseIoVec, edges: &[(u32, u32)]) -> anyhow::Result<FeatureActiveEdges> {
    let axis = FeatureAxis::resolve(&data.row_names()?)?;
    build_feature_active_fine_edges(data, edges, &axis, None, None)
}

/// Break it by indexing `feature_active_edges` with the row instead of the feature and
/// this fails on the very first assert: six entries where there are three features.
#[test]
fn two_channels_give_exactly_the_pooled_single_channel_activities() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let pooled = pooled_input(&dir);
    let channelized = channelized_input(&dir);

    // The axis itself, first: same features, same order, from row names alone.
    let axis = FeatureAxis::resolve(&channelized.row_names()?)?;
    assert!(axis.is_channelized());
    assert_eq!(axis.feature_names(), pooled.row_names()?.as_slice());

    let a = activities(&pooled, &EDGES)?;
    let b = activities(&channelized, &EDGES)?;
    assert_eq!(a.feature_active_edges.len(), 3, "one entry per feature");
    assert_eq!(b.feature_active_edges, a.feature_active_edges);
    assert_eq!(b.feature_active_edge_weights, a.feature_active_edge_weights);
    Ok(())
}

/// An edge's weight is the pair's pooled count of the feature, the two
/// tracks summed, and a feature counted at one end only is active there
/// with that end's count.
#[test]
fn an_edge_weighs_its_pooled_count_at_either_end() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let a = activities(&channelized_input(&dir), &EDGES)?;
    let weight = |g: usize, e: u32| {
        let k = a.feature_active_edges[g].iter().position(|&x| x == e)?;
        Some(a.feature_active_edge_weights[g][k])
    };
    // FEATURE0: cell 0 (5 + 2), cell 1 (3 + 0); cell 5 has none
    assert_eq!(weight(0, 0), Some(10.0), "both ends");
    assert_eq!(weight(0, 4), Some(7.0), "edge (0, 5): cell 0 only");
    // FEATURE1 on edge (0, 5): cell 0 (10 + 1), cell 5 (0 + 25)
    assert_eq!(weight(1, 4), Some(36.0));
    // FEATURE2 is counted at neither end of edge (0, 1)
    assert_eq!(weight(2, 0), None);
    Ok(())
}

/// The weight is the cell's raw count, not a depth-scaled one: two cells of
/// one composition, one nine times as deep, each beside the same third cell,
/// give their edges different weights.
#[test]
fn weights_are_pooled_raw_counts_whatever_the_depth() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    // cells 0 and 1 hold FEATURE0 at 2 and at 18; their neighbour, cell 2, at 4
    let triplets = [(0, 0, 2.0), (0, 1, 18.0), (0, 2, 4.0)];
    let data = sparse_vec(
        dir.path(),
        "depth",
        &numbered("FEATURE", 1),
        &numbered("c", 3),
        &triplets,
    );
    let a = activities(&data, &[(0, 2), (1, 2)])?;
    assert_eq!(a.feature_active_edges[0], vec![0, 1]);
    assert_eq!(a.feature_active_edge_weights[0], vec![6.0, 22.0]);
    Ok(())
}
