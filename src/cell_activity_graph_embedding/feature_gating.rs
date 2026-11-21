//! Per-feature cell activity: which edges each feature is active on, and how
//! strongly.
//!
//! - `FeatureActiveEdges.feature_active_edges[g]` is the precomputed list of
//!   global edge ids where feature `g` was counted at either endpoint.
//!   Built via an edge-driven merge of per-cell sorted feature lists, so
//!   the cost is O(n_edges × avg_features_per_cell) instead of the
//!   per-feature scan over all edges (O(n_features × n_edges)).
//! - `FeatureActiveEdges.feature_active_edge_weights[g][i]` = `x_g[u] + x_g[v]`,
//!   the pair's pooled count of the feature, for the corresponding edge. A
//!   positive is then one count of the feature at a contact, as `senna bge`
//!   draws a cell by its count. The feature-gated sampler rebuilds a
//!   per-call `WeightedIndex` only over this small list.
//!
//! Purely the data-derived sampling weight: this module owns no
//! learnable parameters, and nothing here is fit.

use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use legume_numeric::matrix::utils::generate_minibatch_intervals;
use nalgebra_sparse::{CooMatrix, CsrMatrix};

////////////////////
// FeatureActiveEdges //
////////////////////

pub struct FeatureActiveEdges {
    /// `feature_active_edges[g]`: global edge ids where either endpoint has
    /// a count of feature `g`. Sorted ascending. The struct is
    /// used at TWO levels: fine cell-cell edge ids as built here, then
    /// PB super-edge ids after `fold_active_edges_to_super`.
    pub feature_active_edges: Vec<Vec<u32>>,
    /// `feature_active_edge_weights[g][i]`: weight = `x_g[u] + x_g[v]` for
    /// edge `feature_active_edges[g][i]` (same length).
    pub feature_active_edge_weights: Vec<Vec<f32>>,
}

/// Each feature's count at every cell, and per feature the edges it was
/// counted on with the pair's pooled count `x_g[u] + x_g[v]` as the weight,
/// via an edge-driven merge of per-cell sorted feature lists.
///
/// The count is per FEATURE, not per matrix row: on a splice-channelized input
/// a feature's two rows are summed, so `x_g[c]` is one feature's evidence
/// rather than two half-evidence copies competing for the same positives.
///
/// The count is the cell's own, not scaled to a common library size: the
/// weight is how much of the feature was seen at an edge's two ends, so a
/// deeper cell carries more of it
/// (`feature_gating_tests::weights_are_pooled_raw_counts_whatever_the_depth`).
/// With `fine_to_super`, an edge that folds into no super edge (inside one
/// pseudobulk, or across batches) is skipped: the fold would drop it, and
/// those are most of the edges.
///
/// The counts are not divided by the batch effect: positives are drawn per
/// (feature, batch), within which a feature's `δ` is one constant, so it
/// would not move a single draw. The batch adjustment belongs where pairs
/// are compared across batches.
pub fn build_feature_active_fine_edges(
    data: &SparseIoVec,
    edges: &[(u32, u32)],
    axis: &FeatureAxis,
    fine_to_super: Option<&[Option<usize>]>,
) -> anyhow::Result<FeatureActiveEdges> {
    let n_cells = data.num_columns();
    let n_rows = data.num_rows();
    let n_features = axis.n_features();
    anyhow::ensure!(
        axis.n_rows() == n_rows,
        "feature axis has {} rows, data has {n_rows}",
        axis.n_rows()
    );
    anyhow::ensure!(
        fine_to_super.is_none_or(|m| m.len() == edges.len()),
        "{} edges, but a super-edge map of another length",
        edges.len()
    );

    // Step 1: read counts in column blocks; rayon-fold per-thread
    // (feature, cell, count) entries, then concat once on the main thread.
    // Avoids the per-row mutex on a shared CooMatrix. The COO→CSR conversion
    // SUMS duplicate entries, which is what folds a feature's channel rows
    // together; `log1p` is applied to the summed value afterwards.
    let intervals = generate_minibatch_intervals(n_cells, n_rows.max(1), None);

    let per_block: Vec<Vec<(usize, usize, f32)>> = intervals
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<Vec<(usize, usize, f32)>> {
            let csc = data.read_columns_csc(lb..ub)?;
            let mut local = Vec::new();
            for col in 0..(ub - lb) {
                let global_col = lb + col;
                let column = csc.col(col);
                for (&r, &v) in column.row_indices().iter().zip(column.values().iter()) {
                    if v > 0.0 {
                        local.push((axis.feature_of_row(r), global_col, v));
                    }
                }
            }
            Ok(local)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut coo = CooMatrix::<f32>::new(n_features, n_cells);
    for block in per_block {
        for (g, c, a) in block {
            coo.push(g, c, a);
        }
    }
    // summing the duplicates folds a feature's channel rows together
    let feature_by_cell_csr = CsrMatrix::from(&coo);

    // Step 2: invert to per-cell sorted (feature, activity) lists.
    // Iterating CSR rows in feature order gives each cell's feature-list
    // already sorted ascending, which is what the edge-merge needs.
    let mut cell_to_features: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_cells];
    for g in 0..n_features {
        let row = feature_by_cell_csr.row(g);
        for (&c, &v) in row.col_indices().iter().zip(row.values().iter()) {
            cell_to_features[c].push((g as u32, v));
        }
    }

    // Step 3: edge-driven merge — for each edge (u, v), walk the
    // two sorted lists in tandem, emitting `(edge_idx, x_u + x_v)` for
    // every feature at either end. Features are appended in edge-iteration
    // order, so per-feature edge lists are sorted ascending by construction.
    let mut feature_active_edges: Vec<Vec<u32>> = vec![Vec::new(); n_features];
    let mut feature_active_edge_weights: Vec<Vec<f32>> = vec![Vec::new(); n_features];
    for (e_idx, &(u, v)) in edges.iter().enumerate() {
        if fine_to_super.is_some_and(|m| m[e_idx].is_none()) {
            continue;
        }
        let gu = &cell_to_features[u as usize];
        let gv = &cell_to_features[v as usize];
        let (mut i, mut j) = (0usize, 0usize);
        let mut emit = |g: u32, w: f32| {
            feature_active_edges[g as usize].push(e_idx as u32);
            feature_active_edge_weights[g as usize].push(w);
        };
        while i < gu.len() || j < gv.len() {
            let order = match (gu.get(i), gv.get(j)) {
                (Some(a), Some(b)) => a.0.cmp(&b.0),
                (Some(_), None) => std::cmp::Ordering::Less,
                _ => std::cmp::Ordering::Greater,
            };
            match order {
                std::cmp::Ordering::Equal => {
                    emit(gu[i].0, gu[i].1 + gv[j].1);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    emit(gu[i].0, gu[i].1);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    emit(gv[j].0, gv[j].1);
                    j += 1;
                }
            }
        }
    }

    Ok(FeatureActiveEdges {
        feature_active_edges,
        feature_active_edge_weights,
    })
}

/// Fold per-feature ACTIVE FINE EDGES onto PB-PB super edges: each fine
/// edge's pooled-count weight is summed into the super edge it maps
/// to, and intra-PB fine edges (`fine_to_super[e] == None`) are dropped
/// — they fold into the node, matching `build_super_graph` semantics.
///
/// Returns the same `FeatureActiveEdges` shape one level up, so the
/// existing `(feature, batch)` cache builder consumes it unchanged: edge
/// ids are super-edge ids, sorted ascending per feature, weights parallel.
pub fn fold_active_edges_to_super(
    activities: FeatureActiveEdges,
    fine_to_super: &[Option<usize>],
) -> FeatureActiveEdges {
    use rayon::prelude::*;
    let folded: Vec<(Vec<u32>, Vec<f32>)> = activities
        .feature_active_edges
        .into_par_iter()
        .zip(activities.feature_active_edge_weights.into_par_iter())
        .map(|(edges, weights)| {
            let mut acc: std::collections::BTreeMap<u32, f32> = std::collections::BTreeMap::new();
            for (&e, &w) in edges.iter().zip(weights.iter()) {
                if let Some(s) = fine_to_super[e as usize] {
                    *acc.entry(s as u32).or_insert(0.0) += w;
                }
            }
            acc.into_iter().unzip()
        })
        .collect();
    let (feature_active_edges, feature_active_edge_weights): (Vec<_>, Vec<_>) =
        folded.into_iter().unzip();
    FeatureActiveEdges {
        feature_active_edges,
        feature_active_edge_weights,
    }
}
