//! Per-feature gated draws of super edges (cage's pseudobulk pairs,
//! cage-cell's cell pairs): edge ids here are super-edge ids, weights are
//! per-feature activity sums folded onto those super edges.
//!
//! Precompute the per-(feature, batch) positive distribution once at start
//! (sorted-intersection of feature-active super edges with the batch's
//! retained super edges, plus a `WeightedIndex` weighted by the folded
//! activity), then on every draw look up the cached entry; negatives are
//! other super edges of the batch's pool (`PerBatchUnitSampler::edge_indices`).

use graph_embedding_util::loss::PerBatchUnitSampler;
use rand::{Rng, RngExt};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use rayon::prelude::*;

use crate::cell_activity_graph_embedding::feature_gating::FeatureActiveEdges;

////////////////////
// FeatureExpBatchCache //
////////////////////

/// Precomputed positive distribution for one `(feature, batch)` pair.
pub struct FeatureExpBatchEntry {
    /// `WeightedIndex` over the feature-batch intersected super-edge list.
    /// Weights are per-super-edge SUMS of the fine endpoint products
    /// `a_g[u] · a_g[v]` (or uniform when the sum would underflow).
    pub pos: WeightedIndex<f32>,
    /// Maps each local index in `pos` back to the global SUPER-EDGE id
    /// in `PbFrame::super_edges`. Required by
    /// `sample_unit_chain_batch_with_pos`.
    pub local_to_global: Vec<u32>,
}

/// Per-feature per-batch precomputed positive distributions. Built once at
/// the start of training; consumed (by reference) on every sample call.
///
/// Memory: O(n_features · n_batches · mean_active_edges_per_feature), typically
/// 50-300 MB for an 18k-feature Visium dataset with one batch. Released
/// after training completes.
pub struct FeatureExpBatchCache {
    /// `entries[feature][batch_idx]` — `None` when the feature has no active
    /// edges within that batch (sampling skips this pair).
    pub entries: Vec<Vec<Option<FeatureExpBatchEntry>>>,
}

/// Build the per-(feature, batch) cache by intersecting each feature's
/// active-edge list with each batch's retained-edge list (both sorted
/// ascending → linear merge). Rayon over features; per-feature cost is
/// O(n_batches · (|feature_active_edges| + |batch_edges|)).
pub fn build_feature_exp_batch_cache(
    activities: &FeatureActiveEdges,
    samplers_per_exp_batch: &[Option<PerBatchUnitSampler>],
) -> FeatureExpBatchCache {
    let n_features = activities.feature_active_edges.len();
    let n_batches = samplers_per_exp_batch.len();
    let entries: Vec<Vec<Option<FeatureExpBatchEntry>>> = (0..n_features)
        .into_par_iter()
        .map(|g| {
            let feature_edges = &activities.feature_active_edges[g];
            let feature_weights = &activities.feature_active_edge_weights[g];
            (0..n_batches)
                .map(|b| {
                    build_entry(
                        feature_edges,
                        feature_weights,
                        samplers_per_exp_batch[b].as_ref(),
                    )
                })
                .collect()
        })
        .collect();
    FeatureExpBatchCache { entries }
}

fn build_entry(
    feature_edges: &[u32],
    feature_weights: &[f32],
    batch_sampler: Option<&PerBatchUnitSampler>,
) -> Option<FeatureExpBatchEntry> {
    let bs = batch_sampler?;
    if feature_edges.is_empty() {
        return None;
    }
    let batch_edges = &bs.edge_indices;
    // Linear-time intersection of two sorted u32 lists.
    let (mut i, mut j) = (0usize, 0usize);
    let mut local_edges = Vec::<u32>::new();
    let mut local_weights = Vec::<f32>::new();
    while i < feature_edges.len() && j < batch_edges.len() {
        let a = feature_edges[i];
        let b = batch_edges[j];
        match a.cmp(&b) {
            std::cmp::Ordering::Equal => {
                local_edges.push(a);
                // drawn in proportion to the edge's activity
                local_weights.push(feature_weights[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    if local_edges.is_empty() {
        return None;
    }
    // Guard against an all-zero weight column (e.g., underflow on tiny
    // products). Fall back to uniform within the active set.
    let total: f32 = local_weights.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        local_weights.fill(1.0);
    }
    let pos = WeightedIndex::new(local_weights).ok()?;
    Some(FeatureExpBatchEntry {
        pos,
        local_to_global: local_edges,
    })
}

impl FeatureExpBatchCache {
    /// Diagnostic: count of `(feature, batch)` cells with a non-empty
    /// active-edge intersection. Useful for an early sanity log.
    pub fn n_active_pairs(&self) -> usize {
        self.entries
            .iter()
            .flat_map(|row| row.iter())
            .filter(|e| e.is_some())
            .count()
    }
}

///////////////////////////
// FeatureGatedChainSampler //
///////////////////////////

pub struct FeatureGatedChainSampler<'a> {
    pub samplers_per_exp_batch: &'a [Option<PerBatchUnitSampler>],
    pub cache: &'a FeatureExpBatchCache,
    /// Positives drawn per `(feature, batch)`, the same for every feature.
    ///
    /// Deliberately ONE scalar. A per-feature budget was tried, on the theory that
    /// a feature active on 20 edges should not draw as hard as one active on 3000
    /// (`senna bge` weights its feature draws by count). It cannot work here: the
    /// loss reduces each feature to the MEAN over its positives, so a feature's
    /// expected gradient contribution does not depend on how many it drew — only
    /// its variance does. bge's weighting changes how OFTEN a feature is drawn,
    /// i.e. how many gradient steps it gets, which is a different axis entirely.
    /// If that is wanted here, repeat features in the epoch's visit order rather
    /// than varying this.
    pub positives_per_draw: usize,
    pub n_negatives: usize,
}

/// One `(feature, batch)` draw of super edges: positive pairs, and per
/// positive `n_negatives` other pairs of the batch.
#[derive(Clone, Debug)]
pub struct EdgeChainBatch {
    /// `[B]` positive pair ids (rows of `super_edges`).
    pub pos_edges: Vec<u32>,
    /// `[B * K]` row-major: the negatives of positive `i` at `[i*K..(i+1)*K]`,
    /// pair ids of the same batch, never the positive itself.
    pub neg_edges: Vec<u32>,
    pub n_negatives: usize,
}

impl FeatureGatedChainSampler<'_> {
    /// The draw: positives by the feature's activity over the batch's pairs,
    /// and each one's negatives
    /// drawn uniformly from the batch's other pairs, since a negative must
    /// be a pair with a row of its own. `None` when the feature × batch has
    /// no active pairs.
    pub fn sample_edges<R: Rng>(
        &self,
        feature: usize,
        batch_idx: usize,
        rng: &mut R,
    ) -> Option<EdgeChainBatch> {
        const MAX_TRIES: u32 = 16;
        let entry = self.cache.entries.get(feature)?.get(batch_idx)?.as_ref()?;
        let pool = &self
            .samplers_per_exp_batch
            .get(batch_idx)?
            .as_ref()?
            .edge_indices;
        let b = self.positives_per_draw;
        let k = self.n_negatives;
        let mut out = EdgeChainBatch {
            pos_edges: Vec::with_capacity(b),
            neg_edges: Vec::with_capacity(b * k),
            n_negatives: k,
        };
        for _ in 0..b {
            let e = entry.local_to_global[entry.pos.sample(rng)];
            out.pos_edges.push(e);
            for _ in 0..k {
                // a batch of one pair keeps its last draw
                let mut w = pool[rng.random_range(0..pool.len())];
                for _ in 0..MAX_TRIES {
                    if w != e {
                        break;
                    }
                    w = pool[rng.random_range(0..pool.len())];
                }
                out.neg_edges.push(w);
            }
        }
        Some(out)
    }
}
