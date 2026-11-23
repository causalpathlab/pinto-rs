//! The contrast's negatives and the coordinates its deviations live in.
//!
//! * Matched other-donor pairs: for every pair, the pairs of OTHER donors
//!   nearest to it in the stage-1 embedding. Stage 1 places pairs in the same
//!   shared state together, so these are "the same state, elsewhere": what
//!   the shared state explains cancels between a pair and its matches.
//! * State coordinates: each pair's position in the stage-1 embedding,
//!   reduced to its leading principal axes and scaled to unit variance, with
//!   an intercept first. A donor's deviation of a gene is a linear function
//!   of these, so it can differ between states.

use crate::util::common::*;
use legume_numeric::matrix::knn_match::{ColumnDict, SearchScratch};

/// Every pair's matched other-donor pairs, nearest first, and their
/// Euclidean distances in the stage-1 embedding; both flattened row-major,
/// `matched_pairs_per_pair` entries per pair.
pub struct MatchedOtherDonorPairs {
    pub matched_pair_of_pair: Vec<u32>,
    pub distance_to_matched_pair: Vec<f32>,
    pub matched_pairs_per_pair: usize,
}

impl MatchedOtherDonorPairs {
    /// `pair`'s matched other-donor pairs, nearest first.
    pub fn matched_of(&self, pair: usize) -> &[u32] {
        let k = self.matched_pairs_per_pair;
        &self.matched_pair_of_pair[pair * k..(pair + 1) * k]
    }

    /// Members of the contrast set anchored at `anchor`: the anchor, then its
    /// matched pairs (the layout both fit paths use).
    pub fn members_of_set(&self, anchor: u32) -> impl Iterator<Item = u32> + '_ {
        std::iter::once(anchor).chain(self.matched_of(anchor as usize).iter().copied())
    }

    pub fn members_per_set(&self) -> usize {
        self.matched_pairs_per_pair + 1
    }

    /// For each of the `number_of_pairs` pairs, the pairs that list it among
    /// their matches (the reverse lookup).
    pub fn pairs_matching_each_pair(&self, number_of_pairs: usize) -> Vec<Vec<u32>> {
        let mut matching = vec![Vec::new(); number_of_pairs];
        for (slot, &matched) in self.matched_pair_of_pair.iter().enumerate() {
            matching[matched as usize].push((slot / self.matched_pairs_per_pair) as u32);
        }
        matching
    }
}

/// For every pair, its `matched_pairs_per_pair` nearest pairs of other
/// donors in the stage-1 embedding (`[pairs × dims]`), nearest first.
pub fn find_matched_other_donor_pairs(
    stage1_pair_embedding: &Mat,
    donor_index_of_pair: &[usize],
    number_of_donors: usize,
    matched_pairs_per_pair: usize,
) -> anyhow::Result<MatchedOtherDonorPairs> {
    let embedding_dims = stage1_pair_embedding.ncols();
    // one search index per donor, over that donor's pairs only
    let pairs_in_donor: Vec<usize> = (0..number_of_donors)
        .map(|donor| donor_index_of_pair.iter().filter(|&&d| d == donor).count())
        .collect();
    let index_of_donor: Vec<ColumnDict<u32>> = (0..number_of_donors)
        .map(|donor| {
            let pairs_of_donor: Vec<usize> = (0..donor_index_of_pair.len())
                .filter(|&pair| donor_index_of_pair[pair] == donor)
                .collect();
            ColumnDict::<u32>::from_dmatrix(
                Mat::from_fn(embedding_dims, pairs_of_donor.len(), |dim, j| {
                    stage1_pair_embedding[(pairs_of_donor[j], dim)]
                }),
                pairs_of_donor.iter().map(|&p| p as u32).collect(),
            )
        })
        .collect();
    // per pair, (distance, matched pair), nearest first
    let matched_of_pair: Vec<Vec<(f32, u32)>> = (0..donor_index_of_pair.len())
        .into_par_iter()
        .map_init(SearchScratch::default, |scratch, pair| {
            let query: Vec<f32> = stage1_pair_embedding.row(pair).iter().copied().collect();
            // the nearest of each other donor, then the nearest of them all
            let mut candidates: Vec<(f32, u32)> = Vec::new();
            for (donor, index) in index_of_donor.iter().enumerate() {
                if donor == donor_index_of_pair[pair] || pairs_in_donor[donor] == 0 {
                    continue;
                }
                let (ids, distances) = index
                    .search_by_query_data_reuse(&query, matched_pairs_per_pair, scratch)
                    .map_err(|e| anyhow::anyhow!("matched pairs of pair {pair}: {e}"))?;
                candidates.extend(distances.into_iter().zip(ids));
            }
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
            anyhow::ensure!(
                candidates.len() >= matched_pairs_per_pair,
                "pair {pair}: only {} pairs in the other donors, fewer than \
                 --matched-pairs {matched_pairs_per_pair}",
                candidates.len()
            );
            candidates.truncate(matched_pairs_per_pair);
            Ok(candidates)
        })
        .collect::<anyhow::Result<_>>()?;
    let (distance_to_matched_pair, matched_pair_of_pair) =
        matched_of_pair.into_iter().flatten().unzip();
    Ok(MatchedOtherDonorPairs {
        matched_pair_of_pair,
        distance_to_matched_pair,
        matched_pairs_per_pair,
    })
}

/// `[pairs × (1 + state_axes)]`, row-major in a flat vector, and its width
/// (`1 + state_axes`, the axes capped at the embedding's dims): an intercept
/// column of ones, then the pair's scores on the stage-1 embedding's
/// `state_axes` leading principal axes, each scaled to unit variance.
pub fn state_coordinates_of_pairs(
    stage1_pair_embedding: &Mat,
    state_axes: usize,
) -> (Vec<f32>, usize) {
    let number_of_pairs = stage1_pair_embedding.nrows();
    let embedding_dims = stage1_pair_embedding.ncols();
    let state_axes = state_axes.min(embedding_dims);
    let width = 1 + state_axes;
    let mut centred = stage1_pair_embedding.clone();
    centred.centre_columns_inplace();
    let degrees_of_freedom = (number_of_pairs.max(2) - 1) as f32;
    let covariance = centred.tr_mul(&centred) / degrees_of_freedom;
    let eigen = covariance.symmetric_eigen();
    let mut axes_by_variance: Vec<usize> = (0..embedding_dims).collect();
    axes_by_variance.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));
    // every leading axis's scores in one product, [pairs × state_axes]
    let leading_axes = eigen
        .eigenvectors
        .select_columns(axes_by_variance.iter().take(state_axes));
    let scores = &centred * leading_axes;
    let mut coordinates = vec![0f32; number_of_pairs * width];
    for pair in 0..number_of_pairs {
        coordinates[pair * width] = 1.0;
    }
    for (axis_rank, axis_scores) in scores.column_iter().enumerate() {
        let sd = (axis_scores.norm_squared() / degrees_of_freedom)
            .sqrt()
            .max(1e-12);
        for (pair, score) in axis_scores.iter().enumerate() {
            coordinates[pair * width + 1 + axis_rank] = score / sd;
        }
    }
    (coordinates, width)
}
