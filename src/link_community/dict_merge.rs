//! Cosine-similarity agglomerative merging over the feature × community
//! dictionary.
//!
//! In `pinto lc`, each fitted link community has a posterior feature-expression
//! profile (the "dictionary atom") produced by `compute_feature_community_stat`.
//! Many fine link communities share a near-identical profile — they encode
//! the same cell type or program sitting at different spatial locations or
//! neighbourhoods. To recover a consistent cell-type-level annotation we
//! collapse such columns by hierarchical agglomerative merging on cosine
//! similarity of per-feature-centred log-rates.
//!
//! The output `Vec<BhcMerge>` reuses the merge-tree node type from
//! `data_beans::alg::bhc` purely as a binary-merge-tree carrier. The
//! cut function is the same union-find walk BHC uses; under the
//! cosine alias, `log_bf` carries cosine similarity and `cutoff` is a
//! user-chosen similarity threshold rather than a Bayes-factor break
//! point.
//!
//! Linkage: average linkage (UPGMA). When two clusters merge, the
//! similarity between the new node and any other cluster `o` is the
//! `n_samples`-weighted mean of `(left, o)` and `(right, o)` similarities,
//! which is equivalent to the average pairwise similarity between leaves.

use crate::util::common::{DVec, Mat};
pub use data_beans::alg::bhc::{bhc_cut as cosine_cut, BhcMerge};
use legume_numeric::matrix::traits::MatOps;

/// The effective number of communities, below which a group has no mean of its
/// own in [`remove_group_means`]: closer to one community than to two.
const MIN_EFFECTIVE_COMMUNITIES: f32 = 1.5;

/// Each community's share of its edge ends in each group: `[k × n_groups]`,
/// rows summing to 1 (or to 0 for a community with no edge). `edges` is
/// parallel to `membership`, and `group_of[cell]` is a cell's group; an edge
/// between two groups counts half for each.
pub fn community_group_shares(
    membership: &[usize],
    k: usize,
    edges: &[(usize, usize)],
    group_of: &[u32],
    n_groups: usize,
) -> Mat {
    debug_assert_eq!(membership.len(), edges.len());
    let mut count = vec![0.0f64; k * n_groups];
    for (&c, &(u, v)) in membership.iter().zip(edges) {
        count[c * n_groups + group_of[u] as usize] += 0.5;
        count[c * n_groups + group_of[v] as usize] += 0.5;
    }
    let mut share = Mat::zeros(k, n_groups);
    for c in 0..k {
        let row = &count[c * n_groups..(c + 1) * n_groups];
        let total: f64 = row.iter().sum();
        if total > 0.0 {
            for (g, &x) in row.iter().enumerate() {
                share[(c, g)] = (x / total) as f32;
            }
        }
    }
    share
}

/// The dictionary with each group's mean profile taken out of the
/// communities it holds, for [`cosine_merge`] across sections.
///
/// A section shifts every community it holds the same way (depth, a batch
/// effect, its own composition), and that shared shift is often larger than
/// the difference between two cell types. Compared as they stand, two types
/// in one section then look more alike than one type in two sections, and
/// the merge collapses types within a section before it ever joins a type
/// across sections. With the section's mean removed, a community is compared
/// by how it differs from its own section.
///
/// `share[(c, g)]` is community `c`'s share in group `g`
/// ([`community_group_shares`]); a community with no share anywhere has no
/// edge. A group's mean counts each community by its share there, not by its
/// size, so one large community does not become the mean. A community alone
/// in its group would be its group's mean and come out as nothing, so a group
/// whose effective number of communities (the inverse Simpson index of its
/// shares, `(Σ s)² / Σ s²`) is below [`MIN_EFFECTIVE_COMMUNITIES`] is
/// compared against the mean over all communities with an edge instead: a
/// stray edge end of a second community does not give a group a mean of its
/// own. A community spread over several groups loses the mix of their means
/// in its own proportions.
///
/// A community with no edge comes out as zeros. Its log-rates are the
/// prior's, on another scale from the centred profiles, and left in they
/// would shift the merge's per-feature centring of every other community.
/// With one group this only centres each feature, which the merge does anyway.
pub fn remove_group_means(post_log_mean: &Mat, share: &Mat) -> Mat {
    let (n_features, k) = (post_log_mean.nrows(), post_log_mean.ncols());
    debug_assert_eq!(share.nrows(), k);
    let n_groups = share.ncols();
    let occupied: Vec<usize> = (0..k).filter(|&c| share.row(c).sum() > 0.0).collect();
    // the mean over occupied communities, each counted once
    let mut overall = DVec::zeros(n_features);
    for &c in &occupied {
        overall += post_log_mean.column(c);
    }
    overall /= occupied.len().max(1) as f32;
    // `[n_features × n_groups]`: each group's mean profile
    let mut mean = Mat::zeros(n_features, n_groups);
    for g in 0..n_groups {
        let weight: f32 = share.column(g).sum();
        let effective = weight * weight / share.column(g).norm_squared();
        let mut col = mean.column_mut(g);
        if weight == 0.0 || effective < MIN_EFFECTIVE_COMMUNITIES {
            col.copy_from(&overall);
            continue;
        }
        for c in (0..k).filter(|&c| share[(c, g)] > 0.0) {
            col.axpy(share[(c, g)] / weight, &post_log_mean.column(c), 1.0);
        }
    }
    let mut out = Mat::zeros(n_features, k);
    for &c in &occupied {
        let mut col = out.column_mut(c);
        col.copy_from(&post_log_mean.column(c));
        for g in (0..n_groups).filter(|&g| share[(c, g)] > 0.0) {
            col.axpy(-share[(c, g)], &mean.column(g), 1.0);
        }
    }
    out
}

/// Build an agglomerative average-linkage merge tree over the K columns of
/// `post_log_mean` (feature × community posterior log-mean) using cosine
/// similarity of per-feature-centred community vectors.
///
/// Returns `K - 1` merges in increasing-id order. Each merge records the
/// cosine similarity at which the two children were joined in `log_bf`.
/// The resulting tree can be cut with `data_beans::alg::bhc::bhc_cut`
/// using a cosine-similarity threshold (e.g. 0.4 = collapse columns whose
/// merge happened at cosine ≥ 0.9).
///
/// `keep_features` is a per-row mask of features to score on, indexed like
/// `post_log_mean`'s rows; `None` uses every row. Its length must equal
/// `post_log_mean.nrows()`.
/// Passing the detected-feature mask is strongly recommended — see the comment on
/// step 0 for what leaving undetected features in does to the similarity.
///
/// Returns an empty vector for `K < 2`, or when the mask keeps no features.
pub fn cosine_merge(post_log_mean: &Mat, keep_features: Option<&[bool]>) -> Vec<BhcMerge> {
    let k = post_log_mean.ncols();
    if k < 2 {
        return Vec::new();
    }

    // 0. Restrict to informative features.
    //
    // UNDETECTED FEATURES DOMINATE THIS SIMILARITY IF LEFT IN, and not because they
    // are flat — because they are the LOUDEST rows. A feature with no counts gets a
    // Poisson-Gamma posterior driven entirely by each community's exposure, and
    // `log` of a near-zero rate swings hard between communities, while a
    // well-measured feature's log-rate is stable. Measured, the
    // centred row sum-of-squares was ~13x LARGER for zero-count features than for
    // features seen in >= 20 spots (medians 235 vs 18.5), and
    // `spearman(row SS, nnz) = -0.986`. Cosine is dominated by the
    // largest-magnitude rows, so 19k noise features outvoted 17.7k real ones: 43% of
    // community pairs scored >= 0.9 and the default cut collapsed 50 communities
    // to 4, with 97% of cells in one — a segmentation at chance (1.03x a
    // neighbour-agreement null, against 11.1x before the merge).
    //
    // Filtering by DETECTION is the only criterion that works here. Two plausible
    // alternatives are both backwards, and were measured to be so:
    //   * NB-Fisher `feature_weights` (already computed upstream) give undetected
    //     features weight exactly 1.0 and well-detected ones a median of 0.878 —
    //     they are built to suppress high-dispersion housekeeping features, which is
    //     the opposite question.
    //   * Filtering on low dictionary variance keeps precisely the noise, per the
    //     inverted correlation above.
    // A mismatched mask would silently score a truncated feature set, which looks
    // like a plausible tree and is not one.
    if let Some(mask) = keep_features {
        assert_eq!(
            mask.len(),
            post_log_mean.nrows(),
            "keep_features must be indexed like post_log_mean's rows"
        );
    }
    let rows: Vec<usize> = (0..post_log_mean.nrows())
        .filter(|&g| keep_features.is_none_or(|mask| mask[g]))
        .collect();
    let n_kept = rows.len();
    if n_kept == 0 {
        return Vec::new();
    }

    // 1. Per-feature centring (subtract row mean across communities), so a feature's
    //    shared level across communities drops out and only its contrast
    //    survives. NOTE this is NOT a Pearson correlation between communities —
    //    that would need the COLUMNS centred too. Column centring was measured to
    //    add nothing once the feature filter above is applied.
    let mut z = Mat::zeros(n_kept, k);
    for (gi, &g) in rows.iter().enumerate() {
        let mu: f32 = post_log_mean.row(g).iter().sum::<f32>() / k as f32;
        for j in 0..k {
            z[(gi, j)] = post_log_mean[(g, j)] - mu;
        }
    }

    // 2. L2-normalize columns. Pairwise cosine = Z^T Z (symmetric).
    z.normalize_columns_inplace();

    // 3. Cosine similarity matrix S = Zᵀ Z (K × K, f32 via nalgebra),
    //    promoted to f64 and padded to (2K-1) × (2K-1) for in-place
    //    UPGMA updates as new internal nodes are written into the
    //    extended rows/cols.
    let total_nodes = 2 * k - 1;
    let sim_kk = z.tr_mul(&z);
    let mut sim = vec![vec![0.0f64; total_nodes]; total_nodes];
    for i in 0..k {
        for j in 0..k {
            sim[i][j] = sim_kk[(i, j)] as f64;
        }
    }

    // 4. Agglomeration state. `alive[c]` indicates whether cluster `c` is
    //    still a root (clusters are addressed by their public id, which
    //    starts at 0..k for leaves and grows to 2k-2 at the final merge).
    //    `n_leaves[c]` = number of original leaves under `c` — the
    //    UPGMA weight for similarity averaging.
    let mut alive = vec![false; total_nodes];
    let mut n_leaves = vec![0i32; total_nodes];
    for c in 0..k {
        alive[c] = true;
        n_leaves[c] = 1;
    }

    let mut merges: Vec<BhcMerge> = Vec::with_capacity(k - 1);

    for step in 0..(k - 1) {
        // Find the most-similar live pair.
        let mut best_i = usize::MAX;
        let mut best_j = usize::MAX;
        let mut best_s = f64::NEG_INFINITY;
        for i in 0..(k + step) {
            if !alive[i] {
                continue;
            }
            for j in (i + 1)..(k + step) {
                if !alive[j] {
                    continue;
                }
                if sim[i][j] > best_s {
                    best_s = sim[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        debug_assert!(best_i != usize::MAX);

        let new_id = (k + step) as i32;
        let (left, right) = if (best_i as i32) < (best_j as i32) {
            (best_i as i32, best_j as i32)
        } else {
            (best_j as i32, best_i as i32)
        };
        let n_new = n_leaves[best_i] + n_leaves[best_j];

        merges.push(BhcMerge {
            id: new_id,
            left,
            right,
            log_bf: best_s,
            n_samples: n_new,
        });

        // UPGMA update: similarity from the new cluster to every surviving
        // other cluster `o` is the leaf-count-weighted mean of `(best_i, o)`
        // and `(best_j, o)`.
        let new_idx = new_id as usize;
        let w_i = n_leaves[best_i] as f64;
        let w_j = n_leaves[best_j] as f64;
        let w_sum = w_i + w_j;
        for o in 0..(k + step) {
            if !alive[o] || o == best_i || o == best_j {
                continue;
            }
            let s = (w_i * sim[best_i][o] + w_j * sim[best_j][o]) / w_sum;
            sim[new_idx][o] = s;
            sim[o][new_idx] = s;
        }
        sim[new_idx][new_idx] = 1.0;

        alive[best_i] = false;
        alive[best_j] = false;
        alive[new_idx] = true;
        n_leaves[new_idx] = n_new;
    }

    merges
}
