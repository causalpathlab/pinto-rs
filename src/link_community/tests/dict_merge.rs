use crate::link_community::dict_merge::*;
use crate::util::common::*;

/// Build a dictionary with a clean block structure plus a pile of loud,
/// uninformative rows.
///
/// Signal rows: two genuinely different community groups (columns 0,1 vs 2,3),
/// contrast +-1, so the within-group cosine is +1 and the between-group cosine
/// is -1 once each row is centred.
///
/// Noise rows: no group structure at all, but a LARGE per-column swing shared by
/// every noise row. This is what an undetected feature looks like after the
/// Poisson-Gamma fit -- its log-rate is set by each community's exposure rather
/// than by data, so it is both uninformative AND high-variance. Because cosine is
/// dominated by the largest-magnitude rows, these rows decide the merge unless
/// they are filtered out.
fn dictionary_with_loud_noise(
    n_signal: usize,
    n_noise: usize,
    noise_scale: f32,
) -> (Mat, Vec<bool>) {
    let n = n_signal + n_noise;
    let mut m = Mat::zeros(n, 4);
    for g in 0..n_signal {
        // group A high, group B low (or the reverse) -- real contrast, unit scale
        let flip = if g % 2 == 0 { 1.0 } else { -1.0 };
        m[(g, 0)] = flip;
        m[(g, 1)] = flip;
        m[(g, 2)] = -flip;
        m[(g, 3)] = -flip;
    }
    for g in n_signal..n {
        // identical across noise rows, so it is a single shared direction, and
        // it cuts ACROSS the true grouping (cols 0,2 vs 1,3)
        m[(g, 0)] = noise_scale;
        m[(g, 1)] = -noise_scale;
        m[(g, 2)] = noise_scale;
        m[(g, 3)] = -noise_scale;
    }
    let keep: Vec<bool> = (0..n).map(|g| g < n_signal).collect();
    (m, keep)
}

#[test]
fn loud_undetected_features_hijack_the_merge_when_not_filtered() {
    // 20 real features against 200 loud noise features at 5x amplitude.
    let (m, keep) = dictionary_with_loud_noise(20, 200, 5.0);

    // Unfiltered: the noise direction dominates, so the true pair (0,1) is NOT
    // the first thing to merge -- 0 pairs with 2 instead, which is the noise
    // grouping, not the signal grouping.
    let unfiltered = cosine_merge(&m, None, None);
    assert_ne!(
        (unfiltered[0].left, unfiltered[0].right),
        (0, 1),
        "unfiltered merge should be hijacked by the loud noise rows"
    );

    // Filtered to the detected features: the true pair merges first, at cosine 1.
    let filtered = cosine_merge(&m, Some(&keep), None);
    assert_eq!(
        (filtered[0].left, filtered[0].right),
        (0, 1),
        "filtered merge should recover the true grouping first"
    );
    assert!(
        filtered[0].log_bf > 0.99,
        "same-group columns should merge at cosine ~1 once noise is filtered"
    );
}

#[test]
fn filtering_puts_the_right_communities_together_at_a_default_cut() {
    let (m, keep) = dictionary_with_loud_noise(20, 200, 5.0);
    // True grouping is {0,1} and {2,3}. Both arms may well produce TWO groups at
    // a 0.9 cut, so counting them proves nothing -- what differs is WHICH columns
    // land together.
    let grouped_together =
        |lab: &[i32], a: usize, b: usize| -> bool { lab[a] >= 0 && lab[a] == lab[b] };

    let filtered = cosine_cut(&cosine_merge(&m, Some(&keep), None), 4, 0.9);
    assert!(
        grouped_together(&filtered, 0, 1) && grouped_together(&filtered, 2, 3),
        "filtered cut should recover the true grouping {{0,1}},{{2,3}}"
    );
    assert!(
        !grouped_together(&filtered, 0, 2),
        "filtered cut must not merge across the true groups"
    );

    let unfiltered = cosine_cut(&cosine_merge(&m, None, None), 4, 0.9);
    assert!(
        grouped_together(&unfiltered, 0, 2),
        "unfiltered cut should follow the loud noise rows and merge 0 with 2"
    );
}

#[test]
fn an_all_false_mask_yields_no_merges() {
    let (m, _) = dictionary_with_loud_noise(4, 0, 1.0);
    let none_kept = vec![false; m.nrows()];
    assert!(cosine_merge(&m, Some(&none_kept), None).is_empty());
}

#[test]
fn a_full_mask_matches_passing_none() {
    let (m, _) = dictionary_with_loud_noise(8, 8, 2.0);
    let all_kept = vec![true; m.nrows()];
    let a = cosine_merge(&m, None, None);
    let b = cosine_merge(&m, Some(&all_kept), None);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!((x.left, x.right), (y.left, y.right));
        assert!((x.log_bf - y.log_bf).abs() < 1e-6);
    }
}

////////////////////////////////////////////////
// tree shape and cut invariants, mask-free //
////////////////////////////////////////////////

/// Build an `(n_features × k)` matrix from K column vectors.
fn mat_from_columns(cols: &[Vec<f32>]) -> Mat {
    let n_features = cols[0].len();
    let k = cols.len();
    Mat::from_fn(n_features, k, |g, j| cols[j][g])
}

#[test]
fn merges_two_identical_columns_first() {
    // Three columns: 0 and 1 identical, 2 anti-correlated.
    let cols = vec![
        vec![1.0, 0.0, -1.0, 0.5],
        vec![1.0, 0.0, -1.0, 0.5],
        vec![-1.0, 0.0, 1.0, -0.5],
    ];
    let m = mat_from_columns(&cols);
    let merges = cosine_merge(&m, None, None);
    assert_eq!(merges.len(), 2);
    // First merge should be (0, 1) with cosine very close to 1.
    let first = &merges[0];
    assert!(
        (first.left == 0 && first.right == 1) || (first.left == 1 && first.right == 0),
        "first merge should join cols 0 and 1, got ({}, {})",
        first.left,
        first.right
    );
    assert!(
        (first.log_bf - 1.0).abs() < 1e-6,
        "identical-column merge should have cosine ≈ 1, got {}",
        first.log_bf
    );
    assert_eq!(first.n_samples, 2);
    assert_eq!(first.id, 3);
}

#[test]
fn tree_shape_invariants() {
    // Random-ish 6 columns over 8 features. Just check structural invariants.
    let cols: Vec<Vec<f32>> = (0..6)
        .map(|j| {
            (0..8)
                .map(|g| ((g + j) as f32 * 0.7).sin())
                .collect::<Vec<_>>()
        })
        .collect();
    let m = mat_from_columns(&cols);
    let merges = cosine_merge(&m, None, None);
    let k = 6usize;
    assert_eq!(merges.len(), k - 1);
    // Ids are exactly k, k+1, ..., 2k-2 in order.
    for (step, m) in merges.iter().enumerate() {
        assert_eq!(m.id as usize, k + step);
        assert!(m.left < m.id);
        assert!(m.right < m.id);
        assert!(m.left != m.right);
    }
    // Root node n_samples == K.
    assert_eq!(merges.last().unwrap().n_samples as usize, k);
    // Cosine similarity is monotonically non-increasing as we go up
    // an average-linkage UPGMA tree (the so-called Lance-Williams
    // monotonicity for UPGMA on a similarity).
    for w in merges.windows(2) {
        assert!(
            w[0].log_bf + 1e-9 >= w[1].log_bf,
            "UPGMA similarity should be non-increasing: {} -> {}",
            w[0].log_bf,
            w[1].log_bf
        );
    }
}

#[test]
fn cut_at_threshold_recovers_groups() {
    // Two groups of similar columns: {0, 1, 2} and {3, 4}; group means
    // anti-correlated. Each within-group column has small jitter so
    // cosine within a group is close to 1; across groups close to -1.
    let g0a = vec![1.0, 0.0, -1.0, 0.5, 0.2];
    let g0b = vec![1.01, 0.02, -0.99, 0.51, 0.18];
    let g0c = vec![0.99, -0.01, -1.02, 0.49, 0.22];
    let g1a = vec![-1.0, 0.0, 1.0, -0.5, -0.2];
    let g1b = vec![-1.02, 0.01, 0.98, -0.49, -0.22];
    let m = mat_from_columns(&[g0a, g0b, g0c, g1a, g1b]);
    let merges = cosine_merge(&m, None, None);
    let k = 5usize;
    // Cut at cosine ≥ 0.5: only within-group merges happen.
    let labels = cosine_cut(&merges, k, 0.5);
    assert_eq!(labels.len(), k);
    let n_super: usize = labels.iter().copied().filter(|&v| v >= 0).count();
    let max_label = labels.iter().copied().max().unwrap();
    assert_eq!(n_super, k);
    assert_eq!(max_label, 1, "expected exactly 2 super-clusters");
    // Cols 0, 1, 2 share a label; cols 3, 4 share a label; the two are different.
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn empty_for_k_lt_two() {
    let m = Mat::from_element(4, 1, 0.5);
    assert!(cosine_merge(&m, None, None).is_empty());
}

//////////////////////////////////////////////////
// section means removed before the merge       //
//////////////////////////////////////////////////

/// Four communities over six features: types P and Q, each seen in sections
/// 1 and 2 (columns P1, Q1, P2, Q2). A section adds the same large shift to
/// every community it holds, on its own features, and that shift is larger
/// than the difference between the types.
fn two_types_in_two_sections() -> (Mat, Mat) {
    let p = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0];
    let q = [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let s1 = [0.0, 0.0, 4.0, -4.0, 0.0, 0.0];
    let s2 = [0.0, 0.0, 0.0, 0.0, 4.0, -4.0];
    let col =
        |t: &[f32; 6], s: &[f32; 6]| t.iter().zip(s).map(|(a, b)| a + b).collect::<Vec<f32>>();
    let m = mat_from_columns(&[col(&p, &s1), col(&q, &s1), col(&p, &s2), col(&q, &s2)]);
    // Share of each community (row) in each section (column).
    let share = Mat::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
    (m, share)
}

fn first_pair(merges: &[BhcMerge]) -> (i32, i32) {
    let m = &merges[0];
    (m.left.min(m.right), m.left.max(m.right))
}

#[test]
fn as_they_stand_the_communities_of_one_section_merge_first() {
    let (m, _) = two_types_in_two_sections();
    let first = first_pair(&cosine_merge(&m, None, None));
    assert!(first == (0, 1) || first == (2, 3), "got {first:?}");
}

#[test]
fn with_section_means_removed_one_type_merges_across_sections() {
    let (m, share) = two_types_in_two_sections();
    let merges = cosine_merge(&remove_group_means(&m, &share), None, None);
    let first = first_pair(&merges);
    assert!(first == (0, 2) || first == (1, 3), "got {first:?}");
}

/// With every community in one section there is one mean to remove, and the
/// merge centres each feature anyway: the tree is the same.
#[test]
fn one_section_changes_nothing() {
    let (m, _) = two_types_in_two_sections();
    let one = Mat::from_element(4, 1, 1.0);
    let a = cosine_merge(&m, None, None);
    let b = cosine_merge(&remove_group_means(&m, &one), None, None);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!((x.left, x.right), (y.left, y.right));
        assert!((x.log_bf - y.log_bf).abs() < 1e-5);
    }
}

/// A community split between two sections is compared against the mix of
/// their means in its own proportions.
#[test]
fn a_community_in_two_sections_loses_the_mix_of_their_means() {
    // One feature. Communities 0, 1 wholly in section 0 and 2, 3 wholly in
    // section 1; community 4 half in each.
    let m = Mat::from_row_slice(1, 5, &[2.0, 4.0, 6.0, 8.0, 5.0]);
    let share = Mat::from_row_slice(5, 2, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5]);
    // Section means, each community counted by its share:
    // (2 + 4 + 2.5) / 2.5 = 3.4 and (6 + 8 + 2.5) / 2.5 = 6.6.
    let out = remove_group_means(&m, &share);
    let (mu0, mu1) = (3.4f32, 6.6);
    for (got, want) in [
        (out[(0, 0)], 2.0 - mu0),
        (out[(0, 2)], 6.0 - mu1),
        (out[(0, 4)], 5.0 - 0.5 * (mu0 + mu1)),
    ] {
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }
}

#[test]
fn shares_are_each_communitys_edge_ends_by_section() {
    // Cells 0..=2 in section 0, cells 3..=5 in section 1. Community 0: two
    // edges in section 0, one from section 0 to 1. Community 1: two edges in
    // section 1. Community 2: none.
    let group_of = [0u32, 0, 0, 1, 1, 1];
    let membership = [0usize, 0, 0, 1, 1];
    let edges = [(0usize, 1usize), (1, 2), (2, 3), (3, 4), (4, 5)];
    let share = community_group_shares(&membership, 3, &edges, &group_of, 2);
    assert_eq!((share.nrows(), share.ncols()), (3, 2));
    // an edge across sections counts half for each end's section
    assert_eq!([share[(0, 0)], share[(0, 1)]], [2.5 / 3.0, 0.5 / 3.0]);
    assert_eq!([share[(1, 0)], share[(1, 1)]], [0.0, 1.0]);
    assert_eq!([share[(2, 0)], share[(2, 1)]], [0.0, 0.0]);
}

/// A community alone in its section is not centred to nothing: that section
/// has no mean of its own, so its community is compared against the mean
/// over all communities. A stray edge end of a second community does not
/// change that: by effective number the section's communities are still one.
#[test]
fn a_community_alone_in_its_section_keeps_its_profile() {
    let (m, _) = two_types_in_two_sections();
    let want = m.column(0) - m.column_mean();
    for (stray, tol) in [(0.0f32, 1e-5f32), (0.001, 1e-2)] {
        // community 0 in section 0, with a stray share of community 1
        let share = Mat::from_row_slice(4, 2, &[1.0, 0.0, stray, 1.0 - stray, 0.0, 1.0, 0.0, 1.0]);
        let out = remove_group_means(&m, &share);
        assert!((out.column(0) - &want).norm() < tol, "stray {stray}");
        // only section 1 is centred on a mean of its own
        assert_eq!(groups_with_own_mean(&share), 1);
    }
}

/// A section held by two communities has a mean of its own, however little
/// of each it holds: only a community alone in its section falls back.
#[test]
fn a_section_of_two_partial_communities_is_centred_by_its_own_mean() {
    let (m, _) = two_types_in_two_sections();
    // section 0 holds all of community 0 and half of community 1
    let share = Mat::from_row_slice(4, 2, &[1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0]);
    let out = remove_group_means(&m, &share);
    let mean0 = (m.column(0) + m.column(1) * 0.5) / 1.5;
    let want = m.column(0) - &mean0;
    assert!((out.column(0) - want).norm() < 1e-5);
}

/// Two types in two sections, plus a community with no edge whose log-rates
/// are the prior's, far from the others. It is left out of the tree: no
/// merge names it, so a cut gives it no label, and it does not shift the
/// merge's per-feature centring of the rest. A cut of 0.4 still keeps the
/// two types and joins each across sections.
#[test]
fn a_community_with_no_edge_is_left_out_of_the_tree() {
    let (m4, share4) = two_types_in_two_sections();
    let mut m = Mat::from_element(m4.nrows(), 5, -10.0);
    m.columns_mut(0, 4).copy_from(&m4);
    let mut share = Mat::zeros(5, 2);
    share.rows_mut(0, 4).copy_from(&share4);
    let occupied = [true, true, true, true, false];
    let merges = cosine_merge(&remove_group_means(&m, &share), None, Some(&occupied));
    assert_eq!(merges.len(), 3);
    assert!(merges.iter().all(|x| x.left != 4 && x.right != 4));
    // internal ids follow the full set of leaves, as `cosine_cut` expects
    assert_eq!(
        merges.iter().map(|x| x.id).collect::<Vec<_>>(),
        vec![5, 6, 7]
    );
    let labels = cosine_cut(&merges, 5, 0.4);
    assert_eq!(labels[4], -1);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[1], labels[3]);
    assert_ne!(labels[0], labels[1]);
}
