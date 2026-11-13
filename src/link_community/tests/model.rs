use crate::link_community::model::*;

fn make_synthetic_profiles(n_edges: usize, m: usize, k: usize) -> (LinkProfileStore, Vec<usize>) {
    let mut profiles = vec![0.0f32; n_edges * m];
    let mut labels = vec![0usize; n_edges];

    // Create planted structure: edges in community c have higher values in feature c
    for e in 0..n_edges {
        let c = e % k;
        labels[e] = c;
        for g in 0..m {
            let base_val = 1.0;
            let signal = if g % k == c { 5.0 } else { 0.0 };
            profiles[e * m + g] = base_val + signal;
        }
    }

    (LinkProfileStore::new(profiles, n_edges, m), labels)
}

#[test]
fn test_from_profiles_basic() {
    let (store, labels) = make_synthetic_profiles(20, 6, 3);
    let stats = LinkCommunityStats::from_profiles(&store, 3, &labels);

    assert_eq!(stats.k, 3);
    assert_eq!(stats.m, 6);

    // Each community should have roughly n_edges/k edges
    assert!(stats.edge_count.iter().all(|&n| n > 0));
    let total_count: usize = stats.edge_count.iter().sum();
    assert_eq!(total_count, 20);
}

#[test]
fn test_delta_move_consistency() {
    let (store, labels) = make_synthetic_profiles(30, 4, 3);
    let mut stats = LinkCommunityStats::from_profiles(&store, 3, &labels);

    // Move edge 0 from community 0 to community 1
    let old_c = stats.membership[0];
    let new_c = (old_c + 1) % 3;
    stats.delta_move(0, old_c, new_c, &store);

    // Recompute from scratch and compare
    let stats_recomputed = LinkCommunityStats::from_profiles(&store, 3, &stats.membership);
    let score_delta = stats.total_score();
    let score_recomputed = stats_recomputed.total_score();

    assert!(
        (score_delta - score_recomputed).abs() < 1e-8,
        "Incremental vs recomputed: {} vs {}",
        score_delta,
        score_recomputed
    );
}

#[test]
fn test_size_factors() {
    let profiles = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let store = LinkProfileStore::new(profiles, 2, 3);
    assert_eq!(store.size_factors[0], 6.0);
    assert_eq!(store.size_factors[1], 15.0);
}

fn assert_feature_table_is_fresh(stats: &LinkCommunityStats, store: &LinkProfileStore) {
    let fresh = LinkCommunityStats::from_profiles(store, stats.k, &stats.membership);
    // Whole counts: nothing here can have drifted, so nothing may differ.
    let bits = |cell: [f32; 3]| cell.map(f32::to_bits);
    for i in 0..stats.k * stats.m {
        assert_eq!(
            bits(stats.feature_table.cell(i)),
            bits(fresh.feature_table.cell(i)),
            "cell {i}"
        );
    }
}

#[test]
fn scoring_reads_fresh_counts_after_a_move() {
    let (store, labels) = make_synthetic_profiles(30, 4, 3);
    let mut stats = LinkCommunityStats::from_profiles(&store, 3, &labels);
    let old_c = stats.membership[0];
    stats.delta_move(0, old_c, (old_c + 1) % 3, &store);
    assert_feature_table_is_fresh(&stats, &store);
}

#[test]
fn scoring_reads_fresh_counts_after_a_merged_delta() {
    let (store, labels) = make_synthetic_profiles(30, 4, 3);
    let mut stats = LinkCommunityStats::from_profiles(&store, 3, &labels);
    let old = LinkCommunityStats::local_stats(&store, 3, &labels);
    let moved: Vec<usize> = labels.iter().map(|&c| (c + 1) % 3).collect();
    let new = LinkCommunityStats::local_stats(&store, 3, &moved);
    stats.apply_delta((&old.0, &old.1, &old.2), (&new.0, &new.1, &new.2));
    stats.membership = moved;
    assert_feature_table_is_fresh(&stats, &store);
}
