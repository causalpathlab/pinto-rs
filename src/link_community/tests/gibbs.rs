use crate::link_community::gibbs::*;
use crate::link_community::model::{
    compute_log_probs_for_edge, LinkCommunityStats, LinkProfileStore,
};
use crate::test_support::make_test_graph;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Create a planted partition: edges 0..n/2 belong to community 0,
/// edges n/2..n belong to community 1, with distinct feature signatures.
fn make_planted_profiles(n_edges: usize, m: usize) -> (LinkProfileStore, Vec<usize>) {
    let mut profiles = vec![0.0f32; n_edges * m];
    let mut true_labels = vec![0usize; n_edges];

    for e in 0..n_edges {
        let c = if e < n_edges / 2 { 0 } else { 1 };
        true_labels[e] = c;
        for g in 0..m {
            // Strong signal in first half of features for c=0, second half for c=1
            let signal = if (g < m / 2) == (c == 0) { 10.0 } else { 1.0 };
            profiles[e * m + g] = signal;
        }
    }

    (LinkProfileStore::new(profiles, n_edges, m), true_labels)
}

#[test]
fn test_gibbs_convergence() {
    let (store, _true_labels) = make_planted_profiles(100, 10);
    let k = 2;

    // Start with random labels
    let random_labels: Vec<usize> = (0..100).map(|e| e % k).collect();
    let mut stats = LinkCommunityStats::from_profiles(&store, k, &random_labels);

    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

    let moves1 = sampler.run(&mut stats, &store, 1);
    let _moves_mid = sampler.run(&mut stats, &store, 20);
    let moves_late = sampler.run(&mut stats, &store, 1);

    assert!(
        moves_late <= moves1 || moves1 == 0,
        "Expected convergence: early={}, late={}",
        moves1,
        moves_late
    );
}

#[test]
fn test_greedy_recovers_planted() {
    let (store, true_labels) = make_planted_profiles(100, 10);
    let k = 2;

    // Start with random labels
    let random_labels: Vec<usize> = (0..100).map(|e| (e * 7) % k).collect();
    let mut stats = LinkCommunityStats::from_profiles(&store, k, &random_labels);

    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));
    sampler.run(&mut stats, &store, 50);
    sampler.run_greedy_plain(&mut stats, &store, 20, 0.0);

    // Check that the partition matches the planted one (up to label permutation)
    let match_direct: usize = (0..100)
        .filter(|&e| stats.membership[e] == true_labels[e])
        .count();
    let match_flipped: usize = (0..100)
        .filter(|&e| stats.membership[e] == 1 - true_labels[e])
        .count();

    let best_match = match_direct.max(match_flipped);
    assert!(
        best_match >= 90,
        "Planted partition recovery: {}/100",
        best_match
    );
}

/// Two identical edges: the share of sweeps they spend in one community, against
/// the labelings listed out.
fn share_together_in_truth(store: &LinkProfileStore) -> f64 {
    let log_joint =
        |labels: &[usize]| LinkCommunityStats::from_profiles(store, 2, labels).log_marginal(0.0);
    1.0 / (1.0 + (log_joint(&[0, 1]) - log_joint(&[0, 0])).exp())
}

const IDENTICAL_SWEEPS: usize = 20_000;

fn identical_edges() -> LinkProfileStore {
    LinkProfileStore::new(vec![50.0, 50.0, 50.0, 50.0], 2, 2)
}

/// Share of `IDENTICAL_SWEEPS` sweeps of `step` that end with the edges together.
fn share_together(
    store: &LinkProfileStore,
    mut step: impl FnMut(&mut LinkCommunityStats, &LinkProfileStore),
) -> f64 {
    let mut stats = LinkCommunityStats::from_profiles(store, 2, &[0, 1]);
    let mut together = 0usize;
    for _ in 0..IDENTICAL_SWEEPS {
        step(&mut stats, store);
        together += usize::from(stats.membership[0] == stats.membership[1]);
    }
    together as f64 / IDENTICAL_SWEEPS as f64
}

/// The sequential sweep the cascade runs reaches the posterior.
#[test]
fn the_cascade_sampler_keeps_identical_edges_at_their_posterior() {
    let store = identical_edges();
    let truth = share_together_in_truth(&store);
    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(3));
    let share = share_together(&store, |stats, store| {
        sampler.gibbs_sweep(stats, store, 0.0);
    });
    assert!(
        (share - truth).abs() < 0.03,
        "together {share} against {truth}"
    );
}

/// The check above must be able to fail: updated together against one snapshot,
/// each edge sees its own community empty, and they trade labels.
#[test]
fn updating_both_against_one_snapshot_fails_the_same_check() {
    let store = identical_edges();
    let truth = share_together_in_truth(&store);
    let mut rng = SmallRng::seed_from_u64(3);
    let share = share_together(&store, |stats, store| {
        let mut log_probs = [0.0; 2];
        let proposals: Vec<usize> = (0..2)
            .map(|e| {
                compute_log_probs_for_edge(
                    e,
                    stats,
                    store,
                    0.0,
                    None,
                    &mut log_probs,
                    &mut [0.0; 2],
                );
                sample_categorical_log(&log_probs, &mut rng)
            })
            .collect();
        for (e, &new_c) in proposals.iter().enumerate() {
            let old_c = stats.membership[e];
            if new_c != old_c {
                stats.delta_move(e, old_c, new_c, store);
            }
        }
    });
    assert!(
        (share - truth).abs() > 0.05,
        "the broken update went unnoticed: {share} against {truth}"
    );
}

#[test]
fn test_sample_categorical_log() {
    let mut rng = SmallRng::seed_from_u64(42);

    let log_probs = vec![-100.0, 0.0, -100.0];
    let mut counts = [0usize; 3];
    for _ in 0..1000 {
        let idx = sample_categorical_log(&log_probs, &mut rng);
        counts[idx] += 1;
    }
    assert!(counts[1] > 990, "Expected mostly index 1, got {:?}", counts);
}

/// Test memoized EM Gibbs on a 2-component graph with planted partition.
///
/// Component 0: nodes 0-4, edges among them → community 0 (high in features 0..m/2)
/// Component 1: nodes 5-9, edges among them → community 1 (high in features m/2..m)
#[test]
fn test_memoized_em_two_components() {
    let m = 10;
    let k = 2;

    // Two disconnected cliques: nodes 0-4 and nodes 5-9
    let edges_list = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
        (5, 6),
        (5, 7),
        (5, 8),
        (6, 7),
        (6, 8),
        (7, 8),
        (7, 9),
        (8, 9),
    ];
    let n_edges = edges_list.len();
    let graph = make_test_graph(10, edges_list.clone());

    // Build profiles: edges in component 0 → community 0 signal,
    //                  edges in component 1 → community 1 signal
    let mut profiles = vec![0.0f32; n_edges * m];
    let mut true_labels = vec![0usize; n_edges];
    for (e, &(i, _j)) in edges_list.iter().enumerate() {
        let c = if i < 5 { 0 } else { 1 };
        true_labels[e] = c;
        for g in 0..m {
            let signal = if (g < m / 2) == (c == 0) { 10.0 } else { 1.0 };
            profiles[e * m + g] = signal;
        }
    }
    let store = LinkProfileStore::new(profiles, n_edges, m);

    // Start with random labels
    let mut membership: Vec<usize> = (0..n_edges).map(|e| (e * 7) % k).collect();

    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42));

    // Run memoized EM Gibbs
    let comp_args = ComponentGibbsArgs {
        graph: &graph,
        edges: &edges_list,
        k,
        alpha: 0.0,
        incidence: None,
    };
    let moves = sampler.run_components_em(&mut membership, &store, &comp_args, 50);
    assert!(moves > 0, "Expected some moves");

    // Run memoized greedy
    let greedy_moves = sampler.run_greedy_by_components(&mut membership, &store, &comp_args, 20);
    let _ = greedy_moves;

    // Check planted recovery (up to label permutation)
    let match_direct: usize = (0..n_edges)
        .filter(|&e| membership[e] == true_labels[e])
        .count();
    let match_flipped: usize = (0..n_edges)
        .filter(|&e| membership[e] == 1 - true_labels[e])
        .count();
    let best_match = match_direct.max(match_flipped);
    assert!(
        best_match >= n_edges - 2,
        "Two-component planted recovery: {}/{}",
        best_match,
        n_edges
    );
}

/// Memoized EM on one chain component, split into four blocks of 25 edges,
/// recovers the planted halves.
#[test]
fn test_memoized_em_single_component() {
    let (store, true_labels) = make_planted_profiles(100, 10);
    let k = 2;

    // Build a connected graph for edges 0..100
    // Edges need a graph with nodes. Let's make a simple chain.
    let n_nodes = 101;
    let edges: Vec<(usize, usize)> = (0..100).map(|i| (i, i + 1)).collect();
    let graph = make_test_graph(n_nodes, edges.clone());

    let mut membership: Vec<usize> = (0..100).map(|e| e % k).collect();

    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42)).with_blocks(4, 25);

    let comp_args = ComponentGibbsArgs {
        graph: &graph,
        edges: &edges,
        k,
        alpha: 0.0,
        incidence: None,
    };
    sampler.run_components_em(&mut membership, &store, &comp_args, 10);
    assert_eq!(
        sampler.last_run.block_sizes, [25; 4],
        "one component, four even blocks"
    );
    sampler.run_greedy_by_components(&mut membership, &store, &comp_args, 10);
    let direct = (0..100)
        .filter(|&e| membership[e] == true_labels[e])
        .count();
    let best = direct.max(100 - direct);
    assert!(best >= 98, "split sweeps recover {best}/100 planted labels");
}

/// A component splits into `len / min` near-equal blocks of consecutive
/// edges, one at the least and `blocks` at the most.
#[test]
fn components_split_into_near_equal_blocks() {
    let sizes = |n: usize, blocks: usize, min: usize| {
        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, i + 1)).collect();
        let labels = vec![0; n + 1];
        let split = split_components_into_blocks(&edges, &labels, 1, blocks, min);
        let order: Vec<usize> = split.iter().flatten().copied().collect();
        assert_eq!(order, (0..n).collect::<Vec<_>>(), "edges lost or reordered");
        split.iter().map(Vec::len).collect::<Vec<_>>()
    };
    assert_eq!(sizes(15, 16, 8), [15], "below twice the minimum, one block");
    assert_eq!(sizes(20, 16, 8), [10, 10]);
    assert_eq!(
        sizes(100, 16, 1),
        [6, 6, 6, 7, 6, 6, 6, 7, 6, 6, 6, 7, 6, 6, 6, 7]
    );
}

/// Test that memoized stats delta preserves exact global stats.
#[test]
fn test_memoized_stats_consistency() {
    let m = 6;
    let k = 3;
    let n_edges = 30;

    let (store, labels) = {
        let mut profiles = vec![0.0f32; n_edges * m];
        let mut labels = vec![0usize; n_edges];
        for e in 0..n_edges {
            let c = e % k;
            labels[e] = c;
            for g in 0..m {
                let signal = if g % k == c { 5.0 } else { 1.0 };
                profiles[e * m + g] = signal;
            }
        }
        (LinkProfileStore::new(profiles, n_edges, m), labels)
    };

    let global = LinkCommunityStats::from_profiles(&store, k, &labels);

    // Partition into 2 fake components: edges 0..15 and 15..30
    let comp0: Vec<usize> = (0..15).collect();
    let comp1: Vec<usize> = (15..30).collect();

    let (gs0, ss0, ec0) = LinkCommunityStats::component_stats(&store, k, &comp0, &labels);
    let (gs1, ss1, ec1) = LinkCommunityStats::component_stats(&store, k, &comp1, &labels);

    // Verify: sum of component stats == global stats
    for i in 0..k * m {
        assert!(
            (gs0[i] + gs1[i] - global.feature_sum[i]).abs() < 1e-10,
            "feature_sum mismatch at {}",
            i
        );
    }
    for c in 0..k {
        assert!(
            (ss0[c] + ss1[c] - global.size_sum[c]).abs() < 1e-10,
            "size_sum mismatch at {}",
            c
        );
        assert_eq!(
            ec0[c] + ec1[c],
            global.edge_count[c],
            "edge_count mismatch at {}",
            c
        );
    }
}

// The score is read by `pinto run`, which the viewer feature builds.
#[cfg(feature = "view")]
#[test]
fn a_sweep_bar_scores_the_edges_it_moved() {
    use legume_tui::child::Progress;
    let said = |sweep, moves| Progress::new(0, 9, sweep_message("Gibbs sweeps", sweep, moves));
    assert_eq!(said(0, 7).score(), Some(7.));
    // The sweep's count is not the score, and tells one sweep's from the next.
    assert_eq!(said(11, 0).score(), Some(0.));
    assert_ne!(said(1, 0).what, said(2, 0).what);
}

/// For each pair of edges, the share of labelings with the two together,
/// weighted by the posterior over every labeling listed out.
fn pairs_together_in_truth(store: &LinkProfileStore, k: usize) -> Vec<f64> {
    let n = store.n_edges;
    let labelings: Vec<Vec<usize>> = (0..k.pow(n as u32))
        .map(|code| (0..n).map(|e| code / k.pow(e as u32) % k).collect())
        .collect();
    let log_joint: Vec<f64> = labelings
        .iter()
        .map(|labels| LinkCommunityStats::from_profiles(store, k, labels).log_marginal(0.0))
        .collect();
    let top = log_joint.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weight: Vec<f64> = log_joint.iter().map(|l| (l - top).exp()).collect();
    let total: f64 = weight.iter().sum();
    edge_pairs(n)
        .map(|(a, b)| {
            labelings
                .iter()
                .zip(&weight)
                .filter(|(labels, _)| labels[a] == labels[b])
                .map(|(_, w)| w / total)
                .sum()
        })
        .collect()
}

fn edge_pairs(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n).flat_map(move |a| (a + 1..n).map(move |b| (a, b)))
}

const SPLIT_SWEEPS: usize = 20_000;

/// Sweeps per call to the sampler: each call starts from stats counted afresh
/// and patches them between sweeps.
const SPLIT_SWEEPS_PER_CALL: usize = 20;

/// The global stats the component sweeps patched between sweeps are the
/// stats of the labels they ended on, counted afresh.
fn assert_patched_stats_are_counts(
    sampler: &LinkGibbsSampler,
    store: &LinkProfileStore,
    k: usize,
    membership: &[usize],
) {
    let counted = LinkCommunityStats::from_profiles(store, k, membership);
    let run = &sampler.last_run;
    assert_eq!(run.edge_count, counted.edge_count, "edge counts drift");
    let near = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
    assert!(near(&run.size_sum, &counted.size_sum), "sizes drift");
    assert!(
        near(&run.feature_sum, &counted.feature_sum),
        "feature sums drift"
    );
}

/// The largest gap, over pairs of edges along a chain with two features
/// each, between the share of `SPLIT_SWEEPS` calls to the component sweeps
/// that end with the two together and the posterior's share, listed out.
/// The sweeps split the chain into `blocks` blocks, and the stats they patch
/// must stay the stats of their labels.
fn split_gap(profiles: Vec<f32>, k: usize, blocks: usize) -> f64 {
    let n = profiles.len() / 2;
    let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, i + 1)).collect();
    let graph = make_test_graph(n + 1, edges.clone());
    let store = LinkProfileStore::new(profiles, n, 2);
    let truth = pairs_together_in_truth(&store, k);
    let args = ComponentGibbsArgs {
        graph: &graph,
        edges: &edges,
        k,
        alpha: 0.0,
        incidence: None,
    };
    let mut sampler =
        LinkGibbsSampler::new(SmallRng::seed_from_u64(5)).with_blocks(blocks, n / blocks);
    let mut membership: Vec<usize> = (0..n).map(|e| e % k).collect();
    let mut together = vec![0usize; truth.len()];
    for _ in 0..SPLIT_SWEEPS {
        sampler.run_components_em(&mut membership, &store, &args, SPLIT_SWEEPS_PER_CALL);
        assert_patched_stats_are_counts(&sampler, &store, k, &membership);
        for (count, (a, b)) in together.iter_mut().zip(edge_pairs(n)) {
            *count += usize::from(membership[a] == membership[b]);
        }
    }
    assert_eq!(sampler.last_run.block_sizes, vec![n / blocks; blocks]);
    truth
        .iter()
        .zip(&together)
        .map(|(t, &c)| (t - c as f64 / SPLIT_SWEEPS as f64).abs())
        .fold(0.0, f64::max)
}

/// Eight edges of two kinds, alternating along the chain, so each block of
/// four holds two of each kind and every pair of alike edges has members in
/// both blocks. Each community holds several edges, so one block's moves
/// barely change the snapshot the other block sweeps against, and the split
/// sweeps stay near the posterior. The posterior is fairly confident but not
/// sure of each pair: alike edges are together in about nineteen labelings
/// of twenty, unlike ones in about one. Calls end twenty sweeps apart and are
/// near independent, so a share's sampling error is a small fraction of the
/// bar, which leaves the rest for the snapshot's bias.
#[test]
fn split_blocks_stay_near_the_posterior() {
    let (x, y) = ([36.0, 24.0], [24.0, 36.0]);
    let profiles: Vec<f32> = [x, y, x, y, x, y, x, y].concat();
    let gap = split_gap(profiles, 2, 2);
    assert!(gap < 0.03, "split sweeps stray {gap} from the posterior");
}

/// The check above can fail, where the approximation does: two alike edges
/// at the ends of a chain, each in its own block, with a loud pair between
/// that never parts and a community to spare. Each alike edge, alone in its
/// community, sees the other's community in the snapshot, and they trade
/// labels rather than meet.
#[test]
fn alike_singletons_in_different_blocks_fail_the_same_check() {
    let (alike, loud) = ([50.0, 50.0], [2000.0, 50.0]);
    let profiles: Vec<f32> = [alike, loud, loud, alike].concat();
    let gap = split_gap(profiles, 3, 2);
    assert!(gap > 0.1, "the split went unnoticed: {gap}");
}

/// The blocks the component sweeps split a graph into depend on the data and
/// `--max-threads`, not on the machine: one seed and one `--max-threads` give
/// one partition whatever the pool's size, and another `--max-threads` may
/// split the component otherwise and give another.
#[test]
fn a_seed_and_max_threads_give_one_partition_whatever_the_pool() {
    use rand::RngExt;
    // One chain component, long enough for four blocks at the real layout.
    let (n_edges, m, k) = (4 * GIBBS_MIN_BLOCK_EDGES + 100, 8, 4);
    let edges: Vec<(usize, usize)> = (0..n_edges).map(|i| (i, i + 1)).collect();
    let graph = make_test_graph(n_edges + 1, edges.clone());

    let mut rng = SmallRng::seed_from_u64(7);
    // Four planted communities along the chain, each loud on two features,
    // under noise so the draws matter.
    let profiles: Vec<f32> = (0..n_edges * m)
        .map(|i| {
            let (e, g) = (i / m, i % m);
            let loud = g / 2 == e * k / n_edges;
            rng.random_range(0.0..3.0) + if loud { 2.0 } else { 0.0 }
        })
        .collect();
    let store = LinkProfileStore::new(profiles, n_edges, m);
    // A frozen incidence table, as the fit passes.
    let incidence: Vec<f32> = (0..(n_edges + 1) * k)
        .map(|_| rng.random_range(-1.0..0.0))
        .collect();
    let args = ComponentGibbsArgs {
        graph: &graph,
        edges: &edges,
        k,
        alpha: 0.5,
        incidence: Some(&incidence),
    };

    let partition = |max_threads: usize, threads: usize, n_blocks: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut membership: Vec<usize> = (0..n_edges).map(|e| e % k).collect();
            let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(42))
                .with_blocks(max_threads, GIBBS_MIN_BLOCK_EDGES);
            sampler.run_components_em(&mut membership, &store, &args, 3);
            assert_eq!(
                sampler.last_run.block_sizes.len(),
                n_blocks,
                "the chain is not split as planned"
            );
            sampler.run_greedy_by_components(&mut membership, &store, &args, 3);
            membership
        })
    };
    let max_threads = crate::util::common::DEFAULT_MAX_THREADS;
    let one = partition(max_threads, 1, 4);
    let used = (0..k).filter(|c| one.contains(c)).count();
    assert!(used > 1, "the fit collapsed into one community");
    for threads in [2, 4, 8, 32] {
        let other = partition(max_threads, threads, 4);
        let differ = (0..n_edges).filter(|&e| other[e] != one[e]).count();
        assert_eq!(differ, 0, "{threads} threads move {differ} edges");
    }
    let two = partition(2, 2, 2);
    assert_ne!(two, one, "two blocks drew as four did");
}

/// Each sweep of a call draws afresh. A lone edge's conditional is uniform
/// over the communities whatever the labels, so a call of `n` sweeps ends
/// on the label its last sweep drew. Calls of different lengths from one
/// seed then end on every label; were every sweep of a call to replay one
/// random stream, they would all end on the first sweep's label.
#[test]
fn each_sweep_of_a_call_draws_afresh() {
    let (k, edges) = (4, vec![(0, 1)]);
    let graph = make_test_graph(2, edges.clone());
    let store = LinkProfileStore::new(vec![3.0, 1.0], 1, 2);
    let args = ComponentGibbsArgs {
        graph: &graph,
        edges: &edges,
        k,
        alpha: 0.5,
        incidence: None,
    };
    let mut seen = vec![0usize; k];
    for n in 1..=64 {
        let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(9));
        let mut membership = vec![0];
        sampler.run_components_em(&mut membership, &store, &args, n);
        seen[membership[0]] += 1;
    }
    assert!(seen.iter().all(|&c| c > 0), "labels ended on: {seen:?}");
}
