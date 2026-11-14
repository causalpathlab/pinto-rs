use crate::test_support::make_test_graph;
use crate::util::common::*;
use crate::util::graph_coarsen::*;
use legume_numeric::matrix::knn_graph::KnnGraph;

#[test]
fn test_graph_coarsen_small() {
    // 6 nodes in a path: 0-1-2-3-4-5
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let graph = make_test_graph(6, edges);

    let mut features = Mat::zeros(6, 6);
    for i in 0..6 {
        features[(i, i)] = 1.0;
    }

    let result = graph_coarsen(&graph, &mut features, None, None);
    assert_eq!(result.n_nodes, 6);
    assert_eq!(result.merges.len(), 5);

    // Extract via multilevel with 3 cluster target
    let pairs: Vec<(usize, usize)> = (0..5).map(|i| (i, i + 1)).collect();
    let ml = graph_coarsen_multilevel(
        &make_test_graph(6, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        &mut {
            let mut f = Mat::zeros(6, 6);
            for i in 0..6 {
                f[(i, i)] = 1.0;
            }
            f
        },
        &pairs,
        CoarsenConfig {
            n_clusters: 3,
            num_levels: 1,
            seeding: None,
            modularity_veto: None,
        },
    );
    let unique: HashSet<usize> = ml.all_pair_to_sample[0].iter().cloned().collect();
    assert!(unique.len() <= ml.all_num_samples[0]);
}

#[test]
fn test_graph_coarsen_two_cliques() {
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)];
    let graph = make_test_graph(6, edges.clone());

    let dim = 4;
    let mut features = Mat::zeros(dim, 6);
    for i in 0..3 {
        features[(0, i)] = 1.0;
        features[(1, i)] = 0.1 * (i as f32);
    }
    for i in 3..6 {
        features[(3, i)] = 1.0;
        features[(2, i)] = 0.1 * ((i - 3) as f32);
    }

    let pairs: Vec<(usize, usize)> = edges.clone();

    let ml = graph_coarsen_multilevel(
        &graph,
        &mut features,
        &pairs,
        CoarsenConfig {
            n_clusters: 2,
            num_levels: 1,
            seeding: None,
            modularity_veto: None,
        },
    );
    // With 2 clusters the two cliques should separate
    let p2s = &ml.all_pair_to_sample[0];
    // Intra-clique pairs should share a sample pattern
    // edges: (0,1),(0,2),(1,2) are clique A, (3,4),(3,5),(4,5) are clique B
    // (2,3) is the bridge
    assert_eq!(ml.all_num_samples[0], 3); // (A,A), (B,B), (A,B)
    assert_eq!(p2s[0], p2s[1]); // (0,1) and (0,2) both in clique A
    assert_eq!(p2s[0], p2s[2]); // (1,2) also in clique A
    assert_eq!(p2s[4], p2s[5]); // (3,5) and (4,5) both in clique B
    assert_ne!(p2s[0], p2s[4]); // clique A ≠ clique B
}

#[test]
fn test_cell_labels_to_pair_samples() {
    let pairs = vec![(0, 1), (2, 3), (0, 3), (1, 0)];
    let labels = vec![0, 0, 1, 1];

    let (p2s, n_samples) = cell_labels_to_pair_samples(&labels, &pairs);

    assert_eq!(p2s[0], p2s[3]);
    assert_ne!(p2s[0], p2s[1]);
    assert_ne!(p2s[0], p2s[2]);
    assert_ne!(p2s[1], p2s[2]);
    assert_eq!(n_samples, 3);
}

#[test]
fn test_compute_level_n_clusters() {
    assert_eq!(compute_level_n_clusters(1024, 1), vec![1024]);
    assert_eq!(compute_level_n_clusters(1024, 2), vec![16, 1024]);

    let levels = compute_level_n_clusters(1024, 3);
    assert_eq!(levels.len(), 3);
    assert_eq!(levels[0], 16);
    assert_eq!(levels[2], 1024);
    assert!(levels[1] > 16 && levels[1] < 1024);

    assert_eq!(compute_level_n_clusters(8, 2), vec![8, 8]);
}

#[test]
fn test_graph_coarsen_multilevel() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
    let graph = make_test_graph(6, edges.clone());

    let pairs: Vec<(usize, usize)> = edges.clone();

    let mut features = Mat::zeros(4, 6);
    for i in 0..6 {
        features[(i % 4, i)] = 1.0;
    }

    // 2 levels, finest = 3 clusters
    let ml = graph_coarsen_multilevel(
        &graph,
        &mut features,
        &pairs,
        CoarsenConfig {
            n_clusters: 3,
            num_levels: 2,
            seeding: None,
            modularity_veto: None,
        },
    );

    assert_eq!(ml.all_pair_to_sample.len(), 2);
    assert_eq!(ml.all_num_samples.len(), 2);

    // Coarsest level has fewer or equal samples
    assert!(ml.all_num_samples[0] <= ml.all_num_samples[1]);

    // Each pair_to_sample has one entry per pair
    for (level, p2s) in ml.all_pair_to_sample.iter().enumerate() {
        assert_eq!(p2s.len(), pairs.len());
        for &s in p2s {
            assert!(s < ml.all_num_samples[level]);
        }
    }

    // Single level
    let ml1 = graph_coarsen_multilevel(
        &make_test_graph(6, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        &mut Mat::from_fn(4, 6, |r, c| if r == c % 4 { 1.0 } else { 0.0 }),
        &pairs,
        CoarsenConfig {
            n_clusters: 3,
            num_levels: 1,
            seeding: None,
            modularity_veto: None,
        },
    );
    assert_eq!(ml1.all_pair_to_sample.len(), 1);
    assert_eq!(ml1.all_num_samples.len(), 1);
}

#[test]
fn test_spatial_seeding() {
    // 100 nodes on a 10×10 grid, KNN edges to 4-neighbors
    let n = 100;
    let mut coords = Mat::zeros(n, 2);
    let mut edges = Vec::new();
    for row in 0..10 {
        for col in 0..10 {
            let i = row * 10 + col;
            coords[(i, 0)] = col as f32;
            coords[(i, 1)] = row as f32;
            if col + 1 < 10 {
                edges.push((i, i + 1));
            }
            if row + 1 < 10 {
                edges.push((i, i + 10));
            }
        }
    }
    let graph = make_test_graph(n, edges.clone());

    // Seed with target ~25 super-nodes (5×5 grid)
    let (labels, num_super) = spatial_seed_labels(&coords, &graph, 25, None);
    assert_eq!(labels.len(), n);
    assert!(num_super > 0 && num_super <= 30);
    // Every cell assigned
    assert!(labels.iter().all(|&l| l < num_super));

    // Build super-graph
    let dim = 4;
    let mut features = Mat::from_fn(dim, n, |r, c| if r == c % dim { 1.0 } else { 0.0 });
    let (super_graph, super_features, _) = build_super_graph(&labels, num_super, &graph, &features);
    assert_eq!(super_features.nrows(), dim);
    assert_eq!(super_features.ncols(), num_super);
    assert!(!super_graph.edges.is_empty());
    assert_eq!(super_graph.n_nodes, num_super);

    // Full multilevel with seeding
    let pairs: Vec<(usize, usize)> = edges.clone();
    let sp = SeedingParams {
        coordinates: &coords,
        groups: None,
    };
    let ml = graph_coarsen_multilevel(
        &graph,
        &mut features,
        &pairs,
        CoarsenConfig {
            n_clusters: 10,
            num_levels: 2,
            seeding: Some(sp),
            modularity_veto: None,
        },
    );
    assert_eq!(ml.all_cell_labels.len(), 2);
    for level_labels in &ml.all_cell_labels {
        assert_eq!(level_labels.len(), n);
    }
}

#[test]
fn test_spatial_seeding_capacity_limit() {
    // Dense cluster of 80 cells at (0,0) plus 20 cells spread across (1..20, 0).
    // Without capacity limits, the dense cluster would all join one super-node.
    let n = 100;
    let mut coords = Mat::zeros(n, 2);
    let mut edges = Vec::new();

    // Dense cluster: cells 0..80 all at (0, 0) with chain edges
    for i in 0..80 {
        coords[(i, 0)] = 0.0;
        coords[(i, 1)] = (i as f32) * 0.01; // tiny offsets so they're nearby
        if i + 1 < 80 {
            edges.push((i, i + 1));
        }
    }
    // Connect dense cluster to spread region
    edges.push((79, 80));

    // Spread region: cells 80..100 at increasing x
    for i in 80..100 {
        coords[(i, 0)] = (i - 79) as f32;
        coords[(i, 1)] = 0.0;
        if i + 1 < 100 {
            edges.push((i, i + 1));
        }
    }

    let graph = make_test_graph(n, edges);

    // Target 10 super-nodes → avg 10 cells each, max_cells = 20
    let (labels, num_super) = spatial_seed_labels(&coords, &graph, 10, None);
    assert_eq!(labels.len(), n);
    assert!(labels.iter().all(|&l| l < num_super));

    // Count cells per super-node
    let mut counts = vec![0usize; num_super];
    for &l in &labels {
        counts[l] += 1;
    }

    // No super-node should have more than 2× the average
    let max_cells = (n / num_super) * 2;
    let largest = *counts.iter().max().unwrap();
    assert!(
        largest <= max_cells + 1, // +1 for rounding
        "largest super-node has {} cells, limit is {} (avg={})",
        largest,
        max_cells,
        n / num_super
    );
}

#[test]
fn test_modularity_veto_rejects_bridge() {
    // Two triangles {0,1,2} and {3,4,5} joined by one bridge (2,3). Intra-clique
    // features are nearly identical (high sim) while the bridge endpoints sit
    // on orthogonal axes (zero sim). The modularity veto should reject the
    // bridge merge regardless of how greedy we get.
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)];
    let graph = make_test_graph(6, edges.clone());

    let dim = 2;
    // Clique A lives on axis 0, clique B on axis 1; both L2-normalised.
    let mut features = Mat::zeros(dim, 6);
    for i in 0..3 {
        features[(0, i)] = 1.0;
    }
    for i in 3..6 {
        features[(1, i)] = 1.0;
    }

    // Without the veto, the greedy coarsener will eventually cross the
    // bridge — confirm the baseline produces only a single cluster when
    // asked for one.
    let result_no_veto = graph_coarsen(&graph, &mut features.clone(), None, None);
    assert_eq!(
        result_no_veto.merges.len(),
        5,
        "without veto, coarsening should merge everything"
    );

    // With γ = 1.0, the bridge edge (orthogonal endpoints, sim = 0) should
    // fail the modularity-gain test and be vetoed. The two triangles merge
    // internally (5 intra-clique merges → 2 clusters) but the bridge itself
    // is never taken.
    let veto = ModularityVeto { gamma: 1.0 };
    let result_with_veto = graph_coarsen(&graph, &mut features, None, Some(&veto));
    assert_eq!(
        result_with_veto.merges.len(),
        4,
        "with veto, each triangle produces 2 merges (6 nodes → 2 clusters)"
    );
}

/// Every level is one merge sequence cut short, so each finer cluster lies in
/// exactly one coarser cluster, and a level holds the count asked for.
///
/// The fixture has to yield levels that are really different partitions: a
/// target count high enough that the three cuts differ.
#[test]
fn levels_are_nested_and_hold_the_counts_asked_for() {
    const N: usize = 240;
    // Ring plus chords: one connected piece, and clusters are not forced.
    let mut edges: Vec<(usize, usize)> = (0..N).map(|i| (i, (i + 1) % N)).collect();
    for i in (0..N).step_by(3) {
        edges.push((i, (i + 7) % N));
    }

    let mut features = Mat::zeros(4, N);
    let mut lcg: u64 = 0x2545F491_4F6CDD1D;
    for i in 0..N {
        let th = (i as f32) * std::f32::consts::TAU / (N as f32);
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let jitter = ((lcg >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
        features[(0, i)] = th.cos() + 0.6 * jitter;
        features[(1, i)] = th.sin() - 0.6 * jitter;
        features[(2, i)] = 0.8 * jitter;
        features[(3, i)] = 0.8 * (1.0 - jitter);
    }

    let ml = graph_coarsen_multilevel(
        &make_test_graph(N, edges.clone()),
        &mut features,
        &edges,
        CoarsenConfig {
            n_clusters: 64,
            num_levels: 3,
            seeding: None,
            modularity_veto: None,
        },
    );

    let counts: Vec<usize> = ml
        .all_cell_labels
        .iter()
        .map(|l| l.iter().max().copied().map_or(0, |m| m + 1))
        .collect();
    assert_eq!(counts, compute_level_n_clusters(64, 3));

    for (l, pair) in ml.all_cell_labels.windows(2).enumerate() {
        let (coarse, fine) = (&pair[0], &pair[1]);
        let mut parents: Vec<HashSet<usize>> = vec![Default::default(); counts[l + 1]];
        for (cell, &f) in fine.iter().enumerate() {
            parents[f].insert(coarse[cell]);
        }
        assert!(
            parents.iter().all(|p| p.len() == 1),
            "a cluster of level {} straddles two clusters of level {l}",
            l + 1
        );
    }
}

/// Two 10×10 sections drawn in one frame (the same positions), with no edge
/// between them: grouped, each is seeded on its own grid.
#[test]
fn spatial_seeding_seeds_each_group_in_its_own_frame() {
    let side = 10;
    let per = side * side;
    let n = 2 * per;
    let coords = Mat::from_fn(n, 2, |c, d| {
        let i = c % per;
        [i % side, i / side][d] as f32
    });
    let edges: Vec<(usize, usize)> = (0..2)
        .flat_map(|s| {
            grid_edges(side)
                .into_iter()
                .map(move |(a, b)| (a + s * per, b + s * per))
        })
        .collect();
    let graph = make_test_graph(n, edges);
    let groups: Vec<u32> = (0..n).map(|c| (c / per) as u32).collect();

    let (labels, num_super) = spatial_seed_labels(&coords, &graph, 32, Some(&groups));
    // Each section's grid gets half the seeds, and no super-node crosses.
    let in_group = |g: usize| {
        let mut l: Vec<usize> = labels[g * per..(g + 1) * per].to_vec();
        l.sort();
        l.dedup();
        l
    };
    let (a, b) = (in_group(0), in_group(1));
    assert!(a.iter().all(|l| !b.contains(l)));
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() + b.len(), num_super);
    assert!(
        num_super <= 40,
        "{num_super} super-nodes for a target of 32"
    );
}

//////////////////////////////////////////////
// balance of the clusters a cut hands back //
//////////////////////////////////////////////

/// The edges of a `side × side` four-neighbour grid, cells numbered by row.
fn grid_edges(side: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for r in 0..side {
        for c in 0..side {
            let i = r * side + c;
            if c + 1 < side {
                edges.push((i, i + 1));
            }
            if r + 1 < side {
                edges.push((i, i + side));
            }
        }
    }
    edges
}

/// Unit columns `(1, tilt)`: neighbours on a path that differ a little, the
/// second a little more like the first than like the third.
const TILT: [f32; 3] = [0.0, 0.1, 0.22];
fn tilted(tilt_of: impl Iterator<Item = f32>) -> Mat {
    let tilts: Vec<f32> = tilt_of.collect();
    let mut features = Mat::zeros(2, tilts.len());
    for (c, t) in tilts.into_iter().enumerate() {
        features[(0, c)] = 1.0;
        features[(1, c)] = t;
    }
    features.normalize_columns_inplace();
    features
}

/// A `side × side` four-neighbour grid whose left and right halves hold two
/// orthogonal profiles, each cell its half's profile plus noise of about
/// twice the signal's norm: `(graph, [dim × n] unit columns, half of cell)`.
fn noisy_two_region_grid(side: usize, dim: usize, seed: u64) -> (KnnGraph, Mat, Vec<usize>) {
    use rand::{rngs::SmallRng, SeedableRng};
    use rand_distr::{Distribution, StandardNormal};
    let n = side * side;
    let edges = grid_edges(side);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut features = Mat::zeros(dim, n);
    let mut half = vec![0usize; n];
    let noise = 2.0 / (dim as f32).sqrt();
    for i in 0..n {
        half[i] = usize::from(i % side >= side / 2);
        for d in 0..dim {
            let z: f32 = StandardNormal.sample(&mut rng);
            features[(d, i)] = f32::from(d == half[i]) + noise * z;
        }
        features.column_mut(i).normalize_mut();
    }
    (make_test_graph(n, edges), features, half)
}

/// The cluster of every cell when the grid is cut at `k` clusters.
fn cut_at(graph: &KnnGraph, features: &Mat, k: usize) -> Vec<usize> {
    let ml = graph_coarsen_multilevel(
        graph,
        &mut features.clone(),
        &graph.edges,
        CoarsenConfig {
            n_clusters: k,
            num_levels: 1,
            seeding: None,
            modularity_veto: None,
        },
    );
    ml.all_cell_labels[0].clone()
}

fn sizes(labels: &[usize]) -> Vec<usize> {
    let k = labels.iter().max().map_or(0, |m| m + 1);
    let mut s = vec![0usize; k];
    labels.iter().for_each(|&l| s[l] += 1);
    s
}

/// A noisy cell is more like an averaged cluster than like another noisy
/// cell, so a coarsening that only ever takes a cluster's most similar
/// neighbour feeds a few clusters one cell a round and leaves the rest as
/// single cells: a cut then hands back giants and singletons, never groups
/// of about `n / k`.
#[test]
fn a_cut_hands_back_clusters_of_about_equal_size() {
    let (side, k) = (30, 100);
    let (graph, features, _) = noisy_two_region_grid(side, 16, 7);
    let s = sizes(&cut_at(&graph, &features, k));
    let mean = side * side / k;
    assert_eq!(s.len(), k);
    let largest = *s.iter().max().unwrap();
    assert!(
        largest <= 4 * mean,
        "largest cluster {largest} of mean {mean}"
    );
    let single = s.iter().filter(|&&x| x == 1).count();
    assert!(
        single * 10 <= k,
        "{single} of {k} clusters are single cells"
    );
}

/// Balance must not cost the clusters their meaning: a cluster stays inside
/// one half of the grid.
#[test]
fn balanced_clusters_stay_inside_one_region() {
    let (side, k) = (30, 100);
    let (graph, features, half) = noisy_two_region_grid(side, 16, 7);
    let labels = cut_at(&graph, &features, k);
    let mut in_half = vec![[0usize; 2]; k];
    for (i, &l) in labels.iter().enumerate() {
        in_half[l][half[i]] += 1;
    }
    let majority: usize = in_half.iter().map(|c| c[0].max(c[1])).sum();
    let pure = majority as f64 / labels.len() as f64;
    assert!(
        pure > 0.95,
        "{pure:.3} of cells sit with their region's majority"
    );
}

/// Two triangles of identical cells, orthogonal to each other, joined by one
/// bridge between cells no merge inside a triangle has to wait for. A round
/// that merged whatever two clusters were still free would take the bridge;
/// a cut at two clusters must hand back the triangles.
#[test]
fn a_bridge_between_two_groups_is_the_last_edge_taken() {
    let edges = vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 5)];
    let graph = make_test_graph(6, edges);
    let mut features = Mat::zeros(2, 6);
    for i in 0..6 {
        features[(i / 3, i)] = 1.0;
    }
    let labels = cut_at(&graph, &features, 2);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[3], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

/// The size term counts CELLS. On a path of three equally similar nodes
/// where the first already stands for a hundred cells (a spatial seed), the
/// two single cells merge first.
#[test]
fn a_node_that_stands_for_many_cells_pays_for_them() {
    let graph = make_test_graph(3, vec![(0, 1), (1, 2)]);
    let mut features = tilted(TILT.into_iter());
    // Node 1 is a little more like node 0 than like node 2, so with every
    // node one cell the first merge is (0, 1).
    let by_count = graph_coarsen(&graph, &mut features.clone(), None, None);
    assert_eq!(by_count.merges[0], (0, 1));
    let by_cells = graph_coarsen(&graph, &mut features, Some(&[100.0, 1.0, 1.0]), None);
    assert_eq!(by_cells.merges[0], (1, 2));
}

/// Two neighbouring cells with no signal at all average to a zero centroid,
/// which has no direction to normalize: the merge must not leave a NaN that
/// stops every later merge with that cluster.
#[test]
fn cells_with_no_signal_do_not_stall_the_coarsening() {
    let graph = make_test_graph(4, vec![(0, 1), (1, 2), (2, 3)]);
    let mut features = Mat::zeros(2, 4);
    features[(0, 2)] = 1.0;
    features[(0, 3)] = 1.0;
    let result = graph_coarsen(&graph, &mut features, None, None);
    assert_eq!(
        result.merges.len(),
        3,
        "a connected graph merges down to one"
    );
    assert!(features.iter().all(|v| v.is_finite()));
}

/// The same through the seeded path: a seed of a hundred cells next to two
/// seeds of one cell each.
#[test]
fn a_seed_weighs_the_cells_behind_it() {
    let n = 102usize;
    let labels: Vec<usize> = (0..n).map(|c| c.saturating_sub(99)).collect();
    let edges: Vec<(usize, usize)> = (0..n - 1).map(|c| (c, c + 1)).collect();
    let graph = make_test_graph(n, edges);
    let features = tilted(labels.iter().map(|&l| TILT[l]));
    let result = graph_coarsen_seeds(&labels, 3, &graph, &features, None);
    assert_eq!(result.merges[0], (1, 2));
}

/// A cut can stop partway through a round, so within a round the cheapest
/// merge comes first: here the pair of identical cells, not the lower pair.
#[test]
fn within_a_round_the_cheapest_merge_comes_first() {
    let graph = make_test_graph(4, vec![(0, 1), (2, 3)]);
    let mut features = tilted([0.0, 0.3, 0.0, 0.0].into_iter());
    let result = graph_coarsen(&graph, &mut features, None, None);
    assert_eq!(result.merges, vec![(2, 3), (0, 1)]);
}
