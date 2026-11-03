//! The shared edge-cut flags resolve to the algorithm-side enum with the
//! run's seed, so a k-means cut is reproducible like a Leiden one.

use crate::link_community::profiles::EdgeClustering;
use crate::util::edge_clustering::EdgeClusterArgs;
use clap::Parser;

#[derive(Parser)]
struct Wrap {
    #[command(flatten)]
    edge: EdgeClusterArgs,
}

fn parse(extra: &[&str]) -> EdgeClusterArgs {
    let mut argv = vec!["pinto"];
    argv.extend_from_slice(extra);
    Wrap::try_parse_from(argv).expect("args should parse").edge
}

#[test]
fn leiden_is_the_default_and_carries_the_seed() {
    match parse(&[]).resolve(42) {
        EdgeClustering::Leiden {
            knn,
            resolution,
            target,
            runs,
            seed,
        } => {
            assert_eq!(runs, 10);
            assert_eq!(knn, 30);
            assert_eq!(resolution, 1.0);
            assert_eq!(target, None);
            assert_eq!(seed, 42);
        }
        other => panic!("expected leiden, got {other:?}"),
    }
}

#[test]
fn kmeans_resolves_with_the_count_iterations_and_seed() {
    let resolved = parse(&[
        "--edge-cluster-method",
        "kmeans",
        "--n-edge-clusters",
        "25",
        "--kmeans-max-iter",
        "7",
    ])
    .resolve(9);
    match resolved {
        EdgeClustering::Kmeans {
            n_clusters,
            max_iter,
            seed,
        } => {
            assert_eq!(n_clusters, Some(25));
            assert_eq!(max_iter, 7);
            assert_eq!(seed, 9);
        }
        other => panic!("expected kmeans, got {other:?}"),
    }
}

#[test]
fn runs_must_be_at_least_one() {
    assert!(Wrap::try_parse_from(["pinto", "--leiden-runs", "0"]).is_err());
    match parse(&["--leiden-runs", "1"]).resolve(7) {
        EdgeClustering::Leiden { runs, .. } => assert_eq!(runs, 1),
        other => panic!("expected leiden, got {other:?}"),
    }
}

/// Points on `k` well-separated directions, each with a little seeded noise.
fn planted(k: usize, per: usize, dim: usize, seed: u64) -> (crate::util::common::Mat, Vec<usize>) {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let n = k * per;
    let truth: Vec<usize> = (0..n).map(|i| i / per).collect();
    let m = crate::util::common::Mat::from_fn(n, dim, |i, j| {
        let centre = if j % k == truth[i] { 1.0 } else { 0.0 };
        centre + 0.15 * (rng.random::<f32>() - 0.5)
    });
    (m, truth)
}

fn agree(a: &[usize], b: &[usize]) -> bool {
    // Same partition up to relabelling: the label map is one-to-one both ways.
    let (mut ab, mut ba) = (
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    );
    a.iter()
        .zip(b)
        .all(|(&x, &y)| *ab.entry(x).or_insert(y) == y && *ba.entry(y).or_insert(x) == x)
}

#[test]
fn consensus_recovers_planted_communities_whatever_the_seed() {
    use crate::link_community::profiles::consensus_leiden;
    let (m, truth) = planted(4, 60, 8, 3);
    for seed in [1, 2, 3] {
        let labels = consensus_leiden(&m, 15, 1.0, None, 10, seed).unwrap();
        assert!(agree(&labels, &truth), "seed {seed}: {labels:?}");
    }
    // The same seed gives the same partition.
    assert_eq!(
        consensus_leiden(&m, 15, 1.0, None, 10, 5).unwrap(),
        consensus_leiden(&m, 15, 1.0, None, 10, 5).unwrap()
    );
}

#[test]
fn one_run_is_plain_leiden() {
    use crate::link_community::profiles::consensus_leiden;
    let (m, _) = planted(3, 40, 6, 9);
    for seed in [1, 4] {
        assert_eq!(
            consensus_leiden(&m, 10, 1.0, None, 1, seed).unwrap(),
            legume_numeric::matrix::clustering::leiden_clustering(
                &m,
                10,
                1.0,
                None,
                Some(seed),
                true
            )
            .unwrap()
        );
    }
}
