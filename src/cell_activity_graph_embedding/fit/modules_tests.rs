//! The module count `auto` finds, and the partition a fixed count keeps.

use super::{auto_partition, resolve_modules, PROFILE_SVD_RANK};
use crate::cell_activity_graph_embedding::args::ModulePlan;
use crate::util::common::Mat;
use graph_embedding_util::FeatureModuleArgs;

/// Pseudobulks, 10 cells each.
const N_PB: usize = 64;
/// Features of each of the two planted groups.
const GROUP: usize = 40;
/// Features counted at one flat rate.
const NOISE: usize = 30;
const N_FEATURES: usize = 2 * GROUP + NOISE;

/// A count in `0..span`, fixed by `(g, s)`.
fn jitter(g: usize, s: usize, span: usize) -> f32 {
    ((g * 7919 + s * 104_729 + 13) % 9973 % span) as f32
}

/// `[features x pseudobulks]` counts and the pseudobulk sizes: the first
/// [`GROUP`] features (GENE0..) are counted in the first half of the
/// pseudobulks, the next [`GROUP`] in the second half, and the last
/// [`NOISE`] at one flat rate everywhere.
fn planted() -> (Mat, Vec<f32>) {
    let counts = Mat::from_fn(N_FEATURES, N_PB, |g, s| {
        let first_half = s < N_PB / 2;
        match g / GROUP {
            0 if first_half => 40.0 + jitter(g, s, 5),
            1 if !first_half => 40.0 + jitter(g, s, 5),
            0 | 1 => 1.0 + jitter(g, s, 2),
            _ => 5.0 + jitter(g, s, 2),
        }
    });
    (counts, vec![10.0; N_PB])
}

/// `auto` finds the two planted groups and the background: at least two
/// modules, the flat features in the background module (the last), no
/// planted feature in it, and no module holding features of both groups.
/// The fixture is wide and long enough that the profiles go through the
/// truncated SVD.
#[test]
fn auto_separates_the_planted_groups_and_backgrounds_the_noise() {
    let (counts, sizes) = planted();
    assert!(N_PB > PROFILE_SVD_RANK && 2 * GROUP > PROFILE_SVD_RANK);
    let part = auto_partition(&counts, &sizes, 7).unwrap();
    assert!(part.n_modules >= 2, "{} modules", part.n_modules);
    assert_eq!(part.labels.len(), N_FEATURES);
    assert!(part.labels.iter().all(|&m| (m as usize) < part.n_modules));
    let background = part.background.expect("flat features exist");
    assert_eq!(background as usize, part.n_modules - 1);
    let (groups, noise) = part.labels.split_at(2 * GROUP);
    assert!(
        noise.iter().all(|&m| m == background),
        "flat features outside the background: {noise:?}"
    );
    assert!(
        groups.iter().all(|&m| m != background),
        "planted features in the background: {groups:?}"
    );
    let (a, b) = groups.split_at(GROUP);
    assert!(
        a.iter().all(|m| !b.contains(m)),
        "a module holds both groups: {a:?} vs {b:?}"
    );
}

/// The count is the data's: the same call on the same profiles is the same
/// partition, and with no flat feature there is no background module.
#[test]
fn auto_is_reproducible_and_reserves_no_empty_background() {
    let (counts, sizes) = planted();
    let (one, two) = (
        auto_partition(&counts, &sizes, 7).unwrap(),
        auto_partition(&counts, &sizes, 7).unwrap(),
    );
    assert_eq!(one.labels, two.labels);
    let informative = counts.rows(0, 2 * GROUP).into_owned();
    let part = auto_partition(&informative, &sizes, 7).unwrap();
    assert_eq!(part.background, None);
    assert!(part.n_modules >= 2);
    assert_eq!(
        part.n_modules,
        part.labels.iter().max().map_or(0, |&m| m as usize + 1)
    );
}

/// Features with nothing to group them are one module, which is no layer:
/// `auto` then resolves to a run without modules.
#[test]
fn auto_with_fewer_than_two_modules_is_no_layer() {
    let (counts, sizes) = planted();
    let flat = counts.rows(2 * GROUP, NOISE).into_owned();
    let part = auto_partition(&flat, &sizes, 7).unwrap();
    assert_eq!((part.n_modules, part.background), (1, Some(0)));
    assert!(resolve_modules(ModulePlan::Auto, &flat, &sizes, 7)
        .unwrap()
        .is_none());
}

/// One feature with grouping evidence among flat ones has nothing to be
/// grouped with: every feature shares the one module, and `auto` resolves
/// to a run without the layer.
#[test]
fn a_lone_informative_feature_is_no_layer() {
    let (counts, sizes) = planted();
    let lone = counts.rows(GROUP - 1, 1).into_owned();
    let mut rows = vec![lone.row(0).into_owned()];
    rows.extend((2 * GROUP..N_FEATURES).map(|g| counts.row(g).into_owned()));
    let counts = Mat::from_rows(&rows);
    let informative = data_beans::alg::feature_coarsening::informative_features(&counts, &sizes);
    assert_eq!(informative.iter().filter(|&&i| i).count(), 1);
    assert!(informative[0]);
    let part = auto_partition(&counts, &sizes, 7).unwrap();
    assert_eq!((part.n_modules, part.background), (1, Some(0)));
    assert!(part.labels.iter().all(|&m| m == 0));
    assert!(resolve_modules(ModulePlan::Auto, &counts, &sizes, 7)
        .unwrap()
        .is_none());
}

/// A fixed count is the k-means partition at that count and seed, label for
/// label, with the upstream settings of that count; `auto` carries the
/// count it found into the same settings; off is no layer.
#[test]
fn a_fixed_count_keeps_the_kmeans_partition_and_auto_its_own_count() {
    let (counts, sizes) = planted();
    for (m, seed) in [(4, 7), (9, 42)] {
        let start = resolve_modules(ModulePlan::Fixed(m), &counts, &sizes, seed)
            .unwrap()
            .expect("a layer");
        let kmeans = graph_embedding_util::partition_modules(&counts, &sizes, m, seed).unwrap();
        assert_eq!(start.labels, kmeans, "M = {m}, seed {seed}");
        let upstream = FeatureModuleArgs {
            feature_modules: Some(m),
        }
        .resolve(None)
        .unwrap()
        .unwrap();
        assert_eq!(start.cfg.n_modules, m);
        assert_eq!(format!("{:?}", start.cfg), format!("{upstream:?}"));
    }
    let found = auto_partition(&counts, &sizes, 7).unwrap();
    let start = resolve_modules(ModulePlan::Auto, &counts, &sizes, 7)
        .unwrap()
        .expect("a layer");
    assert_eq!(start.cfg.n_modules, found.n_modules);
    assert_eq!(start.labels, found.labels);
    assert!(resolve_modules(ModulePlan::Off, &counts, &sizes, 7)
        .unwrap()
        .is_none());
}
