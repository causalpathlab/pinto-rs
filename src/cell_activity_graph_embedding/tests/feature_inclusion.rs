//! `--feature-inclusion moran`: the jackknife deletes exactly, spatial
//! features get high inclusion and noise none, and sampling respects π.

use crate::cell_activity_graph_embedding::feature_inclusion::{
    jackknife_moran, normalised_profiles, sample_included, score_sections, section_jack, tile_of,
    tree_blocks, two_group_inclusion, FeatureInclusion,
};
use crate::test_support::rook_grid as grid;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

type Edges = Vec<(usize, usize)>;

/// One section for [`moran_brute`]: a feature's values, edges, cells kept.
type Section = (Vec<f64>, Edges, Vec<bool>);

/// [`FeatureInclusion`] of units held in memory: their profiles, section,
/// block and the edges between them.
fn score(
    n_features: usize,
    profiles: &[Vec<(u32, f32)>],
    section: &[u32],
    block: &[usize],
    edges: &[(usize, usize)],
) -> FeatureInclusion {
    score_sections(n_features, section, edges, "units", |units: &[usize]| {
        Ok((
            units
                .iter()
                .map(|&u| profiles[u].as_slice())
                .collect::<Vec<_>>(),
            units.iter().map(|&u| block[u]).collect(),
        ))
    })
    .unwrap()
}

/// Pooled Moran's I computed from scratch: each section centred on its own
/// mean, over the cells kept and the edges between kept cells.
fn moran_brute(sections: &[Section]) -> f64 {
    let (mut num, mut den, mut n, mut w) = (0.0, 0.0, 0.0, 0.0);
    for (x, edges, keep) in sections {
        let kept: Vec<usize> = (0..x.len()).filter(|&i| keep[i]).collect();
        let mu = kept.iter().map(|&i| x[i]).sum::<f64>() / kept.len() as f64;
        for &i in &kept {
            den += (x[i] - mu).powi(2);
        }
        n += kept.len() as f64;
        for &(i, j) in edges.iter().filter(|&&(i, j)| keep[i] && keep[j]) {
            num += (x[i] - mu) * (x[j] - mu);
            w += 1.0;
        }
    }
    (n / w) * num / den
}

/// The jackknife SE of [`moran_brute`] over the same `edges` in every
/// section, deleting each of `n_blocks` blocks of each section in turn.
fn brute_jackknife_se(xs: &[Vec<f64>], edges: &Edges, block: &[usize], n_blocks: usize) -> f64 {
    let all = vec![true; block.len()];
    let mut loo = Vec::new();
    for s in 0..xs.len() {
        for k in 0..n_blocks {
            let keep: Vec<bool> = block.iter().map(|&b| b != k).collect();
            let secs: Vec<Section> = (0..xs.len())
                .map(|r| {
                    let k = if r == s { keep.clone() } else { all.clone() };
                    (xs[r].clone(), edges.clone(), k)
                })
                .collect();
            loo.push(moran_brute(&secs));
        }
    }
    crate::util::common::jackknife_se(&loo)
}

#[test]
fn the_jackknife_deletes_a_tile_exactly() {
    let mut rng = SmallRng::seed_from_u64(3);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let (xy, edges) = grid(8);
    let tile = tile_of(&xy, 2);
    // two sections, one feature: a gradient plus noise
    let xs: Vec<Vec<f64>> = (0..2)
        .map(|s| {
            xy.iter()
                .map(|&(x, _)| (x as f64) * (1.0 + s as f64) + noise.sample(&mut rng))
                .collect()
        })
        .collect();
    let sums: Vec<_> = xs
        .iter()
        .map(|x| {
            let profiles: Vec<Vec<(u32, f32)>> =
                x.iter().map(|&v| vec![(0u32, v as f32)]).collect();
            section_jack(1, &profiles, &tile, &edges)
        })
        .collect();
    let (moran, se, _) = jackknife_moran(&sums);
    let all = vec![true; xy.len()];
    let full = moran_brute(&[
        (xs[0].clone(), edges.clone(), all.clone()),
        (xs[1].clone(), edges.clone(), all.clone()),
    ]);
    assert!((moran[0] - full).abs() < 1e-4, "{} vs {full}", moran[0]);
    let brute_se = brute_jackknife_se(&xs, &edges, &tile, 4);
    assert!((se[0] - brute_se).abs() < 1e-4, "{} vs {brute_se}", se[0]);
}

#[test]
fn spatial_features_are_included_and_noise_is_not() {
    let mut rng = SmallRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let (xy, edges) = grid(30);
    let tile = tile_of(&xy, 4);
    // 20 smooth features (a left/right domain), 380 pure noise
    let (n_signal, g) = (20usize, 400usize);
    let profiles: Vec<Vec<(u32, f32)>> = xy
        .iter()
        .map(|&(x, _)| {
            (0..g as u32)
                .map(|f| {
                    let signal = if (f as usize) < n_signal && x < 15.0 {
                        2.0
                    } else {
                        0.0
                    };
                    (f, (signal + noise.sample(&mut rng)) as f32)
                })
                .collect()
        })
        .collect();
    let (_, _, t) = jackknife_moran(&[section_jack(g, &profiles, &tile, &edges)]);
    let (pi, pi0) = two_group_inclusion(&t);
    let signal = pi[..n_signal].iter().sum::<f32>() / n_signal as f32;
    let null = pi[n_signal..].iter().sum::<f32>() / (g - n_signal) as f32;
    assert!(signal > 0.9, "signal features' mean π {signal}");
    assert!(null < 0.05, "noise features' mean π {null}");
    assert!(pi0 > 0.8, "null share {pi0}");
}

#[test]
fn inclusion_is_monotone_and_one_sided() {
    let t: Vec<f64> = (0..200)
        .map(|i| (i as f64 - 100.0) / 20.0)
        .chain([12.0, 15.0])
        .collect();
    let (pi, _) = two_group_inclusion(&t);
    let mut order: Vec<usize> = (0..t.len()).collect();
    order.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
    assert!(
        order.windows(2).all(|w| pi[w[0]] <= pi[w[1]]),
        "π decreases somewhere in t"
    );
    // negative autocorrelation is never signal
    assert!(t
        .iter()
        .zip(&pi)
        .filter(|(&v, _)| v < 0.0)
        .all(|(_, &p)| p == 0.0));
    assert!(pi[t.len() - 1] > 0.9);
}

#[test]
fn sampling_follows_pi() {
    let pi = [0.0f32, 1.0, 0.5];
    let mut visits = [0usize; 3];
    for epoch in 0..2000 {
        for g in sample_included(&[0, 1, 2], &pi, 1, epoch) {
            visits[g] += 1;
        }
    }
    assert_eq!(visits[0], 0, "a feature with π = 0 trained");
    assert_eq!(visits[1], 2000, "a feature with π = 1 skipped");
    assert!(
        (900..1100).contains(&visits[2]),
        "π = 0.5 visited {} of 2000",
        visits[2]
    );
}

/// Each feature draws from its own stream: listing a feature with π = 0, or
/// leaving it off, keeps every other feature's draws, and their order.
#[test]
fn a_zero_pi_feature_leaves_the_others_draws_alone() {
    let pi: Vec<f32> = (0..40)
        .map(|g| if g % 3 == 0 { 0.0 } else { 0.5 })
        .collect();
    let all: Vec<usize> = (0..pi.len()).collect();
    let nonzero: Vec<usize> = all.iter().copied().filter(|&g| pi[g] > 0.0).collect();
    for epoch in 0..20 {
        assert_eq!(
            sample_included(&all, &pi, 9, epoch),
            sample_included(&nonzero, &pi, 9, epoch),
            "epoch {epoch}"
        );
    }
}

/// The kept features come out shuffled, a fresh order each epoch, not in
/// the listed order nor by feature index.
#[test]
fn kept_features_come_out_shuffled() {
    let pi = [1.0f32; 30];
    let listed: Vec<usize> = (0..pi.len()).collect();
    let mut orders = Vec::new();
    for epoch in 0..5 {
        let got = sample_included(&listed, &pi, 3, epoch);
        let mut sorted = got.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, listed, "every feature kept once");
        assert_ne!(got, listed, "epoch {epoch} kept the listed order");
        orders.push(got);
    }
    orders.dedup();
    assert_eq!(orders.len(), 5, "an order repeated across epochs");
}

/// When nothing is kept, the one feature drawn instead comes in proportion
/// to π.
#[test]
fn the_fallback_follows_pi() {
    let pi = [1e-7f32, 2e-7, 4e-7];
    let epochs = 7000;
    let mut picks = [0usize; 3];
    for epoch in 0..epochs {
        let f = sample_included(&[0, 1, 2], &pi, 5, epoch);
        assert_eq!(f.len(), 1);
        picks[f[0]] += 1;
    }
    for (g, &n) in picks.iter().enumerate() {
        let share = n as f64 / epochs as f64;
        let expected = f64::from(pi[g]) / 7e-7;
        assert!(
            (share - expected).abs() < 0.03,
            "feature {g}: share {share}, expected {expected}"
        );
    }
}

// ---- units, PB-tree blocks and epochs ------------------------------------

/// Any blocks, not only tiles: a checkerboard over two sections, deleted
/// one at a time, matches Moran's I recomputed from scratch.
#[test]
fn the_jackknife_deletes_any_block_exactly() {
    let mut rng = SmallRng::seed_from_u64(5);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let (xy, edges) = grid(6);
    let n = xy.len();
    let x: Vec<f64> = xy
        .iter()
        .map(|&(a, _)| a as f64 + noise.sample(&mut rng))
        .collect();
    let checker: Vec<usize> = xy
        .iter()
        .map(|&(a, b)| ((a as usize / 2) + (b as usize / 2)) % 3)
        .collect();
    // two sections: the same grid twice, units n..2n the second
    let profiles: Vec<Vec<(u32, f32)>> = (0..2 * n)
        .map(|u| vec![(0u32, x[u % n] as f32 * (1 + u / n) as f32)])
        .collect();
    let section: Vec<u32> = (0..2 * n).map(|u| (u / n) as u32).collect();
    let block: Vec<usize> = (0..2 * n).map(|u| checker[u % n] + 3 * (u / n)).collect();
    let both: Edges = edges
        .iter()
        .flat_map(|&(i, j)| [(i, j), (i + n, j + n)])
        .collect();
    let inc = score(1, &profiles, &section, &block, &both);
    let xs: Vec<Vec<f64>> = (0..2)
        .map(|s| x.iter().map(|v| v * (1 + s) as f64).collect())
        .collect();
    let brute_se = brute_jackknife_se(&xs, &edges, &checker, 3);
    assert!(
        (inc.se[0] - brute_se).abs() < 1e-4,
        "{} vs {brute_se}",
        inc.se[0]
    );
}

/// A unit's profile is its pooled counts, log-normalised to 1e4.
#[test]
fn units_are_log_normalised_pooled_counts() {
    // features × units: unit 0 pools (2, 2), unit 1 pools (0, 4)
    let pooled = crate::util::common::Mat::from_row_slice(2, 2, &[2.0, 0.0, 2.0, 4.0]);
    let p = normalised_profiles(&pooled);
    let half = (1.0f32 + 5e3).ln();
    assert_eq!(p[0], vec![(0, half), (1, half)]);
    assert_eq!(p[1], vec![(1, (1.0f32 + 1e4).ln())]);
}

/// Blocks are coarse super-cells of the PB tree, never spanning two
/// sections, from the coarsest level that gives each section enough.
#[test]
fn blocks_come_from_the_pb_tree() {
    // 8 PBs, sections [0,0,0,0,1,1,1,1]; level 0 has 2 groups that each
    // span both sections, level 1 has 4 groups within sections
    let levels = vec![
        vec![0, 0, 1, 1, 0, 0, 1, 1],
        vec![0, 0, 1, 1, 2, 2, 3, 3],
        vec![0, 1, 2, 3, 4, 5, 6, 7],
    ];
    let section = vec![0u32, 0, 0, 0, 1, 1, 1, 1];
    // two blocks per section: level 0 already gives two once split by section
    let b = tree_blocks(&levels, &section, 2);
    assert_eq!(b, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    // four per section needs the finest level
    let b = tree_blocks(&levels, &section, 4);
    assert_eq!(b, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// An epoch always trains something, however small every π is.
#[test]
fn an_epoch_is_never_empty() {
    let pi = [1e-9f32; 5];
    for epoch in 0..100 {
        assert_eq!(sample_included(&[0, 1, 2, 3, 4], &pi, 2, epoch).len(), 1);
    }
}

/// Inclusion and modules are two ways to handle uninformative features; on
/// the simulations together they did worse than either, so cage takes one.
#[test]
fn inclusion_and_modules_are_exclusive() {
    use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
    use crate::cell_activity_graph_embedding::fit::fit_cell_activity_graph_embedding;
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        cage: CellActivityGraphEmbeddingArgs,
    }
    let a = Cli::try_parse_from([
        "x",
        "-o",
        "out",
        "d.zarr",
        "--feature-inclusion",
        "moran",
        "--feature-modules",
        "20",
    ])
    .unwrap()
    .cage;
    let e = fit_cell_activity_graph_embedding(&a)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("--feature-inclusion") && e.contains("--feature-modules"),
        "{e}"
    );
}

/// An epoch visits every feature, or Σπ of them under inclusion (at least
/// one).
#[test]
fn expected_visits_are_the_sum_of_pi() {
    use crate::cell_activity_graph_embedding::feature_inclusion::expected_visits;
    let pi = [0.0f32, 1.0, 0.6, 0.9, 0.2];
    assert_eq!(expected_visits(&[0, 1, 2, 3], None), 4);
    assert_eq!(expected_visits(&[1, 2, 3, 4], Some(&pi)), 3);
    assert_eq!(expected_visits(&[0, 4], Some(&pi)), 1);
}

/// Inclusion scores spatial signal, so a run without coordinates is
/// refused before any data is read (here there is none to read).
#[test]
fn inclusion_without_coordinates_is_refused_up_front() {
    use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
    use crate::cell_activity_graph_embedding::fit::fit_cell_activity_graph_embedding;
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        cage: CellActivityGraphEmbeddingArgs,
    }
    let a = Cli::try_parse_from(["x", "-o", "out", "d.zarr", "--feature-inclusion", "moran"])
        .unwrap()
        .cage;
    let e = fit_cell_activity_graph_embedding(&a)
        .unwrap_err()
        .to_string();
    assert!(e.contains("needs coordinates"), "{e}");
}

/// A tree whose levels jump past the asked-for count (few coarse groups,
/// then many fine ones) is merged to about that many blocks per section,
/// siblings kept together, rather than taking the whole finer level.
#[test]
fn blocks_are_merged_to_about_the_count_asked_for() {
    // two sections of 12 PBs; the coarse level has 2 groups per section,
    // the next one 12
    let section: Vec<u32> = (0..24).map(|u| (u / 12) as u32).collect();
    let levels = vec![
        (0..24).map(|u| u / 6).collect::<Vec<usize>>(),
        (0..24).collect(),
    ];
    let b = tree_blocks(&levels, &section, 3);
    for s in 0..2 {
        let mut ids: Vec<usize> = (0..24).filter(|&u| u / 12 == s).map(|u| b[u]).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 3, "section {s}: {b:?}");
    }
    // about equal blocks of 4, never across sections
    assert_eq!(
        b,
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    );
    // with no level fine enough, the finest units themselves
    let b = tree_blocks(&levels[..1], &section, 6);
    assert_eq!(b.iter().max(), Some(&11));
}

/// Features that carry no spread (no variation, or no edges) are not part
/// of the two-group fit and get π = 0: were they in it, a majority of them
/// would make the MAD zero and every π one.
#[test]
fn features_without_spread_get_no_inclusion() {
    let mut rng = SmallRng::seed_from_u64(11);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let (xy, edges) = grid(20);
    let tile = tile_of(&xy, 4);
    // 10 signal, 90 noise, 300 never observed
    let (n_signal, n_noise, g) = (10usize, 90usize, 400usize);
    let profiles: Vec<Vec<(u32, f32)>> = xy
        .iter()
        .map(|&(x, _)| {
            (0..(n_signal + n_noise) as u32)
                .map(|f| {
                    let s = if (f as usize) < n_signal && x < 10.0 {
                        3.0
                    } else {
                        0.0
                    };
                    (f, (s + noise.sample(&mut rng)) as f32)
                })
                .collect()
        })
        .collect();
    let inc = score(g, &profiles, &vec![0; xy.len()], &tile, &edges);
    assert!(inc.pi[n_signal + n_noise..].iter().all(|&p| p == 0.0));
    let noise_pi = inc.pi[n_signal..n_signal + n_noise].iter().sum::<f32>() / n_noise as f32;
    assert!(noise_pi < 0.1, "noise features' mean π {noise_pi}");
    assert!(
        inc.pi[..n_signal].iter().all(|&p| p > 0.9),
        "{:?}",
        &inc.pi[..n_signal]
    );
}

/// One extreme t widens the range but not the kernel: the density stays a
/// density, so the null bulk is not called signal.
#[test]
fn one_extreme_t_does_not_inflate_the_density() {
    let mut rng = SmallRng::seed_from_u64(13);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let t: Vec<f64> = (0..999)
        .map(|_| noise.sample(&mut rng))
        .chain([1e6])
        .collect();
    let (pi, _) = two_group_inclusion(&t);
    let null = pi[..999].iter().sum::<f32>() / 999.0;
    assert!(null < 0.05, "null features' mean π {null}");
    assert!(pi[999] > 0.99);
}

/// Cells read section by section in column blocks score as the same units
/// held in memory, tiles as blocks; one coordinate column is refused.
#[test]
fn cells_stream_to_the_same_inclusion() {
    use crate::cell_activity_graph_embedding::feature_inclusion::cell_inclusion;
    use crate::test_support::{numbered, sparse_vec};
    use crate::util::common::{log_normalise, Mat};
    use crate::util::feature_axis::FeatureAxis;
    let dir = tempfile::tempdir().unwrap();
    let (xy, edges) = grid(6);
    let n = xy.len();
    // two sections of the grid; feature 0 left/right, 1 and 2 scattered
    let mut triplets = Vec::new();
    for c in 0..2 * n {
        let (x, y) = xy[c % n];
        triplets.push((0, c as u64, if x < 3.0 { 5.0 } else { 1.0 }));
        triplets.push((1, c as u64, 1.0 + ((c * 7) % 5) as f32));
        if (x as usize + y as usize).is_multiple_of(2) {
            triplets.push((2, c as u64, 2.0));
        }
    }
    let rows = numbered("g", 3);
    let data = sparse_vec(dir.path(), "d", &rows, &numbered("c", 2 * n), &triplets);
    let axis = FeatureAxis::resolve_or_identity(&rows).unwrap();
    let section: Vec<u32> = (0..2 * n).map(|c| (c / n) as u32).collect();
    let all_edges: Edges = edges
        .iter()
        .flat_map(|&(i, j)| [(i, j), (i + n, j + n)])
        .collect();
    let names: Vec<Box<str>> = vec!["x".into(), "y".into()];
    let plane = Mat::from_fn(
        2 * n,
        2,
        |c, k| if k == 0 { xy[c % n].0 } else { xy[c % n].1 },
    );
    let inc = cell_inclusion(
        (&data, None),
        &axis,
        (&plane, &names),
        &section,
        &all_edges,
        4,
        5,
    )
    .unwrap();

    let mut profiles: Vec<Vec<(u32, f32)>> = vec![Vec::new(); 2 * n];
    for &(r, c, v) in &triplets {
        profiles[c as usize].push((r as u32, v));
    }
    profiles.iter_mut().for_each(|p| log_normalise(p));
    let tile = tile_of(&xy, 2);
    let block: Vec<usize> = (0..2 * n).map(|c| tile[c % n] + 4 * (c / n)).collect();
    let held = score(3, &profiles, &section, &block, &all_edges);
    for f in 0..3 {
        assert!((inc.moran[f] - held.moran[f]).abs() < 1e-6, "feature {f}");
        assert!((inc.se[f] - held.se[f]).abs() < 1e-6, "feature {f}");
    }

    let one: Vec<Box<str>> = vec!["x".into()];
    let line = plane.columns(0, 1).into_owned();
    let e = cell_inclusion(
        (&data, None),
        &axis,
        (&line, &one),
        &section,
        &all_edges,
        4,
        5,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("two coordinate columns"), "{e}");
}

/// The linear edge term is built per unit from its edge counts toward each
/// block, not per edge: sparse profiles over several features, scattered
/// blocks (most edges between two) and a repeated edge still match Moran's
/// I and its jackknife recomputed from scratch.
#[test]
fn per_unit_edge_counts_match_the_edges() {
    let mut rng = SmallRng::seed_from_u64(17);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let (xy, mut edges) = grid(5);
    edges.push(edges[3]);
    let n = xy.len();
    let block: Vec<usize> = (0..n).map(|u| (u * 7) % 4).collect();
    // feature f observed on every (f + 1)-th unit
    let xs: Vec<Vec<f64>> = (0..3)
        .map(|f| {
            (0..n)
                .map(|u| {
                    if u % (f + 1) == 0 {
                        xy[u].0 as f64 + noise.sample(&mut rng)
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let profiles: Vec<Vec<(u32, f32)>> = (0..n)
        .map(|u| {
            (0..3)
                .filter(|&f| xs[f][u] != 0.0)
                .map(|f| (f as u32, xs[f][u] as f32))
                .collect()
        })
        .collect();
    let (moran, se, _) = jackknife_moran(&[section_jack(3, &profiles, &block, &edges)]);
    for f in 0..3 {
        let x: Vec<f64> = xs[f].iter().map(|&v| v as f32 as f64).collect();
        let full = moran_brute(&[(x.clone(), edges.clone(), vec![true; n])]);
        assert!(
            (moran[f] - full).abs() < 1e-6,
            "feature {f}: {} vs {full}",
            moran[f]
        );
        let brute_se = brute_jackknife_se(&[x], &edges, &block, 4);
        assert!(
            (se[f] - brute_se).abs() < 1e-6,
            "feature {f}: {} vs {brute_se}",
            se[f]
        );
    }
}

/// PB edges fold the spatial cell edges within a section; the frame's
/// super edges are taken as they are only where its batch filter keeps
/// the same cell edges.
#[test]
fn pb_edges_fold_within_sections_and_reuse_the_frame_only_when_it_agrees() {
    use crate::cell_activity_graph_embedding::feature_inclusion::pb_spatial_edges;
    // six cells in PBs [0, 0, 1, 1, 2, 2]; cells 0..4 in section 0
    let cell_to_pb = [0, 0, 1, 1, 2, 2];
    let section = [0u32, 0, 0, 0, 1, 1];
    let edges = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5)];
    let folded = pb_spatial_edges(&cell_to_pb, &section, &edges, None);
    assert_eq!(folded, vec![(0, 1)]);
    // a frame whose batches are the sections is reused as it is
    let frame: [(u32, u32); 1] = [(0, 1)];
    let reused = pb_spatial_edges(&cell_to_pb, &section, &edges, Some((&frame, &section)));
    assert_eq!(reused, folded);
    // one batch keeps the edge across sections, so the frame is not reused
    let one_batch = [0u32; 6];
    let wide: [(u32, u32); 2] = [(0, 1), (1, 2)];
    let own = pb_spatial_edges(&cell_to_pb, &section, &edges, Some((&wide, &one_batch)));
    assert_eq!(own, folded);
}

/// A stage-1 model on a `side × side` grid in one batch: two communities,
/// the left half community 0 and the right community 1 (mixed `mix` toward
/// the other), each cell's total `n`, over `g` rows: 0..5 high in community
/// 0, 5..10 in community 1, the rest flat. Its tables and divisor on the cells.
fn stage1_model(
    side: usize,
    g: usize,
    n: f32,
    mix: f32,
) -> (
    crate::util::stage1_divisor::Stage1Tables,
    crate::util::stage1_divisor::Stage1Divisor,
) {
    use crate::util::stage1_divisor::{Stage1Divisor, Stage1Tables};
    let (cells, k) = (side * side, 2usize);
    let names: Vec<Box<str>> = (0..cells).map(|i| format!("c{i}").into()).collect();
    let rows: Vec<Box<str>> = (0..g).map(|r| format!("g{r}").into()).collect();
    let mut theta = Vec::with_capacity(cells * k);
    for i in 0..cells {
        let left = (i % side) < side / 2;
        let a = if left { 1.0 - mix } else { mix };
        theta.extend([a, 1.0 - a]);
    }
    let weight = |r: usize, t: usize| match (r, t) {
        (0..5, 0) | (5..10, 1) => 3.0,
        (0..10, _) => 0.2,
        _ => 1.0,
    };
    let mut profiles = vec![0f32; g * k];
    for t in 0..k {
        let total: f32 = (0..g).map(|r| weight(r, t)).sum();
        for r in 0..g {
            profiles[r * k + t] = weight(r, t) / total;
        }
    }
    let mu = |i: usize, r: usize| {
        n * (0..k)
            .map(|t| theta[i * k + t] * profiles[r * k + t])
            .sum::<f32>()
    };
    let pbar: Vec<f32> = (0..g)
        .map(|r| (0..cells).map(|i| mu(i, r)).sum::<f32>() / (n * cells as f32))
        .collect();
    // Poisson: σ² the batch's mean μ
    let sigma2: Vec<f32> = (0..g)
        .map(|r| (0..cells).map(|i| mu(i, r)).sum::<f32>() / cells as f32)
        .collect();
    let tables = Stage1Tables {
        cells: names.clone(),
        n: vec![n; cells],
        batch: vec![0; cells],
        theta,
        k,
        batches: vec!["all".into()],
        features: rows.clone(),
        profiles,
        pbar,
        phi: vec![0.0; g],
        sigma2,
    };
    let s1 = Stage1Divisor::align(&tables, &names, &rows).unwrap();
    (tables, s1)
}

/// Counts drawn from `s1`'s expectation, Poisson, each cell's sorted
/// `(row, count)` nonzeros.
fn draw_counts(s1: &crate::util::stage1_divisor::Stage1Divisor, seed: u64) -> Vec<Vec<(u32, f32)>> {
    use rand_distr::Poisson;
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..s1.n_cells())
        .map(|i| {
            (0..s1.n_rows())
                .filter_map(|r| {
                    let y = Poisson::new(f64::from(s1.mu(i, r)))
                        .unwrap()
                        .sample(&mut rng);
                    (y > 0.0).then_some((r as u32, y as f32))
                })
                .collect()
        })
        .collect()
}

/// Each cell's stage-1 mixture `n_i θ_i`, flat, and its batch.
fn mixtures(s1: &crate::util::stage1_divisor::Stage1Divisor) -> (Vec<f32>, Vec<usize>) {
    let w = (0..s1.n_cells())
        .flat_map(|i| s1.theta(i).iter().map(move |&t| s1.total(i) * t))
        .collect();
    (w, (0..s1.n_cells()).map(|i| s1.batch_of(i)).collect())
}

/// Moran's I on Pearson residuals, the zeros in closed form, equals the
/// same statistic on the dense residuals `(y − μ) / σ` of every cell and
/// row, its jackknife too.
#[test]
fn pearson_moran_equals_the_dense_computation() {
    use crate::cell_activity_graph_embedding::feature_inclusion::{
        section_jack_pearson, PearsonUnits,
    };
    let side = 6;
    let (_, s1) = stage1_model(side, 30, 40.0, 0.2);
    let counts = draw_counts(&s1, 11);
    let (xy, edges) = grid(side);
    let tile = tile_of(&xy, 2);
    let (w, batch) = mixtures(&s1);
    let units = PearsonUnits {
        counts: &counts,
        w: &w,
        batch: &batch,
        stage1: &s1,
    };
    let g = s1.n_rows();
    let closed = jackknife_moran(&[section_jack_pearson(g, &units, &tile, &edges)]);
    let dense: Vec<Vec<(u32, f32)>> = (0..s1.n_cells())
        .map(|i| {
            (0..g)
                .map(|r| {
                    let y = counts[i]
                        .iter()
                        .find(|e| e.0 as usize == r)
                        .map_or(0.0, |e| e.1);
                    let res = (y - s1.mu(i, r)) / s1.sigma(r, 0);
                    (r as u32, res)
                })
                .collect()
        })
        .collect();
    let brute = jackknife_moran(&[section_jack(g, &dense, &tile, &edges)]);
    for f in 0..g {
        for (what, a, b) in [
            ("I", closed.0[f], brute.0[f]),
            ("se", closed.1[f], brute.1[f]),
        ] {
            assert!(
                (a - b).abs() < 1e-4 * b.abs().max(1.0),
                "row {f} {what}: {a} vs {b}"
            );
        }
    }
}

/// Counts drawn from the stage-1 model itself, over many rows: on
/// log-normalised counts the community rows carry the model's own spatial
/// structure, and on Pearson residuals under the model every row reads as
/// the null. The rows' mean Moran's I sits at its null expectation
/// `−1/(N − 1)` within four standard errors of that mean, and the community
/// rows' is no longer apart from the flat rows': a tenth of what it is on
/// the counts at most.
#[test]
fn under_a_null_from_the_stage_1_model_no_row_passes() {
    use crate::cell_activity_graph_embedding::feature_inclusion::{
        section_jack_pearson, PearsonUnits,
    };
    let (side, g) = (24, 400);
    let (_, s1) = stage1_model(side, g, 400.0, 0.1);
    let counts = draw_counts(&s1, 5);
    let (xy, edges) = grid(side);
    let tile = tile_of(&xy, 4);
    let logged = normalised_profiles(&crate::util::common::Mat::from_fn(
        g,
        s1.n_cells(),
        |r, i| {
            counts[i]
                .iter()
                .find(|e| e.0 as usize == r)
                .map_or(0.0, |e| e.1)
        },
    ));
    let (raw, _, t) = jackknife_moran(&[section_jack(g, &logged, &tile, &edges)]);
    let (pi, _) = two_group_inclusion(&t);
    let community = pi[..10].iter().sum::<f32>() / 10.0;
    assert!(
        community > 0.9,
        "log-normalised: community rows' mean π {community}"
    );
    let (w, batch) = mixtures(&s1);
    let units = PearsonUnits {
        counts: &counts,
        w: &w,
        batch: &batch,
        stage1: &s1,
    };
    let (moran, _, _) = jackknife_moran(&[section_jack_pearson(g, &units, &tile, &edges)]);
    let mean = moran.iter().sum::<f64>() / g as f64;
    let sd = (moran.iter().map(|i| (i - mean).powi(2)).sum::<f64>() / (g - 1) as f64).sqrt();
    let e0 = -1.0 / (s1.n_cells() - 1) as f64;
    assert!(
        (mean - e0).abs() < 4.0 * sd / (g as f64).sqrt(),
        "Pearson residuals: mean I {mean}, null {e0}, sd {sd}"
    );
    let abs_mean = |v: &[f64]| v.iter().map(|x| x.abs()).sum::<f64>() / v.len() as f64;
    let (on_counts, on_residuals) = (abs_mean(&raw[..10]), abs_mean(&moran[..10]));
    assert!(
        on_residuals < on_counts / 10.0,
        "community rows' |I|: {on_residuals} on residuals, {on_counts} on counts"
    );
}
