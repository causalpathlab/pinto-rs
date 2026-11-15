//! `--feature-inclusion moran`: the jackknife deletes exactly, spatial
//! features get high inclusion and noise none, and sampling respects π.

use crate::cell_activity_graph_embedding::feature_inclusion::{
    inclusion_of, jackknife_moran, normalised_profiles, sample_included, section_jack, tile_of,
    tree_blocks, two_group_inclusion, UnitGraph,
};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

type Edges = Vec<(usize, usize)>;

/// One section for [`moran_brute`]: a feature's values, edges, cells kept.
type Section = (Vec<f64>, Edges, Vec<bool>);

/// A `side × side` grid's rook edges (each once) and coordinates.
fn grid(side: usize) -> (Vec<(f32, f32)>, Edges) {
    let xy = (0..side * side)
        .map(|c| ((c % side) as f32, (c / side) as f32))
        .collect();
    let mut e = Vec::new();
    for r in 0..side {
        for c in 0..side {
            let i = r * side + c;
            if c + 1 < side {
                e.push((i, i + 1));
            }
            if r + 1 < side {
                e.push((i, i + side));
            }
        }
    }
    (xy, e)
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
    let mut rng = SmallRng::seed_from_u64(1);
    let mut visits = [0usize; 3];
    for _ in 0..2000 {
        let mut f = vec![0, 1, 2];
        sample_included(&mut f, &pi, &mut rng);
        for g in f {
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
    let g = UnitGraph {
        profiles: (0..2 * n)
            .map(|u| vec![(0u32, x[u % n] as f32 * (1 + u / n) as f32)])
            .collect(),
        section: (0..2 * n).map(|u| (u / n) as u32).collect(),
        block: (0..2 * n).map(|u| checker[u % n] + 3 * (u / n)).collect(),
        edges: edges
            .iter()
            .flat_map(|&(i, j)| [(i, j), (i + n, j + n)])
            .collect(),
    };
    let inc = inclusion_of(1, &g);
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
    let mut rng = SmallRng::seed_from_u64(2);
    for _ in 0..100 {
        let mut f = vec![0, 1, 2, 3, 4];
        sample_included(&mut f, &pi, &mut rng);
        assert_eq!(f.len(), 1);
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
    let unit = UnitGraph {
        profiles,
        section: vec![0; xy.len()],
        block: tile,
        edges,
    };
    let inc = inclusion_of(g, &unit);
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
    use crate::cell_activity_graph_embedding::feature_inclusion::{cell_inclusion, log_normalise};
    use crate::test_support::{numbered, sparse_vec};
    use crate::util::common::Mat;
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
    let inc = cell_inclusion(&data, &axis, (&plane, &names), &section, &all_edges, 4, 5).unwrap();

    let mut profiles: Vec<Vec<(u32, f32)>> = vec![Vec::new(); 2 * n];
    for &(r, c, v) in &triplets {
        profiles[c as usize].push((r as u32, v));
    }
    profiles.iter_mut().for_each(|p| log_normalise(p));
    let tile = tile_of(&xy, 2);
    let unit = UnitGraph {
        profiles,
        section,
        block: (0..2 * n).map(|c| tile[c % n] + 4 * (c / n)).collect(),
        edges: all_edges.clone(),
    };
    let held = inclusion_of(3, &unit);
    for f in 0..3 {
        assert!((inc.moran[f] - held.moran[f]).abs() < 1e-6, "feature {f}");
        assert!((inc.se[f] - held.se[f]).abs() < 1e-6, "feature {f}");
    }

    let one: Vec<Box<str>> = vec!["x".into()];
    let line = plane.columns(0, 1).into_owned();
    let section: Vec<u32> = (0..2 * n).map(|c| (c / n) as u32).collect();
    let e = cell_inclusion(&data, &axis, (&line, &one), &section, &all_edges, 4, 5)
        .unwrap_err()
        .to_string();
    assert!(e.contains("two coordinate columns"), "{e}");
}
