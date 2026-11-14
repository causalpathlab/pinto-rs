//! `--feature-inclusion moran`: the jackknife deletes exactly, spatial
//! features get high inclusion and noise none, and sampling respects π.

use crate::cell_activity_graph_embedding::feature_inclusion::{
    jackknife_moran, sample_included, section_sums, tile_of, two_group_inclusion,
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
            section_sums(1, &profiles, &tile, &edges)
        })
        .collect();
    let (moran, se, _) = jackknife_moran(&sums);
    let all = vec![true; xy.len()];
    let full = moran_brute(&[
        (xs[0].clone(), edges.clone(), all.clone()),
        (xs[1].clone(), edges.clone(), all.clone()),
    ]);
    assert!((moran[0] - full).abs() < 1e-4, "{} vs {full}", moran[0]);
    // the jackknife's leave-one-out values, from scratch
    let mut loo = Vec::new();
    for s in 0..2 {
        for k in 0..4 {
            let keep: Vec<bool> = tile.iter().map(|&t| t != k).collect();
            let secs: Vec<_> = (0..2)
                .map(|r| {
                    (
                        xs[r].clone(),
                        edges.clone(),
                        if r == s { keep.clone() } else { all.clone() },
                    )
                })
                .collect();
            loo.push(moran_brute(&secs));
        }
    }
    let m = loo.len() as f64;
    let mean = loo.iter().sum::<f64>() / m;
    let brute_se = ((m - 1.0) / m * loo.iter().map(|v| (v - mean).powi(2)).sum::<f64>()).sqrt();
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
    let (_, _, t) = jackknife_moran(&[section_sums(g, &profiles, &tile, &edges)]);
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
