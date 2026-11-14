use super::graph::toy;
use crate::spatial_baselines::banksy::*;
use crate::spatial_baselines::graph::Neighbours;
use crate::spatial_baselines::input::SpatialData;
use crate::util::common::Mat;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// A cell at the origin and `n` cells evenly on the unit circle.
fn ring(n: usize) -> (Mat, Neighbours) {
    let mut xy = Mat::zeros(n + 1, 2);
    for k in 0..n {
        let a = std::f32::consts::TAU * k as f32 / n as f32;
        xy[(k + 1, 0)] = a.cos();
        xy[(k + 1, 1)] = a.sin();
    }
    let mut index = vec![Vec::new(); n + 1];
    let mut dist = vec![Vec::new(); n + 1];
    index[0] = (1..=n).collect();
    dist[0] = vec![1.0; n];
    (xy, Neighbours { index, dist })
}

#[test]
fn weights_sum_to_one_and_decay_with_distance() {
    let d = [1.0, 2.0, 3.0, 4.0];
    let w = median_gaussian_weights(&d);
    assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(w.windows(2).all(|p| p[0] > p[1]));
    // exp(-(d / 2.5)^2), summed to one
    let raw: Vec<f32> = d.iter().map(|x: &f32| (-(x / 2.5).powi(2)).exp()).collect();
    let s: f32 = raw.iter().sum();
    for (a, b) in w.iter().zip(&raw) {
        assert!((a - b / s).abs() < 1e-6);
    }
    // coincident neighbours weigh evenly
    assert_eq!(median_gaussian_weights(&[0.0, 0.0]), vec![0.5, 0.5]);
}

#[test]
fn harmonics_use_k_geom_times_m_plus_one() {
    let pts: Vec<(f32, f32)> = (0..10).map(|i| (i as f32, 0.0)).collect();
    let d = toy(&pts, &[0; 10]);
    let h = banksy_weights(&d, 2, 1);
    assert_eq!(h.len(), 2);
    assert_eq!(h[0].index[0].len(), 2);
    assert_eq!(h[1].index[0].len(), 4);
    for hm in &h {
        for w in &hm.weight {
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
    // neighbours to the right sit at angle 0
    assert!(h[1].cos[0].iter().all(|&c| (c - 1.0).abs() < 1e-6));
}

#[test]
fn agf_on_a_symmetric_ring_is_zero() {
    let (xy, g) = ring(8);
    let h = harmonics_of(&g, &xy, 8, 1);
    // ring expression flat at 5, the centre different
    let mut x = vec![5.0f32; 9];
    x[0] = 100.0;
    let mut out = vec![0f32; 9];
    neighbour_feature(&h[0], &x, &mut out);
    assert!((out[0] - 5.0).abs() < 1e-5, "mean {}", out[0]);
    neighbour_feature(&h[1], &x, &mut out);
    assert!(out[0].abs() < 1e-5, "AGF {}", out[0]);
    // a gradient along x is picked up: Σ (1/8) cos φ (cos φ) = 1/2
    let xg: Vec<f32> = (0..9).map(|i| xy[(i, 0)]).collect();
    neighbour_feature(&h[1], &xg, &mut out);
    assert!((out[0] - 0.5).abs() < 1e-5, "AGF {}", out[0]);
    // a radially symmetric pattern (cos 2φ) is invisible to m = 1
    let x2: Vec<f32> = (0..9)
        .map(|i| (2.0 * xy[(i, 1)].atan2(xy[(i, 0)])).cos())
        .collect();
    neighbour_feature(&h[1], &x2, &mut out);
    assert!(out[0].abs() < 1e-5, "AGF {}", out[0]);
}

#[test]
fn lambda_weights_match_get_lambdas() {
    let w = lambda_weights(0.8, 2);
    let sq: Vec<f32> = w.iter().map(|v| v * v).collect();
    assert!((sq[0] - 0.2).abs() < 1e-6);
    assert!((sq[1] - 0.8 / 1.5).abs() < 1e-6);
    assert!((sq[2] - 0.4 / 1.5).abs() < 1e-6);
    assert!((sq.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    let w0 = lambda_weights(0.2, 1);
    assert!((w0[0] * w0[0] - 0.8).abs() < 1e-6 && (w0[1] * w0[1] - 0.2).abs() < 1e-6);
}

#[test]
fn scaling_is_per_section() {
    // one column, two sections on different scales; a flat column goes to 0
    let mut m = Mat::from_column_slice(4, 2, &[1.0, 3.0, 100.0, 300.0, 7.0, 7.0, 7.0, 7.0]);
    scale_by_section(&mut m, &[vec![0, 1], vec![2, 3]], 1, &[2.0, 1.0]);
    let s = std::f32::consts::SQRT_2;
    let want = [-s, s, -s, s];
    for (i, w) in want.iter().enumerate() {
        assert!((m[(i, 0)] - w).abs() < 1e-5, "{}", m[(i, 0)]);
        assert_eq!(m[(i, 1)], 0.0);
    }
}

#[test]
fn library_normalisation_scales_to_mean() {
    let mut coo = CooMatrix::new(2, 3);
    coo.push(0, 0, 1.0);
    coo.push(1, 0, 1.0);
    coo.push(0, 1, 4.0);
    let mut x = CscMatrix::from(&coo);
    normalise_library(&mut x);
    // libraries 2, 4, 0; mean 2
    let col0: f32 = x.col(0).values().iter().sum();
    let col1: f32 = x.col(1).values().iter().sum();
    assert!((col0 - 2.0).abs() < 1e-6 && (col1 - 2.0).abs() < 1e-6);
    assert_eq!(x.col(2).nnz(), 0);
}

#[test]
fn snn_graph_is_undirected_and_summed() {
    // two tight groups far apart
    let mut pcs = Mat::zeros(12, 2);
    for i in 0..12 {
        pcs[(i, 0)] = if i < 6 { 0.0 } else { 100.0 } + 0.01 * i as f32;
        pcs[(i, 1)] = 0.02 * (i % 3) as f32;
    }
    let e = snn_graph(&pcs, 5);
    assert!(e.iter().all(|&(i, j, _)| i < j));
    assert!(e.windows(2).all(|p| (p[0].0, p[0].1) < (p[1].0, p[1].1)));
    // within a group every pair shares 4 of 5 neighbours and is mutual
    // (count 5): all kept; no edge crosses the groups
    assert!(e.iter().all(|&(i, j, _)| (i < 6) == (j < 6)));
    // a mutual edge carries both directions' weights
    assert!(e.iter().any(|&(_, _, w)| w > 1.0));
}

#[test]
fn snn_counts_a_mutual_edge_one_more() {
    // cell 0's kNN (k = 6) in distance order: 1, 2, 3 share 5 of its
    // neighbours and are mutual (count 6); 4 shares 4 and is mutual (5:
    // kept by the >= 5 rule at rank 4); 5 shares 4 one way (4: dropped);
    // 6 shares none
    let mut idx = vec![
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3, 4, 5, 6, 0],
        vec![1, 3, 4, 5, 6, 0],
        vec![1, 2, 4, 5, 6, 0],
        vec![1, 2, 3, 5, 0, 7],
        vec![1, 2, 3, 4, 7, 8],
        vec![7, 8, 9, 10, 11, 12],
    ];
    for i in 7..13 {
        idx.push((7..13).filter(|&j| j != i).chain([0]).collect());
    }
    let kept = snn_kept(&idx);
    assert_eq!(kept[0], vec![true, true, true, true, false, false]);
}

/// Two domains on a 12 × 12 grid per section, two sections: left half
/// expresses genes 0..4 high, the right half genes 4..8; plus noise genes.
fn two_domains() -> (SpatialData, Vec<usize>) {
    let side = 12;
    let mut pts = Vec::new();
    let mut section = Vec::new();
    let mut truth = Vec::new();
    for s in 0..2 {
        for r in 0..side {
            for c in 0..side {
                pts.push((c as f32, r as f32));
                section.push(s);
                truth.push(usize::from(c >= side / 2));
            }
        }
    }
    let n = pts.len();
    let n_genes = 12;
    let mut d = toy(&pts, &section);
    let mut coo = CooMatrix::new(n_genes, n);
    let mut state = 7u64;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f32 / (1u64 << 31) as f32
    };
    for (i, &t) in truth.iter().enumerate() {
        for g in 0..n_genes {
            let base = if g < 8 && (g < 4) == (t == 0) {
                6.0
            } else {
                2.0
            };
            let v = (base * (0.5 + rnd())).round();
            if v > 0.0 {
                coo.push(g, i, v);
            }
        }
    }
    d.counts = CscMatrix::from(&coo);
    d.genes = (0..n_genes).map(|g| format!("g{g}").into()).collect();
    (d, truth)
}

fn test_args(n_clusters: Option<usize>) -> Args {
    use clap::Parser;
    #[derive(clap::Parser)]
    struct Cli {
        #[command(flatten)]
        a: Args,
    }
    let mut argv = vec!["x", "d.zarr", "-c", "d.parquet", "-o", "o", "--knn", "15"];
    let k = n_clusters.map(|k| k.to_string());
    if let Some(k) = k.as_deref() {
        argv.extend(["--n-clusters", k]);
    }
    Cli::parse_from(argv).a
}

#[test]
fn defaults_follow_the_reference() {
    let a = test_args(None);
    assert_eq!(a.lambda, 0.8);
    assert_eq!(a.k_geom, 15);
    assert_eq!(a.max_m, 1);
    assert_eq!(a.n_pcs, 20);
    assert_eq!(a.resolution, 1.0);
    assert!(a.validate().is_ok());
}

#[test]
fn recovers_two_domains_end_to_end() {
    let (mut d, truth) = two_domains();
    let args = test_args(Some(2));
    let (pcs, genes) = banksy_pcs(&mut d, &args).unwrap();
    assert_eq!(pcs.nrows(), truth.len());
    assert!(!genes.is_empty());
    let labels = cluster_pcs(&pcs, 15, 1.0, Some(2), 1).unwrap();
    let k = labels.iter().max().unwrap() + 1;
    assert_eq!(k, 2);
    // agreement up to label swap
    let agree = labels.iter().zip(&truth).filter(|(a, b)| a == b).count();
    let acc = agree.max(truth.len() - agree) as f32 / truth.len() as f32;
    assert!(acc > 0.95, "accuracy {acc}");
}
