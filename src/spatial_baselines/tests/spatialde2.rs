use super::graph::toy;
use crate::spatial_baselines::graph::knn_within_sections;
use crate::spatial_baselines::input::SpatialData;
use crate::spatial_baselines::spatialde2::*;
use crate::util::common::Mat;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// A `side × side` grid of unit spacing, in `n_sections` copies.
fn grid(side: usize, n_sections: usize) -> SpatialData {
    let mut xy = Vec::new();
    let mut sec = Vec::new();
    for s in 0..n_sections {
        for r in 0..side {
            for c in 0..side {
                xy.push((r as f32, c as f32));
                sec.push(s);
            }
        }
    }
    toy(&xy, &sec)
}

fn graph_of(data: &SpatialData, k: usize, smoothness: f64) -> WeightedGraph {
    let nb = knn_within_sections(data, k).symmetrised();
    WeightedGraph::from_neighbours(&nb.index, &nb.dist, &data.section, smoothness)
}

/// Two regions (left and right halves of each grid), each with its own
/// marker gene, and a shared gene.
fn two_region_counts(side: usize, n_sections: usize) -> (CellCounts, Vec<f64>, Vec<usize>) {
    let mut rows = Vec::new();
    let mut truth = Vec::new();
    for _ in 0..n_sections {
        for _r in 0..side {
            for c in 0..side {
                let i = rows.len();
                let left = c < side / 2;
                let hi = 20.0 + (i % 3) as f64;
                let lo = 1.0 + (i % 2) as f64;
                let (a, b) = if left { (hi, lo) } else { (lo, hi) };
                rows.push(vec![(0, a), (1, b), (2, 10.0)]);
                truth.push(usize::from(!left));
            }
        }
    }
    let sf: Vec<f64> = rows
        .iter()
        .map(|r| r.iter().map(|&(_, v)| v).sum())
        .collect();
    (CellCounts::from_rows(3, &rows), sf, truth)
}

#[test]
fn weights_scale_with_inverse_squared_distance() {
    let d = grid(3, 1);
    let g = graph_of(&d, 8, 2.0);
    // the centre cell (index 4): 4 neighbours at 1, 4 at sqrt(2)
    let mut w: Vec<f64> = g.neighbours(4).map(|(_, w)| w).collect();
    w.sort_by(f64::total_cmp);
    assert_eq!(w.len(), 8);
    for &x in &w[..4] {
        assert!((x - 2.0).abs() < 1e-6, "{x}");
    }
    for &x in &w[4..] {
        assert!((x - 4.0).abs() < 1e-6, "{x}");
    }
    // symmetric: every edge listed both ways with the same weight
    for i in 0..9 {
        for (j, w) in g.neighbours(i) {
            let back = g.neighbours(j).find(|&(m, _)| m == i).unwrap().1;
            assert!((w - back).abs() < 1e-12);
        }
    }
}

#[test]
fn neighbourhood_prior_is_normalised_and_favours_neighbours() {
    let d = grid(3, 1);
    let g = graph_of(&d, 8, 1.0);
    // centre's four nearest neighbours (weight 2 each) carry label 1, the
    // four diagonals (weight 1 each) label 0
    let labels = vec![0, 1, 0, 1, 2, 1, 0, 1, 0];
    let mut lp = vec![0.0; 3];
    neighbourhood_log_prior_into(&g, &labels, 4, &mut lp);
    let total: f64 = lp.iter().map(|v| v.exp()).sum();
    assert!((total - 1.0).abs() < 1e-12);
    // energies -(W - same): W = 12; class 0: -8, class 1: -4, class 2: -12
    assert!((lp[1] - lp[0] - 4.0).abs() < 1e-6);
    assert!((lp[0] - lp[2] - 4.0).abs() < 1e-6);
}

#[test]
fn stick_terms_follow_the_reference_indexing() {
    let (vsum, v2, v3) = stick_terms(&[], &[]);
    assert_eq!(vsum, vec![0.0]);
    assert!(v2.is_empty() && v3.is_empty());
    let (vsum, v2, v3) = stick_terms(&[2.0, 3.0], &[1.5, 0.5]);
    assert_eq!(vsum.len(), 3);
    // concat((0,), exclusive_cumsum(v3)) + concat(v2, (0,))
    assert!((vsum[0] - v2[0]).abs() < 1e-15);
    assert!((vsum[1] - v2[1]).abs() < 1e-15);
    assert!((vsum[2] - v3[0]).abs() < 1e-15);
    // E[log v] and E[log(1 - v)] of a Beta are negative
    assert!(v2.iter().chain(&v3).all(|&v| v < 0.0));
}

#[test]
fn posteriors_sum_to_one_and_statistics_add_up() {
    let d = grid(4, 1);
    let g = graph_of(&d, 8, 2.0);
    let (counts, sf, _) = two_region_counts(4, 1);
    let p = Params::default();
    let st = State::init(3, counts.n_genes, p.eta_1);
    let labels: Vec<usize> = (0..16).map(|i| i % 3).collect();
    let u = update(
        Vec::new(),
        &counts,
        &sf,
        Some((&g, labels.as_slice())),
        &st,
        3,
        &p,
    );
    for row in u.pi.chunks(3) {
        // φ is ~1e7 at the 1e-6 starting rates: logsumexp keeps ~1e-9
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-6, "{row:?}");
    }
    // gammahat_2 sums the size factors over the classes
    let g2: f64 = u.state.gamma_hat_2.iter().sum::<f64>() - 3.0 * p.gamma_2;
    assert!((g2 - sf.iter().sum::<f64>()).abs() < 1e-6);
    // gammahat_1 sums each gene's counts over the classes
    let gene0: f64 = (0..3).map(|c| u.state.gamma_hat_1[c * 3]).sum();
    let total0: f64 = (0..16)
        .map(|i| if i % 4 < 2 { 20.0 } else { 1.0 })
        .sum::<f64>();
    assert!(gene0 > total0 - 1.0); // counts jitter upward only
    assert!(u.elbo.is_finite());
    assert_eq!(u.state.alpha_1.len(), 2);
}

#[test]
fn prune_drops_unused_classes_and_renumbers() {
    let pi = vec![0.7, 0.0, 0.3, 0.1, 1e-9, 0.9];
    let (idx, labels) = prune(&pi, 3, &[0, 2], 1e-6);
    assert_eq!(idx, vec![0, 2]);
    assert_eq!(labels, vec![0, 1]);
    assert_eq!(argmax_rows(&pi, 3), vec![0, 2]);
}

#[test]
fn segments_two_regions_and_prunes_the_rest() {
    let d = grid(10, 2);
    let g = graph_of(&d, 8, 2.0);
    let (counts, sf, truth) = two_region_counts(10, 2);
    let seg = segment(&counts, &sf, Some(&g), &Params::default(), 7).unwrap();
    // ceil(sqrt(200)) = 15 classes to start, most pruned; the regions
    // found may split a true region (a local optimum from the random
    // start), but none straddles the two
    assert_eq!(seg.nclasses_trace[0], 15);
    assert!(seg.k >= 2 && seg.k < 15, "regions: {}", seg.k);
    assert!(!seg.prune_iterations.is_empty());
    for row in seg.pi.chunks(seg.k) {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
    for c in 0..seg.k {
        let sides: Vec<usize> = (0..200)
            .filter(|&i| seg.labels[i] == c)
            .map(|i| truth[i])
            .collect();
        assert!(!sides.is_empty(), "pruned class {c} has no cell");
        assert!(sides.iter().all(|&t| t == sides[0]), "class {c} straddles");
    }
    // between prunings (updates 2 to 10) the closed-form updates raise
    // the ELBO (×K, undoing the per-class normalisation)
    let raw: Vec<f64> = seg
        .elbo_trace
        .iter()
        .zip(&seg.nclasses_trace)
        .map(|(e, &k)| e * k as f64)
        .collect();
    assert!(seg.nclasses_trace[2..=10]
        .iter()
        .all(|&k| k == seg.nclasses_trace[2]));
    for w in raw[2..=10].windows(2) {
        assert!(w[1] >= w[0] - 1e-12, "{} -> {}", w[0], w[1]);
    }
}

#[test]
fn zero_count_cell_takes_its_prior_and_runs_repeat() {
    let d = grid(10, 2);
    let g = graph_of(&d, 8, 2.0);
    let (_, _, truth) = two_region_counts(10, 2);
    // cell 0 (left half) has no counts and size factor 0, as the
    // reference allows
    let rows: Vec<Vec<(usize, f64)>> = (0..truth.len())
        .map(|i| {
            if i == 0 {
                Vec::new()
            } else {
                let hi = 20.0 + (i % 3) as f64;
                let lo = 1.0 + (i % 2) as f64;
                let (a, b) = if truth[i] == 0 { (hi, lo) } else { (lo, hi) };
                vec![(0, a), (1, b), (2, 10.0)]
            }
        })
        .collect();
    let sf: Vec<f64> = rows
        .iter()
        .map(|r| r.iter().map(|&(_, v)| v).sum())
        .collect();
    assert_eq!(sf[0], 0.0);
    let counts = CellCounts::from_rows(3, &rows);
    let a = segment(&counts, &sf, Some(&g), &Params::default(), 3).unwrap();
    let b = segment(&counts, &sf, Some(&g), &Params::default(), 3).unwrap();
    assert!(a.pi.iter().all(|v| v.is_finite()));
    assert!((a.pi[..a.k].iter().sum::<f64>() - 1.0).abs() < 1e-9);
    // its label follows its neighbours (cells 1 and 10)
    assert!(a.labels[0] == a.labels[1] || a.labels[0] == a.labels[10]);
    // same seed, same bits
    assert_eq!(a.elbo_trace, b.elbo_trace);
    assert_eq!(a.labels, b.labels);
}

#[test]
fn class_cap_and_no_space() {
    let (counts, sf, _) = two_region_counts(6, 1);
    let p = Params {
        nclasses: Some(4),
        ..Params::default()
    };
    // neighbors = 0: a Poisson mixture with a DP prior
    let seg = segment(&counts, &sf, None, &p, 1).unwrap();
    assert_eq!(seg.nclasses_trace[0], 4);
    assert!(seg.k <= 4 && seg.k >= 2);
    let bad = Params {
        smoothness_factor: 0.0,
        ..Params::default()
    };
    assert!(segment(&counts, &sf, None, &bad, 1).is_err());
}

#[test]
fn genes_by_total_counts_and_size_factors_over_all() {
    // genes × cells: gene 0 total 3, gene 1 total 10, gene 2 total 5
    let mut coo = CooMatrix::<f32>::new(3, 2);
    coo.push(0, 0, 1.0);
    coo.push(0, 1, 2.0);
    coo.push(1, 0, 4.0);
    coo.push(1, 1, 6.0);
    coo.push(2, 1, 5.0);
    let mut d = toy(&[(0., 0.), (1., 0.)], &[0, 0]);
    d.counts = CscMatrix::from(&coo);
    d.genes = vec!["a".into(), "b".into(), "c".into()];
    d.xy = Mat::zeros(2, 2);
    let (counts, sf, genes) = select_counts(&d, 2);
    assert_eq!(genes, vec![1, 2]);
    assert_eq!(sf, vec![5.0, 13.0]);
    assert_eq!(counts.n_genes, 2);
    assert_eq!(counts.n_cells(), 2);
    let (all, _, genes) = select_counts(&d, 0);
    assert_eq!(genes, vec![0, 1, 2]);
    assert_eq!(all.n_genes, 3);
}
