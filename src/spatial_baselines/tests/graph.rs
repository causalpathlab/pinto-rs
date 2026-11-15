use crate::spatial_baselines::graph::knn_within_sections;
use crate::spatial_baselines::input::SpatialData;
use crate::test_support::{boxed, numbered};
use crate::util::ablation::{n_sections, section_sizes};
use crate::util::common::Mat;
use nalgebra_sparse::{CooMatrix, CscMatrix};

pub(crate) fn toy(xy: &[(f32, f32)], section: &[usize]) -> SpatialData {
    let n = xy.len();
    let counts = CscMatrix::from(&CooMatrix::<f32>::new(1, n));
    SpatialData {
        counts,
        genes: boxed(&["GENE1"]),
        cells: numbered("c", n),
        xy: Mat::from_fn(n, 2, |i, j| if j == 0 { xy[i].0 } else { xy[i].1 }),
        coord_names: boxed(&["x", "y"]),
        batches: section.iter().map(|s| format!("S{s}").into()).collect(),
        section: section.to_vec(),
        n_sections: n_sections(section),
        delta: None,
    }
}

/// A `side × side` grid of unit spacing, in `n_sections` copies.
pub(crate) fn grid(side: usize, n_sections: usize) -> SpatialData {
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

#[test]
fn neighbours_stay_within_sections() {
    // two sections on the same coordinates
    let pts = [(0., 0.), (1., 0.), (3., 0.), (0., 0.), (1., 0.), (3., 0.)];
    let d = toy(&pts, &[0, 0, 0, 1, 1, 1]);
    let g = knn_within_sections(&d, 5);
    assert_eq!(g.index[0], vec![1, 2]);
    assert_eq!(g.index[3], vec![4, 5]);
    assert_eq!(g.dist[2], vec![2.0, 3.0]);
    let s = g.symmetrised();
    assert_eq!(s.n_edges(), g.n_edges());
}

/// Sections are numbered from 0, so one more than the largest label; a
/// section with no cells (none labelled 1 here) still has its size, 0.
#[test]
fn section_sizes_count_every_section_from_its_labels() {
    assert_eq!(n_sections::<usize>(&[]), 0);
    assert_eq!(n_sections(&[0, 2, 2]), 3);
    assert_eq!(section_sizes(&[0, 2, 2, 0, 2]), vec![2, 0, 3]);
    assert!(section_sizes::<usize>(&[]).is_empty());
    // the reader's u32 labels as well
    assert_eq!(section_sizes(&[1u32, 1, 0]), vec![1, 2]);
}

/// Each edge whose reverse the kNN lacks is added to its other end, the
/// lists re-sorted by distance (ties by index).
#[test]
fn symmetrising_adds_the_missing_reverse_edges() {
    let d = toy(&[(0., 0.), (1., 0.), (2., 0.), (10., 0.)], &[0; 4]);
    let g = knn_within_sections(&d, 1);
    assert_eq!(g.index, vec![vec![1], vec![0], vec![1], vec![2]]);
    let s = g.symmetrised();
    assert_eq!(s.index, vec![vec![1], vec![0, 2], vec![1, 3], vec![2]]);
    assert_eq!(
        s.dist,
        vec![vec![1.0], vec![1.0, 1.0], vec![1.0, 8.0], vec![8.0]]
    );
}
