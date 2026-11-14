use crate::spatial_baselines::graph::knn_within_sections;
use crate::spatial_baselines::input::SpatialData;
use crate::util::common::Mat;
use nalgebra_sparse::{CooMatrix, CscMatrix};

pub(crate) fn toy(xy: &[(f32, f32)], section: &[usize]) -> SpatialData {
    let n = xy.len();
    let counts = CscMatrix::from(&CooMatrix::<f32>::new(1, n));
    SpatialData {
        counts,
        genes: vec!["g".into()],
        cells: (0..n).map(|i| format!("c{i}").into()).collect(),
        xy: Mat::from_fn(n, 2, |i, j| if j == 0 { xy[i].0 } else { xy[i].1 }),
        coord_names: vec!["x".into(), "y".into()],
        batches: section.iter().map(|s| format!("S{s}").into()).collect(),
        section: section.to_vec(),
        n_sections: section.iter().max().map_or(0, |&m| m + 1),
    }
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
