use super::fixture::{synth_cells, synth_communities};
use crate::view::data::{Geometry, NO_CLUSTER};
use crate::view::index::{Grid, Pyramid};

#[test]
fn grid_holds_every_cell_once_in_its_bin() {
    let geom = Geometry::from_cells(synth_cells(5000, 3));
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let mut seen = vec![false; geom.n()];
    for iy in 0..grid.ny {
        for ix in 0..grid.nx {
            for &i in grid.cells(ix, iy) {
                let i = i as usize;
                assert!(!seen[i]);
                seen[i] = true;
                assert_eq!(grid.bin_of(geom.x[i], geom.y[i]), (ix, iy));
            }
        }
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn pyramid_preserves_totals_at_every_level() {
    let geom = Geometry::from_cells(synth_cells(20_000, 2));
    let k = 5;
    let comm = synth_communities(&geom, k);
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let pyr = Pyramid::build(&grid, &comm);

    let top = pyr.levels.last().unwrap();
    assert_eq!((top.nx, top.ny), (1, 1));

    let base_prop: Vec<f32> = (0..k)
        .map(|c| pyr.levels[0].prop.iter().skip(c).step_by(k).sum())
        .collect();
    for (l, level) in pyr.levels.iter().enumerate() {
        let n: u32 = level.count.iter().sum();
        assert_eq!(n as usize, geom.n(), "level {l}");
        for (c, &want) in base_prop.iter().enumerate() {
            let got: f32 = level.prop.iter().skip(c).step_by(k).sum();
            assert!((got - want).abs() / want < 1e-4, "level {l} community {c}");
        }
        let ent: f32 = level.entropy.iter().sum();
        assert!((ent / geom.n() as f32 - 128. / 255.).abs() < 1e-3);
        for (b, &top) in level.top.iter().enumerate() {
            assert_eq!(top == NO_CLUSTER, level.count[b] == 0);
        }
    }
}

/// The cells' pitch: the median distance to each one's nearest neighbour.
#[test]
fn the_pitch_is_the_median_nearest_neighbour_distance() {
    use crate::view::data::Rect;
    use crate::view::index::Grid;
    // A hexagonal lattice of pitch 10, as Visium spots lie.
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for row in 0..30 {
        for col in 0..30 {
            let shift = if row % 2 == 0 { 0. } else { 5. };
            x.push(col as f32 * 10. + shift);
            y.push(row as f32 * 10. * 3f32.sqrt() / 2.);
        }
    }
    let bounds = Rect {
        x0: 0.,
        y0: 0.,
        x1: 300.,
        y1: 270.,
    };
    let grid = Grid::build(&x, &y, bounds, 4.);
    let pitch = grid.pitch(&x, &y).unwrap();
    assert!((pitch - 10.).abs() < 1e-3, "{pitch}");
    let one = Grid::build(&[1.], &[1.], bounds, 4.);
    assert_eq!(one.pitch(&[1.], &[1.]), None);
}
