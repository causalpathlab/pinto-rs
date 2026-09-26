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
