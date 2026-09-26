use super::data::{Communities, Edges, Geometry, NO_CLUSTER};
use super::index::{Grid, Pyramid};
use crate::util::common::*;
use crate::util::parquet_io::CellTable;

/// `n` cells on a jittered square lattice, split into `n_batches` batches
/// that all share one coordinate frame (as pinto writes them).
fn synth_cells(n: usize, n_batches: usize) -> CellTable {
    let side = (n as f64).sqrt().ceil() as usize;
    let names: Vec<Box<str>> = (0..n).map(|i| format!("c{i}").into_boxed_str()).collect();
    let coords: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let jitter = ((i * 7919) % 97) as f32 / 97.;
            ((i % side) as f32 + 0.3 * jitter, (i / side) as f32)
        })
        .collect();
    let batches = (n_batches > 1).then(|| {
        (0..n)
            .map(|i| format!("b{}", i % n_batches).into())
            .collect()
    });
    let index = names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    CellTable {
        names,
        coords,
        batches,
        index,
        coord_col_names: vec!["x".into(), "y".into()],
    }
}

/// Community = vertical stripe of the lattice; propensity 0.75 on it,
/// the rest spread evenly.
fn synth_communities(geom: &Geometry, k: usize) -> Communities {
    let n = geom.n();
    let names = geom.names.clone();
    let mut prop = Mat::from_element(n, k, 0.25 / (k - 1) as f32);
    let mut cluster = Vec::with_capacity(n);
    for i in 0..n {
        let c = (geom.x[i].max(0.) as usize) % k;
        prop[(i, c)] = 0.75;
        cluster.push(c as i64);
    }
    // Half the maximum entropy, ln K, so it quantizes to 128.
    let entropy = Some(vec![0.5 * (k as f32).ln(); n]);
    Communities::join(geom, "final", (prop, cluster, entropy, names))
}

#[test]
fn single_batch_keeps_coordinates() {
    let cells = synth_cells(100, 1);
    let before = cells.coords.clone();
    let geom = Geometry::from_cells(cells);
    assert_eq!(geom.tiles.len(), 1);
    for (i, &(x, y)) in before.iter().enumerate() {
        assert_eq!((geom.x[i], geom.y[i]), (x, y));
    }
}

#[test]
fn batches_are_tiled_without_overlap() {
    let geom = Geometry::from_cells(synth_cells(1000, 4));
    assert_eq!(geom.tiles.len(), 4);
    for (a, ta) in geom.tiles.iter().enumerate() {
        assert_eq!(ta.n_cells, 250);
        for tb in &geom.tiles[a + 1..] {
            let disjoint = ta.bounds.x1 < tb.bounds.x0
                || tb.bounds.x1 < ta.bounds.x0
                || ta.bounds.y1 < tb.bounds.y0
                || tb.bounds.y1 < ta.bounds.y0;
            assert!(disjoint, "{} overlaps {}", ta.name, tb.name);
        }
    }
    for i in 0..geom.n() {
        let t = &geom.tiles[geom.batch[i] as usize].bounds;
        assert!(t.x0 <= geom.x[i] && geom.x[i] <= t.x1);
        assert!(t.y0 <= geom.y[i] && geom.y[i] <= t.y1);
    }
}

#[test]
fn propensity_joins_by_name_and_reports_gaps() {
    let geom = Geometry::from_cells(synth_cells(10, 1));
    let k = 3;
    // Rows in reverse order, one unknown cell, cell c0 missing.
    let names: Vec<Box<str>> = (1..10)
        .rev()
        .map(|i| format!("c{i}").into_boxed_str())
        .chain(std::iter::once("ghost".into()))
        .collect();
    let mut prop = Mat::zeros(names.len(), k);
    let mut cluster = vec![];
    for (r, name) in names.iter().enumerate() {
        let c = name[1..].parse::<usize>().map_or(0, |i| i % k);
        prop[(r, c)] = 1.;
        cluster.push(c as i64);
    }
    let comm = Communities::join(&geom, "L1", (prop, cluster, None, names));

    assert_eq!(comm.n_missing, 1);
    assert_eq!(comm.n_unmatched, 1);
    assert_eq!(comm.cluster[0], NO_CLUSTER);
    for i in 1..10 {
        let c = i % k;
        assert_eq!(comm.cluster[i], c as u16);
        assert_eq!(comm.prop[i * k + c], 255);
    }
    assert!(comm.entropy.is_none());
}

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

#[test]
fn edges_map_names_to_rows() {
    let geom = Geometry::from_cells(synth_cells(4, 1));
    let pairs: Vec<(Box<str>, Box<str>)> = vec![
        ("c0".into(), "c1".into()),
        ("c2".into(), "ghost".into()),
        ("c3".into(), "c2".into()),
    ];
    let edges = Edges::join(&geom, &pairs, &[4, 5, -1]);
    assert_eq!(edges.len(), 2);
    assert_eq!(edges.n_unmatched, 1);
    assert_eq!((edges.a[0], edges.b[0], edges.community[0]), (0, 1, 4));
    assert_eq!(
        (edges.a[1], edges.b[1], edges.community[1]),
        (3, 2, NO_CLUSTER)
    );
}

mod render {
    use super::*;
    use crate::view::color::{palette, BACKGROUND};
    use crate::view::render::{render, Layer, Scene, Style, Viewport};

    /// A 40×40 lattice filling x,y in 0..40, one community per 10-wide stripe.
    fn fixture() -> (Geometry, Communities, Grid, Pyramid) {
        let geom = Geometry::from_cells(synth_cells(1600, 1));
        let comm = synth_communities(&geom, 4);
        let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
        let pyr = Pyramid::build(&grid, &comm);
        (geom, comm, grid, pyr)
    }

    fn pixel(frame: &crate::view::render::Frame, x: usize, y: usize) -> [u8; 3] {
        let o = (y * frame.w + x) * 4;
        [frame.rgba[o], frame.rgba[o + 1], frame.rgba[o + 2]]
    }

    fn draw(w: usize, window: Rect) -> crate::view::render::Frame {
        draw_focused(w, window, None)
    }

    fn draw_focused(w: usize, window: Rect, focus: Option<&[bool]>) -> crate::view::render::Frame {
        let (geom, comm, grid, pyramid) = fixture();
        let scene = Scene {
            geom: &geom,
            comm: &comm,
            grid: &grid,
            pyramid: &pyramid,
            edges: None,
            spacing: 1.,
        };
        let style = Style {
            layer: Layer::Argmax,
            edges: false,
            focus,
            scale_bar: None,
        };
        render(&scene, &Viewport::fit(window, w, w), &style, &palette(4))
    }

    use crate::view::data::Rect;

    #[test]
    fn points_take_their_community_colour() {
        // Zoomed onto one cell: many pixels per unit of cell spacing.
        let (geom, comm, ..) = fixture();
        let i = 2 * 40 + 2;
        let (x, y) = (geom.x[i], geom.y[i]);
        let frame = draw(
            64,
            Rect {
                x0: x - 1.,
                y0: y - 1.,
                x1: x + 1.,
                y1: y + 1.,
            },
        );
        let want = palette(4)[comm.cluster[i] as usize];
        assert_eq!(pixel(&frame, frame.w / 2, frame.h / 2), want);
    }

    #[test]
    fn focus_dims_other_communities() {
        let (geom, comm, ..) = fixture();
        let i = 2 * 40 + 2;
        let (x, y) = (geom.x[i], geom.y[i]);
        let window = Rect {
            x0: x - 1.,
            y0: y - 1.,
            x1: x + 1.,
            y1: y + 1.,
        };
        let c = comm.cluster[i] as usize;
        let mut focus = vec![false; 4];
        focus[(c + 1) % 4] = true;
        let dimmed = draw_focused(64, window, Some(&focus));
        assert_eq!(pixel(&dimmed, 32, 32), crate::view::color::DIMMED);

        focus[c] = true;
        let shown = draw_focused(64, window, Some(&focus));
        assert_eq!(pixel(&shown, 32, 32), palette(4)[c]);
    }

    #[test]
    fn bins_cover_the_tissue_and_leave_the_outside_empty() {
        // 8 pixels for 40 units: 5 units per pixel, well past point mode.
        let frame = draw(
            8,
            Rect {
                x0: 0.,
                y0: 0.,
                x1: 80.,
                y1: 80.,
            },
        );
        let inside = pixel(&frame, 1, 1);
        assert_ne!(inside, BACKGROUND);
        assert_eq!(pixel(&frame, 7, 7), BACKGROUND);
    }

    #[test]
    fn layer_names_parse() {
        assert_eq!("soft".parse::<Layer>().unwrap(), Layer::Soft);
        assert_eq!("C7".parse::<Layer>().unwrap(), Layer::Community(7));
        assert_eq!("c0".parse::<Layer>().unwrap(), Layer::Community(0));
        assert!("blue".parse::<Layer>().is_err());
    }
}

#[test]
fn thousands_groups_digits() {
    use super::thousands;
    assert_eq!(thousands(0), "0");
    assert_eq!(thousands(999), "999");
    assert_eq!(thousands(1000), "1,000");
    assert_eq!(thousands(814243), "814,243");
    assert_eq!(thousands(1234567), "1,234,567");
}
