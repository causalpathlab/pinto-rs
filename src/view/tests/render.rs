use super::fixture::{synth_cells, synth_communities};
use crate::view::color::Theme;

const DARK: Theme = Theme::Dark;

fn palette(k: usize) -> Vec<[u8; 3]> {
    DARK.palette(k)
}
use crate::view::data::{Communities, Edges, Geometry, Rect, NO_CLUSTER};
use crate::view::index::{EdgeIndex, Grid, Links, Pyramid, HETEROTYPIC};
use crate::view::render::{mode, render, zoom, Frame, Layer, Mode, Scene, Style, Viewport};
use crate::view::EdgesShown;

/// A 40×40 lattice filling x,y in 0..40, one community per 10-wide stripe.
fn fixture() -> (Geometry, Communities, Grid, Pyramid) {
    let geom = Geometry::from_cells(synth_cells(1600, 1));
    let comm = synth_communities(&geom, 4);
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let pyr = Pyramid::build(&grid, &comm);
    (geom, comm, grid, pyr)
}

fn pixel(frame: &Frame, x: usize, y: usize) -> [u8; 3] {
    let o = (y * frame.w + x) * 4;
    [frame.rgba[o], frame.rgba[o + 1], frame.rgba[o + 2]]
}

fn draw(w: usize, window: Rect) -> Frame {
    draw_focused(w, window, None)
}

fn draw_focused(w: usize, window: Rect, focus: Option<&[bool]>) -> Frame {
    draw_styled(w, window, focus, 1.)
}

fn draw_styled(w: usize, window: Rect, focus: Option<&[bool]>, point: f32) -> Frame {
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
        edges: None,
        focus,
        theme: DARK,
        point,
    };
    render(&scene, &Viewport::fit(window, w, w), &style, &palette(4))
}

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
    assert_eq!(pixel(&dimmed, 32, 32), DARK.dimmed());

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
    assert_ne!(inside, DARK.background());
    assert_eq!(pixel(&frame, 7, 7), DARK.background());
}

#[test]
fn layer_names_parse() {
    assert_eq!("soft".parse::<Layer>().unwrap(), Layer::Soft);
    assert_eq!("C7".parse::<Layer>().unwrap(), Layer::Community(7));
    assert_eq!("c0".parse::<Layer>().unwrap(), Layer::Community(0));
    assert!("blue".parse::<Layer>().is_err());
}

/// Pixels not in the background colour.
fn painted(frame: &Frame) -> usize {
    (0..frame.h)
        .flat_map(|y| (0..frame.w).map(move |x| (x, y)))
        .filter(|&(x, y)| pixel(frame, x, y) != frame.background)
        .count()
}

#[test]
fn larger_points_cover_more_of_the_frame() {
    // A corner of the lattice, a few pixels per cell: points either way.
    let window = Rect {
        x0: 0.,
        y0: 0.,
        x1: 8.,
        y1: 8.,
    };
    let small = painted(&draw_styled(64, window, None, 0.5));
    let normal = painted(&draw_styled(64, window, None, 1.));
    let large = painted(&draw_styled(64, window, None, 2.));
    assert!(small < normal && normal < large, "{small} {normal} {large}");
}

#[test]
fn larger_points_are_drawn_from_further_out() {
    let (geom, comm, grid, pyramid) = fixture();
    let scene = Scene {
        geom: &geom,
        comm: &comm,
        grid: &grid,
        pyramid: &pyramid,
        edges: None,
        spacing: 1.,
    };
    // About 1.6 pixels per cell: averaged at the default size.
    let vp = Viewport::fit(geom.bounds(), 64, 64);
    assert_ne!(mode(&scene, &vp, 1.), Mode::Points);
    assert_ne!(mode(&scene, &vp, 0.5), Mode::Points);
    assert_eq!(mode(&scene, &vp, 2.), Mode::Points);
}

#[test]
fn a_gene_s_average_leaves_out_the_cells_without_a_level() {
    use crate::view::data::NO_CLUSTER;
    // Every other cell's file never measured the gene; the rest sit at the
    // top of its ramp. Averaged into pixels, the gene stays at its top.
    let geom = Geometry::from_cells(synth_cells(1600, 1));
    let mut comm = Communities::single("GENE1", vec![255; geom.n()]);
    for c in comm.cluster.iter_mut().skip(1).step_by(2) {
        *c = NO_CLUSTER;
    }
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let pyramid = Pyramid::build(&grid, &comm);
    let scene = Scene {
        geom: &geom,
        comm: &comm,
        grid: &grid,
        pyramid: &pyramid,
        edges: None,
        spacing: 1.,
    };
    let style = Style {
        layer: Layer::Gene,
        edges: None,
        focus: None,
        theme: DARK,
        point: 1.,
    };
    let window = Rect {
        x0: 0.,
        y0: 0.,
        x1: 40.,
        y1: 40.,
    };
    let vp = [24, 32, 48, 64, 96]
        .into_iter()
        .map(|w| Viewport::fit(window, w, w))
        .find(|vp| mode(&scene, vp, 1.) == Mode::Average)
        .expect("a size that averages");
    let frame = render(&scene, &vp, &style, &palette(1));
    let top = Layer::Gene.ramp(DARK).at_u8(255);
    let got = pixel(&frame, vp.w / 2, vp.h / 2);
    let off = (0..3).map(|c| got[c].abs_diff(top[c])).max().unwrap();
    assert!(off <= 2, "{got:?} is not the gene's top {top:?}");
}

/// The camera zooms as pinto's always did: a factor above one zooms out,
/// the world point under the anchor stays under it, and the scale stops
/// at its limits.
#[test]
fn the_camera_zooms_out_above_one_about_its_anchor_within_limits() {
    let vp = Viewport::at((10., 20.), 0.5, 200, 100);
    let mut cam = vp.camera();
    let anchor = (30., 70.);
    let before = (vp.x0 + anchor.0 * vp.upp, vp.y0 + anchor.1 * vp.upp);
    zoom(&mut cam, 2., Some(anchor), (0.01, 100.));
    let after = Viewport::of(&cam);
    assert!((after.upp - 1.).abs() < 1e-6, "{}", after.upp);
    let moved = (
        after.x0 + anchor.0 * after.upp,
        after.y0 + anchor.1 * after.upp,
    );
    assert!((moved.0 - before.0).abs() < 1e-4 && (moved.1 - before.1).abs() < 1e-4);
    // Without an anchor, about the centre.
    let mut cam = vp.camera();
    zoom(&mut cam, 0.5, None, (0.01, 100.));
    let c = Viewport::of(&cam).centre();
    assert!((c.0 - 10.).abs() < 1e-4 && (c.1 - 20.).abs() < 1e-4);
    // Held within the limits, in world units a pixel.
    let mut cam = vp.camera();
    zoom(&mut cam, 1000., None, (0.01, 4.));
    assert!((Viewport::of(&cam).upp - 4.).abs() < 1e-5);
    zoom(&mut cam, 1e-6, None, (0.01, 4.));
    assert!((Viewport::of(&cam).upp - 0.01).abs() < 1e-7);
}

/// A window and its camera are one view.
#[test]
fn a_window_and_its_camera_are_the_same_view() {
    let vp = Viewport::at((-3., 7.5), 0.25, 64, 48);
    let back = Viewport::of(&vp.camera());
    assert_eq!((back.w, back.h), (64, 48));
    for (a, b) in [(back.x0, vp.x0), (back.y0, vp.y0), (back.upp, vp.upp)] {
        assert!((a - b).abs() < 1e-5, "{a} {b}");
    }
}

/// Limits given the wrong way round (a batch with no extent, fitted to
/// nothing) still zoom, between them, rather than fail.
#[test]
fn limits_the_wrong_way_round_still_hold() {
    let mut cam = Viewport::at((0., 0.), 1., 10, 10).camera();
    zoom(&mut cam, 2., None, (4., 0.5));
    let upp = Viewport::of(&cam).upp;
    assert!((0.5..=4.).contains(&upp), "{upp}");
}

/// On a grouping, a link within one group is drawn in its colour and one
/// between two in the page's text colour; `edges_shown` drops either. By
/// link community every link is drawn.
#[test]
fn links_are_drawn_by_the_groups_of_their_ends() {
    let (geom, comm, grid, pyramid) = fixture();
    // Groups are x mod 4: a cell's neighbour below is in its group, the
    // one beside it is not.
    let i = 5 * 40 + 5;
    let (below, beside) = (i + 40, i + 1);
    let mut edges = Edges::default();
    for j in [below, beside] {
        edges.push(&geom, &geom.names[i], &geom.names[j], 0);
    }
    let index = EdgeIndex::build(&grid, &geom.x, &geom.y, &edges);
    let by_community = Links {
        edges: edges.into(),
        index: index.into(),
        ends: None,
        kinds: (0, 0),
    };
    let by_ends = by_community.by_ends(&comm.cluster);
    assert_eq!(by_ends.kinds, (1, 1));
    let vp = Viewport::fit(Rect::from_corners(4., 4., 7., 7.), 150, 150);
    let middle = |j: usize| {
        let (x, y) = vp.to_px(0.5 * (geom.x[i] + geom.x[j]), 0.5 * (geom.y[i] + geom.y[j]));
        (x as usize, y as usize)
    };
    let drawn = |links: &Links, shown| {
        let scene = Scene {
            geom: &geom,
            comm: &comm,
            grid: &grid,
            pyramid: &pyramid,
            edges: Some(links),
            spacing: 1.,
        };
        let style = Style {
            layer: Layer::Argmax,
            edges: Some(shown),
            focus: None,
            theme: DARK,
            point: 1.,
        };
        let frame = render(&scene, &vp, &style, &palette(4));
        let at = |(x, y)| pixel(&frame, x, y);
        (at(middle(below)), at(middle(beside)))
    };
    let background = DARK.background();
    let (within, between) = drawn(&by_ends, EdgesShown::All);
    let near = |a: [u8; 3], b: [u8; 3]| a.iter().zip(&b).all(|(&p, &q)| p.abs_diff(q) < 48);
    assert!(
        near(within, palette(4)[comm.cluster[i] as usize]),
        "{within:?}"
    );
    assert!(near(between, DARK.text()), "{between:?}");
    assert_eq!(drawn(&by_ends, EdgesShown::Heterotypic).0, background);
    assert_eq!(drawn(&by_ends, EdgesShown::Isotypic).1, background);
    let (within, between) = drawn(&by_community, EdgesShown::Heterotypic);
    assert_ne!(within, background);
    assert_ne!(between, background);
}

/// A link is in its ends' group when they share one, between groups when
/// they do not, and in none when an end has none; the counts take only
/// the links asked for.
#[test]
fn links_take_the_groups_of_their_ends() {
    let geom = Geometry::from_cells(synth_cells(4, 1));
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), 6.);
    let mut edges = Edges::default();
    for (l, r) in [("c0", "c1"), ("c1", "c2"), ("c2", "c3")] {
        edges.push(&geom, l, r, 7);
    }
    let index = EdgeIndex::build(&grid, &geom.x, &geom.y, &edges);
    let links = Links {
        edges: edges.into(),
        index: index.into(),
        ends: None,
        kinds: (0, 0),
    };
    assert_eq!(links.ends_count(|_| true), None);
    let by_ends = links.by_ends(&[2, 2, 5, NO_CLUSTER]);
    assert_eq!(by_ends.ends, Some(vec![2, HETEROTYPIC, NO_CLUSTER]));
    assert_eq!(by_ends.kinds, (1, 1));
    assert_eq!(by_ends.ends_count(|e| e > 0), Some((0, 1)));
}
