use super::fixture::{synth_cells, synth_communities};
use crate::view::color::{palette, BACKGROUND, DIMMED};
use crate::view::data::{Communities, Geometry, Rect};
use crate::view::index::{Grid, Pyramid};
use crate::view::render::{render, Frame, Layer, Scene, Style, Viewport};

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
    assert_eq!(pixel(&dimmed, 32, 32), DIMMED);

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
