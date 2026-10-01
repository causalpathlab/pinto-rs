use crate::view::data::Rect;
use crate::view::render::{Frame, Viewport};
use crate::view::tui::grid::{blit, clip, columns, move_to, slots, Camera};

fn frame(w: usize, h: usize, c: [u8; 3]) -> Frame {
    Frame {
        background: [0; 3],
        ..Frame::blank(w, h, c)
    }
}

fn pixel(f: &Frame, x: usize, y: usize) -> [u8; 3] {
    let o = f.offset(x, y);
    [f.rgba[o], f.rgba[o + 1], f.rgba[o + 2]]
}

#[test]
fn square_batches_in_a_wide_area_go_side_by_side() {
    assert_eq!(columns(4, (400., 100.), 1., 1.), 4);
    assert_eq!(columns(4, (100., 400.), 1., 1.), 1);
    assert_eq!(columns(4, (200., 200.), 1., 1.), 2);
    // Shorter maps, to leave room for bars, favour more columns.
    assert!(columns(4, (200., 200.), 1., 0.5) >= 2);
}

#[test]
fn slots_stay_apart_and_inside_the_frame() {
    let (w, h) = (300, 200);
    for bars in [false, true] {
        let s = slots(5, (w, h), 3, bars);
        assert_eq!(s.len(), 5);
        for (i, a) in s.iter().enumerate() {
            assert!(a.tile.x + a.tile.w <= w && a.tile.y + a.tile.h <= h);
            assert!(a.map.w > 0 && a.map.h > 0);
            assert_eq!(a.bars.h > 0, bars);
            // The map and its bars sit inside the tile, the bars below.
            assert!(a.map.y + a.map.h <= a.tile.y + a.tile.h);
            if bars {
                assert!(a.bars.y >= a.map.y + a.map.h);
                assert!(a.bars.y + a.bars.h <= a.tile.y + a.tile.h);
            }
            for (j, b) in s.iter().enumerate() {
                if i != j {
                    let apart = a.tile.x + a.tile.w <= b.tile.x
                        || b.tile.x + b.tile.w <= a.tile.x
                        || a.tile.y + a.tile.h <= b.tile.y
                        || b.tile.y + b.tile.h <= a.tile.y;
                    assert!(apart, "{a:?} overlaps {b:?}");
                }
            }
        }
    }
}

#[test]
fn clip_clears_what_lies_outside_the_batch() {
    let mut f = frame(10, 10, [9; 3]);
    let vp = Viewport::fit(
        Rect {
            x0: 0.,
            y0: 0.,
            x1: 10.,
            y1: 10.,
        },
        10,
        10,
    );
    let keep = Rect {
        x0: 2.,
        y0: 3.,
        x1: 6.,
        y1: 7.,
    };
    clip(&mut f, &vp, keep);
    assert_eq!(pixel(&f, 3, 4), [9; 3]);
    assert_eq!(pixel(&f, 1, 4), [0; 3]);
    assert_eq!(pixel(&f, 3, 8), [0; 3]);
    assert_eq!(pixel(&f, 6, 4), [0; 3]);
}

#[test]
fn blit_copies_into_place_and_stops_at_the_edge() {
    let mut dst = frame(8, 8, [0; 3]);
    let src = frame(4, 4, [7; 3]);
    blit(&mut dst, &src, (6, 2));
    assert_eq!(pixel(&dst, 6, 2), [7; 3]);
    assert_eq!(pixel(&dst, 7, 5), [7; 3]);
    assert_eq!(pixel(&dst, 5, 2), [0; 3]);
    assert_eq!(pixel(&dst, 6, 6), [0; 3]);
}

#[test]
fn a_moved_batch_takes_its_new_place_and_the_rest_shift_over() {
    let mut order = vec![0, 1, 2, 3];
    move_to(&mut order, 0, 2);
    assert_eq!(order, [1, 2, 0, 3]);
    move_to(&mut order, 3, 0);
    assert_eq!(order, [3, 1, 2, 0]);
    // Out of range or in place: nothing moves.
    move_to(&mut order, 1, 9);
    move_to(&mut order, 2, 2);
    assert_eq!(order, [3, 1, 2, 0]);
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
    Rect { x0, y0, x1, y1 }
}

#[test]
fn a_fitted_view_is_the_camera_at_rest() {
    let frame = rect(10., 20., 110., 70.);
    let c = Camera::of(&Viewport::fit(frame, 200, 100), frame);
    assert!((c.zoom - 1.).abs() < 1e-5, "{c:?}");
    assert!((c.at.0 - 0.5).abs() < 1e-5 && (c.at.1 - 0.5).abs() < 1e-5);
}

#[test]
fn the_camera_carries_zoom_and_pan_to_another_batch() {
    let a = rect(0., 0., 100., 100.);
    // Zoomed in 4× on the first batch's upper left quarter.
    let fit = Viewport::fit(a, 80, 80);
    let upp = fit.upp / 4.;
    let vp = Viewport {
        x0: 25. - 40. * upp,
        y0: 25. - 40. * upp,
        upp,
        w: 80,
        h: 80,
    };
    let c = Camera::of(&vp, a);
    assert!((c.zoom - 0.25).abs() < 1e-5);
    // The same corner of a batch twice the size, at the same zoom.
    let b = rect(500., 0., 700., 200.);
    let v = c.view(b, 80, 80);
    assert!((v.upp - Viewport::fit(b, 80, 80).upp / 4.).abs() < 1e-4);
    let cx = v.x0 + 40. * v.upp;
    let cy = v.y0 + 40. * v.upp;
    assert!(
        (cx - 550.).abs() < 1e-2 && (cy - 50.).abs() < 1e-2,
        "{cx} {cy}"
    );
}
