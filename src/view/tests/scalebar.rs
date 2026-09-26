use crate::view::color::Theme;
use crate::view::render::{Frame, Viewport};
use crate::view::scalebar::*;

#[test]
fn bars_are_round_and_at_most_a_fifth_of_the_view() {
    let b = bar_for(11_000., Units::Micron);
    assert_eq!((b.length, b.label.as_str()), (2000., "2 mm"));
    let b = bar_for(600., Units::Micron);
    assert_eq!((b.length, b.label.as_str()), (100., "100 µm"));
    let b = bar_for(260., Units::Pixel);
    assert_eq!((b.length, b.label.as_str()), (50., "50 px"));
    let b = bar_for(3., Units::Plain);
    assert_eq!((b.length, b.label.as_str()), (0.5, "0.5"));
}

#[test]
fn units_are_guessed_from_coordinate_columns() {
    let names = |v: &[&str]| v.iter().map(|&s| Box::from(s)).collect::<Vec<Box<str>>>();
    let guess = |v: &[&str]| Units::parse("auto", &names(v)).unwrap();
    assert_eq!(
        guess(&["pxl_row_in_fullres", "pxl_col_in_fullres"]),
        Some(Units::Pixel)
    );
    assert_eq!(
        guess(&["cell_centroid_x", "cell_centroid_y"]),
        Some(Units::Micron)
    );
    assert_eq!(guess(&["x", "y"]), Some(Units::Plain));
    // Whole words only: `um` inside another word is not a unit.
    assert_eq!(guess(&["column_x", "num_y"]), Some(Units::Plain));
    assert_eq!(guess(&["x_um", "y_um"]), Some(Units::Micron));
    assert_eq!(Units::parse("none", &[]).unwrap(), None);
}

#[test]
fn every_label_character_has_a_glyph() {
    for label in ["0123456789", "µm mm px", "0.5"] {
        for c in label.chars() {
            assert!(glyph(c).is_some(), "{c:?}");
        }
    }
}

#[test]
fn the_bar_is_white_at_the_bottom_left() {
    let mut frame = Frame {
        w: 400,
        h: 300,
        rgba: vec![0; 400 * 300 * 4],
        background: Theme::Dark.background(),
    };
    let vp = Viewport {
        x0: 0.,
        y0: 0.,
        upp: 1.,
        w: 400,
        h: 300,
    };
    draw(&mut frame, &vp, Units::Micron, true);
    // 400 units wide → an 50-unit bar: 50 px from x = 6 on a row near the bottom.
    let px = |x: usize, y: usize| &frame.rgba[(y * 400 + x) * 4..(y * 400 + x) * 4 + 3];
    assert_eq!(px(30, 300 - 6 - 1), &[255, 255, 255]);
    assert_eq!(px(200, 300 - 6 - 1), &[0, 0, 0]);
}

#[test]
fn without_a_label_only_the_bar_is_drawn() {
    let blank = |label| {
        let mut frame = Frame {
            w: 400,
            h: 300,
            rgba: vec![0; 400 * 300 * 4],
            background: Theme::Dark.background(),
        };
        let vp = Viewport {
            x0: 0.,
            y0: 0.,
            upp: 1.,
            w: 400,
            h: 300,
        };
        draw(&mut frame, &vp, Units::Micron, label);
        // Anything lit above the bar's outline is label text.
        frame.rgba[..(300 - 6 - 2 - 2) * 400 * 4]
            .iter()
            .all(|&v| v == 0)
    };
    assert!(blank(false));
    assert!(!blank(true));
}

#[test]
fn on_a_light_frame_the_bar_is_black() {
    let mut frame = Frame {
        w: 400,
        h: 300,
        rgba: vec![250; 400 * 300 * 4],
        background: Theme::Light.background(),
    };
    let vp = Viewport {
        x0: 0.,
        y0: 0.,
        upp: 1.,
        w: 400,
        h: 300,
    };
    draw(&mut frame, &vp, Units::Micron, false);
    let o = ((300 - 6 - 1) * 400 + 30) * 4;
    assert_eq!(&frame.rgba[o..o + 3], &[0, 0, 0]);
}
