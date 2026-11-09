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

/// Pixels of a known size read in µm: the bar's length stays in pixels
/// (the world unit), its label is a round length in µm.
#[test]
fn calibrated_pixels_get_a_bar_in_micrometres() {
    // 4000 px at 0.25 µm each is 1000 µm wide: a 200 µm bar, 800 px long.
    let b = bar_for(4000., Units::PixelMicron(0.25));
    assert_eq!((b.length, b.label.as_str()), (800., "200 µm"));
    let b = bar_for(40_000., Units::PixelMicron(0.25));
    assert_eq!((b.length, b.label.as_str()), (8000., "2 mm"));
}

/// A Visium HD file records the pixel size; a standard Visium one does
/// not, and its spot diameter is no reliable measure, so the size comes
/// from the spots' 100 µm pitch over the coordinates' own pitch in pixels.
#[test]
fn space_ranger_scale_factors_give_the_pixel_size() {
    let pitch = |px: f32| move || Some(px);
    let hd = r#"{"microns_per_pixel": 0.2738, "spot_diameter_fullres": 7.3, "bin_size_um": 2.0}"#;
    assert_eq!(microns_per_pixel(hd, pitch(1.0)), Some(0.2738));
    let standard = r#"{"spot_diameter_fullres": 219.2, "tissue_hires_scalef": 0.03}"#;
    assert_eq!(microns_per_pixel(standard, pitch(400.0)), Some(0.25));
    assert_eq!(microns_per_pixel(standard, || None), None);
    // Binned HD output without the size: not a spot lattice, so none.
    let binned = r#"{"spot_diameter_fullres": 29.0, "bin_size_um": 8.0}"#;
    assert_eq!(microns_per_pixel(binned, pitch(29.0)), None);
    assert_eq!(
        microns_per_pixel(r#"{"tissue_hires_scalef": 0.03}"#, pitch(400.0)),
        None
    );
    assert_eq!(
        microns_per_pixel(r#"{"microns_per_pixel": 0}"#, pitch(400.0)),
        None
    );
    assert_eq!(microns_per_pixel("not json", pitch(400.0)), None);
}

/// `auto` reads pixel coordinates in µm when Space Ranger's scale factors
/// sit beside the coordinate file (or in `spatial/` beside it); `px` keeps
/// pixels, as does a run without one coordinate file, and coordinates in
/// other units stay as they are.
#[test]
fn auto_calibrates_pixels_from_the_scale_factors_beside_the_coordinates() {
    let dir = tempfile::tempdir().unwrap();
    let spatial = dir.path().join("spatial");
    std::fs::create_dir(&spatial).unwrap();
    let coords = spatial.join("tissue_positions.csv");
    let top = dir.path().join("positions.csv");
    std::fs::write(&coords, "").unwrap();
    std::fs::write(&top, "").unwrap();
    let pxl: Vec<Box<str>> = vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
    let units = |choice: &str, file: &std::path::Path| {
        Units::for_coordinates(choice, &pxl, Some(file), || Some(400.0)).unwrap()
    };
    let (coords, top) = (coords.as_path(), top.as_path());
    assert_eq!(
        units("auto", coords),
        Some(Units::Pixel),
        "no scale factors yet"
    );
    std::fs::write(
        spatial.join("scalefactors_json.json"),
        r#"{"spot_diameter_fullres": 220.0}"#,
    )
    .unwrap();
    assert_eq!(units("auto", coords), Some(Units::PixelMicron(0.25)));
    assert_eq!(units("auto", top), Some(Units::PixelMicron(0.25)));
    assert_eq!(units("px", coords), Some(Units::Pixel));
    assert_eq!(units("none", coords), None);
    assert_eq!(
        Units::for_coordinates("auto", &pxl, None, || Some(400.0)).unwrap(),
        Some(Units::Pixel),
        "no one coordinate file"
    );
    let xy: Vec<Box<str>> = vec!["x".into(), "y".into()];
    assert_eq!(
        Units::for_coordinates("auto", &xy, Some(coords), || Some(400.0)).unwrap(),
        Some(Units::Plain)
    );
}

/// The view's scale reads in µm where the units are known to be.
#[test]
fn the_scale_per_pixel_reads_in_micrometres_when_it_can() {
    assert_eq!(
        per_pixel(Some(Units::PixelMicron(0.25)), 4.0),
        "1.000 µm/px"
    );
    assert_eq!(per_pixel(Some(Units::Micron), 2.5), "2.500 µm/px");
    assert_eq!(per_pixel(Some(Units::Pixel), 2.0), "2.000 px/px");
    assert_eq!(per_pixel(Some(Units::Plain), 2.0), "2.000 units/px");
    assert_eq!(per_pixel(None, 2.0), "2.000 units/px");
    // Zoomed far in, the scale keeps its digits.
    assert_eq!(per_pixel(Some(Units::Micron), 0.0004), "4.00e-4 µm/px");
}

/// A run's one coordinate file, found beside its manifest; none when it
/// was fit with several (one image each) or the file is gone.
#[test]
fn a_run_has_one_coordinate_file_or_none() {
    use crate::view::data::single_coord_file;
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("run.pinto.json");
    std::fs::write(dir.path().join("a.csv"), "").unwrap();
    std::fs::write(dir.path().join("b.csv"), "").unwrap();
    let one = |f: &str| single_coord_file(Some(f), Some(&manifest));
    assert_eq!(one("a.csv"), Some(dir.path().join("a.csv")));
    assert_eq!(one("a.csv,b.csv"), None);
    assert_eq!(one("gone.csv"), None);
    assert_eq!(single_coord_file(None, Some(&manifest)), None);
}
