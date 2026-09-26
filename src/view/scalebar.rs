//! Scale bar: a round length in world units, drawn white at the bottom left.
//!
//! Raster frames get the bar and its label burnt in with a small built-in
//! pixel font, so a PNG or a terminal frame carries its own scale. The PDF
//! draws the same bar as vector shapes instead.

use super::render::{Frame, Viewport};

/// What one world unit is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Units {
    Micron,
    Pixel,
    /// Unknown: the bar is labelled with a bare number.
    Plain,
}

impl Units {
    /// `auto` guesses from the coordinate columns pinto was fit with:
    /// `pxl_*` are Space Ranger image pixels, Xenium centroids are µm.
    /// `None` means no bar.
    pub fn parse(choice: &str, coord_names: &[Box<str>]) -> anyhow::Result<Option<Units>> {
        Ok(match choice {
            "um" | "µm" | "micron" => Some(Units::Micron),
            "px" | "pixel" => Some(Units::Pixel),
            "none" => None,
            "auto" => {
                let any = |pat: &str| coord_names.iter().any(|c| c.contains(pat));
                Some(if any("pxl") || any("pixel") {
                    Units::Pixel
                } else if any("centroid") || any("um") || any("micron") {
                    Units::Micron
                } else {
                    Units::Plain
                })
            }
            other => anyhow::bail!("--units {other:?}: use auto, um, px or none"),
        })
    }
}

/// A bar of `length` world units and its label.
#[derive(Clone, Debug, PartialEq)]
pub struct Bar {
    pub length: f32,
    pub label: String,
}

/// The largest 1, 2 or 5 × 10ⁿ no longer than a fifth of `view_width`.
pub fn bar_for(view_width: f32, units: Units) -> Bar {
    let target = (view_width / 5.).max(f32::MIN_POSITIVE);
    let mag = 10f32.powf(target.log10().floor());
    let length = [5., 2., 1.]
        .iter()
        .map(|m| m * mag)
        .find(|&l| l <= target)
        .unwrap_or(mag);
    let number = |v: f32| {
        if v >= 1. {
            format!("{}", v.round() as i64)
        } else {
            format!("{v}")
        }
    };
    let label = match units {
        Units::Micron if length >= 1000. => format!("{} mm", number(length / 1000.)),
        Units::Micron => format!("{} µm", number(length)),
        Units::Pixel => format!("{} px", number(length)),
        Units::Plain => number(length),
    };
    Bar { length, label }
}

/// Burn the bar into `frame` at its bottom left.
pub fn draw(frame: &mut Frame, vp: &Viewport, units: Units) {
    let bar = bar_for(vp.w as f32 * vp.upp, units);
    let len_px = (bar.length / vp.upp).round() as i64;
    // Glyph pixel size and bar thickness grow with the frame.
    let s = ((frame.h.min(frame.w) as f32 / 350.).round() as i64).max(1);
    let margin = 6 * s;
    let thick = 2 * s;
    let y_bar = frame.h as i64 - margin - thick;
    let y_text = y_bar - 3 * s - GLYPH_H * s;
    if len_px < 2 || y_text < 0 {
        return;
    }

    // A dark outline first, then white, so the bar reads over bright cells.
    for (pad, colour) in [(s.max(1), [0u8, 0, 0]), (0, [255, 255, 255])] {
        fill(
            frame,
            margin - pad,
            y_bar - pad,
            len_px + 2 * pad,
            thick + 2 * pad,
            colour,
        );
        let mut x = margin;
        for ch in bar.label.chars() {
            if let Some(rows) = glyph(ch) {
                for (gy, row) in rows.iter().enumerate() {
                    for (gx, bit) in row.bytes().enumerate() {
                        if bit == b'#' {
                            let (px, py) = (x + gx as i64 * s, y_text + gy as i64 * s);
                            fill(frame, px - pad, py - pad, s + 2 * pad, s + 2 * pad, colour);
                        }
                    }
                }
            }
            x += (GLYPH_W + 1) * s;
        }
    }
}

fn fill(frame: &mut Frame, x: i64, y: i64, w: i64, h: i64, c: [u8; 3]) {
    let (x0, y0) = (x.max(0), y.max(0));
    let (x1, y1) = ((x + w).min(frame.w as i64), (y + h).min(frame.h as i64));
    for yy in y0..y1 {
        for xx in x0..x1 {
            let o = (yy as usize * frame.w + xx as usize) * 4;
            frame.rgba[o..o + 3].copy_from_slice(&c);
        }
    }
}

const GLYPH_W: i64 = 5;
const GLYPH_H: i64 = 7;

/// 5×7 glyphs for the characters a label can hold.
fn glyph(c: char) -> Option<[&'static str; 7]> {
    Some(match c {
        '0' => [
            " ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### ",
        ],
        '1' => [
            "  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### ",
        ],
        '2' => [
            " ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####",
        ],
        '3' => [
            "#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### ",
        ],
        '4' => [
            "   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # ",
        ],
        '5' => [
            "#####", "#    ", "#### ", "    #", "    #", "#   #", " ### ",
        ],
        '6' => [
            "  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### ",
        ],
        '7' => [
            "#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   ",
        ],
        '8' => [
            " ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### ",
        ],
        '9' => [
            " ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  ",
        ],
        '.' => [
            "     ", "     ", "     ", "     ", "     ", " ##  ", " ##  ",
        ],
        'µ' => [
            "     ", "     ", "#   #", "#   #", "#   #", "#### ", "#    ",
        ],
        'm' => [
            "     ", "     ", "## # ", "# # #", "# # #", "# # #", "# # #",
        ],
        'p' => [
            "     ", "     ", "#### ", "#   #", "#### ", "#    ", "#    ",
        ],
        'x' => [
            "     ", "     ", "#   #", " # # ", "  #  ", " # # ", "#   #",
        ],
        ' ' => ["     "; 7],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        };
        let vp = Viewport {
            x0: 0.,
            y0: 0.,
            upp: 1.,
            w: 400,
            h: 300,
        };
        draw(&mut frame, &vp, Units::Micron);
        // 400 units wide → an 50-unit bar: 50 px from x = 6 on a row near the bottom.
        let px = |x: usize, y: usize| &frame.rgba[(y * 400 + x) * 4..(y * 400 + x) * 4 + 3];
        assert_eq!(px(30, 300 - 6 - 1), &[255, 255, 255]);
        assert_eq!(px(200, 300 - 6 - 1), &[0, 0, 0]);
    }
}
