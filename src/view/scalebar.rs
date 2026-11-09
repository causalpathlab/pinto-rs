//! Scale bar: a round length in world units, drawn white at the bottom left.
//!
//! Raster frames get the bar burnt in. An exported PNG also gets its label,
//! in a small built-in pixel font, so the file carries its own scale; on
//! screen the panel states the length as text instead. The PDF draws the
//! same bar as vector shapes.

use super::color::Theme;
use super::render::{Frame, Viewport};
use std::path::{Path, PathBuf};

/// What one world unit is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Units {
    Micron,
    Pixel,
    /// Image pixels of a known size, µm each: the bar reads in µm.
    PixelMicron(f32),
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
                // Whole words of the column names (`pxl_row_in_fullres` →
                // pxl, row, in, fullres), so `um` never matches `column`.
                let any = |words: &[&str]| {
                    coord_names.iter().any(|c| {
                        c.split(|ch: char| !ch.is_ascii_alphanumeric())
                            .any(|w| words.contains(&w.to_ascii_lowercase().as_str()))
                    })
                };
                Some(if any(&["pxl", "px", "pixel", "pixels"]) {
                    Units::Pixel
                } else if any(&["centroid", "um", "µm", "micron", "microns"]) {
                    Units::Micron
                } else {
                    Units::Plain
                })
            }
            other => anyhow::bail!("--units {other:?}: use auto, um, px or none"),
        })
    }

    /// [`Units::parse`] for coordinates read from `coord_file`: `auto`
    /// reads image pixels in µm when Space Ranger's scale factors beside
    /// that file give their size ([`scalefactors_beside`],
    /// [`microns_per_pixel`], `pitch_px` the coordinates' pitch); without
    /// that file (`None`: the run's coordinates came from several files, or
    /// it is not found) they stay in pixels.
    pub fn for_coordinates(
        choice: &str,
        coord_names: &[Box<str>],
        coord_file: Option<&Path>,
        pitch_px: impl FnOnce() -> Option<f32>,
    ) -> anyhow::Result<Option<Units>> {
        let units = Units::parse(choice, coord_names)?;
        if choice != "auto" || units != Some(Units::Pixel) {
            return Ok(units);
        }
        let Some(factors) = coord_file.and_then(scalefactors_beside) else {
            return Ok(units);
        };
        let size = std::fs::read_to_string(&factors)
            .ok()
            .and_then(|text| microns_per_pixel(&text, pitch_px));
        if let Some(um) = size {
            log::info!("scale bar: {um:.4} µm a pixel, from {}", factors.display());
        }
        Ok(Some(size.map_or(Units::Pixel, Units::PixelMicron)))
    }

    /// World units in these units' terms: µm per world unit and µm for
    /// pixels of a known size, else one and the units themselves.
    fn resolved(self) -> (f32, Units) {
        match self {
            Units::PixelMicron(um) => (um, Units::Micron),
            other => (1., other),
        }
    }
}

/// A view's scale, `upp` world units a pixel, in µm where `units` know
/// their size, and in image pixels for those; its digits kept whatever the
/// zoom.
pub fn per_pixel(units: Option<Units>, upp: f32) -> String {
    let (per_unit, units) = units.map_or((1., None), |u| {
        let (k, base) = u.resolved();
        (k, Some(base))
    });
    let v = upp * per_unit;
    let unit = match units {
        Some(Units::Micron) => "µm/px",
        Some(Units::Pixel) => "px/px",
        _ => "units/px",
    };
    if v != 0. && v.abs() < 0.001 {
        format!("{v:.2e} {unit}")
    } else {
        format!("{v:.3} {unit}")
    }
}

/// Centre-to-centre distance of standard Visium spots, µm.
const VISIUM_PITCH_UM: f32 = 100.;

/// The size of a full-resolution image pixel, µm, from Space Ranger's
/// `scalefactors_json.json`: its `microns_per_pixel` (Visium HD); else, for
/// standard Visium (a spot lattice, no `bin_size_um`), the spots' 100 µm
/// pitch over `pitch_px`, the coordinates' pitch in pixels. The file's spot
/// diameter is not used: it is not the physical spot's.
pub fn microns_per_pixel(
    scalefactors: &str,
    pitch_px: impl FnOnce() -> Option<f32>,
) -> Option<f32> {
    let v: serde_json::Value = serde_json::from_str(scalefactors).ok()?;
    if let Some(um) = v
        .get("microns_per_pixel")
        .and_then(serde_json::Value::as_f64)
    {
        return (um.is_finite() && um > 0.).then_some(um as f32);
    }
    if v.get("spot_diameter_fullres").is_none() || v.get("bin_size_um").is_some() {
        return None;
    }
    pitch_px()
        .filter(|p| p.is_finite() && *p > 0.)
        .map(|p| VISIUM_PITCH_UM / p)
}

/// Space Ranger's `scalefactors_json.json` for the coordinates in
/// `coord_file`: beside it, or in `spatial/` beside it.
pub fn scalefactors_beside(coord_file: &Path) -> Option<PathBuf> {
    let dir = coord_file.parent()?;
    [dir.to_path_buf(), dir.join("spatial")]
        .into_iter()
        .map(|d| d.join("scalefactors_json.json"))
        .find(|f| f.is_file())
}

/// A bar of `length` world units and its label.
#[derive(Clone, Debug, PartialEq)]
pub struct Bar {
    pub length: f32,
    pub label: String,
}

/// The largest 1, 2 or 5 × 10ⁿ no longer than a fifth of `view_width`; for
/// pixels of a known size, round in µm, its length in pixels.
pub fn bar_for(view_width: f32, units: Units) -> Bar {
    let (per_unit, units) = units.resolved();
    let target = (view_width * per_unit / 5.).max(f32::MIN_POSITIVE);
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
        Units::PixelMicron(_) => unreachable!("resolved to µm above"),
        Units::Pixel => format!("{} px", number(length)),
        Units::Plain => number(length),
    };
    Bar {
        length: length / per_unit,
        label,
    }
}

/// Burn the bar into `frame` at its bottom left, with its label above it
/// when `label` is set.
pub fn draw(frame: &mut Frame, vp: &Viewport, units: Units, label: bool) {
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
    let text = if label { bar.label.as_str() } else { "" };

    // The outline first, then the fill, so the bar reads over any cells.
    let (fill_colour, edge) = Theme::for_background(frame.background).bar();
    for (pad, colour) in [(s, edge), (0, fill_colour)] {
        fill(
            frame,
            margin - pad,
            y_bar - pad,
            len_px + 2 * pad,
            thick + 2 * pad,
            colour,
        );
        write(frame, (margin, y_text), s, pad, colour, text);
    }
}

/// Burn `text` into `frame` with its top left at pixel `(x, y)`, each
/// glyph pixel `s` frame pixels, outlined so it reads over any cells.
/// Characters without a glyph are left as gaps.
pub fn label(frame: &mut Frame, (x, y): (i64, i64), s: i64, text: &str) {
    let (fill_colour, edge) = Theme::for_background(frame.background).bar();
    for (pad, colour) in [(s, edge), (0, fill_colour)] {
        write(frame, (x, y), s, pad, colour, text);
    }
}

/// One pass of `text`: each glyph pixel grown by `pad` on every side.
fn write(frame: &mut Frame, (x0, y): (i64, i64), s: i64, pad: i64, colour: [u8; 3], text: &str) {
    let mut x = x0;
    for ch in text.chars() {
        if let Some(rows) = glyph(ch) {
            for (gy, row) in rows.iter().enumerate() {
                for (gx, bit) in row.bytes().enumerate() {
                    if bit == b'#' {
                        let (px, py) = (x + gx as i64 * s, y + gy as i64 * s);
                        fill(frame, px - pad, py - pad, s + 2 * pad, s + 2 * pad, colour);
                    }
                }
            }
        }
        x += (GLYPH_W + 1) * s;
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
pub(super) fn glyph(c: char) -> Option<[&'static str; 7]> {
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
