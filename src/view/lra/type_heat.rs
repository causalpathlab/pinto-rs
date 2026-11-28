//! A pair's links by cell type as a heatmap, drawn by legume-tui's chart:
//! a row per receiver type, a column per sender type, each cell shaded by
//! the share of responders' exposure that sender type gives that receiver
//! type, relative to the largest; blank where no link joins them. Each
//! function takes the pair drawn and gives `None` when it has no cell types.

use super::OnMap;
use crate::lr_activity::ligand_effect::link_cell_types::CellTypeLinks;
use crate::view::color::{to_linear, Theme};
use crate::view::curve;
use crate::view::pdf::{chart_fits, chart_page, CHART_PAGE};
use crate::view::render::Frame;
use legume_tui::chart::heatmap::{self as chart, Heat, HeatStyle, CLIP};
use legume_tui::export::Page;
use legume_tui::map::raster::Frame as Raster;
use legume_tui::map::text::Font;

/// The heatmap in `w × h` pixels, its text drawn for lines `cell_px` tall;
/// with too little room, a note says so where it would be.
pub fn render(on: &OnMap, (w, h): (usize, usize), cell_px: f32, theme: Theme) -> Option<Frame> {
    let grid = Grid::of(on)?;
    Some(grid.raster(Ok((w, h)), cell_px, theme))
}

/// The heatmap as a still at least `w` wide, as tall as it needs, its text
/// sized as [`curve::still`] sizes a curve's.
pub fn still(on: &OnMap, w: usize, theme: Theme) -> Option<Frame> {
    let grid = Grid::of(on)?;
    let cell = (curve::still_height(w) as f32 / 28.).clamp(10., 64.);
    Some(grid.raster(Err(w), cell, theme))
}

/// The heatmap as a PDF page, its text kept as text; too much for a page,
/// why not.
pub fn page(on: &OnMap, theme: Theme) -> Option<Result<Page, &'static str>> {
    let grid = Grid::of(on)?;
    let (heat, style) = (grid.heat(), grid_style(theme));
    let (font, bold) = CHART_PAGE.fonts();
    let (w, h) = chart::heat_size(&heat, font, &style, CHART_PAGE.px(1.).0);
    if chart_fits(w, h).is_err() {
        return Some(Err("too many cell types for a page"));
    }
    let (page, drawn) = chart_page((w, h), theme, |frame| {
        chart::draw_heat(frame, &heat, font, bold, theme.page(), &style)
    });
    Some(drawn.map(|_| page))
}

/// What the chart draws: the title, the receiver types (rows) and sender
/// types (columns) with a link, and each cell's shade.
struct Grid {
    title: String,
    rows: Vec<String>,
    cols: Vec<String>,
    /// In [0, [`CLIP`]], NaN without a link, row-major.
    z: Vec<f32>,
}

impl Grid {
    fn of(on: &OnMap) -> Option<Grid> {
        let t: &CellTypeLinks = on.types.as_ref()?;
        let mut rows: Vec<u32> = t.cells.iter().map(|c| c.receiver_type).collect();
        let mut cols: Vec<u32> = t.cells.iter().map(|c| c.sender_type).collect();
        for v in [&mut rows, &mut cols] {
            v.sort_unstable();
            v.dedup();
        }
        let top = t.cells.iter().map(|c| c.share).fold(0., f64::max);
        let mut z = vec![f32::NAN; rows.len() * cols.len()];
        for c in &t.cells {
            let r = rows.binary_search(&c.receiver_type).expect("listed");
            let s = cols.binary_search(&c.sender_type).expect("listed");
            z[r * cols.len() + s] = if top > 0. {
                (c.share / top) as f32 * CLIP
            } else {
                0.
            };
        }
        let name = |&code: &u32| t.names[code as usize].to_string();
        Some(Grid {
            title: format!(
                "{}:{} · share of responders' exposure, receiver type (rows) by sender \
                 type · darkest the largest",
                on.ligand, on.receptor
            ),
            rows: rows.iter().map(name).collect(),
            cols: cols.iter().map(name).collect(),
            z,
        })
    }

    fn heat(&self) -> Heat<'_> {
        Heat {
            title: &self.title,
            rows: &self.rows,
            cols: &self.cols,
            col_swatch: None,
            z: &self.z,
            clip: CLIP,
            bold_row: None,
        }
    }

    /// The chart on pixels: `size`, or as tall as it needs at the least width.
    fn raster(&self, size: Result<(usize, usize), usize>, cell_px: f32, theme: Theme) -> Frame {
        let (heat, style) = (self.heat(), grid_style(theme));
        let (font, bold): (Font, Font) = legume_tui::export::fonts(cell_px.max(6.));
        let (w, h) = size.unwrap_or_else(|min_w| chart::heat_size(&heat, font, &style, min_w));
        let bg = theme.background();
        let mut frame = Raster::blank_exact(w.max(1), h.max(1), bg);
        // too little room: the chart notes it in place
        let _ = chart::draw_heat(&mut frame, &heat, font, bold, theme.page(), &style);
        Frame::from_raster(&frame, bg)
    }
}

/// The chart's look: no key, as the shade is a share of the largest, not a
/// z-score (the title says so).
fn grid_style(theme: Theme) -> HeatStyle {
    HeatStyle {
        missing: to_linear(theme.dimmed()),
        key: false,
        ..HeatStyle::for_theme(theme.page())
    }
}
