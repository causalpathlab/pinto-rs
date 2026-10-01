//! The strip of saved figures on the left of the map: each export's
//! thumbnail, file name, and how long ago, newest first (`f` hides it).
//! Thumbnails are drawn in block characters, so they show in any terminal
//! beside a map in any graphics protocol.

use super::super::cellart::{Cells, Glyphs};
use super::super::render::Frame;
use super::super::saved::{ago, read_thumb};
use super::App;
use crate::tui::style;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use std::path::PathBuf;

/// Columns of the strip.
pub const SAVED_WIDTH: u16 = 26;

/// Thumbnails drawn, by file and width in cells: their height in cells
/// and their block characters.
pub type Thumbs = std::collections::HashMap<(PathBuf, u16), (u16, Cells)>;

impl App<'_> {
    /// Whether the strip shows beside a main area `width` columns wide.
    pub(super) fn strip_shows(&self, width: u16) -> bool {
        self.show_saved && !self.gallery.entries().is_empty() && width >= 3 * SAVED_WIDTH
    }

    /// `f`: show or hide the saved figures.
    pub(super) fn toggle_saved(&mut self) {
        self.show_saved = !self.show_saved;
        self.status = if self.gallery.entries().is_empty() {
            "nothing saved here yet: s saves the view".into()
        } else if self.show_saved {
            "saved figures on the left, kept in .pinto-view/".into()
        } else {
            "saved figures hidden (f)".into()
        };
        self.need_map = true;
    }

    /// Log a file just saved, with `picture` as its thumbnail.
    pub(super) fn remember(&mut self, path: &str, what: &str, picture: &Frame) -> String {
        match self.gallery.add(std::path::Path::new(path), what, picture) {
            Ok(()) => {
                self.need_map = true;
                String::new()
            }
            Err(e) => format!(" (not listed: {e})"),
        }
    }

    /// Rows a thumbnail of `w × h` pixels takes, `cols` wide.
    fn thumb_rows(&self, (w, h): (usize, usize), cols: u16) -> u16 {
        // Terminal cells are about twice as tall as wide.
        let aspect = self.px_per_cell.1 / self.px_per_cell.0.max(1.);
        let rows = cols as f32 * h as f32 / (w.max(1) as f32 * aspect.max(0.5));
        rows.round().clamp(2., 12.) as u16
    }

    /// Build the block-character thumbnails the strip shows, each read
    /// from disk once; those it no longer shows are dropped.
    pub(super) fn prepare_thumbs(&mut self) {
        if self.strip.width == 0 {
            self.thumbs.clear();
            return;
        }
        let cols = self.strip.width.saturating_sub(2);
        let mut room = self.strip.height.saturating_sub(1);
        let bg = self.base.theme.background();
        let paths: Vec<PathBuf> = self
            .gallery
            .entries()
            .iter()
            .map(|e| self.gallery.thumb(e))
            .collect();
        let mut shown = std::collections::HashSet::new();
        for path in paths {
            let key = (path, cols);
            if !self.thumbs.contains_key(&key) {
                let Some(frame) = read_thumb(&key.0, bg) else {
                    continue;
                };
                let rows = self.thumb_rows((frame.w, frame.h), cols);
                let ppc = (
                    frame.w.div_ceil(cols as usize).max(1),
                    frame.h.div_ceil(rows as usize).max(1),
                );
                let cells = Cells::fit(
                    &frame,
                    Glyphs::Quadrants,
                    ppc,
                    (cols as usize, rows as usize),
                );
                self.thumbs.insert(key.clone(), (rows, cells));
            }
            let rows = self.thumbs[&key].0;
            if room < rows + 3 {
                break;
            }
            room -= rows + 3;
            shown.insert(key);
        }
        self.thumbs.retain(|k, _| shown.contains(k));
    }

    /// Draw the strip into `f`.
    pub(super) fn draw_saved(&self, f: &mut ratatui::Frame, strip: Rect) {
        let block = Block::new()
            .borders(Borders::RIGHT)
            .border_style(style::dim())
            .title(Span::styled(
                format!(" saved ({}) · f hides", self.gallery.entries().len()),
                style::bold(),
            ));
        let inner = block.inner(strip);
        f.render_widget(block, strip);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let dim = style::dim();
        let bold = style::bold();
        let cols = strip.width.saturating_sub(2);
        let mut y = inner.y + 1;
        for e in self.gallery.entries() {
            let Some((rows, cells)) = self.thumbs.get(&(self.gallery.thumb(e), cols)) else {
                continue;
            };
            if y + rows + 2 > inner.bottom() {
                break;
            }
            f.render_widget(cells, Rect::new(inner.x, y, cols, *rows));
            let name: String = e.name().chars().take(cols as usize).collect();
            let what = format!("{} · {}", ago(e.when, now), e.what);
            let what: String = what.chars().take(cols as usize).collect();
            f.render_widget(
                Paragraph::new(vec![Line::styled(name, bold), Line::styled(what, dim)]),
                Rect::new(inner.x, y + rows, inner.width, 2),
            );
            y += rows + 3;
        }
    }
}
