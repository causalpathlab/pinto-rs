//! How `pinto view` and `pinto run` look: legume-tui's chrome on the
//! terminal's own colours, so both read on a dark or a light background.

pub use legume_tui::chrome::{first_row, middle, short, tail, wrap};
use legume_tui::chrome::{At, Chrome};
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Line;

const CHROME: Chrome = Chrome::TERMINAL;

/// Bold text.
#[must_use]
pub fn bold() -> Style {
    CHROME.bold()
}

/// Key hints and other secondary lines.
#[must_use]
pub fn dim() -> Style {
    CHROME.hint()
}

/// The line under a menu's cursor.
#[must_use]
pub fn selected() -> Style {
    CHROME.selected()
}

/// A bordered popup of `lines` in the middle of `area`, at most `max_w`
/// columns wide and as tall as its lines once wrapped.
pub fn popup(f: &mut ratatui::Frame, area: Rect, lines: Vec<Line<'_>>, max_w: u16) {
    CHROME.popup(f, area, lines, max_w, At::Middle);
}
