//! How `pinto view` and `pinto run` look: the terminal's own colours, so
//! both read on a dark or a light background, with bold for what matters,
//! gray for hints, and the line under a cursor reversed.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};

/// Bold text.
#[must_use]
pub fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

/// Key hints and other secondary lines.
#[must_use]
pub fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

/// The line under a menu's cursor.
#[must_use]
pub fn selected() -> Style {
    Style::default().add_modifier(Modifier::REVERSED)
}

/// The first of `n` rows to show in a window `rows` tall so that `row`
/// sits in its middle where it can.
#[must_use]
pub fn first_row(row: usize, rows: usize, n: usize) -> usize {
    row.saturating_sub(rows / 2).min(n.saturating_sub(rows))
}

/// `s` in at most `n` characters, `…` marking a cut at the end.
#[must_use]
pub fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

/// The last `n` characters of `s`, `…` marking a cut at the start.
#[must_use]
pub fn tail(s: &str, n: usize) -> String {
    let len = s.chars().count();
    if len <= n {
        s.to_string()
    } else {
        let mut out = String::from("…");
        out.extend(s.chars().skip(len + 1 - n.max(1)));
        out
    }
}

/// A bordered popup of `lines` in the middle of `area`, at most `max_w`
/// columns wide and as tall as its lines.
pub fn popup(f: &mut ratatui::Frame, area: Rect, lines: Vec<Line<'_>>, max_w: u16) {
    let w = max_w.min(area.width);
    let h = (lines.len() as u16 + 2).min(area.height);
    let r = Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    );
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(Block::bordered().border_style(dim())),
        r,
    );
}

#[cfg(test)]
#[path = "tests/style.rs"]
mod tests;
