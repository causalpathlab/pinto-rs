//! A one-line text field whose text starts selected: typing replaces a
//! suggested value at once, and an arrow key keeps it to edit.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::Style;
use ratatui::text::Span;

pub struct Field {
    pub text: String,
    /// The whole text is selected: the next character replaces it and
    /// backspace clears it.
    selected: bool,
}

impl Field {
    /// `text`, all of it selected.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        let selected = !text.is_empty();
        Field { text, selected }
    }

    /// `text` to go on typing at, nothing selected.
    #[must_use]
    pub fn unselected(text: String) -> Self {
        Field {
            text,
            selected: false,
        }
    }

    /// Set the text, unselected, as a completion does.
    pub fn set(&mut self, text: String) {
        self.text = text;
        self.selected = false;
    }

    /// Edit with `key`; whether the text changed. Ctrl-A selects
    /// everything again, ctrl-U clears, and an arrow drops the selection.
    pub fn key(&mut self, key: KeyEvent) -> bool {
        let chord = key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER);
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let before = self.text.len();
        match key.code {
            KeyCode::Char('a') if ctrl => self.selected = !self.text.is_empty(),
            KeyCode::Char('u') if ctrl => self.set(String::new()),
            KeyCode::Char(_) if chord => {}
            KeyCode::Char(c) => {
                if self.selected {
                    self.text.clear();
                }
                self.text.push(c);
                self.selected = false;
                return true;
            }
            KeyCode::Backspace | KeyCode::Delete if self.selected => self.set(String::new()),
            KeyCode::Backspace => {
                self.text.pop();
            }
            KeyCode::Left | KeyCode::Right | KeyCode::Home | KeyCode::End => {
                self.selected = false;
            }
            _ => {}
        }
        self.text.len() != before
    }

    /// The text wrapped at `width`, one span a line, in `style`: reversed
    /// while selected where `on`, else with a bar for the cursor at its end.
    #[must_use]
    pub fn lines(&self, width: usize, on: bool, style: Style) -> Vec<Span<'static>> {
        let mut pieces = super::style::wrap(&self.text, width, width);
        if pieces.is_empty() {
            pieces.push(String::new());
        }
        let n = pieces.len();
        pieces
            .into_iter()
            .enumerate()
            .map(|(i, p)| match (on, self.selected) {
                (true, true) => Span::styled(p, super::style::selected()),
                (true, false) if i + 1 == n => Span::styled(format!("{p}▏"), style),
                _ => Span::styled(p, style),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(f: &mut Field, code: KeyCode) {
        f.key(KeyEvent::new(code, KeyModifiers::NONE));
    }

    #[test]
    fn typing_replaces_the_selection() {
        let mut f = Field::new("pinto-view-001");
        press(&mut f, KeyCode::Char('a'));
        press(&mut f, KeyCode::Char('b'));
        assert_eq!(f.text, "ab");
    }

    #[test]
    fn an_arrow_keeps_the_text_to_edit() {
        let mut f = Field::new("fig");
        press(&mut f, KeyCode::End);
        press(&mut f, KeyCode::Char('2'));
        assert_eq!(f.text, "fig2");
        press(&mut f, KeyCode::Backspace);
        assert_eq!(f.text, "fig");
    }

    #[test]
    fn backspace_clears_the_selection_and_ctrl_a_reselects() {
        let mut f = Field::new("fig");
        press(&mut f, KeyCode::Backspace);
        assert_eq!(f.text, "");
        let mut f = Field::new("fig");
        press(&mut f, KeyCode::End);
        f.key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL));
        press(&mut f, KeyCode::Char('x'));
        assert_eq!(f.text, "x");
    }
}
