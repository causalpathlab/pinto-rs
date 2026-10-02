use super::*;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

#[test]
fn shift_enter_is_told_from_enter() {
    let is = |code, m| shift_enter(&KeyEvent::new(code, m));
    assert!(is(KeyCode::Enter, KeyModifiers::SHIFT));
    assert!(!is(KeyCode::Enter, KeyModifiers::NONE));
    assert!(!is(KeyCode::Char('H'), KeyModifiers::SHIFT));
}
