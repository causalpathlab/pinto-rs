use super::*;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

#[test]
fn shift_enter_is_told_from_enter() {
    assert!(shift_enter(&KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)));
    assert!(!shift_enter(&KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)));
    assert!(!shift_enter(&KeyEvent::new(KeyCode::Char('H'), KeyModifiers::SHIFT)));
}
