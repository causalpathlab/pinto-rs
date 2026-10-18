use super::*;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

#[test]
fn shift_enter_is_told_from_enter() {
    let is = |code, m| shift_enter(&KeyEvent::new(code, m));
    assert!(is(KeyCode::Enter, KeyModifiers::SHIFT));
    assert!(!is(KeyCode::Enter, KeyModifiers::NONE));
    assert!(!is(KeyCode::Char('H'), KeyModifiers::SHIFT));
}

#[test]
fn a_plain_screen_is_always_attached_and_offers_no_session_keys() {
    let session = plain_session();
    assert!(!session.is_hosted() && !session.offers_keys());
    assert!(session.attached() && !session.stop_requested());
    assert_eq!(session.name(), None);
    assert!(session_keys().is_empty());
}

#[test]
fn every_screen_asks_for_shift_enter_and_its_mouse() {
    let m = modes(legume_tui::Mouse::Full);
    assert_eq!(m.mouse, legume_tui::Mouse::Full);
    assert!(m.keyboard_enhancement.is_some());
    assert_eq!(modes(legume_tui::Mouse::Off).mouse, legume_tui::Mouse::Off);
}
