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
    assert_eq!(session_keys().count(), 0);
}

#[test]
fn every_screen_asks_for_shift_enter_and_its_mouse() {
    let m = modes(legume_tui::Mouse::Full);
    assert_eq!(m.mouse, legume_tui::Mouse::Full);
    assert!(matches!(
        m.keyboard_enhancement,
        legume_tui::Keyboard::IfSupported(_)
    ));
    assert_eq!(modes(legume_tui::Mouse::Off).mouse, legume_tui::Mouse::Off);
}

#[test]
fn a_program_started_from_a_plain_session_is_plain_too() {
    // `--no-detach` reaches what the session starts, so a TUI among them
    // runs plain as well.
    let env = plain_session().child_env();
    let no_detach = std::ffi::OsString::from(legume_tui::NO_DETACH_VAR);
    assert_eq!(env.vars(), [(no_detach, std::ffi::OsString::from("1"))]);
}
