//! Keys every screen and popup of `pinto run` shares: `q` quits, Esc
//! closes, `X` stops, `?` lists the keys.

use super::tests::key;
use super::*;
use ratatui::crossterm::event::KeyCode;

fn app() -> (tempfile::TempDir, App) {
    let dir = tempfile::tempdir().unwrap();
    let a = super::tests::app(dir.path());
    (dir, a)
}

#[test]
fn q_quits_from_the_review_and_esc_only_closes_it() {
    let (_dir, mut a) = app();
    a.confirm = Some(Review::new(Vec::new()));
    key(&mut a, KeyCode::Esc);
    assert!(a.confirm.is_none() && !a.quit, "esc closes, does not quit");
    a.confirm = Some(Review::new(Vec::new()));
    key(&mut a, KeyCode::Char('q'));
    assert!(a.quit, "q quits from the review");
}

#[test]
fn question_mark_shows_the_keys_until_the_next_key() {
    let (_dir, mut a) = app();
    key(&mut a, KeyCode::Char('?'));
    assert!(a.help);
    let text: String = a
        .help_lines()
        .iter()
        .map(|l| l.to_string() + "\n")
        .collect();
    assert!(
        text.contains("keys: Data") && text.contains("q quit"),
        "{text}"
    );
    // The key that closes it does nothing else: `q` does not quit.
    key(&mut a, KeyCode::Char('q'));
    assert!(!a.help && !a.quit);
}
