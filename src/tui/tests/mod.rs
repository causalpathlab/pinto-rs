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

#[test]
fn a_child_pinto_takes_its_parent_s_atomic_outputs_once() {
    let line = |argv: &[&str]| {
        let mut argv: Vec<String> = argv.iter().map(ToString::to_string).collect();
        with_atomic_outputs(&mut argv);
        argv
    };
    assert_eq!(
        line(&["lc", "x.zarr", "--out=o"]),
        ["--atomic-outputs", "lc", "x.zarr", "--out=o"]
    );
    assert_eq!(
        line(&["lc", "--atomic-outputs", "x.zarr"]),
        ["lc", "--atomic-outputs", "x.zarr"]
    );
    // after `--` a word is no flag
    assert_eq!(
        line(&["lra", "--", "--atomic-outputs"]),
        ["--atomic-outputs", "lra", "--", "--atomic-outputs"]
    );
    use clap::Parser;
    let argv = line(&["fit", "lc", "x.zarr", "--out=o"]);
    let cli = crate::Cli::try_parse_from(std::iter::once("pinto".to_string()).chain(argv)).unwrap();
    assert!(cli.atomic_outputs);
}

/// A job of a session started with `--atomic-outputs` takes the flag; a
/// command a server's tool starts is given it by the tool, which may also
/// leave it out, so none is added to it here.
#[test]
fn a_session_s_job_takes_its_atomic_outputs_and_a_tool_s_command_its_own() {
    let line = || vec!["lc".to_string(), "x.zarr".to_string()];
    let (mut plain, mut atomic) = (line(), line());
    inherit_flags(&mut plain);
    crate::util::staging::ENABLED_HERE.set(true);
    inherit_flags(&mut atomic);
    let started = own_command(&line()).unwrap();
    crate::util::staging::ENABLED_HERE.set(false);
    assert_eq!(plain, ["lc", "x.zarr"]);
    assert_eq!(atomic, ["--atomic-outputs", "lc", "x.zarr"]);
    let args: Vec<_> = started.get_args().collect();
    assert_eq!(args, ["lc", "x.zarr"]);
}

#[test]
fn a_child_pinto_takes_its_parent_s_max_threads_unless_it_has_its_own() {
    let line = |argv: &[&str], parent| {
        let mut argv: Vec<String> = argv.iter().map(ToString::to_string).collect();
        with_max_threads(&mut argv, parent);
        argv
    };
    let default = crate::util::common::DEFAULT_MAX_THREADS;
    assert_eq!(
        line(&["lc", "x.zarr", "--out=o"], 4),
        ["--max-threads=4", "lc", "x.zarr", "--out=o"]
    );
    // its own count stands, in either spelling
    assert_eq!(
        line(&["lc", "--max-threads=8", "x.zarr"], 4),
        ["lc", "--max-threads=8", "x.zarr"]
    );
    assert_eq!(
        line(&["lc", "--max-threads", "8"], 4),
        ["lc", "--max-threads", "8"]
    );
    // the default needs no flag; after `--` a word is no flag
    assert_eq!(line(&["lc", "x.zarr"], default), ["lc", "x.zarr"]);
    assert_eq!(
        line(&["lra", "--", "--max-threads=8"], 4),
        ["--max-threads=4", "lra", "--", "--max-threads=8"]
    );
    // as the child parses it
    use clap::Parser;
    let argv = line(&["fit", "lc", "x.zarr", "--out=o"], 4);
    let cli = crate::Cli::try_parse_from(std::iter::once("pinto".to_string()).chain(argv)).unwrap();
    assert_eq!(cli.max_threads, 4);
}
