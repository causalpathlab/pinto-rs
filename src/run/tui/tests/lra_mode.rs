//! `pinto lra`'s screens: lra and its power check on one fitted run,
//! chosen first, written beside it, with no cluster-only flags.

use super::super::data::Pick;
use super::*;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

fn lra_app(dir: &Path) -> App {
    App {
        here: dir.to_path_buf(),
        ..App::new_lra(crate::built_cli(), dir.to_path_buf()).unwrap()
    }
}

fn key(a: &mut App, c: KeyCode) {
    a.key(KeyEvent::new(c, KeyModifiers::NONE));
}

#[test]
fn lra_and_its_power_check_are_the_methods_with_no_shard_and_no_data_screen() {
    let dir = tempfile::tempdir().unwrap();
    let a = lra_app(dir.path());
    let names: Vec<&str> = a.rows.iter().map(|r| r.form.name.as_str()).collect();
    assert_eq!(names, ["lra", "lra-simulate"]);
    assert!(a.rows[0].on && !a.rows[1].on, "lra is queued");
    assert!(a
        .rows
        .iter()
        .all(|r| r.form.fields.iter().all(|f| f.long != "shard")));
    assert!(!a.screens().contains(&Screen::Data));
    assert_eq!(a.screen, Screen::Methods);
    assert!(a.editor.is_none(), "no output header");
    assert_eq!(a.browser.as_ref().map(|b| b.want), Some(Pick::Run));
}

#[test]
fn the_run_chosen_is_read_and_written_beside() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = lra_app(dir.path());
    a.browser = None;
    // Nothing to review before a run is chosen.
    a.open_confirm();
    assert!(a.confirm.is_none());
    assert!(a.message.as_deref().unwrap().contains("f chooses"));

    a.take_run(&dir.path().join("res/tissue.pinto.json"));
    for r in &a.rows {
        assert_eq!(r.form.from(), Some("res/tissue.pinto.json"));
        assert_eq!(r.out, "res/tissue");
    }
    // Run where it writes, beside the run.
    let planned = a.plan();
    let job = &planned[0].job;
    assert_eq!(job.dir, dir.path().join("res"));
    assert_eq!(
        job.argv,
        ["lra", "--out=tissue", "--from", "tissue.pinto.json"]
    );

    // `f` chooses another run.
    key(&mut a, KeyCode::Char('f'));
    assert_eq!(a.browser.as_ref().map(|b| b.want), Some(Pick::Run));
}
