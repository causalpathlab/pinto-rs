use super::super::script::under;
use super::data::Side;
use super::*;

fn cli() -> clap::Command {
    // A stand-in for pinto's command: every method takes data files,
    // --out, coordinate and batch files, and one flag of its own.
    let method = |name: &'static str| {
        clap::Command::new(name)
            .about("fit")
            .arg(
                clap::Arg::new("data_files")
                    .num_args(1..)
                    .value_delimiter(','),
            )
            .arg(clap::Arg::new("out").long("out").required(true))
            .arg(clap::Arg::new("coord").long("coord").value_delimiter(','))
            .arg(
                clap::Arg::new("batch_files")
                    .long("batch-files")
                    .value_delimiter(','),
            )
            .arg(
                clap::Arg::new("steps")
                    .long("steps")
                    .default_value("10")
                    .value_parser(clap::value_parser!(usize)),
            )
    };
    let mut c = clap::Command::new("pinto").subcommands(METHODS.map(method));
    c.build();
    c
}

fn app(dir: &Path) -> App {
    let mut a = App {
        here: dir.to_path_buf(),
        ..App::new(cli(), dir.to_path_buf()).unwrap()
    };
    a.browser = None;
    a.editor = None;
    for m in &mut a.rows {
        m.out = free_out(dir, &m.form.name);
    }
    a
}

fn key(a: &mut App, c: KeyCode) {
    a.key(KeyEvent::new(c, KeyModifiers::NONE));
}

/// ctrl+r, which runs what a screen set up, as in senna.
fn run_key(a: &mut App) {
    a.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
}

/// A fit of method `m` in `dir`, writing to `--out m`.
fn job(dir: &Path, m: &str) -> jobs::Job {
    jobs::Job::fit(m, 0, dir, m, vec![m.into(), "--out".into(), m.into()])
}

/// A stand-in pinto in `dir`: the shell script `body`.
#[cfg(unix)]
fn fake_pinto(dir: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let fake = dir.join("fake-pinto");
    std::fs::write(&fake, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    fake
}

fn data(dir: &Path, names: &[&str]) -> Vec<Pair> {
    names
        .iter()
        .map(|n| {
            let p = dir.join(n);
            std::fs::write(&p, "").unwrap();
            Pair::pending(p)
        })
        .collect()
}

#[test]
fn an_out_already_used_gets_the_next_free_name() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(free_out(dir.path(), "lc"), "lc");
    std::fs::write(dir.path().join("lc.pinto.json"), "{}").unwrap();
    std::fs::write(dir.path().join("lc-2.cmd.sh"), "").unwrap();
    assert_eq!(free_out(dir.path(), "lc"), "lc-3");
}

#[test]
fn queued_methods_share_the_data_and_write_their_own_out() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d1.zarr.zip", "d2.zarr.zip"]);
    a.pairs[0].batch = Some(dir.path().join("b1.tsv"));
    a.pairs[1].batch = Some(dir.path().join("b2.tsv"));
    a.pairs[0].coord = Some(dir.path().join("c1.csv"));
    a.pairs[1].coord = Some(dir.path().join("c2.csv"));
    a.screen = Screen::Methods;
    a.method_row = METHODS.iter().position(|m| *m == "lc").unwrap();
    key(&mut a, KeyCode::Char(' '));
    a.method_row = METHODS.iter().position(|m| *m == "cage").unwrap();
    key(&mut a, KeyCode::Char(' '));
    let planned = a.plan();
    assert_eq!(planned.len(), 2);
    for p in &planned {
        assert_eq!(p.problem, None);
        assert_eq!(
            p.job.argv,
            [
                p.job.method.as_str(),
                "d1.zarr.zip",
                "d2.zarr.zip",
                "--coord",
                "c1.csv,c2.csv",
                "--batch-files",
                "b1.tsv,b2.tsv",
                "--out",
                p.job.method.as_str()
            ]
        );
    }
}

#[test]
fn what_would_overwrite_or_misparse_is_stopped_before_running() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.rows[lc].on = true;
    assert!(a.plan()[0].problem.as_ref().unwrap().contains("no data"));

    a.pairs = data(dir.path(), &["d1.zarr", "d2.zarr"]);
    a.pairs[0].batch = Some(dir.path().join("b1.tsv"));
    assert!(a.plan()[0].problem.as_ref().unwrap().contains("batch"));
    a.pairs[0].batch = None;

    a.pairs[1].coord = Some(dir.path().join("c2.csv"));
    assert!(a.plan()[0]
        .problem
        .as_ref()
        .unwrap()
        .contains("coordinates"));
    a.pairs[1].coord = None;
    assert!(a.coord_warning().unwrap().contains("expression"));

    std::fs::write(dir.path().join("lc.pinto.json"), "{}").unwrap();
    assert!(a.plan()[0].problem.as_ref().unwrap().contains("exists"));
    a.rows[lc].out = "sub/r".into();
    std::fs::write(dir.path().join("sub"), "").unwrap();
    assert!(a.plan()[0]
        .problem
        .as_ref()
        .unwrap()
        .contains("not a folder"));
    std::fs::remove_file(dir.path().join("sub")).unwrap();
    // Not there yet: made when the fit starts.
    let p = &a.plan()[0];
    assert_eq!(p.problem, None);
    assert_eq!(p.job.dir, script::normalize(&dir.path().join("sub")));
    assert_eq!(p.job.argv[1], "../d1.zarr");

    let steps = a.rows[lc]
        .form
        .fields
        .iter()
        .position(|f| f.long == "steps")
        .unwrap();
    a.rows[lc].form.fields[steps].value = "many".into();
    let p = &a.plan()[0];
    assert!(p.problem.is_some());
    assert_eq!(p.blamed.as_deref(), Some("steps"));
}

#[test]
fn enter_on_a_problem_goes_to_the_flag_clap_blamed() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d.zarr"]);
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.rows[lc].on = true;
    a.rows[lc].form.fields[0].value = "many".into();
    key(&mut a, KeyCode::Char('G'));
    assert!(a.confirm.is_none(), "G no longer runs");
    run_key(&mut a);
    assert!(a.confirm.is_some());
    a.confirm.as_mut().unwrap().at -= 2 * CONFIRM_SETTLE;
    run_key(&mut a);
    assert!(a.confirm.is_none() && a.queue.is_none());
    assert_eq!(a.screen, Screen::Params);
    assert_eq!(a.param_method, lc);
}

#[test]
fn flags_are_changed_and_typed_on_the_parameters_screen() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Methods;
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.screen, Screen::Params);
    assert!(a.rows[0].on);
    key(&mut a, KeyCode::Enter);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Char('5'));
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.rows[0].form.fields[0].value, "5");
    assert_eq!(a.rows[0].form.changed(), 1);
    key(&mut a, KeyCode::Char('r'));
    assert_eq!(a.rows[0].form.changed(), 0);
}

#[test]
fn the_filter_narrows_the_flags() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Params;
    assert_eq!(a.visible().len(), 1);
    key(&mut a, KeyCode::Char('/'));
    key(&mut a, KeyCode::Char('z'));
    assert!(a.visible().is_empty());
    key(&mut a, KeyCode::Esc);
    assert_eq!(a.visible().len(), 1);
}

#[test]
fn every_method_of_pinto_parses_from_an_untouched_form() {
    use clap::CommandFactory;
    let mut cli = crate::Cli::command();
    cli.build();
    for m in METHODS {
        let form = Method::new(&cli, m).unwrap();
        assert!(!form.fields.is_empty(), "{m} has no flags");
        assert!(form.takes_batches(), "{m} takes no batch files");
        assert!(form.takes_coords(), "{m} takes no coordinate files");
        let argv = form.argv(
            &["d1.zarr".into(), "d2.zarr".into()],
            &["c1.csv".into(), "c2.csv".into()],
            &["b1.tsv".into(), "b2.tsv".into()],
            "o",
        );
        assert_eq!(form::check(&cli, &argv), Ok(()), "{m}: {argv:?}");
        // A switch flipped on reaches the command line as clap takes it.
        let mut form = form;
        if let Some(f) = form
            .fields
            .iter_mut()
            .find(|f| matches!(f.kind, Kind::Flag { .. }) && !f.advanced)
        {
            f.toggle();
            let argv = form.argv(&["d.zarr".into()], &[], &[], "o");
            assert_eq!(form::check(&cli, &argv), Ok(()), "{m}: {argv:?}");
        }
    }
}

#[test]
fn an_out_folder_given_with_dotdot_still_finds_the_data() {
    let dir = tempfile::tempdir().unwrap();
    let here = dir.path().join("b");
    std::fs::create_dir_all(&here).unwrap();
    std::fs::create_dir_all(dir.path().join("res")).unwrap();
    let mut a = app(&here);
    a.pairs = data(&here, &["d.zarr"]);
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.rows[lc].on = true;
    a.rows[lc].out = "../res/r".into();
    let p = &a.plan()[0];
    assert_eq!(p.problem, None);
    assert_eq!(p.job.argv[1], "../b/d.zarr");
    assert_eq!(p.job.dir, script::normalize(&dir.path().join("res")));
}

#[test]
fn a_parent_start_folder_is_resolved() {
    let a = App::new(cli(), PathBuf::from("..")).unwrap();
    let here = std::env::current_dir().unwrap();
    assert_eq!(a.browse_dir, here.parent().unwrap());
}

#[test]
fn a_flag_with_a_default_cannot_be_cleared_to_unset() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Params;
    key(&mut a, KeyCode::Enter);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Enter);
    let f = &a.rows[0].form.fields[0];
    assert_eq!(f.value, "10");
    assert!(a.message.as_deref().unwrap().contains("default 10"));
}

#[test]
fn the_confirm_popup_scrolls_no_further_than_its_end() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.confirm = Some(Review::new(Vec::new()));
    a.confirm_max.set(3);
    for _ in 0..10 {
        key(&mut a, KeyCode::Down);
    }
    assert_eq!(a.confirm_scroll, 3);
    key(&mut a, KeyCode::Up);
    assert_eq!(a.confirm_scroll, 2);
}

#[test]
fn batch_files_are_described_as_paired() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["s1.zarr", "s2.zarr"]);
    a.take_sides(
        &[dir.path().join("x.tsv"), dir.path().join("y.tsv")],
        Side::Batch,
    );
    assert!(a
        .message
        .as_deref()
        .unwrap()
        .contains("in the order listed"));
}

#[test]
fn data_taken_find_their_coordinates_and_labels_beside_them() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    for n in [
        "s1.zarr.zip",
        "s2.zarr.zip",
        "s1_coords.csv",
        "s2_coords.csv",
        "s1_batch.txt",
    ] {
        std::fs::write(d.join(n), "").unwrap();
    }
    let mut a = app(d);
    a.take_data(vec![d.join("s1.zarr.zip"), d.join("s2.zarr.zip")]);
    assert_eq!(a.pairs.len(), 2);
    assert_eq!(a.pairs[0].coord, Some(d.join("s1_coords.csv")));
    assert_eq!(a.pairs[1].coord, Some(d.join("s2_coords.csv")));
    assert_eq!(a.pairs[0].batch, Some(d.join("s1_batch.txt")));
    assert_eq!(a.pairs[1].batch, None);
    assert!(a.message.as_deref().unwrap().contains("2 with coordinates"));
    // Taking one again adds nothing.
    a.take_data(vec![d.join("s1.zarr.zip")]);
    assert_eq!(a.pairs.len(), 2);
    // One coordinate file goes to the row under the cursor.
    a.pair_row = 1;
    a.take_sides(&[d.join("s1_coords.csv")], Side::Coord);
    assert_eq!(a.pairs[1].coord, Some(d.join("s1_coords.csv")));
    key(&mut a, KeyCode::Char('x'));
    assert_eq!(
        (a.pairs[1].coord.clone(), a.pairs[1].batch.clone()),
        (None, None)
    );
    key(&mut a, KeyCode::Char('X'));
    assert!(a
        .pairs
        .iter()
        .all(|p| p.coord.is_none() && p.batch.is_none()));
}

#[cfg(unix)]
#[test]
fn only_a_finished_fit_opens_in_the_viewer() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    // A stand-in pinto that writes the manifest its --out names, or fails.
    let fake = fake_pinto(
        dir.path(),
        "[ \"$1\" = bad ] && exit 1\nwhile [ $# -gt 0 ]; do [ \"$1\" = --out ] && touch \"$2.pinto.json\"; shift; done",
    );
    let job = |m| job(dir.path(), m);
    a.queue = Some(Queue::start(vec![job("good"), job("bad")], fake));
    a.screen = Screen::Run;
    while a.running() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    key(&mut a, KeyCode::Down);
    key(&mut a, KeyCode::Char('v'));
    assert!(!a.quit && a.view.is_none());
    assert!(a.message.as_deref().unwrap().contains("did not finish"));
    key(&mut a, KeyCode::Up);
    key(&mut a, KeyCode::Char('v'));
    assert!(a.quit);
    assert_eq!(a.view, Some(dir.path().join("good.pinto.json")));
}

#[test]
fn the_parameters_screen_shows_a_queued_method_however_it_is_reached() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    let cage = METHODS.iter().position(|m| *m == "cage").unwrap();
    a.screen = Screen::Methods;
    a.method_row = cage;
    key(&mut a, KeyCode::Char(' '));
    // Tab, not enter: the screen must not keep showing the first method.
    key(&mut a, KeyCode::Tab);
    assert_eq!(a.screen, Screen::Params);
    assert_eq!(a.param_method, cage);
}

#[test]
fn named_batches_reach_the_command_line_as_files_written_for_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d1.zarr", "d2.zarr"]);
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.rows[lc].on = true;
    a.screen = Screen::Data;
    a.pair_row = 1;
    key(&mut a, KeyCode::Char('n'));
    for c in "b1".chars() {
        key(&mut a, KeyCode::Char(c));
    }
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.pairs[1].name.as_deref(), Some("b1"));
    // Its cell count is not known yet: blocked.
    assert!(a.plan()[0].problem.as_ref().unwrap().contains("cell count"));
    a.pairs[0].cells = Some(3);
    a.pairs[1].cells = Some(2);
    let p = &a.plan()[0];
    assert_eq!(p.problem, None);
    let at = p
        .job
        .argv
        .iter()
        .position(|w| w == "--batch-files")
        .unwrap();
    assert_eq!(p.job.argv[at + 1], "lc.batches/d1.txt,lc.batches/d2.txt");
    assert_eq!(
        p.job.made[1],
        (
            script::normalize(dir.path()).join("lc.batches/d2.txt"),
            batch::Made::Repeat("b1".into(), 2)
        )
    );
    std::fs::create_dir(dir.path().join("lc.batches")).unwrap();
    assert!(a.plan()[0].problem.as_ref().unwrap().contains("exists"));
    // Clearing the row's name back to empty: every file its own again.
    std::fs::remove_dir(dir.path().join("lc.batches")).unwrap();
    key(&mut a, KeyCode::Char('n'));
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Enter);
    let p = &a.plan()[0];
    assert!(!p.job.argv.contains(&"--batch-files".to_string()));
    assert!(p.job.made.is_empty());
}

#[test]
fn labels_read_in_the_background_are_renamed_from_their_list() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["s1.zarr"]);
    let labels = dir.path().join("s1_batch.txt");
    std::fs::write(&labels, "A\nB\nA\n").unwrap();
    a.take_sides(std::slice::from_ref(&labels), Side::Batch);
    a.screen = Screen::Data;
    while a.pairs[0].labels.is_none() {
        a.poll();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    key(&mut a, KeyCode::Char('e'));
    assert!(a.labels.is_some());
    key(&mut a, KeyCode::Down);
    key(&mut a, KeyCode::Enter);
    key(&mut a, KeyCode::Backspace);
    key(&mut a, KeyCode::Char('A'));
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.pairs[0].renames.get("B").map(String::as_str), Some("A"));
    key(&mut a, KeyCode::Esc);
    assert!(a.labels.is_none());
    let (batches, _) = batch::summary(&a.pairs);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].cells, Some(3));
}

#[test]
fn the_confirm_popup_wraps_long_commands_and_keeps_its_keys() {
    let dir = tempfile::tempdir().unwrap();
    let deep = dir.path().join("a".repeat(60)).join("b".repeat(60));
    std::fs::create_dir_all(&deep).unwrap();
    let mut a = app(dir.path());
    a.pairs = data(&deep, &["d1.zarr", "d2.zarr"]);
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.rows[lc].on = true;
    a.confirm = Some(Review::new(a.plan()));
    let lines: Vec<String> = a
        .confirm_lines(12, 40)
        .iter()
        .map(ToString::to_string)
        .collect();
    assert!(lines.len() <= 12, "{lines:?}");
    assert!(lines.iter().all(|l| l.chars().count() <= 40), "{lines:?}");
    assert!(lines.last().unwrap().contains("esc back"), "{lines:?}");
    // Scrolling reaches the last wrapped row.
    assert!(a.confirm_max.get() > 0);
}

#[test]
fn many_data_files_are_all_described_by_the_few_workers() {
    let dir = tempfile::tempdir().unwrap();
    let names: Vec<String> = (0..20).map(|i| format!("s{i}.zarr")).collect();
    let paths: Vec<PathBuf> = names
        .iter()
        .map(|n| {
            let p = dir.path().join(n);
            std::fs::write(&p, "").unwrap();
            p
        })
        .collect();
    let mut a = app(dir.path());
    a.take_data(paths);
    assert_eq!(a.describing, 20);
    while a.describing > 0 {
        a.poll();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(a.pairs.iter().all(|p| p.info != "reading…"));
}

#[test]
fn the_keys_wrap_at_whole_hints_and_name_shift_enter() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    let plain: Vec<String> = a.status(60).iter().map(ToString::to_string).collect();
    assert!(
        !plain.iter().any(|l| l.contains("shift-enter")),
        "not named where the terminal cannot tell it: {plain:?}"
    );
    a.shift_enter = true;
    let lines: Vec<String> = a.status(60).iter().map(ToString::to_string).collect();
    assert!(lines.iter().all(|l| l.chars().count() <= 60), "{lines:?}");
    assert!(lines.iter().any(|l| l.contains("shift-enter")), "{lines:?}");
    assert!(lines.last().unwrap().contains("q quit"));
}

#[test]
fn keys_that_need_data_say_so_without_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Data;
    key(&mut a, KeyCode::Char('n'));
    assert!(a.editor.is_none());
    assert!(a.message.as_deref().unwrap().contains("a adds"));
}

#[test]
fn an_out_a_run_wrote_to_is_not_offered_again() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lc.cmd.sh"), "").unwrap();
    assert_eq!(free_out(dir.path(), "lc"), "lc-2");
}

#[test]
fn the_output_header_names_every_out() {
    assert_eq!(under("", "lc"), "lc");
    assert_eq!(under("exp1", "lc"), "exp1_lc");
    for h in ["results/", "exp1_", "exp1-", "exp1."] {
        assert_eq!(under(h, "lc"), format!("{h}lc"));
    }
}

#[test]
fn the_header_is_asked_first_over_the_browser() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = App {
        here: dir.path().to_path_buf(),
        ..App::new(cli(), dir.path().to_path_buf()).unwrap()
    };
    assert!(a.browser.is_some());
    assert_eq!(a.editor.as_ref().map(|e| &e.target), Some(&Target::Header));
    for c in "res/".chars() {
        key(&mut a, KeyCode::Char(c));
    }
    key(&mut a, KeyCode::Enter);
    assert!(a.editor.is_none() && a.browser.is_some());
    assert_eq!(a.header, "res/");
    assert!(a
        .rows
        .iter()
        .all(|r| r.out == format!("res/{}", r.form.name)));

    // Esc: no header.
    let mut a = App {
        here: dir.path().to_path_buf(),
        ..App::new(cli(), dir.path().to_path_buf()).unwrap()
    };
    key(&mut a, KeyCode::Esc);
    assert!(a.editor.is_none() && a.header.is_empty());
    assert!(a.rows.iter().all(|r| r.out == r.form.name));
}

#[test]
fn a_hand_typed_out_keeps_through_header_changes_until_cleared() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Methods;
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.method_row = lc;
    key(&mut a, KeyCode::Char('o'));
    a.editor.as_mut().unwrap().text.text = "mine".into();
    key(&mut a, KeyCode::Enter);
    assert!(a.rows[lc].typed);

    std::fs::write(dir.path().join("exp1_cage.pinto.json"), "{}").unwrap();
    key(&mut a, KeyCode::Char('O'));
    a.editor.as_mut().unwrap().text.text = "exp1".into();
    key(&mut a, KeyCode::Enter);
    let cage = METHODS.iter().position(|m| *m == "cage").unwrap();
    assert_eq!(a.rows[lc].out, "mine");
    assert_eq!(a.rows[cage].out, "exp1_cage-2", "made unique");

    key(&mut a, KeyCode::Char('o'));
    a.editor.as_mut().unwrap().text.text.clear();
    key(&mut a, KeyCode::Enter);
    assert!(!a.rows[lc].typed);
    assert_eq!(a.rows[lc].out, "exp1_lc");
}

#[test]
fn esc_on_a_later_header_edit_keeps_the_header() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Methods;
    key(&mut a, KeyCode::Char('O'));
    a.editor.as_mut().unwrap().text.text = "exp1".into();
    key(&mut a, KeyCode::Enter);
    key(&mut a, KeyCode::Char('O'));
    a.editor.as_mut().unwrap().text.text = "other".into();
    key(&mut a, KeyCode::Esc);
    assert_eq!(a.header, "exp1");
    assert!(a
        .rows
        .iter()
        .all(|r| r.out == format!("exp1_{}", r.form.name)));
}

#[test]
fn a_hand_typed_folder_out_names_the_method_in_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d.zarr"]);
    a.screen = Screen::Methods;
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.method_row = lc;
    a.rows[lc].on = true;
    key(&mut a, KeyCode::Char('o'));
    a.editor.as_mut().unwrap().text.text = "res/sub/".into();
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.rows[lc].out, "res/sub/lc");
    // Not there yet: no problem, the run makes it.
    let p = &a.plan()[0];
    assert_eq!(p.problem, None);
    assert_eq!(p.job.dir, script::normalize(&dir.path().join("res/sub")));
}

#[test]
fn shift_enter_moves_to_the_next_screen_and_enter_on_data_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.screen = Screen::Data;
    key(&mut a, KeyCode::Enter);
    assert_eq!(a.screen, Screen::Data);
    a.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
    assert_eq!(a.screen, Screen::Methods);
    a.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
    assert_eq!(a.screen, Screen::Params);
    assert!(!a.rows.iter().any(|r| r.on), "moving on queues nothing");
}

#[cfg(unix)]
#[test]
fn methods_a_run_finished_are_unqueued_so_the_run_key_again_runs_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    let (lc, cage) = (0, 1);
    a.pairs = data(dir.path(), &["d.zarr"]);
    a.rows[lc].on = true;
    a.rows[cage].on = true;
    // A stand-in pinto that finishes the first method and fails the next.
    let fake = fake_pinto(dir.path(), &format!("[ \"$1\" = {} ]", METHODS[lc]));
    // In the folder as the plan resolves it.
    let at = a.out_of(lc).0;
    let job = |row: usize| jobs::Job {
        row,
        ..job(&at, METHODS[row])
    };
    a.queue = Some(Queue::start(vec![job(lc), job(cage)], fake));
    while a.running() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    a.poll();
    assert!(!a.rows[lc].on && a.rows[lc].done);
    assert!(a.rows[cage].on && !a.rows[cage].done);
    assert!(a
        .message
        .as_deref()
        .unwrap()
        .contains("ctrl+r runs the rest"));
    // The finished one moves on; the failed one keeps its --out, its
    // script there cleared when it runs again.
    assert_eq!(a.rows[lc].out, format!("{}-2", METHODS[lc]));
    assert_eq!(a.rows[cage].out, METHODS[cage]);
    assert!(dir
        .path()
        .join(format!("{}.cmd.sh", METHODS[cage]))
        .exists());
    let p = a.plan();
    assert_eq!(p.len(), 1);
    assert!(p[0].problem.is_none(), "{:?}", p[0].problem);
    assert!(p[0].job.clear);
    // An --out typed anew clears nothing.
    a.rows[cage].out = "mine".into();
    assert!(!a.plan()[0].job.clear);
    a.rows[cage].out = METHODS[cage].into();
    // r on the Run screen reviews what is left.
    a.screen = Screen::Run;
    key(&mut a, KeyCode::Char('r'));
    assert_eq!(a.confirm.as_ref().map(|r| r.planned.len()), Some(1));
    key(&mut a, KeyCode::Esc);
    // Queued again by hand, it stays queued.
    a.screen = Screen::Methods;
    a.method_row = lc;
    key(&mut a, KeyCode::Char(' '));
    a.poll();
    assert!(a.rows[lc].on);
    assert!(a.rows[lc].done, "finished before: the review says so");
}

#[test]
fn the_session_status_says_where_the_fits_are() {
    use super::jobs::State;
    let a = app(tempfile::tempdir().unwrap().path());
    assert_eq!(a.session_status(), "setting up fits");
    let states = [State::Done, State::Running, State::Waiting];
    assert_eq!(fits_status(&states, false), "fits running: 1 of 3 done");
    let states = [State::Done, State::Failed("x".into()), State::Stopped];
    assert_eq!(
        fits_status(&states, true),
        "fits finished: 1 of 3 done, 1 failed, 1 stopped"
    );
}

#[test]
fn v_opens_the_run_as_this_session_was_asked_to() {
    let session = crate::tui::plain_session();
    let mut cmd = view_command("pinto".into(), Path::new("r.pinto.json"));
    session.command(&mut cmd, legume_tui::Group::Shared);
    let args: Vec<_> = cmd
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    assert_eq!(args, ["view", "r.pinto.json"]);
    // `--no-detach` reaches the view, which then runs plain too.
    let env: Vec<_> = cmd.get_envs().collect();
    let no_detach = std::ffi::OsStr::new(legume_tui::NO_DETACH_VAR);
    assert_eq!(env, [(no_detach, Some(std::ffi::OsStr::new("1")))]);
}

#[test]
fn nothing_is_written_to_a_terminal_that_is_not_there() {
    // Without pbcopy, the copy goes through the terminal: refused while
    // detached rather than written into nothing.
    if std::process::Command::new("pbcopy")
        .stdin(std::process::Stdio::null())
        .spawn()
        .is_ok()
    {
        return;
    }
    let e = copy("text", false).unwrap_err();
    assert!(e.to_string().contains("no terminal"), "{e}");
}

#[test]
fn a_double_tap_of_the_run_key_stops_at_the_review() {
    // ctrl+r opens the review and runs from it, as in senna; pressed twice
    // at once, or held, it must not skip the review it opened.
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d.zarr"]);
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.rows[lc].on = true;
    a.rows[lc].form.fields[0].value = "many".into();
    run_key(&mut a);
    run_key(&mut a);
    assert!(a.confirm.is_some(), "the review stays up");
    // A moment later, it runs (here: stops at the problem it found).
    a.confirm.as_mut().unwrap().at -= 2 * CONFIRM_SETTLE;
    run_key(&mut a);
    assert!(a.confirm.is_none());
}

#[test]
fn a_held_run_key_stops_at_the_review() {
    // Key repeat sends the run key again and again, each within the settle
    // time of the last: the review stays up until the key is let go.
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d.zarr"]);
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.rows[lc].on = true;
    a.rows[lc].form.fields[0].value = "many".into();
    run_key(&mut a);
    for _ in 0..3 {
        a.confirm.as_mut().unwrap().at -= CONFIRM_SETTLE * 4 / 5;
        run_key(&mut a);
        assert!(a.confirm.is_some(), "a repeat does not run");
    }
}

#[test]
fn a_running_fit_shows_its_bar_until_its_fit_clears_it_then_its_log() {
    use super::draw::running;
    use legume_tui::child::{Bar, Progress, Said, GRACE};
    assert_eq!(running(None, "reading", 80), "running  reading");
    let mut bar = None;
    Bar::track(
        &mut bar,
        &Said::Progress(Progress::new(2, 10, "sweeps 2: moves=4")),
    );
    assert!(running(bar.as_ref(), "level 1", 80).starts_with('█'));
    // Cleared: kept for the grace (a redraw split between reads), then
    // the fit's own words.
    Bar::track(&mut bar, &Said::Cleared);
    assert!(running(bar.as_ref(), "level 1", 80).starts_with('█'));
    std::thread::sleep(GRACE + std::time::Duration::from_millis(50));
    assert_eq!(running(bar.as_ref(), "level 1", 80), "running  level 1");
}

/// lra and lra-simulate follow the fits, on a run given by `--from`: no
/// data files are passed, files typed where pinto run started are read
/// from where it runs, and what says each is done is its table.
#[test]
fn lra_on_a_run_is_set_up_with_its_flags() {
    use clap::CommandFactory;
    let dir = tempfile::tempdir().unwrap();
    // macOS temp folders sit behind a symlink (/var -> /private/var).
    let here = &dir.path().canonicalize().unwrap();
    std::fs::create_dir_all(here.join("res")).unwrap();
    std::fs::write(here.join("res/run.pinto.json"), "{}").unwrap();
    std::fs::write(here.join("pairs.tsv"), "A\tB\n").unwrap();
    let mut cli = crate::Cli::command();
    cli.build();
    let mut a = App {
        here: here.to_path_buf(),
        ..App::new(cli, here.to_path_buf()).unwrap()
    };
    a.browser = None;
    a.editor = None;
    let names: Vec<&str> = a.rows.iter().map(|r| r.form.name.as_str()).collect();
    assert_eq!(names[..5], ["lc", "cage", "dsvd", "lra", "lra-simulate"]);
    let lra = 3;
    assert!(a.rows[lra].form.from_run && !a.rows[lra].form.takes_coords());

    a.rows[lra].on = true;
    a.rows[lra].out = "out/lra".into();
    // Without --from, the plan says so and blames it.
    let p = &a.plan()[0];
    assert!(
        p.problem.as_ref().unwrap().contains("--from"),
        "{:?}",
        p.problem
    );
    assert_eq!(p.blamed.as_deref(), Some("from"));
    let set = |a: &mut App, long: &str, v: &str| {
        let f = a.rows[lra]
            .form
            .fields
            .iter_mut()
            .find(|f| f.long == long)
            .unwrap();
        f.value = v.into();
    };
    // A run's prefix is taken for its manifest.
    set(&mut a, "from", "res/run");
    set(&mut a, "lr-pairs", "pairs.tsv");
    set(&mut a, "only", "L1:R1, L2");
    set(&mut a, "seed", "7");
    // A value clap refuses is what the plan says, not a missing manifest.
    set(&mut a, "seed", "lots");
    let p = &a.plan()[0];
    let why = p.problem.as_deref().unwrap();
    assert!(why.contains("lots") && !why.contains("manifest"), "{why}");
    assert_eq!(p.blamed.as_deref(), Some("seed"));
    set(&mut a, "seed", "7");
    let planned = a.plan();
    let p = &planned[0];
    assert!(p.problem.is_none(), "{:?}", p.problem);
    let j = &p.job;
    assert_eq!(j.dir, here.join("out"));
    assert_eq!(
        j.argv,
        [
            "lra",
            "--out",
            "lra",
            "--from",
            "../res/run.pinto.json",
            "--seed",
            "7",
            "--lr-pairs",
            "../pairs.tsv",
            "--only",
            "L1:R1,L2"
        ]
    );
    assert_eq!(j.manifest(), j.dir.join("lra.lra.only-l1-r1+l2.parquet"));
    assert_eq!(j.script(), j.dir.join("lra.cmd.sh"));
    assert_eq!(
        script::normalize(j.view()),
        script::normalize(&here.join("res/run.pinto.json"))
    );
    // It owns its script alone: never a batch folder or a manifest.
    assert_eq!(j.leftovers, [j.dir.join("lra.cmd.sh")]);
    form::check(&a.cli, &j.argv).unwrap();
    // The script refuses to run over the table, not over a manifest.
    let text = script::text(&j.argv, &j.out, &j.done_name(), "lra.cmd.sh");
    assert!(
        text.contains("if [ -f lra.lra.only-l1-r1+l2.parquet ]"),
        "{text}"
    );
    assert!(text.contains("bash lra.cmd.sh"), "{text}");

    // lra-simulate: its own table.
    let sim = 4;
    a.rows[lra].on = false;
    a.rows[sim].on = true;
    a.rows[sim].out = "sim".into();
    let f = a.rows[sim]
        .form
        .fields
        .iter_mut()
        .find(|f| f.long == "from")
        .unwrap();
    f.value = "res/run.pinto.json".into();
    let p = &a.plan()[0];
    assert!(p.problem.is_none(), "{:?}", p.problem);
    assert_eq!(
        p.job.argv,
        [
            "lra-simulate",
            "--out",
            "sim",
            "--from",
            "res/run.pinto.json"
        ]
    );
    assert_eq!(p.job.manifest(), here.join("sim.lra-simulate.parquet"));
    // A table there already is in the way.
    std::fs::write(here.join("sim.lra-simulate.parquet"), "").unwrap();
    let p = &a.plan()[0];
    assert!(
        p.problem.as_ref().unwrap().contains("exists"),
        "{:?}",
        p.problem
    );
    // A --from that names no run.
    std::fs::remove_file(here.join("sim.lra-simulate.parquet")).unwrap();
    let f = a.rows[sim]
        .form
        .fields
        .iter_mut()
        .find(|f| f.long == "from")
        .unwrap();
    f.value = "nowhere".into();
    let p = &a.plan()[0];
    assert!(
        p.problem.as_ref().unwrap().contains("no such run"),
        "{:?}",
        p.problem
    );
}

/// `pinto run` on pinto's own commands, started in `here`.
fn pinto_app(here: &Path) -> App {
    use clap::CommandFactory;
    let mut cli = crate::Cli::command();
    cli.build();
    let mut a = App {
        here: here.to_path_buf(),
        ..App::new(cli, here.to_path_buf()).unwrap()
    };
    a.browser = None;
    a.editor = None;
    a
}

/// An lc run at `{here}/res/fit` on two data files beside it, its
/// coordinate columns `cx`, `cy`: the folder made in `here`, canonical.
fn fitted_run(dir: &Path) -> PathBuf {
    let here = dir.canonicalize().unwrap();
    std::fs::create_dir_all(here.join("res")).unwrap();
    for f in ["d1.zarr", "d2.zarr"] {
        std::fs::write(here.join("res").join(f), "").unwrap();
    }
    let prefix = here.join("res/fit").to_string_lossy().into_owned();
    let mut meta = crate::util::metadata::PintoMetadata {
        command: "lc".into(),
        prefix: prefix.clone(),
        data_files: Some(vec!["d1.zarr".into(), "d2.zarr".into()]),
        ..Default::default()
    };
    meta.outputs.coord_columns = Some(vec!["cx".into(), "cy".into()]);
    meta.write(Path::new(&format!("{prefix}.pinto.json")))
        .unwrap();
    here
}

/// The row whose label is `label`.
fn row(a: &App, label: &str) -> usize {
    a.rows
        .iter()
        .position(|r| r.label() == label)
        .unwrap_or_else(|| panic!("no {label} row"))
}

/// Set flag `long` of row `m` to `v`, as typed.
fn set_flag(a: &mut App, m: usize, long: &str, v: &str) {
    let f = a.rows[m]
        .form
        .fields
        .iter_mut()
        .find(|f| f.long == long)
        .unwrap_or_else(|| panic!("no --{long}"));
    f.value = v.into();
}

/// The value after `flag` in `argv`.
fn after<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
    let k = argv.iter().position(|w| w == flag)?;
    argv.get(k + 1).map(String::as_str)
}

/// What `pinto residual --from {here}/res/fit -o {here}/{out}` plans.
fn residual_plan(here: &Path, out: &str) -> crate::residual::Plan {
    use clap::Parser;
    let run = crate::util::metadata::Run::open(&here.join("res/fit").to_string_lossy()).unwrap();
    let cli = crate::Cli::try_parse_from([
        "pinto",
        "residual",
        "--from",
        "res/fit",
        "-o",
        &here.join(out).to_string_lossy(),
    ])
    .unwrap();
    let crate::Commands::Residual(args) = cli.commands else {
        panic!("not residual")
    };
    crate::residual::Plan::new(&run, &args).unwrap()
}

#[test]
fn the_residual_row_is_listed_only_where_pinto_has_the_command() {
    let dir = tempfile::tempdir().unwrap();
    // The stand-in pinto has no residual: none of its rows are listed.
    let labels: Vec<String> = app(dir.path()).rows.iter().map(Row::label).collect();
    assert_eq!(labels, METHODS);
    let a = pinto_app(dir.path());
    let labels: Vec<String> = a.rows.iter().map(Row::label).collect();
    assert_eq!(
        labels,
        [
            "lc",
            "cage",
            "dsvd",
            "lra",
            "lra-simulate",
            "residual",
            "stage-2 dsvd",
            "stage-2 lra"
        ]
    );
    let r = row(&a, "residual");
    assert!(a.rows[r].form.from_run && !a.rows[r].form.takes_coords());
    assert!(!a.rows[row(&a, "stage-2 dsvd")].form.from_run);
    // --run is still taken.
    form::check(
        &a.cli,
        &[
            "residual".into(),
            "--run".into(),
            "r".into(),
            "-o".into(),
            "o".into(),
        ],
    )
    .unwrap();
}

/// A residual queued on a run sets up the second stage: a fit on exactly
/// the files the residual writes, with the run's coordinate columns,
/// started only once the residual is done.
#[test]
fn a_residual_sets_up_a_stage_two_fit_on_its_files() {
    let dir = tempfile::tempdir().unwrap();
    let here = &fitted_run(dir.path());
    let mut a = pinto_app(here);
    let (res, two) = (row(&a, "residual"), row(&a, "stage-2 dsvd"));
    a.screen = Screen::Methods;
    a.method_row = res;
    key(&mut a, KeyCode::Char(' '));
    assert!(
        a.rows[res].on && a.rows[two].on,
        "space on residual queues stage 2"
    );
    // Without --from, the residual says so and stage 2 waits on it.
    let p = a.plan();
    assert_eq!(p.len(), 2);
    assert!(p[0].problem.as_deref().unwrap().contains("--from"));
    assert!(
        p[1].problem.as_deref().unwrap().contains("residual"),
        "{:?}",
        p[1].problem
    );

    set_flag(&mut a, res, "from", "res/fit");
    let planned = a.plan();
    for p in &planned {
        assert!(p.problem.is_none(), "{}: {:?}", p.job.method, p.problem);
        form::check(&a.cli, &p.job.argv).unwrap();
    }
    let plan = residual_plan(here, &a.rows[res].out);
    let (r, s) = (&planned[0].job, &planned[1].job);
    assert_eq!(
        r.argv,
        [
            "residual",
            "--out",
            a.rows[res].out.as_str(),
            "--from",
            "res/fit.pinto.json"
        ]
    );
    // Done when its residual.json is there.
    assert_eq!(r.manifest(), Path::new(&plan.summary));
    assert_eq!(r.after, None);

    assert_eq!(s.method, "stage-2 dsvd");
    assert_eq!(s.after, Some(0), "stage 2 waits on the residual");
    let rel = |p: &String| {
        script::relative(Path::new(p), &s.dir)
            .to_string_lossy()
            .into_owned()
    };
    let zarrs: Vec<String> = plan.zarrs.iter().map(rel).collect();
    let coords: Vec<String> = plan.coords.iter().map(rel).collect();
    assert_eq!(zarrs.len(), 2);
    assert_eq!(s.argv[0], "dsvd");
    assert_eq!(s.argv[1..3], zarrs[..]);
    assert_eq!(after(&s.argv, "--coord"), Some(coords.join(",").as_str()));
    assert_eq!(
        after(&s.argv, "--coord-column-names"),
        Some(plan.coord_names.join(",").as_str())
    );
    assert_eq!(plan.coord_names.join(","), "cx,cy");
    assert_eq!(s.manifest(), s.dir.join(format!("{}.pinto.json", s.out)));
    // The picked data are not stage 2's.
    a.pairs = data(here, &["other.zarr"]);
    let s = &a.plan()[1].job;
    assert!(
        !s.argv.contains(&"../other.zarr".to_string())
            && !s.argv.contains(&"other.zarr".to_string())
    );

    // Stage 2 alone has nothing to fit.
    a.rows[res].on = false;
    let p = a.plan();
    assert_eq!(p.len(), 1);
    assert!(p[0].problem.as_deref().unwrap().contains("residual"));
    // A residual already written is in the way.
    a.rows[res].on = true;
    std::fs::create_dir_all(Path::new(&plan.summary).parent().unwrap()).unwrap();
    std::fs::write(&plan.zarrs[0], "").unwrap();
    let p = a.plan();
    assert!(
        p[0].problem.as_deref().unwrap().contains("exists"),
        "{:?}",
        p[0].problem
    );
}

#[test]
fn the_stage_two_method_is_chosen_on_the_methods_screen() {
    let dir = tempfile::tempdir().unwrap();
    let here = &fitted_run(dir.path());
    let mut a = pinto_app(here);
    let res = row(&a, "residual");
    let two = row(&a, "stage-2 dsvd");
    a.rows[res].on = true;
    a.rows[two].on = true;
    set_flag(&mut a, res, "from", "res/fit");
    a.screen = Screen::Methods;
    a.method_row = two;
    let mut seen = Vec::new();
    for _ in 0..3 {
        key(&mut a, KeyCode::Char('m'));
        let p = &a.plan()[1];
        assert!(p.problem.is_none(), "{:?}", p.problem);
        seen.push(p.job.argv[0].clone());
        assert_eq!(p.job.method, format!("stage-2 {}", p.job.argv[0]));
        assert_eq!(
            a.rows[two].out,
            under("", &format!("residual-{}", p.job.argv[0]))
        );
        // Every one of them reads the residual's coordinate columns.
        assert_eq!(after(&p.job.argv, "--coord-column-names"), Some("cx,cy"));
    }
    assert_eq!(seen, ["lc", "cage", "dsvd"]);
    // Each keeps its own flags.
    key(&mut a, KeyCode::Char('m'));
    let lc = a.rows[two].form.name.clone();
    assert_eq!(lc, "lc");
    let f = a.rows[two]
        .form
        .fields
        .iter()
        .position(|f| f.long == "seed")
        .unwrap();
    a.rows[two].form.fields[f].value = "11".into();
    for _ in 0..3 {
        key(&mut a, KeyCode::Char('m'));
    }
    assert_eq!(a.rows[two].form.name, "lc");
    assert_eq!(after(&a.plan()[1].job.argv, "--seed"), Some("11"));
    // m does nothing to another row.
    a.method_row = res;
    key(&mut a, KeyCode::Char('m'));
    assert_eq!(a.rows[res].form.name, "residual");
}

/// lra on the residual's run is offered on the stage-2 fit too, with its
/// flags, reading the stage-2 run once it is fitted.
#[test]
fn lra_on_the_stage_two_fit_reads_it() {
    let dir = tempfile::tempdir().unwrap();
    let here = &fitted_run(dir.path());
    let mut a = pinto_app(here);
    let (lra, res, two, lra2) = (
        row(&a, "lra"),
        row(&a, "residual"),
        row(&a, "stage-2 dsvd"),
        row(&a, "stage-2 lra"),
    );
    a.screen = Screen::Methods;
    a.method_row = res;
    key(&mut a, KeyCode::Char(' '));
    set_flag(&mut a, res, "from", "res/fit");
    // Not offered without lra on the same run.
    a.method_row = lra2;
    key(&mut a, KeyCode::Char(' '));
    assert!(!a.rows[lra2].on);
    assert!(
        a.message.as_deref().unwrap().contains("lra"),
        "{:?}",
        a.message
    );
    a.rows[lra].on = true;
    set_flag(&mut a, lra, "from", "res/fit.pinto.json");
    set_flag(&mut a, lra, "seed", "7");
    key(&mut a, KeyCode::Char(' '));
    assert!(a.rows[lra2].on, "one key queues it");
    let planned = a.plan();
    let methods: Vec<&str> = planned.iter().map(|p| p.job.method.as_str()).collect();
    assert_eq!(methods, ["lra", "residual", "stage-2 dsvd", "stage-2 lra"]);
    for p in &planned {
        assert!(p.problem.is_none(), "{}: {:?}", p.job.method, p.problem);
        form::check(&a.cli, &p.job.argv).unwrap();
    }
    let (s, l) = (&planned[2].job, &planned[3].job);
    assert_eq!(l.after, Some(2), "it waits on the stage-2 fit");
    let from = after(&l.argv, "--from").unwrap();
    assert_eq!(
        script::normalize(&l.dir.join(from)),
        script::normalize(s.manifest())
    );
    assert_eq!(script::normalize(l.view()), script::normalize(s.manifest()));
    // lra's flags came with it.
    assert_eq!(after(&l.argv, "--seed"), Some("7"));
    assert!(l.manifest().to_string_lossy().ends_with(".lra.parquet"));
    // lra on another run: no longer offered.
    std::fs::create_dir_all(here.join("res2")).unwrap();
    std::fs::copy(
        here.join("res/fit.pinto.json"),
        here.join("res2/fit.pinto.json"),
    )
    .unwrap();
    set_flag(&mut a, lra, "from", "res2/fit");
    let p = a.plan();
    assert!(
        p[3].problem.as_deref().unwrap().contains("lra"),
        "{:?}",
        p[3].problem
    );
    let _ = two;
}

/// The commands copied from the review replay the plan in its order, and
/// each script refuses to run over what says its run is done.
#[test]
fn the_copied_commands_replay_the_residual_then_stage_two() {
    let dir = tempfile::tempdir().unwrap();
    let here = &fitted_run(dir.path());
    let mut a = pinto_app(here);
    let res = row(&a, "residual");
    a.screen = Screen::Methods;
    a.method_row = res;
    key(&mut a, KeyCode::Char(' '));
    set_flag(&mut a, res, "from", "res/fit");
    run_key(&mut a);
    let text = a.copy_text();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{text}");
    assert!(lines[0].starts_with("pinto residual "), "{text}");
    // a pasted replay waits as the queue does: the stage-2 fit runs only
    // once the residual's summary is there
    let done = &a.confirm.as_ref().unwrap().planned[0].job.done_name();
    let guard = format!("[ -f {} ] && pinto dsvd ", script::quote(done));
    assert!(
        lines[1].starts_with(&guard) && lines[1].contains("--coord-column-names cx,cy"),
        "{text}"
    );
    let planned = &a.confirm.as_ref().unwrap().planned;
    for p in planned {
        let j = &p.job;
        let name = j
            .script()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let t = script::text(&j.argv, &j.out, &j.done_name(), &name);
        assert!(
            t.contains(&format!("if [ -f {} ]", script::quote(&j.done_name()))),
            "{t}"
        );
    }
    assert!(planned[0].job.done_name().ends_with("residual.json"));
}

/// A residual that finished while its stage-2 fit (or lra on it) did not
/// keeps its --out, and the run key again fits what it wrote without
/// making it again.
#[cfg(unix)]
#[test]
fn a_stage_two_fit_that_failed_runs_again_on_the_finished_residual() {
    let dir = tempfile::tempdir().unwrap();
    let here = &fitted_run(dir.path());
    let mut a = pinto_app(here);
    let (lra, res, two, lra2) = (
        row(&a, "lra"),
        row(&a, "residual"),
        row(&a, "stage-2 dsvd"),
        row(&a, "stage-2 lra"),
    );
    a.screen = Screen::Methods;
    a.method_row = res;
    key(&mut a, KeyCode::Char(' '));
    set_flag(&mut a, res, "from", "res/fit");
    a.rows[lra].on = true;
    set_flag(&mut a, lra, "from", "res/fit");
    a.method_row = lra2;
    key(&mut a, KeyCode::Char(' '));
    let planned = a.plan();
    assert_eq!(planned.len(), 4);
    let res_out = a.rows[res].out.clone();
    // A stand-in pinto that finishes lra and the residual, and fails the
    // stage-2 fit.
    let fake = fake_pinto(
        here,
        "case \"$1\" in lra) ;; residual) mkdir -p \"$3\" && touch \"$3/residual.json\" ;; *) exit 1 ;; esac",
    );
    let jobs: Vec<jobs::Job> = planned.into_iter().map(|p| p.job).collect();
    a.queue = Some(Queue::start(jobs, fake));
    while a.running() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    a.poll();
    assert!(a.rows[res].done && !a.rows[res].on);
    assert!(a.rows[two].on && a.rows[lra2].on);
    assert_eq!(a.rows[res].out, res_out, "kept for the stage-2 fit");
    let p = a.plan();
    let methods: Vec<&str> = p.iter().map(|p| p.job.method.as_str()).collect();
    assert_eq!(methods, ["stage-2 dsvd", "stage-2 lra"]);
    for q in &p {
        assert!(q.problem.is_none(), "{}: {:?}", q.job.method, q.problem);
    }
    // Nothing to wait on in this queue: the residual is there.
    assert_eq!(p[0].job.after, None);
    assert_eq!(p[1].job.after, Some(0));
    let plan = residual_plan(here, &res_out);
    let zarrs: Vec<String> = plan
        .zarrs
        .iter()
        .map(|z| {
            script::relative(Path::new(z), &p[0].job.dir)
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(p[0].job.argv[1..3], zarrs[..]);
    // A residual no longer there is not read: stage 2 asks for one.
    std::fs::remove_file(&plan.summary).unwrap();
    let p = a.plan();
    assert!(
        p[0].problem.as_deref().unwrap().contains("queue residual"),
        "{:?}",
        p[0].problem
    );
}

/// The queued residual, lra and lra on the stage-2 fit: the app and the
/// rows' indices (lra, residual, stage 2, stage-2 lra).
fn two_stage_with_lra(here: &Path) -> (App, usize, usize, usize, usize) {
    let mut a = pinto_app(here);
    let (lra, res, two, lra2) = (
        row(&a, "lra"),
        row(&a, "residual"),
        row(&a, "stage-2 dsvd"),
        row(&a, "stage-2 lra"),
    );
    a.screen = Screen::Methods;
    a.method_row = res;
    key(&mut a, KeyCode::Char(' '));
    set_flag(&mut a, res, "from", "res/fit");
    a.rows[lra].on = true;
    set_flag(&mut a, lra, "from", "res/fit");
    a.method_row = lra2;
    key(&mut a, KeyCode::Char(' '));
    assert!(a.rows[two].on && a.rows[lra2].on);
    (a, lra, res, two, lra2)
}

/// Taking the residual off takes off what reads it: the stage-2 fit and
/// lra on it.
#[test]
fn taking_the_residual_off_takes_its_stage_two_rows_off() {
    let dir = tempfile::tempdir().unwrap();
    let here = &fitted_run(dir.path());
    let (mut a, lra, res, two, lra2) = two_stage_with_lra(here);
    a.method_row = res;
    key(&mut a, KeyCode::Char(' '));
    assert!(!a.rows[res].on && !a.rows[two].on && !a.rows[lra2].on);
    assert!(a.rows[lra].on, "lra on the first run stays");
}

/// A stage-2 fit that finished while lra on it did not keeps its --out,
/// and the run key again runs only that lra, on the kept fit.
#[cfg(unix)]
#[test]
fn lra_on_a_finished_stage_two_fit_runs_again_on_it() {
    let dir = tempfile::tempdir().unwrap();
    let here = &fitted_run(dir.path());
    let (mut a, _, _, two, lra2) = two_stage_with_lra(here);
    let two_out = a.rows[two].out.clone();
    // A stand-in pinto that finishes lra on the first run, the residual
    // and the stage-2 fit, and fails lra on the stage-2 fit (the one
    // reading a residual fit).
    let fake = fake_pinto(
        here,
        "case \"$1\" in \
         lra) case \"$*\" in *residual*) exit 1 ;; esac ;; \
         residual) mkdir -p \"$3\" && touch \"$3/residual.json\" ;; \
         *) while [ $# -gt 0 ]; do [ \"$1\" = --out ] && touch \"$2.pinto.json\"; shift; done ;; \
         esac",
    );
    let jobs: Vec<jobs::Job> = a.plan().into_iter().map(|p| p.job).collect();
    assert_eq!(jobs.len(), 4);
    a.queue = Some(Queue::start(jobs, fake));
    while a.running() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    a.poll();
    assert!(
        a.rows[two].done && !a.rows[two].on,
        "the stage-2 fit is done"
    );
    assert!(a.rows[lra2].on, "lra on it is still to do");
    assert_eq!(a.rows[two].out, two_out, "kept for lra on it");
    let p = a.plan();
    let methods: Vec<&str> = p.iter().map(|p| p.job.method.as_str()).collect();
    assert_eq!(methods, ["stage-2 lra"]);
    assert!(p[0].problem.is_none(), "{:?}", p[0].problem);
    assert_eq!(p[0].job.after, None, "nothing left to wait on");
    let from = after(&p[0].job.argv, "--from").unwrap();
    assert!(from.ends_with(&format!("{two_out}.pinto.json")), "{from}");
}

/// The residual's default --out moves past a residual already written in
/// the folder, as a fit's moves past a fit.
#[test]
fn the_residual_out_moves_past_a_residual_already_there() {
    let dir = tempfile::tempdir().unwrap();
    let here = &fitted_run(dir.path());
    // outs named in the run's own folder, as pinto run started there would
    let named = |here: &Path| {
        let mut a = pinto_app(here);
        a.refresh_outs();
        a
    };
    let first = named(here);
    let taken = first.rows[row(&first, "residual")].out.clone();
    std::fs::create_dir_all(here.join(&taken)).unwrap();
    std::fs::write(here.join(&taken).join("residual.json"), "{}").unwrap();
    let a = named(here);
    let out = &a.rows[row(&a, "residual")].out;
    assert_ne!(out, &taken, "{out}");
    assert_eq!(out, &format!("{taken}-2"));
}

/// Switching a finished stage-2 row to another method clears its done
/// mark: that method has not run.
#[test]
fn a_new_stage_two_method_is_not_marked_done() {
    let dir = tempfile::tempdir().unwrap();
    let here = &fitted_run(dir.path());
    let mut a = pinto_app(here);
    let two = row(&a, "stage-2 dsvd");
    a.rows[two].done = true;
    a.screen = Screen::Methods;
    a.method_row = two;
    key(&mut a, KeyCode::Char('m'));
    assert!(!a.rows[two].done);
}
