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
    assert!(a.confirm.is_some());
    key(&mut a, KeyCode::Enter);
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
    a.confirm = Some(Vec::new());
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
    let job = |m: &str| jobs::Job {
        method: m.into(),
        dir: dir.path().to_path_buf(),
        out: m.into(),
        argv: vec![m.into(), "--out".into(), m.into()],
        made: Vec::new(),
        clear: false,
    };
    // A stand-in pinto that writes the manifest its --out names, or fails.
    let fake = dir.path().join("fake-pinto");
    std::fs::write(
        &fake,
        "#!/bin/sh\n[ \"$1\" = bad ] && exit 1\nwhile [ $# -gt 0 ]; do [ \"$1\" = --out ] && touch \"$2.pinto.json\"; shift; done\n",
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
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
    a.confirm = Some(a.plan());
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
fn a_started_queue_moves_default_outs_on_so_g_again_is_not_blocked() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(next_out(dir.path(), "lc", Some("lc")), "lc-2");
    let mut a = app(dir.path());
    a.pairs = data(dir.path(), &["d.zarr"]);
    let lc = METHODS.iter().position(|m| *m == "lc").unwrap();
    a.rows[lc].on = true;
    a.rows[METHODS.iter().position(|m| *m == "cage").unwrap()].on = true;
    let cage = METHODS.iter().position(|m| *m == "cage").unwrap();
    a.rows[cage].out = "mine".into();
    a.rows[cage].typed = true;
    a.move_outs_on();
    assert_eq!(a.rows[lc].out, "lc-2");
    assert_eq!(
        a.rows[METHODS.iter().position(|m| *m == "cage").unwrap()].out,
        "mine",
        "a name the user typed is theirs"
    );
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
    a.editor.as_mut().unwrap().text = "mine".into();
    key(&mut a, KeyCode::Enter);
    assert!(a.rows[lc].typed);

    std::fs::write(dir.path().join("exp1_cage.pinto.json"), "{}").unwrap();
    key(&mut a, KeyCode::Char('O'));
    a.editor.as_mut().unwrap().text = "exp1".into();
    key(&mut a, KeyCode::Enter);
    let cage = METHODS.iter().position(|m| *m == "cage").unwrap();
    assert_eq!(a.rows[lc].out, "mine");
    assert_eq!(a.rows[cage].out, "exp1_cage-2", "made unique");

    key(&mut a, KeyCode::Char('o'));
    a.editor.as_mut().unwrap().text.clear();
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
    a.editor.as_mut().unwrap().text = "exp1".into();
    key(&mut a, KeyCode::Enter);
    key(&mut a, KeyCode::Char('O'));
    a.editor.as_mut().unwrap().text = "other".into();
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
    a.editor.as_mut().unwrap().text = "res/sub/".into();
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
fn methods_a_run_finished_are_unqueued_so_g_again_runs_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = app(dir.path());
    let (lc, cage) = (0, 1);
    a.pairs = data(dir.path(), &["d.zarr"]);
    a.rows[lc].on = true;
    a.rows[cage].on = true;
    // As a started queue leaves them: each run's --out kept, the rows
    // moved on.
    for r in &mut a.rows {
        r.last = Some(r.out.clone());
        r.out = format!("{}-2", r.out);
    }
    let job = |m: &str| jobs::Job {
        method: m.into(),
        dir: dir.path().to_path_buf(),
        out: m.into(),
        argv: vec![m.into(), "--out".into(), m.into()],
        made: Vec::new(),
        clear: false,
    };
    // A stand-in pinto that finishes the first method and fails the next.
    let fake = dir.path().join("fake-pinto");
    std::fs::write(&fake, format!("#!/bin/sh\n[ \"$1\" = {} ]\n", METHODS[lc])).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    a.queue = Some(Queue::start(
        vec![job(METHODS[lc]), job(METHODS[cage])],
        fake,
    ));
    while a.running() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    a.poll();
    assert!(!a.rows[lc].on && a.rows[lc].done);
    assert!(a.rows[cage].on && !a.rows[cage].done);
    assert!(a.message.as_deref().unwrap().contains("G runs the rest"));
    // The failed one goes back to its --out, its script there cleared
    // when it runs again.
    assert_eq!(a.rows[cage].out, METHODS[cage]);
    assert!(dir
        .path()
        .join(format!("{}.cmd.sh", METHODS[cage]))
        .exists());
    let p = a.plan();
    assert_eq!(p.len(), 1);
    assert!(p[0].problem.is_none(), "{:?}", p[0].problem);
    assert!(p[0].job.clear && !p[0].again);
    // An --out typed anew clears nothing.
    a.rows[cage].out = "mine".into();
    assert!(!a.plan()[0].job.clear);
    a.rows[cage].out = METHODS[cage].into();
    // r on the Run screen reviews what is left.
    a.screen = Screen::Run;
    key(&mut a, KeyCode::Char('r'));
    assert_eq!(a.confirm.as_ref().map(Vec::len), Some(1));
    key(&mut a, KeyCode::Esc);
    // Queued again by hand, it stays queued.
    a.screen = Screen::Methods;
    a.method_row = lc;
    key(&mut a, KeyCode::Char(' '));
    a.poll();
    assert!(a.rows[lc].on);
    assert!(a.plan().iter().any(|p| p.again), "finished before: said so");
}
