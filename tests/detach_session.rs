//! `pinto run` and `pinto view` as detachable sessions, driven through
//! pseudo-terminals standing in for the user's: each starts hosted, a
//! detach leaves it running and asks for a name, an attach from another
//! terminal repaints it, a TUI started under a session's marker runs
//! plain, and `pinto kill` stops it through its own quit path, leaving no
//! process, socket or label behind. `--no-detach` runs plain, as before.

#![cfg(all(unix, feature = "view"))]

mod common;

use common::write_run;
use legume_tui::testing::{alive, Private, Replies, Setup, Term, PROMPT};
use legume_tui::Size;
use std::time::Duration;

/// The run screen's tab of flags: drawn on every full paint.
const PAINTED: &str = "3 Parameters";

/// The size every test terminal has but where one says otherwise.
const SIZE: Size = Size {
    rows: 40,
    cols: 120,
    width: 1200,
    height: 800,
};

/// A private directory for pinto's sessions; a session a failed test left
/// running is killed with it.
fn private() -> Private {
    Private::new(env!("CARGO_BIN_EXE_pinto")).patience(Duration::from_secs(30))
}

fn term(dir: &Private, args: &[&str], env: &[(&str, &str)]) -> Term {
    dir.term_sized(args, env, SIZE)
}

/// [`term`], on a terminal that answers the program's queries with
/// `replies`.
fn term_answering(dir: &Private, args: &[&str], env: &[(&str, &str)], replies: Replies) -> Term {
    dir.term_with(args, env, Setup::new().size(SIZE).replies(replies))
}

/// Detach `t` by the key, keeping the session unnamed, and wait for it to
/// be listed as `id` detached.
fn detach(dir: &Private, t: &mut Term, id: &str) {
    t.type_(b"\x1b\x04");
    t.answer(1, b"\r");
    assert_eq!(t.exit_code(), 0);
    assert!(t.cooked(), "terminal restored after detach");
    dir.wait_listed(&format!("{id}\tdetached"));
}

#[test]
fn run_detaches_attaches_and_is_killed_through_its_quit_path() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let work = work.path().to_string_lossy().into_owned();

    // Started in a terminal, it is hosted: this terminal is its first
    // client, and the program runs apart from it.
    let mut first = term(&dir, &["run", &work], &[]);
    first.wait_screen(PAINTED);
    assert!(!first.cooked(), "the client's terminal is raw");
    let listed = dir.wait_listed("setting up fits");
    assert!(listed.contains("pinto-run-1\tattached"), "{listed:?}");
    let pid = dir.program_pid();
    assert_ne!(
        pid,
        first.child.id(),
        "the program runs apart from its client"
    );

    // The client lost without a word (as on a dropped connection): the
    // program runs on, detached.
    first.child.kill().unwrap();
    first.child.wait().unwrap();
    drop(first);
    dir.wait_listed("pinto-run-1\tdetached");
    assert!(alive(pid));

    // Attached from a terminal of another size: painted whole there, after
    // the client's reset.
    let size = Size {
        rows: 50,
        cols: 150,
        width: 1500,
        height: 1000,
    };
    // This one has kitty's keyboard enhancement: shift-enter is told from
    // enter there, and the help says so.
    let replies = Replies::plain().keyboard(0);
    let setup = Setup::new().size(size).replies(replies);
    let mut second = dir.term_with(&["attach"], &[], setup);
    for tab in ["1 Data", "2 Methods", PAINTED] {
        second.wait_screen(tab);
    }
    assert!(second.alternate());
    // The keys of the session are listed with the others.
    second.wait_screen("ctrl+alt+d detach");
    second.wait_screen("ctrl+alt+s switch");
    second.wait_screen("tab / shift-tab / shift-enter / 1-4 screens");
    // Keys reach the program: typed into the output header it asks first.
    second.type_(b"zq");
    second.wait_screen("zq▏");

    // A TUI started under the session's marker, as `pinto run` starts
    // `pinto view` when `v` asks for it, runs plain: no second host, no
    // second socket.
    let mut nested = term(&dir, &["view"], &[(legume_tui::SESSION_VAR, "pinto-run-1")]);
    nested.wait_screen("run to open");
    assert_eq!(dir.sockets(), ["pinto-run-1"]);
    nested.type_(b"\x03");
    assert_eq!(nested.exit_code(), 0);
    assert_eq!(dir.sockets(), ["pinto-run-1"]);

    detach(&dir, &mut second, "pinto-run-1");
    second.wait_for("Detached pinto-run-1.");

    // Renamed from a shell: listed and found by its label.
    let said = dir.run(&["rename", "run-1", "fits"]);
    assert!(said.contains("pinto-run-1 is now fits"), "{said:?}");
    dir.wait_listed("fits (pinto-run-1)\tdetached");

    // Killed: it quits as ctrl+c does, and leaves nothing behind.
    let said = dir.run(&["kill", "fits"]);
    assert!(said.contains("pinto-run-1 stopped"), "{said:?}");
    assert!(dir.sockets().is_empty());
    assert!(dir.is_empty(), "no label or status left");
    dir.gone(pid);
    assert!(dir.run(&["sessions"]).contains("no live pinto sessions"));
}

#[test]
fn run_and_view_opted_out_run_plain() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let work = work.path().to_string_lossy().into_owned();

    // By the flag or the variable: in this process, no client between, no
    // socket, and ctrl+c ends it with the terminal restored.
    for (args, env) in [
        (vec!["run", &work, "--no-detach"], vec![]),
        (vec!["run", &work], vec![(legume_tui::NO_DETACH_VAR, "1")]),
        (vec!["view", "--no-detach"], vec![]),
    ] {
        let mut plain = term(&dir, &args, &env);
        plain.wait_screen(if args[0] == "run" {
            PAINTED
        } else {
            "run to open"
        });
        if args[0] == "run" {
            // A terminal that does not say it has kitty's keyboard: no
            // shift-enter.
            plain.wait_screen("tab / shift-tab / 1-4 screens");
        }
        assert!(!plain.cooked());
        assert!(dir.sockets().is_empty(), "{args:?} {env:?}");
        assert!(dir.run(&["sessions"]).contains("no live pinto sessions"));
        assert!(!plain.text().contains("\x1b_Ga=d,d=A"), "no client's reset");
        // No session, no session keys on the screen.
        assert!(!plain.text().contains("ctrl+alt+d"), "{args:?}");
        plain.type_(b"\x03");
        assert_eq!(plain.exit_code(), 0);
        assert!(plain.cooked(), "terminal restored");
    }
}

/// The view's panel title, drawn on every full paint.
const VIEW_PAINTED: &str = "pinto view  run";

/// The last of the terminal's graphics query: its background colour.
const ASKED: &str = "\x1b]11;?\x07";

/// A terminal with cells of 10 by 20 pixels and a white background, no
/// image protocol.
fn light() -> Replies {
    Replies::plain().cell(10, 20).background(255, 255, 255)
}

/// The same, showing kitty's images.
fn kitty_light() -> Replies {
    light().kitty()
}

/// The background of the map in the dark and the light theme, as the
/// colour of a block character's cell.
const DARK_MAP: &str = "17;17;20";
const LIGHT_MAP: &str = "238;238;236";

/// Wait for `t` to ask its graphics query after `from`, which its
/// terminal answers. Where it asked.
fn query_at(t: &Term, from: usize) -> usize {
    t.wait_for_after(from, ASKED);
    from + t.text()[from..].find(ASKED).unwrap()
}

/// A temporary directory for the program, and the environment that gives
/// it that one and keeps it off ssh, whatever the tests run in: kitty's
/// images go through files there.
fn kitty_tmp() -> (tempfile::TempDir, Vec<(&'static str, String)>) {
    let tmp = tempfile::tempdir().unwrap();
    let env = vec![
        ("TMPDIR", tmp.path().to_string_lossy().into_owned()),
        ("SSH_CONNECTION", String::new()),
        ("SSH_CLIENT", String::new()),
        ("SSH_TTY", String::new()),
    ];
    (tmp, env)
}

/// The files kitty's pixels went through, in `tmp`, the program's own
/// temporary directory: named for the terminal's graphics protocol.
fn kitty_files(tmp: &std::path::Path) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(tmp)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("tty-graphics-protocol"))
        })
        .collect()
}

#[test]
fn view_comes_back_on_attach_and_stops_on_kill() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let prefix = write_run(work.path(), 400);

    // A run named starts a new session, hosted. The test terminal answers
    // the graphics query with nothing it has: the map is drawn in block
    // characters, dark.
    let (tmp, env) = kitty_tmp();
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let mut first = term(&dir, &["view", &prefix], &env);
    first.wait_screen(VIEW_PAINTED);
    first.wait_screen("quadrants");
    first.wait_for(DARK_MAP);
    let listed = dir.wait_listed("pinto-view-1\tattached");
    assert!(listed.contains("run · final"), "{listed:?}");
    let pid = dir.program_pid();

    // Named as it is left.
    first.type_(b"\x1b\x04");
    first.answer(1, b"tissue\r");
    assert_eq!(first.exit_code(), 0);
    first.wait_for("detached: tissue");
    dir.wait_listed("tissue (pinto-view-1)\tdetached");

    // Back by its label on another terminal: the screen repainted whole
    // after the client's reset, the terminal asked again and the mouse on
    // once more. This one says it is light: the map is drawn light.
    let mut second = term_answering(&dir, &["attach", "tissue"], &[], light());
    second.wait_for("\x1b[?1049h\x1b[H\x1b[2J");
    let reset = second.text().find("\x1b[?1049h\x1b[H\x1b[2J").unwrap();
    let asked = query_at(&second, reset);
    second.wait_for_after(asked, LIGHT_MAP);
    second.wait_screen(VIEW_PAINTED);
    second.wait_screen("quadrants");
    second.wait_for_after(asked, "\x1b[?1003h");
    second.type_(b"\x1b\x04");
    assert_eq!(second.exit_code(), 0);
    dir.wait_listed("tissue (pinto-view-1)\tdetached");

    // On one that shows kitty's images, the map is sent as one.
    let mut third = term_answering(&dir, &["attach", "tissue"], &[], kitty_light());
    third.wait_for("\x1b[?1049h\x1b[H\x1b[2J");
    let reset = third.text().find("\x1b[?1049h\x1b[H\x1b[2J").unwrap();
    let asked = query_at(&third, reset);
    third.wait_for_after(asked, "\x1b_Ga=T,f=32");
    third.wait_screen("kitty");
    // `?` lists the session keys among the view's.
    third.type_(b"?");
    third.wait_screen("ctrl+alt+d  detach");
    third.wait_screen("ctrl+alt+s  switch");

    // Killed while attached: it quits as ctrl+c does, the client ends with
    // it, and nothing is left, kitty's files among it.
    let said = dir.run(&["kill", "tissue"]);
    assert!(said.contains("pinto-view-1 stopped"), "{said:?}");
    assert_eq!(third.exit_code(), 0);
    assert!(third.cooked(), "terminal restored after the program ended");
    assert!(!third.text().contains(PROMPT));
    dir.gone(pid);
    assert!(dir.is_empty(), "left behind in {:?}", dir.path());
    assert!(kitty_files(tmp.path()).is_empty());
}

#[test]
fn a_kill_while_detached_stops_lupin_and_removes_kittys_files() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let prefix = write_run(work.path(), 400);
    std::fs::write(work.path().join("panel.tsv"), "GENE1\tCT1\nGENE2\tCT2\n").unwrap();

    // A lupin that annotates slowly, deaf to the hangup and the closed
    // pipe its parent's end would bring, then writes.
    let tools = tempfile::tempdir().unwrap();
    let started = tools.path().join("started");
    let written = tools.path().join("written");
    let lupin = tools.path().join("lupin");
    std::fs::write(
        &lupin,
        format!(
            "#!/bin/sh\n\
             case \"$1\" in\n\
             --version) echo lupin 9.9.9 ;;\n\
             annotate)\n\
             trap '' HUP PIPE TERM\n\
             echo > '{}'\n\
             sleep 3 </dev/null >/dev/null 2>&1\n\
             echo > '{}' ;;\n\
             esac\n",
            started.display(),
            written.display()
        ),
    )
    .unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&lupin, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let lupin = lupin.to_string_lossy().into_owned();
    let (tmp, mut env) = kitty_tmp();
    env.push(("PINTO_LUPIN", lupin));
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();

    let mut first = term_answering(&dir, &["view", &prefix], &env, kitty_light());
    first.wait_for("\x1b_Ga=T,f=32");
    first.wait_screen("kitty (file)");
    let pid = dir.program_pid();
    assert!(
        !kitty_files(tmp.path()).is_empty(),
        "the map went through a file"
    );

    // `A` and the panel beside the run: lupin annotates.
    first.type_(b"A");
    first.wait_screen("marker panel for level");
    first.type_(b"\r");
    let start = std::time::Instant::now();
    while !started.exists() {
        assert!(
            start.elapsed() < Duration::from_secs(15),
            "lupin never started"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    detach(&dir, &mut first, "pinto-view-1");

    // Killed with no terminal there: lupin is stopped before it writes,
    // and the files kitty never read are gone with the view.
    let said = dir.run(&["kill", "view-1"]);
    assert!(said.contains("pinto-view-1 stopped"), "{said:?}");
    dir.gone(pid);
    assert!(kitty_files(tmp.path()).is_empty());
    std::thread::sleep(Duration::from_secs(4));
    assert!(!written.exists(), "lupin went on after the view");
}

#[test]
fn the_run_browser_comes_back_on_attach_and_stops_on_kill() {
    let dir = private();

    // With no run named it browses for one, hosted.
    let mut first = term(&dir, &["view"], &[]);
    first.wait_screen("run to open");
    dir.wait_listed("pinto-view-1\tattached");
    let pid = dir.program_pid();
    detach(&dir, &mut first, "pinto-view-1");

    // Started again with nothing named, the live session is offered first:
    // Enter takes it, and the browser is drawn there whole.
    let mut second = term(&dir, &["view"], &[]);
    second.wait_screen("pinto-view-1");
    second.type_(b"\r");
    second.wait_for("\x1b[?1049h\x1b[H\x1b[2J");
    let reset = second.text().find("\x1b[?1049h\x1b[H\x1b[2J").unwrap();
    second.wait_for_after(reset, "run to open");
    detach(&dir, &mut second, "pinto-view-1");

    // Killed while detached: the browser gives up as on ctrl+c.
    let said = dir.run(&["kill", "view-1"]);
    assert!(said.contains("pinto-view-1 stopped"), "{said:?}");
    assert!(dir.sockets().is_empty());
    dir.gone(pid);
}
