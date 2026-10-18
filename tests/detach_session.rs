//! `pinto run` and `pinto view` as detachable sessions, driven through
//! pseudo-terminals standing in for the user's: each starts hosted, a
//! detach leaves it running and asks for a name, an attach from another
//! terminal repaints it, a TUI started under a session's marker runs
//! plain, and `pinto kill` stops it through its own quit path, leaving no
//! process, socket or label behind. `--no-detach` runs plain, as before.

#![cfg(all(unix, feature = "view"))]

use legume_tui::testing::{alive, Private, Term, PROMPT};
use legume_tui::Size;
use std::path::Path;
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
    first.wait_for(PAINTED);
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
    let mut second = dir.term_sized(&["attach"], &[], size);
    second.wait_for("\x1b[?1049h\x1b[H\x1b[2J");
    let reset = second.text().find("\x1b[?1049h\x1b[H\x1b[2J").unwrap();
    for tab in ["1 Data", "2 Methods", PAINTED] {
        second.wait_for_after(reset, tab);
    }
    // The keys of the session are listed with the others.
    second.wait_for("ctrl+alt+d detach");
    second.wait_for("ctrl+alt+s switch");
    // Keys reach the program: typed into the output header it asks first
    // (the second key drawn beside the cursor, the first already there).
    let before = second.seen_len();
    second.type_(b"zq");
    second.wait_for_after(before, "q▏");

    // A TUI started under the session's marker, as `pinto run` starts
    // `pinto view` when `v` asks for it, runs plain: no second host, no
    // second socket.
    let mut nested = term(&dir, &["view"], &[(legume_tui::SESSION_VAR, "pinto-run-1")]);
    nested.wait_for("run to open");
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
        plain.wait_for(if args[0] == "run" {
            PAINTED
        } else {
            "run to open"
        });
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

/// A run of `n` cells on a grid, two communities split down the middle,
/// written as pinto writes them: its cells table, its propensity and its
/// manifest, at `{dir}/run`. The prefix.
fn write_run(dir: &Path, n: usize) -> String {
    use legume_numeric::matrix::traits::IoOps;
    let prefix = dir.join("run").to_string_lossy().into_owned();
    let side = (n as f64).sqrt().ceil() as usize;
    let names: Vec<Box<str>> = (0..n).map(|i| format!("c{i}").into()).collect();
    let xy = nalgebra::DMatrix::<f32>::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i % side) as f32
        } else {
            (i / side) as f32
        }
    });
    let cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    xy.to_parquet_with_names(
        &format!("{prefix}.cells.parquet"),
        (Some(&names), Some("cell")),
        Some(&cols),
    )
    .unwrap();
    let prop = nalgebra::DMatrix::<f32>::from_fn(n, 2, |i, j| {
        let left = i % side < side / 2;
        if left == (j == 0) {
            0.9
        } else {
            0.1
        }
    });
    let cols: Vec<Box<str>> = vec!["C0".into(), "C1".into()];
    prop.to_parquet_with_names(
        &format!("{prefix}.propensity.parquet"),
        (Some(&names), Some("cell")),
        Some(&cols),
    )
    .unwrap();
    let manifest = serde_json::json!({
        "command": "lc",
        "version": "0",
        "timestamp": "",
        "prefix": prefix,
        "n_cells": n,
        "n_features": 0,
        "outputs": {
            "cells": format!("{prefix}.cells.parquet"),
            "propensity": format!("{prefix}.propensity.parquet"),
        },
    });
    std::fs::write(format!("{prefix}.pinto.json"), manifest.to_string()).unwrap();
    prefix
}

/// The view's panel title, drawn on every full paint.
const VIEW_PAINTED: &str = "pinto view  run";

/// The last of the terminal's graphics query: its background colour.
const ASKED: &str = "\x1b]11;?\x07";

/// A terminal's answer to it: cells of 10 by 20 pixels and a white
/// background, no image protocol; the status report ends it.
const LIGHT: &[u8] = b"\x1b[6;20;10t\x1b]11;rgb:ffff/ffff/ffff\x07\x1b[0n";

/// The same from a terminal that shows kitty's images.
const KITTY_LIGHT: &[u8] = b"\x1b_Gi=31;OK\x1b\\\x1b[6;20;10t\x1b]11;rgb:ffff/ffff/ffff\x07\x1b[0n";

/// The background of the map in the dark and the light theme, as the
/// colour of a block character's cell.
const DARK_MAP: &str = "17;17;20";
const LIGHT_MAP: &str = "250;250;248";

/// Wait for `t` to ask its graphics query after `from`, and answer it.
/// Where it asked.
fn answer_query(t: &mut Term, from: usize, answer: &[u8]) -> usize {
    t.wait_for_after(from, ASKED);
    let asked = from + t.text()[from..].find(ASKED).unwrap();
    t.type_(answer);
    asked
}

/// The files kitty's images from the program `pid` go through.
fn kitty_files(pid: u32) -> Vec<std::path::PathBuf> {
    (0..2)
        .map(|slot| {
            std::env::temp_dir().join(format!("tty-graphics-protocol-pinto-{pid}-{slot}.rgba"))
        })
        .collect()
}

#[test]
fn view_comes_back_on_attach_and_stops_on_kill() {
    let dir = private();
    let work = tempfile::tempdir().unwrap();
    let prefix = write_run(work.path(), 400);

    // A run named starts a new session, hosted. The test terminal never
    // answers the graphics query: the map is drawn in block characters,
    // dark.
    let mut first = term(&dir, &["view", &prefix], &[]);
    first.wait_for(VIEW_PAINTED);
    first.wait_for("quadrants");
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
    let mut second = term(&dir, &["attach", "tissue"], &[]);
    second.wait_for("\x1b[?1049h\x1b[H\x1b[2J");
    let reset = second.text().find("\x1b[?1049h\x1b[H\x1b[2J").unwrap();
    let asked = answer_query(&mut second, reset, LIGHT);
    second.wait_for_after(asked, VIEW_PAINTED);
    second.wait_for_after(asked, "quadrants");
    second.wait_for_after(asked, LIGHT_MAP);
    second.wait_for_after(reset, "\x1b[?1003h");
    second.type_(b"\x1b\x04");
    assert_eq!(second.exit_code(), 0);
    dir.wait_listed("tissue (pinto-view-1)\tdetached");

    // On one that shows kitty's images, the map is sent as one.
    let mut third = term(&dir, &["attach", "tissue"], &[]);
    third.wait_for("\x1b[?1049h\x1b[H\x1b[2J");
    let reset = third.text().find("\x1b[?1049h\x1b[H\x1b[2J").unwrap();
    let asked = answer_query(&mut third, reset, KITTY_LIGHT);
    third.wait_for_after(asked, "kitty");
    third.wait_for_after(asked, "\x1b_Ga=T,f=32");
    // `?` lists the session keys among the view's (drawn over the lines
    // before, only what differs: the keys and a word of each).
    third.type_(b"?");
    third.wait_for("ctrl+alt+d");
    third.wait_for("switch");

    // Killed while attached: it quits as ctrl+c does, the client ends with
    // it, and nothing is left, kitty's files among it.
    let said = dir.run(&["kill", "tissue"]);
    assert!(said.contains("pinto-view-1 stopped"), "{said:?}");
    assert_eq!(third.exit_code(), 0);
    assert!(third.cooked(), "terminal restored after the program ended");
    assert!(!third.text().contains(PROMPT));
    dir.gone(pid);
    assert!(dir.is_empty(), "left behind in {:?}", dir.path());
    assert!(kitty_files(pid).iter().all(|f| !f.exists()));
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
    // Not over ssh, whatever the tests run in: kitty's images go through
    // files.
    let env = [
        ("PINTO_LUPIN", lupin.as_str()),
        ("SSH_CONNECTION", ""),
        ("SSH_CLIENT", ""),
        ("SSH_TTY", ""),
    ];

    let mut first = term(&dir, &["view", &prefix], &env);
    let asked = answer_query(&mut first, 0, KITTY_LIGHT);
    first.wait_for_after(asked, "kitty");
    first.wait_for_after(asked, "(file)");
    first.wait_for_after(asked, "\x1b_Ga=T,f=32");
    let pid = dir.program_pid();
    assert!(
        kitty_files(pid).iter().any(|f| f.exists()),
        "the map went through a file"
    );

    // `A` and the panel beside the run: lupin annotates.
    first.type_(b"A");
    first.wait_for("marker panel for level");
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
    assert!(kitty_files(pid).iter().all(|f| !f.exists()));
    std::thread::sleep(Duration::from_secs(4));
    assert!(!written.exists(), "lupin went on after the view");
}

#[test]
fn the_run_browser_comes_back_on_attach_and_stops_on_kill() {
    let dir = private();

    // With no run named it browses for one, hosted.
    let mut first = term(&dir, &["view"], &[]);
    first.wait_for("run to open");
    dir.wait_listed("pinto-view-1\tattached");
    let pid = dir.program_pid();
    detach(&dir, &mut first, "pinto-view-1");

    // Started again with nothing named, the live session is offered first:
    // Enter takes it, and the browser is drawn there whole.
    let mut second = term(&dir, &["view"], &[]);
    second.wait_for("pinto-view-1");
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
