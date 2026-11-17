//! The session flags and subcommands of `pinto run` and `pinto view`.

use super::*;

fn parse(argv: &[&str]) -> Commands {
    Cli::try_parse_from(argv).unwrap().commands
}

#[test]
fn attach_sessions_kill_and_rename_parse() {
    match parse(&["pinto", "session", "attach"]) {
        Commands::Session(Session::Attach(a)) => assert_eq!(a.name, None),
        other => panic!("{other:?}"),
    }
    match parse(&["pinto", "session", "attach", "run-1"]) {
        Commands::Session(Session::Attach(a)) => assert_eq!(a.name.as_deref(), Some("run-1")),
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        parse(&["pinto", "session", "list"]),
        Commands::Session(Session::List)
    ));
    match parse(&["pinto", "session", "kill", "--force", "view-2"]) {
        Commands::Session(Session::Kill(k)) => {
            assert!(k.force);
            assert_eq!(k.name.as_deref(), Some("view-2"));
        }
        other => panic!("{other:?}"),
    }
    match parse(&["pinto", "session", "rename", "run-1", "work"]) {
        Commands::Session(Session::Rename(r)) => {
            assert_eq!(r.session, "run-1");
            assert_eq!(r.label, "work");
        }
        other => panic!("{other:?}"),
    }
    assert!(Cli::try_parse_from(["pinto", "session", "rename", "run-1"]).is_err());
}

#[test]
fn a_target_named_skips_the_resume_picker() {
    // `pinto run` alone may resume, still starting its browser here; a
    // folder named starts a new session there.
    let Commands::Run(r) = parse(&["pinto", "run"]) else {
        panic!("run")
    };
    assert!(r.dir.is_none() && !r.session().target && !r.session().no_detach);
    assert_eq!(r.start(), std::path::PathBuf::from("."));
    let Commands::Run(r) = parse(&["pinto", "run", "data", "--no-detach"]) else {
        panic!("run")
    };
    assert!(r.session().target && r.session().no_detach);
    assert_eq!(r.start(), std::path::PathBuf::from("data"));
    // `pinto view` with a run names it; with none, it may resume.
    let Commands::View(v) = parse(&["pinto", "view", "--session", "x", "results"]) else {
        panic!("view")
    };
    assert_eq!(v.session.session.as_deref(), Some("x"));
    assert!(v.session.options(v.prefix.is_some()).target);
    let Commands::View(v) = parse(&["pinto", "view"]) else {
        panic!("view")
    };
    assert!(!v.session.options(v.prefix.is_some()).target);
}

/// The long help of `name`, as `pinto <name> --help` prints it.
fn long_help(name: &str) -> String {
    let cli = built_cli();
    let sub = crate::cli_path::find(&cli, name).unwrap().1;
    sub.get_long_about().unwrap().to_string()
}

/// `text` on one line, its words a space apart.
pub(crate) fn flat(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn the_session_help_is_the_shared_text_said_as_pinto_says_it() {
    let run = flat(&long_help("run"));
    for said in [
        "the fits go on when the connection drops",
        "`pinto run` with no folder offers the live sessions",
        "`pinto session attach`, `session list`, `session kill` and `session rename`",
        "listed at the foot of the screen.",
        "--no-detach",
        // The braces of the text before it kept as they were.
        "`bash {out}.cmd.sh`",
    ] {
        assert!(run.contains(said), "{said:?} in {run}");
    }
    let view = flat(&long_help("view"));
    for said in [
        "Reads {prefix}.pinto.json",
        "`pinto view` with no run offers the live sessions",
        "`?` shows the keys in effect.",
    ] {
        assert!(view.contains(said), "{said:?} in {view}");
    }
    let attach = flat(&long_help("attach"));
    assert!(attach.contains("`pinto session list` lists them"), "{attach}");
    let kill = flat(&long_help("kill"));
    assert!(
        kill.contains("stops the fits it is running first"),
        "{kill}"
    );
    // Wrapped as the rest of pinto's help, a paragraph apart.
    for help in [long_help("run"), long_help("view")] {
        let ours = help.rsplit("\n\n").next().unwrap();
        assert!(ours.lines().all(|l| l.chars().count() <= 70), "{help}");
        assert!(ours.lines().count() > 1, "{help}");
    }
}
