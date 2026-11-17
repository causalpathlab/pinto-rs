use super::*;

/// Every command of `cli` that does work, under the verbs or at the top,
/// with its path: `fit lc`, `lra`, `lra simulate`, `compare`; not the verbs,
/// nor `help`.
pub(crate) fn commands(cli: &Command) -> Vec<(String, &Command)> {
    let real = |c: &&Command| c.get_name() != "help";
    let mut out = Vec::new();
    for top in cli.get_subcommands().filter(real) {
        if !super::VERBS.contains(&top.get_name()) {
            out.push((top.get_name().to_string(), top));
        }
        for leaf in top.get_subcommands().filter(real) {
            out.push((format!("{} {}", top.get_name(), leaf.get_name()), leaf));
        }
    }
    out
}

fn words(argv: &[&str]) -> Vec<String> {
    argv.iter().map(ToString::to_string).collect()
}

#[test]
fn a_command_is_found_by_its_name_or_old_name_with_its_path() {
    let cli = crate::built_cli();
    for (word, want) in [
        ("lc", vec!["fit", "lc"]),
        ("link-community", vec!["fit", "lc"]),
        ("dsvd", vec!["fit", "dsvd"]),
        ("cage-cell", vec!["fit", "cage-cell"]),
        ("stagate-pb", vec!["fit", "stagate-pb"]),
        ("residual", vec!["apply", "residual"]),
        ("prop", vec!["apply", "propensity"]),
        ("lra", vec!["lra"]),
        ("lra-simulate", vec!["lra", "simulate"]),
        ("srt-simulate", vec!["sim", "new"]),
        ("srt-sim-pairs", vec!["sim", "pairs"]),
        ("compare", vec!["compare"]),
    ] {
        let found = path(find(&cli, word).unwrap_or_else(|| panic!("{word}")));
        assert_eq!(found, want, "{word}");
    }
    assert!(find(&cli, "no-such-command").is_none());
}

#[test]
fn the_leaf_is_found_from_the_path_or_an_old_name() {
    let cli = crate::built_cli();
    for (argv, name, n) in [
        (words(&["fit", "lc", "a.h5", "--out=x"]), "lc", 2),
        (words(&["lc", "a.h5"]), "lc", 1),
        (words(&["lra", "simulate", "--from", "r"]), "simulate", 2),
        (words(&["lra", "--from", "r"]), "lra", 1),
        (words(&["sim", "new", "--out", "s"]), "new", 2),
    ] {
        let (cmd, k) = leaf(&cli, &argv).unwrap();
        assert_eq!((cmd.get_name(), k), (name, n), "{argv:?}");
    }
}

#[test]
fn the_path_length_is_read_from_the_verbs_alone() {
    for (argv, n) in [
        (words(&["fit", "lc", "data"]), 2),
        (words(&["apply", "residual", "--from=r"]), 2),
        (words(&["lra", "simulate", "--from=r"]), 2),
        (words(&["lra", "--from=r"]), 1),
        (words(&["compare", "a", "b"]), 1),
        (words(&["fit"]), 1),
        (vec![], 0),
    ] {
        assert_eq!(path_len(&argv), n, "{argv:?}");
    }
}

#[test]
fn every_command_under_a_verb_is_reached_and_the_verbs_are_listed() {
    let cli = crate::built_cli();
    let parents: Vec<&str> = cli
        .get_subcommands()
        .filter(|c| c.has_subcommands() && c.get_name() != "lra" && c.get_name() != "help")
        .map(clap::Command::get_name)
        .collect();
    // the sessions come with the viewer
    let verbs: Vec<&str> = VERBS
        .into_iter()
        .filter(|v| cfg!(feature = "view") || *v != "session")
        .collect();
    assert_eq!(parents, verbs);
}

#[test]
fn an_old_name_is_moved_under_its_verb_past_the_global_flags() {
    let cli = crate::built_cli();
    let (argv, note) = with_verb(
        &cli,
        words(&[
            "pinto",
            "-v",
            "--max-threads",
            "4",
            "srt-simulate",
            "--out",
            "s",
        ]),
    );
    assert_eq!(
        argv,
        words(&[
            "pinto",
            "-v",
            "--max-threads",
            "4",
            "sim",
            "srt-simulate",
            "--out",
            "s"
        ])
    );
    assert!(note.unwrap().contains("`pinto sim new`"));
    // what is at the top already, or not pinto's, is left as it is
    for line in [
        words(&["pinto", "fit", "lc", "a.h5"]),
        words(&["pinto", "--atomic-outputs", "compare", "a", "b"]),
        words(&["pinto", "nonsense"]),
        words(&["pinto"]),
    ] {
        assert_eq!(with_verb(&cli, line.clone()), (line, None));
    }
}

#[test]
fn an_old_line_parses_as_its_new_one() {
    use clap::Parser;
    let cli = crate::built_cli();
    let (argv, _) = with_verb(
        &cli,
        words(&["pinto", "lra-simulate", "--from", "r.pinto.json"]),
    );
    let parsed = crate::Cli::try_parse_from(argv).unwrap();
    assert!(matches!(
        parsed.commands,
        crate::Commands::Lra(ref a) if matches!(a.simulate, Some(crate::LraSub::Simulate(_)))
    ));
}

#[test]
fn only_a_name_from_before_the_verbs_is_moved_and_help_takes_it_too() {
    let cli = crate::built_cli();
    for new_word in ["simulate", "new", "list", "score", "pairs", "baseline"] {
        let line = words(&["pinto", new_word]);
        assert_eq!(with_verb(&cli, line.clone()), (line, None), "{new_word}");
    }
    let (argv, note) = with_verb(&cli, words(&["pinto", "help", "lc"]));
    assert_eq!(argv, words(&["pinto", "help", "fit", "lc"]));
    assert!(note.is_some());
    // every old name still finds its command under a verb
    let sessions = ["attach", "sessions", "kill", "rename"];
    for old in OLD_NAMES
        .into_iter()
        .filter(|o| cfg!(feature = "view") || !sessions.contains(o))
    {
        let (under, _) = find(&cli, old).unwrap_or_else(|| panic!("{old}"));
        assert!(under.is_some(), "{old}");
    }
}
