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
fn a_command_is_found_by_its_name_or_path() {
    let cli = crate::built_cli();
    for (word, want) in [
        ("lc", vec!["fit", "lc"]),
        ("dsvd", vec!["fit", "dsvd"]),
        ("cage-cell", vec!["fit", "cage-cell"]),
        ("stagate-pb", vec!["fit", "stagate-pb"]),
        ("residual", vec!["apply", "residual"]),
        ("propensity", vec!["apply", "propensity"]),
        ("lra", vec!["lra"]),
        ("lra simulate", vec!["lra", "simulate"]),
        ("sim new", vec!["sim", "new"]),
        ("pairs", vec!["sim", "pairs"]),
        ("compare", vec!["compare"]),
    ] {
        let found = path(find(&cli, word).unwrap_or_else(|| panic!("{word}")));
        assert_eq!(found, want, "{word}");
    }
    for gone in [
        "no-such-command",
        "link-community",
        "srt-simulate",
        "lra-simulate",
    ] {
        assert!(find(&cli, gone).is_none(), "{gone}");
    }
}

#[test]
fn the_leaf_is_found_from_the_path() {
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
