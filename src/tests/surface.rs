//! The command line's surface: clap's own consistency checks, and how many
//! flags each command's `--help` shows.

use super::*;
use clap::CommandFactory;

/// clap checks the definitions (ids, `mut_arg` targets, conflicts, aliases)
/// only when a command is built, so a typo there passes `cargo check`.
#[test]
fn the_command_line_passes_clap_s_own_checks() {
    Cli::command().debug_assert();
}

/// The named flags `pinto {command} --help` lists: hidden ones, help and
/// version left out, `-v` counted.
pub(crate) fn visible_flags(cmd: &clap::Command) -> Vec<String> {
    cmd.get_arguments()
        .filter(|a| !a.is_hide_set())
        .filter(|a| a.get_long().is_some() || a.get_short().is_some())
        .filter(|a| !matches!(a.get_id().as_str(), "help" | "version"))
        .map(|a| a.get_id().to_string())
        .collect()
}

/// Each command shows only the flags a user changes. A flag added to a
/// command's `--help` must be added here on purpose; a tuning knob is
/// hidden, or a constant.
#[test]
fn each_command_s_help_shows_a_short_list_of_flags() {
    const BUDGET: &[(&str, usize)] = &[
        ("fit lc", 27),
        ("fit dsvd", 27),
        ("fit cage", 37),
        ("fit cage-cell", 29),
        ("fit spatialde2", 23),
        ("fit stagate", 28),
        ("fit spatialpca", 23),
        ("fit mefisto", 23),
        ("fit spatialde2-pb", 24),
        ("fit stagate-pb", 29),
        ("fit spatialpca-pb", 24),
        ("fit mefisto-pb", 24),
        ("apply residual", 11),
        ("apply predict", 23),
        ("apply impute", 21),
        ("apply propensity", 11),
        ("lra", 23),
        ("lra simulate", 23),
        ("sim new", 10),
        ("sim score-by-set", 8),
        ("sim score-by-pair", 10),
        ("sim baseline", 8),
        ("compare", 9),
        #[cfg(feature = "view")]
        ("view", 29),
    ];
    let mut cli = Cli::command();
    cli.build();
    let counts: Vec<(String, usize)> = crate::cli_path::tests::commands(&cli)
        .into_iter()
        .map(|(path, c)| (path, visible_flags(c).len()))
        .collect();
    let mut wrong = Vec::new();
    for &(name, budget) in BUDGET {
        let (_, n) = counts
            .iter()
            .find(|(c, _)| c == name)
            .unwrap_or_else(|| panic!("pinto has no {name}"));
        if *n != budget {
            wrong.push(format!("{name}: {n} shown, budget {budget}"));
        }
    }
    assert!(wrong.is_empty(), "{}\nall: {counts:?}", wrong.join("\n"));
}

/// Flags parse by their names: cage's community count by either of its
/// two, compare's levels as one or A,B.
#[test]
fn flags_parse_by_their_names() {
    let parses = |argv: &[&str]| Cli::try_parse_from(argv).map(|c| c.commands);
    let cage = |flag: &str| match parses(&["pinto", "fit", "cage", "d.zarr", "-o", "o", flag, "7"])
    {
        Ok(Commands::Fit(Fit::Cage(a))) => a.edge_clustering.n_edge_clusters,
        other => panic!("{other:?}"),
    };
    assert_eq!(cage("--n-communities"), Some(7));
    assert_eq!(cage("--n-edge-clusters"), Some(7));

    let levels = |extra: &[&str]| match parses(
        &[&["pinto", "compare", "a", "b", "-o", "o"][..], extra].concat(),
    ) {
        Ok(Commands::Compare(a)) => a.levels(),
        other => panic!("{other:?}"),
    };
    let both = |a: &str, b: &str| (a.to_string(), b.to_string());
    assert_eq!(levels(&[]), both("final", "final"));
    assert_eq!(levels(&["--level", "draft"]), both("draft", "draft"));
    assert_eq!(levels(&["--level", "L1,L2"]), both("L1", "L2"));
    assert!(parses(&["pinto", "compare", "a", "b", "-o", "o", "--level", "L1,L2,L3"]).is_err());
}

/// `--max-threads` parses on any command, before or after it, and reaches
/// lc, whose Gibbs blocks it sets.
#[test]
fn max_threads_is_one_flag_on_every_command_and_reaches_lc() {
    let lc = |argv: &[&str]| {
        let mut cli = Cli::try_parse_from(argv).unwrap();
        pass_max_threads(&mut cli);
        match cli.commands {
            Commands::Fit(Fit::LinkCommunity(a)) => a.max_threads,
            other => panic!("{other:?}"),
        }
    };
    let base = ["pinto", "fit", "lc", "d.zarr", "-o", "o"];
    let default = crate::util::common::DEFAULT_MAX_THREADS;
    assert_eq!(lc(&base), default);
    assert_eq!(lc(&[&base[..], &["--max-threads", "40"]].concat()), 40);
    assert_eq!(
        lc(&[&["pinto", "--max-threads", "3"], &base[1..]].concat()),
        3
    );
    assert!(Cli::try_parse_from([&base[..], &["--max-threads", "0"]].concat()).is_err());

    let mut cli = Cli::command();
    cli.build();
    for (_, sub) in crate::cli_path::tests::commands(&cli) {
        let arg = sub
            .get_arguments()
            .find(|a| a.get_long() == Some("max-threads"))
            .unwrap_or_else(|| panic!("{} has no --max-threads", sub.get_name()));
        assert!(!arg.is_hide_set(), "{} hides --max-threads", sub.get_name());
        let shown: Vec<String> = arg
            .get_default_values()
            .iter()
            .map(|v| v.to_string_lossy().into_owned())
            .collect();
        assert_eq!(shown, [default.to_string()], "{}", sub.get_name());
    }
}

/// Every flag a help page shows says what it does.
#[test]
fn every_shown_flag_has_a_help_line() {
    let mut cli = Cli::command();
    cli.build();
    let mut bare = Vec::new();
    for (_, sub) in crate::cli_path::tests::commands(&cli) {
        for a in sub.get_arguments().filter(|a| !a.is_hide_set()) {
            if a.get_help().is_none() && a.get_long_help().is_none() {
                bare.push(format!("{} {}", sub.get_name(), a.get_id()));
            }
        }
    }
    assert!(bare.is_empty(), "no help: {bare:?}");
}
