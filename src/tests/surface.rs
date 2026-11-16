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
        ("delta-svd", 23),
        ("propensity", 10),
        ("link-community", 23),
        ("cage", 34),
        ("cage-cell", 26),
        ("predict", 22),
        ("impute", 20),
        ("lra", 20),
        ("lra-simulate", 19),
        ("residual", 9),
        ("srt-simulate", 7),
        ("srt-sim-score", 7),
        ("srt-sim-pairs", 9),
        ("srt-sim-baseline", 7),
        ("spatialde2", 14),
        ("stagate", 19),
        ("spatialpca", 14),
        ("mefisto", 14),
        ("spatialde2-pb", 15),
        ("stagate-pb", 20),
        ("spatialpca-pb", 15),
        ("mefisto-pb", 15),
        ("compare", 8),
        #[cfg(feature = "view")]
        ("view", 28),
    ];
    let mut cli = Cli::command();
    cli.build();
    let counts: Vec<(String, usize)> = cli
        .get_subcommands()
        .map(|c| (c.get_name().to_string(), visible_flags(c).len()))
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

/// A renamed flag takes its new name and keeps its old one as a hidden
/// alias, so command lines written before the rename still parse.
#[test]
fn renamed_flags_parse_by_their_new_and_old_names() {
    let parses = |argv: &[&str]| Cli::try_parse_from(argv).map(|c| c.commands);
    let cage = |flag: &str| match parses(&["pinto", "cage", "d.zarr", "-o", "o", flag, "7"]) {
        Ok(Commands::Cage(a)) => a.edge_clustering.n_edge_clusters,
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
    assert_eq!(levels(&["--level-a", "draft"]), both("draft", "final"));
    assert!(parses(&["pinto", "compare", "a", "b", "-o", "o", "--level", "L1,L2,L3"]).is_err());
    assert!(parses(&[
        "pinto",
        "compare",
        "a",
        "b",
        "-o",
        "o",
        "--level",
        "L1",
        "--level-b",
        "L2"
    ])
    .is_err());

    for flag in ["-c", "--coord", "--coords"] {
        match parses(&["pinto", "srt-simulate", "-o", "o", flag, "x.csv"]) {
            Ok(Commands::SrtSimulate(a)) => assert_eq!(a.coords.len(), 1, "{flag}"),
            other => panic!("{flag}: {other:?}"),
        }
    }
    for flag in ["--n-clusters", "--k"] {
        match parses(&[
            "pinto",
            "srt-sim-baseline",
            "--truth",
            "t",
            "-o",
            "o",
            flag,
            "4",
        ]) {
            Ok(Commands::SrtSimBaseline(a)) => assert_eq!(a.k, Some(4), "{flag}"),
            other => panic!("{flag}: {other:?}"),
        }
    }
    for flag in ["--data-files", "--expr-data-files", "-d"] {
        let argv = [
            "pinto",
            "prop",
            "-z",
            "l.parquet",
            "-e",
            "p.parquet",
            "-o",
            "o",
            flag,
            "x.zarr",
        ];
        assert!(parses(&argv).is_ok(), "{flag}");
    }
    for flag in ["--feature-name-mode", "--feature-names"] {
        match parses(&["pinto", "residual", "--from", "r", "-o", "o", flag, "gene"]) {
            Ok(Commands::Residual(a)) => assert_ne!(
                a.feature_names,
                crate::cell_activity_graph_embedding::args::FeatureNameMode::Exact,
                "{flag}"
            ),
            other => panic!("{flag}: {other:?}"),
        }
    }
}

/// `--max-threads` parses on any command, before or after it, and reaches
/// lc, whose Gibbs blocks it sets.
#[test]
fn max_threads_is_one_flag_on_every_command_and_reaches_lc() {
    let lc = |argv: &[&str]| {
        let mut cli = Cli::try_parse_from(argv).unwrap();
        pass_max_threads(&mut cli);
        match cli.commands {
            Commands::LinkCommunity(a) => a.max_threads,
            other => panic!("{other:?}"),
        }
    };
    let base = ["pinto", "lc", "d.zarr", "-o", "o"];
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
    for sub in cli.get_subcommands().filter(|s| s.get_name() != "help") {
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
    for sub in cli.get_subcommands() {
        for a in sub.get_arguments().filter(|a| !a.is_hide_set()) {
            if a.get_help().is_none() && a.get_long_help().is_none() {
                bare.push(format!("{} {}", sub.get_name(), a.get_id()));
            }
        }
    }
    assert!(bare.is_empty(), "no help: {bare:?}");
}

/// cage's `--help` lists its flags in the order it always has, and
/// `pinto run`'s form and the MCP schema, which take every argument in
/// order, list the batch options where cage's own stood: `-o` and
/// `--seed` between `--ignore-batch` and `--num-levels`.
#[test]
fn cage_s_help_keeps_its_flag_order() {
    const SHOWN: &[&str] = &[
        "coord_files",
        "coord_columns",
        "coord_column_names",
        "batch_files",
        "auto_batch",
        "ignore_batch",
        "out",
        "seed",
        "cnv_clones",
        "n_pseudobulk",
        "knn_spatial",
        "knn_expr",
        "no_qc",
        "permute_coords",
        "drop_features",
        "mask_cells",
        "ablation_parent",
        "device",
        "embedding_dim",
        "epochs",
        "feature_inclusion",
        "n_hvg",
        "feature_list_file",
        "must_train_features",
        "edge_cluster_method",
        "leiden_resolution",
        "n_edge_clusters",
        "feature_embedding",
        "feature_embedding_bias",
        "feature_embedding_mode",
        "feature_init_mode",
        "feature_modules",
        "verbose",
        "max_threads",
    ];
    const FIRST: &[&str] = &[
        "data_files",
        "coord_files",
        "coord_columns",
        "coord_column_names",
        "coord_header_row",
        "batch_files",
        "auto_batch",
        "ignore_batch",
        "out",
        "seed",
        "num_levels",
        "proj_dim",
        "cnv_clones",
        "n_pseudobulk",
    ];
    let cli = crate::built_cli();
    let cage = cli.find_subcommand("cage").unwrap();
    assert_eq!(visible_flags(cage), SHOWN);
    let all: Vec<String> = cage
        .get_arguments()
        .take(FIRST.len())
        .map(|a| a.get_id().to_string())
        .collect();
    assert_eq!(all, FIRST);
}
