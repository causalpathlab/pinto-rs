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
        ("delta-svd", 22),
        ("propensity", 9),
        ("link-community", 22),
        ("cage", 33),
        ("cage-cell", 24),
        ("predict", 21),
        ("impute", 19),
        ("lra", 19),
        ("lra-simulate", 18),
        ("residual", 8),
        ("srt-simulate", 6),
        ("srt-sim-score", 6),
        ("srt-sim-pairs", 8),
        ("srt-sim-baseline", 6),
        ("spatialde2", 10),
        ("stagate", 15),
        ("compare", 8),
        #[cfg(feature = "view")]
        ("view", 27),
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
