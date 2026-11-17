//! Where a command sits in pinto's verbs: `pinto fit lc`, `pinto apply
//! residual`, `pinto lra simulate`, `pinto sim new`, `pinto session list`.
//!
//! Before 0.17 every command sat at the top (`pinto lc`, `pinto residual`,
//! `pinto lra-simulate`, `pinto srt-simulate`, `pinto sessions`). Those
//! names still run, moved under their verb as the line is read
//! ([`with_verb`]); the leaves keep them as aliases, so [`find`] takes them
//! too.

use clap::Command;

/// The verbs: the commands whose work is done by a command under them.
/// `lra` is not one: it runs itself, and `lra simulate` is under it.
#[cfg(any(feature = "view", test))]
pub(crate) const VERBS: [&str; 4] = ["fit", "apply", "sim", "session"];

/// The command named `word` (its name or an alias) at the top of `cli` or
/// one command down: the command it is under, if any, and the command.
pub(crate) fn find<'c>(cli: &'c Command, word: &str) -> Option<(Option<&'c Command>, &'c Command)> {
    if let Some(c) = cli.find_subcommand(word) {
        return Some((None, c));
    }
    cli.get_subcommands()
        .find_map(|verb| verb.find_subcommand(word).map(|c| (Some(verb), c)))
}

/// The names from the top of a command [`find`] found: `fit lc`.
pub(crate) fn path<'c>((under, cmd): (Option<&'c Command>, &'c Command)) -> Vec<&'c str> {
    under
        .into_iter()
        .chain([cmd])
        .map(Command::get_name)
        .collect()
}

/// The command `argv` (without the program) runs and how many of its
/// leading words name it: its first word as [`find`] takes it, then each
/// word naming a command under the one before.
#[cfg(any(feature = "view", test))]
pub(crate) fn leaf<'c>(cli: &'c Command, argv: &[String]) -> Option<(&'c Command, usize)> {
    let (_, mut cmd) = find(cli, argv.first()?)?;
    let mut n = 1;
    while let Some(c) = argv.get(n).and_then(|w| cmd.find_subcommand(w)) {
        cmd = c;
        n += 1;
    }
    Some((cmd, n))
}

/// The command `matches` ran, as built in `cmd`, and its own matches: the
/// leaf under the verbs.
pub(crate) fn ran<'c>(
    mut cmd: &'c Command,
    mut matches: &'c clap::ArgMatches,
) -> (&'c Command, &'c clap::ArgMatches) {
    while let Some((name, m)) = matches.subcommand() {
        cmd = cmd.find_subcommand(name).expect("a subcommand as parsed");
        matches = m;
    }
    (cmd, matches)
}

/// How many leading words of `argv` (without the program) name its
/// command, by the verbs alone: `fit lc …` and `lra simulate …` two,
/// `lra …` and `compare …` one. For the lines pinto writes, which name
/// the command by its path.
#[cfg(any(feature = "view", test))]
#[must_use]
pub(crate) fn path_len(argv: &[String]) -> usize {
    match argv.first().map(String::as_str) {
        None => 0,
        Some(verb) if VERBS.contains(&verb) => 2.min(argv.len()),
        Some("lra") if argv.get(1).is_some_and(|w| w == "simulate") => 2,
        Some(_) => 1,
    }
}

/// The top-level names before 0.17 of what now sits under a verb, with
/// their aliases: only these are moved ([`with_verb`]), so `pinto
/// simulate` or `pinto list` is no shortcut.
const OLD_NAMES: [&str; 29] = [
    "lc",
    "link-community",
    "dsvd",
    "delta-svd",
    "cage",
    "cge",
    "cage-cell",
    "spatialde2",
    "stagate",
    "spatialpca",
    "mefisto",
    "spatialde2-pb",
    "stagate-pb",
    "spatialpca-pb",
    "mefisto-pb",
    "residual",
    "predict",
    "impute",
    "propensity",
    "prop",
    "lra-simulate",
    "srt-simulate",
    "srt-sim-score",
    "srt-sim-pairs",
    "srt-sim-baseline",
    "attach",
    "sessions",
    "kill",
    "rename",
];

/// `argv` (with the program) with a command named as before 0.17 moved
/// under its verb (`pinto lc …` read as `pinto fit lc …`, `pinto help lc`
/// as `pinto help fit lc`), and a note saying so; unchanged otherwise.
/// Global flags ahead of the command are skipped.
pub(crate) fn with_verb(cli: &Command, mut argv: Vec<String>) -> (Vec<String>, Option<String>) {
    let mut at = 1;
    while let Some(w) = argv.get(at) {
        if w == "--" || !w.starts_with('-') {
            break;
        }
        // a global flag's value may follow as its own word
        at += if takes_value(cli, w) { 2 } else { 1 };
    }
    if argv.get(at).is_some_and(|w| w == "help") {
        at += 1;
    }
    let Some(word) = argv.get(at).filter(|w| OLD_NAMES.contains(&w.as_str())) else {
        return (argv, None);
    };
    let Some(found @ (Some(verb), _)) = find(cli, word) else {
        return (argv, None);
    };
    let note = format!(
        "pinto: `pinto {word}` is now `pinto {}`; the old name goes after 0.17",
        path(found).join(" ")
    );
    argv.insert(at, verb.get_name().to_string());
    (argv, Some(note))
}

/// Whether `word` is a global flag of `cli` given without its value
/// (`--max-threads`, not `--max-threads=4`), so the value is the next word.
fn takes_value(cli: &Command, word: &str) -> bool {
    let long = word.strip_prefix("--");
    cli.get_arguments().any(|a| {
        a.is_global_set()
            && a.get_action().takes_values()
            && long.is_some_and(|l| a.get_long() == Some(l))
    })
}

#[cfg(test)]
#[path = "tests/cli_path.rs"]
pub(crate) mod tests;
