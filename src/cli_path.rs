//! Where a command sits in pinto's verbs: `pinto fit lc`, `pinto apply
//! predict`, `pinto lra simulate`, `pinto sim new`, `pinto session list`.

use clap::Command;

/// The verbs: the commands whose work is done by a command under them.
/// `lra` is not one: it runs itself, and `lra simulate` is under it.
#[cfg(any(feature = "view", test))]
pub(crate) const VERBS: [&str; 4] = ["fit", "apply", "sim", "session"];

/// The command named `word` at the top of `cli` or
/// one command down, or by its path (`lra simulate`, `sim new`): the
/// command it is under, if any, and the command.
#[cfg(any(feature = "view", test))]
pub(crate) fn find<'c>(cli: &'c Command, word: &str) -> Option<(Option<&'c Command>, &'c Command)> {
    if let Some((verb, leaf)) = word.split_once(' ') {
        let verb = cli.find_subcommand(verb)?;
        return verb.find_subcommand(leaf).map(|c| (Some(verb), c));
    }
    if let Some(c) = cli.find_subcommand(word) {
        return Some((None, c));
    }
    cli.get_subcommands()
        .find_map(|verb| verb.find_subcommand(word).map(|c| (Some(verb), c)))
}

/// The names from the top of a command [`find`] found: `fit lc`.
#[cfg(any(feature = "view", test))]
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

#[cfg(test)]
#[path = "tests/cli_path.rs"]
pub(crate) mod tests;
