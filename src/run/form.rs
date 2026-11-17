//! A method's flags as rows of a form, read from its clap definition, and
//! the command line the filled form stands for.
//!
//! Nothing here names a flag of any one method: every row comes from the
//! subcommand's `Arg`s, so a flag added to a method shows up on its own.

use crate::tui::flag_value;
use clap::Command;

/// Flags `pinto run` fills from its own screens or never passes, and the
/// one it takes for the whole session (`--atomic-outputs`).
const OWN: &[&str] = &[
    "out",
    "coord",
    "batch-files",
    "help",
    "version",
    "verbose",
    "atomic-outputs",
];

use legume_tui::form::{emit, many_of, Field, Many};

/// `f`'s values rewritten by `rewrite`, as the command line splits them (a
/// comma list in one value too), joined by commas. A row at its default is
/// not passed, and stays as it is.
pub fn map_values(f: &mut Field, rewrite: impl Fn(&str) -> String) {
    if f.is_default() {
        return;
    }
    // split as legume-tui reads a row's values: on commas, and on any
    // whitespace too when the flag takes several
    let several = f.many != Many::One;
    let vs: Vec<String> = f
        .value
        .trim()
        .split(|c: char| c == ',' || (several && c.is_whitespace()))
        .filter(|v| !v.is_empty())
        .map(rewrite)
        .collect();
    f.value = vs.join(",");
}

/// One method as a form.
#[derive(Clone, Debug)]
pub struct Method {
    /// As `pinto run` and the tools name it: the subcommand or its alias
    /// (`lc`, `residual`, `lra-simulate`).
    pub name: String,
    /// The words that run it, from the top: `fit lc`, `apply residual`,
    /// `lra simulate`.
    pub path: Vec<String>,
    pub about: String,
    pub fields: Vec<Field>,
    /// How the coordinate files are passed, when the method takes them.
    coord: Option<Many>,
    /// How the batch files are passed, when the method takes them.
    batch: Option<Many>,
    /// Whether it reads a prior run (`--from`, as lra does) instead of
    /// the data files: they, their coordinates and batches are not passed.
    pub from_run: bool,
}

impl Method {
    /// The form for command `name` (or an alias of it) of `cli`, at the
    /// top or under its verb ([`crate::cli_path::find`]); `cli` must be
    /// built.
    pub fn new(cli: &Command, name: &str) -> anyhow::Result<Self> {
        let found = crate::cli_path::find(cli, name)
            .ok_or_else(|| anyhow::anyhow!("pinto has no `{name}` command"))?;
        let (path, cmd) = (crate::cli_path::path(found), found.1);
        let slot = |long: &str| {
            cmd.get_arguments()
                .find(|a| a.get_long() == Some(long))
                .map(many_of)
        };
        Ok(Method {
            // a path's words joined, as rows and their folders are named
            name: name.replace(' ', "-"),
            path: path.into_iter().map(String::from).collect(),
            about: cmd.get_about().map(ToString::to_string).unwrap_or_default(),
            // every flag but those `pinto run` fills itself, each heading's
            // together
            fields: legume_tui::form::Form::new(cmd, |a| {
                !a.get_long().is_some_and(|l| OWN.contains(&l))
            })
            .fields,
            coord: slot("coord"),
            batch: slot("batch-files"),
            from_run: slot("from").is_some(),
        })
    }

    /// Whether the method reads coordinate files.
    #[must_use]
    pub fn takes_coords(&self) -> bool {
        self.coord.is_some() && !self.from_run
    }

    /// Whether the method reads batch files.
    #[must_use]
    pub fn takes_batches(&self) -> bool {
        self.batch.is_some() && !self.from_run
    }

    /// The `--from` run, as typed; `None` when unset or not a method that
    /// reads one.
    #[must_use]
    pub fn from(&self) -> Option<&str> {
        self.value("from").filter(|_| self.from_run)
    }

    /// The value of the flag `long`, as typed, when it is not its default.
    #[must_use]
    pub fn value(&self, long: &str) -> Option<&str> {
        let f = &self.fields[self.position(long)?];
        (!f.is_default()).then(|| f.value.trim())
    }

    /// Where the row of the flag named `name` (with or without `--`) is,
    /// by its long name or an alias.
    fn position(&self, name: &str) -> Option<usize> {
        let name = name.trim_start_matches('-');
        self.fields.iter().position(|f| f.is_named(name))
    }

    /// The row of the flag named `name` (with or without `--`), by its
    /// long name or an alias.
    pub fn field_mut(&mut self, name: &str) -> Option<&mut Field> {
        let at = self.position(name)?;
        Some(&mut self.fields[at])
    }

    /// Set the flag `long` to `value`, when the method has it.
    pub fn set(&mut self, long: &str, value: &str) {
        if let Some(f) = self.field_mut(long) {
            f.value = value.to_string();
        }
    }

    /// Flags changed from their defaults.
    #[must_use]
    pub fn changed(&self) -> usize {
        self.fields.iter().filter(|f| !f.is_default()).count()
    }

    /// The command line, without the program: the method's path, data files,
    /// coordinate and batch files, `--out`, then every changed flag.
    #[must_use]
    pub fn argv(
        &self,
        data: &[String],
        coords: &[String],
        batches: &[String],
        out: &str,
    ) -> Vec<String> {
        let mut argv = self.path.clone();
        // A method reading a prior run takes its data from it.
        let (data, coords, batches) = if self.from_run {
            (&[][..], &[][..], &[][..])
        } else {
            (data, coords, batches)
        };
        argv.extend(data.iter().cloned());
        for (flag, many, files) in [
            ("--coord", self.coord, coords),
            ("--batch-files", self.batch, batches),
        ] {
            if let (Some(many), false) = (many, files.is_empty()) {
                argv.extend(emit(flag, many, files.to_vec()));
            }
        }
        argv.push(flag_value("--out", out));
        for f in &self.fields {
            argv.extend(f.argv());
        }
        argv
    }
}

/// Whether clap takes `argv` (without the program); its complaint if not.
pub fn check(cli: &Command, argv: &[String]) -> Result<(), String> {
    matches(cli, argv).map(|_| ())
}

/// `argv` (without the program) as clap parses it; its complaint if not.
pub fn matches(cli: &Command, argv: &[String]) -> Result<clap::ArgMatches, String> {
    // The method's own command parses its line, from the word naming it:
    // no need to copy all of pinto.
    let (method, n) = crate::cli_path::leaf(cli, argv).ok_or_else(|| {
        format!(
            "pinto has no `{}` command",
            argv.first().map_or("", |m| m.as_str())
        )
    })?;
    legume_tui::form::parse(method, &argv[n - 1..])
}

#[cfg(test)]
#[path = "tests/form.rs"]
mod tests;
