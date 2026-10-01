//! What each data file's cells count as batches, and the label files the
//! command line needs for that.
//!
//! A data file's batch is pinto's own rule (the file is its own batch,
//! named by its order: 0, 1, …), one name typed for all its cells, or a
//! label file with one label per cell, whose labels can be renamed. Two
//! files given one name are one batch. When any file is not left to
//! pinto's rule, every file gets a label file: the one given when it is
//! passed as it is, else one written for the run under `{out}.batches/`.

use super::data::{stem, Pair};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What a data file's cells count as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind<'a> {
    /// pinto's own rule: the file is its own batch.
    File,
    /// Every cell in the one batch named.
    Named(&'a str),
    /// One label per cell, from this file.
    Labels(&'a Path),
}

#[must_use]
pub fn kind(p: &Pair) -> Kind<'_> {
    match (&p.name, &p.batch) {
        (Some(n), _) => Kind::Named(n),
        (None, Some(f)) => Kind::Labels(f),
        (None, None) => Kind::File,
    }
}

/// Whether every file is left to pinto's rule: then no batch file is
/// passed at all.
#[must_use]
pub fn all_own(pairs: &[Pair]) -> bool {
    pairs.iter().all(|p| kind(p) == Kind::File)
}

/// What a label file written for a run holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Made {
    /// The label, once per cell.
    Repeat(String, usize),
    /// The lines of a label file, some labels renamed (old to new).
    Renamed(PathBuf, BTreeMap<String, String>),
}

/// A data file's label file on the command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arg {
    /// A label file passed as it is.
    Given(PathBuf),
    /// One written for the run: its name in `{out}.batches/`, and what it
    /// holds.
    Made(String, Made),
}

/// The label file of each of `pairs`, none when every file is left to
/// pinto's rule; why they cannot be made yet otherwise.
pub fn args(pairs: &[Pair]) -> Result<Option<Vec<Arg>>, String> {
    if all_own(pairs) {
        return Ok(None);
    }
    let mut taken: Vec<String> = Vec::new();
    let mut file_name = |p: &Pair| {
        let name = super::first_free(&stem(&p.data), |s| taken.contains(&format!("{s}.txt")));
        let name = format!("{name}.txt");
        taken.push(name.clone());
        name
    };
    let cells = |p: &Pair| {
        p.cells.ok_or_else(|| {
            format!(
                "{}: its cell count is not known yet, so its batch file cannot be written",
                crate::tui::name(&p.data)
            )
        })
    };
    pairs
        .iter()
        .map(|p| {
            Ok(match kind(p) {
                Kind::File => Arg::Made(file_name(p), Made::Repeat(stem(&p.data), cells(p)?)),
                Kind::Named(n) => Arg::Made(file_name(p), Made::Repeat(n.to_string(), cells(p)?)),
                Kind::Labels(f) if p.renames.is_empty() => Arg::Given(f.to_path_buf()),
                Kind::Labels(f) => Arg::Made(
                    file_name(p),
                    Made::Renamed(f.to_path_buf(), p.renames.clone()),
                ),
            })
        })
        .collect::<Result<Vec<_>, String>>()
        .map(Some)
}

/// Write `made` to `path`, never over an existing file.
pub fn write(path: &Path, made: &Made) -> anyhow::Result<()> {
    let text = match made {
        Made::Repeat(label, n) => format!("{label}\n").repeat(*n),
        Made::Renamed(from, renames) => {
            let mut text = String::new();
            for line in crate::util::common::read_lines(&from.to_string_lossy())? {
                text.push_str(renames.get(&*line).map_or(&*line, String::as_str));
                text.push('\n');
            }
            text
        }
    };
    super::script::write_new(path, text.as_bytes())
}

/// A label file's distinct labels with their cell counts, read as pinto
/// reads it: a label per line, gzipped or not.
pub fn label_counts(path: &Path) -> Result<BTreeMap<String, usize>, String> {
    use std::io::BufRead;
    let text = legume_numeric::matrix::common_io::open_buf_reader(&path.to_string_lossy())
        .map_err(|e| e.to_string())?;
    // Streamed: a label per cell, but only tens of labels kept.
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for line in text.lines() {
        let line = line.map_err(|e| e.to_string())?;
        match counts.get_mut(&line) {
            Some(n) => *n += 1,
            None => {
                counts.insert(line, 1);
            }
        }
    }
    Ok(counts)
}

/// One batch the files make.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    pub name: String,
    pub files: usize,
    /// Its cells; `None` while some are not known yet.
    pub cells: Option<usize>,
}

/// The batches the files make, by name, and what is not known yet: label
/// files still being read, or that did not read.
#[must_use]
pub fn summary(pairs: &[Pair]) -> (Vec<Batch>, Vec<String>) {
    let mut out: BTreeMap<String, (std::collections::BTreeSet<usize>, Option<usize>)> =
        BTreeMap::new();
    let mut notes = Vec::new();
    let mut add = |name: String, i: usize, cells: Option<usize>| {
        let e = out.entry(name).or_insert((Default::default(), Some(0)));
        e.0.insert(i);
        e.1 = e.1.zip(cells).map(|(a, b)| a + b);
    };
    let own = all_own(pairs);
    for (i, p) in pairs.iter().enumerate() {
        match kind(p) {
            // pinto numbers the files' batches in order.
            Kind::File if own => add(i.to_string(), i, p.cells),
            Kind::File => add(stem(&p.data), i, p.cells),
            Kind::Named(n) => add(n.to_string(), i, p.cells),
            Kind::Labels(f) => match &p.labels {
                Some(Ok(counts)) => {
                    for (label, n) in counts.iter() {
                        let name = p.renames.get(label).unwrap_or(label);
                        add(name.clone(), i, Some(*n));
                    }
                }
                Some(Err(e)) => notes.push(format!("{}: {e}", crate::tui::name(f))),
                None => notes.push(format!("{}: reading its labels…", crate::tui::name(f))),
            },
        }
    }
    let batches = out
        .into_iter()
        .map(|(name, (files, cells))| Batch {
            name,
            files: files.len(),
            cells,
        })
        .collect();
    (batches, notes)
}

#[cfg(test)]
#[path = "tests/batch.rs"]
mod tests;
