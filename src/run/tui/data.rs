//! The data files a session fits, with the coordinate and batch files of
//! each.

use crate::tui::browse::{is_data, size_of, Header, Wanted};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What `pinto run` browses for: count backends to fit, their coordinate
/// files, or their batch label files; several at once either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    Data,
    Coord,
    Batch,
}

impl Wanted for Pick {
    /// The file's size.
    type About = String;

    fn header(&self) -> Header {
        let (title, what) = match self {
            Pick::Data => ("Data to fit", "backends (.zarr, .zarr.zip, .h5)"),
            Pick::Coord => (
                "Coordinates, one file per data file",
                "coordinate files (.csv, .tsv, .txt, .parquet, .zarr)",
            ),
            Pick::Batch => (
                "Batch labels, one file per data file",
                "label files (.txt, .tsv, .csv, gzipped too)",
            ),
        };
        Header {
            title: title.into(),
            notes: Vec::new(),
            what,
            star: None,
            verb: "take",
            columns: None,
        }
    }

    fn file(&self, path: &Path, name: &str) -> Option<String> {
        let wanted = match self {
            Pick::Data => is_data(name),
            Pick::Coord => is_coord(name),
            Pick::Batch => is_batch(name),
        };
        wanted.then(|| size_of(path))
    }

    fn store(&self, _path: &Path, _name: &str) -> Option<String> {
        (*self != Pick::Batch).then(String::new)
    }

    fn describe<'a>(&self, size: &'a String) -> Cow<'a, str> {
        Cow::Borrowed(size)
    }

    fn many(&self) -> bool {
        true
    }
}

/// Endings of batch label files: plain or gzipped text.
const BATCH_ENDINGS: &[&str] = &[".txt", ".tsv", ".csv", ".txt.gz", ".tsv.gz", ".csv.gz"];

/// Endings of coordinate files: text tables, parquet, or a zarr store.
const COORD_ENDINGS: &[&str] = &[
    ".txt",
    ".tsv",
    ".csv",
    ".txt.gz",
    ".tsv.gz",
    ".csv.gz",
    ".parquet",
    ".zarr.zip",
    ".zarr",
];

fn is_batch(name: &str) -> bool {
    BATCH_ENDINGS.iter().any(|e| name.ends_with(e))
}

fn is_coord(name: &str) -> bool {
    COORD_ENDINGS.iter().any(|e| name.ends_with(e))
}

/// A data file with its coordinates and batch labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    pub data: PathBuf,
    pub coord: Option<PathBuf>,
    /// The batch label file: one label per cell.
    pub batch: Option<PathBuf>,
    /// The one batch every cell is in, as typed; wins over `batch`.
    pub name: Option<String>,
    /// Labels of `batch` renamed: old to new.
    pub renames: BTreeMap<String, String>,
    /// `batch`'s distinct labels with their cell counts, once read; why
    /// it did not read otherwise.
    pub labels: Option<Result<BTreeMap<String, usize>, String>>,
    /// Features × cells, or why the file does not open.
    pub info: String,
    /// The file's cell count, once known.
    pub cells: Option<usize>,
}

impl Pair {
    /// A data file not yet described; [`describe`] fills that in.
    pub fn pending(data: PathBuf) -> Self {
        Pair {
            data,
            coord: None,
            batch: None,
            name: None,
            renames: BTreeMap::new(),
            labels: None,
            info: "reading…".into(),
            cells: None,
        }
    }

    /// The coordinate or batch file of this data file.
    pub fn side(&self, pick: Pick) -> Option<&PathBuf> {
        match pick {
            Pick::Coord => self.coord.as_ref(),
            Pick::Batch => self.batch.as_ref(),
            Pick::Data => Some(&self.data),
        }
    }

    /// Set the coordinate or batch file; the data file stays. A batch file
    /// replaces a typed batch name, and its labels are read again.
    pub fn set(&mut self, pick: Pick, file: Option<PathBuf>) {
        match pick {
            Pick::Coord => self.coord = file,
            Pick::Batch => {
                if file != self.batch {
                    self.renames.clear();
                    self.labels = None;
                }
                if file.is_some() {
                    self.name = None;
                }
                self.batch = file;
            }
            Pick::Data => {}
        }
    }

    /// Back to pinto's own rule: the file is its own batch.
    pub fn clear_batch(&mut self) {
        self.set(Pick::Batch, None);
        self.name = None;
    }
}

/// `2000 features × 5000 cells` and the cell count, or why the file does
/// not open.
pub fn describe(path: &Path) -> (String, Option<usize>) {
    use data_beans::sparse_io::open_sparse_matrix_by_path;
    match open_sparse_matrix_by_path(&path.to_string_lossy()) {
        Ok(m) => match (m.num_rows(), m.num_columns()) {
            (Some(r), Some(c)) => (format!("{r} features × {c} cells"), Some(c)),
            _ => ("opens".to_string(), None),
        },
        Err(e) => (format!("does not open: {e}"), None),
    }
}

/// A file name without the endings data, coordinate and label files carry.
#[must_use]
pub fn stem(path: &Path) -> String {
    let name = crate::tui::name(path);
    let mut name = data_beans::hdf5_io::strip_backend_suffix(&name).to_string();
    while let Some(end) = COORD_ENDINGS
        .iter()
        .filter(|e| name.ends_with(*e))
        .max_by_key(|e| e.len())
    {
        name.truncate(name.len() - end.len());
    }
    name
}

/// Words that say a file holds labels, not which sample it is.
const LABEL_WORDS: &[&str] = &["batch", "batches", "label", "labels", "membership"];

/// Words that say a file holds coordinates, not which sample it is.
const COORD_WORDS: &[&str] = &[
    "coord",
    "coords",
    "coordinate",
    "coordinates",
    "position",
    "positions",
    "location",
    "locations",
    "centroid",
    "centroids",
    "spatial",
    "tissue",
    "xy",
];

/// The words of a name: lower case, split on anything not a letter or
/// digit.
fn words(name: &str) -> Vec<String> {
    name.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// A name reduced to the words that identify its sample: lower case,
/// label and coordinate words dropped. `s1_batch` and `S1` both give
/// `[s1]`.
fn key(name: &str) -> Vec<String> {
    words(name)
        .into_iter()
        .filter(|w| !LABEL_WORDS.contains(&w.as_str()) && !COORD_WORDS.contains(&w.as_str()))
        .collect()
}

/// How a file's name fits a data file's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Fit {
    /// One name's words start the other's: `s1` and `s1_cells_batch`.
    /// Never `s1` and `s10`, or `s1` and `s1b`: words are whole.
    Extends,
    /// The same words, separators aside: `rep2` and `rep_batch_2`.
    Same,
}

fn fit(data: &Path, file: &Path) -> Option<Fit> {
    let (d, f) = (key(&stem(data)), key(&stem(file)));
    if d.is_empty() || f.is_empty() {
        return None;
    }
    if d.concat() == f.concat() {
        return Some(Fit::Same);
    }
    let (short, long) = if d.len() < f.len() {
        (&d, &f)
    } else {
        (&f, &d)
    };
    long.starts_with(short).then_some(Fit::Extends)
}

/// Each of `data`'s file among `files`, by name: the one that fits it
/// best, if only one does. A file named exactly for one data file goes to
/// no other, and a file two data files would take goes to neither.
fn pair_up(data: &[&Path], files: &[PathBuf]) -> Vec<Option<usize>> {
    let fits: Vec<Vec<Option<Fit>>> = data
        .iter()
        .map(|d| files.iter().map(|f| fit(d, f)).collect())
        .collect();
    let exact_for_some = |j: usize| fits.iter().any(|row| row[j] == Some(Fit::Same));
    let chosen: Vec<Option<usize>> = fits
        .iter()
        .map(|row| {
            let best = row.iter().flatten().max()?;
            let cands: Vec<usize> = (0..files.len())
                .filter(|&j| row[j] == Some(*best) && (*best == Fit::Same || !exact_for_some(j)))
                .collect();
            match cands[..] {
                [j] => Some(j),
                _ => None,
            }
        })
        .collect();
    chosen
        .iter()
        .map(|c| c.filter(|j| chosen.iter().filter(|o| **o == Some(*j)).count() == 1))
        .collect()
}

/// How [`assign`] paired the files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Paired {
    /// Each data file found the one file named for it.
    ByName(usize),
    /// No name matched; as many files as data files went in order.
    InOrder,
    /// Neither: what could be matched by name was, the rest is left unset.
    Partly(usize),
}

/// Give each pair the one coordinate or batch file (`pick`) named for its
/// sample ([`pair_up`]). When no name matches at all and the counts
/// agree, files go in the order listed, and the caller says so.
pub fn assign(pairs: &mut [Pair], files: &[PathBuf], pick: Pick) -> Paired {
    let data: Vec<&Path> = pairs.iter().map(|p| p.data.as_path()).collect();
    let none_fit = data
        .iter()
        .all(|d| files.iter().all(|f| fit(d, f).is_none()));
    let chosen = pair_up(&data, files);
    let mut matched = 0;
    for (p, c) in pairs.iter_mut().zip(chosen) {
        if let Some(j) = c {
            p.set(pick, Some(files[j].clone()));
            matched += 1;
        }
    }
    if matched == pairs.len() {
        Paired::ByName(matched)
    } else if none_fit && files.len() == pairs.len() {
        for (p, f) in pairs.iter_mut().zip(files) {
            p.set(pick, Some(f.clone()));
        }
        Paired::InOrder
    } else {
        Paired::Partly(matched)
    }
}

/// Files in `dir`, and in its `spatial/` folder, that say they hold what
/// `pick` wants: its ending, and one of its words in the name but none of
/// the other kind's.
#[must_use]
pub fn side_files_in(dir: &Path, pick: Pick) -> Vec<PathBuf> {
    let (ending, mine, theirs): (fn(&str) -> bool, _, _) = match pick {
        Pick::Coord => (is_coord, COORD_WORDS, LABEL_WORDS),
        Pick::Batch => (is_batch, LABEL_WORDS, COORD_WORDS),
        Pick::Data => return Vec::new(),
    };
    let mut found: Vec<PathBuf> = [dir.to_path_buf(), dir.join("spatial")]
        .iter()
        .flat_map(|d| std::fs::read_dir(d).into_iter().flatten().flatten())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_lowercase();
            let w = words(&n);
            let has = |list: &[&str]| w.iter().any(|x| list.contains(&x.as_str()));
            // Only a `.zarr` store may be a folder.
            ending(&n)
                && has(mine)
                && !has(theirs)
                && (n.ends_with(".zarr") || e.file_type().is_ok_and(|t| !t.is_dir()))
        })
        .map(|e| e.path())
        .collect();
    found.sort();
    found
}

/// The file among `files` of each of `data`, all in one folder: the one
/// named for it ([`pair_up`]). When the folder holds only one data file
/// (`alone`) and no name fits, a single file with a generic name
/// (`tissue_positions`) is its.
#[must_use]
pub fn beside(data: &[&Path], files: &[PathBuf], alone: bool) -> Vec<Option<PathBuf>> {
    let mut out: Vec<Option<PathBuf>> = pair_up(data, files)
        .into_iter()
        .map(|c| c.map(|j| files[j].clone()))
        .collect();
    if let ([d], [one], [None]) = (data, files, &out[..]) {
        if alone && fit(d, one).is_none() && key(&stem(one)).is_empty() {
            out[0] = Some(one.clone());
        }
    }
    out
}

/// How many count backends `dir` holds.
#[must_use]
pub fn data_in(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| is_data(&e.file_name().to_string_lossy()))
        .count()
}

/// Why the coordinate files as given cannot be passed: pinto wants one per
/// data file, or none. (Batches can be mixed: see [`super::batch`].)
#[must_use]
pub fn coord_problem(pairs: &[Pair]) -> Option<String> {
    let with = pairs.iter().filter(|p| p.coord.is_some()).count();
    (with > 0 && with < pairs.len()).then(|| {
        format!(
            "{with} of {} data files have coordinates; give each one, or clear them all",
            pairs.len()
        )
    })
}

#[cfg(test)]
#[path = "tests/data.rs"]
mod tests;
