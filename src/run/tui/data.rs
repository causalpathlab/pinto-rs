//! The data files a session fits, with the coordinate and batch files of
//! each.

use crate::tui::browse::{is_data, size_of, Header, Wanted};
use std::borrow::Cow;
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
    pub batch: Option<PathBuf>,
    /// Features × cells, or why the file does not open.
    pub info: String,
}

impl Pair {
    /// A data file not yet described; [`describe`] fills that in.
    pub fn pending(data: PathBuf) -> Self {
        Pair {
            data,
            coord: None,
            batch: None,
            info: "reading…".into(),
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

    /// Set the coordinate or batch file; the data file stays.
    pub fn set(&mut self, pick: Pick, file: Option<PathBuf>) {
        match pick {
            Pick::Coord => self.coord = file,
            Pick::Batch => self.batch = file,
            Pick::Data => {}
        }
    }
}

/// `2000 features × 5000 cells`, or why the file does not open.
pub fn describe(path: &Path) -> String {
    use data_beans::sparse_io::open_sparse_matrix_by_path;
    match open_sparse_matrix_by_path(&path.to_string_lossy()) {
        Ok(m) => match (m.num_rows(), m.num_columns()) {
            (Some(r), Some(c)) => format!("{r} features × {c} cells"),
            _ => "opens".to_string(),
        },
        Err(e) => format!("does not open: {e}"),
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

/// A name reduced to what identifies its sample: lower case, label and
/// coordinate words and separators dropped. `s1_batch` and `S1` both give
/// `s1`.
fn key(name: &str) -> String {
    words(name)
        .into_iter()
        .filter(|w| !LABEL_WORDS.contains(&w.as_str()) && !COORD_WORDS.contains(&w.as_str()))
        .collect()
}

/// Whether a file's name says it belongs to a data file's: the same key.
/// A longer one that extends it at a word boundary counts too (`s1` and
/// `s1_cells_batch`), never one whose number runs on (`s1` and `s10`).
fn same_sample(data: &Path, file: &Path) -> bool {
    let (d, b) = (key(&stem(data)), key(&stem(file)));
    if d.is_empty() || b.is_empty() {
        return false;
    }
    if d == b {
        return true;
    }
    let (short, long) = if d.len() < b.len() {
        (&d, &b)
    } else {
        (&b, &d)
    };
    let runs_on = |a: Option<char>, b: Option<char>| {
        a.zip(b)
            .is_some_and(|(a, b)| a.is_ascii_digit() == b.is_ascii_digit())
    };
    long.starts_with(short.as_str())
        && !runs_on(short.chars().last(), long[short.len()..].chars().next())
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
/// sample. When no name matches at all and the counts agree, files go in
/// the order listed, and the caller says so. A data file that two files
/// fit, or that shares its one match with another, is left without one.
pub fn assign(pairs: &mut [Pair], files: &[PathBuf], pick: Pick) -> Paired {
    let fits: Vec<Vec<usize>> = pairs
        .iter()
        .map(|p| {
            (0..files.len())
                .filter(|&j| same_sample(&p.data, &files[j]))
                .collect()
        })
        .collect();
    let mut matched = 0;
    for (i, f) in fits.iter().enumerate() {
        let shared = |j: usize| fits.iter().filter(|g| g.contains(&j)).count() > 1;
        if let [j] = f[..] {
            if !shared(j) {
                pairs[i].set(pick, Some(files[j].clone()));
                matched += 1;
            }
        }
    }
    if matched == pairs.len() {
        Paired::ByName(matched)
    } else if fits.iter().all(Vec::is_empty) && files.len() == pairs.len() {
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

/// The one of `files` named for `data`'s sample; none unless exactly one
/// fits. When none is named for any sample and `alone` (the only data file
/// in its folder), a single file with a generic name (`tissue_positions`)
/// is its.
#[must_use]
pub fn beside(data: &Path, files: &[PathBuf], alone: bool) -> Option<PathBuf> {
    match files
        .iter()
        .filter(|l| same_sample(data, l))
        .collect::<Vec<_>>()[..]
    {
        [one] => Some(one.clone()),
        [] if alone => match files {
            [one] if key(&stem(one)).is_empty() => Some(one.clone()),
            _ => None,
        },
        _ => None,
    }
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

/// Why the coordinate or batch files (`pick`) as given cannot be passed:
/// pinto wants one per data file, or none.
#[must_use]
pub fn side_problem(pairs: &[Pair], pick: Pick) -> Option<String> {
    let with = pairs.iter().filter(|p| p.side(pick).is_some()).count();
    let what = match pick {
        Pick::Coord => "coordinates",
        _ => "batch labels",
    };
    (with > 0 && with < pairs.len()).then(|| {
        format!(
            "{with} of {} data files have {what}; give each one, or clear them all",
            pairs.len()
        )
    })
}

#[cfg(test)]
#[path = "tests/data.rs"]
mod tests;
