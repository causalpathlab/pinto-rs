//! A file browser, one folder at a time: its folders, and the files that
//! are wanted, each with a line about it.
//!
//! What is wanted, how a file is described and which one to start on are
//! the caller's, through [`Wanted`]; the browser only moves, narrows, marks
//! and draws. A `.zarr` store is a folder on disk but a file here: it is
//! never opened, only offered when the caller takes stores.

use super::style::{bold, dim, first_row, selected, tail};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// What a browser looks for, and how it speaks of it.
pub(crate) trait Wanted {
    /// What is shown about a listed file: its size, a run's kind, a
    /// panel's counts.
    type About: Clone;

    /// The popup's words.
    fn header(&self) -> Header;

    /// The file `name` at `path` with what to show of it, if it is wanted.
    fn file(&self, path: &Path, name: &str) -> Option<Self::About>;

    /// The `.zarr` store `name` at `path` as a file, when stores are wanted.
    fn store(&self, _path: &Path, _name: &str) -> Option<Self::About> {
        None
    }

    /// A file's row, given the widest name shown (`name_w`): most callers
    /// want [`name_then`].
    fn row(&self, name: &str, about: &Self::About, name_w: usize) -> String;

    /// The file to start on among those listed in `dir`.
    fn best(&self, _dir: &Path, _files: &[(&str, &Self::About)]) -> Option<String> {
        None
    }

    /// Whether space marks several files to take together.
    fn many(&self) -> bool {
        false
    }
}

/// A row of the name padded to `name_w`, then `about`; just the name when
/// there is nothing to say.
#[must_use]
pub fn name_then(name: &str, about: &str, name_w: usize) -> String {
    if about.is_empty() {
        name.to_string()
    } else {
        format!("{name:<name_w$}  {about}")
    }
}

/// What the popup says about what is wanted.
pub(crate) struct Header {
    pub title: String,
    /// The files wanted, as in "no {what} here".
    pub what: &'static str,
    /// What a star by the starting file means; no star without it.
    pub star: Option<&'static str>,
    /// What enter does to a file: "open", "annotate".
    pub verb: &'static str,
    /// A line naming the columns of the rows, over them.
    pub columns: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Entry<A> {
    Up,
    Dir(String),
    File(String, A),
}

impl<A> Entry<A> {
    pub fn name(&self) -> &str {
        match self {
            Entry::Up => "..",
            Entry::Dir(n) | Entry::File(n, _) => n,
        }
    }

    fn is_file(&self) -> bool {
        matches!(self, Entry::File(..))
    }
}

/// What a key did in the browser.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Not a browser key.
    Ignored,
    Moved,
    Cancelled,
    Chosen(Chosen),
}

/// Files chosen, never none: the one under the cursor, or, where several
/// can be taken, the marked ones.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Chosen {
    first: PathBuf,
    rest: Vec<PathBuf>,
}

impl Chosen {
    fn one(path: PathBuf) -> Self {
        Chosen {
            first: path,
            rest: Vec::new(),
        }
    }

    /// The file, where one is taken at a time.
    #[must_use]
    pub fn file(self) -> PathBuf {
        debug_assert!(
            self.rest.is_empty(),
            "several files chosen where one is taken"
        );
        self.first
    }

    /// Every file taken, in order.
    #[must_use]
    pub fn files(self) -> Vec<PathBuf> {
        std::iter::once(self.first).chain(self.rest).collect()
    }
}

pub(crate) struct Browser<W: Wanted> {
    pub dir: PathBuf,
    pub want: W,
    entries: Vec<Entry<W::About>>,
    /// Typed letters narrow the list to names containing them.
    pub filter: String,
    /// Position among the shown entries.
    pub row: usize,
    /// The file to start on, as the caller chose it.
    best: Option<String>,
    /// Files marked with space, in any folder.
    pub marked: BTreeSet<PathBuf>,
    /// Whether hidden files were read: only once the filter asks for them,
    /// so a folder's dotfiles are not opened on every visit.
    hidden: bool,
}

impl<W: Wanted> Browser<W> {
    pub fn open(dir: PathBuf, want: W, select: Option<&str>) -> Self {
        let mut b = Browser {
            dir,
            want,
            entries: Vec::new(),
            filter: String::new(),
            row: 0,
            best: None,
            marked: BTreeSet::new(),
            hidden: false,
        };
        b.read(select);
        b
    }

    /// Read the current folder; the cursor goes to `select` if it is
    /// listed, else to the best file, else to the first entry after `..`.
    pub fn read(&mut self, select: Option<&str>) {
        self.hidden = false;
        self.entries = list_dir(&self.dir, &self.want, Read::Visible);
        let files: Vec<(&str, &W::About)> = self
            .entries
            .iter()
            .filter_map(|e| match e {
                Entry::File(n, a) => Some((n.as_str(), a)),
                _ => None,
            })
            .collect();
        self.best = self.want.best(&self.dir, &files);
        self.filter.clear();
        let want = select.map(str::to_string).or_else(|| self.best.clone());
        self.row = want
            .and_then(|w| self.shown().iter().position(|e| e.name() == w))
            .unwrap_or(usize::from(self.shown().len() > 1));
    }

    /// The entries the filter lets through, `..` always first. Hidden ones
    /// only when the filter starts with `.`.
    pub fn shown(&self) -> Vec<&Entry<W::About>> {
        let f = self.filter.to_lowercase();
        let hidden = f.starts_with('.');
        self.entries
            .iter()
            .filter(|e| {
                matches!(e, Entry::Up)
                    || ((hidden || !e.name().starts_with('.'))
                        && e.name().to_lowercase().contains(&f))
            })
            .collect()
    }

    /// The entry under the cursor.
    pub fn current(&self) -> Option<&Entry<W::About>> {
        self.shown().get(self.row).copied()
    }

    pub fn go_up(&mut self) {
        let from = super::name(&self.dir);
        if let Some(parent) = self.dir.parent().map(Path::to_path_buf) {
            self.dir = parent;
            self.read(Some(&from));
        }
    }

    /// The file under the cursor, if it is on one.
    fn file_here(&self) -> Option<PathBuf> {
        let e = self.current().filter(|e| e.is_file())?;
        Some(self.dir.join(e.name()))
    }

    /// Enter on a file: take the marked files, or this one when none is
    /// (only a browser of [`Wanted::many`] marks any).
    fn take_here(&mut self) -> Outcome {
        let Some(here) = self.file_here() else {
            return Outcome::Ignored;
        };
        let mut marked = std::mem::take(&mut self.marked).into_iter();
        Outcome::Chosen(match marked.next() {
            Some(first) => Chosen {
                first,
                rest: marked.collect(),
            },
            None => Chosen::one(here),
        })
    }

    /// A key: move, open a folder, narrow, mark, cancel, or choose.
    pub fn key(&mut self, k: KeyEvent) -> Outcome {
        if self.want.many() {
            if let Some(o) = self.mark_key(k) {
                return o;
            }
        }
        let last = self.shown().len().saturating_sub(1);
        match k.code {
            KeyCode::Esc => return Outcome::Cancelled,
            KeyCode::Up => self.row = self.row.saturating_sub(1),
            KeyCode::Down => self.row = (self.row + 1).min(last),
            KeyCode::PageUp => self.row = self.row.saturating_sub(10),
            KeyCode::PageDown => self.row = (self.row + 10).min(last),
            KeyCode::Home => self.row = 0,
            KeyCode::End => self.row = last,
            KeyCode::Left => self.go_up(),
            KeyCode::Backspace => {
                if self.filter.pop().is_none() {
                    self.go_up();
                } else {
                    self.row = usize::from(self.shown().len() > 1);
                }
            }
            KeyCode::Char('~') if self.filter.is_empty() => {
                if let Ok(home) = std::env::var("HOME") {
                    self.dir = PathBuf::from(home);
                    self.read(None);
                }
            }
            KeyCode::Enter | KeyCode::Right => match self.current() {
                None => return Outcome::Ignored,
                Some(Entry::Up) => self.go_up(),
                Some(Entry::Dir(name)) => {
                    let name = name.clone();
                    self.dir.push(name);
                    self.read(None);
                }
                Some(Entry::File(..)) if k.code == KeyCode::Enter => return self.take_here(),
                Some(Entry::File(..)) => return Outcome::Ignored,
            },
            KeyCode::Char(c) => {
                let on = self.current().map(|e| e.name().to_string());
                self.filter.push(c);
                if self.filter.starts_with('.') && !self.hidden {
                    self.hidden = true;
                    let hidden = list_dir(&self.dir, &self.want, Read::HiddenFiles);
                    self.entries.extend(hidden);
                    sort_entries(&mut self.entries);
                }
                let shown = self.shown();
                // Stay where the cursor was if it still shows, else on the
                // best file, else on the first entry after `..`.
                let at = |name: &str| shown.iter().position(|e| e.name() == name);
                self.row = on
                    .as_deref()
                    .filter(|n| *n != "..")
                    .and_then(at)
                    .or_else(|| self.best.as_deref().and_then(at))
                    .unwrap_or(usize::from(shown.len() > 1));
            }
            _ => return Outcome::Ignored,
        }
        Outcome::Moved
    }

    /// Space marks or unmarks the file under the cursor and moves on;
    /// ctrl-a marks every file shown; enter on a file takes the marked
    /// ones, or this one when none is. On a folder enter still opens it, so
    /// files marked here and there can be taken together.
    fn mark_key(&mut self, k: KeyEvent) -> Option<Outcome> {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Char(' ') => {
                if let Some(path) = self.file_here() {
                    if !self.marked.remove(&path) {
                        self.marked.insert(path);
                    }
                    self.row = (self.row + 1).min(self.shown().len().saturating_sub(1));
                }
                Some(Outcome::Moved)
            }
            KeyCode::Char('a') if ctrl => {
                let files: Vec<PathBuf> = self
                    .shown()
                    .iter()
                    .filter(|e| e.is_file())
                    .map(|e| self.dir.join(e.name()))
                    .collect();
                self.marked.extend(files);
                Some(Outcome::Moved)
            }
            _ => None,
        }
    }

    /// The browser's lines, at most `height` tall and `width` wide: a side
    /// panel's or the inside of a popup. What the header and footer leave
    /// goes to the entries, so the key hints always show.
    pub fn lines(&self, height: usize, width: usize) -> Vec<Line<'static>> {
        let h = self.want.header();
        let fit = |s: String| -> String { s.chars().take(width).collect() };
        let line = |s: String, style: Style| Line::from(Span::styled(fit(s), style));
        let mut head = vec![line(format!(" {}", h.title), bold())];
        head.push(line(
            format!(
                " {}",
                tail(&super::shown(&self.dir), width.saturating_sub(2))
            ),
            dim(),
        ));
        if !self.filter.is_empty() {
            head.push(line(
                format!(" names with “{}”", self.filter),
                Style::default(),
            ));
        }
        if let Some(c) = h.columns {
            head.push(line(c.to_string(), dim()));
        }

        let shown = self.shown();
        let many = self.want.many();
        let mut foot = Vec::new();
        if !shown.iter().any(|e| e.is_file()) {
            foot.push(line(format!("  no {} here", h.what), dim()));
        }
        foot.push(Line::from(""));
        if let (Some(s), Some(_)) = (h.star, &self.best) {
            foot.push(line(format!(" * {s}"), dim()));
        }
        if many {
            if !self.marked.is_empty() {
                foot.push(line(
                    format!(" {} marked", self.marked.len()),
                    Style::default(),
                ));
            }
            foot.push(line(" space mark  ctrl-a mark all shown".into(), dim()));
            foot.push(line(
                " Enter take the marked, or the one here".into(),
                dim(),
            ));
        }
        foot.push(line(format!(" ↑↓ choose  Enter {}  ← up", h.verb), dim()));
        foot.push(line(" type to narrow  ~ home  Esc cancel".into(), dim()));

        let rows = height.saturating_sub(head.len() + foot.len()).max(3);
        let name_w = shown
            .iter()
            .map(|e| e.name().chars().count() + 1)
            .max()
            .unwrap_or(0)
            .min(36);
        let mut out = head;
        for (i, e) in shown
            .iter()
            .enumerate()
            .skip(first_row(self.row, rows, shown.len()))
            .take(rows)
        {
            let text = match e {
                Entry::Up => " ../".to_string(),
                Entry::Dir(n) => format!(" {n}/"),
                Entry::File(n, about) => {
                    let mark = if many && self.marked.contains(&self.dir.join(n)) {
                        "●"
                    } else if h.star.is_some() && self.best.as_deref() == Some(n) {
                        "*"
                    } else {
                        " "
                    };
                    format!("{mark}{}", self.want.row(n, about, name_w))
                }
            };
            let style = if i == self.row {
                selected()
            } else if e.is_file() {
                Style::default()
            } else {
                dim()
            };
            out.push(line(text, style));
        }
        out.extend(foot);
        out
    }
}

/// Whether `name` is a backend pinto reads (`.zarr`, `.zarr.zip`, `.h5`).
pub(crate) fn is_data(name: &str) -> bool {
    data_beans::hdf5_io::strip_backend_suffix(name) != name
}

/// A file's size, `12.3 MB`.
pub(crate) fn size_of(path: &Path) -> String {
    let bytes = std::fs::metadata(path).map_or(0, |m| m.len()) as f64;
    let (v, unit) = [("kB", 1e3), ("MB", 1e6), ("GB", 1e9)]
        .iter()
        .rev()
        .find(|(_, s)| bytes >= *s)
        .map_or((bytes, "B"), |(u, s)| (bytes / s, *u));
    if unit == "B" {
        format!("{v:.0} B")
    } else {
        format!("{v:.1} {unit}")
    }
}

/// Which entries [`list_dir`] reads.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Read {
    /// Every folder and every wanted file not hidden. Hidden folders are
    /// kept (`shown` lists them only on request): naming them costs
    /// nothing, while a hidden file would be opened to describe it.
    Visible,
    /// Only the hidden files, once the filter asks for them.
    HiddenFiles,
}

/// The entries of `dir` that `read` asks for: `..` first (unless only
/// hidden files are read), folders (not a `.zarr` store) before wanted
/// files, each group by name.
pub(crate) fn list_dir<W: Wanted>(dir: &Path, want: &W, read: Read) -> Vec<Entry<W::About>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let path = e.path();
        let is_dir = path.is_dir();
        let file = !is_dir || name.ends_with(".zarr");
        if file && (name.starts_with('.') != (read == Read::HiddenFiles)) {
            continue;
        }
        if !file {
            if read == Read::Visible {
                out.push(Entry::Dir(name));
            }
            continue;
        }
        let about = if is_dir {
            want.store(&path, &name)
        } else {
            want.file(&path, &name)
        };
        out.extend(about.map(|a| Entry::File(name, a)));
    }
    if read == Read::Visible {
        out.push(Entry::Up);
    }
    sort_entries(&mut out);
    out
}

/// `..` first, then folders, then files, each group by name.
fn sort_entries<A>(entries: &mut [Entry<A>]) {
    let rank = |e: &Entry<A>| match e {
        Entry::Up => 0,
        Entry::Dir(_) => 1,
        Entry::File(..) => 2,
    };
    entries.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.name().cmp(b.name())));
}

#[cfg(test)]
#[path = "tests/browse.rs"]
mod tests;
