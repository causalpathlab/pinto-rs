//! A file browser for the viewer: a run to open (`pinto view` with no
//! run), or a marker panel for `lupin annotate`.
//!
//! ↑ ↓ move, Enter opens a folder or picks a file, ← or Backspace goes up,
//! typing narrows the names, `~` goes home, Esc cancels.

use super::super::lupin::{self, Panel};
use super::tail;
use crate::util::metadata::PintoMetadata;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// What the browser lists besides folders.
pub enum Want {
    /// Run manifests (`*.pinto.json`).
    Runs,
    /// Marker panels, their genes counted against the run's (upper-case
    /// symbols; empty when the run's genes are unknown).
    Panels(HashSet<String>),
}

#[derive(Clone)]
pub enum Entry {
    Up,
    Dir(String),
    /// A run and a one-line description.
    Run(String, String),
    /// A panel: its cell types, genes, and genes found in the run.
    Panel(String, usize, usize, usize),
}

impl Entry {
    fn name(&self) -> &str {
        match self {
            Entry::Up => "..",
            Entry::Dir(n) | Entry::Run(n, _) | Entry::Panel(n, ..) => n,
        }
    }

    fn is_file(&self) -> bool {
        matches!(self, Entry::Run(..) | Entry::Panel(..))
    }
}

pub enum Outcome {
    Moved,
    Cancelled,
    Chosen(PathBuf),
}

pub struct Browser {
    pub dir: PathBuf,
    want: Want,
    entries: Vec<Entry>,
    filter: String,
    row: usize,
    /// The file first chosen: the newest run, or the best-matched panel.
    best: Option<String>,
}

/// Files a panel browser skips outright.
const NOT_PANELS: &[&str] = &[
    "parquet",
    "zarr",
    "zip",
    "h5",
    "h5ad",
    "json",
    "bam",
    "bai",
    "png",
    "pdf",
    "log",
    "safetensors",
    "mtx",
    "idx",
];

/// Largest file read as a candidate panel.
const MAX_PANEL_BYTES: u64 = 8 << 20;

impl Browser {
    pub fn open(dir: PathBuf, want: Want) -> Self {
        let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
        let mut b = Browser {
            dir,
            want,
            entries: Vec::new(),
            filter: String::new(),
            row: 0,
            best: None,
        };
        b.read(None);
        b
    }

    /// List `dir`, landing on `select` (a folder just left), else the best
    /// file, else the first row after `..`.
    fn read(&mut self, select: Option<&str>) {
        self.filter.clear();
        self.entries = list_dir(&self.dir, &self.want);
        self.best = match self.want {
            Want::Runs => newest(&self.dir, &self.entries),
            Want::Panels(_) => best_panel(&self.entries),
        };
        let want = select.map(String::from).or_else(|| self.best.clone());
        self.row = want
            .and_then(|w| self.entries.iter().position(|e| e.name() == w))
            .unwrap_or(usize::from(self.entries.len() > 1));
    }

    fn shown(&self) -> Vec<&Entry> {
        let f = self.filter.to_lowercase();
        self.entries
            .iter()
            .filter(|e| matches!(e, Entry::Up) || e.name().to_lowercase().contains(&f))
            .collect()
    }

    fn go_up(&mut self) {
        let from = lupin::file_name(&self.dir);
        if let Some(parent) = self.dir.parent().map(Path::to_path_buf) {
            self.dir = parent;
            self.read(Some(&from));
        }
    }

    pub fn key(&mut self, k: KeyEvent) -> Outcome {
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
            KeyCode::Enter | KeyCode::Right => {
                let Some(entry) = self.shown().get(self.row).map(|e| (*e).clone()) else {
                    return Outcome::Moved;
                };
                match entry {
                    Entry::Up => self.go_up(),
                    Entry::Dir(name) => {
                        self.dir.push(name);
                        self.read(None);
                    }
                    e if k.code == KeyCode::Enter => {
                        return Outcome::Chosen(self.dir.join(e.name()));
                    }
                    _ => {}
                }
            }
            KeyCode::Char(c) => {
                self.filter.push(c);
                let shown = self.shown();
                self.row = self
                    .best
                    .as_ref()
                    .and_then(|best| shown.iter().position(|e| e.name() == best))
                    .unwrap_or(usize::from(shown.len() > 1));
            }
            _ => {}
        }
        Outcome::Moved
    }

    /// The browser's lines, `rows` entries tall at most, `width` wide.
    pub fn lines(&self, title: &str, rows: usize, width: usize) -> Vec<Line<'static>> {
        let bold = Style::default().add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(Color::DarkGray);
        let rev = Style::default().add_modifier(Modifier::REVERSED);
        let (what, pick) = match self.want {
            Want::Runs => ("runs (*.pinto.json)", "open"),
            Want::Panels(_) => ("marker panels (gene<TAB>type)", "annotate"),
        };
        let mut out = vec![
            Line::styled(format!(" {title}"), bold),
            Line::styled(format!(" {}", tail(&shown(&self.dir), width - 2)), dim),
        ];
        if !self.filter.is_empty() {
            out.push(Line::raw(format!(" names with “{}”", self.filter)));
        }
        if let Want::Panels(_) = self.want {
            out.push(Line::styled("   types  genes  in run  name", dim));
        }
        let shown = self.shown();
        let first = self
            .row
            .saturating_sub(rows / 2)
            .min(shown.len().saturating_sub(rows));
        let name_width = shown
            .iter()
            .map(|e| e.name().chars().count() + 1)
            .max()
            .unwrap_or(0)
            .min(36);
        for (i, e) in shown.iter().enumerate().skip(first).take(rows) {
            let star = if self.best.as_deref() == Some(e.name()) && matches!(e, Entry::Panel(..)) {
                '*'
            } else {
                ' '
            };
            let text = match e {
                Entry::Up => " ../".to_string(),
                Entry::Dir(n) => format!(" {n}/"),
                Entry::Run(n, d) => format!(" {n:<name_width$}  {d}"),
                Entry::Panel(n, types, genes, found) => {
                    format!("{star}{types:>6} {genes:>6} {found:>7}  {n}")
                }
            };
            let text: String = text.chars().take(width).collect();
            let style = if i == self.row {
                rev
            } else if e.is_file() {
                Style::default()
            } else {
                dim
            };
            out.push(Line::styled(text, style));
        }
        if !shown.iter().any(|e| e.is_file()) {
            out.push(Line::styled(format!("  no {what} here"), dim));
        }
        out.push(Line::raw(""));
        if let (Want::Panels(_), Some(_)) = (&self.want, &self.best) {
            out.push(Line::styled(" * covers the most of this run's genes", dim));
        }
        out.push(Line::styled(
            format!(" ↑↓ choose  Enter {pick}  ← up  type to narrow"),
            dim,
        ));
        out.push(Line::styled(" ~ home  Esc cancel", dim));
        out
    }
}

fn list_dir(dir: &Path, want: &Want) -> Vec<Entry> {
    let mut dirs = Vec::new();
    let mut found = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let path = e.path();
        if path.is_dir() {
            if !name.ends_with(".zarr") {
                dirs.push(Entry::Dir(name));
            }
            continue;
        }
        let entry = match want {
            Want::Runs => name
                .ends_with(".pinto.json")
                .then(|| Entry::Run(name, describe_run(&path))),
            Want::Panels(known) => {
                read_panel(&path, known).map(|(t, g, f)| Entry::Panel(name, t, g, f))
            }
        };
        found.extend(entry);
    }
    dirs.sort_by(|a, b| a.name().cmp(b.name()));
    found.sort_by(|a, b| a.name().cmp(b.name()));
    let mut out = vec![Entry::Up];
    out.extend(dirs);
    out.extend(found);
    out
}

/// `cage · 814,243 cells · levels L1 final · 2 rounds`, or why the
/// manifest does not read.
fn describe_run(path: &Path) -> String {
    match PintoMetadata::read(path) {
        Ok(m) => {
            let mut parts = vec![
                m.command.clone(),
                format!("{} cells", super::thousands(m.n_cells)),
            ];
            let tags: Vec<String> = m.level_list().into_iter().map(|l| l.tag).collect();
            if !tags.is_empty() {
                parts.push(format!("levels {}", tags.join(" ")));
            }
            match lupin::latest_rounds(path).len() {
                0 => {}
                1 => parts.push("annotated".into()),
                n => parts.push(format!("{n} rounds")),
            }
            parts.join(" · ")
        }
        Err(_) => "does not read as a run".into(),
    }
}

/// `path` as a marker panel: its cell types, genes and genes found in the
/// run. `None` for a file that is not one: too big, a data format, fewer
/// than two cell types, or (when the run's genes are known) none of its.
fn read_panel(path: &Path, known: &HashSet<String>) -> Option<(usize, usize, usize)> {
    let name = lupin::file_name(path);
    let ext = name
        .trim_end_matches(".gz")
        .rsplit('.')
        .next()
        .unwrap_or("");
    let size = std::fs::metadata(path).ok()?.len();
    if size > MAX_PANEL_BYTES || NOT_PANELS.contains(&ext) {
        return None;
    }
    let panel = Panel::read(path).ok()?;
    // All-numeric labels are a table's other columns, not cell types.
    let types = panel
        .types
        .iter()
        .filter(|(t, _)| t.parse::<f64>().is_err())
        .count();
    if types < 2 {
        return None;
    }
    let genes: HashSet<String> = panel
        .types
        .iter()
        .flat_map(|(_, g)| g.iter().map(|x| x.to_uppercase()))
        .collect();
    let found = genes.iter().filter(|g| known.contains(*g)).count();
    if !known.is_empty() && found == 0 {
        return None;
    }
    Some((types, genes.len(), found))
}

/// The panel with the most of the run's genes.
fn best_panel(entries: &[Entry]) -> Option<String> {
    entries
        .iter()
        .filter_map(|e| match e {
            Entry::Panel(n, _, genes, found) => Some(((*found, *genes), n)),
            _ => None,
        })
        .max_by(|(a, x), (b, y)| a.cmp(b).then_with(|| y.len().cmp(&x.len())))
        .map(|(_, n)| n.clone())
}

/// The most recently written run here.
fn newest(dir: &Path, entries: &[Entry]) -> Option<String> {
    entries
        .iter()
        .filter(|e| matches!(e, Entry::Run(..)))
        .max_by_key(|e| {
            std::fs::metadata(dir.join(e.name()))
                .and_then(|m| m.modified())
                .ok()
        })
        .map(|e| e.name().to_string())
}

/// `p` relative to the working directory when under it, as typed.
fn relative(p: &Path) -> PathBuf {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| p.strip_prefix(cwd).ok().map(Path::to_path_buf))
        .filter(|r| !r.as_os_str().is_empty())
        .unwrap_or_else(|| p.to_path_buf())
}

/// A path as shown: relative to the working directory when under it.
fn shown(p: &Path) -> String {
    relative(p).display().to_string()
}

/// Browse for a run before the viewer opens, full screen: the path picked
/// (relative to the working directory when under it), `None` when
/// cancelled.
pub fn pick_run() -> anyhow::Result<Option<PathBuf>> {
    let mut b = Browser::open(std::env::current_dir()?, Want::Runs);
    let logging = log::max_level();
    log::set_max_level(log::LevelFilter::Off);
    let mut terminal = ratatui::init();
    let picked = (|| -> anyhow::Result<Option<PathBuf>> {
        loop {
            terminal.draw(|f| {
                let area = f.area();
                let rows = usize::from(area.height).saturating_sub(10).max(3);
                let width = usize::from(area.width).min(110);
                let lines = b.lines("pinto view: run to open", rows, width);
                f.render_widget(Paragraph::new(lines), area);
            })?;
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Release {
                    continue;
                }
                if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
                    return Ok(None);
                }
                match b.key(k) {
                    Outcome::Cancelled => return Ok(None),
                    Outcome::Chosen(path) => return Ok(Some(path)),
                    Outcome::Moved => {}
                }
            }
        }
    })();
    ratatui::restore();
    log::set_max_level(logging);
    Ok(picked?.map(|p| relative(&p)))
}
