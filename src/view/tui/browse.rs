//! What the viewer browses for: a run to open (`pinto view` with no run),
//! or a marker panel for `lupin annotate`. The browser itself is
//! [`crate::tui::browse`].

use super::super::lupin::{self, Panel};
use crate::tui::browse::{Browser, Header, Outcome, Wanted};
use crate::util::metadata::PintoMetadata;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::widgets::Paragraph;
use std::borrow::Cow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Run manifests (`*.pinto.json`), each described in a line; the newest
/// first chosen.
pub struct Runs;

impl Wanted for Runs {
    type About = String;

    fn header(&self) -> Header {
        Header {
            title: "pinto view: run to open".into(),
            notes: Vec::new(),
            what: "runs (*.pinto.json)",
            star: None,
            verb: "open",
            columns: None,
        }
    }

    fn file(&self, path: &Path, name: &str) -> Option<String> {
        name.ends_with(".pinto.json").then(|| describe_run(path))
    }

    fn describe<'a>(&self, about: &'a String) -> Cow<'a, str> {
        Cow::Borrowed(about)
    }

    fn best(&self, dir: &Path, files: &[(&str, &String)]) -> Option<String> {
        files
            .iter()
            .max_by_key(|(n, _)| {
                std::fs::metadata(dir.join(n))
                    .and_then(|m| m.modified())
                    .ok()
            })
            .map(|(n, _)| n.to_string())
    }
}

/// Marker panels for level `tag`, their genes counted against the run's
/// (upper-case symbols; empty when the run's genes are unknown). The one
/// covering the most of them is starred.
pub struct Panels {
    pub known: HashSet<String>,
    pub tag: String,
}

/// A panel's cell types, genes, and genes found in the run.
pub type Counts = (usize, usize, usize);

impl Wanted for Panels {
    type About = Counts;

    fn header(&self) -> Header {
        Header {
            title: format!("marker panel for level {}", self.tag),
            notes: Vec::new(),
            what: "marker panels (gene<TAB>type)",
            star: Some("covers the most of this run's genes"),
            verb: "annotate",
            columns: Some("   types  genes  in run  name"),
        }
    }

    fn file(&self, path: &Path, _name: &str) -> Option<Counts> {
        read_panel(path, &self.known)
    }

    fn describe<'a>(&self, &(types, genes, found): &'a Counts) -> Cow<'a, str> {
        format!("{types} types, {genes} genes, {found} in run").into()
    }

    fn row(&self, name: &str, &(types, genes, found): &Counts, _name_w: usize) -> String {
        format!("{types:>6} {genes:>6} {found:>7}  {name}")
    }

    fn best(&self, _dir: &Path, files: &[(&str, &Counts)]) -> Option<String> {
        files
            .iter()
            .max_by(|(x, (_, a, fa)), (y, (_, b, fb))| {
                (fa, a).cmp(&(fb, b)).then_with(|| y.len().cmp(&x.len()))
            })
            .map(|(n, _)| n.to_string())
    }
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
    let name = crate::tui::name(path);
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

/// Browse for a run before the viewer opens, full screen: the path picked
/// (relative to the working directory when under it), `None` when
/// cancelled.
pub fn pick_run() -> anyhow::Result<Option<PathBuf>> {
    let mut b = Browser::open(lupin::canonical(&std::env::current_dir()?), Runs, None);
    let logging = log::max_level();
    log::set_max_level(log::LevelFilter::Off);
    let mut terminal = ratatui::init();
    let picked = (|| -> anyhow::Result<Option<PathBuf>> {
        loop {
            terminal.draw(|f| {
                let area = f.area();
                let width = usize::from(area.width).min(110);
                let lines = b.lines(usize::from(area.height), width);
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
                    Outcome::Chosen(c) => return Ok(Some(c.file())),
                    Outcome::Ignored | Outcome::Moved => {}
                }
            }
        }
    })();
    ratatui::restore();
    log::set_max_level(logging);
    Ok(picked?.map(|p| crate::tui::relative(&p)))
}
