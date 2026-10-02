//! `s`: name the files an export writes. The next free `pinto-*-NNN`
//! comes up selected, so Enter takes it and typing replaces it.

use super::{plots, App};
use crate::tui::field::Field;
use crate::tui::style;
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::text::{Line, Span};
use std::path::Path;

/// What `s` exports, fixed when the dialog opens.
#[derive(Clone, Copy)]
pub enum Export {
    /// The map, with its structure plot where one shows.
    Map {
        structure: bool,
    },
    Grid,
    Heatmap,
}

impl Export {
    fn prefix(self) -> &'static str {
        match self {
            Export::Map { .. } => "pinto-view",
            Export::Grid => "pinto-grid",
            Export::Heatmap => "pinto-heatmap",
        }
    }

    /// The extensions written under the stem itself.
    fn exts(self) -> &'static [&'static str] {
        match self {
            Export::Map { .. } => &["png", "pdf", "txt"],
            Export::Grid => &["png", "txt"],
            Export::Heatmap => &["tsv", "txt"],
        }
    }

    /// Every file an export as `stem` writes: the map's structure plot
    /// goes beside it as `{stem}-structure`.
    fn files(self, stem: &str) -> Vec<String> {
        let mut out: Vec<String> = self.exts().iter().map(|e| format!("{stem}.{e}")).collect();
        if let Export::Map { structure: true } = self {
            out.extend(["png", "txt"].map(|e| format!("{stem}-structure.{e}")));
        }
        out
    }

    /// The first `{prefix}-NNN` none of whose files exist yet.
    fn free_stem(self) -> String {
        (1..)
            .map(|n| format!("{}-{n:03}", self.prefix()))
            .find(|s| self.files(s).iter().all(|f| !Path::new(f).exists()))
            .expect("unbounded")
    }

    /// `name` without an extension this export writes, `~/` expanded.
    fn stem_of(self, name: &str) -> String {
        let name = name.trim();
        let name = match (name.strip_prefix("~/"), std::env::var("HOME")) {
            (Some(rest), Ok(home)) => format!("{home}/{rest}"),
            _ => name.to_string(),
        };
        for ext in self.exts() {
            if let Some(s) = name.strip_suffix(&format!(".{ext}")) {
                return s.to_string();
            }
        }
        name
    }
}

pub struct SaveAs {
    what: Export,
    name: Field,
    /// Enter was pressed on a name whose files exist: Enter again
    /// overwrites them.
    overwrite: bool,
}

impl App<'_> {
    /// `s`: ask for a name, the next free one offered.
    pub(super) fn ask_export(&mut self) {
        let what = if self.view != plots::View::Map {
            if !self.plots.has_heatmap() {
                self.status = "nothing drawn yet".into();
                return;
            }
            Export::Heatmap
        } else if self.grid_shows() {
            Export::Grid
        } else {
            Export::Map {
                structure: self.plots.has_structure() && self.bars.height > 0,
            }
        };
        self.modal = Some(super::annotate::Modal::Save(SaveAs {
            what,
            name: Field::new(what.free_stem()),
            overwrite: false,
        }));
        self.need_panel = true;
    }

    /// A key in the dialog; the dialog back where it stays open.
    pub(super) fn save_key(&mut self, mut s: SaveAs, key: KeyEvent) -> Option<SaveAs> {
        match key.code {
            KeyCode::Esc => {
                self.status = "not saved".into();
                None
            }
            KeyCode::Enter => {
                let stem = s.what.stem_of(&s.name.text);
                if stem.is_empty() || stem.ends_with('/') {
                    self.status = "type a file name".into();
                    return Some(s);
                }
                let parent = Path::new(&stem)
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty());
                if let Some(dir) = parent.filter(|d| !d.is_dir()) {
                    self.status = format!("no folder {}", dir.display());
                    return Some(s);
                }
                let taken: Vec<String> = s
                    .what
                    .files(&stem)
                    .into_iter()
                    .filter(|f| Path::new(f).exists())
                    .collect();
                if !taken.is_empty() && !s.overwrite {
                    self.status = format!(
                        "{} exists: enter overwrites, or type another name",
                        taken.join(", ")
                    );
                    s.overwrite = true;
                    return Some(s);
                }
                let done = match s.what {
                    Export::Map { .. } => self.export_map(&stem),
                    Export::Grid => self.export_grid(&stem),
                    Export::Heatmap => self.export_plot(&stem),
                };
                if let Err(e) = done {
                    self.fail(format!("could not save {stem}: {e:#}"));
                }
                None
            }
            _ => {
                if s.name.key(key) {
                    s.overwrite = false;
                }
                Some(s)
            }
        }
    }

    pub(super) fn save_lines(&self, s: &SaveAs) -> Vec<Line<'static>> {
        let dim = style::dim();
        let width = usize::from(self.side.width).saturating_sub(3).max(10);
        let exts: Vec<String> = s.what.exts().iter().map(|e| format!(".{e}")).collect();
        let mut out = vec![
            Line::styled(" save as", style::bold()),
            Line::styled(format!(" writes {}", exts.join(" ")), dim),
        ];
        for piece in s.name.lines(width, true, style::bold()) {
            out.push(Line::from(vec![Span::raw(" "), piece]));
        }
        out.push(Line::raw(""));
        out.push(Line::styled(" type to replace  ← → keep and edit", dim));
        out.push(Line::styled(" enter save  esc cancel", dim));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_typed_extension_is_dropped_and_home_expanded() {
        let map = Export::Map { structure: false };
        assert_eq!(map.stem_of(" fig.pdf "), "fig");
        assert_eq!(map.stem_of("fig.v2"), "fig.v2");
        assert_eq!(Export::Heatmap.stem_of("t.tsv"), "t");
        let home = std::env::var("HOME").unwrap();
        assert_eq!(map.stem_of("~/a/fig.png"), format!("{home}/a/fig"));
    }

    #[test]
    fn the_structure_files_count_only_where_the_plot_shows() {
        let with = Export::Map { structure: true }.files("f");
        assert!(with.contains(&"f-structure.png".to_string()));
        let without = Export::Map { structure: false }.files("f");
        assert_eq!(without, ["f.png", "f.pdf", "f.txt"]);
    }
}
