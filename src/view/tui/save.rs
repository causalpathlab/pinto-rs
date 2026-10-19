//! `s`: name the files an export writes. The next free `pinto-*-NNN`
//! comes up selected, so Enter takes it and typing replaces it.

use super::{plots, App};
use crate::tui::style;
use legume_tui::save::{self, Field, Numbering};
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::text::{Line, Span};
use std::path::PathBuf;

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

    /// What each file an export adds to its stem: the map's structure
    /// plot goes beside it as `{stem}-structure`.
    fn suffixes(self) -> &'static [&'static str] {
        match self {
            Export::Map { structure: true } => {
                &[".png", ".pdf", ".txt", "-structure.png", "-structure.txt"]
            }
            Export::Map { structure: false } => &[".png", ".pdf", ".txt"],
            Export::Grid => &[".png", ".txt"],
            Export::Heatmap => &[".tsv", ".txt"],
        }
    }

    /// The extensions written under the stem itself.
    fn exts(self) -> Vec<&'static str> {
        self.suffixes()
            .iter()
            .filter_map(|s| s.strip_prefix('.'))
            .collect()
    }

    /// The first `{stem}-NNN` none of whose files exist yet.
    fn free_stem(self, stem: &str) -> String {
        save::free_stem(stem, self.suffixes(), Numbering::Counter { digits: 3 })
    }

    /// The files of an export as `stem` that exist already.
    fn taken(self, stem: &str) -> Vec<PathBuf> {
        save::taken(stem, self.suffixes())
    }

    /// `name` without an extension this export writes, `~/` expanded.
    fn stem_of(self, name: &str) -> String {
        save::stem_of(name, &self.exts())
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
            name: Field::new(what.free_stem(what.prefix())),
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
                if let Some(problem) = save::name_problem(&stem) {
                    self.status = problem;
                    return Some(s);
                }
                let taken = s.what.taken(&stem);
                if !taken.is_empty() && !s.overwrite {
                    let taken: Vec<String> =
                        taken.iter().map(|p| p.display().to_string()).collect();
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
        for piece in s.name.lines(width, true, style::bold(), style::selected()) {
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
    fn the_structure_files_count_only_where_the_plot_shows() {
        let with = Export::Map { structure: true }.suffixes();
        assert!(with.contains(&"-structure.png"));
        let without = Export::Map { structure: false }.suffixes();
        assert_eq!(without, [".png", ".pdf", ".txt"]);
    }

    #[test]
    fn the_name_offered_is_free_of_every_file_an_export_writes() {
        let dir = tempfile::tempdir().unwrap();
        let stem = dir.path().join("pinto-view").to_string_lossy().into_owned();
        let map = Export::Map { structure: true };
        std::fs::write(format!("{stem}-001-structure.png"), "").unwrap();
        assert_eq!(map.free_stem(&stem), format!("{stem}-002"));
        assert_eq!(map.taken(&format!("{stem}-001")).len(), 1);
    }
}
