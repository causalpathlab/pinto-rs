//! The text editor and the file browser, and what they hand back.

use super::super::script::under;
use super::data::{Pair, Pick, Side};
use super::{data, free_out, App, Editor, Target, Task};
use crate::tui::browse::{Browser, Outcome};
use legume_tui::save::Field as TextField;
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use std::path::{Path, PathBuf};

impl App {
    /// Edit `text` for `target`, all of it selected so typing replaces
    /// it; the filter instead goes on narrowing from what it has.
    pub(super) fn edit(&mut self, target: Target, text: String) {
        let text = if target == Target::Filter {
            TextField::unselected(text)
        } else {
            TextField::new(text)
        };
        self.editor = Some(Editor { target, text });
    }

    pub(super) fn editor_key(&mut self, k: KeyEvent) {
        let Some(e) = self.editor.as_mut() else {
            return;
        };
        match k.code {
            KeyCode::Esc => {
                if e.target == Target::Filter {
                    self.filter.clear();
                    self.field_row = 0;
                }
                self.editor = None;
            }
            KeyCode::Enter => {
                let Editor { target, text } = self.editor.take().unwrap();
                let text = text.text;
                match target {
                    Target::Header => {
                        self.header = text.trim().to_string();
                        self.refresh_outs();
                    }
                    Target::Out(i) => {
                        let t = text.trim();
                        let r = &mut self.rows[i];
                        if t.is_empty() {
                            // Cleared: back under the header.
                            r.typed = false;
                            r.out = free_out(&self.here, &under(&self.header, &r.stem()));
                        } else {
                            // A folder: the method's name in it.
                            r.typed = true;
                            r.out = if t.ends_with('/') {
                                under(t, &r.stem())
                            } else {
                                t.to_string()
                            };
                        }
                    }
                    Target::Field(m, f) => {
                        let field = &mut self.rows[m].form.fields[f];
                        if text.trim().is_empty() && !field.default.is_empty() {
                            // Leaving it out means clap's default, so say that.
                            field.reset();
                            self.message = Some(format!(
                                "--{} cannot be left unset: back to its default {}",
                                field.long, field.default
                            ));
                        } else {
                            field.value = text;
                        }
                    }
                    Target::Filter => {}
                    Target::Name(i) => {
                        let p = &mut self.pairs[i];
                        p.clear_batch();
                        let t = text.trim();
                        if !t.is_empty() {
                            p.name = Some(t.to_string());
                        }
                    }
                    Target::Rename(i, label) => {
                        let p = &mut self.pairs[i];
                        let t = text.trim();
                        if t.is_empty() || t == label {
                            p.renames.remove(&label);
                        } else {
                            p.renames.insert(label, t.to_string());
                        }
                    }
                }
            }
            _ => {
                // The filter narrows the flags as it is typed.
                if e.text.key(k) && e.target == Target::Filter {
                    self.filter.clone_from(&e.text.text);
                    self.field_row = 0;
                }
            }
        }
    }

    // ───────────── the file browser ─────────────

    pub(super) fn browse(&mut self, want: Pick) {
        self.browser = Some(Browser::open(self.browse_dir.clone(), want, None));
    }

    pub(super) fn browser_key(&mut self, k: KeyEvent) {
        let Some(b) = self.browser.as_mut() else {
            return;
        };
        // Chosen files, or none when cancelled; either way it closes.
        let chosen = match b.key(k) {
            Outcome::Ignored | Outcome::Moved => return,
            Outcome::Cancelled => None,
            Outcome::Chosen(c) => Some(c.files()),
        };
        let Some(b) = self.browser.take() else { return };
        self.browse_dir = b.dir;
        let Some(paths) = chosen else { return };
        match b.want.side() {
            None => self.take_data(paths),
            Some(side) => self.take_sides(&paths, side),
        }
    }

    pub(super) fn take_data(&mut self, paths: Vec<PathBuf>) {
        let first_new = self.pairs.len();
        for p in paths {
            if self.pairs.iter().any(|q| q.data == p) {
                continue;
            }
            // Opening a large backend takes a while: not here.
            self.read(Task::Data(p.clone()));
            self.pairs.push(Pair::pending(p));
        }
        let added = self.pairs.len() - first_new;
        // Each folder's coordinate and label files go to its data by name,
        // the data taken before counted too, so a file named for one of
        // them goes to no new one.
        let folder = |p: &Pair| p.data.parent().unwrap_or(Path::new(".")).to_path_buf();
        let dirs: std::collections::BTreeSet<PathBuf> =
            self.pairs[first_new..].iter().map(folder).collect();
        for dir in dirs {
            let here: Vec<usize> = (0..self.pairs.len())
                .filter(|&i| folder(&self.pairs[i]) == dir)
                .collect();
            let data: Vec<PathBuf> = here.iter().map(|&i| self.pairs[i].data.clone()).collect();
            let data: Vec<&Path> = data.iter().map(PathBuf::as_path).collect();
            let near = data::near(&dir);
            let alone = near.data <= 1;
            for (side, files) in [(Side::Coord, &near.coords), (Side::Batch, &near.batches)] {
                for (&i, file) in here.iter().zip(data::beside(&data, files, alone)) {
                    if i >= first_new {
                        self.pairs[i].set(side, file);
                    }
                }
            }
        }
        self.pair_row = self.pairs.len().saturating_sub(1);
        let with = |side| self.pairs.iter().filter(|p| p.side(side).is_some()).count();
        self.message = Some(format!(
            "{added} data file{} added; of {}, {} with coordinates, {} with batch labels",
            if added == 1 { "" } else { "s" },
            self.pairs.len(),
            with(Side::Coord),
            with(Side::Batch),
        ));
    }

    /// One coordinate or batch file goes to the data row under the cursor;
    /// several are paired with all rows by name.
    pub(super) fn take_sides(&mut self, paths: &[PathBuf], side: Side) {
        if let [one] = paths {
            if let Some(p) = self.pairs.get_mut(self.pair_row) {
                p.set(side, Some(one.clone()));
            }
            return;
        }
        for p in &mut self.pairs {
            p.set(side, None);
        }
        let n = self.pairs.len();
        let (what, key) = match side {
            Side::Coord => ("coordinate", 'c'),
            Side::Batch => ("batch", 'b'),
        };
        self.message = Some(match data::assign(&mut self.pairs, paths, side) {
            data::Paired::ByName(_) => format!("{n} {what} files paired by name"),
            data::Paired::InOrder => format!(
                "no names match: {n} {what} files paired in the order listed; check them"
            ),
            data::Paired::Partly(k) => format!(
                "{k} of {n} data files matched a {what} file by name; {key} sets the rest one by one"
            ),
        });
    }
}
