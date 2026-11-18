//! `u` and `U`: the cells laid out by a table of theirs in place of the
//! tissue ([`crate::view::layout`]). `u` steps through the layouts and
//! back to the tissue, `U` through the tables. A layout is made on a
//! thread of its own, once for each table and layout, each table coarsened
//! once for all its layouts; the view then draws on it as it draws on the
//! tissue.

use super::App;
use crate::view::layout::{Coarse, Method};
use crate::view::Base;
use legume_tui::work::Pending;
use std::sync::Arc;

/// A table (its place in [`crate::view::layout::sources`]) and a layout of
/// it.
pub(super) type Key = (usize, Method);

/// The table coarsened, and its cells' places in a layout.
type Placed = (Arc<Coarse>, crate::util::common::Mat);

/// A layout being made, and which.
pub(super) struct Laying {
    key: Key,
    pending: Pending<Placed>,
}

impl<'a> App<'a> {
    /// `u`: the next layout of the table shown, after the last the tissue.
    pub(super) fn step_layout(&mut self) {
        if self.still_laying() || self.no_sources() {
            return;
        }
        let next = match self.asked {
            None => Some(Method::ALL[0]),
            Some(m) => m.next(),
        };
        match next {
            Some(m) => self.show_layout((self.layout_source, m)),
            None => self.show_tissue(),
        }
    }

    /// `U`: the next table to lay the cells out by, in the layout shown.
    pub(super) fn step_layout_source(&mut self) {
        if self.still_laying() || self.no_sources() {
            return;
        }
        self.layout_source = (self.layout_source + 1) % self.sources.len();
        match self.asked {
            Some(m) => self.show_layout((self.layout_source, m)),
            None => {
                let name = &self.sources[self.layout_source].name;
                self.status = format!("u lays the cells out by {name}");
            }
        }
    }

    /// Whether the run has no table of its cells, said so.
    fn no_sources(&mut self) -> bool {
        let none = self.sources.is_empty();
        if none {
            self.status = "no table of the cells to lay them out by".into();
        }
        none
    }

    /// Whether a layout is being made: one at a time, as it is heavy work.
    fn still_laying(&mut self) -> bool {
        let Some(key) = self.laying.as_ref().map(|l| l.key) else {
            return false;
        };
        self.status = format!("still laying out {}", self.label(key));
        true
    }

    /// `UMAP of propensity L2`.
    fn label(&self, (source, method): Key) -> String {
        format!("{} of {}", method.name(), self.sources[source].name)
    }

    /// The layout `key`, made first if need be.
    fn show_layout(&mut self, key: Key) {
        self.asked = Some(key.1);
        if let Some(&base) = self.layouts.get(&key) {
            self.switch_base(base, Some(key));
            return;
        }
        let source = self.sources[key.0].clone();
        let coarse = self.coarse.get(&key.0).cloned();
        let coord_names = self.tissue.geom.coord_names.clone();
        self.status = format!("laying the cells out: {} ...", self.label(key));
        let seed = crate::util::common::DEFAULT_SEED;
        self.laying = Some(Laying {
            key,
            pending: Pending::spawn(move || {
                let e = |e: anyhow::Error| format!("{e:#}");
                let coarse = match coarse {
                    Some(c) => c,
                    None => {
                        let (names, table) = source.read(&coord_names).map_err(e)?;
                        Arc::new(Coarse::new(names, &table, seed).map_err(e)?)
                    }
                };
                let xy = coarse.lay_out(key.1, seed).map_err(e)?;
                Ok((coarse, xy))
            }),
        });
        self.need_panel = true;
    }

    /// Back to the tissue.
    fn show_tissue(&mut self) {
        let tissue = self.tissue;
        self.switch_base(tissue, None);
    }

    /// Take a layout that is made, and show it.
    pub(super) fn poll_layout(&mut self) {
        let Some(laying) = self.laying.as_mut() else {
            return;
        };
        let Some(done) = laying.pending.try_take("the layout stopped") else {
            return;
        };
        let key = laying.key;
        self.laying = None;
        let what = self.label(key);
        match done {
            Ok((coarse, xy)) => {
                let geom = self.tissue.geom.laid_out(&coarse.names, &xy, &what);
                self.coarse.insert(key.0, coarse);
                if geom.n() == 0 {
                    self.status = format!("{what}: none of the cells is in the table");
                    return;
                }
                // Kept until the view closes, so switching back is at once:
                // one per table and layout asked for.
                let base: &'a Base = Box::leak(Box::new(self.tissue.laid_out(geom)));
                self.layouts.insert(key, base);
                self.switch_base(base, Some(key));
            }
            Err(e) => self.status = format!("no layout of {what}: {e}"),
        }
    }

    /// Draw on `base` from here on, in the theme shown: the levels it had
    /// loaded back (the others kept for when this base is shown again), the
    /// map fitted to it. What was drawn on the other base's cells goes: a
    /// relabelling (its draft saved), the batch grid, the cell picked, the
    /// plots, the lra pairs and a lupin round (`I` and `,`/`.` open them
    /// again here).
    fn switch_base(&mut self, base: &'a Base, key: Option<Key>) {
        self.leave_relabel();
        let theme = self.base.theme();
        base.set_theme(theme);
        let levels = std::mem::take(&mut self.levels);
        self.stashed.insert(self.laid_out, (theme, levels));
        self.levels = match self.stashed.remove(&key) {
            Some((was, levels)) if was == theme => levels,
            _ => base.run.levels.iter().map(|_| None).collect(),
        };
        self.base = base;
        self.laid_out = key;
        self.asked = key.map(|(_, m)| m);
        self.picked = None;
        self.leave_grid();
        self.plots.forget();
        self.forget_lra();
        self.round = None;
        match self.go_level(self.cur, None) {
            Ok(()) => {
                self.fit_all();
                self.status = match key {
                    Some((s, m)) => format!("{}: {}", m.name(), self.sources[s].name),
                    None => "the tissue".into(),
                };
            }
            Err(e) => self.status = format!("{e:#}"),
        }
        self.need_panel = true;
    }
}
