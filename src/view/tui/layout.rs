//! `u` and `U`: the cells laid out by a table of theirs in place of the
//! tissue ([`crate::view::layout`]). `u` steps through the layouts and
//! back to the tissue, `U` through the tables. A layout is made on a
//! thread of its own, once for each table and layout; the view then draws
//! on it as it draws on the tissue.

use super::App;
use crate::view::layout::{self, Method};
use crate::view::Base;
use legume_tui::work::Pending;

/// A table (its place in [`layout::sources`]) and a layout of it.
pub(super) type Key = (usize, Method);

/// The cells' names and their places in a layout.
type Placed = (Vec<Box<str>>, crate::util::common::Mat);

/// A layout being made, and which.
pub(super) struct Laying {
    key: Key,
    pending: Pending<Placed>,
}

impl<'a> App<'a> {
    /// `u`: the next layout of the table shown, after the last the tissue.
    pub(super) fn step_layout(&mut self) {
        if self.still_laying() {
            return;
        }
        let next = match self.laid_out {
            None => Some(Method::ALL[0]),
            Some((_, m)) => {
                let at = Method::ALL.iter().position(|&x| x == m).unwrap_or(0);
                Method::ALL.get(at + 1).copied()
            }
        };
        match next {
            Some(m) => self.show_layout((self.layout_source, m)),
            None => self.show_tissue(),
        }
    }

    /// `U`: the next table to lay the cells out by, in the layout shown.
    pub(super) fn step_layout_source(&mut self) {
        if self.still_laying() {
            return;
        }
        let n = layout::sources(&self.tissue.run).len();
        if n == 0 {
            self.status = "no table of the cells to lay them out by".into();
            return;
        }
        self.layout_source = (self.layout_source + 1) % n;
        match self.laid_out {
            Some((_, m)) => self.show_layout((self.layout_source, m)),
            None => {
                let name = &layout::sources(&self.tissue.run)[self.layout_source].name;
                self.status = format!("u lays the cells out by {name}");
            }
        }
    }

    /// Whether a layout is being made: one at a time, as it is heavy work.
    fn still_laying(&mut self) -> bool {
        let Some(laying) = &self.laying else {
            return false;
        };
        let source = &layout::sources(&self.tissue.run)[laying.key.0].name;
        self.status = format!("still laying out {} of {source}", laying.key.1.name());
        true
    }

    /// The layout `key`, made first if need be.
    fn show_layout(&mut self, key: Key) {
        if let Some(&base) = self.layouts.get(&key) {
            self.switch_base(base, Some(key));
            return;
        }
        let Some(source) = layout::sources(&self.tissue.run).into_iter().nth(key.0) else {
            self.status = "no table of the cells to lay them out by".into();
            return;
        };
        let coord_names = self.tissue.geom.coord_names.clone();
        let what = format!("{} of {}", key.1.name(), source.name);
        self.status = format!("laying the cells out: {what} ...");
        let seed = crate::util::common::DEFAULT_SEED;
        self.laying = Some(Laying {
            key,
            pending: Pending::spawn(move || {
                let (names, table) =
                    layout::read(&source, &coord_names).map_err(|e| format!("{e:#}"))?;
                let xy = layout::lay_out(&table, key.1, seed).map_err(|e| format!("{e:#}"))?;
                Ok((names, xy))
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
        match done {
            Ok((names, xy)) => {
                let source = &layout::sources(&self.tissue.run)[key.0].name;
                let what = format!("{} of {source}", key.1.name());
                let geom = self
                    .tissue
                    .geom
                    .laid_out(&names, &xy, &what, [key.1.name(), ""]);
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
            Err(e) => self.status = format!("no layout: {e}"),
        }
    }

    /// Draw on `base` from here on, in the theme shown: the levels it had
    /// loaded back (the others kept for when this base is shown again), the
    /// map fitted to it. What was drawn on the other base's cells goes: the
    /// batch grid, the cell picked, the lra pairs' maps and a lupin round
    /// (`I` and `,`/`.` open them again here).
    fn switch_base(&mut self, base: &'a Base, key: Option<Key>) {
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
        self.tile = None;
        self.picked = None;
        self.leave_grid();
        self.lra = None;
        self.lra_on = false;
        self.round = None;
        match self.go_level(self.cur, None) {
            Ok(()) => {
                self.fit(self.base.geom.bounds());
                self.status = match key {
                    Some((s, m)) => {
                        let source = &layout::sources(&self.tissue.run)[s].name;
                        format!("{}: {source}", m.name())
                    }
                    None => "the tissue".into(),
                };
            }
            Err(e) => self.status = format!("{e:#}"),
        }
        self.need_panel = true;
    }
}
