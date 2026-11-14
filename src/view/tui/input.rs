//! Keys and the mouse: what a press or a click on the map means.

use super::super::data::NO_CLUSTER;
use super::super::render::Layer;
use super::{annotate, plots, App, Pick, POINT_STEP, ZOOM};
use ratatui::crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::Position;
use ratatui::DefaultTerminal;

impl App<'_> {
    pub(super) fn handle(
        &mut self,
        ev: Event,
        terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
        self.need_panel = true;
        // What the person changed of the view, not what they pressed,
        // pauses following an agent; F is following itself.
        let resumes = matches!(&ev, Event::Key(k) if k.code == KeyCode::Char('F'));
        let before = (self.follower.is_some() && !resumes).then(|| self.view_key());
        match ev {
            Event::Key(key) if key.kind != KeyEventKind::Release => self.key(key, Some(terminal)),
            Event::Mouse(m) => self.mouse(m),
            Event::Resize(..) => {
                terminal.autoresize()?;
                self.need_map = true;
            }
            _ => {}
        }
        if before.is_some_and(|b| b != self.view_key()) {
            self.pause_following();
        }
        Ok(())
    }

    /// What an agent's view sets, as drawn here now: a change of it is a
    /// move of the person's own.
    fn view_key(&self) -> impl PartialEq {
        let round = self.round.as_ref().map(|r| r.path.clone());
        let gene = self.gene.as_ref().map(|g| (g.feature.clone(), g.clip));
        let cam = (self.cam.cx, self.cam.cy, self.cam.scale);
        let groups = (self.cur, self.show, round, self.focus.clone());
        (
            groups,
            self.layer,
            gene,
            self.source,
            self.edges,
            self.point,
            cam,
        )
    }

    /// A key's action; one that fails shows why in the popup, and the
    /// viewer goes on.
    pub(super) fn key(&mut self, key: KeyEvent, terminal: Option<&mut DefaultTerminal>) {
        if self.error.take().is_some() {
            // Any key closes the popup, and does nothing else.
            self.need_map = true;
            self.need_panel = true;
            return;
        }
        if let Err(e) = self.act(key, terminal) {
            self.fail(format!("{e:#}"));
        }
    }

    fn act(
        &mut self,
        key: KeyEvent,
        mut terminal: Option<&mut DefaultTerminal>,
    ) -> anyhow::Result<()> {
        let big = key.modifiers.contains(KeyModifiers::SHIFT);
        let step = if big { 0.5 } else { 0.125 };
        self.status.clear();
        if self.modal.is_some() {
            return self.modal_key(key);
        }
        // Outside a dialog, shift-enter is H: the next chart.
        let key = if crate::tui::shift_enter(&key) {
            KeyEvent::new(KeyCode::Char('H'), KeyModifiers::NONE)
        } else {
            key
        };
        if self.view == plots::View::Heatmap && !plots::chart_key(key) {
            self.status = "on a chart: H next chart or back to the map · esc the map".into();
            return Ok(());
        }
        if self.lra_on && self.lra_key(key, terminal.as_deref_mut())? {
            return Ok(());
        }
        if self.relabel.is_some() && self.relabel_key(key)? {
            return Ok(());
        }
        if self.grid_shows() && self.grid_key(key) {
            return Ok(());
        }
        match key.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Esc => self.back(),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => self.quit = true,
            KeyCode::Left => self.pan(-step, 0.),
            KeyCode::Right => self.pan(step, 0.),
            KeyCode::Up => self.pan(0., -step),
            KeyCode::Down => self.pan(0., step),
            KeyCode::Char('+' | '=') if self.view == plots::View::Heatmap => self.more_genes(true),
            KeyCode::Char('-' | '_') if self.view == plots::View::Heatmap => self.more_genes(false),
            KeyCode::Char('+' | '=') => self.zoom(1. / ZOOM, None),
            KeyCode::Char('-' | '_') => self.zoom(ZOOM, None),
            // Zoom keys that mean the same in every mode.
            KeyCode::Char('z') => self.zoom(1. / ZOOM, None),
            KeyCode::Char('Z') => self.zoom(ZOOM, None),
            KeyCode::Char('H') => self.cycle_chart(),
            KeyCode::Char('f') => self.toggle_saved(),
            KeyCode::Char('F') => self.resume_following(terminal)?,
            KeyCode::Char('0') => self.fit_all(),
            KeyCode::Char('r') | KeyCode::Home => self.reset(),
            KeyCode::Char('1') => self.set_layer(Layer::Argmax),
            KeyCode::Char('2') => self.set_layer(Layer::Soft),
            KeyCode::Char('3') => self.set_layer(Layer::Entropy),
            KeyCode::Char('4') => {
                self.community = self.shown.unwrap_or(self.community);
                self.set_layer(Layer::Community(self.community))
            }
            KeyCode::Char(']') => self.step_focus(1),
            KeyCode::Char('[') => self.step_focus(-1),
            KeyCode::Char('l') => self.step_level(1, terminal)?,
            KeyCode::Char('L') => self.step_level(-1, terminal)?,
            KeyCode::Char('e') => self.toggle_edges(terminal)?,
            KeyCode::Char('b') => self.next_tile(),
            KeyCode::Char('w') => self.toggle_grid(),
            KeyCode::Char('t') => self.toggle_theme(),
            KeyCode::Char('s') => self.ask_export(),
            KeyCode::Char('x') => self.clear_focus(),
            KeyCode::Char('g') => self.step_gene(1),
            KeyCode::Char('G') => self.step_gene(-1),
            KeyCode::Char('o') => self.toggle_source(),
            KeyCode::Char('p') => self.toggle_clip(),
            KeyCode::Char('?') => self.help = !self.help,
            KeyCode::Char('c') => self.step_show(1, terminal)?,
            KeyCode::Char('C') => self.step_show(-1, terminal)?,
            KeyCode::Char('A') => self.ask_markers(),
            KeyCode::Char('R') => self.toggle_relabel(terminal)?,
            KeyCode::Char('I') => self.toggle_lra(terminal)?,
            KeyCode::Char('<') => self.scale_points(1. / POINT_STEP),
            KeyCode::Char('>') => self.scale_points(POINT_STEP),
            KeyCode::Char('.') => self.next_round(1, terminal)?,
            KeyCode::Char(',') => self.next_round(-1, terminal)?,
            _ => {}
        }
        Ok(())
    }

    fn mouse(&mut self, m: MouseEvent) {
        let (col, row) = (m.column, m.row);
        if self.error.is_some() {
            if let MouseEventKind::Down(_) = m.kind {
                self.error = None;
                self.need_map = true;
                self.need_panel = true;
            }
            return;
        }
        // The view stays as it is while it is being saved.
        if matches!(self.modal, Some(annotate::Modal::Save(_))) {
            return;
        }
        if self.resize_panel(m) {
            return;
        }
        if self.grid_shows() {
            return self.grid_mouse(m);
        }
        // The structure plot: a click picks a community; nothing pans it.
        if self.bars.contains(Position::new(col, row)) {
            match m.kind {
                MouseEventKind::Down(MouseButton::Left) if !self.curve_shows() => {
                    self.bar_click(col, row)
                }
                MouseEventKind::Down(_) | MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                }
                _ => {}
            }
            if !matches!(m.kind, MouseEventKind::Drag(_) | MouseEventKind::Up(_)) {
                return;
            }
        }
        if self.view != plots::View::Map
            && (self.in_map(col, row) || self.below.contains(Position::new(col, row)))
        {
            self.cursor = None;
            if let MouseEventKind::Down(MouseButton::Left) = m.kind {
                self.plot_click(row);
            }
            return;
        }

        match m.kind {
            MouseEventKind::Moved => {
                self.cursor = self.in_map(col, row).then_some((col, row));
            }
            MouseEventKind::ScrollUp if self.in_map(col, row) => {
                self.zoom(1. / ZOOM, Some((col, row)))
            }
            MouseEventKind::ScrollDown if self.in_map(col, row) => {
                self.zoom(ZOOM, Some((col, row)))
            }
            MouseEventKind::Down(MouseButton::Left) if self.in_map(col, row) => {
                self.drag = Some((col, row));
                self.dragged = false;
            }
            MouseEventKind::Down(button) if self.side.contains(Position::new(col, row)) => {
                let add = button == MouseButton::Right || adds(m.modifiers);
                match self.clickable.iter().find(|(r, _)| *r == row) {
                    Some(&(_, Pick::Community(c))) if self.relabel.is_some() => self.visit_group(c),
                    Some(&(_, Pick::Community(c))) => self.select(c, add),
                    Some(&(_, Pick::Gene(i))) => self.pick_gene(i),
                    Some(&(_, Pick::Bar(c))) => self.focus_bar(c),
                    Some(&(_, Pick::Pair(i))) => self.pick_pair(i),
                    None => {}
                }
            }
            MouseEventKind::Down(MouseButton::Right) if self.in_map(col, row) => {
                self.click(col, row, true);
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((c0, r0)) = self.drag {
                    self.dragged |= (c0, r0) != (col, row);
                    let dx = (col as f32 - c0 as f32) * self.px_per_cell.0;
                    let dy = (row as f32 - r0 as f32) * self.px_per_cell.1;
                    self.pan_px(dx, dy);
                    self.drag = Some((col, row));
                }
                self.cursor = self.in_map(col, row).then_some((col, row));
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if self.drag.is_some() && !self.dragged {
                    self.click(col, row, adds(m.modifiers));
                }
                self.drag = None;
            }
            _ => {}
        }
    }

    /// Pick the cell under a click and select its community; a click on
    /// empty tissue clears the selection.
    fn click(&mut self, col: u16, row: u16, add: bool) {
        let at = self.world_at(col, row);
        // Within a cell spacing, or two screen pixels when zoomed far out.
        let radius = self
            .base
            .spacing
            .max(2. * self.px_per_cell.0 * self.cam.upp());
        let geom = &self.base.geom;
        match self.base.grid.nearest(&geom.x, &geom.y, at, radius) {
            Some(i) => {
                self.picked = Some(i);
                let c = self.level().comm.cluster[i];
                if c != NO_CLUSTER {
                    if self.relabel.is_some() {
                        self.visit_group(c as usize);
                    } else {
                        self.select(c as usize, add);
                    }
                }
            }
            None => {
                if !add {
                    self.clear_focus();
                }
                self.status = "no cell here".into();
            }
        }
    }

    /// `Esc`: one step back, never out of the viewer. A plot goes back to
    /// the map, then the mapped gene goes, then the selection.
    fn back(&mut self) {
        if self.bar_focus.take().is_some() {
            self.need_map = true;
        } else if self.view != plots::View::Map || self.structure {
            self.view = plots::View::Map;
            self.structure = false;
            self.need_map = true;
        } else if self.grid_shows() {
            self.toggle_grid();
        } else if self.gene.is_some() {
            self.gene = None;
            self.need_map = true;
        } else if self.focus().is_some() || self.picked.is_some() {
            self.clear_focus();
        } else {
            self.status = "q quits".into();
        }
    }
}

/// Right-click or a held Ctrl/Alt adds to the selection instead of
/// replacing it. (Shift-click is kept by terminals for text selection.)
fn adds(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
}
