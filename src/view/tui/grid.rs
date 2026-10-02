//! Every batch at once (`w`): one tile per batch, each fitted to its own
//! frame and numbered, so batches of different sizes read side by side.
//! With the structure plot on (`H`), each tile carries its own batch's
//! bars under its map, on one stack order, so the batches compare. A click
//! on a map opens that batch; a click on its bars shows where that
//! community lies, as under the single map. A drag moves a batch to
//! another place in the grid.
//!
//! Once a batch is open (a click, or `b`), the grid keeps the map's zoom
//! and pan: every tile shows its own batch as the map shows that one,
//! relative to each batch's fitted frame. In the grid, zoom and point size
//! go to the batch pointed at, which then keeps its own; with Alt, to
//! every batch. `0` fits them all again.

use super::super::render::{self, Frame, Style, Viewport};
use super::super::scalebar;
use super::super::structure::{Drawn, Structure};
use super::super::FIT_MARGIN;
use super::{plots, step_point, App, WorldRect};
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use rayon::prelude::*;
use std::collections::HashMap;

/// Share of a tile its bars take, with the structure plot on.
const BARS_SHARE: f32 = 0.25;

/// A pixel rectangle in the canvas frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Px {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Px {
    fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }

    fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }
}

/// One place in the grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slot {
    /// Its whole tile, its map, and its bars (zero-sized when off).
    pub tile: Px,
    pub map: Px,
    pub bars: Px,
}

/// A batch's own view and point size, once zoomed or resized on its own;
/// `None` follows the map's.
#[derive(Clone, Copy, Debug, Default)]
struct Own {
    cam: Option<Camera>,
    point: Option<f32>,
}

/// What a batch's bars were drawn for: the level, size, and communities
/// in colour.
type BarsKey = (usize, (usize, usize), Option<Vec<bool>>);

#[derive(Default)]
pub struct Grid {
    pub on: bool,
    /// The place under the pointer or the arrow keys.
    pub hover: Option<usize>,
    /// Batch at each place in the grid, as the user arranged them.
    order: Vec<usize>,
    /// The place a drag started from, and whether it has left it.
    drag: Option<(usize, bool)>,
    /// Batches zoomed or resized on their own, by batch.
    own: HashMap<usize, Own>,
    /// Where each place was last drawn on screen.
    slots: Vec<Slot>,
    /// Each batch's bars from one level, by batch, as a plot of its own.
    structure: Option<(usize, Vec<Option<Structure>>)>,
    /// Each batch's bars as last drawn, by batch, with what for.
    bars: HashMap<usize, (BarsKey, Drawn)>,
}

impl Grid {
    /// Every batch back to the map's view and point size.
    pub fn forget_own(&mut self) {
        self.own.clear();
    }
}

/// Columns for `n` tiles of world aspect `aspect` (width / height) in a
/// `w × h` area, each map `map_share` of its tile's height: the shape that
/// draws them largest.
pub fn columns(n: usize, (w, h): (f32, f32), aspect: f32, map_share: f32) -> usize {
    let scale = |cols: usize| {
        let rows = n.div_ceil(cols);
        let (tw, th) = (w / cols as f32, h / rows as f32 * map_share);
        (tw / aspect.max(f32::MIN_POSITIVE)).min(th)
    };
    (1..=n.max(1))
        .max_by(|&a, &b| scale(a).total_cmp(&scale(b)).then(b.cmp(&a)))
        .unwrap_or(1)
}

/// Slots for `n` places in a `w × h` frame, `cols` across.
pub fn slots(n: usize, (w, h): (usize, usize), cols: usize, bars: bool) -> Vec<Slot> {
    let cols = cols.clamp(1, n.max(1));
    let rows = n.div_ceil(cols).max(1);
    let gap = (w.min(h) / 120).max(4);
    (0..n)
        .map(|p| {
            let (c, r) = (p % cols, p / cols);
            let (x0, x1) = (c * w / cols, (c + 1) * w / cols);
            let (y0, y1) = (r * h / rows, (r + 1) * h / rows);
            let tile = Px {
                x: x0,
                y: y0,
                w: x1 - x0,
                h: y1 - y0,
            };
            let inner = Px {
                x: x0 + gap / 2,
                y: y0 + gap / 2,
                w: tile.w.saturating_sub(gap),
                h: tile.h.saturating_sub(gap),
            };
            let bars_h = if bars {
                ((inner.h as f32 * BARS_SHARE) as usize)
                    .max(8)
                    .min(inner.h / 2)
            } else {
                0
            };
            let between = if bars_h > 0 { gap / 2 } else { 0 };
            let map = Px {
                h: inner.h.saturating_sub(bars_h + between),
                ..inner
            };
            let bars = Px {
                y: map.y + map.h + between,
                h: bars_h,
                ..inner
            };
            Slot { tile, map, bars }
        })
        .collect()
}

/// Move the batch at place `from` to place `to`, the others shifting over.
pub fn move_to(order: &mut Vec<usize>, from: usize, to: usize) {
    if from < order.len() && to < order.len() && from != to {
        let b = order.remove(from);
        order.insert(to, b);
    }
}

/// Where the map looks, relative to the frame a batch is fitted in: the
/// centre as a share of the frame's width and height, and the zoom as a
/// multiple of the fitted scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub at: (f32, f32),
    pub zoom: f32,
}

impl Camera {
    /// The camera of `vp` over `frame` (a batch's padded bounds).
    pub fn of(vp: &Viewport, frame: WorldRect) -> Self {
        let fit = Viewport::fit(frame, vp.w, vp.h);
        let (cx, cy) = vp.centre();
        Camera {
            at: (
                (cx - frame.x0) / frame.width().max(f32::MIN_POSITIVE),
                (cy - frame.y0) / frame.height().max(f32::MIN_POSITIVE),
            ),
            zoom: vp.upp / fit.upp,
        }
    }

    /// The same view of another batch's `frame`, at `w × h` pixels.
    pub fn view(&self, frame: WorldRect, w: usize, h: usize) -> Viewport {
        let centre = (
            frame.x0 + self.at.0 * frame.width(),
            frame.y0 + self.at.1 * frame.height(),
        );
        Viewport::at(centre, Viewport::fit(frame, w, h).upp * self.zoom, w, h)
    }
}

/// Paint `frame` outside world rectangle `keep` with its background.
pub fn clip(frame: &mut Frame, vp: &Viewport, keep: WorldRect) {
    let (w, h) = (frame.w, frame.h);
    let (ax, ay) = vp.to_px(keep.x0, keep.y0);
    let (bx, by) = vp.to_px(keep.x1, keep.y1);
    let px = |v: f32, max: usize| (v.round().max(0.) as usize).min(max);
    let (x0, y0) = (px(ax, w), px(ay, h));
    let (x1, y1) = (px(bx, w).max(x0), px(by, h).max(y0));
    if (x0, y0, x1, y1) == (0, 0, w, h) {
        return;
    }
    let [r, g, b] = frame.background;
    for (y, row) in frame.rgba.chunks_exact_mut(4 * w.max(1)).enumerate() {
        let spans = if y < y0 || y >= y1 {
            [(0, w), (0, 0)]
        } else {
            [(0, x0), (x1, w)]
        };
        for (a, z) in spans {
            for p in row[4 * a..4 * z].as_chunks_mut::<4>().0 {
                p.copy_from_slice(&[r, g, b, 255]);
            }
        }
    }
}

/// Copy `src` into `dst` with its top left at `(x, y)`.
pub fn blit(dst: &mut Frame, src: &Frame, (x, y): (usize, usize)) {
    let w = src.w.min(dst.w.saturating_sub(x));
    for row in 0..src.h.min(dst.h.saturating_sub(y)) {
        let (s, d) = (src.offset(0, row), dst.offset(x, y + row));
        dst.rgba[d..d + 4 * w].copy_from_slice(&src.rgba[s..s + 4 * w]);
    }
}

impl App<'_> {
    /// Whether the grid is what the main area shows.
    pub(super) fn grid_shows(&self) -> bool {
        self.grid.on && self.view == plots::View::Map
    }

    pub(super) fn leave_grid(&mut self) {
        self.grid.on = false;
    }

    /// `w`: every batch in a grid, or back to the one map.
    pub(super) fn toggle_grid(&mut self) {
        let n = self.base.geom.tiles.len();
        if self.grid.on {
            self.leave_grid();
            self.status = "map".into();
        } else if n < 2 {
            self.status = "one batch: nothing to grid".into();
            return;
        } else {
            if self.grid.order.len() != n {
                self.grid.order = (0..n).collect();
            }
            self.grid.on = true;
            self.view = plots::View::Map;
            self.status = "every batch: click opens, drag moves; zoom, < >: one (alt: all)".into();
        }
        self.need_map = true;
    }

    /// Leave the grid for batch `b`: as its tile shows it, or fitted to the
    /// map when the grid's batches are fitted.
    pub(super) fn open_batch(&mut self, b: usize) {
        let Some(tile) = self.base.geom.tiles.get(b) else {
            return;
        };
        let (bounds, name) = (tile.bounds, tile.name.to_string());
        let camera = self.own_camera(b).or(self.camera());
        self.leave_grid();
        self.tile = Some(b);
        self.status = format!("batch {name}  w: every batch");
        match camera {
            Some(c) => {
                let (w, h) = self.image_size();
                self.set_view(&c.view(bounds.pad(FIT_MARGIN), w, h));
            }
            None => self.fit(bounds),
        }
    }

    /// The batch at place `p`.
    fn batch_at(&self, p: usize) -> Option<usize> {
        self.grid.order.get(p).copied()
    }

    /// The camera the grid shares with the map: none until a batch is
    /// open, then the map's view of the batch under its centre.
    fn camera(&self) -> Option<Camera> {
        let open = self.tile?;
        let tiles = &self.base.geom.tiles;
        let (cx, cy) = self.center;
        let b = tiles
            .iter()
            .position(|t| {
                let r = t.bounds.pad(FIT_MARGIN);
                (r.x0..=r.x1).contains(&cx) && (r.y0..=r.y1).contains(&cy)
            })
            .unwrap_or(open);
        Some(Camera::of(
            &self.viewport(),
            tiles[b].bounds.pad(FIT_MARGIN),
        ))
    }

    /// Batch `b`'s own view, when it has one.
    fn own_camera(&self, b: usize) -> Option<Camera> {
        self.grid.own.get(&b).and_then(|o| o.cam)
    }

    /// Batch `b`'s point size: its own, or the map's.
    fn own_point(&self, b: usize) -> f32 {
        self.grid
            .own
            .get(&b)
            .and_then(|o| o.point)
            .unwrap_or(self.point)
    }

    /// Batch `b` in a `w × h` map: its own view, the map's, or fitted.
    fn tile_view(&self, b: usize, (w, h): (usize, usize), shared: Option<Camera>) -> Viewport {
        let frame = self.base.geom.tiles[b].bounds.pad(FIT_MARGIN);
        match self.own_camera(b).or(shared) {
            Some(c) => c.view(frame, w, h),
            None => Viewport::fit(frame, w, h),
        }
    }

    /// Zoom the batch at place `p` alone by `f` (> 1 zooms out), keeping
    /// pixel `anchor` of its map still; its centre without one.
    fn tile_zoom(&mut self, p: usize, f: f32, anchor: Option<(f32, f32)>) {
        let (Some(b), Some(slot)) = (self.batch_at(p), self.grid.slots.get(p).copied()) else {
            return;
        };
        let size = (slot.map.w.max(1), slot.map.h.max(1));
        let frame = self.base.geom.tiles[b].bounds.pad(FIT_MARGIN);
        let vp =
            self.tile_view(b, size, self.camera())
                .zoomed(f, anchor, self.zoom_limits(frame, size));
        self.grid.own.entry(b).or_default().cam = Some(Camera::of(&vp, frame));
        let name = &self.base.geom.tiles[b].name;
        self.status = format!("{name} zoomed on its own  alt: every batch  0 fits all");
        self.need_map = true;
    }

    /// Zoom every batch by `f`: each batch's own view, and the map's.
    fn zoom_all(&mut self, f: f32, p: Option<usize>) {
        let zoomed: Vec<usize> = (0..self.grid.order.len())
            .filter(|&q| {
                self.batch_at(q)
                    .is_some_and(|b| self.own_camera(b).is_some())
            })
            .collect();
        for q in zoomed {
            self.tile_zoom(q, f, None);
        }
        self.grid_zoom(f, p);
    }

    /// Zoom the map by `f` (> 1 zooms out), on the batch at place `p` when
    /// no batch is open yet: every batch without its own view follows.
    fn grid_zoom(&mut self, f: f32, p: Option<usize>) {
        if self.tile.is_none() {
            let Some(b) = self.batch_at(p.unwrap_or(0)) else {
                return;
            };
            self.tile = Some(b);
            self.fit(self.base.geom.tiles[b].bounds);
        }
        self.zoom(f, None);
        self.status = "zoom follows the map  0 fits every batch".into();
    }

    /// A zoom key: the batch at place `at`, or with `all` (or none
    /// chosen) every batch.
    fn grid_zoom_key(&mut self, f: f32, at: Option<usize>, all: bool) {
        match at.filter(|_| !all) {
            Some(p) => self.tile_zoom(p, f, None),
            None => self.zoom_all(f, at),
        }
    }

    /// `<`/`>` on the grid: the point size of the batch at place `p`, or
    /// with `all` (or none chosen) of every batch.
    fn grid_points(&mut self, by: f32, p: Option<usize>, all: bool) {
        match p.and_then(|p| self.batch_at(p)).filter(|_| !all) {
            Some(b) => {
                let next = step_point(self.own_point(b), by);
                self.grid.own.entry(b).or_default().point = Some(next);
                let name = &self.base.geom.tiles[b].name;
                self.status = format!("{name} points ×{next:.2}  alt: every batch");
                self.need_map = true;
            }
            None => {
                for x in self.grid.own.values_mut().filter_map(|o| o.point.as_mut()) {
                    *x = step_point(*x, by);
                }
                self.scale_points(by);
            }
        }
    }

    /// A key on the grid; `false` leaves it to the viewer's own keys.
    pub(super) fn grid_key(&mut self, key: KeyEvent) -> bool {
        let n = self.grid.order.len();
        let cols = self
            .grid
            .slots
            .iter()
            .take_while(|s| s.tile.y == 0)
            .count()
            .max(1) as isize;
        let at = self.grid.hover;
        let all = key.modifiers.contains(KeyModifiers::ALT);
        let by = match key.code {
            KeyCode::Left => Some(-1),
            KeyCode::Right => Some(1),
            KeyCode::Up => Some(-cols),
            KeyCode::Down => Some(cols),
            _ => None,
        };
        if let Some(by) = by {
            let to = at.map_or(0, |h| (h as isize + by).clamp(0, n as isize - 1) as usize);
            // Shift and an arrow carries the batch along.
            if let (Some(from), true) = (at, key.modifiers.contains(KeyModifiers::SHIFT)) {
                move_to(&mut self.grid.order, from, to);
                self.need_map = true;
            }
            self.grid.hover = Some(to);
        } else {
            let (zoom, step) = (super::ZOOM, super::POINT_STEP);
            match key.code {
                KeyCode::Esc | KeyCode::Char('w') => self.toggle_grid(),
                KeyCode::Enter => match at.and_then(|p| self.batch_at(p)) {
                    Some(b) => self.open_batch(b),
                    None => self.status = "point at a batch, or arrows to choose one".into(),
                },
                KeyCode::Char('+' | '=' | 'z') => self.grid_zoom_key(1. / zoom, at, all),
                KeyCode::Char('-' | '_' | 'Z') => self.grid_zoom_key(zoom, at, all),
                KeyCode::Char('<') => self.grid_points(1. / step, at, all),
                KeyCode::Char('>') => self.grid_points(step, at, all),
                KeyCode::Char('0') => self.fit_all(),
                _ => return false,
            }
        }
        let quiet = self.grid.on && self.status.is_empty();
        if let Some(b) = self
            .grid
            .hover
            .filter(|_| quiet)
            .and_then(|p| self.batch_at(p))
        {
            let name = &self.base.geom.tiles[b].name;
            self.status = format!("{name}: enter opens, shift+arrows move it");
        }
        true
    }

    /// The wheel over place `p`: that batch around the pointer at canvas
    /// pixel `(x, y)`, or with Alt every batch.
    fn grid_wheel(&mut self, f: f32, p: usize, (x, y): (usize, usize), all: bool) {
        if all {
            return self.zoom_all(f, Some(p));
        }
        let map = self.grid.slots[p].map;
        let anchor = map
            .contains(x, y)
            .then(|| ((x - map.x) as f32, (y - map.y) as f32));
        self.tile_zoom(p, f, anchor);
    }

    pub(super) fn grid_mouse(&mut self, m: MouseEvent) {
        let (col, row) = (m.column, m.row);
        let inside = self
            .canvas
            .contains(ratatui::layout::Position::new(col, row));
        let (x, y) = self.px_in(self.canvas, col, row);
        let (x, y) = (x as usize, y as usize);
        let k = self
            .grid
            .slots
            .iter()
            .position(|s| s.tile.contains(x, y))
            .filter(|_| inside);
        let all = m.modifiers.contains(KeyModifiers::ALT);
        match (m.kind, k) {
            (MouseEventKind::Moved, _) => self.grid.hover = k,
            (MouseEventKind::ScrollUp, Some(p)) => {
                self.grid_wheel(1. / super::ZOOM, p, (x, y), all)
            }
            (MouseEventKind::ScrollDown, Some(p)) => self.grid_wheel(super::ZOOM, p, (x, y), all),
            (MouseEventKind::Down(MouseButton::Left), Some(p)) => {
                let bars = self.grid.slots[p].bars;
                if !bars.contains(x, y) {
                    self.grid.drag = Some((p, false));
                    return;
                }
                let hit = self
                    .batch_at(p)
                    .and_then(|b| self.grid.bars.get(&b))
                    .and_then(|(_, d)| d.at(x - bars.x, y - bars.y));
                match hit {
                    Some(c) => self.focus_bar(c),
                    None => self.status = "between the bars".into(),
                }
            }
            (MouseEventKind::Drag(MouseButton::Left), _) => {
                if let Some((from, moved)) = self.grid.drag.as_mut() {
                    *moved |= k != Some(*from);
                    let from = *from;
                    self.grid.hover = k;
                    if let Some(b) = self.batch_at(from) {
                        let name = &self.base.geom.tiles[b].name;
                        self.status = format!("moving {name}: let go on a tile");
                    }
                }
            }
            (MouseEventKind::Up(MouseButton::Left), _) => match (self.grid.drag.take(), k) {
                (Some((from, false)), Some(to)) if from == to => {
                    if let Some(b) = self.batch_at(from) {
                        self.open_batch(b);
                    }
                }
                (Some((from, _)), Some(to)) if from != to => {
                    move_to(&mut self.grid.order, from, to);
                    self.grid.hover = Some(to);
                    self.status.clear();
                    self.need_map = true;
                }
                (Some(_), _) => self.status.clear(),
                _ => {}
            },
            _ => {}
        }
    }

    /// Bars of every batch, built once per level; `false` when the bars'
    /// level does not load.
    fn batch_structure(&mut self) -> bool {
        let Some(bars) = self.load_bars() else {
            return false;
        };
        if matches!(&self.grid.structure, Some((l, _)) if *l == bars) {
            return true;
        }
        let mut by_name: HashMap<String, Structure> =
            self.by_batch(bars).into_panels().into_iter().collect();
        let per_batch = self
            .base
            .geom
            .tiles
            .iter()
            .map(|t| by_name.remove(t.name.as_ref()))
            .collect();
        self.grid.structure = Some((bars, per_batch));
        self.grid.bars.clear();
        true
    }

    /// Draw the bars of the batch at each place in `slots`, unless already
    /// drawn for the same level, size and colours.
    fn draw_bars(&mut self, slots: &[Slot]) {
        let focus = self.bars_focus();
        let Some((l, per_batch)) = self.grid.structure.as_ref() else {
            return;
        };
        let Some(level) = self.levels[*l].as_ref() else {
            return;
        };
        for (s, &b) in slots.iter().zip(&self.grid.order) {
            let Some(structure) = per_batch.get(b).and_then(Option::as_ref) else {
                continue;
            };
            let key = (*l, s.bars.size(), focus.clone());
            if self.grid.bars.get(&b).is_some_and(|(k, _)| *k == key) {
                continue;
            }
            let drawn = structure.render(
                &level.comm,
                &level.palette,
                s.bars.size(),
                focus.as_deref(),
                self.base.theme,
            );
            self.grid.bars.insert(b, (key, drawn));
        }
    }

    /// Every batch in a `w × h` frame; `on_screen` keeps where each went,
    /// for clicks.
    pub(super) fn grid_frame(&mut self, (w, h): (usize, usize), on_screen: bool) -> Frame {
        let bars = self.structure && self.batch_structure();
        let tiles = &self.base.geom.tiles;
        let (slot_w, slot_h) = tiles.iter().fold((0f32, 0f32), |(w, h), t| {
            (w.max(t.bounds.width()), h.max(t.bounds.height()))
        });
        let share = if bars { 1. - BARS_SHARE } else { 1. };
        let aspect = slot_w / slot_h.max(f32::MIN_POSITIVE);
        let cols = columns(tiles.len(), (w as f32, h as f32), aspect, share);
        let slots = slots(tiles.len(), (w, h), cols, bars);
        if bars {
            self.draw_bars(&slots);
        }

        // Each batch's map, side by side on rayon's threads.
        let camera = self.camera();
        let jobs: Vec<(usize, Px, Viewport, f32)> = slots
            .iter()
            .zip(&self.grid.order)
            .filter(|(s, _)| s.map.w > 0 && s.map.h > 0)
            .map(|(s, &b)| {
                (
                    b,
                    s.map,
                    self.tile_view(b, s.map.size(), camera),
                    self.own_point(b),
                )
            })
            .collect();
        let (level, style) = self.drawn();
        let (base, gene) = (self.base, self.gene.as_ref());
        let maps: Vec<Frame> = jobs
            .par_iter()
            .map(|&(b, _, vp, point)| {
                let style = Style { point, ..style };
                let mut f = base.render(level, &style, gene, &vp);
                // Neighbouring batches sit a gutter away in the world.
                let keep = base.geom.tiles[b].bounds.grow(base.spacing * point.max(1.));
                clip(&mut f, &vp, keep);
                if let Some(units) = base.units {
                    scalebar::draw(&mut f, &vp, units, false);
                }
                let px = ((f.h.min(f.w) as f32 / 300.).round() as i64).clamp(1, 4);
                scalebar::label(&mut f, (3 * px, 3 * px), px, &(b + 1).to_string());
                f
            })
            .collect();

        let mut out = Frame::blank(w, h, self.base.theme.background());
        for (&(_, at, ..), f) in jobs.iter().zip(&maps) {
            blit(&mut out, f, (at.x, at.y));
        }
        for (s, b) in slots.iter().zip(&self.grid.order) {
            if let Some((_, d)) = self.grid.bars.get(b).filter(|_| s.bars.h > 0) {
                blit(&mut out, &d.frame, (s.bars.x, s.bars.y));
            }
        }
        if on_screen {
            if let Some(&(_, _, vp, point)) = jobs.first() {
                self.mode = render::mode(&self.base.scene(level), &vp, point);
            }
            self.grid.slots = slots;
        }
        out
    }

    /// The grid's panel row: how many batches, or the one pointed at with
    /// its own zoom and point size.
    pub(super) fn grid_line(&self) -> String {
        let tiles = &self.base.geom.tiles;
        let Some(b) = self.grid.hover.and_then(|p| self.batch_at(p)) else {
            return format!("{} batches  w: map", tiles.len());
        };
        let t = &tiles[b];
        let name: String = t.name.chars().take(14).collect();
        let mut line = format!("{} {name} {}", b + 1, super::thousands(t.n_cells));
        let own = self.grid.own.get(&b).copied().unwrap_or_default();
        if let Some(c) = own.cam {
            line.push_str(&format!(" z×{:.1}", 1. / c.zoom));
        }
        if let Some(x) = own.point {
            line.push_str(&format!(" p×{x:.1}"));
        }
        line
    }

    /// `s` on the grid: the grid as a PNG at the export scale, its batches
    /// listed beside it.
    pub(super) fn export_grid(&mut self, stem: &str) -> anyhow::Result<()> {
        use std::fmt::Write as _;
        let scale = self.args.export_scale.max(1);
        let w = (self.canvas.width as f32 * self.px_per_cell.0) as usize * scale;
        let h = (self.canvas.height as f32 * self.px_per_cell.1) as usize * scale;
        let frame = self.grid_frame((w.max(1), h.max(1)), false);
        let png = format!("{stem}.png");
        frame.write_png(std::path::Path::new(&png))?;
        let (level, style) = self.drawn();
        let what = format!("grid · {} · {}", level.comm.tag, style.layer);
        let mut notes = String::new();
        writeln!(notes, "# pinto view grid export").ok();
        writeln!(notes, "image    {png} ({}×{})", frame.w, frame.h).ok();
        writeln!(notes, "run      {}", self.base.run.source()).ok();
        writeln!(notes, "level    {}", self.base.run.levels[level.index].tag).ok();
        writeln!(notes, "layer    {}", style.layer).ok();
        if let Some(g) = &self.gene {
            writeln!(notes, "gene     {}, ramp 0..{}", g.title(), g.top_label()).ok();
        }
        if let (true, Some((l, _))) = (self.structure, &self.grid.structure) {
            let tag = &self.levels[*l].as_ref().expect("loaded").comm.tag;
            writeln!(notes, "bars     level {tag} propensities, under each batch").ok();
        }
        writeln!(
            notes,
            "\nbatches in grid order, row by row (number on the tile, name, cells)"
        )
        .ok();
        for &b in &self.grid.order {
            let t = &self.base.geom.tiles[b];
            writeln!(notes, "  {:>3}  {:<24} {:>9}", b + 1, t.name, t.n_cells).ok();
        }
        std::fs::write(format!("{stem}.txt"), notes)?;
        let listed = self.remember(&png, &what, &frame);
        self.status = format!("saved {stem}.png .txt{listed}");
        Ok(())
    }
}
