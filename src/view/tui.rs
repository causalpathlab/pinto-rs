//! Interactive terminal view: the map on the left, a side panel on the right.
//!
//! The map is drawn by [`render`] at the pixel size of its cells and shown
//! through the terminal's graphics protocol: our own kitty transport
//! (`kitty`), ratatui-image for sixel/iTerm2, or coloured block characters
//! (`cellart`, 2×2 pixels per cell) anywhere else.
//!
//! Input is drained before each redraw, so a burst of scroll or drag events
//! costs one frame, not one per event.

mod annotate;
mod browse;
mod gallery;
pub(super) mod grid;
mod plots;

pub use browse::pick_run;

use crate::tui::style;

use super::cellart::{rgb, Cells, Glyphs};
use super::color::{Ramp, Rgb, Theme};
use super::data::Rect as WorldRect;
use super::data::NO_CLUSTER;
use super::gene::{GeneMap, Source};
use super::kitty::{Kitty, Transport};
use super::lupin::Job;
use super::render::{self, Frame, Layer, Mode, Style, Viewport};
use super::round::Round;
use super::scalebar;
use super::{
    focus_name, focused, ids, legend_line, legend_swatch, markers, thousands, write_outputs, Base,
    Graphics, Level, Show, ViewArgs, FIT_MARGIN,
};
use clap::ValueEnum;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::Style as TStyle;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::DefaultTerminal;
use ratatui_image::picker::cap_parser::QueryStdioOptions;
use ratatui_image::picker::{Capability, Picker, ProtocolType};
use ratatui_image::protocol::Protocol;
use ratatui_image::{Image, Resize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Side panel width, cells.
const PANEL: u16 = 34;

/// Side panel width while relabelling or in a dialog, cells.
const WIDE_PANEL: u16 = 46;

/// Zoom step per key press or wheel notch.
const ZOOM: f32 = 1.25;

/// Point size step per `<` or `>`.
const POINT_STEP: f32 = 1.25;

pub fn run(args: &ViewArgs) -> anyhow::Result<()> {
    crate::tui::with_terminal(|terminal| {
        let result = show(args, terminal);
        let _ = execute!(std::io::stdout(), DisableMouseCapture);
        result
    })
}

/// Load the run and show it until the user quits.
fn show(args: &ViewArgs, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
    // The terminal is asked first: its background picks the theme the
    // palette is built in.
    let (gfx, px, background) = Gfx::pick(args.graphics);
    let theme = args
        .theme
        .or(background.map(Theme::for_background))
        .unwrap_or(Theme::Dark);
    terminal.draw(|f| {
        let msg = format!(" loading {} ...", args.prefix());
        f.render_widget(Paragraph::new(msg), f.area());
    })?;
    let base = Base::load(args.prefix(), &args.units, theme)?;
    let first = base.run.level_index(args.level.as_deref())?;
    let level = base.level(first, args.edges)?;

    execute!(std::io::stdout(), EnableMouseCapture)?;
    let mut app = App::new(&base, level, args, (gfx, px));
    app.open_rounds(args.round.as_deref().map(PathBuf::from));
    app.run(terminal)
}

/// How the map reaches the screen.
enum Gfx {
    Kitty(Kitty),
    /// Sixel or iTerm2 through ratatui-image.
    Picker(Picker, Option<Protocol>),
    /// Coloured block characters, fitted once per map change.
    Cells(Glyphs, Option<Cells>),
}

impl Gfx {
    /// The drawing method for `choice`, its frame pixels per cell, and the
    /// terminal's background colour when it reports one.
    fn pick(choice: Graphics) -> (Self, (f32, f32), Option<Rgb>) {
        let options = QueryStdioOptions {
            terminal_background_color_osc: true,
            ..Default::default()
        };
        let picker =
            Picker::from_query_stdio_with_options(options).unwrap_or_else(|_| Picker::halfblocks());
        let background = picker.capabilities().iter().find_map(|c| match c {
            Capability::Background(r, g, b) => Some([*r, *g, *b]),
            _ => None,
        });
        let font = picker.font_size();
        let (fw, fh) = (font.width.max(1) as f32, font.height.max(1) as f32);
        let tmux = picker.tmux_detected();
        let kitty = |transport| {
            let kitty = Kitty::new(transport).through_tmux(tmux);
            (Gfx::Kitty(kitty), (fw, fh))
        };
        let cells = |glyphs: Glyphs| {
            let (pw, ph) = glyphs.pixels_per_cell(fh / fw);
            (Gfx::Cells(glyphs, None), (pw as f32, ph as f32))
        };
        let via_picker = |kind| {
            let mut picker = picker.clone();
            picker.set_protocol_type(kind);
            (Gfx::Picker(picker, None), (fw, fh))
        };
        let (gfx, px) = match choice {
            Graphics::Kitty => kitty(Transport::detect()),
            Graphics::KittyInline => kitty(Transport::Direct),
            Graphics::Sixel => via_picker(ProtocolType::Sixel),
            Graphics::Iterm2 => via_picker(ProtocolType::Iterm2),
            Graphics::Quadrants => cells(Glyphs::Quadrants),
            Graphics::Symbols => cells(Glyphs::Symbols),
            Graphics::Blocks => cells(Glyphs::HalfBlocks),
            // tmux passes kitty graphics through only with allow-passthrough,
            // and then only as best effort: block characters always work.
            Graphics::Auto if tmux => cells(Glyphs::Quadrants),
            Graphics::Auto => match picker.protocol_type() {
                ProtocolType::Kitty => kitty(Transport::detect()),
                ProtocolType::Halfblocks => cells(Glyphs::Quadrants),
                other => via_picker(other),
            },
        };
        (gfx, px, background)
    }

    fn name(&self) -> String {
        match self {
            Gfx::Kitty(k) => {
                let how = match k.transport() {
                    Transport::File => "file",
                    Transport::Direct => "inline",
                };
                let tmux = if k.via_tmux() { ", tmux" } else { "" };
                format!("kitty ({how}{tmux})")
            }
            Gfx::Picker(p, _) => format!("{:?}", p.protocol_type()).to_lowercase(),
            Gfx::Cells(glyphs, _) => glyphs.name().into(),
        }
    }
}

struct App<'a> {
    base: &'a Base,
    args: &'a ViewArgs,
    levels: Vec<Option<Level>>,
    cur: usize,
    /// The level the viewer opened on, for `r`.
    home: usize,
    gfx: Gfx,
    /// Image pixels per terminal cell.
    px_per_cell: (f32, f32),

    center: (f32, f32),
    /// World units per image pixel.
    upp: f32,
    layer: Layer,
    community: usize,
    edges: bool,
    /// Cell disc size, × the default (`<`/`>`).
    point: f32,
    /// Batch tile last jumped to.
    tile: Option<usize>,
    /// Every batch side by side (`w`).
    grid: grid::Grid,

    /// Communities shown in colour, one flag each; all false shows every one.
    focus: Vec<bool>,
    /// Community whose markers the panel lists, and those markers:
    /// feature name, symbol, fold.
    shown: Option<usize>,
    markers: Vec<(Box<str>, String, f32)>,
    /// A feature drawn instead of communities, and which source to prefer.
    gene: Option<GeneMap>,
    source: Source,
    /// Percentile the gene ramp tops out at.
    clip: f32,
    /// Cell last clicked.
    picked: Option<usize>,

    map: Rect,
    side: Rect,
    /// Panel rows that respond to clicks.
    clickable: Vec<(u16, Pick)>,
    cursor: Option<(u16, u16)>,
    drag: Option<(u16, u16)>,
    /// Whether the mouse moved since the button went down.
    dragged: bool,
    help: bool,
    status: String,

    mode: Mode,
    render_time: Duration,
    send_time: Duration,
    send_bytes: usize,

    /// What groups the cells: the level's communities or the round's.
    show: Show,
    /// The lupin round `a` shows, loaded on first use.
    round: Option<Round>,
    /// The newest round of each chain made from this run, newest first;
    /// `n`/`N` step through them.
    rounds: Vec<PathBuf>,
    /// A lupin subprocess running in the background.
    job: Option<Job>,
    /// What the main area shows: the map, a structure plot, a heatmap.
    view: plots::View,
    /// The structure plot under the map (`t`), its panel names under it,
    /// and the community clicked in it while the map shows a round.
    structure: bool,
    bars: Rect,
    below: Rect,
    bar_focus: Option<usize>,
    /// The map and the structure plot, sent as one image.
    canvas: Rect,
    /// Plots kept until what they show changes.
    plots: plots::Cache,
    /// Figures saved in this directory, this session and before.
    gallery: super::saved::Gallery,
    show_saved: bool,
    /// Where the saved figures go, left of the map; empty when hidden.
    strip: Rect,
    thumbs: gallery::Thumbs,
    /// Relabelling the round's clusters (`R`).
    relabel: Option<annotate::Relabel>,
    /// A dialog that takes the keys: file browser, prompt, confirmation.
    modal: Option<annotate::Modal>,

    /// The map must be rendered again.
    need_map: bool,
    /// The panel must be drawn again (always, when the map is).
    need_panel: bool,
    quit: bool,
}

impl<'a> App<'a> {
    fn new(base: &'a Base, level: Level, args: &'a ViewArgs, gfx: (Gfx, (f32, f32))) -> Self {
        let mut levels: Vec<Option<Level>> = base.run.levels.iter().map(|_| None).collect();
        let cur = level.index;
        let community = match args.layer {
            Layer::Community(c) => c.min(level.comm.k.saturating_sub(1)),
            _ => 0,
        };
        let k = level.comm.k;
        levels[cur] = Some(level);
        App {
            base,
            args,
            levels,
            cur,
            home: cur,
            gfx: gfx.0,
            px_per_cell: gfx.1,
            center: (0., 0.),
            upp: 0.,
            layer: args.layer,
            community,
            edges: args.edges,
            point: args.point_size,
            tile: None,
            grid: grid::Grid::default(),
            focus: vec![false; k],
            shown: None,
            markers: Vec::new(),
            gene: None,
            source: Source::Observed,
            clip: args.clip,
            picked: None,
            map: Rect::default(),
            side: Rect::default(),
            clickable: Vec::new(),
            cursor: None,
            drag: None,
            dragged: false,
            help: false,
            status: String::new(),
            mode: Mode::Points,
            render_time: Duration::ZERO,
            send_time: Duration::ZERO,
            send_bytes: 0,
            show: Show::Communities,
            round: None,
            rounds: Vec::new(),
            job: None,
            relabel: None,
            modal: None,
            view: plots::View::Map,
            structure: false,
            bars: Rect::default(),
            below: Rect::default(),
            bar_focus: None,
            canvas: Rect::default(),
            plots: plots::Cache::default(),
            gallery: super::saved::Gallery::here(),
            show_saved: true,
            strip: Rect::default(),
            thumbs: Default::default(),
            need_map: true,
            need_panel: true,
            quit: false,
        }
    }

    /// What the map draws: the current level, or a grouping of the round.
    fn level(&self) -> &Level {
        match (self.show, &self.round) {
            (Show::Types, Some(r)) => &r.types,
            (Show::Clusters, Some(r)) => &r.clusters,
            _ => self.levels[self.cur]
                .as_ref()
                .expect("current level is loaded"),
        }
    }

    fn level_mut(&mut self) -> &mut Level {
        match (self.show, &mut self.round) {
            (Show::Types, Some(r)) => &mut r.types,
            (Show::Clusters, Some(r)) => &mut r.clusters,
            _ => self.levels[self.cur]
                .as_mut()
                .expect("current level is loaded"),
        }
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        self.layout(terminal)?;
        self.fit(self.base.geom.bounds());
        while !self.quit {
            if self.need_map || self.need_panel {
                self.redraw(terminal)?;
            }
            self.poll_job(terminal)?;
            if event::poll(Duration::from_millis(250))? {
                self.handle(event::read()?, terminal)?;
                while !self.quit && event::poll(Duration::ZERO)? {
                    self.handle(event::read()?, terminal)?;
                }
            }
        }
        if let Gfx::Kitty(k) = &mut self.gfx {
            k.clear(&mut std::io::stdout())?;
        }
        Ok(())
    }

    // ── geometry ────────────────────────────────────────────────────────

    fn layout(&mut self, terminal: &DefaultTerminal) -> anyhow::Result<Rect> {
        let size = terminal.size()?;
        let area = Rect::new(0, 0, size.width, size.height);
        let wide = self.relabel.is_some() || self.modal.is_some();
        let panel = if wide { WIDE_PANEL } else { PANEL }.min(area.width / 2);
        let [mut map, side] =
            Layout::horizontal([Constraint::Min(1), Constraint::Length(panel)]).areas(area);
        // Saved figures on the left.
        self.strip = Rect::default();
        if self.strip_shows(map.width) {
            self.strip = Rect::new(map.x, map.y, gallery::SAVED_WIDTH, map.height);
            map.x += gallery::SAVED_WIDTH;
            map.width -= gallery::SAVED_WIDTH;
        }
        // The structure plot under the map, a quarter of its height, and
        // two rows of panel names under that.
        self.bars = Rect::default();
        self.below = Rect::default();
        if self.structure && self.view == plots::View::Map && !self.grid_shows() && map.height >= 16
        {
            let (h, names) = ((map.height / 4).max(6), 2);
            self.below = Rect::new(map.x, map.bottom() - names, map.width, names);
            self.bars = Rect::new(map.x, map.bottom() - names - h, map.width, h);
            map.height -= h + names;
        }
        let canvas = Rect {
            height: map.height + self.bars.height,
            ..map
        };
        if map != self.map || canvas != self.canvas {
            self.map = map;
            self.canvas = canvas;
            self.need_map = true;
        }
        self.side = side;
        Ok(side)
    }

    fn image_size(&self) -> (usize, usize) {
        (
            ((self.map.width as f32 * self.px_per_cell.0) as usize).max(1),
            ((self.map.height as f32 * self.px_per_cell.1) as usize).max(1),
        )
    }

    fn viewport(&self) -> Viewport {
        let (w, h) = self.image_size();
        Viewport::at(self.center, self.upp, w, h)
    }

    /// Show `vp`'s window on the map.
    fn set_view(&mut self, vp: &Viewport) {
        self.upp = vp.upp;
        self.center = vp.centre();
        self.need_map = true;
    }

    /// The zoom limits, in world units per pixel, for a `w × h` frame
    /// that fits `r`: down to single cells, out to 8× the fit.
    fn zoom_limits(&self, r: WorldRect, (w, h): (usize, usize)) -> (f32, f32) {
        (self.base.spacing / 80., Viewport::fit(r, w, h).upp * 8.)
    }

    /// Fit `r`, with a small margin, to the map.
    fn fit(&mut self, r: WorldRect) {
        let r = r.pad(FIT_MARGIN);
        let (w, h) = self.image_size();
        let vp = Viewport::fit(r, w, h);
        self.upp = vp.upp;
        self.center = (0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1));
        self.need_map = true;
    }

    /// Pixel under terminal cell `(col, row)`, in an image placed at `r`.
    fn px_in(&self, r: Rect, col: u16, row: u16) -> (f32, f32) {
        (
            (col.saturating_sub(r.x) as f32 + 0.5) * self.px_per_cell.0,
            (row.saturating_sub(r.y) as f32 + 0.5) * self.px_per_cell.1,
        )
    }

    /// World point under terminal cell `(col, row)`.
    fn world_at(&self, col: u16, row: u16) -> (f32, f32) {
        let vp = self.viewport();
        let (px, py) = self.px_in(self.map, col, row);
        (vp.x0 + px * vp.upp, vp.y0 + py * vp.upp)
    }

    fn in_map(&self, col: u16, row: u16) -> bool {
        self.map.contains(Position::new(col, row))
    }

    /// Scale by `f` (> 1 zooms out) keeping `anchor` fixed on screen.
    fn zoom(&mut self, f: f32, anchor: Option<(u16, u16)>) {
        let limits = self.zoom_limits(self.base.geom.bounds(), self.image_size());
        let anchor = anchor.map(|(c, r)| self.px_in(self.map, c, r));
        let vp = self.viewport().zoomed(f, anchor, limits);
        self.set_view(&vp);
    }

    /// Pan by a fraction of the view.
    fn pan(&mut self, fx: f32, fy: f32) {
        let (w, h) = self.image_size();
        self.center.0 += fx * w as f32 * self.upp;
        self.center.1 += fy * h as f32 * self.upp;
        self.need_map = true;
    }

    // ── input ───────────────────────────────────────────────────────────

    fn handle(&mut self, ev: Event, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        self.need_panel = true;
        match ev {
            Event::Key(key) if key.kind != KeyEventKind::Release => self.key(key, terminal)?,
            Event::Mouse(m) => self.mouse(m),
            Event::Resize(..) => {
                terminal.autoresize()?;
                self.need_map = true;
            }
            _ => {}
        }
        Ok(())
    }

    fn key(&mut self, key: KeyEvent, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let big = key.modifiers.contains(KeyModifiers::SHIFT);
        let step = if big { 0.5 } else { 0.125 };
        self.status.clear();
        if self.modal.is_some() {
            return self.modal_key(key, terminal);
        }
        if self.view == plots::View::Heatmap && !plots::chart_key(key) {
            self.status = "on a chart: H next chart or back to the map · esc the map".into();
            return Ok(());
        }
        if self.relabel.is_some() && self.relabel_key(key, terminal)? {
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
            KeyCode::Char('+' | '=') if self.view == plots::View::Heatmap => self.more_genes(1),
            KeyCode::Char('-' | '_') if self.view == plots::View::Heatmap => self.more_genes(-1),
            KeyCode::Char('+' | '=') => self.zoom(1. / ZOOM, None),
            KeyCode::Char('-' | '_') => self.zoom(ZOOM, None),
            // Zoom keys that mean the same in every mode.
            KeyCode::Char('z') => self.zoom(1. / ZOOM, None),
            KeyCode::Char('Z') => self.zoom(ZOOM, None),
            KeyCode::Char('H') => self.cycle_chart(),
            KeyCode::Char('f') => self.toggle_saved(),
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
            KeyCode::Char('s') => self.export()?,
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
        if self.grid_shows() {
            return self.grid_mouse(m);
        }
        // The structure plot: a click picks a community; nothing pans it.
        if self.bars.contains(Position::new(col, row)) {
            match m.kind {
                MouseEventKind::Down(MouseButton::Left) => self.bar_click(col, row),
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
                    None => {}
                }
            }
            MouseEventKind::Down(MouseButton::Right) if self.in_map(col, row) => {
                self.click(col, row, true);
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if let Some((c0, r0)) = self.drag {
                    self.dragged |= (c0, r0) != (col, row);
                    let dx = (col as f32 - c0 as f32) * self.px_per_cell.0 * self.upp;
                    let dy = (row as f32 - r0 as f32) * self.px_per_cell.1 * self.upp;
                    self.center.0 -= dx;
                    self.center.1 -= dy;
                    self.drag = Some((col, row));
                    self.need_map = true;
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
        let radius = self.base.spacing.max(2. * self.px_per_cell.0 * self.upp);
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

    /// Show community `c` alone, or add/remove it from the shown set.
    /// Selecting the only shown community again shows everything.
    fn select(&mut self, c: usize, add: bool) {
        if c >= self.focus.len() {
            return;
        }
        let only_c = self.focus[c] && self.focus.iter().filter(|&&f| f).count() == 1;
        if add {
            self.focus[c] = !self.focus[c];
        } else {
            self.focus.fill(false);
            self.focus[c] = !only_c;
        }
        self.shown = if self.focus[c] {
            Some(c)
        } else {
            self.focus.iter().position(|&f| f)
        };
        self.list_markers();
        self.need_map = true;
    }

    /// Fill the panel's marker list for the shown community.
    fn list_markers(&mut self) {
        self.markers.clear();
        let Some(shown) = self.shown else {
            return;
        };
        let base = self.base;
        let level = self.level_mut();
        match base.load_features(level) {
            Ok(()) => {
                let rates = level.features.as_ref().expect("just loaded");
                self.markers = rates
                    .top(shown, 12)
                    .into_iter()
                    .map(|m| {
                        let symbol = markers::symbol(&m.name).to_string();
                        (m.name, symbol, m.fold)
                    })
                    .collect();
            }
            Err(e) => self.status = format!("no markers: {e}"),
        }
    }

    /// `Esc`: one step back, never out of the viewer. A plot goes back to
    /// the map, then the mapped gene goes, then the selection.
    fn back(&mut self) {
        if self.bar_focus.take().is_some() {
            self.need_map = true;
        } else if self.view != plots::View::Map {
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

    fn clear_focus(&mut self) {
        self.focus.fill(false);
        self.shown = None;
        self.markers.clear();
        self.gene = None;
        self.picked = None;
        self.need_map = true;
    }

    fn focus(&self) -> Option<&[bool]> {
        self.focus.iter().any(|&f| f).then_some(&self.focus[..])
    }

    /// Draw marker `i` of the shown community; picking the drawn one again
    /// goes back to communities.
    fn pick_gene(&mut self, i: usize) {
        let Some((feature, ..)) = self.markers.get(i).cloned() else {
            return;
        };
        if self.gene.as_ref().is_some_and(|g| g.feature == feature) {
            self.gene = None;
            self.need_map = true;
        } else {
            self.show_gene(&feature);
        }
    }

    /// Map `feature`, on the map (a plot in place of it steps aside).
    fn show_gene(&mut self, feature: &str) {
        self.view = plots::View::Map;
        let (base, source, clip) = (self.base, self.source, self.clip);
        match base.gene(self.level_mut(), feature, source, clip) {
            Ok(g) => {
                if g.fell_back(source) {
                    self.status = "no data file: model-expected level".into();
                }
                self.gene = Some(g);
            }
            Err(e) => self.status = format!("{e}"),
        }
        self.need_map = true;
    }

    /// The next (`by` = 1) or previous marker gene.
    fn step_gene(&mut self, by: isize) {
        let n = self.markers.len() as isize;
        if n == 0 {
            self.status = "select a community first".into();
            return;
        }
        let at = self.gene.as_ref().and_then(|g| {
            self.markers
                .iter()
                .position(|(feature, ..)| *feature == g.feature)
        });
        let next = match at {
            Some(i) => (i as isize + by).rem_euclid(n),
            None if by > 0 => 0,
            None => n - 1,
        };
        let feature = self.markers[next as usize].0.clone();
        self.show_gene(&feature);
    }

    /// Switch the gene ramp's top between the 99th and 95th percentile.
    fn toggle_clip(&mut self) {
        self.clip = if self.clip > 95. { 95. } else { 99. };
        self.status = format!("gene ramp tops at p{}", self.clip);
        self.refresh_gene();
    }

    /// Switch between observed counts and the model-expected level.
    fn toggle_source(&mut self) {
        self.source = self.source.other();
        self.refresh_gene();
    }

    /// Draw the shown gene again after its source or clip changed.
    fn refresh_gene(&mut self) {
        if let Some(feature) = self.gene.as_ref().map(|g| g.feature.clone()) {
            self.show_gene(&feature);
        }
    }

    fn set_layer(&mut self, layer: Layer) {
        self.gene = None;
        if layer == Layer::Entropy && self.level().comm.entropy.is_none() {
            self.status = "this level has no entropy".into();
            return;
        }
        self.layer = layer;
        self.need_map = true;
    }

    /// `]`/`[`: focus the next (`by` = 1) or previous group, largest first;
    /// a map of one community's propensity follows it.
    fn step_focus(&mut self, by: isize) {
        let order = self.level().comm.by_size.clone();
        let n = order.len() as isize;
        if n == 0 {
            return;
        }
        let at = self.shown.and_then(|c| order.iter().position(|&x| x == c));
        let next = match at {
            Some(i) => (i as isize + by).rem_euclid(n),
            None if by > 0 => 0,
            None => n - 1,
        };
        let c = order[next as usize];
        self.focus.fill(false);
        self.focus[c] = true;
        self.shown = Some(c);
        self.gene = None;
        self.list_markers();
        if let Layer::Community(_) = self.layer {
            self.community = c;
            self.layer = Layer::Community(c);
        }
        self.status = format!("{}  ] [ next/prev", self.level().comm.name(c));
        self.need_map = true;
    }

    /// Back to the view the viewer opened on: the whole tissue, the starting
    /// level and layer, edges as asked, nothing selected.
    fn reset(&mut self) {
        self.leave_relabel();
        self.cur = self.home;
        self.show = Show::Communities;
        self.bar_focus = None;
        let k = self.level().comm.k;
        self.focus = vec![false; k];
        self.shown = None;
        self.markers.clear();
        self.gene = None;
        self.picked = None;
        self.source = Source::Observed;
        self.clip = self.args.clip;
        self.layer = self.args.layer;
        self.community = match self.args.layer {
            Layer::Community(c) => c.min(k.saturating_sub(1)),
            _ => 0,
        };
        self.edges = self.args.edges;
        self.point = self.args.point_size;
        self.leave_grid();
        self.fit_all();
        self.status = "back to the start".into();
    }

    fn step_level(&mut self, by: isize, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let n = self.levels.len() as isize;
        let next = (self.cur as isize + by).clamp(0, n - 1) as usize;
        if next == self.cur && self.show == Show::Communities {
            return Ok(());
        }
        self.show = Show::Communities;
        self.bar_focus = None;
        if self.levels[next].is_none() {
            self.busy(
                terminal,
                format!("loading level {} ...", self.base.run.levels[next].tag),
            )?;
            self.levels[next] = Some(self.base.level(next, self.edges)?);
        }
        self.cur = next;
        let k = self.level().comm.k;
        self.focus = vec![false; k];
        self.shown = None;
        self.markers.clear();
        self.gene = None;
        self.community = self.community.min(k.saturating_sub(1));
        if let Layer::Community(_) = self.layer {
            self.layer = Layer::Community(self.community);
        }
        self.need_map = true;
        Ok(())
    }

    fn toggle_edges(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        if self.level().grouping {
            self.status = "edges belong to communities: a to show them".into();
            return Ok(());
        }
        self.edges = !self.edges;
        if self.edges && self.level().edges.is_none() {
            self.busy(terminal, "loading edges ...".into())?;
            let base = self.base;
            base.load_edges(self.level_mut())?;
        }
        if self.edges
            && render::mode(&self.base.scene(self.level()), &self.viewport(), self.point)
                != Mode::Points
        {
            self.status = "edges show once zoomed in".into();
        }
        self.need_map = true;
        Ok(())
    }

    /// `<`/`>`: smaller or larger cell discs, on every map.
    fn scale_points(&mut self, by: f32) {
        self.point = step_point(self.point, by);
        self.status = format!("point size ×{:.2}  < >", self.point);
        self.need_map = true;
    }

    /// `0`: every batch fitted, the whole tissue on the map.
    fn fit_all(&mut self) {
        self.grid.forget_own();
        self.tile = None;
        self.fit(self.base.geom.bounds());
    }

    fn next_tile(&mut self) {
        self.leave_grid();
        let tiles = &self.base.geom.tiles;
        let next = self.tile.map_or(0, |t| (t + 1) % tiles.len());
        self.tile = Some(next);
        let bounds = tiles[next].bounds;
        self.status = format!("batch {}", tiles[next].name);
        self.fit(bounds);
    }

    /// Freeze the current view: a PNG and a PDF at `--export-scale` × the
    /// screen resolution, and a `.txt` with the command that redraws them,
    /// the legend and the shown communities' markers.
    fn export(&mut self) -> anyhow::Result<()> {
        if self.view != plots::View::Map {
            return self.export_plot();
        }
        if self.grid_shows() {
            return self.export_grid();
        }
        let stem = free_stem("pinto-view", "png");
        let (png, pdf, txt) = (
            format!("{stem}.png"),
            format!("{stem}.pdf"),
            format!("{stem}.txt"),
        );

        let vp = self.viewport();
        let scale = self.args.export_scale.max(1);
        let (w, h) = (vp.w * scale, vp.h * scale);
        // Through `fit`, exactly as the redraw command's `--bbox` goes, so
        // the redraw repeats this arithmetic and matches pixel for pixel.
        let window = vp.window();
        let hi = Viewport::fit(window, w, h);
        let (level, style) = self.drawn();
        write_outputs(
            self.base,
            level,
            &style,
            self.gene.as_ref(),
            &hi,
            Some(png.as_ref()),
            Some(pdf.as_ref()),
        )?;
        std::fs::write(&txt, self.export_notes(&png, &pdf, &hi, window))?;
        // A small render of the same window for the saved figures.
        let (tw, th) = (
            super::saved::THUMB_WIDTH,
            super::saved::THUMB_WIDTH * vp.h / vp.w.max(1),
        );
        let small = Viewport::fit(window, tw, th.max(1));
        let (level, style) = self.drawn();
        let thumb = self.base.render(level, &style, self.gene.as_ref(), &small);
        let what = match &self.gene {
            Some(g) => format!("map · {}", markers::symbol(&g.feature)),
            None => format!("map · {} · {}", level.comm.tag, style.layer),
        };
        let listed = self.remember(&pdf, &what, &thumb);
        let bars = if self.bars.height > 0 {
            self.export_structure()?
        } else {
            String::new()
        };
        self.status = format!("saved {stem}.png .pdf .txt{listed}{bars}");
        Ok(())
    }

    /// `window` is the `--bbox` that `vp` was fitted from.
    fn export_notes(&self, png: &str, pdf: &str, vp: &Viewport, window: WorldRect) -> String {
        use std::fmt::Write as _;
        let (level, style) = self.drawn();
        let comm = &level.comm;
        let focused = focused(style.focus);
        // The run's level, also under a round's grouping, whose own tag
        // (`cell types`) is no level `--level` knows.
        let tag = &self.base.run.levels[level.index].tag;

        let win = window;
        let mut cmd = format!(
            // `{}` prints the shortest text that parses back to the same f32.
            "pinto view {} --png {png} --pdf {pdf} --units {} --width {} --height {} \
             --bbox={},{},{},{} --level {} --layer {}",
            self.args.prefix(),
            self.args.units,
            vp.w,
            vp.h,
            win.x0,
            win.y0,
            win.x1,
            win.y1,
            tag,
            style.layer,
        );
        if !focused.is_empty() {
            write!(cmd, " --focus {}", ids(comm, &focused, ",")).ok();
        }
        if style.edges {
            cmd.push_str(" --edges");
        }
        if style.point != 1. {
            write!(cmd, " --point-size {}", style.point).ok();
        }
        let theme = self
            .base
            .theme
            .to_possible_value()
            .expect("no skipped variants");
        write!(cmd, " --theme {}", theme.get_name()).ok();
        let round_drawn = self.show != Show::Communities && self.bar_focus_level().is_none();
        if let (Some(round), true) = (&self.round, round_drawn) {
            let show = self.show.to_possible_value().expect("no skipped variants");
            write!(
                cmd,
                " --round {} --show {}",
                round.path.display(),
                show.get_name()
            )
            .ok();
        }
        if let Some(g) = &self.gene {
            write!(cmd, " --gene {} --clip {}", g.feature, g.clip).ok();
            if g.source == Source::Expected {
                cmd.push_str(" --expected");
            }
        }

        let mut out = String::new();
        writeln!(out, "# pinto view export").ok();
        writeln!(out, "image    {png} ({}×{}), {pdf}", vp.w, vp.h).ok();
        writeln!(out, "redraw   {cmd}").ok();
        writeln!(out, "run      {}", self.base.run.source()).ok();
        writeln!(out, "level    {tag} (K={})", comm.k).ok();
        writeln!(out, "layer    {}", style.layer).ok();
        writeln!(out, "scale    {:.4} units/px", vp.upp).ok();
        if !focused.is_empty() {
            writeln!(out, "focus    {}", ids(comm, &focused, " ")).ok();
        }
        if let Some(g) = &self.gene {
            writeln!(out, "gene     {}, ramp 0..{}", g.title(), g.top_label()).ok();
        }

        writeln!(out, "\nlegend").ok();
        for &c in &comm.by_size {
            writeln!(out, "{}", legend_line(comm, c, level.palette[c])).ok();
        }
        if let (Some(rates), false) = (level.features.as_ref(), focused.is_empty()) {
            writeln!(out, "\nmarkers (fold over the other communities)").ok();
            for &c in &focused {
                writeln!(out, "  {:<5} {}", comm.name(c), rates.summary(c, 20)).ok();
            }
        }
        out
    }

    /// Show `msg` before a blocking load.
    fn busy(&mut self, terminal: &mut DefaultTerminal, msg: String) -> anyhow::Result<()> {
        self.status = msg;
        let side = self.layout(terminal)?;
        let (panel, _) = self.panel(side.height);
        terminal.draw(|f| f.render_widget(Paragraph::new(panel), side))?;
        self.status.clear();
        Ok(())
    }

    /// What the map draws: the level and style on screen, or, for a
    /// community picked in the structure plot on a map of a round's groups,
    /// its level and propensity.
    fn drawn(&self) -> (&Level, Style<'_>) {
        match self.bar_focus_level() {
            Some((level, c)) => (
                level,
                Style {
                    layer: Layer::Community(c),
                    edges: false,
                    focus: None,
                    theme: self.base.theme,
                    point: self.point,
                },
            ),
            None => (self.level(), self.style()),
        }
    }

    fn style(&self) -> Style<'_> {
        Style {
            layer: self.layer,
            edges: self.edges && !self.level().grouping,
            focus: self.focus(),
            theme: self.base.theme,
            point: self.point,
        }
    }

    // ── drawing ─────────────────────────────────────────────────────────

    fn redraw(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        self.need_panel = false;
        let side = self.layout(terminal)?;
        let mut fresh: Option<Frame> = None;
        if self.need_map && self.view != plots::View::Map {
            self.need_map = false;
            self.clear_frame()?;
        }
        if self.need_map && self.grid_shows() {
            self.need_map = false;
            let t = Instant::now();
            let (w, h) = self.image_size();
            let frame = self.grid_frame((w, h), true);
            self.render_time = t.elapsed();
            self.send_frame(frame, &mut fresh)?;
        }
        if self.need_map {
            self.need_map = false;
            let vp = self.viewport();
            let t = Instant::now();
            let (level, style) = self.drawn();
            let scene = self.base.scene(level);
            let mut frame = self.base.render(level, &style, self.gene.as_ref(), &vp);
            // Unlabelled: the panel states the bar's length as text.
            if let Some(units) = self.base.units {
                scalebar::draw(&mut frame, &vp, units, false);
            }
            let mode = render::mode(&scene, &vp, self.point);
            if self.bars.height > 0 {
                frame = self.with_structure(frame);
            }
            self.render_time = t.elapsed();
            self.mode = mode;
            self.send_frame(frame, &mut fresh)?;
        }

        let (panel, clickable) = self.panel(side.height);
        self.clickable = clickable
            .into_iter()
            .map(|(line, pick)| (side.y + line as u16, pick))
            .collect();
        let map = self.map;
        let canvas = self.canvas;
        let below = self.below;
        let text = self.plot_text();
        let names = self.plot_names();
        self.prepare_thumbs();
        let strip = self.strip;
        let app = &*self;
        let gfx = &self.gfx;
        terminal.draw(|f| {
            f.render_widget(Paragraph::new(panel), side);
            if strip.width > 0 {
                app.draw_saved(f, strip);
            }
            if let Some(text) = text {
                f.render_widget(Paragraph::new(text), map);
                return;
            }
            match gfx {
                Gfx::Picker(_, Some(proto)) => f.render_widget(Image::new(proto), canvas),
                Gfx::Cells(_, Some(cells)) => f.render_widget(cells, canvas),
                _ => {}
            }
            if let Some(names) = names {
                f.render_widget(Paragraph::new(names), below);
            }
        })?;

        if let (Gfx::Kitty(kitty), Some(frame)) = (&mut self.gfx, fresh) {
            let t = Instant::now();
            kitty.show(
                &mut std::io::stdout(),
                &frame,
                (canvas.x, canvas.y),
                (canvas.width, canvas.height),
            )?;
            self.send_time = t.elapsed();
            self.send_bytes = kitty.last_bytes;
        }
        Ok(())
    }

    /// Hand a rendered frame to the terminal's drawing method; kitty's is
    /// sent after the text, through `fresh`.
    fn send_frame(&mut self, frame: Frame, fresh: &mut Option<Frame>) -> anyhow::Result<()> {
        {
            match &mut self.gfx {
                Gfx::Kitty(_) => *fresh = Some(frame),
                Gfx::Cells(glyphs, slot) => {
                    let t = Instant::now();
                    let (cols, rows) = (self.canvas.width as usize, self.canvas.height as usize);
                    let ppc = (self.px_per_cell.0 as usize, self.px_per_cell.1 as usize);
                    *slot = Some(Cells::fit(&frame, *glyphs, ppc, (cols, rows)));
                    self.send_time = t.elapsed();
                }
                Gfx::Picker(picker, proto) => {
                    let t = Instant::now();
                    let img =
                        image::RgbaImage::from_raw(frame.w as u32, frame.h as u32, frame.rgba)
                            .expect("frame buffer matches its size");
                    *proto = Some(picker.new_protocol(
                        image::DynamicImage::ImageRgba8(img),
                        self.canvas.as_size(),
                        Resize::Fit(None),
                    )?);
                    self.send_time = t.elapsed();
                }
            }
        }
        Ok(())
    }

    /// Take the image off the screen, for a view drawn as text.
    fn clear_frame(&mut self) -> anyhow::Result<()> {
        match &mut self.gfx {
            Gfx::Kitty(k) => k.clear(&mut std::io::stdout())?,
            Gfx::Cells(_, slot) => *slot = None,
            Gfx::Picker(_, proto) => *proto = None,
        }
        Ok(())
    }

    /// Panel lines, and which of them respond to clicks.
    fn panel(&self, height: u16) -> (Vec<Line<'static>>, Vec<(usize, Pick)>) {
        let level = self.level();
        let comm = &level.comm;
        let dim = style::dim();
        let bold = style::bold();
        let row = |k: &str, v: String| {
            Line::from(vec![Span::styled(format!(" {k:<7}"), dim), Span::raw(v)])
        };

        let mut lines = vec![
            Line::from(Span::styled(
                format!(" pinto view  {}", self.base.run.name()),
                bold,
            )),
            row(
                "level",
                format!(
                    "{} ({}/{})  l L",
                    self.base.run.levels[self.cur].tag,
                    self.cur + 1,
                    self.levels.len()
                ),
            ),
            match &self.gene {
                Some(g) => row(
                    "gene",
                    format!("{}  o: {}", markers::symbol(&g.feature), g.source.name()),
                ),
                None => row("layer", format!("{}  1-4", self.layer)),
            },
            row(
                "cells",
                format!("{}  K={}", thousands(self.base.geom.n()), comm.k),
            ),
            row("show", self.show_line()),
            row("scale", format!("{:.3} /px  {}", self.upp, self.mode)),
            row("point", format!("×{:.2}  < >", self.point)),
            row(
                "bar",
                match self.base.units {
                    Some(units) => {
                        let (w, _) = self.image_size();
                        scalebar::bar_for(w as f32 * self.upp, units).label
                    }
                    None => "off (--units)".into(),
                },
            ),
            row(
                "frame",
                format!(
                    "{:.1} ms + {:.1} ms",
                    ms(self.render_time),
                    ms(self.send_time)
                ),
            ),
            row("screen", self.gfx.name()),
        ];
        if let Some(r) = self.round_line() {
            lines.insert(5, row("round", r));
        }
        if let Gfx::Kitty(_) = self.gfx {
            lines.push(row(
                "sent",
                format!("{:.1} MB", self.send_bytes as f64 / 1e6),
            ));
        }
        if self.edges {
            let n = level.edges.as_ref().map_or(0, |(e, _)| e.len());
            lines.push(row("edges", format!("{} (e)", thousands(n))));
        }
        match self.cursor {
            _ if self.grid_shows() => lines.push(row("grid", self.grid_line())),
            Some((c, r)) => {
                let (x, y) = self.world_at(c, r);
                lines.push(row("at", format!("{x:.1}, {y:.1}")));
            }
            None => lines.push(Line::raw("")),
        }
        if let Some(i) = self.picked {
            lines.extend(self.cell_lines(i));
        }
        lines.push(Line::styled(
            format!(" {}", self.status),
            // Bold in the terminal's own text colour reads on any background.
            style::bold(),
        ));

        let help = help_lines(self.help, self.relabel.is_some());
        let mut clickable = Vec::new();
        if let Some(modal) = &self.modal {
            let room = (height as usize).saturating_sub(lines.len());
            lines.extend(self.modal_lines(modal, room));
            return (lines, clickable);
        }
        if self.relabel.is_some() {
            let room = (height as usize).saturating_sub(lines.len() + help.len());
            for (line, pick) in self.relabel_lines(room) {
                if let Some(pick) = pick {
                    clickable.push((lines.len(), pick));
                }
                lines.push(line);
            }
            let pad = (height as usize).saturating_sub(lines.len() + help.len());
            lines.extend(std::iter::repeat_n(Line::raw(""), pad));
            lines.extend(help);
            return (lines, clickable);
        }
        let free = (height as usize).saturating_sub(lines.len() + help.len() + 1);
        if let Some(c) = self.shown {
            // Leave the legend at least a few rows.
            for (line, pick) in self.marker_lines(c, free.saturating_sub(6).min(12)) {
                if let Some(pick) = pick {
                    clickable.push((lines.len(), pick));
                }
                lines.push(line);
            }
        }
        let room = (height as usize).saturating_sub(lines.len() + help.len() + 1);
        lines.push(Line::raw(""));
        let mut legend = self.legend(room);
        // On a round's map the structure plot's colours are other groups:
        // its communities get a list of their own, to pick one from.
        if self.bars_shown() && self.show != Show::Communities {
            let left = room.saturating_sub(legend.len() + 1);
            legend.extend(self.bars_legend(left));
        }
        for (line, pick) in legend {
            if let Some(pick) = pick {
                clickable.push((lines.len(), pick));
            }
            lines.push(line);
        }
        let pad = (height as usize).saturating_sub(lines.len() + help.len());
        lines.extend(std::iter::repeat_n(Line::raw(""), pad));
        lines.extend(help);
        (lines, clickable)
    }

    /// The picked cell: name, then its strongest communities and entropy.
    fn cell_lines(&self, i: usize) -> Vec<Line<'static>> {
        let comm = &self.level().comm;
        let dim = style::dim();
        let k = comm.k;
        let mut top: Vec<(usize, u8)> = (0..k).map(|c| (c, comm.prop[i * k + c])).collect();
        top.sort_by_key(|&(_, q)| std::cmp::Reverse(q));
        let mut mix: Vec<String> = top
            .iter()
            .take(3)
            .filter(|&&(_, q)| q > 0)
            .map(|&(c, q)| format!("{} {:.2}", comm.name(c), q as f32 / 255.))
            .collect();
        if let Some(h) = comm.entropy.as_ref() {
            mix.push(format!("H {:.2}", h[i] as f32 / 255.));
        }
        let geom = &self.base.geom;
        let mut name: String = geom.names[i].chars().take(25).collect();
        if geom.tiles.len() > 1 {
            name = format!("{name}  ({})", geom.tiles[geom.batch[i] as usize].name);
        }
        vec![
            Line::from(vec![Span::styled(" cell   ", dim), Span::raw(name)]),
            Line::raw(format!("        {}", mix.join("  "))),
        ]
    }

    /// Up to `n` of the shown community `c`'s markers, each clickable.
    fn marker_lines(&self, c: usize, n: usize) -> Vec<(Line<'static>, Option<Pick>)> {
        let level = self.level();
        if self.markers.is_empty() {
            return Vec::new();
        }
        let bold = style::bold();
        let dim = style::dim();
        let mut out = vec![
            (Line::raw(""), None),
            (
                Line::from(vec![
                    Span::raw(" "),
                    Span::styled("██", TStyle::default().fg(rgb(level.palette[c]))),
                    Span::styled(format!(" {} markers", level.comm.name(c)), bold),
                    Span::styled("  fold  g G", dim),
                ]),
                None,
            ),
        ];
        let drawn = self.gene.as_ref().map(|g| g.feature.as_ref());
        out.extend(
            self.markers
                .iter()
                .take(n)
                .enumerate()
                .map(|(i, (feature, symbol, fold))| {
                    let on = drawn.is_some_and(|d| {
                        d == feature.as_ref() || markers::symbol(d) == symbol.as_str()
                    });
                    let mark = if on { " ▸ " } else { "   " };
                    let name: String = symbol.chars().take(20).collect();
                    let style = if on { bold } else { TStyle::default() };
                    let line = Line::from(vec![
                        Span::raw(mark),
                        Span::styled(format!("{name:<20} ×{fold:<6.1}"), style),
                    ]);
                    (line, Some(Pick::Gene(i)))
                }),
        );
        out
    }

    /// Legend lines, each tagged with its community when it names one.
    fn legend(&self, room: usize) -> Vec<(Line<'static>, Option<Pick>)> {
        // A community picked in the structure plot, drawn as its propensity.
        if let Some((level, c)) = self.bar_focus_level() {
            let title = format!("{} propensity  Esc", level.comm.name(c));
            return ramp_legend(Layer::Community(c).ramp(self.base.theme), &title, "1")
                .into_iter()
                .map(|l| (l, None))
                .collect();
        }
        let level = self.level();
        let comm = &level.comm;
        let swatch = |c: Rgb| Span::styled("██", TStyle::default().fg(rgb(c)));
        let untagged = |lines: Vec<Line<'static>>| lines.into_iter().map(|l| (l, None)).collect();
        if let Some(g) = &self.gene {
            return untagged(ramp_legend(
                Layer::Gene.ramp(self.base.theme),
                &format!("{}  o p", g.title()),
                &g.top_label(),
            ));
        }
        match self.layer {
            Layer::Argmax | Layer::Soft => {
                let (order, sizes) = (&comm.by_size, &comm.sizes);
                let focus = self.focus();
                let shown = order.len().min(room);
                let mut out: Vec<(Line, Option<Pick>)> = order[..shown]
                    .iter()
                    .map(|&c| {
                        let (colour, on) = legend_swatch(c, &level.palette, focus, self.base.theme);
                        let text = if on { TStyle::default() } else { style::dim() };
                        let mark = if focus.is_some() && on { "▸" } else { " " };
                        let line = Line::from(vec![
                            Span::raw(mark),
                            swatch(colour),
                            Span::styled(legend_text(comm, c, sizes[c]), text),
                        ]);
                        (line, Some(Pick::Community(c)))
                    })
                    .collect();
                if shown < order.len() {
                    if let Some(last) = out.last_mut() {
                        *last = (
                            Line::raw(format!("  … {} more", order.len() - shown + 1)),
                            None,
                        );
                    }
                }
                out
            }
            Layer::Entropy => untagged(ramp_legend(
                self.layer.ramp(self.base.theme),
                &self.layer.legend_title(),
                "1",
            )),
            // Drawn through `self.gene`, handled above.
            Layer::Gene => Vec::new(),
            Layer::Community(c) => {
                let title = format!("{}  c/C", self.layer.legend_title());
                let mut out = ramp_legend(self.layer.ramp(self.base.theme), &title, "1");
                let size = comm.sizes.get(c).copied().unwrap_or(0);
                out.push(Line::from(vec![
                    Span::raw(" "),
                    swatch(level.palette[c]),
                    Span::raw(format!(" argmax {} cells", thousands(size))),
                ]));
                untagged(out)
            }
        }
    }
}

/// A legend row's text: the name and cell count, fitted to the panel.
fn legend_text(comm: &super::data::Communities, c: usize, n: usize) -> String {
    let name: String = comm.name(c).chars().take(17).collect();
    let width = if comm.names.is_some() { 17 } else { 4 };
    format!(" {name:<width$} {:>9}", thousands(n))
}

/// Right-click or a held Ctrl/Alt adds to the selection instead of
/// replacing it. (Shift-click is kept by terminals for text selection.)
fn adds(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
}

/// What a click in the panel picks.
#[derive(Clone, Copy, Debug)]
enum Pick {
    Community(usize),
    /// Index into the shown community's markers.
    Gene(usize),
    /// A community of the structure plot's bars.
    Bar(usize),
}

/// A colour ramp from 0 to `top`, under `title`.
fn ramp_legend(ramp: &Ramp, title: &str, top: &str) -> Vec<Line<'static>> {
    let bar: Vec<Span> = (0..24)
        .map(|i| Span::styled("█", TStyle::default().fg(rgb(ramp.at(i as f32 / 23.)))))
        .collect();
    let mut first = vec![Span::raw(" 0 ")];
    first.extend(bar);
    first.push(Span::raw(format!(" {top}")));
    vec![Line::raw(format!(" {title}")), Line::from(first)]
}

fn help_lines(full: bool, relabel: bool) -> Vec<Line<'static>> {
    let dim = style::dim();
    let text: &[&str] = if relabel && !full {
        annotate::RELABEL_HELP
    } else if full {
        &[
            " arrows/drag  pan (shift: far)",
            " z/Z +/- or wheel  zoom",
            " < >  smaller/larger points",
            " 0 fit   r back to the start",
            " b next batch  w every batch  e edges",
            " grid: drag or shift+arrows move",
            " grid: zoom, < > one batch; alt all",
            " 1 argmax 2 soft 3 entropy 4 focused",
            " ] [  focus next/prev group",
            " l/L  next/prev level (L1 .. final)",
            " click cell/legend  show group",
            " right/ctrl-click   add to shown",
            " Esc or x  back (chart, gene, focus)",
            " click marker / g G  map a gene",
            " o  observed / model-expected",
            " p  gene ramp top: p99 / p95",
            " c/C  communities / types / clusters",
            " , .  prev/next lupin round",
            " H  structure plot → heatmap → map",
            " s save  f saved figures  q quit",
            " A annotate (lupin)  R relabel",
        ]
    } else {
        &[" s save view  r start over", " Esc back  ? keys  q quit"]
    };
    text.iter().map(|t| Line::styled(*t, dim)).collect()
}

/// `x` stepped by the factor `by`, within `--point-size`'s range, snapped
/// to 1 on the way through so the default is one key away.
fn step_point(x: f32, by: f32) -> f32 {
    let next = (x * by).clamp(*super::POINT_SIZES.start(), *super::POINT_SIZES.end());
    if (next - 1.).abs() < 0.05 {
        1.
    } else {
        next
    }
}

/// The first `{prefix}-NNN` without a `.{ext}` file yet.
fn free_stem(prefix: &str, ext: &str) -> String {
    (1..)
        .map(|n| format!("{prefix}-{n:03}"))
        .find(|s| !std::path::Path::new(&format!("{s}.{ext}")).exists())
        .expect("unbounded")
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
