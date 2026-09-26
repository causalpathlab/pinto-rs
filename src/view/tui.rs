//! Interactive terminal view: the map on the left, a side panel on the right.
//!
//! The map is drawn by [`render`] at the pixel size of its cells and shown
//! through the terminal's graphics protocol: our own kitty transport
//! ([`kitty`]), ratatui-image for sixel/iTerm2, or half-block characters
//! (two pixels per cell) anywhere else.
//!
//! Input is drained before each redraw, so a burst of scroll or drag events
//! costs one frame, not one per event.

use super::color::{Ramp, Rgb, DIMMED};
use super::data::Rect as WorldRect;
use super::data::NO_CLUSTER;
use super::kitty::{Kitty, Transport};
use super::render::{self, Frame, Layer, Mode, Style, Viewport};
use super::scalebar;
use super::{
    focused, ids, legend_line, markers, thousands, write_outputs, Base, Level, ViewArgs, FIT_MARGIN,
};
use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style as TStyle};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::DefaultTerminal;
use ratatui_image::picker::{Picker, ProtocolType};
use ratatui_image::protocol::Protocol;
use ratatui_image::{Image, Resize};
use std::time::{Duration, Instant};

/// Side panel width, cells.
const PANEL: u16 = 34;

/// Zoom step per key press or wheel notch.
const ZOOM: f32 = 1.25;

pub fn run(args: &ViewArgs) -> anyhow::Result<()> {
    eprintln!("loading {} ...", args.prefix);
    let base = Base::load(&args.prefix, &args.units)?;
    let first = base.run.level_index(args.level.as_deref())?;
    let level = base.level(first, args.edges)?;

    let mut terminal = ratatui::init();
    let result = (|| {
        let gfx = Gfx::pick(&args.graphics)?;
        execute!(std::io::stdout(), EnableMouseCapture)?;
        let mut app = App::new(&base, level, args, gfx);
        app.run(&mut terminal)
    })();
    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
    result
}

/// How the map reaches the screen.
enum Gfx {
    Kitty(Kitty),
    /// Sixel or iTerm2 through ratatui-image.
    Picker(Picker, Option<Protocol>),
    /// Half-block characters.
    Blocks(Option<Frame>),
}

impl Gfx {
    fn pick(choice: &str) -> anyhow::Result<(Self, (f32, f32))> {
        let picker = Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks());
        let font = picker.font_size();
        let px = (font.width.max(1) as f32, font.height.max(1) as f32);
        let detected = picker.protocol_type();
        let kind = match choice {
            "auto" => detected,
            "kitty" | "kitty-inline" => ProtocolType::Kitty,
            "sixel" => ProtocolType::Sixel,
            "iterm2" => ProtocolType::Iterm2,
            "blocks" => ProtocolType::Halfblocks,
            other => anyhow::bail!("unknown --graphics {other:?}"),
        };
        Ok(match kind {
            ProtocolType::Kitty => {
                let transport = if choice == "kitty-inline" {
                    Transport::Direct
                } else {
                    Transport::detect()
                };
                (Gfx::Kitty(Kitty::new(transport)), px)
            }
            ProtocolType::Halfblocks => (Gfx::Blocks(None), (1., 2.)),
            other => {
                let mut picker = picker;
                picker.set_protocol_type(other);
                (Gfx::Picker(picker, None), px)
            }
        })
    }

    fn name(&self) -> String {
        match self {
            Gfx::Kitty(k) => match k.transport() {
                Transport::File => "kitty (file)".into(),
                Transport::Direct => "kitty (inline)".into(),
            },
            Gfx::Picker(p, _) => format!("{:?}", p.protocol_type()).to_lowercase(),
            Gfx::Blocks(_) => "half-blocks".into(),
        }
    }
}

struct App<'a> {
    base: &'a Base,
    args: &'a ViewArgs,
    levels: Vec<Option<Level>>,
    cur: usize,
    gfx: Gfx,
    /// Image pixels per terminal cell.
    px_per_cell: (f32, f32),

    center: (f32, f32),
    /// World units per image pixel.
    upp: f32,
    layer: Layer,
    community: usize,
    edges: bool,
    /// Batch tile last jumped to.
    tile: Option<usize>,

    /// Communities shown in colour, one flag each; all false shows every one.
    focus: Vec<bool>,
    /// Community whose markers the panel lists, and those markers.
    shown: Option<usize>,
    markers: Vec<(String, f32)>,
    /// Cell last clicked.
    picked: Option<usize>,

    map: Rect,
    side: Rect,
    /// Panel row of each legend entry, for clicks.
    legend_rows: Vec<(u16, usize)>,
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
            gfx: gfx.0,
            px_per_cell: gfx.1,
            center: (0., 0.),
            upp: 0.,
            layer: args.layer,
            community,
            edges: args.edges,
            tile: None,
            focus: vec![false; k],
            shown: None,
            markers: Vec::new(),
            picked: None,
            map: Rect::default(),
            side: Rect::default(),
            legend_rows: Vec::new(),
            cursor: None,
            drag: None,
            dragged: false,
            help: false,
            status: String::new(),
            mode: Mode::Points,
            render_time: Duration::ZERO,
            send_time: Duration::ZERO,
            send_bytes: 0,
            need_map: true,
            need_panel: true,
            quit: false,
        }
    }

    fn level(&self) -> &Level {
        self.levels[self.cur]
            .as_ref()
            .expect("current level is loaded")
    }

    fn level_mut(&mut self) -> &mut Level {
        self.levels[self.cur]
            .as_mut()
            .expect("current level is loaded")
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        self.layout(terminal)?;
        self.fit(self.base.geom.bounds());
        while !self.quit {
            if self.need_map || self.need_panel {
                self.redraw(terminal)?;
            }
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
        let panel = PANEL.min(area.width / 2);
        let [map, side] =
            Layout::horizontal([Constraint::Min(1), Constraint::Length(panel)]).areas(area);
        if map != self.map {
            self.map = map;
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
        Viewport {
            x0: self.center.0 - 0.5 * w as f32 * self.upp,
            y0: self.center.1 - 0.5 * h as f32 * self.upp,
            upp: self.upp,
            w,
            h,
        }
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

    /// World point under terminal cell `(col, row)`.
    fn world_at(&self, col: u16, row: u16) -> (f32, f32) {
        let vp = self.viewport();
        let px = (col.saturating_sub(self.map.x) as f32 + 0.5) * self.px_per_cell.0;
        let py = (row.saturating_sub(self.map.y) as f32 + 0.5) * self.px_per_cell.1;
        (vp.x0 + px * vp.upp, vp.y0 + py * vp.upp)
    }

    fn in_map(&self, col: u16, row: u16) -> bool {
        self.map.contains(Position::new(col, row))
    }

    /// Scale by `f` (> 1 zooms out) keeping `anchor` fixed on screen.
    fn zoom(&mut self, f: f32, anchor: Option<(u16, u16)>) {
        let b = self.base.geom.bounds();
        let (w, h) = self.image_size();
        let max = Viewport::fit(b, w, h).upp * 8.;
        let min = self.base.spacing / 80.;
        let upp = (self.upp * f).clamp(min, max);
        let f = upp / self.upp;
        let (ax, ay) = anchor.map_or(self.center, |(c, r)| self.world_at(c, r));
        self.center = (ax + (self.center.0 - ax) * f, ay + (self.center.1 - ay) * f);
        self.upp = upp;
        self.need_map = true;
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
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => self.quit = true,
            KeyCode::Char('h' | 'H') | KeyCode::Left => self.pan(-step, 0.),
            KeyCode::Char('l' | 'L') | KeyCode::Right => self.pan(step, 0.),
            KeyCode::Char('k' | 'K') | KeyCode::Up => self.pan(0., -step),
            KeyCode::Char('j' | 'J') | KeyCode::Down => self.pan(0., step),
            KeyCode::Char('+' | '=') => self.zoom(1. / ZOOM, None),
            KeyCode::Char('-' | '_') => self.zoom(ZOOM, None),
            KeyCode::Char('0') => {
                self.tile = None;
                self.fit(self.base.geom.bounds());
            }
            KeyCode::Char('1') => self.set_layer(Layer::Argmax),
            KeyCode::Char('2') => self.set_layer(Layer::Soft),
            KeyCode::Char('3') => self.set_layer(Layer::Entropy),
            KeyCode::Char('4') => self.set_layer(Layer::Community(self.community)),
            KeyCode::Char('c') => self.step_community(1),
            KeyCode::Char('C') => self.step_community(-1),
            KeyCode::Char(']') => self.step_level(1, terminal)?,
            KeyCode::Char('[') => self.step_level(-1, terminal)?,
            KeyCode::Char('e') => self.toggle_edges(terminal)?,
            KeyCode::Char('b') => self.next_tile(),
            KeyCode::Char('s') => self.export()?,
            KeyCode::Char('x') => self.clear_focus(),
            KeyCode::Char('?') => self.help = !self.help,
            _ => {}
        }
        Ok(())
    }

    fn mouse(&mut self, m: MouseEvent) {
        let (col, row) = (m.column, m.row);

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
                if let Some(&(_, c)) = self.legend_rows.iter().find(|(r, _)| *r == row) {
                    self.select(c, add);
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
                    self.select(c as usize, add);
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
        self.markers.clear();
        if let Some(shown) = self.shown {
            let base = self.base;
            let level = self.level_mut();
            match base.load_features(level) {
                Ok(()) => {
                    let rates = level.features.as_ref().expect("just loaded");
                    self.markers = rates
                        .top(shown, 12)
                        .into_iter()
                        .map(|m| (markers::symbol(&m.name).to_string(), m.fold))
                        .collect();
                }
                Err(e) => self.status = format!("no markers: {e}"),
            }
        }
        self.need_map = true;
    }

    fn clear_focus(&mut self) {
        self.focus.fill(false);
        self.shown = None;
        self.markers.clear();
        self.picked = None;
        self.need_map = true;
    }

    fn focus(&self) -> Option<&[bool]> {
        self.focus.iter().any(|&f| f).then_some(&self.focus[..])
    }

    fn set_layer(&mut self, layer: Layer) {
        if layer == Layer::Entropy && self.level().comm.entropy.is_none() {
            self.status = "this level has no entropy".into();
            return;
        }
        self.layer = layer;
        self.need_map = true;
    }

    fn step_community(&mut self, by: isize) {
        let k = self.level().comm.k as isize;
        if k == 0 {
            return;
        }
        self.community = (self.community as isize + by).rem_euclid(k) as usize;
        self.set_layer(Layer::Community(self.community));
    }

    fn step_level(&mut self, by: isize, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let n = self.levels.len() as isize;
        let next = (self.cur as isize + by).clamp(0, n - 1) as usize;
        if next == self.cur {
            return Ok(());
        }
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
        self.community = self.community.min(k.saturating_sub(1));
        if let Layer::Community(_) = self.layer {
            self.layer = Layer::Community(self.community);
        }
        self.need_map = true;
        Ok(())
    }

    fn toggle_edges(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        self.edges = !self.edges;
        if self.edges && self.level().edges.is_none() {
            self.busy(terminal, "loading edges ...".into())?;
            let base = self.base;
            base.load_edges(self.level_mut())?;
        }
        if self.edges
            && render::mode(&self.base.scene(self.level()), &self.viewport()) != Mode::Points
        {
            self.status = "edges show once zoomed in".into();
        }
        self.need_map = true;
        Ok(())
    }

    fn next_tile(&mut self) {
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
        let stem = (1..)
            .map(|n| format!("pinto-view-{n:03}"))
            .find(|s| !std::path::Path::new(&format!("{s}.png")).exists())
            .expect("unbounded");
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
        let style = Style {
            scale_bar: None,
            ..self.style()
        };
        write_outputs(
            self.base,
            self.level(),
            &style,
            &hi,
            Some(png.as_ref()),
            Some(pdf.as_ref()),
        )?;
        std::fs::write(&txt, self.export_notes(&png, &pdf, &hi, window))?;
        self.status = format!("saved {stem}.png .pdf .txt");
        Ok(())
    }

    /// `window` is the `--bbox` that `vp` was fitted from.
    fn export_notes(&self, png: &str, pdf: &str, vp: &Viewport, window: WorldRect) -> String {
        use std::fmt::Write as _;
        let level = self.level();
        let comm = &level.comm;
        let focused = focused(self.focus());

        let win = window;
        let mut cmd = format!(
            // `{}` prints the shortest text that parses back to the same f32.
            "pinto view {} --png {png} --pdf {pdf} --units {} --width {} --height {} \
             --bbox {},{},{},{} --level {} --layer {}",
            self.args.prefix,
            self.args.units,
            vp.w,
            vp.h,
            win.x0,
            win.y0,
            win.x1,
            win.y1,
            comm.tag,
            self.layer,
        );
        if !focused.is_empty() {
            write!(cmd, " --focus {}", ids(&focused, ",")).ok();
        }
        if self.edges {
            cmd.push_str(" --edges");
        }

        let mut out = String::new();
        writeln!(out, "# pinto view export").ok();
        writeln!(out, "image    {png} ({}×{}), {pdf}", vp.w, vp.h).ok();
        writeln!(out, "redraw   {cmd}").ok();
        writeln!(out, "run      {}", self.base.run.manifest.display()).ok();
        writeln!(out, "level    {} (K={})", comm.tag, comm.k).ok();
        writeln!(out, "layer    {}", self.layer).ok();
        writeln!(out, "scale    {:.4} units/px", vp.upp).ok();
        if !focused.is_empty() {
            writeln!(out, "focus    {}", ids(&focused, " ")).ok();
        }

        writeln!(out, "\nlegend").ok();
        for &c in &comm.by_size {
            writeln!(out, "{}", legend_line(c, level.palette[c], comm.sizes[c])).ok();
        }
        if let (Some(rates), false) = (level.features.as_ref(), focused.is_empty()) {
            writeln!(out, "\nmarkers (fold over the other communities)").ok();
            for &c in &focused {
                writeln!(out, "  C{c:<3} {}", rates.summary(c, 20)).ok();
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

    fn style(&self) -> Style<'_> {
        Style {
            layer: self.layer,
            edges: self.edges,
            focus: self.focus(),
            scale_bar: self.base.units,
        }
    }

    // ── drawing ─────────────────────────────────────────────────────────

    fn redraw(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        self.need_panel = false;
        let side = self.layout(terminal)?;
        let mut fresh: Option<Frame> = None;
        if self.need_map {
            self.need_map = false;
            let vp = self.viewport();
            let level = self.level();
            let scene = self.base.scene(level);
            let t = Instant::now();
            let frame = render::render(&scene, &vp, &self.style(), &level.palette);
            let took = t.elapsed();
            let mode = render::mode(&scene, &vp);
            self.render_time = took;
            self.mode = mode;
            match &mut self.gfx {
                Gfx::Kitty(_) => fresh = Some(frame),
                Gfx::Blocks(slot) => *slot = Some(frame),
                Gfx::Picker(picker, proto) => {
                    let t = Instant::now();
                    let img =
                        image::RgbaImage::from_raw(frame.w as u32, frame.h as u32, frame.rgba)
                            .expect("frame buffer matches its size");
                    *proto = Some(picker.new_protocol(
                        image::DynamicImage::ImageRgba8(img),
                        self.map.as_size(),
                        Resize::Fit(None),
                    )?);
                    self.send_time = t.elapsed();
                }
            }
        }

        let (panel, legend_rows) = self.panel(side.height);
        self.legend_rows = legend_rows
            .into_iter()
            .map(|(line, c)| (side.y + line as u16, c))
            .collect();
        let map = self.map;
        let gfx = &self.gfx;
        terminal.draw(|f| {
            f.render_widget(Paragraph::new(panel), side);
            match gfx {
                Gfx::Picker(_, Some(proto)) => f.render_widget(Image::new(proto), map),
                Gfx::Blocks(Some(frame)) => f.render_widget(HalfBlocks(frame), map),
                _ => {}
            }
        })?;

        if let (Gfx::Kitty(kitty), Some(frame)) = (&mut self.gfx, fresh) {
            let t = Instant::now();
            kitty.show(
                &mut std::io::stdout(),
                &frame,
                (map.x, map.y),
                (map.width, map.height),
            )?;
            self.send_time = t.elapsed();
            self.send_bytes = kitty.last_bytes;
        }
        Ok(())
    }

    /// Panel lines, and which of them are legend entries for which community.
    fn panel(&self, height: u16) -> (Vec<Line<'static>>, Vec<(usize, usize)>) {
        let level = self.level();
        let comm = &level.comm;
        let dim = TStyle::default().fg(Color::DarkGray);
        let bold = TStyle::default().add_modifier(Modifier::BOLD);
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
                format!("{} ({}/{})  [ ]", comm.tag, self.cur + 1, self.levels.len()),
            ),
            row("layer", format!("{}  1-4", self.layer)),
            row(
                "cells",
                format!("{}  K={}", thousands(self.base.geom.n()), comm.k),
            ),
            row("scale", format!("{:.3} /px  {}", self.upp, self.mode)),
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
            TStyle::default().fg(Color::Yellow),
        ));

        let help = help_lines(self.help);
        let free = (height as usize).saturating_sub(lines.len() + help.len() + 1);
        if let Some(c) = self.shown {
            // Leave the legend at least a few rows.
            lines.extend(self.marker_lines(c, free.saturating_sub(6).min(12)));
        }
        let room = (height as usize).saturating_sub(lines.len() + help.len() + 1);
        lines.push(Line::raw(""));
        let mut legend_rows = Vec::new();
        for (line, c) in self.legend(room) {
            if let Some(c) = c {
                legend_rows.push((lines.len(), c));
            }
            lines.push(line);
        }
        let pad = (height as usize).saturating_sub(lines.len() + help.len());
        lines.extend(std::iter::repeat_n(Line::raw(""), pad));
        lines.extend(help);
        (lines, legend_rows)
    }

    /// The picked cell: name, then its strongest communities and entropy.
    fn cell_lines(&self, i: usize) -> Vec<Line<'static>> {
        let comm = &self.level().comm;
        let dim = TStyle::default().fg(Color::DarkGray);
        let k = comm.k;
        let mut top: Vec<(usize, u8)> = (0..k).map(|c| (c, comm.prop[i * k + c])).collect();
        top.sort_by_key(|&(_, q)| std::cmp::Reverse(q));
        let mut mix: Vec<String> = top
            .iter()
            .take(3)
            .filter(|&&(_, q)| q > 0)
            .map(|&(c, q)| format!("C{c} {:.2}", q as f32 / 255.))
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

    /// Up to `n` of the shown community `c`'s markers.
    fn marker_lines(&self, c: usize, n: usize) -> Vec<Line<'static>> {
        let level = self.level();
        if self.markers.is_empty() {
            return Vec::new();
        }
        let bold = TStyle::default().add_modifier(Modifier::BOLD);
        let dim = TStyle::default().fg(Color::DarkGray);
        let mut out = vec![
            Line::raw(""),
            Line::from(vec![
                Span::raw(" "),
                Span::styled("██", TStyle::default().fg(rgb(level.palette[c]))),
                Span::styled(format!(" C{c} markers"), bold),
                Span::styled("  fold", dim),
            ]),
        ];
        out.extend(self.markers.iter().take(n).map(|(gene, fold)| {
            let name: String = gene.chars().take(20).collect();
            Line::raw(format!("   {name:<20} ×{fold:<6.1}"))
        }));
        out
    }

    /// Legend lines, each tagged with its community when it names one.
    fn legend(&self, room: usize) -> Vec<(Line<'static>, Option<usize>)> {
        let level = self.level();
        let comm = &level.comm;
        let swatch = |c: Rgb| Span::styled("██", TStyle::default().fg(rgb(c)));
        let untagged = |lines: Vec<Line<'static>>| lines.into_iter().map(|l| (l, None)).collect();
        match self.layer {
            Layer::Argmax | Layer::Soft => {
                let (order, sizes) = (&comm.by_size, &comm.sizes);
                let focus = self.focus();
                let shown = order.len().min(room);
                let mut out: Vec<(Line, Option<usize>)> = order[..shown]
                    .iter()
                    .map(|&c| {
                        let on = focus.is_none_or(|f| f[c]);
                        let text =
                            TStyle::default().fg(if on { Color::Reset } else { Color::DarkGray });
                        let mark = if focus.is_some() && on { "▸" } else { " " };
                        let line = Line::from(vec![
                            Span::raw(mark),
                            if on {
                                swatch(level.palette[c])
                            } else {
                                swatch(DIMMED)
                            },
                            Span::styled(format!(" C{c:<3} {:>9}", thousands(sizes[c])), text),
                        ]);
                        (line, Some(c))
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
            Layer::Entropy => untagged(ramp_legend(self.layer)),
            Layer::Community(c) => {
                let mut out = ramp_legend(self.layer);
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

/// Right-click or a held Ctrl/Alt adds to the selection instead of
/// replacing it. (Shift-click is kept by terminals for text selection.)
fn adds(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
}

fn ramp_legend(layer: Layer) -> Vec<Line<'static>> {
    let ramp: &Ramp = layer.ramp();
    let mut title = layer.legend_title();
    if let Layer::Community(_) = layer {
        title.push_str("  c/C");
    }
    let bar: Vec<Span> = (0..24)
        .map(|i| Span::styled("█", TStyle::default().fg(rgb(ramp.at(i as f32 / 23.)))))
        .collect();
    let mut first = vec![Span::raw(" 0 ")];
    first.extend(bar);
    first.push(Span::raw(" 1"));
    vec![Line::raw(format!(" {title}")), Line::from(first)]
}

fn help_lines(full: bool) -> Vec<Line<'static>> {
    let dim = TStyle::default().fg(Color::DarkGray);
    let text: &[&str] = if full {
        &[
            " hjkl/arrows/drag  pan (shift: far)",
            " +/- or wheel      zoom",
            " 0 fit   b next batch",
            " 1 argmax 2 soft 3 entropy 4 Ck",
            " c/C  next/prev community",
            " [ ]  prev/next level (L1 .. final)",
            " click cell/legend  show community",
            " right/ctrl-click   add to shown",
            " x  show all",
            " e edges  q quit",
            " s export view (PNG, PDF, .txt)",
        ]
    } else {
        &[" s export view (PNG, PDF, .txt)", " ? keys   q quit"]
    };
    text.iter().map(|t| Line::styled(*t, dim)).collect()
}

fn rgb(c: Rgb) -> Color {
    Color::Rgb(c[0], c[1], c[2])
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// A frame as `▀` cells: foreground is the upper pixel, background the lower.
struct HalfBlocks<'f>(&'f Frame);

impl Widget for HalfBlocks<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let f = self.0;
        let px = |x: usize, y: usize| -> Color {
            if x >= f.w || y >= f.h {
                return Color::Reset;
            }
            let o = f.offset(x, y);
            Color::Rgb(f.rgba[o], f.rgba[o + 1], f.rgba[o + 2])
        };
        for r in 0..area.height {
            for c in 0..area.width {
                if let Some(cell) = buf.cell_mut(Position::new(area.x + c, area.y + r)) {
                    let (x, y) = (c as usize, 2 * r as usize);
                    cell.set_char('▀').set_fg(px(x, y)).set_bg(px(x, y + 1));
                }
            }
        }
    }
}
