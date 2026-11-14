//! Interactive terminal view: the map on the left, a side panel on the right.
//!
//! The map is drawn by [`render`] at the pixel size of its cells and shown
//! through the terminal's graphics protocol: legume-tui's kitty transport,
//! ratatui-image for sixel/iTerm2, or legume-tui's coloured block
//! characters anywhere else.
//!
//! Input is drained before each redraw, so a burst of scroll or drag events
//! costs one frame, not one per event.

mod actions;
mod annotate;
mod browse;
mod draw;
mod follow;
mod gallery;
pub(super) mod grid;
mod input;
mod lra;
mod plots;
mod relabel;
mod save;

use draw::{legend_text, ramp_legend};

pub use browse::pick_run;
pub(super) use follow::pick_server;

use legume_tui::map::viewport::Limits;

use super::color::{Rgb, Theme};
use super::data::Rect as WorldRect;
use super::gene::{GeneMap, Source};
use super::lupin::Job;
use super::render::{self, Frame, Layer, Mode, Viewport};
use super::round::Round;
use super::{
    focus_name, legend_line, thousands, value_name, Base, Level, Show, ViewArgs, FIT_MARGIN,
};
use legume_tui::chrome::rgb;
use legume_tui::map::kitty::cellart::{Cells, Glyphs};
use legume_tui::map::kitty::{Kitty, MapGraphics, Resolved};
use ratatui::crossterm::event::{self, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::widgets::Paragraph;
use ratatui::DefaultTerminal;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::Protocol;
use std::path::PathBuf;
use std::time::Duration;

/// Side panel width, cells.
const PANEL: u16 = 34;

/// Side panel width while relabelling or in a dialog, cells.
const WIDE_PANEL: u16 = 46;

/// Narrowest the side panel is dragged to, and the map left beside it.
const MIN_PANEL: u16 = 24;
const MIN_MAP: u16 = 20;

/// Zoom step per key press or wheel notch.
const ZOOM: f32 = 1.25;

/// Point size step per `<` or `>`.
const POINT_STEP: f32 = 1.25;

/// Load the run and show it on `session`'s `terminal` until the user
/// quits, or until `pinto kill` asks the view to stop.
pub fn run(
    session: &legume_tui::Session,
    terminal: &mut DefaultTerminal,
    args: &ViewArgs,
    follow: Option<&legume_mcp::watch::Live>,
) -> anyhow::Result<()> {
    // The terminal is fresh: a repaint asked for by now is this one. It
    // is asked first, before the mouse is on, so no report lands amid its
    // answer: its background picks the theme the palette is built in.
    session.take_repaint();
    let (gfx, px, background) = Gfx::pick(session, args.graphics);
    let theme = args
        .theme
        .or(background.map(Theme::for_background))
        .unwrap_or(Theme::Dark);
    session.draw(terminal, |f| {
        let msg = format!(" loading {} ...", args.prefix());
        f.render_widget(Paragraph::new(msg), f.area());
    })?;
    let base = Base::load(args.prefix(), &args.units, theme)?;
    let first = base.run.level_index(args.level.as_deref())?;
    let level = base.level(first, args.edges)?;

    // The keyboard was asked with the graphics; the session turns the
    // mouse on again after every attach.
    session.set_modes(crate::tui::modes(legume_tui::Mouse::Full))?;
    let mut app = App::new(&base, level, args, (gfx, px), session);
    app.shift_enter = session.keyboard_enhanced();
    app.open_rounds(args.round.as_deref().map(PathBuf::from));
    app.follower = follow.map(follow::Follower::open).transpose()?;
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
    /// The drawing method for `choice` on the terminal `session` is
    /// attached to now, its frame pixels per cell, and the terminal's
    /// background colour when it reports one. The terminal is asked once,
    /// for block characters too (their cell shape, and the background).
    fn pick(session: &legume_tui::Session, choice: MapGraphics) -> (Self, (f32, f32), Option<Rgb>) {
        use legume_tui::graphics::{self, Fallback};
        let (picker, answer) = graphics::pick_protocol(session, choice.protocol(), Fallback::Probe);
        let background = answer.and_then(|a| a.background).map(|(r, g, b)| [r, g, b]);
        let resolved = choice.resolve(
            graphics::in_tmux(session),
            picker.protocol_type(),
            graphics::kitty_transport(session),
        );
        let px = resolved.pixels_per_cell(graphics::cell_size(&picker, answer));
        let gfx = match resolved {
            Resolved::Kitty { transport, tmux } => {
                Gfx::Kitty(Kitty::new(transport).named("pinto").through_tmux(tmux))
            }
            Resolved::Picker(p) => {
                debug_assert_eq!(
                    p,
                    picker.protocol_type(),
                    "the picker holds what was chosen"
                );
                Gfx::Picker(picker, None)
            }
            Resolved::Cells(glyphs) => Gfx::Cells(glyphs, None),
        };
        (gfx, px, background)
    }

    /// Whether the map reaches the screen as an image, so small text burnt
    /// into it (the scale bar's label) stays legible; block characters
    /// would smear it.
    fn shows_images(&self) -> bool {
        !matches!(self, Gfx::Cells(..))
    }

    fn name(&self) -> &'static str {
        match self {
            Gfx::Kitty(k) => Resolved::Kitty {
                transport: k.transport(),
                tmux: k.via_tmux(),
            },
            Gfx::Picker(p, _) => Resolved::Picker(p.protocol_type()),
            Gfx::Cells(glyphs, _) => Resolved::Cells(*glyphs),
        }
        .name()
    }
}

struct App<'a> {
    base: &'a Base,
    args: &'a ViewArgs,
    /// The session the view runs in: whether a terminal is there to draw
    /// on, and what it is.
    session: &'a legume_tui::Session,
    levels: Vec<Option<Level>>,
    cur: usize,
    /// The level the viewer opened on, for `r`.
    home: usize,
    gfx: Gfx,
    /// Image pixels per terminal cell.
    px_per_cell: (f32, f32),

    /// Where the map looks; its frame size is the map's at each use.
    cam: render::MapView,
    layer: Layer,
    community: usize,
    edges: bool,
    /// Cell disc size, × the default (`<`/`>`).
    point: f32,
    /// Batch tile last jumped to.
    tile: Option<usize>,
    /// The theme `--theme` or the theme key chose; `None` follows the
    /// terminal's background.
    theme: Option<Theme>,
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
    /// Whether the terminal tells shift-enter from enter.
    shift_enter: bool,
    status: String,
    /// An error shown in a popup over the map until a key or click.
    error: Option<String>,
    /// Side panel widths dragged by hand: the usual panel, then the wide
    /// one for annotation and dialogs.
    panel_w: [Option<u16>; 2],
    /// Whether the panel's left edge is being dragged.
    resizing: bool,

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
    relabel: Option<relabel::Relabel>,
    /// A dialog that takes the keys: file browser, prompt, confirmation.
    modal: Option<annotate::Modal>,
    /// The agent followed (`--follow`).
    follower: Option<follow::Follower>,

    /// The lra layer (`I`): the run's contact-effect tables, their pairs,
    /// and the pair on the map; read the first time it opens.
    lra: Option<super::lra::LraView>,
    /// Whether the layer is open: the panel lists pairs, the map draws the
    /// selected one.
    lra_on: bool,
    /// The list's cursor, a position among the pairs shown.
    lra_cursor: usize,
    /// The selected pair's curve under the map (`c` in the layer).
    lra_curve: bool,
    /// The gene search being typed (`/` in the layer).
    lra_search: Option<legume_tui::save::Field>,
    /// A `pinto lra` started from the layer.
    lra_job: Option<lra::LraJob>,
    /// The pair selected is to be drawn with the next map.
    pair_pending: bool,

    /// The map must be rendered again.
    need_map: bool,
    /// The panel must be drawn again (always, when the map is).
    need_panel: bool,
    quit: bool,
}

impl<'a> App<'a> {
    fn new(
        base: &'a Base,
        level: Level,
        args: &'a ViewArgs,
        gfx: (Gfx, (f32, f32)),
        session: &'a legume_tui::Session,
    ) -> Self {
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
            session,
            levels,
            cur,
            home: cur,
            gfx: gfx.0,
            px_per_cell: gfx.1,
            // Framed by the fit below, once the map has a size.
            cam: Viewport::at((0., 0.), 1., 1, 1).camera(),
            layer: args.layer,
            community,
            edges: args.edges,
            point: args.point_size,
            tile: None,
            theme: args.theme,
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
            shift_enter: false,
            status: String::new(),
            error: None,
            panel_w: [None; 2],
            resizing: false,
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
            gallery: super::saved::here(),
            show_saved: true,
            strip: Rect::default(),
            thumbs: Default::default(),
            follower: None,
            lra: None,
            lra_on: false,
            lra_cursor: 0,
            lra_curve: false,
            lra_search: None,
            lra_job: None,
            pair_pending: false,
            need_map: true,
            need_panel: true,
            quit: false,
        }
    }

    /// What the map draws: the current level, or a grouping of the round.
    fn level(&self) -> &Level {
        super::shown(&self.levels, self.round.as_ref(), self.show, self.cur)
    }

    fn level_mut(&mut self) -> &mut Level {
        super::shown_mut(&mut self.levels, self.round.as_mut(), self.show, self.cur)
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        self.layout(terminal)?;
        self.fit(self.base.geom.bounds());
        let (session, choice) = (self.session, self.args.graphics);
        while !self.quit {
            // Detached, nothing is drawn or sent (kitty's images among it)
            // and lupin is followed; the map waits for the next attach.
            if session.attached() && (self.need_map || self.need_panel) {
                self.redraw(terminal)?;
            }
            self.poll_job(terminal)?;
            self.poll_lra();
            self.follow_tick(Some(terminal))?;
            session.status(&self.session_status());
            // After an attach the terminal is asked again, before the
            // mouse is on, and the pictures made anew for it.
            let mut repicked = None;
            let round = session.poll_with(terminal, Duration::from_millis(250), || {
                repicked = Some(Gfx::pick(session, choice));
                Ok(())
            })?;
            match round {
                legume_tui::Round::Event(ev) => {
                    self.handle(ev, terminal)?;
                    while !self.quit && event::poll(Duration::ZERO)? {
                        self.handle(event::read()?, terminal)?;
                    }
                }
                legume_tui::Round::Repainted => {
                    if let Some((gfx, px, background)) = repicked {
                        self.repick(gfx, px, background);
                    }
                    self.shift_enter = session.keyboard_enhanced();
                }
                // `pinto kill` stops the view as q does.
                legume_tui::Round::Stopped => self.quit = true,
                _ => {}
            }
        }
        self.end_job(terminal)?;
        if let Some(job) = self.lra_job.take() {
            job.stop();
        }
        if let Gfx::Kitty(k) = &mut self.gfx {
            // Detached, only its files go: there is no screen to clear.
            if session.attached() {
                k.clear(&mut std::io::stdout())?;
            } else {
                k.remove_files();
            }
        }
        Ok(())
    }

    /// On the way out, by `q`, ctrl+c or `pinto kill`: a `relabel --next`
    /// writing the next round is let finish, so the decisions sent are
    /// saved whole; an annotation or a preview is stopped (lupin writes a
    /// round aside and moves it in place last, so none is left half made).
    fn end_job(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let Some(job) = self.job.take() else {
            return Ok(());
        };
        if !job.quit() {
            self.busy(Some(terminal), "finishing lupin's next round ...".into())?;
        }
        job.wait();
        Ok(())
    }

    /// Take `gfx`, drawing `px` frame pixels per cell, for the terminal
    /// attached now. Every picture made for the last one (the map's
    /// protocol or cells, kitty's images, the saved figures' thumbnails)
    /// is made again; the map frames what it framed. Its `background`,
    /// when it says, picks the theme unless `--theme` or the theme key did.
    fn repick(&mut self, gfx: Gfx, px: (f32, f32), background: Option<Rgb>) {
        self.cam.scale *= px.0.max(1.) / self.px_per_cell.0;
        self.gfx = gfx;
        self.px_per_cell = px;
        self.thumbs.clear();
        if let (None, Some(bg)) = (self.theme, background) {
            self.retheme(Theme::for_background(bg));
        }
        self.need_map = true;
        self.need_panel = true;
    }

    /// `t`: the other theme, dark or light (for paper), kept from here on
    /// whatever the terminal's background; saved figures are drawn in it.
    fn toggle_theme(&mut self) {
        let theme = self.base.theme().other();
        self.theme = Some(theme);
        self.retheme(theme);
        self.status = format!("{} background  t: the other", value_name(&theme));
    }

    /// Draw in `theme` from here on: every palette and plot made in the
    /// last one is made again.
    fn retheme(&mut self, theme: Theme) {
        if !super::repalette(self.base, theme, &mut self.levels, self.round.as_mut()) {
            return;
        }
        if let Some(v) = self.lra.as_mut() {
            v.retheme(theme);
        }
        self.plots.forget();
        self.thumbs.clear();
        self.need_map = true;
        self.need_panel = true;
    }

    /// What `pinto sessions` says the view is showing: the run and its
    /// level.
    fn session_status(&self) -> String {
        format!(
            "{} · {}",
            self.base.run.name(),
            self.base.run.levels[self.cur].tag
        )
    }

    // ── geometry ────────────────────────────────────────────────────────

    /// Drag the side panel's left edge to widen or narrow it. Returns
    /// whether `m` was taken for that.
    fn resize_panel(&mut self, m: MouseEvent) -> bool {
        let col = m.column;
        let on_edge =
            self.side.width > 0 && (self.side.x.saturating_sub(1)..=self.side.x).contains(&col);
        match m.kind {
            MouseEventKind::Down(MouseButton::Left) if on_edge => self.resizing = true,
            MouseEventKind::Drag(MouseButton::Left) if self.resizing => {
                let w = self.side.right().saturating_sub(col);
                self.panel_w[usize::from(self.wide())] = Some(w);
                self.need_map = true;
                self.need_panel = true;
            }
            MouseEventKind::Up(MouseButton::Left) if self.resizing => self.resizing = false,
            _ => return false,
        }
        true
    }

    fn layout(&mut self, terminal: &DefaultTerminal) -> anyhow::Result<Rect> {
        let size = terminal.size()?;
        let area = Rect::new(0, 0, size.width, size.height);
        let wide = self.wide();
        let base = if wide { WIDE_PANEL } else { PANEL };
        // Dragged as wide as leaves the map a strip; else half the screen.
        let most = area.width.saturating_sub(MIN_MAP).max(area.width / 2);
        let panel = self.panel_w[usize::from(wide)].map_or(base.min(area.width / 2), |w| {
            w.clamp(MIN_PANEL.min(most), most)
        });
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
        // two rows of panel names under that; the curve, which names its
        // own axes, takes those rows too.
        self.bars = Rect::default();
        self.below = Rect::default();
        let curve = self.curve_shows();
        if (self.structure || curve)
            && self.view == plots::View::Map
            && !self.grid_shows()
            && map.height >= 16
        {
            let names = if curve { 0 } else { 2 };
            let h = (map.height / 4).max(6) + 2 - names;
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

    /// Whether the side panel is the wide one, for annotation or a dialog.
    /// The save dialog fits the usual panel: widening it would shrink
    /// the map that is about to be saved.
    fn wide(&self) -> bool {
        self.relabel.is_some()
            || self.lra_on
            || self
                .modal
                .as_ref()
                .is_some_and(|m| !matches!(m, annotate::Modal::Save(_)))
    }

    /// Columns the wide panel has beyond its usual width, for its text.
    pub(super) fn more(&self) -> usize {
        usize::from(self.side.width.saturating_sub(WIDE_PANEL))
    }

    /// Show `msg` in a popup over the map until a key or click.
    pub(super) fn fail(&mut self, msg: String) {
        self.error = Some(msg);
        self.need_map = true;
        self.need_panel = true;
    }

    fn image_size(&self) -> (usize, usize) {
        (
            ((self.map.width as f32 * self.px_per_cell.0) as usize).max(1),
            ((self.map.height as f32 * self.px_per_cell.1) as usize).max(1),
        )
    }

    fn viewport(&self) -> Viewport {
        Viewport::of(&self.camera_now())
    }

    /// The camera, at the map's size now.
    fn camera_now(&self) -> render::MapView {
        let (w, h) = self.image_size();
        render::MapView { w, h, ..self.cam }
    }

    /// Move the map by `(dx, dy)` image pixels, as a drag does.
    fn pan_px(&mut self, dx: f32, dy: f32) {
        let mut cam = self.camera_now();
        cam.pan_px(dx, dy, &Limits::default());
        self.set_cam(cam);
    }

    /// Look where `cam` looks.
    fn set_cam(&mut self, cam: render::MapView) {
        self.cam = cam;
        self.need_map = true;
    }

    /// The zoom limits, in world units per pixel, for a `w × h` frame
    /// that fits `r`: down to single cells, out to 8× the fit.
    fn zoom_limits(&self, r: WorldRect, (w, h): (usize, usize)) -> (f32, f32) {
        (self.base.spacing / 80., Viewport::fit(r, w, h).upp * 8.)
    }

    /// Fit `r`, with a small margin, to the map.
    fn fit(&mut self, r: WorldRect) {
        self.look_at(r.pad(FIT_MARGIN));
    }

    /// The map on `r` exactly, filling the tighter axis.
    fn look_at(&mut self, r: WorldRect) {
        let (w, h) = self.image_size();
        self.set_cam(Viewport::fit(r, w, h).camera());
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
        let (px, py) = self.px_in(self.map, col, row);
        let [x, y] = self.camera_now().to_data(px, py);
        (x, y)
    }

    fn in_map(&self, col: u16, row: u16) -> bool {
        self.map.contains(Position::new(col, row))
    }

    /// Scale by `f` (> 1 zooms out) keeping `anchor` fixed on screen.
    fn zoom(&mut self, f: f32, anchor: Option<(u16, u16)>) {
        let limits = self.zoom_limits(self.base.geom.bounds(), self.image_size());
        let anchor = anchor.map(|(c, r)| self.px_in(self.map, c, r));
        let mut cam = self.camera_now();
        render::zoom(&mut cam, f, anchor, limits);
        self.set_cam(cam);
    }

    /// Pan by a fraction of the view.
    fn pan(&mut self, fx: f32, fy: f32) {
        let (w, h) = self.image_size();
        self.pan_px(-fx * w as f32, -fy * h as f32);
    }

    /// Show `msg` before a blocking load, while a terminal is there.
    fn busy(&mut self, terminal: Option<&mut DefaultTerminal>, msg: String) -> anyhow::Result<()> {
        let Some(terminal) = terminal.filter(|_| self.session.attached()) else {
            return Ok(());
        };
        self.status = msg;
        let side = self.layout(terminal)?;
        let (panel, _) = self.panel(side.height);
        self.session
            .draw(terminal, |f| f.render_widget(Paragraph::new(panel), side))?;
        self.status.clear();
        Ok(())
    }
}

/// What a click in the panel picks.
#[derive(Clone, Copy, Debug)]
enum Pick {
    Community(usize),
    /// Index into the shown community's markers.
    Gene(usize),
    /// A community of the structure plot's bars.
    Bar(usize),
    /// A pair of the lra list, by its position among the pairs shown.
    Pair(usize),
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

#[cfg(test)]
#[path = "tui/tests.rs"]
mod tests;
