//! Drawing the map, the panel, the legend and the help.

use super::super::color::{Ramp, Rgb};
use super::super::gene::GeneMap;
use super::super::render::{self, Frame, Layer, Style};
use super::super::{legend_swatch, markers, scalebar, thousands, Level, Show};
use super::{lra, plots, relabel, App, Gfx, Pick};
use crate::tui::style;
use legume_tui::chrome::rgb;
use ratatui::style::Style as TStyle;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::DefaultTerminal;
use ratatui_image::{Image, Resize};
use std::time::Duration;
use std::time::Instant;

/// Most side panel rows a status message wraps to.
const STATUS_ROWS: usize = 6;

/// Columns of a side panel row's name (` level  `), the row's text after.
const KEY: usize = 8;

impl App<'_> {
    /// What the map draws: the level and style on screen, or, for a
    /// community picked in the structure plot on a map of a round's groups,
    /// its level and propensity.
    pub(super) fn drawn(&self) -> (&Level, Style<'_>) {
        if let Some((level, style, _)) = self.lra_drawn() {
            return (level, style);
        }
        match self.bar_focus_level() {
            Some((level, c)) => (
                level,
                Style {
                    layer: Layer::Community(c),
                    edges: false,
                    focus: None,
                    theme: self.base.theme(),
                    point: self.point,
                },
            ),
            None => (self.level(), self.style()),
        }
    }

    /// The feature drawn: the lra pair's exposure when the layer draws
    /// it, else the gene picked.
    pub(super) fn drawn_gene(&self) -> Option<&GeneMap> {
        match self.lra_drawn() {
            Some((_, _, gene)) => gene,
            None => self.gene.as_ref(),
        }
    }

    fn style(&self) -> Style<'_> {
        Style {
            layer: self.layer,
            edges: self.edges && !self.level().grouping,
            focus: self.focus(),
            theme: self.base.theme(),
            point: self.point,
        }
    }

    pub(super) fn redraw(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        self.draw_pending_pair(Some(terminal));
        self.need_panel = false;
        let side = self.layout(terminal)?;
        let mut fresh: Option<Frame> = None;
        if self.need_map && (self.view != plots::View::Map || self.error.is_some()) {
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
            let mut frame = self.base.render(level, &style, self.drawn_gene(), &vp);
            // Labelled with its length and unit, as in an exported PNG, so
            // the map says what the bar measures; on block characters the
            // label would smear, and the panel states it instead.
            if let Some(units) = self.base.units {
                scalebar::draw(&mut frame, &vp, units, self.gfx.shows_images());
            }
            let mode = render::mode(&scene, &vp, self.point);
            if self.bars.height > 0 {
                frame = if self.curve_shows() {
                    self.with_curve(frame)
                } else {
                    self.with_structure(frame)
                };
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
            if let Some(e) = &app.error {
                // The map image is cleared meanwhile, so nothing covers it.
                style::popup(f, map, error_lines(e, map.width.min(ERROR_W)), ERROR_W);
                return;
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
            kitty.show_rgba(
                &mut std::io::stdout(),
                (frame.w, frame.h),
                &frame.rgba,
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
                    *slot = Some(frame.cells(*glyphs, ppc, (cols, rows)));
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
    pub(super) fn panel(&self, height: u16) -> (Vec<Line<'static>>, Vec<(usize, Pick)>) {
        let level = self.level();
        let comm = &level.comm;
        let dim = style::dim();
        let bold = style::bold();
        let row = |k: &str, v: String| {
            let k = format!(" {k:<w$}", w = KEY - 1);
            Line::from(vec![Span::styled(k, dim), Span::raw(v)])
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
            row(
                "scale",
                format!(
                    "{}  {}",
                    scalebar::per_pixel(self.base.units, self.cam.upp()),
                    self.mode
                ),
            ),
            row("point", format!("×{:.2}  < >", self.point)),
            row(
                "bar",
                match self.base.units {
                    Some(units) => {
                        let (w, _) = self.image_size();
                        scalebar::bar_for(w as f32 * self.cam.upp(), units).label
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
            row("screen", self.gfx.name().into()),
        ];
        if let Some(r) = self.round_line() {
            lines.insert(5, row("round", r));
        }
        if let Some(f) = &self.follower {
            let width = usize::from(self.side.width).saturating_sub(KEY + 1);
            let mut agent = f.lines(width).into_iter();
            lines.push(row("agent", agent.next().unwrap_or_default()));
            lines.extend(agent.map(|l| Line::styled(format!("{:KEY$}{l}", ""), dim)));
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
        // Wrapped, so a long message (a lupin error, a path) reads in full;
        // bold in the terminal's own text colour reads on any background.
        let w = usize::from(self.side.width).saturating_sub(2).max(10);
        let status = style::wrap(&self.status, w, w);
        lines.extend(
            status
                .iter()
                .take(STATUS_ROWS)
                .map(|s| Line::styled(format!(" {s}"), style::bold())),
        );
        if status.is_empty() {
            lines.push(Line::raw(""));
        }

        let help = help_lines(self.help, self.relabel.is_some(), self.shift_enter);
        let help = if self.lra_on && !self.help {
            lra::LRA_HELP
                .iter()
                .map(|t| Line::styled(*t, style::dim()))
                .collect()
        } else {
            help
        };
        let mut clickable = Vec::new();
        if let Some(modal) = &self.modal {
            let room = (height as usize).saturating_sub(lines.len());
            lines.extend(self.modal_lines(modal, room));
            return (lines, clickable);
        }
        if self.lra_on {
            let room = (height as usize).saturating_sub(lines.len() + help.len());
            for (line, pick) in self.lra_lines(room) {
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
        let mut mix: Vec<String> = comm
            .top_props(i, 3)
            .into_iter()
            .map(|(c, p)| format!("{} {p:.2}", comm.name(c)))
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
            Line::from(vec![
                Span::styled(format!(" {:<w$}", "cell", w = KEY - 1), dim),
                Span::raw(name),
            ]),
            Line::raw(format!("{:KEY$}{}", "", mix.join("  "))),
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
            return ramp_legend(Layer::Community(c).ramp(self.base.theme()), &title, "1")
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
                Layer::Gene.ramp(self.base.theme()),
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
                        let (colour, on) =
                            legend_swatch(c, &level.palette, focus, self.base.theme());
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
                self.layer.ramp(self.base.theme()),
                &self.layer.legend_title(),
                "1",
            )),
            // Drawn through `self.gene`, handled above.
            Layer::Gene => Vec::new(),
            Layer::Community(c) => {
                let title = format!("{}  c/C", self.layer.legend_title());
                let mut out = ramp_legend(self.layer.ramp(self.base.theme()), &title, "1");
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

/// Characters of a group name in a legend row.
pub(super) const LEGEND_NAME: usize = 17;

/// A legend row's text: the name and cell count, fitted to the panel.
pub(super) fn legend_text(comm: &super::super::data::Communities, c: usize, n: usize) -> String {
    // a long name keeps its start and end, so names sharing a prefix differ
    let name = style::middle(&comm.name(c), LEGEND_NAME);
    let width = if comm.names.is_some() { LEGEND_NAME } else { 4 };
    format!(" {name:<width$} {:>9}", thousands(n))
}

/// A colour ramp from 0 to `top`, under `title`.
pub(super) fn ramp_legend(ramp: &Ramp, title: &str, top: &str) -> Vec<Line<'static>> {
    let bar: Vec<Span> = (0..24)
        .map(|i| Span::styled("█", TStyle::default().fg(rgb(ramp.at(i as f32 / 23.)))))
        .collect();
    let mut first = vec![Span::raw(" 0 ")];
    first.extend(bar);
    first.push(Span::raw(format!(" {top}")));
    vec![Line::raw(format!(" {title}")), Line::from(first)]
}

/// Widest the error popup is drawn.
const ERROR_W: u16 = 72;

/// The error popup's lines for a popup `width` columns wide, wrapped here
/// so the popup is as tall as they are.
pub(super) fn error_lines(msg: &str, width: u16) -> Vec<Line<'static>> {
    let w = usize::from(width).saturating_sub(4).max(10);
    let mut out = vec![Line::styled(" ✗ error", style::bold()), Line::raw("")];
    for part in msg.lines() {
        out.extend(
            style::wrap(part, w, w)
                .into_iter()
                .map(|l| Line::raw(format!(" {l}"))),
        );
    }
    out.push(Line::raw(""));
    out.push(Line::styled(" any key or click closes this", style::dim()));
    out
}

fn help_lines(full: bool, relabel: bool, shift_enter: bool) -> Vec<Line<'static>> {
    let dim = style::dim();
    let text: &[&str] = if relabel && !full {
        relabel::RELABEL_HELP
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
            if shift_enter {
                " H shift-enter  next chart"
            } else {
                " H  structure plot → heatmap → map"
            },
            " drag the panel's left edge: width",
            " t dark / light background (paper)",
            " s save as  f saved figures  q quit",
            " A annotate  R relabel  I pairs",
            " F  follow the agent again (--follow)",
        ]
    } else {
        &[" s save as  r start over", " Esc back  ? keys  q quit"]
    };
    let mut lines: Vec<Line<'static>> = text.iter().map(|t| Line::styled(*t, dim)).collect();
    // The detach and switch keys, in a session.
    if full {
        lines.extend(
            crate::tui::session_keys()
                .map(|(key, what)| Line::styled(format!(" {key}  {what}"), dim)),
        );
    }
    lines
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
