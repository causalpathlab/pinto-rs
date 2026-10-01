//! Plots of the grouping on screen: a structure plot of the cells'
//! community mixtures under the map, and a heatmap of the top genes ×
//! groups in place of it; `H` steps map → structure → heatmap → map.
//!
//! Both follow what the map groups by (`a`): the level's communities, or a
//! lupin round's cell types or clusters. The structure plot's bars are
//! always a level's community propensities; a round only sorts the cells
//! into its groups. A click on a community in it shows where that
//! community lies: on a map of communities it is the focus, on a map of a
//! round's groups the map draws its propensity.

use super::super::heatmap::Heatmap;
use super::super::markers;
use super::super::render::Frame;
use super::super::structure::{Drawn, Structure};
use super::super::Level;
use super::{legend_text, rgb, App, Pick, Show};
use crate::tui::style;
use ratatui::style::Style as TStyle;
use ratatui::text::{Line, Span};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum View {
    Map,
    Heatmap,
}

/// What a plot was built from: the level behind it, what groups the cells,
/// and the round.
type Key = (usize, Show, Option<PathBuf>);

/// What a structure plot was drawn for: its content, size in pixels, and
/// the communities in colour.
type DrawnFor = (Key, (usize, usize), Option<Vec<bool>>);

#[derive(Default)]
pub struct Cache {
    structure: Option<(Key, Structure)>,
    /// The structure plot last drawn: its panels, and the community under
    /// each pixel, for clicks; with what it was drawn for, so panning the
    /// map does not draw it again.
    drawn: Option<Drawn>,
    drawn_for: Option<DrawnFor>,
    heatmap: Option<((Key, usize), Heatmap)>,
    /// Genes per group in the heatmap; 0 fits them to the screen.
    per_group: usize,
    /// Screen rows of the heatmap's genes.
    rows: Vec<(u16, Box<str>)>,
}

/// A heatmap cell's colour: its clipped z on the diverging scale; a gene
/// missing from the data in the dimmed tissue colour.
fn z_colour(z: f32, theme: super::super::color::Theme) -> [u8; 3] {
    if z.is_finite() {
        super::super::color::diverging(z / super::super::heatmap::CLIP)
    } else {
        theme.dimmed()
    }
}

/// The heatmap's colour key: -2.5 … 0 … +2.5.
fn heat_key() -> Line<'static> {
    let clip = super::super::heatmap::CLIP;
    let mut spans = vec![Span::raw(format!("  z -{clip} "))];
    for i in 0..21 {
        let t = i as f32 / 10. - 1.;
        spans.push(Span::styled(
            "█",
            TStyle::default().fg(rgb(super::super::color::diverging(t))),
        ));
    }
    spans.push(Span::raw(format!(
        " +{clip}   + - genes  click a gene: map"
    )));
    Line::from(spans)
}

/// Whether the heatmap takes `key`: its own keys (`H`, grouping, genes per
/// group) and the viewer's (save, saved figures, help, quit, back).
pub fn chart_key(key: ratatui::crossterm::event::KeyEvent) -> bool {
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};
    key.modifiers.contains(KeyModifiers::CONTROL)
        || matches!(
            key.code,
            KeyCode::Esc
                | KeyCode::Char('H' | 'c' | 'C' | '+' | '=' | '-' | '_' | 's' | 'f' | '?' | 'q')
        )
}

/// Heatmap rows above the genes: title, key, names, swatches.
const HEAT_HEAD: usize = 4;

/// Characters of a group name written down a heatmap column.
const NAME_ROWS: usize = 8;

impl App<'_> {
    /// `H`: map → map with the structure plot under it → heatmap → map.
    pub(super) fn cycle_chart(&mut self) {
        if self.view == View::Heatmap {
            self.view = View::Map;
            self.structure = false;
            self.status = "map".into();
        } else if self.structure {
            self.view = View::Heatmap;
            self.bar_focus = None;
            self.status = "heatmap: + - genes per group, click one to map it; H or esc: map".into();
        } else {
            self.structure = true;
            self.status =
                "structure plot: click a community to see where it lies; H: heatmap".into();
        }
        self.need_map = true;
    }

    /// The level whose communities make the structure plot's bars: the
    /// round's level when a round groups the cells, else the current one.
    fn bars(&self) -> usize {
        match (&self.round, self.show) {
            (Some(r), Show::Types | Show::Clusters) => r.level,
            _ => self.cur,
        }
    }

    fn plot_key(&self, level: usize) -> Key {
        let round = self
            .round
            .as_ref()
            .filter(|_| self.show != Show::Communities)
            .map(|r| r.path.clone());
        (level, self.show, round)
    }

    /// The community picked in the structure plot while the map shows a
    /// round's groups, with its level: the map draws its propensity.
    pub(super) fn bar_focus_level(&self) -> Option<(&Level, usize)> {
        let c = self.bar_focus?;
        if self.show == Show::Communities || self.gene.is_some() || !self.bars_shown() {
            return None;
        }
        Some((self.levels[self.bars()].as_ref()?, c))
    }

    /// Which of the bars' communities are in colour: the map's focus on a
    /// map of communities, else the one picked; `None` for all.
    pub(super) fn bars_focus(&self) -> Option<Vec<bool>> {
        if self.show == Show::Communities {
            return self.focus().map(<[bool]>::to_vec);
        }
        let c = self.bar_focus?;
        let k = self.levels[self.bars()].as_ref()?.comm.k;
        Some((0..k).map(|j| j == c).collect())
    }

    /// Show where community `c` of the bars lies, or stop showing it.
    pub(super) fn focus_bar(&mut self, c: usize) {
        if self.show == Show::Communities {
            // The bars are the map's own communities.
            self.select(c, false);
            return;
        }
        self.bar_focus = if self.bar_focus == Some(c) {
            None
        } else {
            Some(c)
        };
        self.gene = None;
        self.status = match (self.bar_focus, self.levels[self.bars()].as_ref()) {
            (Some(c), Some(level)) => {
                format!("{}: where it lies (click again or Esc)", level.comm.name(c))
            }
            _ => String::new(),
        };
        self.need_map = true;
    }

    /// A click on the structure plot: the community under it.
    pub(super) fn bar_click(&mut self, col: u16, row: u16) {
        let (x, y) = self.px_in(self.bars, col, row);
        let hit = self
            .plots
            .drawn
            .as_ref()
            .and_then(|d| d.at(x as usize, y as usize));
        match hit {
            Some(c) => self.focus_bar(c),
            None => self.status = "between the bars".into(),
        }
    }

    /// `map` with the structure plot drawn under it, as one frame.
    pub(super) fn with_structure(&mut self, map: Frame) -> Frame {
        let total = ((self.canvas.height as f32 * self.px_per_cell.1) as usize).max(map.h + 1);
        let Some(bars) = self.structure_frame(map.w, total - map.h) else {
            return map;
        };
        let mut rgba = map.rgba;
        rgba.extend_from_slice(&bars.rgba);
        Frame {
            w: map.w,
            h: map.h + bars.h,
            rgba,
            background: map.background,
        }
    }

    /// The structure plot at `w × h`, kept for clicks; `None` when its
    /// level does not load.
    fn structure_frame(&mut self, w: usize, h: usize) -> Option<Frame> {
        let bars = self.load_bars()?;
        let key = self.plot_key(bars);
        if self.plots.structure.as_ref().map(|(k, _)| k) != Some(&key) {
            let structure = self.build_structure(bars);
            self.plots.structure = Some((key.clone(), structure));
        }
        let wanted = Some((key, (w, h), self.bars_focus()));
        if self.plots.drawn_for != wanted || self.plots.drawn.is_none() {
            self.plots.drawn = Some(self.render_structure(w, h));
            self.plots.drawn_for = wanted;
        }
        self.plots.drawn.as_ref().map(|d| d.frame.clone())
    }

    /// Load the level whose communities make the bars, and give its
    /// index; `None` when it does not load.
    pub(super) fn load_bars(&mut self) -> Option<usize> {
        let bars = self.bars();
        if self.levels[bars].is_none() {
            match self.base.level(bars, false) {
                Ok(level) => self.levels[bars] = Some(level),
                Err(e) => {
                    self.status = format!("{e}");
                    return None;
                }
            }
        }
        Some(bars)
    }

    /// Whether structure bars are on screen: under the map, or under
    /// each batch of the grid.
    pub(super) fn bars_shown(&self) -> bool {
        self.bars.height > 0 || (self.grid_shows() && self.structure)
    }

    fn render_structure(&self, w: usize, h: usize) -> Drawn {
        let level = self.levels[self.bars()].as_ref().expect("loaded");
        let (_, structure) = self.plots.structure.as_ref().expect("built");
        let focus = self.bars_focus();
        structure.render(
            &level.comm,
            &level.palette,
            (w, h),
            focus.as_deref(),
            self.base.theme,
        )
    }

    /// Bars from level `bars`, panels from what groups the cells: the
    /// round's groups, else batches, else one panel.
    fn build_structure(&self, bars: usize) -> Structure {
        let comm = &self.levels[bars].as_ref().expect("loaded").comm;
        let geom = &self.base.geom;
        if self.show != Show::Communities && self.round.is_some() {
            let g = &self.level().comm;
            let names: Vec<String> = (0..g.k).map(|c| g.name(c)).collect();
            return Structure::build(comm, &g.cluster, &names, &g.by_size);
        }
        if geom.tiles.len() > 1 {
            return self.by_batch(bars);
        }
        let one = vec![0u16; geom.n()];
        Structure::build(comm, &one, &["all cells".into()], &[0])
    }

    /// Bars from level `bars` (loaded), a panel per batch.
    pub(super) fn by_batch(&self, bars: usize) -> Structure {
        let comm = &self.levels[bars].as_ref().expect("loaded").comm;
        let geom = &self.base.geom;
        let names: Vec<String> = geom.tiles.iter().map(|t| t.name.to_string()).collect();
        let order: Vec<usize> = (0..names.len()).collect();
        Structure::build(comm, &geom.batch, &names, &order)
    }

    /// The bars' communities, to pick one from, when the map shows a
    /// round's groups: at most `room` lines.
    pub(super) fn bars_legend(&self, room: usize) -> Vec<(Line<'static>, Option<Pick>)> {
        let Some(level) = self.levels[self.bars()].as_ref() else {
            return Vec::new();
        };
        if room < 3 {
            return Vec::new();
        }
        let comm = &level.comm;
        let dim = style::dim();
        let mut out = vec![
            (Line::raw(""), None),
            (
                Line::styled(format!(" bars: {} communities", comm.tag), dim),
                None,
            ),
        ];
        for &c in comm.by_size.iter().take(room - 2) {
            let on = self.bar_focus == Some(c);
            let line = Line::from(vec![
                Span::raw(if on { "▸" } else { " " }),
                Span::styled("██", TStyle::default().fg(rgb(level.palette[c]))),
                Span::raw(legend_text(comm, c, comm.sizes[c])),
            ]);
            out.push((line, Some(Pick::Bar(c))));
        }
        out
    }

    /// The structure plot's panel names, under it: two rows, alternating,
    /// so narrow neighbours do not overwrite each other.
    pub(super) fn plot_names(&self) -> Option<Vec<Line<'static>>> {
        if self.below.height == 0 {
            return None;
        }
        let drawn = self.plots.drawn.as_ref()?;
        let width = self.below.width as usize;
        let mut rows = vec![vec![' '; width]; 2];
        let mut next_free = [0usize; 2];
        for (i, p) in drawn.panels.iter().enumerate() {
            let c0 = (p.x0 as f32 / self.px_per_cell.0) as usize;
            let c1 = ((p.x1 + 1) as f32 / self.px_per_cell.0) as usize;
            let r = i % 2;
            let start = c0.max(next_free[r]);
            if start >= width {
                continue;
            }
            // At most the panel's width, or up to the next panel on the row.
            let room = c1.saturating_sub(start).max(8).min(width - start);
            let name: Vec<char> = p.name.chars().take(room.saturating_sub(1)).collect();
            if name.is_empty() {
                continue;
            }
            for (k, ch) in name.iter().enumerate() {
                rows[r][start + k] = *ch;
            }
            next_free[r] = start + name.len() + 1;
        }
        Some(
            rows.into_iter()
                .map(|r| Line::raw(r.into_iter().collect::<String>()))
                .collect(),
        )
    }

    /// The heatmap as text; `None` in the other views.
    pub(super) fn plot_text(&mut self) -> Option<Vec<Line<'static>>> {
        if self.view != View::Heatmap {
            return None;
        }
        let base = self.base;
        if let Err(e) = base.load_features(self.level_mut()) {
            return Some(vec![Line::raw(format!(
                " no gene rates for this level: {e}"
            ))]);
        }
        let groups = self.level().comm.by_size.len().max(1);
        let rows = (self.map.height as usize).saturating_sub(HEAT_HEAD + NAME_ROWS + 1);
        let per_group = match self.plots.per_group {
            0 => (rows / groups).clamp(1, 10),
            n => n,
        };
        let key = (self.plot_key(self.cur), per_group);
        if self.plots.heatmap.as_ref().map(|(k, _)| k) != Some(&key) {
            let level = self.level();
            let rates = level.features.as_ref().expect("just loaded");
            let comm = &level.comm;
            // Observed counts rank the model's candidates when the run's
            // data files read.
            let mut failed = None;
            let observe = |names: &[&str]| match base.expression() {
                Ok(Some(expr)) => match expr.group_means(names, &comm.cluster, comm.k) {
                    Ok(means) => Some(means),
                    Err(e) => {
                        failed = Some(e.to_string());
                        None
                    }
                },
                _ => None,
            };
            let heat = Heatmap::build(rates, comm, per_group, Box::new(observe));
            if let Some(e) = failed {
                self.status = format!("model levels shown: {e}");
            }
            self.plots.heatmap = Some((key, heat));
        }
        let (lines, rows) = self.heat_lines(per_group);
        self.plots.rows = rows;
        Some(lines)
    }

    fn heat_lines(&self, per_group: usize) -> (Vec<Line<'static>>, Vec<(u16, Box<str>)>) {
        let (_, heat) = self.plots.heatmap.as_ref().expect("built");
        let level = self.level();
        let comm = &level.comm;
        let dim = style::dim();
        let bold = style::bold();
        let label_w = 11usize;
        let cols = heat.groups.len();
        let cw = if label_w + 2 * cols <= self.map.width as usize {
            2
        } else {
            1
        };
        let what = self.group_word();
        let mut out = vec![
            Line::from(vec![
                Span::styled(format!(" top {per_group} genes per {what}"), bold),
                Span::styled(
                    if heat.observed {
                        "  mean ln(1+count), z per gene"
                    } else {
                        "  model ln(1+rate) (no data files), z per gene"
                    },
                    dim,
                ),
            ]),
            heat_key(),
        ];
        // Group names written down their columns.
        // A round cluster by its id: `K3 T_cell` → `K3`.
        let short = |c: usize| -> Vec<char> { super::focus_name(comm, c).chars().collect() };
        let tallest = heat
            .groups
            .iter()
            .map(|&c| short(c).len())
            .max()
            .unwrap_or(0)
            .min(NAME_ROWS);
        for r in 0..tallest {
            let mut text = " ".repeat(label_w);
            for &c in &heat.groups {
                let ch = short(c).get(r).copied().unwrap_or(' ');
                text.push(ch);
                if cw == 2 {
                    text.push(' ');
                }
            }
            out.push(Line::raw(text));
        }
        let mut swatches = vec![Span::raw(" ".repeat(label_w))];
        for &c in &heat.groups {
            swatches.push(Span::styled(
                "▀".repeat(cw),
                TStyle::default().fg(rgb(level.palette[c])),
            ));
        }
        out.push(Line::from(swatches));

        let room = (self.map.height as usize).saturating_sub(out.len() + 1);
        let drawn = self
            .gene
            .as_ref()
            .map(|g| markers::symbol(&g.feature).to_string());
        let mut rows = Vec::new();
        for (r, (feature, symbol, _)) in heat.genes.iter().enumerate().take(room) {
            let on = drawn.as_deref() == Some(symbol.as_str());
            let name: String = symbol.chars().take(label_w - 2).collect();
            let mut spans = vec![Span::styled(
                format!(" {name:<w$} ", w = label_w - 2),
                if on { bold } else { TStyle::default() },
            )];
            for &z in &heat.z[r * cols..(r + 1) * cols] {
                spans.push(Span::styled(
                    "█".repeat(cw),
                    TStyle::default().fg(rgb(z_colour(z, self.base.theme))),
                ));
            }
            rows.push((self.map.y + out.len() as u16, feature.clone()));
            out.push(Line::from(spans));
        }
        if heat.genes.len() > room {
            out.push(Line::styled(
                format!(" … {} more genes (-)", heat.genes.len() - room),
                dim,
            ));
        }
        (out, rows)
    }

    /// What a group is in the current grouping.
    fn group_word(&self) -> &'static str {
        match self.show {
            Show::Communities => "community",
            Show::Types => "cell type",
            Show::Clusters => "cluster",
        }
    }

    /// The heatmap as a picture, for its thumbnail: a swatch row of the
    /// groups' colours, then a block per gene and group.
    fn heat_picture(&self, heat: &Heatmap) -> Frame {
        const PX: usize = 6;
        let level = self.level();
        let cols = heat.groups.len().max(1);
        let rows = heat.genes.len() + 1;
        let (w, h) = (cols * PX, rows * PX);
        let bg = self.base.theme.background();
        let mut rgba = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let (r, c) = (y / PX, x / PX);
                let [red, g, b] = if r == 0 {
                    heat.groups.get(c).map_or(bg, |&g| level.palette[g])
                } else {
                    z_colour(
                        heat.z.get((r - 1) * cols + c).copied().unwrap_or(0.),
                        self.base.theme,
                    )
                };
                rgba.extend_from_slice(&[red, g, b, 255]);
            }
        }
        Frame {
            w,
            h,
            rgba,
            background: bg,
        }
    }

    /// `+`/`-` in the heatmap: genes per group.
    pub(super) fn more_genes(&mut self, by: isize) {
        let now = match (&self.plots.heatmap, self.plots.per_group) {
            (_, n) if n > 0 => n,
            (Some(((_, n), _)), _) => *n,
            _ => 3,
        };
        self.plots.per_group = (now as isize + by).clamp(1, 100) as usize;
        self.status = format!("{} genes per group", self.plots.per_group);
        self.need_map = true;
    }

    /// A click in a plot: a heatmap gene goes onto the map.
    pub(super) fn plot_click(&mut self, row: u16) {
        if self.view != View::Heatmap {
            return;
        }
        let Some(feature) = self
            .plots
            .rows
            .iter()
            .find(|(r, _)| *r == row)
            .map(|(_, f)| f.clone())
        else {
            return;
        };
        self.show_gene(&feature);
        self.need_map = true;
    }

    /// The structure plot as a PNG at the export scale, its panels and
    /// stack listed beside it; with the map's export (`s`).
    pub(super) fn export_structure(&mut self) -> anyhow::Result<String> {
        use std::fmt::Write as _;
        if self.plots.structure.is_none() {
            return Ok(String::new());
        }
        let stem = super::free_stem("pinto-structure", "png");
        let scale = self.args.export_scale.max(1);
        let w = (self.bars.width as f32 * self.px_per_cell.0) as usize * scale;
        let h = (self.bars.height as f32 * self.px_per_cell.1) as usize * scale;
        let drawn = self.render_structure(w.max(1), h.max(1));
        drawn
            .frame
            .write_png(std::path::Path::new(&format!("{stem}.png")))?;
        let what = format!("structure · {}", self.group_word());
        let listed = self.remember(&format!("{stem}.png"), &what, &drawn.frame);
        let level = self.levels[self.bars()].as_ref().expect("loaded");
        let (_, structure) = self.plots.structure.as_ref().expect("built");
        let mut notes = String::new();
        writeln!(notes, "run      {}", self.base.run.source()).ok();
        writeln!(notes, "bars     level {} propensities", level.comm.tag).ok();
        writeln!(
            notes,
            "image    {stem}.png ({}×{})",
            drawn.frame.w, drawn.frame.h
        )
        .ok();
        writeln!(notes, "\npanels (first..last pixel column, cells)").ok();
        for (p, (_, cells)) in drawn.panels.iter().zip(&structure.panels) {
            writeln!(
                notes,
                "  {:<24} {:>6}..{:<6} {:>9}",
                p.name,
                p.x0,
                p.x1,
                cells.len()
            )
            .ok();
        }
        writeln!(notes, "\nstack, bottom to top").ok();
        for &c in &structure.stack {
            writeln!(
                notes,
                "{}",
                super::legend_line(&level.comm, c, level.palette[c])
            )
            .ok();
        }
        std::fs::write(format!("{stem}.txt"), notes)?;
        Ok(format!(", {stem}.png{listed}"))
    }

    /// `s` in the heatmap: its values as a table, with a note.
    pub(super) fn export_plot(&mut self) -> anyhow::Result<()> {
        use std::fmt::Write as _;
        let stem = super::free_stem("pinto-heatmap", "tsv");
        let Some((_, heat)) = self.plots.heatmap.as_ref() else {
            self.status = "nothing drawn yet".into();
            return Ok(());
        };
        std::fs::write(format!("{stem}.tsv"), heat.tsv(&self.level().comm))?;
        let picture = self.heat_picture(heat);
        let what = format!("heatmap · {}", self.group_word());
        let listed = self.remember(&format!("{stem}.tsv"), &what, &picture);
        let mut notes = String::new();
        writeln!(notes, "run      {}", self.base.run.source()).ok();
        writeln!(notes, "table    {stem}.tsv: value per gene and group").ok();
        writeln!(notes, "groups   {}", self.level().comm.tag).ok();
        std::fs::write(format!("{stem}.txt"), notes)?;
        self.status = format!("saved {stem}.tsv .txt{listed}");
        Ok(())
    }
}
