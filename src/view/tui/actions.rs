//! What a key does: focus, genes, layers, levels and exports.

use super::super::data::Rect as WorldRect;
use super::super::gene::Source;
use super::super::render::{self, Layer, Mode, Viewport};
use super::super::{focused, ids, legend_line, markers, write_outputs, Show};
use super::{plots, step_point, App};
use ratatui::DefaultTerminal;

impl App<'_> {
    /// Show the groups `focus` flags, listing `shown`'s markers.
    pub(super) fn set_focus(&mut self, focus: Vec<bool>, shown: Option<usize>) {
        self.focus = focus;
        self.shown = shown;
        self.list_markers();
        self.need_map = true;
    }

    pub(super) fn clear_focus(&mut self) {
        self.focus.fill(false);
        self.shown = None;
        self.markers.clear();
        self.gene = None;
        self.picked = None;
        self.need_map = true;
    }

    pub(super) fn focus(&self) -> Option<&[bool]> {
        self.focus.iter().any(|&f| f).then_some(&self.focus[..])
    }

    /// Draw marker `i` of the shown community; picking the drawn one again
    /// goes back to communities.
    pub(super) fn pick_gene(&mut self, i: usize) {
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
    pub(super) fn show_gene(&mut self, feature: &str) {
        self.view = plots::View::Map;
        let (base, source, clip) = (self.base, self.source, self.clip);
        match base.gene(self.level_mut(), feature, source, clip) {
            Ok(g) => {
                if g.fell_back(source) {
                    self.status = "no data file: model-expected level".into();
                }
                self.gene = Some(g);
            }
            Err(e) => self.fail(format!("{e}")),
        }
        self.need_map = true;
    }

    /// The next (`by` = 1) or previous marker gene.
    pub(super) fn step_gene(&mut self, by: isize) {
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
    pub(super) fn toggle_clip(&mut self) {
        self.clip = if self.clip > 95. { 95. } else { 99. };
        self.status = format!("gene ramp tops at p{}", self.clip);
        self.refresh_gene();
    }

    /// Switch between observed counts and the model-expected level.
    pub(super) fn toggle_source(&mut self) {
        self.source = self.source.other();
        self.refresh_gene();
    }

    /// Draw the shown gene again after its source or clip changed.
    fn refresh_gene(&mut self) {
        if let Some(feature) = self.gene.as_ref().map(|g| g.feature.clone()) {
            self.show_gene(&feature);
        }
    }

    pub(super) fn set_layer(&mut self, layer: Layer) {
        self.gene = None;
        if let Err(e) = layer.check(&self.level().comm) {
            self.status = e.to_string();
            return;
        }
        self.layer = layer;
        self.need_map = true;
    }

    /// `]`/`[`: focus the next (`by` = 1) or previous group, largest first;
    /// a map of one community's propensity follows it.
    pub(super) fn step_focus(&mut self, by: isize) {
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
        let mut focus = vec![false; self.focus.len()];
        focus[c] = true;
        self.gene = None;
        self.set_focus(focus, Some(c));
        if let Layer::Community(_) = self.layer {
            self.community = c;
            self.layer = Layer::Community(c);
        }
        self.status = format!("{}  ] [ next/prev", self.level().comm.name(c));
        self.need_map = true;
    }

    /// Back to the view the viewer opened on: the whole tissue, the starting
    /// level and layer, edges as asked, nothing selected.
    pub(super) fn reset(&mut self) {
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

    pub(super) fn step_level(
        &mut self,
        by: isize,
        terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
        let n = self.levels.len() as isize;
        let next = (self.cur as isize + by).clamp(0, n - 1) as usize;
        if next == self.cur && self.show == Show::Communities {
            return Ok(());
        }
        self.go_level(next, terminal)
    }

    /// Level `next`'s communities on the map, loaded first if need be.
    pub(super) fn go_level(
        &mut self,
        next: usize,
        terminal: &mut DefaultTerminal,
    ) -> anyhow::Result<()> {
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

    pub(super) fn toggle_edges(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
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
    pub(super) fn scale_points(&mut self, by: f32) {
        self.point = step_point(self.point, by);
        self.status = format!("point size ×{:.2}  < >", self.point);
        self.need_map = true;
    }

    /// `0`: every batch fitted, the whole tissue on the map.
    pub(super) fn fit_all(&mut self) {
        self.grid.forget_own();
        self.tile = None;
        self.fit(self.base.geom.bounds());
    }

    pub(super) fn next_tile(&mut self) {
        self.leave_grid();
        let tiles = &self.base.geom.tiles;
        let next = self.tile.map_or(0, |t| (t + 1) % tiles.len());
        self.tile = Some(next);
        let bounds = tiles[next].bounds;
        self.status = format!("batch {}", tiles[next].name);
        self.fit(bounds);
    }

    /// Freeze the current view as `stem`: a PNG and a PDF at
    /// `--export-scale` × the screen resolution, and a `.txt` with the
    /// command that redraws them, the legend and the shown communities'
    /// markers.
    pub(super) fn export_map(&mut self, stem: &str) -> anyhow::Result<()> {
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
            self.drawn_gene(),
            &hi,
            Some(png.as_ref()),
            Some(pdf.as_ref()),
        )?;
        std::fs::write(&txt, self.export_notes(&png, &pdf, &hi, window))?;
        // A small render of the same window for the saved figures.
        let (tw, th) = (
            super::super::saved::THUMB_WIDTH,
            super::super::saved::THUMB_WIDTH * vp.h / vp.w.max(1),
        );
        let small = Viewport::fit(window, tw, th.max(1));
        let (level, style) = self.drawn();
        let thumb = self.base.render(level, &style, self.drawn_gene(), &small);
        let what = match (self.lra_drawn(), &self.gene) {
            (Some((level, ..)), _) => format!("map · lra {}", level.comm.tag),
            (None, Some(g)) => format!("map · {}", markers::symbol(&g.feature)),
            (None, None) => format!("map · {} · {}", level.comm.tag, style.layer),
        };
        let listed = self.remember(&pdf, &what, &thumb);
        let bars = if self.curve_shows() {
            self.export_curve(&format!("{stem}-curve"))?
        } else if self.bars.height > 0 {
            self.export_structure(&format!("{stem}-structure"))?
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
        let round_drawn = self.show != Show::Communities && self.bar_focus_level().is_none();
        let still = super::super::Still {
            base: self.base,
            prefix: self.args.prefix(),
            units: &self.args.units,
            level,
            style,
            round: self
                .round
                .as_ref()
                .filter(|_| round_drawn)
                .map(|r| (&*r.path, self.show)),
            gene: self.gene.as_ref(),
            pair: self.still_pair(false),
            vp: *vp,
            window,
        };
        let cmd = still.command(Some(png), Some(pdf));
        let tag = still.tag();

        let mut out = String::new();
        writeln!(out, "# pinto view export").ok();
        writeln!(out, "image    {png} ({}×{}), {pdf}", vp.w, vp.h).ok();
        writeln!(out, "redraw   {cmd}").ok();
        writeln!(out, "run      {}", self.base.run.source()).ok();
        writeln!(out, "level    {tag} (K={})", comm.k).ok();
        writeln!(out, "layer    {}", style.layer).ok();
        let scale = super::super::scalebar::per_pixel(self.base.units, vp.upp);
        writeln!(out, "scale    {scale}").ok();
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

    /// Show community `c` alone, or add/remove it from the shown set.
    /// Selecting the only shown community again shows everything.
    pub(super) fn select(&mut self, c: usize, add: bool) {
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
                    .top(shown, markers::TOP)
                    .into_iter()
                    .map(|m| {
                        let symbol = markers::symbol(&m.name).to_string();
                        (m.name, symbol, m.fold)
                    })
                    .collect();
            }
            Err(e) => self.fail(format!("no markers: {e}")),
        }
    }
}
