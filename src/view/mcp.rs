//! The viewer's tools for `pinto mcp`: open a run, draw it, ask about its
//! groups and cells, write figures. One run and one view at a time, held
//! between calls as the terminal viewer holds them: `view_map` changes
//! only what it is given, then draws the map.
//!
//! lupin rounds are read as they are on disk; making them is lupin's.

use super::color::{hex, Theme};
use super::data::{Rect, NO_CLUSTER};
use super::gene::{GeneMap, Source};
use super::render::{Layer, Style, Viewport};
use super::round::{round_name, Round};
use super::{lupin, markers, scalebar, Base, Level, Show, FIT_MARGIN};
use clap::ValueEnum;
use legume_mcp::{Call, Content, Handler, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

/// Width of a map drawn without one: enough to read a tissue by, small
/// enough to keep the image cheap for the agent.
const WIDTH: usize = 1024;

/// Largest side of a map drawn or exported.
const MAX_SIDE: usize = 4096;

/// The viewer's tools, and the run they have open.
#[derive(Default)]
pub struct Tools {
    open: Option<Open>,
}

/// An open run and the view on it.
struct Open {
    base: Base,
    /// Levels loaded so far, by position in the run's levels.
    levels: Vec<Option<Level>>,
    /// The community level shown, by position.
    at: usize,
    show: Show,
    round: Option<Round>,
    layer: Layer,
    /// One flag per group of the grouping shown.
    focus: Vec<bool>,
    gene: Option<GeneMap>,
    source: Source,
    clip: f32,
    edges: bool,
    point: f32,
    /// The world window asked for; the map drawn covers it.
    window: Rect,
    /// The last map drawn, for `view_cell` pixels.
    drawn: Option<Viewport>,
}

impl Handler for Tools {
    fn tools(&self) -> Vec<ToolSpec> {
        specs()
    }

    fn call(&mut self, name: &str, args: Value, cx: &Call) -> anyhow::Result<Vec<Content>> {
        if name == "view_open" {
            let a: OpenArgs = parse(args)?;
            // One run in memory at a time: a failed open leaves none.
            self.open = None;
            cx.progress(&format!("loading {}", a.prefix));
            let open = Open::new(&a.prefix, a.level.as_deref())?;
            let summary = open.summary();
            cx.state(&open.followed());
            self.open = Some(open);
            return Ok(vec![Content::Json(summary)]);
        }
        let open = self
            .open
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no run open: call view_open first"))?;
        match name {
            "view_map" => {
                let shown = open.map(parse(args)?)?;
                cx.state(&open.followed());
                Ok(shown)
            }
            "view_markers" => open.markers(parse(args)?),
            "view_cell" => open.cell(parse(args)?),
            "view_export" => open.export(parse(args)?),
            _ => unreachable!("the server answers unknown tools itself"),
        }
    }
}

/// The arguments of a call, or what is wrong with them.
fn parse<T: serde::de::DeserializeOwned>(args: Value) -> anyhow::Result<T> {
    serde_json::from_value(args).map_err(|e| anyhow::anyhow!("arguments: {e}"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenArgs {
    prefix: String,
    level: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct MapArgs {
    level: Option<String>,
    round: Option<String>,
    show: Option<String>,
    layer: Option<String>,
    focus: Option<Vec<Value>>,
    gene: Option<String>,
    source: Option<String>,
    clip: Option<f32>,
    edges: Option<bool>,
    point_size: Option<f32>,
    theme: Option<String>,
    bbox: Option<[f32; 4]>,
    fit: Option<bool>,
    batch: Option<String>,
    center: Option<[f32; 2]>,
    zoom: Option<f32>,
    width: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MarkersArgs {
    group: Value,
    n: Option<usize>,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct CellArgs {
    px: Option<[f32; 2]>,
    at: Option<[f32; 2]>,
    name: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct ExportArgs {
    png: Option<String>,
    pdf: Option<String>,
    width: Option<usize>,
    overwrite: bool,
}

impl Open {
    fn new(prefix: &str, level: Option<&str>) -> anyhow::Result<Self> {
        // No terminal to read a background from: dark, as --png draws.
        let base = Base::load(prefix, "auto", Theme::Dark)?;
        let at = base.run.level_index(level)?;
        let mut levels: Vec<Option<Level>> = base.run.levels.iter().map(|_| None).collect();
        let level = base.level(at, false)?;
        let focus = vec![false; level.comm.k];
        levels[at] = Some(level);
        let window = base.geom.bounds().pad(FIT_MARGIN);
        Ok(Open {
            base,
            levels,
            at,
            show: Show::Communities,
            round: None,
            layer: Layer::Argmax,
            focus,
            gene: None,
            source: Source::Observed,
            clip: 99.,
            edges: false,
            point: 1.,
            window,
            drawn: None,
        })
    }

    /// The grouping shown: the level's communities, or the round's.
    fn level(&self) -> &Level {
        super::shown(&self.levels, self.round.as_ref(), self.show, self.at)
    }

    /// The run, and the grouping shown to load into.
    fn parts(&mut self) -> (&Base, &mut Level) {
        let level = super::shown_mut(&mut self.levels, self.round.as_mut(), self.show, self.at);
        (&self.base, level)
    }

    fn focus(&self) -> Option<&[bool]> {
        self.focus.iter().any(|&f| f).then_some(&self.focus[..])
    }

    /// What the run holds, and what `view_map` can be asked to show.
    fn summary(&self) -> Value {
        let run = &self.base.run;
        let geom = &self.base.geom;
        let tags: Vec<&str> = run.levels.iter().map(|l| l.tag.as_str()).collect();
        let rounds: Vec<Value> = self
            .base
            .rounds()
            .iter()
            .map(|p| {
                json!({
                    "round": shown_path(p),
                    "name": round_name(p),
                    "level": lupin::round_level(p, &tags),
                })
            })
            .collect();
        let batches: Vec<Value> = if geom.tiles.len() > 1 {
            geom.tiles
                .iter()
                .map(|t| json!({"batch": t.name, "cells": t.n_cells, "bounds": rect(t.bounds)}))
                .collect()
        } else {
            Vec::new()
        };
        json!({
            "run": run.name(),
            "source": run.source(),
            "cells": geom.n(),
            "coords": geom.coord_names,
            "bounds": rect(geom.bounds()),
            "batches": batches,
            "levels": tags,
            "level": self.level().comm.tag,
            "k": self.level().comm.k,
            "groups": self.groups(),
            "data_files": run.data_files(),
            "rounds": rounds,
        })
    }

    /// The groups, largest first: id, colour and cells, and the name when
    /// it says more than the id. Out of focus they are drawn `dimmed`.
    fn groups(&self) -> Vec<Value> {
        let level = self.level();
        let comm = &level.comm;
        comm.by_size
            .iter()
            .map(|&c| {
                let (id, name) = (super::focus_name(comm, c), comm.name(c));
                let mut g = json!({
                    "group": id,
                    "colour": hex(level.palette[c]),
                    "cells": comm.sizes[c],
                });
                if name != id {
                    g["name"] = name.into();
                }
                g
            })
            .collect()
    }

    fn map(&mut self, a: MapArgs) -> anyhow::Result<Vec<Content>> {
        let next = self.next(&a)?;
        self.commit(next);
        let vp = self.viewport(a.width.unwrap_or(WIDTH));
        let mut frame = self.render(&vp);
        if let Some(units) = self.base.units {
            scalebar::draw(&mut frame, &vp, units, true);
        }
        self.drawn = Some(vp);
        // The state as text, not structuredContent: a client that reads
        // that alone would never show the agent the map.
        Ok(vec![
            Content::Png(frame.small_png()?),
            Content::Text(self.state(&vp).to_string()),
        ])
    }

    /// The view `a` asks for, made in full before any of it is kept: what
    /// it loads is cached, but a call that fails leaves the view as it was.
    fn next(&mut self, a: &MapArgs) -> anyhow::Result<Next> {
        let mut show = a
            .show
            .as_deref()
            .map(|s| value_enum(s, "show"))
            .transpose()?;
        let mut round = None;
        if let Some(path) = &a.round {
            let path = Path::new(path);
            if !self
                .round
                .as_ref()
                .is_some_and(|r| lupin::same_file(&r.path, path))
            {
                round = Some(self.base.round(path)?);
            }
            // A round given is a round to look at.
            show = show.or(Some(Show::Clusters));
        }
        let mut at = self.at;
        if let Some(tag) = &a.level {
            at = self.base.run.level_index(Some(tag))?;
            show = show.or(Some(Show::Communities));
        }
        let show = show.unwrap_or(self.show);
        if show != Show::Communities && round.is_none() && self.round.is_none() {
            let newest = self.base.rounds();
            let path = newest.first().ok_or_else(|| {
                anyhow::anyhow!(
                    "show {}: no lupin round of this run; give round",
                    super::value_name(&show)
                )
            })?;
            round = Some(self.base.round(path)?);
        }
        if self.levels[at].is_none() {
            self.levels[at] = Some(self.base.level(at, false)?);
        }
        let regrouped = drawn(show, at, round.as_ref().or(self.round.as_ref()))
            != drawn(self.show, self.at, self.round.as_ref());

        let level = super::shown_mut(
            &mut self.levels,
            round.as_mut().or(self.round.as_mut()),
            show,
            at,
        );
        let edges = a.edges.unwrap_or(self.edges);
        if edges {
            // A round's groupings have no edges; they load none.
            self.base.load_edges(level)?;
        }
        let comm = &level.comm;
        let mut layer = if regrouped {
            self.layer.fit(comm)
        } else {
            self.layer
        };
        if let Some(asked) = &a.layer {
            layer = asked.parse()?;
            layer.check(comm)?;
        }
        let focus = match &a.focus {
            Some(ids) => {
                let ids = ids
                    .iter()
                    .map(group_id)
                    .collect::<anyhow::Result<Vec<_>>>()?;
                super::parse_focus(&ids, comm)?
            }
            None if regrouped => vec![false; comm.k],
            None => self.focus.clone(),
        };
        let point = match a.point_size {
            Some(p) => super::check_point_size(p).map_err(anyhow::Error::msg)?,
            None => self.point,
        };
        let source = match &a.source {
            Some(s) => value_enum(s, "source")?,
            None => self.source,
        };
        let clip = match a.clip {
            Some(c) => super::check_clip(c).map_err(anyhow::Error::msg)?,
            None => self.clip,
        };

        let feature = match a.gene.as_deref() {
            Some("") => None,
            Some(feature) => Some(feature),
            // A layer asked for is drawn in place of the gene, as `1`–`4` do.
            None if a.layer.is_some() => None,
            None => self.gene.as_ref().map(|g| &*g.feature),
        };
        // A map made is kept unless its source or clip changed, or it is
        // the model's expected level of a grouping no longer shown.
        let kept = feature.is_some_and(|f| {
            self.gene.as_ref().is_some_and(|g| {
                (*g.feature == *f || markers::symbol(&g.feature) == f)
                    && (g.source, g.clip) == (source, clip)
                    && !(regrouped && g.source == Source::Expected)
            })
        });
        let gene = match feature {
            _ if kept => None,
            Some(f) => Some(Some(self.base.gene(level, f, source, clip)?)),
            None => Some(None),
        };

        let theme = a
            .theme
            .as_deref()
            .map(|t| value_enum(t, "theme"))
            .transpose()?;
        Ok(Next {
            at,
            show,
            round,
            theme,
            layer,
            focus,
            edges,
            point,
            source,
            clip,
            gene,
            window: self.window(a)?,
        })
    }

    fn commit(&mut self, n: Next) {
        (self.at, self.show) = (n.at, n.show);
        if let Some(round) = n.round {
            self.round = Some(round);
        }
        if let Some(theme) = n.theme {
            super::repalette(&self.base, theme, &mut self.levels, self.round.as_mut());
        }
        (self.layer, self.focus, self.edges, self.point) = (n.layer, n.focus, n.edges, n.point);
        (self.source, self.clip) = (n.source, n.clip);
        if let Some(gene) = n.gene {
            self.gene = gene;
        }
        self.window = n.window;
    }

    /// The window asked for: a box, the whole tissue, a batch, or the one
    /// shown moved and zoomed.
    fn window(&self, a: &MapArgs) -> anyhow::Result<Rect> {
        let geom = &self.base.geom;
        let mut window = self.window;
        if let Some([x0, y0, x1, y1]) = a.bbox {
            window = Rect::from_corners(x0, y0, x1, y1);
        } else if a.fit == Some(true) {
            window = geom.bounds().pad(FIT_MARGIN);
        } else if let Some(name) = &a.batch {
            let tile = geom
                .tiles
                .iter()
                .find(|t| *t.name == **name)
                .ok_or_else(|| {
                    let names: Vec<&str> = geom.tiles.iter().map(|t| &*t.name).collect();
                    anyhow::anyhow!("no batch {name:?}; batches: {}", names.join(", "))
                })?;
            window = tile.bounds.pad(FIT_MARGIN);
        }
        if a.center.is_some() || a.zoom.is_some() {
            let w = window;
            let (cx, cy) = match a.center {
                Some([x, y]) => (x, y),
                None => (0.5 * (w.x0 + w.x1), 0.5 * (w.y0 + w.y1)),
            };
            let zoom = a.zoom.unwrap_or(1.);
            anyhow::ensure!(zoom > 0., "zoom {zoom}: above 1 zooms in, below 1 out");
            let (hw, hh) = (0.5 * w.width() / zoom, 0.5 * w.height() / zoom);
            window = Rect::from_corners(cx - hw, cy - hh, cx + hw, cy + hh);
        }
        let (w, h) = (window.width(), window.height());
        anyhow::ensure!(
            w.is_finite() && h.is_finite() && w > 0. && h > 0.,
            "window {:?}: no area to draw",
            rect(window)
        );
        Ok(window)
    }

    /// The window in a frame `width` wide, as tall as the window's shape.
    fn viewport(&self, width: usize) -> Viewport {
        let w = width.clamp(1, MAX_SIDE);
        Viewport::fit(self.window, w, self.window.height_at(w).min(MAX_SIDE))
    }

    fn style(&self) -> Style<'_> {
        Style {
            layer: self.layer,
            edges: self.edges,
            focus: self.focus(),
            theme: self.base.theme(),
            point: self.point,
        }
    }

    fn render(&self, vp: &Viewport) -> super::render::Frame {
        self.base
            .render(self.level(), &self.style(), self.gene.as_ref(), vp)
    }

    /// The view as someone following the server sees it (`pinto view
    /// --follow`): `view_map`'s arguments that draw it again, with the
    /// run they are of.
    fn followed(&self) -> Value {
        let comm = &self.level().comm;
        let focus: Vec<String> = super::focused(self.focus())
            .iter()
            .map(|&c| super::focus_name(comm, c))
            .collect();
        let round = match drawn(self.show, self.at, self.round.as_ref()) {
            (_, _, Some(path)) => Some(shown_path(path)),
            _ => None,
        };
        json!({
            "run": shown_path(Path::new(&self.base.run.prefix)),
            "level": self.base.run.levels[self.at].tag,
            "round": round,
            "show": super::value_name(&self.show),
            "layer": self.layer.to_string(),
            "focus": focus,
            "gene": self.gene.as_ref().map_or("", |g| &g.feature),
            "source": super::value_name(&self.source),
            "clip": self.clip,
            "edges": self.edges,
            "point_size": self.point,
            "bbox": rect(self.window),
        })
    }

    /// What the map shows: enough to read it and to ask for another.
    fn state(&self, vp: &Viewport) -> Value {
        let comm = &self.level().comm;
        let theme = self.base.theme();
        let ramp = |title: String, layer: Layer| {
            json!({
                "ramp": title,
                "low": hex(layer.ramp(theme).at(0.)),
                "high": hex(layer.ramp(theme).at(1.)),
            })
        };
        let legend = match (&self.gene, self.layer) {
            (Some(g), _) => {
                let mut legend = ramp(g.title(), Layer::Gene);
                legend["top"] = g.top_label().into();
                legend
            }
            (None, Layer::Argmax | Layer::Soft) => json!({"groups": self.groups()}),
            (None, layer) => ramp(layer.legend_title(), layer),
        };
        let focus: Vec<String> = super::focused(self.focus())
            .iter()
            .map(|&c| super::focus_name(comm, c))
            .collect();
        json!({
            "run": self.base.run.name(),
            "level": self.base.run.levels[drawn(self.show, self.at, self.round.as_ref()).1].tag,
            "show": super::value_name(&self.show),
            "round": self.round.as_ref().map(|r| shown_path(&r.path)),
            "k": comm.k,
            "layer": self.layer.to_string(),
            "focus": focus,
            "gene": self.gene.as_ref().map(|g| json!({
                "feature": g.feature, "source": g.source.name(), "clip": g.clip,
            })),
            "edges": self.edges,
            "point_size": self.point,
            "theme": super::value_name(&theme),
            "image": [vp.w, vp.h],
            "window": rect(vp.window()),
            "units_per_px": vp.upp,
            "background": hex(theme.background()),
            "dimmed": hex(theme.dimmed()),
            "legend": legend,
        })
    }

    fn markers(&mut self, a: MarkersArgs) -> anyhow::Result<Vec<Content>> {
        let id = group_id(&a.group)?;
        let focus = super::parse_focus(std::slice::from_ref(&id), &self.level().comm)?;
        let c = focus.iter().position(|&f| f).expect("one group named");
        // A round's groupings come with theirs.
        let (base, level) = self.parts();
        base.load_features(level)?;
        let level = self.level();
        let rates = level.features.as_ref().expect("just loaded");
        let top: Vec<Value> = rates
            .top(c, a.n.unwrap_or(markers::TOP))
            .into_iter()
            .map(|m| json!({"feature": m.name, "symbol": markers::symbol(&m.name), "fold": m.fold}))
            .collect();
        Ok(vec![Content::Json(json!({
            "group": super::focus_name(&level.comm, c),
            "name": level.comm.name(c),
            "cells": level.comm.sizes[c],
            "fold": "rate in the group over the mean rate in the others, among features at or above the group's median rate",
            "markers": top,
        }))])
    }

    fn cell(&mut self, a: CellArgs) -> anyhow::Result<Vec<Content>> {
        let geom = &self.base.geom;
        let i = match (a.name, a.px, a.at) {
            (Some(name), None, None) => *geom
                .index
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("no cell {name:?}"))?,
            (None, Some([px, py]), None) => {
                let vp = self
                    .drawn
                    .ok_or_else(|| anyhow::anyhow!("px: no map drawn yet; call view_map"))?;
                self.nearest(vp.to_world(px, py), self.base.spacing.max(2. * vp.upp))?
            }
            (None, None, Some([x, y])) => self.nearest((x, y), self.base.spacing)?,
            _ => anyhow::bail!("give one of name, px or at"),
        };
        let comm = &self.level().comm;
        let group = comm.cluster[i];
        let props: Vec<Value> = comm
            .top_props(i, 3)
            .into_iter()
            .map(|(c, p)| json!({"group": super::focus_name(comm, c), "p": p}))
            .collect();
        Ok(vec![Content::Json(json!({
            "cell": geom.names[i],
            "batch": (geom.tiles.len() > 1).then(|| &geom.tiles[geom.batch[i] as usize].name),
            "at": [geom.x[i], geom.y[i]],
            "group": (group != NO_CLUSTER).then(|| super::focus_name(comm, group as usize)),
            "propensity": props,
            "entropy": comm.entropy.as_ref().map(|h| h[i] as f32 / 255.),
        }))])
    }

    fn nearest(&self, at: (f32, f32), radius: f32) -> anyhow::Result<usize> {
        let geom = &self.base.geom;
        self.base
            .grid
            .nearest(&geom.x, &geom.y, at, radius)
            .ok_or_else(|| anyhow::anyhow!("no cell within {radius} of {at:?}"))
    }

    fn export(&mut self, a: ExportArgs) -> anyhow::Result<Vec<Content>> {
        anyhow::ensure!(a.png.is_some() || a.pdf.is_some(), "give png, pdf or both");
        let paths: Vec<&Path> = [a.png.as_deref(), a.pdf.as_deref()]
            .into_iter()
            .flatten()
            .map(Path::new)
            .collect();
        // Only figures are written, so no other file (a script, a config,
        // a run's .json) can be written over.
        for (path, ext) in [(&a.png, "png"), (&a.pdf, "pdf")] {
            if let Some(path) = path {
                let ok = Path::new(path)
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case(ext));
                anyhow::ensure!(ok, "{ext} {path:?}: a .{ext} file");
            }
        }
        for p in &paths {
            anyhow::ensure!(
                a.overwrite || !p.exists(),
                "{} exists; give another path, or overwrite",
                shown_path(p)
            );
        }
        // The figure lists the focused groups' markers.
        if self.focus().is_some() && a.pdf.is_some() {
            let (base, level) = self.parts();
            base.load_features(level)?;
        }
        let vp = self.viewport(a.width.unwrap_or(2400));
        super::write_outputs(
            &self.base,
            self.level(),
            &self.style(),
            self.gene.as_ref(),
            &vp,
            a.png.as_deref().map(Path::new),
            a.pdf.as_deref().map(Path::new),
        )?;
        let shown = |p: Option<&String>| p.map(|p| shown_path(Path::new(p)));
        let (png, pdf) = (shown(a.png.as_ref()), shown(a.pdf.as_ref()));
        let round = match drawn(self.show, self.at, self.round.as_ref()) {
            (show, _, Some(path)) => Some((lupin::absolute(path), show)),
            _ => None,
        };
        let still = super::Still {
            base: &self.base,
            prefix: &shown_path(Path::new(&self.base.run.prefix)),
            units: "auto",
            level: self.level(),
            style: self.style(),
            round: round.as_ref().map(|(path, show)| (path.as_path(), *show)),
            gene: self.gene.as_ref(),
            vp,
            window: vp.window(),
        };
        Ok(vec![Content::Json(json!({
            "wrote": png.iter().chain(&pdf).collect::<Vec<_>>(),
            "image": [vp.w, vp.h],
            // What draws the same files again, without an agent.
            "command": still.command(png.as_deref(), pdf.as_deref()),
        }))])
    }
}

/// What a `view_map` call changes the view to.
struct Next {
    at: usize,
    show: Show,
    /// A round newly read, shown from now on.
    round: Option<Round>,
    theme: Option<Theme>,
    layer: Layer,
    focus: Vec<bool>,
    edges: bool,
    point: f32,
    source: Source,
    clip: f32,
    /// The gene map from now on, when it is another.
    gene: Option<Option<GeneMap>>,
    window: Rect,
}

/// What the map draws, as told apart from what else it could: what is
/// shown, of which level, from which round.
fn drawn(show: Show, at: usize, round: Option<&Round>) -> (Show, usize, Option<&Path>) {
    match (show, round) {
        (Show::Types | Show::Clusters, Some(r)) => (show, r.level, Some(&r.path)),
        _ => (Show::Communities, at, None),
    }
}

/// `v` as `--focus` spells a group: a name, or a number.
fn group_id(v: &Value) -> anyhow::Result<Box<str>> {
    match v {
        Value::String(s) => Ok(s.as_str().into()),
        Value::Number(n) => Ok(n.to_string().into()),
        _ => anyhow::bail!("group {v}: a name or a number"),
    }
}

fn value_enum<T: ValueEnum>(s: &str, what: &str) -> anyhow::Result<T> {
    T::from_str(s, true).map_err(|_| {
        let names: Vec<String> = T::value_variants().iter().map(super::value_name).collect();
        anyhow::anyhow!("{what} {s:?}: one of {}", names.join(", "))
    })
}

fn rect(r: Rect) -> [f32; 4] {
    [r.x0, r.y0, r.x1, r.y1]
}

/// `p` as results give paths: from the root.
fn shown_path(p: &Path) -> String {
    lupin::absolute(p).to_string_lossy().into_owned()
}

/// For the agent: what the viewer is for and the order of calls.
pub const INSTRUCTIONS: &str = "\
pinto: spatial link-community runs on the tissue. Start with view_open on a run \
prefix (the -o of a pinto fit, or its .pinto.json): it lists the run's levels, \
batches, groups and any lupin rounds. view_map draws the map as a PNG with its \
legend; its arguments stay set until changed, so pass only what changes. \
view_markers lists a group's top features, view_cell names the cell under a \
pixel of the last map (or at a world point, or by name), and view_export writes \
the current view as PNG/PDF files. World coordinates are those view_open \
reports, with y running down the map. lupin rounds are read here as they are on \
disk; annotating and relabelling them is lupin's own server.";

fn specs() -> Vec<ToolSpec> {
    let spec = ToolSpec::new;
    let group = json!({
        "type": ["string", "integer"],
        "description": "C3 or 3 for a community; a lupin cluster id (K3 or 3) or a cell type when a round is shown",
    });
    vec![
        spec(
            "view_open",
            "Open a pinto run and summarize it: cells, coordinates, bounds, batches, \
             levels, the level's groups with their map colours, data files and lupin rounds. \
             Opening another run closes the one open.",
            json!({
                "type": "object",
                "properties": {
                    "prefix": {"type": "string", "description": "Output prefix of the run (the -o it was fit with) or its .pinto.json"},
                    "level": {"type": "string", "description": "Level tag to show (L1, L2, …, final) [default: the last]"},
                },
                "required": ["prefix"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_map",
            "Draw the map as a PNG, with the state and legend as JSON. Every argument is \
             optional and stays set until changed; with none, the same view is drawn again.",
            json!({
                "type": "object",
                "properties": {
                    "level": {"type": "string", "description": "Level tag; shows its communities"},
                    "round": {"type": "string", "description": "A lupin round ({out}.lupin.json); shows its clusters unless show says otherwise"},
                    "show": {"enum": ["communities", "types", "clusters"], "description": "What groups the cells; types and clusters are the round's (the newest round when none is given)"},
                    "layer": {"type": "string", "description": "argmax (most likely group), soft (colours mixed by propensity), entropy (propensity entropy / ln K), or C<k> (one community's propensity)"},
                    "focus": {"type": "array", "items": group, "description": "Groups to show; the rest are dimmed. [] shows every group"},
                    "gene": {"type": "string", "description": "Colour cells by one feature, by symbol or full name; \"\" shows the groups again"},
                    "source": {"enum": ["observed", "expected"], "description": "The gene's observed ln(1+count), or the model's expected level"},
                    "clip": {"type": "number", "minimum": 50, "maximum": 100, "description": "Top of the gene's colour ramp, a percentile of its positive values [default: 99]"},
                    "edges": {"type": "boolean", "description": "Draw adjacent cell pairs by link community, where zoomed in far enough"},
                    "point_size": {"type": "number", "minimum": 0.25, "maximum": 8, "description": "Cell disc size, × the default"},
                    "theme": {"enum": ["dark", "light"], "description": "Map colours for a dark or light background [default: dark]"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "World window [x0, y0, x1, y1]"},
                    "fit": {"type": "boolean", "description": "true: the whole tissue"},
                    "batch": {"type": "string", "description": "Fit one batch, by the name view_open lists"},
                    "center": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "Move the window's centre to this world point"},
                    "zoom": {"type": "number", "exclusiveMinimum": 0, "description": "Magnify the window about its centre: 2 zooms in, 0.5 out"},
                    "width": {"type": "integer", "minimum": 1, "maximum": MAX_SIDE, "description": "Image width in pixels, this call only [default: 1024]"},
                },
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_markers",
            "A group's top marker features, by fold over the other groups.",
            json!({
                "type": "object",
                "properties": {
                    "group": group,
                    "n": {"type": "integer", "minimum": 1, "description": "How many [default: 12]"},
                },
                "required": ["group"],
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_cell",
            "The cell nearest a pixel of the last map, a world point, or by name: \
             its batch, position, group, strongest propensities and entropy.",
            json!({
                "type": "object",
                "properties": {
                    "px": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[column, row] pixel of the last view_map image"},
                    "at": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[x, y] world point"},
                    "name": {"type": "string", "description": "Cell name"},
                },
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_export",
            "Write the current view as a PNG and/or a one-page PDF figure (title, scale \
             bar, legend, focused groups' markers). Refuses to replace a file unless overwrite.",
            json!({
                "type": "object",
                "properties": {
                    "png": {"type": "string", "description": "PNG path, ending .png"},
                    "pdf": {"type": "string", "description": "PDF path, ending .pdf"},
                    "width": {"type": "integer", "minimum": 1, "maximum": MAX_SIDE, "description": "Image width in pixels [default: 2400]"},
                    "overwrite": {"type": "boolean", "description": "Replace files that exist"},
                },
                "additionalProperties": false,
            }),
        ),
    ]
}
