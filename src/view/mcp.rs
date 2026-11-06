//! The viewer's tools for `pinto mcp`: open a run, draw it, ask about its
//! groups and cells, write figures. One run and one view at a time, held
//! between calls as the terminal viewer holds them: `view_map` changes
//! only what it is given, then draws the map.
//!
//! lupin rounds are read as they are on disk; making them is lupin's.

use super::color::{hex, Theme};
use super::data::{Rect, NO_CLUSTER};
use super::gene::{GeneMap, Source};
use super::lra::{self as lra_layer, LraView, PairShow, Sort};
use super::render::{Layer, Style, Viewport};
use super::round::{round_name, Round};
use super::{lupin, markers, scalebar, Base, Level, Show, FIT_MARGIN};
use crate::lr_activity::ligand_effect::output::{Verdict, Q_SCOPE_ALL};
use crate::lr_activity::outputs::opt_finite;
use crate::mcp::{parse, shown_path};
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
    /// The view last published to followers, so an unchanged one is not
    /// published again.
    published: Option<super::Followed>,
}

/// The server's name, which `pinto view --follow` looks for.
pub const SERVER: &str = "pinto";

impl Tools {
    /// Publish the open view to followers, when it changed.
    fn publish(&mut self, cx: &Call) {
        let Some(view) = self.open.as_ref().map(Open::followed) else {
            return;
        };
        if self.published.as_ref() != Some(&view) {
            cx.state(&serde_json::to_value(&view).expect("plain fields"));
            self.published = Some(view);
        }
    }
}

/// An open run and the view on it.
pub(super) struct Open {
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
    /// The lra layer, read on the first `view_lra`.
    pub(super) lra: Option<LraView>,
    /// Whether the map draws the lra pair selected, in place of the groups.
    pair_drawn: bool,
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
            self.open = Some(open);
            self.publish(cx);
            return Ok(vec![Content::Json(summary)]);
        }
        let open = self
            .open
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no run open: call view_open first"))?;
        match name {
            "view_map" => {
                let shown = open.map(parse(args)?)?;
                self.publish(cx);
                Ok(shown)
            }
            "view_markers" => open.markers(parse(args)?),
            "view_cell" => open.cell(parse(args)?),
            "view_lra" => open.lra(parse(args)?, cx),
            "view_export" => open.export(parse(args)?, cx),
            _ => unreachable!("the server answers unknown tools itself"),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenArgs {
    prefix: String,
    level: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(super) struct MapArgs {
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
    curve: bool,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(super) struct LraArgs {
    table: Option<String>,
    verdict: Option<String>,
    search: Option<String>,
    sort: Option<String>,
    limit: Option<usize>,
    pair: Option<String>,
    show: Option<String>,
}

/// Pairs `view_lra` lists without a limit.
const LIST: usize = 20;

impl Open {
    pub(super) fn new(prefix: &str, level: Option<&str>) -> anyhow::Result<Self> {
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
            lra: None,
            pair_drawn: false,
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

    pub(super) fn map(&mut self, a: MapArgs) -> anyhow::Result<Vec<Content>> {
        let next = self.next(&a)?;
        self.commit(next);
        // Asking for groups, a layer or a gene draws them instead of the pair.
        let regroup = a.level.is_some()
            || a.round.is_some()
            || a.show.is_some()
            || a.layer.is_some()
            || a.focus.is_some()
            || a.gene.is_some();
        if regroup {
            self.pair_drawn = false;
        }
        if let Some(theme) = a.theme.as_deref() {
            let theme: Theme = value_enum(theme, "theme")?;
            if let Some(v) = self.lra.as_mut() {
                v.retheme(theme);
            }
        }
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
        if let Some(on) = self.pair_on_map() {
            let lra = self.lra.as_ref().expect("a pair is drawn");
            let (level, style, gene) = on.drawn(&self.base, lra.show, self.point);
            return self.base.render(level, &style, gene, vp);
        }
        self.base
            .render(self.level(), &self.style(), self.gene.as_ref(), vp)
    }

    /// The lra pair the map draws, when it draws one.
    fn pair_on_map(&self) -> Option<&lra_layer::OnMap> {
        self.lra.as_ref().filter(|_| self.pair_drawn)?.shown()
    }

    /// The view as someone following the server sees it.
    fn followed(&self) -> super::Followed {
        let comm = &self.level().comm;
        let round = match drawn(self.show, self.at, self.round.as_ref()) {
            (_, _, Some(path)) => Some(shown_path(path)),
            _ => None,
        };
        super::Followed {
            run: shown_path(Path::new(&self.base.run.prefix)),
            level: self.base.run.levels[self.at].tag.clone(),
            round,
            show: self.show,
            layer: self.layer.to_string(),
            focus: super::focused(self.focus())
                .iter()
                .map(|&c| super::focus_name(comm, c).into())
                .collect(),
            gene: self.gene.as_ref().map(|g| g.feature.to_string()),
            source: self.source,
            clip: self.clip,
            edges: self.edges,
            point_size: self.point,
            bbox: rect(self.window),
        }
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
        if let Some(on) = self.pair_on_map() {
            let show = self.lra.as_ref().map_or(PairShow::Classes, |v| v.show);
            let legend = match show {
                PairShow::Classes => json!({"groups": on.legend().iter().map(|(name, colour, n)| {
                    json!({"group": name, "colour": hex(*colour), "cells": n})
                }).collect::<Vec<_>>()}),
                PairShow::Exposure => {
                    let exposure = on.exposure(&self.base);
                    json!({
                        "ramp": exposure.title(),
                        "low": hex(Layer::Gene.ramp(theme).at(0.)),
                        "high": hex(Layer::Gene.ramp(theme).at(1.)),
                        "top": exposure.top_label(),
                    })
                }
            };
            return json!({
                "run": self.base.run.name(),
                "pair": format!("{}:{}", on.ligand, on.receptor),
                "pair_show": super::value_name(&show),
                "threshold": on.map.threshold,
                "competence": on.map.competence,
                "point_size": self.point,
                "theme": super::value_name(&theme),
                "image": [vp.w, vp.h],
                "window": rect(vp.window()),
                "units_per_px": vp.upp,
                "background": hex(theme.background()),
                "no_receiver": hex(theme.no_community()),
                "legend": legend,
                "note": "the lra pair is drawn; view_map with level, round, show, layer, focus or gene draws groups again, view_lra pair: \"\" too",
            });
        }
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

    /// `view_lra`: the run's contact-effect tables and their pairs, listed
    /// as asked; with `pair`, that pair on the map from now on. What it is
    /// given stays set, as `view_map`'s arguments do.
    fn lra(&mut self, a: LraArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
        let reply = self.lra_with(a, &|msg| cx.progress(msg))?;
        Ok(vec![Content::Json(reply)])
    }

    /// [`Open::lra`]'s work and reply, telling `progress` what it reads.
    pub(super) fn lra_with(
        &mut self,
        a: LraArgs,
        progress: &dyn Fn(&str),
    ) -> anyhow::Result<Value> {
        // As the manifest says now: a run_lra may have ended since it was opened.
        let run = &self.base.run;
        let tables =
            super::data::Run::open(&run.prefix).map_or_else(|_| run.tables(), |now| now.tables());
        match self.lra.as_mut() {
            Some(v) => v.add_tables(tables),
            None => {
                let view = LraView::new(tables)?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "{} has no lra table: run_lra on {} first",
                        run.name(),
                        run.manifest
                            .as_deref()
                            .map_or_else(|| run.prefix.clone(), shown_path)
                    )
                })?;
                self.lra = Some(view);
            }
        }
        // What changes, checked before any of it is kept.
        let v = self.lra.as_mut().expect("read above");
        let mut filter = v.filter().clone();
        if let Some(code) = a.verdict.as_deref() {
            filter.verdict = match code {
                "" | "all" => None,
                code => Some(code.parse().map_err(|_| {
                    let codes: Vec<&str> = Verdict::ALL.iter().map(|v| v.code()).collect();
                    anyhow::anyhow!(
                        "verdict {code:?}: one of {}, or \"\" for all",
                        codes.join(", ")
                    )
                })?),
            };
        }
        if let Some(search) = &a.search {
            filter.search = search.trim().to_string();
        }
        if let Some(sort) = a.sort.as_deref() {
            filter.sort = value_enum::<Sort>(sort, "sort")?;
        }
        let show = a
            .show
            .as_deref()
            .map(|s| value_enum::<PairShow>(s, "show"))
            .transpose()?;
        let staged = match a.table.as_deref() {
            Some(t) => Some(match find_table(&v.tables, t) {
                Some(at) => v.stage(lra_layer::Which::At(at))?,
                None => v.stage(lra_layer::Which::Path(Box::new(lra_layer::TableRef::at(
                    Path::new(t),
                    &self.base.run,
                ))))?,
            }),
            None => None,
        };
        // The pair selected from now on, in the table shown from now on.
        let table = staged.as_ref().map_or_else(|| v.table(), |s| v.staged(s));
        let selected = match a.pair.as_deref() {
            Some(name) => LraView::resolve(table, name)?,
            None => v.selection_in(table),
        };
        // A pair asked for is drawn, and one drawn stays drawn in the
        // table switched to: drawn before anything is kept, as it is the
        // one thing left that can fail.
        let draw = selected.is_some() && (a.pair.is_some() || self.pair_drawn);
        if let (true, Some(i)) = (draw, selected) {
            let (row, options) = (table.rows[i].clone(), table.r.options.clone());
            if v.needs_receivers_for(&options) {
                progress("reading the run's receivers as lra saw them");
            }
            v.draw(&self.base, &options, &row, self.at)?;
        }
        if let Some(s) = staged {
            v.commit(s);
        }
        v.selected = selected;
        self.pair_drawn = draw;
        v.set_filter(filter);
        if let Some(show) = show {
            v.show = show;
        }
        Ok(self.lra_reply(a.limit.unwrap_or(LIST)))
    }

    /// The lra layer as `view_lra` says it: the table, the verdict counts,
    /// the pairs listed, and the pair selected.
    fn lra_reply(&self, limit: usize) -> Value {
        let v = self.lra.as_ref().expect("the layer is read");
        let table = v.table();
        let rows = &table.rows;
        let counts: serde_json::Map<String, Value> =
            Verdict::counts(rows.iter().map(|r| r.verdict))
                .iter()
                .map(|(v, n)| (v.code().into(), (*n).into()))
                .collect();
        let shown = v.visible();
        let list: Vec<Value> = shown
            .iter()
            .take(limit)
            .map(|&i| {
                let r = &rows[i];
                json!({
                    "pair": r.name(),
                    "verdict": r.verdict.code(),
                    "change": lra_layer::change(r),
                    "q": opt_finite(r.effect_q),
                    "marked": v.marked.contains(&r.name()),
                })
            })
            .collect();
        let q_scope: String = match &table.q_scope {
            _ if !table.subset_corrected() => Q_SCOPE_ALL.into(),
            Some(scope) => scope.to_string(),
            None => "q corrected within the --only subset".into(),
        };
        let selected = v.selected_row().map(|r| {
            let mut out = json!({
                "pair": r.name(),
                "verdict": r.verdict.code(),
                "note": r.note,
                "warnings": r.warnings,
                "change": lra_layer::change(r),
                "effect": opt_finite(r.effect),
                "effect_se": opt_finite(r.effect_se),
                "q": opt_finite(r.effect_q),
                "coexpr_q": opt_finite(r.coexpr_q),
                "subset_corrected": table.subset_corrected(),
                "threshold": opt_finite(r.threshold),
                "receivers": r.n_receivers,
                "competent": r.n_competent,
                "exposed": r.n_exposed,
                "curve_points": table.curve(v.selected.expect("selected")).len(),
            });
            if let Some(on) = v.shown() {
                let classes: serde_json::Map<_, _> = on
                    .legend()
                    .iter()
                    .map(|(name, _, n)| ((*name).to_string(), Value::from(*n)))
                    .collect();
                let count = |f: &[bool]| f.iter().filter(|&&x| x).count();
                out["map"] = json!({
                    "classes": classes,
                    "exposed": count(&on.map.exposed),
                    "competent": count(&on.map.competent),
                    "threshold": on.map.threshold,
                    "competence": on.map.competence,
                });
                out["drawn"] = self.pair_drawn.into();
            }
            out
        });
        let f = v.filter();
        json!({
            "table": table.r.label(),
            "path": shown_path(&table.r.path),
            "q_scope": q_scope,
            "tables": v.tables.iter().map(|t| json!({"table": t.label(), "path": shown_path(&t.path)})).collect::<Vec<_>>(),
            "pairs": rows.len(),
            "verdicts": counts,
            "filter": {
                "verdict": f.verdict.map(|v| v.code()),
                "search": f.search,
                "sort": f.sort.name(),
            },
            "shown": shown.len(),
            "list": list,
            "selected": selected,
            "show": super::value_name(&v.show),
            "reading": "change: the headline effect as a percent change (exp(effect) - 1) with \
                        its 95% CI; q: Benjamini-Hochberg over the table's pairs (q_scope). With \
                        a pair, view_map draws it: senders (ligand+), receivers exposed or not \
                        (contact exposure above threshold) and receptor+ or not; show: exposure \
                        draws the exposure itself. view_export curve: true writes its \
                        threshold-response curve.",
        })
    }

    fn nearest(&self, at: (f32, f32), radius: f32) -> anyhow::Result<usize> {
        let geom = &self.base.geom;
        self.base
            .grid
            .nearest(&geom.x, &geom.y, at, radius)
            .ok_or_else(|| anyhow::anyhow!("no cell within {radius} of {at:?}"))
    }

    /// Write the view's figures by running the `pinto view` command that
    /// draws them, so the project's records (`runall.sh`, `INDEX.tsv`)
    /// can make them again.
    fn export(&mut self, a: ExportArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
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
        let mut vp = self.viewport(a.width.unwrap_or(2400));
        let shown = |p: Option<&String>| p.map(|p| shown_path(Path::new(p)));
        let (png, pdf) = (shown(a.png.as_ref()), shown(a.pdf.as_ref()));
        // The pair drawn, or with `curve` the selected pair's curve.
        let lra = self.lra.as_ref();
        let selected = lra.and_then(LraView::selected_row);
        anyhow::ensure!(
            !a.curve || selected.is_some(),
            "curve: no lra pair selected; view_lra pair first"
        );
        let pair = lra
            .filter(|_| a.curve || self.pair_drawn)
            .and_then(|v| v.still_pair(a.curve));
        if a.curve {
            let w = vp.w;
            vp = Viewport::fit(vp.window(), w, (w * 3 / 5).max(1));
        }
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
            pair,
            vp,
            window: vp.window(),
        };
        let wrote: Vec<&String> = png.iter().chain(&pdf).collect();
        let summary = match &still.pair {
            Some(p) if p.curve => format!("lra curve of {} on {}", p.name, self.base.run.name()),
            Some(p) => format!("lra pair {} on {}", p.name, self.base.run.name()),
            None => format!(
                "{} of {} {}",
                still.style.layer,
                self.base.run.name(),
                self.level().comm.tag
            ),
        };
        let outputs: Vec<std::path::PathBuf> = wrote.iter().map(Into::into).collect();
        let args = still.args(png.as_deref(), pdf.as_deref());
        cx.run_making(legume_mcp::own_command(&args)?, &outputs, &summary)?;
        Ok(vec![Content::Json(json!({
            "wrote": wrote,
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

/// The table `name` names among `tables`: `full`, or a subset by its
/// list (`L1,L2`, order and case aside), or by its path.
fn find_table(tables: &[lra_layer::TableRef], name: &str) -> Option<usize> {
    let list: Vec<Box<str>> = name.split(',').map(|s| s.trim().into()).collect();
    let canonical = crate::lr_activity::ligand_effect::only::canonical(&list);
    let path = shown_path(Path::new(name));
    tables.iter().position(|t| {
        shown_path(&t.path) == path
            || match &t.subset {
                None => name.eq_ignore_ascii_case("full"),
                Some(s) => s.only == canonical,
            }
    })
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
                    "layer": {
                        "anyOf": [
                            {"enum": ["argmax", "soft", "entropy"]},
                            {"type": "string", "pattern": "^[Cc]?[0-9]+$"},
                        ],
                        "description": "argmax (most likely group), soft (colours mixed by propensity), entropy (propensity entropy / ln K), or C<k> (one community's propensity, e.g. C3)",
                    },
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
            "view_lra",
            "The run's ligand-receptor tables (pinto lra: the full table and any --only \
             subsets) and their pooled pairs: verdict counts and a list filtered by verdict \
             and gene, sorted by q, effect or name, each with its change (percent, exp(effect) \
             - 1, with 95% CI) and q. With pair, returns that pair's row (verdict, note, \
             threshold, receivers) and view_map draws it from then on: senders (ligand+), \
             receivers exposed or not (contact exposure above the table's threshold) and \
             receptor-competent or not, as lra computed them (show: exposure draws the \
             exposure itself). Arguments stay set until changed. A subset's q is corrected \
             within the subset only (q_scope). view_export curve: true writes the pair's \
             threshold-response curve.",
            json!({
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "full, a subset by its --only list (L1,L2), or a table's path [default: the full table]"},
                    "verdict": {"type": "string", "description": "List only this verdict (supported, effect_without_coexpression, colocation_only, no_evidence, control_failed, celltype_inseparable, insufficient_data); \"\" lists all"},
                    "search": {"type": "string", "description": "Genes: each word must be in the ligand's or receptor's name; \"\" for all"},
                    "sort": {"enum": ["q", "effect", "name"], "description": "Order of the list [default: q]"},
                    "limit": {"type": "integer", "minimum": 0, "description": "Pairs listed, this call only [default: 20]"},
                    "pair": {"type": "string", "description": "LIGAND:RECEPTOR to select and draw on the map; \"\" draws groups again"},
                    "show": {"enum": ["classes", "exposure"], "description": "What the pair's map shows [default: classes]"},
                },
                "additionalProperties": false,
            }),
        ),
        spec(
            "view_export",
            "Write the current view as a PNG and/or a one-page PDF figure (title, scale \
             bar, legend, focused groups' markers); with curve, the selected lra pair's \
             threshold-response curve instead. Refuses to replace a file unless overwrite.",
            json!({
                "type": "object",
                "properties": {
                    "png": {"type": "string", "description": "PNG path, ending .png"},
                    "pdf": {"type": "string", "description": "PDF path, ending .pdf"},
                    "width": {"type": "integer", "minimum": 1, "maximum": MAX_SIDE, "description": "Image width in pixels [default: 2400]"},
                    "overwrite": {"type": "boolean", "description": "Replace files that exist"},
                    "curve": {"type": "boolean", "description": "Write the selected lra pair's threshold-response curve (view_lra pair) instead of the map"},
                },
                "additionalProperties": false,
            }),
        ),
    ]
}
