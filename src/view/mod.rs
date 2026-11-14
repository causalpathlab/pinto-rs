//! `pinto view`: multi-resolution viewer for a pinto run.
//!
//! Loading ([`data`]) joins the run's parquet outputs onto one cell geometry;
//! [`index`] buckets cells into a grid and summarizes it as a pyramid, and
//! [`render`] draws a viewport from them, so a frame costs about the number
//! of screen pixels, not the number of cells.

mod args;
mod color;
mod curve;
pub(crate) mod data;
mod draft;
mod gene;
mod heatmap;
mod index;
mod lra;
mod lupin;
mod markers;
pub mod mcp;
mod pdf;
mod render;
mod round;
mod saved;
mod scalebar;
mod still;
mod structure;
mod tui;

#[cfg(test)]
mod tests;

use args::{check_clip, check_point_size, POINT_SIZES};
pub use args::{Show, ViewArgs};
use color::Theme;
use data::{Communities, Edges, Geometry, Run};
use gene::{Expression, GeneMap, Source};
use index::{EdgeIndex, Grid, Pyramid};
use markers::FeatureRates;
use render::{Frame, Layer, Scene, Style, Viewport};
use scalebar::Units;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use still::{write_outputs, write_still, Still, StillPair};

/// Average cells per finest grid bin.
const CELLS_PER_BIN: f32 = 6.;

/// A gene map draws as its single pseudo-community's level, every cell shown.
/// Community `c`'s legend swatch, and whether it is shown: its colour, or
/// the dimmed tissue colour when other communities are focused.
fn legend_swatch(
    c: usize,
    palette: &[color::Rgb],
    focus: Option<&[bool]>,
    theme: Theme,
) -> (color::Rgb, bool) {
    let on = focus.is_none_or(|f| f[c]);
    (if on { palette[c] } else { theme.dimmed() }, on)
}

/// Margin around the whole tissue when a view fits it, per side, as a
/// fraction of its larger extent. An explicit `--bbox` gets none.
const FIT_MARGIN: f32 = 0.01;

/// An agent's view on `pinto mcp`, as `pinto view --follow` draws it: the
/// run, and `view_map`'s settings by its own names and values. A `gene` or
/// `round` of none means there is none, where `view_map` keeps what it has.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Followed {
    run: String,
    /// The level whose communities are shown, when they are.
    level: String,
    round: Option<String>,
    show: Show,
    layer: String,
    focus: Vec<Box<str>>,
    gene: Option<String>,
    source: Source,
    clip: f32,
    edges: bool,
    point_size: f32,
    bbox: [f32; 4],
}

pub fn run_view(args: &ViewArgs) -> anyhow::Result<()> {
    // A summary or a still prints and exits; the rest is on screen.
    if args.summary || args.png.is_some() || args.pdf.is_some() {
        anyhow::ensure!(
            args.prefix.is_some(),
            "--summary, --png and --pdf need the run: pinto view <prefix>"
        );
        return if args.summary {
            summarize(args)
        } else {
            write_still(args)
        };
    }
    // An agent's view: its server, and the run it has open, found before
    // the session so a miss is said plainly.
    let with;
    let (args, follow) = match &args.follow {
        None => (args, None),
        Some(server) => {
            let (live, run) = tui::pick_server(server)?;
            with = ViewArgs {
                prefix: Some(run.into()),
                ..args.clone()
            };
            (&with, Some(live))
        }
    };
    // On screen, the session first. Hosted, this process only relays the
    // terminal from here on, and the program the host starts reads the
    // run; with none named, live views are offered before a new one.
    let session = crate::tui::TOOL.session("view", &args.session.options(args.prefix.is_some()))?;
    // One screen for the run browser and the view after it.
    let modes = crate::tui::modes(legume_tui::Mouse::Off);
    crate::tui::with_terminal(&session, modes, |terminal| {
        let picked;
        let args = match &args.prefix {
            Some(_) => args,
            None => {
                let Some(path) = tui::pick_run(&session, terminal)? else {
                    return Ok(());
                };
                let mut with = args.clone();
                with.prefix = Some(path.to_string_lossy().into());
                picked = with;
                &picked
            }
        };
        tui::run(&session, terminal, args, follow.as_ref())
    })
}

/// A run's cell geometry and grid, shared by every level.
struct Base {
    run: Run,
    geom: Geometry,
    grid: Grid,
    /// Scale bar units; `None` draws no bar.
    units: Option<Units>,
    /// Typical distance between neighbouring cells, world units.
    spacing: f32,
    /// The run's expression data, opened on the first observed gene.
    expression: OnceLock<Result<Option<Expression>, String>>,
    /// Colours of every frame and palette drawn from this run; the view
    /// takes another when a terminal of another background attaches.
    theme: std::sync::RwLock<Theme>,
}

impl Base {
    fn load(prefix: &str, units: &str, theme: Theme) -> anyhow::Result<Self> {
        let run = Run::open(prefix)?;
        let t = Instant::now();
        let geom = run.load_geometry()?;
        log::info!("cells: {} in {:.2?}", geom.n(), t.elapsed());
        let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), CELLS_PER_BIN);
        let coord_file = run.single_coord_file();
        let units =
            Units::for_coordinates(units, &geom.coord_names, coord_file.as_deref(), || {
                grid.pitch(&geom.x, &geom.y)
            })?;

        // Spacing over occupied bins only: the bounding box of a tissue
        // section is mostly empty around the edges.
        let area = grid.occupied_bins() as f32 * grid.bin * grid.bin;
        let spacing = (area / geom.n().max(1) as f32).sqrt();
        Ok(Base {
            run,
            geom,
            grid,
            units,
            spacing,
            expression: OnceLock::new(),
            theme: theme.into(),
        })
    }

    /// The colours frames and palettes are drawn in now.
    fn theme(&self) -> Theme {
        *self.theme.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Draw in `theme` from here on; the palettes already made keep theirs.
    fn set_theme(&self, theme: Theme) {
        *self.theme.write().unwrap_or_else(|e| e.into_inner()) = theme;
    }

    /// The expression data, opened once; `None` when the run names no data
    /// file and none sits next to its outputs.
    fn expression(&self) -> anyhow::Result<Option<&Expression>> {
        let opened = self.expression.get_or_init(|| {
            let files = self.run.data_files();
            if files.is_empty() {
                return Ok(None);
            }
            let t = Instant::now();
            let expr = Expression::open(&files, &self.geom).map_err(|e| e.to_string())?;
            log::info!("data: {} in {:.2?}", files.join(", "), t.elapsed());
            Ok(Some(expr))
        });
        opened
            .as_ref()
            .map(Option::as_ref)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// `feature` on the map, observed when asked and the data files are
    /// there, model-expected otherwise; the map says which it is.
    fn gene(
        &self,
        level: &mut Level,
        feature: &str,
        source: Source,
        clip: f32,
    ) -> anyhow::Result<GeneMap> {
        if source == Source::Observed {
            if let Some(expr) = self.expression()? {
                return expr.observed(feature, &self.geom, clip, &self.grid);
            }
        }
        self.load_features(level)?;
        let rates = level.features.as_ref().expect("just loaded");
        GeneMap::expected(feature, rates, &level.comm, clip, &self.grid)
    }

    /// Render `vp`: the level's communities in `style`, or `gene`.
    fn render(&self, level: &Level, style: &Style, gene: Option<&GeneMap>, vp: &Viewport) -> Frame {
        match gene {
            Some(g) => {
                // The gene map's values in place of the level's communities,
                // every cell shown.
                let scene = Scene {
                    comm: &g.comm,
                    pyramid: &g.pyramid,
                    edges: None,
                    ..self.scene(level)
                };
                let style = Style {
                    layer: Layer::Gene,
                    edges: false,
                    focus: None,
                    theme: self.theme(),
                    point: style.point,
                };
                render::render(&scene, vp, &style, &[[255; 3]])
            }
            None => render::render(&self.scene(level), vp, style, &level.palette),
        }
    }

    /// Load level `i` of `run.levels`, with its edges when asked.
    fn level(&self, i: usize, with_edges: bool) -> anyhow::Result<Level> {
        let info = &self.run.levels[i];
        let t = Instant::now();
        let comm = self.run.load_communities(&self.geom, info)?;
        log::info!("level {}: K={} in {:.2?}", comm.tag, comm.k, t.elapsed());
        let pyramid = Pyramid::build(&self.grid, &comm);
        let palette = self.theme().palette(comm.k);
        let mut level = Level {
            index: i,
            comm,
            pyramid,
            palette,
            edges: None,
            features: None,
            grouping: false,
        };
        if with_edges {
            self.load_edges(&mut level)?;
        }
        Ok(level)
    }

    fn load_edges(&self, level: &mut Level) -> anyhow::Result<()> {
        if level.edges.is_none() && !level.grouping {
            let t = Instant::now();
            let edges = self
                .run
                .load_edges(&self.geom, &self.run.levels[level.index])?;
            let index = EdgeIndex::build(&self.grid, &self.geom.x, &self.geom.y, &edges);
            log::info!("edges: {} in {:.2?}", edges.len(), t.elapsed());
            level.edges = Some((edges, index));
        }
        Ok(())
    }

    fn load_features(&self, level: &mut Level) -> anyhow::Result<()> {
        if level.features.is_none() {
            level.features = Some(self.run.load_feature_rates(&self.run.levels[level.index])?);
        }
        Ok(())
    }

    /// Every chain's newest lupin round made from this run, newest first.
    fn rounds(&self) -> Vec<std::path::PathBuf> {
        self.run
            .manifest
            .as_deref()
            .map(lupin::latest_rounds)
            .unwrap_or_default()
    }

    /// The lupin round at `path`, on the level it annotated.
    fn round(&self, path: &Path) -> anyhow::Result<round::Round> {
        let tags: Vec<&str> = self.run.levels.iter().map(|l| l.tag.as_str()).collect();
        let level = lupin::round_level(path, &tags)
            .and_then(|t| tags.iter().position(|&x| x == t))
            .unwrap_or(tags.len() - 1);
        let t = Instant::now();
        let round = round::Round::load(path, level, self)?;
        log::info!(
            "round {}: {} clusters in {:.2?}",
            path.display(),
            round.ids.len(),
            t.elapsed()
        );
        Ok(round)
    }

    /// A grouping that is not one of pinto's levels (a lupin round's cell
    /// types or clusters) as a level drawn like the others. `index` is the
    /// level it was made from; its marker rates come with it, never from
    /// the level's own files.
    fn grouping(&self, index: usize, comm: Communities, rates: Option<FeatureRates>) -> Level {
        let pyramid = Pyramid::build(&self.grid, &comm);
        let palette = self.theme().palette(comm.k);
        let features = rates.unwrap_or_else(|| FeatureRates {
            names: Vec::new(),
            rates: crate::util::common::Mat::zeros(0, comm.k),
        });
        Level {
            index,
            comm,
            pyramid,
            palette,
            edges: None,
            features: Some(features),
            grouping: true,
        }
    }

    fn scene<'a>(&'a self, level: &'a Level) -> Scene<'a> {
        Scene {
            geom: &self.geom,
            comm: &level.comm,
            grid: &self.grid,
            pyramid: &level.pyramid,
            edges: level.edges.as_ref().map(|(e, i)| (e, i)),
            spacing: self.spacing,
        }
    }
}

/// The grouping drawn: a round's cell types or clusters when `show` asks
/// for them and a round is open, else community level `at`.
fn shown<'a>(
    levels: &'a [Option<Level>],
    round: Option<&'a round::Round>,
    show: Show,
    at: usize,
) -> &'a Level {
    match (show, round) {
        (Show::Types, Some(r)) => &r.types,
        (Show::Clusters, Some(r)) => &r.clusters,
        _ => levels[at].as_ref().expect("the level shown is loaded"),
    }
}

/// [`shown`], to load into or redraw.
fn shown_mut<'a>(
    levels: &'a mut [Option<Level>],
    round: Option<&'a mut round::Round>,
    show: Show,
    at: usize,
) -> &'a mut Level {
    match (show, round) {
        (Show::Types, Some(r)) => &mut r.types,
        (Show::Clusters, Some(r)) => &mut r.clusters,
        _ => levels[at].as_mut().expect("the level shown is loaded"),
    }
}

/// Draw in `theme` from here on: every palette of `levels` and `round`
/// made again. False when it was the theme already.
fn repalette(
    base: &Base,
    theme: Theme,
    levels: &mut [Option<Level>],
    round: Option<&mut round::Round>,
) -> bool {
    if theme == base.theme() {
        return false;
    }
    base.set_theme(theme);
    let round = round
        .into_iter()
        .flat_map(|r| [&mut r.clusters, &mut r.types]);
    for level in levels.iter_mut().flatten().chain(round) {
        level.palette = theme.palette(level.comm.k);
    }
    true
}

/// One community level, loaded on demand.
struct Level {
    /// Position in `run.levels`.
    index: usize,
    comm: Communities,
    pyramid: Pyramid,
    palette: Vec<color::Rgb>,
    edges: Option<(Edges, EdgeIndex)>,
    /// Feature rates per community, loaded when markers are first asked for.
    features: Option<FeatureRates>,
    /// A lupin round's grouping rather than the level's communities: it
    /// has no edges of its own.
    grouping: bool,
}

/// Communities whose focus flag is set; empty with no focus.
fn focused(focus: Option<&[bool]>) -> Vec<usize> {
    focus.map_or_else(Vec::new, |f| (0..f.len()).filter(|&c| f[c]).collect())
}

/// `[3, 17]` → `C3{sep}C17`; a round's groups by their `--focus` names
/// (`K3`, or the cell type).
fn ids(comm: &Communities, cs: &[usize], sep: &str) -> String {
    cs.iter()
        .map(|&c| focus_name(comm, c))
        .collect::<Vec<_>>()
        .join(sep)
}

/// How `--focus` names group `c`: `C3`, a round cluster's `K3`, a type.
fn focus_name(comm: &Communities, c: usize) -> String {
    match comm.ids.as_ref().and_then(|ids| ids.get(c)) {
        Some(id) => id.to_string(),
        None => comm.name(c),
    }
}

/// One legend row: name, hex colour, cell count.
fn legend_line(comm: &Communities, c: usize, colour: color::Rgb) -> String {
    let n = comm.sizes[c];
    format!("  {:<5} {} {n:>9} cells", comm.name(c), color::hex(colour))
}

/// How the command line spells `v`.
fn value_name<T: clap::ValueEnum>(v: &T) -> String {
    v.to_possible_value()
        .expect("no skipped variants")
        .get_name()
        .to_string()
}

/// `814243` → `814,243`.
fn thousands(n: usize) -> String {
    indicatif::HumanCount(n as u64).to_string()
}

/// `["C7", "3"]` → one flag per community; a `round`'s groups by name
/// (`K3`, a cell type). A level whose communities the manifest names
/// takes the names, and `C7` or `7` too.
fn parse_focus(given: &[Box<str>], comm: &Communities, round: bool) -> anyhow::Result<Vec<bool>> {
    let k = comm.k;
    let mut focus = vec![false; k];
    for id in given {
        // A round's cluster by its bare id too, as lupin gives it.
        let cluster = comm.ids.is_some() && id.parse::<i64>().is_ok();
        let id: Box<str> = if cluster {
            format!("K{id}").into()
        } else {
            id.clone()
        };
        let named = comm
            .names
            .as_ref()
            .and_then(|_| (0..k).find(|&c| focus_name(comm, c) == id.as_ref()));
        let c = match named {
            Some(c) => c,
            None if round => anyhow::bail!("--focus: no group {id:?} in this round"),
            None => {
                let c = render::community_id(&id).ok_or_else(|| match comm.names {
                    Some(_) => {
                        anyhow::anyhow!("--focus: no group {id:?} in this level (a name, C7 or 7)")
                    }
                    None => anyhow::anyhow!("--focus: {id:?} is not a community (C7 or 7)"),
                })?;
                anyhow::ensure!(c < k, "--focus: C{c}, but the level has K={k}");
                c
            }
        };
        focus[c] = true;
    }
    Ok(focus)
}

fn summarize(args: &ViewArgs) -> anyhow::Result<()> {
    let run = Run::open(args.prefix())?;
    let meta = &run.meta;
    println!("run      {}", run.source());
    if run.manifest.is_some() {
        println!(
            "manifest {} v{}: {} cells, {} features, {} edges",
            meta.command,
            meta.version,
            meta.n_cells,
            meta.n_features,
            meta.n_edges.map_or("?".to_string(), |e| e.to_string())
        );
    }
    let tags: Vec<&str> = run.levels.iter().map(|l| l.tag.as_str()).collect();
    println!("levels   {}", tags.join(" "));

    let t = Instant::now();
    let geom = run.load_geometry()?;
    let t_geom = t.elapsed();
    let b = geom.bounds();
    println!(
        "cells    {} loaded in {:.2?}; coords [{}], bounds x {:.1}..{:.1}, y {:.1}..{:.1}",
        geom.n(),
        t_geom,
        geom.coord_names.join(", "),
        b.x0,
        b.x1,
        b.y0,
        b.y1
    );
    let outside = geom.n_outside_graph();
    if outside > 0 {
        println!("         {outside} outside the graph (dropped by QC, or without neighbours)");
    }
    if geom.tiles.len() > 1 {
        println!("batches  {} (tiled):", geom.tiles.len());
        for tile in &geom.tiles {
            println!("  {:<20} {:>9} cells", tile.name, tile.n_cells);
        }
    }

    let level = &run.levels[run.level_index(args.level.as_deref())?];
    let t = Instant::now();
    let comm = run.load_communities(&geom, level)?;
    let t_comm = t.elapsed();
    print_communities(&comm, t_comm);
    if let Some(n) = args.markers {
        let rates = run.load_feature_rates(level)?;
        println!("markers  (fold over the other communities)");
        for &c in &comm.by_size {
            println!("  C{c:<3} {}", rates.summary(c, n));
        }
    }

    let t = Instant::now();
    match run.load_edges(&geom, level) {
        Ok(edges) => println!(
            "edges    {} adjacent pairs in {:.2?} ({} unmatched)",
            edges.len(),
            t.elapsed(),
            edges.n_unmatched
        ),
        Err(e) => println!("edges    unavailable: {e}"),
    }

    let t = Instant::now();
    let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), CELLS_PER_BIN);
    let t_grid = t.elapsed();
    let t = Instant::now();
    let pyr = Pyramid::build(&grid, &comm);
    println!(
        "index    grid {}×{} bins of {:.2} ({:.2?}); pyramid {} levels, {:.1} MB ({:.2?})",
        grid.nx,
        grid.ny,
        grid.bin,
        t_grid,
        pyr.levels.len(),
        pyr.bytes() as f64 / 1e6,
        t.elapsed()
    );
    Ok(())
}

fn print_communities(comm: &Communities, took: Duration) {
    println!(
        "level    {}: K={} ({} non-empty), entropy {}, loaded in {:.2?}",
        comm.tag,
        comm.k,
        comm.by_size.len(),
        if comm.entropy.is_some() { "yes" } else { "no" },
        took
    );
    if comm.n_missing > 0 || comm.n_unmatched > 0 {
        println!(
            "         {} graph cells without propensity, {} rows matching no cell",
            comm.n_missing, comm.n_unmatched
        );
    }
    let top: Vec<String> = comm
        .by_size
        .iter()
        .take(8)
        .map(|&c| format!("C{c}:{}", comm.sizes[c]))
        .collect();
    println!("largest  {}", top.join(" "));
}
