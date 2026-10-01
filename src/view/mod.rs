//! `pinto view`: multi-resolution viewer for a pinto run.
//!
//! Loading ([`data`]) joins the run's parquet outputs onto one cell geometry;
//! [`index`] buckets cells into a grid and summarizes it as a pyramid, and
//! [`render`] draws a viewport from them, so a frame costs about the number
//! of screen pixels, not the number of cells.

mod cellart;
mod color;
mod data;
mod draft;
mod gene;
mod heatmap;
mod index;
mod kitty;
mod lupin;
mod markers;
mod pdf;
mod render;
mod round;
mod saved;
mod scalebar;
mod structure;
mod tui;

#[cfg(test)]
mod tests;

use clap::Args;
use color::Theme;
use data::{Communities, Edges, Geometry, Rect, Run};
use gene::{Expression, GeneMap, Source};
use index::{EdgeIndex, Grid, Pyramid};
use markers::FeatureRates;
use render::{Frame, Layer, Scene, Style, Viewport};
use scalebar::Units;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

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

#[derive(Args, Clone, Debug)]
pub struct ViewArgs {
    #[arg(
        help = "Run prefix (as given to -o) or its .pinto.json [default: browse]",
        long_help = "Output prefix of a pinto run (the -o it was fit with),\n\
                     or the path of its {prefix}.pinto.json manifest. Without it,\n\
                     the terminal viewer opens a browser to pick a run."
    )]
    pub prefix: Option<Box<str>>,

    #[arg(
        long,
        help = "Community level to show (e.g. L2, final)",
        long_help = "Community level to show, by the tag the manifest lists\n\
                     (L1, L2, ... for cascade levels, final for the last).\n\
                     Defaults to the last level."
    )]
    pub level: Option<Box<str>>,

    #[arg(long, help = "Print what the run holds and exit")]
    pub summary: bool,

    #[arg(
        long,
        value_name = "FILE",
        help = "Draw one view to a PNG file and exit",
        long_help = "Draw one view to a PNG file and exit, without a terminal.\n\
                     The view is the whole run, or --bbox."
    )]
    pub png: Option<Box<str>>,

    #[arg(
        long,
        value_name = "FILE",
        help = "Draw one view to a PDF figure and exit",
        long_help = "Draw one view to a one-page PDF figure and exit: the map as an\n\
                     image, with title, scale bar, legend and (with --focus) the\n\
                     focused communities' markers as vector text. Combines with --png."
    )]
    pub pdf: Option<Box<str>>,

    #[arg(
        long,
        default_value = "auto",
        help = "Coordinate units for the scale bar: auto, um, px, or none",
        long_help = "Coordinate units for the scale bar:\n\
                     \x20 auto  px for Space Ranger pxl_* columns, µm for Xenium\n\
                     \x20       centroids, otherwise a bare number\n\
                     \x20 um    micrometres (1000 µm shows as 1 mm)\n\
                     \x20 px    image pixels\n\
                     \x20 none  no scale bar"
    )]
    pub units: Box<str>,

    #[arg(
        long,
        value_name = "FEATURE",
        help = "Colour cells by one feature, e.g. CD3E (--png/--pdf)",
        long_help = "Colour cells by one feature instead of communities: its symbol\n\
                     (CD3E) or full name (ENSG00000198851_CD3E). Shows the observed\n\
                     ln(1+count) from the run's data files, or, with --expected or\n\
                     when no data file is found, the community model's expected level.\n\
                     In the terminal, click a marker in the panel instead."
    )]
    pub gene: Option<Box<str>>,

    #[arg(
        long,
        help = "With --gene, show the model-expected level, not observed counts"
    )]
    pub expected: bool,

    #[arg(
        long,
        default_value_t = 99.,
        value_name = "PERCENTILE",
        value_parser = parse_clip,
        help = "Top of a gene's colour ramp: this percentile of its positive values",
        long_help = "Top of a gene's colour ramp, as a percentile of its positive\n\
                     values; cells above it show the top colour, so a few extreme\n\
                     cells do not darken the rest. 95 clips harder, 100 is the maximum.\n\
                     In the terminal, p switches between 99 and 95."
    )]
    pub clip: f32,

    #[arg(long, default_value_t = 2400, help = "Image width in pixels")]
    pub width: usize,

    #[arg(
        long,
        help = "Image height in pixels [default: from the view's aspect ratio]"
    )]
    pub height: Option<usize>,

    #[arg(
        long,
        default_value = "argmax",
        help = "What colours cells: argmax, soft, entropy, or C<k>",
        long_help = "What colours cells:\n\
                     \x20 argmax   the cell's most likely community\n\
                     \x20 soft     community colours mixed by propensity\n\
                     \x20 entropy  propensity entropy / ln K (viridis)\n\
                     \x20 C<k>     community k's propensity (magma), e.g. C3"
    )]
    pub layer: Layer,

    #[arg(
        long,
        value_delimiter = ',',
        allow_negative_numbers = true,
        value_name = "X0,Y0,X1,Y1",
        help = "World window to draw (tiled coordinates; see --summary)",
        long_help = "World window to draw, in the tiled coordinates --summary reports.\n\
                     Write --bbox=X0,Y0,X1,Y1 when X0 is negative, so it is not\n\
                     read as a flag."
    )]
    pub bbox: Option<Vec<f32>>,

    #[arg(
        long,
        help = "Draw edges coloured by link community",
        long_help = "Draw adjacent cell pairs coloured by their link community.\n\
                     They appear only where the view is zoomed in far enough\n\
                     for an edge to span a few pixels."
    )]
    pub edges: bool,

    #[arg(
        long,
        default_value_t = 1.,
        value_name = "X",
        value_parser = parse_point_size,
        help = "Cell disc size, × the default",
        long_help = "Cell disc size, as a multiple of the default (half the typical\n\
                     cell spacing across). Above 1, discs are also drawn from further\n\
                     out, where cells would otherwise be averaged per pixel.\n\
                     In the terminal, < and > change it."
    )]
    pub point_size: f32,

    #[arg(
        long,
        value_delimiter = ',',
        value_name = "C1,C7,...",
        help = "Show only these communities; dim the rest (--png)",
        long_help = "Show only these communities, comma separated (C7 or 7);\n\
                     other cells are dimmed and their edges hidden.\n\
                     In the terminal, click a cell or a legend entry instead."
    )]
    pub focus: Option<Vec<Box<str>>>,

    #[arg(
        long,
        value_name = "N",
        num_args = 0..=1,
        default_missing_value = "8",
        help = "With --summary, list each community's top N marker features",
        long_help = "With --summary, list each community's top N marker\n\
                     features [default N: 8]. A feature's fold is its rate in\n\
                     the community over its mean rate in the other ones,\n\
                     among features at or above the community's median rate."
    )]
    pub markers: Option<usize>,

    #[arg(
        long,
        default_value_t = 2,
        value_name = "N",
        help = "In the terminal, `s` exports the view at N × screen resolution"
    )]
    pub export_scale: usize,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "How the map is drawn in the terminal",
        long_help = "How the map is drawn in the terminal:\n\
                     \x20 auto          ask the terminal (default); quadrants when it\n\
                     \x20               has no graphics protocol, and inside tmux\n\
                     \x20 kitty         kitty graphics; pixels go through a temp file,\n\
                     \x20               or inline when over ssh (kitty, Ghostty, WezTerm)\n\
                     \x20 kitty-inline  kitty graphics, always inline\n\
                     \x20               (in tmux, through passthrough, switched on for the\n\
                     \x20               viewer's pane; best effort)\n\
                     \x20 sixel         sixel graphics (iTerm2, WezTerm, foot, xterm)\n\
                     \x20 iterm2        iTerm2 inline images\n\
                     \x20 quadrants     block characters, 2×2 pixels per character;\n\
                     \x20               any terminal\n\
                     \x20 symbols       block characters with boundaries at eighths of a\n\
                     \x20               character: finer shapes, mixed colours averaged\n\
                     \x20 blocks        half-blocks, 1×2 pixels per character"
    )]
    pub graphics: Graphics,

    #[arg(
        long,
        value_name = "FILE",
        help = "Open this lupin round ({out}.lupin.json) [default: the newest]",
        long_help = "Open this lupin annotation round ({out}.lupin.json). Without it\n\
                     the terminal viewer finds the newest round made from this run\n\
                     (press a to show it); --png/--pdf need it to draw one."
    )]
    pub round: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value = "communities",
        help = "What groups cells: communities, types or clusters (--round)",
        long_help = "What groups cells:\n\
                     \x20 communities  the level's communities\n\
                     \x20 types        the lupin round's cell types\n\
                     \x20 clusters     the lupin round's clusters (merges applied)"
    )]
    pub show: Show,

    #[arg(
        long,
        value_enum,
        help = "Map colours for a dark or light background [default: the terminal's]",
        long_help = "Map colours for a dark or a light background: background,\n\
                     dimmed tissue, palette, ramps and scale bar. The terminal viewer\n\
                     defaults to the terminal's own background; --png and --pdf\n\
                     default to dark."
    )]
    pub theme: Option<Theme>,
}

/// What groups the cells on the map (`--show`).
#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
pub enum Show {
    /// The level's communities.
    Communities,
    /// A lupin round's cell types.
    Types,
    /// A lupin round's clusters.
    Clusters,
}

impl ViewArgs {
    /// The run, once [`run_view`] has made sure there is one.
    pub fn prefix(&self) -> &str {
        self.prefix.as_deref().expect("run_view picks a run first")
    }
}

/// How the map reaches the terminal (`--graphics`).
#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
pub enum Graphics {
    Auto,
    Kitty,
    KittyInline,
    Sixel,
    Iterm2,
    Quadrants,
    Symbols,
    Blocks,
}

pub fn run_view(args: &ViewArgs) -> anyhow::Result<()> {
    if args.prefix.is_none() {
        anyhow::ensure!(
            !args.summary && args.png.is_none() && args.pdf.is_none(),
            "--summary, --png and --pdf need the run: pinto view <prefix>"
        );
        let Some(path) = tui::pick_run()? else {
            return Ok(());
        };
        let mut args = args.clone();
        args.prefix = Some(path.to_string_lossy().into());
        return run_view(&args);
    }
    if args.summary {
        return summarize(args);
    }
    if args.png.is_some() || args.pdf.is_some() {
        write_still(args)
    } else {
        tui::run(args)
    }
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
    /// Colours of every frame and palette drawn from this run.
    theme: Theme,
}

impl Base {
    fn load(prefix: &str, units: &str, theme: Theme) -> anyhow::Result<Self> {
        let run = Run::open(prefix)?;
        let t = Instant::now();
        let geom = run.load_geometry()?;
        let units = Units::parse(units, &geom.coord_names)?;
        log::info!("cells: {} in {:.2?}", geom.n(), t.elapsed());
        let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), CELLS_PER_BIN);

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
            theme,
        })
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
                    theme: self.theme,
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
        let palette = self.theme.palette(comm.k);
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
        let palette = self.theme.palette(comm.k);
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

fn write_still(args: &ViewArgs) -> anyhow::Result<()> {
    // Without a terminal to ask, `auto` means dark.
    let theme = args.theme.unwrap_or(Theme::Dark);
    let base = Base::load(args.prefix(), &args.units, theme)?;
    let mut level = match args.show {
        Show::Communities => {
            base.level(base.run.level_index(args.level.as_deref())?, args.edges)?
        }
        show => {
            let path = args.round.as_deref().ok_or_else(|| {
                anyhow::anyhow!("--show {show:?} needs --round FILE (a lupin round)")
            })?;
            let round = base.round(std::path::Path::new(path))?;
            if show == Show::Types {
                round.types
            } else {
                round.clusters
            }
        }
    };
    if let Layer::Community(c) = args.layer {
        anyhow::ensure!(c < level.comm.k, "C{c}: level has K={}", level.comm.k);
    }

    let window = match args.bbox.as_deref() {
        Some(&[x0, y0, x1, y1]) => Rect {
            x0: x0.min(x1),
            y0: y0.min(y1),
            x1: x0.max(x1),
            y1: y0.max(y1),
        },
        Some(v) => anyhow::bail!("--bbox takes 4 numbers X0,Y0,X1,Y1, got {}", v.len()),
        None => base.geom.bounds().pad(FIT_MARGIN),
    };
    let w = args.width.max(1);
    let h = args.height.unwrap_or_else(|| {
        ((w as f32 * window.height() / window.width().max(f32::MIN_POSITIVE)).round() as usize)
            .max(1)
    });
    let vp = Viewport::fit(window, w, h);
    let focus = args
        .focus
        .as_deref()
        .map(|ids| parse_focus(ids, &level.comm))
        .transpose()?;
    if focus.is_some() && args.pdf.is_some() {
        base.load_features(&mut level)?;
    }
    let style = Style {
        layer: args.layer,
        edges: args.edges,
        focus: focus.as_deref(),
        theme,
        point: args.point_size,
    };
    let gene = match args.gene.as_deref() {
        Some(feature) => {
            let source = if args.expected {
                Source::Expected
            } else {
                Source::Observed
            };
            let g = base.gene(&mut level, feature, source, args.clip)?;
            if g.fell_back(source) {
                eprintln!("no data file found for this run; showing the model-expected level");
            }
            Some(g)
        }
        None => None,
    };

    let t = Instant::now();
    write_outputs(
        &base,
        &level,
        &style,
        gene.as_ref(),
        &vp,
        args.png.as_deref().map(Path::new),
        args.pdf.as_deref().map(Path::new),
    )?;
    let took = t.elapsed();
    for out in [args.png.as_deref(), args.pdf.as_deref()]
        .into_iter()
        .flatten()
    {
        println!("wrote {out}");
    }
    println!("{w}×{h}, {:.3} units/px, in {took:.2?}", vp.upp);
    if matches!(args.layer, Layer::Argmax | Layer::Soft) {
        for &c in &level.comm.by_size {
            println!("{}", legend_line(&level.comm, c, level.palette[c]));
        }
    }
    Ok(())
}

/// Render `vp` once and write it as a PDF figure and/or a PNG. The PDF
/// takes the map clean and draws its own vector scale bar; the PNG gets the
/// bar burnt in afterwards, on the same buffer.
fn write_outputs(
    base: &Base,
    level: &Level,
    style: &Style,
    gene: Option<&GeneMap>,
    vp: &Viewport,
    png: Option<&Path>,
    pdf_path: Option<&Path>,
) -> anyhow::Result<()> {
    let mut frame = base.render(level, style, gene, vp);
    if let Some(path) = pdf_path {
        pdf::write(&figure(base, level, style, gene, vp, &frame), path)?;
    }
    if let Some(path) = png {
        if let Some(units) = base.units {
            scalebar::draw(&mut frame, vp, units, true);
        }
        frame.write_png(path)?;
    }
    Ok(())
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
fn legend_line(comm: &Communities, c: usize, [r, g, b]: color::Rgb) -> String {
    let n = comm.sizes[c];
    format!("  {:<5} #{r:02x}{g:02x}{b:02x} {n:>9} cells", comm.name(c))
}

/// The PDF page for a rendered view: title, legend for the layer (or the
/// gene), and the markers of each focused community.
fn figure<'a>(
    base: &Base,
    level: &Level,
    style: &Style,
    gene: Option<&GeneMap>,
    vp: &Viewport,
    frame: &'a Frame,
) -> pdf::Figure<'a> {
    let comm = &level.comm;
    let focused = focused(style.focus);
    let mut subtitle = format!(
        "{} cells · K={} · {:.3} units/px",
        thousands(base.geom.n()),
        comm.k,
        vp.upp
    );
    if !focused.is_empty() {
        subtitle.push_str(&format!(" · focus {}", ids(comm, &focused, " ")));
    }

    let ramp = |title: String, layer: Layer, top: String| pdf::Legend::Ramp {
        title,
        stops: (0..64)
            .map(|i| layer.ramp(base.theme).at(i as f32 / 63.))
            .collect(),
        top,
    };
    let legend = match (gene, style.layer) {
        (Some(g), _) => ramp(g.title(), Layer::Gene, g.top_label()),
        (None, Layer::Argmax | Layer::Soft) => pdf::Legend::Communities(
            comm.by_size
                .iter()
                .map(|&c| {
                    let (colour, on) = legend_swatch(c, &level.palette, style.focus, base.theme);
                    pdf::Entry {
                        label: comm.name(c),
                        count: comm.sizes[c],
                        colour,
                        on,
                    }
                })
                .collect(),
        ),
        (None, layer) => ramp(layer.legend_title(), layer, "1".into()),
    };

    let markers = match level.features.as_ref() {
        Some(rates) => focused
            .iter()
            .take(4)
            .map(|&c| pdf::MarkerBlock {
                title: format!("{} markers", comm.name(c)),
                colour: level.palette[c],
                genes: rates
                    .top(c, 12)
                    .into_iter()
                    .map(|m| (markers::symbol(&m.name).to_string(), m.fold))
                    .collect(),
            })
            .collect(),
        None => Vec::new(),
    };

    pdf::Figure {
        frame,
        vp: *vp,
        title: {
            let what = gene.map_or_else(|| style.layer.to_string(), GeneMap::title);
            format!("{} · {} · {what}", base.run.name(), comm.tag)
        },
        subtitle,
        units: base.units,
        legend,
        markers,
    }
}

/// `814243` → `814,243`.
fn thousands(n: usize) -> String {
    indicatif::HumanCount(n as u64).to_string()
}

fn parse_clip(s: &str) -> Result<f32, String> {
    match s.parse::<f32>() {
        Ok(p) if (50. ..=100.).contains(&p) => Ok(p),
        _ => Err(format!("{s:?}: a percentile from 50 to 100")),
    }
}

/// Smallest and largest `--point-size`.
const POINT_SIZES: std::ops::RangeInclusive<f32> = 0.25..=8.;

fn parse_point_size(s: &str) -> Result<f32, String> {
    match s.parse::<f32>() {
        Ok(x) if POINT_SIZES.contains(&x) => Ok(x),
        _ => Err(format!(
            "{s:?}: a size from {} to {}",
            POINT_SIZES.start(),
            POINT_SIZES.end()
        )),
    }
}

/// `["C7", "3"]` → one flag per community; a round's groups by name
/// (`K3`, a cell type).
fn parse_focus(given: &[Box<str>], comm: &Communities) -> anyhow::Result<Vec<bool>> {
    let k = comm.k;
    let mut focus = vec![false; k];
    for id in given {
        let c = if comm.names.is_some() {
            (0..k)
                .find(|&c| focus_name(comm, c) == id.as_ref())
                .ok_or_else(|| anyhow::anyhow!("--focus: no group {id:?} in this round"))?
        } else {
            let c = render::community_id(id)
                .ok_or_else(|| anyhow::anyhow!("--focus: {id:?} is not a community (C7 or 7)"))?;
            anyhow::ensure!(c < k, "--focus: C{c}, but the level has K={k}");
            c
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
