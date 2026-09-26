//! `pinto view`: multi-resolution viewer for a pinto run.
//!
//! Loading ([`data`]) joins the run's parquet outputs onto one cell geometry;
//! [`index`] buckets cells into a grid and summarizes it as a pyramid, and
//! [`render`] draws a viewport from them, so a frame costs about the number
//! of screen pixels, not the number of cells.

mod color;
mod data;
mod index;
mod render;

#[cfg(test)]
mod tests;

use clap::Args;
use data::{Communities, Edges, Geometry, Rect, Run};
use index::{EdgeIndex, Grid, Pyramid};
use render::{Layer, Scene, Style, Viewport};
use std::time::{Duration, Instant};

/// Average cells per finest grid bin.
const CELLS_PER_BIN: f32 = 6.;

#[derive(Args, Debug)]
pub struct ViewArgs {
    #[arg(
        help = "Run prefix (as given to -o) or its .pinto.json",
        long_help = "Output prefix of a pinto run (the -o it was fit with),\n\
                     or the path of its {prefix}.pinto.json manifest."
    )]
    pub prefix: Box<str>,

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

    #[arg(long, default_value_t = 2400, help = "PNG width in pixels")]
    pub width: usize,

    #[arg(
        long,
        help = "PNG height in pixels [default: from the view's aspect ratio]"
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
        help = "World window to draw (tiled coordinates; see --summary)"
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
}

pub fn run_view(args: &ViewArgs) -> anyhow::Result<()> {
    if args.summary {
        return summarize(args);
    }
    match args.png.as_deref() {
        Some(out) => write_png(args, out),
        None => {
            anyhow::bail!("the interactive viewer is not built yet; use --summary or --png FILE")
        }
    }
}

/// A run loaded and indexed at one level.
struct Loaded {
    geom: Geometry,
    comm: Communities,
    edges: Option<(Edges, EdgeIndex)>,
    grid: Grid,
    pyramid: Pyramid,
    spacing: f32,
}

impl Loaded {
    fn load(run: &Run, level: Option<&str>, with_edges: bool) -> anyhow::Result<Self> {
        let level = run.level(level)?;
        let t = Instant::now();
        let geom = run.load_geometry()?;
        log::info!("cells: {} in {:.2?}", geom.n(), t.elapsed());

        let t = Instant::now();
        let comm = run.load_communities(&geom, level)?;
        log::info!("level {}: K={} in {:.2?}", comm.tag, comm.k, t.elapsed());

        let grid = Grid::build(&geom.x, &geom.y, geom.bounds(), CELLS_PER_BIN);
        let pyramid = Pyramid::build(&grid, &comm);

        let edges = if with_edges {
            let t = Instant::now();
            let edges = run.load_edges(&geom, level)?;
            let index = EdgeIndex::build(&grid, &geom.x, &geom.y, &edges);
            log::info!("edges: {} in {:.2?}", edges.len(), t.elapsed());
            Some((edges, index))
        } else {
            None
        };

        // Spacing over occupied bins only: the bounding box of a tissue
        // section is mostly empty around the edges.
        let occupied = pyramid.levels[0].count.iter().filter(|&&c| c > 0).count();
        let area = occupied as f32 * grid.bin * grid.bin;
        let spacing = (area / geom.n().max(1) as f32).sqrt();

        Ok(Loaded {
            geom,
            comm,
            edges,
            grid,
            pyramid,
            spacing,
        })
    }

    fn scene(&self) -> Scene<'_> {
        Scene {
            geom: &self.geom,
            comm: &self.comm,
            grid: &self.grid,
            pyramid: &self.pyramid,
            edges: self.edges.as_ref().map(|(e, i)| (e, i)),
            spacing: self.spacing,
        }
    }
}

fn write_png(args: &ViewArgs, out: &str) -> anyhow::Result<()> {
    let run = Run::open(&args.prefix)?;
    let loaded = Loaded::load(&run, args.level.as_deref(), args.edges)?;
    if let Layer::Community(c) = args.layer {
        anyhow::ensure!(c < loaded.comm.k, "C{c}: level has K={}", loaded.comm.k);
    }

    let window = match args.bbox.as_deref() {
        Some(&[x0, y0, x1, y1]) => Rect {
            x0: x0.min(x1),
            y0: y0.min(y1),
            x1: x0.max(x1),
            y1: y0.max(y1),
        },
        Some(v) => anyhow::bail!("--bbox takes 4 numbers X0,Y0,X1,Y1, got {}", v.len()),
        None => loaded.geom.bounds(),
    };
    let w = args.width.max(1);
    let h = args.height.unwrap_or_else(|| {
        ((w as f32 * window.height() / window.width().max(f32::MIN_POSITIVE)).round() as usize)
            .max(1)
    });
    let vp = Viewport::fit(window, w, h);
    let palette = color::palette(loaded.comm.k);
    let style = Style {
        layer: args.layer,
        edges: args.edges,
    };

    let t = Instant::now();
    let frame = render::render(&loaded.scene(), &vp, &style, &palette);
    let took = t.elapsed();
    frame.write_png(std::path::Path::new(out))?;
    println!(
        "wrote {out}: {w}×{h}, {:.3} units/px, rendered in {took:.2?}",
        vp.upp
    );
    if matches!(args.layer, Layer::Argmax | Layer::Soft) {
        print_legend(&loaded.comm, &palette);
    }
    Ok(())
}

fn print_legend(comm: &Communities, palette: &[color::Rgb]) {
    let sizes = comm.sizes();
    for (c, &[r, g, b]) in palette.iter().enumerate() {
        if sizes[c] > 0 {
            println!("  C{c:<3} #{r:02x}{g:02x}{b:02x} {:>9} cells", sizes[c]);
        }
    }
}

fn summarize(args: &ViewArgs) -> anyhow::Result<()> {
    let run = Run::open(&args.prefix)?;
    let meta = &run.meta;
    println!(
        "run      {} ({} v{})",
        run.manifest.display(),
        meta.command,
        meta.version
    );
    println!(
        "manifest {} cells, {} features, {} edges",
        meta.n_cells,
        meta.n_features,
        meta.n_edges.map_or("?".to_string(), |e| e.to_string())
    );
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
    if geom.tiles.len() > 1 {
        println!("batches  {} (tiled):", geom.tiles.len());
        for tile in &geom.tiles {
            println!("  {:<20} {:>9} cells", tile.name, tile.n_cells);
        }
    }

    let level = run.level(args.level.as_deref())?;
    let t = Instant::now();
    let comm = run.load_communities(&geom, level)?;
    let t_comm = t.elapsed();
    print_communities(&comm, t_comm);

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
    let sizes = comm.sizes();
    let used = sizes.iter().filter(|&&s| s > 0).count();
    println!(
        "level    {}: K={} ({} non-empty), entropy {}, loaded in {:.2?}",
        comm.tag,
        comm.k,
        used,
        if comm.entropy.is_some() { "yes" } else { "no" },
        took
    );
    if comm.n_missing > 0 || comm.n_unmatched > 0 {
        println!(
            "         {} cells without propensity, {} rows matching no cell",
            comm.n_missing, comm.n_unmatched
        );
    }
    let mut order: Vec<usize> = (0..comm.k).collect();
    order.sort_by_key(|&c| std::cmp::Reverse(sizes[c]));
    let top: Vec<String> = order
        .iter()
        .take(8)
        .filter(|&&c| sizes[c] > 0)
        .map(|&c| format!("C{c}:{}", sizes[c]))
        .collect();
    println!("largest  {}", top.join(" "));
}
