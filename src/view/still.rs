//! Writing a still figure of a view to a file, without a terminal.

use super::args::{Show, ViewArgs};
use super::color::Theme;
use super::data::Rect;
use super::gene::{GeneMap, Source};
use super::legend_swatch;
use super::markers;
use super::render::{Frame, Layer, Style, Viewport};
use super::thousands;
use super::{
    color, curve, data, focused, ids, legend_line, lra, parse_focus, pdf, scalebar, value_name,
    Base, Level, FIT_MARGIN,
};
use anyhow::Context as _;
use legume_numeric::matrix::common_io::mkdir_parent;
use std::path::Path;
use std::time::Instant;

pub(super) fn write_still(args: &ViewArgs) -> anyhow::Result<()> {
    // Without a terminal to ask, `auto` means dark.
    let theme = args.theme.unwrap_or(Theme::Dark);
    if let Some(pair) = args.pair.as_deref() {
        return write_pair_still(args, theme, pair);
    }
    anyhow::ensure!(
        !args.curve,
        "--curve draws an lra pair's curve: give --pair"
    );
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

    let vp = still_viewport(args, &base)?;
    let focus = args
        .focus
        .as_deref()
        .map(|ids| parse_focus(ids, &level.comm, level.grouping))
        .transpose()?;
    // The PDF lists the focused groups' markers, when the level has them.
    let has_markers = base.run.levels[level.index].feature_community.is_some();
    if focus.is_some() && args.pdf.is_some() && has_markers {
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
    print_written(args);
    let scale = scalebar::per_pixel(base.units, vp.upp);
    println!("{}×{}, {scale}, in {took:.2?}", vp.w, vp.h);
    if matches!(args.layer, Layer::Argmax | Layer::Soft) {
        for &c in &level.comm.by_size {
            println!("{}", legend_line(&level.comm, c, level.palette[c]));
        }
    }
    Ok(())
}

/// The map's viewport for a still: `--bbox`, else the whole tissue, at
/// `--width` and `--height` (else the height that keeps its aspect).
fn still_viewport(args: &ViewArgs, base: &Base) -> anyhow::Result<Viewport> {
    let window = match args.bbox.as_deref() {
        Some(&[x0, y0, x1, y1]) => Rect::from_corners(x0, y0, x1, y1),
        Some(v) => anyhow::bail!("--bbox takes 4 numbers X0,Y0,X1,Y1, got {}", v.len()),
        None => base.geom.bounds().pad(FIT_MARGIN),
    };
    let w = args.width.max(1);
    let h = args.height.unwrap_or_else(|| window.height_at(w));
    Ok(Viewport::fit(window, w, h))
}

/// Say each file a still wrote.
fn print_written(args: &ViewArgs) {
    for out in [args.png.as_deref(), args.pdf.as_deref()]
        .into_iter()
        .flatten()
    {
        println!("wrote {out}");
    }
}

/// The lra layer for `--pair`: the table `--lra-table` names, else the
/// run's first, with `pair` selected.
fn pair_layer(run: &data::Run, table: Option<&str>, pair: &str) -> anyhow::Result<lra::LraView> {
    let tables = match table {
        Some(path) => vec![lra::TableRef::at(Path::new(path), run)],
        None => run.tables(),
    };
    let mut view = lra::LraView::new(tables)?.ok_or_else(|| {
        anyhow::anyhow!(
            "--pair: the run has no lra table; run `pinto lra --from {}` first, or give \
             --lra-table",
            run.manifest
                .as_deref()
                .map_or_else(|| run.prefix.clone(), |m| m.display().to_string())
        )
    })?;
    view.select(pair)?;
    Ok(view)
}

/// `--pair`: the pair on the map, or with `--curve` its curve (which needs
/// only the run, not its cells), as a PNG and/or a PDF figure.
fn write_pair_still(args: &ViewArgs, theme: Theme, pair: &str) -> anyhow::Result<()> {
    let t = Instant::now();
    if args.curve {
        let run = data::Run::open(args.prefix())?;
        let view = pair_layer(&run, args.lra_table.as_deref(), pair)?;
        let curve = view.curve().expect("a pair is selected");
        let w = args.width.max(1);
        let h = args.height.unwrap_or(w * 3 / 5);
        if let Some(path) = args.png.as_deref() {
            writing(Path::new(path), |p| {
                curve::still(&curve, (w, h), theme).write_png(p)
            })?;
        }
        if let Some(path) = args.pdf.as_deref() {
            let page = curve::page(&curve, theme).map_err(anyhow::Error::msg)?;
            writing(Path::new(path), |p| Ok(pdf::write_chart(page, p)?))?;
        }
    } else {
        let base = Base::load(args.prefix(), &args.units, theme)?;
        let mut view = pair_layer(&base.run, args.lra_table.as_deref(), pair)?;
        let level = base.run.level_index(args.level.as_deref())?;
        let on = view.on_map(&base, level)?.expect("a pair is selected");
        let vp = still_viewport(args, &base)?;
        let (level, style, gene) = on.drawn(&base, args.pair_show, args.point_size);
        write_outputs(
            &base,
            level,
            &style,
            gene,
            &vp,
            args.png.as_deref().map(Path::new),
            args.pdf.as_deref().map(Path::new),
        )?;
        if args.pair_show == lra::PairShow::Classes {
            for (name, colour, n) in on.legend() {
                println!("  {name:<22} {} {n:>9} receivers", color::hex(colour));
            }
        }
    }
    print_written(args);
    println!("in {:.2?}", t.elapsed());
    Ok(())
}

/// The lra pair a still draws, to say the command that draws it again.
pub(super) struct StillPair<'a> {
    pub name: String,
    pub show: lra::PairShow,
    /// The table, when it is not the run's first.
    pub table: Option<&'a Path>,
    pub curve: bool,
}

/// One view as `pinto view` draws it without a terminal, to say the
/// command that draws it again.
pub(super) struct Still<'a> {
    pub(super) base: &'a Base,
    pub(super) prefix: &'a str,
    pub(super) units: &'a str,
    pub(super) level: &'a Level,
    pub(super) style: Style<'a>,
    /// The round drawn, and what of it.
    pub(super) round: Option<(&'a Path, Show)>,
    pub(super) gene: Option<&'a GeneMap>,
    /// An lra pair drawn in place of the groups (its exposure map is the
    /// pair's, not `gene`).
    pub(super) pair: Option<StillPair<'a>>,
    pub(super) vp: Viewport,
    pub(super) window: Rect,
}

impl Still<'_> {
    /// The run's level, also under a round's grouping, whose own tag
    /// (`cell types`) is no level `--level` knows.
    pub(super) fn tag(&self) -> &str {
        &self.base.run.levels[self.level.index].tag
    }

    /// The `pinto view` command that draws this view into `png` and `pdf`.
    pub(super) fn command(&self, png: Option<&str>, pdf: Option<&str>) -> String {
        crate::tui::pinto_line(&self.args(png, pdf))
    }

    /// [`Still::command`]'s arguments, after `pinto`.
    pub(super) fn args(&self, png: Option<&str>, pdf: Option<&str>) -> Vec<String> {
        let (style, win) = (&self.style, self.window);
        let mut args = vec!["view".into(), self.prefix.to_string()];
        // each value in its flag's word, so a negative one is not read as
        // a flag
        let mut flag = |name: &str, value: Option<String>| {
            args.push(value.map_or_else(|| name.into(), |v| crate::tui::flag_value(name, v)));
        };
        for (name, path) in [("--png", png), ("--pdf", pdf)] {
            if let Some(path) = path {
                flag(name, Some(path.into()));
            }
        }
        flag("--units", Some(self.units.into()));
        flag("--width", Some(self.vp.w.to_string()));
        flag("--height", Some(self.vp.h.to_string()));
        // `{}` prints the shortest text that parses back to the same f32
        let bbox = format!("{},{},{},{}", win.x0, win.y0, win.x1, win.y1);
        flag("--bbox", Some(bbox));
        flag("--level", Some(self.tag().into()));
        flag("--layer", Some(style.layer.to_string()));
        let focused = focused(style.focus);
        if !focused.is_empty() {
            flag("--focus", Some(ids(&self.level.comm, &focused, ",")));
        }
        if style.edges {
            flag("--edges", None);
        }
        if style.point != 1. {
            flag("--point-size", Some(style.point.to_string()));
        }
        flag("--theme", Some(value_name(&style.theme)));
        if let Some((round, show)) = self.round {
            flag("--round", Some(round.display().to_string()));
            flag("--show", Some(value_name(&show)));
        }
        if let Some(p) = &self.pair {
            flag("--pair", Some(p.name.clone()));
            flag("--pair-show", Some(value_name(&p.show)));
            if let Some(t) = p.table {
                flag("--lra-table", Some(t.display().to_string()));
            }
            if p.curve {
                flag("--curve", None);
            }
            return args;
        }
        if let Some(g) = self.gene {
            flag("--gene", Some(g.feature.to_string()));
            flag("--clip", Some(g.clip.to_string()));
            if g.source == Source::Expected {
                flag("--expected", None);
            }
        }
        args
    }
}

/// Render `vp` once and write it as a PDF figure and/or a PNG. The PDF
/// takes the map clean and draws its own vector scale bar; the PNG gets the
/// bar burnt in afterwards, on the same buffer.
pub(super) fn write_outputs(
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
        writing(path, |p| {
            pdf::write(&figure(base, level, style, gene, vp, &frame), p)
        })?;
    }
    if let Some(path) = png {
        if let Some(units) = base.units {
            scalebar::draw(&mut frame, vp, units, true);
        }
        writing(path, |p| frame.write_png(p))?;
    }
    Ok(())
}

/// Write `path` by `write`, its folder made first; a failure names it.
fn writing(path: &Path, write: impl FnOnce(&Path) -> anyhow::Result<()>) -> anyhow::Result<()> {
    mkdir_parent(&path.to_string_lossy())
        .and_then(|()| write(path))
        .with_context(|| format!("writing {}", path.display()))
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
        "{} cells · K={} · {}",
        thousands(base.geom.n()),
        comm.k,
        scalebar::per_pixel(base.units, vp.upp)
    );
    if !focused.is_empty() {
        subtitle.push_str(&format!(" · focus {}", ids(comm, &focused, " ")));
    }

    let ramp = |title: String, layer: Layer, top: String| pdf::Legend::Ramp {
        title,
        stops: (0..64)
            .map(|i| layer.ramp(base.theme()).at(i as f32 / 63.))
            .collect(),
        top,
    };
    let legend = match (gene, style.layer) {
        (Some(g), _) => ramp(g.title(), Layer::Gene, g.top_label()),
        (None, Layer::Argmax | Layer::Soft) => pdf::Legend::Communities(
            comm.by_size
                .iter()
                .map(|&c| {
                    let (colour, on) = legend_swatch(c, &level.palette, style.focus, base.theme());
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
                    .top(c, markers::TOP)
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
