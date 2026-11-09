//! The options of `pinto view`, and the checks on their values.

use super::color::Theme;
use super::lra;
use super::render::Layer;
use clap::Args;

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
                     \x20 auto  µm for Space Ranger pxl_* columns when its\n\
                     \x20       scalefactors_json.json sits beside the coordinates\n\
                     \x20       (or in spatial/), else px; µm for Xenium centroids;\n\
                     \x20       otherwise a bare number\n\
                     \x20 um    micrometres (1000 µm shows as 1 mm)\n\
                     \x20 px    image pixels\n\
                     \x20 none  no scale bar"
    )]
    pub units: Box<str>,

    #[arg(
        long,
        value_name = "FEATURE",
        help = "Colour cells by one feature, by its symbol or full name (--png/--pdf)",
        long_help = "Colour cells by one feature instead of communities: its symbol\n\
                     (SYMBOL) or full name (ENSG…_SYMBOL). Shows the observed\n\
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
        help = "How the map is drawn in the terminal"
    )]
    pub graphics: legume_tui::map::kitty::MapGraphics,

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
                     defaults to the terminal's own background, and `t` switches it;\n\
                     --png and --pdf default to dark."
    )]
    pub theme: Option<Theme>,

    #[arg(
        long,
        value_name = "SERVER",
        num_args = 0..=1,
        default_missing_value = "",
        conflicts_with_all = ["prefix", "summary", "png", "pdf"],
        help = "Follow what an agent shows on `pinto mcp` [default: the newest]",
        long_help = "Follow the view an agent has on a running `pinto mcp`: the\n\
                     newest, or the one SERVER names (its pid, or its watch folder).\n\
                     The map follows the agent's as it changes; a move of your own\n\
                     pauses that, and F follows again. Nothing goes back to the\n\
                     agent. The panel lists its last calls."
    )]
    pub follow: Option<Box<str>>,

    #[arg(
        long,
        value_name = "LIGAND:RECEPTOR",
        help_heading = "Ligand-receptor pairs (pinto lra)",
        help = "Colour cells by an lra pair: senders, exposure, competence (--png/--pdf)",
        long_help = "Colour cells by a ligand-receptor pair of the run's lra table\n\
                     (pinto lra): senders (cells carrying the ligand), and receivers by\n\
                     exposed or not (contact exposure above the table's threshold) and\n\
                     receptor-competent or not, computed as lra computes them. In the\n\
                     terminal, I opens the pair list instead."
    )]
    pub pair: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value = "classes",
        help_heading = "Ligand-receptor pairs (pinto lra)",
        help = "With --pair: classes (senders, exposed, competent) or exposure"
    )]
    pub pair_show: lra::PairShow,

    #[arg(
        long,
        value_name = "FILE",
        help_heading = "Ligand-receptor pairs (pinto lra)",
        help = "With --pair: the lra table [default: the run's full table]",
        long_help = "With --pair: the contact-effect table the pair is read from, such\n\
                     as a subset of pinto lra --only ({prefix}.lra.only-<tag>.parquet).\n\
                     Defaults to the full table the run's manifest records, else its\n\
                     first subset."
    )]
    pub lra_table: Option<Box<str>>,

    #[arg(
        long,
        help_heading = "Ligand-receptor pairs (pinto lra)",
        help = "With --pair: draw its threshold-response curve, not the map"
    )]
    pub curve: bool,

    #[command(flatten)]
    pub session: legume_tui::cli::SessionArgs,
}

/// What groups the cells on the map (`--show`).
#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Show {
    /// The level's communities.
    Communities,
    /// A lupin round's cell types.
    Types,
    /// A lupin round's clusters.
    Clusters,
}

impl ViewArgs {
    /// The run, once [`run_view`](super::run_view) has made sure there is one.
    pub fn prefix(&self) -> &str {
        self.prefix.as_deref().expect("run_view picks a run first")
    }
}

/// Percentiles `--clip` takes.
const CLIPS: std::ops::RangeInclusive<f32> = 50.0..=100.;

fn parse_clip(s: &str) -> Result<f32, String> {
    s.parse::<f32>()
        .ok()
        .and_then(|p| check_clip(p).ok())
        .ok_or_else(|| format!("{s:?}: a percentile from 50 to 100"))
}

pub(super) fn check_clip(p: f32) -> Result<f32, String> {
    in_range(p, &CLIPS, "a percentile")
}

pub(super) fn check_point_size(x: f32) -> Result<f32, String> {
    in_range(x, &POINT_SIZES, "a size")
}

/// `x` when `range` holds it; else what it should be.
fn in_range(x: f32, range: &std::ops::RangeInclusive<f32>, what: &str) -> Result<f32, String> {
    if range.contains(&x) {
        Ok(x)
    } else {
        Err(format!(
            "{x}: {what} from {} to {}",
            range.start(),
            range.end()
        ))
    }
}

/// Smallest and largest `--point-size`.
pub(super) const POINT_SIZES: std::ops::RangeInclusive<f32> = 0.25..=8.;

fn parse_point_size(s: &str) -> Result<f32, String> {
    s.parse::<f32>()
        .ok()
        .and_then(|x| check_point_size(x).ok())
        .ok_or_else(|| {
            format!(
                "{s:?}: a size from {} to {}",
                POINT_SIZES.start(),
                POINT_SIZES.end()
            )
        })
}
