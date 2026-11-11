//! Ablations: a fit of the same data with one thing taken away or scrambled,
//! to set against its parent run with `pinto compare`.
//!
//! Three kinds, one per run:
//!
//! - `coord_permutation` (`--permute-coords`): the graph is built on
//!   scrambled positions. `within-batch` shuffles cells among their own
//!   section's spot positions, which keeps the spot density exactly; `block`
//!   tiles each section into squares and moves whole tiles, which keeps
//!   neighbourhoods below the tile size and breaks them above it. The cells'
//!   own coordinates are still the ones written out, so a map of the
//!   ablation shows its communities where the cells really are.
//! - `feature_drop` (`--drop-features`): rows are removed before QC.
//! - `cell_mask` (`--mask-cells`): cells are removed after QC. By default
//!   only the edges touching them are cut (`--mask-mode cut-edges`): the
//!   graph is built with them and they are then taken out, so cells on either
//!   side of a masked region are not joined across the hole as a graph built
//!   without them would join them.
//!
//! What was done is recorded as the manifest's `ablation` block, with the
//! parent run it is to be set against.

use crate::util::common::*;
use crate::util::knn_graph::KnnGraph;
use crate::util::metadata::run_manifest;
use clap::{Args, ValueEnum};
use legume_numeric::matrix::common_io::read_name_list;
use legume_numeric::matrix::knn_graph::symmetric_adjacency;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermuteCoords {
    /// Shuffle cells among their own section's positions
    WithinBatch,
    /// Move whole square tiles of each section
    Block,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MaskMode {
    /// Build the graph with the masked cells, then cut their edges
    #[default]
    CutEdges,
    /// Leave the masked cells out before the graph is built
    Drop,
}

/// What an ablation took away; one per run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AblationKind {
    CoordPermutation,
    FeatureDrop,
    CellMask,
}

impl AblationKind {
    /// Its name in the manifest's `ablation.kind`.
    pub fn name(self) -> &'static str {
        match self {
            AblationKind::CoordPermutation => "coord_permutation",
            AblationKind::FeatureDrop => "feature_drop",
            AblationKind::CellMask => "cell_mask",
        }
    }
}

/// For the MCP tools, which only the `view` build has.
#[cfg(feature = "view")]
impl AblationKind {
    pub const ALL: [AblationKind; 3] = [
        AblationKind::CoordPermutation,
        AblationKind::FeatureDrop,
        AblationKind::CellMask,
    ];

    /// The flag that asks for it, without its dashes.
    pub fn flag(self) -> &'static str {
        match self {
            AblationKind::CoordPermutation => "permute-coords",
            AblationKind::FeatureDrop => "drop-features",
            AblationKind::CellMask => "mask-cells",
        }
    }
}

#[derive(Args, Debug, Clone, Default)]
#[command(
    next_help_heading = "Ablation",
    group(clap::ArgGroup::new("ablation_kind").multiple(false))
)]
pub struct AblationArgs {
    #[arg(
        long,
        value_enum,
        group = "ablation_kind",
        requires = "coord_files",
        help = "Ablation: build the graph on scrambled positions",
        long_help = "Ablation: build the spatial graph on scrambled cell positions.\n\
                     within-batch shuffles the cells among their own section's\n\
                     positions (a section is one data file and batch), which keeps\n\
                     the spot density exactly. block tiles each section into\n\
                     squares (--permute-block per side) and moves whole tiles among\n\
                     the occupied ones, keeping neighbourhoods inside a tile.\n\
                     Seeded by --ablation-seed. The written coordinates stay the\n\
                     cells' own. Needs --coord."
    )]
    pub permute_coords: Option<PermuteCoords>,

    #[arg(
        long,
        default_value_t = 8,
        help = "Tiles along a section's longer side for --permute-coords block"
    )]
    pub permute_block: usize,

    #[arg(
        long,
        value_name = "FILE|REGEX",
        group = "ablation_kind",
        help = "Ablation: drop features named in a file, or matching a regex",
        long_help = "Ablation: drop features before QC. An existing file lists names,\n\
                     one per line; anything else is a regular expression. A row is\n\
                     dropped when its name, or one of its `_`-separated parts (before\n\
                     any `/`), equals a listed name or matches the expression:\n\
                     `^MT-` drops ENSG…_MT-CO1."
    )]
    pub drop_features: Option<Box<str>>,

    #[arg(
        long,
        value_name = "FILE",
        group = "ablation_kind",
        help = "Ablation: mask the cells listed in a file (one per line)",
        long_help = "Ablation: mask cells after QC. The file lists cell names as the\n\
                     run names them (barcode@sample), one per line; the first\n\
                     tab-, comma- or space-separated field counts and a `cell` header\n\
                     is skipped. A .geojson (or .json) file is read as polygons\n\
                     instead, as QuPath exports them: each in its section's own\n\
                     coordinates, the section named by a `section` (or `sample`,\n\
                     `name`) property matching the end of the cells' @sample tag;\n\
                     the cells inside are masked. See --mask-mode."
    )]
    pub mask_cells: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = MaskMode::CutEdges,
        help = "How --mask-cells takes cells out of the graph",
        long_help = "cut-edges builds the spatial graph with the masked cells and\n\
                     then removes them with their edges, so the cells around a\n\
                     masked region lose those neighbours. drop leaves them out\n\
                     before the graph is built, so the KNN search joins cells across\n\
                     the hole. Without --coord the two are the same."
    )]
    pub mask_mode: MaskMode,

    #[arg(
        long,
        value_name = "RUN",
        requires = "ablation_kind",
        help = "The run this ablation is set against (its prefix or .pinto.json)",
        long_help = "The run this ablation is set against: its prefix or .pinto.json,\n\
                     recorded in the manifest's ablation block (and, given as the\n\
                     .pinto.json, linked to it in a project's records)."
    )]
    pub ablation_parent: Option<Box<str>>,

    #[arg(
        long,
        help = "Seed for --permute-coords [default: --seed]",
        hide = true
    )]
    pub ablation_seed: Option<u64>,
}

/// The manifest's `ablation` block: pinto's own record of what this run took
/// away from which parent.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Ablation {
    /// The parent run, as given (a path or prefix, relative as typed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// `coord_permutation`, `feature_drop` or `cell_mask`.
    pub kind: String,
    /// What was done, and how much it touched.
    pub spec: Value,
    /// For the random kinds only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl AblationArgs {
    /// What clap cannot say: the files named exist, and a polygon file has
    /// coordinates to place cells by. (One kind per run, a parent only with
    /// one, and --permute-coords only with --coord are clap rules.) Runs
    /// before any data is read.
    pub fn validate(&self, has_coords: bool) -> anyhow::Result<()> {
        if self.permute_coords == Some(PermuteCoords::Block) {
            anyhow::ensure!(
                self.permute_block >= 2,
                "--permute-block must be at least 2"
            );
        }
        if let Some(f) = &self.mask_cells {
            anyhow::ensure!(Path::new(f.as_ref()).is_file(), "--mask-cells: no file {f}");
            anyhow::ensure!(
                has_coords || !is_region_file(f),
                "--mask-cells {f} is a polygon file: it needs --coord to place the cells"
            );
        }
        if let Some(p) = &self.ablation_parent {
            parent_manifest(p)?;
        }
        if self.kind().is_some() && self.ablation_parent.is_none() {
            warn!("an ablation without --ablation-parent records no run to set it against");
        }
        Ok(())
    }

    /// The kind asked for, if any.
    pub fn kind(&self) -> Option<AblationKind> {
        if self.permute_coords.is_some() {
            Some(AblationKind::CoordPermutation)
        } else if self.drop_features.is_some() {
            Some(AblationKind::FeatureDrop)
        } else if self.mask_cells.is_some() {
            Some(AblationKind::CellMask)
        } else {
            None
        }
    }

    /// The seed the random kinds use.
    pub fn seed(&self, run_seed: u64) -> u64 {
        self.ablation_seed.unwrap_or(run_seed)
    }

    /// The manifest block for what was done, `spec` filled by the caller.
    /// The parent is recorded as its manifest, the name records link by.
    pub fn record(&self, kind: AblationKind, spec: Value, seed: Option<u64>) -> Ablation {
        Ablation {
            parent: self.ablation_parent.as_deref().map(run_manifest),
            kind: kind.name().to_string(),
            spec,
            seed,
        }
    }
}

/// The parent's manifest: `run` itself when it is one, else `{run}.pinto.json`.
pub fn parent_manifest(run: &str) -> anyhow::Result<String> {
    let manifest = run_manifest(run);
    anyhow::ensure!(
        Path::new(&manifest).is_file(),
        "--ablation-parent {run}: no manifest {manifest}"
    );
    Ok(manifest)
}

/// The parts of a row name a listed name or a pattern is tried against: the
/// whole name, and its `_`-separated parts before any `/`.
fn name_parts(name: &str) -> impl Iterator<Item = &str> {
    let head = name.split('/').next().unwrap_or(name);
    std::iter::once(name).chain(head.split('_').filter(|p| !p.is_empty()))
}

/// Which rows survive `--drop-features`: `keep[r]` is false for a dropped row.
pub fn feature_keep(row_names: &[Box<str>], spec: &str) -> anyhow::Result<(Vec<bool>, Value)> {
    let (keep, how): (Vec<bool>, Value) = if Path::new(spec).is_file() {
        let listed: HashSet<Box<str>> = read_name_list(spec)?.into_iter().collect();
        anyhow::ensure!(
            !listed.is_empty(),
            "--drop-features {spec}: the file lists no names"
        );
        let keep = row_names
            .iter()
            .map(|r| !name_parts(r).any(|p| listed.contains(p)))
            .collect();
        (keep, json!({ "file": spec, "n_listed": listed.len() }))
    } else {
        let re = regex::Regex::new(spec).map_err(|e| {
            anyhow::anyhow!("--drop-features {spec:?}: neither a file nor a regex: {e}")
        })?;
        let keep = row_names
            .iter()
            .map(|r| !name_parts(r).any(|p| re.is_match(p)))
            .collect();
        (keep, json!({ "regex": spec }))
    };
    let n_dropped = keep.iter().filter(|&&k| !k).count();
    anyhow::ensure!(
        n_dropped > 0,
        "--drop-features {spec:?} matches no feature: nothing would be ablated"
    );
    anyhow::ensure!(
        n_dropped < keep.len(),
        "--drop-features {spec:?} matches every feature"
    );
    let mut how = how;
    how["n_dropped"] = n_dropped.into();
    how["n_rows"] = keep.len().into();
    Ok((keep, how))
}

/// The cell names a `--mask-cells` file lists.
pub fn read_cell_list(path: &str) -> anyhow::Result<HashSet<Box<str>>> {
    // The shared name-list reader, which knows gene-like headers; a `cell` or
    // `barcode` header (what run_region writes) is dropped here.
    let cells: HashSet<Box<str>> = read_name_list(path)?
        .into_iter()
        .filter(|c| !matches!(c.as_ref(), "cell" | "barcode"))
        .collect();
    anyhow::ensure!(
        !cells.is_empty(),
        "--mask-cells {path}: the file lists no cells"
    );
    Ok(cells)
}

/// Whether a `--mask-cells` file is a polygon file rather than a cell list.
pub fn is_region_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.ends_with(".geojson") || lower.ends_with(".json")
}

/// A polygon (its rings: the outline, then any holes), and the section whose
/// coordinates it is drawn in, when it names one.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    pub section: Option<String>,
    pub rings: Vec<Vec<(f64, f64)>>,
}

/// The polygons of a GeoJSON file: a FeatureCollection, a Feature, or a bare
/// Polygon / MultiPolygon geometry. A feature's section is its `section`
/// property (or `sample`).
pub fn read_regions(path: &str) -> anyhow::Result<Vec<Region>> {
    let text =
        std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("--mask-cells {path}: {e}"))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("--mask-cells {path}: not JSON: {e}"))?;
    let regions = parse_regions(&value)?;
    anyhow::ensure!(!regions.is_empty(), "--mask-cells {path}: no polygon in it");
    Ok(regions)
}

/// [`read_regions`] on parsed GeoJSON.
pub fn parse_regions(value: &Value) -> anyhow::Result<Vec<Region>> {
    let mut out = Vec::new();
    collect_regions(value, None, &mut out)?;
    Ok(out)
}

fn collect_regions(v: &Value, section: Option<&str>, out: &mut Vec<Region>) -> anyhow::Result<()> {
    let ring = |r: &Value| -> anyhow::Result<Vec<(f64, f64)>> {
        r.as_array()
            .ok_or_else(|| anyhow::anyhow!("a polygon ring is not a list of points"))?
            .iter()
            .map(|p| match p.as_array().map(|a| a.as_slice()) {
                Some([x, y, ..]) => Ok((
                    x.as_f64()
                        .ok_or_else(|| anyhow::anyhow!("a point is not numeric"))?,
                    y.as_f64()
                        .ok_or_else(|| anyhow::anyhow!("a point is not numeric"))?,
                )),
                _ => anyhow::bail!("a point is not [x, y]"),
            })
            .collect()
    };
    let polygon = |rings: &Value| -> anyhow::Result<Region> {
        let rings: Vec<Vec<(f64, f64)>> = rings
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("a polygon is not a list of rings"))?
            .iter()
            .map(ring)
            .collect::<anyhow::Result<_>>()?;
        anyhow::ensure!(
            rings.first().is_some_and(|r| r.len() >= 3),
            "a polygon needs at least three points"
        );
        Ok(Region {
            section: section.map(str::to_string),
            rings,
        })
    };
    match v.get("type").and_then(Value::as_str) {
        Some("FeatureCollection") => {
            for f in v
                .get("features")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                collect_regions(f, section, out)?;
            }
        }
        Some("Feature") => {
            let props = v.get("properties");
            // `name` is not one: QuPath writes the annotation's label there.
            let named = ["section", "sample"]
                .iter()
                .find_map(|k| props.and_then(|p| p.get(*k)).and_then(Value::as_str))
                .or(section);
            if let Some(g) = v.get("geometry") {
                collect_regions(g, named, out)?;
            }
        }
        Some("Polygon") => out.push(polygon(&v["coordinates"])?),
        Some("MultiPolygon") => {
            for p in v["coordinates"].as_array().into_iter().flatten() {
                out.push(polygon(p)?);
            }
        }
        Some("GeometryCollection") => {
            for g in v
                .get("geometries")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                collect_regions(g, section, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Whether `(x, y)` lies inside `rings` by the even-odd rule, so a hole ring
/// takes its area back out.
pub fn inside(rings: &[Vec<(f64, f64)>], x: f64, y: f64) -> bool {
    let mut odd = false;
    for ring in rings {
        let n = ring.len();
        for i in 0..n {
            let (xi, yi) = ring[i];
            let (xj, yj) = ring[(i + n - 1) % n];
            if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                odd = !odd;
            }
        }
    }
    odd
}

/// A cell's section: what follows the last `@` of its name (the sample a
/// multi-file run tags it with), or "" for a single sample.
pub fn section_of(name: &str) -> &str {
    name.rsplit_once('@').map_or("", |(_, s)| s)
}

/// Whether a region's section name picks out a cell's section: the same, or
/// its `_`-separated end (`S1` picks `ID1_S1`).
pub fn section_matches(cell_section: &str, wanted: &str) -> bool {
    cell_section == wanted
        || cell_section
            .strip_suffix(wanted)
            .is_some_and(|head| head.ends_with('_'))
}

/// The cells inside any of `regions`, with each region's count. A region
/// without a section applies to every cell when there is one section, and is
/// refused when there are several: their coordinates overlap.
pub fn cells_in_regions(
    names: &[Box<str>],
    x: &[f32],
    y: &[f32],
    regions: &[Region],
) -> anyhow::Result<(HashSet<Box<str>>, Vec<usize>)> {
    // Each cell's section once; a region then picks sections, not cells.
    let mut sections: Vec<&str> = Vec::new();
    let section_id: Vec<usize> = names
        .iter()
        .map(|n| {
            let s = section_of(n);
            sections.iter().position(|&t| t == s).unwrap_or_else(|| {
                sections.push(s);
                sections.len() - 1
            })
        })
        .collect();
    let mut picked: HashSet<Box<str>> = HashSet::default();
    let mut counts = Vec::with_capacity(regions.len());
    for (r, region) in regions.iter().enumerate() {
        let wanted = region.section.as_deref();
        anyhow::ensure!(
            wanted.is_some() || sections.len() <= 1,
            "polygon {} names no section, and the run has {} whose coordinates overlap: give it \
             a `section` property (one of {})",
            r + 1,
            sections.len(),
            sections
                .iter()
                .take(5)
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        );
        let in_region: Vec<bool> = sections
            .iter()
            .map(|s| wanted.is_none_or(|w| section_matches(s, w)))
            .collect();
        let n_sections = in_region.iter().filter(|&&b| b).count();
        anyhow::ensure!(
            n_sections > 0,
            "polygon {}: no cell is in section {:?}",
            r + 1,
            wanted.unwrap_or("")
        );
        anyhow::ensure!(
            wanted.is_none() || n_sections == 1,
            "polygon {}: section {:?} names {} sections, whose coordinates overlap; give one \
             in full",
            r + 1,
            wanted.unwrap_or(""),
            n_sections
        );
        let mut n = 0;
        for i in (0..names.len()).filter(|&i| in_region[section_id[i]]) {
            if inside(&region.rings, f64::from(x[i]), f64::from(y[i])) {
                picked.insert(names[i].clone());
                n += 1;
            }
        }
        counts.push(n);
    }
    Ok((picked, counts))
}

/// Which cells survive `--mask-cells`: `keep[i]` is false for a masked cell.
pub fn cell_keep(names: &[Box<str>], listed: &HashSet<Box<str>>) -> anyhow::Result<Vec<bool>> {
    let keep: Vec<bool> = names.iter().map(|n| !listed.contains(n)).collect();
    let n_masked = keep.iter().filter(|&&k| !k).count();
    anyhow::ensure!(
        n_masked > 0,
        "--mask-cells: none of the {} listed cells is among the run's {} (names look like {})",
        listed.len(),
        names.len(),
        names.first().map_or("", |n| n.as_ref())
    );
    anyhow::ensure!(n_masked < names.len(), "--mask-cells masks every cell");
    if n_masked < listed.len() {
        warn!(
            "--mask-cells: {} of the {} listed cells are not in the run (QC may have dropped them)",
            listed.len() - n_masked,
            listed.len()
        );
    }
    Ok(keep)
}

/// Each cell's section: one id per (data file, batch label) pair, numbered in
/// order of first appearance.
pub fn sections(file_sizes: &[usize], batches: &[Box<str>]) -> Vec<u32> {
    let mut ids: HashMap<(usize, &str), u32> = HashMap::default();
    let mut out = Vec::with_capacity(batches.len());
    let mut i = 0;
    for (f, &n) in file_sizes.iter().enumerate() {
        for b in &batches[i..i + n] {
            let next = ids.len() as u32;
            out.push(*ids.entry((f, b.as_ref())).or_insert(next));
        }
        i += n;
    }
    out
}

fn group(sections: &[u32]) -> Vec<Vec<usize>> {
    let n = sections.iter().map(|&s| s as usize + 1).max().unwrap_or(0);
    let mut groups = vec![Vec::new(); n];
    for (i, &s) in sections.iter().enumerate() {
        groups[s as usize].push(i);
    }
    groups
}

/// `coords` with its rows shuffled within each section.
pub fn permute_within(coords: &Mat, sections: &[u32], seed: u64) -> Mat {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut from: Vec<usize> = (0..coords.nrows()).collect();
    for members in group(sections) {
        let mut shuffled = members.clone();
        shuffled.shuffle(&mut rng);
        for (&i, &j) in members.iter().zip(&shuffled) {
            from[i] = j;
        }
    }
    coords.select_rows(from.iter())
}

/// `coords` with each section cut into square tiles, `per_side` along its
/// longer side, and whole occupied tiles moved onto one another: a cell keeps
/// its place inside its tile. Columns `xy` are the plane; the others (the
/// batch offset) are untouched. Returns the coordinates and the number of
/// occupied tiles shuffled.
pub fn permute_blocks(
    coords: &Mat,
    xy: (usize, usize),
    sections: &[u32],
    per_side: usize,
    seed: u64,
) -> (Mat, usize) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut out = coords.clone();
    let mut n_tiles = 0;
    for members in group(sections) {
        if members.is_empty() {
            continue;
        }
        let (mut lo_x, mut lo_y) = (f32::INFINITY, f32::INFINITY);
        let (mut hi_x, mut hi_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &i in &members {
            let (x, y) = (coords[(i, xy.0)], coords[(i, xy.1)]);
            lo_x = lo_x.min(x);
            lo_y = lo_y.min(y);
            hi_x = hi_x.max(x);
            hi_y = hi_y.max(y);
        }
        let side = (hi_x - lo_x).max(hi_y - lo_y) / per_side as f32;
        if side <= 0.0 || !side.is_finite() {
            continue;
        }
        let last = per_side - 1;
        let tile = |v: f32, lo: f32| (((v - lo) / side) as usize).min(last);
        let tile_of: Vec<(usize, usize)> = members
            .iter()
            .map(|&i| (tile(coords[(i, xy.0)], lo_x), tile(coords[(i, xy.1)], lo_y)))
            .collect();
        let mut occupied: Vec<(usize, usize)> = tile_of.clone();
        occupied.sort_unstable();
        occupied.dedup();
        let mut onto = occupied.clone();
        onto.shuffle(&mut rng);
        let moved: HashMap<(usize, usize), (usize, usize)> =
            occupied.iter().copied().zip(onto.iter().copied()).collect();
        n_tiles += occupied.len();
        for (&i, &t) in members.iter().zip(&tile_of) {
            let u = moved[&t];
            out[(i, xy.0)] += (u.0 as f32 - t.0 as f32) * side;
            out[(i, xy.1)] += (u.1 as f32 - t.1 as f32) * side;
        }
    }
    (out, n_tiles)
}

/// The plane's two columns: the first two coordinate columns that are not
/// the batch offset.
pub fn plane_columns(names: &[Box<str>]) -> anyhow::Result<(usize, usize)> {
    let mut cols = names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.as_ref() != "batch")
        .map(|(j, _)| j);
    match (cols.next(), cols.next()) {
        (Some(x), Some(y)) => Ok((x, y)),
        _ => anyhow::bail!("--permute-coords block needs two coordinate columns"),
    }
}

/// `graph` restricted to the kept nodes, renumbered in order: every edge
/// touching a dropped node goes with it.
pub fn induced_subgraph(graph: &KnnGraph, keep: &[bool]) -> KnnGraph {
    let mut new_id = vec![usize::MAX; keep.len()];
    let mut n = 0;
    for (i, &k) in keep.iter().enumerate() {
        if k {
            new_id[i] = n;
            n += 1;
        }
    }
    let (edges, distances): (Vec<(usize, usize)>, Vec<f32>) = graph
        .edges
        .iter()
        .zip(&graph.distances)
        .filter(|(&(i, j), _)| keep[i] && keep[j])
        .map(|(&(i, j), &d)| ((new_id[i], new_id[j]), d))
        .unzip();
    KnnGraph {
        adjacency: symmetric_adjacency(n, &edges, &distances),
        edges,
        distances,
        n_nodes: n,
    }
}
