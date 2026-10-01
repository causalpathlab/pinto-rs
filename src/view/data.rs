//! Load a pinto run for viewing: manifest → cell geometry ⋈ propensity.
//!
//! Everything comes from files a pinto run already writes, through the same
//! readers the rest of the crate uses: `coord_pairs` for positions and batch
//! labels, `propensity` for the per-cell community mixture (one per level),
//! and `link_community` for the per-edge labels.

use super::markers::FeatureRates;
use crate::util::common::*;
use crate::util::input::read_one_coord_file;
use crate::util::metadata::{LevelInfo, PintoMetadata};
use crate::util::parquet_io::{
    read_cells_from_coord_pairs, read_cells_table, read_feature_community, read_propensity,
    visit_link_community, CellTable, PropensityRead,
};
use std::path::{Path, PathBuf};

/// Marks a cell (or bin) with no community: absent from the propensity
/// table, or a bin with no cells.
pub const NO_CLUSTER: u16 = u16::MAX;

/// Axis-aligned rectangle in world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Rect {
    pub const UNIT: Rect = Rect {
        x0: 0.,
        y0: 0.,
        x1: 1.,
        y1: 1.,
    };

    /// Grown by `frac` of its larger side on every edge.
    pub fn pad(&self, frac: f32) -> Rect {
        self.grow(frac * self.width().max(self.height()))
    }

    /// Grown by `m` on every side.
    pub fn grow(&self, m: f32) -> Rect {
        Rect {
            x0: self.x0 - m,
            y0: self.y0 - m,
            x1: self.x1 + m,
            y1: self.y1 + m,
        }
    }

    pub fn width(&self) -> f32 {
        self.x1 - self.x0
    }

    pub fn height(&self) -> f32 {
        self.y1 - self.y0
    }

    pub fn union(&self, o: &Rect) -> Rect {
        Rect {
            x0: self.x0.min(o.x0),
            y0: self.y0.min(o.y0),
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
        }
    }
}

/// A pinto run on disk: its `.pinto.json`, or, for a run without one, what
/// [`PintoMetadata::discover`] finds from its file names.
pub struct Run {
    /// Output prefix: every output is `{prefix}.…`.
    pub prefix: String,
    /// The manifest read, if the run has one.
    pub manifest: Option<PathBuf>,
    pub meta: PintoMetadata,
    /// Community levels, coarse cascade levels first and `final` last.
    pub levels: Vec<LevelInfo>,
}

impl Run {
    /// Open `{prefix}.pinto.json`, or the manifest itself when given one; a
    /// prefix without a manifest is discovered from its file names.
    pub fn open(prefix_or_manifest: &str) -> anyhow::Result<Self> {
        let (prefix, manifest) = match prefix_or_manifest.strip_suffix(".pinto.json") {
            Some(prefix) => (prefix.to_string(), Some(PathBuf::from(prefix_or_manifest))),
            None => {
                let path = PathBuf::from(format!("{prefix_or_manifest}.pinto.json"));
                (
                    prefix_or_manifest.to_string(),
                    path.exists().then_some(path),
                )
            }
        };
        let meta = match &manifest {
            Some(path) => PintoMetadata::read(path)
                .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?,
            None => PintoMetadata::discover(&prefix)?,
        };
        let levels = meta.level_list();
        anyhow::ensure!(!levels.is_empty(), "{prefix}: no propensity output listed");
        Ok(Run {
            prefix,
            manifest,
            meta,
            levels,
        })
    }

    /// The run's expression data files: those the manifest lists, else
    /// `{prefix}.zarr.zip`, `.zarr` or `.h5` next to the outputs.
    pub fn data_files(&self) -> Vec<Box<str>> {
        match self.meta.data_files.as_deref() {
            Some(files) if !files.is_empty() => files
                .iter()
                .map(|f| self.resolve(f).to_string_lossy().into())
                .collect(),
            _ => ["zarr.zip", "zarr", "h5"]
                .iter()
                .map(|ext| format!("{}.{ext}", self.prefix))
                .find(|p| Path::new(p).exists())
                .map(|p| vec![p.into_boxed_str()])
                .unwrap_or_default(),
        }
    }

    /// Where the run was read from, for messages.
    pub fn source(&self) -> String {
        match &self.manifest {
            Some(path) => path.display().to_string(),
            None => format!(
                "{} (no .pinto.json; outputs found by file name)",
                self.prefix
            ),
        }
    }

    /// Output paths are `{prefix}.…` as typed at fit time, so they are
    /// relative to wherever pinto ran. Use one as-is when it exists, else
    /// the same path under the prefix's directory (the run opened from
    /// elsewhere), else the same file name next to the prefix.
    pub fn resolve(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.exists() {
            return p.to_path_buf();
        }
        let Some(dir) = Path::new(&self.prefix).parent() else {
            return p.to_path_buf();
        };
        let under = dir.join(p);
        if p.is_relative() && under.exists() {
            return under;
        }
        match p.file_name() {
            Some(name) => dir.join(name),
            None => p.to_path_buf(),
        }
    }

    /// The run's name: the last part of its prefix.
    pub fn name(&self) -> String {
        Path::new(&self.prefix)
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
    }

    /// Position of the level tagged `tag` (e.g. `L2`, `final`); the last
    /// level if `None`.
    pub fn level_index(&self, tag: Option<&str>) -> anyhow::Result<usize> {
        match tag {
            None => Ok(self.levels.len() - 1),
            Some(t) => self.levels.iter().position(|l| l.tag == t).ok_or_else(|| {
                let tags: Vec<&str> = self.levels.iter().map(|l| l.tag.as_str()).collect();
                anyhow::anyhow!("no level {t:?}; this run has {tags:?}")
            }),
        }
    }

    /// Every cell's position: from the run's cells table when it wrote one;
    /// for older runs, from the cells `coord_pairs` names plus the rest from
    /// the coordinate file the run was fit with.
    pub fn load_geometry(&self) -> anyhow::Result<Geometry> {
        let coord_columns = self.meta.outputs.coord_columns.as_deref();
        if let Some(table) = self.meta.outputs.cells.as_deref() {
            let path = self.resolve(table);
            if path.exists() {
                match read_cells_table(&path, coord_columns) {
                    Ok(cells) => return Ok(Geometry::from_cells(cells)),
                    Err(e) => log::warn!("{}: {e}; reading coord_pairs instead", path.display()),
                }
            }
        }
        let pairs = self.meta.outputs.coord_pairs.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "{}: no coord_pairs output listed; was the run fit with --coord?",
                self.source()
            )
        })?;
        let mut cells = read_cells_from_coord_pairs(&self.resolve(pairs), coord_columns)?;
        self.add_cells_without_edges(&mut cells);
        Ok(Geometry::from_cells(cells))
    }

    /// For runs written before the cells table: `coord_pairs` only names
    /// cells with an edge, and the coordinate file the run was fit with has
    /// every cell; add the rest from it. Only for a
    /// single batch: the file does not say which batch a lone cell is in.
    fn add_cells_without_edges(&self, cells: &mut CellTable) {
        let Some(file) = self.meta.coord_file.as_deref() else {
            return;
        };
        if cells.batches.is_some() || file.contains(',') {
            return;
        }
        let path = self.resolve(file);
        let read = read_one_coord_file(&path.to_string_lossy(), &[], &cells.coord_col_names, None);
        let coords = match read {
            Ok(coords) if coords.mat.ncols() >= 2 => coords,
            Ok(_) => return,
            Err(e) => {
                log::warn!("{}: {e}; cells without edges are left out", path.display());
                return;
            }
        };
        let before = cells.names.len();
        let mut in_graph = cells.in_graph.take().unwrap_or_else(|| vec![true; before]);
        for (r, name) in coords.rows.iter().enumerate() {
            if !cells.index.contains_key(name) {
                cells.index.insert(name.clone(), cells.names.len());
                cells.names.push(name.clone());
                cells.coords.push((coords.mat[(r, 0)], coords.mat[(r, 1)]));
                in_graph.push(false);
            }
        }
        cells.in_graph = Some(in_graph);
        log::info!(
            "{} cells outside the graph (dropped by QC, or without neighbours) \
             added from {}",
            cells.names.len() - before,
            path.display()
        );
    }

    pub fn load_communities(
        &self,
        geom: &Geometry,
        level: &LevelInfo,
    ) -> anyhow::Result<Communities> {
        let exclude: HashSet<Box<str>> = geom.coord_names.iter().cloned().collect();
        let read = read_propensity(&self.resolve(&level.propensity), &exclude)?;
        Ok(Communities::join(geom, &level.tag, read))
    }

    pub fn load_feature_rates(&self, level: &LevelInfo) -> anyhow::Result<FeatureRates> {
        let path = level.feature_community.as_deref().ok_or_else(|| {
            anyhow::anyhow!("level {} has no feature_community output", level.tag)
        })?;
        let (rates, names) = read_feature_community(&self.resolve(path))?;
        Ok(FeatureRates { names, rates })
    }

    pub fn load_edges(&self, geom: &Geometry, level: &LevelInfo) -> anyhow::Result<Edges> {
        let path = level
            .link_community
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("level {} has no link_community output", level.tag))?;
        let mut edges = Edges::default();
        visit_link_community(&self.resolve(path), |l, r, c| edges.push(geom, l, r, c))?;
        Ok(edges)
    }
}

/// One batch's block in the tiled layout.
#[derive(Clone, Debug)]
pub struct Tile {
    pub name: Box<str>,
    /// Bounds in world (tiled) coordinates.
    pub bounds: Rect,
    pub n_cells: usize,
}

/// Cell positions in world coordinates.
///
/// A multi-batch run keeps every batch in its own original frame, so the
/// sections overlap. Each batch is translated into its own slot of a
/// near-square grid of tiles so they sit side by side.
pub struct Geometry {
    pub names: Vec<Box<str>>,
    pub index: HashMap<Box<str>, usize>,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    /// Index into `tiles` per cell.
    pub batch: Vec<u16>,
    pub tiles: Vec<Tile>,
    /// Coordinate column names, `[x, y]`.
    pub coord_names: Vec<Box<str>>,
    /// Whether each cell became a graph node; `None` when every cell did.
    /// The others (dropped by QC, or without neighbours) have no community.
    pub in_graph: Option<Vec<bool>>,
}

impl Geometry {
    pub fn from_cells(cells: CellTable) -> Self {
        let CellTable {
            names,
            coords,
            batches,
            index,
            in_graph,
            coord_col_names,
        } = cells;
        let mut x: Vec<f32> = coords.iter().map(|c| c.0).collect();
        let mut y: Vec<f32> = coords.iter().map(|c| c.1).collect();

        let labels: Vec<Box<str>> = batches.unwrap_or_else(|| vec!["all".into(); names.len()]);
        let (batch, tile_names) = factorize(&labels);
        let tiles = tile_batches(&mut x, &mut y, &batch, tile_names);

        Geometry {
            names,
            index,
            x,
            y,
            batch,
            tiles,
            coord_names: coord_col_names,
            in_graph,
        }
    }

    /// Cells that are not graph nodes.
    pub fn n_outside_graph(&self) -> usize {
        self.in_graph
            .as_ref()
            .map_or(0, |g| g.iter().filter(|&&k| !k).count())
    }

    pub fn n(&self) -> usize {
        self.x.len()
    }

    pub fn bounds(&self) -> Rect {
        self.tiles
            .iter()
            .map(|t| t.bounds)
            .reduce(|a, b| a.union(&b))
            .unwrap_or(Rect::UNIT)
    }
}

/// Map labels to dense ids in sorted label order.
fn factorize(labels: &[Box<str>]) -> (Vec<u16>, Vec<Box<str>>) {
    let mut uniq: Vec<Box<str>> = labels.to_vec();
    uniq.sort();
    uniq.dedup();
    let id: HashMap<&str, u16> = uniq
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_ref(), i as u16))
        .collect();
    (labels.iter().map(|s| id[s.as_ref()]).collect(), uniq)
}

/// Move each batch into its own slot of a `cols × rows` grid, in place.
/// Slots are as large as the largest batch plus a 5% gutter.
fn tile_batches(x: &mut [f32], y: &mut [f32], batch: &[u16], names: Vec<Box<str>>) -> Vec<Tile> {
    let nb = names.len();
    let mut bounds = vec![
        Rect {
            x0: f32::INFINITY,
            y0: f32::INFINITY,
            x1: f32::NEG_INFINITY,
            y1: f32::NEG_INFINITY,
        };
        nb
    ];
    let mut counts = vec![0usize; nb];
    for i in 0..x.len() {
        let b = batch[i] as usize;
        let r = &mut bounds[b];
        r.x0 = r.x0.min(x[i]);
        r.y0 = r.y0.min(y[i]);
        r.x1 = r.x1.max(x[i]);
        r.y1 = r.y1.max(y[i]);
        counts[b] += 1;
    }

    let slot_w = bounds.iter().map(Rect::width).fold(0f32, f32::max);
    let slot_h = bounds.iter().map(Rect::height).fold(0f32, f32::max);
    let gutter = 0.05 * slot_w.max(slot_h);
    let cols = (nb as f64).sqrt().ceil().max(1.) as usize;

    // A single batch stays in its own frame.
    let offsets: Vec<(f32, f32)> = (0..nb)
        .map(|b| {
            if nb == 1 {
                return (0., 0.);
            }
            let (col, row) = (b % cols, b / cols);
            let ox = col as f32 * (slot_w + gutter) - bounds[b].x0;
            let oy = row as f32 * (slot_h + gutter) - bounds[b].y0;
            (ox, oy)
        })
        .collect();

    for i in 0..x.len() {
        let (ox, oy) = offsets[batch[i] as usize];
        x[i] += ox;
        y[i] += oy;
    }

    names
        .into_iter()
        .enumerate()
        .map(|(b, name)| {
            let r = bounds[b];
            let (ox, oy) = offsets[b];
            Tile {
                name,
                bounds: Rect {
                    x0: r.x0 + ox,
                    y0: r.y0 + oy,
                    x1: r.x1 + ox,
                    y1: r.y1 + oy,
                },
                n_cells: counts[b],
            }
        })
        .collect()
}

/// One level's per-cell communities, aligned to [`Geometry`] rows.
///
/// Propensities are quantized to `u8` (p·255): colour needs no finer steps,
/// and it keeps a million cells × K=50 at 50 MB instead of 200 MB.
pub struct Communities {
    pub tag: String,
    pub k: usize,
    /// `n × k`, row-major.
    pub prop: Vec<u8>,
    /// Argmax community per cell, [`NO_CLUSTER`] when the cell is absent.
    pub cluster: Vec<u16>,
    /// Entropy / ln K, quantized to `u8`.
    pub entropy: Option<Vec<u8>>,
    /// Cells per community, argmax assignment.
    pub sizes: Vec<usize>,
    /// Non-empty communities, largest first.
    pub by_size: Vec<usize>,
    /// Geometry cells with no propensity row.
    pub n_missing: usize,
    /// Propensity rows naming no known cell.
    pub n_unmatched: usize,
    /// A name per community, for groupings that are not pinto's own
    /// (a lupin round's clusters or cell types); `None` names them `C{c}`.
    pub names: Option<Vec<Box<str>>>,
    /// Short ids `--focus` takes, when the names are longer (a round's
    /// cluster `K3`, named `K3 T_cell`); `None` uses the names.
    pub ids: Option<Vec<Box<str>>>,
}

impl Communities {
    pub fn join(geom: &Geometry, tag: &str, read: PropensityRead) -> Self {
        let (prop_mat, cluster_in, entropy_in, rows) = read;
        let n = geom.n();
        let k = prop_mat.ncols();
        let ln_k = (k.max(2) as f32).ln();

        let mut prop = vec![0u8; n * k];
        let mut cluster = vec![NO_CLUSTER; n];
        let mut entropy = entropy_in.as_ref().map(|_| vec![0u8; n]);
        let mut n_unmatched = 0usize;

        for (r, name) in rows.iter().enumerate() {
            let Some(&i) = geom.index.get(name) else {
                n_unmatched += 1;
                continue;
            };
            for c in 0..k {
                prop[i * k + c] = quantize(prop_mat[(r, c)]);
            }
            cluster[i] = cluster_id(cluster_in[r]);
            if let (Some(out), Some(h)) = (entropy.as_mut(), entropy_in.as_ref()) {
                out[i] = quantize(h[r] / ln_k);
            }
        }
        // Cells outside the graph have no community by design; only graph
        // nodes without a propensity row are a mismatch.
        let in_graph = |i: usize| geom.in_graph.as_ref().is_none_or(|g| g[i]);
        let n_missing = (0..n)
            .filter(|&i| cluster[i] == NO_CLUSTER && in_graph(i))
            .count();
        if n_missing > 0 || n_unmatched > 0 {
            warn!(
                "level {tag}: {n_missing} graph cells have no propensity, \
                 {n_unmatched} propensity rows match no cell"
            );
        }

        let mut sizes = vec![0usize; k];
        for &c in &cluster {
            if let Some(s) = sizes.get_mut(c as usize) {
                *s += 1;
            }
        }
        let mut by_size: Vec<usize> = (0..k).filter(|&c| sizes[c] > 0).collect();
        by_size.sort_by_key(|&c| std::cmp::Reverse(sizes[c]));

        Communities {
            tag: tag.to_string(),
            k,
            prop,
            cluster,
            entropy,
            sizes,
            by_size,
            n_missing,
            n_unmatched,
            names: None,
            ids: None,
        }
    }

    /// One pseudo-community holding `values` (already scaled to `u8`) for
    /// every cell, so per-cell values draw like a community's propensity.
    pub fn single(tag: &str, values: Vec<u8>) -> Self {
        let n = values.len();
        Communities {
            tag: tag.to_string(),
            k: 1,
            prop: values,
            cluster: vec![0; n],
            entropy: None,
            sizes: vec![n],
            by_size: vec![0],
            n_missing: 0,
            n_unmatched: 0,
            names: None,
            ids: None,
        }
    }

    /// A hard grouping as communities: cell `i` belongs wholly to group
    /// `group[i]` ([`NO_CLUSTER`] for none), named `names`. Propensities are
    /// one-hot, so every layer draws it as it draws pinto's own levels.
    pub fn from_groups(tag: &str, group: Vec<u16>, names: Vec<Box<str>>) -> Self {
        let k = names.len();
        let mut prop = vec![0u8; group.len() * k];
        let mut sizes = vec![0usize; k];
        for (i, &g) in group.iter().enumerate() {
            if let Some(s) = sizes.get_mut(g as usize) {
                *s += 1;
                prop[i * k + g as usize] = 255;
            }
        }
        let mut by_size: Vec<usize> = (0..k).filter(|&c| sizes[c] > 0).collect();
        by_size.sort_by_key(|&c| std::cmp::Reverse(sizes[c]));
        Communities {
            tag: tag.to_string(),
            k,
            prop,
            cluster: group,
            entropy: None,
            sizes,
            by_size,
            n_missing: 0,
            n_unmatched: 0,
            names: Some(names),
            ids: None,
        }
    }

    /// The same groups, known by `ids` on the command line.
    pub fn with_ids(self, ids: Vec<Box<str>>) -> Self {
        Communities {
            ids: Some(ids),
            ..self
        }
    }

    /// Community `c`'s name: its given name, else `C{c}`.
    pub fn name(&self, c: usize) -> String {
        match self.names.as_ref().and_then(|n| n.get(c)) {
            Some(name) => name.to_string(),
            None => format!("C{c}"),
        }
    }
}

/// A community id as stored, [`NO_CLUSTER`] when out of range (e.g. -1).
fn cluster_id(c: i64) -> u16 {
    if (0..NO_CLUSTER as i64).contains(&c) {
        c as u16
    } else {
        NO_CLUSTER
    }
}

fn quantize(v: f32) -> u8 {
    (v.clamp(0., 1.) * 255.).round() as u8
}

/// Adjacent cell pairs of one level, as geometry row indices.
#[derive(Default)]
pub struct Edges {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub community: Vec<u16>,
    /// Pairs naming a cell absent from the geometry.
    pub n_unmatched: usize,
}

impl Edges {
    /// Add the pair `l`–`r` of `community`, if both cells are on the map.
    pub fn push(&mut self, geom: &Geometry, l: &str, r: &str, community: i64) {
        match (geom.index.get(l), geom.index.get(r)) {
            (Some(&i), Some(&j)) => {
                self.a.push(i as u32);
                self.b.push(j as u32);
                self.community.push(cluster_id(community));
            }
            _ => self.n_unmatched += 1,
        }
    }

    pub fn len(&self) -> usize {
        self.a.len()
    }
}
