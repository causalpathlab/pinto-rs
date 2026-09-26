//! Load a pinto run for viewing: manifest → cell geometry ⋈ propensity.
//!
//! Everything comes from files a pinto run already writes, through the same
//! readers the rest of the crate uses: `coord_pairs` for positions and batch
//! labels, `propensity` for the per-cell community mixture (one per level),
//! and `link_community` for the per-edge labels.

use super::markers::FeatureRates;
use crate::util::common::*;
use crate::util::metadata::{LevelInfo, PintoMetadata};
use crate::util::parquet_io::{
    read_cells_from_coord_pairs, read_feature_community, read_link_community, read_propensity,
    CellTable, PropensityRead,
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
        let m = frac * self.width().max(self.height());
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
    /// look for the same file name next to the prefix.
    pub fn resolve(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.exists() {
            return p.to_path_buf();
        }
        match (Path::new(&self.prefix).parent(), p.file_name()) {
            (Some(dir), Some(name)) => dir.join(name),
            _ => p.to_path_buf(),
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

    pub fn load_geometry(&self) -> anyhow::Result<Geometry> {
        let pairs = self.meta.outputs.coord_pairs.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "{}: no coord_pairs output listed; was the run fit with --coord?",
                self.source()
            )
        })?;
        let cells = read_cells_from_coord_pairs(
            &self.resolve(pairs),
            self.meta.outputs.coord_columns.as_deref(),
        )?;
        Ok(Geometry::from_cells(cells))
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
        let (pairs, community, _) = read_link_community(&self.resolve(path))?;
        Ok(Edges::join(geom, &pairs, &community))
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
}

impl Geometry {
    pub fn from_cells(cells: CellTable) -> Self {
        let CellTable {
            names,
            coords,
            batches,
            index,
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
        }
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
        let n_missing = n - (rows.len() - n_unmatched);
        if n_missing > 0 || n_unmatched > 0 {
            warn!(
                "level {tag}: {n_missing} cells have no propensity, \
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
        }
    }
}

impl Communities {
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
pub struct Edges {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub community: Vec<u16>,
    /// Pairs naming a cell absent from the geometry.
    pub n_unmatched: usize,
}

impl Edges {
    pub fn join(geom: &Geometry, pairs: &[(Box<str>, Box<str>)], community: &[i64]) -> Self {
        let mut a = Vec::with_capacity(pairs.len());
        let mut b = Vec::with_capacity(pairs.len());
        let mut comm = Vec::with_capacity(pairs.len());
        let mut n_unmatched = 0usize;
        for ((l, r), &c) in pairs.iter().zip(community) {
            match (geom.index.get(l), geom.index.get(r)) {
                (Some(&i), Some(&j)) => {
                    a.push(i as u32);
                    b.push(j as u32);
                    comm.push(cluster_id(c));
                }
                _ => n_unmatched += 1,
            }
        }
        Edges {
            a,
            b,
            community: comm,
            n_unmatched,
        }
    }

    pub fn len(&self) -> usize {
        self.a.len()
    }
}
