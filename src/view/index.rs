//! Spatial index and summary pyramid.
//!
//! [`Grid`] buckets cells into square bins (CSR: offsets + cell ids), so a
//! viewport touches only the cells inside it. [`Pyramid`] summarizes those
//! bins at successively coarser 2×2 merges: when a screen pixel covers many
//! cells, the renderer reads one bin instead of every cell under it.

use super::data::{Communities, Edges, Rect, NO_CLUSTER};
use crate::util::common::*;

/// Square bins over the world bounds, cells stored bin by bin.
pub struct Grid {
    pub origin: (f32, f32),
    /// Bin side, world units.
    pub bin: f32,
    pub nx: usize,
    pub ny: usize,
    cells: Csr,
}

impl Grid {
    /// Size bins for about `per_bin` cells each on average over `bounds`.
    pub fn build(x: &[f32], y: &[f32], bounds: Rect, per_bin: f32) -> Self {
        let n = x.len().max(1) as f32;
        let area = (bounds.width() * bounds.height()).max(f32::MIN_POSITIVE);
        let mut bin = (area * per_bin / n).sqrt();
        if !bin.is_finite() || bin <= 0. {
            bin = bounds.width().max(bounds.height()).max(1.);
        }
        let nx = ((bounds.width() / bin).floor() as usize + 1).max(1);
        let ny = ((bounds.height() / bin).floor() as usize + 1).max(1);

        let mut grid = Grid {
            origin: (bounds.x0, bounds.y0),
            bin,
            nx,
            ny,
            cells: Csr::default(),
        };

        let bins: Vec<u32> = x
            .iter()
            .zip(y)
            .map(|(&xi, &yi)| grid.bin_id(xi, yi))
            .collect();
        grid.cells = Csr::bucket(&bins, nx * ny);
        grid
    }

    /// Bin holding world point `(x, y)`, clamped to the grid.
    pub fn bin_of(&self, x: f32, y: f32) -> (usize, usize) {
        let ix = ((x - self.origin.0) / self.bin).floor().max(0.) as usize;
        let iy = ((y - self.origin.1) / self.bin).floor().max(0.) as usize;
        (ix.min(self.nx - 1), iy.min(self.ny - 1))
    }

    pub fn bin_id(&self, x: f32, y: f32) -> u32 {
        let (ix, iy) = self.bin_of(x, y);
        (iy * self.nx + ix) as u32
    }

    /// The cell nearest `at` within `radius`, if any. `x`, `y` are the
    /// coordinates the grid was built from.
    pub fn nearest(&self, x: &[f32], y: &[f32], at: (f32, f32), radius: f32) -> Option<usize> {
        let (ix0, iy0) = self.bin_of(at.0 - radius, at.1 - radius);
        let (ix1, iy1) = self.bin_of(at.0 + radius, at.1 + radius);
        let mut best: Option<(f32, usize)> = None;
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                for &i in self.cells(ix, iy) {
                    let i = i as usize;
                    let d = (x[i] - at.0).hypot(y[i] - at.1);
                    if d <= radius && best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, i));
                    }
                }
            }
        }
        best.map(|(_, i)| i)
    }

    /// Bins holding at least one cell.
    pub fn occupied_bins(&self) -> usize {
        self.cells.start.windows(2).filter(|w| w[1] > w[0]).count()
    }

    /// Cells in bin `(ix, iy)`.
    pub fn cells(&self, ix: usize, iy: usize) -> &[u32] {
        self.cells.get(iy * self.nx + ix)
    }
}

/// Items grouped by bin: bin `b` holds `ids[start[b]..start[b + 1]]`.
#[derive(Default)]
struct Csr {
    start: Vec<u32>,
    ids: Vec<u32>,
}

impl Csr {
    /// Counting sort of item `i` into bin `bins[i]`.
    fn bucket(bins: &[u32], n_bins: usize) -> Self {
        let mut start = vec![0u32; n_bins + 1];
        for &b in bins {
            start[b as usize + 1] += 1;
        }
        for i in 0..n_bins {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut ids = vec![0u32; bins.len()];
        for (i, &b) in bins.iter().enumerate() {
            let slot = &mut fill[b as usize];
            ids[*slot as usize] = i as u32;
            *slot += 1;
        }
        Csr { start, ids }
    }

    fn get(&self, b: usize) -> &[u32] {
        &self.ids[self.start[b] as usize..self.start[b + 1] as usize]
    }
}

/// Edges bucketed by the grid bin of their first endpoint. A view reads the
/// bins it covers, widened by `max_len`, the longest edge.
pub struct EdgeIndex {
    nx: usize,
    edges: Csr,
    pub max_len: f32,
}

impl EdgeIndex {
    pub fn build(grid: &Grid, x: &[f32], y: &[f32], edges: &Edges) -> Self {
        let bins: Vec<u32> = edges
            .a
            .iter()
            .map(|&a| grid.bin_id(x[a as usize], y[a as usize]))
            .collect();
        let csr = Csr::bucket(&bins, grid.nx * grid.ny);
        let max_len = edges
            .a
            .iter()
            .zip(&edges.b)
            .map(|(&a, &b)| {
                let (a, b) = (a as usize, b as usize);
                (x[a] - x[b]).hypot(y[a] - y[b])
            })
            .fold(0f32, f32::max);
        EdgeIndex {
            nx: grid.nx,
            edges: csr,
            max_len,
        }
    }

    /// Edges whose first endpoint is in bin `(ix, iy)`.
    pub fn edges(&self, ix: usize, iy: usize) -> &[u32] {
        self.edges.get(iy * self.nx + ix)
    }
}

/// One pyramid level: `nx × ny` bins of side `bin`.
pub struct PyramidLevel {
    pub nx: usize,
    pub ny: usize,
    pub bin: f32,
    pub count: Vec<u32>,
    /// Summed propensity, `nx·ny × k`.
    pub prop: Vec<f32>,
    /// Summed normalized entropy (0 when the level has none).
    pub entropy: Vec<f32>,
    /// Community with the largest summed propensity, [`NO_CLUSTER`] if empty.
    pub top: Vec<u16>,
    /// Half the mean cell count of occupied bins: the count at which a bin
    /// is drawn fully opaque.
    pub full: f32,
}

impl PyramidLevel {
    fn empty(nx: usize, ny: usize, bin: f32, k: usize) -> Self {
        let nb = nx * ny;
        PyramidLevel {
            nx,
            ny,
            bin,
            count: vec![0; nb],
            prop: vec![0.; nb * k],
            entropy: vec![0.; nb],
            top: vec![NO_CLUSTER; nb],
            full: 1.,
        }
    }

    /// Fill the per-bin summaries derived from the sums.
    fn finish(&mut self, k: usize) {
        let prop = &self.prop;
        self.top.par_iter_mut().enumerate().for_each(|(b, top)| {
            *top = argmax(&prop[b * k..(b + 1) * k]);
        });
        let (filled, total) = self
            .count
            .iter()
            .filter(|&&c| c > 0)
            .fold((0usize, 0u64), |(n, s), &c| (n + 1, s + c as u64));
        self.full = 0.5 * total as f32 / filled.max(1) as f32;
    }

    pub fn bytes(&self) -> usize {
        self.count.len() * 4 + self.prop.len() * 4 + self.entropy.len() * 4 + self.top.len() * 2
    }
}

fn argmax(v: &[f32]) -> u16 {
    let mut best = NO_CLUSTER;
    let mut best_v = 0f32;
    for (c, &p) in v.iter().enumerate() {
        if p > best_v {
            best_v = p;
            best = c as u16;
        }
    }
    best
}

/// Level 0 is the [`Grid`] itself; each next level merges 2×2 bins, down
/// to a single bin.
pub struct Pyramid {
    pub levels: Vec<PyramidLevel>,
}

impl Pyramid {
    pub fn build(grid: &Grid, comm: &Communities) -> Self {
        let k = comm.k;
        let mut base = PyramidLevel::empty(grid.nx, grid.ny, grid.bin, k);
        let nx = grid.nx;

        base.prop
            .par_chunks_mut(k.max(1))
            .zip(base.count.par_iter_mut())
            .zip(base.entropy.par_iter_mut())
            .enumerate()
            .for_each(|(b, ((prop, count), ent))| {
                let cells = grid.cells(b % nx, b / nx);
                *count = cells.len() as u32;
                for &i in cells {
                    let i = i as usize;
                    let row = &comm.prop[i * k..(i + 1) * k];
                    for (acc, &q) in prop.iter_mut().zip(row) {
                        *acc += q as f32 / 255.;
                    }
                    if let Some(h) = comm.entropy.as_ref() {
                        *ent += h[i] as f32 / 255.;
                    }
                }
            });
        base.finish(k);

        let mut levels = vec![base];
        loop {
            let prev = levels.last().expect("non-empty");
            if prev.nx == 1 && prev.ny == 1 {
                break;
            }
            let next = coarsen(prev, k);
            levels.push(next);
        }
        Pyramid { levels }
    }

    pub fn bytes(&self) -> usize {
        self.levels.iter().map(PyramidLevel::bytes).sum()
    }
}

fn coarsen(prev: &PyramidLevel, k: usize) -> PyramidLevel {
    let nx = prev.nx.div_ceil(2);
    let ny = prev.ny.div_ceil(2);
    let mut next = PyramidLevel::empty(nx, ny, prev.bin * 2., k);

    next.prop
        .par_chunks_mut(k.max(1))
        .zip(next.count.par_iter_mut())
        .zip(next.entropy.par_iter_mut())
        .enumerate()
        .for_each(|(b, ((prop, count), ent))| {
            let (bx, by) = (b % nx, b / nx);
            for (cx, cy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (px, py) = (2 * bx + cx, 2 * by + cy);
                if px >= prev.nx || py >= prev.ny {
                    continue;
                }
                let pb = py * prev.nx + px;
                *count += prev.count[pb];
                *ent += prev.entropy[pb];
                for (acc, &p) in prop.iter_mut().zip(&prev.prop[pb * k..(pb + 1) * k]) {
                    *acc += p;
                }
            }
        });
    next.finish(k);
    next
}
