//! Spatial neighbour graphs, built within each section only: two sections
//! never neighbour each other, whatever their coordinates.
//!
//! Neighbours come from the exact all-pairs Euclidean kNN of
//! [`knn_rows_l2`] on each section's coordinates (ties by index), so a
//! graph is sparse, `n × k`, and never an `n × n` matrix.

use super::input::SpatialData;
use legume_numeric::matrix::knn::all_pairs::knn_rows_l2;
use rayon::prelude::*;

/// Each cell's neighbours (global indices, the cell itself excluded),
/// nearest first, with their Euclidean distances.
#[derive(Debug, Clone)]
pub struct Neighbours {
    pub index: Vec<Vec<usize>>,
    pub dist: Vec<Vec<f32>>,
}

impl Neighbours {
    pub fn n_cells(&self) -> usize {
        self.index.len()
    }

    pub fn n_edges(&self) -> usize {
        self.index.iter().map(Vec::len).sum()
    }

    /// The same graph with every edge in both directions, each listed once
    /// per cell, neighbours re-sorted by distance (ties by index).
    pub fn symmetrised(&self) -> Neighbours {
        let n = self.n_cells();
        let mut index: Vec<Vec<usize>> = self.index.clone();
        let mut dist: Vec<Vec<f32>> = self.dist.clone();
        for i in 0..n {
            for (k, &j) in self.index[i].iter().enumerate() {
                if !self.index[j].contains(&i) {
                    index[j].push(i);
                    dist[j].push(self.dist[i][k]);
                }
            }
        }
        for i in 0..n {
            let mut order: Vec<usize> = (0..index[i].len()).collect();
            order.sort_by(|&a, &b| {
                dist[i][a]
                    .total_cmp(&dist[i][b])
                    .then(index[i][a].cmp(&index[i][b]))
            });
            index[i] = order.iter().map(|&a| index[i][a]).collect();
            dist[i] = order.iter().map(|&a| dist[i][a]).collect();
        }
        Neighbours { index, dist }
    }
}

/// The `k` nearest neighbours of each cell within its section (fewer when
/// the section has fewer other cells).
pub fn knn_within_sections(data: &SpatialData, k: usize) -> Neighbours {
    let n = data.cells.len();
    let mut index = vec![Vec::new(); n];
    let mut dist = vec![Vec::new(); n];
    let sections = data.section_cells();
    let found: Vec<_> = sections
        .par_iter()
        .map(|cells| knn_rows_l2(&data.xy.select_rows(cells.iter()), k))
        .collect();
    for (cells, (idx, d)) in sections.iter().zip(found) {
        for (r, (js, ds)) in idx.into_iter().zip(d).enumerate() {
            index[cells[r]] = js.into_iter().map(|j| cells[j]).collect();
            dist[cells[r]] = ds;
        }
    }
    Neighbours { index, dist }
}

/// Every other cell of the same section within `radius` (inclusive),
/// searched among the `max_k` nearest; logs how many cells hit the cap.
pub fn radius_within_sections(data: &SpatialData, radius: f32, max_k: usize) -> Neighbours {
    let mut g = knn_within_sections(data, max_k);
    let mut capped = 0usize;
    for (js, ds) in g.index.iter_mut().zip(g.dist.iter_mut()) {
        let keep = ds.iter().take_while(|&&d| d <= radius).count();
        if keep == js.len() && keep == max_k {
            capped += 1;
        }
        js.truncate(keep);
        ds.truncate(keep);
    }
    if capped > 0 {
        log::warn!("{capped} cells have {max_k} or more neighbours within {radius}; kept {max_k}");
    }
    g
}
