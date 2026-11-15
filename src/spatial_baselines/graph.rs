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
        let sorted: Vec<Vec<usize>> = self
            .index
            .par_iter()
            .map(|js| {
                let mut s = js.clone();
                s.sort_unstable();
                s
            })
            .collect();
        // each edge i → j whose reverse is missing, as (j, i, d)
        let mut reverse: Vec<(usize, usize, f32)> = (0..self.n_cells())
            .into_par_iter()
            .flat_map_iter(|i| {
                let sorted = &sorted;
                self.index[i]
                    .iter()
                    .zip(&self.dist[i])
                    .filter(move |(&j, _)| sorted[j].binary_search(&i).is_err())
                    .map(move |(&j, &d)| (j, i, d))
                    .collect::<Vec<_>>()
            })
            .collect();
        reverse.par_sort_unstable_by_key(|&(j, i, _)| (j, i));
        let mut at = 0;
        let mut lists: Vec<Vec<(f32, usize)>> = Vec::with_capacity(self.n_cells());
        for (i, (js, ds)) in self.index.iter().zip(&self.dist).enumerate() {
            let mut l: Vec<(f32, usize)> = ds.iter().copied().zip(js.iter().copied()).collect();
            while at < reverse.len() && reverse[at].0 == i {
                l.push((reverse[at].2, reverse[at].1));
                at += 1;
            }
            lists.push(l);
        }
        lists
            .par_iter_mut()
            .for_each(|l| l.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))));
        let (index, dist) = lists
            .into_iter()
            .map(|l| {
                let (d, j): (Vec<f32>, Vec<usize>) = l.into_iter().unzip();
                (j, d)
            })
            .unzip();
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
