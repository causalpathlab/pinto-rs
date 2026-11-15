//! The super-cells `spatialde2-pb` fits on, built as cage builds its pseudobulks, and the way back to cells.
//!
//! The cells are coarsened by cage's graph-constrained agglomeration
//! ([`graph_coarsen_multilevel`]) along a kNN graph within each section, on
//! the random projection cage coarsens on (batch-corrected when there is a
//! δ), to about `-d --n-pseudobulk` super-cells. A super-cell never spans
//! two sections: the graph has no edge between them, and the spatial seeds
//! of a large input are laid on each section's own grid. A super-cell's
//! counts are its cells' counts summed, its coordinates their centroid.
//! The method's own cell graph is then folded onto the super-cells
//! ([`fold_edges_to_super`]): an edge inside one super-cell folds into it,
//! and each super-edge records the cell-cell edges it collapses.
//!
//! A `-pb` fit is unravelled to cells by conditional re-inference on each
//! cell's own counts and the cell graph, the fit's parameters held fixed
//! ([`super::spatialde2`] says how); the super-cell's label broadcast to
//! its cells is kept only as a diagnostic.

use super::graph::{knn_within_sections, Neighbours};
use super::input::SpatialData;
use crate::util::common::Mat;
use crate::util::graph_coarsen::{
    fold_edges_to_super, graph_coarsen_multilevel, CoarsenConfig, SeedingParams,
};
use legume_numeric::matrix::knn_graph::{symmetric_adjacency, KnnGraph};
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// Neighbours per cell of the graph the cells are coarsened along: cage's
/// `-k` default.
pub const COARSEN_K: usize = 5;

/// The super-cells of an input.
pub struct SuperCells {
    /// Each cell's super-cell.
    pub of_cell: Vec<usize>,
    /// The super-cells as units: summed counts, centroids, their section
    /// (and its batch label); named `PB0`, `PB1`, ....
    pub data: SpatialData,
}

/// Each undirected edge `(i, j)`, `i < j`, of neighbour lists, once.
pub fn undirected(nb: &Neighbours) -> (Vec<(usize, usize)>, Vec<f32>) {
    let mut pairs: Vec<((usize, usize), f32)> = nb
        .index
        .iter()
        .zip(&nb.dist)
        .enumerate()
        .flat_map(|(i, (js, ds))| {
            js.iter()
                .zip(ds)
                .filter(move |(&j, _)| j != i)
                .map(move |(&j, &d)| ((i.min(j), i.max(j)), d))
        })
        .collect();
    pairs.sort_unstable_by_key(|p| p.0);
    pairs.dedup_by_key(|p| p.0);
    pairs.into_iter().unzip()
}

/// Coarsen `data`'s cells to about `target` super-cells, as cage does
/// (see the module documentation); `data.projection` must be read.
pub fn super_cells(data: &SpatialData, target: usize) -> anyhow::Result<SuperCells> {
    let n = data.cells.len();
    anyhow::ensure!(target >= 1, "--n-pseudobulk must be positive");
    let proj = data
        .projection
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("the -pb commands read the cells' projection"))?;
    let (edges, distances) = undirected(&knn_within_sections(data, COARSEN_K));
    let graph = KnnGraph {
        adjacency: symmetric_adjacency(n, &edges, &distances),
        edges,
        distances,
        n_nodes: n,
    };
    let groups: Vec<u32> = data.section.iter().map(|&s| s as u32).collect();
    let ml = graph_coarsen_multilevel(
        &graph,
        &mut proj.clone(),
        &graph.edges,
        CoarsenConfig {
            n_clusters: target.min(n),
            num_levels: 1,
            seeding: Some(SeedingParams {
                coordinates: &data.xy,
                groups: Some(&groups),
            }),
            modularity_veto: None,
        },
    );
    let of_cell = ml
        .all_cell_labels
        .last()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("coarsening produced no level"))?;
    Ok(SuperCells {
        data: collapse(data, &of_cell),
        of_cell,
    })
}

/// The units of `of_cell`'s super-cells: counts summed, coordinates the
/// centroid, section and batch label of their cells (the coarsening keeps
/// a super-cell within one section).
pub fn collapse(data: &SpatialData, of_cell: &[usize]) -> SpatialData {
    let m = of_cell.iter().max().map_or(0, |&x| x + 1);
    let mut coo = CooMatrix::<f32>::new(data.counts.nrows(), m);
    for (c, col) in data.counts.col_iter().enumerate() {
        for (&g, &v) in col.row_indices().iter().zip(col.values()) {
            coo.push(g, of_cell[c], v);
        }
    }
    let dims = data.xy.ncols();
    let mut xy = Mat::zeros(m, dims);
    let mut size = vec![0f32; m];
    let mut section = vec![0usize; m];
    let mut batches: Vec<Box<str>> = vec![Box::from(""); m];
    for (c, &s) in of_cell.iter().enumerate() {
        size[s] += 1.0;
        for j in 0..dims {
            xy[(s, j)] += data.xy[(c, j)];
        }
        section[s] = data.section[c];
        batches[s] = data.batches[c].clone();
    }
    for (s, &k) in size.iter().enumerate() {
        for j in 0..dims {
            xy[(s, j)] /= k;
        }
    }
    SpatialData {
        // CSC from COO sums duplicate entries
        counts: CscMatrix::from(&coo),
        genes: data.genes.clone(),
        cells: (0..m).map(|s| format!("PB{s}").into()).collect(),
        xy,
        coord_names: data.coord_names.clone(),
        batches,
        section,
        n_sections: data.n_sections,
        delta: data.delta.clone(),
        projection: None,
    }
}

/// A cell graph's undirected edges `edges` folded onto super-cells
/// `of_cell`: the super-edges `(a, b)`, `a < b`, each with the sum of
/// `weight` over the cell-cell edges it collapses (with unit weights, how
/// many it collapses). Edges inside one super-cell fold into it.
pub fn fold_weighted(
    of_cell: &[usize],
    edges: &[(usize, usize)],
    weight: impl Fn(usize) -> f64,
) -> (Vec<(usize, usize)>, Vec<f64>) {
    let (super_edges, fine_to_super) = fold_edges_to_super(of_cell, edges, None);
    let mut w = vec![0f64; super_edges.len()];
    for (e, s) in fine_to_super.iter().enumerate() {
        if let Some(s) = *s {
            w[s] += weight(e);
        }
    }
    (super_edges, w)
}

/// What a `-pb` fit's unravelled labels share with the super-cells'
/// labels broadcast to their cells: the fraction of cells whose label
/// stayed, for `baseline.json` (broadcasting is a diagnostic only).
pub fn broadcast_agreement(unravelled: &[usize], broadcast: &[usize]) -> f64 {
    let same = unravelled
        .iter()
        .zip(broadcast)
        .filter(|(a, b)| a == b)
        .count();
    same as f64 / unravelled.len().max(1) as f64
}
