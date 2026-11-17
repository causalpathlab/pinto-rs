//! The super-cells the `-pb` commands fit on, built as cage builds its
//! pseudobulks, and the way back to cells.
//!
//! The cells are coarsened by cage's graph-constrained agglomeration
//! ([`graph_coarsen_multilevel`]) along a kNN graph within each section, on
//! the random projection cage coarsens on (batch-corrected when there is a
//! δ), to about `-d --n-pseudobulk` super-cells. A super-cell never spans
//! two sections or two batches (under `--auto-batch` a section may hold
//! several): the graph has no edge between them, and the spatial seeds of
//! a large input are laid on each (section, batch) pair's own grid. A
//! super-cell's counts are its cells' counts summed, its coordinates their
//! centroid.
//! The method's own cell graph is then folded onto the super-cells
//! ([`fold_edges_to_super`]): an edge inside one super-cell folds into it,
//! and each super-edge records the cell-cell edges it collapses.
//!
//! A `-pb` fit is unravelled to cells by conditional re-inference on each
//! cell's own counts and the cell graph, the fit's parameters held fixed
//! (each method's module says how); the super-cell's label broadcast to
//! its cells is kept only as a diagnostic.

use super::input::SpatialData;
use crate::util::ablation::seeding_inputs;
use crate::util::common::Mat;
use crate::util::graph_coarsen::{
    fold_edges_to_super, graph_coarsen_multilevel, CoarsenConfig, SeedingParams,
};
use legume_numeric::matrix::knn_graph::{symmetric_adjacency, KnnGraph};
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// A `-pb` command's arguments: its method's, `fit`, and the super-cells
/// to fit on.
#[derive(clap::Args, Debug, Clone)]
pub struct PbArgs<F: clap::Args> {
    #[command(flatten)]
    pub fit: F,

    /// Target number of super-cells to fit on (as cage's -d: the cells are
    /// coarsened until about this many remain)
    #[arg(
        long,
        short = 'd',
        default_value_t = 1024,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub n_pseudobulk: usize,
}

/// Neighbours per cell of the graph the cells are coarsened along, pinto's
/// cell-pair graph at its `-k` default: what its memory is charged at.
pub const COARSEN_K: usize = crate::util::input::DEFAULT_KNN;

/// The super-cells of an input.
pub struct SuperCells {
    /// Each cell's super-cell.
    pub of_cell: Vec<usize>,
    /// The super-cells as units: summed counts, centroids, their section
    /// (and its batch label); named `PB0`, `PB1`, ....
    pub data: SpatialData,
}

/// Coarsen `data`'s cells to about `target` super-cells, as cage does
/// (see the module documentation), along pinto's cell-pair graph
/// ([`SpatialData::pair_graph`], taken from `data`), on `proj`, the cells'
/// projection ([`SpatialData::projection`], taken from `data`), which the
/// coarsening normalizes in place and drops.
pub fn super_cells(
    data: &mut SpatialData,
    mut proj: Mat,
    target: usize,
) -> anyhow::Result<SuperCells> {
    let n = data.cells.len();
    anyhow::ensure!(target >= 1, "--n-pseudobulk must be positive");
    // an edge between two sections or two batches is dropped: pinto's
    // graph is built over all sections at once, and under `--auto-batch` a
    // section may hold several batches; a super-cell spans none of them
    let (edges, distances) = data
        .pair_graph
        .take()
        .ok_or_else(|| anyhow::anyhow!("a -pb command reads the cell-pair graph"))?;
    let (edges, distances): (Vec<_>, Vec<_>) = edges
        .into_iter()
        .zip(distances)
        .filter(|&((i, j), _)| {
            data.section[i] == data.section[j] && data.batches[i] == data.batches[j]
        })
        .unzip();
    let graph = KnnGraph {
        adjacency: symmetric_adjacency(n, &edges, &distances),
        edges,
        distances,
        n_nodes: n,
    };
    // each (section, batch) pair is its own seeding group
    let sections: Vec<u32> = data.section.iter().map(|&s| s as u32).collect();
    let (plane, groups) = seeding_inputs(&data.xy, &data.coord_names, &sections, &data.batches);
    let ml = graph_coarsen_multilevel(
        &graph,
        &mut proj,
        &graph.edges,
        CoarsenConfig {
            n_clusters: target.min(n),
            num_levels: 1,
            seeding: Some(SeedingParams {
                coordinates: &plane,
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
    drop(proj);
    Ok(collapse(data, of_cell))
}

/// The super-cells `of_cell` labels as units: counts summed, coordinates
/// the centroid, section and batch labels of their cells (the coarsening
/// keeps a super-cell within one section and batch). The labels are dense,
/// `0..m` each held by a cell, as every level of the coarsening gives them,
/// so no super-cell is empty (no centroid or size factor of nothing).
pub fn collapse(data: &SpatialData, of_cell: Vec<usize>) -> SuperCells {
    let m = of_cell.iter().max().map_or(0, |&s| s + 1);
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
    debug_assert!(size.iter().all(|&k| k > 0.0), "a super-cell with no cell");
    for (s, &k) in size.iter().enumerate() {
        for j in 0..dims {
            xy[(s, j)] /= k;
        }
    }
    let data = SpatialData {
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
        pair_graph: None,
        ablation: data.ablation.clone(),
    };
    SuperCells { of_cell, data }
}

/// A cell graph's undirected edges `edges` folded onto super-cells
/// `of_cell`: the super-edges `(a, b)`, `a < b`, each with the sum of
/// `weight` (one per edge) over the cell-cell edges it collapses. Edges
/// inside one super-cell fold into it.
pub fn fold_weighted(
    of_cell: &[usize],
    edges: &[(usize, usize)],
    weight: &[f64],
) -> (Vec<(usize, usize)>, Vec<f64>) {
    let (super_edges, fine_to_super) = fold_edges_to_super(of_cell, edges, None);
    let mut w = vec![0f64; super_edges.len()];
    for (s, &we) in fine_to_super.iter().zip(weight) {
        if let Some(s) = *s {
            w[s] += we;
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
