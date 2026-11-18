//! The cells laid out in two dimensions by a table of theirs (any `N × K`
//! table a run writes: a level's propensity, a cell embedding), drawn in
//! place of the tissue: legume-numeric's layouts on a kNN graph of the
//! table's rows.
//!
//! UMAP and tUMAP run on every cell. The others hold every pair of points
//! (PHATE's and the stress layout's distances, the diffusion map's and
//! LLE's eigensolves, ForceAtlas2's repulsion), so past [`LANDMARKS`] cells
//! they lay out that many landmark cells, drawn at random from the seed,
//! and place every other cell from its nearest landmarks in the table
//! ([`project_via_knn`]), as legume-numeric does for many points.

use crate::util::common::Mat;
use legume_numeric::matrix::knn::ColumnDict;
use legume_numeric::matrix::knn_graph::{KnnGraph, KnnGraphArgs};
use legume_numeric::matrix::landmark_init::landmark_init_2d;
use legume_numeric::matrix::layout::{
    diffusion_map_2d_on_graph, forceatlas2_layout_2d_on_graph, lle_layout_2d_on_graph,
    phate_layout_2d_on_graph, project_via_knn, stress_layout_2d_on_graph, PhateArgs,
};
use legume_numeric::matrix::umap::Umap;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// A layout of the cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    Umap,
    /// UMAP with its kernel's `a = b = 1`: tighter clusters.
    Tumap,
    Phate,
    Diffusion,
    ForceAtlas2,
    Lle,
    Stress,
}

impl Method {
    /// Every layout, in the order the view cycles through them.
    pub const ALL: [Method; 7] = [
        Method::Umap,
        Method::Tumap,
        Method::Phate,
        Method::Diffusion,
        Method::ForceAtlas2,
        Method::Lle,
        Method::Stress,
    ];

    /// Its name, as the view shows it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Method::Umap => "UMAP",
            Method::Tumap => "tUMAP",
            Method::Phate => "PHATE",
            Method::Diffusion => "diffusion map",
            Method::ForceAtlas2 => "ForceAtlas2",
            Method::Lle => "LLE",
            Method::Stress => "stress",
        }
    }

    /// Whether it lays out every cell, not only landmarks.
    fn scales(self) -> bool {
        matches!(self, Method::Umap | Method::Tumap)
    }
}

/// Neighbours per cell in the graph the layouts run on.
const KNN: usize = 15;

/// The most cells a layout that holds every pair of points is run on;
/// past it, as many landmark cells.
pub const LANDMARKS: usize = 2000;

/// Landmarks of the kNN graph that set UMAP's start.
const START_LANDMARKS: usize = 64;

/// Iterations of ForceAtlas2.
const FORCEATLAS2_ITERATIONS: usize = 300;

/// The rows of `table` (`N × K`, one row per cell) laid out by `method`,
/// `N × 2` in the same order, from `seed`.
pub fn lay_out(table: &Mat, method: Method, seed: u64) -> anyhow::Result<Mat> {
    let n = table.nrows();
    if n < 3 {
        return Ok(Mat::zeros(n, 2));
    }
    if method.scales() || n <= LANDMARKS {
        return on_graph(table, method, seed);
    }
    let mut picked: Vec<usize> = (0..n).collect();
    picked.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
    picked.truncate(LANDMARKS);
    picked.sort_unstable();
    let landmarks = table.select_rows(&picked);
    let coords = on_graph(&landmarks, method, seed)?;
    let dict =
        ColumnDict::<usize>::from_dmatrix(landmarks.transpose(), (0..picked.len()).collect());
    Ok(project_via_knn(
        &table.transpose(),
        &dict,
        &coords,
        KNN,
        PhateArgs::default().alpha,
    ))
}

/// `method` on every row of `table`, on their kNN graph.
fn on_graph(table: &Mat, method: Method, seed: u64) -> anyhow::Result<Mat> {
    let n = table.nrows();
    let args = KnnGraphArgs {
        knn: KNN.min(n - 1),
        block_size: 1000,
        reciprocal: false,
    };
    let (graph, weights) = KnnGraph::from_rows_fuzzy(table, args)?;
    let edges: Vec<(usize, usize, f32)> = graph
        .edges
        .iter()
        .zip(&weights)
        .map(|(&(i, j), &w)| (i, j, w))
        .collect();
    let umap = |u: Umap| {
        let start = landmark_init_2d(table, &edges, START_LANDMARKS, seed);
        let xy = Umap { seed, ..u }.fit(&edges, n, &start);
        Mat::from_row_slice(n, 2, &xy)
    };
    Ok(match method {
        Method::Umap => umap(Umap::default()),
        Method::Tumap => umap(Umap::tumap()),
        Method::Phate => phate_layout_2d_on_graph(n, &edges, &PhateArgs::default()),
        Method::Diffusion => diffusion_map_2d_on_graph(n, &edges, 1),
        Method::ForceAtlas2 => {
            forceatlas2_layout_2d_on_graph(n, &edges, FORCEATLAS2_ITERATIONS, seed)
        }
        Method::Lle => lle_layout_2d_on_graph(table, &edges),
        Method::Stress => stress_layout_2d_on_graph(n, &edges, seed),
    })
}

/// A table of the run's cells a layout is made of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    /// As the view names it: `propensity L2`, `cell embedding`.
    pub name: String,
    pub path: std::path::PathBuf,
    /// A propensity table (communities, then `cluster`, `entropy` and any
    /// coordinates), not a plain matrix.
    propensity: bool,
}

/// Every `N × K` table of `run`'s cells there is: each level's propensity,
/// then the cell embedding when the run has one.
#[must_use]
pub fn sources(run: &super::data::Run) -> Vec<Source> {
    let mut out: Vec<Source> = run
        .levels
        .iter()
        .map(|l| Source {
            name: format!("propensity {}", l.tag),
            path: run.resolve(&l.propensity),
            propensity: true,
        })
        .collect();
    if let Some(e) = run.meta.outputs.cell_embedding.as_deref() {
        out.push(Source {
            name: "cell embedding".into(),
            path: run.resolve(e),
            propensity: false,
        });
    }
    out
}

/// `source`'s cell names and rows (`N × K`); `coord_names`, the columns a
/// propensity table may carry after its own, are left out.
pub fn read(source: &Source, coord_names: &[Box<str>]) -> anyhow::Result<(Vec<Box<str>>, Mat)> {
    use crate::util::parquet_io::{read_labelled_matrix, read_propensity};
    if source.propensity {
        let exclude = coord_names.iter().cloned().collect();
        let (prop, _, _, cells) = read_propensity(&source.path, &exclude)?;
        Ok((cells, prop))
    } else {
        let m = read_labelled_matrix(&source.path)?;
        Ok((m.rows, m.mat))
    }
}

#[cfg(test)]
#[path = "tests/layout.rs"]
mod tests;
