//! The cells laid out in two dimensions by a table of theirs (any `N × K`
//! table a run writes: a level's propensity, a cell embedding), drawn in
//! place of the tissue, by legume-numeric's coarsened layouts: the cells'
//! fuzzy kNN graph coarsened into pseudobulks, a layout of the pseudobulk
//! graph, and every cell placed from the pseudobulks near it in that graph
//! ([`Coarsened::place_cells`]). Every layout runs on at most
//! [`PSEUDOBULKS`] points, however many cells there are.

use crate::util::common::Mat;
use legume_numeric::matrix::layout::{
    coarsened, diffusion_map_2d_on_graph, forceatlas2_layout_2d_on_graph, lle_layout_2d_on_graph,
    phate_layout_2d_on_graph, stress_layout_2d_on_graph, tsne_layout_2d_on_graph,
    umap_layout_2d_on_graph, CoarsenArgs, Coarsened, PhateArgs, TsneArgs,
};
use legume_numeric::matrix::umap::Umap;

/// A layout of the cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    Umap,
    /// UMAP with its kernel's `a = b = 1`: tighter clusters.
    Tumap,
    Tsne,
    Phate,
    Diffusion,
    ForceAtlas2,
    Lle,
    Stress,
}

impl Method {
    /// Every layout, in the order the view cycles through them.
    pub const ALL: [Method; 8] = [
        Method::Umap,
        Method::Tumap,
        Method::Tsne,
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
            Method::Tsne => "t-SNE",
            Method::Phate => "PHATE",
            Method::Diffusion => "diffusion map",
            Method::ForceAtlas2 => "ForceAtlas2",
            Method::Lle => "LLE",
            Method::Stress => "stress",
        }
    }

    /// The layout after it in [`Method::ALL`]; `None` after the last.
    #[must_use]
    pub fn next(self) -> Option<Method> {
        let at = Method::ALL.iter().position(|&m| m == self)?;
        Method::ALL.get(at + 1).copied()
    }
}

/// Neighbours per cell, counting the cell itself, in the graph coarsened.
const KNN: usize = 15;

/// The most pseudobulks the cells are coarsened into: the points a layout
/// is run on.
pub const PSEUDOBULKS: usize = 1000;

/// Pseudobulks a cell is placed from: few, so a cell stays by its own.
const PLACE_KNN: usize = 3;

/// Exponent of the kernel that weighs those pseudobulks.
const PLACE_ALPHA: f32 = 10.0;

/// Iterations of ForceAtlas2.
const FORCEATLAS2_ITERATIONS: usize = 300;

/// A table of the cells coarsened into pseudobulks, once for every layout
/// of it.
pub struct Coarse {
    /// The cells, in the table's order.
    pub names: Vec<Box<str>>,
    /// The table, `K × N`: one column per cell.
    table_kn: Mat,
    /// Its pseudobulks; `None` under 3 cells, which are all put at 0.
    pbs: Option<Coarsened>,
}

impl Coarse {
    /// The rows of `table` (`N × K`, one row per cell, named `names`)
    /// coarsened from `seed`.
    pub fn new(names: Vec<Box<str>>, table: &Mat, seed: u64) -> anyhow::Result<Self> {
        let n = table.nrows();
        let table_kn = table.transpose();
        let pbs = (n >= 3)
            .then(|| {
                coarsened(
                    &table_kn,
                    &CoarsenArgs {
                        knn: KNN.min(n - 1),
                        n_landmarks: PSEUDOBULKS,
                        seed,
                        ..CoarsenArgs::default()
                    },
                )
            })
            .transpose()?;
        Ok(Coarse {
            names,
            table_kn,
            pbs,
        })
    }

    /// The cells laid out by `method`, `N × 2` in the table's order, from
    /// `seed`.
    pub fn lay_out(&self, method: Method, seed: u64) -> anyhow::Result<Mat> {
        let Some(pbs) = &self.pbs else {
            return Ok(Mat::zeros(self.table_kn.ncols(), 2));
        };
        let pb_coords = on_pseudobulks(pbs, method, seed)?;
        Ok(pbs.place_cells(&self.table_kn, &pb_coords, PLACE_KNN, PLACE_ALPHA))
    }
}

/// `method` on the pseudobulk graph of `pbs`, `n_pb × 2`.
fn on_pseudobulks(pbs: &Coarsened, method: Method, seed: u64) -> anyhow::Result<Mat> {
    let (n, edges) = (pbs.n_pb(), &pbs.pb_edges);
    Ok(match method {
        Method::Umap => umap_layout_2d_on_graph(
            n,
            edges,
            &Umap {
                seed,
                ..Umap::default()
            },
        ),
        Method::Tumap => umap_layout_2d_on_graph(
            n,
            edges,
            &Umap {
                seed,
                ..Umap::tumap()
            },
        ),
        Method::Tsne => tsne_layout_2d_on_graph(
            n,
            edges,
            &TsneArgs {
                seed,
                ..TsneArgs::default()
            },
        )?,
        Method::Phate => phate_layout_2d_on_graph(n, edges, &PhateArgs::default()),
        Method::Diffusion => diffusion_map_2d_on_graph(n, edges, 1),
        Method::ForceAtlas2 => {
            forceatlas2_layout_2d_on_graph(n, edges, FORCEATLAS2_ITERATIONS, seed)
        }
        Method::Lle => lle_layout_2d_on_graph(&pbs.pb_features.transpose(), edges),
        Method::Stress => stress_layout_2d_on_graph(n, edges, seed),
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

impl Source {
    /// Its cell names and rows (`N × K`); `coord_names`, the columns a
    /// propensity table may carry after its own, are left out.
    pub fn read(&self, coord_names: &[Box<str>]) -> anyhow::Result<(Vec<Box<str>>, Mat)> {
        use crate::util::parquet_io::{read_labelled_matrix, read_propensity};
        if self.propensity {
            let exclude = coord_names.iter().cloned().collect();
            let (prop, _, _, cells) = read_propensity(&self.path, &exclude)?;
            Ok((cells, prop))
        } else {
            let m = read_labelled_matrix(&self.path)?;
            Ok((m.rows, m.mat))
        }
    }
}

#[cfg(test)]
#[path = "tests/layout.rs"]
mod tests;
