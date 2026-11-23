//! The cells laid out in two dimensions by a table of theirs (any `N × K`
//! table a run writes: a level's propensity, a cell embedding), drawn in
//! place of the tissue, by legume-numeric's coarsened layouts: the cells'
//! fuzzy kNN graph coarsened into pseudobulks, a layout of the pseudobulk
//! graph, and every cell placed from the pseudobulks near it in that graph
//! ([`Coarsened::place_cells`]). Every layout runs on at most
//! [`PSEUDOBULKS`] points, however many cells there are. The settings
//! are `senna layout`'s, so pinto and senna lay out a table alike.

use crate::util::common::Mat;
use legume_numeric::matrix::dense_mat_io::l2_normalize_rows_inplace;
use legume_numeric::matrix::layout::{
    coarsened, diffusion_map_2d_on_graph, forceatlas2_layout_2d_on_graph, lle_layout_2d_on_graph,
    phate_layout_2d_on_graph, radial_tree_layout_2d_on_graph, stress_layout_2d_on_graph,
    sugiyama_layout_2d_on_graph, tsne_layout_2d_on_graph, umap_layout_2d_on_graph, CoarsenArgs,
    Coarsened, PhateArgs, TsneArgs,
};
use legume_numeric::matrix::umap::Umap;

/// A layout of the cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, clap::ValueEnum)]
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
    /// Layered drawing by breadth-first level from a root.
    Sugiyama,
    /// Radial drawing of the maximum spanning tree.
    Radial,
}

impl Method {
    /// Every layout, in the order the view cycles through them.
    pub const ALL: [Method; 10] = [
        Method::Umap,
        Method::Tumap,
        Method::Tsne,
        Method::Phate,
        Method::Diffusion,
        Method::ForceAtlas2,
        Method::Lle,
        Method::Stress,
        Method::Sugiyama,
        Method::Radial,
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
            Method::Sugiyama => "Sugiyama",
            Method::Radial => "radial tree",
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
const FORCEATLAS2_ITERATIONS: usize = 500;

/// PHATE's diffusion time, as `senna layout phate` sets it.
const PHATE_T: usize = 20;

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
                weights: Some(pbs.pb_size.clone()),
                seed,
                ..TsneArgs::default()
            },
        )?,
        Method::Phate => phate_layout_2d_on_graph(
            n,
            edges,
            &PhateArgs {
                t: PHATE_T,
                ..PhateArgs::default()
            },
        ),
        Method::Diffusion => diffusion_map_2d_on_graph(n, edges, 1),
        Method::ForceAtlas2 => {
            forceatlas2_layout_2d_on_graph(n, edges, FORCEATLAS2_ITERATIONS, seed)
        }
        Method::Lle => lle_layout_2d_on_graph(&pbs.pb_features.transpose(), edges),
        Method::Stress => stress_layout_2d_on_graph(n, edges, seed),
        Method::Sugiyama => sugiyama_layout_2d_on_graph(n, edges, None),
        Method::Radial => radial_tree_layout_2d_on_graph(n, edges, None),
    })
}

/// Any table of rows to lay out (`N × K` parquet, its first column the rows'
/// names), read as a layout reads its kind: community columns (`C0`, `C1`,
/// …) as a propensity, by their square roots (Hellinger); otherwise every
/// column as an embedding, each row by its direction (cosine). The rows can
/// be cells or cell pairs (a pair latent).
pub fn read_any_table(path: &std::path::Path) -> anyhow::Result<(Vec<Box<str>>, Mat)> {
    use crate::util::parquet_io::{parse_community_col_name, read_labelled_matrix};
    let m = read_labelled_matrix(path)?;
    let communities: Vec<usize> = m
        .cols
        .iter()
        .enumerate()
        .filter(|(_, c)| parse_community_col_name(c).is_some())
        .map(|(j, _)| j)
        .collect();
    if communities.is_empty() {
        let mut mat = m.mat;
        l2_normalize_rows_inplace(&mut mat);
        Ok((m.rows, mat))
    } else {
        let mut mat = m.mat.select_columns(&communities);
        mat.apply(|v| *v = v.max(0.0).sqrt());
        Ok((m.rows, mat))
    }
}

/// `pinto view --layout`: lay out `table` by `method` and write each row's
/// `x`, `y` to `out` (parquet, the rows keyed as the table's first column).
pub fn write_layout(table: &std::path::Path, method: Method, out: &str) -> anyhow::Result<()> {
    use legume_numeric::matrix::parquet::{write_named_table, Column};
    let key = crate::util::parquet_io::field_names(table)?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("{table:?}: no columns"))?;
    let (names, mat) = read_any_table(table)?;
    let seed = crate::util::common::DEFAULT_SEED;
    log::info!(
        "{} layout of {} rows of {table:?}",
        method.name(),
        names.len()
    );
    let xy = Coarse::new(names.clone(), &mat, seed)?.lay_out(method, seed)?;
    let x: Vec<f32> = xy.column(0).iter().copied().collect();
    let y: Vec<f32> = xy.column(1).iter().copied().collect();
    let columns = [("x".into(), Column::F32(&x)), ("y".into(), Column::F32(&y))];
    write_named_table(out, &key, &names, &columns)
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
    /// Its cell names and rows (`N × K`), as a layout reads them, as
    /// `senna layout` reads a table of their kind: a propensity, a mixture
    /// per cell, by its square roots (the Hellinger distance between
    /// cells); an embedding by each cell's direction (cosine).
    /// `coord_names`, the columns a propensity table may carry after its
    /// own, are left out.
    pub fn read(&self, coord_names: &[Box<str>]) -> anyhow::Result<(Vec<Box<str>>, Mat)> {
        use crate::util::parquet_io::{read_labelled_matrix, read_propensity};
        if self.propensity {
            let exclude = coord_names.iter().cloned().collect();
            let (mut prop, _, _, cells) = read_propensity(&self.path, &exclude)?;
            prop.apply(|v| *v = v.max(0.0).sqrt());
            Ok((cells, prop))
        } else {
            let mut m = read_labelled_matrix(&self.path)?;
            l2_normalize_rows_inplace(&mut m.mat);
            Ok((m.rows, m.mat))
        }
    }
}

#[cfg(test)]
#[path = "tests/layout.rs"]
mod tests;
