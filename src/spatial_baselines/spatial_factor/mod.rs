//! One sparse spatial factor core under two baselines, [`super::spatialpca`]
//! and [`super::mefisto`]: both are the latent factor model
//!
//! `Y (genes × cells) = W Z + E`, each factor `z_k ~ N(0, v_k R(rho_k))`,
//!
//! with a spatial prior on each factor's values over the cells. The papers
//! put a dense Gaussian-kernel (SpatialPCA) or squared-exponential
//! Gaussian-process (MEFISTO) prior on them; both commands replace it, with
//! no mode that keeps it, by the sparse precision of [`prior`]: `R(rho) = (I + rho L)^{-1} / cbar(rho)` on the per-section kNN
//! graph's Laplacian, so no `N × N` matrix and no `O(N^3)` step is met.
//!
//! The core holds what both share: the implicit expression ([`prep`]), the
//! graph prior, its spectrum and solves ([`prior`]), the numerical kit
//! ([`kit`]), the gene selection, the super-cells of the `-pb` commands and
//! the step from factors to domains. Each method keeps its own estimator.

pub mod bandwidth;
pub mod kit;
pub mod prep;
pub mod prior;
pub mod sct;
pub mod sparkx;

use super::cluster::{cluster_embedding, refine_by_neighbours};
use super::graph::{knn_within_sections, Neighbours};
use super::input::{SpatialData, SpatialInputArgs};
use super::pseudobulk::{fold_weighted, super_cells, SuperCells, COARSEN_K};
use super::stagate::seurat_v3_hvg;
use crate::util::common::Mat;
use kit::Dense;
use prior::{Laplacian, Spectrum, SpectrumOpts};

/// Spatial neighbours per cell of the sparse prior's graph: pinto's
/// shared spatial default.
pub const KNN: usize = crate::util::input::DEFAULT_KNN;
pub const SLQ_PROBES: usize = 32;
pub const SLQ_STEPS: usize = 60;
pub const EXACT_EIG_MAX: usize = 2000;
pub const RESOLUTION: f64 = 1.0;

/// The flags both factor baselines share.
#[derive(clap::Args, Debug, Clone)]
pub struct CoreArgs {
    #[command(flatten)]
    pub input: SpatialInputArgs,

    /// Number of domains: the factors clustered by pinto's seeded k-means
    /// into exactly this many. Not given, by Leiden on the 15-NN graph of
    /// the factors, which picks its own number
    #[arg(long)]
    pub n_clusters: Option<usize>,

    /// Leiden's modularity resolution without --n-clusters
    #[arg(long, default_value_t = RESOLUTION, hide = true)]
    pub resolution: f64,

    /// Spatial neighbours per cell of the sparse prior's graph, within a
    /// section, symmetrised
    #[arg(long, default_value_t = KNN, hide = true)]
    pub knn: usize,

    /// Stochastic Lanczos quadrature: probes per section
    #[arg(long, default_value_t = SLQ_PROBES, hide = true)]
    pub slq_probes: usize,

    /// Stochastic Lanczos quadrature: Lanczos steps per probe
    #[arg(long, default_value_t = SLQ_STEPS, hide = true)]
    pub slq_steps: usize,

    /// Sections of at most this many cells take their Laplacian's exact
    /// eigenvalues instead of quadrature
    #[arg(long, default_value_t = EXACT_EIG_MAX, hide = true)]
    pub exact_eig_max: usize,
}

impl CoreArgs {
    pub fn check(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.knn >= 1, "--knn must be at least 1");
        anyhow::ensure!(
            self.n_clusters.is_none_or(|k| k >= 1),
            "--n-clusters must be at least 1"
        );
        anyhow::ensure!(
            self.slq_probes >= 1 && self.slq_steps >= 1,
            "SLQ sizes must be positive"
        );
        Ok(())
    }

    pub fn spectrum_opts(&self) -> SpectrumOpts {
        SpectrumOpts {
            exact_eig_max: self.exact_eig_max,
            probes: self.slq_probes,
            steps: self.slq_steps,
            seed: self.input.seed,
        }
    }
}

/// The sparse prior's graph on cells: each cell's `knn` nearest within its
/// section, symmetrised, every edge of weight one; the neighbour lists and
/// the Laplacian.
pub fn cell_graph(data: &SpatialData, knn: usize) -> (Neighbours, Laplacian) {
    let nb = knn_within_sections(data, knn).symmetrised();
    let lap = Laplacian::from_neighbours(&nb, &data.section, data.n_sections);
    (nb, lap)
}

/// The prior's spectrum on `lap`.
pub fn spectrum(lap: &Laplacian, core: &CoreArgs) -> Spectrum {
    let s = Spectrum::new(lap, &core.spectrum_opts());
    let exact = s.exact.iter().filter(|&&e| e).count();
    log::info!(
        "graph prior: {} nodes, {} edges, mean degree {:.2}; {exact} of {} sections exact, \
         the rest by quadrature",
        lap.n(),
        lap.nnz() / 2,
        lap.mean_degree(),
        s.exact.len()
    );
    s
}

/// The `n_genes` most variable genes (seurat_v3 flavour on the counts as
/// read, sections pooled), in gene order.
pub fn select_genes(data: &SpatialData, n_genes: usize) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(n_genes >= 1, "--n-genes must be at least 1");
    let (genes, _) = seurat_v3_hvg(&data.counts, n_genes);
    anyhow::ensure!(!genes.is_empty(), "no genes to fit on");
    Ok(genes)
}

/// The factors' domains: clustered ([`cluster_embedding`]), then, with
/// `refine` neighbour lists, SpaGCN's majority refinement
/// ([`refine_by_neighbours`]). The labels and the clustering's name.
pub fn domains(
    z: &Mat,
    core: &CoreArgs,
    refine: Option<&Neighbours>,
) -> anyhow::Result<(Vec<usize>, &'static str)> {
    let (labels, how) = cluster_embedding(z, core.n_clusters, core.resolution, core.input.seed)?;
    Ok(match refine {
        Some(nb) => (refine_by_neighbours(&labels, nb), how),
        None => (labels, how),
    })
}

/// `z` (f64, cells × k) as the f32 embedding the outputs write.
pub fn to_mat(z: &Dense) -> Mat {
    Mat::from_fn(z.nrows(), z.ncols(), |i, j| z[(i, j)] as f32)
}

/// What a `-pb` command fits on: the super-cells, as cage builds them, and
/// the cell graph `nb` (symmetric lists) collapsed onto them, each
/// super-edge weighted by the cell-cell edges it collapses.
pub struct Coarse {
    pub cells: SuperCells,
    pub lap: Laplacian,
    pub n_super_edges: usize,
}

/// Coarsen `data` (read with its projection) to about `target`
/// super-cells along each cell's [`COARSEN_K`] nearest, and fold the
/// symmetric cell graph `nb` onto them.
pub fn coarsen(data: &mut SpatialData, target: usize, nb: &Neighbours) -> anyhow::Result<Coarse> {
    let proj = data
        .projection
        .take()
        .ok_or_else(|| anyhow::anyhow!("a -pb command reads the cells' projection"))?;
    let sc = super_cells(data, proj, target, knn_within_sections(data, COARSEN_K))?;
    let (edges, _) = nb.undirected();
    let ones = vec![1.0; edges.len()];
    let (super_edges, w) = fold_weighted(&sc.of_cell, &edges, &ones);
    let lap = Laplacian::from_edges(
        sc.data.cells.len(),
        &super_edges,
        &w,
        &sc.data.section,
        sc.data.n_sections,
    );
    Ok(Coarse {
        n_super_edges: super_edges.len(),
        cells: sc,
        lap,
    })
}

/// The `PseudobulkInput` a `-pb` run records.
pub fn pseudobulk_input(target: usize, coarse: &Coarse) -> crate::util::metadata::PseudobulkInput {
    crate::util::metadata::PseudobulkInput {
        n_pseudobulk: target,
        n_super_cells: coarse.cells.data.cells.len(),
        n_super_edges: coarse.n_super_edges,
        edge_weighting: "number of cell-cell edges collapsed".into(),
    }
}
