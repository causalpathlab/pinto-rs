//! MEFISTO (Velten, Braunger, Argelaguet, Arnol, Wirbel, Bredikhin, Zeller
//! and Stegle 2022, "Identifying temporal and spatial patterns of variation
//! from multimodal data using MEFISTO", Nature Methods 19:179,
//! doi:10.1038/s41592-021-01343-9): multi-group Bayesian factor analysis
//! (MOFA+; Argelaguet et al. 2020, Genome Biology 21:111,
//! doi:10.1186/s13059-020-02015-1) whose factors are spatially smooth.
//!
//! Written from the paper and its Supplementary Information (SI) alone; the
//! reference implementation is copyleft and none of its code is read or
//! translated. One view (RNA), the sections as groups, the coordinates as
//! the covariate.
//!
//! # Steps
//!
//! 1. Genes: the `--n-genes` most variable (2000, the paper's
//!    `FindVariableFeatures`, the seurat_v3 flavour on counts), sections
//!    pooled.
//! 2. `LogNormalize` (`ln(1 + 1e4 x / lib)`, `lib` over all genes), each
//!    gene centred within each section (SI section 2.1), held implicitly.
//! 3. Variational Bayes ([`vb`]): spike-and-slab weights with ARD,
//!    per-gene per-section noise, each factor's posterior multivariate over
//!    the cells, coordinate ascent (SI (28)-(68)); the factors' prior
//!    hyperparameters re-fitted from iteration `--gp-start` every
//!    `--gp-every` iterations (SI (31), (39)-(40)); until the ELBO's
//!    relative change is below `--tol` at an iteration whose
//!    hyperparameter step moved none by more than `--hyper-tol` (log
//!    scale), or `--max-iter` iterations.
//! 4. The factors' prior ([`sparse`]), in place of the paper's Gaussian
//!    processes (a squared-exponential kernel times a group kernel, SI
//!    (8)-(13)), which with their inducing-point approximation cost too
//!    much time and memory at real cell counts; pinto offers the model
//!    with a sparse graph prior instead, and `mefisto-pb` for scale. The
//!    precision `c_k (I + rho_k L)` on the
//!    per-section kNN graph, one strength `rho_k` per factor by a log grid
//!    then Brent; traces and log-determinants by the graph's spectrum,
//!    means by conjugate gradients.
//! 5. Domains, a step the paper does not take: `E[Z]`, each factor scaled
//!    by its weights' norm `||E[w_k]||` ([`domain_embedding`]), clustered
//!    by pinto's seeded k-means (`--n-clusters`) or Leiden.
//!
//! # Deviations
//!
//! - **Batch effects** (δ): with two or more batches the counts are divided
//!   by the input's own δ before the libraries, the genes and the
//!   normalization; `--ignore-batch` runs the paper's model (the sections
//!   still its groups).
//! - **The sparse prior** (cost, above): the graph prior in place of the kernel (one
//!   `rho_k` for the lengthscale and smooth share; the group kernel the
//!   identity, sections independent a priori), each `rho_k` maximizing
//!   the ELBO jointly with `q(z_k)` (the factor's collapsed bound) where
//!   the paper's step holds `q` fixed, spectral sums by
//!   stochastic Lanczos quadrature on sections of more than 2000 cells,
//!   solves to tolerance; the smoothness reported as the prior
//!   correlation of graph neighbours, not `1 - zeta_k`.
//! - **Exact reformulations**: implicit centring; the weight updates
//!   written for per-group noise; only per-group sums of the factors'
//!   variances kept.
//! - **Domains**: an added step (no GPL package involved).
//! - **Not ported** (unused by the paper's spatial analysis): covariate
//!   alignment, non-Gaussian likelihoods, the Kronecker path, interpolation.
//!
//! The paper is silent on the start, the order of updates within an
//! iteration and the convergence rule (pinto's also asks the
//! hyperparameters to have settled); pinto's choices are documented where
//! they are made ([`vb::State::init`], [`vb::fit`]).
//!
//! `mefisto-pb` fits the sparse model on cage's super-cells and the
//! collapsed graph, then unravels to cells: the weights, ARD and sparsity
//! held fixed; the noise, the factors and each `rho_k` re-fitted on the
//! cells and their graph, from the super-cells' factors broadcast.

pub mod sparse;
pub mod vb;

use super::input::{InputShape, SpatialData};
use super::memory::{factor_peak_bytes, FactorShape, FACTOR_PROCESS_BYTES};
use super::output::{write_loadings, write_run, Fitted, PbFit};
use super::spatial_factor::kit::Dense;
use super::spatial_factor::prep::{library, select_rows, SparseOp};
pub use super::spatial_factor::Variant;
use super::spatial_factor::{
    cell_graph, check_stored, coarsen, domains, merge, names_of, pseudobulk_input, read_shape,
    refined_domains, select_genes, spectrum, to_mat, CoreArgs,
};
use crate::util::memory::{ensure_fits, Budget};
use sparse::GraphPrior;
use vb::{FactorPrior, State, Trace, VbData, VbOpts};

pub const N_FACTORS: usize = 4;
pub const N_GENES: usize = 2000;
pub const CG_TOL: f64 = 1e-6;
pub const TOL: f64 = 5e-6;
pub const HYPER_TOL: f64 = 1e-3;
pub const MAX_ITER: usize = 1000;
pub const GP_START: usize = 20;
pub const GP_EVERY: usize = 10;
pub const N_GRID: usize = 20;
pub const UNRAVEL_MAX_ITER: usize = 200;

/// The model's arguments, shared by `mefisto` and `mefisto-pb`.
#[derive(clap::Args, Debug, Clone)]
pub struct FitArgs {
    #[command(flatten)]
    pub core: CoreArgs,

    /// Number of factors
    #[arg(long, default_value_t = N_FACTORS)]
    pub n_factors: usize,

    /// Genes modelled: the most variable (seurat_v3 flavour), at most this
    /// many
    #[arg(long, default_value_t = N_GENES)]
    pub n_genes: usize,

    /// Neighbours of a majority refinement of the domains (0: none, as the
    /// paper has no clustering step)
    #[arg(long, default_value_t = 0, hide = true)]
    pub refine_k: usize,

    /// Conjugate gradients' relative residual
    #[arg(long, default_value_t = CG_TOL, hide = true)]
    pub cg_tol: f64,

    /// Relative ELBO change at convergence
    #[arg(long, default_value_t = TOL, hide = true)]
    pub tol: f64,

    /// Largest change of a hyperparameter (log scale) over a step at
    /// convergence
    #[arg(long, default_value_t = HYPER_TOL, hide = true)]
    pub hyper_tol: f64,

    /// Iterations at most
    #[arg(long, default_value_t = MAX_ITER, hide = true)]
    pub max_iter: usize,

    /// First iteration of the prior's hyperparameter step
    #[arg(long, default_value_t = GP_START, hide = true)]
    pub gp_start: usize,

    /// Iterations between the prior's hyperparameter steps
    #[arg(long, default_value_t = GP_EVERY, hide = true)]
    pub gp_every: usize,

    /// Points of the hyperparameter grid (lengthscales, or rho)
    #[arg(long, default_value_t = N_GRID, hide = true)]
    pub n_grid: usize,
}

/// `mefisto`'s arguments.
#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub fit: FitArgs,
}

/// `mefisto-pb`'s arguments: the sparse model's, and the super-cells.
#[derive(clap::Args, Debug, Clone)]
pub struct PbArgs {
    #[command(flatten)]
    pub fit: FitArgs,

    /// Target number of super-cells to fit on (as cage's -d: the cells are
    /// coarsened until about this many remain)
    #[arg(
        long,
        short = 'd',
        default_value_t = 1024,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub n_pseudobulk: usize,

    /// Iterations of the unravelling at most
    #[arg(long, default_value_t = UNRAVEL_MAX_ITER, hide = true)]
    pub unravel_max_iter: usize,
}

impl FitArgs {
    fn check(&self) -> anyhow::Result<()> {
        self.core.check()?;
        anyhow::ensure!(self.n_factors >= 1, "--n-factors must be at least 1");
        anyhow::ensure!(self.n_genes >= 2, "--n-genes must be at least 2");
        anyhow::ensure!(
            self.cg_tol > 0.0 && self.tol > 0.0 && self.hyper_tol > 0.0,
            "tolerances must be positive"
        );
        anyhow::ensure!(self.gp_every >= 1, "--gp-every must be at least 1");
        Ok(())
    }

    fn opts(&self) -> VbOpts {
        VbOpts {
            tol: self.tol,
            max_iter: self.max_iter,
            gp_start: self.gp_start,
            gp_every: self.gp_every,
            frozen_weights: false,
            hyper_tol: self.hyper_tol,
        }
    }

    pub(crate) fn memory_shape(
        &self,
        input: &InputShape,
        m: usize,
        variant: Variant,
    ) -> FactorShape {
        let k = self.n_factors;
        let g = input.section_sizes.len().max(1);
        FactorShape {
            input: input.clone(),
            genes: m,
            knn: self.core.knn,
            exact_eig_max: self.core.exact_eig_max,
            cell_vectors: 2 * k * g + 2 * k + 8,
            gene_vectors: 8 * k + 2 * k * g + 4 * g,
            super_cells: variant.super_cells(),
            proj_dim: self.core.input.collapse.proj_dim,
            dense_bytes: 0,
            nonzero_bytes: super::memory::FACTOR_NONZERO_BYTES,
            freed_before_graph: 0,
        }
    }
}

const LOWER_MEMORY: &str = "lower --n-genes or --n-factors";

fn check_before_read(fit: &FitArgs, variant: Variant) -> anyhow::Result<Budget> {
    fit.check()?;
    check_stored("mefisto", &fit.core.input, LOWER_MEMORY, |s| {
        fit.memory_shape(s, s.genes.min(fit.n_genes), variant)
    })
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    let budget = check_before_read(&args.fit, Variant::Sparse)?;
    run_on(
        args,
        args.fit
            .core
            .input
            .read_within(false, &budget, FACTOR_PROCESS_BYTES)?,
        &budget,
    )
}

pub fn run_pb(args: &PbArgs) -> anyhow::Result<()> {
    let budget = check_before_read(&args.fit, Variant::Pseudobulk(args.n_pseudobulk))?;
    run_pb_on(
        args,
        args.fit
            .core
            .input
            .read_within(true, &budget, FACTOR_PROCESS_BYTES)?,
        &budget,
    )
}

/// The log expression of `genes`, centred per section, the counts freed.
fn expression(data: &mut SpatialData, genes: &[usize]) -> SparseOp {
    let lib = library(&data.counts);
    let sel = select_rows(&data.counts, genes);
    data.counts = nalgebra_sparse::CscMatrix::zeros(0, 0);
    SparseOp::log_centred(&sel, &lib, &data.section, data.n_sections)
}

/// Each factor's share of the centred expression's sum of squares that its
/// own term `w_k z_k^T` explains.
pub fn variance_explained(data: &VbData<'_>, st: &State) -> Vec<f64> {
    let total: f64 = data.yy.sum();
    let wm = st.w_mean();
    let yw = data.op.apply_t(&wm); // cells × K: Y^T E[w]
    (0..st.k())
        .map(|k| {
            let z = st.mu.column(k);
            let w2: f64 = wm.column(k).norm_squared();
            let cross = 2.0 * z.dot(&yw.column(k));
            let own = w2 * z.norm_squared();
            (cross - own) / total.max(1e-300)
        })
        .collect()
}

/// What pinto's domain step clusters (the paper has none): each factor's
/// `E[z_k]` scaled by the norm of its weights `||E[w_k]||`, so a factor
/// weighs in by the expression it carries, as SpatialPCA's PCs carry their
/// eigenvalues' scale. MEFISTO's factors have unit prior variance whatever
/// they explain, so unscaled a factor with almost no weight (one left
/// unstructured, or pruned by ARD) counts as much as the main one.
pub fn domain_embedding(mu: &Dense, w: &Dense) -> Dense {
    let mut z = mu.clone();
    for k in 0..z.ncols() {
        let norm = w.column(k).norm();
        z.column_mut(k).scale_mut(norm);
    }
    z
}

/// The domains of the fit `st`, refined over each cell's `refine_k`
/// nearest within its section when set.
fn cell_domains(
    fit: &FitArgs,
    data: &SpatialData,
    st: &State,
) -> anyhow::Result<(Vec<usize>, &'static str)> {
    let z = domain_embedding(&st.mu, &st.w_mean());
    refined_domains(&to_mat(&z), &fit.core, data, fit.refine_k)
}

fn weights_out(
    fit: &FitArgs,
    data: &SpatialData,
    genes: &[usize],
    st: &State,
) -> anyhow::Result<()> {
    write_loadings(
        &fit.core.input.out,
        "mefisto_weights",
        &names_of(&data.genes, genes),
        &to_mat(&st.w_mean()),
        "mefisto",
    )
}

/// [`run`] on `data` as read.
pub(crate) fn run_on(args: &Args, mut data: SpatialData, budget: &Budget) -> anyhow::Result<()> {
    let fit = &args.fit;
    let genes = select_genes(&data, fit.n_genes)?;
    ensure_fits(
        "mefisto",
        factor_peak_bytes(&fit.memory_shape(&read_shape(&data), genes.len(), Variant::Sparse)),
        budget,
        LOWER_MEMORY,
    )?;
    let op = expression(&mut data, &genes);
    let vd = VbData::new(&op, &data.section, data.n_sections);
    let mut st = State::init(&vd, fit.n_factors, fit.core.input.seed);
    let (trace, extra) = {
        let (_, lap) = cell_graph(&data, fit.core.knn);
        let spec = spectrum(&lap, &fit.core);
        let mut prior = GraphPrior::new(
            &lap,
            &spec,
            st.k(),
            GraphPrior::start_rho(&lap),
            fit.cg_tol,
            fit.n_grid,
        );
        let trace = vb::fit(&vd, &mut st, &mut prior, &fit.opts());
        let smooth: Vec<f64> = (0..st.k())
            .map(|k| prior.neighbour_correlation(k, fit.core.slq_probes, fit.core.input.seed))
            .collect();
        let mut extra = serde_json::json!({
            "prior": "sparse",
            "graph": format!("kNN within a section, k = {}, symmetrized", fit.core.knn),
            "neighbour_correlation": smooth,
            "slq_probes": fit.core.slq_probes,
            "slq_steps": fit.core.slq_steps,
            "exact_eig_max": fit.core.exact_eig_max,
            "cg_tol": fit.cg_tol,
        });
        merge(&mut extra, state_settings(&vd, &st, &prior));
        (trace, extra)
    };
    log::info!(
        "mefisto: {} iterations, ELBO {:.6e}{}",
        trace.iterations,
        trace.elbo.last().copied().unwrap_or(f64::NAN),
        if trace.converged {
            ""
        } else {
            " (not converged)"
        }
    );
    let (labels, clustering) = cell_domains(fit, &data, &st)?;
    let mut s = base_settings(fit, &genes, &data.genes, clustering, &trace);
    merge(&mut s, extra);
    weights_out(fit, &data, &genes, &st)?;
    write_run(
        &fit.core.input,
        &data,
        Fitted {
            method: "mefisto",
            labels: &labels,
            membership: None,
            embedding: &to_mat(&st.mu),
            embedding_name: "mefisto",
            settings: s,
            input_layout: None,
            pseudobulk: None,
        },
    )
}

/// What `baseline.json` records of a fit on `genes` (of `names`).
fn base_settings(
    fit: &FitArgs,
    genes: &[usize],
    names: &[Box<str>],
    clustering: &str,
    trace: &Trace,
) -> serde_json::Value {
    serde_json::json!({
        "paper": "Velten et al. 2022, doi:10.1038/s41592-021-01343-9",
        "n_factors": fit.n_factors,
        "n_genes": fit.n_genes,
        "genes_used": names_of(names, genes),
        "gene_selection": "seurat_v3 highly variable genes",
        "normalization": "LogNormalize (1e4), centred per section",
        "elbo_trace": trace.elbo,
        "iterations": trace.iterations,
        "converged": trace.converged,
        "tol": fit.tol,
        "max_iter": fit.max_iter,
        "gp_start": fit.gp_start,
        "gp_every": fit.gp_every,
        "n_grid": fit.n_grid,
        "clustering": clustering,
        "clustered": "E[z], each factor scaled by ||E[w_k]||",
        "n_clusters": fit.core.n_clusters,
        "resolution": fit.core.n_clusters.is_none().then_some(fit.core.resolution),
        "refine_k": fit.refine_k,
    })
}

/// What the fit's state adds: variance explained and the prior's state.
fn state_settings(vd: &VbData<'_>, st: &State, prior: &dyn FactorPrior) -> serde_json::Value {
    serde_json::json!({
        "variance_explained": variance_explained(vd, st),
        "prior_state": prior.report(),
    })
}

/// `mefisto-pb` on `data` as read with its projection.
pub(crate) fn run_pb_on(
    args: &PbArgs,
    mut data: SpatialData,
    budget: &Budget,
) -> anyhow::Result<()> {
    let fit = &args.fit;
    let genes = select_genes(&data, fit.n_genes)?;
    ensure_fits(
        "mefisto-pb",
        factor_peak_bytes(&fit.memory_shape(
            &read_shape(&data),
            genes.len(),
            Variant::Pseudobulk(args.n_pseudobulk),
        )),
        budget,
        LOWER_MEMORY,
    )?;
    let (nb, lap) = cell_graph(&data, fit.core.knn);
    let mut coarse = coarsen(&mut data, args.n_pseudobulk, &nb)?;
    drop(nb);
    let pb_op = expression(&mut coarse.cells.data, &genes);
    let op = expression(&mut data, &genes);
    let seed = fit.core.input.seed;

    let pb_vd = VbData::new(&pb_op, &coarse.cells.data.section, data.n_sections);
    let pb_spec = spectrum(&coarse.lap, &fit.core);
    let mut pb_prior = GraphPrior::new(
        &coarse.lap,
        &pb_spec,
        fit.n_factors,
        GraphPrior::start_rho(&coarse.lap),
        fit.cg_tol,
        fit.n_grid,
    );
    let mut pb_st = State::init(&pb_vd, fit.n_factors, seed);
    let pb_trace = vb::fit(&pb_vd, &mut pb_st, &mut pb_prior, &fit.opts());
    let (pb_labels, _) = domains(
        &to_mat(&domain_embedding(&pb_st.mu, &pb_st.w_mean())),
        &fit.core,
        None,
    )?;

    let vd = VbData::new(&op, &data.section, data.n_sections);
    let spec = spectrum(&lap, &fit.core);
    let mut prior = GraphPrior::new(
        &lap,
        &spec,
        pb_st.k(),
        GraphPrior::start_rho(&lap),
        fit.cg_tol,
        fit.n_grid,
    );
    let mut st = unravel_start(&vd, &pb_st, &coarse.cells.of_cell);
    let opts = VbOpts {
        max_iter: args.unravel_max_iter,
        gp_start: 1,
        frozen_weights: true,
        ..fit.opts()
    };
    let trace = vb::fit(&vd, &mut st, &mut prior, &opts);
    log::info!(
        "mefisto-pb: super-cells {} iterations; cells unravelled in {}; rho {:?}",
        pb_trace.iterations,
        trace.iterations,
        prior.rho
    );
    let (labels, clustering) = cell_domains(fit, &data, &st)?;
    let of_cell = &coarse.cells.of_cell;
    let broadcast: Vec<usize> = of_cell.iter().map(|&s| pb_labels[s]).collect();
    let mut s = base_settings(fit, &genes, &data.genes, clustering, &trace);
    merge(&mut s, state_settings(&vd, &st, &prior));
    s["prior"] = "sparse".into();
    s["super_cell_fit"] = serde_json::json!({
        "iterations": pb_trace.iterations,
        "converged": pb_trace.converged,
        "elbo_final": pb_trace.elbo.last(),
        "prior_state": pb_prior.report(),
    });
    s["unravel"] = serde_json::json!({
        "method": "weights, ARD and sparsity fixed from the super-cell fit; noise, factors and rho refitted on the cells and their graph, from the super-cells' factors broadcast",
        "max_iter": args.unravel_max_iter,
        "broadcast": "the super-cells' own labels, clustered apart, in cell_pb.parquet (a diagnostic)",
    });
    weights_out(fit, &data, &genes, &pb_st)?;
    write_run(
        &fit.core.input,
        &data,
        Fitted {
            method: "mefisto-pb",
            labels: &labels,
            membership: None,
            embedding: &to_mat(&st.mu),
            embedding_name: "mefisto",
            settings: s,
            input_layout: None,
            pseudobulk: Some(PbFit {
                input: pseudobulk_input(args.n_pseudobulk, &coarse),
                super_cell: of_cell,
                broadcast: &broadcast,
            }),
        },
    )
}

/// The unravelling's start on the cells: the super-cells' weights, ARD and
/// sparsity, each cell's factors its super-cell's, no variance; the noise
/// is re-fitted first.
pub fn unravel_start(vd: &VbData<'_>, pb: &State, of_cell: &[usize]) -> State {
    let mu = Dense::from_fn(of_cell.len(), pb.k(), |c, k| pb.mu[(of_cell[c], k)]);
    let mut st = State::start(vd, mu);
    st.gamma = pb.gamma.clone();
    st.mu_w = pb.mu_w.clone();
    st.s2_w = pb.s2_w.clone();
    st.alpha = pb.alpha.clone();
    st.theta = pb.theta.clone();
    st
}
