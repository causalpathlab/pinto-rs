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
//!    relative change is below `--tol` or `--max-iter` iterations.
//! 4. The factors' prior:
//!    - `--reference` ([`reference`]): the paper's Gaussian process, a
//!      squared-exponential kernel times a group kernel (SI (8)-(13)), on
//!      the cells when there are at most `--n-inducing` (1000, the paper's
//!      spatial run), else on that many inducing cells (SI section 4); the
//!      lengthscale grid and the box-constrained search over the smooth
//!      share and the group kernel (SI section 3.2.2).
//!    - the default, sparse ([`sparse`]): the precision `c_k (I + rho_k L)`
//!      on the per-section kNN graph, one strength `rho_k` per factor by a
//!      log grid then Brent; traces and log-determinants by the graph's
//!      spectrum, means by conjugate gradients.
//! 5. Domains, a step the paper does not take: `E[Z]` clustered by pinto's
//!    seeded k-means (`--n-clusters`) or Leiden.
//!
//! # Deviations
//!
//! - **Batch effects** (δ): with two or more batches the counts are divided
//!   by the input's own δ before the libraries, the genes and the
//!   normalisation; `--ignore-batch` runs the paper's model (the sections
//!   still its groups).
//! - **Sparse mode only**: the graph prior in place of the kernel (one
//!   `rho_k` for the lengthscale and smooth share; the group kernel the
//!   identity, sections independent a priori), spectral sums by
//!   stochastic Lanczos quadrature on sections of more than 2000 cells,
//!   solves to tolerance; the smoothness reported as the prior
//!   correlation of graph neighbours, not `1 - zeta_k`.
//! - **Exact reformulations**: implicit centring; the weight updates
//!   written for per-group noise; only per-group sums of the factors'
//!   variances kept; `Sigma_ZU` formed in row blocks; the extreme pairwise
//!   distances by nearest neighbours and the convex hull.
//! - **Domains**: an added step (no GPL package involved).
//! - **Not ported** (unused by the paper's spatial analysis): covariate
//!   alignment, non-Gaussian likelihoods, the Kronecker path, interpolation.
//!
//! The paper is silent on the start, the order of updates within an
//! iteration, the convergence rule and the inducing cells' placement in
//! 2-D; pinto's choices are documented where they are made ([`vb::State::init`],
//! [`vb::fit`], [`reference::inducing_cells`]).
//!
//! `mefisto-pb` fits the sparse model on cage's super-cells and the
//! collapsed graph, then unravels to cells: the weights, ARD and sparsity
//! held fixed; the noise, the factors and each `rho_k` re-fitted on the
//! cells and their graph, from the super-cells' factors broadcast.

pub mod reference;
pub mod sparse;
pub mod vb;

use super::graph::knn_within_sections;
use super::input::{InputShape, SpatialData};
use super::memory::{check_stored_shape, factor_peak_bytes, FactorShape, FACTOR_PROCESS_BYTES};
use super::output::{write_loadings, write_run, Fitted, PbFit};
use super::spatial_factor::kit::Dense;
use super::spatial_factor::prep::{library, select_rows, ResidualOp};
use super::spatial_factor::{
    cell_graph, coarsen, domains, pseudobulk_input, select_genes, spectrum, to_mat, CoreArgs,
};
use crate::util::ablation::section_sizes;
use crate::util::memory::{ensure_fits, Budget};
use sparse::GraphPrior;
use vb::{FactorPrior, State, Trace, VbData, VbOpts};

pub const N_FACTORS: usize = 4;
pub const N_GENES: usize = 2000;
pub const CG_TOL: f64 = 1e-6;
pub const TOL: f64 = 5e-6;
pub const MAX_ITER: usize = 1000;
pub const GP_START: usize = 20;
pub const GP_EVERY: usize = 10;
pub const N_GRID: usize = 20;
pub const N_INDUCING: usize = 1000;
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

    /// Run the paper's own formulation instead of pinto's sparse one (see
    /// --help): each factor a Gaussian process, on inducing cells when
    /// there are many
    #[arg(long, default_value_t = false)]
    pub reference: bool,

    /// --reference: the Gaussian processes' inducing cells, at most
    #[arg(long, default_value_t = N_INDUCING, hide = true)]
    pub n_inducing: usize,

    /// --reference: rank of the group kernel
    #[arg(long, default_value_t = 1, hide = true)]
    pub group_rank: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Sparse,
    Reference,
    /// A `-pb` run, its target number of super-cells.
    Pseudobulk(usize),
}

impl FitArgs {
    fn check(&self) -> anyhow::Result<()> {
        self.core.check()?;
        anyhow::ensure!(self.n_factors >= 1, "--n-factors must be at least 1");
        anyhow::ensure!(self.n_genes >= 2, "--n-genes must be at least 2");
        anyhow::ensure!(
            self.cg_tol > 0.0 && self.tol > 0.0,
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
        }
    }

    pub(crate) fn memory_shape(
        &self,
        input: &InputShape,
        m: usize,
        variant: Variant,
        n_inducing: usize,
    ) -> FactorShape {
        let k = self.n_factors;
        let g = input.section_sizes.len().max(1);
        let n: usize = input.section_sizes.iter().sum();
        FactorShape {
            input: input.clone(),
            genes: m,
            knn: self.core.knn,
            exact_eig_max: self.core.exact_eig_max,
            cell_vectors: 2 * k * g + 2 * k + 8,
            gene_vectors: 8 * k + 2 * k * g + 4 * g,
            super_cells: match variant {
                Variant::Pseudobulk(target) => target,
                _ => 0,
            },
            reference: match variant {
                Variant::Reference => {
                    let mm = n.min(n_inducing);
                    // per factor q(u) (its covariance), and an update's or a
                    // hyperparameter step's few M × M matrices and row block
                    8 * (k * mm * mm + 8 * mm * mm + 3 * 4096.min(n) * mm)
                }
                _ => 0,
            },
        }
    }
}

const LOWER_MEMORY: &str = "lower --n-genes or --n-factors";

fn check_before_read(fit: &FitArgs, variant: Variant, n_inducing: usize) -> anyhow::Result<Budget> {
    fit.check()?;
    let budget = Budget::now()?;
    let stored = fit.core.input.stored()?;
    check_stored_shape(
        "mefisto",
        stored.as_ref().map(|s| {
            let peak = factor_peak_bytes(&fit.memory_shape(
                s,
                s.genes.min(fit.n_genes),
                variant,
                n_inducing,
            ));
            fit.core
                .input
                .peak_with_delta(peak, FACTOR_PROCESS_BYTES, s)
        }),
        &budget,
        LOWER_MEMORY,
    )?;
    Ok(budget)
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    let variant = if args.reference {
        Variant::Reference
    } else {
        Variant::Sparse
    };
    let budget = check_before_read(&args.fit, variant, args.n_inducing)?;
    run_on(args, args.fit.core.input.read(false)?, &budget)
}

pub fn run_pb(args: &PbArgs) -> anyhow::Result<()> {
    let budget = check_before_read(&args.fit, Variant::Pseudobulk(args.n_pseudobulk), 0)?;
    run_pb_on(args, args.fit.core.input.read(true)?, &budget)
}

fn read_shape(data: &SpatialData) -> InputShape {
    InputShape {
        genes: data.genes.len(),
        nonzeros: data.counts.nnz(),
        section_sizes: section_sizes(&data.section),
        batches: data.n_batches(),
    }
}

/// The log expression of `genes`, centred per section, the counts freed.
fn expression(data: &mut SpatialData, genes: &[usize]) -> ResidualOp {
    let lib = library(&data.counts);
    let sel = select_rows(&data.counts, genes);
    data.counts = nalgebra_sparse::CscMatrix::zeros(0, 0);
    ResidualOp::log_centred(&sel, &lib, &data.section, data.n_sections)
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

fn cell_domains(
    fit: &FitArgs,
    data: &SpatialData,
    z: &Dense,
) -> anyhow::Result<(Vec<usize>, &'static str)> {
    let refine = (fit.refine_k > 0).then(|| knn_within_sections(data, fit.refine_k));
    domains(&to_mat(z), &fit.core, refine.as_ref())
}

fn weights_out(
    fit: &FitArgs,
    data: &SpatialData,
    genes: &[usize],
    st: &State,
) -> anyhow::Result<()> {
    let names: Vec<Box<str>> = genes.iter().map(|&g| data.genes[g].clone()).collect();
    write_loadings(
        &fit.core.input.out,
        "mefisto_weights",
        &names,
        &to_mat(&st.w_mean()),
        "mefisto",
    )
}

/// [`run`] on `data` as read.
pub(crate) fn run_on(args: &Args, mut data: SpatialData, budget: &Budget) -> anyhow::Result<()> {
    let fit = &args.fit;
    let genes = select_genes(&data, fit.n_genes)?;
    let variant = if args.reference {
        Variant::Reference
    } else {
        Variant::Sparse
    };
    ensure_fits(
        "mefisto",
        factor_peak_bytes(&fit.memory_shape(
            &read_shape(&data),
            genes.len(),
            variant,
            args.n_inducing,
        )),
        budget,
        LOWER_MEMORY,
    )?;
    let op = expression(&mut data, &genes);
    let section = data.section.clone();
    let vd = VbData::new(&op, &section, data.n_sections);
    let mut st = State::init(&vd, fit.n_factors, fit.core.input.seed);
    let (trace, extra) = if args.reference {
        if data.n_sections > 1 {
            log::warn!(
                "mefisto --reference: {} sections share one coordinate frame in the paper's \
                 group kernel; pinto's sections have their own, so cross-section terms \
                 overlay them as they are",
                data.n_sections
            );
        }
        let xy: Vec<Vec<f64>> = (0..op.n)
            .map(|i| {
                (0..data.xy.ncols())
                    .map(|j| data.xy[(i, j)] as f64)
                    .collect()
            })
            .collect();
        let mut prior = reference::GpPrior::new(
            xy,
            section.clone(),
            data.n_sections,
            reference::GpSizes {
                k: st.k(),
                n_inducing: args.n_inducing,
                n_grid: fit.n_grid,
                rank: args.group_rank,
                seed: fit.core.input.seed,
            },
        );
        let trace = vb::fit(&vd, &mut st, &mut prior, &fit.opts());
        let extra = serde_json::json!({
            "prior": "reference",
            "kernel": "squared exponential times a group kernel (SI (8)-(13))",
            "n_inducing": prior.inducing.len(),
            "group_rank": args.group_rank,
        });
        let mut extra = extra;
        merge(&mut extra, state_settings(&vd, &st, &prior));
        (trace, extra)
    } else {
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
        let extra = serde_json::json!({
            "prior": "sparse",
            "graph": format!("kNN within a section, k = {}, symmetrised", fit.core.knn),
            "neighbour_correlation": smooth,
            "slq_probes": fit.core.slq_probes,
            "slq_steps": fit.core.slq_steps,
            "exact_eig_max": fit.core.exact_eig_max,
            "cg_tol": fit.cg_tol,
        });
        let mut extra = extra;
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
    let (labels, clustering) = cell_domains(fit, &data, &st.mu)?;
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
            reference: args.reference,
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
        "genes_used": genes.iter().map(|&g| names[g].as_ref()).collect::<Vec<_>>(),
        "gene_selection": "seurat_v3 highly variable genes",
        "normalisation": "LogNormalize (1e4), centred per section",
        "elbo_trace": trace.elbo,
        "iterations": trace.iterations,
        "converged": trace.converged,
        "tol": fit.tol,
        "max_iter": fit.max_iter,
        "gp_start": fit.gp_start,
        "gp_every": fit.gp_every,
        "n_grid": fit.n_grid,
        "clustering": clustering,
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

fn merge(into: &mut serde_json::Value, from: serde_json::Value) {
    if let (Some(a), serde_json::Value::Object(b)) = (into.as_object_mut(), from) {
        a.extend(b);
    }
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
            0,
        )),
        budget,
        LOWER_MEMORY,
    )?;
    let (nb, lap) = cell_graph(&data, fit.core.knn);
    let mut coarse = coarsen(&mut data, args.n_pseudobulk, &nb)?;
    drop(nb);
    let pb_section = coarse.cells.data.section.clone();
    let pb_op = expression(&mut coarse.cells.data, &genes);
    let op = expression(&mut data, &genes);
    let seed = fit.core.input.seed;

    let pb_vd = VbData::new(&pb_op, &pb_section, data.n_sections);
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
    let (pb_labels, _) = domains(&to_mat(&pb_st.mu), &fit.core, None)?;

    let section = data.section.clone();
    let vd = VbData::new(&op, &section, data.n_sections);
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
    let (labels, clustering) = cell_domains(fit, &data, &st.mu)?;
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
            reference: false,
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
