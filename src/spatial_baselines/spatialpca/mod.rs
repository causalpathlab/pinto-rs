//! SpatialPCA (Shang and Zhou 2022, "Spatially aware dimension reduction
//! for spatial transcriptomics", Nature Communications 13:7203,
//! doi:10.1038/s41467-022-34879-1): probabilistic PCA (Tipping and Bishop
//! 1999, JRSS B 61:611, doi:10.1111/1467-9868.00196) whose factors, the
//! spatial PCs, carry a spatial kernel prior (the generalised probabilistic
//! PCA of Gu and Shen 2020, JMLR 21:1), clustered into domains.
//!
//! Written from the paper's Methods and Supplementary Note 1 alone; the
//! reference implementation is GPL-licensed and none of its code is read or
//! translated. Equation labels: Eq. (k) of the Methods, SN(k) of
//! Supplementary Note 1.
//!
//! # The model and its fit
//!
//! `Y = (X B)^T + W Z + E` (Eq. 1) with the intercept `X = 1`, `W^T W =
//! I_d`, noise `sigma0^2`, and each spatial PC `Z_l ~ N(0, sigma0^2 tau K)`
//! (Eq. 2), `tau` shared by the PCs. `B` is integrated out; `sigma0^2`
//! (SN(6)) and `W` (SN(8), the top `d` eigenvectors of `G(tau)`) are
//! profiled; `tau` maximises the profile SN(7); the PCs are the posterior
//! means SN(10).
//!
//! - `--reference` ([`reference`]): the paper's Gaussian kernel `K(s, s') =
//!   exp(-||s - s'||^2 / gamma)`, one block per section (the paper's
//!   multi-sample extension), each section's coordinates standardised per
//!   axis, `gamma` per section by the paper's rule (Silverman above 5000
//!   cells, Sheather-Jones at or below, its pairwise sums on R's `bw.SJ`
//!   binning of 1000 bins, per gene on the standardised expression, the
//!   median over genes); the kernel's eigenpairs, the
//!   Woodbury form SN(11) on the rank-`r` kernel and SN(13) at the
//!   paper's ranks (the 90% rule at or below 5000 cells; 20
//!   for the `tau` search and `ceil(0.1 n)` for the PCs above);
//!   `tau` by Brent on `log tau` after a 41-point grid on `[1e-4, 1e4]`.
//! - The default, sparse ([`sparse`]): the kernel replaced by the sparse
//!   prior `R(rho) = (I + rho L)^{-1} / cbar(rho)` on the per-section kNN
//!   graph ([`super::spatial_factor::prior`]), `tau` and `rho` both by
//!   maximum likelihood on the same profile, which then has the exact
//!   simplified form of [`sparse`]; the counts stay sparse.
//!
//! Domains: the PCs are clustered (pinto's seeded k-means into
//! `--n-clusters`, Leiden without), then each cell takes the majority label
//! of itself and its 6 nearest cells in its section (the paper's
//! refinement after SpaGCN; Hu et al. 2021, Nature Methods 18:1342,
//! doi:10.1038/s41592-021-01255-8).
//!
//! # Deviations
//!
//! - **Batch effects** (δ): with two or more batches the counts are divided
//!   by the input's own δ before the library sizes, the genes and the
//!   normalisation, in place of the paper's multi-sample Seurat
//!   integration; `--ignore-batch` runs the paper's model on the pooled
//!   sections.
//! - **Clustering** (licence): the paper's Walktrap and Louvain (GPL
//!   packages) are replaced by k-means (`--n-clusters`) or Leiden.
//! - **Gene selection**: the paper's SPARK / SPARK-X spatially variable
//!   genes are not available here (SPARK is GPL; SPARK-X is not yet
//!   written from its paper), so the `--n-genes` most variable genes
//!   (seurat_v3 flavour) are modelled, one of the gene sets the paper
//!   itself reports runs on.
//! - **Normalisation**: SCTransform's VST is replaced, in both modes, by
//!   analytic Pearson residuals of a Poisson model (Lause, Berens and
//!   Kobak 2021, Genome Biology 22:258, doi:10.1186/s13059-021-02451-7),
//!   clipped at `+-sqrt(N)`, then standardised per gene as the paper does.
//! - **Sparse mode only**: the sparse prior and its `rho` by maximum
//!   likelihood; spectral sums by stochastic Lanczos quadrature on
//!   sections of more than 2000 cells; solves by conjugate gradients and
//!   `G`'s eigenpairs by subspace iteration, to tolerance; `(tau, rho)` by a
//!   projected L-BFGS from the best of a 3 × 3 grid.
//! - **Low rank** (`--reference`): `G`, `g` and `q` of the `tau` profile
//!   from SN(11)'s Woodbury identity on the rank-`r` kernel `U D U^T`, in
//!   place of SN(12) as printed, which keeps the kernel's discarded
//!   directions at full weight in `G` (the two agree at full rank).
//! - **Exact reformulations**: the scalar forms of SN(11) with `q = 1` and
//!   centred `Y`; the kernel applied in tiles; `R = I` with each loading's
//!   largest entry positive; the refinement over the `k` nearest cells
//!   (the lattice neighbours on a regular lattice).
//!
//! `spatialpca-pb` fits the sparse model on cage's super-cells and the
//! collapsed graph, then unravels to cells: `W` held fixed, `sigma0^2`,
//! `tau` and `rho` refitted on the cells and their graph, each cell's PCs
//! `C^{-1} Y^T W`.

pub mod reference;
pub mod sparse;

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
use sparse::Profile;

pub const N_FACTORS: usize = 20;
pub const N_GENES: usize = 3000;
pub const REFINE_K: usize = 6;
pub const CG_TOL: f64 = 1e-8;
pub const MAX_EVALS: usize = 50;
pub const MAX_KERNEL_EVALS: f64 = 1e14;

/// The model's arguments, shared by `spatialpca` and `spatialpca-pb`.
#[derive(clap::Args, Debug, Clone)]
pub struct FitArgs {
    #[command(flatten)]
    pub core: CoreArgs,

    /// Number of spatial PCs
    #[arg(long, visible_alias = "n-pcs", default_value_t = N_FACTORS)]
    pub n_factors: usize,

    /// Genes modelled: the most variable (seurat_v3 flavour), at most this
    /// many
    #[arg(long, default_value_t = N_GENES)]
    pub n_genes: usize,

    /// Neighbours of the majority refinement of the domains (0: none)
    #[arg(long, default_value_t = REFINE_K, hide = true)]
    pub refine_k: usize,

    /// Conjugate gradients' relative residual
    #[arg(long, default_value_t = CG_TOL, hide = true)]
    pub cg_tol: f64,

    /// Evaluations of the sparse profile's optimiser at most
    #[arg(long, default_value_t = MAX_EVALS, hide = true)]
    pub max_evals: usize,
}

/// `spatialpca`'s arguments.
#[derive(clap::Args, Debug, Clone)]
pub struct Args {
    #[command(flatten)]
    pub fit: FitArgs,

    /// Run the paper's own formulation instead of pinto's sparse one (see
    /// --help): the dense Gaussian kernel with its low-rank approximation
    #[arg(long, default_value_t = false)]
    pub reference: bool,

    /// --reference: the kernel bandwidth gamma, in place of the paper's rule
    #[arg(long, hide = true, requires = "reference")]
    pub bandwidth: Option<f64>,

    /// --reference: the kernel rank, in place of the paper's rule
    #[arg(long, hide = true, requires = "reference")]
    pub rank: Option<usize>,

    /// --reference: kernel evaluations a run may take at most
    #[arg(long, default_value_t = MAX_KERNEL_EVALS, hide = true)]
    pub max_kernel_evals: f64,
}

/// `spatialpca-pb`'s arguments: the sparse model's, and the super-cells.
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
}

/// Which run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Sparse,
    /// `--reference`, its `--rank` when set.
    Reference(Option<usize>),
    /// A `-pb` run, its target number of super-cells.
    Pseudobulk(usize),
}

impl FitArgs {
    fn check(&self) -> anyhow::Result<()> {
        self.core.check()?;
        anyhow::ensure!(self.n_factors >= 1, "--n-factors must be at least 1");
        anyhow::ensure!(self.n_genes >= 2, "--n-genes must be at least 2");
        anyhow::ensure!(self.cg_tol > 0.0, "--cg-tol must be positive");
        Ok(())
    }

    /// What a `variant` run on `input`, `m` genes modelled, holds.
    pub(crate) fn memory_shape(
        &self,
        input: &InputShape,
        m: usize,
        variant: Variant,
    ) -> FactorShape {
        let d = self.n_factors.min(m);
        FactorShape {
            input: input.clone(),
            genes: m,
            knn: self.core.knn,
            exact_eig_max: self.core.exact_eig_max,
            cell_vectors: 7 * (d + 10) + 2 * d,
            gene_vectors: 3 * (d + 10) + d,
            super_cells: match variant {
                Variant::Pseudobulk(target) => target,
                _ => 0,
            },
            reference: match variant {
                Variant::Reference(rank) => reference_bytes(input, m, d, rank),
                _ => 0,
            },
        }
    }
}

/// What `--reference` holds beyond the sparse path's: `Y` dense (m × n),
/// `G`, its rank-one term, its eigenvectors and their workspace (m × m), the kernel's
/// eigenvectors over all cells (`n × sum_s min(want, n_s)`) with one
/// block's own while it is taken (its dense kernel and eigenvectors, three
/// copies as sorted, at or below the paper's 5000 cells), `Y U` and its
/// scaled copy (m × r_est), and the PCs' `K M U` (n × r_Z) and products
/// (n × d).
fn reference_bytes(input: &InputShape, m: usize, d: usize, rank: Option<usize>) -> usize {
    let sizes = &input.section_sizes;
    let n: usize = sizes.iter().sum();
    let want = reference::kernel_want(n, rank);
    let width: usize = sizes.iter().map(|&s| want.min(s)).sum();
    let largest = sizes.iter().copied().max().unwrap_or(0);
    let block = if largest <= reference::DENSE_KERNEL_MAX {
        3 * largest * largest
    } else {
        // subspace iteration: a few blocks of want + 10 vectors
        4 * largest * (want + 10)
    };
    // the ranks at most: all of the kernel's at or below 5000 cells
    let (r_est, r_z) = if n > reference::DENSE_KERNEL_MAX {
        (
            rank.unwrap_or(20).min(width),
            rank.unwrap_or(n.div_ceil(10)).min(width),
        )
    } else {
        (width, width)
    };
    8 * (m * n + 4 * m * m + block + n * width + 2 * m * r_est + n * r_z + 4 * n * d)
}

/// Kernel evaluations of a `--reference` run, from its section sizes:
/// each block's eigenpairs (dense at or below the paper's 5000 cells,
/// else about 30 rounds of a block `r_Z + 10` wide) and the PCs' products.
pub fn kernel_evals(sizes: &[usize], d: usize) -> f64 {
    let n: usize = sizes.iter().sum();
    let rz = (n as f64 * 0.1).ceil();
    sizes
        .iter()
        .map(|&g| {
            let g = g as f64;
            let eig = if g as usize <= reference::DENSE_KERNEL_MAX {
                g * g
            } else {
                30.0 * g * g * (rz + 10.0)
            };
            eig + g * g * (d as f64 + 1.0)
        })
        .sum()
}

const LOWER_MEMORY: &str = "lower --n-genes or --n-factors, or run without --reference";

fn check_before_read(fit: &FitArgs, variant: Variant) -> anyhow::Result<Budget> {
    fit.check()?;
    let budget = Budget::now()?;
    let stored = fit.core.input.stored()?;
    check_stored_shape(
        "spatialpca",
        stored.as_ref().map(|s| {
            let peak = factor_peak_bytes(&fit.memory_shape(s, s.genes.min(fit.n_genes), variant));
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
        Variant::Reference(args.rank)
    } else {
        Variant::Sparse
    };
    let budget = check_before_read(&args.fit, variant)?;
    if args.reference {
        if let Some(s) = args.fit.core.input.stored()? {
            check_kernel_evals(args, &s.section_sizes)?;
        }
    }
    run_on(args, args.fit.core.input.read(false)?, &budget)
}

fn check_kernel_evals(args: &Args, sizes: &[usize]) -> anyhow::Result<()> {
    let evals = kernel_evals(sizes, args.fit.n_factors);
    anyhow::ensure!(
        evals <= args.max_kernel_evals,
        "spatialpca --reference would take about {evals:.2e} kernel evaluations, over \
         --max-kernel-evals {:.2e}; the default sparse mode has no kernel",
        args.max_kernel_evals
    );
    Ok(())
}

pub fn run_pb(args: &PbArgs) -> anyhow::Result<()> {
    let budget = check_before_read(&args.fit, Variant::Pseudobulk(args.n_pseudobulk))?;
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

/// The genes, the expression on them and its libraries, the counts freed.
fn expression(data: &mut SpatialData, genes: &[usize]) -> ResidualOp {
    let lib = library(&data.counts);
    let sel = select_rows(&data.counts, genes);
    data.counts = nalgebra_sparse::CscMatrix::zeros(0, 0);
    ResidualOp::pearson(&sel, &lib)
}

/// The domains of the PCs `z`, refined over each cell's `refine_k`
/// nearest within its section.
fn cell_domains(
    fit: &FitArgs,
    data: &SpatialData,
    z: &Dense,
) -> anyhow::Result<(Vec<usize>, &'static str)> {
    let refine = (fit.refine_k > 0).then(|| knn_within_sections(data, fit.refine_k));
    domains(&to_mat(z), &fit.core, refine.as_ref())
}

fn base_settings(
    fit: &FitArgs,
    genes: &[usize],
    names: &[Box<str>],
    clustering: &str,
) -> serde_json::Value {
    serde_json::json!({
        "paper": "Shang and Zhou 2022, doi:10.1038/s41467-022-34879-1",
        "d": fit.n_factors,
        "n_genes": fit.n_genes,
        "genes_used": genes.iter().map(|&g| names[g].as_ref()).collect::<Vec<_>>(),
        "gene_selection": "seurat_v3 highly variable genes",
        "normalisation": "analytic Pearson residuals, clipped at sqrt(N), standardised per gene",
        "clustering": clustering,
        "n_clusters": fit.core.n_clusters,
        "resolution": fit.core.n_clusters.is_none().then_some(fit.core.resolution),
        "refine_k": fit.refine_k,
    })
}

/// [`run`] on `data` as read.
pub(crate) fn run_on(args: &Args, mut data: SpatialData, budget: &Budget) -> anyhow::Result<()> {
    let fit = &args.fit;
    let genes = select_genes(&data, fit.n_genes)?;
    let variant = if args.reference {
        Variant::Reference(args.rank)
    } else {
        Variant::Sparse
    };
    ensure_fits(
        "spatialpca",
        factor_peak_bytes(&fit.memory_shape(&read_shape(&data), genes.len(), variant)),
        budget,
        LOWER_MEMORY,
    )?;
    let op = expression(&mut data, &genes);
    let d = fit.n_factors.min(op.m);
    let n = op.n;
    let (z, w, mut settings) = if args.reference {
        check_kernel_evals(args, &section_sizes(&data.section))?;
        let y = op.dense();
        drop(op);
        let xy: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..data.xy.ncols())
                    .map(|j| data.xy[(i, j)] as f64)
                    .collect()
            })
            .collect();
        let blocks = reference::blocks(&y, &xy, &data.section_cells(), args.bandwidth)?;
        let want = reference::kernel_want(n, args.rank);
        let eig = reference::kernel_eigen(&blocks, n, want, fit.core.input.seed);
        let (r_est, r_z) = reference::ranks(&eig.values, n, args.rank);
        let prof = reference::TauProfile::new(&y, &eig, r_est, d);
        let (best, evals) = prof.fit();
        let z = reference::pcs(&y, &best.w, best.tau, &eig, r_z, &blocks);
        let mut w = best.w.clone();
        let mut zz = z.clone();
        sparse::fix_signs(&mut w, &mut zz);
        let m = y.nrows();
        log::info!(
            "spatialpca --reference: tau {:.4e}, sigma0^2 {:.4e}, ranks {r_est} and {r_z}",
            best.tau,
            best.s / (m as f64 * (n as f64 - 1.0))
        );
        let settings = serde_json::json!({
            "prior": "reference",
            "kernel": "Gaussian exp(-d^2 / gamma), one block per section, coordinates standardised per axis",
            "gamma": blocks.iter().map(|b| b.gamma).collect::<Vec<_>>(),
            "bandwidth_rule": if args.bandwidth.is_some() { "--bandwidth" } else { "Silverman above 5000 cells, Sheather-Jones at or below; median over genes" },
            "tau_hat": best.tau,
            "sigma0_sq_hat": best.s / (m as f64 * (n as f64 - 1.0)),
            "eigenvalues_g": best.eigvals,
            "r_est": r_est,
            "r_z": r_z,
            "low_rank_variant": "SN(11) Woodbury on the rank-r kernel",
            "ell": best.ell,
            "evaluations": evals,
        });
        (zz, w, settings)
    } else {
        let (_, lap) = cell_graph(&data, fit.core.knn);
        let spec = spectrum(&lap, &fit.core);
        let prof = Profile::new(&op, &lap, &spec, d, fit.cg_tol, fit.core.input.seed);
        let (best, evals) = prof.fit(fit.max_evals);
        log::info!(
            "spatialpca: tau {:.4e}, rho {:.4e}, sigma0^2 {:.4e} after {evals} evaluations",
            best.tau,
            best.rho,
            best.sigma2(op.m, n)
        );
        let settings = sparse_settings(&best, op.m, n, evals, fit);
        (best.z, best.w, settings)
    };
    let (labels, clustering) = cell_domains(fit, &data, &z)?;
    let base = base_settings(fit, &genes, &data.genes, clustering);
    merge(&mut settings, base);
    let loadings = to_mat(&w);
    let gene_names: Vec<Box<str>> = genes.iter().map(|&g| data.genes[g].clone()).collect();
    write_loadings(
        &fit.core.input.out,
        "spatialpca_loadings",
        &gene_names,
        &loadings,
        "spatialpc",
    )?;
    write_run(
        &fit.core.input,
        &data,
        Fitted {
            method: "spatialpca",
            labels: &labels,
            membership: None,
            embedding: &to_mat(&z),
            embedding_name: "spatialpc",
            settings,
            reference: args.reference,
            input_layout: None,
            pseudobulk: None,
        },
    )
}

fn merge(into: &mut serde_json::Value, from: serde_json::Value) {
    if let (Some(a), serde_json::Value::Object(b)) = (into.as_object_mut(), from) {
        a.extend(b);
    }
}

fn sparse_settings(
    best: &sparse::Eval,
    m: usize,
    n: usize,
    evals: usize,
    fit: &FitArgs,
) -> serde_json::Value {
    serde_json::json!({
        "prior": "sparse",
        "graph": format!("kNN within a section, k = {}, symmetrised", fit.core.knn),
        "tau_hat": best.tau,
        "rho_hat": best.rho,
        "t": best.t,
        "log_det_term": best.h,
        "sigma0_sq_hat": best.sigma2(m, n),
        "explained": best.explained,
        "ell": best.ell,
        "evaluations": evals,
        "slq_probes": fit.core.slq_probes,
        "slq_steps": fit.core.slq_steps,
        "exact_eig_max": fit.core.exact_eig_max,
        "cg_tol": fit.cg_tol,
    })
}

/// `spatialpca-pb` on `data` as read with its projection.
pub(crate) fn run_pb_on(
    args: &PbArgs,
    mut data: SpatialData,
    budget: &Budget,
) -> anyhow::Result<()> {
    let fit = &args.fit;
    let genes = select_genes(&data, fit.n_genes)?;
    ensure_fits(
        "spatialpca-pb",
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
    let (m, n, d) = (op.m, op.n, fit.n_factors.min(op.m));
    let pb_spec = spectrum(&coarse.lap, &fit.core);
    let seed = fit.core.input.seed;
    let (pb_fit, pb_evals) =
        Profile::new(&pb_op, &coarse.lap, &pb_spec, d, fit.cg_tol, seed).fit(fit.max_evals);
    let (pb_labels, _) = domains(&to_mat(&pb_fit.z), &fit.core, None)?;
    let spec = spectrum(&lap, &fit.core);
    let (best, evals) = Profile::new(&op, &lap, &spec, d, fit.cg_tol, seed)
        .with_loadings(pb_fit.w.clone())
        .fit(fit.max_evals);
    log::info!(
        "spatialpca-pb: super-cells tau {:.3e} rho {:.3e}; cells tau {:.3e} rho {:.3e}",
        pb_fit.tau,
        pb_fit.rho,
        best.tau,
        best.rho
    );
    let (labels, clustering) = cell_domains(fit, &data, &best.z)?;
    let of_cell = &coarse.cells.of_cell;
    let broadcast: Vec<usize> = of_cell.iter().map(|&s| pb_labels[s]).collect();
    let mut settings = sparse_settings(&best, m, n, evals, fit);
    merge(
        &mut settings,
        base_settings(fit, &genes, &data.genes, clustering),
    );
    settings["super_cell_fit"] = sparse_settings(&pb_fit, m, pb_op.n, pb_evals, fit);
    settings["unravel"] = serde_json::json!({
        "method": "W fixed from the super-cell fit; sigma0^2, tau and rho refitted on the cells and their graph",
        "broadcast": "the super-cells' own labels, clustered apart, in cell_pb.parquet (a diagnostic)",
    });
    write_loadings(
        &fit.core.input.out,
        "spatialpca_loadings",
        &genes
            .iter()
            .map(|&g| data.genes[g].clone())
            .collect::<Vec<_>>(),
        &to_mat(&pb_fit.w),
        "spatialpc",
    )?;
    write_run(
        &fit.core.input,
        &data,
        Fitted {
            method: "spatialpca-pb",
            labels: &labels,
            membership: None,
            embedding: &to_mat(&best.z),
            embedding_name: "spatialpc",
            settings,
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
