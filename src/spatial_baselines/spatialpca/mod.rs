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
//! The paper's dense Gaussian kernel `K(s, s') = exp(-||s - s'||^2 /
//! gamma)` and its low-rank approximation cost too much time and memory at
//! real cell counts, so pinto offers the model with a sparse graph prior
//! in its place, and `spatialpca-pb` for scale; there is no `--reference`.
//!
//! - The prior ([`sparse`]): the kernel replaced by the sparse
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
//! # Preprocessing, as the paper's
//!
//! Genes: the spatially variable genes by SPARK-X at a
//! Benjamini-Yekutieli FDR of 0.05, at most `--n-genes` (3000, the paper's
//! cap) by p-value. Expression: SCTransform's Pearson residuals
//! (Hafemeister and Satija 2019, Genome Biology 20:296,
//! doi:10.1186/s13059-019-1874-1; [`super::spatial_factor::sct`], written
//! from its paper) on those genes, each gene standardised to mean 0 and
//! SD 1. The residuals are dense, held as f32 rows (genes × cells).
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
//! - **Gene selection**: SPARK-X on every data set (Zhu, Sun and Zhou
//!   2021, Genome Biology 22:184, doi:10.1186/s13059-021-02404-0;
//!   [`super::spatial_factor::sparkx`], written from its paper), the
//!   paper's choice for large data; the paper takes SPARK on small data
//!   sets, a GPL package not ported. Each section is centred apart (the
//!   test's covariate form).
//! - **The sparse prior** (cost, above): its `rho` by maximum
//!   likelihood; spectral sums by stochastic Lanczos quadrature on
//!   sections of more than 2000 cells; solves by conjugate gradients and
//!   `G`'s eigenpairs by subspace iteration, to tolerance; `(tau, rho)` by a
//!   projected L-BFGS from the best of a 3 × 3 grid.
//! - **Exact reformulations**: the scalar forms of SN(11) with `q = 1` and
//!   centred `Y`; `R = I` with each loading's
//!   largest entry positive; the refinement over the `k` nearest cells
//!   (the lattice neighbours on a regular lattice).
//!
//! `spatialpca-pb` fits the sparse model on cage's super-cells and the
//! collapsed graph, then unravels to cells: `W` held fixed, `sigma0^2`,
//! `tau` and `rho` refitted on the cells and their graph, each cell's PCs
//! `C^{-1} Y^T W`.

pub mod sparse;

use super::input::{InputShape, SpatialData};
use super::memory::{factor_peak_bytes, FactorShape};
use super::output::{write_loadings, write_run, Fitted, PbFit};
use super::spatial_factor::kit::Dense;
use super::spatial_factor::prep::{library, DenseRows};
use super::spatial_factor::sct::SctReport;
use super::spatial_factor::sparkx;
pub use super::spatial_factor::Variant;
use super::spatial_factor::{
    cell_graph, check_stored, coarsen, domains, merge, names_of, pseudobulk_input, read_shape,
    refined_domains, spectrum, to_mat, CoreArgs,
};
use crate::util::common::Mat;
use crate::util::memory::{ensure_fits, Budget};
use sparse::Profile;

pub const N_FACTORS: usize = 20;
pub const N_GENES: usize = 3000;
pub const REFINE_K: usize = 6;
pub const CG_TOL: f64 = 1e-8;
pub const MAX_EVALS: usize = 50;

/// The model's arguments, shared by `spatialpca` and `spatialpca-pb`.
#[derive(clap::Args, Debug, Clone)]
pub struct FitArgs {
    #[command(flatten)]
    pub core: CoreArgs,

    /// Number of spatial PCs
    #[arg(long, visible_alias = "n-pcs", default_value_t = N_FACTORS)]
    pub n_factors: usize,

    /// Genes modelled: SPARK-X's spatially variable genes (FDR 0.05), at
    /// most this many by p-value
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
        let n: usize = input.section_sizes.iter().sum();
        let super_cells = variant.super_cells();
        let (sections, threads) = (input.section_sizes.len(), rayon::current_num_threads());
        FactorShape {
            input: input.clone(),
            genes: m,
            knn: self.core.knn,
            exact_eig_max: self.core.exact_eig_max,
            cell_vectors: 7 * (d + 10) + 2 * d,
            gene_vectors: 3 * (d + 10) + d,
            super_cells,
            // SPARK-X's accumulators over every gene (three coordinates at
            // most) and its kernels' coordinates (`n x d` for each of the
            // eleven, and three more as they are formed); SCTransform's
            // residuals as f32 rows, the cells' and with -pb the
            // super-cells', and each gene's residuals in f64 as formed
            dense_bytes: sparkx::accumulator_bytes(input.genes, sections, 3, threads)
                + 8 * n * 3 * (sparkx::KERNELS + 3)
                + 4 * m * (n + super_cells)
                + 8 * n * threads,
            nonzero_bytes: super::memory::SCT_NONZERO_BYTES,
            // on cells, the counts (12) and SCTransform's copy (8) are freed
            // before the graph; -pb builds it with the counts held
            freed_before_graph: match variant {
                Variant::Sparse => 20,
                Variant::Pseudobulk(_) => 0,
            },
        }
    }
}

const LOWER_MEMORY: &str = "lower --n-genes or --n-factors, or run spatialpca-pb";

fn check_before_read(fit: &FitArgs, variant: Variant) -> anyhow::Result<Budget> {
    fit.check()?;
    check_stored("spatialpca", &fit.core.input, LOWER_MEMORY, |s| {
        fit.memory_shape(s, s.genes.min(fit.n_genes), variant)
    })
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    let budget = check_before_read(&args.fit, Variant::Sparse)?;
    run_on(args, args.fit.core.input.read(false)?, &budget)
}

pub fn run_pb(args: &PbArgs) -> anyhow::Result<()> {
    let budget = check_before_read(&args.fit, Variant::Pseudobulk(args.n_pseudobulk))?;
    run_pb_on(args, args.fit.core.input.read(true)?, &budget)
}

/// SpatialPCA's genes: SPARK-X's spatially variable genes at a
/// Benjamini-Yekutieli FDR of [`sparkx::FDR`], at most `n_genes` by
/// p-value; and how many were significant.
fn spatial_genes(data: &SpatialData, n_genes: usize) -> anyhow::Result<(Vec<usize>, usize)> {
    let xy = nalgebra::DMatrix::from_fn(data.xy.nrows(), data.xy.ncols(), |i, j| {
        data.xy[(i, j)] as f64
    });
    let p = sparkx::sparkx(&data.counts, &xy, &data.section, data.n_sections);
    let (genes, significant) = sparkx::significant(&p, n_genes);
    log::info!(
        "spatialpca: SPARK-X finds {significant} spatially variable genes, {} modelled",
        genes.len()
    );
    anyhow::ensure!(
        genes.len() >= 2,
        "SPARK-X finds {significant} spatially variable genes at FDR {}; spatialpca needs \
         at least 2",
        sparkx::FDR
    );
    Ok((genes, significant))
}

/// SCTransform's standardised residuals on `genes`, the counts freed.
fn expression(data: &mut SpatialData, genes: &[usize], seed: u64) -> (DenseRows, SctReport) {
    let lib = library(&data.counts);
    let out = DenseRows::sct(&data.counts, &lib, genes, seed);
    data.counts = nalgebra_sparse::CscMatrix::zeros(0, 0);
    out
}

/// `{out}.spatialpca_loadings.parquet`: the loadings `w` on `genes`.
fn loadings_out(
    fit: &FitArgs,
    data: &SpatialData,
    genes: &[usize],
    w: &Dense,
) -> anyhow::Result<()> {
    write_loadings(
        &fit.core.input.out,
        "spatialpca_loadings",
        &names_of(&data.genes, genes),
        &to_mat(w),
        "spatialpc",
    )
}

fn base_settings(
    fit: &FitArgs,
    genes: &[usize],
    names: &[Box<str>],
    clustering: &str,
    pre: &Preprocessing,
) -> serde_json::Value {
    serde_json::json!({
        "paper": "Shang and Zhou 2022, doi:10.1038/s41467-022-34879-1",
        "d": fit.n_factors,
        "n_genes": fit.n_genes,
        "genes_used": names_of(names, genes),
        "gene_selection": "SPARK-X, Benjamini-Yekutieli FDR 0.05, at most n_genes by p",
        "n_significant": pre.significant,
        "normalisation": "SCTransform v1 Pearson residuals, clipped at sqrt(N), standardised per gene",
        "sctransform": pre.sct,
        "clustering": clustering,
        "n_clusters": fit.core.n_clusters,
        "resolution": fit.core.n_clusters.is_none().then_some(fit.core.resolution),
        "refine_k": fit.refine_k,
    })
}

/// [`run`] on `data` as read.
pub(crate) fn run_on(args: &Args, mut data: SpatialData, budget: &Budget) -> anyhow::Result<()> {
    let fit = &args.fit;
    // before SPARK-X, at the most genes it can select
    ensure_fits(
        "spatialpca",
        factor_peak_bytes(&fit.memory_shape(
            &read_shape(&data),
            data.genes.len().min(fit.n_genes),
            Variant::Sparse,
        )),
        budget,
        LOWER_MEMORY,
    )?;
    let (genes, significant) = spatial_genes(&data, fit.n_genes)?;
    let (op, sct) = expression(&mut data, &genes, fit.core.input.seed);
    let pre = Preprocessing { significant, sct };
    let d = fit.n_factors.min(op.m);
    let n = op.n;
    let (z, w, mut settings) = {
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
    let z: Mat = to_mat(&z);
    let (labels, clustering) = refined_domains(&z, &fit.core, &data, fit.refine_k)?;
    let base = base_settings(fit, &genes, &data.genes, clustering, &pre);
    merge(&mut settings, base);
    loadings_out(fit, &data, &genes, &w)?;
    write_run(
        &fit.core.input,
        &data,
        Fitted {
            method: "spatialpca",
            labels: &labels,
            membership: None,
            embedding: &z,
            embedding_name: "spatialpc",
            settings,
            reference: false,
            input_layout: None,
            pseudobulk: None,
        },
    )
}

/// What a run's preprocessing records.
struct Preprocessing {
    significant: usize,
    sct: SctReport,
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
    ensure_fits(
        "spatialpca-pb",
        factor_peak_bytes(&fit.memory_shape(
            &read_shape(&data),
            data.genes.len().min(fit.n_genes),
            Variant::Pseudobulk(args.n_pseudobulk),
        )),
        budget,
        LOWER_MEMORY,
    )?;
    let (genes, significant) = spatial_genes(&data, fit.n_genes)?;
    let (nb, lap) = cell_graph(&data, fit.core.knn);
    let mut coarse = coarsen(&mut data, args.n_pseudobulk, &nb)?;
    drop(nb);
    let seed = fit.core.input.seed;
    let (pb_op, pb_sct) = expression(&mut coarse.cells.data, &genes, seed);
    let (op, sct) = expression(&mut data, &genes, seed);
    let pre = Preprocessing { significant, sct };
    let (m, n, d) = (op.m, op.n, fit.n_factors.min(op.m));
    let pb_spec = spectrum(&coarse.lap, &fit.core);
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
    let z = to_mat(&best.z);
    let (labels, clustering) = refined_domains(&z, &fit.core, &data, fit.refine_k)?;
    let of_cell = &coarse.cells.of_cell;
    let broadcast: Vec<usize> = of_cell.iter().map(|&s| pb_labels[s]).collect();
    let mut settings = sparse_settings(&best, m, n, evals, fit);
    merge(
        &mut settings,
        base_settings(fit, &genes, &data.genes, clustering, &pre),
    );
    settings["super_cell_fit"] = sparse_settings(&pb_fit, m, pb_op.n, pb_evals, fit);
    settings["super_cell_fit"]["sctransform"] = serde_json::json!(pb_sct);
    settings["unravel"] = serde_json::json!({
        "method": "W fixed from the super-cell fit; sigma0^2, tau and rho refitted on the cells and their graph",
        "broadcast": "the super-cells' own labels, clustered apart, in cell_pb.parquet (a diagnostic)",
    });
    loadings_out(fit, &data, &genes, &pb_fit.w)?;
    write_run(
        &fit.core.input,
        &data,
        Fitted {
            method: "spatialpca-pb",
            labels: &labels,
            membership: None,
            embedding: &z,
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
