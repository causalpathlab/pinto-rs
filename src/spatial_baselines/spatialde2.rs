// Ported from SpatialDE (https://github.com/PMBio/SpatialDE), MIT License,
// Copyright (c) 2020 Ilia Kats, Stegle Group and Copyright (c) 2018
// Teichmann Group. Its license and the commit followed are in
// THIRD_PARTY_NOTICES.md at the crate root.

//! SpatialDE2 tissue segmentation (Kats, Vento-Tormo and Stegle 2021,
//! "SpatialDE2: Fast and localized variance component analysis of spatial
//! transcriptomics", bioRxiv doi:10.1101/2021.10.27.466045): a Poisson
//! hidden Markov random field with a Dirichlet-process (stick-breaking)
//! prior over regions, fitted by closed-form variational updates
//! (Supplementary Note section 1, after Chatzis and Tsechpenakis 2010, "The
//! infinite hidden Markov random field model", IEEE TNN 21:1004).
//!
//! Ported from `PMBio/SpatialDE` at commit
//! `6dbdb559a566d639da0f6fd3acc9e17104c0674f`, file `SpatialDE/dp_hmrf.py`
//! (`tissue_segmentation`, `_segment`, `TissueSegmentationParameters`),
//! with `SpatialDE/_internal/util_mixture.py` (`prune_components`,
//! `prune_labels`) and `SpatialDE/_internal/util.py` (`calc_sizefactors`);
//! gene selection from the paper's analysis notebooks, `PMBio/spatialde2-paper`
//! at commit `c68c965e3e7330e3e47c1b2644a4f74ba77fde03` (`figure_2.ipynb`,
//! `figure_3_4.ipynb`) and Supplementary Note section 5.
//!
//! # Steps, as the reference code runs them
//!
//! - **Genes** (`figure_2.ipynb`): the paper segments on the spatially
//!   variable genes at FDR 0.001 (SpatialDE2's GP score test), the 2000
//!   with the highest total counts. *Deviation:* the GP test is not ported;
//!   the 2000 genes with the highest total counts are used
//!   (`--n-top-genes`, 0 for every gene, which is also
//!   `tissue_segmentation`'s own default `genes=None`).
//! - **Size factors** (`tissue_segmentation` → `calc_sizefactors`): each
//!   cell's total count over *all* genes (computed before the gene subset).
//! - **Spatial energy** (`tissue_segmentation`, Supplementary eq. 2): the
//!   pairwise weight is `w_nm = 2 · smoothness_factor · min d² / d²_nm`,
//!   `min d²` the smallest squared distance among the graph's pairs (here
//!   within each section; see Sections below), and
//!   `E(x_n = c) = Σ_m w_nm [x̂_m ≠ c]`; `smoothness_factor` defaults to 2.
//!   The kNN graph is symmetrised (`tensor_scatter_nd_update` on the
//!   transposed indices).
//! - **Classes** (`tissue_segmentation`): at most `nclasses =
//!   ceil(sqrt(N))`; labels start uniformly at random
//!   (`rng.choice(nclasses, N)`); `alphahat_1 = alphahat_2 = 1` (length
//!   `C - 1`), `etahat_2 = eta_1 + C - 1`, `gammahat_1 = gammahat_2 = 1e-6`.
//!   Priors `gamma_1 = gamma_2 = 1e-14` (Poisson rate Gamma prior), `eta_1 =
//!   eta_2 = 1` (Gamma hyperprior on the DP concentration).
//! - **One update** (`_segment`), all from the previous iterate: the
//!   neighbourhood log prior `log p(x_n = c | x̂_N(n)) = -E(c) -
//!   logsumexp_h(-E(h))` from the previous labels; `E[λ] = γ̂₁/γ̂₂`, `E[log λ]
//!   = ψ(γ̂₁) - log γ̂₂`; stick terms `v̂₂ = ψ(α̂₁) - ψ(α̂₁+α̂₂)`, `v̂₃ = ψ(α̂₂) -
//!   ψ(α̂₁+α̂₂)`; concentration `α̂ = (eta_1 + C₀ - 1) / η̂₂`; then
//!   `φ_cn = log p(c | x̂_N) + vsum_c + Σ_g E[log λ_cg] y_gn - s_n Σ_g E[λ_cg]`
//!   and `π̂ = softmax_c φ`; then `γ̂₁ = gamma_1 + π̂ Y`, `γ̂₂ = gamma_2 + π̂ s`,
//!   `η̂₂ = eta_2 - Σ v̂₃`, `α̂₁ = 1 + G Σ_n π̂_cn`, `α̂₂ = G Σ_n Σ_{l>c} π̂_ln + α̂`
//!   (the first `C - 1` classes); labels become `argmax_c π̂`.
//! - **ELBO** (`_segment`), divided by `C · G · N`, `C` the current class
//!   count and `C₀` the starting one (the code keeps `fnclasses` fixed).
//! - **Stopping** (`tissue_segmentation`): after at most `maxiter = 1000`
//!   updates, or when the ELBO moves by less than `abstol = 1e-12`, or by
//!   less than `reltol = 1e-12` of the smaller of the two ELBOs.
//! - **Pruning** (`prune_components(..., everything=True)`): after updates
//!   0, 1, 10, 20, ... (`i == 1 or not i % 10`) and once at the end, classes
//!   whose posterior exceeds `class_prune_threshold = 1e-6` in no cell are
//!   dropped and the labels renumbered; `α̂` keeps the retained indices but
//!   the last (`idx[:-1]`). At the end the posteriors of the retained classes
//!   are renormalised to sum to 1 per cell. (Two cases differ from
//!   `prune_labels`, where only leading classes die and where a label's own
//!   class is pruned; see Deliberate deviations.)
//!
//! Two quirks of the reference code are kept as they are, since they are
//! the model as it runs: the stick-breaking sum is `vsum_c = v̂₂_c +
//! Σ_{l < c-1} v̂₃_l` (`concat((0,), exclusive_cumsum(v̂₃)) + concat(v̂₂,
//! (0,))`, one class later than the textbook `Σ_{l<c}`), and `α̂₁`, `α̂₂`
//! count each cell `G` times.
//!
//! # Deliberate deviations
//!
//! - **Sparse graph.** The reference builds a dense `N × N` distance matrix
//!   in every mode, and its default `neighbors=None` connects every pair of
//!   spots. Here the graph is the `--neighbors` nearest cells within each
//!   section ([`super::graph::knn_within_sections`]), symmetrised, so memory
//!   stays linear in cells; the energy is the same sum, truncated to the
//!   graph. `--neighbors 0` drops space, as `neighbors=0` does in the
//!   reference (a Poisson mixture with a DP prior).
//!
//!   The default of 100 neighbours is a choice of this port, not a
//!   reference value. Cutting the `1/d²` sum does more than truncate it:
//!   in 2-D the far shells carry much of each cell's total weight, so a
//!   cut graph has a weaker effective `smoothness_factor`. On the 6 × 1600
//!   spot simulation the regions found went from shared ARI 0.638 at
//!   `--neighbors 100` to 0.753 with every pair in a section connected
//!   (`--neighbors` at least the section size, which is the reference's
//!   model within each section, at `k · N` memory).
//! - **Memory at the default cap.** The posteriors are `N × K₀` with the
//!   reference's default `K₀ = ceil(sqrt(N))`, so memory and work per
//!   update grow as `N^1.5` (about 8 GB for the posteriors alone at 1M
//!   cells, and a second copy while pruning), not linearly in cells. The
//!   default is kept for fidelity; `--n-clusters` bounds `K₀` and with it
//!   the memory. A warning is logged when the posteriors pass 4 GB.
//! - **Cells with no counts.** Kept, as in the reference: the size factor
//!   enters only as `-s_n Σ_g E[λ_cg]` and in `π̂ s`, never as a divisor,
//!   so such a cell's posterior is its prior (neighbourhood and stick).
//! - **Order of sums.** The sums over cells run over fixed chunks added in
//!   order, so a given `--seed` gives the same ELBO trace and labels on
//!   every run.
//! - **Sections.** All sections are fitted as one model (one set of region
//!   rate profiles), neighbours taken only within a section; `min d²` is
//!   taken within each section, since the reference sees one dataset: a
//!   section's weights do not depend on another section's spot spacing.
//!   Coincident cells (distance 0) take the weight of the smallest positive
//!   distance, where the reference divides by zero.
//! - **Pruning when only leading classes die.** When the retained classes
//!   have no gap but do not start at 0 (say class 0 dies and 1 to 4 live),
//!   the reference's `prune_labels` takes its no-gap branch: it returns
//!   `idx = range(len)` (0 to 3) and leaves the labels as they are. So it
//!   keeps the dead class's `γ̂` and `α̂`, drops the last live class, and
//!   leaves labels past `nclasses`. The port keeps the retained classes
//!   (1 to 4) and renumbers, as the gap branch does. The reference loses a
//!   live class at each such prune and so under-counts the regions; since
//!   the `vsum` quirk gives class 0 no prior advantage over class 1, class
//!   0 dies often enough for this to show, most of all with
//!   `--neighbors 0`. This is a bug fix, not the model as it runs.
//! - **A label on a pruned class.** With `--class-prune-threshold` at or
//!   above `1/K` a cell's most probable class can be pruned; its label moves
//!   to its most probable retained class (the reference's renumbering
//!   shifts it with the labels above it, which can leave it negative or
//!   past the last class).
//! - **ELBO terms.** The reference's three terms `Σ π̂ log p(c|x̂)`,
//!   `Σ π̂ (φ - log p(c|x̂))` and `-Σ π̂ φ` sum to zero exactly and are not
//!   computed.
//! - **Prune threshold** must be below 1 (the reference accepts 1, which
//!   prunes every class and fails).
//! - **Status.** The reference reports a relative-tolerance stop as
//!   `AbsoluteToleranceReached`; here it is logged as relative.
//! - **K.** SpatialDE2 picks the number of regions itself; it has no
//!   resolution to tune. `--n-clusters` sets the reference's own cap,
//!   `nclasses`, the most faithful handle on K: the DP may still prune to
//!   fewer regions.
//!
//! Outputs: labels = `argmax` of the posteriors; membership and the cell
//! embedding (`spatialde2_prob_*`) = the posteriors over the retained
//! classes, cells × K.

use super::graph::knn_within_sections;
use super::input::{n_sections, section_sizes, OpenedInput, SpatialData, SpatialInputArgs};
use super::memory::{knn_edges, knn_gram_bytes, spatialde2_peak_bytes, Spatialde2Shape};
use super::output::{write_run, Fitted};
use crate::util::common::Mat;
use crate::util::memory::{ensure_fits, Budget};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use special::Gamma;

const SMOOTHNESS_FACTOR: f64 = 2.0;
const CLASS_PRUNE_THRESHOLD: f64 = 1e-6;
const ABSTOL: f64 = 1e-12;
const RELTOL: f64 = 1e-12;
const MAXITER: usize = 1000;
const GAMMA_1: f64 = 1e-14;
const GAMMA_2: f64 = 1e-14;
const ETA_1: f64 = 1.0;
const ETA_2: f64 = 1.0;
/// `gammahat_1`, `gammahat_2` at the start (`tf.fill(..., 1e-6)`).
const GAMMAHAT_INIT: f64 = 1e-6;
/// The paper's segmentation genes: the top 2000 by total counts.
const N_TOP_GENES: usize = 2000;
/// Neighbours per cell in place of the reference's fully connected graph
/// (a choice of this port, not a reference value).
const NEIGHBORS: usize = 100;
/// Cells per chunk in the sums over cells, fixed so they run in one order.
const SUM_CHUNK: usize = 4096;
/// Posterior size above which a warning names `--n-clusters`.
const PI_WARN_BYTES: f64 = 4e9;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(flatten)]
    pub input: SpatialInputArgs,

    /// Maximum number of regions, SpatialDE2's `nclasses` cap (default:
    /// ceil(sqrt(cells)), as the reference). SpatialDE2 has no resolution:
    /// its Dirichlet process prunes unused regions, so a run may end with
    /// fewer. The posteriors take cells × cap memory, N^1.5 at the default;
    /// set this to bound it on large data
    #[arg(long)]
    pub n_clusters: Option<usize>,

    /// Spatial neighbours per cell, within a section (100 is this port's
    /// choice: the reference connects every pair, and cutting its 1/d^2
    /// sum weakens the smoothing; the section size reproduces it at
    /// cells × section memory; 0 ignores space: a Poisson DP mixture)
    #[arg(long, default_value_t = NEIGHBORS)]
    pub neighbors: usize,

    /// Spatial smoothness; larger values give finer segmentations
    /// (multiplied by the smallest squared distance, so dimensionless)
    #[arg(long, default_value_t = SMOOTHNESS_FACTOR)]
    pub smoothness_factor: f64,

    /// Genes to segment on: the top N by total counts (0: every gene). The
    /// paper takes the 2000 most expressed spatially variable genes; the
    /// SVG test is not ported
    #[arg(long, default_value_t = N_TOP_GENES)]
    pub n_top_genes: usize,

    /// Posterior below which, in every cell, a region is pruned
    #[arg(long, default_value_t = CLASS_PRUNE_THRESHOLD)]
    pub class_prune_threshold: f64,

    /// Absolute ELBO convergence tolerance
    #[arg(long, default_value_t = ABSTOL)]
    pub abstol: f64,

    /// Relative ELBO convergence tolerance
    #[arg(long, default_value_t = RELTOL)]
    pub reltol: f64,

    /// Maximum number of iterations
    #[arg(long, default_value_t = MAXITER)]
    pub max_iter: usize,
}

/// `TissueSegmentationParameters`, as run.
#[derive(Debug, Clone)]
pub struct Params {
    pub nclasses: Option<usize>,
    pub smoothness_factor: f64,
    pub class_prune_threshold: f64,
    pub abstol: f64,
    pub reltol: f64,
    pub maxiter: usize,
    pub gamma_1: f64,
    pub gamma_2: f64,
    pub eta_1: f64,
    pub eta_2: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            nclasses: None,
            smoothness_factor: SMOOTHNESS_FACTOR,
            class_prune_threshold: CLASS_PRUNE_THRESHOLD,
            abstol: ABSTOL,
            reltol: RELTOL,
            maxiter: MAXITER,
            gamma_1: GAMMA_1,
            gamma_2: GAMMA_2,
            eta_1: ETA_1,
            eta_2: ETA_2,
        }
    }
}

impl Params {
    /// `TissueSegmentationParameters.__post_init__`.
    fn check(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.nclasses.is_none_or(|k| k >= 1),
            "--n-clusters must be at least 1"
        );
        anyhow::ensure!(
            self.smoothness_factor > 0.0,
            "--smoothness-factor must be > 0"
        );
        anyhow::ensure!(
            (0.0..1.0).contains(&self.class_prune_threshold),
            "--class-prune-threshold must be in [0, 1)"
        );
        anyhow::ensure!(self.abstol > 0.0, "--abstol must be > 0");
        anyhow::ensure!(self.reltol > 0.0, "--reltol must be > 0");
        anyhow::ensure!(self.maxiter >= 1, "--max-iter must be at least 1");
        anyhow::ensure!(
            self.gamma_1 > 0.0 && self.gamma_2 > 0.0 && self.eta_1 > 0.0 && self.eta_2 > 0.0,
            "hyperparameters must be > 0"
        );
        Ok(())
    }
}

/// The counts the model reads over the selected genes, sparse, kept both
/// cell by cell (for `φ`) and gene by gene (for `π̂ Y`).
pub struct CellCounts {
    pub n_genes: usize,
    offsets: Vec<usize>,
    genes: Vec<u32>,
    values: Vec<f64>,
    gene_offsets: Vec<usize>,
    gene_cells: Vec<u32>,
    gene_values: Vec<f64>,
}

impl CellCounts {
    /// From `(gene, count)` lists, one per cell.
    pub fn from_rows(n_genes: usize, rows: &[Vec<(usize, f64)>]) -> Self {
        let mut offsets = vec![0];
        let mut genes = Vec::new();
        let mut values = Vec::new();
        for r in rows {
            for &(g, v) in r {
                genes.push(g as u32);
                values.push(v);
            }
            offsets.push(genes.len());
        }
        let mut gene_offsets = vec![0usize; n_genes + 1];
        for &g in &genes {
            gene_offsets[g as usize + 1] += 1;
        }
        for g in 0..n_genes {
            gene_offsets[g + 1] += gene_offsets[g];
        }
        let mut next = gene_offsets.clone();
        let mut gene_cells = vec![0u32; genes.len()];
        let mut gene_values = vec![0.0; genes.len()];
        for i in 0..rows.len() {
            for x in offsets[i]..offsets[i + 1] {
                let slot = &mut next[genes[x] as usize];
                gene_cells[*slot] = i as u32;
                gene_values[*slot] = values[x];
                *slot += 1;
            }
        }
        CellCounts {
            n_genes,
            offsets,
            genes,
            values,
            gene_offsets,
            gene_cells,
            gene_values,
        }
    }

    pub fn n_cells(&self) -> usize {
        self.offsets.len() - 1
    }

    fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.offsets[i]..self.offsets[i + 1];
        self.genes[r.clone()]
            .iter()
            .zip(&self.values[r])
            .map(|(&g, &v)| (g as usize, v))
    }

    fn column(&self, g: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.gene_offsets[g]..self.gene_offsets[g + 1];
        self.gene_cells[r.clone()]
            .iter()
            .zip(&self.gene_values[r])
            .map(|(&i, &v)| (i as usize, v))
    }
}

/// Each cell's spatial neighbours with their pairwise weights `w_nm`.
pub struct WeightedGraph {
    offsets: Vec<usize>,
    index: Vec<usize>,
    weight: Vec<f64>,
}

impl WeightedGraph {
    /// `w_nm = 2 s min d² / d²_nm` on symmetric neighbour lists (cell
    /// indices and Euclidean distances), `min d²` over the listed pairs of
    /// the cell's own section (`section[i]`), as the reference computes it
    /// over one dataset: a section's weights do not depend on another
    /// section's spot spacing.
    pub fn from_neighbours(
        index: &[Vec<usize>],
        dist: &[Vec<f32>],
        section: &[usize],
        smoothness: f64,
    ) -> Self {
        let mut min_d2 = vec![f64::INFINITY; n_sections(section)];
        for (ds, &s) in dist.iter().zip(section) {
            for &d in ds {
                let d2 = (d as f64) * (d as f64);
                if d2 > 0.0 && d2 < min_d2[s] {
                    min_d2[s] = d2;
                }
            }
        }
        let mut offsets = vec![0];
        let mut flat = Vec::new();
        let mut weight = Vec::new();
        for ((js, ds), &s) in index.iter().zip(dist).zip(section) {
            let min_d2 = min_d2[s];
            for (&j, &d) in js.iter().zip(ds) {
                let d2 = ((d as f64) * (d as f64)).max(min_d2);
                flat.push(j);
                weight.push(2.0 * smoothness * min_d2 / d2);
            }
            offsets.push(flat.len());
        }
        WeightedGraph {
            offsets,
            index: flat,
            weight,
        }
    }

    pub fn neighbours(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.offsets[i]..self.offsets[i + 1];
        self.index[r.clone()]
            .iter()
            .zip(&self.weight[r])
            .map(|(&j, &w)| (j, w))
    }
}

/// `p_x_neighborhood` for one cell: `log p(x_n = c | x̂_N(n))` over the
/// `k` classes, the log-normalised `-Σ_m w_nm [x̂_m ≠ c]`.
/// One cell's spatial log prior, written into `row` (one entry per class).
pub fn neighbourhood_log_prior_into(
    graph: &WeightedGraph,
    labels: &[usize],
    i: usize,
    row: &mut [f64],
) {
    let k = row.len();
    row.fill(0.0);
    let mut total = 0.0;
    for (j, w) in graph.neighbours(i) {
        total += w;
        if labels[j] < k {
            row[labels[j]] += w;
        }
    }
    for r in row.iter_mut() {
        *r = -(total - *r);
    }
    let lse = logsumexp(row);
    for r in row.iter_mut() {
        *r -= lse;
    }
}

fn logsumexp(x: &[f64]) -> f64 {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + x.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
}

fn ln_gamma(x: f64) -> f64 {
    Gamma::ln_gamma(x).0
}

/// The stick-breaking prior's per-class term in `φ`, `vsum`, and its parts
/// `(v̂₂, v̂₃)`, from `alphahat_1`, `alphahat_2` (length `k - 1`), as
/// `_segment` computes them.
pub fn stick_terms(alpha_1: &[f64], alpha_2: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = alpha_1.len() + 1;
    let (v2, v3): (Vec<f64>, Vec<f64>) = alpha_1
        .iter()
        .zip(alpha_2)
        .map(|(&a, &b)| {
            let dg = (a + b).digamma();
            (a.digamma() - dg, b.digamma() - dg)
        })
        .unzip();
    // exclusive cumulative sum of v̂₃ (`tf.cumsum(vhat3) - vhat3`)
    let mut excl = Vec::with_capacity(k - 1);
    let mut acc = 0.0;
    for &v in &v3 {
        excl.push(acc);
        acc += v;
    }
    let vsum = (0..k)
        .map(|c| {
            let head = if c == 0 { 0.0 } else { excl[c - 1] };
            head + if c + 1 < k { v2[c] } else { 0.0 }
        })
        .collect();
    (vsum, v2, v3)
}

/// The variational state between updates.
#[derive(Debug, Clone)]
pub struct State {
    pub k: usize,
    pub alpha_1: Vec<f64>,
    pub alpha_2: Vec<f64>,
    pub eta_hat_2: f64,
    /// `gammahat_1`, k × G row-major.
    pub gamma_hat_1: Vec<f64>,
    /// `gammahat_2`, k.
    pub gamma_hat_2: Vec<f64>,
}

impl State {
    /// The reference's starting state for `k` classes.
    pub fn init(k: usize, n_genes: usize, eta_1: f64) -> Self {
        State {
            k,
            alpha_1: vec![1.0; k - 1],
            alpha_2: vec![1.0; k - 1],
            eta_hat_2: eta_1 + k as f64 - 1.0,
            gamma_hat_1: vec![GAMMAHAT_INIT; k * n_genes],
            gamma_hat_2: vec![GAMMAHAT_INIT; k],
        }
    }
}

/// What one `_segment` call returns: the posteriors (cells × k, row-major),
/// the next state and the ELBO.
pub struct Update {
    pub pi: Vec<f64>,
    pub state: State,
    pub elbo: f64,
}

/// One `_segment` update. `k0` is the starting class count (`fnclasses`).
/// `scratch` is reused for the new posteriors (the update never reads the
/// old ones), so only one n × k buffer lives between updates. `prior` is
/// the spatial graph with the current labels (none: no spatial prior).
pub fn update(
    scratch: Vec<f64>,
    counts: &CellCounts,
    sizefactors: &[f64],
    prior: Option<(&WeightedGraph, &[usize])>,
    st: &State,
    k0: usize,
    p: &Params,
) -> Update {
    let n = counts.n_cells();
    let g = counts.n_genes;
    let k = st.k;
    let eta_1_nclasses = p.eta_1 + k0 as f64;

    // E[λ], E[log λ]; E[log λ] stored genes × k for the per-count sweep
    let mut lam1 = vec![0.0; k * g];
    let mut lam2 = vec![0.0; k * g];
    let mut lam2_t = vec![0.0; g * k];
    let mut lam1_sum = vec![0.0; k];
    for c in 0..k {
        let g2 = st.gamma_hat_2[c];
        let lg2 = g2.ln();
        for j in 0..g {
            let g1 = st.gamma_hat_1[c * g + j];
            lam1[c * g + j] = g1 / g2;
            lam2[c * g + j] = g1.digamma() - lg2;
            lam2_t[j * k + c] = lam2[c * g + j];
            lam1_sum[c] += lam1[c * g + j];
        }
    }
    let (vsum, v2, v3) = stick_terms(&st.alpha_1, &st.alpha_2);
    let alphahat = (eta_1_nclasses - 1.0) / st.eta_hat_2;

    // φ, π̂ = softmax φ, and logsumexp φ per cell
    let mut pi = scratch;
    pi.clear();
    pi.resize(n * k, 0.0);
    let lse: Vec<f64> = pi
        .par_chunks_mut(k)
        .enumerate()
        .map(|(i, row)| {
            if let Some((gr, labels)) = prior {
                neighbourhood_log_prior_into(gr, labels, i, row);
            }
            for (r, (&v, &l)) in row.iter_mut().zip(vsum.iter().zip(&lam1_sum)) {
                *r += v - sizefactors[i] * l;
            }
            for (j, y) in counts.row(i) {
                for (r, &l) in row.iter_mut().zip(&lam2_t[j * k..(j + 1) * k]) {
                    *r += l * y;
                }
            }
            let m = logsumexp(row);
            for r in row.iter_mut() {
                *r = (*r - m).exp();
            }
            m
        })
        .collect();

    // sufficient statistics: π̂ Y (genes × k, gene by gene), then π̂ s,
    // Σ_n π̂ and Σ_n Σ_{l>c} π̂ (k each)
    let py: Vec<Vec<f64>> = (0..g)
        .into_par_iter()
        .map(|j| {
            let mut acc = vec![0.0; k];
            for (i, y) in counts.column(j) {
                for (a, &r) in acc.iter_mut().zip(&pi[i * k..(i + 1) * k]) {
                    *a += r * y;
                }
            }
            acc
        })
        .collect();
    // fixed-size chunks summed in order, so the rounding (and with it the
    // stopping test and argmax ties) is the same on every run for a seed
    let parts: Vec<[Vec<f64>; 3]> = pi
        .par_chunks(k * SUM_CHUNK)
        .zip(sizefactors.par_chunks(SUM_CHUNK))
        .map(|(rows, sf)| {
            let (mut ps, mut pn, mut pc) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
            for (row, &s) in rows.chunks(k).zip(sf) {
                let mut tail = 0.0;
                for c in (0..k).rev() {
                    pc[c] += tail;
                    tail += row[c];
                    pn[c] += row[c];
                    ps[c] += row[c] * s;
                }
            }
            [ps, pn, pc]
        })
        .collect();
    let (mut ps, mut pn, mut pc) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
    for [a, b, c] in &parts {
        add(&mut ps, a);
        add(&mut pn, b);
        add(&mut pc, c);
    }

    let fg = g as f64;
    let gamma_hat_1: Vec<f64> = (0..k * g).map(|x| p.gamma_1 + py[x % g][x / g]).collect();
    let gamma_hat_2: Vec<f64> = ps.iter().map(|&v| p.gamma_2 + v).collect();
    let eta_hat_2 = p.eta_2 - v3.iter().sum::<f64>();
    let alpha_1: Vec<f64> = pn[..k - 1].iter().map(|&v| 1.0 + fg * v).collect();
    let alpha_2: Vec<f64> = pc[..k - 1].iter().map(|&v| fg * v + alphahat).collect();

    let mut elbo = 0.0;
    elbo += alpha_2
        .iter()
        .zip(&v3)
        .map(|(&a2, &v)| (alphahat - a2) * v)
        .sum::<f64>();
    for (c, &g2) in gamma_hat_2.iter().enumerate() {
        let lg2 = g2.ln();
        for x in c * g..(c + 1) * g {
            let g1 = gamma_hat_1[x];
            elbo += (p.gamma_1 - g1) * lam2[x] + (g2 - p.gamma_2) * lam1[x];
            elbo -= g1 * lg2 - ln_gamma(g1);
        }
    }
    elbo += lse.iter().sum::<f64>();
    elbo -= alpha_1
        .iter()
        .zip(&v2)
        .map(|(&a1, &v)| (a1 - 1.0) * v)
        .sum::<f64>();
    elbo += alpha_1
        .iter()
        .zip(&alpha_2)
        .map(|(&a, &b)| ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b))
        .sum::<f64>();
    elbo -= eta_1_nclasses * eta_hat_2.ln();
    elbo += (eta_hat_2 - p.eta_2) * alphahat;
    elbo /= (k * g * n) as f64;

    Update {
        pi,
        state: State {
            k,
            alpha_1,
            alpha_2,
            eta_hat_2,
            gamma_hat_1,
            gamma_hat_2,
        },
        elbo,
    }
}

fn add(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Each cell's most probable class (first on ties, as `tf.math.argmax`).
pub fn argmax_rows(pi: &[f64], k: usize) -> Vec<usize> {
    pi.chunks(k)
        .map(|row| argmax(row.iter().copied()))
        .collect()
}

/// The position of the largest value, the first on ties (0 when empty).
fn argmax(values: impl Iterator<Item = f64>) -> usize {
    let mut best = (0, f64::NEG_INFINITY);
    for (c, v) in values.enumerate() {
        if v > best.1 {
            best = (c, v);
        }
    }
    best.0
}

/// `prune_components(..., everything=True)`: the classes whose posterior
/// exceeds `threshold` in some cell, in order, and the labels renumbered
/// onto them. A label on a dropped class (possible once `threshold` is at
/// least `1/k`) moves to the cell's most probable retained class.
pub fn prune(pi: &[f64], k: usize, labels: &[usize], threshold: f64) -> (Vec<usize>, Vec<usize>) {
    let mut keep = vec![false; k];
    for row in pi.chunks(k) {
        for (c, &v) in row.iter().enumerate() {
            keep[c] |= v > threshold;
        }
    }
    let idx: Vec<usize> = (0..k).filter(|&c| keep[c]).collect();
    let mut new = vec![usize::MAX; k];
    for (a, &c) in idx.iter().enumerate() {
        new[c] = a;
    }
    let labels = labels
        .iter()
        .zip(pi.chunks(k))
        .map(|(&l, row)| match new[l] {
            usize::MAX => argmax(idx.iter().map(|&c| row[c])),
            a => a,
        })
        .collect();
    (idx, labels)
}

const ALL_PRUNED: &str = "spatialde2: every region fell below --class-prune-threshold";

/// Why the iterations stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Status {
    AbsoluteTolerance,
    RelativeTolerance,
    MaximumIterations,
}

/// What [`segment`] returns.
pub struct Segmentation {
    /// Posteriors over the retained classes, cells × k, row-major, rows
    /// summing to 1.
    pub pi: Vec<f64>,
    pub k: usize,
    pub labels: Vec<usize>,
    pub status: Status,
    /// Updates run (the reference's `niter`, 0-based last index).
    pub niter: usize,
    pub elbo_trace: Vec<f64>,
    pub nclasses_trace: Vec<usize>,
    pub prune_iterations: Vec<usize>,
}

/// Regions at the start: `nclasses`, else ceil(sqrt(cells)), one at least.
pub fn initial_regions(n: usize, nclasses: Option<usize>) -> usize {
    nclasses
        .unwrap_or_else(|| (n as f32).sqrt().ceil() as usize)
        .max(1)
}

/// `tissue_segmentation`: the iterations, pruning and the final prune.
pub fn segment(
    counts: &CellCounts,
    sizefactors: &[f64],
    graph: Option<&WeightedGraph>,
    p: &Params,
    seed: u64,
) -> anyhow::Result<Segmentation> {
    p.check()?;
    let n = counts.n_cells();
    anyhow::ensure!(n > 0, "no cells");
    anyhow::ensure!(sizefactors.len() == n, "one size factor per cell");
    let k0 = initial_regions(n, p.nclasses);
    let pi_bytes = n as f64 * k0 as f64 * 8.0;
    if pi_bytes > PI_WARN_BYTES {
        log::warn!(
            "spatialde2: posteriors of {n} cells × {k0} regions take {:.1} GB \
             (twice that while pruning); --n-clusters bounds it",
            pi_bytes / 1e9
        );
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..k0)).collect();
    let mut st = State::init(k0, counts.n_genes, p.eta_1);

    let mut last = f64::NEG_INFINITY;
    let mut status = Status::MaximumIterations;
    let mut elbo_trace = Vec::new();
    let mut nclasses_trace = Vec::new();
    let mut prune_iterations = Vec::new();
    let mut pi = Vec::new();
    let mut niter = 0;
    for i in 0..p.maxiter {
        niter = i;
        let u = update(
            std::mem::take(&mut pi),
            counts,
            sizefactors,
            graph.map(|g| (g, labels.as_slice())),
            &st,
            k0,
            p,
        );
        pi = u.pi;
        st = u.state;
        labels = argmax_rows(&pi, st.k);
        elbo_trace.push(u.elbo);
        nclasses_trace.push(st.k);
        let diff = (u.elbo - last).abs();
        if diff < p.abstol {
            status = Status::AbsoluteTolerance;
            break;
        } else if diff / u.elbo.abs().min(last.abs()) < p.reltol {
            status = Status::RelativeTolerance;
            break;
        }
        last = u.elbo;
        if i == 1 || i % 10 == 0 {
            let k_old = st.k;
            let (idx, relabelled) = prune(&pi, k_old, &labels, p.class_prune_threshold);
            anyhow::ensure!(!idx.is_empty(), "{ALL_PRUNED}");
            if idx.len() < k_old {
                prune_iterations.push(i);
                let g = counts.n_genes;
                let sticks = &idx[..idx.len() - 1];
                st = State {
                    k: idx.len(),
                    alpha_1: sticks.iter().map(|&c| st.alpha_1[c]).collect(),
                    alpha_2: sticks.iter().map(|&c| st.alpha_2[c]).collect(),
                    eta_hat_2: st.eta_hat_2,
                    gamma_hat_1: idx
                        .iter()
                        .flat_map(|&c| st.gamma_hat_1[c * g..(c + 1) * g].iter().copied())
                        .collect(),
                    gamma_hat_2: idx.iter().map(|&c| st.gamma_hat_2[c]).collect(),
                };
                // keep `pi` on the retained classes, so it and the
                // renumbered labels agree should the loop end here
                pi = pi
                    .chunks(k_old)
                    .flat_map(|row| idx.iter().map(|&c| row[c]))
                    .collect();
            }
            labels = relabelled;
        }
    }
    let k_now = pi.len() / n;
    let (idx, labels) = prune(&pi, k_now, &labels, p.class_prune_threshold);
    anyhow::ensure!(!idx.is_empty(), "{ALL_PRUNED}");
    let k = idx.len();
    let pi: Vec<f64> = pi
        .chunks(k_now)
        .flat_map(|row| {
            let s: f64 = idx.iter().map(|&c| row[c]).sum();
            idx.iter().map(move |&c| row[c] / s)
        })
        .collect();
    Ok(Segmentation {
        pi,
        k,
        labels,
        status,
        niter,
        elbo_trace,
        nclasses_trace,
        prune_iterations,
    })
}

/// The `n_top` genes with the highest totals, in index order (every gene
/// when `n_top` is 0 or covers them), and each cell's total over all genes
/// (`calc_sizefactors`).
pub fn top_genes(data: &SpatialData, n_top: usize) -> (Vec<usize>, Vec<f64>) {
    let n_genes = data.genes.len();
    let csc = &data.counts;
    let mut total = vec![0.0f64; n_genes];
    let mut sizefactors = vec![0.0f64; data.cells.len()];
    for (i, col) in csc.col_iter().enumerate() {
        for (&g, &v) in col.row_indices().iter().zip(col.values()) {
            total[g] += v as f64;
            sizefactors[i] += v as f64;
        }
    }
    let mut order: Vec<usize> = (0..n_genes).collect();
    order.sort_by(|&a, &b| total[b].total_cmp(&total[a]).then(a.cmp(&b)));
    if n_top > 0 && n_top < n_genes {
        order.truncate(n_top);
    }
    order.sort_unstable();
    (order, sizefactors)
}

/// `genes` (sorted) renumbered `0..genes.len()`, by gene index; other
/// genes `usize::MAX`.
fn gene_map(n_genes: usize, genes: &[usize]) -> Vec<usize> {
    let mut map = vec![usize::MAX; n_genes];
    for (new, &g) in genes.iter().enumerate() {
        map[g] = new;
    }
    map
}

/// The counts of `genes` (sorted), cells by genes.
pub fn selected_counts(data: &SpatialData, genes: &[usize]) -> CellCounts {
    let map = gene_map(data.genes.len(), genes);
    let rows: Vec<Vec<(usize, f64)>> = data
        .counts
        .col_iter()
        .map(|col| {
            col.row_indices()
                .iter()
                .zip(col.values())
                .filter(|(&g, &v)| map[g] != usize::MAX && v != 0.0)
                .map(|(&g, &v)| (map[g], v as f64))
                .collect()
        })
        .collect();
    CellCounts::from_rows(genes.len(), &rows)
}

/// Nonzeros [`selected_counts`] keeps.
pub fn selected_nonzeros(data: &SpatialData, genes: &[usize]) -> usize {
    let map = gene_map(data.genes.len(), genes);
    data.counts
        .triplet_iter()
        .filter(|&(g, _, &v)| map[g] != usize::MAX && v != 0.0)
        .count()
}

/// What would lower SpatialDE2's peak memory, for [`ensure_fits`].
const LOWER_MEMORY: &str = "lower --n-top-genes, --neighbors or --n-clusters";

impl Args {
    fn params(&self) -> Params {
        Params {
            nclasses: self.n_clusters,
            smoothness_factor: self.smoothness_factor,
            class_prune_threshold: self.class_prune_threshold,
            abstol: self.abstol,
            reltol: self.reltol,
            maxiter: self.max_iter,
            ..Params::default()
        }
    }
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    args.params().check()?;
    let budget = Budget::now()?;
    let input = args.input.open()?;
    check_input(args, &input, &budget)?;
    run_on(args, input.read()?, &budget)
}

/// The first memory check, from the files before their counts are read:
/// every stored nonzero counted as selected too, so it bounds the check
/// after the read. A file that does not store its nonzero count leaves the
/// read unchecked.
fn check_input(args: &Args, input: &OpenedInput, budget: &Budget) -> anyhow::Result<()> {
    let Some(estimate) = input_peak_bytes(args, input) else {
        log::info!("spatialde2: a file does not store its nonzero count; its read is not checked");
        return Ok(());
    };
    ensure_fits(
        "spatialde2 (from the input's stored shape)",
        estimate,
        budget,
        LOWER_MEMORY,
    )
}

/// The peak [`check_input`] checks, `None` when a file does not store its
/// nonzero count.
pub(crate) fn input_peak_bytes(args: &Args, input: &OpenedInput) -> Option<usize> {
    let s = input.shape()?;
    let shape = memory_shape(args, &s.section_sizes, s.nonzeros, s.nonzeros);
    Some(spatialde2_peak_bytes(&shape))
}

/// What a run on `nonzeros` counts, `selected` of them on the genes
/// segmented on, and sections of `section_sizes` cells holds
/// ([`Spatialde2Shape`]).
fn memory_shape(
    args: &Args,
    section_sizes: &[usize],
    nonzeros: usize,
    selected: usize,
) -> Spatialde2Shape {
    let n = section_sizes.iter().sum();
    Spatialde2Shape {
        cells: n,
        nonzeros,
        selected_nonzeros: selected,
        // symmetrising at most doubles the edges
        edges: 2 * knn_edges(section_sizes, args.neighbors),
        regions: initial_regions(n, args.n_clusters),
        gram: if args.neighbors > 0 {
            knn_gram_bytes(section_sizes, rayon::current_num_threads())
        } else {
            0
        },
    }
}

/// [`run`] on `data` as read: its peak memory is checked against the
/// budget (`budget`, taken before the read) before the counts are copied
/// and the graph is built.
pub(crate) fn run_on(args: &Args, mut data: SpatialData, budget: &Budget) -> anyhow::Result<()> {
    let params = args.params();
    let (genes, sizefactors) = top_genes(&data, args.n_top_genes);
    let n = data.cells.len();
    let sizes = section_sizes(&data.section);
    let spatial = args.neighbors > 0;
    let shape = memory_shape(
        args,
        &sizes,
        data.counts.nnz(),
        selected_nonzeros(&data, &genes),
    );
    ensure_fits(
        "spatialde2",
        spatialde2_peak_bytes(&shape),
        budget,
        LOWER_MEMORY,
    )?;
    let counts = selected_counts(&data, &genes);
    // the model reads its own copy from here on
    data.counts = nalgebra_sparse::CscMatrix::zeros(0, 0);
    log::info!(
        "spatialde2: {} genes by total counts, size factors over all {}",
        genes.len(),
        data.genes.len()
    );
    let graph = spatial.then(|| {
        let nb = knn_within_sections(&data, args.neighbors).symmetrised();
        log::info!(
            "spatialde2: {} spatial edges (k = {}, symmetrised)",
            nb.n_edges(),
            args.neighbors
        );
        WeightedGraph::from_neighbours(&nb.index, &nb.dist, &data.section, params.smoothness_factor)
    });
    let seg = segment(
        &counts,
        &sizefactors,
        graph.as_ref(),
        &params,
        args.input.seed,
    )?;
    log::info!(
        "spatialde2: {:?} after {} iterations, {} regions, ELBO {:.6e}",
        seg.status,
        seg.niter + 1,
        seg.k,
        seg.elbo_trace.last().copied().unwrap_or(f64::NAN)
    );
    let probs = Mat::from_fn(n, seg.k, |i, c| seg.pi[i * seg.k + c] as f32);
    let k0 = seg.nclasses_trace.first().copied().unwrap_or(0);
    let settings = serde_json::json!({
        "nclasses": k0,
        "n_clusters": args.n_clusters,
        "neighbors": args.neighbors,
        "smoothness_factor": params.smoothness_factor,
        "n_top_genes": args.n_top_genes,
        "n_genes_used": genes.len(),
        "genes_used": genes.iter().map(|&g| data.genes[g].as_ref()).collect::<Vec<_>>(),
        "class_prune_threshold": params.class_prune_threshold,
        "abstol": params.abstol,
        "reltol": params.reltol,
        "maxiter": params.maxiter,
        "gamma_1": params.gamma_1,
        "gamma_2": params.gamma_2,
        "eta_1": params.eta_1,
        "eta_2": params.eta_2,
        "status": seg.status,
        "iterations": seg.niter + 1,
        "n_regions": seg.k,
        "prune_iterations": seg.prune_iterations,
        "elbo_final": seg.elbo_trace.last(),
        "reference": "PMBio/SpatialDE@6dbdb559a566d639da0f6fd3acc9e17104c0674f SpatialDE/dp_hmrf.py",
    });
    write_run(
        &args.input,
        &data,
        Fitted {
            method: "spatialde2",
            labels: &seg.labels,
            membership: Some(&probs),
            embedding: &probs,
            embedding_name: "spatialde2_prob",
            settings,
        },
    )
}
