//! STAGATE (Dong and Zhang 2022, "Deciphering spatial domains from
//! spatially resolved transcriptomics with an adaptive graph attention
//! auto-encoder", Nature Communications 13:1739,
//! doi:10.1038/s41467-022-29439-6): a graph-attention autoencoder over the
//! spatial neighbour graph, whose 30-d bottleneck is clustered into domains.
//!
//! Ported at the model level from the PyTorch reference, STAGATE_pyG
//! (github.com/QIFEIDKN/STAGATE_pyG, commit ae1158c, 2023-11-10, the
//! version the authors' tutorials now use), and read against the TF1
//! original (github.com/zhanglabtools/STAGATE, commit d6dd9f4) where the
//! two differ. Preprocessing is that of the tutorials
//! (stagate.readthedocs.io, T1_DLPFC and T7_pyG).
//!
//! Each step, with the file and function it follows:
//!
//! 1. Genes: `sc.pp.highly_variable_genes(flavor="seurat_v3",
//!    n_top_genes=3000)` on the raw counts ([`seurat_v3_hvg`], after
//!    scanpy's `_highly_variable_genes_seurat_v3`): per gene the mean and
//!    unbiased variance of the counts; a degree-2 loess (span 0.3, tricube
//!    weights, Gaussian family) of log10 variance on log10 mean over the
//!    genes with any variance ([`loess_quadratic`]); each count clipped at
//!    `mean + sqrt(N) * reg_sd`, `reg_sd = sqrt(10^fit)`; the variance of the
//!    clipped counts standardised by `reg_sd²`; the `n_top_genes` largest
//!    (all genes when there are fewer). Sections are pooled (no
//!    `batch_key`), as in the tutorials.
//! 2. `sc.pp.normalize_total(target_sum=1e4)` then `sc.pp.log1p` on every
//!    gene ([`normalise_log`]), then the variable genes only
//!    (`Train_STAGATE.py:train_STAGATE`, `adata[:, highly_variable]`), dense.
//! 3. The spatial network (`utils.py:Cal_Spatial_Net`): `model='KNN'` links
//!    each cell to its `k_cutoff` nearest cells, `model='Radius'` to every
//!    cell within `rad_cutoff` (inclusive), within its own section; edges of
//!    distance 0 are dropped (`Distance > 0`). `Transfer_pytorch_Data` then
//!    adds a self loop to every cell. An edge runs from `Cell1` (the cell
//!    whose neighbour list it is) to `Cell2` (the neighbour), and messages
//!    are summed at `Cell2` (PyG's `source_to_target` flow; the TF version's
//!    `prepare_graph_data` transposes to the same thing). Several sections
//!    train together as one graph with no edges between them.
//! 4. The model (`STAGATE.py:STAGATE`, `gat_conv.py:GATConv`, one head, no
//!    bias, no dropout, hidden dims `[512, 30]`):
//!    `h1 = elu(A(x W1) x W1)`, `h2 = h1 W2` (the embedding),
//!    `h3 = elu(A(x W1) h2 W3)`, `h4 = h3 W4`, with `W3 = W2ᵀ` and
//!    `W4 = W1ᵀ` (tied) and the decoder's attention the encoder's own
//!    (`tied_attention=conv1.attentions`). Attention ([`attention`]): per
//!    node `s = (x W1) a_src`, `d = (x W1) a_dst`; per edge `j → i` the
//!    logit `s_j + d_i`, then `sigmoid` (the 2022-08-23 change from
//!    LeakyReLU, as in the TF code), then a softmax over the edges into
//!    `i`. Aggregation ([`aggregate`]) gathers and scatter-adds over the
//!    edge list: no dense adjacency is ever formed.
//! 5. Initialisation (`GATConv.__init__`): every weight and attention vector
//!    Xavier normal with gain 1.414.
//! 6. Training (`Train_STAGATE.py:train_STAGATE`): loss `F.mse_loss(x, h4)`
//!    (the mean over every entry); full batch; `n_epochs = 1000`; Adam with
//!    `lr = 0.001`, `weight_decay = 1e-4` (L2 added to the gradient, torch's
//!    `Adam`, not AdamW), betas (0.9, 0.999), eps 1e-8; before each step the
//!    gradients' global L2 norm clipped at 5 (`clip_grad_norm_`).
//! 7. The tie, as the code does it: `conv3.lin_src.data =
//!    conv2.lin_src.transpose(0, 1)` makes the decoder weight a separate
//!    parameter that shares the encoder weight's storage. So the encoder
//!    weight receives two Adam updates per step, its own and the decoder
//!    parameter's (each with its own moments, each clipped in the one global
//!    norm), both with weight decay on the pre-step values (torch's
//!    multi-tensor Adam). [`Stagate::step`] does exactly that.
//! 8. The embedding: `h2` from a last forward pass after training.
//! 9. Clustering. With `--n-clusters K`, the tutorials' `mclust_R(num_cluster
//!    = K, modelNames='EEE')`: a Gaussian mixture with one full covariance
//!    shared by all components, fitted by EM ([`eee_gmm`]) until the
//!    log-likelihood changes by less than 1e-5 of itself (mclust's `tol`).
//!    Without it, the pyG tutorial's `sc.pp.neighbors(use_rep='STAGATE')`
//!    (15 neighbours) and `sc.tl.louvain(resolution=0.5)`, as Leiden at
//!    modularity resolution 0.5 on the UMAP-weighted kNN graph.
//!
//! Memory is linear in the cells: the input is dense cells × variable genes,
//! as in the reference, and the graph, attention and messages live on the
//! edge list (cells × (k + 1) entries).
//!
//! Deliberate deviations:
//!
//! - The loess is evaluated exactly at every gene (`surface="direct"`),
//!   where scikit-misc's default interpolates a kd-tree of vertex fits
//!   (`surface="interpolate"`); the two agree to the interpolation error.
//! - mclust starts EM from model-based agglomerative clustering (`hc`), which
//!   is quadratic in the cells; this runs EM from 10 seeded k-means
//!   partitions (100 Lloyd rounds each), keeps the run of highest
//!   log-likelihood, and stops EM after 1000 rounds. A single k-means start
//!   is not enough: on the simulation fixture it often lands in a poor
//!   optimum.
//! - Louvain is replaced by Leiden (the one pinto has); both maximise
//!   modularity at the same resolution.
//! - The default spatial network is `KNN` with `k_cutoff = 6`, the
//!   neighbourhood of the tutorials' spot data (Visium, `rad_cutoff = 150`,
//!   "5.80 neighbors per cell"); the tutorials themselves pass a radius,
//!   which `--rad-cutoff` gives. Radius neighbours are searched among each
//!   cell's 256 nearest.
//! - The seed is pinto's `--seed` (default 42) rather than the code's 0;
//!   random streams differ from torch's anyway. On CUDA the scatter-adds
//!   are atomic, so two runs differ in the last digits (as PyG's do); on
//!   the CPU a seed gives one result.
//! - The compute device defaults to the CPU, as `cage`; the reference uses
//!   `cuda:0` when there is one. Pass `--device cuda`.
//! - The TF original differs from the PyTorch one in ways not followed here:
//!   loss `sqrt(sum of squares)` plus `1e-4 * l2_loss` of the last weight
//!   only, `lr = 1e-4`, 500 epochs, Glorot uniform initialisation, and the
//!   cell type-aware module (`alpha > 0`, a pruned second graph). The
//!   tutorials' `alpha=0` disables that module, and the pyG version does
//!   not have it.

use super::graph::{knn_within_sections, radius_within_sections, Neighbours};
use super::input::{SpatialData, SpatialInputArgs};
use super::output::{write_run, Fitted};
use crate::simulate::baseline::normalise_log;
use crate::util::common::Mat;
use crate::util::device::ComputeDevice;
use anyhow::Context;
use legume_numeric::candle::candle_core::{DType, Device, Tensor, Var};
use legume_numeric::candle::fast_index::{gather_rows, index_add_rows};
use log::info;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CscMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

const N_TOP_GENES: usize = 3000;
const LOESS_SPAN: f64 = 0.3;
const HIDDEN: usize = 512;
const LATENT: usize = 30;
const N_EPOCHS: usize = 1000;
const LR: f64 = 1e-3;
const WEIGHT_DECAY: f64 = 1e-4;
const GRAD_CLIP: f64 = 5.0;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const ADAM_EPS: f64 = 1e-8;
const INIT_GAIN: f64 = 1.414;
const K_CUTOFF: usize = 6;
const RADIUS_MAX_K: usize = 256;
const LOUVAIN_RESOLUTION: f64 = 0.5;
const SCANPY_NEIGHBOURS: usize = 15;
const MCLUST_TOL: f64 = 1e-5;
const EM_MAX_ROUNDS: usize = 1000;
const KMEANS_ROUNDS: usize = 100;
const KMEANS_STARTS: u64 = 10;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(flatten)]
    pub input: SpatialInputArgs,

    /// Number of domains. Given, the embedding is clustered as the STAGATE
    /// tutorials do, by mclust's EEE Gaussian mixture with this many
    /// components; not given, by Leiden at --resolution (the pyG
    /// tutorial's Louvain at 0.5), which picks its own number
    #[arg(long)]
    pub n_clusters: Option<usize>,

    /// Modularity resolution of the Leiden step used when --n-clusters is
    /// not given (the pyG tutorial's sc.tl.louvain resolution)
    #[arg(long, default_value_t = LOUVAIN_RESOLUTION)]
    pub resolution: f64,

    /// Spatial network: each cell's k nearest cells in its section
    /// (Cal_Spatial_Net model='KNN', k_cutoff); ignored with --rad-cutoff
    #[arg(long, default_value_t = K_CUTOFF)]
    pub k_cutoff: usize,

    /// Spatial network: every cell within this distance, in coordinate
    /// units (Cal_Spatial_Net model='Radius', rad_cutoff; the tutorials use
    /// 150 for Visium pixels, 50 for Slide-seq/Stereo-seq). Overrides
    /// --k-cutoff
    #[arg(long)]
    pub rad_cutoff: Option<f32>,

    /// Number of highly variable genes (seurat_v3 flavour, on counts)
    #[arg(long, default_value_t = N_TOP_GENES)]
    pub n_top_genes: usize,

    /// Hidden and embedding dimensions of the encoder
    #[arg(long, value_delimiter(','), default_values_t = [HIDDEN, LATENT])]
    pub hidden_dims: Vec<usize>,

    /// Training epochs (full batch)
    #[arg(long, default_value_t = N_EPOCHS)]
    pub n_epochs: usize,

    /// Adam learning rate
    #[arg(long, default_value_t = LR)]
    pub lr: f64,

    /// Adam weight decay (L2 on the gradient, as torch's Adam)
    #[arg(long, default_value_t = WEIGHT_DECAY)]
    pub weight_decay: f64,

    /// Gradient clipping: the global L2 norm of the gradients
    #[arg(long, default_value_t = GRAD_CLIP)]
    pub gradient_clipping: f64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,
}

// ---------------------------------------------------------------- genes

/// Degree-2 local regression of `y` on `x` (Cleveland's loess, Gaussian
/// family, no robustness rounds), evaluated at every `x`: each point fits
/// its `floor(span * n)` nearest points (at least 3), weighted by the
/// tricube of distance over the farthest of them.
pub fn loess_quadratic(x: &[f64], y: &[f64], span: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let q = ((span * n as f64).floor() as usize).clamp(3.min(n), n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    // the q nearest of each sorted point form a window [lo, lo + q)
    let mut lo = 0usize;
    let starts: Vec<usize> = (0..n)
        .map(|p| {
            while lo + q < n && xs[lo + q] - xs[p] < xs[p] - xs[lo] {
                lo += 1;
            }
            lo
        })
        .collect();
    let fitted_sorted: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|p| {
            let x0 = xs[p];
            let lo = starts[p];
            let win = lo..lo + q;
            let dmax = (x0 - xs[lo]).max(xs[lo + q - 1] - x0);
            let mut ata = [[0f64; 3]; 3];
            let mut aty = [0f64; 3];
            for i in win {
                let u = xs[i] - x0;
                let w = if dmax > 0.0 {
                    let r = (u.abs() / dmax).min(1.0);
                    (1.0 - r * r * r).powi(3)
                } else {
                    1.0
                };
                let b = [1.0, u, u * u];
                for a in 0..3 {
                    aty[a] += w * b[a] * ys[i];
                    for c in 0..3 {
                        ata[a][c] += w * b[a] * b[c];
                    }
                }
            }
            solve_local(&ata, &aty)
        })
        .collect();
    let mut out = vec![0f64; n];
    for (p, &i) in order.iter().enumerate() {
        out[i] = fitted_sorted[p];
    }
    out
}

/// The intercept of the local fit: quadratic, else linear, else the mean.
fn solve_local(ata: &[[f64; 3]; 3], aty: &[f64; 3]) -> f64 {
    let m = nalgebra::Matrix3::from_fn(|i, j| ata[i][j]);
    let scale = m.diagonal().max().max(1e-300);
    if let Some(inv) = (m / scale).try_inverse() {
        let beta = inv * nalgebra::Vector3::from_column_slice(aty) / scale;
        if beta[0].is_finite() && (m.determinant() / scale.powi(3)).abs() > 1e-10 {
            return beta[0];
        }
    }
    let m2 = nalgebra::Matrix2::new(ata[0][0], ata[0][1], ata[1][0], ata[1][1]);
    if m2.determinant().abs() > 1e-12 * ata[0][0].max(1e-300).powi(2) {
        if let Some(inv) = m2.try_inverse() {
            return (inv * nalgebra::Vector2::new(aty[0], aty[1]))[0];
        }
    }
    aty[0] / ata[0][0].max(1e-300)
}

/// scanpy's `highly_variable_genes(flavor="seurat_v3")` on raw counts
/// (genes × cells), one batch: each gene's standardised variance, the
/// `n_top` largest picked (ties by gene order), returned in gene order with
/// every gene's standardised variance.
pub fn seurat_v3_hvg(counts: &CscMatrix<f32>, n_top: usize) -> (Vec<usize>, Vec<f64>) {
    let (n_genes, n) = (counts.nrows(), counts.ncols());
    let nf = n as f64;
    let mut sum = vec![0f64; n_genes];
    for (g, _, &v) in counts.triplet_iter() {
        sum[g] += v as f64;
    }
    let mean: Vec<f64> = sum.iter().map(|s| s / nf).collect();
    let mut nnz = vec![0f64; n_genes];
    let mut ss = vec![0f64; n_genes];
    for (g, _, &v) in counts.triplet_iter() {
        ss[g] += (v as f64 - mean[g]).powi(2);
        nnz[g] += 1.0;
    }
    let var: Vec<f64> = (0..n_genes)
        .map(|g| (ss[g] + (nf - nnz[g]) * mean[g] * mean[g]) / (nf - 1.0).max(1.0))
        .collect();
    let varying: Vec<usize> = (0..n_genes).filter(|&g| var[g] > 0.0).collect();
    let lx: Vec<f64> = varying.iter().map(|&g| mean[g].log10()).collect();
    let ly: Vec<f64> = varying.iter().map(|&g| var[g].log10()).collect();
    let fit = loess_quadratic(&lx, &ly, LOESS_SPAN);
    let mut estimated = vec![0f64; n_genes];
    for (&g, &f) in varying.iter().zip(&fit) {
        estimated[g] = f;
    }
    let reg_sd: Vec<f64> = estimated.iter().map(|e| 10f64.powf(*e).sqrt()).collect();
    let vmax = nf.sqrt();
    let mut csum = vec![0f64; n_genes];
    let mut csq = vec![0f64; n_genes];
    for (g, _, &v) in counts.triplet_iter() {
        let c = (v as f64).min(reg_sd[g] * vmax + mean[g]);
        csum[g] += c;
        csq[g] += c * c;
    }
    let norm_var: Vec<f64> = (0..n_genes)
        .map(|g| {
            let m = mean[g];
            (nf * m * m + csq[g] - 2.0 * csum[g] * m) / ((nf - 1.0).max(1.0) * reg_sd[g].powi(2))
        })
        .collect();
    let mut order: Vec<usize> = (0..n_genes).collect();
    order.sort_by(|&a, &b| norm_var[b].total_cmp(&norm_var[a]).then(a.cmp(&b)));
    order.truncate(n_top);
    order.sort_unstable();
    (order, norm_var)
}

/// Rows `genes` of log data (genes × cells) as a dense cells × genes
/// row-major buffer.
fn dense_rows(log: &CscMatrix<f32>, genes: &[usize]) -> Vec<f32> {
    let mut at = vec![None; log.nrows()];
    for (j, &g) in genes.iter().enumerate() {
        at[g] = Some(j);
    }
    let p = genes.len();
    let mut x = vec![0f32; log.ncols() * p];
    for (g, c, &v) in log.triplet_iter() {
        if let Some(j) = at[g] {
            x[c * p + j] = v;
        }
    }
    x
}

// ---------------------------------------------------------------- graph

/// Cal_Spatial_Net then Transfer_pytorch_Data: edges `(source, target)`,
/// each cell's spatial neighbours as `(cell, neighbour)` with distance 0
/// dropped, then one self loop per cell; duplicates merged.
pub fn spatial_edges(nb: &Neighbours) -> (Vec<u32>, Vec<u32>) {
    let n = nb.n_cells();
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(nb.n_edges() + n);
    for (i, (js, ds)) in nb.index.iter().zip(&nb.dist).enumerate() {
        for (&j, &d) in js.iter().zip(ds) {
            if d > 0.0 {
                pairs.push((i as u32, j as u32));
            }
        }
        pairs.push((i as u32, i as u32));
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs.into_iter().unzip()
}

/// The edge list on the device.
pub struct EdgeIndex {
    pub src: Tensor,
    pub dst: Tensor,
    pub n_nodes: usize,
}

impl EdgeIndex {
    pub fn new(src: &[u32], dst: &[u32], n_nodes: usize, dev: &Device) -> anyhow::Result<Self> {
        Ok(Self {
            src: Tensor::from_slice(src, src.len(), dev)?,
            dst: Tensor::from_slice(dst, dst.len(), dev)?,
            n_nodes,
        })
    }
}

// ---------------------------------------------------------------- model

/// GATConv's attention on edges `j → i` for node features `h` (n × C):
/// `softmax_{j into i}(sigmoid(h_j·a_src + h_i·a_dst))`, as an E × 1 column.
pub fn attention(
    h: &Tensor,
    a_src: &Tensor,
    a_dst: &Tensor,
    g: &EdgeIndex,
) -> anyhow::Result<Tensor> {
    let s = h.matmul(a_src)?;
    let d = h.matmul(a_dst)?;
    let logit = (gather_rows(&s, &g.src)? + gather_rows(&d, &g.dst)?)?;
    let sig = (logit.neg()?.exp()? + 1.0)?.recip()?;
    // sigmoid lies in (0, 1): no max to subtract before exp
    let e = sig.exp()?;
    let zeros = Tensor::zeros((g.n_nodes, 1), DType::F32, h.device())?;
    let denom = index_add_rows(&zeros, &g.dst, &e)?;
    Ok((e / gather_rows(&denom, &g.dst)?)?)
}

/// `out_i = Σ_{j → i} alpha_ji h_j`, over the edge list.
pub fn aggregate(alpha: &Tensor, h: &Tensor, g: &EdgeIndex) -> anyhow::Result<Tensor> {
    let msg = gather_rows(h, &g.src)?.broadcast_mul(alpha)?;
    let zeros = Tensor::zeros((g.n_nodes, h.dim(1)?), DType::F32, h.device())?;
    Ok(index_add_rows(&zeros, &g.dst, &msg)?)
}

/// torch's Adam moments for one parameter.
struct Moments {
    m: Tensor,
    v: Tensor,
}

impl Moments {
    fn new(like: &Tensor) -> anyhow::Result<Self> {
        Ok(Self {
            m: like.zeros_like()?,
            v: like.zeros_like()?,
        })
    }

    /// Adam's change at step `t` for gradient `g` (weight decay already in).
    fn delta(&mut self, g: &Tensor, t: usize, lr: f64) -> anyhow::Result<Tensor> {
        self.m = ((&self.m * BETA1)? + (g * (1.0 - BETA1))?)?;
        self.v = ((&self.v * BETA2)? + (g.sqr()? * (1.0 - BETA2))?)?;
        let bc1 = 1.0 - BETA1.powi(t as i32);
        let bc2 = 1.0 - BETA2.powi(t as i32);
        let denom = ((self.v.sqrt()? / bc2.sqrt())? + ADAM_EPS)?;
        Ok(((&self.m / denom)? * (lr / bc1))?)
    }
}

/// The training settings of [`Stagate::fit`].
#[derive(Clone, Debug)]
pub struct TrainOpts {
    pub n_epochs: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub gradient_clipping: f64,
}

/// STAGATE's four tied GATConv layers.
pub struct Stagate {
    /// conv1.lin_src, in × hidden
    pub w1: Var,
    /// conv1.att_src, conv1.att_dst, hidden × 1
    pub a_src: Var,
    pub a_dst: Var,
    /// conv2.lin_src, hidden × latent
    pub w2: Var,
    moments: Vec<Moments>,
    t: usize,
}

fn xavier_normal(
    rows: usize,
    cols: usize,
    fan_sum: usize,
    rng: &mut SmallRng,
    dev: &Device,
) -> anyhow::Result<Var> {
    let sd = INIT_GAIN * (2.0 / fan_sum as f64).sqrt();
    let v: Vec<f32> = (0..rows * cols)
        .map(|_| {
            let z: f64 = StandardNormal.sample(rng);
            (z * sd) as f32
        })
        .collect();
    Ok(Var::from_tensor(&Tensor::from_vec(v, (rows, cols), dev)?)?)
}

/// The forward pass's outputs.
pub struct Forward {
    pub z: Tensor,
    pub recon: Tensor,
}

impl Stagate {
    pub fn new(
        n_in: usize,
        hidden: usize,
        latent: usize,
        seed: u64,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let mut rng = SmallRng::seed_from_u64(seed);
        // xavier on lin_src (in, out): fans in + out; att (1, 1, C): C + C
        let w1 = xavier_normal(n_in, hidden, n_in + hidden, &mut rng, dev)?;
        let a_src = xavier_normal(hidden, 1, 2 * hidden, &mut rng, dev)?;
        let a_dst = xavier_normal(hidden, 1, 2 * hidden, &mut rng, dev)?;
        let w2 = xavier_normal(hidden, latent, hidden + latent, &mut rng, dev)?;
        let moments = [
            w1.as_tensor(),
            a_src.as_tensor(),
            a_dst.as_tensor(),
            w2.as_tensor(),
        ]
        .into_iter()
        .chain([&w2.as_tensor().t()?, &w1.as_tensor().t()?])
        .map(Moments::new)
        .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            w1,
            a_src,
            a_dst,
            w2,
            moments,
            t: 0,
        })
    }

    /// STAGATE.forward with the decoder weights `w3` (latent × hidden) and
    /// `w4` (hidden × in) given.
    pub fn forward_with(
        &self,
        x: &Tensor,
        g: &EdgeIndex,
        w3: &Tensor,
        w4: &Tensor,
    ) -> anyhow::Result<Forward> {
        let xw1 = x.matmul(self.w1.as_tensor())?;
        let alpha = attention(&xw1, self.a_src.as_tensor(), self.a_dst.as_tensor(), g)?;
        let h1 = aggregate(&alpha, &xw1, g)?.elu(1.0)?;
        let z = h1.matmul(self.w2.as_tensor())?;
        let h3 = aggregate(&alpha, &z.matmul(w3)?, g)?.elu(1.0)?;
        let recon = h3.matmul(w4)?;
        Ok(Forward { z, recon })
    }

    /// The forward pass with the decoder tied to the encoder.
    pub fn forward(&self, x: &Tensor, g: &EdgeIndex) -> anyhow::Result<Forward> {
        let w3 = self.w2.as_tensor().t()?.contiguous()?;
        let w4 = self.w1.as_tensor().t()?.contiguous()?;
        self.forward_with(x, g, &w3, &w4)
    }

    /// One training step (Train_STAGATE.py's loop body); the loss before
    /// the step.
    pub fn step(&mut self, x: &Tensor, g: &EdgeIndex, opts: &TrainOpts) -> anyhow::Result<Tensor> {
        // conv3/conv4.lin_src: their own leaves, set to the transposes
        let w3 = Var::from_tensor(&self.w2.as_tensor().t()?.contiguous()?)?;
        let w4 = Var::from_tensor(&self.w1.as_tensor().t()?.contiguous()?)?;
        let out = self.forward_with(x, g, w3.as_tensor(), w4.as_tensor())?;
        // F.mse_loss, the mean over every entry: by rows, then over cells,
        // since one f32 sum over every entry drifts on the CPU
        let loss = (x - &out.recon)?.sqr()?.mean(1)?.mean_all()?;
        let grads = loss.backward()?;
        let params = [
            self.w1.as_tensor(),
            self.a_src.as_tensor(),
            self.a_dst.as_tensor(),
            self.w2.as_tensor(),
            w3.as_tensor(),
            w4.as_tensor(),
        ];
        let gs: Vec<Tensor> = params
            .iter()
            .map(|p| {
                grads
                    .get(p)
                    .map(Tensor::detach)
                    .context("a STAGATE parameter has no gradient")
            })
            .collect::<anyhow::Result<_>>()?;
        // clip_grad_norm_: one coefficient min(1, max / (norm + 1e-6))
        let mut sq = gs[0].sqr()?.sum_all()?;
        for gr in &gs[1..] {
            sq = (sq + gr.sqr()?.sum_all()?)?;
        }
        let coef = ((sq.sqrt()? + 1e-6)?.recip()? * opts.gradient_clipping)?.clamp(0f32, 1f32)?;
        self.t += 1;
        let mut deltas = Vec::with_capacity(6);
        for ((gr, p), m) in gs.iter().zip(params).zip(self.moments.iter_mut()) {
            let gr = (gr.broadcast_mul(&coef)? + (p.detach() * opts.weight_decay)?)?;
            deltas.push(m.delta(&gr, self.t, opts.lr)?);
        }
        // conv3 and conv4 alias conv2 and conv1: their updates land there too
        let w1 = ((self.w1.as_tensor() - &deltas[0])? - deltas[5].t()?)?;
        let w2 = ((self.w2.as_tensor() - &deltas[3])? - deltas[4].t()?)?;
        let a_src = (self.a_src.as_tensor() - &deltas[1])?;
        let a_dst = (self.a_dst.as_tensor() - &deltas[2])?;
        self.w1.set(&w1.contiguous()?)?;
        self.w2.set(&w2.contiguous()?)?;
        self.a_src.set(&a_src)?;
        self.a_dst.set(&a_dst)?;
        Ok(loss.detach())
    }

    /// Train for `opts.n_epochs`; the loss of every epoch.
    pub fn fit(&mut self, x: &Tensor, g: &EdgeIndex, opts: &TrainOpts) -> anyhow::Result<Vec<f32>> {
        let mut losses = Vec::with_capacity(opts.n_epochs);
        let mut pending = Vec::new();
        for epoch in 1..=opts.n_epochs {
            pending.push(self.step(x, g, opts)?);
            if epoch % 100 == 0 || epoch == opts.n_epochs {
                for l in pending.drain(..) {
                    losses.push(l.to_scalar::<f32>()?);
                }
                info!("epoch {epoch}: loss {:.6}", losses[losses.len() - 1]);
            }
        }
        Ok(losses)
    }
}

// ---------------------------------------------------------------- clusters

/// mclust's EEE Gaussian mixture (one full covariance shared by every
/// component) on the rows of `z`, by EM from each of `KMEANS_STARTS` seeded
/// k-means partitions, the run of highest log-likelihood kept; returns the
/// hard classification and the posterior (cells × K).
pub fn eee_gmm(z: &Mat, k: usize, seed: u64) -> anyhow::Result<(Vec<usize>, Mat)> {
    let n = z.nrows();
    anyhow::ensure!(k >= 1 && n >= k, "EEE mixture needs at least {k} cells");
    let x: DMatrix<f64> = z.map(f64::from);
    let mut best: Option<(DMatrix<f64>, f64)> = None;
    for start in 0..KMEANS_STARTS {
        let (_, init) = legume_numeric::matrix::kmeans::kmeans_centroids_seeded(
            z,
            k,
            KMEANS_ROUNDS,
            seed.wrapping_add(start),
        );
        let (post, loglik) = eee_em(&x, &init, k)?;
        if best.as_ref().is_none_or(|(_, b)| loglik > *b) {
            best = Some((post, loglik));
        }
    }
    let (post, loglik) = best.context("no EEE start")?;
    info!("EEE mixture: best of {KMEANS_STARTS} starts, log-likelihood {loglik:.3}");
    let labels: Vec<usize> = (0..n).map(|i| post.row(i).transpose().argmax().0).collect();
    Ok((labels, post.map(|v| v as f32)))
}

/// EM for the EEE mixture from a hard partition `init`, until the
/// log-likelihood changes by at most `MCLUST_TOL` of itself; the posterior
/// and the log-likelihood.
fn eee_em(x: &DMatrix<f64>, init: &[usize], k: usize) -> anyhow::Result<(DMatrix<f64>, f64)> {
    let (n, d) = (x.nrows(), x.ncols());
    let mut post = DMatrix::<f64>::zeros(n, k);
    for (i, &l) in init.iter().enumerate() {
        post[(i, l)] = 1.0;
    }
    let mut loglik_old = f64::NEG_INFINITY;
    for round in 0..EM_MAX_ROUNDS {
        // M step
        let nk: Vec<f64> = (0..k).map(|c| post.column(c).sum()).collect();
        let mu: Vec<DVector<f64>> = (0..k)
            .map(|c| x.tr_mul(&post.column(c)) / nk[c].max(1e-300))
            .collect();
        let mut cov = DMatrix::<f64>::zeros(d, d);
        for (c, m) in mu.iter().enumerate() {
            let mut xc = x.clone();
            for (i, mut row) in xc.row_iter_mut().enumerate() {
                let w = post[(i, c)].sqrt();
                for j in 0..d {
                    row[j] = (row[j] - m[j]) * w;
                }
            }
            cov += xc.tr_mul(&xc);
        }
        cov /= n as f64;
        let chol = match cov.clone().cholesky() {
            Some(c) => c,
            None => {
                let ridge = 1e-6 * cov.diagonal().mean().max(1e-12);
                (cov + DMatrix::identity(d, d) * ridge)
                    .cholesky()
                    .context("EEE covariance is singular")?
            }
        };
        let logdet = 2.0 * chol.l_dirty().diagonal().map(f64::ln).sum();
        let log_pro: Vec<f64> = nk.iter().map(|v| (v / n as f64).max(1e-300).ln()).collect();
        let cst = -0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + logdet);
        // E step
        let rows: Vec<(Vec<f64>, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let xi = x.row(i).transpose();
                let lp: Vec<f64> = (0..k)
                    .map(|c| {
                        let y = chol
                            .l_dirty()
                            .solve_lower_triangular(&(&xi - &mu[c]))
                            .unwrap_or_else(|| DVector::from_element(d, f64::INFINITY));
                        log_pro[c] + cst - 0.5 * y.norm_squared()
                    })
                    .collect();
                let mx = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let s: f64 = lp.iter().map(|v| (v - mx).exp()).sum();
                let lse = mx + s.ln();
                (lp.iter().map(|v| (v - lse).exp()).collect(), lse)
            })
            .collect();
        let mut loglik = 0.0;
        for (i, (p, l)) in rows.into_iter().enumerate() {
            loglik += l;
            for (c, v) in p.into_iter().enumerate() {
                post[(i, c)] = v;
            }
        }
        let err = (loglik - loglik_old).abs() / (1.0 + loglik.abs());
        loglik_old = loglik;
        if err <= MCLUST_TOL {
            log::debug!("EEE EM: {} rounds, log-likelihood {loglik:.3}", round + 1);
            break;
        }
        if round + 1 == EM_MAX_ROUNDS {
            log::warn!(
                "EEE mixture: stopped after {EM_MAX_ROUNDS} rounds (relative change {err:.2e})"
            );
        }
    }
    Ok((post, loglik_old))
}

/// `sc.pp.neighbors` (15 neighbours, the cell counted) then modularity
/// clustering at a fixed `resolution`, with Leiden for Louvain.
fn leiden_at(z: &Mat, resolution: f64, seed: u64) -> anyhow::Result<Vec<usize>> {
    use legume_numeric::matrix::knn_graph::{
        compact_labels, modularity_to_cpm_resolution, run_leiden, KnnGraph, KnnGraphArgs,
    };
    let n = z.nrows();
    anyhow::ensure!(n >= 2, "Leiden needs at least 2 cells");
    let args = KnnGraphArgs {
        knn: SCANPY_NEIGHBOURS - 1,
        block_size: 1000,
        reciprocal: false,
    };
    let (graph, weights) = KnnGraph::from_rows_fuzzy(z, args)?;
    let (network, total) = graph.to_leiden_network_with(&weights);
    let res = modularity_to_cpm_resolution(resolution, total);
    let mut labels = run_leiden(&network, n, res, Some(seed as usize));
    compact_labels(&mut labels);
    Ok(labels)
}

// ---------------------------------------------------------------- run

pub fn run(args: &Args) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.hidden_dims.len() == 2 && args.hidden_dims.iter().all(|&h| h > 0),
        "--hidden-dims takes two positive sizes (hidden, embedding)"
    );
    anyhow::ensure!(args.n_top_genes >= 1, "--n-top-genes must be positive");
    if let Some(k) = args.n_clusters {
        anyhow::ensure!(k >= 1, "--n-clusters must be positive");
    }
    let data = args.input.read()?;
    let n = data.cells.len();

    let (genes, _) = seurat_v3_hvg(&data.counts, args.n_top_genes);
    anyhow::ensure!(!genes.is_empty(), "no genes to train on");
    let mut log = data.counts.clone();
    normalise_log(&mut log);
    let x = dense_rows(&log, &genes);
    drop(log);
    info!(
        "{n} cells, {} highly variable genes (seurat_v3)",
        genes.len()
    );

    let nb = spatial_network(&data, args);
    let (src, dst) = spatial_edges(&nb);
    info!(
        "spatial network: {} edges, {:.4} neighbours per cell, plus self loops",
        src.len() - n,
        (src.len() - n) as f64 / n as f64
    );

    let dev = args.device.to_device(args.device_no)?;
    let g = EdgeIndex::new(&src, &dst, n, &dev)?;
    let xt = Tensor::from_vec(x, (n, genes.len()), &dev)?;
    let mut model = Stagate::new(
        genes.len(),
        args.hidden_dims[0],
        args.hidden_dims[1],
        args.input.seed,
        &dev,
    )?;
    let opts = TrainOpts {
        n_epochs: args.n_epochs,
        lr: args.lr,
        weight_decay: args.weight_decay,
        gradient_clipping: args.gradient_clipping,
    };
    let t0 = std::time::Instant::now();
    let losses = model.fit(&xt, &g, &opts)?;
    info!(
        "trained {} epochs in {:.1?} on {}",
        args.n_epochs,
        t0.elapsed(),
        args.device
    );
    let z = model.forward(&xt, &g)?.z.to_vec2::<f32>()?;
    let emb = Mat::from_fn(n, args.hidden_dims[1], |i, j| z[i][j]);

    let (labels, membership, clustering) = match args.n_clusters {
        Some(k) => {
            let (l, p) = eee_gmm(&emb, k, args.input.seed)?;
            (l, Some(p), "mclust EEE")
        }
        None => (
            leiden_at(&emb, args.resolution, args.input.seed)?,
            None,
            "leiden",
        ),
    };
    let found = labels.iter().max().map_or(0, |&m| m + 1);
    info!("{clustering}: {found} domains");

    let settings = serde_json::json!({
        "reference": "STAGATE_pyG ae1158c (Dong and Zhang 2022)",
        "n_top_genes": args.n_top_genes,
        "n_hvg": genes.len(),
        "hvg_flavor": "seurat_v3",
        "spatial_net": match args.rad_cutoff {
            Some(r) => serde_json::json!({"model": "Radius", "rad_cutoff": r}),
            None => serde_json::json!({"model": "KNN", "k_cutoff": args.k_cutoff}),
        },
        "n_edges_with_self_loops": src.len(),
        "hidden_dims": args.hidden_dims,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "gradient_clipping": args.gradient_clipping,
        "device": args.device.to_string(),
        "final_loss": losses.last(),
        "clustering": clustering,
        "n_clusters": args.n_clusters,
        "resolution": args.n_clusters.is_none().then_some(args.resolution),
        "n_domains": found,
    });
    write_run(
        &args.input,
        &data,
        Fitted {
            method: "stagate",
            labels: &labels,
            membership: membership.as_ref(),
            embedding: &emb,
            embedding_name: "STAGATE",
            settings,
        },
    )?;
    info!("wrote {}", args.input.out);
    Ok(())
}

fn spatial_network(data: &SpatialData, args: &Args) -> Neighbours {
    match args.rad_cutoff {
        Some(r) => radius_within_sections(data, r, RADIUS_MAX_K),
        None => knn_within_sections(data, args.k_cutoff),
    }
}
