//! `pinto banksy`: BANKSY (Singhal, Chou, Lee, Lee, Chou, Wu, Leong, Tan,
//! Lim, Sit, Prabhakar et al. 2024, "BANKSY unifies cell typing and tissue
//! domain segmentation for scalable spatial omics data analysis", Nature
//! Genetics 56:431-441, doi:10.1038/s41588-024-01664-3): each cell's own
//! expression, the weighted mean of its spatial neighbours' and the
//! azimuthal Gabor filter (AGF) of their gradient, z-scored, weighted by
//! `lambda`, then PCA and Leiden on a shared-nearest-neighbour graph of the
//! PCs.
//!
//! Read: the R package <https://github.com/prabhakarlab/Banksy> at commit
//! `70175fbfe10d5ee38efeaba86575bc55965c1ed3` (the reference) and the
//! Python package <https://github.com/prabhakarlab/Banksy_py> at commit
//! `9278996c39e376277d57ef95278000447ba6c57c`. Where the two differ this
//! port follows R, except where noted. Step by step:
//!
//! 1. Normalisation ([`normalise_library`]): each cell's counts scaled to
//!    the mean library size, no log, as the R README's
//!    `computeLibraryFactors` + `normalizeCounts(se, log = FALSE)`. (Python,
//!    `banksy_utils/filter_utils.py:normalize_total`, scales to the median
//!    library size, scanpy's default; also unlogged.)
//! 2. Genes ([`select_genes`]): the `--n-hvg` (2000) most variable genes by
//!    scanpy's "seurat" flavour on a `log1p` copy, as Python's
//!    `filter_utils.py:filter_hvg(n_top_genes=2000, flavor="seurat")`; BANKSY
//!    itself then runs on the unlogged normalised values of those genes.
//!    Deviation: the R multi-sample vignette (`vignettes/multi-sample.Rmd`)
//!    takes the union of Seurat's `vst` HVGs per sample; this port pools the
//!    sections for one "seurat"-flavour list (pinto's [`hvg`]), as Python does.
//! 3. Neighbours ([`banksy_weights`]): per section, the `k_geom` (15)
//!    nearest cells by
//!    Euclidean distance, the cell itself excluded (`dbscan::kNN`), weighted
//!    by the median-scaled Gaussian `exp(-d² / median(d)²)` over the cell's
//!    own neighbour distances, then summed to one
//!    (`R/computation.R:withKNNmedian`, `spatial_mode = "kNN_median"`;
//!    Python's `nbr_weight_decay = "scaled_gaussian"`,
//!    `banksy/main.py:generate_spatial_weights_fixed_nbrs`). Harmonic `m`
//!    uses `k_geom (m + 1)` neighbours, as Python's
//!    `generate_spatial_weights_fixed_nbrs(num_neighbours * (m + 1))` and
//!    the R examples' `computeBanksy(..., M = 1, k_geom = c(15, 30))`; R's
//!    bare default would reuse 15 for every harmonic. The angle of each
//!    neighbour is `atan2(dy, dx)` over the first two coordinates
//!    (`R/computation.R:getPhi`). Within each section: on the `M ≥ 1` path
//!    `R/computation.R:computeBanksy` finds neighbours over all cells'
//!    coordinates with no grouping, and the multi-sample vignettes keep
//!    samples apart by staggering their coordinates first
//!    (`vignettes/batch-correction.Rmd`, "Stagger spatial coordinates");
//!    per-section kNN is equivalent to that, and is what
//!    `R/reduction.R:runBanksyPCA` does with `group` on its lazy path
//!    (which itself accepts only `M = 0`).
//! 4. Neighbourhood matrices ([`neighbour_features`],
//!    `R/computation.R:computeHarmonics`): `H0 = |Σ_j w_ij x_j|`, the
//!    weighted neighbour mean, and for `m ≥ 1` the AGF
//!    `H_m = |Σ_j w_ij e^{i m φ_ij} (x_j − x̄_i)|`, `x̄_i` the unweighted mean
//!    over the same neighbours (`center = TRUE`; Python's
//!    `embed_banksy.py:create_nbr_matrix` since its Sep 2024 "match Banksy
//!    R" edit).
//! 5. Scaling ([`scale_by_section`], `R/computation.R:getBanksyMatrix` with
//!    `scale = TRUE, group = sample`): each feature of each block z-scored
//!    within each section with the sample SD, a flat feature set to 0
//!    (`scaler`). Python z-scores over all cells with the population SD
//!    (`banksy/main.py:zscore`); R's per-sample scaling is the reference's
//!    multi-sample path.
//! 6. Weighting ([`lambda_weights`], `R/computation.R:getLambdas`): block
//!    weights `sqrt(1 − λ)` for own expression and
//!    `sqrt(λ 2^-m / Σ_{m'} 2^-m')` for `H_m`; with `M = 1` and `λ = 0.8`,
//!    squares 0.2, 0.533 and 0.267 (Python's `concatenate_all`, the same).
//!    The default `λ` is 0.8, the domain-segmentation value of the README,
//!    the STARmap vignette (`vignettes/domain-segment.Rmd`) and Python's
//!    domain scripts; 0.2 is cell typing (R's `runBanksyPCA` default), and
//!    the README advises 0.2 for domains on Visium-scale (55-100 um) spots.
//! 7. PCA (`R/reduction.R:runBanksyPCA`, non-lazy path): the concatenated
//!    `[own, H0, H1, ...]`, centred and not rescaled, `npcs = 20` (`--n-pcs`; R and
//!    Python default), by randomized SVD (R uses `irlba`). R drops
//!    zero-SD features first; a zero-SD feature is zero once centred and
//!    does not move the PCs, so it is kept here.
//! 8. Clustering ([`cluster_pcs`], `R/cluster.R:clusterBanksy` with
//!    `algo = "leiden"` and `getGraph`): the 50 (`k_neighbors`, `--knn`) nearest
//!    cells in PC space; each kNN edge `i → j` carries the shared count
//!    `|N(i) ∩ N(j)|`, plus 1 when `i ∈ N(j)` (`dbscan::sNN`,
//!    `src/JP.cpp:SNN_sim_int`, dbscan commit
//!    `61f64a55d5c0563d508ca1caca1f753e45dfe8bb`) and weight
//!    `1 / (1 + d_ij)`; a cell keeps
//!    its 3 edges with most shared neighbours and every edge with 5 or more;
//!    the kept edges become undirected with summed weights
//!    (`igraph::as_undirected`); Leiden with the modularity
//!    (RBConfiguration) objective at resolution 1 (R's default; Python's
//!    `LeidenPartition.partition` uses 0.7). With `--n-clusters`, the
//!    resolution is tuned toward that many clusters instead (pinto's
//!    convention, as `srt-sim-baseline`'s [`cluster`]).
//!
//! Deviations, besides those above: Leiden runs up to 10 rounds of
//! `legume-numeric`'s implementation (R's `n.iterations = -1` runs until
//! stable), and the PCA is randomized SVD, not `irlba`. No batch correction:
//! the original does none by default. The optional label smoothing of the
//! multi-sample vignette (`smoothLabels`) is not run, as it is not part of
//! `clusterBanksy`.
//!
//! [`hvg`]: crate::simulate::baseline::hvg
//! [`cluster`]: crate::simulate::baseline::cluster

use super::graph::{knn_within_sections, Neighbours};
use super::input::{SpatialData, SpatialInputArgs};
use super::output::{write_run, Fitted};
use crate::simulate::baseline::{dense_spots_by_genes, hvg, pca};
use crate::util::common::Mat;
use legume_numeric::matrix::knn::knn_rows;
use legume_numeric::matrix::knn_graph::{
    compact_labels, modularity_to_cpm_resolution, run_leiden, tune_leiden_resolution,
};
use log::info;
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;

pub const LAMBDA: f32 = 0.8;
pub const K_GEOM: usize = 15;
pub const MAX_M: usize = 1;
pub const N_PCS: usize = 20;
pub const N_HVG: usize = 2000;
pub const KNN: usize = 50;
pub const RESOLUTION: f64 = 1.0;
/// `getGraph`: an edge is kept when among a cell's top 3 by shared
/// neighbours, or sharing at least 5.
pub const SNN_MAX_RANK: usize = 3;
pub const SNN_MIN_SHARED: usize = 5;

#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(flatten)]
    pub input: SpatialInputArgs,

    /// Spatial weight lambda in [0, 1]: 0.8 for domain segmentation (the
    /// default), 0.2 for cell typing, 0 for non-spatial clustering; the
    /// BANKSY README advises 0.2 for domains on 55-100 um spot data (Visium)
    #[arg(long, default_value_t = LAMBDA)]
    pub lambda: f32,

    /// Spatial neighbours of the mean neighbourhood (harmonic m uses
    /// k_geom (m + 1))
    #[arg(long = "k-geom", default_value_t = K_GEOM)]
    pub k_geom: usize,

    /// Highest azimuthal Gabor filter harmonic (0: neighbour mean only)
    #[arg(long = "max-m", default_value_t = MAX_M)]
    pub max_m: usize,

    /// Principal components of the BANKSY matrix (the embedding written;
    /// R's npcs)
    #[arg(long, default_value_t = N_PCS)]
    pub n_pcs: usize,

    /// Highly variable genes ("seurat" flavour on log1p; fewer if fewer
    /// genes vary)
    #[arg(long = "n-hvg", default_value_t = N_HVG)]
    pub n_hvg: usize,

    /// Nearest neighbours in PC space for the shared-nearest-neighbour graph
    /// (R's k_neighbors)
    #[arg(long, default_value_t = KNN)]
    pub knn: usize,

    /// Leiden modularity resolution, used when --n-clusters is not given
    #[arg(long, default_value_t = RESOLUTION)]
    pub resolution: f64,

    /// Clusters to tune the Leiden resolution toward (default: none, the
    /// resolution is --resolution as BANKSY's clusterBanksy uses it)
    #[arg(long = "n-clusters")]
    pub n_clusters: Option<usize>,
}

impl Args {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.lambda),
            "--lambda must be in [0, 1]"
        );
        anyhow::ensure!(self.k_geom >= 1, "--k-geom must be at least 1");
        anyhow::ensure!(self.n_pcs >= 1, "--n-pcs must be at least 1");
        anyhow::ensure!(self.n_hvg >= 2, "--n-hvg must be at least 2");
        anyhow::ensure!(self.knn >= 1, "--knn must be at least 1");
        anyhow::ensure!(self.resolution > 0.0, "--resolution must be positive");
        anyhow::ensure!(
            self.n_clusters != Some(0),
            "--n-clusters must be at least 1"
        );
        Ok(())
    }
}

/// Each cell's counts (genes × cells) scaled to the mean library size, as
/// scuttle's `normalizeCounts(log = FALSE)` with library size factors; an
/// empty cell stays empty.
pub fn normalise_library(x: &mut CscMatrix<f32>) {
    let n = x.ncols();
    let lib: Vec<f64> = (0..n)
        .map(|c| x.col(c).values().iter().map(|&v| v as f64).sum())
        .collect();
    let mean = lib.iter().sum::<f64>() / n.max(1) as f64;
    let offsets = x.col_offsets().to_vec();
    let values = x.values_mut();
    for c in 0..n {
        if lib[c] > 0.0 {
            let s = (mean / lib[c]) as f32;
            values[offsets[c]..offsets[c + 1]]
                .iter_mut()
                .for_each(|v| *v *= s);
        }
    }
}

/// The `n_hvg` most variable genes of normalised counts (genes × cells),
/// by the "seurat" flavour on a `log1p` copy, in gene order.
pub fn select_genes(norm: &CscMatrix<f32>, n_hvg: usize) -> Vec<usize> {
    let mut log = norm.clone();
    log.values_mut().iter_mut().for_each(|v| *v = v.ln_1p());
    hvg(&log, n_hvg)
}

/// One harmonic's neighbourhood: each cell's neighbours (global indices)
/// and their weights, summing to one, with `cos(m φ)` and `sin(m φ)` of
/// their angles.
#[derive(Debug, Clone)]
pub struct Harmonic {
    pub m: usize,
    pub index: Vec<Vec<usize>>,
    pub weight: Vec<Vec<f32>>,
    pub cos: Vec<Vec<f32>>,
    pub sin: Vec<Vec<f32>>,
}

/// Median-scaled Gaussian weights `exp(-d² / median(d)²)` of one cell's
/// neighbour distances, summed to one (`withKNNmedian`). R's `median`
/// averages the two middle values of an even count. All-zero distances
/// (coincident cells) weigh evenly.
pub fn median_gaussian_weights(dist: &[f32]) -> Vec<f32> {
    if dist.is_empty() {
        return Vec::new();
    }
    let mut s = dist.to_vec();
    s.sort_by(f32::total_cmp);
    let h = s.len() / 2;
    let med = if s.len() % 2 == 1 {
        s[h]
    } else {
        0.5 * (s[h - 1] + s[h])
    };
    let w: Vec<f32> = if med > 0.0 {
        dist.iter().map(|d| (-(d / med).powi(2)).exp()).collect()
    } else {
        vec![1.0; dist.len()]
    };
    let total: f32 = w.iter().sum();
    w.iter().map(|v| v / total).collect()
}

/// The neighbourhoods of harmonics `0..=max_m` from one spatial kNN of
/// `k_geom (max_m + 1)` neighbours per cell (within sections), harmonic `m`
/// keeping the nearest `k_geom (m + 1)`.
pub fn banksy_weights(data: &SpatialData, k_geom: usize, max_m: usize) -> Vec<Harmonic> {
    let g = knn_within_sections(data, k_geom * (max_m + 1));
    harmonics_of(&g, &data.xy, k_geom, max_m)
}

/// [`banksy_weights`] from neighbours already found.
pub fn harmonics_of(g: &Neighbours, xy: &Mat, k_geom: usize, max_m: usize) -> Vec<Harmonic> {
    (0..=max_m)
        .map(|m| {
            let k = k_geom * (m + 1);
            let n = g.n_cells();
            let mut h = Harmonic {
                m,
                index: Vec::with_capacity(n),
                weight: Vec::with_capacity(n),
                cos: Vec::with_capacity(n),
                sin: Vec::with_capacity(n),
            };
            for i in 0..n {
                let kk = k.min(g.index[i].len());
                let js = &g.index[i][..kk];
                let w = median_gaussian_weights(&g.dist[i][..kk]);
                let phi: Vec<f32> = js
                    .iter()
                    .map(|&j| {
                        let dx = xy[(j, 0)] - xy[(i, 0)];
                        let dy = if xy.ncols() > 1 {
                            xy[(j, 1)] - xy[(i, 1)]
                        } else {
                            0.0
                        };
                        dy.atan2(dx)
                    })
                    .collect();
                h.cos
                    .push(phi.iter().map(|p| (m as f32 * p).cos()).collect());
                h.sin
                    .push(phi.iter().map(|p| (m as f32 * p).sin()).collect());
                h.index.push(js.to_vec());
                h.weight.push(w);
            }
            h
        })
        .collect()
}

/// One gene's harmonic-`m` neighbourhood feature for every cell, from that
/// gene's values `x` over cells: `|Σ_j w_ij x_j|` for `m = 0`, else the
/// centred AGF `|Σ_j w_ij e^{i m φ_ij} (x_j − x̄_i)|` (`computeHarmonics`).
pub fn neighbour_feature(h: &Harmonic, x: &[f32], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let js = &h.index[i];
        if js.is_empty() {
            *o = 0.0;
            continue;
        }
        if h.m == 0 {
            let s: f32 = js.iter().zip(&h.weight[i]).map(|(&j, w)| w * x[j]).sum();
            *o = s.abs();
        } else {
            let mean = js.iter().map(|&j| x[j]).sum::<f32>() / js.len() as f32;
            let (mut re, mut im) = (0f32, 0f32);
            for (k, &j) in js.iter().enumerate() {
                let v = h.weight[i][k] * (x[j] - mean);
                re += v * h.cos[i][k];
                im += v * h.sin[i][k];
            }
            *o = (re * re + im * im).sqrt();
        }
    }
}

/// The unscaled BANKSY matrix, cells × genes (M + 2): own expression `x`
/// (cells × genes) then `H0`, `H1`, ... `H_M`, each a block of `genes`
/// columns. Built one column at a time, so memory is that of the result.
pub fn neighbour_features(x: &Mat, harmonics: &[Harmonic]) -> Mat {
    let (n, g) = (x.nrows(), x.ncols());
    let mut joint = Mat::zeros(n, g * (harmonics.len() + 1));
    let (own, rest) = joint.as_mut_slice().split_at_mut(n * g);
    own.copy_from_slice(x.as_slice());
    rest.par_chunks_mut(n.max(1))
        .enumerate()
        .for_each(|(c, col)| {
            let (b, gene) = (c / g, c % g);
            let xs = &x.as_slice()[gene * n..(gene + 1) * n];
            neighbour_feature(&harmonics[b], xs, col);
        });
    joint
}

/// BANKSY's block weights for own expression and harmonics `0..n_harmonics`
/// (`getLambdas`): `sqrt(1 − λ)`, then `sqrt(λ 2^-m / Σ 2^-m')`.
pub fn lambda_weights(lambda: f32, n_harmonics: usize) -> Vec<f32> {
    let decay: Vec<f64> = (0..n_harmonics).map(|m| 0.5f64.powi(m as i32)).collect();
    let total: f64 = decay.iter().sum();
    let mut out = vec![(1.0 - lambda as f64).max(0.0).sqrt() as f32];
    out.extend(
        decay
            .iter()
            .map(|d| (lambda as f64 * d / total).sqrt() as f32),
    );
    out
}

/// Each column of `joint` z-scored within each section (sample SD; a flat
/// column 0, as `scaler`), then column `c` multiplied by
/// `block_weight[c / block]`.
pub fn scale_by_section(
    joint: &mut Mat,
    sections: &[Vec<usize>],
    block: usize,
    block_weight: &[f32],
) {
    let n = joint.nrows().max(1);
    joint
        .as_mut_slice()
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(c, col)| {
            let lam = block_weight[c / block.max(1)];
            for cells in sections {
                let k = cells.len() as f64;
                if k == 0.0 {
                    continue;
                }
                let mean = cells.iter().map(|&i| col[i] as f64).sum::<f64>() / k;
                let ss: f64 = cells.iter().map(|&i| (col[i] as f64 - mean).powi(2)).sum();
                let sd = (ss / (k - 1.0).max(1.0)).sqrt();
                for &i in cells {
                    col[i] = if sd > 0.0 {
                        (((col[i] as f64 - mean) / sd) as f32) * lam
                    } else {
                        0.0
                    };
                }
            }
        });
}

/// dbscan's `sNN` count for each kNN edge `i → idx[i][e]`
/// (`src/JP.cpp:SNN_sim_int`): `|N(i) ∩ N(j)|`, each point outside its own
/// list, plus 1 when `i ∈ N(j)` (a mutual edge); then `getGraph`'s filter:
/// a cell keeps its top [`SNN_MAX_RANK`] edges by that count (ties in
/// distance order, as `setorder` is stable) and every edge counting
/// [`SNN_MIN_SHARED`] or more.
pub fn snn_kept(idx: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let mut sorted: Vec<Vec<usize>> = idx.to_vec();
    sorted.par_iter_mut().for_each(|v| v.sort_unstable());
    let shared = |a: &[usize], b: &[usize]| -> usize {
        let (mut i, mut j, mut c) = (0, 0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    c += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        c
    };
    (0..idx.len())
        .into_par_iter()
        .map(|i| {
            let s: Vec<usize> = idx[i]
                .iter()
                .map(|&j| {
                    let mutual = sorted[j].binary_search(&i).is_ok();
                    shared(&sorted[i], &sorted[j]) + usize::from(mutual)
                })
                .collect();
            let mut order: Vec<usize> = (0..s.len()).collect();
            // stable: equal counts stay in distance order
            order.sort_by(|&a, &b| s[b].cmp(&s[a]));
            let mut keep = vec![false; s.len()];
            for (rank, &e) in order.iter().enumerate() {
                keep[e] = rank < SNN_MAX_RANK || s[e] >= SNN_MIN_SHARED;
            }
            keep
        })
        .collect()
}

/// R's `getGraph` on the rows of `pcs`: the kNN graph (`k` neighbours,
/// self excluded) with each edge weighted `1 / (1 + d)`, filtered by
/// [`snn_kept`], as undirected edges `(i < j)` with the weights of both
/// directions summed (`igraph::as_undirected`).
pub fn snn_graph(pcs: &Mat, k: usize) -> Vec<(usize, usize, f32)> {
    let (idx, dist) = knn_rows(pcs, k);
    let kept = snn_kept(&idx);
    let mut edges: Vec<(usize, usize, f32)> = (0..idx.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            let (js, ds, keep) = (&idx[i], &dist[i], &kept[i]);
            (0..js.len())
                .filter(move |&e| keep[e])
                .map(move |e| {
                    let j = js[e];
                    (i.min(j), i.max(j), 1.0 / (1.0 + ds[e]))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    edges.par_sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    let mut out: Vec<(usize, usize, f32)> = Vec::with_capacity(edges.len());
    for (i, j, w) in edges {
        match out.last_mut() {
            Some(last) if last.0 == i && last.1 == j => last.2 += w,
            _ => out.push((i, j, w)),
        }
    }
    out
}

/// Leiden with the modularity objective on [`snn_graph`] of `pcs`, at
/// modularity resolution `resolution`, or tuned toward `n_clusters` from
/// there; labels compacted to `0..K`.
pub fn cluster_pcs(
    pcs: &Mat,
    k: usize,
    resolution: f64,
    n_clusters: Option<usize>,
    seed: u64,
) -> anyhow::Result<Vec<usize>> {
    let n = pcs.nrows();
    anyhow::ensure!(n >= 2, "Leiden needs at least 2 cells");
    let edges = snn_graph(pcs, k);
    let mut degree = vec![0f32; n];
    let mut n_edges = vec![0usize; n];
    let mut total = 0f64;
    for &(i, j, w) in &edges {
        degree[i] += w;
        degree[j] += w;
        n_edges[i] += 1;
        n_edges[j] += 1;
        total += w as f64;
    }
    let mut network = legume_numeric::leiden::Network::with_nodes(&degree, &n_edges);
    for &(i, j, w) in &edges {
        network.add_edge(i, j, w);
    }
    info!("sNN graph: {n} cells, {} edges", edges.len());
    let res = modularity_to_cpm_resolution(resolution, total);
    let seed = Some(seed as usize);
    let mut labels = match n_clusters {
        Some(k) => tune_leiden_resolution(&network, n, k, res, seed),
        None => run_leiden(&network, n, res, seed),
    };
    compact_labels(&mut labels);
    Ok(labels)
}

/// The BANKSY PCs (cells × PCs) of the data, with the genes used. The
/// counts are normalised in place and then released: nothing after this
/// reads them.
pub fn banksy_pcs(data: &mut SpatialData, args: &Args) -> anyhow::Result<(Mat, Vec<usize>)> {
    normalise_library(&mut data.counts);
    let genes = select_genes(&data.counts, args.n_hvg);
    anyhow::ensure!(genes.len() >= 2, "fewer than two variable genes");
    let x = dense_spots_by_genes(&data.counts, &genes);
    data.counts = CscMatrix::zeros(0, 0);
    let harmonics = banksy_weights(data, args.k_geom, args.max_m);
    info!(
        "{} genes; harmonics 0..={} with k_geom {}..{}",
        genes.len(),
        args.max_m,
        args.k_geom,
        args.k_geom * (args.max_m + 1)
    );
    let mut joint = neighbour_features(&x, &harmonics);
    drop(x);
    let weights = lambda_weights(args.lambda, harmonics.len());
    scale_by_section(&mut joint, &data.section_cells(), genes.len(), &weights);
    let dims = args.n_pcs.min(joint.nrows()).min(joint.ncols());
    let pcs = pca(joint, dims)?;
    info!("BANKSY matrix to {} PCs", pcs.ncols());
    Ok((pcs, genes))
}

pub fn run(args: &Args) -> anyhow::Result<()> {
    args.validate()?;
    let mut data = args.input.read()?;
    let (pcs, genes) = banksy_pcs(&mut data, args)?;
    let labels = cluster_pcs(
        &pcs,
        args.knn,
        args.resolution,
        args.n_clusters,
        args.input.seed,
    )?;
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    info!("{k} clusters");
    let settings = serde_json::json!({
        "lambda": args.lambda,
        "k_geom": args.k_geom,
        "max_m": args.max_m,
        "spatial_mode": "kNN_median",
        "n_pcs": pcs.ncols(),
        "n_hvg": args.n_hvg,
        "n_genes_used": genes.len(),
        "knn": args.knn,
        "resolution": args.resolution,
        "n_clusters": args.n_clusters,
        "n_clusters_found": k,
        "scale": "per section",
    });
    write_run(
        &args.input,
        &data,
        Fitted {
            method: "banksy",
            labels: &labels,
            membership: None,
            embedding: &pcs,
            embedding_name: "banksy_pc",
            settings,
        },
    )
}
