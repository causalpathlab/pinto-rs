//! `pinto srt-sim-baseline`: standard single-cell pipelines run on a
//! simulation, as baselines for `srt-sim-score`.
//!
//! The shared pipeline is the scanpy recipe (Wolf, Angerer and Theis 2018,
//! "SCANPY: large-scale single-cell gene expression data analysis", Genome
//! Biology 19:15, doi:10.1186/s13059-017-1382-0): [`normalise_log`], [`hvg`],
//! [`scale`], [`pca`] and [`cluster`]. `pooled` runs it on all sections
//! together, with no batch handling; `harmony` corrects the PCs for the
//! donor with [`harmonize`] before clustering them the same way;
//! `per-donor` runs it on each donor alone and carries the labels across
//! with [`match_to_reference`].

use super::harmony::{harmonize, HarmonyParams};
use super::score::TruthSpots;
use crate::util::common::HashMap;
use crate::util::metadata::Run;
use legume_numeric::matrix::pca::pc_scores;
use legume_numeric::matrix::traits::{MatElemOps, MatOps};
use log::info;
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

/// Each spot's counts scaled to 1e4 in total, then `ln(1 + x)`
/// (genes × spots; an empty spot stays empty).
pub fn normalise_log(x: &mut CscMatrix<f32>) {
    x.sum_to_one_columns_inplace();
    x.values_mut().iter_mut().for_each(|v| *v *= 1e4);
    x.log1p_inplace();
}

/// Per gene of log data (genes × spots), on the counts scale (`expm1`):
/// `ln(1 + mean)` and `ln(variance / mean)`, the variance unbiased; NaN
/// where the gene has no variance, as scanpy's "seurat" flavour. The
/// variance is taken about the mean (two passes), so a flat gene at a high
/// level has none rather than a rounding error's worth.
pub fn gene_mean_dispersion(x: &CscMatrix<f32>) -> (Vec<f64>, Vec<f64>) {
    let (n_genes, n) = (x.nrows(), x.ncols() as f64);
    let mut sum = vec![0f64; n_genes];
    let mut nnz = vec![0f64; n_genes];
    for (g, _, &v) in x.triplet_iter() {
        sum[g] += (v as f64).exp_m1();
        nnz[g] += 1.0;
    }
    let mean: Vec<f64> = sum.iter().map(|s| s / n).collect();
    // the spots a gene is absent from each add mean²
    let mut ss: Vec<f64> = (0..n_genes)
        .map(|g| (n - nnz[g]) * mean[g] * mean[g])
        .collect();
    for (g, _, &v) in x.triplet_iter() {
        ss[g] += ((v as f64).exp_m1() - mean[g]).powi(2);
    }
    (0..n_genes)
        .map(|g| {
            let mean = mean[g];
            let var = ss[g] / (n - 1.0).max(1.0);
            let disp = if var > 1e-12 * mean.max(1.0).powi(2) && mean > 0.0 {
                (var / mean).ln()
            } else {
                f64::NAN
            };
            (mean.ln_1p(), disp)
        })
        .unzip()
}

/// Each gene's dispersion as a z-score among the genes in its bin of
/// `n_bins` equal-width bins of mean; a gene alone in its bin scores 1, as
/// in Satija et al. 2015 ("Spatial reconstruction of single-cell gene
/// expression data", Nature Biotechnology 33:495, doi:10.1038/nbt.3192)
/// and scanpy's "seurat" flavour. NaN dispersions stay NaN and count for
/// no bin.
pub fn normalised_dispersion(mean: &[f64], disp: &[f64], n_bins: usize) -> Vec<f64> {
    let (lo, hi) = mean
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &m| {
            (a.min(m), b.max(m))
        });
    let width = (hi - lo) / n_bins as f64;
    let bin = |m: f64| {
        if width > 0.0 {
            (((m - lo) / width) as usize).min(n_bins - 1)
        } else {
            0
        }
    };
    let mut members: Vec<Vec<f64>> = vec![Vec::new(); n_bins];
    for (&m, &d) in mean.iter().zip(disp).filter(|(_, d)| !d.is_nan()) {
        members[bin(m)].push(d);
    }
    let stats: Vec<(f64, f64)> = members
        .iter()
        .map(|v| {
            let k = v.len() as f64;
            let mu = v.iter().sum::<f64>() / k;
            let sd = (v.iter().map(|d| (d - mu).powi(2)).sum::<f64>() / (k - 1.0)).sqrt();
            (mu, sd)
        })
        .collect();
    mean.iter()
        .zip(disp)
        .map(|(&m, &d)| {
            let b = bin(m);
            let (mu, sd) = stats[b];
            if d.is_nan() {
                f64::NAN
            } else if members[b].len() == 1 {
                1.0
            } else if sd > 0.0 {
                (d - mu) / sd
            } else {
                0.0
            }
        })
        .collect()
}

/// The `n_top` most variable genes of log data (genes × spots) by
/// normalised dispersion over 20 mean bins (scanpy's default), in gene
/// order. Genes with no dispersion are never picked.
pub fn hvg(x: &CscMatrix<f32>, n_top: usize) -> Vec<usize> {
    let (mean, disp) = gene_mean_dispersion(x);
    let z = normalised_dispersion(&mean, &disp, 20);
    let mut order: Vec<usize> = (0..z.len()).filter(|&g| !z[g].is_nan()).collect();
    order.sort_by(|&a, &b| z[b].total_cmp(&z[a]).then(a.cmp(&b)));
    order.truncate(n_top);
    order.sort_unstable();
    order
}

/// The genes `genes` of log data (genes × spots) as a dense spots × genes
/// matrix.
pub fn dense_spots_by_genes(x: &CscMatrix<f32>, genes: &[usize]) -> DMatrix<f32> {
    let mut at = vec![None; x.nrows()];
    for (j, &g) in genes.iter().enumerate() {
        at[g] = Some(j);
    }
    // genes × spots first, so each spot's writes are contiguous
    let mut d = DMatrix::zeros(genes.len(), x.ncols());
    for (g, s, &v) in x.triplet_iter() {
        if let Some(j) = at[g] {
            d[(j, s)] = v;
        }
    }
    d.transpose()
}

/// Each column (gene) of spots × genes to zero mean and unit SD (unbiased;
/// a flat gene stays zero), clipped to [-10, 10], as scanpy's `pp.scale`
/// with `max_value=10`.
pub fn scale(x: &mut DMatrix<f32>) {
    use rayon::prelude::*;
    let rows = x.nrows().max(1);
    let n = x.nrows() as f32;
    // column-major: each chunk is one gene's column
    x.as_mut_slice().par_chunks_mut(rows).for_each(|col| {
        let mean = col.iter().sum::<f32>() / n;
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
        let sd = if var > 0.0 { var.sqrt() } else { 1.0 };
        col.iter_mut()
            .for_each(|v| *v = ((*v - mean) / sd).clamp(-10.0, 10.0));
    });
}

/// The first `n_pcs` principal component scores of spots × genes
/// (centred first), by randomized SVD (Halko, Martinsson and Tropp 2011,
/// "Finding structure with randomness", SIAM Review 53:217,
/// doi:10.1137/090771806).
pub fn pca(mut x: DMatrix<f32>, n_pcs: usize) -> anyhow::Result<DMatrix<f32>> {
    x.centre_columns_inplace();
    pc_scores(&x, n_pcs, 0)
}

/// Leiden (Traag, Waltman and van Eck 2019, "From Louvain to Leiden:
/// guaranteeing well-connected communities", Scientific Reports 9:5233,
/// doi:10.1038/s41598-019-41721-w) on the Euclidean kNN graph of the PC
/// scores with UMAP's fuzzy edge weights (McInnes, Healy and Melville 2018,
/// arXiv:1802.03426), as scanpy's `pp.neighbors` and `tl.leiden`. `knn`
/// counts the spot itself, as scanpy's `n_neighbors` does. The modularity
/// resolution starts at scanpy's 1 and is tuned toward `k` clusters. The
/// scores go in as they are: `leiden_clustering`'s Euclidean path z-scores
/// the PCs first, which scanpy does not.
pub fn cluster(pcs: &DMatrix<f32>, knn: usize, k: usize, seed: u64) -> anyhow::Result<Vec<usize>> {
    use legume_numeric::matrix::knn_graph::{
        compact_labels, modularity_to_cpm_resolution, tune_leiden_resolution, KnnGraph,
        KnnGraphArgs,
    };
    let n = pcs.nrows();
    anyhow::ensure!(n >= 2, "Leiden needs at least 2 spots");
    let args = KnnGraphArgs {
        knn: knn.saturating_sub(1).max(1),
        block_size: 1000,
        reciprocal: false,
    };
    let (graph, weights) = KnnGraph::from_rows_fuzzy(pcs, args)?;
    let (network, total) = graph.to_leiden_network_with(&weights);
    let res = modularity_to_cpm_resolution(1.0, total);
    let mut labels = tune_leiden_resolution(&network, n, k, res, Some(seed as usize));
    compact_labels(&mut labels);
    Ok(labels)
}

/// The shared pipeline on log data (genes × spots, from [`normalise_log`]),
/// up to the PC scores (spots × PCs); also the HVGs it used.
pub fn scanpy_pcs(
    log: &CscMatrix<f32>,
    args: &BaselineArgs,
) -> anyhow::Result<(DMatrix<f32>, Vec<usize>)> {
    let genes = hvg(log, args.n_hvg);
    anyhow::ensure!(genes.len() >= 2, "fewer than two variable genes");
    let mut x = dense_spots_by_genes(log, &genes);
    scale(&mut x);
    let pcs = pca(x, args.n_pcs)?;
    info!(
        "{} spots, {} HVGs, {} PCs",
        pcs.nrows(),
        genes.len(),
        pcs.ncols()
    );
    Ok((pcs, genes))
}

/// Marker-correlation matching to a reference donor, the usual way
/// annotation is carried from one donor to the next: each cluster takes the
/// label of the reference cluster whose mean profile it correlates with
/// most, as scmap-cluster projects onto reference centroids (Kiselev, Yiu
/// and Hemberg 2018, "scmap: projection of single-cell RNA-seq data across
/// data sets", Nature Methods 15:359, doi:10.1038/nmeth.4644). `means[d]` holds donor
/// `d`'s cluster means (clusters × genes) of log data over the same genes.
/// Each donor's rows are first centred by that donor's own mean profile
/// (the mean of its cluster means), so a donor-wide shift does not decide
/// the match; each cluster then takes the index of the reference cluster
/// whose centred profile it correlates with most (Pearson), and the
/// reference's clusters keep their own. A donor with one cluster has no
/// profile left once centred, so it is matched on its mean as it is. Two
/// clusters can map to one label,
/// and a cluster found in one donor only can never get a label of its own:
/// that is the failure this baseline shows.
pub fn match_to_reference(means: &[DMatrix<f32>], reference: usize) -> Vec<Vec<usize>> {
    let centred = |m: &DMatrix<f32>| -> DMatrix<f32> {
        let mean = m.row_mean();
        DMatrix::from_fn(m.nrows(), m.ncols(), |c, g| m[(c, g)] - mean[g])
    };
    let pearson = |a: &[f32], b: &[f32]| -> f32 {
        let (ma, mb) = (mean_of(a), mean_of(b));
        let (mut ab, mut aa, mut bb) = (0f32, 0f32, 0f32);
        for (x, y) in a.iter().zip(b) {
            ab += (x - ma) * (y - mb);
            aa += (x - ma).powi(2);
            bb += (y - mb).powi(2);
        }
        ab / (aa * bb).sqrt().max(f32::MIN_POSITIVE)
    };
    let rows = |m: &DMatrix<f32>| -> Vec<Vec<f32>> {
        m.row_iter().map(|r| r.iter().copied().collect()).collect()
    };
    let best = |r: &[f32], refs: &[Vec<f32>]| -> usize {
        refs.iter()
            .map(|q| pearson(r, q))
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |(i, _)| i)
    };
    let (refs, refs_raw) = (rows(&centred(&means[reference])), rows(&means[reference]));
    means
        .iter()
        .enumerate()
        .map(|(d, m)| match (d == reference, m.nrows()) {
            (true, n) => (0..n).collect(),
            (false, 1) => vec![best(&rows(m)[0], &refs_raw)],
            (false, _) => rows(&centred(m)).iter().map(|r| best(r, &refs)).collect(),
        })
        .collect()
}

fn mean_of(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

/// The columns `cols` of `x`, in that order.
fn select_columns(x: &CscMatrix<f32>, cols: &[usize]) -> CscMatrix<f32> {
    let (mut offsets, mut rows, mut values) = (vec![0], Vec::new(), Vec::new());
    for &c in cols {
        let col = x.col(c);
        rows.extend_from_slice(col.row_indices());
        values.extend_from_slice(col.values());
        offsets.push(rows.len());
    }
    CscMatrix::try_from_csc_data(x.nrows(), cols.len(), offsets, rows, values)
        .expect("columns of a valid CSC matrix")
}

/// Each cluster's mean of log data (genes × spots) over `genes`, as
/// clusters × genes: spot `spots[i]` is in cluster `labels[i]`. Summed from
/// the sparse columns.
fn cluster_means(
    log: &CscMatrix<f32>,
    genes: &[usize],
    spots: &[usize],
    labels: &[usize],
) -> DMatrix<f32> {
    let mut at = vec![None; log.nrows()];
    for (j, &g) in genes.iter().enumerate() {
        at[g] = Some(j);
    }
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    let mut sum = DMatrix::zeros(k, genes.len());
    let mut n = vec![0f32; k];
    for (&s, &l) in spots.iter().zip(labels) {
        let col = log.col(s);
        for (&g, &v) in col.row_indices().iter().zip(col.values()) {
            if let Some(j) = at[g] {
                sum[(l, j)] += v;
            }
        }
        n[l] += 1.0;
    }
    for (l, &nl) in n.iter().enumerate() {
        sum.row_mut(l).unscale_mut(nl.max(1.0));
    }
    sum
}

/// The donor with the most spots; ties to the first, so equal donors do not
/// silently make the last one (often a simulation's patch owner) the
/// reference.
pub fn reference_donor(sizes: &[usize]) -> usize {
    let most = sizes.iter().copied().max().unwrap_or(0);
    sizes.iter().position(|&n| n == most).unwrap_or(0)
}

/// The number of distinct domains among each donor's spots: the true K of
/// a donor alone, as the pooled run is given the true K of all of them.
pub fn domains_per_donor(domain: &[usize], donor: &[usize]) -> Vec<usize> {
    let n_donors = donor.iter().max().map_or(0, |&m| m + 1);
    let mut held = vec![std::collections::BTreeSet::new(); n_donors];
    for (&k, &d) in domain.iter().zip(donor) {
        held[d].insert(k);
    }
    held.iter().map(|h| h.len()).collect()
}

/// The scanpy recipe on each donor alone (its own HVGs, PCs, graph and
/// Leiden toward `k[d]`), then [`match_to_reference`] over the union of the
/// donors' HVGs, the reference being [`reference_donor`].
fn per_donor_labels(
    log: &CscMatrix<f32>,
    donors: &[usize],
    k: &[usize],
    args: &BaselineArgs,
) -> anyhow::Result<Vec<usize>> {
    let mut of_donor = vec![Vec::new(); k.len()];
    for (s, &d) in donors.iter().enumerate() {
        of_donor[d].push(s);
    }
    let (ids, spots): (Vec<usize>, Vec<Vec<usize>>) = of_donor
        .into_iter()
        .enumerate()
        .filter(|(_, v)| !v.is_empty())
        .unzip();
    let mut union = Vec::new();
    let mut fits = Vec::new();
    for (&d, s) in ids.iter().zip(&spots) {
        let (pcs, genes) = scanpy_pcs(&select_columns(log, s), args)?;
        union.extend(genes);
        fits.push(cluster(&pcs, args.knn, k[d], args.seed)?);
    }
    union.sort_unstable();
    union.dedup();
    let means: Vec<DMatrix<f32>> = spots
        .iter()
        .zip(&fits)
        .map(|(s, l)| cluster_means(log, &union, s, l))
        .collect();
    let reference = reference_donor(&spots.iter().map(Vec::len).collect::<Vec<_>>());
    info!(
        "per donor: reference {}, {} genes to match on",
        super::panel::donor_label(ids[reference]),
        union.len()
    );
    let matched = match_to_reference(&means, reference);
    let mut labels = vec![0; donors.len()];
    for ((s, l), m) in spots.iter().zip(&fits).zip(&matched) {
        for (&spot, &c) in s.iter().zip(l) {
            labels[spot] = m[c];
        }
    }
    Ok(labels)
}

pub const N_PCS: usize = 30;
pub const N_HVG: usize = 2000;
pub const KNN: usize = 15;
pub const SEED: u64 = 1;

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// All sections together, no batch handling
    Pooled,
    /// Harmony on the PCs, batch = donor
    Harmony,
    /// Each donor alone, matched to the largest donor's clusters
    PerDonor,
}

#[derive(clap::Args, Debug)]
pub struct BaselineArgs {
    /// The simulation's output directory (as given to srt-simulate -o)
    #[arg(long)]
    pub truth: Box<str>,

    /// Baseline to run
    #[arg(long, value_enum, default_value = "pooled")]
    pub method: Method,

    /// Clusters to tune Leiden toward (default: the simulation's domains; per
    /// donor, the domains that donor holds)
    #[arg(long)]
    pub k: Option<usize>,

    /// Principal components
    #[arg(long, default_value_t = N_PCS)]
    pub n_pcs: usize,

    /// Highly variable genes
    #[arg(long, default_value_t = N_HVG)]
    pub n_hvg: usize,

    /// Neighbours in the kNN graph, the spot itself included (scanpy's
    /// n_neighbors)
    #[arg(long, default_value_t = KNN)]
    pub knn: usize,

    /// Leiden and Harmony seed
    #[arg(long, default_value_t = SEED)]
    pub seed: u64,

    #[command(flatten)]
    pub harmony: HarmonyParams,

    /// Labels as a cell<TAB>label table
    #[arg(short, long)]
    pub out: Box<str>,
}

/// The simulation's counts (genes × spots) and cell names, as pinto reads
/// the truth manifest's data files.
pub(crate) fn read_counts(truth: &str) -> anyhow::Result<(CscMatrix<f32>, Vec<Box<str>>)> {
    let run = Run::open(&format!("{truth}/truth/truth"))?;
    let files = run
        .meta
        .data_files
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("{}: no data files", run.manifest.display()))?
        .iter()
        .map(|f| {
            run.resolve(f)
                .map(|p| p.to_string_lossy().into())
                .ok_or_else(|| anyhow::anyhow!("data file {f} not found"))
        })
        .collect::<anyhow::Result<Vec<Box<str>>>>()?;
    let data = crate::util::input::read_expr_data(&files)?;
    let cells = data.column_names()?;
    Ok((data.read_columns_csc(0..cells.len())?, cells))
}

/// Each data column's row in the truth table, matched by cell name.
fn truth_rows(truth: &TruthSpots, cells: &[Box<str>]) -> anyhow::Result<Vec<usize>> {
    let of_cell: HashMap<&str, usize> = truth
        .cells
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_ref(), i))
        .collect();
    cells
        .iter()
        .map(|c| {
            of_cell
                .get(c.as_ref())
                .copied()
                .ok_or_else(|| anyhow::anyhow!("spot {c} not in the truth table"))
        })
        .collect()
}

impl BaselineArgs {
    /// Settings the pipeline can run with, checked before any data is read.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.k != Some(0), "--k must be at least 1");
        anyhow::ensure!(self.n_pcs >= 1, "--n-pcs must be at least 1");
        anyhow::ensure!(self.n_hvg >= 2, "--n-hvg must be at least 2");
        anyhow::ensure!(
            self.knn >= 2,
            "--knn must be at least 2 (it counts the spot itself)"
        );
        self.harmony.validate()
    }
}

pub fn run_baseline(args: &BaselineArgs) -> anyhow::Result<()> {
    args.validate()?;
    let truth = TruthSpots::read(&args.truth)?;
    let (mut log, cells) = read_counts(&args.truth)?;
    normalise_log(&mut log);
    let rows = truth_rows(&truth, &cells)?;
    let donors: Vec<usize> = rows.iter().map(|&i| truth.donor[i]).collect();
    let domains: Vec<usize> = rows.iter().map(|&i| truth.domain[i]).collect();
    // the true K: the domains among the spots clustered together
    let k = |of: &[usize]| -> Vec<usize> {
        domains_per_donor(&domains, of)
            .into_iter()
            .map(|h| args.k.unwrap_or(h))
            .collect()
    };
    let labels = match args.method {
        Method::Pooled | Method::Harmony => {
            let k = k(&vec![0; domains.len()])[0];
            info!("Leiden toward {k} clusters");
            let mut pcs = scanpy_pcs(&log, args)?.0;
            if args.method == Method::Harmony {
                pcs = harmonize(&pcs, &donors, &args.harmony, args.seed)?;
            }
            cluster(&pcs, args.knn, k, args.seed)?
        }
        Method::PerDonor => {
            let k = k(&donors);
            info!("Leiden toward {k:?} clusters, by donor");
            per_donor_labels(&log, &donors, &k, args)?
        }
    };
    let text: String = std::iter::once("cell\tlabel\n".to_string())
        .chain(
            cells
                .iter()
                .zip(&labels)
                .map(|(c, l)| format!("{c}\tC{}\n", l + 1)),
        )
        .collect();
    std::fs::write(args.out.as_ref(), text)?;
    Ok(())
}
