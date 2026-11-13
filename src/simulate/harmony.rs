//! Harmony (Korsunsky et al. 2019, "Fast, sensitive and accurate
//! integration of single-cell data with Harmony", Nature Methods
//! 16:1289-1296, doi:10.1038/s41592-019-0619-0) on a PCA embedding, with
//! one batch variable.
//!
//! Section and equation numbers are those of the paper's Online Methods;
//! iteration counts and tolerances are those of its reference
//! implementations (R `harmony`, `harmonypy`).
//!
//! - Clustering runs on the cells L2-normalised, where the squared distance
//!   to a unit centroid is `2 (1 - cosine)` (section 2.2, eq. 4).
//! - Centroids start from k-means on the normalised cells, the best of 10
//!   k-means++ starts, then L2-normalised (section 2.6.2); the starting `R`
//!   is the soft assignment `exp(-d / sigma)`, each cell's column summed to 1.
//! - Maximum diversity clustering (section 2) minimises eq. 3: the soft
//!   k-means distance, `sigma` times the entropy term `sum R ln R`, and the
//!   diversity penalty `sigma theta sum R ln((O + 1) / (E + 1))` over each
//!   cell's batch, with 1 added to `O` and `E` (section 2.6.3). Each round
//!   sets the centroids to the normalised `R`-weighted sum of the cells
//!   (Dhillon's spherical update, section 2.4.2), then `R` by eq. 8,
//!   `R_ki ∝ exp(-d_ki / sigma) ((E_kb + 1) / (O_kb + 1))^theta`, in blocks of
//!   5% of the cells in random order, `O` and `E` counted without the block
//!   (section 2.6.1). Rounds stop when the sum of the last 3 objectives moves
//!   by less than 1e-5 of the previous 3, or after 20 rounds.
//! - The correction (section 3, eqs. 9 to 13) fits, per cluster `k`, the
//!   ridge mixture of experts `W_k = (Φ* diag(R_k) Φ*ᵀ + Λ)⁻¹ Φ* diag(R_k) Zᵀ`,
//!   `Φ*` the batch one-hot design with an intercept row and `Λ =
//!   diag(0, lambda, ...)`, and subtracts the batch rows only:
//!   `Ẑ = Z - Σ_k W_k[1..]ᵀ Φ diag(R_k)`, always from the original `Z`
//!   (section 1.1). Clustering then resumes on `Ẑ` normalised.
//! - Harmony rounds stop when the objective falls by less than 1e-4 of its
//!   value, or after `max_rounds` rounds.
//!
//! Simplification: theta is not discounted for small batches (section
//! 2.6.4 with tau = 0, the reference implementations' default).

use legume_numeric::matrix::kmeans::kmeans_centroids_seeded;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const MAX_CLUSTER_ROUNDS: usize = 20;
const BLOCK_FRAC: f64 = 0.05;
const EPS_CLUSTER: f64 = 1e-5;
const EPS_HARMONY: f64 = 1e-4;
const WINDOW: usize = 3;
const KMEANS_STARTS: u64 = 10;
const KMEANS_ROUNDS: usize = 25;

const THETA: f64 = 2.0;
const SIGMA: f64 = 0.1;
const LAMBDA: f64 = 1.0;
const MAX_ROUNDS: usize = 10;

#[derive(clap::Args, Debug, Clone)]
pub struct HarmonyParams {
    /// Harmony diversity penalty (>= 0)
    #[arg(long, default_value_t = THETA)]
    pub theta: f64,

    /// Harmony soft-clustering width (> 0)
    #[arg(long, default_value_t = SIGMA)]
    pub sigma: f64,

    /// Harmony ridge penalty (> 0)
    #[arg(long, default_value_t = LAMBDA)]
    pub lambda: f64,

    /// Harmony clusters (default: min(100, N / 30))
    #[arg(long)]
    pub harmony_k: Option<usize>,

    /// Harmony rounds at most
    #[arg(long, default_value_t = MAX_ROUNDS)]
    pub max_rounds: usize,
}

impl Default for HarmonyParams {
    fn default() -> Self {
        Self {
            theta: THETA,
            sigma: SIGMA,
            lambda: LAMBDA,
            harmony_k: None,
            max_rounds: MAX_ROUNDS,
        }
    }
}

impl HarmonyParams {
    /// Settings Harmony can correct with: no ridge penalty leaves every
    /// cluster's system singular, and no width every assignment NaN.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.theta >= 0.0, "--theta must be at least 0");
        anyhow::ensure!(self.sigma > 0.0, "--sigma must be positive");
        anyhow::ensure!(self.lambda > 0.0, "--lambda must be positive");
        anyhow::ensure!(self.max_rounds >= 1, "--max-rounds must be at least 1");
        Ok(())
    }
}

/// Each column scaled to unit length (a zero column stays zero).
fn unit_columns(mut z: DMatrix<f64>) -> DMatrix<f64> {
    for mut col in z.column_iter_mut() {
        let n = col.norm();
        if n > 0.0 {
            col /= n;
        }
    }
    z
}

/// The mixture-of-experts ridge correction (section 3) of `z` (dims ×
/// cells) under soft clusters `r` (clusters × cells), each cell in batch
/// `batch[i] < n_batches`.
pub fn correct(
    z: &DMatrix<f64>,
    r: &DMatrix<f64>,
    batch: &[usize],
    n_batches: usize,
    lambda: f64,
) -> DMatrix<f64> {
    let (d, n) = z.shape();
    let mut out = z.clone();
    for k in 0..r.nrows() {
        // Φ* diag(R_k) Φ*ᵀ + Λ and Φ* diag(R_k) Zᵀ; row 0 is the intercept
        let mut a = DMatrix::<f64>::zeros(n_batches + 1, n_batches + 1);
        let mut rhs = DMatrix::<f64>::zeros(n_batches + 1, d);
        for i in 0..n {
            let (w, j) = (r[(k, i)], batch[i] + 1);
            for (p, q) in [(0, 0), (0, j), (j, 0), (j, j)] {
                a[(p, q)] += w;
            }
            for c in 0..d {
                rhs[(0, c)] += w * z[(c, i)];
                rhs[(j, c)] += w * z[(c, i)];
            }
        }
        for j in 1..=n_batches {
            a[(j, j)] += lambda;
        }
        // singular only for a cluster holding no weight, which moves no cell
        let Some(w) = a.lu().solve(&rhs) else {
            continue;
        };
        for i in 0..n {
            let (rk, j) = (r[(k, i)], batch[i] + 1);
            for c in 0..d {
                out[(c, i)] -= rk * w[(j, c)];
            }
        }
    }
    out
}

/// Centroids (dims × clusters, unit columns) from k-means on the cells of
/// `z_cos` (dims × cells): the best of [`KMEANS_STARTS`] seeds.
fn initial_centroids(z_cos: &DMatrix<f64>, k: usize, seed: u64) -> DMatrix<f64> {
    let rows = z_cos.transpose().map(|v| v as f32);
    let mut best: Option<(f64, DMatrix<f32>)> = None;
    for s in 0..KMEANS_STARTS {
        let (cents, labels) = kmeans_centroids_seeded(&rows, k, KMEANS_ROUNDS, seed + s);
        let inertia: f64 = labels
            .iter()
            .enumerate()
            .map(|(i, &l)| (rows.row(i) - cents.row(l)).norm_squared() as f64)
            .sum();
        if best.as_ref().is_none_or(|(b, _)| inertia < *b) {
            best = Some((inertia, cents));
        }
    }
    let cents = best.map(|(_, c)| c).unwrap_or_else(|| DMatrix::zeros(k, 0));
    unit_columns(cents.transpose().map(|v| v as f64))
}

/// Maximum diversity clustering's state (section 2).
struct Clustering<'a> {
    batch: &'a [usize],
    /// Each batch's share of the cells.
    pr_b: Vec<f64>,
    sigma: f64,
    theta: f64,
    /// Centroids, dims × clusters.
    y: DMatrix<f64>,
    /// Soft assignments, clusters × cells.
    r: DMatrix<f64>,
    /// Observed and expected cells per cluster and batch, clusters × batches.
    o: DMatrix<f64>,
    e: DMatrix<f64>,
    /// Every objective so far, over all Harmony rounds.
    objectives: Vec<f64>,
    rng: SmallRng,
}

impl Clustering<'_> {
    /// Squared distances `2 (1 - yᵀ z)`, clusters × cells.
    fn distances(&self, z_cos: &DMatrix<f64>) -> DMatrix<f64> {
        (self.y.transpose() * z_cos).map(|c| 2.0 * (1.0 - c))
    }

    /// `±` cell `i`'s assignment in `O` and `E`.
    fn tally(&mut self, i: usize, sign: f64) {
        let b = self.batch[i];
        for k in 0..self.r.nrows() {
            let v = sign * self.r[(k, i)];
            self.o[(k, b)] += v;
            for (bb, &p) in self.pr_b.iter().enumerate() {
                self.e[(k, bb)] += v * p;
            }
        }
    }

    /// Eq. 3.
    fn objective(&self, dist: &DMatrix<f64>) -> f64 {
        let mut obj = 0.0;
        for (i, &b) in self.batch.iter().enumerate() {
            for k in 0..self.r.nrows() {
                let r = self.r[(k, i)];
                if r > 0.0 {
                    let penalty = ((self.o[(k, b)] + 1.0) / (self.e[(k, b)] + 1.0)).ln();
                    obj += r * dist[(k, i)] + self.sigma * r * (r.ln() + self.theta * penalty);
                }
            }
        }
        obj
    }

    /// Eq. 8, in shuffled blocks (section 2.6.1).
    fn update_r(&mut self, dist: &DMatrix<f64>) {
        let n = self.batch.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut self.rng);
        let block = ((n as f64 * BLOCK_FRAC).ceil() as usize).max(1);
        for cells in order.chunks(block) {
            cells.iter().for_each(|&i| self.tally(i, -1.0));
            for &i in cells {
                let b = self.batch[i];
                let col = dist.column(i);
                let min = col.min();
                let mut total = 0.0;
                for k in 0..self.r.nrows() {
                    let diversity =
                        ((self.e[(k, b)] + 1.0) / (self.o[(k, b)] + 1.0)).powf(self.theta);
                    let v = (-(col[k] - min) / self.sigma).exp() * diversity;
                    self.r[(k, i)] = v;
                    total += v;
                }
                self.r.column_mut(i).unscale_mut(total);
            }
            cells.iter().for_each(|&i| self.tally(i, 1.0));
        }
    }

    /// Clustering rounds on `z_cos` until the windowed objective settles.
    fn cluster(&mut self, z_cos: &DMatrix<f64>) {
        for round in 0..MAX_CLUSTER_ROUNDS {
            self.y = unit_columns(z_cos * self.r.transpose());
            let dist = self.distances(z_cos);
            self.update_r(&dist);
            self.objectives.push(self.objective(&dist));
            if round > WINDOW && self.settled() {
                break;
            }
        }
    }

    fn settled(&self) -> bool {
        let m = self.objectives.len();
        let old: f64 = self.objectives[m - 1 - WINDOW..m - 1].iter().sum();
        let new: f64 = self.objectives[m - WINDOW..].iter().sum();
        (old - new).abs() / old.abs() < EPS_CLUSTER
    }
}

/// The Harmony-corrected embedding of `x` (cells × PCs), cell `i` in batch
/// `batch[i]`; `seed` drives the k-means starts and the block order.
pub fn harmonize(
    x: &DMatrix<f32>,
    batch: &[usize],
    p: &HarmonyParams,
    seed: u64,
) -> anyhow::Result<DMatrix<f32>> {
    p.validate()?;
    let n = x.nrows();
    anyhow::ensure!(
        batch.len() == n,
        "{} batch labels for {n} cells",
        batch.len()
    );
    let n_batches = batch.iter().max().map_or(0, |&b| b + 1);
    let k = p
        .harmony_k
        .unwrap_or(((n as f64 / 30.0).round() as usize).min(100))
        .clamp(1, n.max(1));
    let z = x.transpose().map(|v| v as f64);
    let mut z_cos = unit_columns(z.clone());
    let mut pr_b = vec![0.0; n_batches];
    batch.iter().for_each(|&b| pr_b[b] += 1.0 / n as f64);

    let mut c = Clustering {
        batch,
        pr_b,
        sigma: p.sigma,
        theta: p.theta,
        y: initial_centroids(&z_cos, k, seed),
        r: DMatrix::zeros(k, n),
        o: DMatrix::zeros(k, n_batches),
        e: DMatrix::zeros(k, n_batches),
        objectives: Vec::new(),
        rng: SmallRng::seed_from_u64(seed),
    };
    let dist = c.distances(&z_cos);
    for i in 0..n {
        let col = dist.column(i);
        let min = col.min();
        let mut r = col.map(|d| (-(d - min) / p.sigma).exp());
        r.unscale_mut(r.sum());
        c.r.set_column(i, &r);
        c.tally(i, 1.0);
    }
    let mut old = c.objective(&dist);
    c.objectives.push(old);

    let mut z_hat = z.clone();
    for _ in 0..p.max_rounds {
        c.cluster(&z_cos);
        z_hat = correct(&z, &c.r, batch, n_batches, p.lambda);
        z_cos = unit_columns(z_hat.clone());
        let new = c.objectives[c.objectives.len() - 1];
        if (old - new) / old.abs() < EPS_HARMONY {
            break;
        }
        old = new;
    }
    Ok(z_hat.transpose().map(|v| v as f32))
}
