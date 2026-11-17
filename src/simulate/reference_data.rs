//! Reference data for `pinto sim new --reference-data`: a real data set's
//! expression in place of the planted genes.
//!
//! data-beans fits the reference data ([`fit_global_copula`]): each gene's
//! count distribution in each of its cell groups as a Poisson mixture
//! (Kiefer and Wolfowitz 1956; fitted by EM as in Laird 1978), each gene's
//! coupling to the cells' library-size coordinate, and a low-rank Gaussian
//! copula of the highly variable genes' latent correlation beyond depth
//! (Sklar 1959; scDesign2, Sun et al. 2021; scDesign3, Song et al. 2023).
//! pinto draws the counts in its own loop, on its keyed streams (`rng.rs`),
//! so the simulator without reference data is untouched and every draw
//! keeps its key.
//!
//! The planted cell states take the reference data's cell groups, largest
//! first (the donor-only state, when there is one, the next; never the pool
//! data-beans fits the labels of fewer than 100 cells in): state `c`'s mean
//! count of gene `g` is its cell group's, `m[c,g]`. A spot's expected count
//! adds the nuisance on the count scale:
//!
//! ```text
//! μ[g,i] = exp(D[g,d] + S[g,s]) · Σ_c w[i,c] · m[c,g] · exp(X[g,c,d])
//! z[g,i] = ρ[c*,g] · zd[i] + √(1 − ρ[c*,g]²) · e[g,i]
//! e[g,i] = L[h(g)] · η[i] + σ[h(g)] · ε[g,i]     (ε[g,i] for a gene outside the copula)
//! y[g,i] = F⁻¹_{c*,g}(Φ(z[g,i]); counts scaled by μ[g,i] / m[c*,g])
//! ```
//!
//! `c*` is the spot's dominant state, `F_{c,g}` its cell group's count
//! distribution of the gene (a Poisson of mean `μ` where the group never
//! shows it), `zd` the spot's depth coordinate, `L` the copula's loadings
//! and `σ` each gene's own part. A spot of one state with no nuisance draws
//! its cell group's counts exactly; depth and the noise beyond the states
//! are the reference data's own. The nuisance is a share of the reference
//! data's own log-count variance between states: `V = T / pve_type`, `T`
//! the variance over the spots of a gene's log mixture mean, averaged over
//! detected genes weighted by their mean count
//! ([`cell_type_log_count_variance`] over [`pve_genes`]), so the realized PVE
//! over those genes (the manifest's `realized_pve_expressed_genes`) meets the
//! targets in expectation.

use super::params::SimParams;
use data_beans::sim::copula::groups::{MIN_GROUP_CELLS, SMALL_LABELS};
use data_beans::sim::copula::mixture::Prepared as CountDistribution;
use data_beans::sim::copula::{fit_global_copula, GlobalCopulaArgs, GroupSpec};
use nalgebra::{DMatrix, DVector};

/// The fit's settings: data-beans' defaults for its own reference-data
/// simulators (`ReferenceFitArgs`).
pub const HIGHLY_VARIABLE_GENES: usize = 2000;
pub const COPULA_RANK: usize = 100;
/// The least share of each copula gene's latent variance kept as its own.
pub const COPULA_GENE_VARIANCE_FLOOR: f32 = 1e-3;
/// The least share of cells a gene must be detected in to join the copula.
pub const COPULA_MIN_DETECTION: f32 = 0.005;

/// Cell groups the reference data is fitted in without labels, at least.
pub const MIN_CELL_GROUPS: usize = 8;

/// The parameters reference data replaces: refused beside it, by name.
pub const PARAMS_REFERENCE_DATA_REPLACES: [&str; 10] = [
    "n_genes",
    "log_mean",
    "log_mean_sd",
    "dispersion",
    "dispersion_sd",
    "depth_sd",
    "patch_cosine",
    "log_rate_var",
    "copula_rank",
    "copula_strength",
];

/// Reference data, fitted, for the cell states it backs.
pub struct ReferenceData {
    pub gene_names: Vec<Box<str>>,
    pub n_cells: usize,
    /// Each cell group's name and share of the reference data's cells.
    pub cell_groups: Vec<(String, f64)>,
    /// Each planted cell state's cell group.
    pub cell_group_of_state: Vec<usize>,
    /// `mean_count_by_state[c][g]`: state `c`'s mean count of gene `g`.
    pub mean_count_by_state: Vec<Vec<f64>>,
    /// `log_mean_count_by_state[c][g]`: its log, the mean floored at the
    /// detection limit of state `c`'s cell group (half a count over its
    /// cells, [`detection_limit`]): a gene its cells never show is below
    /// that, not at `−∞`. The cell-type term of the log mean.
    pub log_mean_count_by_state: Vec<Vec<f64>>,
    /// `count_distribution_by_state[c][g]`: its count distribution.
    pub count_distribution_by_state: Vec<Vec<CountDistribution>>,
    /// `depth_coupling_by_state[c][g]`: gene `g`'s latent correlation with
    /// depth in state `c`'s cell group.
    pub depth_coupling_by_state: Vec<Vec<f64>>,
    /// Gene `g`'s row of the copula, when it is in it.
    pub copula_row_of_gene: Vec<Option<u32>>,
    /// The copula's loadings (copula genes × rank) and each copula gene's
    /// own standard deviation.
    pub copula_loadings: DMatrix<f32>,
    pub copula_gene_sd: DVector<f32>,
    /// Genes detected in at least one cell; the rest are never drawn.
    pub detected: Vec<bool>,
    /// Every gene's mean count over all the cells, and the extra-Poisson
    /// variance of its counts (`cv²` of its mixture's rates: the NB
    /// dispersion of the same variance).
    pub pooled_mean_count: Vec<f64>,
    pub pooled_extra_poisson_variance: Vec<f64>,
    /// The derived log-count variance the nuisance is a share of.
    pub log_count_variance: f64,
    /// A hash of the reference data's content (genes, cells, every gene's
    /// mean count).
    pub content_fingerprint: String,
}

impl ReferenceData {
    /// Fit the reference data at `path` (opened as pinto opens data files)
    /// in the cell groups `labels` names (one per cell, or `cell<TAB>label`
    /// lines), else in `max(MIN_CELL_GROUPS, n states)` k-means groups, for
    /// states holding shares `spot_share_by_state` of the spots, whose
    /// mixtures are `spot_mixtures` (per section, per spot).
    pub fn fit(
        path: &str,
        labels: Option<&str>,
        spot_share_by_state: &[f64],
        spot_mixtures: &[Vec<Vec<f64>>],
        p: &SimParams,
    ) -> anyhow::Result<Self> {
        let n_states = spot_share_by_state.len();
        anyhow::ensure!(
            p.pve_type > 0.0,
            "--reference-data needs pve_type > 0: the nuisance is a share of the reference \
             data's variance between cell states"
        );
        let data = data_beans::convert::try_open_or_convert(path)?;
        let n_cells = data.num_columns().ok_or_else(|| {
            anyhow::anyhow!("--reference-data {path}: its cell count is not stored")
        })?;
        let grouping = match labels {
            Some(l) => GroupSpec::Labels(l.into()),
            None => GroupSpec::Cluster(MIN_CELL_GROUPS.max(n_states)),
        };
        let fit = fit_global_copula(&GlobalCopulaArgs {
            sc: &data,
            n_hvg: HIGHLY_VARIABLE_GENES,
            copula_rank: COPULA_RANK,
            regularization: COPULA_GENE_VARIANCE_FLOOR,
            hvg_min_detection: COPULA_MIN_DETECTION,
            groups: grouping,
        })?;
        // the pooled small labels are no one cell type: never a state's
        let mut largest_first: Vec<usize> = (0..fit.groups.len())
            .filter(|&k| fit.groups[k].name != SMALL_LABELS)
            .collect();
        largest_first.sort_by(|&a, &b| {
            fit.groups[b]
                .share
                .total_cmp(&fit.groups[a].share)
                .then(a.cmp(&b))
        });
        anyhow::ensure!(
            largest_first.len() >= n_states,
            "--reference-data {path}: {} cell group(s) for {n_states} planted cell states \
             (labels of fewer than {MIN_GROUP_CELLS} cells are pooled, and the pool backs no \
             state); give --reference-data-labels with more groups, or a scenario with fewer \
             states",
            largest_first.len()
        );
        let cell_group_of_state: Vec<usize> = largest_first[..n_states].to_vec();
        let floor_by_state: Vec<f64> = cell_group_of_state
            .iter()
            .map(|&k| detection_limit(fit.groups[k].share, n_cells))
            .collect();
        let n_genes = fit.n_genes;
        let mean_count_by_state: Vec<Vec<f64>> = cell_group_of_state
            .iter()
            .map(|&k| fit.groups[k].marginals.iter().map(|m| m.mean()).collect())
            .collect();
        let log_mean_count_by_state: Vec<Vec<f64>> = mean_count_by_state
            .iter()
            .zip(&floor_by_state)
            .map(|(means, &floor)| means.iter().map(|m| m.max(floor).ln()).collect())
            .collect();
        let count_distribution_by_state = cell_group_of_state
            .iter()
            .map(|&k| {
                fit.groups[k]
                    .marginals
                    .iter()
                    .map(|m| m.prepare())
                    .collect()
            })
            .collect();
        let depth_coupling_by_state = cell_group_of_state
            .iter()
            .map(|&k| {
                fit.groups[k]
                    .depth_rho
                    .iter()
                    .map(|&r| f64::from(r))
                    .collect()
            })
            .collect();
        let mut detected = vec![false; n_genes];
        for &g in &fit.active_genes {
            detected[g] = true;
        }
        let pooled_mean_count: Vec<f64> = fit.marginals.iter().map(|m| m.mean()).collect();
        let pooled_extra_poisson_variance = fit.marginals.iter().map(|m| m.rate_cv2()).collect();
        let genes = pve_genes(&log_mean_count_by_state, spot_share_by_state, &detected);
        let log_count_variance =
            cell_type_log_count_variance(&log_mean_count_by_state, spot_mixtures, &genes)
                / p.pve_type;
        let content_fingerprint = content_fingerprint(&fit.gene_names, n_cells, &fit.mu_hat);
        Ok(Self {
            gene_names: fit.gene_names,
            n_cells,
            cell_groups: fit
                .groups
                .iter()
                .map(|g| (g.name.clone(), g.share))
                .collect(),
            cell_group_of_state,
            mean_count_by_state,
            log_mean_count_by_state,
            count_distribution_by_state,
            depth_coupling_by_state,
            copula_row_of_gene: fit.hvg_pos,
            copula_loadings: fit.copula.factor,
            copula_gene_sd: fit.copula.ridge_sd,
            detected,
            pooled_mean_count,
            pooled_extra_poisson_variance,
            log_count_variance,
            content_fingerprint,
        })
    }

    pub fn n_genes(&self) -> usize {
        self.gene_names.len()
    }

    /// Each state's marker genes: the `k` detected genes whose mean count in
    /// that state most exceeds their mean over the other states (log ratio,
    /// counts floored at a thousandth), each gene one state's alone, as the
    /// planted programs are sets of genes.
    pub fn marker_genes_by_state(&self, k: usize) -> Vec<Vec<f64>> {
        let means = &self.mean_count_by_state;
        let n_states = means.len();
        let floor = 1e-3;
        let mut program = vec![vec![0.0; self.n_genes()]; n_states];
        let mut taken = vec![false; self.n_genes()];
        let ranked: Vec<Vec<usize>> = (0..n_states)
            .map(|c| {
                let log_ratio: Vec<f64> = (0..self.n_genes())
                    .map(|g| {
                        let others: f64 = (0..n_states)
                            .filter(|&o| o != c)
                            .map(|o| means[o][g])
                            .sum::<f64>()
                            / (n_states - 1).max(1) as f64;
                        (means[c][g].max(floor) / others.max(floor)).ln()
                    })
                    .collect();
                let mut genes: Vec<usize> = (0..self.n_genes())
                    .filter(|&g| self.detected[g] && means[c][g] > 0.0 && log_ratio[g] > 0.0)
                    .collect();
                genes.sort_by(|&a, &b| log_ratio[b].total_cmp(&log_ratio[a]).then(a.cmp(&b)));
                genes
            })
            .collect();
        // round robin, so no state takes another's best
        for _ in 0..k {
            for (c, genes) in ranked.iter().enumerate() {
                if let Some(g) = genes.iter().copied().find(|&g| !taken[g]) {
                    program[c][g] = 1.0;
                    taken[g] = true;
                }
            }
        }
        program
    }
}

/// The least mean count a cell group of `share` of `n_cells` cells can
/// tell from zero: half a count over its cells.
pub fn detection_limit(share: f64, n_cells: usize) -> f64 {
    0.5 / (share * n_cells as f64).max(1.0)
}

/// Genes the nuisance is scaled on and its realized PVE read over: the
/// [`PVE_GENES`] detected genes of the largest mean count at the states'
/// shares of the spots, each with that mean count as its weight.
pub const PVE_GENES: usize = 2000;

/// The [`PVE_GENES`] genes and their weights, by decreasing weight.
pub fn pve_genes(
    log_mean_count_by_state: &[Vec<f64>],
    spot_share_by_state: &[f64],
    detected: &[bool],
) -> Vec<(usize, f64)> {
    let mut genes: Vec<(usize, f64)> = (0..detected.len())
        .filter(|&g| detected[g])
        .map(|g| {
            let weight: f64 = log_mean_count_by_state
                .iter()
                .zip(spot_share_by_state)
                .map(|(m, share)| share * m[g].exp())
                .sum();
            (g, weight)
        })
        .collect();
    genes.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    genes.truncate(PVE_GENES);
    genes
}

/// `T`: the variance over spots of a gene's log mixture mean count,
/// `ln Σ_c w[i,c] m[c,g]` with each state's mean floored at its detection
/// limit (`log_mean_count_by_state`), averaged over `genes` at their
/// weights ([`pve_genes`]): what the realized PVE measures the cell-type
/// term by, so the nuisance's shares of it are its targets in expectation.
pub fn cell_type_log_count_variance(
    log_mean_count_by_state: &[Vec<f64>],
    spot_mixtures: &[Vec<Vec<f64>>],
    genes: &[(usize, f64)],
) -> f64 {
    use rayon::prelude::*;
    let spots: Vec<&Vec<f64>> = spot_mixtures.iter().flatten().collect();
    let (weighted_variance, total_weight) = genes
        .par_iter()
        .map(|&(g, weight)| {
            let log_means: Vec<f64> = spots
                .iter()
                .map(|mixture| {
                    mixture
                        .iter()
                        .zip(log_mean_count_by_state)
                        .map(|(w, m)| w * m[g].exp())
                        .sum::<f64>()
                        .ln()
                })
                .collect();
            (weight * super::field::mean_sd(&log_means).1.powi(2), weight)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    if total_weight > 0.0 {
        weighted_variance / total_weight
    } else {
        0.0
    }
}

/// FNV-1a over the genes' names, the cell count and every gene's mean count:
/// the data set's identity, which a record can carry without its path.
pub fn content_fingerprint(gene_names: &[Box<str>], n_cells: usize, mean_counts: &[f32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for name in gene_names {
        eat(name.as_bytes());
        eat(&[0]);
    }
    eat(&(n_cells as u64).to_le_bytes());
    for m in mean_counts {
        eat(&m.to_le_bytes());
    }
    format!("{h:016x}")
}

#[cfg(test)]
#[path = "tests/reference_data.rs"]
mod tests;
