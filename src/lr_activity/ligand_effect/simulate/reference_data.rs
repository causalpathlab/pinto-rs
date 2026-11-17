//! Reference data for `pinto lra simulate --reference-data`: the program
//! genes' counts drawn with a real data set's count distributions and
//! gene-gene correlation, around the tissue's own mean counts.
//!
//! data-beans fits the reference data ([`fit_global_copula`]): each gene's
//! count distribution in each cell group as a Poisson mixture, and a
//! low-rank Gaussian copula of its highly variable genes. Each identity bin
//! of the tissue takes the cell group whose mean counts over the program
//! genes correlate best with the bin's (Pearson, on `ln(1 + mean)`). A
//! program gene the reference data has is then drawn per receiver `i` in
//! bin `b` as
//!
//! ```text
//! m[i,g] = sf[i] · μ[b,g] · lift[i] · e^{θ_i}       (the tissue's mean, as without reference data)
//! z[i,g] = L[h(g)] · η[i] + σ[h(g)] · ε[i,g]         (ε[i,g] for a gene outside the copula)
//! y[i,g] = F⁻¹_{k(b),g}(Φ(z[i,g]); counts scaled by m[i,g] / m_ref[k(b),g])
//! ```
//!
//! in place of the Poisson log-normal draw: the mean is the tissue's, the
//! shape around it (extra-Poisson spread, zeros) and the genes'
//! co-variation the reference data's. Both potential outcomes read the one
//! `z[i,g]`, so they differ by the effect alone. A gene the reference data
//! does not have keeps the Poisson log-normal draw.

use crate::simulate::reference_data::{
    content_fingerprint, COPULA_GENE_VARIANCE_FLOOR, COPULA_MIN_DETECTION, COPULA_RANK,
    HIGHLY_VARIABLE_GENES,
};
use crate::util::common::HashMap;
use data_beans::sim::copula::groups::SMALL_LABELS;
use data_beans::sim::copula::mixture::Prepared as CountDistribution;
use data_beans::sim::copula::{fit_global_copula, GlobalCopulaArgs, GroupSpec};

/// Cell groups the reference data is fitted in without labels.
pub const CELL_GROUPS: usize = 8;

/// The reference data, fitted, kept for the program genes alone.
pub struct ReferenceProgramGenes {
    /// Each identity bin's cell group.
    pub cell_group_of_bin: Vec<usize>,
    /// Each cell group's name and share of the reference data's cells.
    pub cell_groups: Vec<(String, f64)>,
    pub n_cells: usize,
    /// A hash of the reference data's content (genes, cells, every gene's
    /// mean count).
    pub content_fingerprint: String,
    /// Each kept gene's position here, by the tissue's name for it.
    position_of_gene: HashMap<Box<str>, usize>,
    /// `mean_count[k][j]`, `count_distribution[k][j]`: kept gene `j`'s mean
    /// count and count distribution in cell group `k`.
    mean_count: Vec<Vec<f64>>,
    count_distribution: Vec<Vec<CountDistribution>>,
    /// Kept gene `j`'s copula loadings and own standard deviation, when it
    /// is in the copula.
    copula_loadings: Vec<Option<(Vec<f64>, f64)>>,
    copula_rank: usize,
}

impl ReferenceProgramGenes {
    /// Fit the reference data at `path` (opened as pinto opens data) in the
    /// cell groups `labels` names, else [`CELL_GROUPS`] k-means groups,
    /// keeping the tissue's program genes `genes` it has, and match each of
    /// the tissue's `n_bins` identity bins to a cell group by
    /// `bin_mean_counts` (bins × `genes`, row-major).
    pub fn fit(
        path: &str,
        labels: Option<&str>,
        genes: &[Box<str>],
        bin_mean_counts: &[f64],
        n_bins: usize,
    ) -> anyhow::Result<Self> {
        let data = data_beans::convert::try_open_or_convert(path)?;
        let n_cells = data.num_columns().unwrap_or(0);
        let fit = fit_global_copula(&GlobalCopulaArgs {
            sc: &data,
            n_hvg: HIGHLY_VARIABLE_GENES,
            copula_rank: COPULA_RANK,
            regularization: COPULA_GENE_VARIANCE_FLOOR,
            hvg_min_detection: COPULA_MIN_DETECTION,
            groups: match labels {
                Some(l) => GroupSpec::Labels(l.into()),
                None => GroupSpec::Cluster(CELL_GROUPS),
            },
        })?;
        // the tissue's program genes the reference data has, by exact name:
        // (the tissue's position, the reference data's)
        let reference_gene_by_name: HashMap<&str, usize> = fit
            .gene_names
            .iter()
            .enumerate()
            .map(|(g, n)| (n.as_ref(), g))
            .collect();
        let kept: Vec<(usize, usize)> = genes
            .iter()
            .enumerate()
            .filter_map(|(t, n)| reference_gene_by_name.get(n.as_ref()).map(|&r| (t, r)))
            .collect();
        anyhow::ensure!(
            !kept.is_empty(),
            "--reference-data {path}: none of the {} program genes is among its {} genes, by \
             name",
            genes.len(),
            fit.n_genes
        );
        let cell_groups_used: Vec<usize> = (0..fit.groups.len())
            .filter(|&k| fit.groups[k].name != SMALL_LABELS)
            .collect();
        anyhow::ensure!(
            !cell_groups_used.is_empty(),
            "--reference-data {path}: no cell group to use"
        );
        let mean_count: Vec<Vec<f64>> = cell_groups_used
            .iter()
            .map(|&k| {
                kept.iter()
                    .map(|&(_, r)| fit.groups[k].marginals[r].mean())
                    .collect()
            })
            .collect();
        let count_distribution = cell_groups_used
            .iter()
            .map(|&k| {
                kept.iter()
                    .map(|&(_, r)| fit.groups[k].marginals[r].prepare())
                    .collect()
            })
            .collect();
        let loadings = &fit.copula.factor;
        let copula_loadings = kept
            .iter()
            .map(|&(_, r)| {
                fit.hvg_pos[r].map(|h| {
                    let h = h as usize;
                    let row = (0..loadings.ncols())
                        .map(|k| f64::from(loadings[(h, k)]))
                        .collect();
                    (row, f64::from(fit.copula.ridge_sd[h]))
                })
            })
            .collect();
        let bin_profiles: Vec<Vec<f64>> = (0..n_bins)
            .map(|b| {
                kept.iter()
                    .map(|&(t, _)| bin_mean_counts[b * genes.len() + t].ln_1p())
                    .collect()
            })
            .collect();
        let cell_group_profiles: Vec<Vec<f64>> = mean_count
            .iter()
            .map(|m| m.iter().map(|x| x.ln_1p()).collect())
            .collect();
        let cell_group_of_bin = bin_profiles
            .iter()
            .map(|profile| most_correlated_profile(profile, &cell_group_profiles))
            .collect();
        let position_of_gene = kept
            .iter()
            .enumerate()
            .map(|(j, &(t, _))| (genes[t].clone(), j))
            .collect();
        Ok(Self {
            cell_group_of_bin,
            cell_groups: cell_groups_used
                .iter()
                .map(|&k| (fit.groups[k].name.clone(), fit.groups[k].share))
                .collect(),
            n_cells,
            content_fingerprint: content_fingerprint(&fit.gene_names, n_cells, &fit.mu_hat),
            position_of_gene,
            mean_count,
            count_distribution,
            copula_loadings,
            copula_rank: loadings.ncols(),
        })
    }

    /// What a record says of it: its size and a fingerprint of its content
    /// (never its path), its cell groups, each identity bin's, and how many
    /// program genes it draws.
    pub fn provenance(&self) -> serde_json::Value {
        serde_json::json!({
            "n_cells": self.n_cells,
            "content_fingerprint": self.content_fingerprint,
            "cell_groups": self.cell_groups.iter()
                .map(|(name, share)| serde_json::json!({"cell_group": name, "share": share}))
                .collect::<Vec<_>>(),
            "cell_group_of_bin": self.cell_group_of_bin.iter()
                .map(|&k| &self.cell_groups[k].0)
                .collect::<Vec<_>>(),
            "program_genes_drawn": self.position_of_gene.len(),
        })
    }

    /// The position here of the tissue gene named `name`, if the reference
    /// data has it.
    pub fn gene_position(&self, name: &str) -> Option<usize> {
        self.position_of_gene.get(name).copied()
    }

    /// The copula's rank: how many factors a receiver draws.
    pub fn copula_rank(&self) -> usize {
        self.copula_rank
    }

    /// Gene `j`'s copula latent at a receiver of copula factors `factors`
    /// and own draw `own`.
    pub fn copula_latent(&self, j: usize, factors: &[f64], own: f64) -> f64 {
        match &self.copula_loadings[j] {
            Some((row, sd)) => row.iter().zip(factors).map(|(a, b)| a * b).sum::<f64>() + sd * own,
            None => own,
        }
    }

    /// Gene `j`'s count at quantile `u` in bin `b`'s cell group, its counts
    /// scaled to mean `mean`; `poisson(u, mean)` where the group never shows
    /// the gene.
    pub fn count_at_quantile(
        &self,
        b: usize,
        j: usize,
        u: f64,
        mean: f64,
        poisson: impl Fn(f64, f64) -> f64,
    ) -> f64 {
        let k = self.cell_group_of_bin[b];
        let group_mean = self.mean_count[k][j];
        if group_mean > 0.0 {
            f64::from(self.count_distribution[k][j].inverse_cdf(u, mean / group_mean))
        } else {
            poisson(u, mean)
        }
    }
}

/// The position of the profile in `candidates` most correlated with
/// `profile` (Pearson; the first on ties, and the first when none varies).
pub fn most_correlated_profile(profile: &[f64], candidates: &[Vec<f64>]) -> usize {
    let correlation = |other: &[f64]| {
        let n = profile.len() as f64;
        let mean_a = profile.iter().sum::<f64>() / n;
        let mean_b = other.iter().sum::<f64>() / n;
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for (a, b) in profile.iter().zip(other) {
            ab += (a - mean_a) * (b - mean_b);
            aa += (a - mean_a).powi(2);
            bb += (b - mean_b).powi(2);
        }
        if aa > 0.0 && bb > 0.0 {
            ab / (aa * bb).sqrt()
        } else {
            f64::NEG_INFINITY
        }
    };
    let mut best = (0, f64::NEG_INFINITY);
    for (k, other) in candidates.iter().enumerate() {
        let r = correlation(other);
        if r > best.1 {
            best = (k, r);
        }
    }
    best.0
}

#[cfg(test)]
#[path = "../tests/simulate_reference_data.rs"]
mod tests;
