//! Who touches whom and receptor competence.
//!
//! - [`Contacts`]: each cell's spatial contacts, from the run's adjacent
//!   (not expression-similar) pairs.
//! - [`Contacts::sum`] / [`Contacts::mean`]: a receiver's exposure is the
//!   sum of a per-cell quantity over its contacts; neighbour-average
//!   confounders use the mean.
//! - [`median_contact_length`]: the unit for tiles and smooths, by
//!   [`upper_median`].
//! - [`competence`]: whether a receiver's cell type expresses the receptor,
//!   read from its identity bin rather than its own count, which the
//!   ligand itself could change.

use crate::util::common::*;

/// Each row's contact partners (row indices), symmetric and deduplicated.
pub struct Contacts {
    pub adj: Vec<Vec<u32>>,
}

impl Contacts {
    pub fn from_pairs(n: usize, pairs: &[(usize, usize)]) -> Self {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(i, j) in pairs {
            if i != j {
                adj[i].push(j as u32);
                adj[j].push(i as u32);
            }
        }
        for a in &mut adj {
            a.sort_unstable();
            a.dedup();
        }
        Self { adj }
    }

    pub fn degree(&self, i: usize) -> usize {
        self.adj[i].len()
    }

    pub fn sum(&self, v: &[f64]) -> Vec<f64> {
        self.adj
            .par_iter()
            .map(|a| a.iter().map(|&j| v[j as usize]).sum())
            .collect()
    }

    pub fn mean(&self, v: &[f64]) -> Vec<f64> {
        self.adj
            .par_iter()
            .map(|a| {
                if a.is_empty() {
                    0.0
                } else {
                    a.iter().map(|&j| v[j as usize]).sum::<f64>() / a.len() as f64
                }
            })
            .collect()
    }
}

/// Median contact length per section, the unit for tiles and smooths.
/// Sections without contacts get the overall median.
pub fn median_contact_length(
    coords: &[(f32, f32)],
    section: &[u32],
    n_sections: usize,
    contacts: &Contacts,
) -> Vec<f64> {
    let mut per: Vec<Vec<f64>> = vec![Vec::new(); n_sections];
    for (i, a) in contacts.adj.iter().enumerate() {
        for &j in a {
            let j = j as usize;
            if i < j {
                let (dx, dy) = (
                    (coords[i].0 - coords[j].0) as f64,
                    (coords[i].1 - coords[j].1) as f64,
                );
                per[section[i] as usize].push((dx * dx + dy * dy).sqrt());
            }
        }
    }
    let mut all: Vec<f64> = per.iter().flatten().copied().collect();
    let overall = upper_median(&mut all).unwrap_or(1.0).max(1e-9);
    per.iter_mut()
        .map(|v| upper_median(v).unwrap_or(overall).max(1e-9))
        .collect()
}

/// The `q` quantile of `sorted` (ascending), the element at
/// `round(q (len − 1))`; `None` when empty.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    sorted.get((q * last as f64).round() as usize).copied()
}

/// The element at `len / 2` of `v` sorted (the upper median), found by
/// selection; `None` when empty. Reorders `v`.
pub fn upper_median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mid = v.len() / 2;
    Some(*v.select_nth_unstable_by(mid, f64::total_cmp).1)
}

/// `true` where the share of cells in a row's identity bin detecting the
/// receptor is at least `threshold`.
pub fn competence(bin: &[usize], detected: &[bool], threshold: f64) -> Vec<bool> {
    competence_grid(bin, detected, &[threshold])
        .pop()
        .unwrap_or_default()
}

/// [`competence`] at each threshold, the bins' detection shares counted
/// once.
pub fn competence_grid(bin: &[usize], detected: &[bool], thresholds: &[f64]) -> Vec<Vec<bool>> {
    let nb = bin.iter().copied().max().map_or(0, |b| b + 1);
    let mut hit = vec![0usize; nb];
    let mut tot = vec![0usize; nb];
    for (&b, &d) in bin.iter().zip(detected) {
        tot[b] += 1;
        hit[b] += d as usize;
    }
    let share: Vec<Option<f64>> = (0..nb)
        .map(|b| (tot[b] > 0).then(|| hit[b] as f64 / tot[b] as f64))
        .collect();
    thresholds
        .iter()
        .map(|&c| {
            let on: Vec<bool> = share.iter().map(|s| s.is_some_and(|s| s >= c)).collect();
            bin.iter().map(|&b| on[b]).collect()
        })
        .collect()
}
