//! Who touches whom, the distant ring and receptor competence.
//!
//! - [`Contacts`]: each cell's spatial contacts, from the run's adjacent
//!   (not expression-similar) pairs.
//! - [`Contacts::sum`] / [`Contacts::mean`]: a receiver's exposure is the
//!   sum of a per-cell quantity over its contacts; neighbour-average
//!   confounders use the mean.
//! - [`median_contact_length`]: the unit for tiles, smooths and the ring, by
//!   [`upper_median`].
//! - [`ring_partners`]: the negative-control exposure. For each receiver,
//!   as many cells as it has contacts, drawn from a ring well beyond
//!   contact range. They share the receiver's region but cannot signal to
//!   it by contact.
//! - [`competence`]: whether a receiver's cell type expresses the receptor,
//!   read from its identity bin rather than its own count, which the
//!   ligand itself could change.

use crate::util::common::*;
use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

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

/// Median contact length per section, the unit for tiles, smooths and the
/// ring. Sections without contacts get the overall median.
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

/// The element at `len / 2` of `v` sorted (the upper median), found by
/// selection; `None` when empty. Reorders `v`.
pub fn upper_median(v: &mut [f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mid = v.len() / 2;
    Some(*v.select_nth_unstable_by(mid, f64::total_cmp).1)
}

/// For each row, `degree(i)` partners at distance in `[inner, outer]` (in
/// that section's contact lengths), drawn without replacement, seeded per
/// row. Fewer when the ring holds fewer cells.
pub fn ring_partners(
    coords: &[(f32, f32)],
    section: &[u32],
    unit: &[f64],
    contacts: &Contacts,
    inner: f64,
    outer: f64,
    seed: u64,
) -> Vec<Vec<u32>> {
    let n = coords.len();
    // Grid of cell side `outer` per section: the ring of any point lies in
    // its own and the eight surrounding grid cells.
    let mut grid: HashMap<(u32, i64, i64), Vec<u32>> = HashMap::default();
    let key = |i: usize| {
        let s = outer * unit[section[i] as usize];
        (
            section[i],
            (coords[i].0 as f64 / s).floor() as i64,
            (coords[i].1 as f64 / s).floor() as i64,
        )
    };
    for i in 0..n {
        grid.entry(key(i)).or_default().push(i as u32);
    }
    (0..n)
        .into_par_iter()
        .map(|i| {
            let want = contacts.degree(i);
            if want == 0 {
                return Vec::new();
            }
            let u = unit[section[i] as usize];
            let (lo, hi) = ((inner * u).powi(2), (outer * u).powi(2));
            let (b, gx, gy) = key(i);
            let mut cand: Vec<u32> = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(cell) = grid.get(&(b, gx + dx, gy + dy)) {
                        for &j in cell {
                            let j_ = j as usize;
                            let (ex, ey) = (
                                (coords[i].0 - coords[j_].0) as f64,
                                (coords[i].1 - coords[j_].1) as f64,
                            );
                            let d2 = ex * ex + ey * ey;
                            if d2 >= lo && d2 <= hi {
                                cand.push(j);
                            }
                        }
                    }
                }
            }
            let mut rng = SmallRng::seed_from_u64(mix_seed(seed, i as u64));
            cand.shuffle(&mut rng);
            cand.truncate(want);
            cand
        })
        .collect()
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
