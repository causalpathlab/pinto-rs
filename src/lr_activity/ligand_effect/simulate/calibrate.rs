//! What the simulation takes from the real tissue: per-bin levels and who makes
//! the ligand.

use crate::lr_activity::ligand_effect::tissue::Tissue;

/// Floor on a producer's true rate, in expected counts at median depth.
const R_FLOOR: f64 = 0.5;

/// Log-rate lift of the programme's baseline in the marker type (signed
/// per gene, about a doubling).
pub const MARKER_LIFT: f64 = 0.7;

/// Real ingredients of one pair that the simulator keeps.
pub struct SimBase<'a> {
    pub ligand: &'a str,
    pub receptor: &'a str,
    /// The real ligand's counts per row: only its per-bin mean and
    /// detection are used.
    pub ligand_counts: &'a [f64],
    /// Signed weight of each programme gene.
    pub target_weights: &'a [f64],
    /// Each programme gene's mean count / size factor per identity bin,
    /// bins × genes, row-major ([`target_bin_means`]).
    pub target_bin_means: &'a [f64],
    pub competent: &'a [bool],
    /// The real receptor's counts per row (the scarcest subunit's): its
    /// per-bin mean calibrates the simulated receptor, and it is what the
    /// baselines read when the receptor is not simulated.
    pub receptor_counts: &'a [f64],
    /// Counts per row of a real gene outside the benchmarked pairs, for
    /// [`super::ConfounderKind::RealGene`].
    pub confounder_gene: Option<&'a [f64]>,
}

pub(super) fn n_bins(tissue: &Tissue) -> usize {
    tissue
        .identity_bin
        .iter()
        .copied()
        .max()
        .map_or(0, |b| b + 1)
}

/// Mean count / size factor of each gene (non-zeros by row) per identity
/// bin: bins × genes, row-major.
pub fn target_bin_means(tissue: &Tissue, genes: &[&[(u32, f32)]]) -> Vec<f64> {
    let nb = n_bins(tissue);
    let ng = genes.len();
    let mut cells = vec![0usize; nb];
    for &b in &tissue.identity_bin {
        cells[b] += 1;
    }
    let mut out = vec![0.0; nb * ng];
    for (g, nz) in genes.iter().enumerate() {
        for &(r, x) in *nz {
            let r = r as usize;
            out[tissue.identity_bin[r] * ng + g] += x as f64 / tissue.size_factor[r];
        }
    }
    for b in 0..nb {
        for v in &mut out[b * ng..(b + 1) * ng] {
            *v /= cells[b].max(1) as f64;
        }
    }
    out
}

/// One bin's calibrated producers: share `p`, rate among producers `r`.
#[derive(Clone, Copy, Debug)]
pub struct Producers {
    pub p: f64,
    pub r: f64,
    pub cells: usize,
}

/// The zero-inflated Poisson moment fit of the module docs, per bin.
pub fn calibrate_producers(tissue: &Tissue, counts: &[f64]) -> Vec<Producers> {
    let sf = &tissue.size_factor;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_bins(tissue)];
    for (i, &b) in tissue.identity_bin.iter().enumerate() {
        members[b].push(i);
    }
    members
        .iter()
        .map(|rows| {
            let k = rows.len() as f64;
            let none = Producers {
                p: 0.0,
                r: 0.0,
                cells: rows.len(),
            };
            if rows.is_empty() {
                return none;
            }
            let m = rows.iter().map(|&i| counts[i] / sf[i]).sum::<f64>() / k;
            let d = rows.iter().filter(|&&i| counts[i] > 0.0).count() as f64 / k;
            if m <= 0.0 || d <= 0.0 {
                return none;
            }
            let detect = |r: f64| rows.iter().map(|&i| 1.0 - (-sf[i] * r).exp()).sum::<f64>() / k;
            // Predicted minus observed detection at producer rate r, with
            // the share set by the mean; decreasing in r.
            let excess = |r: f64| m / r * detect(r) - d;
            let r = if excess(m) <= 0.0 {
                m
            } else {
                let (mut lo, mut hi) = (m, 2.0 * m);
                while excess(hi) > 0.0 && hi < 1e6 * m.max(1.0) {
                    lo = hi;
                    hi *= 2.0;
                }
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if excess(mid) > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                0.5 * (lo + hi)
            };
            let r = r.max(R_FLOOR);
            Producers {
                p: (m / r).min(1.0),
                r,
                cells: rows.len(),
            }
        })
        .collect()
}

/// The marker type: the top quarter (at least one) of occupied identity
/// bins by mean ligand.
pub fn marker_bins(cal: &[Producers]) -> Vec<bool> {
    let mut occupied: Vec<usize> = (0..cal.len()).filter(|&b| cal[b].cells > 0).collect();
    occupied.sort_by(|&x, &y| (cal[y].p * cal[y].r).total_cmp(&(cal[x].p * cal[x].r)));
    let k = ((occupied.len() as f64 / 4.0).round() as usize).max(1);
    let mut out = vec![false; cal.len()];
    for &b in occupied.iter().take(k) {
        out[b] = true;
    }
    out
}
