//! The threshold-response curve and its simultaneous band.

use super::averaging::combine;
use super::estimate::timing;
use super::inference::Estimate;
use super::options::{BAND_DRAWS, CURVE_QUANTILES};
use super::rows::CurvePoint;

/// The curve's thresholds as `(τ, quantile)`: τ = 0 (any ligand on a
/// neighbour), the [`super::options::CURVE_QUANTILES`] of the positive exposures, and the
/// headline τ; each kept only when it leaves `min_arm` `eligible` rows on
/// either side. `quantile` is the share of positive exposures at or below
/// τ. Strictly increasing in τ.
pub fn curve_taus(a: &[f64], headline: f64, eligible: &[bool], min_arm: usize) -> Vec<(f64, f64)> {
    let mut pos: Vec<f64> = a.iter().copied().filter(|&v| v > 0.0).collect();
    if pos.is_empty() {
        return Vec::new();
    }
    pos.sort_by(f64::total_cmp);
    let at = |q: f64| pos[(q * (pos.len() - 1) as f64).round() as usize];
    let mut cand: Vec<f64> = std::iter::once(0.0)
        .chain(CURVE_QUANTILES.iter().map(|&q| at(q)))
        .chain(std::iter::once(headline))
        .filter(|t| t.is_finite())
        .collect();
    cand.sort_by(f64::total_cmp);
    cand.dedup();
    cand.into_iter()
        .filter(|&tau| {
            let (mut above, mut below) = (0usize, 0usize);
            for (&v, &e) in a.iter().zip(eligible) {
                if e {
                    if v > tau {
                        above += 1;
                    } else {
                        below += 1;
                    }
                }
            }
            above >= min_arm && below >= min_arm
        })
        .map(|tau| {
            let q = pos.partition_point(|&v| v <= tau) as f64 / pos.len() as f64;
            (tau, q)
        })
        .collect()
}

/// The sup-t critical value of a simultaneous 95% band over `K` estimates,
/// by a Gaussian multiplier bootstrap over clusters.
///
/// Each row of `stacked` is one cluster's sums of ψ for the `K` estimates
/// (one row per cluster and analysis, analyses being the competence and weighting choices
/// the estimates average over, so `ΣᵀΣ` is their summed covariance). A
/// bootstrap draw multiplies each row by an independent standard normal,
/// `T_k = Σ_r g_r S_rk / √V_kk` with `V_kk = Σ_r S_rk²`, and the critical
/// value is the 95% quantile of `max_k |T_k|` over [`super::options::BAND_DRAWS`] draws.
/// Columns with `V_kk = 0` are left out; NaN when none is left.
pub fn sup_t_crit(stacked: &[Vec<f64>], k: usize, seed: u64) -> f64 {
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut v = vec![0.0f64; k];
    for row in stacked {
        for (vk, x) in v.iter_mut().zip(row) {
            *vk += x * x;
        }
    }
    let keep: Vec<usize> = (0..k).filter(|&j| v[j] > 0.0).collect();
    if keep.is_empty() {
        return f64::NAN;
    }
    let sd: Vec<f64> = keep.iter().map(|&j| v[j].sqrt()).collect();
    let rows: Vec<&Vec<f64>> = stacked
        .iter()
        .filter(|r| keep.iter().any(|&j| r[j] != 0.0))
        .collect();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut t = vec![0.0f64; keep.len()];
    let mut sup: Vec<f64> = (0..BAND_DRAWS)
        .map(|_| {
            t.fill(0.0);
            for row in &rows {
                let g: f64 = StandardNormal.sample(&mut rng);
                for (tj, &j) in t.iter_mut().zip(&keep) {
                    *tj += g * row[j];
                }
            }
            t.iter()
                .zip(&sd)
                .map(|(x, s)| (x / s).abs())
                .fold(0.0, f64::max)
        })
        .collect();
    sup.sort_by(f64::total_cmp);
    sup[((0.95 * BAND_DRAWS as f64).ceil() as usize).min(BAND_DRAWS) - 1]
}

/// The curve from its points' combined estimates and the stacked cluster
/// sums. The bootstrap's critical value is a normal-reference one; with few
/// clusters each point's is scaled by its t reference over the normal's,
/// `c_k = c · t_{df_k} / z`, and never below the pointwise `t_{df_k}`.
pub fn build_curve(
    taus: &[(f64, f64)],
    est: &[Estimate],
    stacked: &[Vec<f64>],
    seed: u64,
) -> Vec<CurvePoint> {
    let ok: Vec<bool> = est
        .iter()
        .map(|e| e.theta.is_finite() && e.se.is_finite())
        .collect();
    let masked: Vec<Vec<f64>> = stacked
        .iter()
        .map(|r| {
            r.iter()
                .zip(&ok)
                .map(|(&x, &o)| if o { x } else { 0.0 })
                .collect()
        })
        .collect();
    let crit = sup_t_crit(&masked, est.len(), seed);
    taus.iter()
        .zip(est)
        .map(|(&(tau, q), e)| {
            let c = crit * e.crit() / super::inference::Z_975;
            CurvePoint::new(tau, q, *e, c)
        })
        .collect()
}

/// The curve at `taus` from the fitted `points`, in the order of its thresholds:
/// one row of stacked sums per variance cluster that carries any.
pub(super) fn threshold_curve(
    n_clusters: usize,
    taus: &[(f64, f64)],
    points: &[(f64, Estimate, Vec<f64>)],
    seed: u64,
) -> Vec<CurvePoint> {
    if taus.is_empty() {
        return Vec::new();
    }
    let point = |tau: f64| points.iter().find(|p| p.0 == tau);
    let estimates: Vec<Estimate> = taus
        .iter()
        .map(|t| point(t.0).map_or_else(|| combine(&[]), |p| p.1))
        .collect();
    let mut stacked: Vec<Vec<f64>> = Vec::new();
    for c in 0..n_clusters {
        let row: Vec<f64> = taus
            .iter()
            .map(|t| point(t.0).and_then(|p| p.2.get(c).copied()).unwrap_or(0.0))
            .collect();
        if row.iter().any(|&x| x != 0.0) {
            stacked.push(row);
        }
    }
    timing::time(timing::BAND, || {
        build_curve(taus, &estimates, &stacked, seed)
    })
}
