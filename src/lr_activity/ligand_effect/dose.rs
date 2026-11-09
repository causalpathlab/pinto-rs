//! The contact dose each receiver gets, the exposure called from it at a
//! threshold, and how much of it neighbour identity explains.

use super::averaging::AbfSummary;
use super::contacts::Contacts;
use super::options::EffectOpts;
use super::tissue::Tissue;

/// The exposure side of a pair's per-row quantities: the same under every
/// weighting of its programme.
pub struct Exposure {
    /// The exposure threshold used.
    pub tau: f64,
    pub t: Vec<bool>,
    /// `log1p` of the receiver's own ligand / size factor.
    pub own: Vec<f64>,
    /// `log1p` of the mean ligand / size factor over second-order
    /// neighbours.
    pub niche: Vec<f64>,
    /// Ligand / size factor summed over each row's contacts, which the
    /// threshold is read from.
    pub dose: Vec<f64>,
}

impl Exposure {
    /// The exposure side from the ligand counts the estimator sees.
    pub fn build(tissue: &Tissue, ligand_counts: &[f64], opts: &EffectOpts) -> Self {
        let (lnorm, dose) = contact_dose(&tissue.contacts, &tissue.size_factor, ligand_counts);
        let tau = choose_threshold(&dose, opts);
        Self {
            tau,
            t: exposed(&dose, tau),
            own: lnorm.iter().map(|v| v.ln_1p()).collect(),
            niche: niche_ligand(tissue, &lnorm),
            dose,
        }
    }

    /// The exposure at the threshold an average leaned on, else the headline's.
    pub fn leaned_on(&self, abf: AbfSummary) -> Vec<bool> {
        if abf.top_tau.is_finite() && abf.top_tau != self.tau {
            exposed(&self.dose, abf.top_tau)
        } else {
            self.t.clone()
        }
    }

    /// The same exposure read at another threshold.
    pub fn at(&self, tau: f64) -> Self {
        Self {
            tau,
            t: exposed(&self.dose, tau),
            own: self.own.clone(),
            niche: self.niche.clone(),
            dose: self.dose.clone(),
        }
    }
}

pub fn exposed(a: &[f64], tau: f64) -> Vec<bool> {
    a.iter().map(|&v| v > tau).collect()
}

/// The exposure threshold for one pair: the requested τ, else 0 (any
/// ligand on a neighbour) when that leaves `min_arm` receivers on each
/// side, else the median exposure. An abundant ligand reaches nearly every
/// receiver, and its contrast is then high against low exposure rather
/// than some against none.
pub fn choose_threshold(a: &[f64], opts: &EffectOpts) -> f64 {
    if let Some(t) = opts.threshold {
        return t;
    }
    let pos = a.iter().filter(|&&v| v > 0.0).count();
    if pos >= opts.min_arm && a.len() - pos >= opts.min_arm {
        return 0.0;
    }
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.total_cmp(y));
    // Strictly-above split: step below the median when ties at the median
    // would leave the exposed arm empty.
    let med = s[s.len() / 2];
    if a.iter().filter(|&&v| v > med).count() >= opts.min_arm {
        med
    } else {
        s.iter().rev().copied().find(|&v| v < med).unwrap_or(0.0)
    }
}

/// η² of `x` by `group` over `rows`: between-group sum of squares over
/// the total. 1 when `x` is a function of the group, near 0 when unrelated
/// (biased up by about `(groups − 1) / n`); NaN when `x` is constant.
pub fn eta_squared(group: &[usize], x: &[f64], rows: &[u32]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    let mean = rows.iter().map(|&i| x[i as usize]).sum::<f64>() / rows.len() as f64;
    let total: f64 = rows.iter().map(|&i| (x[i as usize] - mean).powi(2)).sum();
    if total <= 1e-12 {
        return f64::NAN;
    }
    // Ordered, so the sum is the same in every run.
    let mut by: std::collections::BTreeMap<usize, (f64, usize)> = Default::default();
    for &i in rows {
        let e = by.entry(group[i as usize]).or_default();
        e.0 += x[i as usize];
        e.1 += 1;
    }
    let between: f64 = by
        .values()
        .map(|&(s, c)| c as f64 * (s / c as f64 - mean).powi(2))
        .sum();
    (between / total).min(1.0)
}

/// Each row's ligand over its size factor, and its contact dose: that
/// summed over its contacts, which the exposure threshold reads.
pub fn contact_dose(
    contacts: &Contacts,
    size_factor: &[f64],
    ligand_counts: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let lnorm: Vec<f64> = ligand_counts
        .iter()
        .zip(size_factor)
        .map(|(c, s)| c / s)
        .collect();
    let a = contacts.sum(&lnorm);
    (lnorm, a)
}

/// `log1p` of the mean of `lnorm` (ligand / size factor) over each row's
/// second-order neighbours.
pub fn niche_ligand(tissue: &Tissue, lnorm: &[f64]) -> Vec<f64> {
    tissue
        .second_mean(lnorm)
        .into_iter()
        .map(f64::ln_1p)
        .collect()
}
