//! The contact dose each receiver gets, and the exposure called from it at a threshold.

use super::averaging::AbfSummary;
use super::options::EffectOpts;
use super::response::niche_ligand;
use super::tissue::Tissue;

/// The exposure side of a pair's per-row quantities: the same under every
/// weighting of its programme.
pub struct Exposure {
    /// The exposure threshold used.
    pub tau: f64,
    pub t: Vec<bool>,
    pub t_ring: Vec<bool>,
    /// `log1p` of the receiver's own ligand / size factor.
    pub own: Vec<f64>,
    /// `log1p` of the mean ligand / size factor over second-order
    /// neighbours.
    pub niche: Vec<f64>,
    /// Ligand / size factor summed over each row's contacts, which the
    /// threshold is read from.
    pub dose: Vec<f64>,
    /// The same over its distant ring. The ring holds as many partners as
    /// there are contacts, so the same τ means the same dose.
    pub ring_dose: Vec<f64>,
}

impl Exposure {
    /// The exposure side from the ligand counts the estimator sees.
    pub fn build(tissue: &Tissue, ligand_counts: &[f64], opts: &EffectOpts) -> Self {
        let (lnorm, dose) =
            super::receivers::contact_dose(&tissue.contacts, &tissue.size_factor, ligand_counts);
        let ring_dose = tissue.ring_sum(&lnorm);
        let tau = choose_threshold(&dose, opts);
        Self {
            tau,
            t: exposed(&dose, tau),
            t_ring: exposed(&ring_dose, tau),
            own: lnorm.iter().map(|v| v.ln_1p()).collect(),
            niche: niche_ligand(tissue, &lnorm),
            dose,
            ring_dose,
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
            t_ring: exposed(&self.ring_dose, tau),
            own: self.own.clone(),
            niche: self.niche.clone(),
            dose: self.dose.clone(),
            ring_dose: self.ring_dose.clone(),
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
