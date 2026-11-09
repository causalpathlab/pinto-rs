//! Putting estimates together: Rubin's rules across choices and approximate Bayes
//! factors across thresholds.

use super::contacts::upper_median;
use super::inference::{usable, Estimate};

/// Rubin's rules over analyses: mean estimate, variance = mean within-analysis
/// variance + `(1 + 1/R)` × between-analysis variance. Analyses that were not
/// estimable (see [`super::inference::usable`]) are left out; none left gives
/// NaN. Degrees of freedom by Barnard & Rubin (1999, Biometrika 86:948), with
/// the analyses' smallest df as the complete-data df: the between-analysis
/// spread, estimated from `R` analyses, costs degrees of freedom in proportion
/// to its share of the variance.
pub fn combine(parts: &[Estimate]) -> Estimate {
    let ok: Vec<&Estimate> = parts.iter().filter(|e| usable(e)).collect();
    let Some(first) = ok.first() else {
        let e = parts
            .first()
            .copied()
            .unwrap_or_else(|| Estimate::nan(0, 0));
        // Every analysis in one variance cluster: say so, as each did.
        let one_cluster = !parts.is_empty()
            && parts
                .iter()
                .all(|d| d.df == super::inference::ONE_CLUSTER_DF);
        let nan = Estimate::nan(e.n, e.n_treated);
        let df = if one_cluster { e.df } else { nan.df };
        return Estimate { df, ..nan };
    };
    let r = ok.len() as f64;
    let mean = |f: &dyn Fn(&Estimate) -> f64| ok.iter().map(|e| f(e)).sum::<f64>() / r;
    let theta = mean(&|e| e.theta);
    let within = mean(&|e| e.se * e.se);
    let between = if ok.len() > 1 {
        ok.iter().map(|e| (e.theta - theta).powi(2)).sum::<f64>() / (r - 1.0)
    } else {
        0.0
    };
    let total = within + (1.0 + 1.0 / r) * between;
    let df_com = ok.iter().map(|e| e.df).fold(f64::INFINITY, f64::min);
    let lambda = if total > 0.0 {
        (1.0 + 1.0 / r) * between / total
    } else {
        0.0
    };
    let df_obs = if df_com.is_finite() {
        (df_com + 1.0) / (df_com + 3.0) * df_com * (1.0 - lambda)
    } else {
        f64::INFINITY
    };
    let df = if ok.len() < 2 {
        df_com
    } else if lambda <= 0.0 {
        df_obs
    } else {
        let df_old = (r - 1.0) / (lambda * lambda);
        df_old * df_obs / (df_old + df_obs)
    };
    Estimate {
        theta,
        se: total.sqrt(),
        n: first.n,
        n_treated: first.n_treated,
        mean_h: mean(&|e| e.mean_h),
        ess: mean(&|e| e.ess),
        df,
    }
}

/// What the data say about the exposure threshold, for one estimand: the
/// evidence for an effect at any threshold against none, and the threshold
/// the average leans on. NaN without an estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbfSummary {
    /// log10 of the mean Bayes factor over the thresholds.
    pub log10_bf: f64,
    /// The threshold of the largest weight, and that weight.
    pub top_tau: f64,
    pub top_weight: f64,
}

impl AbfSummary {
    pub const NONE: Self = Self {
        log10_bf: f64::NAN,
        top_tau: f64::NAN,
        top_weight: f64::NAN,
    };
}

/// Each usable estimate's posterior weight under a uniform prior over them,
/// from its approximate Bayes factor against no effect (Wakefield 2009): the
/// estimate as `N(θ, se²)` with the prior `θ ~ N(0, σ²)`, `σ²` being
/// `prior_scale²` times the upper median of the usable estimates' squared
/// standard errors. Scaled so, the weights do not depend on the response's
/// units. An estimate with a zero standard error has no z and takes no weight
/// (a guard: the pipeline's estimates with one variance cluster are already
/// unusable).
pub fn abf_weights(parts: &[Estimate], prior_scale: f64) -> (Vec<(usize, f64)>, f64) {
    let scored: Vec<(usize, f64, f64)> = parts
        .iter()
        .enumerate()
        .filter(|(_, e)| usable(e) && e.se > 0.0)
        .map(|(k, e)| (k, e.se * e.se, e.z()))
        .collect();
    let mut variances: Vec<f64> = scored.iter().map(|x| x.1).collect();
    let Some(median) = upper_median(&mut variances) else {
        return (Vec::new(), f64::NAN);
    };
    let prior_var = prior_scale * prior_scale * median;
    let log_bf: Vec<(usize, f64)> = scored
        .iter()
        .map(|&(k, v, z)| {
            let shrink = prior_var / (v + prior_var);
            (k, 0.5 * (1.0 - shrink).ln() + 0.5 * z * z * shrink)
        })
        .collect();
    let top = log_bf.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    let relative: Vec<f64> = log_bf.iter().map(|x| (x.1 - top).exp()).collect();
    let total: f64 = relative.iter().sum();
    let weights: Vec<(usize, f64)> = log_bf
        .iter()
        .zip(&relative)
        .map(|(&(k, _), r)| (k, r / total))
        .collect();
    let mean_log = top + (total / log_bf.len() as f64).ln();
    (weights, mean_log / std::f64::consts::LN_10)
}

/// The largest of `weights`.
fn heaviest(weights: &[(usize, f64)]) -> (usize, f64) {
    weights
        .iter()
        .copied()
        .fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a })
}

/// The evidence behind [`abf_average`] of `parts`, the estimates at `taus`.
pub fn abf_summary(parts: &[Estimate], taus: &[f64], prior_scale: f64) -> AbfSummary {
    let (weights, log10_bf) = abf_weights(parts, prior_scale);
    if weights.is_empty() {
        return AbfSummary::NONE;
    }
    let (k, top_weight) = heaviest(&weights);
    AbfSummary {
        log10_bf,
        top_tau: taus[k],
        top_weight,
    }
}

/// Estimates of one estimand at several exposure thresholds, averaged by their
/// posterior weights: `Σ w θ`, with the variance `Σ w (se² + (θ − θ̄)²)` of the
/// mixture, so thresholds that disagree widen it. Unusable ones are left out;
/// one usable estimate is itself. The degrees of freedom are the weighted
/// harmonic mean of the estimates'.
///
/// The weights read the estimates they average, so the average leans toward
/// the largest z: its null is not the t reference's, which is why a finding also
/// has to exceed the negative control averaged the same way.
pub fn abf_average(parts: &[Estimate], prior_scale: f64) -> Estimate {
    let (weights, _) = abf_weights(parts, prior_scale);
    if weights.is_empty() {
        return combine(parts);
    }
    if let [(k, _)] = weights[..] {
        return parts[k];
    }
    let sum = |f: &dyn Fn(&Estimate) -> f64| -> f64 {
        weights.iter().map(|&(k, w)| w * f(&parts[k])).sum()
    };
    let theta = sum(&|e| e.theta);
    let var = sum(&|e| e.se * e.se + (e.theta - theta).powi(2));
    let top = parts[heaviest(&weights).0];
    Estimate {
        theta,
        se: var.sqrt(),
        n: top.n,
        n_treated: top.n_treated,
        mean_h: sum(&|e| e.mean_h),
        ess: sum(&|e| e.ess),
        df: 1.0 / sum(&|e| 1.0 / e.df),
    }
}
