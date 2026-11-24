//! Efron's two-group model and local false discovery rate.
//!
//! Many statistics `z_i`, most from a null, a few from signal:
//! `f(z) = π₀ f₀(z) + (1 − π₀) f₁(z)`, and the local fdr of `z` is
//! `lfdr(z) = π₀ f₀(z) / f(z)`, the posterior probability that a statistic
//! there is null. Nothing is thresholded: the lfdr is reported, and the
//! reader chooses.
//!
//! - **Null** `f₀ = N(μ₀, σ₀²)`: either the theoretical `N(0, 1)`, or
//!   fitted to the bulk of the statistics (median and MAD), Efron's
//!   empirical null. The empirical null absorbs what moves every statistic
//!   alike (dependence between tests, a residual confounder widening every
//!   z) instead of reading it as signal.
//! - **Null share** `π₀`: the share of statistics within one null sd of the
//!   null mean, over the 68.27% a normal puts there, at most 1.
//! - **Mixture density** `f`: a Gaussian kernel estimate (Silverman's
//!   bandwidth) binned on a grid.
//! - **One-sided**: signal lies above the null mean; at or below it the
//!   lfdr is 1.
//! - **Monotone**: above the null mean the lfdr never increases outward, so
//!   a statistic never looks more null than a less extreme one.
//! - **Infinite** statistics (a p that underflowed, an SE of 0) are signal
//!   whatever the density: lfdr 0 at +∞, 1 at −∞. NaN stays NaN.
//!
//! References: B. Efron, R. Tibshirani, J. D. Storey and V. Tusher,
//! "Empirical Bayes analysis of a microarray experiment", JASA 96 (2001);
//! B. Efron, "Large-scale simultaneous hypothesis testing: the choice of a
//! null hypothesis", JASA 99 (2004); B. Efron, "Correlation and large-scale
//! simultaneous significance testing", JASA 102 (2007); B. W. Silverman,
//! "Density Estimation for Statistics and Data Analysis" (1986).

use crate::util::common::{quantile_sorted, upper_median};

/// A normal null `N(mean, sd²)`.
#[derive(Clone, Copy, Debug)]
pub struct NormalNull {
    pub mean: f64,
    pub sd: f64,
}

impl NormalNull {
    /// The theoretical null of a z statistic.
    pub const THEORETICAL: Self = Self { mean: 0.0, sd: 1.0 };

    /// Efron's empirical null fitted to the bulk: the median and the MAD
    /// (scaled to a normal's sd) of the finite `z`; `None` with fewer than
    /// three or no spread.
    pub fn empirical(z: &[f64]) -> Option<Self> {
        let mut v: Vec<f64> = z.iter().copied().filter(|x| x.is_finite()).collect();
        if v.len() < 3 {
            return None;
        }
        let mean = upper_median(&mut v)?;
        let mut dev: Vec<f64> = v.iter().map(|x| (x - mean).abs()).collect();
        let sd = 1.4826 * upper_median(&mut dev)?;
        (sd > 0.0).then_some(Self { mean, sd })
    }

    fn density(&self, z: f64) -> f64 {
        std_normal_pdf((z - self.mean) / self.sd) / self.sd
    }
}

/// The two-group model fitted to a set of statistics.
#[derive(Clone, Debug)]
pub struct TwoGroups {
    /// `π₀`.
    pub null_share: f64,
    /// Per statistic, in the order given; NaN where the statistic is NaN.
    pub lfdr: Vec<f64>,
}

/// The standard normal density.
fn std_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// The one-sided two-group model on `z` under `null`, signal above its mean,
/// fitted to the finite statistics (at least three: callers fit the
/// empirical null first, which needs as many; with fewer, every lfdr is NaN).
pub fn two_groups(z: &[f64], null: NormalNull) -> TwoGroups {
    let mut lfdr: Vec<f64> = z
        .iter()
        .map(|&v| match v {
            f64::INFINITY => 0.0,
            f64::NEG_INFINITY => 1.0,
            _ => f64::NAN,
        })
        .collect();
    let finite: Vec<usize> = (0..z.len()).filter(|&i| z[i].is_finite()).collect();
    let n = finite.len();
    if n < 3 {
        return TwoGroups {
            null_share: f64::NAN,
            lfdr,
        };
    }
    let t: Vec<f64> = finite.iter().map(|&i| z[i]).collect();
    // the bulk within one null sd holds 68.27% of the null
    let bulk = t
        .iter()
        .filter(|&&v| (v - null.mean).abs() < null.sd)
        .count() as f64
        / n as f64;
    let null_share = (bulk / 0.682_689_492).min(1.0);
    let f = mixture_density(&t, null);
    // At or below the null mean, null; above it, walking outward, the lfdr
    // is the smallest seen so far, so no statistic is more null than a less
    // extreme one.
    let (mut above, below): (Vec<usize>, Vec<usize>) = (0..n).partition(|&k| t[k] > null.mean);
    for k in below {
        lfdr[finite[k]] = 1.0;
    }
    above.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
    let mut run = f64::INFINITY;
    for k in above {
        let raw = null_share * null.density(t[k]) / f[k].max(f64::MIN_POSITIVE);
        run = run.min(raw.clamp(0.0, 1.0));
        lfdr[finite[k]] = run;
    }
    TwoGroups { null_share, lfdr }
}

/// The kernel density estimate of `t` (at least three, finite) at each of
/// them: a Gaussian kernel with Silverman's bandwidth, binned on a grid and
/// convolved once.
///
/// The grid spans the data, but no further than 40 null sds from the null
/// mean, where the null density has vanished (a statistic beyond it is
/// signal whatever the mixture density, and is binned at the edge); its step
/// is at most h/4 unless that takes more than `GRID_MAX` points, and the
/// kernel is normalized on the grid, so the estimate stays a density however
/// coarse the step.
fn mixture_density(t: &[f64], null: NormalNull) -> Vec<f64> {
    const GRID_MAX: usize = 1 << 16;
    let n = t.len();
    let mut s = t.to_vec();
    s.sort_by(f64::total_cmp);
    let at = |q: f64| quantile_sorted(&s, q).unwrap_or(0.0);
    let mean = t.iter().sum::<f64>() / n as f64;
    let sd = (t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let iqr = (at(0.75) - at(0.25)) / 1.34;
    let spread = if iqr > 0.0 { sd.min(iqr) } else { sd }.max(f64::EPSILON);
    let h = 0.9 * spread * (n as f64).powf(-0.2);
    let lo = s[0].max(null.mean - 40.0 * null.sd) - 4.0 * h;
    let hi = s[n - 1].min(null.mean + 40.0 * null.sd) + 4.0 * h;
    let grid = (((hi - lo) / (h / 4.0)).ceil() as usize + 1).clamp(2, GRID_MAX);
    let step = (hi - lo) / (grid - 1) as f64;
    let bins: Vec<usize> = t
        .iter()
        .map(|&v| (((v - lo) / step).round().max(0.0) as usize).min(grid - 1))
        .collect();
    let mut hist = vec![0f64; grid];
    for &b in &bins {
        hist[b] += 1.0;
    }
    let reach = ((4.0 * h) / step).ceil() as usize;
    let shape: Vec<f64> = (0..=reach)
        .map(|d| std_normal_pdf(d as f64 * step / h))
        .collect();
    let mass = step * (2.0 * shape.iter().sum::<f64>() - shape[0]);
    let kernel: Vec<f64> = shape.iter().map(|k| k / (n as f64 * mass)).collect();
    let density: Vec<f64> = (0..grid)
        .map(|x| {
            let (a, b) = (x.saturating_sub(reach), (x + reach).min(grid - 1));
            (a..=b).map(|y| hist[y] * kernel[x.abs_diff(y)]).sum()
        })
        .collect();
    bins.iter().map(|&b| density[b]).collect()
}
