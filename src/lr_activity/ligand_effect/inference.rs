//! An estimate with its standard error, and the t and normal reference it is
//! read against.

use special::{Beta, Error as _};

/// The `df` of a not-estimable estimate whose rows lie in fewer than two
/// variance clusters, so the verdict can say why.
pub const ONE_CLUSTER_DF: f64 = 0.0;

/// One subgroup's estimate.
#[derive(Clone, Copy, Debug)]
pub struct Estimate {
    pub theta: f64,
    pub se: f64,
    pub n: usize,
    pub n_treated: usize,
    /// Mean overlap weight; 0.25 is perfect balance, near 0 means the
    /// subgroup's exposure was nearly determined by its confounders.
    pub mean_h: f64,
    /// Kish effective sample size of the overlap weights.
    pub ess: f64,
    /// Degrees of freedom of the t reference: clusters carrying the
    /// estimate minus one (after Rubin's rules, Barnard & Rubin's).
    pub df: f64,
}

impl Estimate {
    pub fn nan(n: usize, n_treated: usize) -> Self {
        Self {
            theta: f64::NAN,
            se: f64::NAN,
            n,
            n_treated,
            mean_h: f64::NAN,
            ess: f64::NAN,
            df: f64::NAN,
        }
    }

    pub fn z(&self) -> f64 {
        self.theta / self.se
    }

    /// Two-sided p of `θ = 0` against the t reference; NaN stays NaN.
    pub fn p(&self) -> f64 {
        t_two_sided_p(self.z(), self.df)
    }

    /// The two-sided 5% critical value of the t reference.
    pub fn crit(&self) -> f64 {
        t_crit_975(self.df)
    }

    /// The 95% interval of the t reference.
    pub fn interval(&self) -> (f64, f64) {
        let half = self.crit() * self.se;
        (self.theta - half, self.theta + half)
    }

    /// Whether the 95% interval excludes 0.
    pub fn rejects(&self) -> bool {
        self.z().abs() > self.crit()
    }
}

/// Two-sided 5% normal critical value.
pub const Z_975: f64 = 1.959_964;

/// Two-sided p of `t` with `df` degrees of freedom:
/// `I_{df/(df+t²)}(df/2, 1/2)`; the normal when `df` is infinite, NaN for
/// a NaN `t` or a `df` that is NaN or not positive (Rubin's rules can give
/// a fractional df below 1).
pub fn t_two_sided_p(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if df.is_infinite() {
        return 2.0 * upper_normal_p(t.abs());
    }
    let (a, b) = (0.5 * df, 0.5);
    let x = df / (df + t * t);
    x.inc_beta(a, b, a.ln_beta(b))
}

/// Upper-tail normal p, `P(Z ≥ z)`; NaN stays NaN.
pub fn upper_normal_p(z: f64) -> f64 {
    0.5 * (z / std::f64::consts::SQRT_2).compl_error()
}

/// The upper 2.5% point of t with `df` degrees of freedom; the normal's
/// when `df` is infinite, NaN when `df` is NaN or not positive.
pub fn t_crit_975(df: f64) -> f64 {
    if df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if df.is_infinite() {
        return Z_975;
    }
    let (a, b) = (0.5 * df, 0.5);
    let x = 0.05f64.inv_inc_beta(a, b, a.ln_beta(b));
    (df * (1.0 - x) / x).sqrt()
}

/// `sqrt(Σ_c s_c²)` over the cluster sums `s_c` of ψ, scaled by
/// `C / (C − 1)` for few clusters, and the t reference's degrees of freedom
/// `C − 1` (NaN below two clusters), `C` counting the clusters that carry
/// `ψ`.
pub(super) fn clustered_sd(sums: &[f64]) -> (f64, f64) {
    let used = sums.iter().filter(|s| **s != 0.0).count();
    let ss: f64 = sums.iter().map(|s| s * s).sum();
    let scale = if used > 1 {
        used as f64 / (used as f64 - 1.0)
    } else {
        1.0
    };
    let df = if used > 1 {
        used as f64 - 1.0
    } else {
        f64::NAN
    };
    ((ss * scale).sqrt(), df)
}

/// Whether an analysis's estimate can enter Rubin's rules: a finite estimate
/// and SE and a positive df. A NaN df (fewer than two variance clusters)
/// makes the analysis not estimable; it never reads as infinite df.
pub fn usable(e: &Estimate) -> bool {
    e.theta.is_finite() && e.se.is_finite() && e.df > 0.0
}
