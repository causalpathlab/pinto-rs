//! Augmented overlap-weighted effect of a binary neighbourhood exposure.
//!
//! The estimand for a treatment `T` (here: a receiver's contact neighbours
//! carry ligand above a threshold), confounders `X` and outcome `Y` is the
//! overlap-weighted average treatment effect
//!
//! ```text
//! θ = E[h(X) (Y(1) − Y(0))] / E[h(X)],   h(x) = e(x)(1 − e(x)),  e(x) = P(T=1 | X=x)
//! ```
//!
//! (Li, Morgan & Zaslavsky 2018, JASA 113:390). It is the numerically
//! stable member of the weighted-ATE family: the weight is bounded by 1/4
//! and nothing is divided by a propensity, so a receiver whose exposure was
//! almost certain, or almost impossible, contributes little instead of
//! exploding the estimate. Its target population is the receivers whose
//! exposure could plausibly have gone either way.
//!
//! It is estimated in the augmented, doubly robust form (Mao, Li & Greene
//! 2019, Stat Methods Med Res 28:3221):
//!
//! ```text
//! θ̂ = Σ [h(m₁ − m₀) + T(1 − e)(Y − m₁) − (1 − T)e(Y − m₀)] / Σ h
//! ```
//!
//! with `m_t(x) = E[Y | X=x, T=t]`, and every nuisance cross-fitted: each
//! receiver's `e`, `m₀`, `m₁` come from models fitted on the other folds.
//! Folds are spatial tiles, so a receiver is never predicted by a model that
//! saw its own neighbourhood. Standard errors come from the influence
//! function summed within clusters (tiles), since receivers sharing a tile
//! share senders and are not independent. With a few dozen tiles the
//! cluster-robust variance is itself noisy, so tests and intervals use a
//! t reference with `C − 1` degrees of freedom over the `C` clusters that
//! carry the subgroup (the usual small-cluster reference for a
//! cluster-robust mean; Cameron & Miller 2015, J Hum Resour 50:317),
//! rather than the normal.
//!
//! The nuisance models are deliberately plain: a logistic ridge for `e` and
//! a linear ridge for `m`, both on a shared confounder matrix. Double
//! robustness means either one being right is enough for consistency;
//! plainness is what lets one Gram matrix per fold serve every
//! ligand-receptor pair. Confounders read off the response enter `m` alone
//! ([`PairInputs::outcome_only`]), so one fitted `e` ([`propensity`])
//! serves every fit with the same exposure side ([`overlap_terms_with`]).

use super::regression::{
    fold_products, logistic_ridge, ridge_bordered, sigmoid, standardize, subsample,
};
use nalgebra::DMatrix;
use rayon::prelude::*;
use special::{Beta, Error as _};

/// Ridge penalty on the standardized confounders; the intercept is free.
pub const DEFAULT_RIDGE: f64 = 1.0;

/// Fold assignment and clusters for cross-fitting and the variance.
pub struct CrossFit {
    /// Fold of each row, `0..k`.
    pub fold: Vec<u8>,
    pub k: usize,
    /// Cluster of each row for the variance (a spatial tile).
    pub cluster: Vec<u32>,
    n_clusters: usize,
    /// Training and held-out rows of each fold, ascending.
    train: Vec<Vec<usize>>,
    test: Vec<Vec<usize>>,
}

impl CrossFit {
    pub fn new(fold: Vec<u8>, k: usize, cluster: Vec<u32>) -> Self {
        let n_clusters = cluster.iter().copied().max().map_or(0, |c| c as usize + 1);
        let mut test: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &f) in fold.iter().enumerate() {
            test[f as usize].push(i);
        }
        let train = (0..k)
            .map(|f| (0..fold.len()).filter(|&i| fold[i] as usize != f).collect())
            .collect();
        Self {
            fold,
            k,
            cluster,
            n_clusters,
            train,
            test,
        }
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }
}

/// The confounder matrix with one Gram matrix per held-out fold.
///
/// Column 0 is the intercept; every other column is standardized, so one
/// ridge penalty means the same thing for every confounder.
pub struct Design {
    /// Rows are receivers, column-major `n × p`, stored in f32 to halve the
    /// largest allocation of the run; every sum over it is taken in f64.
    pub x: DMatrix<f32>,
    /// `gram[k] = Σ_{fold(i) ≠ k} x_i x_iᵀ`: the training Gram for fold `k`.
    gram: Vec<DMatrix<f64>>,
}

impl Design {
    /// Standardize the non-intercept columns and precompute the per-fold
    /// training Grams. A constant column is zeroed rather than divided by 0.
    pub fn new(mut x: DMatrix<f32>, cf: &CrossFit) -> Self {
        let n = x.nrows().max(1) as f64;
        for j in 1..x.ncols() {
            let mut col = x.column_mut(j);
            let mean = col.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = col.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let sd = var.sqrt();
            for v in col.iter_mut() {
                *v = if sd > 1e-12 {
                    ((*v as f64 - mean) / sd) as f32
                } else {
                    0.0
                };
            }
        }
        if x.ncols() > 0 {
            x.column_mut(0).fill(1.0);
        }
        let gram = fold_grams(&x, cf);
        Self { x, gram }
    }

    pub fn p(&self) -> usize {
        self.x.ncols()
    }
}

// * Nested designs

/// A borrowed design: the first `p` columns of `x` with their per-fold
/// training Grams. The full [`Design`] and each [`DesignPrefix`] lend one.
#[derive(Clone, Copy)]
pub struct DesignView<'a> {
    pub x: &'a DMatrix<f32>,
    pub p: usize,
    gram: &'a [DMatrix<f64>],
}

/// The first `p` columns of a [`Design`] as a nested adjustment set. It
/// owns only its `p × p` Gram per fold (the Gram of a column subset is the
/// submatrix of the full Gram) and borrows the matrix, so a step costs
/// `k p²` numbers rather than another `n × p` copy.
pub struct DesignPrefix {
    p: usize,
    gram: Vec<DMatrix<f64>>,
}

impl Design {
    pub fn view(&self) -> DesignView<'_> {
        DesignView {
            x: &self.x,
            p: self.p(),
            gram: &self.gram,
        }
    }

    /// The nested design on columns `0..p` (column 0, the intercept,
    /// always included).
    pub fn prefix(&self, p: usize) -> DesignPrefix {
        let p = p.clamp(1, self.p());
        DesignPrefix {
            p,
            gram: self
                .gram
                .iter()
                .map(|g| g.view((0, 0), (p, p)).clone_owned())
                .collect(),
        }
    }
}

impl DesignPrefix {
    pub fn p(&self) -> usize {
        self.p
    }

    /// This prefix over `design`, the design it was cut from.
    pub fn view<'a>(&'a self, design: &'a Design) -> DesignView<'a> {
        debug_assert!(self.p <= design.p());
        DesignView {
            x: &design.x,
            p: self.p,
            gram: &self.gram,
        }
    }
}

/// Rows `rows` of the first `p` columns of `x`, row-major in f64, read
/// column by column: `x` is column-major, so a row read strides by `n`.
pub(super) fn gather_rows(x: &DMatrix<f32>, p: usize, rows: &[usize], buf: &mut Vec<f64>) {
    buf.clear();
    buf.resize(rows.len() * p, 0.0);
    for j in 0..p {
        let col = x.column(j);
        let col = col.as_slice();
        for (r, &i) in rows.iter().enumerate() {
            buf[r * p + j] = col[i] as f64;
        }
    }
}

/// Rows per [`gather_rows`] block: small enough to stay in cache.
pub(super) const ROW_BLOCK: usize = 256;

/// `Σ_{i in fold k} x_i x_iᵀ` for each fold, then training = total − fold.
/// Accumulated over row chunks in f64, rows gathered a block at a time.
fn fold_grams(x: &DMatrix<f32>, cf: &CrossFit) -> Vec<DMatrix<f64>> {
    let (n, p) = x.shape();
    let rows: Vec<usize> = (0..n).collect();
    let per_fold = rows
        .par_chunks(4096)
        .map(|chunk| {
            let mut acc = vec![DMatrix::<f64>::zeros(p, p); cf.k];
            let mut buf = Vec::new();
            for block in chunk.chunks(ROW_BLOCK) {
                gather_rows(x, p, block, &mut buf);
                for (xi, &i) in buf.chunks_exact(p.max(1)).zip(block) {
                    let g = &mut acc[cf.fold[i] as usize];
                    for a in 0..p {
                        let xa = xi[a];
                        if xa == 0.0 {
                            continue;
                        }
                        for b in a..p {
                            g[(a, b)] += xa * xi[b];
                        }
                    }
                }
            }
            acc
        })
        .reduce(
            || vec![DMatrix::<f64>::zeros(p, p); cf.k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );
    let mut per_fold = per_fold;
    for g in &mut per_fold {
        for a in 0..p {
            for b in 0..a {
                g[(a, b)] = g[(b, a)];
            }
        }
    }
    let mut total = DMatrix::<f64>::zeros(p, p);
    for g in &per_fold {
        total += g;
    }
    per_fold.into_iter().map(|g| &total - g).collect()
}

/// Options for the nuisance fits.
#[derive(Clone, Copy, Debug)]
pub struct NuisanceOpts {
    pub ridge: f64,
    /// Rows used to fit each fold's propensity model; the rest are only
    /// predicted. The logistic fit is the one nuisance whose Gram cannot be
    /// shared across pairs, so it is the one that is subsampled.
    pub max_propensity_rows: usize,
    pub seed: u64,
}

impl Default for NuisanceOpts {
    fn default() -> Self {
        Self {
            ridge: DEFAULT_RIDGE,
            max_propensity_rows: 20_000,
            seed: 42,
        }
    }
}

/// Per-row pieces of the augmented overlap estimator, all cross-fitted.
pub struct OverlapTerms {
    /// `h = e(1 − e)`.
    pub h: Vec<f64>,
    /// `h(m₁ − m₀) + T(1 − e)(Y − m₁) − (1 − T)e(Y − m₀)`.
    pub num: Vec<f64>,
    /// `h(m₁ − m₀)`, the outcome regression without the augmentation.
    pub outcome: Vec<f64>,
    /// The overlap weight of the row's own arm: `1 − e` if exposed, `e` if not.
    pub arm_weight: Vec<f64>,
}

/// Per-pair inputs to [`overlap_terms_with`].
#[derive(Clone, Copy)]
pub struct PairInputs<'a> {
    pub t: &'a [bool],
    pub y: &'a [f64],
    /// Extra confounders specific to this pair, standardized
    /// ([`standardize`]). They enter both nuisance models.
    pub extra: &'a [&'a [f64]],
    /// Extra confounders of the outcome model alone (raw scale;
    /// standardized here): those read off the response (the neighbours'
    /// response, bleed). They leave the propensity a function of the
    /// exposure side, so one fit serves every programme ([`propensity`]);
    /// the estimator stays doubly robust, consistent when the outcome model
    /// holds.
    pub outcome_only: &'a [&'a [f64]],
    /// Effect modifiers: the outcome model carries `T · m` for each, so
    /// `m₁ − m₀` can differ between subgroups estimated from these terms.
    pub modifiers: &'a [&'a [f64]],
}

/// Cross-fit both nuisances on a (possibly nested) design and return the
/// per-row estimator pieces, `pair.extra` given on the raw scale. The
/// estimator itself fits the propensity once for every fit that shares it
/// ([`propensity`], [`overlap_terms_with`]).
#[cfg(test)]
pub fn overlap_terms(
    design: DesignView<'_>,
    cf: &CrossFit,
    pair: &PairInputs<'_>,
    opts: &NuisanceOpts,
) -> anyhow::Result<OverlapTerms> {
    let extra: Vec<Vec<f64>> = pair.extra.iter().map(|c| standardize(c)).collect();
    let extra: Vec<&[f64]> = extra.iter().map(|c| c.as_slice()).collect();
    let pair = PairInputs {
        extra: &extra,
        ..*pair
    };
    let e = propensity(design, cf, pair.t, &extra, opts, &mut FoldStarts::default())?;
    overlap_terms_with(design, cf, &pair, &e, opts)
}

/// `x_i·β₀ + Σ_c cols[c][i]·β_c` for each row `i` of `rows`, handed to `f`
/// with the row: `β₀` is the first `p` coefficients of `beta`, `β_c` the
/// ones after.
fn linear_predictor(
    design: DesignView<'_>,
    rows: &[usize],
    beta: &[f64],
    cols: &[&[f64]],
    mut f: impl FnMut(usize, f64),
) {
    let p = design.p;
    let mut buf = Vec::new();
    for block in rows.chunks(ROW_BLOCK) {
        gather_rows(design.x, p, block, &mut buf);
        for (xi, &i) in buf.chunks_exact(p.max(1)).zip(block) {
            let mut eta: f64 = (0..p).map(|j| xi[j] * beta[j]).sum();
            for (c, col) in cols.iter().enumerate() {
                eta += col[i] * beta[p + c];
            }
            f(i, eta);
        }
    }
}

/// Each fold's last solution of one propensity model, where the same fold
/// of the next threshold starts: its training tiles are the same, so the solution
/// is close. Another fold's is not a good start: the folds differ in space,
/// and Newton takes more steps from it than from the marginal log-odds. The
/// problem is strictly convex, so a start changes the steps, not the optimum.
#[derive(Clone, Debug, Default)]
pub struct FoldStarts(Vec<Option<nalgebra::DVector<f64>>>);

impl FoldStarts {
    /// These solutions, of a model on the first `p` design columns and pair
    /// columns after them, cut down to a nested model on the first `nested`
    /// design columns and the same pair columns.
    pub fn nested(&self, p: usize, nested: usize) -> Self {
        Self(
            self.0
                .iter()
                .map(|b| {
                    b.as_ref().map(|b| {
                        let s = b.as_slice();
                        nalgebra::DVector::from_iterator(
                            nested + s.len() - p,
                            s[..nested].iter().chain(&s[p..]).copied(),
                        )
                    })
                })
                .collect(),
        )
    }
}

/// The cross-fitted propensity `e` of each row: a logistic ridge of `t` on
/// the design and `extra` (standardized, [`standardize`]), fitted per fold
/// on the other folds' rows, each from its entry in `starts`, which it
/// updates.
pub fn propensity(
    design: DesignView<'_>,
    cf: &CrossFit,
    t: &[bool],
    extra: &[&[f64]],
    opts: &NuisanceOpts,
    starts: &mut FoldStarts,
) -> anyhow::Result<Vec<f64>> {
    use super::estimate::timing;
    let n = design.x.nrows();
    anyhow::ensure!(t.len() == n, "row count mismatch");
    let tf: Vec<f64> = t.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect();
    let p = design.p;
    let mut e = vec![0.0; n];
    starts.0.resize(cf.k, None);
    for k in 0..cf.k {
        let (train, test) = (&cf.train[k], &cf.test[k]);
        if test.is_empty() {
            continue;
        }
        anyhow::ensure!(!train.is_empty(), "fold {k} leaves no training rows");
        let beta = timing::time(timing::LOGISTIC, || {
            logistic_ridge(
                design.x,
                p,
                extra,
                &tf,
                &subsample(train, opts.max_propensity_rows, opts.seed ^ k as u64),
                opts.ridge,
                starts.0[k].as_ref(),
            )
        })?;
        linear_predictor(design, test, beta.as_slice(), extra, |i, eta| {
            e[i] = sigmoid(eta)
        });
        starts.0[k] = Some(beta);
    }
    Ok(e)
}

/// The per-row estimator pieces with the cross-fitted propensity `e` given
/// ([`propensity`] on the same design, folds, `t` and `extra`): only the
/// outcome model is fitted.
pub fn overlap_terms_with(
    design: DesignView<'_>,
    cf: &CrossFit,
    pair: &PairInputs<'_>,
    e: &[f64],
    opts: &NuisanceOpts,
) -> anyhow::Result<OverlapTerms> {
    let n = design.x.nrows();
    anyhow::ensure!(
        pair.t.len() == n && pair.y.len() == n && e.len() == n,
        "row count mismatch"
    );
    let outcome_only: Vec<Vec<f64>> = pair.outcome_only.iter().map(|c| standardize(c)).collect();
    let modifiers = pair.modifiers;
    let tf: Vec<f64> = pair.t.iter().map(|&t| if t { 1.0 } else { 0.0 }).collect();
    let products: Vec<Vec<f64>> = modifiers
        .iter()
        .map(|m| m.iter().zip(&tf).map(|(a, b)| a * b).collect())
        .collect();

    // Outcome model columns beyond X: extra (both kinds), T, T·modifier.
    let n_extra = pair.extra.len() + outcome_only.len();
    let mut o_cols: Vec<&[f64]> = pair.extra.to_vec();
    o_cols.extend(outcome_only.iter().map(|c| c.as_slice()));
    o_cols.push(&tf);
    o_cols.extend(products.iter().map(|c| c.as_slice()));

    use super::estimate::timing;
    let products = timing::time(timing::RIDGE, || {
        fold_products(design.x, design.p, &o_cols, pair.y, &cf.fold, cf.k)
    });
    let mut h = vec![0.0; n];
    let mut num = vec![0.0; n];
    let mut outcome = vec![0.0; n];
    let mut arm_weight = vec![0.0; n];
    for (k, fold) in products.iter().enumerate() {
        let (train, test) = (&cf.train[k], &cf.test[k]);
        if test.is_empty() {
            continue;
        }
        anyhow::ensure!(!train.is_empty(), "fold {k} leaves no training rows");

        let beta_o = timing::time(timing::RIDGE, || {
            ridge_bordered(&design.gram[k], fold, train.len(), opts.ridge)
        })?;

        let p = design.p;
        let beta_o = beta_o.as_slice();
        linear_predictor(design, test, beta_o, &o_cols[..n_extra], |i, base| {
            let mut effect = beta_o[p + n_extra];
            for (c, m) in modifiers.iter().enumerate() {
                effect += m[i] * beta_o[p + n_extra + 1 + c];
            }
            let m0 = base;
            let m1 = base + effect;
            let ei = e[i];
            let w = ei * (1.0 - ei);
            let yi = pair.y[i];
            h[i] = w;
            outcome[i] = w * (m1 - m0);
            if pair.t[i] {
                arm_weight[i] = 1.0 - ei;
                num[i] = outcome[i] + (1.0 - ei) * (yi - m1);
            } else {
                arm_weight[i] = ei;
                num[i] = outcome[i] - ei * (yi - m0);
            }
        });
    }
    Ok(OverlapTerms {
        h,
        num,
        outcome,
        arm_weight,
    })
}

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

    /// Whether the 95% interval excludes 0.
    pub fn rejects(&self) -> bool {
        self.z().abs() > self.crit()
    }
}

// * The t reference

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

/// `θ̂` over `rows` (ascending row indices), with the cluster sums of its
/// linearization `ψ_i = (num_i − h_i θ̂) / Σ h` (zero outside `rows`),
/// which give the variance and the simultaneous band.
///
/// The counts (`n`, `n_treated`) and the overlap summaries (`mean_h`,
/// `ess = (Σh)² / Σh²`) are over the rows, so the guards
/// on them ([`super::estimate::EffectOpts::guard`]) read the information
/// the estimate rests on.
pub fn subset_estimate(
    terms: &OverlapTerms,
    t: &[bool],
    rows: &[u32],
    cf: &CrossFit,
) -> (Estimate, Vec<f64>) {
    let (mut sh, mut sh2, mut snum) = (0.0, 0.0, 0.0);
    let mut treated = 0usize;
    for &i in rows {
        let i = i as usize;
        sh += terms.h[i];
        sh2 += terms.h[i] * terms.h[i];
        snum += terms.num[i];
        treated += t[i] as usize;
    }
    let cnt = rows.len();
    let mut sums = vec![0.0; cf.n_clusters()];
    if cnt == 0 || treated == 0 || treated == cnt || sh <= 1e-12 {
        return (Estimate::nan(cnt, treated), sums);
    }
    let theta = snum / sh;
    for &i in rows {
        let i = i as usize;
        sums[cf.cluster[i] as usize] += (terms.num[i] - terms.h[i] * theta) / sh;
    }
    let (se, df) = clustered_sd(&sums);
    (
        Estimate {
            theta,
            se,
            n: cnt,
            n_treated: treated,
            mean_h: sh / cnt as f64,
            ess: sh * sh / sh2,
            df,
        },
        sums,
    )
}

/// `a − b` for two subgroup estimates from the same fit, with the
/// variance of the difference from their linearizations' cluster sums.
pub fn difference(a: &Estimate, sums_a: &[f64], b: &Estimate, sums_b: &[f64]) -> Estimate {
    if !a.theta.is_finite() || !b.theta.is_finite() {
        return Estimate::nan(a.n + b.n, a.n_treated + b.n_treated);
    }
    let diff: Vec<f64> = sums_a.iter().zip(sums_b).map(|(x, y)| x - y).collect();
    let (se, df) = clustered_sd(&diff);
    Estimate {
        theta: a.theta - b.theta,
        se,
        n: a.n + b.n,
        n_treated: a.n_treated + b.n_treated,
        mean_h: f64::NAN,
        ess: f64::NAN,
        df,
    }
}

/// `sqrt(Σ_c s_c²)` over the cluster sums `s_c` of ψ, scaled by
/// `C / (C − 1)` for few clusters, and the t reference's degrees of freedom
/// `C − 1` (NaN below two clusters), `C` counting the clusters that carry
/// `ψ`.
fn clustered_sd(sums: &[f64]) -> (f64, f64) {
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

/// The weighted mean response of the exposed minus that of the unexposed over
/// `rows`, with its tile-clustered variance. Weight 1 is the crude contrast,
/// what a co-expression style comparison would report on the same scale; the
/// overlap weight of a row's own arm ([`OverlapTerms::arm_weight`]) is inverse
/// probability weighting, with no outcome model.
pub fn arm_contrast(
    y: &[f64],
    t: &[bool],
    rows: &[u32],
    cf: &CrossFit,
    weight: impl Fn(usize) -> f64,
) -> Estimate {
    let (mut w1, mut s1, mut w0, mut s0) = (0.0, 0.0, 0.0, 0.0);
    let mut treated = 0usize;
    for &i in rows {
        let i = i as usize;
        let w = weight(i);
        if t[i] {
            treated += 1;
            w1 += w;
            s1 += w * y[i];
        } else {
            w0 += w;
            s0 += w * y[i];
        }
    }
    if w1 <= 1e-12 || w0 <= 1e-12 {
        return Estimate::nan(rows.len(), treated);
    }
    let (m1, m0) = (s1 / w1, s0 / w0);
    let mut sums = vec![0.0; cf.n_clusters()];
    for &i in rows {
        let i = i as usize;
        sums[cf.cluster[i] as usize] += if t[i] {
            weight(i) * (y[i] - m1) / w1
        } else {
            -weight(i) * (y[i] - m0) / w0
        };
    }
    let (se, df) = clustered_sd(&sums);
    Estimate {
        theta: m1 - m0,
        se,
        df,
        ..Estimate::nan(rows.len(), treated)
    }
}

/// The outcome regression alone over `rows`, `Σ h(m₁ − m₀) / Σ h`, read from the
/// fit behind `effect`, the augmented estimate on the same rows. Its variance is
/// `effect`'s: the coefficient's sampling error in a linear model is
/// `Σ T̃ (Y − m) / Σ T̃²`, and the augmentation term is that with `T̃ = T − e`.
pub fn outcome_estimate(terms: &OverlapTerms, effect: &Estimate, rows: &[u32]) -> Estimate {
    let (mut outcome, mut h) = (0.0, 0.0);
    for &i in rows {
        outcome += terms.outcome[i as usize];
        h += terms.h[i as usize];
    }
    Estimate {
        theta: if h > 1e-12 { outcome / h } else { f64::NAN },
        ..*effect
    }
}

/// Pearson's correlation of `y` and `x` over `rows`, with the tile-clustered
/// variance of its linearization: the marginal association, no threshold and
/// no adjustment. A constant `y` or `x` has none.
pub fn marginal_correlation(y: &[f64], x: &[f64], rows: &[u32], cf: &CrossFit) -> Estimate {
    let nan = Estimate::nan(rows.len(), 0);
    let n = rows.len() as f64;
    let mean = |v: &[f64]| rows.iter().map(|&i| v[i as usize]).sum::<f64>() / n;
    let (my, mx) = (mean(y), mean(x));
    let spread = |v: &[f64], m: f64| {
        (rows
            .iter()
            .map(|&i| (v[i as usize] - m).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    };
    let (sy, sx) = (spread(y, my), spread(x, mx));
    if rows.is_empty() || sy <= 1e-12 || sx <= 1e-12 {
        return nan;
    }
    let scaled = |i: u32| ((y[i as usize] - my) / sy, (x[i as usize] - mx) / sx);
    let r = rows
        .iter()
        .map(|&i| {
            let (a, b) = scaled(i);
            a * b
        })
        .sum::<f64>()
        / n;
    let mut sums = vec![0.0; cf.n_clusters()];
    for &i in rows {
        let (a, b) = scaled(i);
        sums[cf.cluster[i as usize] as usize] += (a * b - 0.5 * r * (a * a + b * b)) / n;
    }
    let (se, df) = clustered_sd(&sums);
    Estimate {
        theta: r,
        se,
        df,
        ..nan
    }
}

// * Combining analyses

/// Whether an analysis's estimate can enter Rubin's rules: a finite estimate
/// and SE and a positive df. A NaN df (fewer than two variance clusters)
/// makes the analysis not estimable; it never reads as infinite df.
pub fn usable(e: &Estimate) -> bool {
    e.theta.is_finite() && e.se.is_finite() && e.df > 0.0
}

/// Rubin's rules over analyses: mean estimate, variance = mean within-analysis
/// variance + `(1 + 1/R)` × between-analysis variance. Analyses that were not
/// estimable (see [`usable`]) are left out; none left gives NaN. Degrees of freedom by
/// Barnard & Rubin (1999, Biometrika 86:948), with the analyses' smallest df
/// as the complete-data df: the between-analysis spread, estimated from `R`
/// analyses, costs degrees of freedom in proportion to its share of the
/// variance.
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
                .all(|d| d.df == super::estimate::ONE_CLUSTER_DF);
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
/// estimate as `N(θ, se²)` with the prior `θ ~ N(0, prior_var)`.
fn abf_weights(parts: &[Estimate], prior_var: f64) -> (Vec<(usize, f64)>, f64) {
    let log_bf: Vec<(usize, f64)> = parts
        .iter()
        .enumerate()
        .filter(|(_, e)| usable(e))
        .map(|(k, e)| {
            let v = e.se * e.se;
            let shrink = prior_var / (v + prior_var);
            (k, 0.5 * (1.0 - shrink).ln() + 0.5 * e.z() * e.z() * shrink)
        })
        .collect();
    if log_bf.is_empty() {
        return (Vec::new(), f64::NAN);
    }
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
pub fn abf_summary(parts: &[Estimate], taus: &[f64], prior_var: f64) -> AbfSummary {
    let (weights, log10_bf) = abf_weights(parts, prior_var);
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
/// has to exceed negative controls averaged the same way.
pub fn abf_average(parts: &[Estimate], prior_var: f64) -> Estimate {
    let (weights, _) = abf_weights(parts, prior_var);
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

/// A pair's background programmes against its effect: their mean, and the effect over
/// that mean as one more draw from what programmes like it show, with the
/// variance `s² (1 + 1/K)` and `K − 1` degrees of freedom of a predictive test.
/// The spread never reads below the background programmes' own sampling variance, so identical
/// background programmes are not taken for certainty. Fewer than two usable background programmes say nothing.
pub fn background_comparison(effect: &Estimate, backgrounds: &[Estimate]) -> (Estimate, Estimate) {
    let ok: Vec<&Estimate> = backgrounds.iter().filter(|e| usable(e)).collect();
    let nan = Estimate::nan(effect.n, effect.n_treated);
    if ok.len() < 2 {
        return (nan, nan);
    }
    let k = ok.len() as f64;
    let mean = ok.iter().map(|e| e.theta).sum::<f64>() / k;
    let between = ok.iter().map(|e| (e.theta - mean).powi(2)).sum::<f64>() / (k - 1.0);
    let within = ok.iter().map(|e| e.se * e.se).sum::<f64>() / k;
    let var = between.max(within);
    // The effect's own uncertainty sets the width when it is the larger: its
    // analyses can disagree in a way the background programmes, read on one, cannot show.
    let effect_var = effect.se * effect.se;
    let (over_var, over_df) = if effect_var > var {
        (effect_var, effect.df)
    } else {
        (var, k - 1.0)
    };
    let of = |theta: f64, v: f64| Estimate {
        theta,
        se: v.sqrt(),
        df: k - 1.0,
        ..nan
    };
    let over = if effect.theta.is_finite() {
        Estimate {
            df: over_df,
            ..of(effect.theta - mean, over_var + var / k)
        }
    } else {
        nan
    };
    (of(mean, var / k), over)
}
