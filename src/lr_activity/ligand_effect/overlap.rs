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
/// of the next draw starts: its training tiles are the same, so the solution
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
            num[i] = if pair.t[i] {
                w * (m1 - m0) + (1.0 - ei) * (yi - m1)
            } else {
                w * (m1 - m0) - ei * (yi - m0)
            };
        });
    }
    Ok(OverlapTerms { h, num })
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
/// `ess = (Σh)² / Σh²`) are over the rows, the estimator's own points: on
/// pseudobulk a unit counts once, as it weighs once in `θ̂`, so the guards
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

/// The unadjusted contrast `mean(Y | T=1) − mean(Y | T=0)` over `rows`:
/// what a co-expression style comparison would report on the same scale.
pub fn crude_difference(y: &[f64], t: &[bool], rows: &[u32]) -> f64 {
    let (mut s1, mut n1, mut s0, mut n0) = (0.0, 0usize, 0.0, 0usize);
    for &i in rows {
        let i = i as usize;
        if t[i] {
            s1 += y[i];
            n1 += 1;
        } else {
            s0 += y[i];
            n0 += 1;
        }
    }
    if n1 == 0 || n0 == 0 {
        return f64::NAN;
    }
    s1 / n1 as f64 - s0 / n0 as f64
}
