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

use super::crossfit::{gather_rows, CrossFit, DesignView, DEFAULT_RIDGE, ROW_BLOCK};
use super::inference::{clustered_sd, Estimate};
use super::regression::{
    fold_products, logistic_ridge, ridge_bordered, sigmoid, standardize, subsample,
};

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

/// `θ̂` over `rows` (ascending row indices), with the cluster sums of its
/// linearization `ψ_i = (num_i − h_i θ̂) / Σ h` (zero outside `rows`),
/// which give the variance and the simultaneous band.
///
/// The counts (`n`, `n_treated`) and the overlap summaries (`mean_h`,
/// `ess = (Σh)² / Σh²`) are over the rows, so the guards
/// on them ([`super::options::EffectOpts::guard`]) read the information
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
