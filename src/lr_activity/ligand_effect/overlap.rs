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
//! t reference (the usual small-cluster reference for a cluster-robust
//! mean; Cameron & Miller 2015, J Hum Resour 50:317), rather than the
//! normal, with `C_eff − 1` degrees of freedom: `C_eff` counts the clusters
//! the variance rests on, at most the `C` that carry the subgroup
//! ([`super::inference::effective_clusters`]), so a region inside a few
//! tiles is read on few.
//!
//! The nuisance models are deliberately plain: a logistic ridge for `e` and
//! a linear ridge for `m`, both on a shared confounder matrix. Double
//! robustness means either one being right is enough for consistency;
//! plainness is what lets one Gram matrix per fold serve every
//! ligand-receptor pair. A fit whose confounders read off the response
//! borders the shared exposure-side propensity ([`propensity`]) with those
//! columns rather than refitting it ([`logistic_bordered`]).

use super::crossfit::{gather_rows, CrossFit, DesignView, DEFAULT_RIDGE, ROW_BLOCK};
use super::fits::indicator;
use super::inference::{clustered_sd, Estimate};
use super::regression::{
    fold_products, logistic_base, logistic_bordered, logistic_from, ridge_bordered, sigmoid,
    subsample, LogisticBase,
};
use nalgebra::DVector;
use rayon::prelude::*;
use std::sync::Arc;

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
    /// What deleting a tile moves the outcome model by, when asked for
    /// ([`overlap_terms_with`]'s `influence`).
    pub influence: Option<OutcomeInfluence>,
}

/// The outcome model's infinitesimal jackknife (Giordano, Stephenson, Liu,
/// Jordan & Broderick 2019, AISTATS, PMLR 89:1139): deleting tile `c` moves
/// fold `k`'s ridge coefficients by one Newton step from the full solution
/// on its full penalized Gram, `Δβ_k = −A_k⁻¹ g_ck` with
/// `g_ck = Σ_{i∈c} x̃_i (y_i − x̃_i β_k)`; no model is refitted. The outcome
/// regression's sampling error lives in those coefficients, which a
/// jackknife holding the fits fixed would leave out; the augmented estimate
/// is first-order insensitive to them and does not read this.
pub struct OutcomeInfluence {
    /// Per fold, the factor of its penalized Gram `A_k`; `None` for a fold
    /// with no held-out rows.
    factors: Vec<Option<nalgebra::Cholesky<f64, nalgebra::Dyn>>>,
    /// Per fold, `g_ck` of each tile `c` it trains on, `d` numbers a tile.
    gradients: Vec<Vec<f64>>,
    d: usize,
    /// The coefficient of `T`; those of `T · modifier` follow.
    effect: usize,
    /// Per row, `(1, modifier…)`: what multiplies the effect coefficients.
    z: Vec<f64>,
    /// Each tile's fold.
    tile_fold: Vec<u8>,
}

impl OutcomeInfluence {
    /// Per fold, `v_k = A_k⁻¹ M_k`, `M_k` holding `Σ h_i z_i` over the rows of
    /// `rows` the fold holds out at the effect coefficients, zero elsewhere:
    /// `M_k · Δβ_k` is the change of `Σ h(m₁ − m₀)` over those rows.
    pub fn directions(&self, h: &[f64], rows: &[u32], cf: &CrossFit) -> Vec<Option<DVector<f64>>> {
        let w = self.z.len() / h.len().max(1);
        let mut m = vec![DVector::<f64>::zeros(self.d); self.factors.len()];
        for &i in rows {
            let i = i as usize;
            let f = cf.fold[i] as usize;
            let Some(mk) = m.get_mut(f) else { continue };
            for j in 0..w {
                mk[self.effect + j] += h[i] * self.z[i * w + j];
            }
        }
        m.into_iter()
            .zip(&self.factors)
            .map(|(mk, a)| a.as_ref().map(|a| a.solve(&mk)))
            .collect()
    }

    /// The change of `Σ h(m₁ − m₀)` over the rows of `v`
    /// ([`OutcomeInfluence::directions`]) when tile `c` is deleted:
    /// `−Σ_{k ≠ fold(c)} g_ck · v_k`.
    pub fn shift(&self, v: &[Option<DVector<f64>>], c: u32) -> f64 {
        let c = c as usize;
        let own = self.tile_fold[c] as usize;
        let mut total = 0.0;
        for (k, vk) in v.iter().enumerate() {
            let Some(vk) = vk.as_ref().filter(|_| k != own) else {
                continue;
            };
            let g = &self.gradients[k][c * self.d..(c + 1) * self.d];
            total -= g.iter().zip(vk.iter()).map(|(a, b)| a * b).sum::<f64>();
        }
        total
    }
}

/// Per-pair inputs to [`overlap_terms_with`].
#[derive(Clone, Copy)]
pub struct PairInputs<'a> {
    pub t: &'a [bool],
    pub y: &'a [f64],
    /// Extra confounders specific to this pair, standardized
    /// ([`super::regression::standardize`]). They enter both nuisance models.
    pub extra: &'a [&'a [f64]],
    /// Effect modifiers: the outcome model carries `T · m` for each, so
    /// `m₁ − m₀` can differ between subgroups estimated from these terms.
    pub modifiers: &'a [&'a [f64]],
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
///
#[derive(Clone, Default)]
pub struct FoldStarts(Vec<Option<FoldStart>>);

/// One fold's start: a solution, or the whole fit it came from, which a
/// jackknife replicate of the same folds (rows left out,
/// [`super::crossfit::LEFT_OUT`]) starts from: its Newton factor less those
/// rows ([`LogisticBase::leaving_out`]), rather than a fresh Hessian.
#[derive(Clone)]
enum FoldStart {
    Beta(nalgebra::DVector<f64>),
    Fit(Arc<LogisticBase>),
}

impl FoldStart {
    fn beta(&self) -> &nalgebra::DVector<f64> {
        match self {
            Self::Beta(b) => b,
            Self::Fit(f) => &f.beta,
        }
    }
}

impl FoldStarts {
    /// These solutions, of a model on the first `p` design columns and pair
    /// columns after them, cut down to a nested model on the first `nested`
    /// design columns and the same pair columns. The fits do not carry over:
    /// they are of another design.
    pub fn nested(&self, p: usize, nested: usize) -> Self {
        Self(
            self.0
                .iter()
                .map(|b| {
                    b.as_ref().map(|b| {
                        let s = b.beta().as_slice();
                        FoldStart::Beta(nalgebra::DVector::from_iterator(
                            nested + s.len() - p,
                            s[..nested].iter().chain(&s[p..]).copied(),
                        ))
                    })
                })
                .collect(),
        )
    }
}

/// The cross-fitted propensity `e` of each row: a logistic ridge of `t` on
/// the design and `extra` (standardized, [`super::regression::standardize`]), fitted per fold
/// on the other folds' rows, each from its entry in `starts`, which it
/// updates.
pub fn propensity(
    design: DesignView<'_>,
    cf: &CrossFit,
    t: &[bool],
    extra: &[&[f64]],
    opts: &NuisanceOpts,
    starts: &mut FoldStarts,
) -> anyhow::Result<Propensity> {
    use super::timing;
    let n = design.x.nrows();
    anyhow::ensure!(t.len() == n, "row count mismatch");
    let tf = indicator(t);
    let p = design.p;
    starts.0.resize(cf.k, None);
    // Folds are independent: fit them side by side on this worker's pool.
    let folds: Vec<Option<Arc<LogisticBase>>> = (0..cf.k)
        .into_par_iter()
        .map(|k| {
            let (train, test) = (&cf.train[k], &cf.test[k]);
            if test.is_empty() {
                return Ok(None);
            }
            anyhow::ensure!(!train.is_empty(), "fold {k} leaves no training rows");
            let start = starts.0[k].as_ref().map(FoldStart::beta);
            // A jackknife replicate: the full fit's rows less the ones left
            // out, from its factor less theirs.
            let replicate = match &starts.0[k] {
                Some(FoldStart::Fit(b)) => b.leaving_out(&cf.fold, design.x, p, extra),
                _ => None,
            };
            timing::time(timing::LOGISTIC, || match replicate {
                Some(rep) => logistic_from(design.x, p, extra, &tf, opts.ridge, start, rep),
                None => {
                    let rows = subsample(train, opts.max_propensity_rows, opts.seed ^ k as u64);
                    logistic_base(design.x, p, extra, &tf, &rows, opts.ridge, start)
                }
            })
            .map(|b| Some(Arc::new(b)))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut e = vec![0.0; n];
    for (k, base) in folds.iter().enumerate() {
        let Some(base) = base else { continue };
        linear_predictor(
            design,
            &cf.test[k],
            base.beta.as_slice(),
            extra,
            |i, eta| e[i] = sigmoid(eta),
        );
        starts.0[k] = Some(FoldStart::Fit(Arc::clone(base)));
    }
    Ok(Propensity { e, folds })
}

/// A cross-fitted propensity ([`propensity`]) with each fold's fit, from
/// which a propensity with further columns starts ([`propensity_bordered`]).
pub struct Propensity {
    pub e: Vec<f64>,
    folds: Vec<Option<Arc<LogisticBase>>>,
}

impl Propensity {
    /// Each fold's solution, as the warm start of a fit that nests in or
    /// follows this one.
    pub fn starts(&self) -> FoldStarts {
        FoldStarts(
            self.folds
                .iter()
                .map(|f| f.as_ref().map(|f| FoldStart::Fit(Arc::clone(f))))
                .collect(),
        )
    }
}

/// The cross-fitted propensity on the design, `extra` and `added`, each fold
/// fitted from `base`'s fit of the same fold on the design and `extra`
/// ([`logistic_bordered`]): the optimum [`propensity`] would reach on all
/// the columns, at a fraction of the cost.
pub fn propensity_bordered(
    design: DesignView<'_>,
    cf: &CrossFit,
    t: &[bool],
    extra: &[&[f64]],
    added: &[&[f64]],
    base: &Propensity,
) -> anyhow::Result<Vec<f64>> {
    use super::timing;
    let n = design.x.nrows();
    anyhow::ensure!(t.len() == n, "row count mismatch");
    anyhow::ensure!(base.folds.len() == cf.k, "base propensity has other folds");
    let tf = indicator(t);
    let mut all: Vec<&[f64]> = extra.to_vec();
    all.extend_from_slice(added);
    let betas: Vec<Option<DVector<f64>>> = base
        .folds
        .par_iter()
        .map(|fold| {
            let Some(fold) = fold else { return Ok(None) };
            timing::time(timing::LOGISTIC, || {
                logistic_bordered(design.x, design.p, extra, added, &tf, fold)
            })
            .map(|(beta, _)| Some(beta))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut e = vec![0.0; n];
    for (k, beta) in betas.iter().enumerate() {
        let Some(beta) = beta else { continue };
        linear_predictor(design, &cf.test[k], beta.as_slice(), &all, |i, eta| {
            e[i] = sigmoid(eta)
        });
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
    influence: bool,
) -> anyhow::Result<OverlapTerms> {
    let n = design.x.nrows();
    anyhow::ensure!(
        pair.t.len() == n && pair.y.len() == n && e.len() == n,
        "row count mismatch"
    );
    let modifiers = pair.modifiers;
    let tf = indicator(pair.t);
    let products: Vec<Vec<f64>> = modifiers
        .iter()
        .map(|m| m.iter().zip(&tf).map(|(a, b)| a * b).collect())
        .collect();

    // Outcome model columns beyond X: extra, T, T·modifier.
    let n_extra = pair.extra.len();
    let mut o_cols: Vec<&[f64]> = pair.extra.to_vec();
    o_cols.push(&tf);
    o_cols.extend(products.iter().map(|c| c.as_slice()));

    use super::timing;
    let products = timing::time(timing::RIDGE, || {
        fold_products(design.x, design.p, &o_cols, pair.y, &cf.fold, cf.k)
    });
    let mut h = vec![0.0; n];
    let mut num = vec![0.0; n];
    let mut outcome = vec![0.0; n];
    let mut arm_weight = vec![0.0; n];
    let mut fits = Vec::with_capacity(cf.k);
    for (k, fold) in products.iter().enumerate() {
        let (train, test) = (&cf.train[k], &cf.test[k]);
        if test.is_empty() {
            fits.push(None);
            continue;
        }
        anyhow::ensure!(!train.is_empty(), "fold {k} leaves no training rows");

        let (beta_o, factor) = timing::time(timing::RIDGE, || {
            ridge_bordered(&design.gram[k], fold, train.len(), opts.ridge)
        })?;
        fits.push(Some((beta_o.clone(), factor)));

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
    let influence = influence.then(|| {
        let effect = design.p + n_extra;
        outcome_influence(design, cf, &o_cols, pair, fits, effect)
    });
    Ok(OverlapTerms {
        h,
        num,
        outcome,
        arm_weight,
        influence,
    })
}

/// A fold's ridge coefficients and the factor of its penalized Gram.
type RidgeFit = (DVector<f64>, nalgebra::Cholesky<f64, nalgebra::Dyn>);

/// [`OutcomeInfluence`] of the folds' ridge `fits` of `pair.y` on the design
/// and `o_cols`, the coefficient of `T` at `effect`.
fn outcome_influence(
    design: DesignView<'_>,
    cf: &CrossFit,
    o_cols: &[&[f64]],
    pair: &PairInputs<'_>,
    fits: Vec<Option<RidgeFit>>,
    effect: usize,
) -> OutcomeInfluence {
    use super::regression::gather_design_rows;
    let d = design.p + o_cols.len();
    let n_tiles = cf.n_clusters();
    let gradients = fits
        .par_iter()
        .enumerate()
        .map(|(k, fit)| {
            let mut g = vec![0.0; n_tiles * d];
            let Some((beta, _)) = fit else { return g };
            let mut buf: Vec<f64> = Vec::new();
            for block in cf.train[k].chunks(ROW_BLOCK) {
                gather_design_rows(design.x, design.p, o_cols, block, &mut buf);
                for (zi, &i) in buf.chunks_exact(d.max(1)).zip(block) {
                    let r = pair.y[i] - zi.iter().zip(beta.iter()).map(|(a, b)| a * b).sum::<f64>();
                    let c = cf.cluster[i] as usize;
                    for (gj, zj) in g[c * d..(c + 1) * d].iter_mut().zip(zi) {
                        *gj += r * zj;
                    }
                }
            }
            g
        })
        .collect();
    let w = 1 + pair.modifiers.len();
    let n = pair.y.len();
    let mut z = vec![1.0; n * w];
    for (j, m) in pair.modifiers.iter().enumerate() {
        for i in 0..n {
            z[i * w + 1 + j] = m[i];
        }
    }
    let mut tile_fold = vec![0u8; n_tiles];
    for (i, &c) in cf.cluster.iter().enumerate() {
        tile_fold[c as usize] = cf.fold[i];
    }
    OutcomeInfluence {
        factors: fits.into_iter().map(|f| f.map(|(_, a)| a)).collect(),
        gradients,
        d,
        effect,
        z,
        tile_fold,
    }
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
    ratio_estimate(&terms.num, &terms.h, t, rows, cf)
}

/// The outcome regression alone over `rows`, `Σ h(m₁ − m₀) / Σ h`, read as
/// [`subset_estimate`] reads the augmented one, the nuisance fits taken as
/// known: its SE weighs the thresholds of the average, and the tile
/// jackknife replaces it ([`super::jackknife::EstimatorJackknife`]).
pub fn outcome_estimate(terms: &OverlapTerms, t: &[bool], rows: &[u32], cf: &CrossFit) -> Estimate {
    ratio_estimate(&terms.outcome, &terms.h, t, rows, cf).0
}

/// `Σ num / Σ h` over `rows`, with its linearization's cluster sums.
fn ratio_estimate(
    num: &[f64],
    h: &[f64],
    t: &[bool],
    rows: &[u32],
    cf: &CrossFit,
) -> (Estimate, Vec<f64>) {
    let (mut sh, mut sh2, mut snum) = (0.0, 0.0, 0.0);
    let mut treated = 0usize;
    for &i in rows {
        let i = i as usize;
        sh += h[i];
        sh2 += h[i] * h[i];
        snum += num[i];
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
        sums[cf.cluster[i] as usize] += (num[i] - h[i] * theta) / sh;
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
