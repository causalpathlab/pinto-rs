//! Nuisance regressions for the overlap estimator: a ridge regression that
//! borders a precomputed Gram matrix, and a logistic ridge fitted by
//! iteratively reweighted least squares. Both leave the intercept
//! (column 0) unpenalized.

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

pub(super) fn standardize(c: &[f64]) -> Vec<f64> {
    let n = c.len().max(1) as f64;
    let mean = c.iter().sum::<f64>() / n;
    let sd = (c.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    if sd > 1e-12 {
        c.iter().map(|v| (v - mean) / sd).collect()
    } else {
        vec![0.0; c.len()]
    }
}

pub(super) fn sigmoid(eta: f64) -> f64 {
    let eta = eta.clamp(-30.0, 30.0);
    1.0 / (1.0 + (-eta).exp())
}

/// Deterministic subsample of at most `m` rows, evenly strided from a
/// seeded offset, so a refit sees the same rows.
pub(super) fn subsample(rows: &[usize], m: usize, seed: u64) -> Vec<usize> {
    if rows.len() <= m || m == 0 {
        return rows.to_vec();
    }
    let stride = rows.len() as f64 / m as f64;
    let offset = (seed % 1_000_003) as f64 / 1_000_003.0 * stride;
    (0..m)
        .map(|j| rows[((offset + j as f64 * stride) as usize).min(rows.len() - 1)])
        .collect()
}

/// The cross-products a fold's outcome ridge needs beyond its Gram:
/// `[X | cols]ᵀ cols` and `[X | cols]ᵀ y` over the fold's training rows.
pub(super) struct FoldProducts {
    a: DMatrix<f64>,
    b: DVector<f64>,
}

/// [`FoldProducts`] of every fold in one pass over the rows: each row adds to
/// its own fold's sums, and a fold's training sums are the total less its
/// own, as for the Grams. `X` is the first `p` columns of `x`.
pub(super) fn fold_products(
    x: &DMatrix<f32>,
    p: usize,
    cols: &[&[f64]],
    y: &[f64],
    fold: &[u8],
    k: usize,
) -> Vec<FoldProducts> {
    use super::crossfit::ROW_BLOCK;
    debug_assert!(p <= x.ncols());
    let q = cols.len();
    let d = p + q;
    let rows: Vec<usize> = (0..x.nrows()).collect();
    let empty = || -> Vec<(DMatrix<f64>, DVector<f64>)> {
        vec![(DMatrix::zeros(d, q), DVector::zeros(d)); k]
    };
    let partial: Vec<Vec<(DMatrix<f64>, DVector<f64>)>> = rows
        .par_chunks(8192)
        .map(|rows| {
            let mut sums = empty();
            let mut buf = Vec::new();
            for block in rows.chunks(ROW_BLOCK) {
                gather_design_rows(x, p, cols, block, &mut buf);
                for (zi, &i) in buf.chunks_exact(d.max(1)).zip(block) {
                    let (a, b) = &mut sums[fold[i] as usize];
                    let yi = y[i];
                    for r in 0..d {
                        b[r] += zi[r] * yi;
                    }
                    for c in 0..q {
                        let zc = zi[p + c];
                        let out = a.column_mut(c);
                        for (o, &z) in out.into_iter().zip(zi) {
                            *o += z * zc;
                        }
                    }
                }
            }
            sums
        })
        .collect();
    let mut per_fold = empty();
    for sums in partial {
        for ((a, b), (pa, pb)) in per_fold.iter_mut().zip(sums) {
            *a += pa;
            *b += pb;
        }
    }
    let mut total = (DMatrix::<f64>::zeros(d, q), DVector::<f64>::zeros(d));
    for (a, b) in &per_fold {
        total.0 += a;
        total.1 += b;
    }
    per_fold
        .into_iter()
        .map(|(a, b)| FoldProducts {
            a: &total.0 - a,
            b: &total.1 - b,
        })
        .collect()
}

/// Ridge regression of `y` on `[X | cols]` over `n_train` training rows,
/// bordering their precomputed `XᵀX` with the extra columns' cross-products
/// ([`fold_products`]). The intercept (column 0) is unpenalized. Returns
/// `p + q` coefficients.
pub(super) fn ridge_bordered(
    xtx: &DMatrix<f64>,
    products: &FoldProducts,
    n_train: usize,
    ridge: f64,
) -> anyhow::Result<DVector<f64>> {
    let p = xtx.nrows();
    let FoldProducts { a, b } = products;
    let d = a.nrows();
    let q = a.ncols();
    debug_assert_eq!(d, p + q);
    let mut g = DMatrix::<f64>::zeros(d, d);
    g.view_mut((0, 0), (p, p)).copy_from(xtx);
    for c in 0..q {
        for r in 0..d {
            g[(r, p + c)] = a[(r, c)];
            g[(p + c, r)] = a[(r, c)];
        }
    }
    let scale = n_train.max(1) as f64;
    for j in 1..d {
        g[(j, j)] += ridge;
    }
    // A tiny relative jitter keeps an all-zero column (a modifier absent
    // from the training rows) from making the system singular.
    for j in 0..d {
        g[(j, j)] += 1e-9 * scale;
    }
    Ok(factor_spd(g)?.solve(b))
}

/// Newton steps a logistic fit may take at most; every step moves, so this
/// is a guard ([`super::timing::newton`] counts the fits it stops).
const MAX_NEWTON_STEPS: usize = 500;

/// Largest coefficient change at which a Newton step's Hessian may serve the
/// next step: close to the optimum the weights barely move.
const REUSE_HESSIAN_BELOW: f64 = 0.05;

/// Largest ratio of successive step sizes at which a bordered factor
/// ([`logistic_bordered`]) is kept.
const BORDERED_KEEP_BELOW: f64 = 0.9;

thread_local! {
    /// A logistic fit's rows, kept between fits: allocating them anew for
    /// every fit costs a page fault per page, and many workers fault at once.
    static LOGISTIC_ROWS: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A converged logistic fit with what a refit adding columns reuses
/// ([`logistic_bordered`]): the factor of the last penalized Hessian its
/// Newton steps computed, near the optimum, and the weights `e(1 − e)` over
/// the rows it was computed at.
pub(super) struct LogisticBase {
    pub beta: DVector<f64>,
    hessian: nalgebra::Cholesky<f64, nalgebra::Dyn>,
    pub weights: Vec<f64>,
    /// The ridge it was fitted with, which the bordered refit shares.
    ridge: f64,
}

/// Logistic ridge by iteratively reweighted least squares on `[X | cols]`
/// over `rows`, `X` being the first `p` columns of `x`. Intercept
/// unpenalized. `init` starts Newton from a nearby solution (the same fold's
/// on the previous threshold): the penalized problem is strictly convex, so the
/// start changes only how many steps reach the one optimum.
///
/// Memory traffic, not arithmetic, is what limits many fits side by side, so
/// the rows are held once, row-major, and one pass gives the objective, the
/// gradient and the weights; and close to the optimum a step reuses the last
/// Hessian's factor, which moves the iterate as far toward the same optimum
/// (the gradient is exact) at a fraction of the cost, for as long as steps
/// keep shrinking.
///
/// Returns the fit with what [`logistic_bordered`] reuses.
pub(super) fn logistic_base(
    x: &DMatrix<f32>,
    p: usize,
    cols: &[&[f64]],
    t: &[f64],
    rows: &[usize],
    ridge: f64,
    init: Option<&DVector<f64>>,
) -> anyhow::Result<LogisticBase> {
    let d = p + cols.len();
    LOGISTIC_ROWS.with_borrow_mut(|z| {
        gather_design_rows(x, p, cols, rows, z);
        let tt: Vec<f64> = rows.iter().map(|&i| t[i]).collect();
        let fit = newton_on_rows(z, d, &tt, ridge, init, None)?;
        let (hessian, weights) = fit.last_fresh.expect("a fit given no factor computes one");
        Ok(LogisticBase {
            beta: fit.beta,
            hessian,
            weights,
            ridge,
        })
    })
}

/// The logistic ridge on `[X | cols | added]` over the rows `base` was fitted
/// on (its fit on `[X | cols]`), started from `base`'s solution with zeros for
/// `added`. Its Newton steps use `base`'s Hessian bordered by the added
/// columns' cross-products at `base`'s weights, solved through the Schur
/// complement of the added block: `O(m q d)` to set up and `O(m d)` a step,
/// where a fresh Hessian costs `O(m d²)`. The gradient is exact, so the steps
/// reach the same optimum a cold fit does; a fresh Hessian is computed only
/// when the bordered one stops pointing uphill. Returns the solution and the
/// count of fresh Hessians.
pub(super) fn logistic_bordered(
    x: &DMatrix<f32>,
    p: usize,
    cols: &[&[f64]],
    added: &[&[f64]],
    t: &[f64],
    rows: &[usize],
    base: &LogisticBase,
) -> anyhow::Result<(DVector<f64>, usize)> {
    let ridge = base.ridge;
    let d0 = p + cols.len();
    anyhow::ensure!(base.beta.len() == d0, "base fit has another design");
    anyhow::ensure!(base.weights.len() == rows.len(), "base fit has other rows");
    let mut all: Vec<&[f64]> = cols.to_vec();
    all.extend_from_slice(added);
    let d = d0 + added.len();
    LOGISTIC_ROWS.with_borrow_mut(|z| {
        gather_design_rows(x, p, &all, rows, z);
        let tt: Vec<f64> = rows.iter().map(|&i| t[i]).collect();
        let border = Bordered::new(z, d, base)?;
        let mut init = DVector::<f64>::zeros(d);
        init.rows_mut(0, d0).copy_from(&base.beta);
        let fit = newton_on_rows(
            z,
            d,
            &tt,
            ridge,
            Some(&init),
            Some(Factor::Bordered(border)),
        )?;
        Ok((fit.beta, fit.hessians))
    })
}

/// `ZᵀWZ + λD`, factored, with the jitter every Newton Hessian carries.
pub(super) fn penalized_hessian(
    z: &[f64],
    d: usize,
    weights: &[f64],
    ridge: f64,
) -> anyhow::Result<nalgebra::Cholesky<f64, nalgebra::Dyn>> {
    let mut hmat = weighted_gram(z, d, weights);
    for j in 1..d {
        hmat[(j, j)] += ridge;
    }
    for j in 0..d {
        hmat[(j, j)] += 1e-9 * weights.len() as f64;
    }
    factor_spd(hmat)
}

/// A base fit's Hessian `A` bordered by added columns `S`:
/// `[[A, B], [Bᵀ, C]]` with `B = Z₀ᵀWS`, `C = SᵀWS + λI` at the base weights,
/// solved through `A⁻¹B` and the Schur complement `C − BᵀA⁻¹B`.
pub(super) struct Bordered<'a> {
    a: &'a nalgebra::Cholesky<f64, nalgebra::Dyn>,
    a_inv_b: DMatrix<f64>,
    b: DMatrix<f64>,
    schur: nalgebra::Cholesky<f64, nalgebra::Dyn>,
}

impl<'a> Bordered<'a> {
    pub(super) fn new(z: &[f64], d: usize, base: &'a LogisticBase) -> anyhow::Result<Self> {
        let d0 = base.beta.len();
        let q = d - d0;
        let m = base.weights.len();
        let mut b = DMatrix::<f64>::zeros(d0, q);
        let mut c = DMatrix::<f64>::zeros(q, q);
        for (zr, &w) in z.chunks_exact(d).zip(&base.weights) {
            let (head, tail) = zr.split_at(d0);
            for (k, &s) in tail.iter().enumerate() {
                let ws = w * s;
                for (j, &a) in head.iter().enumerate() {
                    b[(j, k)] += a * ws;
                }
                for (l, &s2) in tail.iter().enumerate() {
                    c[(l, k)] += s2 * ws;
                }
            }
        }
        for k in 0..q {
            c[(k, k)] += base.ridge + 1e-9 * m as f64;
        }
        let a_inv_b = base.hessian.solve(&b);
        let schur = factor_spd(c - b.transpose() * &a_inv_b)?;
        Ok(Self {
            a: &base.hessian,
            a_inv_b,
            b,
            schur,
        })
    }

    pub(super) fn solve(&self, g: &DVector<f64>) -> DVector<f64> {
        let d0 = self.a_inv_b.nrows();
        let q = self.a_inv_b.ncols();
        let u = self.a.solve(&g.rows(0, d0).into_owned());
        let rhs = g.rows(d0, q) - self.b.transpose() * &u;
        let lower = self.schur.solve(&rhs);
        let upper = u - &self.a_inv_b * &lower;
        let mut out = DVector::<f64>::zeros(d0 + q);
        out.rows_mut(0, d0).copy_from(&upper);
        out.rows_mut(d0, q).copy_from(&lower);
        out
    }
}

/// The factor a Newton step solves with.
enum Factor<'a> {
    /// The Hessian at an iterate.
    Fresh(nalgebra::Cholesky<f64, nalgebra::Dyn>),
    /// A base fit's Hessian bordered by added columns ([`logistic_bordered`]).
    Bordered(Bordered<'a>),
}

impl Factor<'_> {
    fn solve(&self, g: &DVector<f64>) -> DVector<f64> {
        match self {
            Self::Fresh(c) => c.solve(g),
            Self::Bordered(b) => b.solve(g),
        }
    }
}

/// Drop the factor in use, keeping it in `last_fresh` when it is a fresh one.
fn retire(
    factor: &mut Option<Factor<'_>>,
    last_fresh: &mut Option<nalgebra::Cholesky<f64, nalgebra::Dyn>>,
) {
    if let Some(Factor::Fresh(c)) = factor.take() {
        *last_fresh = Some(c);
    }
}

/// What [`newton_on_rows`] returns.
struct NewtonFit {
    beta: DVector<f64>,
    /// The last fresh Hessian's factor and the weights it was computed at;
    /// `None` when every step used the factor it was given.
    last_fresh: Option<(nalgebra::Cholesky<f64, nalgebra::Dyn>, Vec<f64>)>,
    /// Fresh Hessians computed.
    hessians: usize,
}

/// Rows `rows` of `[X | cols]`, row-major, gathered a cache-sized block at a
/// time: `x` is column-major.
fn gather_design_rows(
    x: &DMatrix<f32>,
    p: usize,
    cols: &[&[f64]],
    rows: &[usize],
    z: &mut Vec<f64>,
) {
    use super::crossfit::ROW_BLOCK;
    let d = p + cols.len();
    z.clear();
    z.resize(rows.len() * d, 0.0);
    if d == 0 {
        return;
    }
    let columns: Vec<&[f32]> = (0..p)
        .map(|j| {
            let start = j * x.nrows();
            &x.as_slice()[start..start + x.nrows()]
        })
        .collect();
    for (out, block) in z.chunks_mut(ROW_BLOCK * d).zip(rows.chunks(ROW_BLOCK)) {
        for (j, col) in columns.iter().enumerate() {
            for (zr, &i) in out.chunks_exact_mut(d).zip(block) {
                zr[j] = col[i] as f64;
            }
        }
        for (c, col) in cols.iter().enumerate() {
            for (zr, &i) in out.chunks_exact_mut(d).zip(block) {
                zr[p + c] = col[i];
            }
        }
    }
}

/// What one pass over the rows gives at a `beta`.
struct RowPass {
    /// Penalized log-likelihood, the objective Newton climbs.
    objective: f64,
    /// `Zᵀ(t − e) − λDβ`.
    gradient: Vec<f64>,
    /// `e(1 − e)`, floored.
    weights: Vec<f64>,
}

impl RowPass {
    fn new(m: usize, d: usize) -> Self {
        Self {
            objective: f64::NEG_INFINITY,
            gradient: vec![0.0; d],
            weights: vec![0.0; m],
        }
    }

    fn at(&mut self, z: &[f64], t: &[f64], beta: &[f64], ridge: f64) {
        self.gradient.fill(0.0);
        let ll = row_pass_best(z, t, beta, &mut self.gradient, &mut self.weights);
        let mut penalty = 0.0;
        for (b, g) in beta.iter().zip(self.gradient.iter_mut()).skip(1) {
            penalty += b * b;
            *g -= ridge * b;
        }
        self.objective = ll - 0.5 * ridge * penalty;
    }
}

/// [`row_pass`] on the widest vectors the machine has.
fn row_pass_best(
    z: &[f64],
    t: &[f64],
    beta: &[f64],
    gradient: &mut [f64],
    weights: &mut [f64],
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the check above found the feature the function enables.
        return unsafe { row_pass_avx2(z, t, beta, gradient, weights) };
    }
    row_pass(z, t, beta, gradient, weights)
}

/// The log-likelihood, adding each row's score to `gradient` and writing its
/// weight. Written in fixed lanes without fused multiply-add, so wider
/// vectors give the same bits.
#[inline(always)]
fn row_pass(z: &[f64], t: &[f64], beta: &[f64], gradient: &mut [f64], weights: &mut [f64]) -> f64 {
    let d = beta.len().max(1);
    let mut ll = 0.0;
    for ((zr, &tr), wr) in z.chunks_exact(d).zip(t).zip(weights.iter_mut()) {
        let mut lanes = [0.0f64; 4];
        let (head, tail) = zr.as_chunks::<4>();
        let (bhead, btail) = beta.as_chunks::<4>();
        for (a, b) in head.iter().zip(bhead) {
            for k in 0..4 {
                lanes[k] += a[k] * b[k];
            }
        }
        let mut eta = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        for (a, b) in tail.iter().zip(btail) {
            eta += a * b;
        }
        let e = sigmoid(eta);
        *wr = (e * (1.0 - e)).max(1e-10);
        ll += tr * eta - softplus(eta);
        let residual = tr - e;
        for (g, &a) in gradient.iter_mut().zip(zr) {
            *g += residual * a;
        }
    }
    ll
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn row_pass_avx2(
    z: &[f64],
    t: &[f64],
    beta: &[f64],
    gradient: &mut [f64],
    weights: &mut [f64],
) -> f64 {
    row_pass(z, t, beta, gradient, weights)
}

/// Newton on rows `z` (row-major, `d` columns); see [`logistic_base`].
/// `first`, when given, is the factor the first steps solve with
/// ([`logistic_bordered`]), kept for as long as its steps keep shrinking.
fn newton_on_rows<'a>(
    z: &[f64],
    d: usize,
    t: &[f64],
    ridge: f64,
    init: Option<&DVector<f64>>,
    first: Option<Factor<'a>>,
) -> anyhow::Result<NewtonFit> {
    use super::timing;
    let m = t.len();
    let mut beta = match init {
        Some(b) if b.len() == d && b.iter().all(|v| v.is_finite()) => b.clone(),
        _ => {
            // Start at the marginal log-odds, clamped away from separation.
            let mut b = DVector::<f64>::zeros(d);
            let mean_t = t.iter().sum::<f64>() / m.max(1) as f64;
            let m0 = mean_t.clamp(1e-4, 1.0 - 1e-4);
            b[0] = (m0 / (1.0 - m0)).ln();
            b
        }
    };
    let mut at = RowPass::new(m, d);
    let mut next = RowPass::new(m, d);
    at.at(z, t, beta.as_slice(), ridge);
    let mut trial = beta.clone();
    let mut factor = first;
    let mut last_amax = f64::INFINITY;
    let mut hessians = 0usize;
    let mut last_fresh = None;
    // The weights the factor last computed fresh was computed at.
    let mut fresh_weights = Vec::new();
    let mut converged = false;
    // Curvature pairs `(s, y, 1/sᵀy)` of the steps taken with a bordered
    // factor, which refine it into a quasi-Newton inverse ([`lbfgs_direction`]).
    let mut pairs: std::collections::VecDeque<(DVector<f64>, DVector<f64>, f64)> =
        std::collections::VecDeque::new();
    for _ in 0..MAX_NEWTON_STEPS {
        let reused = factor.is_some();
        let chol: &Factor = match &factor {
            Some(c) => c,
            None => {
                hessians += 1;
                // Newton step: (ZᵀWZ + λD) Δ = Zᵀ(t − e) − λDβ.
                let fresh = penalized_hessian(z, d, &at.weights, ridge)?;
                fresh_weights.clone_from(&at.weights);
                factor.insert(Factor::Fresh(fresh))
            }
        };
        let bordered = matches!(chol, Factor::Bordered(_));
        let gradient = DVector::from_column_slice(&at.gradient);
        let mut step = if bordered {
            lbfgs_direction(chol, &pairs, gradient)
        } else {
            chol.solve(&gradient)
        };
        // Halve the step until the objective does not fall: undamped Newton
        // can oscillate on a nearly separated fold and never converge;
        // halving keeps every step uphill on the concave objective, so the
        // iterations reach its one optimum.
        let mut accepted = false;
        let mut halved = false;
        for _ in 0..30 {
            trial.copy_from(&beta);
            trial += &step;
            next.at(z, t, trial.as_slice(), ridge);
            if next.objective >= at.objective - 1e-12 * (1.0 + at.objective.abs()) {
                accepted = true;
                break;
            }
            step *= 0.5;
            halved = true;
        }
        if !accepted {
            // A reused factor may point poorly: only a fresh one that finds
            // no ascent at machine precision says this is the optimum.
            if reused {
                retire(&mut factor, &mut last_fresh);
                pairs.clear();
                continue;
            }
            converged = true;
            break;
        }
        let gain = next.objective - at.objective;
        if bordered {
            // The objective is strictly concave, so `sᵀy > 0` for every step
            // but one lost to rounding.
            let y = DVector::from_column_slice(&at.gradient)
                - DVector::from_column_slice(&next.gradient);
            let sy = step.dot(&y);
            if sy > 1e-12 * step.norm() * y.norm() {
                if pairs.len() == LBFGS_MEMORY {
                    pairs.pop_front();
                }
                pairs.push_back((step.clone(), y, 1.0 / sy));
            }
        }
        std::mem::swap(&mut beta, &mut trial);
        std::mem::swap(&mut at, &mut next);
        let amax = step.amax();
        // A reused factor's step says less about the distance left than a
        // fresh one's, so only the size of a full step may end the fit then:
        // a halved one is small because the factor pointed poorly.
        let small_gain = !reused && gain.abs() < 1e-12 * (1.0 + at.objective.abs());
        let small_step = amax < 1e-8 && !(reused && halved);
        if small_step || small_gain {
            converged = true;
            break;
        }
        // Keep the factor only while the steps shrink fast. A bordered
        // factor is kept from the first step, and while its steps shrink at
        // all: at width d one fresh Hessian costs about d/2 of its steps,
        // so even slow linear convergence is the cheaper way to the optimum.
        let shrinking = !halved
            && if bordered {
                amax < BORDERED_KEEP_BELOW * last_amax
            } else {
                amax < REUSE_HESSIAN_BELOW && amax < 0.5 * last_amax
            };
        if !shrinking {
            retire(&mut factor, &mut last_fresh);
            pairs.clear();
        }
        last_amax = amax;
    }
    timing::newton(hessians, converged);
    retire(&mut factor, &mut last_fresh);
    Ok(NewtonFit {
        beta,
        last_fresh: last_fresh.map(|c| (c, fresh_weights)),
        hessians,
    })
}

/// Curvature pairs a bordered fit keeps ([`lbfgs_direction`]).
const LBFGS_MEMORY: usize = 5;

/// The L-BFGS step for gradient `g` (two-loop recursion; Nocedal & Wright
/// 2006, Algorithm 7.4) with `factor` as the initial inverse Hessian: the
/// bordered factor's step, corrected by the curvature the steps since have
/// seen. With an exact gradient the steps reach the same optimum; the
/// corrections make the convergence superlinear where the factor alone
/// converges linearly.
fn lbfgs_direction(
    factor: &Factor,
    pairs: &std::collections::VecDeque<(DVector<f64>, DVector<f64>, f64)>,
    mut q: DVector<f64>,
) -> DVector<f64> {
    let mut alpha = Vec::with_capacity(pairs.len());
    for (s, y, rho) in pairs.iter().rev() {
        let a = rho * s.dot(&q);
        q.axpy(-a, y, 1.0);
        alpha.push(a);
    }
    let mut r = factor.solve(&q);
    for ((s, y, rho), a) in pairs.iter().zip(alpha.iter().rev()) {
        let b = rho * y.dot(&r);
        r.axpy(a - b, s, 1.0);
    }
    r
}

/// Rows of a [`weighted_gram`] tile: the tile and its weighted copy stay in
/// cache while one matrix product adds them in.
const GRAM_TILE: usize = 256;

/// `Σ_r w_r z_r z_rᵀ`, the `z_r` contiguous in `rows`, a tile of rows at a
/// time as one matrix product `Z_tᵀ (W_t Z_t)`: a blocked kernel instead of
/// one rank-one update per row, so each entry of the result is loaded once
/// per tile rather than once per row.
pub(super) fn weighted_gram(rows: &[f64], d: usize, w: &[f64]) -> DMatrix<f64> {
    let mut h = DMatrix::<f64>::zeros(d, d);
    if d == 0 {
        return h;
    }
    let mut scaled = vec![0.0; GRAM_TILE * d];
    for (tile, wt) in rows.chunks(GRAM_TILE * d).zip(w.chunks(GRAM_TILE)) {
        let r = wt.len();
        for ((s, z), &wr) in scaled.chunks_exact_mut(d).zip(tile.chunks_exact(d)).zip(wt) {
            for (a, &b) in s.iter_mut().zip(z) {
                *a = wr * b;
            }
        }
        // Column-major views: `zt` is `d × r` (row r of the tile is its
        // column r), `zw` the `r × d` weighted rows.
        let zt = nalgebra::DMatrixView::from_slice(tile, d, r);
        let zw = nalgebra::DMatrixView::from_slice_with_strides(&scaled[..r * d], r, d, d, 1);
        h.gemm(1.0, &zt, &zw, 1.0);
    }
    h
}

/// `log(1 + e^x)` without overflow.
fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn factor_spd(a: DMatrix<f64>) -> anyhow::Result<nalgebra::Cholesky<f64, nalgebra::Dyn>> {
    let d = a.nrows();
    match a.clone().cholesky() {
        Some(ch) => Ok(ch),
        None => {
            let mut a = a;
            let bump = 1e-6 * (0..d).map(|j| a[(j, j)].abs()).fold(1.0, f64::max);
            for j in 0..d {
                a[(j, j)] += bump;
            }
            a.cholesky()
                .ok_or_else(|| anyhow::anyhow!("nuisance system is not positive definite"))
        }
    }
}
