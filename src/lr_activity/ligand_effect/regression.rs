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
    use super::overlap::{gather_rows, ROW_BLOCK};
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
            let mut zi = vec![0.0; d];
            let mut buf = Vec::new();
            for block in rows.chunks(ROW_BLOCK) {
                gather_rows(x, p, block, &mut buf);
                for (xi, &i) in buf.chunks_exact(p.max(1)).zip(block) {
                    zi[..p].copy_from_slice(&xi[..p]);
                    for (c, col) in cols.iter().enumerate() {
                        zi[p + c] = col[i];
                    }
                    let (a, b) = &mut sums[fold[i] as usize];
                    let yi = y[i];
                    for r in 0..d {
                        b[r] += zi[r] * yi;
                    }
                    for c in 0..q {
                        let zc = zi[p + c];
                        let out = a.column_mut(c);
                        for (o, &z) in out.into_iter().zip(&zi) {
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
/// is a guard ([`super::estimate::timing::newton`] counts the fits it stops).
const MAX_NEWTON_STEPS: usize = 500;

/// Largest coefficient change at which a Newton step's Hessian may serve the
/// next step: close to the optimum the weights barely move.
const REUSE_HESSIAN_BELOW: f64 = 0.05;

thread_local! {
    /// A logistic fit's rows, kept between fits: allocating them anew for
    /// every fit costs a page fault per page, and many workers fault at once.
    static LOGISTIC_ROWS: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Logistic ridge by iteratively reweighted least squares on `[X | cols]`
/// over `rows`, `X` being the first `p` columns of `x`. Intercept
/// unpenalized. `init` starts Newton from a nearby solution (the same fold's
/// on the previous draw): the penalized problem is strictly convex, so the
/// start changes only how many steps reach the one optimum.
///
/// Memory traffic, not arithmetic, is what limits many fits side by side, so
/// the rows are held once, row-major, and one pass gives the objective, the
/// gradient and the weights; and close to the optimum a step reuses the last
/// Hessian's factor, which moves the iterate as far toward the same optimum
/// (the gradient is exact) at a fraction of the cost, for as long as steps
/// keep shrinking.
pub(super) fn logistic_ridge(
    x: &DMatrix<f32>,
    p: usize,
    cols: &[&[f64]],
    t: &[f64],
    rows: &[usize],
    ridge: f64,
    init: Option<&DVector<f64>>,
) -> anyhow::Result<DVector<f64>> {
    debug_assert!(p <= x.ncols());
    let d = p + cols.len();
    LOGISTIC_ROWS.with_borrow_mut(|z| {
        gather_design_rows(x, p, cols, rows, z);
        let tt: Vec<f64> = rows.iter().map(|&i| t[i]).collect();
        newton_on_rows(z, d, &tt, ridge, init)
    })
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
    use super::overlap::ROW_BLOCK;
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
        let d = beta.len();
        self.gradient.fill(0.0);
        #[allow(unused_mut)]
        let mut ll = None;
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the check above found the feature the function enables.
            ll = Some(unsafe { row_pass_avx2(z, t, beta, &mut self.gradient, &mut self.weights) });
        }
        let ll = ll.unwrap_or_else(|| row_pass(z, t, beta, &mut self.gradient, &mut self.weights));
        let mut penalty = 0.0;
        for (b, g) in beta.iter().zip(self.gradient.iter_mut()).take(d).skip(1) {
            penalty += b * b;
            *g -= ridge * b;
        }
        self.objective = ll - 0.5 * ridge * penalty;
    }
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

/// Newton on rows `z` (row-major, `d` columns); see [`logistic_ridge`].
fn newton_on_rows(
    z: &[f64],
    d: usize,
    t: &[f64],
    ridge: f64,
    init: Option<&DVector<f64>>,
) -> anyhow::Result<DVector<f64>> {
    use super::estimate::timing;
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
    let mut factor = None;
    let mut last_amax = f64::INFINITY;
    let mut hessians = 0usize;
    let mut converged = false;
    for _ in 0..MAX_NEWTON_STEPS {
        let reused = factor.is_some();
        let chol: &nalgebra::Cholesky<f64, nalgebra::Dyn> = match &factor {
            Some(c) => c,
            None => {
                hessians += 1;
                // Newton step: (ZᵀWZ + λD) Δ = Zᵀ(t − e) − λDβ.
                let mut hmat = weighted_gram(z, d, &at.weights);
                for j in 1..d {
                    hmat[(j, j)] += ridge;
                }
                for j in 0..d {
                    hmat[(j, j)] += 1e-9 * m as f64;
                }
                factor.insert(factor_spd(hmat)?)
            }
        };
        let mut step = chol.solve(&DVector::from_column_slice(&at.gradient));
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
                factor = None;
                continue;
            }
            converged = true;
            break;
        }
        let gain = next.objective - at.objective;
        std::mem::swap(&mut beta, &mut trial);
        std::mem::swap(&mut at, &mut next);
        let amax = step.amax();
        // A reused factor's step says less about the distance left than a
        // fresh one's, so only the step's size may end the fit then.
        let small_gain = !reused && gain.abs() < 1e-12 * (1.0 + at.objective.abs());
        if amax < 1e-8 || small_gain {
            converged = true;
            break;
        }
        // Keep the factor only while the steps shrink fast.
        let shrinking = amax < REUSE_HESSIAN_BELOW && amax < 0.5 * last_amax && !halved;
        if !shrinking {
            factor = None;
        }
        last_amax = amax;
    }
    timing::newton(hessians, converged);
    Ok(beta)
}

/// `Σ_r w_r z_r z_rᵀ`, the `z_r` contiguous in `rows`: each row adds to the
/// upper triangle, which is then mirrored. Every entry is a sum over the rows
/// in order, without fused multiply-add, so wider vectors give the same bits.
pub(super) fn weighted_gram(rows: &[f64], d: usize, w: &[f64]) -> DMatrix<f64> {
    let mut h = vec![0.0; d * d];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the check above found the feature the function enables.
        unsafe { accumulate_gram_avx2(rows, d, w, &mut h) };
        return mirrored(h, d);
    }
    accumulate_gram(rows, d, w, &mut h);
    mirrored(h, d)
}

#[inline(always)]
fn accumulate_gram(rows: &[f64], d: usize, w: &[f64], h: &mut [f64]) {
    for (z, &wr) in rows.chunks_exact(d.max(1)).zip(w) {
        for a in 0..d {
            let za = wr * z[a];
            let out = &mut h[a * d + a..(a + 1) * d];
            for (o, &zb) in out.iter_mut().zip(&z[a..]) {
                *o += za * zb;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn accumulate_gram_avx2(rows: &[f64], d: usize, w: &[f64], h: &mut [f64]) {
    accumulate_gram(rows, d, w, h);
}

/// The symmetric matrix of the upper triangle of `h`, row-major.
fn mirrored(mut h: Vec<f64>, d: usize) -> DMatrix<f64> {
    for a in 0..d {
        for b in 0..a {
            h[a * d + b] = h[b * d + a];
        }
    }
    DMatrix::from_vec(d, d, h)
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
