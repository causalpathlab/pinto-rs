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

/// Ridge regression of `y` on `[X | cols]` over `train`, reusing the
/// precomputed `XᵀX` for those rows and bordering it with the extra
/// columns. `X` is the first `p = xtx.nrows()` columns of `x`, so a nested
/// design reuses the full matrix. The intercept (column 0) is unpenalized.
/// Returns `p + q` coefficients.
pub(super) fn ridge_bordered(
    x: &DMatrix<f32>,
    xtx: &DMatrix<f64>,
    cols: &[&[f64]],
    y: &[f64],
    train: &[usize],
    ridge: f64,
) -> anyhow::Result<DVector<f64>> {
    use super::overlap::{gather_rows, ROW_BLOCK};
    let p = xtx.nrows();
    debug_assert!(p <= x.ncols());
    let q = cols.len();
    let d = p + q;
    // One pass over the training rows for every cross-product involving
    // the extra columns and `y`; chunked in parallel, rows gathered a
    // block at a time.
    let chunk = 8192;
    let partial: Vec<(DMatrix<f64>, DVector<f64>)> = train
        .par_chunks(chunk)
        .map(|rows| {
            let mut a = DMatrix::<f64>::zeros(d, q);
            let mut b = DVector::<f64>::zeros(d);
            let mut zi = vec![0.0; d];
            let mut buf = Vec::new();
            for block in rows.chunks(ROW_BLOCK) {
                gather_rows(x, p, block, &mut buf);
                for (xi, &i) in buf.chunks_exact(p.max(1)).zip(block) {
                    zi[..p].copy_from_slice(&xi[..p]);
                    for (c, col) in cols.iter().enumerate() {
                        zi[p + c] = col[i];
                    }
                    let yi = y[i];
                    for r in 0..d {
                        b[r] += zi[r] * yi;
                        for c in 0..q {
                            a[(r, c)] += zi[r] * zi[p + c];
                        }
                    }
                }
            }
            (a, b)
        })
        .collect();
    let mut a = DMatrix::<f64>::zeros(d, q);
    let mut b = DVector::<f64>::zeros(d);
    for (pa, pb) in partial {
        a += pa;
        b += pb;
    }
    let mut g = DMatrix::<f64>::zeros(d, d);
    g.view_mut((0, 0), (p, p)).copy_from(xtx);
    for c in 0..q {
        for r in 0..d {
            g[(r, p + c)] = a[(r, c)];
            g[(p + c, r)] = a[(r, c)];
        }
    }
    let scale = train.len().max(1) as f64;
    for j in 1..d {
        g[(j, j)] += ridge;
    }
    // A tiny relative jitter keeps an all-zero column (a modifier absent
    // from the training rows) from making the system singular.
    for j in 0..d {
        g[(j, j)] += 1e-9 * scale;
    }
    solve_spd(g, b)
}

/// Logistic ridge by iteratively reweighted least squares on `[X | cols]`
/// over `rows`, `X` being the first `p` columns of `x`. Intercept
/// unpenalized. `init` starts Newton from a nearby solution (another
/// fold's): the penalized problem is strictly convex, so the start changes
/// only how many steps reach the one optimum.
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
    let m = rows.len();
    // Column by column: both `x` and `z` are column-major.
    let mut z = DMatrix::<f64>::zeros(m, d);
    for j in 0..p {
        let src = x.column(j);
        let src = src.as_slice();
        for (dst, &i) in z.column_mut(j).iter_mut().zip(rows) {
            *dst = src[i] as f64;
        }
    }
    for (c, col) in cols.iter().enumerate() {
        for (dst, &i) in z.column_mut(p + c).iter_mut().zip(rows) {
            *dst = col[i];
        }
    }
    let tt: DVector<f64> = DVector::from_iterator(m, rows.iter().map(|&i| t[i]));
    let mean_t = tt.sum() / m.max(1) as f64;
    let mut beta = match init {
        Some(b) if b.len() == d && b.iter().all(|v| v.is_finite()) => b.clone(),
        _ => {
            // Start at the marginal log-odds, clamped away from separation.
            let mut b = DVector::<f64>::zeros(d);
            let m0 = mean_t.clamp(1e-4, 1.0 - 1e-4);
            b[0] = (m0 / (1.0 - m0)).ln();
            b
        }
    };
    // Penalized log-likelihood, the objective Newton climbs.
    let objective = |beta: &DVector<f64>, eta: &DVector<f64>| -> f64 {
        let ll: f64 = eta
            .iter()
            .zip(tt.iter())
            .map(|(&e, &t)| t * e - softplus(e))
            .sum();
        ll - 0.5 * ridge * beta.iter().skip(1).map(|b| b * b).sum::<f64>()
    };
    let mut zw = z.clone();
    // Zᵀ once: the Hessian is a blocked product with it every step
    // (`tr_mul` takes column dot products, about 8× slower here).
    let zt = z.transpose();
    let mut eta = &z * &beta;
    let mut obj = objective(&beta, &eta);
    let mut w = DVector::<f64>::zeros(m);
    let mut grad_r = DVector::<f64>::zeros(m);
    let mut trial = beta.clone();
    let mut eta_t = eta.clone();
    for _ in 0..50 {
        for r in 0..m {
            let e = sigmoid(eta[r]);
            w[r] = (e * (1.0 - e)).max(1e-10);
            grad_r[r] = tt[r] - e;
        }
        // Newton step: (ZᵀWZ + λD) Δ = Zᵀ(t − e) − λDβ.
        zw.copy_from(&z);
        for (r, mut row) in zw.row_iter_mut().enumerate() {
            row *= w[r];
        }
        let mut hmat = &zt * &zw;
        let mut g = &zt * &grad_r;
        for j in 1..d {
            hmat[(j, j)] += ridge;
            g[j] -= ridge * beta[j];
        }
        for j in 0..d {
            hmat[(j, j)] += 1e-9 * m as f64;
        }
        let mut step = solve_spd(hmat, g)?;
        // Halve the step until the objective does not fall: undamped Newton
        // can oscillate on a nearly separated fold and never converge;
        // halving keeps every step uphill on the concave objective, so the
        // iterations reach its one optimum.
        let mut accepted = false;
        for _ in 0..30 {
            trial.copy_from(&beta);
            trial += &step;
            eta_t.gemv(1.0, &z, &trial, 0.0);
            let obj_t = objective(&trial, &eta_t);
            if obj_t >= obj - 1e-12 * (1.0 + obj.abs()) {
                let gain = obj_t - obj;
                std::mem::swap(&mut beta, &mut trial);
                std::mem::swap(&mut eta, &mut eta_t);
                obj = obj_t;
                accepted = true;
                if step.amax() < 1e-8 || gain.abs() < 1e-12 * (1.0 + obj.abs()) {
                    return Ok(beta);
                }
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            // No ascent left at machine precision: at the optimum.
            break;
        }
    }
    Ok(beta)
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

fn solve_spd(a: DMatrix<f64>, b: DVector<f64>) -> anyhow::Result<DVector<f64>> {
    let d = a.nrows();
    match a.clone().cholesky() {
        Some(ch) => Ok(ch.solve(&b)),
        None => {
            let mut a = a;
            let bump = 1e-6 * (0..d).map(|j| a[(j, j)].abs()).fold(1.0, f64::max);
            for j in 0..d {
                a[(j, j)] += bump;
            }
            a.cholesky()
                .map(|ch| ch.solve(&b))
                .ok_or_else(|| anyhow::anyhow!("nuisance system is not positive definite"))
        }
    }
}
