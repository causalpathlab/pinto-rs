//! The numerical kit both engines share, written from the cited
//! algorithms: Brent's one-dimensional minimiser, a projected L-BFGS for
//! box constraints, and subspace iteration for the top eigenpairs of a
//! symmetric operator given only its products.

use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// Dense f64 matrix, column-major.
pub type Dense = DMatrix<f64>;

/// Brent's minimiser on `[a, b]` (Brent 1973, "Algorithms for
/// Minimization without Derivatives", Prentice-Hall, chapter 5: golden
/// section with parabolic interpolation), to absolute tolerance `tol` on
/// `x`, at most `max_eval` evaluations. The minimiser, its value and the
/// evaluations used.
pub fn brent_min(
    mut f: impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_eval: usize,
) -> (f64, f64, usize) {
    let golden = 0.5 * (3.0 - 5f64.sqrt());
    let (mut a, mut b) = (a.min(b), a.max(b));
    let mut x = a + golden * (b - a);
    let (mut w, mut v) = (x, x);
    let mut fx = f(x);
    let (mut fw, mut fv) = (fx, fx);
    let mut evals = 1;
    let (mut d, mut e) = (0f64, 0f64);
    while evals < max_eval {
        let m = 0.5 * (a + b);
        let tol1 = 1e-10 * x.abs() + tol / 3.0;
        let tol2 = 2.0 * tol1;
        if (x - m).abs() <= tol2 - 0.5 * (b - a) {
            break;
        }
        let mut golden_step = true;
        if e.abs() > tol1 {
            // parabola through x, w, v
            let r = (x - w) * (fx - fv);
            let q0 = (x - v) * (fx - fw);
            let mut p = (x - v) * q0 - (x - w) * r;
            let mut q = 2.0 * (q0 - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            if p.abs() < (0.5 * q * e).abs() && p > q * (a - x) && p < q * (b - x) {
                e = d;
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = if x < m { tol1 } else { -tol1 };
                }
                golden_step = false;
            }
        }
        if golden_step {
            e = if x < m { b - x } else { a - x };
            d = golden * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u);
        evals += 1;
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            (v, fv, w, fw, x, fx) = (w, fw, x, fx, u, fu);
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                (v, fv, w, fw) = (w, fw, u, fu);
            } else if fu <= fv || v == x || v == w {
                (v, fv) = (u, fu);
            }
        }
    }
    (x, fx, evals)
}

/// A grid search then Brent: `f` at each of `grid` (sorted ascending), then
/// [`brent_min`] on the bracket of the best grid point's neighbours. The
/// minimiser, its value and the evaluations used.
pub fn grid_then_brent(
    mut f: impl FnMut(f64) -> f64,
    grid: &[f64],
    tol: f64,
    max_eval: usize,
) -> (f64, f64, usize) {
    let values: Vec<f64> = grid.iter().map(|&x| f(x)).collect();
    let best = (0..grid.len())
        .min_by(|&i, &j| values[i].total_cmp(&values[j]))
        .expect("a grid of at least one point");
    if grid.len() < 3 {
        return (grid[best], values[best], grid.len());
    }
    let lo = grid[best.saturating_sub(1)];
    let hi = grid[(best + 1).min(grid.len() - 1)];
    let (x, fx, used) = brent_min(&mut f, lo, hi, tol, max_eval.saturating_sub(grid.len()));
    if fx <= values[best] {
        (x, fx, grid.len() + used)
    } else {
        (grid[best], values[best], grid.len() + used)
    }
}

/// What [`lbfgs_box`] found.
#[derive(Debug, Clone)]
pub struct BoxMin {
    pub x: Vec<f64>,
    pub evals: usize,
}

/// Minimise `fg` (value and gradient) over the box `[lo, hi]` from `x0`,
/// by a projected limited-memory BFGS: the L-BFGS two-loop direction
/// (Nocedal 1980, Math. Comp. 35:773, doi:10.1090/S0025-5718-1980-0572855-7)
/// on the variables off their bounds, projected back onto the box with a
/// backtracking Armijo search, in the spirit of L-BFGS-B (Byrd, Lu,
/// Nocedal and Zhu 1995, SIAM J. Sci. Comput. 16:1190,
/// doi:10.1137/0916069) without its generalised Cauchy point. Stops at a
/// projected gradient below `pgtol`, a relative change of `f` below
/// `ftol`, or `max_evals` evaluations. `fg` returns, beside the value and
/// gradient, what its evaluation computed (`T`); the result carries that
/// of the point it stops at, so no evaluation is repeated there.
pub fn lbfgs_box<T>(
    mut fg: impl FnMut(&[f64]) -> (f64, Vec<f64>, T),
    x0: &[f64],
    lo: &[f64],
    hi: &[f64],
    max_evals: usize,
    pgtol: f64,
    ftol: f64,
) -> (BoxMin, T) {
    const MEMORY: usize = 8;
    let p = x0.len();
    let clamp = |x: &mut [f64]| {
        for i in 0..x.len() {
            x[i] = x[i].clamp(lo[i], hi[i]);
        }
    };
    let projected = |x: &[f64], g: &[f64]| -> Vec<f64> {
        (0..p)
            .map(|i| {
                if (x[i] <= lo[i] && g[i] > 0.0) || (x[i] >= hi[i] && g[i] < 0.0) {
                    0.0
                } else {
                    g[i]
                }
            })
            .collect()
    };
    let mut x = x0.to_vec();
    clamp(&mut x);
    let (mut f, mut g, mut at) = fg(&x);
    let mut evals = 1;
    let mut hist: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    let norm_inf = |v: &[f64]| v.iter().fold(0f64, |m, &a| m.max(a.abs()));
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    loop {
        let pg = projected(&x, &g);
        if norm_inf(&pg) < pgtol || evals >= max_evals {
            return (BoxMin { x, evals }, at);
        }
        // two-loop recursion on the projected gradient
        let mut q = pg.clone();
        let mut alphas = Vec::with_capacity(hist.len());
        for (s, y, rho) in hist.iter().rev() {
            let a = rho * dot(s, &q);
            q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
            alphas.push(a);
        }
        if let Some((s, y, _)) = hist.last() {
            let gamma = dot(s, y) / dot(y, y);
            q.iter_mut().for_each(|v| *v *= gamma);
        }
        for ((s, y, rho), a) in hist.iter().zip(alphas.iter().rev()) {
            let b = rho * dot(y, &q);
            q.iter_mut().zip(s).for_each(|(qi, si)| *qi += (a - b) * si);
        }
        let mut d: Vec<f64> = q
            .iter()
            .zip(&pg)
            .map(|(&v, &pgi)| if pgi == 0.0 { 0.0 } else { -v })
            .collect();
        if dot(&d, &g) >= 0.0 {
            d = pg.iter().map(|v| -v).collect();
        }
        // the first step moves no variable by more than one unit
        let mut step = if hist.is_empty() {
            (1.0 / norm_inf(&d).max(1e-300)).min(1.0)
        } else {
            1.0
        };
        let mut accepted = None;
        for _ in 0..30 {
            if evals >= max_evals {
                break;
            }
            let mut xn: Vec<f64> = x.iter().zip(&d).map(|(xi, di)| xi + step * di).collect();
            clamp(&mut xn);
            let dx: Vec<f64> = xn.iter().zip(&x).map(|(a, b)| a - b).collect();
            if norm_inf(&dx) == 0.0 {
                break;
            }
            let (fnew, gnew, tnew) = fg(&xn);
            evals += 1;
            if fnew.is_finite() && fnew <= f + 1e-4 * dot(&g, &dx) {
                accepted = Some((xn, fnew, gnew, dx, tnew));
                break;
            }
            step *= 0.5;
        }
        let Some((xn, fnew, gnew, s, tnew)) = accepted else {
            return (BoxMin { x, evals }, at);
        };
        let y: Vec<f64> = gnew.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 * dot(&y, &y).max(1e-300) {
            hist.push((s, y, 1.0 / sy));
            if hist.len() > MEMORY {
                hist.remove(0);
            }
        }
        let change = (f - fnew).abs() / f.abs().max(fnew.abs()).max(1e-300);
        x = xn;
        f = fnew;
        g = gnew;
        at = tnew;
        if change < ftol {
            return (BoxMin { x, evals }, at);
        }
    }
}

/// An orthonormal basis of the columns of `v` (thin QR).
pub fn orthonormal(v: Dense) -> Dense {
    let k = v.ncols().min(v.nrows());
    let q = v.qr().q();
    q.columns(0, k).into_owned()
}

/// A seeded `rows × cols` standard normal matrix.
pub fn gaussian(rows: usize, cols: usize, seed: u64) -> Dense {
    let mut rng = SmallRng::seed_from_u64(seed);
    Dense::from_fn(rows, cols, |_, _| StandardNormal.sample(&mut rng))
}

/// Eigenpairs of a symmetric matrix, sorted by eigenvalue, largest first.
pub fn sym_eig_desc(a: &Dense) -> (Vec<f64>, Dense) {
    let sym = (a + a.transpose()) * 0.5;
    let eig = SymmetricEigen::new(sym);
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&i, &j| eig.eigenvalues[j].total_cmp(&eig.eigenvalues[i]));
    let vals = order.iter().map(|&i| eig.eigenvalues[i]).collect();
    let vecs = Dense::from_fn(a.nrows(), order.len(), |r, c| {
        eig.eigenvectors[(r, order[c])]
    });
    (vals, vecs)
}

/// Settings of [`top_eigs`].
#[derive(Debug, Clone, Copy)]
pub struct EigOpts {
    pub d: usize,
    pub block: usize,
    pub seed: u64,
    pub tol: f64,
    pub max_iter: usize,
}

/// Blocks of a restart's Krylov basis in [`top_eigs`].
const KRYLOV_DEPTH: usize = 3;

/// The top `d` eigenpairs of the symmetric positive semi-definite operator
/// `apply` on `dim`-vectors, given only its products with `dim × b`
/// blocks: a restarted block Krylov method (block Lanczos with full
/// reorthogonalisation, restarted from the best `block` Ritz vectors;
/// Saad 2011, "Numerical Methods for Large Eigenvalue Problems", SIAM,
/// doi:10.1137/1.9781611970739, chapters 5 and 6): each round
/// Rayleigh-Ritz on `[V, G V, G^2 V]`, from `start` (a warm block) or a
/// seeded Gaussian one (subspace iteration when `dim` has room for one
/// block only), until the top `d` Ritz values change by less than `tol`
/// relative, at most `max_iter` rounds. A `dim` no larger than
/// the block is solved densely. The eigenvalues, the eigenvectors (`dim ×
/// d`) and the final block, to warm-start the next call.
pub fn top_eigs(
    mut apply: impl FnMut(&Dense) -> Dense,
    dim: usize,
    start: Option<&Dense>,
    o: &EigOpts,
) -> (Vec<f64>, Dense, Dense) {
    let EigOpts {
        d,
        block,
        seed,
        tol,
        max_iter,
    } = *o;
    let d = d.min(dim);
    let max_iter = max_iter.max(1);
    if dim <= block {
        let g = apply(&Dense::identity(dim, dim));
        let (vals, vecs) = sym_eig_desc(&g);
        let top = vecs.columns(0, d).into_owned();
        return (vals[..d].to_vec(), top, vecs);
    }
    let mut v = match start {
        Some(s) if s.nrows() == dim && s.ncols() == block => orthonormal(s.clone()),
        _ => orthonormal(gaussian(dim, block, seed)),
    };
    let depth = (dim / block).clamp(1, KRYLOV_DEPTH);
    let mut prev: Option<Vec<f64>> = None;
    for it in 0..max_iter {
        // the block Krylov basis [V, G V, G^2 V, ...], each block
        // orthogonalised against those before it (twice) and orthonormal
        let mut basis: Vec<Dense> = vec![v.clone()];
        let mut products: Vec<Dense> = Vec::with_capacity(depth);
        for j in 0..depth {
            let w = apply(&basis[j]);
            if j + 1 < depth {
                let mut x = w.clone();
                for _ in 0..2 {
                    for q in &basis {
                        x -= q * (q.transpose() * &x);
                    }
                    x = orthonormal(x);
                }
                basis.push(x);
            }
            products.push(w);
        }
        let q = Dense::from_columns(
            &basis
                .iter()
                .flat_map(|b| b.column_iter())
                .collect::<Vec<_>>(),
        );
        let gq = Dense::from_columns(
            &products
                .iter()
                .flat_map(|b| b.column_iter())
                .collect::<Vec<_>>(),
        );
        let h = q.transpose() * &gq;
        let (vals, e) = sym_eig_desc(&h);
        let ritz = &q * &e;
        let converged = prev.as_ref().is_some_and(|p| {
            (0..d).all(|l| (vals[l] - p[l]).abs() <= tol * vals[l].abs().max(1e-300))
        });
        if converged || it + 1 == max_iter {
            if !converged {
                log::warn!("block Krylov: top {d} eigenvalues not converged in {max_iter} rounds");
            }
            let top = ritz.columns(0, d).into_owned();
            return (vals[..d].to_vec(), top, ritz.columns(0, block).into_owned());
        }
        prev = Some(vals);
        // with room for one block only, a power step: G applied to the
        // Ritz vectors
        v = match depth {
            1 => orthonormal((&gq * &e).columns(0, block).into_owned()),
            _ => orthonormal(ritz.columns(0, block).into_owned()),
        };
    }
    unreachable!("max_iter is at least 1")
}
