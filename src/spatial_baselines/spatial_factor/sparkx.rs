//! SPARK-X (Zhu, Sun and Zhou 2021, "SPARK-X: non-parametric modeling
//! enables scalable and robust detection of spatial expression patterns
//! for large spatial transcriptomic studies", Genome Biology 22:184,
//! doi:10.1186/s13059-021-02404-0), written from the paper's Methods: a
//! covariance test of each gene's expression against the coordinates,
//! linear in the cells.
//!
//! For a gene with counts `y` over `n` cells, `H` the centring matrix
//! (here on each section's indicator, the paper's covariate form, so each
//! section is centred apart) and a coordinate matrix `S` (n × d), the
//! projection kernel `Sigma = S (S^T S)^{-1} S^T`, and the statistic
//! `T = y^T H Sigma H y / (y^T H y / n)`, which under independence follows
//! `sum_j lambda_j chi^2_1`, `lambda_j` the eigenvalues of `(S^T S)^{-1}
//! S^T H S` (the paper's `n^2 tr(E_C Sigma_C) / n` with `y` scaled to unit
//! variance). Eleven kernels: the coordinates themselves (each axis
//! centred and scaled within its section), five Gaussian transforms
//! `exp(-s^2 / (2 sigma^2))` and five cosine transforms `cos(2 pi s /
//! phi)`, `sigma` and `phi` per axis at the 20%, 40%, 60%, 80% and 100%
//! quantiles of `|s|`. Each kernel's p-value is combined by the Cauchy
//! rule (Liu and Xie 2020, JASA 115:393, doi:10.1080/01621459.2018.1554485),
//! equal weights. Only `y^T y`, the per-section sums of `y` and `(HS)^T y`
//! are read per gene, from its nonzero counts.
//!
//! Deviation: the tail of `sum_j lambda_j chi^2_1` is computed exactly by
//! one-dimensional quadrature per weight beyond the first (two weights in
//! 2-D) in place of Davies' algorithm (Davies 1980, JRSS C 29:323), which
//! computes the same probability.
//!
//! SpatialPCA declares genes significant at a Benjamini-Yekutieli FDR of
//! 0.05 (Benjamini and Yekutieli 2001, Annals of Statistics 29:1165,
//! doi:10.1214/aos/1013699998) and models up to 3000 of them
//! ([`significant`]).

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CscMatrix;
use rayon::prelude::*;

/// Quantiles of `|s|` the transforms' parameters are taken at.
const QUANTILES: [f64; 5] = [0.2, 0.4, 0.6, 0.8, 1.0];
/// FDR at which a gene is spatially variable.
pub const FDR: f64 = 0.05;
/// Gauss-Legendre nodes per panel of the tail quadrature.
const GL_NODES: usize = 16;
/// Panels of the tail quadrature, graded toward the end where the
/// integrand peaks.
const PANELS: usize = 14;

/// One kernel: `(H S)` (n × d), `(S^T S)^{-1}` and the null's weights.
struct Kernel {
    hs: DMatrix<f64>,
    w: DMatrix<f64>,
    lambda: Vec<f64>,
}

fn kernel(s: DMatrix<f64>, section: &[usize], n_sections: usize) -> Kernel {
    let hs = centred(&s, section, n_sections);
    let g = s.transpose() * &s;
    let w = g.clone().try_inverse().unwrap_or_else(|| {
        // a degenerate transform (a constant column): its pseudo-inverse
        g.clone().pseudo_inverse(1e-12).expect("a symmetric matrix")
    });
    let a = hs.transpose() * &hs;
    // the eigenvalues of W A, symmetrised through W's square root
    let e = w.clone().symmetric_eigen();
    let root = &e.eigenvectors
        * DMatrix::from_diagonal(&e.eigenvalues.map(|v| v.max(0.0).sqrt()))
        * e.eigenvectors.transpose();
    let m = &root * a * &root;
    let mut lambda: Vec<f64> = m
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .filter(|&v| v > 1e-10)
        .collect();
    lambda.sort_by(|a, b| b.total_cmp(a));
    Kernel { hs, w, lambda }
}

/// Columns of `s` centred within each section.
fn centred(s: &DMatrix<f64>, section: &[usize], n_sections: usize) -> DMatrix<f64> {
    let mut sum = DMatrix::<f64>::zeros(n_sections, s.ncols());
    let mut count = vec![0f64; n_sections];
    for (c, &g) in section.iter().enumerate() {
        count[g] += 1.0;
        for j in 0..s.ncols() {
            sum[(g, j)] += s[(c, j)];
        }
    }
    DMatrix::from_fn(s.nrows(), s.ncols(), |c, j| {
        s[(c, j)] - sum[(section[c], j)] / count[section[c]].max(1.0)
    })
}

/// R's default quantile (type 7) of sorted `x` at `p`.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let h = (sorted.len() as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// The eleven kernels of coordinates `xy` (cells × d): each axis centred
/// and scaled within its section, then as they are, Gaussian- and
/// cosine-transformed.
fn kernels(xy: &DMatrix<f64>, section: &[usize], n_sections: usize) -> Vec<Kernel> {
    let (n, d) = xy.shape();
    let mut s = centred(xy, section, n_sections);
    let mut ss = DMatrix::<f64>::zeros(n_sections, d);
    let mut count = vec![0f64; n_sections];
    for (c, &g) in section.iter().enumerate() {
        count[g] += 1.0;
        for j in 0..d {
            ss[(g, j)] += s[(c, j)] * s[(c, j)];
        }
    }
    for c in 0..n {
        let g = section[c];
        for j in 0..d {
            let sd = (ss[(g, j)] / (count[g] - 1.0).max(1.0)).sqrt();
            if sd > 0.0 {
                s[(c, j)] /= sd;
            }
        }
    }
    let abs_sorted: Vec<Vec<f64>> = (0..d)
        .map(|j| {
            let mut v: Vec<f64> = s.column(j).iter().map(|x| x.abs()).collect();
            v.sort_by(f64::total_cmp);
            v
        })
        .collect();
    let mut out = vec![kernel(s.clone(), section, n_sections)];
    for &q in &QUANTILES {
        let par: Vec<f64> = abs_sorted
            .iter()
            .map(|v| quantile(v, q).max(1e-12))
            .collect();
        let gauss = DMatrix::from_fn(n, d, |c, j| {
            (-s[(c, j)].powi(2) / (2.0 * par[j].powi(2))).exp()
        });
        out.push(kernel(gauss, section, n_sections));
    }
    for &q in &QUANTILES {
        let par: Vec<f64> = abs_sorted
            .iter()
            .map(|v| quantile(v, q).max(1e-12))
            .collect();
        let cosine = DMatrix::from_fn(n, d, |c, j| {
            (2.0 * std::f64::consts::PI * s[(c, j)] / par[j]).cos()
        });
        out.push(kernel(cosine, section, n_sections));
    }
    out
}

/// Gauss-Legendre nodes and weights on `[-1, 1]` (Newton on the Legendre
/// polynomial from the Chebyshev guesses).
fn gauss_legendre(k: usize) -> Vec<(f64, f64)> {
    (0..k)
        .map(|i| {
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (k as f64 + 0.5)).cos();
            loop {
                let (mut p0, mut p1) = (1.0, x);
                for m in 2..=k {
                    let p2 = ((2 * m - 1) as f64 * x * p1 - (m - 1) as f64 * p0) / m as f64;
                    p0 = p1;
                    p1 = p2;
                }
                let dp = k as f64 * (x * p1 - p0) / (x * x - 1.0);
                let dx = p1 / dp;
                x -= dx;
                if dx.abs() < 1e-15 {
                    let (mut p0, mut p1) = (1.0, x);
                    for m in 2..=k {
                        let p2 = ((2 * m - 1) as f64 * x * p1 - (m - 1) as f64 * p0) / m as f64;
                        p0 = p1;
                        p1 = p2;
                    }
                    let dp = k as f64 * (x * p1 - p0) / (x * x - 1.0);
                    return (x, 2.0 / ((1.0 - x * x) * dp * dp));
                }
            }
        })
        .collect()
}

fn erfc(x: f64) -> f64 {
    special::Error::compl_error(x)
}

/// `P(sum_j lambda_j Z_j^2 > q)` for standard normal `Z_j`, `lambda`
/// positive and in decreasing order: with `z_max = sqrt(q / lambda_1)`,
/// `P(lambda_1 Z^2 > q) + 2 int_0^{z_max} phi(z) P(rest > q - lambda_1
/// z^2) dz`, the integral on `z = z_max sin(t)` (smooth in `t`) by
/// Gauss-Legendre panels graded toward `t = pi / 2`.
pub fn chisq_mix_tail(lambda: &[f64], q: f64) -> f64 {
    if q <= 0.0 {
        return 1.0;
    }
    match lambda {
        [] => 0.0,
        [l] => erfc((q / (2.0 * l)).sqrt()),
        [l1, rest @ ..] => {
            let gl = gauss_legendre(GL_NODES);
            let zmax = (q / l1).sqrt();
            let mut edges = vec![0.0];
            edges.extend(
                (1..PANELS).map(|k| std::f64::consts::FRAC_PI_2 * (1.0 - 0.5f64.powi(k as i32))),
            );
            edges.push(std::f64::consts::FRAC_PI_2);
            let mut integral = 0.0;
            for w in edges.windows(2) {
                let (a, b) = (w[0], w[1]);
                for &(x, wt) in &gl {
                    let t = 0.5 * (b - a) * x + 0.5 * (a + b);
                    let z = zmax * t.sin();
                    let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
                    let c = t.cos();
                    integral +=
                        0.5 * (b - a) * wt * phi * chisq_mix_tail(rest, q * c * c) * zmax * c;
                }
            }
            (erfc(zmax / std::f64::consts::SQRT_2) + 2.0 * integral).min(1.0)
        }
    }
}

/// The Cauchy combination of `p` with equal weights: `T = mean tan((1/2 -
/// p) pi)`, `p = 1/2 - arctan(T) / pi`, in forms exact for small `p`.
pub fn cauchy_combine(p: &[f64]) -> f64 {
    let t: f64 = p
        .iter()
        .map(|&pi| {
            let pi = pi.clamp(1e-300, 1.0);
            // tan((1/2 - p) pi) = 1 / tan(p pi)
            1.0 / (pi * std::f64::consts::PI).tan()
        })
        .sum::<f64>()
        / p.len() as f64;
    if t > 1.0 {
        (1.0 / t).atan() / std::f64::consts::PI
    } else {
        0.5 - t.atan() / std::f64::consts::PI
    }
}

/// Benjamini-Yekutieli adjusted p-values.
pub fn by_adjust(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let cm: f64 = (1..=m).map(|i| 1.0 / i as f64).sum();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut adj = vec![1.0; m];
    let mut running = 1f64;
    for (rank, &i) in order.iter().enumerate().rev() {
        running = running
            .min(p[i] * m as f64 * cm / (rank + 1) as f64)
            .min(1.0);
        adj[i] = running;
    }
    adj
}

/// Each gene's statistic and null weights under each kernel (none for a
/// gene with no variance within its sections), from `counts` (genes ×
/// cells) at coordinates `xy` (cells × d) in sections `section`.
pub fn kernel_stats(
    counts: &CscMatrix<f32>,
    xy: &DMatrix<f64>,
    section: &[usize],
    n_sections: usize,
) -> Vec<Vec<(f64, Vec<f64>)>> {
    let (m, n) = (counts.nrows(), counts.ncols());
    let ks = kernels(xy, section, n_sections);
    let d = xy.ncols();
    let width = ks.len() * d;
    // per gene: sum y^2, the per-section sums and (HS_k)^T y for every k
    let stride = 1 + n_sections + width;
    let cols: Vec<(usize, _)> = counts.col_iter().enumerate().collect();
    let sums: Vec<f64> = cols
        .par_chunks(1024)
        .map(|chunk| {
            let mut acc = vec![0f64; m * stride];
            for (c, col) in chunk {
                let g = section[*c];
                for (&gene, &v) in col.row_indices().iter().zip(col.values()) {
                    let v = v as f64;
                    let row = &mut acc[gene * stride..(gene + 1) * stride];
                    row[0] += v * v;
                    row[1 + g] += v;
                    for (k, ker) in ks.iter().enumerate() {
                        for j in 0..d {
                            row[1 + n_sections + k * d + j] += v * ker.hs[(*c, j)];
                        }
                    }
                }
            }
            acc
        })
        .reduce(
            || vec![0f64; m * stride],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    let mut size = vec![0f64; n_sections];
    for &g in section {
        size[g] += 1.0;
    }
    (0..m)
        .into_par_iter()
        .map(|gene| {
            let row = &sums[gene * stride..(gene + 1) * stride];
            let yhy = row[0]
                - (0..n_sections)
                    .map(|g| row[1 + g] * row[1 + g] / size[g].max(1.0))
                    .sum::<f64>();
            if yhy <= 1e-12 * row[0].max(1e-300) {
                return Vec::new();
            }
            ks.iter()
                .enumerate()
                .map(|(k, ker)| {
                    let u =
                        DVector::from_iterator(d, (0..d).map(|j| row[1 + n_sections + k * d + j]));
                    (u.dot(&(&ker.w * &u)) / (yhy / n as f64), ker.lambda.clone())
                })
                .collect()
        })
        .collect()
}

/// Each gene's p-value, its kernels' combined by the Cauchy rule (1 for a
/// gene with no variance within its sections).
pub fn sparkx(
    counts: &CscMatrix<f32>,
    xy: &DMatrix<f64>,
    section: &[usize],
    n_sections: usize,
) -> Vec<f64> {
    kernel_stats(counts, xy, section, n_sections)
        .into_par_iter()
        .map(|stats| {
            if stats.is_empty() {
                return 1.0;
            }
            let p: Vec<f64> = stats
                .iter()
                .map(|(t, lambda)| chisq_mix_tail(lambda, *t))
                .collect();
            cauchy_combine(&p)
        })
        .collect()
}

/// The genes SpatialPCA models: those with a Benjamini-Yekutieli adjusted
/// p-value below [`FDR`] among `p` (tested genes), the `at_most` smallest
/// p-values, in increasing order of p; and how many were significant.
pub fn significant(p: &[f64], at_most: usize) -> (Vec<usize>, usize) {
    let adj = by_adjust(p);
    let mut genes: Vec<usize> = (0..p.len()).filter(|&g| adj[g] < FDR).collect();
    let count = genes.len();
    genes.sort_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)));
    genes.truncate(at_most);
    (genes, count)
}
