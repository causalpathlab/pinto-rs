//! SpatialPCA's preprocessing: SPARK-X's tests and null distribution
//! against direct computations, its detection of planted spatial genes;
//! SCTransform's per-gene fits against direct maximisation and its
//! residuals' variance on null genes, each with a broken variant the check
//! must catch.

use crate::spatial_baselines::spatial_factor::kit::{gaussian, Dense};
use crate::spatial_baselines::spatial_factor::sct::{
    fit_gene, nb_loglik, residuals, GeneCounts, GeneModel,
};
use crate::spatial_baselines::spatial_factor::sparkx::{
    by_adjust, cauchy_combine, chisq_mix_tail, kernel_stats, significant, sparkx,
};
use nalgebra::DMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// A draw from a negative binomial of mean `mu` and dispersion `theta`
/// (a gamma-Poisson mixture; `theta = inf` is Poisson).
fn nb(rng: &mut SmallRng, mu: f64, theta: f64) -> f64 {
    let lam = if theta.is_finite() {
        // gamma(theta, mu / theta) by Marsaglia and Tsang
        let (a, scale) = (theta, mu / theta);
        let boost = if a < 1.0 {
            rng.random_range(f64::MIN_POSITIVE..1.0).powf(1.0 / a)
        } else {
            1.0
        };
        let a = if a < 1.0 { a + 1.0 } else { a };
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let z = normal(rng);
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                break d * v * scale * boost;
            }
        }
    } else {
        mu
    };
    poisson(rng, lam)
}

fn normal(rng: &mut SmallRng) -> f64 {
    let u: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
    let v: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

fn poisson(rng: &mut SmallRng, lam: f64) -> f64 {
    if lam > 30.0 {
        return (lam + lam.sqrt() * normal(rng)).round().max(0.0);
    }
    let l = (-lam).exp();
    let (mut k, mut p) = (0.0, 1.0);
    loop {
        p *= rng.random_range(0.0..1.0f64);
        if p <= l {
            return k;
        }
        k += 1.0;
    }
}

fn csc(rows: &[Vec<f64>], n: usize) -> CscMatrix<f32> {
    let mut coo = CooMatrix::<f32>::new(rows.len(), n);
    for (g, r) in rows.iter().enumerate() {
        for (c, &v) in r.iter().enumerate() {
            if v != 0.0 {
                coo.push(g, c, v as f32);
            }
        }
    }
    CscMatrix::from(&coo)
}

fn sparse_gene(r: &[f64]) -> GeneCounts {
    r.iter()
        .enumerate()
        .filter(|(_, &v)| v != 0.0)
        .map(|(c, &v)| (c as u32, v))
        .collect()
}

/// The tail of a weighted sum of chi-squares: one weight's is the
/// chi-square's, equal weights' the exponential's (exact far into the
/// tail), unequal weights' the Monte Carlo frequency; without its first
/// term the tail is wrong.
#[test]
fn chisq_mixture_tail_is_exact() {
    for q in [0.5, 3.0, 20.0] {
        let want = special::Error::compl_error((q / 2.0f64).sqrt());
        assert!((chisq_mix_tail(&[1.0], q) - want).abs() < 1e-14);
    }
    for (l, q) in [(1.0, 3.0), (0.7, 30.0), (1.0, 200.0), (2.0, 900.0)] {
        let got = chisq_mix_tail(&[l, l], q);
        let want = (-q / (2.0 * l)).exp();
        assert!((got / want - 1.0).abs() < 1e-8, "{l} {q}: {got} vs {want}");
    }
    let mut rng = SmallRng::seed_from_u64(1);
    for (w, q) in [(vec![1.0, 0.25], 2.0), (vec![1.0, 0.5, 0.2], 3.0)] {
        let draws = 400_000;
        let hits = (0..draws)
            .filter(|_| w.iter().map(|l| l * normal(&mut rng).powi(2)).sum::<f64>() > q)
            .count();
        let mc = hits as f64 / draws as f64;
        let got = chisq_mix_tail(&w, q);
        assert!((got - mc).abs() < 4e-3, "{w:?}: {got} vs {mc}");
    }
    // broken: the integral alone
    let full = chisq_mix_tail(&[1.0, 0.25], 2.0);
    let first = special::Error::compl_error((2.0f64 / 2.0).sqrt());
    assert!((full - first) < full - 1e-3);
}

/// The Cauchy combination of one p-value is itself, and of equal small
/// p-values about them; Benjamini-Yekutieli scales by `m c(m) / rank`.
#[test]
fn cauchy_and_by_adjustment() {
    for p in [0.3, 1e-4, 1e-40] {
        assert!((cauchy_combine(&[p]) / p - 1.0).abs() < 1e-8, "{p}");
        assert!((cauchy_combine(&[p; 5]) / p - 1.0).abs() < 1e-6, "{p}");
    }
    let p = [0.01, 0.04, 0.03, 0.5];
    let c: f64 = 1.0 + 0.5 + 1.0 / 3.0 + 0.25;
    let adj = by_adjust(&p);
    // sorted 0.01, 0.03, 0.04, 0.5; running minimum from the top
    let want3 = (0.04 * 4.0 * c / 3.0).min(1.0);
    let want2 = (0.03 * 4.0 * c / 2.0).min(want3);
    assert!((adj[1] - want3).abs() < 1e-12 && (adj[2] - want2).abs() < 1e-12);
    assert!((adj[0] - (0.01 * 4.0 * c).min(want2)).abs() < 1e-12);
}

/// On one section, each kernel's statistic is the paper's `n^2 tr(E_C
/// Sigma_C) / n` with the gene scaled to unit variance and the coordinates
/// centred and scaled, its null weights the nonzero eigenvalues of
/// `Sigma_C`; without the centring the statistic differs.
#[test]
fn sparkx_statistics_are_the_papers_traces() {
    let n = 80;
    let mut rng = SmallRng::seed_from_u64(2);
    let xy = DMatrix::from_fn(n, 2, |_, _| rng.random_range(0.0..10.0));
    let y: Vec<f64> = (0..n)
        .map(|c| poisson(&mut rng, 1.0 + xy[(c, 0)] * 0.3))
        .collect();
    let counts = csc(std::slice::from_ref(&y), n);
    let section = vec![0usize; n];
    let stats = kernel_stats(&counts, &xy, &section, 1);
    // the coordinates as the test standardises them
    let mut s = xy.clone();
    for j in 0..2 {
        let mean = s.column(j).mean();
        let sd =
            (s.column(j).iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        s.column_mut(j)
            .iter_mut()
            .for_each(|v| *v = (*v - mean) / sd);
    }
    let h = Dense::identity(n, n) - Dense::from_element(n, n, 1.0 / n as f64);
    let ym = y.iter().sum::<f64>() / n as f64;
    let ysd = (y.iter().map(|v| (v - ym).powi(2)).sum::<f64>() / n as f64).sqrt();
    let yv = Dense::from_fn(n, 1, |c, _| (y[c] - ym) / ysd);
    let e = &yv * (yv.transpose() * &yv).try_inverse().unwrap() * yv.transpose();
    let check = |k: usize, sk: &Dense| {
        let sigma = sk * (sk.transpose() * sk).try_inverse().unwrap() * sk.transpose();
        let (ec, sc) = (&h * &e * &h, &h * &sigma * &h);
        let want = n as f64 * (&ec * &sc).trace();
        let (t, lambda) = &stats[0][k];
        assert!(
            (t - want).abs() < 1e-9 * want.abs().max(1.0),
            "kernel {k}: {t} vs {want}"
        );
        let mut ev: Vec<f64> = sc
            .clone()
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .copied()
            .filter(|v| *v > 1e-9)
            .collect();
        ev.sort_by(|a, b| b.total_cmp(a));
        assert_eq!(ev.len(), lambda.len());
        for (a, b) in ev.iter().zip(lambda) {
            assert!((a - b).abs() < 1e-9, "kernel {k}: {ev:?} vs {lambda:?}");
        }
        // broken: the gene's scale from its uncentred counts
        let raw = Dense::from_fn(n, 1, |c, _| y[c]);
        let broken =
            n as f64 * (&h * &raw).norm_squared() / raw.norm_squared() * (&ec * &sc).trace();
        assert!((broken - want).abs() > 1e-6 * want.abs());
    };
    check(0, &s);
    // the first Gaussian transform, at the 20% quantile of |s| per axis
    let q20: Vec<f64> = (0..2)
        .map(|j| {
            let mut v: Vec<f64> = s.column(j).iter().map(|x| x.abs()).collect();
            v.sort_by(f64::total_cmp);
            let hh = (n as f64 - 1.0) * 0.2;
            let lo = hh.floor() as usize;
            v[lo] + (hh - lo as f64) * (v[lo + 1] - v[lo])
        })
        .collect();
    let g = Dense::from_fn(n, 2, |c, j| {
        (-s[(c, j)].powi(2) / (2.0 * q20[j].powi(2))).exp()
    });
    check(1, &g);
}

/// Bar, set before measuring: on a 30 × 30 grid, genes planted with the
/// paper's simulated patterns (a gradient, a central hotspot or a streak,
/// at three-fold) are all declared spatially variable at the FDR, and no
/// more than one null gene is; with the coordinates shuffled, at most one
/// planted gene is.
#[test]
fn sparkx_detects_planted_spatial_genes_and_not_null_ones() {
    let side = 30;
    let n = side * side;
    let mut rng = SmallRng::seed_from_u64(3);
    let xy = DMatrix::from_fn(n, 2, |c, j| {
        if j == 0 {
            (c / side) as f64
        } else {
            (c % side) as f64
        }
    });
    let mut rows = Vec::new();
    for g in 0..30 {
        let row: Vec<f64> = (0..n)
            .map(|c| {
                let (x, y) = (xy[(c, 0)], xy[(c, 1)]);
                let base = 0.5;
                let mu = match g % 3 {
                    0 => base * (1.0 + 2.0 * x / side as f64),
                    1 => {
                        // at the section's centre, where the Gaussian transforms
                        // (centred there, as the paper writes them) look
                        let hot = (x - 14.5).powi(2) + (y - 14.5).powi(2) < 36.0;
                        base * if hot { 3.0 } else { 1.0 }
                    }
                    _ => base * if (12.0..18.0).contains(&y) { 3.0 } else { 1.0 },
                };
                nb(&mut rng, mu, 5.0)
            })
            .collect();
        rows.push(row);
    }
    for g in 0..60 {
        let mu = 0.3 + 0.05 * g as f64;
        rows.push((0..n).map(|_| nb(&mut rng, mu, 5.0)).collect());
    }
    let counts = csc(&rows, n);
    let section = vec![0usize; n];
    let p = sparkx(&counts, &xy, &section, 1);
    let (genes, count) = significant(&p, 3000);
    let planted = genes.iter().filter(|&&g| g < 30).count();
    let null = genes.iter().filter(|&&g| g >= 30).count();
    assert_eq!(count, genes.len());
    assert_eq!(planted, 30, "planted detected {planted}: {:?}", &p[..30]);
    assert!(null <= 1, "null declared {null}");
    // shuffled coordinates break the pattern
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.random_range(0..=i));
    }
    let shuffled = DMatrix::from_fn(n, 2, |c, j| xy[(order[c], j)]);
    let (genes, _) = significant(&sparkx(&counts, &shuffled, &section, 1), 3000);
    assert!(genes.iter().filter(|&&g| g < 30).count() <= 1, "{genes:?}");
}

/// Libraries spread over a decade and a half, a cell's `log10 m`.
fn libraries(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| 10f64.powf(rng.random_range(2.5..4.0)))
        .collect()
}

/// One gene's fit is the direct fit: the Poisson GLM's score is zero at
/// its betas, and no `theta` on a fine log grid beats its `theta` in the
/// negative binomial likelihood at those means; it recovers the planted
/// slope and dispersion roughly.
#[test]
fn sctransform_gene_fit_is_the_direct_fit() {
    let n = 600;
    let lib = libraries(n, 4);
    let t: Vec<f64> = lib.iter().map(|m| m.log10()).collect();
    let mut rng = SmallRng::seed_from_u64(5);
    let x: Vec<f64> = lib.iter().map(|&m| nb(&mut rng, 2e-3 * m, 3.0)).collect();
    let xs = sparse_gene(&x);
    let fit = fit_gene(&xs, &t);
    let mu: Vec<f64> = t
        .iter()
        .map(|&ti| (fit.beta0 + fit.beta1 * ti).exp())
        .collect();
    let g0: f64 = (0..n).map(|c| x[c] - mu[c]).sum();
    let g1: f64 = (0..n).map(|c| (x[c] - mu[c]) * t[c]).sum();
    let scale: f64 = x.iter().sum();
    assert!(
        g0.abs() < 1e-6 * scale && g1.abs() < 1e-6 * scale,
        "{g0} {g1}"
    );
    let best = nb_loglik(&xs, &mu, fit.theta);
    for i in 0..400 {
        let th = 10f64.powf(-2.0 + 5.0 * i as f64 / 399.0);
        assert!(
            nb_loglik(&xs, &mu, th) <= best + 1e-8 * best.abs(),
            "theta {th} beats {}",
            fit.theta
        );
    }
    // the full likelihood (theta-free terms included) agrees on the order
    let full = |th: f64| -> f64 {
        let lg = |v: f64| special::Gamma::ln_gamma(v).0;
        (0..n)
            .map(|c| {
                lg(x[c] + th) - lg(th) - lg(x[c] + 1.0)
                    + th * (th / (th + mu[c])).ln()
                    + x[c] * (mu[c] / (th + mu[c])).ln()
            })
            .sum()
    };
    assert!(full(fit.theta) >= full(fit.theta * 1.2) && full(fit.theta) >= full(fit.theta / 1.2));
    assert!(
        (fit.beta1 - std::f64::consts::LN_10).abs() < 0.15,
        "{}",
        fit.beta1
    );
    assert!((fit.theta / 3.0).ln().abs() < 0.4, "{}", fit.theta);
}

/// Bar, set before measuring: on null genes (means proportional to the
/// library, one dispersion), the residuals at the regularised models have
/// variance near one (median within 0.1, nine in ten within 0.3), on the
/// genes inside the range of gene means (at its ends the kernel
/// regression's own bias moves the intercept, in the paper's method as
/// here); Poisson residuals (no dispersion) are over-dispersed.
#[test]
fn sctransform_residuals_have_unit_variance_on_null_genes() {
    use crate::spatial_baselines::spatial_factor::sct::sctransform;
    let n = 2000;
    let lib = libraries(n, 6);
    let t: Vec<f64> = lib.iter().map(|m| m.log10()).collect();
    let mut rng = SmallRng::seed_from_u64(7);
    let genes: Vec<GeneCounts> = (0..300)
        .map(|g| {
            let p = 10f64.powf(-4.0 + 2.5 * g as f64 / 299.0);
            let x: Vec<f64> = lib.iter().map(|&m| nb(&mut rng, p * m, 4.0)).collect();
            sparse_gene(&x)
        })
        .collect();
    let wanted: Vec<usize> = (0..300).collect();
    let (models, report) = sctransform(&genes, &lib, &wanted, 8);
    assert_eq!(report.genes_fitted, 300);
    let var = |z: &[f64]| {
        let m = z.iter().sum::<f64>() / z.len() as f64;
        z.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (z.len() - 1) as f64
    };
    // the genes inside the range, with a mean of at least 0.1 a cell
    let mut vs: Vec<f64> = (100..250)
        .map(|g| var(&residuals(&genes[g], &t, &models[g])))
        .collect();
    vs.sort_by(f64::total_cmp);
    let med = vs[vs.len() / 2];
    let within = vs.iter().filter(|v| (*v - 1.0).abs() < 0.3).count();
    eprintln!(
        "residual variance: median {med:.3}, {within} of {} within 0.3",
        vs.len()
    );
    assert!((med - 1.0).abs() < 0.1, "median {med}");
    assert!(within * 10 >= vs.len() * 9, "{within}");
    // broken: Poisson residuals
    let poisson_var: Vec<f64> = (200..250)
        .map(|g| {
            let m = GeneModel {
                theta: f64::INFINITY,
                ..models[g]
            };
            var(&residuals(&genes[g], &t, &m))
        })
        .collect();
    assert!(poisson_var.iter().all(|&v| v > 1.5), "{poisson_var:?}");
}

/// The dense-row operator's products are its matrix's.
#[test]
fn dense_rows_operator_matches_its_matrix() {
    use crate::spatial_baselines::spatial_factor::prep::ResidualOp;
    let (m, n) = (5, 700);
    let y = gaussian(m, n, 9).map(|v| (v as f32) as f64);
    let rows: Vec<f32> = (0..m)
        .flat_map(|g| (0..n).map(move |c| (g, c)))
        .map(|(g, c)| y[(g, c)] as f32)
        .collect();
    let op = ResidualOp::from_rows(m, n, rows);
    let v = gaussian(n, 3, 10);
    let u = gaussian(m, 2, 11);
    assert!((op.apply(&v) - &y * &v).amax() < 1e-10 * (&y * &v).amax());
    assert!((op.apply_t(&u) - y.transpose() * &u).amax() < 1e-10 * (y.transpose() * &u).amax());
    let section: Vec<usize> = (0..n).map(|c| c % 3).collect();
    let ss = op.section_sumsq(&section, 3);
    for s in 0..3 {
        let want: f64 = (0..n)
            .filter(|&c| section[c] == s)
            .map(|c| y[(0, c)].powi(2))
            .sum();
        assert!((ss[(0, s)] - want).abs() < 1e-9 * want);
    }
}
