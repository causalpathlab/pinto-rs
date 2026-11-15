//! The spatial factor core: its numerical kit, implicit expression and
//! graph prior, each against a dense computation, with a broken variant
//! that each check must catch.

use super::graph::{grid, toy};
use crate::spatial_baselines::graph::knn_within_sections;
use crate::spatial_baselines::spatial_factor::kit::*;
use crate::spatial_baselines::spatial_factor::prep::*;
use crate::spatial_baselines::spatial_factor::prior::*;
use nalgebra::DMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix};

fn max_rel(a: &Dense, b: &Dense) -> f64 {
    let scale = b.amax().max(1e-300);
    (a - b).amax() / scale
}

/// The dense Laplacian of `lap`, built from its products with the unit
/// vectors.
fn dense_laplacian(lap: &Laplacian) -> Dense {
    let n = lap.n();
    let mut out = Dense::zeros(n, n);
    let mut e = vec![0.0; n];
    let mut y = vec![0.0; n];
    for j in 0..n {
        e[j] = 1.0;
        lap.apply(&e, &mut y);
        for i in 0..n {
            out[(i, j)] = y[i];
        }
        e[j] = 0.0;
    }
    out
}

fn two_section_lap(side: usize, k: usize) -> Laplacian {
    let d = grid(side, 2);
    let nb = knn_within_sections(&d, k).symmetrised();
    Laplacian::from_neighbours(&nb, &d.section, d.n_sections)
}

/// `L` of a symmetrised kNN is symmetric, annihilates the constants of each
/// section, and is the hand-built `D - A`; a super-graph's carries its
/// edge weights.
#[test]
fn laplacians_are_d_minus_a() {
    let d = toy(&[(0., 0.), (1., 0.), (2., 0.), (10., 0.)], &[0; 4]);
    let nb = knn_within_sections(&d, 1).symmetrised();
    let lap = Laplacian::from_neighbours(&nb, &d.section, 1);
    let want = DMatrix::from_row_slice(
        4,
        4,
        &[
            1., -1., 0., 0., -1., 2., -1., 0., 0., -1., 2., -1., 0., 0., -1., 1.,
        ],
    );
    assert_eq!(dense_laplacian(&lap), want);

    let lap = Laplacian::from_edges(3, &[(0, 1), (1, 2)], &[2.0, 0.5], &[0, 0, 0], 1);
    let l = dense_laplacian(&lap);
    assert_eq!(l[(0, 0)], 2.0);
    assert_eq!(l[(1, 1)], 2.5);
    assert_eq!(l[(1, 2)], -0.5);

    let lap = two_section_lap(5, 4);
    let l = dense_laplacian(&lap);
    assert!((&l - l.transpose()).amax() < 1e-15);
    for s in &lap.sections {
        let mut one = vec![0.0; lap.n()];
        s.iter().for_each(|&i| one[i] = 1.0);
        let mut y = vec![0.0; lap.n()];
        lap.apply(&one, &mut y);
        assert!(y.iter().all(|v| v.abs() < 1e-12));
    }
}

/// A dense solve of `(a_s I + b L) X = B`.
fn dense_solve(lap: &Laplacian, shift: &[f64], b: f64, rhs: &Dense, with_l: bool) -> Dense {
    let n = lap.n();
    let l = dense_laplacian(lap);
    let a = Dense::from_fn(n, n, |i, j| {
        let d = if i == j { shift[lap.section[i]] } else { 0.0 };
        d + if with_l { b * l[(i, j)] } else { 0.0 }
    });
    a.lu().solve(rhs).unwrap()
}

/// Conjugate gradients agree with a dense solve; one without the `b L`
/// term does not.
#[test]
fn cg_solves_the_shifted_laplacian() {
    let lap = two_section_lap(6, 5);
    let n = lap.n();
    let rhs = gaussian(n, 3, 1);
    let shift = [0.7, 2.0];
    let (x, steps) = cg_solve(&lap, &shift, 3.0, &rhs, None, 1e-13);
    assert!(steps > 0);
    let want = dense_solve(&lap, &shift, 3.0, &rhs, true);
    assert!(max_rel(&x, &want) < 1e-10, "{}", max_rel(&x, &want));
    let broken = dense_solve(&lap, &shift, 3.0, &rhs, false);
    assert!(max_rel(&broken, &want) > 1e-3);
    // a warm start from the answer takes no step
    let (_, again) = cg_solve(&lap, &shift, 3.0, &rhs, Some(&x), 1e-8);
    assert_eq!(again, 0);
}

/// Spectral sums by quadrature within 2% of the exact eigenvalue sums
/// (forced on every section), the exact path to rounding; quadrature
/// weighted by `e[0]` in place of `e[0]^2` misses.
#[test]
fn slq_matches_exact_spectral_sums() {
    let lap = two_section_lap(12, 5);
    let exact = Spectrum::new(
        &lap,
        &SpectrumOpts {
            exact_eig_max: 10_000,
            probes: 1,
            steps: 1,
            seed: 0,
        },
    );
    let slq = Spectrum::new(
        &lap,
        &SpectrumOpts {
            exact_eig_max: 0,
            probes: 32,
            steps: 60,
            seed: 7,
        },
    );
    assert!(exact.exact.iter().all(|&e| e) && slq.exact.iter().all(|&e| !e));
    type F = Box<dyn Fn(f64) -> f64>;
    let fs: [(&str, F); 3] = [
        ("log(1 + 2 mu)", Box::new(|t: f64| (1.0 + 2.0 * t).ln())),
        ("1 / (0.5 + 3 mu)", Box::new(|t: f64| 1.0 / (0.5 + 3.0 * t))),
        ("mu / (1 + mu)", Box::new(|t: f64| t / (1.0 + t))),
    ];
    let per = 144.0;
    let mut broken = slq.clone();
    for nodes in &mut broken.nodes {
        for (_, w) in nodes.iter_mut() {
            *w = (per / 32.0) * (*w * 32.0 / per).sqrt();
        }
    }
    for (name, f) in &fs {
        for s in 0..2 {
            let (e, q) = (exact.sum(s, f), slq.sum(s, f));
            assert!((q - e).abs() <= 0.02 * e.abs(), "{name}: {q} vs {e}");
            let b = broken.sum(s, f);
            assert!((b - e).abs() > 0.02 * e.abs(), "{name}: broken {b} vs {e}");
        }
    }
    // the trace of the identity is the section's size
    assert!((exact.n() - 288.0).abs() < 1e-9);
}

/// `cbar(rho)` is the mean of `1 / (1 + rho mu)`, its log-derivative the
/// central difference.
#[test]
fn cbar_and_its_log_derivative() {
    let lap = two_section_lap(6, 5);
    let sp = Spectrum::new(
        &lap,
        &SpectrumOpts {
            exact_eig_max: 10_000,
            probes: 1,
            steps: 1,
            seed: 0,
        },
    );
    let n = lap.n();
    let rho: f64 = 1.7;
    let (c, dlog) = sp.cbar(rho, n);
    assert!(c > 0.0 && c < 1.0);
    let h: f64 = 1e-5;
    let (cp, _) = sp.cbar(rho * h.exp(), n);
    let (cm, _) = sp.cbar(rho * (-h).exp(), n);
    let fd = (cp.ln() - cm.ln()) / (2.0 * h);
    assert!((fd - dlog).abs() < 1e-7, "{fd} vs {dlog}");
    assert_eq!(sp.cbar(0.0, n).0, 1.0);
}

#[test]
fn brent_finds_a_known_minimum() {
    let (x, fx, evals) = brent_min(|x| (x - 0.3).powi(2) + 1.0, -2.0, 5.0, 1e-10, 100);
    assert!(
        (x - 0.3).abs() < 1e-8 && (fx - 1.0).abs() < 1e-12,
        "{x} {fx}"
    );
    assert!(evals < 50);
    let (x, _, _) = grid_then_brent(
        |x: f64| (x.exp() - 3.0).powi(2),
        &[-2.0, 0.0, 1.0, 2.0, 4.0],
        1e-10,
        100,
    );
    assert!((x - 3f64.ln()).abs() < 1e-7, "{x}");
}

/// The box minimiser stops on the bound a quadratic presses against, and
/// finds Rosenbrock's minimum inside the box.
#[test]
fn box_minimiser_respects_active_bounds() {
    let quad = |x: &[f64]| {
        let f = (x[0] - 2.0).powi(2) + 3.0 * (x[1] + 0.5).powi(2);
        (f, vec![2.0 * (x[0] - 2.0), 6.0 * (x[1] + 0.5)])
    };
    let r = lbfgs_box(quad, &[0.0, 0.0], &[-1.0, 0.0], &[1.0, 1.0], 200, 1e-9, 0.0);
    assert!(
        (r.x[0] - 1.0).abs() < 1e-9 && r.x[1].abs() < 1e-9,
        "{:?}",
        r.x
    );
    let rosen = |x: &[f64]| {
        let (a, b) = (x[0], x[1]);
        let f = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
        (
            f,
            vec![
                -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
                200.0 * (b - a * a),
            ],
        )
    };
    let r = lbfgs_box(
        rosen,
        &[-1.2, 1.0],
        &[-2.0, -2.0],
        &[2.0, 2.0],
        2000,
        1e-8,
        0.0,
    );
    assert!(
        (r.x[0] - 1.0).abs() < 1e-5 && (r.x[1] - 1.0).abs() < 1e-5,
        "{:?}",
        r.x
    );
    // with an active bound on Rosenbrock
    let r = lbfgs_box(
        rosen,
        &[-1.2, 1.0],
        &[-2.0, -2.0],
        &[0.5, 2.0],
        2000,
        1e-8,
        0.0,
    );
    assert!((r.x[0] - 0.5).abs() < 1e-9, "{:?}", r.x);
    assert!((r.x[1] - 0.25).abs() < 1e-5, "{:?}", r.x);
}

/// Subspace iteration finds the top eigenpairs of a matrix given only its
/// products, warm or cold.
#[test]
fn subspace_iteration_matches_a_dense_eigensolver() {
    let n = 60;
    let b = gaussian(n, n, 3);
    let mut a = &b * b.transpose();
    // a clear gap below the top 4
    for i in 0..4 {
        a += 50.0 * (i + 1) as f64 * b.column(i) * b.column(i).transpose();
    }
    let (want, wvec) = sym_eig_desc(&a);
    let o = EigOpts {
        d: 4,
        block: 12,
        seed: 1,
        tol: 1e-13,
        max_iter: 2000,
    };
    let (vals, vecs, block) = top_eigs(|v| &a * v, n, None, &o);
    for l in 0..4 {
        assert!((vals[l] - want[l]).abs() < 1e-10 * want[0], "{l}");
        let c = vecs.column(l).dot(&wvec.column(l)).abs();
        assert!((c - 1.0).abs() < 1e-8, "{l}: {c}");
    }
    let (again, _, _) = top_eigs(|v| &a * v, n, Some(&block), &o);
    assert!((again[0] - want[0]).abs() < 1e-10 * want[0]);
}

/// Counts with a positive clip (a large count) and a negative one (a zero
/// count where the mean passes N), on two sections.
fn counts_fixture() -> (CscMatrix<f32>, Vec<f64>, Vec<usize>) {
    let n = 6;
    let mut coo = CooMatrix::<f32>::new(4, n);
    let entries = [
        (0, 0, 30.0),
        (0, 1, 1.0),
        (1, 1, 2.0),
        (1, 2, 900.0),
        (2, 3, 4.0),
        (3, 4, 1.0),
        (0, 5, 2.0),
        (2, 5, 1.0),
        (1, 0, 1.0),
        (3, 3, 2.0),
    ];
    for (g, c, v) in entries {
        coo.push(g, c, v);
    }
    let counts = CscMatrix::from(&coo);
    // libraries over more genes than the four kept
    let lib = vec![40.0, 8.0, 5000.0, 9.0, 6.0, 5.0];
    (counts, lib, vec![0, 0, 0, 1, 1, 1])
}

fn dense_counts(c: &CscMatrix<f32>) -> Dense {
    let mut d = Dense::zeros(c.nrows(), c.ncols());
    for (g, j, &v) in c.triplet_iter() {
        d[(g, j)] = v as f64;
    }
    d
}

/// The Pearson recipe, densified by hand: residuals, the clip at
/// `+-sqrt(N)`, standardised genes.
fn pearson_dense(counts: &CscMatrix<f32>, lib: &[f64]) -> Dense {
    let x = dense_counts(counts);
    let (m, n) = x.shape();
    let total: f64 = lib.iter().sum();
    let clip = (n as f64).sqrt();
    let mut r = Dense::from_fn(m, n, |g, c| {
        let mu = x.row(g).sum() / total * lib[c];
        ((x[(g, c)] - mu) / mu.sqrt()).clamp(-clip, clip)
    });
    for g in 0..m {
        let mean = r.row(g).mean();
        let var = r.row(g).iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        for c in 0..n {
            r[(g, c)] = (r[(g, c)] - mean) / var.sqrt();
        }
    }
    r
}

/// The implicit operators' products are the densified matrices'.
#[test]
fn implicit_expression_matches_its_dense_form() {
    let (counts, lib, section) = counts_fixture();
    let op = ResidualOp::pearson(&counts, &lib);
    assert!(op.clipped_zeros > 0, "the negative clip binds");
    let want = pearson_dense(&counts, &lib);
    let v = gaussian(6, 3, 2);
    let u = gaussian(4, 2, 3);
    assert!(max_rel(&op.apply(&v), &(&want * &v)) < 1e-5);
    assert!(max_rel(&op.apply_t(&u), &(want.transpose() * &u)) < 1e-5);
    assert!(max_rel(&op.dense(), &want) < 1e-5);
    let fro = op.frob2(&section, 2);
    assert!((fro - want.norm_squared()).abs() < 1e-4 * fro);
    // without the clip, the products differ
    let unclipped = {
        let x = dense_counts(&counts);
        let total: f64 = lib.iter().sum();
        Dense::from_fn(4, 6, |g, c| {
            let mu = x.row(g).sum() / total * lib[c];
            (x[(g, c)] - mu) / mu.sqrt()
        })
    };
    assert!(max_rel(&unclipped, &want) > 1e-2);

    assert_eq!(library(&counts), vec![31.0, 3.0, 900.0, 6.0, 1.0, 3.0]);
    let sub = select_rows(&counts, &[2, 0]);
    assert_eq!(dense_counts(&sub).row(0), dense_counts(&counts).row(2));
}
