//! MEFISTO: the sparse graph prior and the Gaussian-process prior against
//! dense oracles, the variational updates against the ELBO they ascend,
//! each with a broken variant the check must catch; then recovery of
//! planted factors and domains.

use super::graph::toy;
use super::pseudobulk::{ari, labels_of, read_labels};
use super::spatialpca::{parse, planted, simulated_section};
use crate::spatial_baselines::graph::knn_within_sections;
use crate::spatial_baselines::mefisto::reference::{inducing_cells, GpPrior, GpSizes};
use crate::spatial_baselines::mefisto::sparse::GraphPrior;
use crate::spatial_baselines::mefisto::vb::{self, FactorPrior, State, VbData, VbOpts};
use crate::spatial_baselines::spatial_factor::kit::{gaussian, Dense};
use crate::spatial_baselines::spatial_factor::prep::ResidualOp;
use crate::spatial_baselines::spatial_factor::prior::{Laplacian, Spectrum, SpectrumOpts};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn jittered_grid(side: usize, sections: usize, seed: u64) -> (Vec<(f32, f32)>, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut xy = Vec::new();
    let mut sec = Vec::new();
    for s in 0..sections {
        for r in 0..side {
            for c in 0..side {
                xy.push((
                    r as f32 + rng.random_range(-0.2..0.2f32),
                    c as f32 + rng.random_range(-0.2..0.2f32),
                ));
                sec.push(s);
            }
        }
    }
    (xy, sec)
}

fn lap_of(xy: &[(f32, f32)], section: &[usize]) -> Laplacian {
    let d = toy(xy, section);
    let nb = knn_within_sections(&d, 5).symmetrised();
    Laplacian::from_neighbours(&nb, &d.section, d.n_sections)
}

fn dense_lap(lap: &Laplacian) -> Dense {
    let n = lap.n();
    let mut out = Dense::zeros(n, n);
    let mut e = vec![0.0; n];
    let mut y = vec![0.0; n];
    for j in 0..n {
        e[j] = 1.0;
        lap.apply(&e, &mut y);
        out.column_mut(j).copy_from_slice(&y);
        e[j] = 0.0;
    }
    out
}

fn exact() -> SpectrumOpts {
    SpectrumOpts {
        exact_eig_max: 100_000,
        probes: 1,
        steps: 1,
        seed: 0,
    }
}

fn logdet(a: &Dense) -> f64 {
    2.0 * a
        .clone()
        .cholesky()
        .unwrap()
        .l()
        .diagonal()
        .iter()
        .map(|v| v.ln())
        .sum::<f64>()
}

/// The dense factor update and its ELBO term for precision `prec` of the
/// prior and likelihood precision `s` per section: the mean, the
/// per-section variance sums, and the term.
fn dense_update(
    prec: &Dense,
    s: &[f64],
    section: &[usize],
    b: &[f64],
) -> (Vec<f64>, Vec<f64>, f64) {
    let n = prec.nrows();
    let sdiag = Dense::from_fn(n, n, |i, j| if i == j { s[section[i]] } else { 0.0 });
    let a = (&sdiag + prec).try_inverse().unwrap();
    let mu = &a * Dense::from_column_slice(n, 1, b);
    let mut tr = vec![0.0; s.len()];
    for i in 0..n {
        tr[section[i]] += a[(i, i)];
    }
    let term = 0.5 * logdet(prec)
        - 0.5 * (prec * &a).trace()
        - 0.5 * (mu.transpose() * prec * &mu)[(0, 0)]
        + 0.5 * logdet(&a)
        + n as f64 / 2.0;
    (mu.iter().copied().collect(), tr, term)
}

/// One factor update of the graph prior is the dense solve of `(S + Q)`,
/// its variance sums the dense inverse's, its ELBO term the dense one;
/// without the `c rho L` term, or with `c` held at 1, they are not.
#[test]
fn graph_prior_update_is_the_dense_update() {
    let (xy, section) = jittered_grid(8, 2, 1);
    let n = xy.len();
    let lap = lap_of(&xy, &section);
    let spec = Spectrum::new(&lap, &exact());
    let rho = 2.5;
    let mut prior = GraphPrior::new(&lap, &spec, 1, rho, 1e-12, 20);
    let c = spec.cbar(rho, n).0;
    let l = dense_lap(&lap);
    let q = (Dense::identity(n, n) + &l * rho) * c;
    let s = [3.0, 0.7];
    let b: Vec<f64> = gaussian(n, 1, 2).iter().copied().collect();
    let mut mu = vec![0.0; n];
    let tr = prior.update(0, &s, &b, &mut mu);
    let (want_mu, want_tr, want_term) = dense_update(&q, &s, &section, &b);
    let rel = |a: &[f64], b: &[f64]| {
        let scale = b.iter().fold(0f64, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / scale
    };
    assert!(rel(&mu, &want_mu) < 1e-8, "{}", rel(&mu, &want_mu));
    assert!(rel(&tr, &want_tr) < 1e-10);
    let term = prior.kl_term(0, &mu);
    assert!(
        (term - want_term).abs() < 1e-8 * want_term.abs(),
        "{term} vs {want_term}"
    );
    // broken: no c rho L in the solve; c held at 1 in the prior
    let (no_l, _, _) = dense_update(&(Dense::identity(n, n) * c), &s, &section, &b);
    assert!(rel(&no_l, &want_mu) > 1e-3);
    let q1 = Dense::identity(n, n) + &l * rho;
    let (_, _, t1) = dense_update(&q1, &s, &section, &b);
    assert!((t1 - want_term).abs() > 1e-3 * want_term.abs());
}

/// The graph prior's `rho` step lands on the maximum over `rho` of the
/// factor's part of the ELBO with `q(z)` at its optimum, densely, and
/// leaves `q` there; holding `q` fixed (the paper's step) lands elsewhere.
#[test]
fn graph_prior_rho_step_maximises_the_collapsed_bound() {
    let (xy, section) = jittered_grid(8, 1, 3);
    let n = xy.len();
    let lap = lap_of(&xy, &section);
    let spec = Spectrum::new(&lap, &exact());
    let mut prior = GraphPrior::new(&lap, &spec, 1, 0.3, 1e-12, 30);
    // a smooth mean so the step has something to find
    let b: Vec<f64> = (0..n).map(|i| 4.0 * (xy[i].0 as f64 / 3.0).sin()).collect();
    let mut mu = vec![0.0; n];
    let s = [1.0];
    prior.update(0, &s, &b, &mut mu);
    let l = dense_lap(&lap);
    let q_of = |r: f64| (Dense::identity(n, n) + &l * r) * spec.cbar(r, n).0;
    let (_, _, before) = dense_update(&q_of(0.3), &s, &section, &b);
    let mu0 = mu.clone();
    let tr = prior.hyper(0, &s, &b, &mut mu).unwrap();
    let rho = prior.rho[0];
    let (want_mu, want_tr, _) = dense_update(&q_of(rho), &s, &section, &b);
    assert!(mu.iter().zip(&want_mu).all(|(a, w)| (a - w).abs() < 1e-6));
    assert!((tr[0] - want_tr[0]).abs() < 1e-8 * want_tr[0]);
    // the collapsed bound, densely: the factor's part at q's optimum
    let bound = |r: f64| {
        let (m, t, term) = dense_update(&q_of(r), &s, &section, &b);
        let lin: f64 = (0..n).map(|c| b[c] * m[c] - 0.5 * s[0] * m[c] * m[c]).sum();
        lin - 0.5 * s[0] * t[0] + term
    };
    let at = bound(rho);
    let lin0: f64 = (0..n)
        .map(|c| b[c] * mu0[c] - 0.5 * s[0] * mu0[c] * mu0[c])
        .sum();
    let a0 = (Dense::identity(n, n) * s[0] + q_of(0.3))
        .try_inverse()
        .unwrap();
    assert!(at >= lin0 - 0.5 * s[0] * a0.trace() + before - 1e-8 * at.abs());
    for i in 0..60 {
        let r = 10f64.powf(-3.0 + 6.0 * i as f64 / 59.0);
        assert!(bound(r) <= at + 1e-6 * at.abs(), "rho {r} beats {rho}");
    }
    // the fixed-q objective's maximum is not the joint one's
    let m0 = Dense::from_column_slice(n, 1, &mu0);
    let fixed = |r: f64| {
        let q = q_of(r);
        0.5 * logdet(&q) - 0.5 * (&q * &a0).trace() - 0.5 * (m0.transpose() * &q * &m0)[(0, 0)]
    };
    let best_fixed = (0..200)
        .map(|i| 10f64.powf(-3.0 + 6.0 * i as f64 / 199.0))
        .max_by(|a, b| fixed(*a).total_cmp(&fixed(*b)))
        .unwrap();
    assert!(
        (best_fixed.ln() - rho.ln()).abs() > 0.05,
        "{best_fixed} vs {rho}"
    );
}

/// The Gaussian-process prior with every cell inducing is the dense GP
/// update, mean and variance sums.
#[test]
fn gp_prior_with_every_cell_inducing_is_the_dense_gp() {
    let (xy, section) = jittered_grid(5, 2, 4);
    let n = xy.len();
    let pts: Vec<Vec<f64>> = xy.iter().map(|p| vec![p.0 as f64, p.1 as f64]).collect();
    let mut prior = GpPrior::new(
        pts.clone(),
        section.clone(),
        2,
        GpSizes {
            k: 1,
            n_inducing: n,
            n_grid: 5,
            rank: 1,
            seed: 0,
        },
    );
    assert_eq!(prior.inducing.len(), n);
    let h = prior.hyper[0].clone();
    // the dense kernel, by hand
    let kt = Dense::from_fn(2, 2, |a, b| {
        h.x[a] * h.x[b] + if a == b { h.sigma * h.sigma } else { 0.0 }
    });
    let kg = Dense::from_fn(2, 2, |a, b| kt[(a, b)] / (kt[(a, a)] * kt[(b, b)]).sqrt());
    let sigma = Dense::from_fn(n, n, |i, j| {
        let d2 = (pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2);
        let e = (-d2 / (2.0 * h.lengthscale * h.lengthscale)).exp();
        (1.0 - h.zeta) * e * kg[(section[i], section[j])] + if i == j { h.zeta } else { 0.0 }
    });
    let s = [2.0, 0.5];
    let b: Vec<f64> = gaussian(n, 1, 5).iter().copied().collect();
    let mut mu = vec![0.0; n];
    let tr = prior.update(0, &s, &b, &mut mu);
    let prec = sigma.clone().try_inverse().unwrap();
    let (want_mu, want_tr, want_term) = dense_update(&prec, &s, &section, &b);
    for (a, w) in mu.iter().zip(&want_mu) {
        assert!((a - w).abs() < 1e-7 * want_mu.iter().fold(0f64, |m, v| m.max(v.abs())));
    }
    for (a, w) in tr.iter().zip(&want_tr) {
        assert!((a - w).abs() < 1e-7 * w, "{a} vs {w}");
    }
    let term = prior.kl_term(0, &mu);
    assert!(
        (term - want_term).abs() < 1e-6 * want_term.abs(),
        "{term} vs {want_term}"
    );
    // the hyperparameter step never lowers the factor's part of the ELBO
    let before = factor_part(&prior, &s, &b, &section, &mu, &tr);
    let tr = prior.hyper(0, &s, &b, &mut mu).unwrap();
    let after = factor_part(&prior, &s, &b, &section, &mu, &tr);
    assert!(after >= before - 1e-8 * before.abs(), "{after} < {before}");
}

/// Factor 0's part of the ELBO given the likelihood's `s` and `b`: `b .
/// E[z] - sum_n s_g E[z_n^2] / 2 + E[log p(z)] - E[log q(z)]`.
fn factor_part(
    prior: &GpPrior,
    s: &[f64],
    b: &[f64],
    section: &[usize],
    mu: &[f64],
    tr: &[f64],
) -> f64 {
    let lin: f64 = (0..mu.len())
        .map(|c| b[c] * mu[c] - 0.5 * s[section[c]] * mu[c] * mu[c])
        .sum();
    let var: f64 = tr.iter().zip(s).map(|(t, sg)| 0.5 * sg * t).sum();
    lin - var + prior.kl_term(0, mu)
}

fn gp_prior(pts: &[Vec<f64>], section: &[usize], groups: usize, inducing: usize) -> GpPrior {
    GpPrior::new(
        pts.to_vec(),
        section.to_vec(),
        groups,
        GpSizes {
            k: 1,
            n_inducing: inducing,
            n_grid: 4,
            rank: 1,
            seed: 0,
        },
    )
}

fn points(xy: &[(f32, f32)]) -> Vec<Vec<f64>> {
    xy.iter().map(|p| vec![p.0 as f64, p.1 as f64]).collect()
}

/// With fewer inducing cells than cells, the hyperparameter objective is
/// the factor's part of the ELBO at the moments its hyperparameters give
/// (`q(u)` fixed), on one group and two; its gradient is the central
/// difference's. The prior's term alone is not that part.
#[test]
fn gp_objective_with_inducing_cells_is_the_elbo_at_its_moments() {
    for groups in [1usize, 2] {
        let (xy, section) = jittered_grid(7, groups, 20 + groups as u64);
        let n = xy.len();
        let pts = points(&xy);
        let mut prior = gp_prior(&pts, &section, groups, 15);
        assert!(prior.inducing.len() < n);
        let s: Vec<f64> = (0..groups).map(|g| 1.5 + g as f64).collect();
        let b: Vec<f64> = (0..n)
            .map(|i| 2.0 * (pts[i][0] / 2.0).sin() + 0.3 * pts[i][1])
            .collect();
        let mut mu = vec![0.0; n];
        prior.update(0, &s, &b, &mut mu);
        let h0 = prior.hyper[0].clone();
        for (zeta, sigma, x) in [(0.3, 0.4, 0.6), (0.7, 0.8, -0.2)] {
            let mut h = h0.clone();
            h.lengthscale *= 0.7;
            h.zeta = zeta;
            h.sigma = sigma;
            h.x.iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = x + 0.1 * i as f64);
            let f = prior.factor_term(0, &s, &b, &h);
            let grad = prior.factor_grad(0, &s, &b, &h);
            let tr = prior.set_hyper(0, h.clone(), &mut mu);
            let want = factor_part(&prior, &s, &b, &section, &mu, &tr);
            assert!(
                (f - want).abs() < 1e-8 * want.abs(),
                "{groups}: {f} vs {want}"
            );
            assert!((prior.kl_term(0, &mu) - want).abs() > 1e-3 * want.abs());
            // the free parameters: zeta, and with two groups sigma and x
            let free = if groups > 1 { 2 + h.x.len() } else { 1 };
            for i in 0..free {
                let step = 1e-5;
                let at = |d: f64| {
                    let mut hh = h.clone();
                    match i {
                        0 => hh.zeta += d,
                        1 => hh.sigma += d,
                        _ => hh.x[i - 2] += d,
                    }
                    prior.factor_term(0, &s, &b, &hh)
                };
                let fd = (at(step) - at(-step)) / (2.0 * step);
                assert!(
                    (grad[i] - fd).abs() < 1e-5 * fd.abs().max(1.0),
                    "{groups} groups, parameter {i}: {} vs {fd}",
                    grad[i]
                );
            }
            prior.set_hyper(0, h0.clone(), &mut mu);
        }
    }
}

/// One group: the eigenbasis objective is the general one at every
/// `zeta`, with fewer inducing cells than cells and with every cell
/// inducing; and the eigenbasis step's choice is at least as good as the
/// general step's (the factor's part of the ELBO after each, from the same
/// state).
#[test]
fn one_group_fast_search_is_the_general_search() {
    for inducing in [15usize, 1000] {
        let (xy, section) = jittered_grid(7, 1, 50);
        let n = xy.len();
        let pts = points(&xy);
        let s = [1.7];
        let b: Vec<f64> = (0..n)
            .map(|i| 2.0 * (pts[i][0] / 2.0).sin() + 0.3 * pts[i][1])
            .collect();
        let mut fast = gp_prior(&pts, &section, 1, inducing);
        let mut general = gp_prior(&pts, &section, 1, inducing);
        general.fast = false;
        let (mut mu_f, mut mu_g) = (vec![0.0; n], vec![0.0; n]);
        fast.update(0, &s, &b, &mut mu_f);
        general.update(0, &s, &b, &mut mu_g);
        for zeta in [1e-4, 0.03, 0.4, 0.9, 1.0] {
            let mut h = fast.hyper[0].clone();
            h.zeta = zeta;
            let want = fast.factor_term(0, &s, &b, &h);
            let got = fast.fast_factor_term(0, &s, &b, &h);
            assert!(
                (got - want).abs() < 1e-9 * want.abs(),
                "{inducing} inducing, zeta {zeta}: {got} vs {want}"
            );
        }
        let tr_f = fast.hyper(0, &s, &b, &mut mu_f).unwrap();
        let tr_g = general.hyper(0, &s, &b, &mut mu_g).unwrap();
        let f = factor_part(&fast, &s, &b, &section, &mu_f, &tr_f);
        let g = factor_part(&general, &s, &b, &section, &mu_g, &tr_g);
        assert!(f >= g - 1e-8 * g.abs(), "{inducing} inducing: {f} < {g}");
        assert!(
            (f - g).abs() < 1e-4 * g.abs(),
            "{inducing} inducing: {f} vs {g}"
        );
    }
}

/// The unstructured factor (`l = 0`) with fewer inducing cells than
/// cells: every cell its own `N(0, 1)` factor, `E[z_n] = b_n / (1 +
/// s_g)`, not zero off the inducing cells.
#[test]
fn unstructured_factor_covers_every_cell_with_inducing_cells() {
    let (xy, section) = jittered_grid(10, 2, 30);
    let n = xy.len();
    let pts = points(&xy);
    let mut prior = gp_prior(&pts, &section, 2, 20);
    assert_eq!(prior.inducing.len(), 20);
    prior.hyper[0].lengthscale = 0.0;
    let s = [2.0, 0.5];
    let b: Vec<f64> = gaussian(n, 1, 31).iter().copied().collect();
    let mut mu = vec![0.0; n];
    let tr = prior.update(0, &s, &b, &mut mu);
    for c in 0..n {
        let want = b[c] / (1.0 + s[section[c]]);
        assert!(
            (mu[c] - want).abs() < 1e-12,
            "cell {c}: {} vs {want}",
            mu[c]
        );
    }
    for g in 0..2 {
        assert!((tr[g] - 100.0 / (1.0 + s[g])).abs() < 1e-10);
    }
    // its term is the independent prior's, densely
    let sec: Vec<usize> = section.clone();
    let (want_mu, want_tr, want_term) = dense_update(&Dense::identity(n, n), &s, &sec, &b);
    assert!(mu.iter().zip(&want_mu).all(|(a, w)| (a - w).abs() < 1e-10));
    assert!(tr.iter().zip(&want_tr).all(|(a, w)| (a - w).abs() < 1e-9));
    let term = prior.kl_term(0, &mu);
    assert!(
        (term - want_term).abs() < 1e-8 * want_term.abs(),
        "{term} vs {want_term}"
    );
    // zeta = 1 is the same factor
    let mut again = gp_prior(&pts, &section, 2, 20);
    again.hyper[0].zeta = 1.0;
    let mut mu2 = vec![0.0; n];
    again.update(0, &s, &b, &mut mu2);
    assert_eq!(mu2, mu);
}

/// With fewer inducing cells than cells, on one section and two, the ELBO
/// never falls across iterations that each end with a hyperparameter step.
#[test]
fn gp_elbo_never_falls_across_hyperparameter_steps_with_inducing_cells() {
    for sections in [1usize, 2] {
        let (mut y, _, xy, section) = planted_factors_on(10, 40, 40, sections);
        centre_sections(&mut y, &section, sections);
        let op = ResidualOp::from_dense(&y);
        let vd = VbData::new(&op, &section, sections);
        let pts = points(&xy);
        let mut prior = GpPrior::new(
            pts,
            section.clone(),
            sections,
            GpSizes {
                k: 2,
                n_inducing: 40,
                n_grid: 5,
                rank: 1,
                seed: 0,
            },
        );
        assert!(prior.inducing.len() < y.ncols());
        let mut st = State::init(&vd, 2, 1);
        let opts = VbOpts {
            tol: 1e-14,
            max_iter: 12,
            gp_start: 2,
            gp_every: 1,
            frozen_weights: false,
            hyper_tol: 1e-12,
        };
        let trace = vb::fit(&vd, &mut st, &mut prior, &opts);
        for (i, w) in trace.elbo.windows(2).enumerate() {
            assert!(
                w[1] >= w[0] - 1e-9 * w[0].abs(),
                "{sections} sections, iteration {}: ELBO fell {} -> {}",
                i + 2,
                w[0],
                w[1]
            );
        }
        // the planted smooth factors keep a lengthscale
        assert!(
            prior.hyper.iter().any(|h| !h.is_unstructured()),
            "{sections} sections: {:?}",
            prior.hyper
        );
    }
}

/// Inducing cells: every cell when they are few; else the count asked,
/// each section its share, no cell twice.
#[test]
fn inducing_cells_are_spread_over_sections() {
    let (xy, section) = jittered_grid(10, 2, 6);
    let pts: Vec<Vec<f64>> = xy.iter().map(|p| vec![p.0 as f64, p.1 as f64]).collect();
    assert_eq!(inducing_cells(&pts, &section, 2, 500, 0).len(), 200);
    let u = inducing_cells(&pts, &section, 2, 40, 0);
    assert_eq!(u.len(), 40);
    let mut d = u.clone();
    d.dedup();
    assert_eq!(d.len(), 40);
    assert_eq!(u.iter().filter(|&&c| section[c] == 0).count(), 20);
}

/// Data from the sparse model: `K` factors over two sections, sparse
/// weights, Gaussian noise; the expression (genes × cells) and the factors.
fn planted_factors(
    side: usize,
    genes: usize,
    seed: u64,
) -> (Dense, Dense, Vec<(f32, f32)>, Vec<usize>) {
    planted_factors_on(side, genes, seed, 2)
}

/// [`planted_factors`] on `sections` sections.
fn planted_factors_on(
    side: usize,
    genes: usize,
    seed: u64,
    sections: usize,
) -> (Dense, Dense, Vec<(f32, f32)>, Vec<usize>) {
    let (xy, section) = jittered_grid(side, sections, seed);
    let n = xy.len();
    let mut rng = SmallRng::seed_from_u64(seed);
    let z = Dense::from_fn(n, 2, |i, k| {
        let (x, y) = (xy[i].0 as f64, xy[i].1 as f64);
        match k {
            0 => (x / 3.0).sin() * 1.5,
            _ => (y / 4.0).cos() * 1.5,
        }
    });
    let w = Dense::from_fn(genes, 2, |g, k| {
        if g % 2 == k {
            rng.random_range(0.5..1.5)
        } else {
            0.0
        }
    });
    let noise = gaussian(genes, n, seed + 1) * 0.5;
    let y = &w * z.transpose() + noise;
    (y, z, xy, section)
}

/// Each gene centred within each section, as MEFISTO centres its groups.
fn centre_sections(y: &mut Dense, section: &[usize], sections: usize) {
    for s in 0..sections {
        let cells: Vec<usize> = (0..y.ncols()).filter(|&c| section[c] == s).collect();
        for g in 0..y.nrows() {
            let m = cells.iter().map(|&c| y[(g, c)]).sum::<f64>() / cells.len() as f64;
            cells.iter().for_each(|&c| y[(g, c)] -= m);
        }
    }
}

/// The fit stops with each `rho_k` (log scale) close to where a long run
/// with tight tolerances settles; stopping after any hyperparameter step
/// (no hyperparameter tolerance) does not.
#[test]
fn mefisto_stops_with_rho_fitted() {
    let (mut y, _, xy, section) = planted_factors(10, 40, 7);
    centre_sections(&mut y, &section, 2);
    let op = ResidualOp::from_dense(&y);
    let vd = VbData::new(&op, &section, 2);
    let lap = lap_of(&xy, &section);
    let spec = Spectrum::new(&lap, &exact());
    let run = |o: &VbOpts| {
        let mut prior = GraphPrior::new(&lap, &spec, 2, GraphPrior::start_rho(&lap), 1e-10, 20);
        let mut st = State::init(&vd, 2, 1);
        let trace = vb::fit(&vd, &mut st, &mut prior, o);
        (prior.rho.clone(), trace)
    };
    let default = VbOpts {
        tol: crate::spatial_baselines::mefisto::TOL,
        max_iter: crate::spatial_baselines::mefisto::MAX_ITER,
        gp_start: crate::spatial_baselines::mefisto::GP_START,
        gp_every: crate::spatial_baselines::mefisto::GP_EVERY,
        frozen_weights: false,
        hyper_tol: crate::spatial_baselines::mefisto::HYPER_TOL,
    };
    let long = VbOpts {
        tol: 1e-13,
        max_iter: 20_000,
        gp_every: 1,
        hyper_tol: 1e-9,
        ..default
    };
    let (rho, trace) = run(&default);
    let (want, long_trace) = run(&long);
    assert!(trace.converged && long_trace.converged);
    let gap = |a: &[f64]| {
        a.iter()
            .zip(&want)
            .map(|(x, w)| (x.ln() - w.ln()).abs())
            .fold(0f64, f64::max)
    };
    eprintln!(
        "rho {rho:?} after {} iterations; long run {want:?} after {}",
        trace.iterations, long_trace.iterations
    );
    assert!(gap(&rho) < 0.05, "{rho:?} vs {want:?}");
    let early = VbOpts {
        hyper_tol: f64::INFINITY,
        ..default
    };
    let (rho_early, _) = run(&early);
    assert!(gap(&rho_early) > 0.05, "{rho_early:?} vs {want:?}");
}

/// The section-wise data terms and each factor's likelihood terms are the
/// dense masked products'.
#[test]
fn vb_terms_are_the_dense_masked_products() {
    let (mut y, _, _, section) = planted_factors(5, 12, 3);
    centre_sections(&mut y, &section, 2);
    let op = ResidualOp::from_dense(&y);
    let yd = op.dense();
    let vd = VbData::new(&op, &section, 2);
    let mut st = State::init(&vd, 2, 1);
    let mut rng = SmallRng::seed_from_u64(4);
    st.gamma = Dense::from_fn(12, 2, |_, _| rng.random_range(0.2..0.9));
    st.mu_w = gaussian(12, 2, 5);
    st.s2_w = Dense::from_element(12, 2, 0.1);
    st.tau_b
        .iter_mut()
        .for_each(|v| *v *= rng.random_range(0.5..2.0));
    let n = y.ncols();
    let (per, gram) = vd.data_terms(&st.mu);
    for s in 0..2 {
        let masked = Dense::from_fn(
            n,
            2,
            |c, k| if section[c] == s { st.mu[(c, k)] } else { 0.0 },
        );
        let want = &yd * &masked;
        assert!((&per[s] - &want).amax() < 1e-10 * want.amax());
        let wg = masked.transpose() * &masked;
        assert!((&gram[s] - &wg).amax() < 1e-10 * wg.amax());
    }
    let terms = vb::FactorTerms::new(&vd, &st);
    let tau = st.tau_a.component_div(&st.tau_b);
    let wm = st.w_mean();
    let w2 = st
        .gamma
        .component_mul(&(st.mu_w.component_mul(&st.mu_w) + &st.s2_w));
    for k in 0..2 {
        let (s, b) = terms.of(&vd, k, &st.mu);
        for g in 0..2 {
            let want: f64 = (0..12).map(|d| w2[(d, k)] * tau[(d, g)]).sum();
            assert!((s[g] - want).abs() < 1e-12 * want);
        }
        for c in 0..n {
            let g = section[c];
            let mut want: f64 = (0..12).map(|d| tau[(d, g)] * wm[(d, k)] * yd[(d, c)]).sum();
            for l in (0..2).filter(|&l| l != k) {
                let cross: f64 = (0..12).map(|d| tau[(d, g)] * wm[(d, k)] * wm[(d, l)]).sum();
                want -= cross * st.mu[(c, l)];
            }
            assert!((b[c] - want).abs() < 1e-10 * want.abs().max(1.0), "{c}");
        }
    }
}

/// The domain step clusters factors scaled by their weights' norm: on a
/// planted two-domain section with a second, unstructured factor carrying
/// almost no weight but wider spread, the domains follow the planted one;
/// unscaled, the unstructured factor drives them.
#[test]
fn mefisto_domains_are_not_driven_by_a_factor_without_weight() {
    use crate::spatial_baselines::mefisto::domain_embedding;
    use crate::spatial_baselines::spatial_factor::{domains, to_mat};
    let (xy, _) = jittered_grid(20, 1, 40);
    let n = xy.len();
    let mut rng = SmallRng::seed_from_u64(41);
    let planted: Vec<usize> = xy.iter().map(|p| usize::from(p.0 >= 10.0)).collect();
    let mu = Dense::from_fn(n, 2, |c, k| match k {
        0 => 2.0 * planted[c] as f64 - 1.0 + rng.random_range(-0.3..0.3),
        _ => {
            if rng.random_bool(0.5) {
                3.0
            } else {
                -3.0
            }
        }
    });
    let w = Dense::from_fn(
        30,
        2,
        |g, k| if k == 0 { 1.0 } else { 0.01 * (g % 3) as f64 },
    );
    let crate::Commands::Mefisto(a) = parse(
        "mefisto",
        &["d.zarr".into(), "-c".into(), "c.csv".into()],
        "x",
        &["--n-clusters", "2"],
    ) else {
        unreachable!()
    };
    let (scaled, _) = domains(&to_mat(&domain_embedding(&mu, &w)), &a.fit.core, None).unwrap();
    let (raw, _) = domains(&to_mat(&mu), &a.fit.core, None).unwrap();
    let (a_scaled, a_raw) = (ari(&planted, &scaled), ari(&planted, &raw));
    assert!(a_scaled > 0.9, "scaled {a_scaled}");
    assert!(a_raw < 0.5, "unscaled {a_raw}");
}

/// Bar, set before measuring: on data drawn from the model, the sparse fit
/// recovers each planted factor with |correlation| at least 0.9, and with
/// the hyperparameter step off the ELBO never falls (to rounding).
#[test]
fn mefisto_recovers_planted_factors_and_ascends_its_elbo() {
    let (mut y, z, xy, section) = planted_factors(10, 40, 7);
    centre_sections(&mut y, &section, 2);
    let op = ResidualOp::from_dense(&y);
    let vd = VbData::new(&op, &section, 2);
    let lap = lap_of(&xy, &section);
    let spec = Spectrum::new(&lap, &exact());
    let mut prior = GraphPrior::new(&lap, &spec, 2, GraphPrior::start_rho(&lap), 1e-10, 20);
    let mut st = State::init(&vd, 2, 1);
    let flat = VbOpts {
        tol: 1e-12,
        max_iter: 30,
        gp_start: 1000,
        gp_every: 10,
        frozen_weights: false,
        hyper_tol: 1e-3,
    };
    let trace = vb::fit(&vd, &mut st, &mut prior, &flat);
    for w in trace.elbo.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-8 * w[0].abs(),
            "ELBO fell: {} -> {}",
            w[0],
            w[1]
        );
    }
    let full = VbOpts {
        tol: 5e-6,
        max_iter: 300,
        gp_start: 1,
        ..flat
    };
    let trace = vb::fit(&vd, &mut st, &mut prior, &full);
    assert!(trace.iterations > 1);
    let corr = |a: &Dense, i: usize, b: &Dense, j: usize| {
        let (x, y) = (a.column(i), b.column(j));
        let (mx, my) = (x.mean(), y.mean());
        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(p, q)| (p - mx) * (q - my))
            .sum();
        let vx: f64 = x.iter().map(|p| (p - mx).powi(2)).sum();
        let vy: f64 = y.iter().map(|q| (q - my).powi(2)).sum();
        (cov / (vx * vy).sqrt()).abs()
    };
    for k in 0..2 {
        let best = (0..2).map(|j| corr(&z, k, &st.mu, j)).fold(0f64, f64::max);
        assert!(best >= 0.9, "factor {k}: |corr| {best}");
    }
    assert!(prior.rho.iter().any(|&r| r > 0.0), "{:?}", prior.rho);
}

/// Bar, set before measuring: on a simulated section of 576 cells the
/// sparse fit's domains recover the planted ones at an ARI of at least 0.5;
/// the reference (a full GP at this size) runs to the same outputs.
#[test]
fn mefisto_recovers_planted_domains() {
    let dir = tempfile::tempdir().unwrap();
    let (sim, files) = simulated_section(dir.path(), 24, 200, 4);
    let probe = dir.path().join("probe").to_string_lossy().to_string();
    let crate::Commands::Mefisto(a) = parse("mefisto", &files, &probe, &[]) else {
        unreachable!()
    };
    let data = a.fit.core.input.read(false).unwrap();
    let (domain, k) = planted(&sim, &data.cells);
    let k = k.to_string();
    let out = dir.path().join("sparse").to_string_lossy().to_string();
    let crate::Commands::Mefisto(a) = parse("mefisto", &files, &out, &["--n-clusters", &k]) else {
        unreachable!()
    };
    crate::spatial_baselines::mefisto::run(&a).unwrap();
    let score = ari(&domain, &labels_of(&data.cells, &read_labels(&out)));
    eprintln!("mefisto ARI {score:.3}");
    for f in [
        "pinto.json",
        "cell_embedding.parquet",
        "baseline.json",
        "mefisto_weights.parquet",
    ] {
        assert!(std::path::Path::new(&format!("{out}.{f}")).exists(), "{f}");
    }
    assert!(score >= 0.5, "ARI {score}");
}

/// The reference runs end to end on a small section: a full GP below the
/// inducing count, then on inducing cells above it.
#[test]
fn mefisto_reference_runs_dense_and_inducing() {
    let dir = tempfile::tempdir().unwrap();
    let (_, files) = simulated_section(dir.path(), 10, 80, 5);
    for (name, inducing) in [("dense", "1000"), ("inducing", "40")] {
        let out = dir.path().join(name).to_string_lossy().to_string();
        let crate::Commands::Mefisto(a) = parse(
            "mefisto",
            &files,
            &out,
            &[
                "--reference",
                "--n-clusters",
                "3",
                "--n-factors",
                "2",
                "--max-iter",
                "25",
                "--n-grid",
                "4",
                "--n-inducing",
                inducing,
            ],
        ) else {
            unreachable!()
        };
        crate::spatial_baselines::mefisto::run(&a).unwrap();
        let s: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}.baseline.json")).unwrap())
                .unwrap();
        assert_eq!(s["run"]["mode"], "reference");
        let n_ind = s["settings"]["n_inducing"].as_u64().unwrap();
        assert_eq!(n_ind, if name == "dense" { 100 } else { 40 });
    }
}

/// Bar, set before measuring: on a simulated section of 576 cells, above
/// `--n-inducing`, the reference fit keeps a lengthscale on some factor
/// and its domains (at the planted count) are within 0.1 ARI of the
/// sparse mode's.
#[test]
fn mefisto_reference_with_inducing_cells_keeps_spatial_factors() {
    let dir = tempfile::tempdir().unwrap();
    let (sim, files) = simulated_section(dir.path(), 24, 200, 4);
    let probe = dir.path().join("probe").to_string_lossy().to_string();
    let crate::Commands::Mefisto(a) = parse("mefisto", &files, &probe, &[]) else {
        unreachable!()
    };
    let data = a.fit.core.input.read(false).unwrap();
    let (domain, k) = planted(&sim, &data.cells);
    let k = k.to_string();
    let mut scores = Vec::new();
    for (name, extra) in [
        ("sparse", vec![]),
        ("reference", vec!["--reference", "--n-inducing", "300"]),
    ] {
        let out = dir.path().join(name).to_string_lossy().to_string();
        let mut flags = vec!["--n-clusters", k.as_str()];
        flags.extend(extra);
        let crate::Commands::Mefisto(a) = parse("mefisto", &files, &out, &flags) else {
            unreachable!()
        };
        crate::spatial_baselines::mefisto::run(&a).unwrap();
        scores.push(ari(&domain, &labels_of(&data.cells, &read_labels(&out))));
        if name == "reference" {
            let s: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(format!("{out}.baseline.json")).unwrap(),
            )
            .unwrap();
            let st = &s["settings"]["prior_state"];
            assert_eq!(st["n_inducing"], 300);
            let ls: Vec<f64> = st["factors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| f["lengthscale"].as_f64().unwrap())
                .collect();
            eprintln!("reference lengthscales {ls:?}");
            assert!(ls.iter().any(|&l| l > 0.0), "{ls:?}");
        }
    }
    eprintln!(
        "mefisto ARI sparse {:.3}, reference {:.3}",
        scores[0], scores[1]
    );
    assert!(scores[1] >= scores[0] - 0.1, "{scores:?}");
}
