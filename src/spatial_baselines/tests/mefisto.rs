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

/// The graph prior's `rho` step lands on the dense objective's maximum
/// over `rho`.
#[test]
fn graph_prior_rho_step_maximises_the_dense_objective() {
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
    let before = prior.kl_term(0, &mu);
    prior.hyper(0, &mut mu);
    let after = prior.kl_term(0, &mu);
    assert!(after >= before - 1e-9, "{after} < {before}");
    // no other rho on a fine grid beats it
    let rho = prior.rho[0];
    let l = dense_lap(&lap);
    let a = (Dense::identity(n, n) * (s[0] + 0.0) + {
        let c0 = spec.cbar(0.3, n).0;
        (Dense::identity(n, n) + &l * 0.3) * c0
    })
    .try_inverse()
    .unwrap();
    let m = Dense::from_column_slice(n, 1, &mu);
    let f = |r: f64| {
        let c = spec.cbar(r, n).0;
        let q = (Dense::identity(n, n) + &l * r) * c;
        0.5 * logdet(&q) - 0.5 * (&q * &a).trace() - 0.5 * (m.transpose() * &q * &m)[(0, 0)]
    };
    let fr = f(rho);
    for i in 0..60 {
        let r = 10f64.powf(-3.0 + 6.0 * i as f64 / 59.0);
        assert!(f(r) <= fr + 1e-6 * fr.abs(), "rho {r} beats {rho}");
    }
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
    // the hyperparameter step never lowers its objective (the factor term)
    let before = prior.kl_term(0, &mu);
    prior.hyper(0, &mut mu);
    assert!(prior.kl_term(0, &mu) >= before - 1e-8 * before.abs());
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
    let (xy, section) = jittered_grid(side, 2, seed);
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

/// Bar, set before measuring: on data drawn from the model, the sparse fit
/// recovers each planted factor with |correlation| at least 0.9, and with
/// the hyperparameter step off the ELBO never falls (to rounding).
#[test]
fn mefisto_recovers_planted_factors_and_ascends_its_elbo() {
    let (mut y, z, xy, section) = planted_factors(10, 40, 7);
    // centred within each section, as MEFISTO centres its groups
    for s in 0..2 {
        let cells: Vec<usize> = (0..y.ncols()).filter(|&c| section[c] == s).collect();
        for g in 0..y.nrows() {
            let m = cells.iter().map(|&c| y[(g, c)]).sum::<f64>() / cells.len() as f64;
            cells.iter().for_each(|&c| y[(g, c)] -= m);
        }
    }
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
