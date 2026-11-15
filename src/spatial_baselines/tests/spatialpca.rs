//! SpatialPCA: the sparse profile and the reference kernel forms against
//! brute-force likelihoods and dense algebra, each with a broken variant
//! the check must catch; then recovery of planted domains.

use super::graph::toy;
use super::pseudobulk::{ari, labels_of, read_column, read_labels};
use crate::spatial_baselines::graph::knn_within_sections;
use crate::spatial_baselines::spatial_factor::kit::{gaussian, sym_eig_desc, Dense};
use crate::spatial_baselines::spatial_factor::prep::ResidualOp;
use crate::spatial_baselines::spatial_factor::prior::{Laplacian, Spectrum, SpectrumOpts};
use crate::spatial_baselines::spatialpca::reference::{self, Eigen, KernelBlock, TauProfile};
use crate::spatial_baselines::spatialpca::sparse::Profile;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// `m × n` integer expression, each gene summing to 0 exactly.
fn centred_integers(m: usize, n: usize, seed: u64) -> Dense {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut y = Dense::from_fn(m, n, |_, _| rng.random_range(-5i32..=5) as f64);
    for g in 0..m {
        let s: f64 = (0..n - 1).map(|c| y[(g, c)]).sum();
        y[(g, n - 1)] = -s;
    }
    y
}

fn random_points(n: usize, seed: u64) -> Vec<(f32, f32)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (rng.random_range(0.0..5.0f32), rng.random_range(0.0..5.0f32)))
        .collect()
}

fn exact() -> SpectrumOpts {
    SpectrumOpts {
        exact_eig_max: 100_000,
        probes: 1,
        steps: 1,
        seed: 0,
    }
}

fn lap_of(xy: &[(f32, f32)], section: &[usize], k: usize) -> Laplacian {
    let d = toy(xy, section);
    let nb = knn_within_sections(&d, k).symmetrised();
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

/// The restricted (intercepts integrated out under a flat prior) Gaussian
/// log-likelihood of `Y` under `Y = b 1^T + W Z + E`, `Z_l ~ N(0, sigma2
/// tau P)`, `E ~ N(0, sigma2)`, from the `mn × mn` covariance of `vec(Y)`,
/// up to a constant in `(m, n)`; `reml = false` drops the intercepts' term
/// (`M = I`, a broken variant).
fn restricted_loglik(y: &Dense, w: &Dense, sigma2: f64, tau: f64, p: &Dense, reml: bool) -> f64 {
    let (m, n) = y.shape();
    let ww = w * w.transpose();
    let big = m * n;
    let sigma = Dense::from_fn(big, big, |a, b| {
        let (j, i) = (a / n, a % n);
        let (k, l) = (b / n, b % n);
        sigma2 * (tau * ww[(j, k)] * p[(i, l)] + if a == b { 1.0 } else { 0.0 })
    });
    let chol = sigma.clone().cholesky().unwrap();
    let logdet = 2.0 * chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
    let vy = Dense::from_fn(big, 1, |a, _| y[(a / n, a % n)]);
    let x = Dense::from_fn(big, m, |a, j| if a / n == j { 1.0 } else { 0.0 });
    let si_y = chol.solve(&vy);
    if !reml {
        return -0.5 * logdet - 0.5 * (vy.transpose() * si_y)[(0, 0)];
    }
    let si_x = chol.solve(&x);
    let xsx = x.transpose() * &si_x;
    let xsy = x.transpose() * &si_y;
    let xchol = xsx.clone().cholesky().unwrap();
    let xlogdet = 2.0 * xchol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
    let quad = (vy.transpose() * &si_y)[(0, 0)] - (xsy.transpose() * xchol.solve(&xsy))[(0, 0)];
    -0.5 * logdet - 0.5 * xlogdet - 0.5 * quad
}

/// The sparse prior's normalised covariance `R(rho) = (I + rho L)^{-1} /
/// cbar(rho)`, densely.
fn r_of(lap: &Laplacian, rho: f64) -> Dense {
    let n = lap.n();
    let inv = (Dense::identity(n, n) + dense_lap(lap) * rho)
        .try_inverse()
        .unwrap();
    let cbar = inv.trace() / n as f64;
    inv / cbar
}

/// The sparse profile `ell` differs across `(tau, rho)` exactly as the
/// brute-force restricted likelihood at the profiled `sigma0^2` and `W`;
/// dropping its `-log(1 + t)` term or the intercepts' term breaks that.
/// No orthonormal perturbation of `W` raises the likelihood.
#[test]
fn sparse_profile_is_the_restricted_likelihood() {
    let (m, n, d) = (4, 12, 2);
    let xy = random_points(n, 1);
    let lap = lap_of(&xy, &vec![0; n], 3);
    let spec = Spectrum::new(&lap, &exact());
    let y = centred_integers(m, n, 2);
    let op = ResidualOp::from_dense(&y);
    let prof = Profile::new(&op, &lap, &spec, d, 1e-14, 0);
    let mut evals = Vec::new();
    for (tau, rho) in [(0.3, 0.5), (1.0, 2.0), (3.0, 0.7), (0.8, 6.0)] {
        let e = prof.eval(f64::ln(tau), f64::ln(rho));
        let r = r_of(&lap, rho);
        let s2 = e.sigma2(m, n);
        let brute = restricted_loglik(&y, &e.w, s2, tau, &r, true);
        let plain = restricted_loglik(&y, &e.w, s2, tau, &r, false);
        let k = m as f64 * (n as f64 - 1.0) / 2.0;
        let dropped = -k * e.s.ln() - d as f64 / 2.0 * (e.h + e.t.ln_1p());
        evals.push((e.ell, brute, plain, dropped));
        // W is the maximiser at this (tau, rho)
        let mut rng = SmallRng::seed_from_u64(9);
        for _ in 0..5 {
            let mut skew = Dense::zeros(m, m);
            for a in 0..m {
                for b in a + 1..m {
                    let v = rng.random_range(-0.05..0.05);
                    skew[(a, b)] = v;
                    skew[(b, a)] = -v;
                }
            }
            let rot = (Dense::identity(m, m) - &skew).try_inverse().unwrap()
                * (Dense::identity(m, m) + &skew);
            let w2 = &rot * &e.w;
            let wt = w2.transpose() * &y;
            let explained: f64 = (0..d)
                .map(|l| {
                    let u = wt.row(l).transpose();
                    let c =
                        Dense::identity(n, n) * (1.0 + 1.0 / e.t) + dense_lap(&lap) * (rho / e.t);
                    (u.transpose() * c.lu().solve(&u).unwrap())[(0, 0)]
                })
                .sum();
            let s2b = (y.norm_squared() - explained) / (m as f64 * (n as f64 - 1.0));
            let other = restricted_loglik(&y, &w2, s2b, tau, &r, true);
            assert!(other <= brute + 1e-9, "{other} > {brute}");
        }
    }
    for w in evals.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let want = b.1 - a.1;
        assert!(
            ((b.0 - a.0) - want).abs() < 1e-8 * want.abs().max(1.0),
            "{} vs {want}",
            b.0 - a.0
        );
        assert!(((b.3 - a.3) - want).abs() > 1e-4, "-log(1 + t) dropped");
        assert!(((b.0 - a.0) - (b.2 - a.2)).abs() > 1e-4, "M = I");
    }
}

/// The analytic gradient in `(log tau, log rho)` is the central difference.
#[test]
fn sparse_profile_gradient_is_the_central_difference() {
    let n = 60;
    let xy = random_points(n, 3);
    let section: Vec<usize> = (0..n).map(|i| i / 30).collect();
    let lap = lap_of(&xy, &section, 5);
    let spec = Spectrum::new(&lap, &exact());
    let y = centred_integers(8, n, 4);
    let op = ResidualOp::from_dense(&y);
    let prof = Profile::new(&op, &lap, &spec, 3, 1e-13, 0);
    let (lt, lr) = (0.4f64, -0.3f64);
    let e = prof.eval(lt, lr);
    let h = 1e-5;
    let fd0 = (prof.eval(lt + h, lr).ell - prof.eval(lt - h, lr).ell) / (2.0 * h);
    let fd1 = (prof.eval(lt, lr + h).ell - prof.eval(lt, lr - h).ell) / (2.0 * h);
    for (a, b) in [(e.grad[0], fd0), (e.grad[1], fd1)] {
        assert!((a - b).abs() < 1e-5 * b.abs().max(1.0), "{a} vs {b}");
    }
    // with the loadings fixed too
    let fixed = Profile::new(&op, &lap, &spec, 3, 1e-13, 0).with_loadings(e.w.clone());
    let f = fixed.eval(lt, lr);
    let fd = (fixed.eval(lt, lr + h).ell - fixed.eval(lt, lr - h).ell) / (2.0 * h);
    assert!((f.grad[1] - fd).abs() < 1e-5 * fd.abs().max(1.0));
}

/// On two sections, `G`'s top eigenvalues and the PCs are the dense twin's
/// (`C` inverted densely); a fit with quadrature in place of the exact
/// spectrum lands within 10% (in log scale) of the exact fit.
#[test]
fn sparse_profile_matches_its_dense_twin() {
    let n = 200;
    let xy = random_points(n, 5);
    let section: Vec<usize> = (0..n).map(|i| i / 100).collect();
    let lap = lap_of(&xy, &section, 5);
    let spec = Spectrum::new(&lap, &exact());
    let mut y = gaussian(20, n, 6);
    // a smooth signal for the profile to find
    for g in 0..5 {
        for c in 0..n {
            y[(g, c)] += 2.0 * (xy[c].0 as f64).sin();
        }
    }
    let y = y.map(|v| (v as f32) as f64);
    let op = ResidualOp::from_dense(&y);
    // centre through the operator's dense form for the twin
    let yd = op.dense();
    let prof = Profile::new(&op, &lap, &spec, 3, 1e-12, 0);
    for (tau, rho) in [(0.5, 1.0), (4.0, 0.2), (2.0, 8.0)] {
        let e = prof.eval(f64::ln(tau), f64::ln(rho));
        let c = Dense::identity(n, n) * (1.0 + 1.0 / e.t) + dense_lap(&lap) * (rho / e.t);
        let cinv = c.try_inverse().unwrap();
        let g = &yd * &cinv * yd.transpose();
        let (vals, _) = sym_eig_desc(&g);
        for l in 0..3 {
            assert!((e.explained[l] - vals[l]).abs() < 1e-7 * vals[l], "{l}");
        }
        let z = &cinv * yd.transpose() * &e.w;
        assert!((&e.z - &z).amax() < 1e-6 * z.amax());
    }
    let (fit, _) = prof.fit(50);
    let slq = Spectrum::new(
        &lap,
        &SpectrumOpts {
            exact_eig_max: 0,
            probes: 32,
            steps: 60,
            seed: 1,
        },
    );
    let (approx, _) = Profile::new(&op, &lap, &slq, 3, 1e-10, 0).fit(50);
    assert!(
        (approx.tau / fit.tau).ln().abs() < 0.1,
        "{} vs {}",
        approx.tau,
        fit.tau
    );
    assert!(
        (approx.rho / fit.rho).ln().abs() < 0.1,
        "{} vs {}",
        approx.rho,
        fit.rho
    );
    assert!((approx.sigma2(20, n) / fit.sigma2(20, n) - 1.0).abs() < 1e-2);
}

/// One kernel block over `xy` at bandwidth `gamma`, its exact eigenpairs.
fn kernel(xy: &[(f32, f32)], gamma: f64) -> (Vec<KernelBlock>, Eigen, Dense) {
    let n = xy.len();
    let pts: Vec<Vec<f64>> = xy.iter().map(|p| vec![p.0 as f64, p.1 as f64]).collect();
    let block = KernelBlock {
        cells: (0..n).collect(),
        xy: reference::standardised(&pts),
        gamma,
    };
    let blocks = vec![block];
    let eig = reference::kernel_eigen(&blocks, n, n, 0);
    let k = reference::kernel_apply(&blocks, &Dense::identity(n, n));
    (blocks, eig, k)
}

/// At full rank the reference's `G` (SN(11), SN(12)) is
/// `Y M (M + tau^{-1} K^{-1})^{-1} M Y^T`, its PCs (SN(13)) are SN(10)'s,
/// and its profile differs across `tau` as the brute-force restricted
/// likelihood; without the intercepts' rank-one term `G` is wrong.
#[test]
fn reference_forms_are_the_paper_s_at_full_rank() {
    let (m, n, d) = (4, 12, 2);
    let xy = random_points(n, 7);
    let (blocks, eig, k) = kernel(&xy, 2.0);
    let y = centred_integers(m, n, 8);
    let prof = TauProfile::new(&y, &eig, n, d);
    let mm = Dense::identity(n, n) - Dense::from_element(n, n, 1.0 / n as f64);
    let kinv = k.clone().try_inverse().unwrap();
    let mut prev: Option<(f64, f64, f64)> = None;
    for tau in [0.2, 1.0, 5.0] {
        let e = prof.eval(tau);
        let inner = (&mm + &kinv / tau).try_inverse().unwrap();
        let g = &y * &mm * &inner * &mm * y.transpose();
        let (vals, _) = sym_eig_desc(&g);
        for l in 0..d {
            assert!((e.eigvals[l] - vals[l]).abs() < 1e-8 * vals[0], "{l}");
        }
        // the broken G, without the intercepts' term
        let f = (Dense::identity(n, n) + &kinv / tau).try_inverse().unwrap();
        let yfy = &y * f * y.transpose();
        let (bvals, _) = sym_eig_desc(&yfy);
        assert!((bvals[0] - vals[0]).abs() > 1e-6 * vals[0]);
        let z = reference::pcs(&y, &e.w, tau, &eig, n, &blocks);
        let want = &inner * &mm * y.transpose() * &e.w;
        assert!((&z - &want).amax() < 1e-8 * want.amax(), "SN(13) vs SN(10)");
        let s2 = e.s / (m as f64 * (n as f64 - 1.0));
        let brute = restricted_loglik(&y, &e.w, s2, tau, &k, true);
        let plain = restricted_loglik(&y, &e.w, s2, tau, &k, false);
        if let Some((pe, pb, pp)) = prev {
            let want = brute - pb;
            assert!(
                ((e.ell - pe) - want).abs() < 1e-7 * want.abs().max(1.0),
                "{} vs {want}",
                e.ell - pe
            );
            assert!(((e.ell - pe) - (plain - pp)).abs() > 1e-4, "M = I");
        }
        prev = Some((e.ell, brute, plain));
    }
}

/// The rank rule: 90% of the trace at or below 5000 cells; 20 and a tenth
/// of the cells above; a set rank for both.
#[test]
fn reference_ranks_follow_the_paper_s_rule() {
    let vals = [5.0, 3.0, 1.0, 0.5, 0.5];
    assert_eq!(reference::ranks(&vals, 10, None), (3, 3));
    let many = vec![1.0; 6000];
    assert_eq!(reference::ranks(&many, 6000, None), (20, 600));
    assert_eq!(reference::ranks(&vals, 10, Some(2)), (2, 2));
}

/// Silverman's rule on a known sample; Sheather-Jones near the normal
/// reference value on a normal sample, and 0 with no spread.
#[test]
fn bandwidth_rules() {
    let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
    // sd 1.5811, IQR 2 (type 7), n^(-1/5)
    let want = 0.9 * (2.0f64 / 1.34).min(1.581_138_830_084_189_8) * 5f64.powf(-0.2);
    assert!((reference::silverman(&x) - want).abs() < 1e-12);
    let g = gaussian(1000, 1, 11);
    let x: Vec<f64> = g.iter().copied().collect();
    let h = reference::sheather_jones(&x);
    assert!(h > 0.2 && h < 0.4, "{h}");
    assert_eq!(reference::sheather_jones(&[1.0; 50]), 0.0);
    assert_eq!(
        reference::bandwidth(&vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]; 3]),
        reference::sheather_jones(&[0.0, 1.0, 2.0, 3.0, 4.0])
    );
}

// ------------------------------------------------- end to end

/// A simulated section of `side × side` cells with `genes` genes: its
/// directory and the data arguments.
pub(crate) fn simulated_section(
    dir: &std::path::Path,
    side: usize,
    genes: usize,
    seed: u64,
) -> (String, Vec<String>) {
    let sim = dir.join("sim").to_string_lossy().to_string();
    crate::simulate::run::run_simulate(&crate::simulate::run::SimulateArgs {
        scenario: "L0".into(),
        set: [
            format!("grid_side={side}"),
            format!("n_genes={genes}"),
            "n_donors=1".into(),
            "sections_per_donor=1".into(),
            format!("seed={seed}"),
        ]
        .into(),
        coords: Vec::new(),
        coord_column_names: vec!["row_um".into(), "col_um".into()],
        out: sim.as_str().into(),
    })
    .unwrap();
    let args = vec![
        format!("{sim}/sections/S1.zarr"),
        "-c".into(),
        format!("{sim}/sections/S1.coords.parquet"),
        "--coord-column-names".into(),
        "row_um,col_um".into(),
    ];
    (sim, args)
}

pub(crate) fn parse(cmd: &str, files: &[String], out: &str, extra: &[&str]) -> crate::Commands {
    let mut argv = vec!["pinto".to_string(), cmd.into()];
    argv.extend(files.iter().cloned());
    argv.extend(["-o".into(), out.into()]);
    argv.extend(extra.iter().map(|s| s.to_string()));
    <crate::Cli as clap::Parser>::try_parse_from(argv)
        .unwrap_or_else(|e| panic!("{e}"))
        .commands
}

/// The planted domains of a simulation, in the order of `cells`, and their
/// number.
pub(crate) fn planted(sim: &str, cells: &[Box<str>]) -> (Vec<usize>, usize) {
    let truth = read_column(&format!("{sim}/truth/spots.parquet"), "barcode", "domain");
    let d = labels_of(cells, &truth);
    let k = d.iter().max().map_or(0, |&m| m + 1);
    (d, k)
}

/// Bar, set before measuring: on a simulated section of 576 cells, both
/// modes recover the planted domains at an ARI of at least 0.5. The run
/// writes the shared outputs and its loadings.
#[test]
fn spatialpca_recovers_planted_domains() {
    let dir = tempfile::tempdir().unwrap();
    let (sim, files) = simulated_section(dir.path(), 24, 200, 4);
    let probe = dir.path().join("probe").to_string_lossy().to_string();
    let crate::Commands::Spatialpca(a) = parse("spatialpca", &files, &probe, &[]) else {
        unreachable!()
    };
    let data = a.fit.core.input.read(false).unwrap();
    let (domain, k) = planted(&sim, &data.cells);
    let k = k.to_string();
    let mut scores = Vec::new();
    for (name, extra) in [("sparse", vec![]), ("reference", vec!["--reference"])] {
        let out = dir.path().join(name).to_string_lossy().to_string();
        let mut flags = vec!["--n-clusters", k.as_str(), "--n-factors", "10"];
        flags.extend(extra);
        let crate::Commands::Spatialpca(a) = parse("spatialpca", &files, &out, &flags) else {
            unreachable!()
        };
        crate::spatial_baselines::spatialpca::run(&a).unwrap();
        let labels = labels_of(&data.cells, &read_labels(&out));
        scores.push((name, ari(&domain, &labels)));
        for f in [
            "pinto.json",
            "cell_embedding.parquet",
            "baseline.json",
            "spatialpca_loadings.parquet",
        ] {
            assert!(std::path::Path::new(&format!("{out}.{f}")).exists(), "{f}");
        }
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}.pinto.json")).unwrap())
                .unwrap();
        assert_eq!(meta["command"], "spatialpca");
    }
    eprintln!("spatialpca ARI: {scores:?}");
    for (name, a) in scores {
        assert!(a >= 0.5, "{name}: ARI {a}");
    }
}

/// Counts divided by a δ of one everywhere give the run `--ignore-batch`
/// gives, bit for bit.
#[test]
fn a_delta_of_one_is_ignore_batch() {
    let dir = tempfile::tempdir().unwrap();
    let (_, files) = simulated_section(dir.path(), 12, 120, 5);
    let out_a = dir.path().join("a").to_string_lossy().to_string();
    let out_b = dir.path().join("b").to_string_lossy().to_string();
    let run = |out: &str, delta: bool| {
        let crate::Commands::Spatialpca(a) = parse(
            "spatialpca",
            &files,
            out,
            &["--ignore-batch", "--n-clusters", "3", "--n-factors", "4"],
        ) else {
            unreachable!()
        };
        let mut data = a.fit.core.input.read(false).unwrap();
        if delta {
            data.delta = Some(crate::spatial_baselines::input::AppliedDelta {
                batches: vec!["B1".into()],
            });
        }
        let budget = crate::util::memory::Budget::measured(Some(usize::MAX));
        crate::spatial_baselines::spatialpca::run_on(&a, data, &budget).unwrap();
        std::fs::read(format!("{out}.cell_embedding.parquet")).unwrap()
    };
    assert_eq!(run(&out_a, false), run(&out_b, true));
}
