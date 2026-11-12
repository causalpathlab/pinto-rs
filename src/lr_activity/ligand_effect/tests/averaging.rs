//! Choices averaged rather than asked for, and what each pair reports
//! beside its headline: the competence grid combined by Rubin's rules, the
//! average over exposure thresholds, the threshold–response curve with
//! its simultaneous band, the two-scale
//! moments and the per-cell-type rows.

use super::{estimate_pair, fitted_effects};
use crate::lr_activity::ligand_effect::averaging::{abf_average, abf_weights, combine};
use crate::lr_activity::ligand_effect::curve::{curve_taus, sup_t_crit};
use crate::lr_activity::ligand_effect::dose::Exposure;
use crate::lr_activity::ligand_effect::effects::{effects_on, Sensitivity};
use crate::lr_activity::ligand_effect::estimate::*;
use crate::lr_activity::ligand_effect::fits::{fit_main, Fit, Rows, ThresholdFits};
use crate::lr_activity::ligand_effect::inference::{Estimate, Z_975};
use crate::lr_activity::ligand_effect::options::EffectOpts;
use crate::lr_activity::ligand_effect::report::{report, ReportOpts};
use crate::lr_activity::ligand_effect::response::Response;
use crate::lr_activity::ligand_effect::rows::EffectRow;
use crate::lr_activity::ligand_effect::rows::Moments;
use crate::lr_activity::ligand_effect::tissue::{Tissue, TissueInputs, TissueOpts};
use crate::util::common::Mat;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson, StandardNormal};

const SIDE: usize = 40;

fn est(theta: f64, se: f64, df: f64) -> Estimate {
    Estimate {
        theta,
        se,
        n: 100,
        n_treated: 40,
        mean_h: 0.2,
        ess: 80.0,
        df,
    }
}

struct World {
    tissue: Tissue,
    ligand: Vec<f64>,
    receptor: Vec<f64>,
    response: Vec<(u32, f32)>,
}

fn poisson(rng: &mut SmallRng, rate: f64) -> f32 {
    Poisson::new(rate.max(1e-9)).unwrap().sample(rng) as f32
}

/// A grid whose cells touch their four neighbours, one section, three
/// identity bins: bin 0 detects the receptor in every cell, bin 1 in about
/// 15% (competent at the grid's 0.05 and 0.1, not at 0.2), bin 2 never.
/// A random 30% of cells carry ligand. Competent receivers' programme rate
/// rises with dose, `2 exp(dose · a)`, `a` the number of ligand-carrying
/// contacts. Two cell types in 8×8 patches, named when `named`.
fn world(seed: u64, dose: f64, named: bool) -> World {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = SIDE * SIDE;
    let coords: Vec<Option<(f32, f32)>> = (0..n)
        .map(|i| Some(((i % SIDE) as f32 * 10.0, (i / SIDE) as f32 * 10.0)))
        .collect();
    let mut pairs = Vec::new();
    for i in 0..n {
        if i % SIDE + 1 < SIDE {
            pairs.push((i, i + 1));
        }
        if i / SIDE + 1 < SIDE {
            pairs.push((i, i + SIDE));
        }
    }
    let section: Vec<Box<str>> = vec!["s".into(); n];
    let identity_bin: Vec<usize> = (0..n).map(|i| (i * 7 + i / SIDE) % 3).collect();
    let ty: Vec<u32> = (0..n)
        .map(|i| (((i % SIDE) / 8 + (i / SIDE) / 8) % 2) as u32)
        .collect();
    let identity = Mat::from_fn(4, n, |_, _| {
        let e: f64 = StandardNormal.sample(&mut rng);
        e as f32
    });
    let log1p_depth = vec![(500.0f32).ln_1p(); n];
    let mut tissue = Tissue::build(
        &TissueInputs {
            coords: &coords,
            section: &section,
            pairs: &pairs,
            identity: &identity,
            identity_bin: &identity_bin,
            log1p_depth: &log1p_depth,
            cell_type: Some(&ty),
        },
        &TissueOpts {
            folds: 4,
            tile: 6.0,
            ..TissueOpts::default()
        },
    )
    .unwrap();
    if named {
        tissue = tissue.with_cell_type_names(vec!["epi".into(), "fib".into()]);
    }
    let receptor: Vec<f64> = identity_bin
        .iter()
        .map(|&b| match b {
            0 => 1.0,
            1 => (rng.random::<f64>() < 0.15) as u8 as f64,
            _ => 0.0,
        })
        .collect();
    let ligand: Vec<f64> = (0..n)
        .map(|_| if rng.random::<f64>() < 0.3 { 3.0 } else { 0.0 })
        .collect();
    let response: Vec<(u32, f32)> = (0..n)
        .filter_map(|i| {
            let a = tissue.contacts.adj[i]
                .iter()
                .filter(|&&j| ligand[j as usize] > 0.0)
                .count() as f64;
            let competent = identity_bin[i] < 2;
            let rate = 2.0 * if competent { (dose * a).exp() } else { 1.0 };
            let x = poisson(&mut rng, rate);
            (x > 0.0).then_some((i as u32, x))
        })
        .collect();
    World {
        tissue,
        ligand,
        receptor,
        response,
    }
}

fn data<'a>(w: &'a World, targets: &'a [(f64, &'a [(u32, f32)])]) -> PairData<'a> {
    PairData {
        ligand: "L",
        receptor: "R",
        ligand_counts: &w.ligand,
        receptor_counts: &w.receptor,
        targets,
        reweighted: None,
        n_removed: 0,
    }
}

fn opts(competences: &'static [f64], curve: bool) -> EffectOpts {
    EffectOpts {
        threshold: None,
        competences,
        curve,
        ..EffectOpts::default()
    }
}

// * Competence grid

/// Averaging over the grid is Rubin's rules over the single-threshold
/// estimates, so the spread between thresholds widens the interval.
#[test]
fn competence_grid_combines_the_single_estimates() {
    use crate::lr_activity::ligand_effect::contacts::competence;
    let w = world(1, 0.15, false);
    let targets = [(1.0, w.response.as_slice())];
    let pick =
        |o: EffectOpts| estimate_pair(&w.tissue, &data(&w, &targets), &o).unwrap()[0].clone();
    static GRID: [f64; 2] = [0.05, 0.2];
    let both = pick(opts(&GRID, false));
    // The grid is one joint fit; its per-threshold estimates, combined by
    // Rubin's rules, are the reported one.
    let detected: Vec<bool> = w.receptor.iter().map(|&r| r > 0.0).collect();
    let levels: Vec<Vec<bool>> = GRID
        .iter()
        .map(|&c| competence(&w.tissue.identity_bin, &detected, c))
        .collect();
    let x = Exposure::build(&w.tissue, &w.ligand, &opts(&[0.1], false));
    let r = Response::build(&w.tissue, &targets);
    let per = fitted_effects(
        Rows::cells(&w.tissue).pooled(),
        &x,
        &r,
        &levels,
        Sensitivity::ALL,
        &opts(&GRID, false),
    )
    .unwrap();
    let (ja, jb) = (
        per.levels[0].scopes[0].competent,
        per.levels[1].scopes[0].competent,
    );
    let want = combine(&[ja, jb]);
    let got = both.effects.competent;
    assert!((got.theta - want.theta).abs() < 1e-9, "{got:?} vs {want:?}");
    assert!((got.se - want.se).abs() < 1e-9);
    let within = (0.5 * (ja.se.powi(2) + jb.se.powi(2))).sqrt();
    assert!(got.se >= within - 1e-12);
    // Competence differs across the grid: bin 1 is in at 0.05, out at 0.2.
    assert_ne!(ja.n, jb.n);
    // Each threshold's estimate from the joint fit stays close to its own
    // single-threshold fit: the joint models nest the single ones.
    let a = pick(opts(&[0.05], false)).effects.competent;
    let b = pick(opts(&[0.2], false)).effects.competent;
    for (j, s) in [(ja, a), (jb, b)] {
        assert_eq!(j.n, s.n);
        assert!(
            (j.theta - s.theta).abs() < 0.25 * s.se,
            "joint {j:?} single {s:?}"
        );
    }
}

// * The average over exposure thresholds

/// The headline is the average of the curve's estimates, each with its own
/// fits, by their evidence; a fixed threshold is its own estimate.
#[test]
fn the_headline_averages_over_the_exposure_thresholds() {
    let w = world(2, 0.3, false);
    let targets = [(1.0, w.response.as_slice())];
    let o = opts(&[0.1], true);
    let r = &estimate_pair(&w.tissue, &data(&w, &targets), &o).unwrap()[0];
    assert!(r.curve.len() >= 2, "{} thresholds", r.curve.len());
    let points: Vec<Estimate> = r.curve.iter().map(|p| p.effect).collect();
    let want = abf_average(&points, o.abf_prior_scale);
    let got = r.effects.competent;
    assert!((got.theta - want.theta).abs() < 1e-12, "{got:?} {want:?}");
    assert!((got.se - want.se).abs() < 1e-12);
    assert!(r.effects.abf.log10_bf.is_finite());
    assert!(r.curve.iter().any(|p| p.tau == r.effects.abf.top_tau));
    assert!(r.effects.abf.top_weight > 0.0 && r.effects.abf.top_weight <= 1.0);
    // The sensitivity fits read the headline threshold alone.
    assert!(r.effects.steps[0].theta.is_finite());

    let fixed = EffectOpts {
        threshold: Some(r.curve[0].tau),
        ..o
    };
    let r = &estimate_pair(&w.tissue, &data(&w, &targets), &fixed).unwrap()[0];
    let point = r
        .curve
        .iter()
        .find(|p| Some(p.tau) == fixed.threshold)
        .expect("the fixed threshold is on the curve");
    assert_eq!(r.effects.competent.theta, point.effect.theta);
}

// * Threshold–response curve

#[test]
fn curve_thresholds_are_increasing_and_leave_both_arms() {
    let a: Vec<f64> = (0..400)
        .map(|i| if i < 100 { 0.0 } else { (i % 10) as f64 + 1.0 })
        .collect();
    let all = vec![true; a.len()];
    let taus = curve_taus(&a, 0.0, &all, 20);
    assert_eq!(taus[0], (0.0, 0.0), "τ = 0 first");
    assert!(taus.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1));
    assert!(taus.len() >= 3, "{taus:?}");
    for &(tau, q) in &taus {
        let above = a.iter().filter(|&&v| v > tau).count();
        assert!(above >= 20 && a.len() - above >= 20);
        assert!((0.0..=1.0).contains(&q));
    }
    // A headline outside the quartiles joins the grid.
    let taus = curve_taus(&a, 9.0, &all, 20);
    assert!(taus.iter().any(|&(t, _)| t == 9.0));
    // A strict arm minimum drops the high thresholds.
    let few = curve_taus(&a, 0.0, &all, 250);
    assert!(few
        .iter()
        .all(|&(t, _)| a.iter().filter(|&&v| v > t).count() >= 250));
}

#[test]
fn sup_t_critical_value_lies_between_pointwise_and_bonferroni() {
    let mut rng = SmallRng::seed_from_u64(9);
    let rows = 200;
    let indep: Vec<Vec<f64>> = (0..rows)
        .map(|_| {
            (0..4)
                .map(|_| {
                    let e: f64 = StandardNormal.sample(&mut rng);
                    e
                })
                .collect()
        })
        .collect();
    let c4 = sup_t_crit(&indep, 4, 1);
    // Šidák for four independent: 2.49.
    assert!((c4 - 2.49).abs() < 0.12, "independent {c4}");
    let same: Vec<Vec<f64>> = indep.iter().map(|r| vec![r[0]; 4]).collect();
    let c1 = sup_t_crit(&same, 4, 1);
    assert!((c1 - Z_975).abs() < 0.12, "identical columns {c1}");
    assert!(sup_t_crit(&[vec![0.0, 0.0]], 2, 1).is_nan());
}

/// On a planted dose response the curve rises with τ, every band holds
/// its pointwise interval, and the headline threshold is one of its points.
#[test]
fn curve_rises_with_dose_and_band_covers_pointwise() {
    let w = world(3, 0.25, false);
    let targets = [(1.0, w.response.as_slice())];
    let r = &estimate_pair(&w.tissue, &data(&w, &targets), &opts(&[0.1], true)).unwrap()[0];
    let c = &r.curve;
    assert!(c.len() >= 3, "{c:?}");
    assert!(c.windows(2).all(|w| w[0].tau < w[1].tau));
    for p in c {
        assert!(p.effect.theta.is_finite(), "{p:?}");
        assert!(p.band_lo <= p.lo && p.band_hi >= p.hi, "{p:?}");
        assert!(
            p.band_hi > p.hi,
            "a band over several points is wider: {p:?}"
        );
    }
    assert!(c.iter().any(|p| p.tau == r.threshold), "headline threshold");
    for w in c.windows(2) {
        assert!(
            w[1].effect.theta > w[0].effect.theta - w[0].effect.se,
            "not rising: {c:?}"
        );
    }
    assert!(c.last().unwrap().effect.theta > c[0].effect.theta);
}

// * Moments

#[test]
fn moments_on_both_scales() {
    let e = est(0.2, 0.05, f64::INFINITY);
    let m = Moments::of(&e);
    assert!((m.lo - (0.2 - Z_975 * 0.05)).abs() < 1e-9);
    assert!((m.pct - 100.0 * (0.2f64.exp() - 1.0)).abs() < 1e-9);
    assert!((m.pct_sd - 100.0 * 0.2f64.exp() * 0.05).abs() < 1e-9);
    assert!((m.pct_lo - 100.0 * m.lo.exp_m1()).abs() < 1e-9);
    assert!((m.pct_hi - 100.0 * m.hi.exp_m1()).abs() < 1e-9);
    assert!(m.pct_lo < m.pct && m.pct < m.pct_hi);
    // Few clusters: the t reference widens the interval.
    let t = Moments::of(&est(0.2, 0.05, 5.0));
    assert!(t.hi - t.lo > m.hi - m.lo);
    assert!(Moments::of(&Estimate::nan(0, 0)).pct.is_nan());
}

// * By receiver cell type

#[test]
fn named_cell_types_add_one_row_each() {
    let w = world(5, 0.15, true);
    let targets = [(1.0, w.response.as_slice())];
    let rows = estimate_pair(&w.tissue, &data(&w, &targets), &opts(&[0.1], false)).unwrap();
    let labels: Vec<&str> = rows.iter().map(|r| &*r.section).collect();
    assert_eq!(labels, ["all", "type:epi", "type:fib"]);
    let n: usize = rows[1..].iter().map(|r| r.n_receivers).sum();
    assert_eq!(n, rows[0].n_receivers, "types partition the receivers");
    for r in &rows[1..] {
        assert!(r.effects.competent.theta.is_finite(), "{}", r.section);
        assert!(r.effects.steps[0].theta.is_nan(), "steps pooled only");
        assert!(r.curve.is_empty());
    }
    // Without names, no type rows.
    let w = world(5, 0.15, false);
    let rows = estimate_pair(&w.tissue, &data(&w, &targets), &opts(&[0.1], false)).unwrap();
    assert_eq!(rows.len(), 1);
}

// * Skipped fits

/// A pair whose competent receivers leave an arm thin at every threshold
/// skips the fits, and reports the counts the fitted guard would have.
#[test]
fn hopeless_pairs_skip_the_fits_with_the_guards_counts() {
    let mut w = world(7, 0.3, false);
    // Ligand on three cells: a dozen exposed receivers at most.
    w.ligand = (0..w.ligand.len())
        .map(|i| if i % 500 == 7 { 3.0 } else { 0.0 })
        .collect();
    let targets = [(1.0, w.response.as_slice())];
    static GRID: [f64; 2] = [0.05, 0.2];
    let o = opts(&GRID, true);
    let mut rows = estimate_pair(&w.tissue, &data(&w, &targets), &o).unwrap();
    let head = rows[0].effects.competent;
    assert!(head.theta.is_nan());
    assert!(rows[0].effects.incompetent.theta.is_nan(), "no fit ran");
    assert!(rows[0].effects.near_determined.is_nan(), "no main fit");
    assert!(rows[0].curve.iter().all(|c| c.effect.theta.is_nan()));

    // The fitted guard's counts at the middle threshold, which decides.
    use crate::lr_activity::ligand_effect::contacts::competence;
    let detected: Vec<bool> = w.receptor.iter().map(|&r| r > 0.0).collect();
    let levels: Vec<Vec<bool>> = GRID
        .iter()
        .map(|&c| competence(&w.tissue.identity_bin, &detected, c))
        .collect();
    let x = Exposure::build(&w.tissue, &w.ligand, &o);
    let r = Response::build(&w.tissue, &targets);
    let fitted = fitted_effects(
        Rows::cells(&w.tissue),
        &x,
        &r,
        &levels,
        Sensitivity::NONE,
        &EffectOpts {
            fit_thin: true,
            ..o
        },
    )
    .unwrap();
    let want = fitted.levels[levels.len() / 2].scopes[0].competent;
    assert!(want.theta.is_nan());
    assert_eq!((head.n, head.n_treated), (want.n, want.n_treated));
    report(&mut rows, &ReportOpts::default());
    assert_eq!(
        rows[0].verdict,
        crate::lr_activity::ligand_effect::verdict::Verdict::InsufficientData
    );
    assert!(
        rows[0].warnings.starts_with("not estimable: thin arm"),
        "{}",
        rows[0].warnings
    );
    assert!(rows[0].warnings.contains("competent receivers;"));
}

/// Every row (every column of the contact-effect table, the curve rows
/// included) of a pair with its hopeless fits skipped, beside the same pair
/// with every fit run
/// ([`crate::lr_activity::ligand_effect::options::EffectOpts::fit_thin`]).
fn skipped_and_fitted(w: &World, o: &EffectOpts) -> (String, String) {
    let targets = [(1.0, w.response.as_slice())];
    let run = |fit_thin: bool| {
        let o = EffectOpts { fit_thin, ..*o };
        let mut rows = estimate_pair(&w.tissue, &data(w, &targets), &o).unwrap();
        report(&mut rows, &ReportOpts::default());
        format!("{rows:#?}")
    };
    (run(false), run(true))
}

/// Skipping a fit whose every subset leaves an arm thin changes nothing in the
/// output: for a hopeless pair, and for a pair whose controls alone would be
/// estimable (no competent effect, so the incompetent effect is not read
/// either); every fit is skipped for both.
#[test]
fn skipped_fits_change_no_output() {
    let mut hopeless = world(7, 0.3, true);
    hopeless.ligand = (0..hopeless.ligand.len())
        .map(|i| if i % 500 == 7 { 3.0 } else { 0.0 })
        .collect();
    let mut controls_only = world(8, 0.3, false);
    // Only bin 1 detects the receptor, in about 15% of its cells: nobody
    // is competent at 0.2.
    controls_only.receptor = controls_only
        .receptor
        .iter()
        .zip(&controls_only.tissue.identity_bin)
        .map(|(&r, &b)| if b == 1 { r } else { 0.0 })
        .collect();
    static ONLY: [f64; 1] = [0.2];
    static GRID: [f64; 2] = [0.05, 0.2];
    for (w, grid) in [(&hopeless, &GRID[..]), (&controls_only, &ONLY[..])] {
        let (skipped, fitted) = skipped_and_fitted(w, &opts(grid, true));
        assert_eq!(skipped, fitted);
    }
}

/// A thin competent arm skips the fits: the receptor-negative control checks
/// an estimate the pair does not have, and does not enter its family's
/// q-values.
#[test]
fn a_thin_headline_drops_its_controls() {
    let mut w = world(8, 0.3, false);
    w.receptor = w
        .receptor
        .iter()
        .zip(&w.tissue.identity_bin)
        .map(|(&r, &b)| if b == 1 { r } else { 0.0 })
        .collect();
    let targets = [(1.0, w.response.as_slice())];
    static ONLY: [f64; 1] = [0.2];
    let o = opts(&ONLY, false);
    let row = &estimate_pair(&w.tissue, &data(&w, &targets), &o).unwrap()[0];
    assert_eq!(row.n_competent, 0);
    assert!(row.effects.competent.theta.is_nan());
    assert!(row.effects.incompetent.theta.is_nan());
    assert!(
        row.effects.incompetent.n > 0,
        "its counts are still reported"
    );
}

/// Competent only at a threshold looser than the middle one: no estimate, on
/// any threshold, with every fit skipped.
#[test]
fn a_pair_empty_at_the_middle_threshold_has_no_estimate() {
    let mut w = world(8, 0.3, false);
    w.receptor = w
        .receptor
        .iter()
        .zip(&w.tissue.identity_bin)
        .map(|(&r, &b)| if b == 1 { r } else { 0.0 })
        .collect();
    let targets = [(1.0, w.response.as_slice())];
    static GRID: [f64; 2] = [0.05, 0.2];
    let o = opts(&GRID, false);
    let row = &estimate_pair(&w.tissue, &data(&w, &targets), &o).unwrap()[0];
    assert_eq!(row.n_competent, 0);
    assert!(row.effects.competent.theta.is_nan());
    assert!(row.effects.incompetent.theta.is_nan());
    // The looser threshold alone does estimate this pair.
    static LOOSE: [f64; 1] = [0.05];
    let row = &estimate_pair(&w.tissue, &data(&w, &targets), &opts(&LOOSE, false)).unwrap()[0];
    assert!(row.effects.competent.theta.is_finite());
}

/// The step fits need the steps' propensities: asked for on threshold
/// fits built without them, they are an error, not a panic.
#[test]
fn step_fits_without_their_propensities_are_an_error() {
    let w = world(1, 0.3, false);
    let targets = [(1.0, w.response.as_slice())];
    let o = opts(&[0.1], false);
    let x = Exposure::build(&w.tissue, &w.ligand, &o);
    let r = Response::build(&w.tissue, &targets);
    let detected: Vec<bool> = w.receptor.iter().map(|&r| r > 0.0).collect();
    let levels = vec![crate::lr_activity::ligand_effect::contacts::competence(
        &w.tissue.identity_bin,
        &detected,
        0.1,
    )];
    let rows = Rows::cells(&w.tissue);
    let fits = ThresholdFits::new(rows, &x, &levels, &o, &mut Default::default()).unwrap();
    let (design, cf) = (rows.design.view(), rows.cf);
    assert!(fit_main(design, cf, &fits, &r, Fit::Main, None, &o).is_ok());
    assert!(fit_main(design, cf, &fits, &r, Fit::Main, Some(0), &o).is_err());
    assert!(effects_on(rows, &fits, &r, Sensitivity::NONE, &o).is_ok());
    assert!(effects_on(rows, &fits, &r, Sensitivity::ALL, &o).is_err());
}

/// The prior is set in multiples of the estimand's own standard error, so the
/// weights over thresholds do not depend on the response's units: the same
/// estimates on a scale a thousand times larger weigh the same.
#[test]
fn threshold_weights_do_not_depend_on_the_response_units() {
    let parts = [
        est(0.004, 0.002, 30.0),
        est(0.006, 0.0025, 30.0),
        est(0.001, 0.0015, 30.0),
    ];
    let scaled = parts.map(|e| est(1000.0 * e.theta, 1000.0 * e.se, e.df));
    let scale = EffectOpts::default().abf_prior_scale;
    let (a, _) = abf_weights(&parts, scale);
    let (b, _) = abf_weights(&scaled, scale);
    assert_eq!((a.len(), b.len()), (parts.len(), parts.len()));
    for ((ka, wa), (kb, wb)) in a.iter().zip(&b) {
        assert_eq!(ka, kb);
        assert!((wa - wb).abs() < 1e-12, "{wa} vs {wb}");
    }
}

/// With the prior a multiple `c` of the median standard error, each threshold's
/// log Bayes factor is `½ ln(1 − s) + ½ z² s` with `s = c² m / (se² + c² m)`,
/// `m` the median squared standard error: the weights are those factors,
/// normalized.
#[test]
fn threshold_weights_follow_the_relative_prior() {
    let parts = [
        est(0.004, 0.002, 30.0),
        est(0.009, 0.003, 30.0),
        est(-0.001, 0.001, 30.0),
    ];
    let c: f64 = 10.0;
    let m = 0.002f64.powi(2);
    let log_bf: Vec<f64> = parts
        .iter()
        .map(|e| {
            let s = c * c * m / (e.se * e.se + c * c * m);
            0.5 * (1.0 - s).ln() + 0.5 * e.z() * e.z() * s
        })
        .collect();
    let top = log_bf.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = log_bf.iter().map(|l| (l - top).exp()).sum();
    let (w, _) = abf_weights(&parts, c);
    assert_eq!(w.len(), parts.len());
    for ((k, got), l) in w.iter().zip(&log_bf) {
        let want = (l - top).exp() / total;
        assert!((got - want).abs() < 1e-12, "threshold {k}: {got} vs {want}");
    }
}

/// Thresholds whose standard error is zero cannot set the prior's scale: the
/// scale comes from the positive ones, and the weights stay finite.
#[test]
fn a_zero_standard_error_leaves_the_weights_finite() {
    let parts = [
        est(0.0, 0.0, 30.0),
        est(0.0, 0.0, 30.0),
        est(0.3, 0.1, 30.0),
    ];
    let (w, log10_bf) = abf_weights(&parts, 10.0);
    assert!(
        !w.is_empty() && w.iter().all(|&(_, x)| x.is_finite()),
        "{w:?}"
    );
    assert!(log10_bf.is_finite(), "{log10_bf}");
    assert!(abf_average(&parts, 10.0).theta.is_finite());
}

// * Sharing a ligand's propensity

/// Every estimand of `a` and `b`, theta and se, equal up to the solver's
/// tolerance (NaN only where the other is NaN too).
fn assert_same_effects(what: &str, a: &EffectRow, b: &EffectRow) {
    let pick = |r: &EffectRow| -> Vec<(&'static str, Estimate)> {
        use crate::lr_activity::ligand_effect::tissue::STEP_NAMES;
        let e = &r.effects;
        let mut all = vec![
            ("effect", e.effect),
            ("competent", e.competent),
            ("unadjusted_diff", e.unadjusted_diff),
            ("ipw", e.ipw),
            ("outcome", e.outcome),
            ("incompetent", e.incompetent),
            ("nospill", e.nospill),
        ];
        all.extend(STEP_NAMES.into_iter().zip(e.steps));
        all
    };
    for ((name, x), (_, y)) in pick(a).into_iter().zip(pick(b)) {
        for (part, u, v) in [("theta", x.theta, y.theta), ("se", x.se, y.se)] {
            let same = (u.is_nan() && v.is_nan()) || (u - v).abs() <= 1e-6 * (1.0 + u.abs());
            assert!(same, "{what} {name}.{part}: alone {u} vs shared {v}");
        }
    }
}

/// Pairs that share a ligand share its propensity: one fit per threshold on
/// the design and the ligand's own columns, which each pair borders with its
/// receptor's competence columns. Bordering reaches the optimum the pair's
/// own fit reaches, so every estimand agrees with the pair fitted alone. The
/// bases start cold, so a pair's estimates are the same bits whichever of
/// the ligand's pairs ran first; and a later pair whose thresholds the first
/// already fitted adds no base of its own.
#[test]
fn a_ligands_shared_propensity_is_each_pairs_own() {
    use crate::lr_activity::ligand_effect::estimate::estimate_pair_shared;
    use crate::lr_activity::ligand_effect::fits::LigandShare;
    let w = world(31, 0.3, false);
    let targets = [(1.0, w.response.as_slice())];
    // A second receptor of the same ligand, detected in other bins.
    let other: Vec<f64> = w
        .tissue
        .identity_bin
        .iter()
        .map(|&b| (b != 1) as u8 as f64)
        .collect();
    let pairs = [
        data(&w, &targets),
        PairData {
            receptor: "R2",
            receptor_counts: &other,
            ..data(&w, &targets)
        },
    ];
    let o = opts(&[0.05, 0.1, 0.2], true);
    let alone: Vec<Vec<EffectRow>> = pairs
        .iter()
        .map(|p| estimate_pair(&w.tissue, p, &o).unwrap())
        .collect();
    // Each pair's shared rows, the pairs run in `order`.
    let shared_in = |order: [usize; 2]| -> Vec<Vec<EffectRow>> {
        let share = LigandShare::default();
        let mut out: Vec<Vec<EffectRow>> = vec![Vec::new(), Vec::new()];
        for (i, &k) in order.iter().enumerate() {
            let before = share.fits();
            out[k] = estimate_pair_shared(&w.tissue, &pairs[k], &o, &share).unwrap();
            if i == 0 {
                assert!(share.fits() > 0, "the first pair fits the ligand's bases");
            } else {
                assert_eq!(share.fits(), before, "pair {k} reuses every base");
            }
        }
        out
    };
    let (forward, backward) = (shared_in([0, 1]), shared_in([1, 0]));
    for k in 0..2 {
        assert_eq!(alone[k].len(), forward[k].len());
        for (a, b) in alone[k].iter().zip(&forward[k]) {
            assert_same_effects(&format!("pair {k} {}", a.section), a, b);
        }
        for (f, b) in forward[k].iter().zip(&backward[k]) {
            let bits = |r: &EffectRow| {
                (
                    r.effects.competent.theta.to_bits(),
                    r.effects.competent.se.to_bits(),
                )
            };
            assert_eq!(
                bits(f),
                bits(b),
                "pair {k} {} depends on the order",
                f.section
            );
        }
    }
}
