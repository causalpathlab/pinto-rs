//! The overlap estimator on data whose effect is known: it removes
//! measured confounding the crude contrast keeps, separates subgroups, and
//! stays finite when exposure is nearly determined.

use super::rows_of;
use crate::lr_activity::ligand_effect::folds::spatial_folds;
use crate::lr_activity::ligand_effect::overlap::*;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

struct World {
    design: Design,
    cf: CrossFit,
    z: Vec<f64>,
    c: Vec<f64>,
}

/// `n` rows on a 2-D grid, one confounder `z` (smooth in space plus noise)
/// and a binary modifier `c`.
fn world(n: usize, seed: u64) -> World {
    let mut rng = SmallRng::seed_from_u64(seed);
    let side = (n as f64).sqrt().ceil() as usize;
    let coords: Vec<(f32, f32)> = (0..n)
        .map(|i| ((i % side) as f32, (i / side) as f32))
        .collect();
    let z: Vec<f64> = coords
        .iter()
        .map(|&(x, y)| {
            let e: f64 = StandardNormal.sample(&mut rng);
            (x as f64 / 9.0).sin() + (y as f64 / 13.0).cos() + 0.5 * e
        })
        .collect();
    let c: Vec<f64> = (0..n)
        .map(|_| (rng.random::<f64>() < 0.5) as u8 as f64)
        .collect();
    let section = vec![0u32; n];
    let cf = spatial_folds(&coords, &section, &[6.0], 5, seed);
    let x = DMatrix::<f32>::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { z[i] as f32 });
    World {
        design: Design::new(x, &cf),
        cf,
        z,
        c,
    }
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// Exposure confounded by `z` with strength `a`; response `θ T (+ θ_c T c)
/// + b z + noise`.
fn draw(w: &World, a: f64, b: f64, theta: f64, theta_c: f64, seed: u64) -> (Vec<bool>, Vec<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = w.z.len();
    let t: Vec<bool> = (0..n)
        .map(|i| rng.random::<f64>() < sigmoid(a * w.z[i]))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let e: f64 = StandardNormal.sample(&mut rng);
            let tt = t[i] as u8 as f64;
            theta * tt + theta_c * tt * w.c[i] + b * w.z[i] + 0.5 * e
        })
        .collect();
    (t, y)
}

fn all_rows(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

#[test]
fn confounding_fools_the_crude_contrast_not_the_estimate() {
    let w = world(4000, 1);
    let (t, y) = draw(&w, 1.5, 1.0, 0.0, 0.0, 2);
    let terms = overlap_terms(
        w.design.view(),
        &w.cf,
        &PairInputs {
            t: &t,
            y: &y,
            extra: &[],
            outcome_only: &[],
            modifiers: &[],
        },
        &NuisanceOpts::default(),
    )
    .unwrap();
    let (e, _) = subset_estimate(&terms, &t, &all_rows(t.len()), &w.cf);
    let crude = arm_contrast(&y, &t, &all_rows(t.len()), &w.cf, |_| 1.0).theta;
    assert!(
        crude > 0.5,
        "crude contrast should be confounded upward: {crude}"
    );
    assert!(
        e.z().abs() < 3.0,
        "no effect, but estimate {:.3} ± {:.3}",
        e.theta,
        e.se
    );
}

#[test]
fn a_planted_effect_is_recovered() {
    let w = world(4000, 3);
    let (t, y) = draw(&w, 1.0, 1.0, 0.4, 0.0, 4);
    let terms = overlap_terms(
        w.design.view(),
        &w.cf,
        &PairInputs {
            t: &t,
            y: &y,
            extra: &[],
            outcome_only: &[],
            modifiers: &[],
        },
        &NuisanceOpts::default(),
    )
    .unwrap();
    let (e, _) = subset_estimate(&terms, &t, &all_rows(t.len()), &w.cf);
    assert!(
        ((e.theta - 0.4) / e.se).abs() < 3.0,
        "{:.3} ± {:.3}",
        e.theta,
        e.se
    );
    assert!(e.se < 0.1);
}

#[test]
fn subgroups_and_their_difference() {
    let w = world(5000, 5);
    // Effect only where c = 1.
    let (t, y) = draw(&w, 1.0, 1.0, 0.0, 0.5, 6);
    let terms = overlap_terms(
        w.design.view(),
        &w.cf,
        &PairInputs {
            t: &t,
            y: &y,
            extra: &[&w.c],
            outcome_only: &[],
            modifiers: &[&w.c],
        },
        &NuisanceOpts::default(),
    )
    .unwrap();
    let on: Vec<bool> = w.c.iter().map(|&v| v > 0.5).collect();
    let off: Vec<bool> = on.iter().map(|b| !b).collect();
    let (e1, p1) = subset_estimate(&terms, &t, &rows_of(&on), &w.cf);
    let (e0, p0) = subset_estimate(&terms, &t, &rows_of(&off), &w.cf);
    let d = difference(&e1, &p1, &e0, &p0);
    assert!(
        ((e1.theta - 0.5) / e1.se).abs() < 3.0,
        "on: {:.3} ± {:.3}",
        e1.theta,
        e1.se
    );
    assert!(
        (e0.theta / e0.se).abs() < 3.0,
        "off: {:.3} ± {:.3}",
        e0.theta,
        e0.se
    );
    assert!(
        ((d.theta - 0.5) / d.se).abs() < 3.0,
        "difference: {:.3} ± {:.3}",
        d.theta,
        d.se
    );
}

#[test]
fn nearly_determined_exposure_stays_finite() {
    let w = world(3000, 7);
    // Exposure almost a step function of z: inverse-propensity weights
    // would explode; overlap weights shrink toward the boundary instead.
    let (t, y) = draw(&w, 8.0, 1.0, 0.3, 0.0, 8);
    let terms = overlap_terms(
        w.design.view(),
        &w.cf,
        &PairInputs {
            t: &t,
            y: &y,
            extra: &[],
            outcome_only: &[],
            modifiers: &[],
        },
        &NuisanceOpts::default(),
    )
    .unwrap();
    assert!(terms.h.iter().all(|&h| (0.0..=0.25).contains(&h)));
    let (e, _) = subset_estimate(&terms, &t, &all_rows(t.len()), &w.cf);
    assert!(e.theta.is_finite() && e.se.is_finite());
    assert!(e.mean_h < 0.1, "overlap should be poor here: {}", e.mean_h);
    assert!(
        ((e.theta - 0.3) / e.se).abs() < 3.5,
        "{:.3} ± {:.3}",
        e.theta,
        e.se
    );
}

#[test]
fn a_group_with_one_arm_is_not_estimable() {
    let w = world(500, 9);
    let t = vec![true; 500];
    let y = vec![1.0; 500];
    let terms = OverlapTerms {
        h: vec![0.2; 500],
        num: vec![0.0; 500],
        outcome: vec![0.0; 500],
        arm_weight: vec![0.5; 500],
    };
    let (e, _) = subset_estimate(&terms, &t, &all_rows(500), &w.cf);
    assert!(e.theta.is_nan());
    assert!(arm_contrast(&y, &t, &all_rows(500), &w.cf, |_| 1.0)
        .theta
        .is_nan());
}

/// Each half of the estimator alone: under measured confounding and a known
/// effect, overlap weights and the outcome regression both shed what the crude
/// contrast keeps.
#[test]
fn ipw_and_outcome_regression_each_adjust() {
    let w = world(4000, 21);
    let n = w.z.len();
    let mut rng = SmallRng::seed_from_u64(22);
    let t: Vec<bool> =
        w.z.iter()
            .map(|z| rng.random::<f64>() < sigmoid(1.5 * z))
            .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let e: f64 = StandardNormal.sample(&mut rng);
            w.z[i] + 0.3 * t[i] as u8 as f64 + 0.3 * e
        })
        .collect();
    let terms = overlap_terms(
        w.design.view(),
        &w.cf,
        &PairInputs {
            t: &t,
            y: &y,
            extra: &[],
            outcome_only: &[],
            modifiers: &[],
        },
        &NuisanceOpts::default(),
    )
    .unwrap();
    let rows = all_rows(n);
    let (effect, _) = subset_estimate(&terms, &t, &rows, &w.cf);
    let crude = arm_contrast(&y, &t, &rows, &w.cf, |_| 1.0);
    let ipw = arm_contrast(&y, &t, &rows, &w.cf, |i| terms.arm_weight[i]);
    let outcome = outcome_estimate(&terms, &effect, &rows);
    assert!(crude.theta > 0.6, "crude {:.3}", crude.theta);
    assert!((ipw.theta - 0.3).abs() < 0.1, "ipw {:.3}", ipw.theta);
    assert!(
        (outcome.theta - 0.3).abs() < 0.1,
        "outcome {:.3}",
        outcome.theta
    );
    assert_eq!(outcome.se, effect.se);
    assert!(crude.se > 0.0 && ipw.se > 0.0 && ipw.df > 0.0);
}

/// The clustered correlation agrees with the formula, reads a perfect line
/// as 1 with no spread, and a constant as nothing.
#[test]
fn the_marginal_correlation_is_pearson_with_a_clustered_se() {
    let w = world(1000, 23);
    let n = w.z.len();
    let rows = all_rows(n);
    let mut rng = SmallRng::seed_from_u64(24);
    let x: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
    let y: Vec<f64> = x.iter().map(|v| v + rng.random::<f64>()).collect();
    let r = marginal_correlation(&y, &x, &rows, &w.cf);
    // Two uniforms summed: cor = 1/sqrt(2).
    assert!((r.theta - 0.707).abs() < 0.06, "{:.3}", r.theta);
    assert!(r.se > 0.0 && r.se < 0.1, "{:.3}", r.se);
    let line = marginal_correlation(&x, &x, &rows, &w.cf);
    assert!((line.theta - 1.0).abs() < 1e-9 && line.se < 1e-9);
    assert!(marginal_correlation(&vec![1.0; n], &x, &rows, &w.cf)
        .theta
        .is_nan());
}

/// The Hessian kernel against the plain sum.
#[test]
fn weighted_gram_is_the_plain_sum() {
    use crate::lr_activity::ligand_effect::regression::weighted_gram;
    let mut rng = SmallRng::seed_from_u64(3);
    let (m, d) = (37, 9);
    let rows: Vec<f64> = (0..m * d)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    let w: Vec<f64> = (0..m).map(|_| rng.random::<f64>()).collect();
    let h = weighted_gram(&rows, d, &w);
    for a in 0..d {
        for b in 0..d {
            let plain: f64 = (0..m)
                .map(|r| w[r] * rows[r * d + a] * rows[r * d + b])
                .sum();
            assert!((h[(a, b)] - plain).abs() < 1e-12, "({a}, {b})");
        }
    }
}

/// A fit from the marginal log-odds and one from a distant start reach the
/// one optimum of the penalized likelihood.
#[test]
fn logistic_ridge_reaches_one_optimum_from_any_start() {
    use crate::lr_activity::ligand_effect::regression::logistic_ridge;
    let w = world(900, 5);
    let (t, _) = draw(&w, 1.5, 0.0, 0.0, 0.0, 6);
    let tf: Vec<f64> = t.iter().map(|&t| t as u8 as f64).collect();
    let rows: Vec<usize> = (0..tf.len()).collect();
    let x = &w.design.x;
    let cold = logistic_ridge(x, 2, &[], &tf, &rows, 1.0, None).unwrap();
    let far = nalgebra::DVector::from_vec(vec![3.0, -4.0]);
    let warm = logistic_ridge(x, 2, &[], &tf, &rows, 1.0, Some(&far)).unwrap();
    assert!((cold - warm).amax() < 1e-6);
}
