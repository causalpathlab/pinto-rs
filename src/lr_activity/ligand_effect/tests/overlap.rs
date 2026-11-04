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
    let crude = crude_difference(&y, &t, &all_rows(t.len()));
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
    };
    let (e, _) = subset_estimate(&terms, &t, &all_rows(500), &w.cf);
    assert!(e.theta.is_nan());
    assert!(crude_difference(&y, &t, &all_rows(500)).is_nan());
}
