//! Sequential adjustment: on a tissue where cell type drives both who is
//! exposed and how strongly a receiver responds, with no ligand effect at
//! all, the crude, section-only and spatial estimates are fooled and the
//! identity step is not. The nested designs share the full matrix.

use super::{estimate_pair, overlap_terms};
use crate::lr_activity::ligand_effect::crossfit::Design;
use crate::lr_activity::ligand_effect::estimate::PairData;
use crate::lr_activity::ligand_effect::folds::spatial_folds;
use crate::lr_activity::ligand_effect::options::EffectOpts;
use crate::lr_activity::ligand_effect::overlap::*;
use crate::lr_activity::ligand_effect::tissue::{Tissue, TissueInputs, TissueOpts, STEP_NAMES};
use crate::util::common::Mat;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma, Poisson, StandardNormal};

const SIDE: usize = 60;

struct World {
    tissue: Tissue,
    ligand: Vec<f64>,
    receptor: Vec<f64>,
    response: Vec<(u32, f32)>,
}

/// Two sections (left and right halves) of a grid whose cells touch their
/// four neighbours. Cell type comes in random 2×2 clumps, finer than any
/// spatial smooth: type 1 makes the ligand, so its receivers are more often
/// exposed, and responds more strongly on its own. The ligand does nothing.
/// The identity projection is pure noise; type enters only through
/// `cell_type`.
fn world(seed: u64) -> World {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = SIDE * SIDE;
    let half = SIDE / 2;
    let coords: Vec<Option<(f32, f32)>> = (0..n)
        .map(|i| Some(((i % SIDE) as f32 * 10.0, (i / SIDE) as f32 * 10.0)))
        .collect();
    let section: Vec<Box<str>> = (0..n)
        .map(|i| if i % SIDE < half { "a" } else { "b" }.into())
        .collect();
    let mut pairs = Vec::new();
    for i in 0..n {
        let (x, y) = (i % SIDE, i / SIDE);
        if x + 1 < SIDE && x + 1 != half {
            pairs.push((i, i + 1));
        }
        if y + 1 < SIDE {
            pairs.push((i, i + SIDE));
        }
    }
    let clumps = SIDE / 2;
    let clump_type: Vec<u32> = (0..clumps * clumps)
        .map(|_| (rng.random::<f64>() < 0.5) as u32)
        .collect();
    let ty: Vec<u32> = (0..n)
        .map(|i| clump_type[(i / SIDE / 2) * clumps + (i % SIDE) / 2])
        .collect();
    let identity = Mat::from_fn(4, n, |_, _| {
        let e: f64 = StandardNormal.sample(&mut rng);
        e as f32
    });
    // Receptor in identity bin 0 only, independent of type.
    let identity_bin: Vec<usize> = (0..n).map(|i| (i % 3 == 0) as usize).collect();
    let receptor: Vec<f64> = identity_bin
        .iter()
        .map(|&b| (b == 0) as u8 as f64)
        .collect();
    let log1p_depth = vec![(500.0f32).ln_1p(); n];
    let tissue = Tissue::build(
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
    assert_eq!(tissue.n(), n, "every cell has a contact");
    let ligand: Vec<f64> = ty
        .iter()
        .map(|&t| {
            let p = if t == 1 { 0.4 } else { 0.05 };
            if rng.random::<f64>() < p {
                3.0
            } else {
                0.0
            }
        })
        .collect();
    let response: Vec<(u32, f32)> = ty
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| {
            // Overdispersed, so a neighbour mean (the spillover column) is
            // a poor proxy for a receiver's type.
            let mean = if t == 1 { 6.0 } else { 1.5 };
            let rate: f64 = Gamma::new(1.0, mean).unwrap().sample(&mut rng);
            let x = Poisson::new(rate.max(1e-9)).unwrap().sample(&mut rng) as f32;
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

#[test]
fn cell_type_confounding_is_removed_by_the_identity_step() {
    let w = world(7);
    let targets = [(1.0, w.response.as_slice())];
    let rows = estimate_pair(
        &w.tissue,
        &PairData {
            ligand: "L",
            receptor: "R",
            ligand_counts: &w.ligand,
            receptor_counts: &w.receptor,
            targets: &targets,
            reweighted: None,
            n_removed: 0,
        },
        &EffectOpts {
            threshold: Some(0.0),
            competences: &[0.5],
            nuisance: NuisanceOpts::default(),
            ..EffectOpts::default()
        },
    )
    .unwrap();
    assert_eq!(rows.len(), 3, "pooled plus two sections");
    let pooled = &rows[0];
    let steps = [
        pooled.effects.steps[0],
        pooled.effects.steps[1],
        pooled.effects.steps[2],
        pooled.effects.competent,
    ];
    let report = format!(
        "crude {:.3}; {}",
        pooled.effects.crude.theta,
        STEP_NAMES
            .iter()
            .chain(&["full"])
            .zip(&steps)
            .map(|(s, e)| format!("{s} {:.3} ± {:.3}", e.theta, e.se))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(
        pooled.effects.crude.theta > 0.2,
        "crude contrast not fooled: {report}"
    );
    for (name, e) in STEP_NAMES.iter().zip(&steps).take(2) {
        assert!(e.z() > 3.0, "{name} step not fooled: {report}");
    }
    for e in &steps[2..] {
        assert!(e.z().abs() < 3.0, "identity step still biased: {report}");
        assert!(
            e.theta.abs() < 0.25 * pooled.effects.crude.theta,
            "{report}"
        );
    }
    for r in &rows[1..] {
        assert!(r.effects.steps[0].theta.is_nan());
        assert!(r.effects.steps[1].se.is_nan());
        assert!(r.effects.steps[2].theta.is_nan());
        assert!(r.effects.competent.theta.is_finite());
    }
}

/// The steps are nested column prefixes of the one matrix: each view
/// borrows `design.x` (no other `n × p` matrix exists; a step owns only its
/// `p × p` Grams), and the cell-type columns sit inside the identity step.
#[test]
fn steps_are_nested_prefixes_of_one_matrix() {
    let w = world(3);
    let t = &w.tissue;
    let o = TissueOpts::default();
    // Intercept and one section indicator; then the smooths; then receiver
    // and neighbour identity (4 + 4) and one cell-type indicator; depth last.
    let section = 2;
    let spatial = section + o.scales.len() * o.n_freq;
    let identity = spatial + 4 + 4 + 1;
    let widths: Vec<usize> = t.steps.iter().map(|s| s.design.p()).collect();
    assert_eq!(widths, [section, spatial, identity]);
    assert_eq!(t.design.p(), identity + 2);
    for s in &t.steps {
        assert!(std::ptr::eq(s.design.view(&t.design).x, &t.design.x));
    }
}

/// The overlap terms on a prefix, which reuses the full Grams' submatrix,
/// equal those of a design built from a copy of the prefix columns.
#[test]
fn prefix_terms_match_a_copied_design() {
    let n = 900;
    let mut rng = SmallRng::seed_from_u64(5);
    let coords: Vec<(f32, f32)> = (0..n).map(|i| ((i % 30) as f32, (i / 30) as f32)).collect();
    let cf = spatial_folds(&coords, &vec![0u32; n], &[6.0], 4, 5);
    let x = DMatrix::<f32>::from_fn(n, 5, |_, j| {
        if j == 0 {
            1.0
        } else {
            let e: f64 = StandardNormal.sample(&mut rng);
            e as f32
        }
    });
    let full = Design::new(x, &cf);
    let small = Design::new(full.x.columns(0, 3).clone_owned(), &cf);
    let t: Vec<bool> = (0..n)
        .map(|i| rng.random::<f64>() < 1.0 / (1.0 + (-full.x[(i, 1)] as f64).exp()))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| full.x[(i, 1)] as f64 + 0.3 * t[i] as u8 as f64 + full.x[(i, 4)] as f64)
        .collect();
    let extra: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let pair = PairInputs {
        t: &t,
        y: &y,
        extra: &[&extra],
        outcome_only: &[],
        modifiers: &[],
    };
    let opts = NuisanceOpts::default();
    let prefix = full.prefix(3);
    let a = overlap_terms(prefix.view(&full), &cf, &pair, &opts).unwrap();
    let b = overlap_terms(small.view(), &cf, &pair, &opts).unwrap();
    let c = overlap_terms(full.view(), &cf, &pair, &opts).unwrap();
    for i in 0..n {
        assert!((a.h[i] - b.h[i]).abs() < 1e-6, "h at {i}");
        assert!((a.num[i] - b.num[i]).abs() < 1e-5, "num at {i}");
    }
    let differs = (0..n).any(|i| (a.num[i] - c.num[i]).abs() > 1e-3);
    assert!(differs, "dropping a confounder changes the fit");
}
