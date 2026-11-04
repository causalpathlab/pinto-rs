//! The benchmark's own guarantees: its truth does not come from the
//! estimator, the grid stays small, the ligand calibration reads producers
//! out of the real counts, and the marker type fools the crude contrast.

use super::run_sim;
use super::simulate::{base, fixture, opts};
use crate::lr_activity::ligand_effect::simulate::{
    calibrate_producers, default_grid, marker_bins, ConfoundShape, ConfounderKind, GridOpts,
    Scenario, SimBase,
};

#[test]
fn grid_and_calibration() {
    let grid = default_grid(&GridOpts {
        reps: 1,
        ..GridOpts::default()
    });
    // The original grid ran 16 scenarios per replicate.
    assert!(grid.len() <= 40, "{} scenarios per replicate", grid.len());
    let has = |p: &dyn Fn(&Scenario) -> bool| grid.iter().any(p);
    assert!(has(&|s| s.marker));
    assert!(has(&|s| s.heterogeneous && s.effect > 0.0));
    assert!(has(
        &|s| s.confounder == ConfounderKind::RealGene && s.scale == 4.0
    ));
    assert!(has(&|s| s.shape == ConfoundShape::Quadratic));
    assert!(!has(
        &|s| s.confounder == ConfounderKind::Fourier && s.confound_ligand != 0.0
    ));

    // Six counts in a third of type 1: a minority of strong producers, not
    // every cell producing weakly. Type 0 has one in forty.
    let f = fixture();
    let cal = calibrate_producers(&f.tissue, &f.ligand);
    assert!(
        (0.01..0.05).contains(&cal[0].p) && cal[0].r > 3.0,
        "type 0: share {:.3}, rate {:.3}",
        cal[0].p,
        cal[0].r
    );
    assert!(
        (0.2..0.5).contains(&cal[1].p) && cal[1].r > 3.0,
        "type 1: share {:.3}, rate {:.3}",
        cal[1].p,
        cal[1].r
    );
    assert_eq!(marker_bins(&cal), [false, true]);
}

/// The truth is fixed by what receivers respond to: changing the
/// estimator (a far heavier ridge on its nuisance models, on a sparse
/// ligand that dropout misclassifies) leaves it bit for bit, while the
/// estimator-weighted truth moves. Spillover, which changes the
/// measurement, moves both (`covariates::spillover_truth_is_on_the_measured_scale`).
#[test]
fn truth_is_free_of_the_estimator() {
    let f = fixture();
    let n = f.tissue.n();
    let ty = &f.tissue.identity_bin;
    // Single counts in a quarter of type 1: weak producers, often missed.
    let sparse: Vec<f64> = (0..n)
        .map(|i| if ty[i] == 1 && i % 4 == 0 { 1.0 } else { 0.0 })
        .collect();
    let b = SimBase {
        ligand_counts: &sparse,
        ..base(&f)
    };
    let scen = Scenario {
        effect: 0.5,
        ..Scenario::default()
    };
    let plain = run_sim(&f.tissue, &b, &scen, &opts(), 5).unwrap();
    let mut heavy = opts();
    heavy.nuisance.ridge = 1e4;
    let other = run_sim(&f.tissue, &b, &scen, &heavy, 5).unwrap();
    assert!(plain.truth > 0.1, "truth {}", plain.truth);
    assert_eq!(
        plain.truth, other.truth,
        "the estimator does not move the truth"
    );
    assert_ne!(plain.truth_weighted, other.truth_weighted);
    assert!(
        plain.misclassified > 0.05,
        "dropout should misclassify exposure ({:.3})",
        plain.misclassified
    );
}

/// Ligand, response and a raised programme confined to one type: the crude
/// contrast reads cell type as an effect; the estimator, adjusting for
/// identity, does not (or has no overlap to try).
#[test]
fn a_marker_type_fools_the_crude_contrast() {
    let f = fixture();
    for rep in 0..2 {
        let scen = Scenario {
            marker: true,
            rep,
            ..Scenario::default()
        };
        let r = run_sim(&f.tissue, &base(&f), &scen, &opts(), 6).unwrap();
        assert_eq!(r.truth, 0.0);
        assert!(
            r.crude / r.crude_se > 3.0,
            "crude {:.3} ± {:.3}",
            r.crude,
            r.crude_se
        );
        assert!(
            !r.estimable() || r.estimate.z().abs() < 3.0,
            "estimate {:.3} ± {:.3}",
            r.estimate.theta,
            r.estimate.se
        );
    }
}

/// A real gene's smoothed expression as the hidden field, quadratic.
#[test]
fn a_real_gene_confounder_is_adjusted() {
    let f = fixture();
    let scen = Scenario {
        confound_ligand: 1.0,
        confound_response: 1.0,
        confounder: ConfounderKind::RealGene,
        shape: ConfoundShape::Quadratic,
        scale: 32.0,
        ..Scenario::default()
    };
    let r = run_sim(&f.tissue, &base(&f), &scen, &opts(), 7).unwrap();
    assert_eq!(r.truth, 0.0);
    assert!(
        r.estimate.z().abs() < 3.0,
        "estimate z {:.2}",
        r.estimate.z()
    );
    let none = SimBase {
        confounder_gene: None,
        ..base(&f)
    };
    assert!(run_sim(&f.tissue, &none, &scen, &opts(), 7).is_err());
}
