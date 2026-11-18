//! The benchmark's own guarantees: the grid stays small, the ligand
//! calibration reads producers out of the real counts, and a real-gene
//! confounder needs its gene.

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
    assert!(has(&|s| s.heterogeneous && s.effect_pve > 0.0));
    assert!(has(
        &|s| s.confounder == ConfounderKind::RealGene && s.scale == 4.0
    ));
    assert!(has(&|s| s.shape == ConfoundShape::Quadratic));
    assert!(!has(
        &|s| s.confounder == ConfounderKind::Fourier && s.ligand_field_pve != 0.0
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

/// A real-gene confounder without a gene is refused.
#[test]
fn a_real_gene_confounder_needs_a_gene() {
    let f = fixture();
    let scen = Scenario {
        ligand_field_pve: 0.5,
        response_field_pve: 0.5,
        confounder: ConfounderKind::RealGene,
        ..Scenario::default()
    };
    let none = SimBase {
        confounder_gene: None,
        reference_position_of_target: None,
        ..base(&f)
    };
    assert!(run_sim(&f.tissue, &none, &scen, &opts(), 7).is_err());
}
