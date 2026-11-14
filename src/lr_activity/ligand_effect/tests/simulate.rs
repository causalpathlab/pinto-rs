//! End to end on a synthetic tissue: a grid of cells touching their four
//! neighbours, two cell types in patches, uneven depth. The simulator
//! plants or withholds an effect under a hidden regional confounder; the
//! estimate should track the truth while the unadjusted difference is fooled.

use super::{lattice, run_sim};
use crate::lr_activity::ligand_effect::contacts::{competence, Contacts};
use crate::lr_activity::ligand_effect::options::EffectOpts;
use crate::lr_activity::ligand_effect::overlap::NuisanceOpts;
use crate::lr_activity::ligand_effect::simulate::{ConfounderKind, Scenario, SimBase};
use crate::lr_activity::ligand_effect::tissue::{Tissue, TissueInputs, TissueOpts};
use crate::util::common::Mat;

const SIDE: usize = 60;

pub(super) struct Fixture {
    pub(super) tissue: Tissue,
    pub(super) ligand: Vec<f64>,
    /// Signed programme weights, and each gene's mean per identity bin
    /// (bins × genes).
    pub(super) weights: Vec<f64>,
    pub(super) bin_means: Vec<f64>,
    pub(super) competent: Vec<bool>,
    /// Receptor counts: two in every cell that detects it.
    pub(super) receptor: Vec<f64>,
    /// A spatially patterned gene for the real-gene confounder.
    pub(super) gene: Vec<f64>,
}

pub(super) fn fixture() -> Fixture {
    let n = SIDE * SIDE;
    let (coords, pairs) = lattice(SIDE);
    // Two types in 10×10 patches; type 1 makes the ligand, type 0 expresses
    // the receptor more.
    let ty: Vec<usize> = (0..n)
        .map(|i| ((i % SIDE) / 10 + (i / SIDE) / 10) % 2)
        .collect();
    let identity = Mat::from_fn(4, n, |k, i| {
        let base = if ty[i] == 1 { 1.0 } else { -1.0 };
        base * (k as f32 + 1.0) + ((i * 31 + k * 7) % 11) as f32 * 0.05
    });
    let section: Vec<Box<str>> = vec!["s0".into(); n];
    let log1p_depth: Vec<f32> = (0..n)
        .map(|i| (200.0 + (i % 17) as f32 * 30.0).ln_1p())
        .collect();
    let tissue = Tissue::build(
        &TissueInputs {
            coords: &coords,
            section: &section,
            pairs: &pairs,
            identity: &identity,
            identity_bin: &ty,
            log1p_depth: &log1p_depth,
            cell_type: None,
            region: None,
        },
        &TissueOpts {
            folds: 4,
            tile: 6.0,
            ..TissueOpts::default()
        },
    )
    .unwrap();
    // Type 1 makes the ligand: a third of its cells detect six counts,
    // which the calibration reads as a share of strong producers. Type 0
    // detects as many in one cell of forty: rare producers, which reach
    // receivers inside type-0 patches too.
    let ligand: Vec<f64> = (0..n)
        .map(|i| match (ty[i], i % 3 == 0, i % 40 == 7) {
            (1, true, _) => 6.0,
            (0, _, true) => 6.0,
            _ => 0.0,
        })
        .collect();
    // One programme gene, higher in type 0.
    let weights = vec![1.0];
    let bin_means = vec![3.0, 2.0];
    let gene: Vec<f64> = (0..n)
        .map(|i| {
            let (x, y) = ((i % SIDE) as f64, (i / SIDE) as f64);
            (2.0 + 2.0 * (x / 9.0).sin() * (y / 13.0).cos()).round()
        })
        .collect();
    let detected: Vec<bool> = (0..n).map(|i| ty[i] == 0 || i % 5 == 0).collect();
    let competent = competence(&tissue.identity_bin, &detected, 0.5);
    let receptor: Vec<f64> = detected.iter().map(|&d| 2.0 * d as u8 as f64).collect();
    Fixture {
        tissue,
        ligand,
        weights,
        bin_means,
        competent,
        receptor,
        gene,
    }
}

pub(super) fn opts() -> EffectOpts {
    EffectOpts {
        threshold: Some(0.0),
        competences: &[0.5],
        nuisance: NuisanceOpts::default(),
        ..EffectOpts::default()
    }
}

pub(super) fn base(f: &Fixture) -> SimBase<'_> {
    SimBase {
        ligand: "L",
        receptor: "R",
        ligand_counts: &f.ligand,
        target_weights: &f.weights,
        target_bin_means: &f.bin_means,
        competent: &f.competent,
        receptor_counts: &f.receptor,
        confounder_gene: Some(&f.gene),
    }
}

#[test]
fn contacts_and_competence() {
    let c = Contacts::from_pairs(4, &[(0, 1), (1, 0), (1, 2), (2, 2)]);
    assert_eq!(c.adj[1], [0, 2], "symmetric, deduplicated, no self loops");
    assert_eq!(c.sum(&[1.0, 10.0, 100.0, 0.0])[1], 101.0);
    assert_eq!(c.mean(&[1.0, 10.0, 100.0, 0.0])[3], 0.0);

    let f = fixture();
    let t = &f.tissue;
    let unit = t.unit[0];
    assert!((unit - 10.0).abs() < 1e-6, "median contact length {unit}");
    assert_eq!(
        competence(&[0, 0, 1, 1], &[true, false, false, false], 0.5),
        [true, true, false, false]
    );
}

/// Random Gaussian bumps at the regional scale, outside the adjustment
/// smooth's own Fourier basis: the smooth still has to absorb them.
#[test]
fn no_effect_under_regional_confounding() {
    let f = fixture();
    let scen = Scenario {
        effect_pve: 0.0,
        ligand_field_pve: 0.5,
        response_field_pve: 0.5,
        confounder: ConfounderKind::Bumps,
        scale: 32.0,
        ..Scenario::default()
    };
    let (mut est_rej, mut unadjusted_rej, mut z_sum) = (0, 0, 0.0);
    let runs = 4;
    for rep in 0..runs {
        let r = run_sim(&f.tissue, &base(&f), &Scenario { rep, ..scen }, &opts(), 11).unwrap();
        assert!(
            r.truth.abs() < 1e-12,
            "no planted effect means a zero truth"
        );
        est_rej += r.rejected() as usize;
        unadjusted_rej += r.unadjusted_diff.rejects() as usize;
        z_sum += r.estimate.z();
    }
    assert!(
        unadjusted_rej >= 3,
        "the unadjusted difference should be fooled most runs ({unadjusted_rej}/{runs})"
    );
    assert!(
        est_rej <= 1,
        "the estimate rejected {est_rej}/{runs} null runs"
    );
    assert!(
        (z_sum / runs as f64).abs() < 2.0,
        "mean z {:.2}",
        z_sum / runs as f64
    );
}

#[test]
fn a_planted_effect_is_recovered_on_the_tissue() {
    let f = fixture();
    let scen = Scenario {
        effect_pve: 0.1,
        ligand_field_pve: 0.5,
        response_field_pve: 0.5,
        ..Scenario::default()
    };
    let r = run_sim(&f.tissue, &base(&f), &scen, &opts(), 21).unwrap();
    assert!(r.truth > 0.05, "truth {}", r.truth);
    assert!(
        ((r.estimate.theta - r.truth) / r.estimate.se).abs() < 3.0,
        "estimate {:.3} ± {:.3}, truth {:.3}",
        r.estimate.theta,
        r.estimate.se,
        r.truth
    );
    assert!(r.incompetent.theta.is_nan() || (r.incompetent.z()).abs() < 3.0);
}

#[test]
fn the_truth_contrasts_the_realized_effect_between_seen_arms() {
    use crate::lr_activity::ligand_effect::simulate::seen_contrast;
    // Two exposed receivers with effect 2 and two unexposed with none.
    let effect = [2.0, 2.0, 0.0, 0.0];
    let comp = [true; 4];
    let even = [1.0; 4];
    // Seen correctly, the contrast is the effect among the exposed.
    assert_eq!(
        seen_contrast(&effect, &[true, true, false, false], &comp, &even),
        2.0
    );
    // One exposed receiver seen as unexposed: the unexposed arm carries a third
    // of its weight in effect, and the contrast falls.
    let missed = seen_contrast(&effect, &[true, false, false, false], &comp, &even);
    assert!((missed - (2.0 - 2.0 / 3.0)).abs() < 1e-12, "{missed}");
    // Arm weights and competence both count.
    let heavy = seen_contrast(
        &effect,
        &[true, false, false, false],
        &comp,
        &[1.0, 3.0, 1.0, 0.0],
    );
    assert!((heavy - (2.0 - 6.0 / 4.0)).abs() < 1e-12, "{heavy}");
    let skip = seen_contrast(
        &effect,
        &[true, false, false, false],
        &[true, false, true, true],
        &even,
    );
    assert_eq!(skip, 2.0);
    // An empty arm has no contrast.
    assert!(seen_contrast(&effect, &[true; 4], &comp, &even).is_nan());
}

#[test]
fn a_dose_response_truth_tracks_the_estimate_and_keeps_the_exposed_effect_apart() {
    let f = fixture();
    let scen = Scenario {
        effect_pve: 0.1,
        dose_response: true,
        dose_midpoint: 0.5,
        dose_order: 2,
        ..Scenario::default()
    };
    let r = run_sim(&f.tissue, &base(&f), &scen, &opts(), 22).unwrap();
    assert!(
        r.truth > 0.0 && r.truth_exposed > 0.0,
        "{} {}",
        r.truth,
        r.truth_exposed
    );
    assert!(
        ((r.estimate.theta - r.truth) / r.estimate.se).abs() < 3.0,
        "estimate {:.3} ± {:.3}, truth {:.3}",
        r.estimate.theta,
        r.estimate.se,
        r.truth
    );
}

/// A field PVE is its share of the linear predictor's variance: on a log
/// rate the coefficient's square, on a logit beside the logistic's own
/// `π²/3`; the sign of a log-rate PVE is the direction of the push.
#[test]
fn field_coefficients_give_their_share_of_variance() {
    use crate::lr_activity::ligand_effect::simulate::{log_rate_coef, logit_coef};
    let logistic = std::f64::consts::PI.powi(2) / 3.0;
    for pve in [0.05, 0.25, 0.5, 0.9] {
        let a = logit_coef(pve);
        assert!(
            (a * a / (a * a + logistic) - pve).abs() < 1e-12,
            "logit {pve}"
        );
        let b = log_rate_coef(pve);
        assert!((b * b - pve).abs() < 1e-12, "log rate {pve}");
        assert_eq!(log_rate_coef(-pve), -b);
    }
    assert_eq!(logit_coef(0.0), 0.0);
}

/// The effect coefficient gives the exposure term `effect_pve` of the log
/// rate's variance among responding receivers, beyond their identity bin:
/// bin offsets and non-responders do not count.
#[test]
fn the_effect_coefficient_explains_its_pve_within_bins() {
    use crate::lr_activity::ligand_effect::simulate::effect_coef;
    // Two bins with different exposure rates, and non-responders exposed too.
    let n = 400;
    let bin: Vec<usize> = (0..n).map(|i| i % 2).collect();
    let responds: Vec<bool> = (0..n).map(|i| i % 5 != 0).collect();
    let x: Vec<f64> = (0..n)
        .map(|i| if (i % 7) < 2 + 3 * bin[i] { 1.0 } else { 0.0 })
        .collect();
    let pve = 0.04;
    let theta = effect_coef(pve, &x, &responds, &bin);
    let rows: Vec<usize> = (0..n).filter(|&i| responds[i]).collect();
    let mut ss = 0.0;
    for b in 0..2 {
        let r: Vec<usize> = rows.iter().copied().filter(|&i| bin[i] == b).collect();
        let m = r.iter().map(|&i| x[i]).sum::<f64>() / r.len() as f64;
        ss += r.iter().map(|&i| (theta * (x[i] - m)).powi(2)).sum::<f64>();
    }
    assert!((ss / rows.len() as f64 - pve).abs() < 1e-12);
    assert_eq!(effect_coef(0.0, &x, &responds, &bin), 0.0);
}
