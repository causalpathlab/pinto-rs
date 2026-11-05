//! End to end on a synthetic tissue: a grid of cells touching their four
//! neighbours, two cell types in patches, uneven depth. The simulator
//! plants or withholds an effect under a hidden regional confounder; the
//! estimate should track the truth while the crude contrast is fooled.

use super::run_sim;
use crate::lr_activity::ligand_effect::estimate::EffectOpts;
use crate::lr_activity::ligand_effect::exposure::{competence, ring_partners, Contacts};
use crate::lr_activity::ligand_effect::overlap::NuisanceOpts;
use crate::lr_activity::ligand_effect::simulate::{
    simulate_decoy, ConfounderKind, Scenario, SimBase,
};
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
    let coords: Vec<Option<(f32, f32)>> = (0..n)
        .map(|i| Some(((i % SIDE) as f32 * 10.0, (i / SIDE) as f32 * 10.0)))
        .collect();
    let mut pairs = Vec::new();
    for i in 0..n {
        let (x, y) = (i % SIDE, i / SIDE);
        if x + 1 < SIDE {
            pairs.push((i, i + 1));
        }
        if y + 1 < SIDE {
            pairs.push((i, i + SIDE));
        }
    }
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
fn contacts_ring_and_competence() {
    let c = Contacts::from_pairs(4, &[(0, 1), (1, 0), (1, 2), (2, 2)]);
    assert_eq!(c.adj[1], [0, 2], "symmetric, deduplicated, no self loops");
    assert_eq!(c.sum(&[1.0, 10.0, 100.0, 0.0])[1], 101.0);
    assert_eq!(c.mean(&[1.0, 10.0, 100.0, 0.0])[3], 0.0);

    let f = fixture();
    let t = &f.tissue;
    let unit = t.unit[0];
    assert!((unit - 10.0).abs() < 1e-6, "median contact length {unit}");
    let ring = ring_partners(&t.coords, &t.section, &t.unit, &t.contacts, 2.0, 4.0, 1);
    for (i, r) in ring.iter().enumerate().take(500) {
        assert!(r.len() <= t.contacts.degree(i));
        for &j in r {
            let (a, b) = (t.coords[i], t.coords[j as usize]);
            let d = (((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)) as f64).sqrt();
            assert!((20.0..=40.0).contains(&d), "ring partner at {d}");
        }
    }
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
        effect: 0.0,
        confound_ligand: 1.5,
        confound_response: 1.5,
        confounder: ConfounderKind::Bumps,
        scale: 32.0,
        ..Scenario::default()
    };
    let (mut est_rej, mut crude_rej, mut z_sum) = (0, 0, 0.0);
    let runs = 4;
    for rep in 0..runs {
        let r = run_sim(&f.tissue, &base(&f), &Scenario { rep, ..scen }, &opts(), 11).unwrap();
        assert!(
            r.truth.abs() < 1e-12,
            "no planted effect means a zero truth"
        );
        est_rej += r.rejected() as usize;
        crude_rej += r.crude_rejected() as usize;
        z_sum += r.estimate.z();
    }
    assert!(
        crude_rej >= 3,
        "the crude contrast should be fooled most runs ({crude_rej}/{runs})"
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
        effect: 0.5,
        confound_ligand: 1.0,
        confound_response: 1.0,
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
    assert_eq!(r.ring_truth, 0.0, "a contact-only effect has no ring truth");
    assert!(r.ring.z().abs() < 3.0, "ring z {:.2}", r.ring.z());
}

/// Segmentation spillover from senders that carry the programme
/// themselves: receivers touching them measure it, which the crude
/// contrast reads as an effect. Nothing is planted. Neighbour identity
/// already accounts for how much programme bleeds in, so the fit without
/// the neighbour-response covariate stays at zero; the full fit removes
/// most of the spurious contrast, though that covariate, which carries the
/// receiver's own counts bled back, leaves it slightly below zero here.
#[test]
fn spillover_is_absorbed() {
    let f = fixture();
    let sender_heavy = vec![3.0, 12.0];
    let b = SimBase {
        target_bin_means: &sender_heavy,
        ..base(&f)
    };
    let (mut crude_rej, mut est, mut crude, mut z_bare) = (0, 0.0, 0.0, 0.0);
    let runs = 3;
    for rep in 0..runs {
        let scen = Scenario {
            spillover: 0.3,
            rep,
            ..Scenario::default()
        };
        let r = run_sim(&f.tissue, &b, &scen, &opts(), 31).unwrap();
        assert!(r.truth.abs() < 1e-12, "spillover is not an effect");
        assert!(
            r.no_spill.theta.is_finite(),
            "the no-spill fit runs under spillover"
        );
        crude_rej += (r.crude_rejected() && r.crude > 0.0) as usize;
        est += r.estimate.theta / runs as f64;
        crude += r.crude / runs as f64;
        z_bare += r.no_spill.z() / runs as f64;
    }
    assert!(
        crude_rej >= 2,
        "spillover should fool the crude contrast ({crude_rej}/{runs})"
    );
    assert!(
        est.abs() < 0.5 * crude,
        "mean estimate {est:.3} against crude {crude:.3}"
    );
    assert!(z_bare.abs() < 2.5, "no-spill mean z {z_bare:.2}");
}

/// The estimator sees the ligand field displaced by four contact lengths:
/// under the regional confounder the crude contrast still finds an
/// association, but the planted contact effect is out of reach.
#[test]
fn a_shifted_field_has_no_contact_effect() {
    let f = fixture();
    let scen = Scenario {
        effect: 0.5,
        confound_ligand: 1.5,
        confound_response: 1.5,
        shift: 4.0,
        ..Scenario::default()
    };
    let r = run_sim(&f.tissue, &base(&f), &scen, &opts(), 41).unwrap();
    assert_eq!(r.truth, 0.0, "the seen exposure has no effect");
    assert!(
        r.estimate.z().abs() < 3.0,
        "estimate z {:.2}",
        r.estimate.z()
    );
    assert!(
        r.crude / r.crude_se > 3.0,
        "crude {:.3} ± {:.3}",
        r.crude,
        r.crude_se
    );
}

/// An effect from ring partners alone: the ring estimate finds it, the
/// contact estimate does not.
#[test]
fn an_effect_at_ring_range_is_seen_by_the_ring_estimate() {
    let f = fixture();
    let scen = Scenario {
        ring_effect: 0.5,
        ..Scenario::default()
    };
    let r = run_sim(&f.tissue, &base(&f), &scen, &opts(), 51).unwrap();
    assert!(
        r.truth.abs() < 1e-12 && r.ring_truth > 0.05,
        "truths {} {}",
        r.truth,
        r.ring_truth
    );
    assert!(
        r.estimate.z().abs() < 3.0,
        "contact z {:.2}",
        r.estimate.z()
    );
    assert!(
        ((r.ring.theta - r.ring_truth) / r.ring.se).abs() < 3.0,
        "ring {:.3} ± {:.3}, truth {:.3}",
        r.ring.theta,
        r.ring.se,
        r.ring_truth
    );
}

/// The simulated decoy runs beside the programme without touching it: it does
/// not move the estimate, it ignores a planted effect, it follows the hidden
/// field as far as it is told to, and the run carries `lra`'s verdict.
#[test]
fn the_decoy_sees_the_field_and_not_the_effect() {
    let f = fixture();
    let b = base(&f);
    let run = |scen: &Scenario, seed: u64| run_sim(&f.tissue, &b, scen, &opts(), seed).unwrap();

    let planted = Scenario {
        effect: 0.6,
        ..Scenario::default()
    };
    let r = run(&planted, 11);
    assert!(r.estimate.rejects(), "{:?}", r.estimate);
    assert!(r.decoy.theta.is_finite());
    assert!(r.decoy.z().abs() < 3.0, "{:?}", r.decoy);
    assert_eq!(r.survived(), r.verdict.survived());

    let field = |decoy_confounding: f64| Scenario {
        confound_ligand: 1.0,
        confound_response: 1.0,
        decoy_confounding,
        ..Scenario::default()
    };
    let (on, off) = (run(&field(1.0), 3), run(&field(0.0), 3));
    assert_eq!(on.estimate.theta, off.estimate.theta);
}

/// Decoy counts follow the hidden field as far as they are told to.
#[test]
fn the_decoy_counts_follow_the_field_they_are_given() {
    let f = fixture();
    let b = base(&f);
    let n = f.tissue.n();
    let field: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let high_share = |beta: f64| {
        let nz = simulate_decoy(&f.tissue, &b, &field, beta, &[false; 64], 0.0, 5);
        let (mut high, mut all) = (0.0, 0.0);
        for (genes, &w) in nz.iter().zip(b.target_weights) {
            let up = w > 0.0;
            for &(i, v) in genes {
                let on_high = (field[i as usize] > 0.0) == up;
                high += if on_high { v as f64 } else { 0.0 };
                all += v as f64;
            }
        }
        high / all
    };
    assert!((high_share(0.0) - 0.5).abs() < 0.05, "{}", high_share(0.0));
    assert!(high_share(1.0) > 0.75, "{}", high_share(1.0));
}
