//! End to end on a synthetic tissue: a grid of cells touching their four
//! neighbours, two cell types in patches, uneven depth. The simulator
//! plants or withholds an effect under a hidden regional confounder; the
//! estimate should track the truth while the crude contrast is fooled.

use super::run_sim;
use crate::lr_activity::ligand_effect::contacts::{competence, Contacts};
use crate::lr_activity::ligand_effect::options::EffectOpts;
use crate::lr_activity::ligand_effect::overlap::NuisanceOpts;
use crate::lr_activity::ligand_effect::simulate::communities::{
    between_sum, draw_communities, on_boundary,
};
use crate::lr_activity::ligand_effect::simulate::{
    default_grid, sign_lifts, simulate_background, CommunityLayout, ConfounderKind, GridOpts,
    Scenario, SimBase, Study,
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
        crude_rej += r.crude.rejects() as usize;
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
        crude_rej += (r.crude.rejects() && r.crude.theta > 0.0) as usize;
        est += r.estimate.theta / runs as f64;
        crude += r.crude.theta / runs as f64;
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
        r.crude.z() > 3.0,
        "crude {:.3} ± {:.3}",
        r.crude.theta,
        r.crude.se
    );
}

/// The simulated background runs beside the programme without touching it: it does
/// not move the estimate, it ignores a planted effect, it follows the hidden
/// field as far as it is told to, and the run carries `lra`'s verdict.
#[test]
fn the_background_sees_the_field_and_not_the_effect() {
    let f = fixture();
    let b = base(&f);
    let run = |scen: &Scenario, seed: u64| run_sim(&f.tissue, &b, scen, &opts(), seed).unwrap();

    let planted = Scenario {
        effect: 0.6,
        ..Scenario::default()
    };
    let r = run(&planted, 11);
    assert!(r.estimate.rejects(), "{:?}", r.estimate);
    assert!(r.background.theta.is_finite());
    assert!(r.background.z().abs() < 3.0, "{:?}", r.background);

    let field = |background_confounding: f64| Scenario {
        confound_ligand: 1.0,
        confound_response: 1.0,
        background_confounding,
        ..Scenario::default()
    };
    let (on, off) = (run(&field(1.0), 3), run(&field(0.0), 3));
    assert_eq!(on.estimate.theta, off.estimate.theta);
}

/// Background counts follow the hidden field as far as they are told to.
#[test]
fn the_background_counts_follow_the_field_they_are_given() {
    let f = fixture();
    let b = base(&f);
    let n = f.tissue.n();
    let field: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let high_share = |beta: f64| {
        let lifts = sign_lifts(
            &field,
            beta,
            &vec![0.0; n],
            &[false; 64],
            &f.tissue.identity_bin,
        );
        let nz = simulate_background(&f.tissue, &b, &lifts, 0.0, 5);
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
        effect: 0.5,
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

fn layout(number: usize, dominant: f64, mixing: f64) -> CommunityLayout {
    CommunityLayout {
        number,
        dominant,
        mixing,
    }
}

/// Number, skew and mixing each do what the layout says, whatever the seed.
#[test]
fn community_layouts_follow_their_settings() {
    let f = fixture();
    let n = f.tissue.n() as f64;
    let share = |labels: &[u32], c: u32| labels.iter().filter(|&&l| l == c).count() as f64 / n;
    let boundary = |labels: &[u32]| {
        let on = on_boundary(&f.tissue.contacts, labels);
        on.iter().filter(|&&b| b).count() as f64 / n
    };
    for seed in [1, 2, 3] {
        let plain = draw_communities(&f.tissue, &layout(4, 0.0, 0.0), seed);
        assert!(plain.iter().all(|&l| l < 4));
        let skewed = draw_communities(&f.tissue, &layout(4, 0.7, 0.0), seed);
        assert!((share(&skewed, 0) - 0.7).abs() < 0.01);
        let mixed = draw_communities(&f.tissue, &layout(4, 0.0, 0.3), seed);
        assert!(boundary(&mixed) > boundary(&plain));
    }
}

#[test]
fn between_sum_counts_only_contacts_in_another_community() {
    let contacts = Contacts::from_pairs(4, &[(0, 1), (0, 2), (2, 3)]);
    let labels = [0, 0, 1, 1];
    let v = [1.0, 10.0, 100.0, 1000.0];
    assert_eq!(between_sum(&contacts, &labels, &v), [100.0, 0.0, 1.0, 0.0]);
    assert_eq!(on_boundary(&contacts, &labels), [true, false, true, false]);
}

/// A planted effect through contacts between communities leaves everyone else
/// untouched, so today's arms over all contacts see less of it than an effect
/// through every contact.
#[test]
fn an_effect_between_communities_reaches_fewer_receivers() {
    let f = fixture();
    let every = Scenario {
        effect: 0.5,
        ..Scenario::default()
    };
    let between = Scenario {
        layout: Some(layout(4, 0.0, 0.0)),
        ..every
    };
    let (a, b) = (
        run_sim(&f.tissue, &base(&f), &every, &opts(), 31).unwrap(),
        run_sim(&f.tissue, &base(&f), &between, &opts(), 31).unwrap(),
    );
    assert!(a.boundary_share.is_nan());
    assert!(b.boundary_share > 0.0 && b.boundary_share < 1.0);
    assert!(
        b.truth_exposed > 0.0,
        "truth among exposed {}",
        b.truth_exposed
    );
    assert!(b.truth < a.truth, "between {} every {}", b.truth, a.truth);
}

#[test]
fn community_fields_need_a_layout() {
    let f = fixture();
    let scen = Scenario {
        confounder: ConfounderKind::Boundary,
        confound_ligand: 1.0,
        confound_response: 1.0,
        ..Scenario::default()
    };
    assert!(run_sim(&f.tissue, &base(&f), &scen, &opts(), 5).is_err());
    let with_layout = Scenario {
        layout: Some(layout(4, 0.0, 0.0)),
        ..scen
    };
    assert!(run_sim(&f.tissue, &base(&f), &with_layout, &opts(), 5).is_ok());
}

/// A layout adds its own scenarios and changes none of the rest.
#[test]
fn the_grid_adds_three_scenarios_per_layout_and_rep() {
    let plain = GridOpts {
        reps: 1,
        effect_sizes: vec![],
        ..GridOpts::default()
    };
    let with = GridOpts {
        communities: vec![layout(4, 0.0, 0.0), layout(12, 0.6, 0.2)],
        ..plain.clone()
    };
    let (a, b) = (default_grid(&plain), default_grid(&with));
    assert_eq!(b.len(), a.len() + 6);
    assert!(a.iter().all(|s| s.layout.is_none()));
    assert_eq!(b.iter().filter(|s| s.layout.is_some()).count(), 6);
}

#[test]
fn a_study_rejects_an_impossible_layout() {
    for bad in [
        r#"{"scenarios":{"communities":[{"number":1}]}}"#,
        r#"{"scenarios":{"communities":[{"number":4,"dominant":1.0}]}}"#,
        r#"{"scenarios":{"communities":[{"number":4,"mixing":1.5}]}}"#,
        r#"{"scenarios":{"communities":[{"number":4,"share":0.5}]}}"#,
    ] {
        assert!(Study::from_json(bad).is_err(), "{bad}");
    }
    assert!(Study::from_json(r#"{"scenarios":{"communities":[{"number":4}]}}"#).is_ok());
}

/// At community boundaries the comparison targets the effect through contacts
/// between communities, and recovers it; under no effect it stays at zero.
#[test]
fn the_boundary_comparison_tracks_its_own_truth() {
    let f = fixture();
    for (effect, seed) in [(0.5, 41), (0.0, 43)] {
        let scen = Scenario {
            effect,
            layout: Some(layout(6, 0.0, 0.1)),
            ..Scenario::default()
        };
        let r = run_sim(&f.tissue, &base(&f), &scen, &opts(), seed).unwrap();
        let b = r.boundary;
        assert!(b.estimate.n_treated > 0 && b.estimate.n > b.estimate.n_treated);
        let z = (b.estimate.theta - b.truth) / b.estimate.se;
        assert!(
            z.abs() < 3.0,
            "effect {effect}: boundary {:.4} ± {:.4}, truth {:.4}",
            b.estimate.theta,
            b.estimate.se,
            b.truth
        );
        if effect > 0.0 {
            assert!(
                b.truth > r.truth,
                "boundary truth {} vs {}",
                b.truth,
                r.truth
            );
        } else {
            assert_eq!(b.truth, 0.0);
        }
    }
    let plain = run_sim(&f.tissue, &base(&f), &Scenario::default(), &opts(), 3).unwrap();
    assert!(plain.boundary.estimate.theta.is_nan() && plain.boundary.truth.is_nan());
}
