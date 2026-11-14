//! The co-expression baselines: calibrated when ligand and receptor are
//! unrelated, fooled by cell-type expression, and blind to a planted
//! response effect (they score co-expression, not response).

use super::run_sim;
use super::simulate::{base, fixture, opts, Fixture};
use crate::lr_activity::ligand_effect::baselines::{baselines, rejects, Baselines, METHODS};
use crate::lr_activity::ligand_effect::simulate::{ConfounderKind, Scenario, SimBase};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson};

fn rejections(b: &Baselines) -> [bool; 4] {
    METHODS.map(|(_, p)| rejects(p(b)))
}

/// Ligand and receptor drawn independently of each other, of cell type and
/// of place: every baseline rejects about 5% of runs.
#[test]
fn baselines_are_calibrated_without_association() {
    let f = fixture();
    let n = f.tissue.n();
    let runs = 40;
    let mut hits = [0usize; 4];
    for rep in 0..runs {
        let mut rng = SmallRng::seed_from_u64(1000 + rep as u64);
        let pois = |rng: &mut SmallRng, m: f64| Poisson::new(m).unwrap().sample(rng);
        let l: Vec<f64> = (0..n)
            .map(|_| {
                if rng.random::<f64>() < 0.3 {
                    pois(&mut rng, 3.0)
                } else {
                    0.0
                }
            })
            .collect();
        let r: Vec<f64> = (0..n).map(|_| pois(&mut rng, 1.0)).collect();
        let b = baselines(&f.tissue, &l, &r, rep as u64);
        assert!(b.celltype_tested > 0 && b.moran_p.is_finite() && b.coexpr_p.is_finite());
        for (h, x) in hits.iter_mut().zip(rejections(&b)) {
            *h += x as usize;
        }
    }
    for ((name, _), h) in METHODS.iter().zip(hits) {
        assert!(h <= 6, "{name} rejected {h}/{runs} unrelated pairs");
    }
}

/// Ligand and receptor at the same rate in both cell types, so only the
/// hidden field can tie them together.
fn even(f: &Fixture) -> (Vec<f64>, Vec<f64>) {
    let n = f.tissue.n();
    let l = (0..n).map(|i| if i % 3 == 0 { 6.0 } else { 0.0 }).collect();
    let r = (0..n).map(|i| if i % 2 == 0 { 6.0 } else { 0.0 }).collect();
    (l, r)
}

fn confounded(effect: f64, rep: usize) -> Scenario {
    Scenario {
        effect_pve: effect,
        ligand_field_pve: 0.5,
        response_field_pve: 0.5,
        confounder: ConfounderKind::Bumps,
        scale: 32.0,
        rep,
        ..Scenario::default()
    }
}

/// Rejections per baseline, and of the estimator, over `runs` null runs.
fn null_rejections(
    f: &Fixture,
    b: &SimBase<'_>,
    scen: Scenario,
    runs: usize,
) -> ([usize; 4], usize) {
    let (mut hits, mut est) = ([0usize; 4], 0usize);
    for rep in 0..runs {
        let row = run_sim(&f.tissue, b, &Scenario { rep, ..scen }, &opts(), 61).unwrap();
        assert_eq!(row.truth, 0.0);
        for (h, x) in hits.iter_mut().zip(rejections(&row.baselines)) {
            *h += x as usize;
        }
        est += row.rejected() as usize;
    }
    (hits, est)
}

/// Ligand made by one cell type and receptor carried by the other, nothing
/// planted and no field: the cell-type baseline reads the types'
/// expression as an interaction; the estimator, adjusting for identity,
/// does not. (The types sit in patches, so their contacts are mostly
/// homotypic and the contact baselines see less co-expression than chance.)
#[test]
fn cell_type_expression_fools_the_cell_type_baseline_not_the_estimator() {
    let f = fixture();
    let scen = Scenario::default();
    let runs = 4;
    let (hits, est) = null_rejections(&f, &base(&f), scen, runs);
    assert!(est <= 1, "the estimate rejected {est}/{runs} null runs");
    assert!(
        hits[0] >= 3,
        "celltype should be fooled by cell type ({}/{runs})",
        hits[0]
    );
}

/// The same runs with an effect planted: ligand and receptor are drawn
/// before the response, from their own streams, so every baseline is
/// unchanged while the truth is not.
#[test]
fn a_planted_effect_leaves_the_baselines_unchanged() {
    let f = fixture();
    let (l, r) = even(&f);
    let b = SimBase {
        ligand_counts: &l,
        receptor_counts: &r,
        ..base(&f)
    };
    let none = run_sim(&f.tissue, &b, &confounded(0.0, 0), &opts(), 71).unwrap();
    let some = run_sim(&f.tissue, &b, &confounded(0.2, 0), &opts(), 71).unwrap();
    assert!(some.truth > 0.05, "truth {}", some.truth);
    for (name, p) in METHODS {
        assert_eq!(p(&none.baselines), p(&some.baselines), "{name}");
    }
}

/// Every statistic of one run, in [`Baselines`]' field order.
fn values(b: &Baselines) -> [f64; 10] {
    [
        b.celltype_score,
        b.celltype_p,
        b.celltype_tested as f64,
        b.moran_i,
        b.moran_p,
        b.contact_score,
        b.contact_p,
        b.coexpr_log_or,
        b.coexpr_se,
        b.coexpr_p,
    ]
}

/// What `lra-simulate` reports for two pairs on the fixture, pinned: the
/// cell-type test on identity bins and the contact statistics, value for
/// value, so a change to the code `lra` shares cannot move the benchmark.
/// The first pair's verdicts settle early, the second's run every
/// permutation.
#[test]
fn the_benchmark_baselines_are_pinned() {
    let f = fixture();
    let b = baselines(&f.tissue, &f.ligand, &f.receptor, 7);
    assert_eq!(
        values(&b),
        [
            0.7742558873543621,
            0.0009995002498750624,
            2.0,
            -0.25864063578360863,
            1.0,
            1782.1335134361896,
            1.0,
            -1.5296461582183838,
            0.06737641245126724,
            1.0,
        ]
    );
    let (l, r) = even(&f);
    let b = baselines(&f.tissue, &l, &r, 8);
    assert_eq!(
        values(&b),
        [
            0.6727535228830798,
            1.0,
            4.0,
            0.012633672125407561,
            0.0004997501249375312,
            9611.818943933049,
            0.00849575212393803,
            0.03821168839931488,
            0.05047428607940674,
            0.22450918162937955,
        ]
    );
}
