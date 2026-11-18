//! The co-expression baselines: blind to a planted response effect (they
//! score co-expression, not response), and pinned.

use super::run_sim;
use super::simulate::{base, fixture, opts, Fixture};
use crate::lr_activity::ligand_effect::baselines::{baselines, Baselines, METHODS};
use crate::lr_activity::ligand_effect::simulate::{ConfounderKind, Scenario, SimBase};

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

/// Every statistic of one run, named, in [`Baselines`]' field order.
fn values(b: &Baselines) -> [(&'static str, f64); 10] {
    [
        (stringify!(celltype_score), b.celltype_score),
        (stringify!(celltype_p), b.celltype_p),
        (stringify!(celltype_tested), b.celltype_tested as f64),
        (stringify!(moran_i), b.moran_i),
        (stringify!(moran_p), b.moran_p),
        (stringify!(contact_score), b.contact_score),
        (stringify!(contact_p), b.contact_p),
        (stringify!(coexpr_log_or), b.coexpr_log_or),
        (stringify!(coexpr_se), b.coexpr_se),
        (stringify!(coexpr_p), b.coexpr_p),
    ]
}

/// Each statistic of `b` equals its `want` to a relative 1e-12.
#[track_caller]
fn assert_pinned(b: &Baselines, want: [f64; 10]) {
    for ((field, got), want) in values(b).into_iter().zip(want) {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs(),
            "{field}: {got} != {want}"
        );
    }
}

/// What `lra-simulate` reports for two pairs on the fixture, pinned: the
/// cell-type test on identity bins and the contact statistics, value for
/// value, so a change to the code `lra` shares cannot move the benchmark.
/// The first pair's verdicts settle early, the second's run every
/// permutation. Values match to a part in 1e12, the rounding platforms
/// differ by; a change to the code moves them far more.
#[test]
fn the_benchmark_baselines_are_pinned() {
    let f = fixture();
    let b = baselines(&f.tissue, &f.ligand, &f.receptor, 7);
    assert_pinned(
        &b,
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
        ],
    );
    let (l, r) = even(&f);
    let b = baselines(&f.tissue, &l, &r, 8);
    assert_pinned(
        &b,
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
        ],
    );
}
