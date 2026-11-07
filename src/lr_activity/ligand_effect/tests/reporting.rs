//! Inference across pairs: BH and Simes on known vectors, the verdict
//! table with its overrides, and the cell-type identifiability diagnostic
//! on the synthetic grid tissue.

use super::rows_of;
use super::simulate::fixture;
use crate::lr_activity::ligand_effect::averaging::AbfSummary;
use crate::lr_activity::ligand_effect::dose::eta_squared;
use crate::lr_activity::ligand_effect::effects::ScopeEffects;
use crate::lr_activity::ligand_effect::inference::{t_crit_975, t_two_sided_p, Estimate};
use crate::lr_activity::ligand_effect::report::{bh, report, simes, ReportOpts};
use crate::lr_activity::ligand_effect::rows::EffectRow;
use crate::lr_activity::ligand_effect::verdict::{verdict, Evidence, Verdict};

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-12)
}

// * p-values, BH and Simes

#[test]
fn normal_tails() {
    let inf = f64::INFINITY;
    assert!((t_two_sided_p(1.959_964, inf) - 0.05).abs() < 1e-6);
    assert!((t_two_sided_p(-1.959_964, inf) - 0.05).abs() < 1e-6);
    assert!(
        (crate::lr_activity::ligand_effect::inference::upper_normal_p(1.644_854) - 0.05).abs()
            < 1e-6
    );
    assert!(
        (crate::lr_activity::ligand_effect::inference::upper_normal_p(-1.644_854) - 0.95).abs()
            < 1e-6
    );
    assert!(t_two_sided_p(f64::NAN, inf).is_nan());
}

/// Tabulated t: the upper 2.5% points at 1, 10 and 70 degrees of freedom,
/// and the p they give back.
#[test]
fn t_reference_matches_tables() {
    for (df, q) in [(1.0, 12.706_2), (10.0, 2.228_14), (70.0, 1.994_44)] {
        assert!(
            (t_crit_975(df) - q).abs() < 1e-3,
            "df {df}: {}",
            t_crit_975(df)
        );
        assert!((t_two_sided_p(q, df) - 0.05).abs() < 1e-4);
    }
    assert!((t_crit_975(f64::INFINITY) - 1.959_964).abs() < 1e-6);
    assert!(t_crit_975(0.0).is_nan() && t_crit_975(f64::NAN).is_nan());
    assert!(t_crit_975(0.5) > t_crit_975(1.0));
    // Heavier tails than the normal at the same statistic.
    assert!(t_two_sided_p(2.0, 20.0) > t_two_sided_p(2.0, f64::INFINITY));
}

#[test]
fn bh_known_vectors() {
    assert!(close(&bh(&[0.01, 0.02, 0.03, 0.04, 0.05]), &[0.05; 5]));
    // Sorted 0.005, 0.01, 0.03, 0.04 → raw 0.02, 0.02, 0.04, 0.04.
    assert!(close(
        &bh(&[0.01, 0.04, 0.03, 0.005]),
        &[0.02, 0.04, 0.04, 0.02]
    ));
    // Raw 0.2, 0.15, 0.27: the running minimum from the largest p down
    // lowers the smallest p's q to 0.15.
    assert!(close(&bh(&[0.2, 0.1, 0.09]), &[0.2, 0.15, 0.15]));
    // NaN is left out of m.
    assert!(close(&bh(&[0.5, f64::NAN, 0.01]), &[0.5, f64::NAN, 0.02]));
    assert!(close(&bh(&[0.9, 0.95]), &[0.95, 0.95]));
}

#[test]
fn simes_known_vectors() {
    // 0.01·3/1 = 0.03, 0.04·3/2 = 0.06, 0.5.
    assert!((simes(&[0.04, 0.01, 0.5]) - 0.03).abs() < 1e-12);
    assert!((simes(&[0.2, f64::NAN]) - 0.2).abs() < 1e-12);
    assert!(simes(&[f64::NAN]).is_nan());
    assert!((simes(&[0.9, 0.95]) - 0.95).abs() < 1e-12);
}

// * Verdicts

#[test]
fn verdict_table() {
    let ev = |estimable, effect, coexpr, control_failed, confounded| Evidence {
        estimable,
        effect,
        coexpr,
        control_failed,
        confounded,
    };
    let cases = [
        (ev(true, true, true, false, false), Verdict::Supported),
        (ev(true, false, true, false, false), Verdict::ColocationOnly),
        (
            ev(true, true, false, false, false),
            Verdict::EffectWithoutCoexpression,
        ),
        (ev(true, false, false, false, false), Verdict::NoEvidence),
        // A failed control vetoes an effect, not co-expression.
        (ev(true, true, true, true, false), Verdict::ControlFailed),
        (ev(true, true, false, true, false), Verdict::ControlFailed),
        (ev(true, false, true, true, false), Verdict::ColocationOnly),
        (ev(true, false, false, true, false), Verdict::NoEvidence),
        // Confounding vetoes everything estimable, controls included.
        (
            ev(true, true, true, false, true),
            Verdict::CelltypeInseparable,
        ),
        (
            ev(true, true, false, true, true),
            Verdict::CelltypeInseparable,
        ),
        (
            ev(true, false, true, false, true),
            Verdict::CelltypeInseparable,
        ),
        (ev(false, true, true, true, true), Verdict::InsufficientData),
        (
            ev(false, false, false, false, false),
            Verdict::InsufficientData,
        ),
    ];
    for (e, want) in cases {
        assert_eq!(verdict(e), want, "{e:?}");
    }
    for v in Verdict::ALL {
        assert!(cases.iter().any(|(_, w)| *w == v), "{v:?} unreachable");
    }
}

fn est(theta: f64, se: f64) -> Estimate {
    Estimate {
        theta,
        se,
        n: 400,
        n_treated: 150,
        mean_h: 0.2,
        ess: 300.0,
        // The normal reference, so p-values read as the z's below.
        df: f64::INFINITY,
    }
}

/// A row with a headline z, a co-expression z, control z's and η².
fn row(section: &str, ligand: &str, z: f64, zc: f64, z3: f64, eta2: f64) -> EffectRow {
    let nan = Estimate::nan(0, 0);
    EffectRow {
        section: section.into(),
        ligand: ligand.into(),
        receptor: "R".into(),
        n_targets: 10,
        n_removed: 0,
        n_receivers: 1000,
        n_competent: 400,
        n_exposed: 300,
        threshold: 0.0,
        coexpr_log_or: zc * 0.1,
        coexpr_se: 0.1,
        marginal_cor: nan,
        effects: ScopeEffects {
            effect: est((z - z3) * 0.05, 0.05),
            competent: est(z * 0.05, 0.05),
            crude: nan,
            ipw: nan,
            outcome: nan,
            incompetent: est(z3 * 0.05, 0.05),
            steps: [nan; 3],
            nospill: nan,
            near_determined: 0.0,
            abf: AbfSummary::NONE,
        },
        exposure_eta2: eta2,
        n_competent_bins: 1,
        n_incompetent_bins: 1,
        background_effect: nan,
        n_background_genes: 0,
        effect_over_background: nan,
        curve: Vec::new(),
        background_q: f64::NAN,
        effect_p: f64::NAN,
        effect_q: f64::NAN,
        coexpr_p: f64::NAN,
        coexpr_q: f64::NAN,
        effect_incompetent_q: f64::NAN,
        effect_contrast_q: f64::NAN,
        effect_over_background_q: f64::NAN,
        n_pairs_same_ligand: 0,
        ligand_q: f64::NAN,
        verdict: Verdict::InsufficientData,
        warnings: String::new(),
    }
}

#[test]
fn report_labels_and_overrides() {
    let mut rows = vec![
        row("all", "L1", 10.0, 10.0, 0.0, 0.1),
        row("all", "L1", 10.0, 10.0, 10.0, 0.1),
        row("all", "L2", 10.0, 10.0, 0.0, 0.9),
        row("all", "L3", f64::NAN, 10.0, 0.0, 0.1),
        row("all", "L4", 0.1, 0.0, 0.0, 0.1),
        row("all", "L5", 10.0, -3.0, 0.0, 0.1),
        row("all", "L6", 0.0, 10.0, 0.0, 0.1),
        // Alone in its section, so its q is its p.
        row("s1", "L1", 1.0, 0.0, 0.0, 0.1),
    ];
    rows[3].effects.competent = Estimate::nan(400, 3);
    report(&mut rows, &ReportOpts::default());
    let got: Vec<&str> = rows.iter().map(|r| r.verdict.code()).collect();
    assert_eq!(
        got,
        [
            "supported",
            "control_failed",
            "celltype_inseparable",
            "insufficient_data",
            "no_evidence",
            "effect_without_coexpression",
            "colocation_only",
            "no_evidence",
        ]
    );
    assert!(rows[0].warnings.is_empty());
    assert!(rows[1].warnings.contains("receptor-negative"));
    assert!(rows[2].warnings.contains("neighbour cell type"));
    assert!(rows[2].warnings.contains("cell-type artefact"));
    assert!(
        rows[3].warnings.contains("thin arm"),
        "{}",
        rows[3].warnings
    );

    // Ligand families within a section; the s1 row is its own family.
    assert_eq!(rows[0].n_pairs_same_ligand, 2);
    assert_eq!(rows[1].n_pairs_same_ligand, 2);
    assert_eq!(rows[2].n_pairs_same_ligand, 1);
    assert_eq!(rows[7].n_pairs_same_ligand, 1);
    assert!((rows[7].effect_q - rows[7].effect_p).abs() < 1e-12);
    assert!((rows[7].effect_p - t_two_sided_p(1.0, f64::INFINITY)).abs() < 1e-9);
    assert!((rows[7].ligand_q - rows[7].effect_p).abs() < 1e-12);
    // The NaN headline is out of the family: 7 finite p in "all".
    assert!(rows[3].effect_p.is_nan() && rows[3].effect_q.is_nan());
    // z = 0 is the largest p (1) of the family, and keeps q = p.
    assert!((rows[6].effect_q - 1.0).abs() < 1e-12);
    // One-sided: a negative co-expression score is not significant.
    assert!(rows[5].coexpr_p > 0.99);
}

#[test]
fn fdr_level_moves_verdicts() {
    // z = 2.2 gives p ≈ 0.028: significant alone at 0.05, not at 0.01.
    let mut rows = vec![row("all", "L1", 2.2, 0.0, 0.0, 0.1)];
    report(&mut rows, &ReportOpts::default());
    assert_eq!(rows[0].verdict, Verdict::EffectWithoutCoexpression);
    report(
        &mut rows,
        &ReportOpts {
            fdr: 0.01,
            ..ReportOpts::default()
        },
    );
    assert_eq!(rows[0].verdict, Verdict::NoEvidence);
    // Many nulls beside it: BH no longer lets it through at 0.05.
    let mut rows: Vec<EffectRow> = (0..20)
        .map(|k| row("all", &format!("N{k}"), 0.05 * k as f64, 0.0, 0.0, 0.1))
        .collect();
    rows.push(row("all", "L1", 2.2, 0.0, 0.0, 0.1));
    report(&mut rows, &ReportOpts::default());
    assert_eq!(rows[20].verdict, Verdict::NoEvidence);
}

// * Identifiability

#[test]
fn exposure_eta2_tracks_neighbour_identity() {
    let fx = fixture();
    let tissue = &fx.tissue;
    let n = tissue.n();
    let nb = tissue.neighbour_bin.clone();
    assert!(nb.iter().all(|&b| b < 2), "every grid cell has contacts");
    let all = rows_of(&vec![true; n]);

    // Exposure a function of the dominant neighbour's bin.
    let t: Vec<f64> = nb.iter().map(|&b| (b == 1) as u8 as f64).collect();
    let e = eta_squared(&nb, &t, &all);
    assert!((e - 1.0).abs() < 1e-12, "{e}");

    // Exposure independent of it: a hash of the row.
    let t: Vec<f64> = (0..n)
        .map(|i| ((i.wrapping_mul(2_654_435_761) >> 7) & 1) as f64)
        .collect();
    let e = eta_squared(&nb, &t, &all);
    assert!(e < 0.01, "{e}");

    // The real fixture's exposure: type-1 neighbours carry the ligand.
    let a = tissue.contacts.sum(&fx.ligand);
    let t: Vec<f64> = a.iter().map(|&v| (v > 1.0) as u8 as f64).collect();
    let e = eta_squared(&nb, &t, &all);
    assert!(e > 0.5, "ligand marks a cell type: {e}");

    // A constant exposure has no variance to explain.
    assert!(eta_squared(&nb, &vec![1.0; n], &all).is_nan());
}

/// The controls are corrected among the pairs with a significant effect:
/// the others show none and warn of none, and what they show cannot move a
/// significant pair's control.
#[test]
fn controls_are_corrected_among_significant_effects() {
    let e3_q = |rows: &[EffectRow], i: usize| rows[i].effect_incompetent_q;
    // No significant effect: no control is read, however strong.
    let mut rows = vec![row("all", "L1", 0.0, 0.0, 10.0, 0.1)];
    report(&mut rows, &ReportOpts::default());
    assert!(e3_q(&rows, 0).is_nan());
    assert_eq!(rows[0].verdict.code(), "no_evidence");
    assert!(
        !rows[0].warnings.contains("respond"),
        "{}",
        rows[0].warnings
    );

    // Alone among the significant, its control's q is its p: unrelated pairs,
    // with controls strong or weak, change neither it nor its verdict.
    let alone = |others: &[f64]| {
        let mut rows = vec![row("all", "L1", 10.0, 10.0, 8.5, 0.1)];
        for (k, &z3) in others.iter().enumerate() {
            rows.push(row("all", &format!("L{}", k + 2), 0.0, 0.0, z3, 0.1));
        }
        report(&mut rows, &ReportOpts::default());
        (e3_q(&rows, 0), rows[0].verdict.code())
    };
    let (q, verdict) = alone(&[]);
    assert_eq!(verdict, "control_failed");
    assert_eq!((q, verdict), alone(&[10.0; 20]));
    assert_eq!((q, verdict), alone(&[0.0; 20]));

    // A significant effect whose control has no estimate has not held.
    let mut rows = vec![row("all", "L1", 10.0, 10.0, 0.0, 0.1)];
    rows[0].effects.incompetent = Estimate::nan(10, 1);
    rows[0].effects.effect = Estimate::nan(10, 1);
    report(&mut rows, &ReportOpts::default());
    assert!(e3_q(&rows, 0).is_nan());
    assert_eq!(rows[0].verdict.code(), "control_failed");

    // Each section is its own family.
    let mut rows = vec![
        row("all", "L1", 10.0, 10.0, 2.5, 0.1),
        row("s1", "L1", 0.0, 10.0, 2.5, 0.1),
    ];
    report(&mut rows, &ReportOpts::default());
    assert!(e3_q(&rows, 0) < 0.05 && e3_q(&rows, 1).is_nan());
}

/// Beside a significant effect the conservative verdict: the effect must
/// exceed each negative control, the same way, and a control that could not
/// run has not held, but for a background programme where the pool matched none.
#[test]
fn a_verdict_needs_a_specific_effect_and_every_control_checked() {
    let judged = |change: &dyn Fn(&mut EffectRow)| {
        let mut rows = vec![row("all", "L1", 10.0, 10.0, 0.0, 0.1)];
        change(&mut rows[0]);
        report(&mut rows, &ReportOpts::default());
        (rows[0].verdict.code(), rows[0].warnings.clone())
    };
    assert_eq!(judged(&|_| {}).0, "supported");

    // Receptor-negative receivers are quiet, but the contrast does not show
    // the effect exceeding theirs.
    let (v, w) = judged(&|r| r.effects.effect = est(0.05, 0.05));
    assert_eq!(v, "control_failed");
    assert!(w.contains("does not exceed"), "{w}");
    // Significant, the wrong way: the receptor-negative effect is the larger.
    let (v, _) = judged(&|r| r.effects.effect = est(-0.5, 0.05));
    assert_eq!(v, "control_failed");

    // What decides is the comparison: a control can be finite while its
    // comparison could not run, and that has not held either.
    let (v, w) = judged(&|r| r.effects.effect = Estimate::nan(10, 1));
    assert_eq!(v, "control_failed");
    assert!(w.contains("receptor-negative receivers could not"), "{w}");
    let (v, w) = judged(&|r| r.n_background_genes = 5);
    assert_eq!(v, "control_failed");
    assert!(w.contains("background programme could not"), "{w}");

    let (v, w) = judged(&|r| {
        r.n_background_genes = 5;
        r.effect_over_background = est(0.05, 0.05);
    });
    assert_eq!(v, "control_failed");
    assert!(w.contains("exceed the one on a matched background"), "{w}");
    // Larger than the background, but the wrong way round.
    let (v, _) = judged(&|r| {
        r.n_background_genes = 5;
        r.effect_over_background = est(-0.5, 0.05);
    });
    assert_eq!(v, "control_failed");
}

/// A hidden field moves the controls too, so a significant control does not
/// rule out an effect that clearly exceeds it: the verdict stands and the
/// warning says what responded.
#[test]
fn a_significant_control_the_effect_exceeds_is_a_warning_not_a_veto() {
    let mut rows = vec![row("all", "L1", 20.0, 10.0, 5.0, 0.1)];
    rows[0].n_background_genes = 5;
    rows[0].background_effect = est(0.25, 0.05);
    rows[0].effect_over_background = est(0.75, 0.05);
    report(&mut rows, &ReportOpts::default());
    assert_eq!(rows[0].verdict, Verdict::Supported);
    let w = &rows[0].warnings;
    assert!(w.contains("receptor-negative receivers respond"), "{w}");
    assert!(w.contains("background programme responds too"), "{w}");
}
