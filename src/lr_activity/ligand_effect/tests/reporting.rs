//! Inference across pairs: the t and normal references, the local fdr over
//! a family of pairs, the verdict table with its overrides, and the
//! cell-type identifiability diagnostic on the synthetic grid tissue.

use super::rows_of;
use super::simulate::fixture;
use crate::lr_activity::ligand_effect::averaging::AbfSummary;
use crate::lr_activity::ligand_effect::dose::eta_squared;
use crate::lr_activity::ligand_effect::effects::ScopeEffects;
use crate::lr_activity::ligand_effect::inference::{
    normal_score_of_two_sided_p, t_crit_975, t_two_sided_p, upper_normal_p, Estimate,
};
use crate::lr_activity::ligand_effect::report::{report, ReportOpts};
use crate::lr_activity::ligand_effect::rows::{EffectRow, LocalFdr};
use crate::lr_activity::ligand_effect::verdict::{verdict, Evidence, Verdict};

// * p-values and normal scores

#[test]
fn normal_tails() {
    let inf = f64::INFINITY;
    assert!((t_two_sided_p(1.959_964, inf) - 0.05).abs() < 1e-6);
    assert!((t_two_sided_p(-1.959_964, inf) - 0.05).abs() < 1e-6);
    assert!((upper_normal_p(1.644_854) - 0.05).abs() < 1e-6);
    assert!((upper_normal_p(-1.644_854) - 0.95).abs() < 1e-6);
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

/// The normal score gives back its p, including where `1 − p` rounds to 1.
#[test]
fn normal_score_inverts_the_two_sided_p() {
    assert!((normal_score_of_two_sided_p(0.05) - 1.959_964).abs() < 1e-5);
    assert_eq!(normal_score_of_two_sided_p(1.0), 0.0);
    assert!(normal_score_of_two_sided_p(f64::NAN).is_nan());
    for z in [0.3, 1.0, 2.5, 8.0, 20.0, 37.0] {
        let p = 2.0 * upper_normal_p(z);
        let back = normal_score_of_two_sided_p(p);
        assert!((back - z).abs() < 1e-8 * z.max(1.0), "{z}: {back}");
    }
    // A t on few degrees of freedom scores less than its statistic, and a p
    // that underflowed is infinite.
    let few = Estimate {
        df: 4.0,
        ..est(-0.15, 0.05)
    };
    let s = normal_score_of_two_sided_p(few.p());
    assert!(s > 0.0 && s < 3.0, "{s}");
    assert_eq!(normal_score_of_two_sided_p(0.0), f64::INFINITY);
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

/// A row with a headline z, a co-expression z, the receptor-negative z and η².
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
            unadj: nan,
            ipw: nan,
            outcome: nan,
            incompetent: est(z3 * 0.05, 0.05),
            versus_rest: nan,
            steps: [nan; 3],
            nospill: nan,
            near_determined: 0.0,
            abf: AbfSummary::NONE,
        },
        exposure_eta2: eta2,
        n_competent_bins: 1,
        n_incompetent_bins: 1,
        curve: Vec::new(),
        marginal: None,
        jackknife: crate::lr_activity::ligand_effect::jackknife::Jackknife::NONE,
        effect_dr_p: f64::NAN,
        effect_dr_lfdr: LocalFdr::NAN,
        coexpr_p: f64::NAN,
        coexpr_lfdr: LocalFdr::NAN,
        effect_dr_incompetent_lfdr: LocalFdr::NAN,
        effect_dr_contrast_lfdr: LocalFdr::NAN,
        n_pairs_same_ligand: 0,
        verdict: Verdict::InsufficientData,
        warnings: String::new(),
    }
}

/// Null pairs in section `all`, enough for a local fdr: every statistic at
/// the normal's quantiles, each in its own order so no two are tied.
const NULL_PAIRS: usize = 150;

fn quantile(k: usize) -> f64 {
    let u = (k as f64 + 0.5) / NULL_PAIRS as f64;
    normal_score_of_two_sided_p(2.0 * u.min(1.0 - u)) * (u - 0.5).signum()
}

/// `rows`, then [`NULL_PAIRS`] null pairs after them.
fn among_nulls(mut rows: Vec<EffectRow>) -> Vec<EffectRow> {
    rows.extend((0..NULL_PAIRS).map(|k| {
        let (zc, z3) = (quantile(NULL_PAIRS - 1 - k), quantile(k * 7 % NULL_PAIRS));
        row("all", &format!("N{k}"), quantile(k), zc, z3, 0.1)
    }));
    rows
}

#[test]
fn report_labels_and_overrides() {
    let mut rows = among_nulls(vec![
        row("all", "L1", 10.0, 10.0, 0.0, 0.1),
        row("all", "L1", 10.0, 10.0, 10.0, 0.1),
        row("all", "L2", 10.0, 10.0, 0.0, 0.9),
        row("all", "L3", f64::NAN, 10.0, 0.0, 0.1),
        row("all", "L4", 0.1, 0.0, 0.0, 0.1),
        row("all", "L5", 10.0, -3.0, 0.0, 0.1),
        row("all", "L6", 0.0, 10.0, 0.0, 0.1),
        // Alone in its section: too few for a local fdr, so its p is tested
        // alone, and passes.
        row("s1", "L1", 10.0, 10.0, 0.0, 0.1),
    ]);
    rows[3].effects.competent = Estimate::nan(400, 3);
    report(&mut rows, &ReportOpts::default());
    let got: Vec<&str> = rows[..8].iter().map(|r| r.verdict.code()).collect();
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
            "supported",
        ]
    );
    assert!(rows[0].warnings.is_empty(), "{}", rows[0].warnings);
    assert!(rows[1].warnings.contains("receptor-negative"));
    assert!(rows[2].warnings.contains("neighbour cell type"));
    assert!(rows[2].warnings.contains("cell-type artefact"));
    assert!(
        rows[3].warnings.contains("thin arm"),
        "{}",
        rows[3].warnings
    );

    // Ligands are counted within a section.
    assert_eq!(rows[0].n_pairs_same_ligand, 2);
    assert_eq!(rows[2].n_pairs_same_ligand, 1);
    assert_eq!(rows[7].n_pairs_same_ligand, 1);
    // A section too small has p-values and no local fdr.
    assert!((rows[7].effect_dr_p - t_two_sided_p(10.0, f64::INFINITY)).abs() < 1e-12);
    assert!(rows[7].effect_dr_lfdr.empirical_null.is_nan());
    // The NaN headline is out of the family and has no lfdr.
    assert!(rows[3].effect_dr_p.is_nan() && rows[3].effect_dr_lfdr.empirical_null.is_nan());
    // Far out the null is gone, at its centre it is all there is.
    let l = |i: usize| rows[i].effect_dr_lfdr;
    assert!(l(0).empirical_null < 1e-6 && l(0).theoretical_null < 1e-6);
    assert!(l(6).empirical_null > 0.9 && l(6).theoretical_null > 0.9);
    // One-sided: a negative co-expression score is null.
    assert!(rows[5].coexpr_p > 0.99);
    assert_eq!(rows[5].coexpr_lfdr.empirical_null, 1.0);
}

#[test]
fn local_fdr_level_moves_verdicts() {
    let judged = |level: f64| {
        let mut rows = among_nulls(vec![row("all", "L1", 3.0, 0.0, -6.0, 0.1)]);
        let opts = ReportOpts {
            local_fdr_level: level,
            ..ReportOpts::default()
        };
        report(&mut rows, &opts);
        (rows[0].effect_dr_lfdr.empirical_null, rows[0].verdict)
    };
    let (l, _) = judged(0.2);
    assert!(l > 0.0 && l < 1.0, "{l}");
    assert_eq!(judged(l * 1.01).1, Verdict::EffectWithoutCoexpression);
    assert_eq!(judged(l * 0.99).1, Verdict::NoEvidence);
}

/// Every statistic is moved alike (a residual confounder widening every z):
/// the null fitted to the bulk widens with it, the theoretical one does not.
#[test]
fn an_inflated_family_is_read_against_its_own_null() {
    let mut rows = among_nulls(vec![row("all", "L1", 4.5, 0.0, 0.0, 0.1)]);
    for r in rows.iter_mut() {
        r.effects.competent.theta *= 1.6;
    }
    report(&mut rows, &ReportOpts::default());
    let l = rows[0].effect_dr_lfdr;
    assert!(l.empirical_null > l.theoretical_null, "{l:?}");
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

/// The controls are fitted over every pair, like the effect; a control
/// responding is a warning beside an effect only.
#[test]
fn controls_are_fitted_over_every_pair() {
    let e3 = |rows: &[EffectRow], i: usize| rows[i].effect_dr_incompetent_lfdr.empirical_null;
    // No effect: the control has its lfdr, and warns of nothing.
    let mut rows = among_nulls(vec![row("all", "L1", 0.0, 0.0, 10.0, 0.1)]);
    report(&mut rows, &ReportOpts::default());
    assert!(e3(&rows, 0) < 1e-6, "{}", e3(&rows, 0));
    assert_eq!(rows[0].verdict.code(), "no_evidence");
    assert!(
        !rows[0].warnings.contains("respond"),
        "{}",
        rows[0].warnings
    );

    // A significant effect whose control has no estimate has not held.
    let mut rows = among_nulls(vec![row("all", "L1", 10.0, 10.0, 0.0, 0.1)]);
    rows[0].effects.incompetent = Estimate::nan(10, 1);
    rows[0].effects.effect = Estimate::nan(10, 1);
    report(&mut rows, &ReportOpts::default());
    assert!(e3(&rows, 0).is_nan());
    assert_eq!(rows[0].verdict.code(), "control_failed");
}

/// Beside a significant effect the conservative verdict: the effect must
/// exceed the negative control, the same way, and a control that could not
/// run has not held.
#[test]
fn a_verdict_needs_a_specific_effect_and_the_control_checked() {
    let judged = |change: &dyn Fn(&mut EffectRow)| {
        let mut rows = among_nulls(vec![row("all", "L1", 10.0, 10.0, 0.0, 0.1)]);
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
}

/// A hidden field moves the control too, so a significant control does not
/// rule out an effect that clearly exceeds it: the verdict stands and the
/// warning says what responded.
#[test]
fn a_significant_control_the_effect_exceeds_is_a_warning_not_a_veto() {
    let mut rows = among_nulls(vec![row("all", "L1", 20.0, 10.0, 5.0, 0.1)]);
    report(&mut rows, &ReportOpts::default());
    assert_eq!(rows[0].verdict, Verdict::Supported);
    let w = &rows[0].warnings;
    assert!(w.contains("receptor-negative receivers respond"), "{w}");
}

/// A family too small for a local fdr tests its p-values by Bonferroni over
/// its finite ones: a strong pair passes, a merely nominal one does not.
#[test]
fn a_small_family_tests_its_p_values_by_bonferroni() {
    let mut rows: Vec<EffectRow> = (0..40)
        .map(|k| row("all", &format!("N{k}"), quantile(k * 3), 0.0, 0.0, 0.1))
        .collect();
    rows[0] = row("all", "L1", 8.0, 8.0, -2.0, 0.1);
    // p ≈ 0.012: below 0.05 alone, above 0.05 / 40.
    rows[1] = row("all", "L2", 2.5, 8.0, -6.0, 0.1);
    report(&mut rows, &ReportOpts::default());
    assert!(rows[0].effect_dr_lfdr.empirical_null.is_nan());
    assert_eq!(rows[0].verdict, Verdict::Supported, "{}", rows[0].warnings);
    assert_eq!(rows[1].verdict, Verdict::ColocationOnly);
}

/// A p that underflowed, or an SE of 0, keeps its pair in the fit at the
/// largest score rather than dropping it.
#[test]
fn the_strongest_pairs_stay_in_the_local_fdr() {
    let mut rows = among_nulls(vec![row("all", "L1", 10.0, 10.0, 0.0, 0.1)]);
    rows[0].effects.competent = est(400.0, 1.0);
    rows[0].coexpr_se = 0.0;
    report(&mut rows, &ReportOpts::default());
    assert!(rows[0].effect_dr_lfdr.empirical_null < 1e-6);
    assert!(rows[0].coexpr_lfdr.empirical_null < 1e-6);
    assert_eq!(rows[0].verdict, Verdict::Supported, "{}", rows[0].warnings);
}
