//! Across pairs: the t and normal references, the p-values and warnings,
//! co-expression's local fdr over a family of pairs, and the cell-type
//! identifiability diagnostic on the synthetic grid tissue.

use super::rows_of;
use super::simulate::fixture;
use crate::lr_activity::ligand_effect::averaging::AbfSummary;
use crate::lr_activity::ligand_effect::dose::eta_squared;
use crate::lr_activity::ligand_effect::effects::ScopeEffects;
use crate::lr_activity::ligand_effect::inference::{
    t_crit_975, t_two_sided_p, upper_normal_p, Estimate,
};
use crate::lr_activity::ligand_effect::report::report;
use crate::lr_activity::ligand_effect::rows::{EffectRow, LocalFdr};
use crate::lr_activity::ligand_effect::INTERNALS;

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

/// A row with a headline z and a co-expression z.
fn row(section: &str, ligand: &str, z: f64, zc: f64) -> EffectRow {
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
            effect: est(z * 0.05, 0.05),
            competent: est(z * 0.05, 0.05),
            unadj: nan,
            ipw: nan,
            outcome: nan,
            incompetent: est(0.0, 0.05),
            versus_rest: nan,
            steps: [nan; 3],
            nospill: nan,
            near_determined: 0.0,
            abf: AbfSummary::NONE,
        },
        exposure_eta2: 0.1,
        n_competent_bins: 1,
        n_incompetent_bins: 1,
        curve: Vec::new(),
        marginal: None,
        jackknife: crate::lr_activity::ligand_effect::jackknife::Jackknife::NONE,
        effect_dr_p: f64::NAN,
        coexpr_p: f64::NAN,
        coexpr_lfdr: LocalFdr::NAN,
        n_pairs_same_ligand: 0,
        warnings: String::new(),
    }
}

/// Null pairs in section `all`, enough for co-expression's local fdr.
const NULL_PAIRS: usize = 150;

/// The normal's quantile at `u`, by bisection on its upper tail.
fn normal_quantile(u: f64) -> f64 {
    let (mut lo, mut hi) = (-10.0, 10.0);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if 1.0 - upper_normal_p(mid) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// `rows`, then [`NULL_PAIRS`] null pairs after them, every statistic at the
/// normal's quantiles.
fn among_nulls(mut rows: Vec<EffectRow>) -> Vec<EffectRow> {
    let q = |k: usize| normal_quantile((k as f64 + 0.5) / NULL_PAIRS as f64);
    rows.extend((0..NULL_PAIRS).map(|k| row("all", &format!("N{k}"), q(k), q(NULL_PAIRS - 1 - k))));
    rows
}

/// Every row gets its p-values, the not estimable say why, and nothing is
/// judged.
#[test]
fn report_gives_p_values_and_says_why_a_pair_has_no_estimate() {
    let mut rows = among_nulls(vec![
        row("all", "L1", 10.0, 10.0),
        row("all", "L1", 0.0, -3.0),
        row("all", "L3", f64::NAN, 10.0),
        // Alone in its section: too few for a local fdr.
        row("s1", "L1", 10.0, 10.0),
    ]);
    rows[2].effects.competent = Estimate::nan(400, 3);
    report(&mut rows, INTERNALS.min_arm);
    assert!((rows[0].effect_dr_p - t_two_sided_p(10.0, f64::INFINITY)).abs() < 1e-12);
    assert!(rows[0].warnings.is_empty(), "{}", rows[0].warnings);
    assert!(rows[2].effect_dr_p.is_nan());
    assert!(
        rows[2].warnings.contains("thin arm"),
        "{}",
        rows[2].warnings
    );
    // Ligands are counted within a section.
    assert_eq!(rows[0].n_pairs_same_ligand, 2);
    assert_eq!(rows[3].n_pairs_same_ligand, 1);
    // A section too small has p-values and no local fdr.
    assert!((rows[3].effect_dr_p - rows[0].effect_dr_p).abs() < 1e-12);
    assert!(rows[3].coexpr_lfdr.empirical_null.is_nan());
    // Far out the null is gone, at its centre it is all there is.
    let l = |i: usize| rows[i].coexpr_lfdr;
    assert!(l(0).empirical_null < 1e-6 && l(0).theoretical_null < 1e-6);
    // One-sided: a negative co-expression score is null.
    assert!(rows[1].coexpr_p > 0.99);
    assert_eq!(l(1).empirical_null, 1.0);
}

/// Spatial autocorrelation moves every co-expression score alike: the null
/// fitted to the bulk widens with it, the theoretical one does not.
#[test]
fn an_inflated_coexpression_family_is_read_against_its_own_null() {
    let mut rows = among_nulls(vec![row("all", "L1", 0.0, 4.5)]);
    for r in rows.iter_mut() {
        r.coexpr_log_or *= 1.6;
    }
    report(&mut rows, INTERNALS.min_arm);
    let l = rows[0].coexpr_lfdr;
    assert!(l.empirical_null > l.theoretical_null, "{l:?}");
}

/// A co-expression score that is infinite (an SE of 0) stays in the fit as
/// signal rather than dropping out.
#[test]
fn the_strongest_coexpression_stays_in_the_local_fdr() {
    let mut rows = among_nulls(vec![row("all", "L1", 10.0, 10.0)]);
    rows[0].coexpr_se = 0.0;
    report(&mut rows, INTERNALS.min_arm);
    assert_eq!(rows[0].coexpr_lfdr.empirical_null, 0.0);
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
