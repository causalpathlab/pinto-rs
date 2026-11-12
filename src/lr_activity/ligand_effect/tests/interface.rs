//! What users see: the verdict vocabulary and file names.

use crate::lr_activity::ligand_effect::effects::ScopeEffects;
use crate::lr_activity::ligand_effect::inference::Estimate;
use crate::lr_activity::ligand_effect::output::{
    lra_path, lra_simulate_path, write_effects, Q_SCOPE_ALL,
};
use crate::lr_activity::ligand_effect::output::{percent_short, percent_text};
use crate::lr_activity::ligand_effect::rows::{EffectRow, Moments};
use crate::lr_activity::ligand_effect::verdict;
use crate::lr_activity::ligand_effect::verdict::{verdict_help, Evidence, Verdict};

// * Verdicts

#[test]
fn every_verdict_has_one_code_and_a_note() {
    let codes: Vec<&str> = Verdict::ALL.iter().map(|v| v.code()).collect();
    assert_eq!(
        codes,
        [
            "supported",
            "effect_without_coexpression",
            "colocation_only",
            "no_evidence",
            "control_failed",
            "celltype_inseparable",
            "insufficient_data",
        ]
    );
    for (k, v) in Verdict::ALL.into_iter().enumerate() {
        // `counts` indexes by `index`: it is the place in ALL.
        assert_eq!(v.index(), k);
        assert_eq!(Verdict::counts([v])[k], (v, 1));
        assert_eq!(v.code().parse::<Verdict>().unwrap(), v);
        assert!(
            !v.note().is_empty() && !v.note().contains("  "),
            "{}",
            v.code()
        );
        assert!(v.code().chars().all(|c| c.is_ascii_lowercase() || c == '_'));
    }
    assert!("both".parse::<Verdict>().is_err());
    // Every combination of evidence the judge can see lands on a verdict,
    // and every verdict is reachable.
    let mut seen = std::collections::HashSet::new();
    for bits in 0..32u8 {
        let e = Evidence {
            estimable: bits & 1 != 0,
            effect: bits & 2 != 0,
            coexpr: bits & 4 != 0,
            control_failed: bits & 8 != 0,
            confounded: bits & 16 != 0,
        };
        seen.insert(verdict::verdict(e));
    }
    assert_eq!(seen.len(), Verdict::ALL.len());
    let help = verdict_help();
    assert_eq!(help.lines().count(), Verdict::ALL.len());
    assert!(Verdict::ALL.iter().all(|v| help.contains(v.code())));
}

/// The README's verdict table lists every verdict, code and note, in order.
#[test]
fn readme_verdict_table_matches_the_enum() {
    let readme = include_str!("../../../../README.md");
    let table: Vec<(String, String)> = readme
        .lines()
        .filter_map(|l| {
            let cells: Vec<&str> = l.split('|').map(str::trim).collect();
            let code = cells.get(1)?.strip_prefix('`')?.strip_suffix('`')?;
            code.parse::<Verdict>().ok()?;
            Some((code.to_string(), cells.get(2)?.to_string()))
        })
        .collect();
    let want: Vec<(String, String)> = Verdict::ALL
        .iter()
        .map(|v| (v.code().to_string(), v.note().to_string()))
        .collect();
    assert_eq!(table, want);
    // The viewer's `v` steps through the verdicts in that order too.
    let codes: Vec<&str> = Verdict::ALL.iter().map(|v| v.code()).collect();
    let cycle = format!("with `v`/`V` ({}, or\nall)", codes.join(", "));
    let flat = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        flat(readme).contains(&flat(&cycle)),
        "README's v/V cycle: {}",
        flat(&cycle)
    );
}

/// A log-rate effect as a percent change with its 95% interval.
fn percent_change(e: &Estimate) -> String {
    let m = Moments::of(e);
    percent_text(m.pct, m.pct_lo, m.pct_hi)
}

#[test]
fn an_effect_reads_as_a_percent_change_with_its_interval() {
    let e = Estimate {
        theta: 0.0,
        se: 0.0,
        df: f64::INFINITY,
        ..Estimate::nan(100, 50)
    };
    // Under a percent, whole percents would all read +0%.
    assert_eq!(percent_change(&e), "+0.00% (95% CI +0.00% to +0.00%)");
    let e = Estimate {
        theta: 2f64.ln(),
        ..e
    };
    assert!(percent_change(&e).starts_with("+100%"));
}

/// The digits follow the size of the numbers, so a column of small effects
/// does not round some to whole percents, and nothing reads `-0`.
#[test]
fn a_percent_change_keeps_its_digits_and_never_reads_minus_zero() {
    assert_eq!(
        percent_text(-0.6, -1.5, 0.3),
        "-0.6% (95% CI -1.5% to +0.3%)"
    );
    assert_eq!(
        percent_text(0.59, 0.25, 0.9),
        "+0.59% (95% CI +0.25% to +0.90%)"
    );
    assert_eq!(percent_text(12.4, 3.0, 22.0), "+12% (95% CI +3% to +22%)");
    assert_eq!(
        percent_text(-0.001, -0.004, 0.002),
        "+0.00% (95% CI +0.00% to +0.00%)"
    );
    assert_eq!(percent_short(-0.3, -1.2, 0.6), "-0.3%");
    assert_eq!(percent_short(-0.2, -9.0, 30.0), "+0%");
    assert_eq!(percent_short(f64::NAN, 0.0, 0.0), "—");
}

// * The effects table

#[test]
fn the_table_gives_ipw_an_interval_and_the_regression_a_point_estimate() {
    use legume_numeric::matrix::parquet::read_table_columns;
    let ipw = Estimate {
        theta: 0.2,
        se: 0.05,
        df: f64::INFINITY,
        ..Estimate::nan(100, 50)
    };
    let outcome = Estimate { theta: 0.3, ..ipw };
    let none = Estimate::nan(0, 0);
    let effects = ScopeEffects {
        ipw,
        outcome,
        ..ScopeEffects::zip_with(&[], |_| none)
    };
    let row = EffectRow::pending(effects, (f64::NAN, f64::NAN), f64::NAN);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o.lra.parquet");
    let path = path.to_str().unwrap();
    write_effects(path, &[row], Q_SCOPE_ALL).unwrap();

    let names = [
        "effect_ipw",
        "effect_ipw_se",
        "effect_ipw_lo",
        "effect_ipw_hi",
        "effect_outcome",
    ];
    let (_, x) = read_table_columns(path, &["ligand"], &names).unwrap();
    let (lo, hi) = ipw.interval();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
    assert!(close(x[0][0], 0.2) && close(x[1][0], 0.05));
    assert!(lo < hi && close(x[2][0], lo) && close(x[3][0], hi));
    assert!(close(x[4][0], 0.3));
    assert!(read_table_columns(path, &["ligand"], &["effect_outcome_se"]).is_err());
    // The ring and background controls were removed: none of their columns
    // are written.
    for gone in [
        "ring_effect",
        "ring_effect_q",
        "effect_over_ring",
        "effect_over_ring_q",
        "background_effect",
        "background_effect_se",
        "background_effect_df",
        "background_q",
        "effect_over_background",
        "effect_over_background_se",
        "effect_over_background_q",
        "n_background_genes",
    ] {
        assert!(
            read_table_columns(path, &["ligand"], &[gone]).is_err(),
            "{gone}"
        );
    }
}

// * Files

#[test]
fn outputs_are_named_lra() {
    assert_eq!(lra_path("res/run"), "res/run.lra.parquet");
    assert_eq!(lra_simulate_path("res/run"), "res/run.lra-simulate.parquet");
}

/// A section literally named like the pooled row, or a cell-type row, is
/// relabelled so the per-section rows cannot collide with them.
#[test]
fn section_labels_never_collide_with_the_pooled_row() {
    use crate::lr_activity::ligand_effect::tissue::section_label;
    use crate::lr_activity::ligand_effect::tissue::ALL_SECTIONS;
    assert_eq!(&*section_label("slide1"), "slide1");
    assert_ne!(&*section_label(ALL_SECTIONS), ALL_SECTIONS);
    assert_eq!(&*section_label("type:T"), "section:type:T");
    assert_ne!(section_label("section:all"), section_label("all"));
}
