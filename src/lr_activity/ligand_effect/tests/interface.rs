//! What users see: the verdict vocabulary, the study JSON, file names.

use crate::lr_activity::ligand_effect::inference::Estimate;
use crate::lr_activity::ligand_effect::output::percent_text;
use crate::lr_activity::ligand_effect::output::{lra_path, lra_simulate_path};
use crate::lr_activity::ligand_effect::rows::Moments;
use crate::lr_activity::ligand_effect::simulate::{default_grid, ConfounderKind, Study};
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

// * Study

#[test]
fn the_example_study_round_trips_to_the_default() {
    let text = Study::example_json();
    let mut back = Study::from_json(&text).unwrap();
    assert!(back.doc.is_some() && back.scenarios.doc.is_some());
    back.doc = None;
    back.scenarios.doc = None;
    assert_eq!(back.grid(), Study::default().grid());
    assert_eq!((back.reps, back.pairs), (3, 3));
    assert_eq!(
        default_grid(&back.grid()).len(),
        default_grid(&Study::default().grid()).len()
    );
}

#[test]
fn a_partial_study_keeps_the_defaults_and_a_typo_is_refused() {
    let s = Study::from_json(
        r#"{"reps": 2, "scenarios": {"effect_sizes": [0.2, 0.5], "confounders": ["fourier"],
            "shapes": ["linear", "quadratic"], "marker": false}}"#,
    )
    .unwrap();
    assert_eq!(s.reps, 2);
    assert_eq!(s.pairs, Study::default().pairs);
    assert_eq!(s.scenarios.scales, Study::default().scenarios.scales);
    let grid = default_grid(&s.grid());
    assert!(grid.iter().any(|x| x.effect == 0.5) && grid.iter().any(|x| x.effect == 0.0));
    assert!(grid.iter().all(|x| !x.marker));
    assert!(grid
        .iter()
        .filter(|x| x.confound_ligand != 0.0)
        .all(|x| x.confounder == ConfounderKind::Fourier));
    assert!(Study::from_json(r#"{"scenarios": {"effect_size": [0.2]}}"#).is_err());
    assert!(Study::from_json(r#"{"repz": 2}"#).is_err());
    assert!(Study::from_json(r#"{"scenarios": {"shapes": ["interaction"]}}"#).is_err());
    assert!(Study::from_json(r#"{"reps": 0}"#).is_err());
    assert!(Study::from_json(r#"{"scenarios": {"shapes": ["cubic"]}}"#).is_err());
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
