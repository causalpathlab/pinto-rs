//! Region labels: their confounder columns and their scopes.

use crate::lr_activity::ligand_effect::confounders::region_levels;
use crate::lr_activity::ligand_effect::tissue::{scope_rows, section_label, Scope};

/// Within each section every region but the most common gets a column, so
/// a region that two sections share gets one in each.
#[test]
fn region_columns_are_per_section_without_the_commonest() {
    let section = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    let region = [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 0];
    assert_eq!(region_levels(&section, &region), [(0, 1), (1, 0), (1, 2)]);
    // A tie goes to the smaller code as the reference.
    assert_eq!(region_levels(&[0, 0], &[3, 1]), [(0, 3)]);
    // One region everywhere: no column.
    assert!(region_levels(&[0, 1], &[4, 4]).is_empty());
}

#[test]
fn region_scopes_take_their_rows() {
    let scopes = [
        Scope::All,
        Scope::Region(1),
        Scope::Section(0),
        Scope::Region(0),
    ];
    let section = [0, 1, 0, 1];
    let region = [1, 1, 0, 2];
    let rows = scope_rows(&scopes, &section, None, Some(&region));
    assert_eq!(rows, [vec![0, 1, 2, 3], vec![0, 1], vec![0, 2], vec![2]]);
}

#[test]
fn a_section_named_like_a_region_row_is_prefixed() {
    assert_eq!(&*section_label("region:x"), "section:region:x");
    assert_eq!(&*section_label("S1"), "S1");
}

// * Planting in one region (`lra-simulate --plant-region`)

use crate::lr_activity::input::UNASSIGNED;
use crate::lr_activity::ligand_effect::baselines::Baselines;
use crate::lr_activity::ligand_effect::inference::Estimate;
use crate::lr_activity::ligand_effect::output::sim_summary;
use crate::lr_activity::ligand_effect::pipeline::planted_region;
use crate::lr_activity::ligand_effect::simulate::{Scenario, SimRow};
use crate::lr_activity::ligand_effect::tissue::{Tissue, TissueInputs, TissueOpts};
use crate::lr_activity::ligand_effect::verdict::Verdict;
use crate::util::common::Mat;

/// A grid, its left half region R1 and its right half R2.
fn region_tissue() -> Tissue {
    let side = 30;
    let n = side * side;
    let coords: Vec<Option<(f32, f32)>> = (0..n)
        .map(|i| Some(((i % side) as f32 * 10.0, (i / side) as f32 * 10.0)))
        .collect();
    let mut pairs = Vec::new();
    for i in 0..n {
        if i % side + 1 < side {
            pairs.push((i, i + 1));
        }
        if i / side + 1 < side {
            pairs.push((i, i + side));
        }
    }
    let bin: Vec<usize> = (0..n).map(|i| (i / side) / 10).collect();
    let identity = Mat::from_fn(2, n, |k, i| bin[i] as f32 * (k as f32 + 1.0));
    let section: Vec<Box<str>> = vec!["s0".into(); n];
    let depth = vec![100f32.ln_1p(); n];
    let codes: Vec<u32> = (0..n).map(|i| u32::from(i % side >= side / 2)).collect();
    let names: Vec<Box<str>> = ["R1", "R2", UNASSIGNED].map(Into::into).to_vec();
    Tissue::build(
        &TissueInputs {
            coords: &coords,
            section: &section,
            pairs: &pairs,
            identity: &identity,
            identity_bin: &bin,
            log1p_depth: &depth,
            cell_type: None,
            region: Some((&codes, &names)),
        },
        &TissueOpts {
            folds: 4,
            tile: 6.0,
            ..TissueOpts::default()
        },
    )
    .unwrap()
}

/// `--plant-region` names a region of `--region`; the error for one that
/// is not lists the regions it could name, `unassigned` not among them.
#[test]
fn a_planted_region_is_one_of_the_regions_named() {
    let t = region_tissue();
    assert_eq!(planted_region(&t, Some("R2")).unwrap(), Some(1));
    assert_eq!(planted_region(&t, None).unwrap(), None);
    for wrong in ["nope", UNASSIGNED] {
        let e = planted_region(&t, Some(wrong)).unwrap_err().to_string();
        assert!(e.contains("(has R1, R2)"), "{e}");
    }
}

/// One benchmark run planted in region `code`, every estimate missing.
pub(super) fn planted_run(code: u32) -> SimRow {
    let e = Estimate::nan(0, 0);
    let nan = f64::NAN;
    SimRow {
        ligand: "GENE1".into(),
        receptor: "GENE2".into(),
        scenario: Scenario {
            effect_pve: 0.1,
            region: Some(code),
            ..Scenario::default()
        },
        truth: nan,
        truth_exposed: nan,
        estimate: e,
        marginal_cor: e,
        unadjusted_diff: e,
        ipw: e,
        outcome: nan,
        incompetent: e,
        steps: [e; 3],
        contrast: e,
        verdict: Verdict::InsufficientData,
        misclassified: nan,
        baselines: Baselines {
            celltype_score: nan,
            celltype_p: nan,
            celltype_tested: 0,
            moran_i: nan,
            moran_p: nan,
            contact_score: nan,
            contact_p: nan,
            coexpr_log_or: nan,
            coexpr_se: nan,
            coexpr_p: nan,
        },
        seconds: 0.0,
        baseline_seconds: 0.0,
        region: None,
        field_sha256: None,
    }
}

/// The planted region goes by its name in the run's summary.
#[test]
fn the_planted_region_is_printed_by_name() {
    let summary = sim_summary(&[planted_run(1)], Some("R2"));
    assert!(summary.contains("planted in region R2"), "{summary}");
    assert!(!summary.contains("region 1"), "{summary}");
}

/// The table names the planted region, and gives it back to the
/// comparison read from it.
#[cfg(feature = "view")]
#[test]
fn the_planted_region_is_written_by_name() {
    use crate::lr_activity::ligand_effect::output::{
        compare_runs, read_planted_region, read_sim_runs, write_sim,
    };
    let rows = [planted_run(1)];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o.lra-simulate.parquet");
    let path = path.to_str().unwrap();
    write_sim(path, &rows, Some("R2")).unwrap();
    let planted = read_planted_region(path);
    assert_eq!(planted.as_deref(), Some("R2"));
    let c = compare_runs(&read_sim_runs(path).unwrap(), planted.as_deref());
    assert!(
        c.scenarios[0].scenario.ends_with("planted in region R2"),
        "{}",
        c.scenarios[0].scenario
    );
    // Planted everywhere, the table names no region.
    let everywhere = SimRow {
        scenario: Scenario {
            region: None,
            ..rows[0].scenario
        },
        ..rows[0].clone()
    };
    write_sim(path, &[everywhere], None).unwrap();
    assert_eq!(read_planted_region(path), None);
}
