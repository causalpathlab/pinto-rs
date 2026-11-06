//! `pinto lra --only`: the entries, matching them to pairs on their genes,
//! the errors for entries no pair answers, the subset's file names, and
//! its record in the manifest apart from the full table.

use super::super::only::{self, canonical, edit_distance, parse, tag, Wanted};
use super::super::pairs::Pair;
use super::super::pipeline::record;
use crate::util::metadata::{LraOptions, LraSubset, PintoMetadata};
use data_beans::utilities::name_matching::GeneIndex;

fn list(entries: &[&str]) -> Vec<Box<str>> {
    entries.iter().map(|&e| e.into()).collect()
}

fn pair(ligand: &str, receptor: &str, lg: &[usize], rg: &[usize]) -> Pair {
    Pair {
        ligand: ligand.into(),
        receptor: receptor.into(),
        ligand_genes: lg.to_vec(),
        receptor_genes: rg.to_vec(),
        targets: Vec::new(),
        n_removed: 0,
    }
}

/// A panel and the pairs resolved on it.
fn panel() -> (Vec<Box<str>>, GeneIndex, Vec<Pair>) {
    let names: Vec<Box<str>> = [
        "TGFB1", "TGFBR1", "TGFBR2", "SPP1", "CD44", "ITGB1", "ACVRL1",
    ]
    .iter()
    .map(|&g| g.into())
    .collect();
    let index = GeneIndex::build(&names);
    let pairs = vec![
        pair("TGFB1", "TGFBR1_TGFBR2", &[0], &[1, 2]),
        pair("TGFB1", "ACVRL1", &[0], &[6]),
        pair("SPP1", "CD44", &[3], &[4]),
        pair("SPP1", "ITGB1", &[3], &[5]),
    ];
    (names, index, pairs)
}

fn select(entries: &[&str]) -> anyhow::Result<Vec<usize>> {
    let keep = select_mask(entries)?;
    Ok((0..keep.len()).filter(|&k| keep[k]).collect())
}

fn select_mask(entries: &[&str]) -> anyhow::Result<Vec<bool>> {
    let (names, index, pairs) = panel();
    only::select(&parse(&list(entries))?, &pairs, &names, &index)
}

// * Entries

#[test]
fn entries_are_pairs_or_bare_ligands() {
    let w = parse(&list(&[" TGFB1:TGFBR1_TGFBR2 ", "spp1", ""])).unwrap();
    assert_eq!(
        w,
        [
            Wanted {
                ligand: "TGFB1".into(),
                receptor: Some("TGFBR1_TGFBR2".into())
            },
            Wanted {
                ligand: "spp1".into(),
                receptor: None
            },
        ]
    );
    assert_eq!(w[0].to_string(), "TGFB1:TGFBR1_TGFBR2");
    for bad in ["A:B:C", ":CD44", "SPP1:", ":"] {
        let e = parse(&list(&[bad])).unwrap_err().to_string();
        assert!(e.contains("--only"), "{bad}: {e}");
    }
    assert!(parse(&list(&["", " "])).is_err());
}

// * Matching

#[test]
fn entries_match_on_genes_case_and_subunit_order_aside() {
    assert_eq!(select(&["tgfb1:tgfbr2_tgfbr1"]).unwrap(), [0]);
    assert_eq!(select(&["TGFB1:TGFBR1+TGFBR2"]).unwrap(), [0]);
    // A bare ligand takes all of its pairs; pairs stay in table order.
    assert_eq!(select(&["SPP1", "tgfb1:acvrl1"]).unwrap(), [1, 2, 3]);
    // Named twice, kept once.
    assert_eq!(select(&["SPP1:CD44", "spp1"]).unwrap(), [2, 3]);
}

#[test]
fn an_entry_no_pair_answers_names_the_nearest() {
    let e = select(&["SPP1:CD4"]).unwrap_err().to_string();
    assert!(e.contains("SPP1:CD4: no gene CD4 in the data"), "{e}");
    assert!(e.contains("nearest: SPP1:CD44"), "{e}");
    let e = select(&["TGFB2"]).unwrap_err().to_string();
    assert!(e.contains("no gene TGFB2"), "{e}");
    assert!(e.contains("nearest: TGFB1"), "{e}");
    // Genes known, but no such pair: the ligand's pairs come first.
    let e = select(&["TGFB1:CD44"]).unwrap_err().to_string();
    assert!(e.contains("no such pair"), "{e}");
    assert!(e.contains("TGFB1:ACVRL1"), "{e}");
    // Every unknown entry is listed, the known ones are not.
    let e = select(&["SPP1", "FOO", "BAR:CD44"])
        .unwrap_err()
        .to_string();
    assert!(e.contains("FOO") && e.contains("BAR:CD44"), "{e}");
    assert!(!e.contains("SPP1: no"), "{e}");
}

#[test]
fn edit_distance_counts_edits() {
    assert_eq!(edit_distance("", ""), 0);
    assert_eq!(edit_distance("CD4", "CD44"), 1);
    assert_eq!(edit_distance("kitten", "sitting"), 3);
    assert_eq!(edit_distance("abc", ""), 3);
}

// * Files

#[test]
fn a_subset_is_named_by_its_list_not_over_the_full_table() {
    let short = list(&["TGFB1", "spp1"]);
    assert_eq!(canonical(&short), ["SPP1", "TGFB1"]);
    assert_eq!(tag(&short), "spp1+tgfb1");
    // Order, case and spacing do not change the name.
    assert_eq!(tag(&list(&[" Spp1", "TGFB1 ", "tgfb1"])), "spp1+tgfb1");
    assert_eq!(
        only::table_path("res/run", &short),
        "res/run.lra.only-spp1+tgfb1.parquet"
    );
    assert_eq!(
        only::curve_path("res/run", &short),
        "res/run.lra.only-spp1+tgfb1.curve.parquet"
    );
    assert_ne!(
        only::table_path("res/run", &short),
        super::super::lra_path("res/run")
    );
    // A long list is named by a short stable hash.
    let long = list(&["TGFB1:TGFBR1_TGFBR2", "SPP1:CD44", "CXCL12:CXCR4"]);
    let t = tag(&long);
    assert_eq!(t.len(), 8);
    assert!(t.chars().all(|c| c.is_ascii_hexdigit()), "{t}");
    let reordered = list(&["cxcl12:cxcr4", "SPP1:CD44", "TGFB1:TGFBR1_TGFBR2"]);
    assert_eq!(tag(&reordered), t);
    assert_ne!(tag(&list(&["TGFB1:TGFBR1_TGFBR2", "SPP1:CD44"])), t);
    // `:` makes a readable `-`.
    assert_eq!(tag(&list(&["SPP1:CD44"])), "spp1-cd44");
}

#[test]
fn lra_outputs_follow_only() {
    use crate::lr_activity::args::LraMethod;
    let names = |only: &[Box<str>]| -> Vec<String> {
        crate::lr_activity::lra_outputs(LraMethod::ContactEffect, "o", only)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    };
    assert_eq!(names(&[]), ["o.lra.parquet", "o.lra.curve.parquet"]);
    assert_eq!(
        names(&list(&["SPP1"])),
        ["o.lra.only-spp1.parquet", "o.lra.only-spp1.curve.parquet"]
    );
}

// * Manifest

fn subset(table: &str, pairs: usize) -> LraSubset {
    LraSubset {
        only: vec!["SPP1".into()],
        pairs,
        of: 10,
        table: table.into(),
        curve: table.replace(".parquet", ".curve.parquet"),
        options: Some(LraOptions::default()),
    }
}

#[test]
fn a_subset_is_recorded_apart_from_the_full_table() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("run.pinto.json");
    PintoMetadata {
        command: "lc".into(),
        prefix: "run".into(),
        ..Default::default()
    }
    .write(&manifest)
    .unwrap();
    record(&manifest, |o| {
        o.table = Some("/r/run.lra.parquet".into());
        o.options = Some(LraOptions::default());
    })
    .unwrap();
    record(&manifest, |o| {
        o.record_subset(subset("/r/run.lra.only-spp1.parquet", 2))
    })
    .unwrap();
    record(&manifest, |o| {
        o.record_subset(subset("/r/run.lra.only-cd44.parquet", 1))
    })
    .unwrap();
    // The same subset again replaces its entry.
    record(&manifest, |o| {
        o.record_subset(subset("/r/run.lra.only-spp1.parquet", 3))
    })
    .unwrap();
    let lra = PintoMetadata::read(&manifest).unwrap().outputs.lra;
    assert_eq!(lra.table.as_deref(), Some("/r/run.lra.parquet"));
    assert_eq!(lra.subsets.len(), 2);
    assert_eq!(lra.subsets[0].pairs, 3);
    assert_eq!(lra.subsets[1].table, "/r/run.lra.only-cd44.parquet");
    // A manifest from before subsets reads, and writes none.
    let old: PintoMetadata = serde_json::from_str(
        r#"{"command":"lc","version":"0.16.3","timestamp":"0","prefix":"run","n_cells":1,
            "n_features":1,"outputs":{"lra":{"table":"t"}}}"#,
    )
    .unwrap();
    assert!(old.outputs.lra.subsets.is_empty() && old.outputs.lra.options.is_none());
    let text = serde_json::to_string(&old).unwrap();
    assert!(
        !text.contains("subsets") && !text.contains("options"),
        "{text}"
    );
}
