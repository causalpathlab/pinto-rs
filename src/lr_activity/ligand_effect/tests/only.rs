//! `pinto lra --only`: the entries, matching them to pairs on their genes,
//! the errors for entries no pair answers, the subset's file names, and
//! its record in the manifest apart from the full table.

use super::super::only::{self, canonical, edit_distance, parse, tag, Wanted};
use super::super::{record, Pair};
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
        "LIGB1", "RECE1", "RECE2", "LIGA1", "RECA1", "RECC1", "RECB1",
    ]
    .iter()
    .map(|&g| g.into())
    .collect();
    let index = GeneIndex::build(&names);
    let pairs = vec![
        pair("LIGB1", "RECE1_RECE2", &[0], &[1, 2]),
        pair("LIGB1", "RECB1", &[0], &[6]),
        pair("LIGA1", "RECA1", &[3], &[4]),
        pair("LIGA1", "RECC1", &[3], &[5]),
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
    let w = parse(&list(&[" LIGB1:RECE1_RECE2 ", "liga1", ""])).unwrap();
    assert_eq!(
        w,
        [
            Wanted {
                ligand: "LIGB1".into(),
                receptor: Some("RECE1_RECE2".into())
            },
            Wanted {
                ligand: "liga1".into(),
                receptor: None
            },
        ]
    );
    assert_eq!(w[0].to_string(), "LIGB1:RECE1_RECE2");
    for bad in ["A:B:C", ":RECA1", "LIGA1:", ":"] {
        let e = parse(&list(&[bad])).unwrap_err().to_string();
        assert!(e.contains("--only"), "{bad}: {e}");
    }
    assert!(parse(&list(&["", " "])).is_err());
}

// * Matching

#[test]
fn entries_match_on_genes_case_and_subunit_order_aside() {
    assert_eq!(select(&["ligb1:rece2_rece1"]).unwrap(), [0]);
    assert_eq!(select(&["LIGB1:RECE1+RECE2"]).unwrap(), [0]);
    // A bare ligand takes all of its pairs; pairs stay in table order.
    assert_eq!(select(&["LIGA1", "ligb1:recb1"]).unwrap(), [1, 2, 3]);
    // Named twice, kept once.
    assert_eq!(select(&["LIGA1:RECA1", "liga1"]).unwrap(), [2, 3]);
}

#[test]
fn an_entry_no_pair_answers_names_the_nearest() {
    let e = select(&["LIGA1:RECA"]).unwrap_err().to_string();
    assert!(e.contains("LIGA1:RECA: no gene RECA in the data"), "{e}");
    assert!(e.contains("nearest: LIGA1:RECA1"), "{e}");
    let e = select(&["LIGB2"]).unwrap_err().to_string();
    assert!(e.contains("no gene LIGB2"), "{e}");
    assert!(e.contains("nearest: LIGB1"), "{e}");
    // Genes known, but no such pair: the ligand's pairs come first.
    let e = select(&["LIGB1:RECA1"]).unwrap_err().to_string();
    assert!(e.contains("no such pair"), "{e}");
    assert!(e.contains("LIGB1:RECB1"), "{e}");
    // Every unknown entry is listed, the known ones are not.
    let e = select(&["LIGA1", "FOO", "BAR:RECA1"])
        .unwrap_err()
        .to_string();
    assert!(e.contains("FOO") && e.contains("BAR:RECA1"), "{e}");
    assert!(!e.contains("LIGA1: no"), "{e}");
}

#[test]
fn edit_distance_counts_edits() {
    assert_eq!(edit_distance("", ""), 0);
    assert_eq!(edit_distance("RECA", "RECA1"), 1);
    assert_eq!(edit_distance("kitten", "sitting"), 3);
    assert_eq!(edit_distance("abc", ""), 3);
}

// * Files

#[test]
fn a_subset_is_named_by_its_list_not_over_the_full_table() {
    let short = list(&["LIGB1", "liga1"]);
    assert_eq!(canonical(&short), ["LIGA1", "LIGB1"]);
    assert_eq!(tag(&short), "liga1+ligb1");
    // Order, case and spacing do not change the name.
    assert_eq!(tag(&list(&[" Liga1", "LIGB1 ", "ligb1"])), "liga1+ligb1");
    assert_eq!(
        only::table_path("res/run", &short),
        "res/run.lra.only-liga1+ligb1.parquet"
    );
    assert_eq!(
        only::curve_path("res/run", &short),
        "res/run.lra.only-liga1+ligb1.curve.parquet"
    );
    assert_ne!(
        only::table_path("res/run", &short),
        super::super::lra_path("res/run")
    );
    // A long list is named by a short stable hash.
    let long = list(&["LIGB1:RECE1_RECE2", "LIGA1:RECA1", "LIGA0:RECD1"]);
    let t = tag(&long);
    assert_eq!(t.len(), 8);
    assert!(t.chars().all(|c| c.is_ascii_hexdigit()), "{t}");
    let reordered = list(&["liga0:recd1", "LIGA1:RECA1", "LIGB1:RECE1_RECE2"]);
    assert_eq!(tag(&reordered), t);
    assert_ne!(tag(&list(&["LIGB1:RECE1_RECE2", "LIGA1:RECA1"])), t);
    // `:` makes a readable `-`.
    assert_eq!(tag(&list(&["LIGA1:RECA1"])), "liga1-reca1");
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
        names(&list(&["LIGA1"])),
        ["o.lra.only-liga1.parquet", "o.lra.only-liga1.curve.parquet"]
    );
}

// * Manifest

fn subset(table: &str, pairs: usize) -> LraSubset {
    LraSubset {
        only: vec!["LIGA1".into()],
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
        o.record_subset(subset("/r/run.lra.only-liga1.parquet", 2))
    })
    .unwrap();
    record(&manifest, |o| {
        o.record_subset(subset("/r/run.lra.only-reca1.parquet", 1))
    })
    .unwrap();
    // The same subset again replaces its entry.
    record(&manifest, |o| {
        o.record_subset(subset("/r/run.lra.only-liga1.parquet", 3))
    })
    .unwrap();
    let lra = PintoMetadata::read(&manifest).unwrap().outputs.lra;
    assert_eq!(lra.table.as_deref(), Some("/r/run.lra.parquet"));
    assert_eq!(lra.subsets.len(), 2);
    assert_eq!(lra.subsets[0].pairs, 3);
    assert_eq!(lra.subsets[1].table, "/r/run.lra.only-reca1.parquet");
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
