//! Pair resolution: complexes split into subunits before gene matching,
//! the scarcest-subunit rule, and deduplication on the matched genes.

use super::super::targets::TargetSets;
use super::super::{resolve_pairs, subunit_genes};
use crate::util::common::*;
use data_beans::utilities::name_matching::GeneIndex;

fn panel(genes: &[&str]) -> (Vec<Box<str>>, GeneIndex) {
    let names: Vec<Box<str>> = genes.iter().map(|g| (*g).into()).collect();
    let index = GeneIndex::build(&names);
    (names, index)
}

#[test]
fn a_complex_matches_every_subunit_not_its_trailing_one() {
    let (names, index) = panel(&["RECB2", "RECC1", "LIGG1"]);
    assert_eq!(
        subunit_genes("RECB2_RECC1", &names, &index),
        Some(vec![0, 1])
    );
    assert_eq!(
        subunit_genes("recb2+recc1", &names, &index),
        Some(vec![0, 1])
    );
    assert_eq!(subunit_genes("LIGG1", &names, &index), Some(vec![2]));
}

#[test]
fn a_complex_with_a_missing_subunit_is_absent() {
    let (names, index) = panel(&["RECC1", "LIGG1"]);
    assert_eq!(subunit_genes("RECB2_RECC1", &names, &index), None);
}

#[test]
fn a_panel_gene_with_an_underscore_is_one_gene() {
    let (names, index) = panel(&["ENSG0001_FOO_BAR", "FOO", "BAR"]);
    assert_eq!(subunit_genes("FOO_BAR", &names, &index), Some(vec![0]));
}

fn programme() -> TargetSets {
    let mut s = TargetSets::default();
    s.add_table(
        "receptor\ttarget\nRECB2_RECC1\tT1\nRECB2_RECC1\tT2\nRECB2_RECC1\tT3\n\
         RECB2_RECC1\tT4\nRECB2_RECC1\tT5\nRECB2_RECC1\tRECB2\n",
        false,
    )
    .unwrap();
    s
}

#[test]
fn pairs_resolve_complex_ligands_and_receptors_and_dedup_on_genes() {
    let (names, index) = panel(&[
        "LIGH1", "LIGH2", "RECB2", "RECC1", "T1", "T2", "T3", "T4", "T5",
    ]);
    let raw: Vec<(Box<str>, Box<str>)> = vec![
        ("LIGH1_LIGH2".into(), "RECB2_RECC1".into()),
        // The same genes spelt otherwise: kept once.
        ("ligh2+ligh1".into(), "RECC1_RECB2".into()),
        // A subunit absent from the data: dropped.
        ("LIGH1".into(), "RECB2_RECC2".into()),
    ];
    let sets = programme();
    let pairs = resolve_pairs(raw, &names, &index, &sets, &HashMap::default());
    assert_eq!(pairs.len(), 1);
    let p = &pairs[0];
    assert_eq!(p.ligand_genes, [0, 1]);
    assert_eq!(p.receptor_genes, [2, 3]);
    // The receptor subunit listed as a target is purged.
    let targets: Vec<usize> = p.targets.iter().map(|t| t.0).collect();
    assert_eq!(targets, [4, 5, 6, 7, 8]);
}

#[test]
fn a_complex_counts_as_its_scarcest_subunit() {
    let cols = vec![vec![3.0, 0.0, 2.0], vec![1.0, 5.0, 4.0]];
    assert_eq!(super::super::scarcest_subunit(&cols), [1.0, 0.0, 2.0]);
}
