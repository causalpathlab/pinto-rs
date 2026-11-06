//! Response programmes: formats, precedence, merging, and the purge of
//! signalling genes.

use crate::lr_activity::ligand_effect::targets::TargetSets;

fn names(set: &[(Box<str>, f32)]) -> Vec<&str> {
    set.iter().map(|(g, _)| g.as_ref()).collect()
}

#[test]
fn receptor_keyed_table_with_signed_weights() {
    let mut s = TargetSets::default();
    s.add_table(
        "receptor\ttarget\tweight\nRECB1\tGENE2\t0.8\nRECB1\tGENE3\t-0.4\nRECB1\tLIGB1\t0.5\n",
        false,
    )
    .unwrap();
    let (set, removed) = s.for_pair("LIGB1", "RECB1", &["RECB1", "RECB2"]).unwrap();
    assert_eq!(names(&set), ["GENE2", "GENE3"]);
    assert_eq!(set[1].1, -0.4);
    assert_eq!(removed, 1, "the ligand listed as a target is purged");
}

#[test]
fn pair_beats_receptor_beats_ligand() {
    let mut s = TargetSets::default();
    s.add_gmt("LIGB1\tby ligand\tA\tB\nRECB1\tby receptor\tC\nLIGB1_RECB1\tby pair\tD\n");
    s.settle_gmt_names(&[]);
    assert_eq!(names(&s.for_pair("LIGB1", "RECB1", &[]).unwrap().0), ["D"]);
    assert_eq!(
        names(&s.for_pair("LIGB1", "RECB3", &[]).unwrap().0),
        ["A", "B"]
    );
    assert!(s.for_pair("LIGD1", "RECD1", &[]).is_none());
}

#[test]
fn files_merge_and_duplicates_sum() {
    let mut s = TargetSets::default();
    s.add_table("ligand,target,weight\nLIGE1,GENE1,1\n", true)
        .unwrap();
    s.add_table("ligand\ttarget\nLIGE1\tgene1\nLIGE1\tGENE4\n", false)
        .unwrap();
    let (set, _) = s.for_pair("lige1", "RECE1", &[]).unwrap();
    assert_eq!(names(&set), ["GENE1", "GENE4"]);
    assert_eq!(
        set[0].1, 2.0,
        "a target listed twice keeps the sum of its weights"
    );
}

#[test]
fn every_receptor_of_the_ligand_is_purged() {
    let mut s = TargetSets::default();
    s.add_gmt("LIGB1\tx\tRECB2\tRECB3\tGENE5\n");
    let (set, removed) = s.for_pair("LIGB1", "RECB1", &["RECB2", "RECB3"]).unwrap();
    assert_eq!(names(&set), ["GENE5"]);
    assert_eq!(removed, 2);
}

#[test]
fn a_table_needs_a_key_and_a_target() {
    let mut s = TargetSets::default();
    assert!(s.add_table("gene\tweight\nA\t1\n", false).is_err());
    assert!(s.add_table("target\nA\n", false).is_err());
    assert!(s
        .add_table("ligand\ttarget\tweight\nA\tB\tlots\n", false)
        .is_err());
}

#[test]
fn hub_targets_score_low_specificity() {
    let mut s = TargetSets::default();
    s.add_gmt("R1\tx\tGENE6\tA\nR2\tx\tGENE6\tB\nR3\tx\tGENE6\tC\n");
    let spec = s.specificity();
    // GENE6 is in all 3 sets: ln(4/4) = 0; A in one: ln(4/2).
    assert!(spec["GENE6"].abs() < 1e-6);
    assert!((spec["A"] - (4.0f32 / 2.0).ln()).abs() < 1e-6);
}

fn pairs(list: &[(&str, &str)]) -> Vec<(Box<str>, Box<str>)> {
    list.iter()
        .map(|(l, r)| ((*l).into(), (*r).into()))
        .collect()
}

#[test]
fn a_gmt_name_used_whole_as_a_complex_keys_that_receptor() {
    let mut s = TargetSets::default();
    s.add_gmt("RECB1_RECB2\tcomplex\tGENE5\nLIGB1_RECB1_RECB2\tpair\tGENE2\n");
    s.settle_gmt_names(&pairs(&[
        ("LIGB1", "RECB1_RECB2"),
        ("LIGB2", "RECB1_RECB2"),
        ("RECB1", "RECB2"),
    ]));
    // The complex's set keys the receptor, not the pair RECB1-RECB2.
    assert_eq!(
        names(&s.for_pair("LIGB2", "RECB1_RECB2", &[]).unwrap().0),
        ["GENE5"]
    );
    assert!(s.for_pair("RECB1", "RECB2", &[]).is_none());
    // A pair name splits where it names a listed pair.
    assert_eq!(
        names(&s.for_pair("LIGB1", "RECB1_RECB2", &[]).unwrap().0),
        ["GENE2"]
    );
}

#[test]
fn an_unlisted_gmt_name_with_underscore_is_a_ligand_receptor_pair() {
    let mut s = TargetSets::default();
    s.add_gmt("LIGE1_RECE1\tpair\tGENE1\n");
    s.settle_gmt_names(&pairs(&[("LIGB1", "RECB1")]));
    assert_eq!(
        names(&s.for_pair("LIGE1", "RECE1", &[]).unwrap().0),
        ["GENE1"]
    );
}
