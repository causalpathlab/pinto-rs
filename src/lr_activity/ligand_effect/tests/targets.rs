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
        "receptor\ttarget\tweight\nRECE1\tTGTC1\t0.8\nRECE1\tTGTE1\t-0.4\nRECE1\tLIGB1\t0.5\n",
        false,
    )
    .unwrap();
    let (set, removed) = s.for_pair("LIGB1", "RECE1", &["RECE1", "RECE2"]).unwrap();
    assert_eq!(names(&set), ["TGTC1", "TGTE1"]);
    assert_eq!(set[1].1, -0.4);
    assert_eq!(removed, 1, "the ligand listed as a target is purged");
}

#[test]
fn pair_beats_receptor_beats_ligand() {
    let mut s = TargetSets::default();
    s.add_gmt("LIGB1\tby ligand\tA\tB\nRECE1\tby receptor\tC\nLIGB1_RECE1\tby pair\tD\n");
    s.settle_gmt_names(&[]);
    assert_eq!(names(&s.for_pair("LIGB1", "RECE1", &[]).unwrap().0), ["D"]);
    assert_eq!(
        names(&s.for_pair("LIGB1", "RECB1", &[]).unwrap().0),
        ["A", "B"]
    );
    assert!(s.for_pair("LIGE1", "RECG1", &[]).is_none());
}

#[test]
fn files_merge_and_duplicates_sum() {
    let mut s = TargetSets::default();
    s.add_table("ligand,target,weight\nLIGD1,TGTA1,1\n", true)
        .unwrap();
    s.add_table("ligand\ttarget\nLIGD1\ttgta1\nLIGD1\tTGTF1\n", false)
        .unwrap();
    let (set, _) = s.for_pair("ligd1", "RECF1", &[]).unwrap();
    assert_eq!(names(&set), ["TGTA1", "TGTF1"]);
    assert_eq!(
        set[0].1, 2.0,
        "a target listed twice keeps the sum of its weights"
    );
}

#[test]
fn every_receptor_of_the_ligand_is_purged() {
    let mut s = TargetSets::default();
    s.add_gmt("LIGB1\tx\tRECE2\tRECB1\tTGTB1\n");
    let (set, removed) = s.for_pair("LIGB1", "RECE1", &["RECE2", "RECB1"]).unwrap();
    assert_eq!(names(&set), ["TGTB1"]);
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
    s.add_gmt("R1\tx\tTGTD1\tA\nR2\tx\tTGTD1\tB\nR3\tx\tTGTD1\tC\n");
    let spec = s.specificity();
    // TGTD1 is in all 3 sets: ln(4/4) = 0; A in one: ln(4/2).
    assert!(spec["TGTD1"].abs() < 1e-6);
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
    s.add_gmt("RECE1_RECE2\tcomplex\tTGTB1\nLIGB1_RECE1_RECE2\tpair\tTGTC1\n");
    s.settle_gmt_names(&pairs(&[
        ("LIGB1", "RECE1_RECE2"),
        ("LIGB2", "RECE1_RECE2"),
        ("RECE1", "RECE2"),
    ]));
    // The complex's set keys the receptor, not the pair RECE1-RECE2.
    assert_eq!(
        names(&s.for_pair("LIGB2", "RECE1_RECE2", &[]).unwrap().0),
        ["TGTB1"]
    );
    assert!(s.for_pair("RECE1", "RECE2", &[]).is_none());
    // A pair name splits where it names a listed pair.
    assert_eq!(
        names(&s.for_pair("LIGB1", "RECE1_RECE2", &[]).unwrap().0),
        ["TGTC1"]
    );
}

#[test]
fn an_unlisted_gmt_name_with_underscore_is_a_ligand_receptor_pair() {
    let mut s = TargetSets::default();
    s.add_gmt("LIGD1_RECF1\tpair\tTGTA1\n");
    s.settle_gmt_names(&pairs(&[("LIGB1", "RECE1")]));
    assert_eq!(
        names(&s.for_pair("LIGD1", "RECF1", &[]).unwrap().0),
        ["TGTA1"]
    );
}
