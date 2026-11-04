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
        "receptor\ttarget\tweight\nTGFBR1\tSERPINE1\t0.8\nTGFBR1\tMYC\t-0.4\nTGFBR1\tTGFB1\t0.5\n",
        false,
    )
    .unwrap();
    let (set, removed) = s
        .for_pair("TGFB1", "TGFBR1", &["TGFBR1", "TGFBR2"])
        .unwrap();
    assert_eq!(names(&set), ["SERPINE1", "MYC"]);
    assert_eq!(set[1].1, -0.4);
    assert_eq!(removed, 1, "the ligand listed as a target is purged");
}

#[test]
fn pair_beats_receptor_beats_ligand() {
    let mut s = TargetSets::default();
    s.add_gmt("TGFB1\tby ligand\tA\tB\nTGFBR1\tby receptor\tC\nTGFB1_TGFBR1\tby pair\tD\n");
    s.settle_gmt_names(&[]);
    assert_eq!(names(&s.for_pair("TGFB1", "TGFBR1", &[]).unwrap().0), ["D"]);
    assert_eq!(
        names(&s.for_pair("TGFB1", "ACVRL1", &[]).unwrap().0),
        ["A", "B"]
    );
    assert!(s.for_pair("WNT5A", "FZD5", &[]).is_none());
}

#[test]
fn files_merge_and_duplicates_sum() {
    let mut s = TargetSets::default();
    s.add_table("ligand,target,weight\nIL6,SOCS3,1\n", true)
        .unwrap();
    s.add_table("ligand\ttarget\nIL6\tsocs3\nIL6\tJUNB\n", false)
        .unwrap();
    let (set, _) = s.for_pair("il6", "IL6R", &[]).unwrap();
    assert_eq!(names(&set), ["SOCS3", "JUNB"]);
    assert_eq!(
        set[0].1, 2.0,
        "a target listed twice keeps the sum of its weights"
    );
}

#[test]
fn every_receptor_of_the_ligand_is_purged() {
    let mut s = TargetSets::default();
    s.add_gmt("TGFB1\tx\tTGFBR2\tACVRL1\tSMAD7\n");
    let (set, removed) = s
        .for_pair("TGFB1", "TGFBR1", &["TGFBR2", "ACVRL1"])
        .unwrap();
    assert_eq!(names(&set), ["SMAD7"]);
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
    s.add_gmt("R1\tx\tFOS\tA\nR2\tx\tFOS\tB\nR3\tx\tFOS\tC\n");
    let spec = s.specificity();
    // FOS is in all 3 sets: ln(4/4) = 0; A in one: ln(4/2).
    assert!(spec["FOS"].abs() < 1e-6);
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
    s.add_gmt("TGFBR1_TGFBR2\tcomplex\tSMAD7\nTGFB1_TGFBR1_TGFBR2\tpair\tSERPINE1\n");
    s.settle_gmt_names(&pairs(&[
        ("TGFB1", "TGFBR1_TGFBR2"),
        ("TGFB2", "TGFBR1_TGFBR2"),
        ("TGFBR1", "TGFBR2"),
    ]));
    // The complex's set keys the receptor, not the pair TGFBR1-TGFBR2.
    assert_eq!(
        names(&s.for_pair("TGFB2", "TGFBR1_TGFBR2", &[]).unwrap().0),
        ["SMAD7"]
    );
    assert!(s.for_pair("TGFBR1", "TGFBR2", &[]).is_none());
    // A pair name splits where it names a listed pair.
    assert_eq!(
        names(&s.for_pair("TGFB1", "TGFBR1_TGFBR2", &[]).unwrap().0),
        ["SERPINE1"]
    );
}

#[test]
fn an_unlisted_gmt_name_with_underscore_is_a_ligand_receptor_pair() {
    let mut s = TargetSets::default();
    s.add_gmt("IL6_IL6R\tpair\tSOCS3\n");
    s.settle_gmt_names(&pairs(&[("TGFB1", "TGFBR1")]));
    assert_eq!(names(&s.for_pair("IL6", "IL6R", &[]).unwrap().0), ["SOCS3"]);
}
