use crate::util::common::*;
use crate::view::markers::*;

#[test]
fn markers_rank_by_fold_over_other_communities() {
    // g0 is everywhere, g1 marks community 1, g2 marks it more weakly,
    // g3 is specific to 1 but too faint to count. The median rate in
    // community 1 is 6, so g0 (5) falls below the floor too.
    let names: Vec<Box<str>> = ["g0", "g1", "g2", "g3"].map(Box::from).to_vec();
    let rates = Mat::from_row_slice(
        4,
        3,
        &[
            5., 5., 5., //
            1., 9., 1., //
            2., 6., 2., //
            0., 0.01, 0., //
        ],
    );
    let fr = FeatureRates { names, rates };
    let top = fr.top(1, 3);
    let names: Vec<&str> = top.iter().map(|m| m.name.as_ref()).collect();
    assert_eq!(names, ["g1", "g2"]);
    // 9 / 1, softened by the pseudocount (1% of the median rate).
    assert!((8.0..9.0).contains(&top[0].fold));
    assert!(fr.top(7, 3).is_empty());
}

#[test]
fn symbols_drop_the_ensembl_id() {
    assert_eq!(symbol("ENSG00000116824_CD2"), "CD2");
    assert_eq!(symbol("CD45RA"), "CD45RA");
    assert_eq!(symbol("HLA_DRA"), "HLA_DRA");
}
