use crate::util::batch_effects::strata_of;
use legume_numeric::matrix::parquet::{write_named_table, Column};

#[test]
fn clone_strata_align_to_cells_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.clones.parquet");
    let path = path.to_str().unwrap();
    let cell: Vec<Box<str>> = ["c1@d1", "c2@d1", "c3@d2"].map(Box::from).to_vec();
    let donor: Vec<Box<str>> = ["d1", "d1", "d2"].map(Box::from).to_vec();
    let stratum = [0i32, 1, 2];
    let cols = [
        ("donor".into(), Column::Str(&donor)),
        ("stratum".into(), Column::I32(&stratum)),
    ];
    write_named_table(path, "cell", &cell, &cols).unwrap();
    // A cell the table does not list mixes freely.
    let cells: Vec<Box<str>> = ["c2@d1", "other", "c3@d2", "c1@d1"].map(Box::from).to_vec();
    assert_eq!(strata_of(path, &cells).unwrap(), [1, 0, 2, 0]);
}
