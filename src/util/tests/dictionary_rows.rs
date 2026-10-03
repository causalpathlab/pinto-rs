//! Marking a mixed-type dictionary's matchable rows by its types table.

use crate::util::dictionary_rows::matchable_rows;
use data_beans::aux::feature_types::write_feature_types;

#[test]
fn rows_are_marked_by_the_types_table_that_lists_them() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let path = format!("{prefix}.feature_embedding.parquet");
    let names: Vec<Box<str>> = ["CD4", "CD4", "chr1:0-5000", "apoptosis"]
        .map(Box::from)
        .to_vec();

    // No types table: every row may match.
    assert_eq!(matchable_rows(&path, &names).unwrap(), [true; 4]);

    let types: Vec<Box<str>> = ["cell_type", "gene", "region", "word"]
        .map(Box::from)
        .to_vec();
    write_feature_types(&prefix, &names, &types).unwrap();
    assert_eq!(
        matchable_rows(&path, &names).unwrap(),
        [false, true, true, false]
    );

    // A table that lists other rows says nothing of this one.
    let other: Vec<Box<str>> = ["CD4", "MYC", "chr1:0-5000", "apoptosis"]
        .map(Box::from)
        .to_vec();
    assert_eq!(matchable_rows(&path, &other).unwrap(), [true; 4]);
}
