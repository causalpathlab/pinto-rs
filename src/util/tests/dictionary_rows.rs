//! Marking a mixed-type dictionary's matchable rows by its types table.

use crate::util::dictionary_rows::{matchable_rows, stem};
use data_beans::aux::feature_types::write_feature_types;

#[test]
fn the_stem_is_the_one_module_tables_are_found_by() {
    assert_eq!(stem("/a/v1.2/run.feature_embedding.parquet"), "/a/v1.2/run");
    assert_eq!(stem("/a/v1.2/run.dictionary.parquet"), "/a/v1.2/run");
    assert_eq!(stem("./mydict.parquet"), "./mydict");
    assert_eq!(stem("/a/.cache/x/emb.parquet"), "/a/.cache/x/emb");
}

#[test]
fn rows_are_marked_by_the_types_table_that_lists_them() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let path = format!("{prefix}.feature_embedding.parquet");
    let names: Vec<Box<str>> = ["CD4", "CD4", "chr1:0-5000", "apoptosis"]
        .map(Box::from)
        .to_vec();

    // No types table: every row may match.
    assert_eq!(matchable_rows(&path, &names), [true; 4]);

    let types: Vec<Box<str>> = ["cell_type", "gene", "region", "word"]
        .map(Box::from)
        .to_vec();
    write_feature_types(&prefix, &names, &types).unwrap();
    assert_eq!(matchable_rows(&path, &names), [false, true, true, false]);

    // A table that lists other rows says nothing of this one.
    let other: Vec<Box<str>> = ["CD4", "MYC", "chr1:0-5000", "apoptosis"]
        .map(Box::from)
        .to_vec();
    assert_eq!(matchable_rows(&path, &other), [true; 4]);

    // Nor does one that cannot be read.
    std::fs::write(format!("{prefix}.feature_types.parquet"), b"not parquet").unwrap();
    assert_eq!(matchable_rows(&path, &names), [true; 4]);
}
