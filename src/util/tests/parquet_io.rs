//! The column-wise readers against the row-wise ones they replace.

use crate::util::common::*;
use crate::util::parquet_io::read_labelled_matrix;

#[test]
fn labelled_matrix_matches_the_row_reader() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.propensity.parquet");
    let n = 70_000; // several record batches
    let names: Vec<Box<str>> = (0..n).map(|i| format!("cell-{i}").into()).collect();
    let cols: Vec<Box<str>> = ["C0", "C1", "cluster", "entropy"].map(Box::from).to_vec();
    let mat = Mat::from_fn(n, 4, |i, j| ((i * 7 + j * 13) % 101) as f32 / 101.);
    mat.to_parquet_with_names(
        path.to_str().unwrap(),
        (Some(&names), Some("cell")),
        Some(&cols),
    )
    .unwrap();

    let rows = Mat::from_parquet(path.to_str().unwrap()).unwrap();
    let cols_wise = read_labelled_matrix(&path).unwrap();
    assert_eq!(cols_wise.rows, rows.rows);
    assert_eq!(cols_wise.cols, rows.cols);
    assert_eq!(cols_wise.mat, rows.mat);
}
