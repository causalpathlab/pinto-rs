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

#[test]
fn cells_table_round_trips_every_cell() {
    use crate::util::parquet_io::{cells_table_path, read_cells_table, write_cells_table};
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().to_string();
    let names: Vec<Box<str>> = ["a", "b", "c"].map(Box::from).to_vec();
    // x, y, and the internal batch offset column, which is not written.
    let coords = Mat::from_row_slice(3, 3, &[1., 10., 0., 2., 20., 0., 3., 30., 100.]);
    let coord_names: Vec<Box<str>> = ["x_um", "y_um", "batch"].map(Box::from).to_vec();
    let batches: Vec<Box<str>> = ["s1", "s1", "s2"].map(Box::from).to_vec();
    write_cells_table(
        &prefix,
        &names,
        &coords,
        &coord_names,
        &batches,
        &[true, false, true],
    )
    .unwrap();

    let path = std::path::PathBuf::from(cells_table_path(&prefix));
    let cols =
        legume_numeric::matrix::parquet::peek_parquet_field_names(path.to_str().unwrap()).unwrap();
    let cols: Vec<&str> = cols.iter().map(|c| c.as_ref()).collect();
    assert_eq!(cols, ["cell", "x_um", "y_um", "batch", "in_graph"]);

    let cells = read_cells_table(&path, None).unwrap();
    assert_eq!(cells.names, names);
    assert_eq!(cells.coords, vec![(1., 10.), (2., 20.), (3., 30.)]);
    assert_eq!(cells.batches.unwrap(), batches);
    assert_eq!(cells.index["c"], 2);
    assert_eq!(cells.in_graph, Some(vec![true, false, true]));
    assert_eq!(
        cells.coord_col_names,
        ["x_um", "y_um"].map(Box::from).to_vec()
    );
}
