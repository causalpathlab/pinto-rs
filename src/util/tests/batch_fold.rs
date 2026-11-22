//! The saved batch fold: written by a fit, read back by predict and impute.

use crate::predict::fold::fitted_cells_fold_divisor;
use crate::util::batch_fold::{
    write_cell_fold_tables, BatchFoldFiles, FoldTableNames, FoldWritten, SavedCellFold,
    CELL_ON_DELTA,
};
use crate::util::common::Mat;
use data_beans::alg::collapse_data::state_aware_batch_fold::StateAwareBatchFold;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|&s| s.into()).collect()
}

/// 2 features, 2 nodes, 2 batches; cell c2 kept δ.
fn fold() -> (StateAwareBatchFold, Mat) {
    let fold = StateAwareBatchFold {
        n_batches: 2,
        n_nodes: 2,
        log_fold: Mat::from_row_slice(2, 4, &[0.1, -0.1, 0.4, -0.4, 0.2, -0.2, -0.3, 0.3]),
        log_delta: Mat::from_row_slice(2, 2, &[0.05, -0.05, 0.0, 0.0]),
        node_of_unit: vec![Some(0), Some(1), None],
    };
    let composition =
        Mat::from_row_slice(2, 2, &[0.9f32.ln(), 0.2f32.ln(), 0.1f32.ln(), 0.8f32.ln()]);
    (fold, composition)
}

fn written(dir: &std::path::Path) -> (BatchFoldFiles, StateAwareBatchFold, Mat) {
    let prefix = dir.join("run");
    let files =
        BatchFoldFiles::written(prefix.to_str().expect("utf8"), FoldWritten::Cells).expect("files");
    let (f, comp) = fold();
    write_cell_fold_tables(
        &files,
        FoldTableNames {
            features: &names(&["g1", "g2"]),
            cells: &names(&["c0", "c1", "c2"]),
            batches: &names(&["B1", "B2"]),
        },
        &f,
        &comp,
    )
    .expect("write");
    (files, f, comp)
}

/// Written and read back: the same features, batches, folds, compositions,
/// and each cell's node, a cell on δ as "delta".
#[test]
fn saved_cell_fold_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (files, f, comp) = written(dir.path());
    let saved = SavedCellFold::read_files(&files.cells, &files.cell_nodes).expect("read");
    assert_eq!(saved.features, names(&["g1", "g2"]));
    assert_eq!(saved.batches, names(&["B1", "B2"]));
    assert_eq!(saved.log_fold, f.log_fold);
    assert_eq!(saved.log_mean_composition, comp);
    assert_eq!(saved.node_of_cell[&Box::<str>::from("c0")], Some(0));
    assert_eq!(saved.node_of_cell[&Box::<str>::from("c1")], Some(1));
    assert_eq!(saved.node_of_cell[&Box::<str>::from("c2")], None);
    let (strs, _) = legume_numeric::matrix::parquet::read_table_columns(
        &files.cell_nodes,
        &["cell", "node"],
        &[],
    )
    .expect("table");
    assert_eq!(strs[1][2].as_ref(), CELL_ON_DELTA);
}

/// A fitted cell is divided by exactly the fold the fit divided it by (its
/// node's column in its batch); a cell the fit left on δ, by its batch's δ.
#[test]
fn fitted_cells_take_the_fold_they_were_fitted_under() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (files, f, _) = written(dir.path());
    let saved = SavedCellFold::read_files(&files.cells, &files.cell_nodes).expect("read");
    let delta = f.log_delta.map(f32::exp);
    let (table, col) = fitted_cells_fold_divisor(
        &saved,
        &names(&["g2", "g1"]),
        &names(&["c0", "c1", "c2"]),
        &names(&["B2", "B1", "B2"]),
        (
            &Mat::from_fn(2, 2, |r, b| delta[(1 - r, b)]),
            &names(&["B1", "B2"]),
        ),
    );
    // rows in the reference's order (g2, g1): c0 at node 0 in B2, c1 at node 1 in B1
    for (cell, (u, b)) in [(0usize, (0usize, 1usize)), (1, (1, 0))] {
        let want = f.unit_log_fold(u, b).map(f32::exp);
        let got = table.column(col[cell].expect("a column"));
        assert!((got[0] - want[1]).abs() < 1e-6 && (got[1] - want[0]).abs() < 1e-6);
    }
    // c2 kept δ: its batch's (B2) δ column
    let got = table.column(col[2].expect("a column"));
    assert!((got[0] - delta[(1, 1)]).abs() < 1e-6 && (got[1] - delta[(0, 1)]).abs() < 1e-6);
}

/// A run whose manifest records no fold (older, or δ alone) reads none, so
/// its readers keep δ.
#[test]
fn a_run_without_a_saved_fold_reads_delta() {
    let run = crate::util::metadata::Run {
        manifest: "nowhere.pinto.json".into(),
        prefix: "nowhere".into(),
        meta: crate::util::metadata::PintoMetadata::default(),
    };
    assert!(SavedCellFold::read(&run).expect("no error").is_none());
}
