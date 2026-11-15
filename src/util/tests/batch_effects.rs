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

/// What the estimate writes is what the reader gives back: the posterior
/// mean, on the matrix rows and the sorted batch labels.
#[test]
fn the_delta_file_reads_back_as_the_estimate() {
    use super::ignore_batch::{preprocess, two_batches};
    use crate::util::batch_effects::read_delta;
    let dir = tempfile::tempdir().unwrap();
    let files = two_batches(dir.path(), 60);
    let out = dir.path().join("run").to_str().unwrap().to_string();
    let pre = preprocess(&files, &out, &[]);
    let estimate = pre.batch_effects.expect("two files, two batches");
    let back = read_delta(&format!("{out}.delta.parquet")).unwrap();
    assert_eq!(back.rows, pre.data_vec.row_names().unwrap());
    assert_eq!(back.batches, pre.data_vec.batch_names().unwrap());
    assert!((back.mean - estimate).abs().max() < 1e-6);
}

/// Rows are matched by name, whatever order the file lists them in; a row
/// the file lacks has no match.
#[test]
fn delta_rows_are_matched_by_name() {
    use crate::util::batch_effects::read_delta;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.delta.parquet");
    let path = path.to_str().unwrap();
    let feature: Vec<Box<str>> = ["GENE2", "GENE1", "GENE2", "GENE1"].map(Box::from).to_vec();
    let batch: Vec<Box<str>> = ["b", "b", "a", "a"].map(Box::from).to_vec();
    let mean = [4.0f32, 2.0, 0.25, 0.5];
    let cols = [
        ("batch".into(), Column::Str(&batch)),
        ("mean".into(), Column::F32(&mean)),
    ];
    write_named_table(path, "feature", &feature, &cols).unwrap();
    let d = read_delta(path).unwrap();
    assert_eq!(d.batches, ["a", "b"].map(Box::from).to_vec());
    let at = |g: &str, b: usize| d.mean[(d.rows.iter().position(|r| r.as_ref() == g).unwrap(), b)];
    assert_eq!((at("GENE1", 0), at("GENE1", 1)), (0.5, 2.0));
    assert_eq!((at("GENE2", 0), at("GENE2", 1)), (0.25, 4.0));
    let rows: Vec<Box<str>> = ["GENE1", "GENE3", "GENE2"].map(Box::from).to_vec();
    let (on, read) = d.on_rows(&rows);
    assert_eq!(read, [true, false, true]);
    let on = on.mean;
    assert_eq!(on.nrows(), 3);
    assert_eq!((on[(0, 1)], on[(2, 0)]), (2.0, 0.25));
    assert_eq!(
        (on[(1, 0)], on[(1, 1)]),
        (1.0, 1.0),
        "a row the run lacks is left as is"
    );
}

/// A run's δ is the file its manifest names; a `delta.parquet` left from an
/// earlier fit under the same prefix is not read when the manifest names
/// none. Without a manifest the file is taken as it is.
#[test]
fn a_runs_delta_is_the_one_its_manifest_names() {
    use crate::util::batch_effects::run_delta_path;
    use crate::util::metadata::{manifest_path, PintoMetadata};
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_str().unwrap().to_string();
    let file = format!("{prefix}.delta.parquet");
    assert_eq!(run_delta_path(&prefix), None);
    std::fs::write(&file, b"").unwrap();
    assert_eq!(run_delta_path(&prefix).as_deref(), Some(file.as_str()));
    let mut meta = PintoMetadata::default();
    let manifest = manifest_path(&prefix);
    meta.write(std::path::Path::new(&manifest)).unwrap();
    assert_eq!(run_delta_path(&prefix), None, "a stale file was read");
    meta.outputs.batch_effects = Some(file.clone());
    meta.write(std::path::Path::new(&manifest)).unwrap();
    assert_eq!(run_delta_path(&prefix).as_deref(), Some(file.as_str()));
}

/// Each count is divided by its row's δ in its column's batch; a column
/// with no batch (or no use) is left as it is, and so is a count whose δ
/// is not positive.
#[test]
fn columns_are_divided_by_their_batchs_delta() {
    use crate::util::batch_effects::divide_columns;
    use crate::util::common::Mat;
    use nalgebra_sparse::{CooMatrix, CscMatrix};
    // rows x cells: cell 0 in batch 1, cell 1 skipped, cell 2 in batch 0
    let mut coo = CooMatrix::new(2, 3);
    for (r, c, v) in [(0, 0, 4.0f32), (1, 0, 6.0), (0, 1, 5.0), (1, 2, 9.0)] {
        coo.push(r, c, v);
    }
    let mut x = CscMatrix::from(&coo);
    // rows x batches
    let delta = Mat::from_row_slice(2, 2, &[2.0, 2.0, 3.0, 0.0]);
    let batch = [Some(1), None, Some(0)];
    divide_columns(&mut x, &delta, |j| batch[j]);
    let at = |r: usize, c: usize| x.get_entry(r, c).unwrap().into_value();
    assert_eq!((at(0, 0), at(1, 0)), (2.0, 6.0), "δ 0 leaves the count");
    assert_eq!(at(0, 1), 5.0, "a skipped column is left as is");
    assert_eq!(at(1, 2), 3.0);
}

/// A δ written by `RunDelta::write` has the fits' columns and reads back
/// whole, every plane.
#[test]
fn a_written_delta_reads_back_with_its_spread() {
    use crate::util::batch_effects::{read_delta, RunDelta};
    use crate::util::common::Mat;
    use legume_numeric::matrix::parquet::read_table_columns;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.delta.parquet");
    let path = path.to_str().unwrap();
    let names = |v: &[&str]| v.iter().map(|&s| Box::from(s)).collect::<Vec<Box<str>>>();
    let d = RunDelta {
        rows: names(&["GENE1", "GENE2"]),
        batches: names(&["a", "b"]),
        mean: Mat::from_row_slice(2, 2, &[0.5, 2.0, 1.5, 0.75]),
        sd: Mat::from_row_slice(2, 2, &[0.1, 0.2, 0.3, 0.4]),
        log_mean: Mat::from_row_slice(2, 2, &[-0.7, 0.7, 0.4, -0.3]),
        log_sd: Mat::from_row_slice(2, 2, &[0.01, 0.02, 0.03, 0.04]),
    };
    d.write(path).unwrap();
    read_table_columns(
        path,
        &["feature", "batch"],
        &["mean", "sd", "log_mean", "log_sd"],
    )
    .expect("the fits' columns");
    let back = read_delta(path).unwrap();
    assert_eq!((back.rows, back.batches), (d.rows, d.batches));
    for (a, b) in [
        (&back.mean, &d.mean),
        (&back.sd, &d.sd),
        (&back.log_mean, &d.log_mean),
        (&back.log_sd, &d.log_sd),
    ] {
        assert!((a - b).abs().max() < 1e-6);
    }
}

/// Preprocessing estimates batch effects when asked to always, never when
/// told to skip, and, against a fitted model, only when a batch is new;
/// with every one of two batches known it still corrects for them.
#[test]
fn batches_are_estimated_only_when_one_is_new() {
    use crate::util::srt_pipeline::{batch_plan, BatchEstimate};
    let labels: Vec<Box<str>> = vec!["a".into(), "b".into()];
    assert_eq!(batch_plan(&BatchEstimate::Always, &labels), (true, false));
    assert_eq!(batch_plan(&BatchEstimate::Skip, &labels), (false, false));
    let known = |l: &[Box<str>]| l.iter().all(|b| b.as_ref() == "a" || b.as_ref() == "b");
    assert_eq!(
        batch_plan(&BatchEstimate::New(&known), &labels),
        (false, true)
    );
    let with_new: Vec<Box<str>> = vec!["a".into(), "c".into()];
    assert_eq!(
        batch_plan(&BatchEstimate::New(&known), &with_new),
        (true, false)
    );
    let one: Vec<Box<str>> = vec!["a".into()];
    assert_eq!(
        batch_plan(&BatchEstimate::New(&known), &one),
        (false, false)
    );
}
