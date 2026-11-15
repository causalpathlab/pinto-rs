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
    let on = d.on_rows(&rows);
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
