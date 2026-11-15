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
/// in full, on the matrix rows and the sorted batch labels; and each cell's
/// batch is written beside it.
#[test]
fn the_delta_file_reads_back_as_the_estimate() {
    use super::ignore_batch::{preprocess, two_batches};
    use crate::util::batch_effects::{delta_cells_path, read_cell_batches, read_delta};
    let dir = tempfile::tempdir().unwrap();
    let files = two_batches(dir.path(), 60);
    let out = dir.path().join("run").to_str().unwrap().to_string();
    let mut pre = preprocess(&files, &out, &[]);
    let estimate = pre
        .batch_effects
        .take()
        .expect("two files, two batches")
        .into_delta()
        .unwrap();
    let back = read_delta(&format!("{out}.delta.parquet")).unwrap();
    assert_eq!(back.rows, pre.data_vec.row_names().unwrap());
    assert_eq!(back.batches, pre.data_vec.batch_names().unwrap());
    assert_eq!(
        (&estimate.rows, &estimate.batches),
        (&back.rows, &pre.batch_labels)
    );
    for (a, b) in back.planes().into_iter().zip(estimate.planes()) {
        assert!((a - b).abs().max() < 1e-6);
    }
    assert!(estimate.sd.max() > 0.0, "the spread is carried");

    let cells = read_cell_batches(&delta_cells_path(&out)).unwrap();
    let names = pre.data_vec.column_names().unwrap();
    assert_eq!(cells.len(), names.len());
    for (cell, batch) in names.iter().zip(&pre.batch_membership) {
        assert_eq!(&cells[cell], batch, "{cell}");
    }
}

/// Asked not to write, preprocessing still hands back the posterior in
/// full and writes neither table: `predict` writes the δ it divides by,
/// once.
#[test]
fn an_estimate_not_to_be_written_is_only_returned() {
    use super::ignore_batch::{preprocess_with, two_batches};
    use crate::util::batch_effects::delta_cells_path;
    use crate::util::srt_pipeline::BatchEstimate;
    let dir = tempfile::tempdir().unwrap();
    let files = two_batches(dir.path(), 60);
    let out = dir.path().join("run").to_str().unwrap().to_string();
    let pre = preprocess_with(
        &files,
        &out,
        &[],
        BatchEstimate::Query {
            all_known: false,
            base_corrected: false,
        },
    );
    let estimate = pre
        .batch_effects
        .expect("two files, two batches")
        .into_delta()
        .unwrap();
    assert_eq!(estimate.batches, pre.batch_labels);
    assert!(estimate.sd.max() > 0.0, "the spread is carried");
    assert!(!std::path::Path::new(&format!("{out}.delta.parquet")).exists());
    assert!(!std::path::Path::new(&delta_cells_path(&out)).exists());
}

/// With every batch known (no estimate, the projection still corrected),
/// nothing is estimated or written.
#[test]
fn nothing_is_estimated_when_every_batch_is_known() {
    use super::ignore_batch::{preprocess_with, two_batches};
    use crate::util::srt_pipeline::BatchEstimate;
    let dir = tempfile::tempdir().unwrap();
    let files = two_batches(dir.path(), 60);
    let out = dir.path().join("run").to_str().unwrap().to_string();
    let pre = preprocess_with(
        &files,
        &out,
        &[],
        BatchEstimate::Query {
            all_known: true,
            base_corrected: false,
        },
    );
    assert!(pre.batch_effects.is_none());
    assert_eq!(pre.batch_labels.len(), 2);
    assert!(!std::path::Path::new(&format!("{out}.delta.parquet")).exists());
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

/// A run's δ is the file its manifest names: an absolute path as is, a
/// relative one beside the manifest, and a record of the fit's own
/// `{prefix}.delta.parquet` also the run's own file now; never one from
/// the working directory (a decoy there is another run's). A
/// `delta.parquet` left from an earlier fit under the same prefix is not
/// read when the manifest names none, nor for a recorded file that is gone,
/// which is an error. Without a manifest the file is taken as it is.
#[test]
fn a_runs_delta_is_the_one_its_manifest_names() {
    use crate::util::batch_effects::run_delta_path;
    use crate::util::metadata::{manifest_path, PintoMetadata, Run};
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_str().unwrap().to_string();
    let file = format!("{prefix}.delta.parquet");
    let found = |run: Option<&Run>| run_delta_path(&prefix, run).unwrap();
    assert_eq!(found(None), None);
    std::fs::write(&file, b"").unwrap();
    assert_eq!(found(None).as_deref(), Some(file.as_str()));

    let manifest = manifest_path(&prefix);
    let open = |meta: &PintoMetadata| {
        meta.write(std::path::Path::new(&manifest)).unwrap();
        Run::open(&prefix).unwrap()
    };
    let mut meta = PintoMetadata::default();
    assert_eq!(found(Some(&open(&meta))), None, "a stale file was read");
    meta.outputs.batch_effects = Some(file.clone());
    assert_eq!(
        found(Some(&open(&meta))).as_deref(),
        Some(file.as_str()),
        "an absolute path as recorded"
    );

    // recorded elsewhere: that file, not the one under the prefix
    let elsewhere = dir.path().join("other.delta.parquet");
    std::fs::write(&elsewhere, b"").unwrap();
    meta.outputs.batch_effects = Some("other.delta.parquet".into());
    assert_eq!(
        found(Some(&open(&meta))).map(std::path::PathBuf::from),
        Some(elsewhere),
        "the recorded file, beside the manifest"
    );

    // recorded elsewhere and gone: an error, though a stale file is under
    // the prefix
    meta.outputs.batch_effects = Some("kept/run.delta.parquet".into());
    assert!(
        run_delta_path(&prefix, Some(&open(&meta))).is_err(),
        "a stale file under the prefix was taken for a recorded one that is gone"
    );

    // the fit's own file, recorded relative to the directory it was fitted
    // from (not beside the manifest), and a file of that name there from
    // the working directory: another run's, never read; the run's own is
    let decoy = CwdDecoy::new(dir.path(), "delta.parquet");
    meta.prefix = decoy.stem.clone();
    meta.outputs.batch_effects = Some(decoy.name.clone());
    let run = open(&meta);
    assert_eq!(
        found(Some(&run)).as_deref(),
        Some(file.as_str()),
        "the working directory's file was read, not the run's own"
    );
    std::fs::remove_file(&file).unwrap();
    assert!(
        run_delta_path(&prefix, Some(&run)).is_err(),
        "with the run's own file gone, the working directory's was read"
    );

    // recorded but nowhere: an error, not a run without a δ
    meta.outputs.batch_effects = Some("moved/away.delta.parquet".into());
    let err = run_delta_path(&prefix, Some(&open(&meta))).unwrap_err();
    assert!(err.to_string().contains("moved/away"), "{err}");
}

/// A file `{stem}.{suffix}` in the working directory under a stem no other
/// test uses, removed when dropped.
pub(crate) struct CwdDecoy {
    pub stem: String,
    pub name: String,
}

impl CwdDecoy {
    pub(crate) fn new(unique: &std::path::Path, suffix: &str) -> Self {
        let tag = unique.file_name().unwrap().to_string_lossy();
        let stem = format!("decoy-{tag}");
        let name = format!("{stem}.{suffix}");
        assert!(!std::path::Path::new(&name).exists());
        std::fs::write(&name, b"").unwrap();
        Self { stem, name }
    }
}

impl Drop for CwdDecoy {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.name);
    }
}

/// An estimate's mean must be on its rows and batches; only a spread plane
/// the estimate left unallocated is filled (no spread, log 0).
#[test]
fn a_posterior_mean_off_its_rows_and_batches_is_an_error() {
    use crate::util::batch_effects::BatchPosterior;
    use legume_numeric::param::dmatrix_gamma::GammaMatrix;
    use legume_numeric::param::traits::TwoStatParam;
    let names = |v: &[&str]| v.iter().map(|&s| Box::from(s)).collect::<Vec<Box<str>>>();
    let posterior = |rows: &[&str]| BatchPosterior {
        // the mean allocated, the spread planes not
        gamma: GammaMatrix::new((2, 2), 1.0, 1.0),
        rows: names(rows),
        batches: names(&["a", "b"]),
    };
    let d = posterior(&["GENE1", "GENE2"]).into_delta().unwrap();
    assert_eq!(d.mean.shape(), (2, 2));
    for p in [&d.sd, &d.log_mean, &d.log_sd] {
        assert_eq!(p.shape(), (2, 2));
        assert!(p.iter().all(|&v| v == 0.0));
    }
    assert_eq!(
        posterior(&["GENE1", "GENE2"]).into_mean().unwrap().shape(),
        (2, 2)
    );
    let off = ["GENE1", "GENE2", "GENE3"];
    assert!(posterior(&off).into_delta().is_err(), "a mean was filled");
    assert!(
        posterior(&off).into_mean().is_err(),
        "a mean was taken off its rows"
    );
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

/// Preprocessing estimates batch effects when asked to, never under
/// `--ignore-batch`, and without an estimate corrects the projection by
/// batch only when told every batch is known and there are two or more.
#[test]
fn batches_are_estimated_only_when_asked() {
    use crate::util::srt_pipeline::{batch_plan, BatchEstimate};
    assert_eq!(batch_plan(BatchEstimate::Fit, false, 2), (true, false));
    assert_eq!(batch_plan(BatchEstimate::Fit, true, 2), (false, false));
    assert_eq!(batch_plan(BatchEstimate::Skip, false, 2), (false, false));
    let new = BatchEstimate::Query {
        all_known: false,
        base_corrected: false,
    };
    assert_eq!(batch_plan(new, false, 2), (true, false));
    assert_eq!(batch_plan(new, true, 2), (false, false));
    let known = BatchEstimate::Query {
        all_known: true,
        base_corrected: false,
    };
    assert_eq!(batch_plan(known, false, 2), (false, true));
    assert_eq!(batch_plan(known, false, 1), (false, false));
    assert_eq!(batch_plan(known, true, 2), (false, false));
}

/// Each cell's batch is the table the manifest records, found as the δ is
/// (beside the manifest, else the run's own, never from the working
/// directory); a run that records none has none (its coord_pairs carry the
/// labels), and one that records a table that is nowhere is an error.
#[test]
fn a_runs_cell_batches_are_the_table_its_manifest_names() {
    use crate::util::batch_effects::{delta_cells_path, run_delta_cells_path};
    use crate::util::metadata::{manifest_path, PintoMetadata, Run};
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_str().unwrap().to_string();
    let own = delta_cells_path(&prefix);
    std::fs::write(&own, b"").unwrap();
    let open = |meta: &PintoMetadata| {
        meta.write(std::path::Path::new(&manifest_path(&prefix)))
            .unwrap();
        Run::open(&prefix).unwrap()
    };
    let mut meta = PintoMetadata::default();
    assert_eq!(run_delta_cells_path(&open(&meta)).unwrap(), None);

    meta.outputs.delta_cells = Some("run.delta_cells.parquet".into());
    assert_eq!(
        run_delta_cells_path(&open(&meta)).unwrap().as_deref(),
        Some(own.as_str()),
        "beside the manifest"
    );

    meta.outputs.delta_cells = Some("kept/run.delta_cells.parquet".into());
    assert!(
        run_delta_cells_path(&open(&meta)).is_err(),
        "a stale table under the prefix was taken for a recorded one that is gone"
    );

    let decoy = CwdDecoy::new(dir.path(), "delta_cells.parquet");
    meta.prefix = decoy.stem.clone();
    meta.outputs.delta_cells = Some(decoy.name.clone());
    let run = open(&meta);
    assert_eq!(
        run_delta_cells_path(&run).unwrap().as_deref(),
        Some(own.as_str()),
        "the working directory's table was read, not the run's own"
    );
    std::fs::remove_file(&own).unwrap();
    assert!(
        run_delta_cells_path(&run).is_err(),
        "a recorded table that is gone was taken from the working directory or as none"
    );
}
