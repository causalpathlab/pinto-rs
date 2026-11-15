//! The model's pair encoder is found from the model prefix alone, so the
//! resolution is checked without a model.

use super::{cage_only_flags, default_args, pair_encoder_path};

#[test]
fn the_pair_encoder_is_required_and_found_beside_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("run").to_string_lossy().to_string();
    let file = format!("{model}.pair_encoder.safetensors");

    let err = pair_encoder_path(&model).unwrap_err();
    assert!(err.to_string().contains(&file), "{err}");

    std::fs::write(&file, b"").unwrap();
    assert_eq!(pair_encoder_path(&model).unwrap(), file);
}

#[test]
fn cage_only_flags_are_named_when_set() {
    let d = default_args().unwrap();
    assert!(cage_only_flags(&d, &d).is_empty());
    let mut set = default_args().unwrap();
    set.no_init_features = true;
    set.device_no = 1;
    assert_eq!(
        cage_only_flags(&set, &d),
        vec!["--device-no", "--no-init-features"]
    );
}

/// A model that is not there is refused before any data is read, by its
/// path.
#[test]
fn a_model_that_is_not_there_is_refused_by_its_path() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("nope").to_string_lossy().to_string();
    let mut args = default_args().unwrap();
    args.model = model.clone().into();
    args.common.data_files = vec![dir.path().join("d.zarr").to_string_lossy().into()];
    let e = format!("{:#}", super::predict(&args).unwrap_err());
    assert!(e.contains(&format!("{model}.pinto.json")), "{e}");
    assert!(e.contains("not there"), "{e}");
}

mod batch_effects {
    use super::super::delta::{combine, match_batches, QueryLabels};
    use crate::test_support::boxed;
    use crate::util::common::Mat;

    /// Named labels match by name. Per-file labels match a query file only
    /// when it is one of the model's own files (the model's labels are
    /// file positions too); a new file at a known position is new.
    /// Components never match.
    #[test]
    fn query_batches_are_matched_by_name_or_by_file() {
        let model = boxed(&["0", "1"]);
        let q = boxed(&["1", "x"]);
        assert_eq!(
            match_batches(&q, &model, None, &QueryLabels::Named),
            vec![Some(1), None]
        );
        let dir = tempfile::tempdir().unwrap();
        let file = |n: &str| {
            let p = dir.path().join(n);
            std::fs::write(&p, b"").unwrap();
            p.to_str().unwrap().to_string()
        };
        let (a, b, c) = (file("a.zarr"), file("b.zarr"), file("c.zarr"));
        let model_files = vec![a.clone(), b.clone()];
        let query_files = boxed(&[&c, &a]);
        let how = QueryLabels::Files {
            query_files: &query_files,
            model_files: &model_files,
        };
        // query "0" is c (new), query "1" is a (the model's "0")
        assert_eq!(
            match_batches(&boxed(&["0", "1"]), &model, None, &how),
            vec![None, Some(0)]
        );
        // a model whose labels are not file positions shares none by file,
        // even labels that look like positions
        for labels in [&["d1", "d2"][..], &["0", "1", "2"][..]] {
            assert_eq!(
                match_batches(&boxed(&["0", "1"]), &boxed(labels), None, &how),
                vec![None, None]
            );
        }
        assert_eq!(
            match_batches(&q, &model, None, &QueryLabels::Components),
            vec![None, None]
        );
    }

    /// A manifest that records how the model's batches were named is
    /// matched on that, not on how the labels look: a model labelled by
    /// file never matches `--batch-files` names, and one labelled by name
    /// never matches by file, even where the labels read as positions.
    #[test]
    fn batches_match_only_labels_of_the_models_kind() {
        use crate::util::metadata::BatchLabels;
        let model = boxed(&["0", "1"]);
        let q = boxed(&["0", "1"]);
        let named = QueryLabels::Named;
        assert_eq!(
            match_batches(&q, &model, Some(BatchLabels::Files), &named),
            vec![None, None]
        );
        assert_eq!(
            match_batches(&q, &model, Some(BatchLabels::BatchFiles), &named),
            vec![Some(0), Some(1)]
        );
        let dir = tempfile::tempdir().unwrap();
        let file = |n: &str| {
            let p = dir.path().join(n);
            std::fs::write(&p, b"").unwrap();
            p.to_str().unwrap().to_string()
        };
        let model_files = vec![file("a.zarr"), file("b.zarr")];
        let query_files: Vec<Box<str>> = model_files.iter().map(|f| f.as_str().into()).collect();
        let how = QueryLabels::Files {
            query_files: &query_files,
            model_files: &model_files,
        };
        assert_eq!(
            match_batches(&q, &model, Some(BatchLabels::BatchFiles), &how),
            vec![None, None]
        );
        assert_eq!(
            match_batches(&q, &model, Some(BatchLabels::Files), &how),
            vec![Some(0), Some(1)]
        );
    }

    /// A known batch takes the model's column; a new one the query's
    /// estimate times the model-over-estimate ratio of the shared batches,
    /// its spread scaled alike; a row the model lacks keeps the estimate.
    /// Every plane of the fits' table is filled.
    #[test]
    fn a_new_batch_is_carried_into_the_model_frame() {
        use crate::util::batch_effects::RunDelta;
        let planes = |mean: &[f32], sd: f32| RunDelta {
            rows: boxed(&["r0", "r1"]),
            batches: boxed(&["b0", "b1"]),
            mean: Mat::from_row_slice(2, 2, mean),
            sd: Mat::from_element(2, 2, sd),
            log_mean: Mat::from_row_slice(2, 2, mean).map(f32::ln),
            log_sd: Mat::from_element(2, 2, 0.1),
        };
        // model: rows x batches (m0, m1)
        let model = planes(&[0.5, 2.0, 1.0, 1.0], 0.2);
        // query batches (q0 = m1, q1 new); its own estimate pinned over q0, q1
        let estimate = planes(&[0.5, 2.0, 3.0, 1.0 / 3.0], 0.01);
        let matched = [Some(1), None];
        let q = boxed(&["q0", "q1"]);
        let d = combine(&model, &[true, false], Some(&estimate), &matched, &q);
        assert_eq!(d.batches, q);
        assert_eq!(d.mean[(0, 0)], 2.0, "known batch: the model's column");
        assert_eq!(d.sd[(0, 0)], 0.2, "known batch: the model's spread");
        // new: 2.0 (estimate) x 2.0 / 0.5 (model over estimate on q0)
        assert!((d.mean[(0, 1)] - 8.0).abs() < 1e-5, "{}", d.mean[(0, 1)]);
        assert!((d.sd[(0, 1)] - 0.04).abs() < 1e-6, "{}", d.sd[(0, 1)]);
        assert!((d.log_mean[(0, 1)] - 8f32.ln()).abs() < 1e-5);
        assert_eq!(d.log_sd[(0, 1)], 0.1);
        assert_eq!(
            (d.mean[(1, 0)], d.mean[(1, 1)]),
            (3.0, 1.0 / 3.0),
            "no model row"
        );
    }
}

/// The query is divided only by a model's δ: a model without one (one
/// batch, `--ignore-batch`, or an older cage-cell fit) trained its encoder
/// on undivided counts; `--ignore-batch` on predict leaves it unused.
#[test]
fn the_query_is_divided_only_when_the_model_has_a_delta() {
    use crate::util::batch_effects::RunDelta;
    use crate::util::common::Mat;
    use crate::util::metadata::{manifest_path, PintoMetadata};
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("run").to_string_lossy().to_string();
    assert!(super::model_delta(&model, false).unwrap().is_none());

    let file = format!("{model}.delta.parquet");
    RunDelta {
        rows: vec!["GENE1".into()],
        batches: vec!["0".into(), "1".into()],
        mean: Mat::from_row_slice(1, 2, &[0.5, 2.0]),
        sd: Mat::zeros(1, 2),
        log_mean: Mat::zeros(1, 2),
        log_sd: Mat::zeros(1, 2),
    }
    .write(&file)
    .unwrap();
    let mut meta = PintoMetadata::default();
    meta.write(std::path::Path::new(&manifest_path(&model)))
        .unwrap();
    assert!(
        super::model_delta(&model, false).unwrap().is_none(),
        "a delta the manifest does not name"
    );
    meta.outputs.batch_effects = Some(file);
    meta.write(std::path::Path::new(&manifest_path(&model)))
        .unwrap();
    let (d, _) = super::model_delta(&model, false).unwrap().unwrap();
    assert_eq!(d.mean[(0, 1)], 2.0);
    assert!(super::model_delta(&model, true).unwrap().is_none());
}

/// A manifest's data files: canonical ones as recorded; an older relative
/// entry beside the manifest, whatever the working directory.
#[test]
fn model_data_files_resolve_beside_the_manifest() {
    use crate::util::metadata::{manifest_path, PintoMetadata};
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("run").to_string_lossy().to_string();
    std::fs::write(dir.path().join("a.zarr"), b"").unwrap();
    let meta = PintoMetadata {
        data_files: Some(vec!["a.zarr".into(), "gone.zarr".into()]),
        ..Default::default()
    };
    meta.write(std::path::Path::new(&manifest_path(&model)))
        .unwrap();
    let files = super::model_data_files(&model);
    assert_eq!(
        std::path::Path::new(&files[0]),
        dir.path().join("a.zarr"),
        "a relative entry beside the manifest"
    );
    assert_eq!(files[1], "gone.zarr", "kept as written");
}
