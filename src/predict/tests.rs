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
            match_batches(&q, &model, &QueryLabels::Named),
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
            match_batches(&boxed(&["0", "1"]), &model, &how),
            vec![None, Some(0)]
        );
        // a model whose labels are not file positions shares none by file,
        // even labels that look like positions
        for labels in [&["d1", "d2"][..], &["0", "1", "2"][..]] {
            assert_eq!(
                match_batches(&boxed(&["0", "1"]), &boxed(labels), &how),
                vec![None, None]
            );
        }
        assert_eq!(
            match_batches(&q, &model, &QueryLabels::Components),
            vec![None, None]
        );
    }

    /// A known batch takes the model's column; a new one the query's
    /// estimate times the model-over-estimate ratio of the shared batches;
    /// a row the model lacks keeps the estimate.
    #[test]
    fn a_new_batch_is_carried_into_the_model_frame() {
        // model: rows x batches (m0, m1)
        let model = Mat::from_row_slice(2, 2, &[0.5, 2.0, 1.0, 1.0]);
        // query batches (q0 = m1, q1 new); its own estimate pinned over q0, q1
        let estimate = Mat::from_row_slice(2, 2, &[0.5, 2.0, 3.0, 1.0 / 3.0]);
        let matched = [Some(1), None];
        let d = combine(&model, &[true, false], Some(&estimate), &matched);
        assert_eq!(d[(0, 0)], 2.0, "known batch: the model's column");
        // new: 2.0 (estimate) x 2.0 / 0.5 (model over estimate on q0)
        assert!((d[(0, 1)] - 8.0).abs() < 1e-5, "{}", d[(0, 1)]);
        assert_eq!((d[(1, 0)], d[(1, 1)]), (3.0, 1.0 / 3.0), "no model row");
    }
}

/// The query is corrected for batch only when the model was: a model
/// whose manifest records no batch effects (one batch, `--ignore-batch`,
/// or an older cage-cell fit) trained its encoder on uncorrected counts.
/// A model without a readable manifest keeps the correction.
#[test]
fn the_query_is_batch_corrected_only_when_the_model_was() {
    use crate::util::metadata::{manifest_path, PintoMetadata};
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("run").to_string_lossy().to_string();
    assert!(super::model_batch_corrected(&model));

    let mut meta = PintoMetadata::default();
    meta.write(std::path::Path::new(&manifest_path(&model)))
        .unwrap();
    assert!(!super::model_batch_corrected(&model));

    meta.outputs.batch_effects = Some(format!("{model}.delta.parquet"));
    meta.write(std::path::Path::new(&manifest_path(&model)))
        .unwrap();
    assert!(super::model_batch_corrected(&model));
}
