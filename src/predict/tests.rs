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
