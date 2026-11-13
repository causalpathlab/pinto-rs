use crate::simulate::run::{run_simulate, SimulateArgs};
use data_beans::sparse_io::open_sparse_matrix_by_path;

fn args(out: &str, scenario: &str, extra: &[&str]) -> SimulateArgs {
    let mut set: Vec<String> = ["grid_side=12", "n_genes=60", "programme_genes=8", "seed=3"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    set.extend(extra.iter().map(|s| s.to_string()));
    SimulateArgs {
        scenario: scenario.into(),
        set,
        coords: Vec::new(),
        coord_column_names: vec!["row_um".into(), "col_um".into()],
        out: out.into(),
    }
}

/// A run writes one readable backend and coordinate file per section, the
/// truth tables, and a manifest holding every parameter.
#[test]
fn a_run_writes_backends_truth_and_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "L2", &[])).unwrap();
    for s in 1..=6 {
        let m = open_sparse_matrix_by_path(&format!("{out}/sections/S{s}.zarr")).unwrap();
        assert_eq!((m.num_rows(), m.num_columns()), (Some(60), Some(144)));
        assert!(std::path::Path::new(&format!("{out}/sections/S{s}.coords.parquet")).exists());
    }
    for t in ["spots", "domains", "states", "genes"] {
        assert!(
            std::path::Path::new(&format!("{out}/truth/{t}.parquet")).exists(),
            "{t}"
        );
    }
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(m["params"]["n_genes"], 60);
    assert_eq!(m["params"]["patch_kind"], "new-state");
    assert_eq!(m["sections"].as_array().unwrap().len(), 6);
    assert!(m["realised_pve_programme_genes"]["type"].as_f64().unwrap() > 0.0);
}

/// The same seed gives the same counts.
#[test]
fn a_run_is_reproducible() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a").to_string_lossy().to_string();
    let b = dir.path().join("b").to_string_lossy().to_string();
    run_simulate(&args(&a, "L1", &[])).unwrap();
    run_simulate(&args(&b, "L1", &[])).unwrap();
    let read = |o: &str| {
        let m = open_sparse_matrix_by_path(&format!("{o}/sections/S3.zarr")).unwrap();
        m.read_columns_csc((0..144).collect::<Vec<usize>>())
            .unwrap()
    };
    assert_eq!(read(&a), read(&b));
}

#[test]
fn an_impossible_setting_is_refused_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("bad").to_string_lossy().to_string();
    assert!(run_simulate(&args(&out, "L0", &["pve_donor=0.9"])).is_err());
    assert!(!std::path::Path::new(&out).exists());
}
