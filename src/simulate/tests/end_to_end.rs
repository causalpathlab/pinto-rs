use crate::simulate::run::{run_simulate, SimulateArgs};
use data_beans::sparse_io::open_sparse_matrix_by_path;

fn truth_meta(out: &str) -> crate::util::metadata::PintoMetadata {
    let path = std::path::PathBuf::from(format!("{out}/truth/truth.pinto.json"));
    crate::util::metadata::PintoMetadata::read(&path).unwrap()
}

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

/// The truth is written as a run with a cells table and the domain, state
/// and donor levels.
#[test]
fn the_truth_is_written_as_a_run() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "L2", &[])).unwrap();
    let meta = truth_meta(&out);
    let tags: Vec<String> = meta.levels.unwrap().iter().map(|l| l.tag.clone()).collect();
    assert_eq!(tags, ["domain", "state", "donor"]);
    assert_eq!(meta.n_cells, 6 * 144);
    assert_eq!(meta.data_files.unwrap().len(), 6);
}

/// The manifest's data files resolve beside it, and every backend column,
/// read the way pinto reads data files, names a truth cell: with one section
/// (bare barcodes) and with several (`barcode@section`).
#[test]
fn backend_columns_join_onto_the_truth_cells() {
    use crate::util::input::read_expr_data;
    use crate::util::metadata::resolve_beside;
    for (tag, extra) in [
        ("one", vec!["n_donors=1", "sections_per_donor=1"]),
        ("six", vec![]),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join(tag).to_string_lossy().to_string();
        run_simulate(&args(&out, "L0", &extra)).unwrap();
        let manifest = std::path::PathBuf::from(format!("{out}/truth/truth.pinto.json"));
        let files: Vec<Box<str>> = truth_meta(&out)
            .data_files
            .unwrap()
            .iter()
            .map(|f| {
                assert!(std::path::Path::new(f).is_relative(), "{f}");
                let p = resolve_beside(&manifest, f).expect("a data file beside the manifest");
                p.to_string_lossy().into()
            })
            .collect();
        let columns = read_expr_data(&files).unwrap().column_names().unwrap();
        let cells: std::collections::HashSet<Box<str>> = {
            use legume_numeric::matrix::parquet::read_parquet_string_column;
            read_parquet_string_column(&format!("{out}/truth/truth.cells.parquet"), 0)
                .unwrap()
                .into_iter()
                .collect()
        };
        assert_eq!(columns.len(), cells.len(), "{tag}");
        assert!(columns.iter().all(|c| cells.contains(c)), "{tag}");
    }
}

/// `pinto view` opens the truth run, places every spot, and names the
/// communities of each level as the truth tables do.
#[cfg(feature = "view")]
#[test]
fn the_viewer_opens_the_truth() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "L2", &[])).unwrap();
    let run = crate::view::data::Run::open(&format!("{out}/truth/truth")).unwrap();
    let geometry = run.load_geometry().unwrap();
    assert_eq!(geometry.names.len(), 6 * 144);
    assert_eq!(geometry.tiles.len(), 6);
    let levels = run.meta.level_list();
    let domain = run.load_communities(&geometry, &levels[0]).unwrap();
    let names: Vec<&str> = domain
        .names
        .as_ref()
        .unwrap()
        .iter()
        .map(|n| n.as_ref())
        .collect();
    assert_eq!(names, ["DM1", "DM2", "DM3", "DM4", "DM5"]);
}
