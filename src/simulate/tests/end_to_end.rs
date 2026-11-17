use crate::simulate::run::{run_simulate, SimulateArgs};
use data_beans::sparse_io::open_sparse_matrix_by_path;

fn truth_meta(out: &str) -> crate::util::metadata::PintoMetadata {
    let path = std::path::PathBuf::from(format!("{out}/truth/truth.pinto.json"));
    crate::util::metadata::PintoMetadata::read(&path).unwrap()
}

fn args(out: &str, scenario: &str, extra: &[&str]) -> SimulateArgs {
    let mut set: Vec<String> = ["grid_side=12", "n_genes=60", "program_genes=8", "seed=3"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    set.extend(extra.iter().map(|s| s.to_string()));
    SimulateArgs {
        scenario: scenario.into(),
        set,
        coords: Vec::new(),
        coord_column_names: vec!["row_um".into(), "col_um".into()],
        reference_data: None,
        reference_data_labels: None,
        out: out.into(),
    }
}

/// A run writes one readable backend and coordinate file per section, the
/// truth tables, and a manifest holding every parameter.
#[test]
fn a_run_writes_backends_truth_and_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "donor-confounded-donor-only-cell-type", &[])).unwrap();
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
    assert!(m["realized_pve_program_genes"]["type"].as_f64().unwrap() > 0.0);
}

/// The same seed gives the same counts, the scenario named by its earlier
/// name or its own, and the manifest records its own name.
#[test]
fn a_run_is_reproducible() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a").to_string_lossy().to_string();
    let b = dir.path().join("b").to_string_lossy().to_string();
    run_simulate(&args(&a, "donor-confounded", &[])).unwrap();
    run_simulate(&args(&b, "L1", &[])).unwrap();
    let read = |o: &str| {
        let m = open_sparse_matrix_by_path(&format!("{o}/sections/S3.zarr")).unwrap();
        m.read_columns_csc((0..144).collect::<Vec<usize>>())
            .unwrap()
    };
    assert_eq!(read(&a), read(&b));
    let scenario = |o: &str| {
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{o}/manifest.json")).unwrap())
                .unwrap();
        m["scenario"].clone()
    };
    assert_eq!(scenario(&a), "donor-confounded");
    assert_eq!(scenario(&b), "donor-confounded");
}

#[test]
fn an_impossible_setting_is_refused_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("bad").to_string_lossy().to_string();
    assert!(run_simulate(&args(&out, "unconfounded", &["pve_donor=0.9"])).is_err());
    assert!(!std::path::Path::new(&out).exists());
}

/// The truth is written as a run with a cells table and the domain, state
/// and donor levels.
#[test]
fn the_truth_is_written_as_a_run() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "donor-confounded-donor-only-cell-type", &[])).unwrap();
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
        run_simulate(&args(&out, "unconfounded", &extra)).unwrap();
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
    run_simulate(&args(&out, "donor-confounded-donor-only-cell-type", &[])).unwrap();
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
    assert_eq!(
        names,
        [
            "shared_domain_1",
            "shared_domain_2",
            "shared_domain_3",
            "shared_domain_4",
            "D3_specific_domain_1"
        ]
    );
}

/// Scoring a simulation's own truth, written as a cell<TAB>label table,
/// gives perfect scores on both tests; labels that follow the donor fail
/// the first.
#[test]
fn scoring_the_truth_and_the_donors() {
    use crate::sim_eval::score::{read_labels, score, TruthSpots};
    use crate::simulate::panel::donor_label;
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim").to_string_lossy().to_string();
    run_simulate(&args(&out, "donor-confounded-donor-only-cell-type", &[])).unwrap();
    let t = TruthSpots::read(&out).unwrap();
    let write = |name: &str, f: &dyn Fn(usize) -> String| -> String {
        let path = dir.path().join(name).to_string_lossy().to_string();
        let text: String = std::iter::once("cell\tlabel\n".to_string())
            .chain((0..t.cells.len()).map(|i| format!("{}\t{}\n", t.cells[i], f(i))))
            .collect();
        std::fs::write(&path, text).unwrap();
        path
    };
    let truth = write("truth.tsv", &|i| t.domain_labels[t.domain[i]].to_string());
    let donors = write("donors.tsv", &|i| donor_label(t.donor[i]).to_string());
    let s = score(&t, &read_labels(&truth, "final", "label").unwrap()).unwrap();
    assert_eq!((s.n_scored, s.donor_dependence), (t.cells.len(), 0.0));
    assert!((s.shared_ari - 1.0).abs() < 1e-12);
    assert_eq!(s.patches.len(), 1);
    assert_eq!(s.patches[0].f1, 1.0);
    let s = score(&t, &read_labels(&donors, "final", "label").unwrap()).unwrap();
    assert!((s.donor_dependence - 1.0).abs() < 1e-9);
    for d in &s.donor_dependence_by_domain {
        let x = d.dependence.unwrap();
        assert!((x - 1.0).abs() < 1e-9, "{}: {x}", d.domain);
    }
}

/// The outputs appear at `out` only once all are written: they are written
/// to `{out}.partial` and moved, and a partial folder an earlier stopped run
/// left is replaced.
#[test]
fn a_simulation_appears_only_when_whole() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sim");
    let partial = dir.path().join("sim.partial");
    std::fs::create_dir_all(partial.join("sections")).unwrap();
    std::fs::write(partial.join("sections/stale"), "").unwrap();
    crate::simulate::run::run_simulate(&args(&out.to_string_lossy(), "unconfounded", &[])).unwrap();
    assert!(out.join("manifest.json").is_file());
    assert!(!out.join("sections/stale").exists());
    assert!(!partial.exists());
    let run = std::fs::read_to_string(out.join("truth/truth.pinto.json")).unwrap();
    assert!(
        !run.contains("partial"),
        "the run names where it was moved: {run}"
    );
}

/// An out given with a trailing slash, or already there as an empty folder
/// (as the records' replay script makes it), still gets the whole run, and
/// no partial folder is left beside it.
#[test]
fn a_simulation_into_a_slashed_or_empty_out_is_whole() {
    let dir = tempfile::tempdir().unwrap();
    let slashed = dir.path().join("a");
    crate::simulate::run::run_simulate(&args(
        &format!("{}/", slashed.display()),
        "unconfounded",
        &[],
    ))
    .unwrap();
    assert!(slashed.join("manifest.json").is_file());
    let empty = dir.path().join("b");
    std::fs::create_dir(&empty).unwrap();
    crate::simulate::run::run_simulate(&args(&empty.to_string_lossy(), "unconfounded", &[]))
        .unwrap();
    assert!(empty.join("truth/truth.pinto.json").is_file());
    for left in ["a.partial", "b.partial", "a/.partial"] {
        assert!(!dir.path().join(left).exists(), "{left} left behind");
    }
}

/// Each donor's status is in the truth: a `confounded` column per donor in
/// `donors.parquet` and per spot in `spots.parquet`, and in the manifest the
/// unconfounded donors and whether they were drawn.
#[test]
fn the_truth_records_each_donors_status() {
    use legume_numeric::matrix::parquet::read_parquet_string_columns_by_name;
    let dir = tempfile::tempdir().unwrap();
    for (name, set, drawn) in [
        ("drawn", "n_unconfounded_donors=1", true),
        ("listed", "unconfounded_donors=[1]", false),
    ] {
        let out = dir.path().join(name).to_string_lossy().to_string();
        run_simulate(&args(&out, "donor-confounded", &[set])).unwrap();
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}/manifest.json")).unwrap())
                .unwrap();
        assert_eq!(m["unconfounded_donors"]["drawn"], drawn, "{name}");
        let free: Vec<String> = m["unconfounded_donors"]["donors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_str().unwrap().to_string())
            .collect();
        assert_eq!(free.len(), 1, "{name}");
        if !drawn {
            assert_eq!(free, ["D2"]);
        }
        let status = |d: &str| {
            if free.iter().any(|f| f == d) {
                "false"
            } else {
                "true"
            }
        };
        let donors = read_parquet_string_columns_by_name(
            &format!("{out}/truth/donors.parquet"),
            &["donor", "confounded"],
        )
        .unwrap();
        assert_eq!(donors[0].len(), 3, "{name}");
        for (d, c) in donors[0].iter().zip(&donors[1]) {
            assert_eq!(c.as_ref(), status(d), "{name} {d}");
        }
        let spots = read_parquet_string_columns_by_name(
            &format!("{out}/truth/spots.parquet"),
            &["donor", "confounded"],
        )
        .unwrap();
        assert_eq!(spots[0].len(), 6 * 144);
        for (d, c) in spots[0].iter().zip(&spots[1]) {
            assert_eq!(c.as_ref(), status(d), "{name} {d}");
        }
    }
}

/// A donor is confounded only when some shift reaches it: with no nuisance
/// term on (the unconfounded scenario, `nuisance_off`, or every nuisance
/// share at 0) every donor and spot is written `false`.
#[test]
fn a_donor_with_no_shift_is_not_confounded() {
    use legume_numeric::matrix::parquet::read_parquet_string_columns_by_name;
    let dir = tempfile::tempdir().unwrap();
    for (name, scenario, set) in [
        ("default", "unconfounded", &[][..]),
        ("off", "donor-confounded", &["nuisance_off=true"][..]),
        (
            "zero",
            "donor-confounded",
            &["pve_donor=0", "pve_inter=0", "pve_section=0"][..],
        ),
    ] {
        let out = dir.path().join(name).to_string_lossy().to_string();
        run_simulate(&args(&out, scenario, set)).unwrap();
        for table in ["donors", "spots"] {
            let col = read_parquet_string_columns_by_name(
                &format!("{out}/truth/{table}.parquet"),
                &["confounded"],
            )
            .unwrap();
            assert!(!col[0].is_empty(), "{name} {table}");
            assert!(
                col[0].iter().all(|c| c.as_ref() == "false"),
                "{name} {table}"
            );
        }
    }
}
