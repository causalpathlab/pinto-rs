//! `--from`: a run's files are found beside its manifest.

use crate::lr_activity::args::LraInputArgs;
use crate::util::metadata::{resolve_beside, PintoMetadata};

fn manifest(dir: &std::path::Path, data: &str) -> std::path::PathBuf {
    let path = dir.join("run.pinto.json");
    PintoMetadata {
        command: "lc".into(),
        prefix: "run".into(),
        data_files: Some(vec![data.into()]),
        ..Default::default()
    }
    .write(&path)
    .unwrap();
    path
}

#[test]
fn relative_paths_resolve_against_the_manifest_directory() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("d.zarr"), b"").unwrap();
    let m = manifest(dir.path(), "d.zarr");
    assert_eq!(
        resolve_beside(&m, "d.zarr"),
        Some(dir.path().join("d.zarr"))
    );
    assert_eq!(resolve_beside(&m, "missing.zarr"), None);
    let abs = dir.path().join("d.zarr");
    assert_eq!(resolve_beside(&m, &abs.to_string_lossy()), Some(abs));

    let input = LraInputArgs {
        from: Some(m.to_string_lossy().into()),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    assert_eq!(
        input.data_files,
        vec![dir.path().join("d.zarr").to_string_lossy().into()]
    );
    // The prefix is the manifest's own path, wherever the run was made.
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    assert_eq!(input.out(None).unwrap(), prefix);
    assert_eq!(input.out(Some("o")).unwrap(), "o");
    // Data files on the command line win.
    let input = LraInputArgs {
        from: Some(m.to_string_lossy().into()),
        data_files: vec!["other.zarr".into()],
        ..Default::default()
    }
    .resolve()
    .unwrap();
    assert_eq!(input.data_files, vec!["other.zarr".into()]);
}

#[test]
fn errors_name_the_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let m = manifest(dir.path(), "gone.zarr");
    let err = LraInputArgs {
        from: Some(m.to_string_lossy().into()),
        ..Default::default()
    }
    .resolve()
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("gone.zarr") && err.contains("run.pinto.json"),
        "{err}"
    );

    std::fs::write(dir.path().join("gone.zarr"), b"").unwrap();
    let input = LraInputArgs {
        from: Some(m.to_string_lossy().into()),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let err = input.lc_prefix().unwrap_err().to_string();
    assert!(err.contains("run.link_community.parquet"), "{err}");

    let err = LraInputArgs {
        from: Some("nowhere.pinto.json".into()),
        ..Default::default()
    }
    .resolve()
    .unwrap_err()
    .to_string();
    assert!(err.contains("nowhere.pinto.json"), "{err}");
    let err = LraInputArgs::default().resolve().unwrap_err().to_string();
    assert!(err.contains("--from"), "{err}");
    assert!(LraInputArgs::default()
        .resolve()
        .is_err_and(|e| e.to_string().contains("--coord")));
}

/// A cell whose only contact straddles two batches still gets its own
/// batch, not the fallback label, which reads as the pooled row's.
#[test]
fn a_cell_without_a_within_batch_contact_keeps_its_batch() -> anyhow::Result<()> {
    use crate::util::common::HashMap;
    use legume_numeric::matrix::parquet::{write_named_table, Column};
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("r.coord_pairs.parquet");
    let left: Vec<Box<str>> = vec!["c0".into(), "c1".into()];
    let right: Vec<Box<str>> = vec!["c1".into(), "c2".into()];
    let lb: Vec<Box<str>> = vec!["A".into(), "A".into()];
    let rb: Vec<Box<str>> = vec!["A".into(), "B".into()];
    let rows: Vec<Box<str>> = vec!["0".into(), "1".into()];
    write_named_table(
        &path.to_string_lossy(),
        "cell_pair",
        &rows,
        &[
            ("left_cell".into(), Column::Str(&left)),
            ("right_cell".into(), Column::Str(&right)),
            ("left_batch".into(), Column::Str(&lb)),
            ("right_batch".into(), Column::Str(&rb)),
        ],
    )?;
    let cols: HashMap<Box<str>, usize> = ["c0", "c1", "c2", "c3"]
        .iter()
        .enumerate()
        .map(|(i, c)| ((*c).into(), i))
        .collect();
    let crate::lr_activity::input::SpatialContacts {
        pairs,
        cell_batch: batch,
        ..
    } = crate::lr_activity::input::read_spatial_contacts(&path, &cols, 4)?;
    assert_eq!(pairs, [(0, 1), (1, 2)]);
    assert_eq!(batch[2].as_deref(), Some("B"), "{batch:?}");
    assert_eq!(batch[1].as_deref(), Some("A"));
    assert_eq!(batch[3], None, "a cell with no contact has none");
    Ok(())
}

/// Cell types are optional: a run with no lupin round beside it has none.
#[test]
fn a_run_without_a_lupin_round_has_no_cell_types() {
    let dir = tempfile::tempdir().unwrap();
    let m = manifest(dir.path(), "d.zarr");
    assert_eq!(crate::lr_activity::input::newest_round(&m), None);
}

/// A label table names cells by its `cell` column and the label column
/// asked for, else by its first two columns; unnamed and `unassigned`
/// cells share the last code.
#[test]
fn cell_labels_read_by_name_or_position() {
    use crate::lr_activity::input::read_labels;
    use crate::util::common::HashMap;
    let dir = tempfile::tempdir().unwrap();
    let cells: HashMap<Box<str>, usize> = ["c0", "c1", "c2", "c3"]
        .iter()
        .enumerate()
        .map(|(i, n)| (Box::<str>::from(*n), i))
        .collect();
    let named = dir.path().join("named.tsv");
    std::fs::write(
        &named,
        "x\tregion\tcell\n1\tpatch\tc1\n2\tunassigned\tc2\n3\tbulk\tc3\n4\tbulk\tzz\n",
    )
    .unwrap();
    let (codes, names) = read_labels(&named, &["region", "label"], &cells, 4).unwrap();
    assert_eq!(
        names,
        ["bulk".into(), "patch".into(), "unassigned".into()] as [Box<str>; 3]
    );
    assert_eq!(codes, [2, 1, 2, 0]);
    let bare = dir.path().join("bare.csv");
    std::fs::write(&bare, "spot,domain\nc0,d1\nc3,d2\n").unwrap();
    let (codes, names) = read_labels(&bare, &["region", "label"], &cells, 4).unwrap();
    assert_eq!(names.len(), 3);
    assert_eq!(codes, [0, 2, 2, 1]);
    let none = dir.path().join("none.csv");
    std::fs::write(&none, "cell,region\nq,a\n").unwrap();
    let e = read_labels(&none, &["region"], &cells, 4)
        .unwrap_err()
        .to_string();
    assert!(e.contains("c0"), "{e}");
}

/// The cell-type caveat names where the labels came from: a `--cell-types`
/// file is no lupin round.
#[test]
fn the_cell_type_caveat_names_its_source() {
    let file = crate::lr_activity::input::cell_type_caveat("--cell-types ct.tsv");
    assert!(
        file.starts_with("cell types from --cell-types ct.tsv adjust the estimate"),
        "{file}"
    );
    assert!(!file.contains("lupin"), "{file}");
    let round = crate::lr_activity::input::cell_type_caveat("lupin round r.lupin.json");
    assert!(
        round.starts_with("cell types from lupin round r.lupin.json"),
        "{round}"
    );
}
