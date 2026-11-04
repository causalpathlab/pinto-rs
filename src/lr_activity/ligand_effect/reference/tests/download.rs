//! The download cache, offline.

use crate::lr_activity::ligand_effect::reference::download::{
    fetch_all, part_path, sha256_bytes, Source,
};
use std::path::Path;

fn source(name: &'static str) -> Source {
    Source {
        name,
        url: format!("https://example.invalid/{name}"),
        sha256: None,
    }
}

#[test]
fn sha256_of_abc() {
    assert_eq!(
        sha256_bytes(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn cached_files_are_hashed_not_fetched() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), b"abc").unwrap();
    let got = fetch_all(dir.path(), &[source("a.txt")], false, true).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].bytes, 3);
    assert_eq!(got[0].sha256, sha256_bytes(b"abc"));
}

#[test]
fn offline_names_every_missing_file_and_the_cache() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), b"abc").unwrap();
    let err = fetch_all(
        dir.path(),
        &[source("a.txt"), source("b.tsv"), source("c.gz")],
        false,
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("PINTO_OFFLINE"), "{err}");
    assert!(err.contains("b.tsv") && err.contains("c.gz"), "{err}");
    assert!(!err.contains("a.txt"), "{err}");
    assert!(err.contains(&dir.path().display().to_string()), "{err}");
}

#[test]
fn offline_refresh_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), b"abc").unwrap();
    let err = fetch_all(dir.path(), &[source("a.txt")], true, true).unwrap_err();
    assert!(err.to_string().contains("refresh"), "{err}");
}

#[test]
fn part_file_sits_beside_its_target_and_is_unique() {
    let (a, b) = (
        part_path(Path::new("/x/y/file.tsv")),
        part_path(Path::new("/x/y/file.tsv")),
    );
    assert_eq!(a.parent(), Some(Path::new("/x/y")));
    let name = a.file_name().unwrap().to_string_lossy().into_owned();
    assert!(
        name.starts_with("file.tsv.") && name.ends_with(".part"),
        "{name}"
    );
    assert!(name.contains(&std::process::id().to_string()), "{name}");
    assert_ne!(a, b, "two writers never share a temporary file");
}

#[test]
fn a_build_lock_excludes_a_second_holder_until_dropped() {
    use crate::lr_activity::ligand_effect::reference::download::BuildLock;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".build.lock");
    let lock = BuildLock::acquire(&path).unwrap();
    assert!(path.is_file());
    let (tx, rx) = std::sync::mpsc::channel();
    let p = path.clone();
    let waiter = std::thread::spawn(move || {
        let _second = BuildLock::acquire(&p).unwrap();
        tx.send(()).unwrap();
    });
    assert!(rx
        .recv_timeout(std::time::Duration::from_millis(300))
        .is_err());
    drop(lock);
    rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    waiter.join().unwrap();
    assert!(!path.exists());
}

#[test]
fn write_atomic_leaves_no_temporary_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.tsv");
    crate::lr_activity::ligand_effect::reference::download::write_atomic(&path, b"x").unwrap();
    let names: Vec<_> = std::fs::read_dir(dir.path()).unwrap().flatten().collect();
    assert_eq!(names.len(), 1);
    assert_eq!(std::fs::read(&path).unwrap(), b"x");
}
