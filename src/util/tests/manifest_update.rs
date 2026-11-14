//! Concurrent changes to one run manifest: every read-change-write holds
//! the manifest's lock, and every write replaces the file whole, so runs on
//! one fit (lra and its power check, a sweep's cells) keep each other's
//! updates and a reader never sees half a file.

use crate::util::metadata::{LevelInfo, PintoMetadata};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// A level that only names itself: what one update adds.
fn level(tag: String) -> LevelInfo {
    LevelInfo {
        tag,
        level_index: 0,
        propensity: String::new(),
        link_community: None,
        feature_community: None,
        entropy_present: None,
        community_names: None,
    }
}

/// A manifest at `path` large enough that writing it takes a while, so a
/// reader racing an unatomic write would catch it half done.
fn seed(path: &Path) {
    PintoMetadata {
        command: "lc".into(),
        levels: Some((0..400).map(|i| level(format!("seed{i}"))).collect()),
        ..Default::default()
    }
    .write(path)
    .unwrap();
}

/// A seeded `run.pinto.json` in a folder of its own.
fn seeded() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.pinto.json");
    seed(&path);
    (dir, path)
}

/// One update: read, add the level `tag` after a pause (the time a real
/// update spends between its read and its write), write.
fn add(path: &Path, tag: String) {
    PintoMetadata::backfill(path, |m| {
        std::thread::sleep(Duration::from_millis(2));
        m.levels.get_or_insert_with(Vec::new).push(level(tag));
    });
}

/// Read `path` until `done`, counting reads and the ones that failed.
fn read_until(path: &Path, done: &AtomicBool) -> (usize, usize) {
    let (mut reads, mut failed) = (0, 0);
    while !done.load(Ordering::Relaxed) {
        reads += 1;
        if PintoMetadata::read(path).is_err() {
            failed += 1;
        }
    }
    (reads, failed)
}

/// Every read, of `reads`, parsed.
fn assert_reads_parsed((reads, failed): (usize, usize)) {
    assert!(reads > 0);
    assert_eq!(
        failed, 0,
        "{failed} of {reads} reads saw an unparseable manifest"
    );
}

/// The tags of `want` that the manifest's levels lack.
fn lost(path: &Path, want: &[String]) -> Vec<String> {
    let meta = PintoMetadata::read(path).unwrap();
    let have: std::collections::HashSet<String> =
        meta.level_list().into_iter().map(|l| l.tag).collect();
    want.iter()
        .filter(|t| !have.contains(*t))
        .cloned()
        .collect()
}

/// The manifest holds every tag of `want`.
#[track_caller]
fn assert_kept(path: &Path, want: &[String]) {
    let lost = lost(path, want);
    assert!(
        lost.is_empty(),
        "{} of {} updates lost",
        lost.len(),
        want.len()
    );
}

#[test]
fn concurrent_updates_from_threads_all_survive_and_every_read_parses() {
    const WRITERS: usize = 4;
    const EACH: usize = 5;
    let (_dir, path) = seeded();

    let done = AtomicBool::new(false);
    let counts = std::thread::scope(|s| {
        let reader = s.spawn(|| read_until(&path, &done));
        let writers: Vec<_> = (0..WRITERS)
            .map(|w| {
                let path = &path;
                s.spawn(move || {
                    for j in 0..EACH {
                        add(path, format!("w{w}-{j}"));
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        done.store(true, Ordering::Relaxed);
        reader.join().unwrap()
    });

    assert_reads_parsed(counts);
    let want: Vec<String> = (0..WRITERS)
        .flat_map(|w| (0..EACH).map(move |j| format!("w{w}-{j}")))
        .collect();
    assert_kept(&path, &want);
}

/// Writes back to back, with readers all the while: none of the readers'
/// reads finds a manifest half written.
#[test]
fn a_reader_never_sees_a_manifest_half_written() {
    const READERS: usize = 4;
    const WRITES: usize = 40;
    let (_dir, path) = seeded();
    let meta = PintoMetadata::read(&path).unwrap();

    let done = AtomicBool::new(false);
    let counts: Vec<(usize, usize)> = std::thread::scope(|s| {
        let readers: Vec<_> = (0..READERS)
            .map(|_| s.spawn(|| read_until(&path, &done)))
            .collect();
        for _ in 0..WRITES {
            meta.write(&path).unwrap();
        }
        done.store(true, Ordering::Relaxed);
        readers.into_iter().map(|r| r.join().unwrap()).collect()
    });

    assert_reads_parsed((
        counts.iter().map(|c| c.0).sum(),
        counts.iter().map(|c| c.1).sum(),
    ));
}

/// Set in a child of [`concurrent_updates_from_processes_all_survive`]:
/// the manifest it updates, and the worker it is.
const MANIFEST_ENV: &str = "PINTO_TEST_MANIFEST_UPDATE";
const WORKER_ENV: &str = "PINTO_TEST_MANIFEST_WORKER";

#[test]
fn concurrent_updates_from_processes_all_survive() {
    const WORKERS: usize = 4;
    const EACH: usize = 4;
    let tag = |w: usize, j: usize| format!("p{w}-{j}");

    // A child: its share of the updates, then out.
    if let (Ok(path), Ok(w)) = (std::env::var(MANIFEST_ENV), std::env::var(WORKER_ENV)) {
        let w: usize = w.parse().unwrap();
        for j in 0..EACH {
            add(Path::new(&path), tag(w, j));
        }
        return;
    }

    let (_dir, path) = seeded();
    // This test by its name in the harness, run alone in each child.
    let name = module_path!().split_once("::").unwrap().1;
    let name = format!("{name}::concurrent_updates_from_processes_all_survive");
    let exe = std::env::current_exe().unwrap();
    let children: Vec<_> = (0..WORKERS)
        .map(|w| {
            std::process::Command::new(&exe)
                .args([name.as_str(), "--exact", "--test-threads=1", "-q"])
                .env(MANIFEST_ENV, &path)
                .env(WORKER_ENV, w.to_string())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap()
        })
        .collect();

    let done = AtomicBool::new(false);
    let counts = std::thread::scope(|s| {
        let reader = s.spawn(|| read_until(&path, &done));
        for mut c in children {
            assert!(c.wait().unwrap().success(), "a worker failed");
        }
        done.store(true, Ordering::Relaxed);
        reader.join().unwrap()
    });

    assert_reads_parsed(counts);
    let want: Vec<String> = (0..WORKERS)
        .flat_map(|w| (0..EACH).map(move |j| tag(w, j)))
        .collect();
    assert_kept(&path, &want);
}

/// Writing and updating leave the manifest and its hidden lock, which no
/// reader of `*.pinto.json` lists, and no temporary file.
#[test]
fn writes_leave_the_manifest_and_a_hidden_lock_only() {
    let (dir, path) = seeded();
    add(&path, "x".into());
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, [".run.pinto.json.lock", "run.pinto.json"]);
    assert_kept(&path, &["x".into()]);
}

/// A missing manifest is left missing, with no lock made beside it.
#[test]
fn backfilling_a_missing_manifest_makes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gone.pinto.json");
    let mut called = false;
    PintoMetadata::backfill(&path, |_| called = true);
    assert!(!called);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

/// `write_if_missing` writes a manifest that is not there and keeps one
/// that is.
#[test]
fn write_if_missing_keeps_an_existing_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.pinto.json");
    let fresh = |command: &str| PintoMetadata {
        command: command.into(),
        ..Default::default()
    };
    fresh("first").write_if_missing(&path).unwrap();
    fresh("second").write_if_missing(&path).unwrap();
    assert_eq!(PintoMetadata::read(&path).unwrap().command, "first");
}

/// A manifest named through a link is updated where it lives, and the
/// link kept.
#[cfg(unix)]
#[test]
fn an_update_through_a_link_keeps_the_link() {
    let (dir, real) = seeded();
    let link = dir.path().join("link.pinto.json");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    add(&link, "x".into());
    assert!(std::fs::symlink_metadata(&link)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_kept(&real, &["x".into()]);
}

/// A manifest made read-only is left as it is, as a write in place left it.
#[cfg(unix)]
#[test]
fn a_read_only_manifest_is_not_replaced() {
    use std::os::unix::fs::PermissionsExt;
    // Root writes a read-only file anyway, as a write in place did.
    // SAFETY: geteuid has no preconditions and cannot fail.
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let (_dir, path) = seeded();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
    add(&path, "x".into());
    assert_eq!(lost(&path, &["x".into()]), ["x"]);
}

/// A manifest with a second hard link is written in place, so both names
/// still name one manifest after the update.
#[cfg(unix)]
#[test]
fn a_hard_linked_manifest_stays_one_file() {
    let (dir, path) = seeded();
    let twin = dir.path().join("twin.pinto.json");
    std::fs::hard_link(&path, &twin).unwrap();
    add(&path, "x".into());
    assert_kept(&twin, &["x".into()]);
}

/// A chain of links to a manifest not written yet is followed to its end:
/// the manifest is made there and every link in the chain kept.
#[cfg(unix)]
#[test]
fn a_chain_of_links_is_followed_to_its_end() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("real")).unwrap();
    let end = dir.path().join("real").join("run.pinto.json");
    let b = dir.path().join("b.pinto.json");
    let a = dir.path().join("a.pinto.json");
    std::os::unix::fs::symlink(&end, &b).unwrap();
    std::os::unix::fs::symlink(&b, &a).unwrap();
    PintoMetadata::default().write(&a).unwrap();
    assert!(end.is_file(), "made at the end of the chain");
    for link in [&a, &b] {
        let kind = std::fs::symlink_metadata(link).unwrap().file_type();
        assert!(kind.is_symlink(), "{} kept", link.display());
    }
}
