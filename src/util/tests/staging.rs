//! `--atomic-outputs`' staging: a staged run's outputs land under their
//! final names only on commit, the files that mark it complete last and
//! naming the final paths; an abort leaves nothing; a commit that fails
//! part way keeps what it had not moved and puts back what it replaced.

use crate::util::metadata::{LraSubset, OutputFiles, PintoMetadata};
use crate::util::staging::Staging;
use serde_json::json;
use std::path::{Path, PathBuf};

/// The names in `dir`, sorted.
fn names(dir: &Path) -> Vec<String> {
    let mut n: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    n.sort();
    n
}

/// The names in `dir` that are not hidden, sorted.
fn shown(dir: &Path) -> Vec<String> {
    let mut n = names(dir);
    n.retain(|n| !n.starts_with('.'));
    n
}

fn text(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

/// The staging folder of the prefix staging `s`.
fn staging_of(s: &Staging) -> PathBuf {
    PathBuf::from(s.out()).parent().unwrap().to_path_buf()
}

/// A run staged for `{dir}/run`: two outputs and a manifest that records
/// them, one by its absolute staged path and one relative.
fn staged_run(dir: &Path) -> Staging {
    let out = format!("{}/run", dir.display());
    let s = Staging::prefix(&out).unwrap();
    let staged = s.out().to_string();
    assert_ne!(staged, out);
    std::fs::write(format!("{staged}.a.parquet"), "a").unwrap();
    std::fs::write(format!("{staged}.b.json"), r#"{"self": "x"}"#).unwrap();
    let rel = Path::new(&staged)
        .strip_prefix(dir)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    PintoMetadata {
        command: "lc".into(),
        prefix: staged.clone(),
        outputs: OutputFiles {
            propensity: Some(format!("{staged}.a.parquet")),
            scores: Some(format!("{rel}.b.json")),
            ..Default::default()
        },
        ..Default::default()
    }
    .write(Path::new(&format!("{staged}.pinto.json")))
    .unwrap();
    s
}

/// An earlier run's manifest at `{dir}/run.pinto.json`, and its text.
fn earlier_manifest(dir: &Path) -> String {
    let path = dir.join("run.pinto.json");
    PintoMetadata {
        command: "earlier".into(),
        ..Default::default()
    }
    .write(&path)
    .unwrap();
    text(&path)
}

#[test]
fn a_commit_moves_every_output_and_the_manifest_last() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let s = staged_run(dir);
    let staging = staging_of(&s);
    let manifest = dir.join("run.pinto.json");
    let mut landed = Vec::new();
    s.commit_observed(|p| {
        if p == manifest {
            // every other output is in place before the manifest is
            assert!(dir.join("run.a.parquet").is_file());
            assert!(dir.join("run.b.json").is_file());
            let left: Vec<String> = shown(&staging)
                .into_iter()
                .filter(|n| !n.ends_with(".pinto.json"))
                .collect();
            assert!(left.is_empty(), "staged still: {left:?}");
        } else {
            assert!(!manifest.exists(), "manifest before {}", p.display());
        }
        landed.push(p.to_path_buf());
    })
    .unwrap();
    assert_eq!(landed.last(), Some(&manifest));
    assert_eq!(landed.len(), 3);
    assert!(!staging.exists(), "the staging folder is removed");
    // the manifest writer's lock is the only other file, and no output
    assert_eq!(
        names(dir),
        [
            ".run.pinto.json.lock",
            "run.a.parquet",
            "run.b.json",
            "run.pinto.json"
        ]
    );
}

#[test]
fn a_committed_manifest_records_the_final_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    staged_run(dir).commit().unwrap();
    let meta = PintoMetadata::read(&dir.join("run.pinto.json")).unwrap();
    let out = format!("{}/run", dir.display());
    assert_eq!(meta.prefix, out);
    assert_eq!(meta.outputs.propensity, Some(format!("{out}.a.parquet")));
    assert_eq!(meta.outputs.scores.as_deref(), Some("run.b.json"));
    // a JSON file that names no staged path is left as written
    assert_eq!(text(&dir.join("run.b.json")), r#"{"self": "x"}"#);
}

/// Only the string that names a staged path changes: escapes, other
/// strings and the spacing stay byte for byte.
#[test]
fn a_recorded_path_is_made_final_and_the_rest_of_its_file_kept() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let s = staged_run(dir);
    let staged = s.out().to_string();
    let doc = |at: &str| {
        format!(
            "{{ \"note\" :\t\"a \\\"quoted\\\" back\\\\slash \\u00e9 é\",\n\n  \"at\": \"{at}.x\",\"n\":[1 ,2]}}"
        )
    };
    std::fs::write(format!("{staged}.c.json"), doc(&staged)).unwrap();
    // not JSON, though named so: left as it is
    let broken = format!("{{\"at\": \"{staged}.x\"");
    std::fs::write(format!("{staged}.d.json"), &broken).unwrap();
    s.commit().unwrap();
    let out = format!("{}/run", dir.display());
    assert_eq!(text(&dir.join("run.c.json")), doc(&out));
    assert_eq!(text(&dir.join("run.d.json")), broken);
}

#[test]
fn an_abort_leaves_no_output_and_no_staging_folder() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("other"), "kept").unwrap();
    staged_run(dir).abort();
    assert_eq!(names(dir), ["other"]);
}

#[test]
fn a_commit_replaces_earlier_outputs_and_leaves_the_rest() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("run.a.parquet"), "old").unwrap();
    std::fs::write(dir.join("run.c.parquet"), "old, not rewritten").unwrap();
    staged_run(dir).commit().unwrap();
    assert_eq!(text(&dir.join("run.a.parquet")), "a");
    assert_eq!(text(&dir.join("run.c.parquet")), "old, not rewritten");
}

/// Over an earlier run, its manifest is gone before the first new output
/// is in place: no manifest describes a mix of two runs' outputs.
#[test]
fn an_earlier_manifest_never_sits_beside_the_new_outputs() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let old = earlier_manifest(dir);
    std::fs::write(dir.join("run.a.parquet"), "old").unwrap();
    let manifest = dir.join("run.pinto.json");
    let mut landed = 0;
    staged_run(dir)
        .commit_observed(|p| {
            landed += 1;
            if p != manifest {
                assert!(!manifest.exists(), "at {}", p.display());
            }
        })
        .unwrap();
    assert_eq!(landed, 3);
    let new = text(&manifest);
    assert!(new != old && new.contains("\"lc\""), "{new}");
}

/// Whether file permissions bind this user. They do not bind root: a test
/// of a failure that permissions make says so there and is skipped, and
/// runs in full for every other user.
#[cfg(unix)]
fn permissions_bind(test: &str) -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    let root = unsafe { libc::geteuid() } == 0;
    if root {
        eprintln!("{test}: SKIPPED, run as root, whom no permission binds");
    }
    !root
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

/// An earlier manifest that could only be written in place (hard-linked,
/// read-only, a link) cannot be taken away and put back whole: the run is
/// refused before anything is moved, at its start and again at its commit,
/// and the earlier run stays as it was.
#[cfg(unix)]
#[test]
fn an_earlier_manifest_written_in_place_is_refused_before_anything_moves() {
    for why in ["hard-linked", "read-only", "a link"] {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let old = earlier_manifest(dir);
        let manifest = dir.join("run.pinto.json");
        std::fs::write(dir.join("run.a.parquet"), "old").unwrap();
        match why {
            "hard-linked" => std::fs::hard_link(&manifest, dir.join("link.json")).unwrap(),
            "read-only" => set_mode(&manifest, 0o444),
            _ => {
                std::fs::rename(&manifest, dir.join("real.json")).unwrap();
                std::os::unix::fs::symlink("real.json", &manifest).unwrap();
            }
        }
        let before = names(dir);
        let refused = |err: anyhow::Error| {
            let said = format!("{err:#}");
            let named = format!("the manifest {} is {why}", manifest.display());
            assert!(said.contains(&named), "{why}: {said}");
        };
        // at the start, with no staging folder made
        let out = format!("{}/run", dir.display());
        refused(crate::util::staging::stage_run(&out).unwrap_err());
        assert_eq!(names(dir), before, "{why}");
        // and at the commit, should it come to be so during the run
        let s = staged_run(dir);
        let staging = staging_of(&s);
        refused(s.commit().unwrap_err());
        assert_eq!(text(&manifest), old, "{why}");
        assert_eq!(text(&dir.join("run.a.parquet")), "old", "{why}");
        assert_eq!(manifest.is_symlink(), why == "a link");
        // all is still staged, and nothing was moved aside
        assert_eq!(
            shown(&staging),
            ["run.a.parquet", "run.b.json", "run.pinto.json"],
            "{why}"
        );
        assert!(!staging.join(".replaced").exists(), "{why}");
    }
}

/// What is in an output's place and of another kind is not replaced: the
/// commit is refused before its first move, and names it.
#[test]
fn a_destination_of_another_kind_is_refused_before_anything_moves() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let old = earlier_manifest(dir);
    // a folder where the second output, a file, lands
    std::fs::create_dir(dir.join("run.b.json")).unwrap();
    std::fs::write(dir.join("run.b.json/mine"), "kept").unwrap();
    let s = staged_run(dir);
    let staging = staging_of(&s);
    let said = format!("{:#}", s.commit().unwrap_err());
    assert!(said.contains("run.b.json is a folder"), "{said}");
    assert!(said.contains(&*staging.to_string_lossy()), "{said}");
    assert_eq!(text(&dir.join("run.b.json/mine")), "kept");
    assert_eq!(text(&dir.join("run.pinto.json")), old);
    assert_eq!(shown(dir), ["run.b.json", "run.pinto.json"]);
    assert_eq!(
        shown(&staging),
        ["run.a.parquet", "run.b.json", "run.pinto.json"]
    );
    assert!(!staging.join(".replaced").exists());
    std::fs::remove_dir_all(&staging).unwrap();
    std::fs::remove_dir_all(dir.join("run.b.json")).unwrap();

    // a file where a store, a folder, lands
    std::fs::write(dir.join("run.s.zarr"), "a file").unwrap();
    let s = staged_run(dir);
    let staging = staging_of(&s);
    std::fs::create_dir(staging.join("run.s.zarr")).unwrap();
    let said = format!("{:#}", s.commit().unwrap_err());
    assert!(said.contains("run.s.zarr is a file"), "{said}");
    assert_eq!(text(&dir.join("run.s.zarr")), "a file");
    assert_eq!(shown(dir), ["run.pinto.json", "run.s.zarr"]);
    std::fs::remove_dir_all(&staging).unwrap();
    std::fs::remove_file(dir.join("run.s.zarr")).unwrap();

    // a link where an output lands
    #[cfg(unix)]
    {
        std::fs::write(dir.join("elsewhere"), "linked to").unwrap();
        std::os::unix::fs::symlink("elsewhere", dir.join("run.b.json")).unwrap();
        let said = format!("{:#}", staged_run(dir).commit().unwrap_err());
        assert!(said.contains("run.b.json is a link"), "{said}");
        assert!(dir.join("run.b.json").is_symlink());
        assert_eq!(text(&dir.join("elsewhere")), "linked to");
        assert!(!dir.join("run.a.parquet").exists());
    }
}

/// A commit that fails after its first move: the staging folder is kept
/// with what was not moved and the error names it, a store it had moved
/// aside is back, and the earlier manifest is not (an output landed).
#[test]
fn a_commit_failing_part_way_keeps_the_staging_and_restores_what_it_replaced() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    earlier_manifest(dir);
    std::fs::create_dir(dir.join("run.b.zarr")).unwrap();
    std::fs::write(dir.join("run.b.zarr/chunk"), "old").unwrap();
    let s = staged_run(dir);
    let staging = staging_of(&s);
    let store = staging.join("run.b.zarr");
    std::fs::create_dir(&store).unwrap();
    std::fs::write(store.join("chunk"), "new").unwrap();
    // the moves go in name order, the store last: once run.a.parquet has
    // landed the staged store is taken away, so its move fails after the
    // earlier store was moved aside
    let mut seen = Vec::new();
    let err = s
        .commit_observed(|p| {
            seen.push(p.to_path_buf());
            if p == dir.join("run.a.parquet") {
                std::fs::remove_dir_all(&store).unwrap();
            }
        })
        .unwrap_err();
    assert_eq!(seen, [dir.join("run.a.parquet"), dir.join("run.b.json")]);
    let said = format!("{err:#}");
    assert!(said.contains(&*staging.to_string_lossy()), "{said}");
    assert_eq!(text(&dir.join("run.b.zarr/chunk")), "old");
    assert!(!dir.join("run.pinto.json").exists());
    assert_eq!(shown(dir), ["run.a.parquet", "run.b.json", "run.b.zarr"]);
    // what did not land is still staged, and the earlier manifest is held
    assert_eq!(shown(&staging), ["run.pinto.json"]);
    assert!(text(&staging.join(".replaced/0")).contains("earlier"));
}

/// A commit that fails before any output lands puts the earlier run's
/// manifest back: that run is whole, and still marked so.
#[cfg(unix)]
#[test]
fn a_commit_failing_before_any_output_lands_puts_the_earlier_manifest_back() {
    if !permissions_bind("a_commit_failing_before_any_output_lands_puts_the_earlier_manifest_back")
    {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let out = format!("{}/sim", dir.display());
    let sim = dir.join("sim");
    std::fs::create_dir_all(sim.join("truth")).unwrap();
    std::fs::write(sim.join("manifest.json"), "earlier").unwrap();
    let s = Staging::folder(&out).unwrap();
    let staged = PathBuf::from(s.out());
    std::fs::create_dir(staged.join("truth")).unwrap();
    std::fs::write(staged.join("truth/t.parquet"), "new").unwrap();
    std::fs::write(staged.join("manifest.json"), "new").unwrap();
    // nothing can be renamed out of a folder that takes no change
    set_mode(&staged.join("truth"), 0o555);
    assert!(std::fs::write(staged.join("truth/probe"), "").is_err());
    let err = s.commit().unwrap_err();
    set_mode(&staged.join("truth"), 0o755);
    assert!(format!("{err:#}").contains("t.parquet"), "{err:#}");
    assert_eq!(text(&sim.join("manifest.json")), "earlier");
    assert_eq!(names(&sim.join("truth")), Vec::<String>::new());
    assert!(staged.join("manifest.json").is_file());
}

/// The same of a run's manifest, taken away under its lock: a first move
/// that fails puts it back as it was.
#[cfg(unix)]
#[test]
fn a_commit_failing_at_its_first_move_puts_the_earlier_run_s_manifest_back() {
    if !permissions_bind("a_commit_failing_at_its_first_move_puts_the_earlier_run_s_manifest_back")
    {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let old = earlier_manifest(dir);
    let s = staged_run(dir);
    let staging = staging_of(&s);
    // a store that sorts first and takes no change: it cannot be moved
    // into another folder
    let store = staging.join("run.0.zarr");
    std::fs::create_dir(&store).unwrap();
    set_mode(&store, 0o555);
    let err = s.commit().unwrap_err();
    set_mode(&store, 0o755);
    let said = format!("{err:#}");
    assert!(said.contains("run.0.zarr"), "{said}");
    assert!(said.contains(&*staging.to_string_lossy()), "{said}");
    assert_eq!(text(&dir.join("run.pinto.json")), old);
    assert_eq!(shown(dir), ["run.pinto.json"]);
}

/// A folder that cannot be moved into place whole: the error names it,
/// and the staged folder is kept as it was.
#[cfg(unix)]
#[test]
fn a_folder_that_cannot_land_whole_is_kept_staged() {
    if !permissions_bind("a_folder_that_cannot_land_whole_is_kept_staged") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("in");
    std::fs::create_dir(&dir).unwrap();
    let s = Staging::folder(&format!("{}/sim", dir.display())).unwrap();
    fill(&s, "one");
    let staged = PathBuf::from(s.out());
    set_mode(&dir, 0o555);
    let err = s.commit().unwrap_err();
    set_mode(&dir, 0o755);
    let said = format!("{err:#}");
    assert!(said.contains(&*dir.join("sim").to_string_lossy()), "{said}");
    assert!(said.contains(&*staged.to_string_lossy()), "{said}");
    assert!(!dir.join("sim").exists());
    assert_eq!(names(&staged), ["a.zarr", "manifest.json", "truth"]);
}

/// A bare prefix is staged beside it, and its paths stay relative.
#[test]
fn a_bare_prefix_stages_beside_it_and_its_paths_stay_relative() {
    let s = Staging::prefix_named("bare", ".bare.partial-1-ab".into());
    assert_eq!(s.out(), ".bare.partial-1-ab/bare");
    assert_eq!(
        s.final_path(".bare.partial-1-ab/bare.x").as_deref(),
        Some("bare.x")
    );
    assert_eq!(s.final_path(".bare.partial-1-ab").as_deref(), Some("."));
    let s = Staging::prefix_named("/top", ".top.partial-1-ab".into());
    assert_eq!(s.final_path("/.top.partial-1-ab").as_deref(), Some("/"));
}

/// What a run says names the final paths, wherever in a line they stand.
#[test]
fn a_message_names_the_final_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().display().to_string();
    let s = Staging::prefix(&format!("{dir}/r")).unwrap();
    let said = format!("writing {}.x: full; see {dir}/other", s.out());
    assert_eq!(
        s.final_text(&said),
        format!("writing {dir}/r.x: full; see {dir}/other")
    );
    s.abort();
    let s = Staging::folder(&format!("{dir}/sim/")).unwrap();
    let said = format!("Wrote {}/truth and {0}/manifest.json", s.out());
    assert_eq!(
        s.final_text(&said),
        format!("Wrote {dir}/sim/truth and {dir}/sim/manifest.json")
    );
    s.abort();
}

#[test]
fn only_a_path_through_the_staging_folder_is_staged() {
    let tmp = tempfile::tempdir().unwrap();
    let out = format!("{}/r", tmp.path().display());
    let s = Staging::prefix(&out).unwrap();
    let staged = s.out().to_string();
    let marker = staging_of(&s)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(
        s.final_path(&format!("{staged}.x")),
        Some(format!("{out}.x"))
    );
    assert_eq!(
        s.final_path(&format!("a/{marker}/b")).as_deref(),
        Some("a/b")
    );
    // a name that only holds the marker is another file's
    assert_eq!(s.final_path(&format!("a/{marker}x/b")), None);
    assert_eq!(s.final_path(&out), None);
    s.abort();
}

/// Each staging makes a folder of its own, named for the process, and
/// removes no other: a second run on the same out, or a folder left by a
/// killed one, is left alone.
#[test]
fn a_staging_folder_is_the_run_s_own() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let out = format!("{}/run", dir.display());
    let stale = dir.join(format!(".run.partial-{}", std::process::id()));
    std::fs::create_dir(&stale).unwrap();
    std::fs::write(stale.join("kept"), "").unwrap();
    let (a, b) = (
        Staging::prefix(&out).unwrap(),
        Staging::prefix(&out).unwrap(),
    );
    let (at_a, at_b) = (staging_of(&a), staging_of(&b));
    assert_ne!(at_a, at_b);
    let named = format!(".run.partial-{}-", std::process::id());
    for at in [&at_a, &at_b] {
        let name = at.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with(&named), "{name}");
    }
    a.abort();
    assert!(!at_a.exists() && at_b.is_dir());
    b.abort();
    assert!(stale.join("kept").is_file());

    let folder = format!("{}/sim", dir.display());
    let (a, b) = (
        Staging::folder(&folder).unwrap(),
        Staging::folder(&folder).unwrap(),
    );
    assert_ne!(a.out(), b.out());
    a.abort();
    assert!(Path::new(b.out()).is_dir());
    b.abort();
}

/// The staged simulation folder `s`, its files holding `tag`: a store, a
/// plain folder with a table and a run's manifest, and the manifest.
fn fill(s: &Staging, tag: &str) {
    let d = PathBuf::from(s.out());
    std::fs::create_dir_all(d.join("truth")).unwrap();
    std::fs::create_dir_all(d.join("a.zarr")).unwrap();
    std::fs::write(d.join("a.zarr/chunk"), tag).unwrap();
    std::fs::write(d.join("truth/t.parquet"), tag).unwrap();
    std::fs::write(d.join("truth/truth.pinto.json"), "{}").unwrap();
    std::fs::write(
        d.join("manifest.json"),
        json!({ "truth": d.join("truth").to_string_lossy() }).to_string(),
    )
    .unwrap();
}

#[test]
fn a_folder_lands_whole_when_missing_and_file_by_file_into_one_in_use() {
    let tmp = tempfile::tempdir().unwrap();
    let out = format!("{}/sim/", tmp.path().display());
    let s = Staging::folder(&out).unwrap();
    fill(&s, "one");
    s.commit().unwrap();
    let sim = tmp.path().join("sim");
    let manifest: serde_json::Value =
        serde_json::from_str(&text(&sim.join("manifest.json"))).unwrap();
    assert_eq!(
        manifest["truth"],
        sim.join("truth").to_string_lossy().as_ref()
    );
    assert_eq!(names(tmp.path()), ["sim"]);

    std::fs::write(sim.join("truth/old.parquet"), "old").unwrap();
    std::fs::write(sim.join("a.zarr/stale"), "old").unwrap();
    let s = Staging::folder(&out).unwrap();
    fill(&s, "two");
    // the markers last, the outermost after the one inside, and the
    // earlier manifest gone while the outputs land
    let (outer, inner) = (
        sim.join("manifest.json"),
        sim.join("truth/truth.pinto.json"),
    );
    let mut landed = Vec::new();
    s.commit_observed(|p| {
        if p != outer {
            assert!(!outer.exists(), "the manifest at {}", p.display());
        }
        landed.push(p.to_path_buf());
    })
    .unwrap();
    assert_eq!(landed.len(), 4, "{landed:?}");
    assert_eq!(landed[2..], [inner, outer]);
    assert_eq!(text(&sim.join("truth/t.parquet")), "two");
    // a plain folder is merged into; a store is replaced whole
    assert!(sim.join("truth/old.parquet").is_file());
    assert_eq!(names(&sim.join("a.zarr")), ["chunk"]);
    assert_eq!(names(tmp.path()), ["sim"]);
}

/// An empty folder takes the run whole; a run that wrote nothing makes no
/// folder.
#[test]
fn a_folder_lands_whole_over_an_empty_one_and_not_at_all_when_nothing_was_written() {
    let tmp = tempfile::tempdir().unwrap();
    let out = format!("{}/sim", tmp.path().display());
    let sim = tmp.path().join("sim");
    Staging::folder(&out).unwrap().commit().unwrap();
    assert_eq!(names(tmp.path()), Vec::<String>::new());
    std::fs::create_dir(&sim).unwrap();
    let s = Staging::folder(&out).unwrap();
    fill(&s, "one");
    let mut landed = Vec::new();
    s.commit_observed(|p| landed.push(p.to_path_buf())).unwrap();
    assert_eq!(landed, std::slice::from_ref(&sim));
    assert_eq!(names(&sim), ["a.zarr", "manifest.json", "truth"]);
    assert_eq!(names(tmp.path()), ["sim"]);
}

/// The files that mark a call complete land after the outputs they speak
/// for, whatever their names sort as: residual's marker in a folder in
/// use, a single scenario's sidecar beside its table.
#[test]
fn every_completion_marker_lands_last() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let res = dir.join("res");
    std::fs::create_dir(&res).unwrap();
    std::fs::write(res.join("notes.txt"), "mine").unwrap();
    let s = Staging::folder(&format!("{}/res", dir.display())).unwrap();
    let staged = PathBuf::from(s.out());
    std::fs::create_dir(staged.join("S1.zarr")).unwrap();
    std::fs::write(staged.join("residual.json"), "{}").unwrap();
    std::fs::write(staged.join("z.coords.parquet"), "").unwrap();
    let mut landed = Vec::new();
    s.commit_observed(|p| landed.push(p.to_path_buf())).unwrap();
    assert_eq!(landed.len(), 3);
    assert_eq!(landed[2], res.join("residual.json"));

    let out = format!("{}/one", dir.display());
    let s = Staging::prefix(&out).unwrap();
    std::fs::write(format!("{}.lra-simulate.json", s.out()), "{}").unwrap();
    std::fs::write(format!("{}.lra-simulate.parquet", s.out()), "").unwrap();
    std::fs::write(dir.join("one.lra-simulate.json"), "earlier").unwrap();
    let sidecar = dir.join("one.lra-simulate.json");
    let mut landed = Vec::new();
    s.commit_observed(|p| {
        if p != sidecar {
            assert!(!sidecar.exists(), "the earlier sidecar by the new table");
        }
        landed.push(p.to_path_buf());
    })
    .unwrap();
    assert_eq!(landed, [dir.join("one.lra-simulate.parquet"), sidecar]);

    // a comparison's report after its table, though it sorts first
    let s = Staging::prefix(&format!("{}/cmp", dir.display())).unwrap();
    std::fs::write(format!("{}.compare.json", s.out()), "{}").unwrap();
    std::fs::write(format!("{}.contingency.parquet", s.out()), "").unwrap();
    let mut landed = Vec::new();
    s.commit_observed(|p| landed.push(p.to_path_buf())).unwrap();
    assert_eq!(landed[1], dir.join("cmp.compare.json"));
}

/// A link to a folder is staged beside the folder it leads to, and the
/// run lands there; what could take no rename is refused at once.
#[cfg(unix)]
#[test]
fn a_linked_folder_is_staged_beside_the_folder_it_leads_to() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().canonicalize().unwrap();
    let real = dir.join("elsewhere/real");
    std::fs::create_dir_all(&real).unwrap();
    std::os::unix::fs::symlink(&real, dir.join("link")).unwrap();
    let out = format!("{}/link", dir.display());
    for round in ["empty", "in use"] {
        let s = Staging::folder(&out).unwrap();
        let staged = PathBuf::from(s.out());
        assert_eq!(staged.parent(), real.parent(), "{round}");
        assert_eq!(
            s.final_path(&format!("{}/x", s.out())),
            Some(format!("{}/x", real.display()))
        );
        std::fs::write(staged.join(format!("{round}.txt")), round).unwrap();
        std::fs::write(staged.join("manifest.json"), round).unwrap();
        s.commit().unwrap();
        assert!(dir.join("link").is_symlink(), "{round}");
        assert_eq!(text(&real.join("manifest.json")), round);
    }
    assert_eq!(names(&real), ["empty.txt", "in use.txt", "manifest.json"]);
    assert_eq!(names(&dir.join("elsewhere")), ["real"]);

    std::os::unix::fs::symlink(dir.join("nowhere"), dir.join("dangling")).unwrap();
    std::fs::write(dir.join("file"), "").unwrap();
    for (bad, why) in [
        ("dangling", "a link to no folder"),
        ("file", "not a folder"),
    ] {
        let err = Staging::folder(&format!("{}/{bad}", dir.display()))
            .err()
            .unwrap_or_else(|| panic!("{bad} was staged"));
        assert!(err.to_string().contains(why), "{err}");
    }
    assert_eq!(names(&dir), ["dangling", "elsewhere", "file", "link"]);
}

/// An out that ends in `.` or `..`, or in a slash, names an ordinary
/// folder: it is staged beside the folder it leads to, made or not.
#[test]
fn a_folder_named_with_dots_is_staged_beside_the_folder_it_leads_to() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().canonicalize().unwrap();
    let at = |rest: &str| format!("{}/{rest}", dir.display());
    std::fs::create_dir_all(dir.join("top/inner")).unwrap();
    for (out, lands) in [
        // not made yet, and its parents neither
        (at("new/deep/sim/"), dir.join("new/deep/sim")),
        (at("fresh/."), dir.join("fresh")),
        (at("slashes/.//"), dir.join("slashes")),
        // made, and named from inside it
        (at("top/inner/.."), dir.join("top")),
        (at("top/inner/."), dir.join("top/inner")),
    ] {
        let s = Staging::folder(&out).unwrap_or_else(|e| panic!("{out}: {e:#}"));
        let staged = PathBuf::from(s.out());
        assert_eq!(staged.parent(), lands.parent(), "{out}");
        std::fs::write(staged.join("manifest.json"), &out).unwrap();
        assert_eq!(
            s.final_path(&format!("{}/manifest.json", s.out())),
            Some(format!("{}/manifest.json", lands.display())),
            "{out}"
        );
        s.commit().unwrap();
        assert_eq!(text(&lands.join("manifest.json")), out);
    }
    assert_eq!(names(&dir), ["fresh", "new", "slashes", "top"]);
    assert_eq!(names(&dir.join("top")), ["inner", "manifest.json"]);

    // only the root has no folder to stage beside
    for root in ["/", "/.", "//"] {
        let err = Staging::folder(root)
            .err()
            .unwrap_or_else(|| panic!("{root} was staged"));
        assert!(
            err.to_string().contains("no folder to stage beside"),
            "{err}"
        );
    }
    let err = Staging::folder(&at("nowhere/..")).err().unwrap();
    assert!(err.to_string().contains("names no folder"), "{err}");
}

/// A folder reached through a linked folder above it is an ordinary one.
#[cfg(unix)]
#[test]
fn an_out_under_a_linked_folder_is_staged_as_any_other() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::create_dir(dir.join("real")).unwrap();
    std::os::unix::fs::symlink("real", dir.join("via")).unwrap();
    let s = Staging::folder(&format!("{}/via/sim", dir.display())).unwrap();
    fill(&s, "one");
    s.commit().unwrap();
    assert_eq!(text(&dir.join("real/sim/truth/t.parquet")), "one");
    staged_run(&dir.join("via")).commit().unwrap();
    assert!(dir.join("real/run.pinto.json").is_file());
}

/// The folders a run made above its staging folder go with it when it
/// lands nothing: an abort leaves the place as it found it. One that is
/// there already, or that something else was put in meanwhile, stays.
#[test]
fn an_abort_removes_the_folders_made_for_the_staging() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::create_dir(dir.join("mine")).unwrap();
    let at = |rest: &str| format!("{}/{rest}", dir.display());
    Staging::prefix(&at("new/deep/run")).unwrap().abort();
    Staging::folder(&at("other/deep/sim")).unwrap().abort();
    Staging::prefix(&at("mine/run")).unwrap().abort();
    assert_eq!(names(dir), ["mine"]);

    let s = Staging::prefix(&at("new/deep/run")).unwrap();
    std::fs::write(dir.join("new/theirs"), "").unwrap();
    s.abort();
    assert_eq!(names(&dir.join("new")), ["theirs"]);

    // so does a folder run that wrote nothing
    Staging::folder(&at("empty/deep/sim"))
        .unwrap()
        .commit()
        .unwrap();
    assert_eq!(names(dir), ["mine", "new"]);
}

/// A subset recorded as `lra --only` records one./// A subset recorded as `lra --only` records one.
fn subset(table: &str, pairs: usize) -> LraSubset {
    LraSubset {
        only: vec!["GENE1".into()],
        pairs,
        of: 9,
        table: table.into(),
        curve: format!("{table}.curve"),
        celltype: None,
        options: None,
    }
}

#[test]
fn another_run_s_manifest_changes_when_the_outputs_land() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let upstream = dir.join("up.pinto.json");
    PintoMetadata {
        command: "lc".into(),
        n_cells: 3,
        ..Default::default()
    }
    .write(&upstream)
    .unwrap();
    PintoMetadata::update(&upstream, |m| {
        m.outputs.lra.record_subset(subset("/t/a", 1))
    })
    .unwrap();
    let mut s = staged_run(dir);
    s.update_elsewhere(&upstream, |m| m.n_edges = Some(7));
    // a second change sees the first; the same subset again replaces it
    s.update_elsewhere(&upstream, |m| {
        assert_eq!(m.n_edges, Some(7));
        m.outputs.lra.record_subset(subset("/t/a", 2));
    });
    let before = PintoMetadata::read(&upstream).unwrap();
    assert_eq!(
        (before.n_edges, before.outputs.lra.subsets[0].pairs),
        (None, 1)
    );
    // what another process records meanwhile is kept, a list entry too
    PintoMetadata::update(&upstream, |m| {
        m.n_communities = Some(2);
        m.outputs.lra.record_subset(subset("/t/b", 5));
    })
    .unwrap();
    s.commit().unwrap();
    let m = PintoMetadata::read(&upstream).unwrap();
    assert_eq!(
        (m.n_cells, m.n_edges, m.n_communities),
        (3, Some(7), Some(2))
    );
    let recorded: Vec<(&str, usize)> = m
        .outputs
        .lra
        .subsets
        .iter()
        .map(|s| (s.table.as_str(), s.pairs))
        .collect();
    assert_eq!(recorded, [("/t/a", 2), ("/t/b", 5)]);
}

/// A run's own manifest at a final name is written when its outputs are
/// in place: anew when none is there, else the one there changed.
#[test]
fn a_run_s_own_manifest_elsewhere_is_written_when_the_outputs_land() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let own = dir.join("own.pinto.json");
    let fresh = |command: &str| PintoMetadata {
        command: command.into(),
        ..Default::default()
    };
    let mut s = staged_run(dir);
    s.write_elsewhere_if_missing(&own, fresh("lra"));
    s.update_elsewhere(&own, |m| m.n_edges = Some(4));
    assert!(!own.exists());
    s.commit().unwrap();
    let m = PintoMetadata::read(&own).unwrap();
    assert_eq!((m.command.as_str(), m.n_edges), ("lra", Some(4)));

    // one written meanwhile is kept, and changed
    PintoMetadata::update(&own, |m| m.n_communities = Some(2)).unwrap();
    let mut s = staged_run(dir);
    s.write_elsewhere_if_missing(&own, fresh("again"));
    s.update_elsewhere(&own, |m| m.n_edges = Some(5));
    s.commit().unwrap();
    let m = PintoMetadata::read(&own).unwrap();
    assert_eq!(
        (m.command.as_str(), m.n_edges, m.n_communities),
        ("lra", Some(5), Some(2))
    );
}

/// Another run's manifest that cannot be changed costs a warning, as it
/// does without staging: the run's outputs and its own manifest land.
#[test]
fn another_run_s_manifest_that_cannot_be_changed_fails_no_commit() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let (unreadable, gone) = (dir.join("bad.pinto.json"), dir.join("gone.pinto.json"));
    std::fs::write(&unreadable, "not a manifest").unwrap();
    PintoMetadata::default().write(&gone).unwrap();
    let mut s = staged_run(dir);
    s.update_elsewhere(&unreadable, |m| m.n_edges = Some(1));
    s.update_elsewhere(&gone, |m| m.n_edges = Some(1));
    std::fs::remove_file(&gone).unwrap();
    s.commit().unwrap();
    assert!(dir.join("run.pinto.json").is_file());
    assert_eq!(text(&unreadable), "not a manifest");
    assert!(!gone.exists());
}

#[test]
fn an_abort_leaves_another_run_s_manifest_alone() {
    let tmp = tempfile::tempdir().unwrap();
    let upstream = tmp.path().join("up.pinto.json");
    PintoMetadata::default().write(&upstream).unwrap();
    let before = text(&upstream);
    let mut s = staged_run(tmp.path());
    s.update_elsewhere(&upstream, |m| m.n_edges = Some(1));
    s.write_elsewhere_if_missing(&tmp.path().join("own.pinto.json"), PintoMetadata::default());
    s.abort();
    assert_eq!(text(&upstream), before);
    assert!(!tmp.path().join("own.pinto.json").exists());
}

/// A manifest of the run's own that is already there and cannot be changed
/// costs a warning, as it does without staging, and the outputs land.
#[cfg(unix)]
#[test]
fn a_run_s_own_manifest_that_is_there_and_cannot_be_changed_fails_no_commit() {
    if !permissions_bind("a_run_s_own_manifest_that_is_there_and_cannot_be_changed_fails_no_commit")
    {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let own = dir.join("own.pinto.json");
    PintoMetadata {
        command: "lra".into(),
        ..Default::default()
    }
    .write(&own)
    .unwrap();
    let before = text(&own);
    set_mode(&own, 0o444);
    let mut s = staged_run(dir);
    s.write_elsewhere_if_missing(&own, PintoMetadata::default());
    s.update_elsewhere(&own, |m| m.n_edges = Some(4));
    s.commit().unwrap();
    assert_eq!(text(&own), before);
    assert!(dir.join("run.pinto.json").is_file());
    assert!(dir.join("run.a.parquet").is_file());
}

/// A manifest of the run's own that has to be made and cannot be is an
/// error, as it is without staging: it names the manifest.
#[test]
fn a_run_s_own_manifest_that_cannot_be_made_fails_the_commit() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    // under a file, where no manifest can be made
    std::fs::write(dir.join("blocked"), "").unwrap();
    let own = dir.join("blocked/own.pinto.json");
    let mut s = staged_run(dir);
    s.write_elsewhere_if_missing(&own, PintoMetadata::default());
    s.update_elsewhere(&own, |m| m.n_edges = Some(4));
    let said = format!("{:#}", s.commit().unwrap_err());
    assert!(
        said.contains(&format!("writing {}", own.display())),
        "{said}"
    );
    // the run's manifest, its marker, did not land
    assert!(!dir.join("run.pinto.json").exists());
}
