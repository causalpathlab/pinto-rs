//! `--atomic-outputs` through the binary, per kind of command: a run that
//! succeeds leaves its usual outputs, naming their final paths in what it
//! records and in what it says, and no staging folder; one that fails
//! leaves nothing under its final names. A failure is a natural one where
//! a command fails after it has written, else the debug build's stand-in
//! for a run that fails once all its outputs are written.

mod reference;

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

/// The debug build's switch for a run that fails (`1`) or panics (`panic`)
/// at its end; a release build has none.
const SWITCH: &str = "PINTO_TEST_FAIL_BEFORE_COMMIT";
const HAS_SWITCH: bool = cfg!(debug_assertions);

/// pinto to run in `dir` with `args`, the switch set to `switch`. lra
/// reads the empty reference of the fixture, offline.
fn pinto(dir: &Path, args: &[&str], switch: Option<&str>) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_pinto"));
    cmd.args(["--max-threads", "2"])
        .args(args)
        .current_dir(dir)
        .env_remove(SWITCH)
        .env("PINTO_REFERENCE_DIR", fixture().join("reference"))
        .env("PINTO_OFFLINE", "1");
    if let Some(value) = switch {
        cmd.env(SWITCH, value);
    }
    cmd
}

/// [`pinto`], run to its end.
fn run(dir: &Path, args: &[&str], switch: Option<&str>) -> Output {
    pinto(dir, args, switch).output().unwrap()
}

/// What a run printed, on both streams.
fn said(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn ok(dir: &Path, args: &[&str]) {
    let out = run(dir, args, None);
    assert!(
        out.status.success(),
        "pinto {args:?} failed:\n{}",
        said(&out)
    );
}

/// `args` under `--atomic-outputs`, saying all it says: the run succeeds,
/// and names no staged path.
fn ok_atomic(dir: &Path, args: &[&str]) {
    let args = [args, &["--atomic-outputs", "--verbose"]].concat();
    let out = run(dir, &args, None);
    let said = said(&out);
    assert!(out.status.success(), "pinto {args:?} failed:\n{said}");
    assert!(!said.contains(".partial-"), "pinto {args:?} said:\n{said}");
}

/// `args` under `--atomic-outputs` fail of themselves, naming no staged
/// path.
fn fails_atomic(dir: &Path, args: &[&str]) {
    let args = [args, &["--atomic-outputs"]].concat();
    let out = run(dir, &args, None);
    let said = said(&out);
    assert!(!out.status.success(), "pinto {args:?} succeeded");
    assert!(!said.contains(".partial-"), "pinto {args:?} said:\n{said}");
}

/// `args` under `--atomic-outputs` made to fail once all is written.
/// Nothing is run by a build without the switch: what the caller then
/// asserts of a failed run holds of no run at all.
fn fails_at_end(dir: &Path, args: &[&str]) {
    if !HAS_SWITCH {
        return;
    }
    let args = [args, &["--atomic-outputs"]].concat();
    let out = run(dir, &args, Some("1"));
    assert!(!out.status.success(), "pinto {args:?} succeeded");
    assert!(said(&out).contains(SWITCH), "{}", said(&out));
}

/// The names in `dir`, sorted; none when it is missing.
fn names(dir: &Path) -> Vec<String> {
    let mut n: Vec<String> = std::fs::read_dir(dir)
        .map(|d| {
            d.map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    n.sort();
    n
}

/// Every name in `dir` and below that is a staging folder or names one.
fn staged_anywhere(dir: &Path) -> Vec<String> {
    let mut found = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.to_string_lossy().contains(".partial-") {
            found.push(p.display().to_string());
        }
        if p.is_dir() {
            found.extend(staged_anywhere(&p));
        } else if p.extension().is_some_and(|x| x == "json")
            && std::fs::read_to_string(&p).unwrap().contains(".partial-")
        {
            found.push(format!("{} names a staged path", p.display()));
        }
    }
    found
}

fn assert_none_staged(dir: &Path) {
    let found = staged_anywhere(dir);
    assert!(found.is_empty(), "{found:?}");
}

/// A small simulation into `out`, its seed `seed`.
fn sim_args<'a>(seed: &'a str, out: &'a str) -> Vec<&'a str> {
    vec![
        "srt-simulate",
        "--scenario",
        "donor-confounded",
        "--set",
        "grid_side=24",
        "--set",
        "n_genes=40",
        "--set",
        "programme_genes=5",
        "--set",
        seed,
        "-o",
        out,
    ]
}

/// The simulation's first two sections and their coordinates, as a
/// command reads them.
const DATA: &[&str] = &[
    "sim/sections/S1.zarr,sim/sections/S2.zarr",
    "-c",
    "sim/sections/S1.coords.parquet,sim/sections/S2.coords.parquet",
    "--coord-column-names",
    "row_um,col_um",
];

/// lc on the simulation, writing `out`, with `extra`.
fn lc_args<'a>(out: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut a = vec!["lc"];
    a.extend(DATA);
    a.extend([
        "--no-qc",
        "--n-communities",
        "3",
        "--num-gibbs",
        "10",
        "-d",
        "8",
        "-o",
        out,
    ]);
    a.extend(extra);
    a
}

/// What every test starts from, made once by each test process in a
/// folder of its own in the build's temporary folder, so two processes
/// that share a build never share a fixture: a simulation at `sim`, an lc
/// fit of it at `fit/lc`, a pair and its programme for lra, and an empty
/// lra reference. No process removes its fixture at its end, so each
/// removes those left a day or more ago.
fn fixture() -> &'static Path {
    const NAMED: &str = "atomic-outputs-fixture-";
    static MADE: OnceLock<PathBuf> = OnceLock::new();
    MADE.get_or_init(|| {
        let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
        let day = std::time::Duration::from_secs(24 * 60 * 60);
        for e in std::fs::read_dir(tmp).into_iter().flatten().flatten() {
            let old = e
                .metadata()
                .and_then(|m| m.modified())
                .is_ok_and(|at| at.elapsed().is_ok_and(|age| age > day));
            if old && e.file_name().to_string_lossy().starts_with(NAMED) {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
        let dir = tmp.join(format!("{NAMED}{}", std::process::id()));
        // one a process of the same id left
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        reference::empty_reference(&dir);
        std::fs::write(dir.join("lr.tsv"), "ligand\treceptor\nGENE1\tGENE2\n").unwrap();
        let mut programme = String::from("ligand\treceptor\ttarget\tweight\n");
        for g in 3..=12 {
            programme.push_str(&format!("GENE1\tGENE2\tGENE{g}\t1\n"));
        }
        std::fs::write(dir.join("targets.tsv"), programme).unwrap();
        dir
    })
}

/// `from` and all in it, copied to `to`.
fn copy_all(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if src.is_dir() {
            copy_all(&src, &dst);
        } else {
            std::fs::copy(&src, &dst).unwrap();
        }
    }
}

/// A folder of a test's own, holding the fixture's `parts`.
fn scene(parts: &[&str]) -> tempfile::TempDir {
    static BUILT: OnceLock<()> = OnceLock::new();
    let fixture = fixture();
    BUILT.get_or_init(|| {
        ok(fixture, &sim_args("seed=3", "sim"));
        ok(fixture, &lc_args("fit/lc", &[]));
    });
    let tmp = tempfile::tempdir().unwrap();
    for part in parts {
        let (src, dst) = (fixture.join(part), tmp.path().join(part));
        if src.is_dir() {
            copy_all(&src, &dst);
        } else {
            std::fs::copy(&src, &dst).unwrap();
        }
    }
    tmp
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn a_fit_lands_whole_and_names_its_final_files() {
    let tmp = scene(&["sim", "fit"]);
    let d = tmp.path();
    ok_atomic(d, &lc_args("atomic/lc", &[]));
    // the same files as a plain run, and no staging folder
    let plain: Vec<String> = names(&d.join("fit"))
        .into_iter()
        .filter(|n| !n.starts_with('.'))
        .collect();
    let atomic: Vec<String> = names(&d.join("atomic"))
        .into_iter()
        .filter(|n| !n.ends_with(".lock"))
        .collect();
    assert_eq!(atomic, plain);
    assert_none_staged(d);
    let m = json(&d.join("atomic/lc.pinto.json"));
    assert_eq!(m["prefix"], "atomic/lc");
    for (slot, file) in m["outputs"].as_object().unwrap() {
        if let Some(f) = file.as_str().filter(|f| f.ends_with(".parquet")) {
            // from the working directory, or beside the manifest
            let found = d.join(f).is_file() || d.join("atomic").join(f).is_file();
            assert!(found, "{slot}: {f}");
        }
    }
}

#[test]
fn a_fit_that_fails_after_writing_leaves_nothing() {
    let tmp = scene(&["sim"]);
    let d = tmp.path();
    // a feature network that names no feature of the data: refused once
    // the cells table is written
    std::fs::write(d.join("net.tsv"), "NOPE1\tNOPE2\n").unwrap();
    let bad = ["--feature-network", "net.tsv"];
    let plain = run(d, &lc_args("plain/lc", &bad), None);
    assert!(!plain.status.success());
    assert!(
        d.join("plain/lc.cells.parquet").is_file(),
        "the failure comes after a write: {:?}",
        names(&d.join("plain"))
    );
    fails_atomic(d, &lc_args("atomic/lc", &bad));
    assert_eq!(names(&d.join("atomic")), Vec::<String>::new());
}

#[test]
fn a_fit_failing_at_its_end_leaves_an_earlier_run_as_it_was() {
    let tmp = scene(&["sim", "fit"]);
    let d = tmp.path();
    let before = std::fs::read(d.join("fit/lc.pinto.json")).unwrap();
    let listed = names(&d.join("fit"));
    fails_at_end(d, &lc_args("fit/lc", &["--seed", "9"]));
    assert_eq!(names(&d.join("fit")), listed);
    assert_eq!(std::fs::read(d.join("fit/lc.pinto.json")).unwrap(), before);
    // and a run that succeeds replaces it, manifest and all
    ok_atomic(d, &lc_args("fit/lc", &["--seed", "9"]));
    assert_ne!(std::fs::read(d.join("fit/lc.pinto.json")).unwrap(), before);
    assert_eq!(names(&d.join("fit")), listed);
}

/// A panic unwinds through the same removal as an error.
#[cfg(debug_assertions)]
#[test]
fn a_run_that_panics_leaves_nothing() {
    let tmp = scene(&["fit"]);
    let d = tmp.path();
    let compare = ["compare", "fit/lc", "fit/lc", "-o", "cmp/x"];
    let out = run(
        d,
        &[&compare[..], &["--atomic-outputs"]].concat(),
        Some("panic"),
    );
    assert!(!out.status.success());
    assert!(said(&out).contains("panicked"), "{}", said(&out));
    assert_eq!(names(&d.join("cmp")), Vec::<String>::new());
    assert_none_staged(d);
    // any other value of the switch is no switch
    let out = run(
        d,
        &[&compare[..], &["--atomic-outputs"]].concat(),
        Some("0"),
    );
    assert!(out.status.success(), "{}", said(&out));
}

#[test]
fn lra_records_into_its_run_only_once_its_outputs_land() {
    let tmp = scene(&["sim", "fit", "lr.tsv"]);
    let d = tmp.path();
    let lra = [
        "lra",
        "--from",
        "fit/lc.pinto.json",
        "--method",
        "coexpression-scores",
        "--lr-pairs",
        "lr.tsv",
    ];
    let before = std::fs::read(d.join("fit/lc.pinto.json")).unwrap();
    fails_at_end(d, &lra);
    assert!(!d.join("fit/lc.lr_scores.parquet").exists());
    assert_eq!(std::fs::read(d.join("fit/lc.pinto.json")).unwrap(), before);
    assert_none_staged(d);

    ok_atomic(d, &lra);
    let table = d.join("fit/lc.lr_scores.parquet");
    assert!(table.is_file());
    let recorded = json(&d.join("fit/lc.pinto.json"))["outputs"]["lr_scores"].clone();
    assert_eq!(
        Path::new(recorded.as_str().unwrap())
            .canonicalize()
            .unwrap(),
        table.canonicalize().unwrap()
    );
    assert_none_staged(d);
}

/// lra with an `--out` of its own and no `--from` keeps its manifest
/// there: written with the outputs, changed and never replaced by a later
/// run, and the same `--only` list run again takes the place of its
/// earlier entry.
#[test]
fn lra_with_its_own_out_changes_the_manifest_there_and_never_replaces_it() {
    let tmp = scene(&["sim", "lr.tsv", "targets.tsv"]);
    let d = tmp.path();
    let lra = |seed: &'static str| {
        let mut a = vec!["lra"];
        a.extend(DATA);
        a.extend([
            "--lr-pairs",
            "lr.tsv",
            "--targets",
            "targets.tsv",
            "--only",
            "GENE1",
            "--seed",
            seed,
            "-o",
            "own/x",
        ]);
        a
    };
    let manifest = d.join("own/x.pinto.json");
    fails_at_end(d, &lra("1"));
    assert_eq!(names(&d.join("own")), Vec::<String>::new());
    ok_atomic(d, &lra("1"));
    assert_none_staged(d);
    let first = json(&manifest);
    let subsets = first["outputs"]["lra"]["subsets"].as_array().unwrap();
    assert_eq!(subsets.len(), 1, "{first}");
    assert_eq!(subsets[0]["options"]["seed"], 1, "{first}");
    let table = subsets[0]["table"].as_str().unwrap();
    assert!(Path::new(table).is_file(), "{table}");
    assert_eq!(first["prefix"], "own/x");

    // what another run recorded there meanwhile
    let mut edited = first.clone();
    edited["n_communities"] = 77.into();
    std::fs::write(&manifest, serde_json::to_string_pretty(&edited).unwrap()).unwrap();
    let before = std::fs::read(&manifest).unwrap();
    fails_at_end(d, &lra("2"));
    assert_eq!(std::fs::read(&manifest).unwrap(), before);

    ok_atomic(d, &lra("2"));
    let second = json(&manifest);
    assert_eq!(second["n_communities"], 77, "{second}");
    let subsets = second["outputs"]["lra"]["subsets"].as_array().unwrap();
    assert_eq!(subsets.len(), 1, "{second}");
    assert_eq!(subsets[0]["options"]["seed"], 2, "{second}");
    assert_eq!(subsets[0]["table"], table);
    assert_none_staged(d);
}

/// A single lra-simulate scenario clears nothing at the final names
/// before it has succeeded: a call that fails leaves the earlier call's
/// table and sidecar as they were. Without the flag they are cleared
/// first, so they cannot pass for the failed call's.
#[test]
fn a_single_scenario_that_fails_leaves_the_earlier_call_s_outputs() {
    let tmp = scene(&["sim", "fit"]);
    let d = tmp.path();
    let earlier = ["one/x.lra-simulate.parquet", "one/x.lra-simulate.json"];
    let write_earlier = || {
        std::fs::create_dir_all(d.join("one")).unwrap();
        for f in earlier {
            std::fs::write(d.join(f), "earlier").unwrap();
        }
    };
    // no such pairs file: refused once the data is being read
    let single = [
        "lra-simulate",
        "--from",
        "fit/lc.pinto.json",
        "--one",
        "--lr-pairs",
        "nowhere.tsv",
        "-o",
        "one/x",
    ];
    write_earlier();
    assert!(!run(d, &single, None).status.success());
    assert_eq!(names(&d.join("one")), Vec::<String>::new());
    write_earlier();
    fails_atomic(d, &single);
    for f in earlier {
        assert_eq!(std::fs::read_to_string(d.join(f)).unwrap(), "earlier");
    }
    assert_eq!(names(&d.join("one")).len(), 2);
}

#[test]
fn a_simulation_folder_lands_whole_or_not_at_all() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    let small = |seed: &'static str| {
        let mut a = sim_args(seed, "sim");
        a[4] = "grid_side=10";
        a
    };
    fails_at_end(d, &small("seed=3"));
    assert_eq!(names(d), Vec::<String>::new());

    ok_atomic(d, &small("seed=3"));
    assert_eq!(
        names(&d.join("sim")),
        ["manifest.json", "sections", "truth"]
    );
    assert_none_staged(d);

    // into a folder in use: file by file, the rest left as it was
    std::fs::write(d.join("sim/notes.txt"), "mine").unwrap();
    let before = std::fs::read(d.join("sim/manifest.json")).unwrap();
    fails_at_end(d, &small("seed=4"));
    assert_eq!(std::fs::read(d.join("sim/manifest.json")).unwrap(), before);
    ok_atomic(d, &small("seed=4"));
    assert_ne!(std::fs::read(d.join("sim/manifest.json")).unwrap(), before);
    assert!(d.join("sim/notes.txt").is_file());
    assert_eq!(
        names(&d.join("sim")),
        ["manifest.json", "notes.txt", "sections", "truth"]
    );
    assert_none_staged(d);
}

#[test]
fn a_residual_folder_lands_whole_or_not_at_all() {
    let tmp = scene(&["sim", "fit"]);
    let d = tmp.path();
    let residual = ["residual", "--from", "fit/lc.pinto.json", "-o", "res"];
    fails_at_end(d, &residual);
    assert!(!d.join("res").exists());
    // into a folder in use: file by file, the rest left as it was
    std::fs::create_dir(d.join("res")).unwrap();
    std::fs::write(d.join("res/notes.txt"), "mine").unwrap();
    ok_atomic(d, &residual);
    assert_eq!(
        names(&d.join("res")),
        [
            "S1.coords.parquet",
            "S1.zarr",
            "S2.coords.parquet",
            "S2.zarr",
            "notes.txt",
            "residual.json"
        ]
    );
    assert_none_staged(d);
    // its outputs at their final names are still not written over
    let out = run(d, &[&residual[..], &["--atomic-outputs"]].concat(), None);
    assert!(!out.status.success());
    let said = said(&out);
    assert!(
        said.contains("exists") && !said.contains(".partial-"),
        "{said}"
    );
}

#[test]
fn a_comparison_lands_whole_or_not_at_all() {
    let tmp = scene(&["fit"]);
    let d = tmp.path();
    let compare = ["compare", "fit/lc", "fit/lc", "-o", "cmp/x"];
    fails_at_end(d, &compare);
    assert_eq!(names(&d.join("cmp")), Vec::<String>::new());
    ok_atomic(d, &compare);
    assert_eq!(
        names(&d.join("cmp")),
        ["x.compare.json", "x.contingency.parquet"]
    );
    assert_none_staged(d);
    // a comparison at its final names is still not written over
    let out = run(d, &[&compare[..], &["--atomic-outputs"]].concat(), None);
    assert!(!out.status.success());
    let said = said(&out);
    assert!(said.contains("cmp/x.compare.json exists"), "{said}");
}

#[test]
fn a_file_output_lands_whole_or_not_at_all() {
    let tmp = scene(&["sim"]);
    let d = tmp.path();
    let baseline = ["srt-sim-baseline", "--truth", "sim", "-o", "base/b.tsv"];
    fails_at_end(d, &baseline);
    assert_eq!(names(&d.join("base")), Vec::<String>::new());
    ok_atomic(d, &baseline);
    assert_eq!(names(&d.join("base")), ["b.tsv"]);

    let score = [
        "srt-sim-score",
        "--truth",
        "sim",
        "--labels",
        "base/b.tsv",
        "-o",
        "base/s.json",
    ];
    fails_at_end(d, &score);
    assert_eq!(names(&d.join("base")), ["b.tsv"]);
    ok_atomic(d, &score);
    assert_eq!(names(&d.join("base")), ["b.tsv", "s.json"]);
    assert_none_staged(d);
}

/// The other fits, each from the simulation: delta-svd, propensity on the
/// tables it wrote, and a baseline.
#[test]
fn every_kind_of_fit_lands_whole_and_names_its_final_files() {
    let tmp = scene(&["sim"]);
    let d = tmp.path();
    let fit = |method: &'static str, out: &'static str, extra: &[&'static str]| {
        let mut a = vec![method];
        a.extend(DATA);
        a.extend(["-o", out]);
        a.extend(extra);
        a
    };
    let dsvd = fit(
        "delta-svd",
        "atomic/dsvd",
        &["--no-qc", "--n-latent-topics", "5", "--n-communities", "3"],
    );
    fails_at_end(d, &dsvd);
    assert_eq!(names(&d.join("atomic")), Vec::<String>::new());
    ok_atomic(d, &dsvd);
    ok_atomic(
        d,
        &[
            "propensity",
            "-z",
            "atomic/dsvd.latent.parquet",
            "-e",
            "atomic/dsvd.coord_pairs.parquet",
            "--n-communities",
            "3",
            "-o",
            "atomic/prop",
        ],
    );
    ok_atomic(
        d,
        &fit(
            "spatialde2",
            "atomic/sde",
            &["--n-clusters", "3", "--neighbors", "8"],
        ),
    );
    for run in ["dsvd", "prop", "sde"] {
        let m = json(&d.join(format!("atomic/{run}.pinto.json")));
        assert_eq!(m["prefix"], format!("atomic/{run}"), "{run}");
        let table = format!("atomic/{run}.propensity.parquet");
        assert!(d.join(&table).is_file(), "{table}");
    }
    assert_none_staged(d);
}

/// An earlier manifest that could only be written in place is refused at
/// once, before the fit does any work, and the earlier run stays whole.
#[cfg(unix)]
#[test]
fn a_fit_over_a_read_only_manifest_is_refused_at_once() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = scene(&["fit"]);
    let d = tmp.path();
    let manifest = d.join("fit/lc.pinto.json");
    std::fs::set_permissions(&manifest, std::fs::Permissions::from_mode(0o444)).unwrap();
    let before = std::fs::read(&manifest).unwrap();
    let listed = names(&d.join("fit"));
    // the data is not there: the refusal comes before it is looked for
    let out = run(
        d,
        &[&lc_args("fit/lc", &[])[..], &["--atomic-outputs"]].concat(),
        None,
    );
    let said = said(&out);
    assert!(!out.status.success(), "{said}");
    assert!(
        said.contains("fit/lc.pinto.json is read-only") && said.contains("--atomic-outputs"),
        "{said}"
    );
    assert_eq!(names(&d.join("fit")), listed);
    assert_eq!(std::fs::read(&manifest).unwrap(), before);
}

/// `--out .` names the folder the run stands in: it is staged beside that
/// folder and filled, and stays the same folder.
#[cfg(unix)]
#[test]
fn a_simulation_into_the_folder_it_stands_in_fills_that_folder() {
    use std::os::unix::fs::MetadataExt;
    let tmp = tempfile::tempdir().unwrap();
    let here = tmp.path().join("here");
    std::fs::create_dir(&here).unwrap();
    let inode = here.metadata().unwrap().ino();
    let mut small = sim_args("seed=3", ".");
    small[4] = "grid_side=10";
    fails_at_end(&here, &small);
    assert_eq!(names(&here), Vec::<String>::new());
    ok_atomic(&here, &small);
    assert_eq!(names(&here), ["manifest.json", "sections", "truth"]);
    assert_eq!(here.metadata().unwrap().ino(), inode, "the folder itself");
    assert_eq!(names(tmp.path()), ["here"]);
    assert_none_staged(tmp.path());
}

/// A comparison of the fixture's fit with itself into `out`, under the
/// flag.
fn compare_atomic(out: &'static str) -> [&'static str; 6] {
    ["compare", "fit/lc", "fit/lc", "-o", out, "--atomic-outputs"]
}

/// What a run says on stdout at its end comes once its outputs are in
/// place, the same lines as without the flag; a run that fails says none
/// of it.
#[test]
fn a_closing_summary_follows_the_outputs() {
    let tmp = scene(&["fit"]);
    let d = tmp.path();
    let out = run(d, &compare_atomic("cmp/said"), None);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "{}", said(&out));
    let wrote = "wrote cmp/said.compare.json and cmp/said.contingency.parquet\n";
    assert!(
        stdout.contains("ARI") && stdout.ends_with(wrote),
        "{stdout}"
    );
    let plain = run(d, &compare_atomic("cmp/plain")[..5], None);
    assert_eq!(
        stdout.replace("cmp/said", "cmp/plain"),
        String::from_utf8_lossy(&plain.stdout)
    );
    if HAS_SWITCH {
        let out = run(d, &compare_atomic("cmp/failed"), Some("1"));
        assert!(!out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout), "");
    }
}

/// A reader that has gone ends a run that writes to it. It cannot end a
/// staged run with its outputs still staged: they are in place before the
/// run writes its summary.
#[test]
fn a_closed_stdout_ends_no_run_before_its_outputs_land() {
    let tmp = scene(&["fit"]);
    let d = tmp.path();
    let mut child = pinto(d, &compare_atomic("cmp/unread"), None)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // closed before the run has anything to say
    drop(child.stdout.take());
    child.wait().unwrap();
    assert_eq!(
        names(&d.join("cmp")),
        ["unread.compare.json", "unread.contingency.parquet"]
    );
    assert_none_staged(d);
}
