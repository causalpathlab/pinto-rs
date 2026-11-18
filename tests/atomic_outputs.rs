//! `--atomic-outputs` through the binary, on lc: a run that succeeds
//! leaves its usual outputs, naming their final paths in what it records
//! and in what it says, and no staging folder; one that fails after it has
//! written leaves nothing under its final names; one over a read-only
//! manifest is refused at once. The staging itself (every kind of output,
//! failures part way, other runs' manifests) is `util::staging`'s tests.

mod staging;

use staging::assert_none_staged;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::sync::OnceLock;

/// pinto run to its end in `dir` with `args`, without the debug build's
/// switch for a run that fails at its end.
fn run(dir: &Path, args: &[&str]) -> Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_pinto"))
        .args(["--max-threads", "2"])
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap()
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
    let out = run(dir, args);
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
    let out = run(dir, &args);
    let said = said(&out);
    assert!(out.status.success(), "pinto {args:?} failed:\n{said}");
    assert!(!said.contains(".partial-"), "pinto {args:?} said:\n{said}");
}

/// `args` under `--atomic-outputs` fail of themselves, naming no staged
/// path.
fn fails_atomic(dir: &Path, args: &[&str]) {
    let args = [args, &["--atomic-outputs"]].concat();
    let out = run(dir, &args);
    let said = said(&out);
    assert!(!out.status.success(), "pinto {args:?} succeeded");
    assert!(!said.contains(".partial-"), "pinto {args:?} said:\n{said}");
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

/// A small simulation into `out`, its seed `seed`.
fn sim_args<'a>(seed: &'a str, out: &'a str) -> Vec<&'a str> {
    vec![
        "sim",
        "new",
        "--scenario",
        "donor-confounded",
        "--set",
        "grid_side=24",
        "--set",
        "n_genes=40",
        "--set",
        "program_genes=5",
        "--set",
        seed,
        "-o",
        out,
    ]
}

/// The simulation's first two sections and their coordinates, as a
/// command reads them.
const DATA: &[&str] = &[
    "sim/sections/S1.zarr.zip,sim/sections/S2.zarr.zip",
    "-c",
    "sim/sections/S1.coords.parquet,sim/sections/S2.coords.parquet",
    "--coord-column-names",
    "row_um,col_um",
];

/// lc on the simulation, writing `out`, with `extra`.
fn lc_args<'a>(out: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut a = vec!["fit", "lc"];
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
/// that share a build never share a fixture: a simulation at `sim` and an
/// lc fit of it at `fit/lc`. No process removes its fixture at its end, so
/// each removes those left a day or more ago.
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
    let plain = run(d, &lc_args("plain/lc", &bad));
    assert!(!plain.status.success());
    assert!(
        d.join("plain/lc.cells.parquet").is_file(),
        "the failure comes after a write: {:?}",
        names(&d.join("plain"))
    );
    fails_atomic(d, &lc_args("atomic/lc", &bad));
    assert_eq!(names(&d.join("atomic")), Vec::<String>::new());
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
