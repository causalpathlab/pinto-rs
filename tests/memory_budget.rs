//! `PINTO_MEMORY_BUDGET` as a runner sets it per child process: a value
//! that is not a number of bytes stops the command before it reads data.

use std::process::Command;

#[test]
fn a_memory_budget_that_does_not_parse_stops_lra_simulate_at_the_start() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("x");
    let run = |budget: &str| {
        Command::new(env!("CARGO_BIN_EXE_pinto"))
            .args([
                "lra-simulate",
                "missing.zarr",
                "--coord",
                "missing.csv",
                "--one",
                "-o",
            ])
            .arg(&out)
            .env("PINTO_MEMORY_BUDGET", budget)
            .current_dir(dir.path())
            .output()
            .unwrap()
    };
    let refused = run("lots");
    assert!(!refused.status.success());
    let e = String::from_utf8_lossy(&refused.stderr);
    assert!(
        e.contains("PINTO_MEMORY_BUDGET") && !e.contains("missing.zarr"),
        "{e}"
    );
    // A budget that parses gets as far as the data, which is missing.
    let read = run("4000000000");
    assert!(!read.status.success());
    let e = String::from_utf8_lossy(&read.stderr);
    assert!(!e.contains("PINTO_MEMORY_BUDGET"), "{e}");
}

#[test]
fn a_memory_budget_that_does_not_parse_stops_the_spatial_baselines_at_the_start() {
    let dir = tempfile::tempdir().unwrap();
    for cmd in ["stagate", "spatialde2"] {
        let run = |budget: &str| {
            Command::new(env!("CARGO_BIN_EXE_pinto"))
                .args([cmd, "missing.zarr", "--coord", "missing.csv", "-o", "x"])
                .env("PINTO_MEMORY_BUDGET", budget)
                .current_dir(dir.path())
                .output()
                .unwrap()
        };
        let refused = run("lots");
        assert!(!refused.status.success());
        let e = String::from_utf8_lossy(&refused.stderr);
        assert!(
            e.contains("PINTO_MEMORY_BUDGET") && !e.contains("missing.zarr"),
            "{cmd}: {e}"
        );
        let read = run("4000000000");
        assert!(!read.status.success());
        let e = String::from_utf8_lossy(&read.stderr);
        assert!(!e.contains("PINTO_MEMORY_BUDGET"), "{cmd}: {e}");
    }
}

/// A budget that is not UTF-8 is refused, not taken as unset.
#[cfg(unix)]
#[test]
fn a_memory_budget_that_is_not_utf8_stops_the_command_at_the_start() {
    use std::os::unix::ffi::OsStrExt;
    let dir = tempfile::tempdir().unwrap();
    for cmd in ["stagate", "spatialde2", "lra-simulate"] {
        let mut argv = vec![cmd, "missing.zarr", "--coord", "missing.csv", "-o", "x"];
        if cmd == "lra-simulate" {
            argv.insert(4, "--one");
        }
        let out = Command::new(env!("CARGO_BIN_EXE_pinto"))
            .args(&argv)
            .env(
                "PINTO_MEMORY_BUDGET",
                std::ffi::OsStr::from_bytes(b"\xff4000"),
            )
            .current_dir(dir.path())
            .output()
            .unwrap();
        assert!(!out.status.success());
        let e = String::from_utf8_lossy(&out.stderr);
        assert!(
            e.contains("PINTO_MEMORY_BUDGET") && !e.contains("missing.zarr"),
            "{cmd}: {e}"
        );
    }
}
