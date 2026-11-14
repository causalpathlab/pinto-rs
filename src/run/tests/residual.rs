use super::*;
use crate::mcp::shown_path;
use crate::run::mcp::checked;
use crate::util::metadata::PintoMetadata;
use serde_json::Value;
use std::path::Path;

fn cli() -> clap::Command {
    use clap::CommandFactory;
    let mut cli = crate::Cli::command();
    cli.build();
    cli
}

fn args(v: Value) -> ResidualToolArgs {
    crate::mcp::parse(v).unwrap()
}

/// A manifest of a `command` run on `files`, with coordinate columns
/// `CT1_x`, `CT1_y`, at `{dir}/fit`: the prefix.
fn manifest(dir: &Path, command: &str, files: &[&str]) -> String {
    let prefix = dir.join("fit").to_string_lossy().into_owned();
    let mut meta = PintoMetadata {
        command: command.into(),
        prefix: prefix.clone(),
        data_files: Some(files.iter().map(|f| f.to_string()).collect()),
        ..Default::default()
    };
    meta.outputs.coord_columns = Some(vec!["CT1_x".into(), "CT1_y".into()]);
    meta.write(Path::new(&format!("{prefix}.pinto.json")))
        .unwrap();
    prefix
}

#[test]
fn the_residual_tool_is_listed_with_its_schema() {
    let tools = specs();
    let spec = tools.iter().find(|t| t.name == "run_residual").unwrap();
    assert_eq!(spec.input_schema["additionalProperties"], false);
    assert_eq!(spec.input_schema["required"], json!(["run"]));
    let props = spec.input_schema["properties"].as_object().unwrap();
    for arg in [
        "run",
        "out",
        "data_files",
        "excess",
        "adapt",
        "adapt_cells",
        "feature_names",
        "background",
    ] {
        assert!(props.contains_key(arg), "run_residual lacks {arg}");
    }
    // the choices are the command's own
    assert_eq!(props["excess"]["enum"], json!(["round", "floor"]));
    assert_eq!(props["adapt"]["enum"], json!(["none", "batch"]));
    assert!(props["feature_names"]["enum"]
        .as_array()
        .unwrap()
        .contains(&json!("exact")));
}

#[test]
fn a_residual_call_becomes_its_command_line_with_only_what_was_given() {
    let bare = residual_argv(&args(json!({"run": "r"})), "/p/res").unwrap();
    assert_eq!(
        bare,
        [
            "residual",
            "--from",
            &shown_path(Path::new("r")),
            "-o",
            "/p/res"
        ]
    );
    checked(&cli(), &bare).unwrap();

    let full = args(json!({
        "run": "r",
        "data_files": ["a.zarr", "b.zarr"],
        "excess": "floor",
        "adapt": "none",
        "adapt_cells": 20.0,
        "feature_names": "gene",
    }));
    let files = format!(
        "{},{}",
        shown_path(Path::new("a.zarr")),
        shown_path(Path::new("b.zarr"))
    );
    let argv = residual_argv(&full, "o").unwrap();
    let want: Vec<String> = [
        "residual",
        "--from",
        &shown_path(Path::new("r")),
        "--data-files",
        &files,
        "--excess=floor",
        "--adapt=none",
        "--adapt-cells=20",
        "--feature-names=gene",
        "-o",
        "o",
    ]
    .map(String::from)
    .to_vec();
    assert_eq!(argv, want);
    let parsed: ResidualArgs = parsed(&cli(), &argv).unwrap();
    assert_eq!(parsed.excess, Excess::Floor);
    assert_eq!(parsed.adapt, Adapt::None);
    assert_eq!(parsed.adapt_cells, 20.0);

    let comma = args(json!({"run": "r", "data_files": ["a,b.zarr"]}));
    assert!(residual_argv(&comma, "o").is_err());
}

/// A value the command refuses is refused by the command's own parser,
/// with its own reason: a negative number is one token, not read as a
/// flag.
#[test]
fn a_value_the_command_refuses_is_refused_with_its_reason() {
    for x in [-1.0, -0.5] {
        let bad = args(json!({"run": "r", "adapt_cells": x}));
        let argv = residual_argv(&bad, "o").unwrap();
        let err = parsed::<ResidualArgs>(&cli(), &argv)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be finite and not negative"), "{err}");
        assert!(err.contains("adapt-cells"), "{err}");
    }
    let good = args(json!({"run": "r", "adapt_cells": 0.5}));
    let argv = residual_argv(&good, "o").unwrap();
    let parsed: ResidualArgs = parsed(&cli(), &argv).unwrap();
    assert_eq!(parsed.adapt_cells, 0.5);
}

/// The run is named by `--from`, the command's own flag; the schema says
/// `--run` works too.
#[test]
fn the_run_is_named_by_the_commands_own_flag() {
    let tools = specs();
    let spec = tools.iter().find(|t| t.name == "run_residual").unwrap();
    let run = spec.input_schema["properties"]["run"]["description"]
        .as_str()
        .unwrap();
    assert!(run.ends_with("(--from; --run also works)"), "{run}");
    let argv = residual_argv(&args(json!({"run": "r"})), "o").unwrap();
    assert_eq!(argv[1], "--from");
}

/// Only an lc, cage or dsvd run is taken, refused while the call is set up.
#[test]
fn a_run_that_is_not_lc_cage_or_dsvd_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("S1.zarr");
    std::fs::create_dir_all(&data).unwrap();
    let data = data.to_string_lossy().into_owned();
    let prefix = manifest(dir.path(), "srt-simulate", &[&data]);
    let a = args(json!({"run": prefix}));
    let out = dir.path().join("res").to_string_lossy().into_owned();
    let err = residual_plan(&cli(), &a, &out)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(err.contains("lc, cage or dsvd"), "{err}");
}

/// The reply lists the files `pinto residual` writes, in the run's order,
/// with the coordinate columns they carry, and the call makes exactly them
/// with residual.json.
#[test]
fn the_reply_lists_the_residual_files_to_fit_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let files: Vec<String> = ["S2.zarr", "S1.zarr"]
        .iter()
        .map(|f| {
            let p = dir.path().join("raw").join(f);
            std::fs::create_dir_all(&p).unwrap();
            p.to_string_lossy().into_owned()
        })
        .collect();
    let refs: Vec<&str> = files.iter().map(String::as_str).collect();
    let prefix = manifest(dir.path(), "lc", &refs);
    let out = dir.path().join("res").to_string_lossy().into_owned();
    let (argv, plan) = residual_plan(&cli(), &args(json!({"run": prefix})), &out).unwrap();
    assert_eq!(argv[0], "residual");
    let fit = fit_lists(&plan);
    assert_eq!(
        fit,
        json!({
            "data": [format!("{out}/S2.zarr"), format!("{out}/S1.zarr")],
            "coords": [format!("{out}/S2.coords.parquet"), format!("{out}/S1.coords.parquet")],
            "flags": {"coord-column-names": "CT1_x,CT1_y"},
        })
    );
    let made: Vec<String> = plan
        .made()
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        made,
        [
            format!("{out}/S2.zarr"),
            format!("{out}/S2.coords.parquet"),
            format!("{out}/S1.zarr"),
            format!("{out}/S1.coords.parquet"),
            format!("{out}/residual.json"),
        ]
    );
}
