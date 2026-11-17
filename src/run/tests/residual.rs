use super::*;
use crate::built_cli as cli;
use crate::mcp::shown_path;
use crate::run::mcp::checked;
use crate::util::metadata::PintoMetadata;
use serde_json::Value;
use std::path::Path;

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
    crate::test_support::record_tables(&mut meta, &prefix);
    meta.write(Path::new(&format!("{prefix}.pinto.json")))
        .unwrap();
    prefix
}

#[test]
fn the_residual_tool_is_listed_with_its_schema() {
    let tools = specs(&crate::built_cli());
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
        "feature_name_mode",
        "background",
    ] {
        assert!(props.contains_key(arg), "run_residual lacks {arg}");
    }
    // the choices are the command's own
    assert_eq!(props["excess"]["enum"], json!(["round", "floor"]));
    assert_eq!(props["adapt"]["enum"], json!(["none", "batch"]));
    assert!(props["feature_name_mode"]["enum"]
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
            "apply".to_string(),
            "residual".to_string(),
            format!("--from={}", shown_path(Path::new("r"))),
            "--out=/p/res".into()
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
        "apply",
        "residual",
        &format!("--from={}", shown_path(Path::new("r"))),
        &format!("--data-files={files}"),
        "--excess=floor",
        "--adapt=none",
        "--adapt-cells=20",
        "--feature-name-mode=gene",
        "--out=o",
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
    let tools = specs(&crate::built_cli());
    let spec = tools.iter().find(|t| t.name == "run_residual").unwrap();
    let run = spec.input_schema["properties"]["run"]["description"]
        .as_str()
        .unwrap();
    assert!(run.ends_with("(--from; --run also works)"), "{run}");
    let argv = residual_argv(&args(json!({"run": "r"})), "o").unwrap();
    assert!(argv[2].starts_with("--from="), "{argv:?}");
}

/// A run is taken by what its manifest records: one that records no
/// propensity output is refused while the call is set up, whatever wrote it.
#[test]
fn a_run_that_records_no_propensity_output_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("S1.zarr");
    std::fs::create_dir_all(&data).unwrap();
    let data = data.to_string_lossy().into_owned();
    let prefix = manifest(dir.path(), "lc", &[&data]);
    let a = args(json!({"run": prefix}));
    let out = dir.path().join("res").to_string_lossy().into_owned();
    residual_plan(&cli(), &a, &out).expect("a run with its tables is planned");
    PintoMetadata::update(Path::new(&format!("{prefix}.pinto.json")), |m| {
        m.outputs.propensity = None
    })
    .unwrap();
    let err = residual_plan(&cli(), &a, &out)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(err.contains("records no propensity output"), "{err}");
}

/// The reply lists what stage 2 fits, in the run's order, with the
/// coordinate columns: under subtract the residual's files, under divide
/// (the default) the run's own data files and the folder as `offset`; and
/// the call makes exactly what the plan names, with residual.json.
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
    let sub = args(json!({"run": prefix, "residual": "subtract"}));
    let (argv, plan) = residual_plan(&cli(), &sub, &out).unwrap();
    assert_eq!(argv[..2], ["apply", "residual"]);
    assert!(
        argv.contains(&"--residual=subtract".to_string()),
        "{argv:?}"
    );
    let fit = fit_lists(&plan, &out);
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

    let (_, plan) = residual_plan(&cli(), &args(json!({"run": prefix})), &out).unwrap();
    let fit = fit_lists(&plan, &out);
    assert_eq!(fit["data"], json!(plan.files), "{fit}");
    assert_eq!(fit["flags"]["offset"], json!(out), "{fit}");
    let made: Vec<String> = plan
        .made()
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        made,
        [
            format!("{out}/S2.coords.parquet"),
            format!("{out}/S1.coords.parquet"),
            format!("{out}/stage1.cells.parquet"),
            format!("{out}/stage1.profiles.parquet"),
            format!("{out}/stage1.features.parquet"),
            format!("{out}/residual.json"),
        ]
    );
}
