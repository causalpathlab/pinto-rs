use super::*;

fn flags(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn an_ablation_fit_is_named_for_its_kind_and_parent() {
    let f = flags(json!({"permute-coords": "block", "ablation-parent": "res/base.pinto.json"}));
    assert_eq!(
        ablation_summary(&f).unwrap(),
        "coord_permutation ablation of res/base.pinto.json"
    );
    let f = flags(json!({"--mask-cells": "region.txt"}));
    assert_eq!(
        ablation_summary(&f).unwrap(),
        "cell_mask ablation of an unnamed parent"
    );
    assert!(ablation_summary(&flags(json!({"n-communities": 12}))).is_none());
}

#[test]
fn a_run_given_by_prefix_is_named_by_its_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run");
    let prefix = prefix.to_str().unwrap();
    assert_eq!(as_run(prefix), shown_path(Path::new(prefix)));
    std::fs::write(format!("{prefix}.pinto.json"), "{}").unwrap();
    assert!(as_run(prefix).ends_with("run.pinto.json"));
    let args = CompareArgs {
        a: Runs(vec![format!("{prefix}.pinto.json")]),
        b: Runs(vec!["x/other".into()]),
        out: None,
        level_a: Some("draft".into()),
        level_b: Some("final".into()),
        tau: None,
        paired: false,
        donor_map: None,
        bootstrap: None,
    };
    assert_eq!(default_name(&args), "run.draft_vs_other");
}

#[test]
fn the_lra_tools_are_listed_and_take_no_unknown_arguments() {
    let tools = specs();
    for name in ["run_lra", "run_lra_simulate"] {
        let spec = tools
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is not listed"));
        assert_eq!(spec.input_schema["additionalProperties"], false, "{name}");
    }
    let lra = tools.iter().find(|t| t.name == "run_lra").unwrap();
    let props = lra.input_schema["properties"].as_object().unwrap();
    for arg in [
        "run",
        "method",
        "out",
        "lr_pairs",
        "targets",
        "species",
        "background",
    ] {
        assert!(props.contains_key(arg), "run_lra lacks {arg}");
    }
    // The schema's methods are the command line's.
    use clap::ValueEnum;
    let methods: Vec<String> = crate::lr_activity::args::LraMethod::value_variants()
        .iter()
        .map(|m| m.to_possible_value().unwrap().get_name().to_string())
        .collect();
    assert_eq!(props["method"]["enum"], json!(methods));
}

#[test]
fn the_lra_tools_take_every_flag_the_command_line_shows() {
    use clap::CommandFactory;
    let cli = crate::Cli::command();
    let tools = specs();
    for (tool, command) in [("run_lra", "lra"), ("run_lra_simulate", "lra-simulate")] {
        let spec = tools.iter().find(|t| t.name == tool).unwrap();
        let props = spec.input_schema["properties"].as_object().unwrap();
        let flags = cli.find_subcommand(command).unwrap().get_arguments();
        for arg in flags.filter(|a| !a.is_hide_set()) {
            let id = arg.get_id().as_str();
            let name = if id == "from" { "run" } else { id };
            assert!(props.contains_key(name), "{tool} lacks {name}");
        }
    }
}

/// A writer the test reads back.
#[derive(Clone, Default)]
struct Sink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `name` called once on a server of the run tools: whether it failed,
/// and what it said.
fn call_once(name: &str, args: Value) -> (bool, String) {
    let call = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": name, "arguments": args}});
    let sink = Sink::default();
    let info = legume_mcp::ServerInfo {
        name: "test".into(),
        version: "0.0.0".into(),
        instructions: None,
    };
    let input = std::io::Cursor::new(format!("{call}\n"));
    legume_mcp::serve_on(input, sink.clone(), info, Tools::default()).unwrap();
    let text = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
    let reply: Value = text
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|m| m["id"] == json!(1))
        .unwrap();
    let result = &reply["result"];
    let said = result["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    (result["isError"] == json!(true), said)
}

/// A run of four cells at the corners of a unit square, as its manifest
/// lists it; its prefix.
fn square_run(dir: &Path) -> String {
    use legume_numeric::matrix::traits::IoOps;
    let prefix = dir.join("run").to_string_lossy().into_owned();
    let names: Vec<Box<str>> = (0..4).map(|i| format!("c{i}@S1").into()).collect();
    let xy = nalgebra::DMatrix::<f32>::from_row_slice(4, 2, &[0., 0., 1., 0., 0., 1., 1., 1.]);
    let cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    xy.to_parquet_with_names(
        &format!("{prefix}.cells.parquet"),
        (Some(&names), Some("cell")),
        Some(&cols),
    )
    .unwrap();
    let prop = nalgebra::DMatrix::<f32>::from_element(4, 2, 0.5);
    let comms: Vec<Box<str>> = vec!["C0".into(), "C1".into()];
    prop.to_parquet_with_names(
        &format!("{prefix}.propensity.parquet"),
        (Some(&names), Some("cell")),
        Some(&comms),
    )
    .unwrap();
    let manifest = json!({
        "command": "lc", "version": "0", "timestamp": "", "prefix": prefix,
        "n_cells": 4, "n_features": 1,
        "outputs": {
            "cells": format!("{prefix}.cells.parquet"),
            "propensity": format!("{prefix}.propensity.parquet"),
        },
    });
    std::fs::write(format!("{prefix}.pinto.json"), manifest.to_string()).unwrap();
    prefix
}

/// `run_region` writes its cell list into a folder it makes.
#[test]
fn run_region_makes_the_folder_of_its_out() {
    let dir = tempfile::tempdir().unwrap();
    let run = square_run(dir.path());
    let out = dir.path().join("new/sub/region");
    let (failed, said) = call_once(
        "run_region",
        json!({"run": run, "polygon": [[-1, -1], [2, -1], [2, 2], [-1, 2]],
               "out": out.to_str().unwrap()}),
    );
    assert!(!failed, "{said}");
    let list = std::fs::read_to_string(format!("{}.cells.txt", out.display())).unwrap();
    assert_eq!(list.lines().count(), 5, "{list}");
}

/// A GeoJSON `run_region` cannot read is named as its `geojson`, from the
/// root, not as a flag it was never given.
#[test]
fn run_region_names_its_geojson_when_it_cannot_read_it() {
    let dir = tempfile::tempdir().unwrap();
    let run = square_run(dir.path());
    // Given relative to where the server runs.
    let missing = crate::test_support::from_cwd(&dir.path().join("missing.geojson"));
    let (failed, said) = call_once(
        "run_region",
        json!({"run": run, "geojson": missing.to_str().unwrap()}),
    );
    assert!(failed);
    let named = format!("geojson {}", shown_path(&missing));
    assert!(said.contains(&named), "{said}");
    assert!(!said.contains("--mask-cells"), "{said}");
}

/// `run_predict` refuses a model that is not there before it starts a job.
#[test]
fn run_predict_refuses_a_model_that_is_not_there() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("nope");
    let data = dir.path().join("d.zarr");
    let (failed, said) = call_once(
        "run_predict",
        json!({"model": model.to_str().unwrap(), "data": [data.to_str().unwrap()],
               "out": dir.path().join("pred").to_str().unwrap(), "background": false}),
    );
    assert!(failed, "{said}");
    assert!(
        said.contains(&format!("{}.pinto.json", model.display())),
        "{said}"
    );
    assert!(said.contains("not there"), "{said}");
}

/// A tau `pinto compare` refuses is refused with its reason: the negative
/// number reaches its check as the flag's value, not as a flag of its own.
#[test]
fn run_compare_refuses_a_negative_tau_with_the_command_s_reason() {
    let dir = tempfile::tempdir().unwrap();
    let run = square_run(dir.path());
    let out = dir.path().join("cmp");
    let (failed, said) = call_once(
        "run_compare",
        json!({"a": run, "b": run, "tau": -0.5, "out": out.to_str().unwrap()}),
    );
    assert!(failed, "{said}");
    assert!(said.contains("-0.5 is not in (0, 1]"), "{said}");
    assert!(!said.contains("unexpected argument"), "{said}");
}
