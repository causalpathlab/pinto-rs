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
