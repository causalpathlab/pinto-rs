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
        level: Some("draft,final".into()),
        tau: None,
        paired: false,
        donor_map: None,
        bootstrap: None,
        max_threads: None,
        atomic_outputs: None,
    };
    assert_eq!(default_name(&args), "run.draft_vs_other");
}

#[test]
fn the_lra_tools_are_listed_and_take_no_unknown_arguments() {
    let tools = specs(&crate::built_cli());
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
    assert!(said.contains("no such manifest"), "{said}");
}

/// `run_methods` lists every method `pinto run` offers, the spatial
/// baselines with them, each taking coordinate and batch files; with a
/// baseline it gives that baseline's flags, as with lc.
#[test]
fn run_methods_lists_the_baselines_as_it_lists_lc() {
    let tools = Tools::default();
    let first = |c: Vec<Content>| match c.into_iter().next() {
        Some(Content::Json(v)) => v,
        _ => panic!("no JSON reply"),
    };
    let all = first(tools.methods(MethodsArgs { method: None }).unwrap());
    let listed = all["methods"].as_array().unwrap();
    let names: Vec<&str> = listed
        .iter()
        .map(|m| m["method"].as_str().unwrap())
        .collect();
    assert_eq!(names, METHODS);
    for m in listed {
        assert_eq!(m["coords"], true, "{m}");
        assert_eq!(m["batches"], true, "{m}");
    }
    for &name in BASELINES {
        let one = first(
            tools
                .methods(MethodsArgs {
                    method: Some(name.into()),
                })
                .unwrap(),
        );
        assert_eq!(one["method"], name);
        let flags: Vec<&str> = one["flags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["flag"].as_str().unwrap())
            .collect();
        assert!(flags.contains(&"--seed"), "{name}: {flags:?}");
        assert!(!flags.contains(&"--out"), "{name}: {flags:?}");
    }
    assert!(tools
        .methods(MethodsArgs {
            method: Some("nope".into())
        })
        .is_err());
}

/// `run_methods` lists cage's choices, the inclusion mode and the module
/// count, and leaves its tuning knobs (hidden flags) out.
#[test]
fn run_methods_lists_cage_s_choices_and_no_hidden_knob() {
    let tools = Tools::default();
    let reply = match tools
        .methods(MethodsArgs {
            method: Some("cage".into()),
        })
        .unwrap()
        .into_iter()
        .next()
    {
        Some(Content::Json(v)) => v,
        _ => panic!("no JSON reply"),
    };
    let flags = reply["flags"].as_array().unwrap();
    let flag = |name: &str| flags.iter().find(|f| f["flag"] == name);
    let inclusion = flag("--feature-inclusion").expect("listed");
    assert_eq!(inclusion["choices"], json!(["all", "moran"]));
    // the module count's help says what its default is and how to turn it off
    let modules = flag("--feature-modules").expect("listed")["help"]
        .as_str()
        .expect("a help line");
    for word in ["auto (default)", "0 turns them off"] {
        assert!(modules.contains(word), "{modules}");
    }
    assert!(flag("--inclusion-blocks").is_none());
    assert!(flags.iter().all(|f| f.get("advanced").is_none()));
}

/// `run_methods` lists `--feature-modules` for every method, its help
/// naming the default and how to turn it off.
#[test]
fn run_methods_lists_feature_modules_for_every_method() {
    let tools = Tools::default();
    for method in [
        "lc",
        "dsvd",
        "cage",
        "cage-cell",
        "spatialde2",
        "stagate",
        "spatialpca",
        "mefisto",
        "spatialde2-pb",
        "stagate-pb",
        "spatialpca-pb",
        "mefisto-pb",
    ] {
        let reply = match tools
            .methods(MethodsArgs {
                method: Some(method.into()),
            })
            .unwrap()
            .into_iter()
            .next()
        {
            Some(Content::Json(v)) => v,
            _ => panic!("no JSON reply"),
        };
        let flags = reply["flags"].as_array().unwrap();
        let help = flags
            .iter()
            .find(|f| f["flag"] == "--feature-modules")
            .unwrap_or_else(|| panic!("{method}: --feature-modules not listed"))["help"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(help.contains("auto (default)"), "{method}: {help}");
    }
}

/// run_fit's method is one of METHODS, and a baseline's fit is checked as
/// lc's is: an unknown flag is refused, and so is a run already there.
#[test]
fn run_fit_takes_a_baseline_as_it_takes_lc() {
    let spec = specs(&crate::built_cli())
        .into_iter()
        .find(|t| t.name == "run_fit")
        .unwrap();
    assert_eq!(
        spec.input_schema["properties"]["method"]["enum"],
        json!(METHODS)
    );
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("m");
    let out = out.to_str().unwrap();
    for &name in BASELINES {
        let (failed, said) = call_once(
            "run_fit",
            json!({"method": name, "data": ["a.zarr"], "coords": ["a.csv"],
                   "out": out, "flags": {"no-such-flag": 1}}),
        );
        assert!(failed && said.contains("no flag --no-such-flag"), "{said}");
    }
    std::fs::write(format!("{out}.pinto.json"), "{}").unwrap();
    for &name in BASELINES {
        let (failed, said) = call_once(
            "run_fit",
            json!({"method": name, "data": ["a.zarr"], "coords": ["a.csv"], "out": out}),
        );
        assert!(failed && said.contains("exists"), "{said}");
    }
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

/// The lra and residual tools describe each flag they pass as clap does:
/// its short help (the long one where it carries a vocabulary), then the
/// flag, then clap's default. Every default a
/// description states is clap's.
#[test]
fn the_lra_and_residual_tools_describe_their_flags_by_clap() {
    let cli = crate::built_cli();
    let mut tools = specs(&cli);
    tools.extend(super::super::residual::specs(&cli));
    for (tool, command) in [
        ("run_lra", "lra"),
        ("run_lra_simulate", "lra simulate"),
        ("run_residual", "residual"),
    ] {
        let spec = tools.iter().find(|t| t.name == tool).unwrap();
        let props = spec.input_schema["properties"].as_object().unwrap();
        let sub = crate::cli_path::find(&cli, command).unwrap().1;
        let mut described = 0;
        for arg in sub.get_arguments().filter(|a| !a.is_hide_set()) {
            let id = arg.get_id().as_str();
            // The run and where it is written are the tool's own to explain.
            // max_threads and atomic_outputs default to the server's own,
            // not clap's.
            if matches!(
                id,
                "from" | "run" | "out" | "max_threads" | "atomic_outputs"
            ) {
                continue;
            }
            let Some(p) = props.get(id) else { continue };
            let said = p["description"].as_str().unwrap();
            let (short, long) = (arg.get_help(), arg.get_long_help());
            let words = |h: Option<&clap::builder::StyledStr>| {
                let h = h.map(ToString::to_string).unwrap_or_default();
                h.split_whitespace().collect::<Vec<_>>().join(" ")
            };
            // the short help, or the long one where the tool asks for it
            let help = [words(short.or(long)), words(long.or(short))]
                .into_iter()
                .find(|h| said.starts_with(h))
                .unwrap_or_else(|| panic!("{tool}.{id}: {said:?} is neither of its helps"));
            let default: Vec<String> = arg
                .get_default_values()
                .iter()
                .map(|v| v.to_string_lossy().into_owned())
                .collect();
            for (at, _) in said.match_indices("[default: ") {
                let stated = &said[at + "[default: ".len()..];
                let stated = &stated[..stated.find(']').unwrap()];
                assert!(
                    stated == default.join(",") || help.contains(&format!("[default: {stated}]")),
                    "{tool}.{id} states default {stated}, clap's is {default:?}"
                );
            }
            if !default.is_empty() {
                assert!(
                    said.ends_with(&format!("[default: {}]", default.join(","))),
                    "{tool}.{id}: {said:?} leaves out clap's default"
                );
            }
            described += 1;
        }
        assert!(described > 3, "{tool}: {described} flags described");
    }
    let lra = tools.iter().find(|t| t.name == "run_lra").unwrap();
    let seed = lra.input_schema["properties"]["seed"]["description"]
        .as_str()
        .unwrap();
    assert_eq!(
        seed,
        "Random seed for reproducibility (--seed) [default: 42]"
    );
}

/// Every tool pinto serves, run and view alike.
fn all_tools() -> Vec<ToolSpec> {
    legume_mcp::Handler::tools(&crate::mcp::Server::default())
}

/// A tool whose fields pass a command's flags: the tool, the command, the
/// fields named apart from the argument they pass, and the command's
/// visible arguments the tool takes in another way (or not at all).
type Mirror = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
    &'static [&'static str],
);

/// The tools whose fields pass a command's flags.
const MIRRORS: &[Mirror] = &[
    ("run_lra", "lra", &[("run", "from")], &[]),
    ("run_lra_simulate", "lra simulate", &[("run", "from")], &[]),
    (
        "run_residual",
        "residual",
        &[("feature_name_mode", "feature_names")],
        &[],
    ),
    ("run_compare", "compare", &[], &[]),
    ("run_simulate", "sim new", &[], &[]),
    // These two run in the call, in the server's own threads.
    (
        "run_sim_score",
        "sim score",
        &[],
        &["out", "max_threads", "atomic_outputs"],
    ),
    (
        "run_sim_pairs",
        "sim pairs",
        &[],
        &["out", "max_threads", "atomic_outputs"],
    ),
    (
        "run_sim_baseline",
        "sim baseline",
        &[("n_clusters", "k")],
        &[],
    ),
];

/// MCP mirrors the command line: a tool passing a command's flags has a
/// field for each flag its `--help` shows, and none for a hidden one.
#[test]
fn each_tool_s_fields_are_its_command_s_visible_flags() {
    let cli = crate::built_cli();
    let tools = all_tools();
    for &(tool, command, renamed, not_fields) in MIRRORS {
        let spec = tools.iter().find(|t| t.name == tool).unwrap();
        let mut fields: Vec<String> = spec.input_schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| *k != "background")
            .map(|k| {
                renamed
                    .iter()
                    .find(|(f, _)| f == k)
                    .map_or(k.clone(), |(_, id)| id.to_string())
            })
            .collect();
        fields.sort();
        fields.dedup();
        let sub = crate::cli_path::find(&cli, command).unwrap().1;
        let mut visible: Vec<String> = sub
            .get_arguments()
            .filter(|a| !a.is_hide_set())
            .map(|a| a.get_id().to_string())
            .filter(|id| !matches!(id.as_str(), "help" | "version" | "verbose"))
            .filter(|id| !not_fields.contains(&id.as_str()))
            .collect();
        visible.sort();
        assert_eq!(fields, visible, "{tool} against pinto {command}");
    }
}

/// A tool's `atomic_outputs` reaches the child pinto as `--atomic-outputs`,
/// which the command parses; left out, the child has none of its own. Every
/// tool that starts a command builds its line through [`shared_flags`].
#[test]
fn atomic_outputs_reaches_the_child_s_command_line() {
    use super::super::{lra, residual, sim};
    let cli = crate::built_cli();
    assert_eq!(
        shared_flags(Some(3), Some(true)).collect::<Vec<_>>(),
        ["--max-threads=3", "--atomic-outputs"]
    );
    assert_eq!(shared_flags(None, None).count(), 0);
    assert_eq!(shared_flags(None, Some(false)).count(), 0);
    // left out, it is the server's own; given, true or false, the tool's
    crate::util::staging::ENABLED_HERE.set(true);
    let under_the_flag = [None, Some(true), Some(false)].map(|on| shared_flags(None, on).count());
    crate::util::staging::ENABLED_HERE.set(false);
    assert_eq!(under_the_flag, [1, 1, 0]);
    // each command line with the field given and left out
    let lines = |on: bool| -> Vec<Vec<String>> {
        let with = |mut v: Value| {
            if on {
                v["atomic_outputs"] = json!(true);
            }
            v
        };
        let baseline: sim::SimBaselineToolArgs =
            serde_json::from_value(with(json!({"truth": "t"}))).unwrap();
        let simulate: sim::SimulateToolArgs = serde_json::from_value(with(json!({}))).unwrap();
        let res: residual::ResidualToolArgs =
            serde_json::from_value(with(json!({"run": "r"}))).unwrap();
        let lra_a: lra::LraToolArgs =
            crate::mcp::parse(with(json!({"run": "r", "lr_pairs": "p.tsv"}))).unwrap();
        let lra_sim: lra::LraSimulateToolArgs =
            crate::mcp::parse(with(json!({"run": "r", "species": "human"}))).unwrap();
        vec![
            sim::baseline_argv(&baseline, "t", "o.tsv"),
            sim::simulate_argv(&simulate, "o"),
            residual::residual_argv(&res, "o").unwrap(),
            lra::lra_argv(
                &lra_a,
                crate::lr_activity::args::LraMethod::CoexpressionScores,
                Some("/p/r.pinto.json"),
                "/p/o",
            ),
            lra::lra_simulate_argv(&lra_sim, Some("/p/r.pinto.json"), "/p/o"),
        ]
    };
    for argv in lines(true) {
        let n = argv.iter().filter(|a| *a == "--atomic-outputs").count();
        assert_eq!(n, 1, "{argv:?}");
        checked(&cli, &argv).unwrap();
    }
    for argv in lines(false) {
        assert!(
            !argv.iter().any(|a| a.contains("atomic-outputs")),
            "{argv:?}"
        );
    }

    for tool in [
        "run_fit",
        "run_predict",
        "run_compare",
        "run_lra",
        "run_lra_simulate",
        "run_residual",
        "run_simulate",
        "run_sim_baseline",
    ] {
        let spec = all_tools().into_iter().find(|t| t.name == tool).unwrap();
        let said = spec.input_schema["properties"]["atomic_outputs"]["description"]
            .as_str()
            .unwrap_or_else(|| panic!("{tool} has no atomic_outputs"))
            .to_string();
        let tail = format!("(--atomic-outputs) {}", super::SERVER_ATOMIC_OUTPUTS);
        assert!(said.ends_with(&tail), "{tool}: {said}");
    }
}

/// A tool's `max_threads` reaches the child pinto as `--max-threads`, which
/// the command parses; `run_fit` takes it as its own field, not in `flags`.
#[test]
fn max_threads_reaches_the_child_s_command_line() {
    let cli = crate::built_cli();
    let a: super::super::sim::SimBaselineToolArgs =
        serde_json::from_value(json!({"truth": "t", "max_threads": 8})).unwrap();
    let argv = super::super::sim::baseline_argv(&a, "t", "o.tsv");
    assert!(argv.contains(&"--max-threads=8".to_string()), "{argv:?}");
    checked(&cli, &argv).unwrap();
    let none: super::super::sim::SimBaselineToolArgs =
        serde_json::from_value(json!({"truth": "t"})).unwrap();
    let argv = super::super::sim::baseline_argv(&none, "t", "o.tsv");
    assert!(!argv.iter().any(|a| a.contains("max-threads")), "{argv:?}");

    for tool in [
        "run_fit",
        "run_predict",
        "run_compare",
        "run_lra",
        "run_residual",
    ] {
        let spec = all_tools().into_iter().find(|t| t.name == tool).unwrap();
        let said = spec.input_schema["properties"]["max_threads"]["description"]
            .as_str()
            .unwrap_or_else(|| panic!("{tool} has no max_threads"))
            .to_string();
        let tail = format!("(--max-threads) {}", super::SERVER_MAX_THREADS);
        assert!(said.ends_with(&tail), "{tool}: {said}");
    }
}

/// `run_methods` lists exactly the flags a method's `--help` shows, less
/// those `run_fit` takes as its own fields.
#[test]
fn run_methods_lists_exactly_each_method_s_visible_flags() {
    let cli = crate::built_cli();
    let tools = Tools::default();
    for &name in &METHODS {
        let reply = match tools
            .methods(MethodsArgs {
                method: Some(name.into()),
            })
            .unwrap()
            .into_iter()
            .next()
        {
            Some(Content::Json(v)) => v,
            _ => panic!("no JSON reply"),
        };
        let listed: Vec<String> = reply["flags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                f["flag"]
                    .as_str()
                    .unwrap()
                    .trim_start_matches('-')
                    .to_string()
            })
            .collect();
        let shown: Vec<String> = crate::cli_path::find(&cli, name)
            .unwrap()
            .1
            .get_arguments()
            .filter(|a| !a.is_hide_set())
            .filter_map(|a| a.get_long())
            .filter(|l| {
                ![
                    "out",
                    "coord",
                    "batch-files",
                    "help",
                    "version",
                    "verbose",
                    "max-threads",
                    "atomic-outputs",
                ]
                .contains(l)
            })
            .map(str::to_string)
            .collect();
        // the same flags; the listing groups them by help heading
        let sorted = |mut v: Vec<String>| {
            v.sort();
            v
        };
        assert_eq!(sorted(listed), sorted(shown), "run_methods {name}");
    }
}

/// A flag hidden on every command that has it is named by no tool and not
/// in the instructions: an agent is never steered to a knob the user does
/// not change.
#[test]
fn no_tool_names_a_hidden_flag() {
    let cli = crate::built_cli();
    let mut hidden: Vec<&str> = Vec::new();
    let mut shown: Vec<&str> = Vec::new();
    for (_, sub) in crate::cli_path::tests::commands(&cli) {
        for a in sub.get_arguments() {
            if let Some(long) = a.get_long() {
                if a.is_hide_set() {
                    hidden.push(long);
                } else {
                    shown.push(long);
                }
            }
        }
    }
    hidden.retain(|h| h.contains('-') && !shown.contains(h));
    let mut texts = vec![("instructions".to_string(), crate::mcp::instructions())];
    for t in all_tools() {
        texts.push((
            t.name.clone(),
            format!("{} {}", t.description, t.input_schema),
        ));
    }
    for (name, text) in &texts {
        for h in &hidden {
            let named = text.match_indices(h).any(|(at, _)| {
                let before = text[..at].chars().next_back();
                let after = text[at + h.len()..].chars().next();
                let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '-');
                !word(before.filter(|&c| c != '-')) && !word(after)
            });
            assert!(!named, "{name} names the hidden flag --{h}");
        }
    }
}
