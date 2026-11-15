use super::*;
use crate::run::mcp::checked;
use legume_numeric::matrix::parquet::read_parquet_string_columns_by_name;
use serde_json::Value;

fn cli() -> clap::Command {
    use clap::CommandFactory;
    let mut cli = crate::Cli::command();
    cli.build();
    cli
}

/// A tool call's arguments, as the server parses them.
fn args<T: serde::de::DeserializeOwned>(v: Value) -> T {
    crate::mcp::parse(v).unwrap()
}

fn sim_args(v: Value) -> SimulateToolArgs {
    args(v)
}

/// The parameters a `run_simulate` call sets, checked as it checks them.
fn params(v: Value) -> anyhow::Result<SimParams> {
    simulation(&cli(), &sim_args(v), "o").map(|(_, _, p)| p)
}

const TINY: [&str; 3] = ["grid_side=8", "n_genes=40", "programme_genes=5"];

#[test]
fn the_sim_tools_are_listed_with_their_schemas() {
    let tools = specs();
    for (name, args) in [
        (
            "run_simulate",
            &["scenario", "set", "coords", "coord_column_names", "out"][..],
        ),
        (
            "run_sim_score",
            &["truth", "labels", "level", "label_column"][..],
        ),
        (
            "run_sim_pairs",
            &["truth", "vectors", "level", "pairs", "seed", "knn"][..],
        ),
        (
            "run_sim_baseline",
            &["truth", "method", "k", "seed", "out", "background"][..],
        ),
    ] {
        let spec = tools
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is not listed"));
        assert_eq!(spec.input_schema["additionalProperties"], false, "{name}");
        let props = spec.input_schema["properties"].as_object().unwrap();
        for arg in args {
            assert!(props.contains_key(*arg), "{name} lacks {arg}");
        }
    }
    let simulate = tools.iter().find(|t| t.name == "run_simulate").unwrap();
    let about = simulate.input_schema["properties"]["scenario"]["description"]
        .as_str()
        .unwrap();
    for sc in SCENARIOS {
        assert!(
            about.contains(sc.name) && about.contains(sc.about),
            "{about}"
        );
    }
    let set = simulate.input_schema["properties"]["set"]["description"]
        .as_str()
        .unwrap();
    for dial in [
        "grid_side",
        "pve_donor",
        "patch_kind",
        "nuisance_off",
        "seed",
    ] {
        assert!(set.contains(dial), "{dial}: {set}");
    }
}

#[test]
fn a_simulate_call_becomes_its_command_line() {
    let a = sim_args(json!({
        "scenario": "L2",
        "set": ["grid_side=8", "seed=2"],
        "coords": ["a.parquet", "b.parquet"],
        "coord_column_names": ["x", "y"],
    }));
    let argv = simulate_argv(&a, "/p/sim");
    let coords = format!(
        "{},{}",
        shown_path(Path::new("a.parquet")),
        shown_path(Path::new("b.parquet"))
    );
    let want: Vec<String> = [
        "srt-simulate",
        "--scenario=L2",
        "--set=grid_side=8",
        "--set=seed=2",
        &format!("--coords={coords}"),
        "--coord-column-names=x,y",
        "--out=/p/sim",
    ]
    .map(String::from)
    .to_vec();
    assert_eq!(argv, want);
    checked(&cli(), &argv).unwrap();
    // Only what was given: no scenario (the command's default), no other flag.
    let bare = simulate_argv(&sim_args(json!({})), "o");
    assert_eq!(bare, ["srt-simulate", "--out=o"]);
    // column names name the columns of given coordinate files only
    let names_alone = sim_args(json!({"coord_column_names": ["-x", "y"]}));
    assert_eq!(simulate_argv(&names_alone, "o"), bare);
    checked(&cli(), &bare).unwrap();
    // A simulation is a folder: its manifest and the truth run inside it.
    let made = sim_outputs("/p/sim");
    assert_eq!(
        made,
        [
            PathBuf::from("/p/sim"),
            PathBuf::from("/p/sim/manifest.json"),
            PathBuf::from("/p/sim/truth/truth.pinto.json")
        ]
    );
}

#[test]
fn the_reply_lists_the_sections_to_fit_in_order() {
    let p = params(json!({"scenario": "L2", "set": TINY})).unwrap();
    let n = p.n_donors * p.sections_per_donor;
    assert!(n > 1, "{n}");
    let fit = fit_lists("/p/sim", &p);
    let data: Vec<String> = (1..=n)
        .map(|s| format!("/p/sim/sections/S{s}.zarr"))
        .collect();
    let coords: Vec<String> = (1..=n)
        .map(|s| format!("/p/sim/sections/S{s}.coords.parquet"))
        .collect();
    // the sections' columns are not among run_fit's default coordinate names
    let flags = json!({"coord-column-names": "row_um,col_um"});
    assert_eq!(fit, json!({"data": data, "coords": coords, "flags": flags}));
}

/// An `out` given as a folder, ending in `/`, names the sections, the
/// manifest and the truth run once under it.
#[test]
fn a_folder_out_names_its_files_without_a_doubled_slash() {
    let p = params(json!({"scenario": "L2", "set": TINY})).unwrap();
    let fit = fit_lists("/p/sim/", &p);
    assert_eq!(fit["data"][0], "/p/sim/sections/S1.zarr");
    assert_eq!(fit["coords"][0], "/p/sim/sections/S1.coords.parquet");
    // as text: two paths that differ only by a doubled slash compare equal
    let made: Vec<String> = sim_outputs("/p/sim/")[1..]
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        made,
        ["/p/sim/manifest.json", "/p/sim/truth/truth.pinto.json"]
    );
}

#[test]
fn a_bad_setting_is_refused_before_launching() {
    assert_eq!(params(json!({"set": TINY})).unwrap().grid_side, 8);
    for (call, says) in [
        (json!({"set": ["pve_type=0"]}), "pve_type must be positive"),
        (json!({"set": ["no_such=1"]}), "no such parameter"),
        (json!({"set": ["grid_side"]}), "expected key=value"),
        (json!({"scenario": "L9"}), "unknown scenario"),
        // The coordinate files are read before it starts, and a comma, which
        // the command line splits on, is refused.
        (json!({"coords": ["nowhere.parquet"]}), "nowhere.parquet"),
        (json!({"coords": ["a,b.parquet"]}), "a comma"),
        (json!({"coord_column_names": ["x,y"]}), "a comma"),
    ] {
        let e = format!("{:#}", params(call.clone()).unwrap_err());
        assert!(e.contains(says), "{call}: {e}");
    }
    let e = crate::mcp::parse::<SimulateToolArgs>(json!({"reps": 3}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("unknown field `reps`"), "{e}");
}

/// A table of each spot's planted domain, as a labelling.
fn truth_labels(sim: &str, path: &Path) {
    let cols = read_parquet_string_columns_by_name(
        &format!("{sim}/truth/spots.parquet"),
        &["cell", "domain"],
    )
    .unwrap();
    let mut tsv = String::from("cell\tlabel\n");
    for (c, d) in cols[0].iter().zip(&cols[1]) {
        tsv.push_str(&format!("{c}\t{d}\n"));
    }
    std::fs::write(path, tsv).unwrap();
}

#[test]
fn scoring_the_truth_itself_is_perfect() {
    let dir = tempfile::tempdir().unwrap();
    let sim = dir.path().join("sim").to_string_lossy().into_owned();
    let a = sim_args(json!({"scenario": "L2", "set": TINY}));
    crate::simulate::run_simulate(&simulation(&cli(), &a, &sim).unwrap().1).unwrap();
    let table = dir.path().join("truth.tsv");
    truth_labels(&sim, &table);

    let call: SimScoreToolArgs = crate::mcp::parse(json!({
        "truth": sim,
        "labels": [table.to_string_lossy()],
    }))
    .unwrap();
    assert_eq!(
        (call.level.as_str(), call.label_column.as_str()),
        ("final", "label")
    );
    let reply = sim_score(call).unwrap();
    let [Content::Text(text), Content::Json(v)] = &reply[..] else {
        panic!("a table and its JSON: {reply:?}");
    };
    let s = &v["scores"][0]["score"];
    assert_eq!(s["shared_ari"], 1.0, "{v}");
    assert_eq!(s["donor_dependence"], 0.0, "{v}");
    for d in s["donor_dependence_by_domain"].as_array().unwrap() {
        assert_eq!(d["dependence"], 0.0, "{v}");
    }
    assert_eq!(s["n_scored"], s["n_spots"], "{v}");
    assert_eq!(s["n_unmatched"], 0, "{v}");
    assert!(v.get("warnings").is_none(), "{v}");
    let patches = s["patches"].as_array().unwrap();
    assert!(!patches.is_empty(), "L2 plants a donor-specific domain");
    for p in patches {
        for k in ["f1", "precision", "recall", "owner_share"] {
            assert_eq!(p[k], 1.0, "{k}: {p}");
        }
    }
    // One row per labelling and patch, under the header.
    assert_eq!(text.lines().count(), 1 + patches.len(), "{text}");
    for (row, p) in text.lines().skip(1).zip(patches) {
        assert!(row.contains(p["domain"].as_str().unwrap()), "{row}");
    }
    assert!(
        text.lines().next().unwrap().contains("shared_ari"),
        "{text}"
    );

    // The truth run, or its folder, names the simulation as well.
    for truth in [
        format!("{sim}/truth/truth.pinto.json"),
        format!("{sim}/truth"),
    ] {
        let call: SimScoreToolArgs =
            crate::mcp::parse(json!({"truth": truth, "labels": [table.to_string_lossy()]}))
                .unwrap();
        let reply = sim_score(call).unwrap();
        let [_, Content::Json(w)] = &reply[..] else {
            panic!("{reply:?}");
        };
        assert_eq!(w["scores"], v["scores"], "{truth}");
    }

    // Rows of cells the truth lacks are warned of, by labelling.
    let more = dir.path().join("more.tsv");
    let rows = std::fs::read_to_string(&table).unwrap() + "elsewhere_1\tD1\n";
    std::fs::write(&more, rows).unwrap();
    let call: SimScoreToolArgs =
        crate::mcp::parse(json!({"truth": sim, "labels": [more.to_string_lossy()]})).unwrap();
    let reply = sim_score(call).unwrap();
    let [_, Content::Json(w)] = &reply[..] else {
        panic!("{reply:?}");
    };
    assert_eq!(w["scores"][0]["score"]["n_unmatched"], 1, "{w}");
    let warned = w["warnings"][0]["warning"].as_str().unwrap_or_default();
    assert!(warned.contains("not in the truth"), "{w}");

    // A labelling that cannot be read is named.
    let bad: SimScoreToolArgs =
        crate::mcp::parse(json!({"truth": sim, "labels": ["nowhere.tsv"]})).unwrap();
    let e = format!("{:#}", sim_score(bad).unwrap_err());
    assert!(e.contains("nowhere.tsv"), "{e}");
}

/// The truth run's own domains, as vectors, score 1 on every pair set; the
/// reply is a table, one row per donor-specific domain, and the JSON with
/// how the pairs were drawn. Vectors that cannot be read are named.
#[test]
fn pair_scoring_the_truth_itself_is_perfect() {
    let dir = tempfile::tempdir().unwrap();
    let sim = dir.path().join("sim").to_string_lossy().into_owned();
    let a = sim_args(json!({"scenario": "L2", "set": TINY}));
    crate::simulate::run_simulate(&simulation(&cli(), &a, &sim).unwrap().1).unwrap();
    let truth = format!("{sim}/truth/truth.pinto.json");
    let call: SimPairsToolArgs = args(json!({
        "truth": sim,
        "vectors": [truth, "nowhere.tsv"],
        "level": "domain",
        "pairs": 2000,
    }));
    let reply = sim_pairs(call).unwrap();
    let [Content::Text(text), Content::Json(v)] = &reply[..] else {
        panic!("a table and its JSON: {reply:?}");
    };
    let scored = &v["scores"][0];
    assert_eq!(
        (&scored["pairs"], &scored["seed"]),
        (&json!(2000), &json!(1))
    );
    let s = &scored["score"];
    for set in ["cross_donor", "within_donor", "spatial"] {
        assert_eq!(s[set]["auroc"], 1.0, "{set}: {v}");
        assert_eq!(s[set]["auprc"], 1.0, "{set}: {v}");
    }
    let patches = s["patches"].as_array().unwrap();
    assert!(!patches.is_empty(), "L2 plants a donor-specific domain");
    assert_eq!(patches[0]["pairs"]["auroc"], 1.0, "{v}");
    assert_eq!(text.lines().count(), 1 + patches.len(), "{text}");
    assert_eq!(v["failed"][0]["vectors"], "nowhere.tsv", "{v}");
    let bad: SimPairsToolArgs = args(json!({"truth": sim, "vectors": ["nowhere.tsv"]}));
    let e = format!("{:#}", sim_pairs(bad).unwrap_err());
    assert!(e.contains("nowhere.tsv"), "{e}");
    let few: SimPairsToolArgs = args(json!({"truth": sim, "vectors": [truth], "pairs": 1}));
    assert!(sim_pairs(few).is_err());
}

/// A baseline is `pinto srt-sim-baseline` with only what was given; an
/// unknown method is refused as the command line refuses it.
#[test]
fn the_baseline_command_line_has_only_what_was_given() {
    let a = args::<SimBaselineToolArgs>(
        json!({"truth": "/p/sim", "method": "per-donor", "k": 5, "seed": 2}),
    );
    let argv = baseline_argv(&a, "/p/sim", "/p/out.tsv");
    assert_eq!(
        argv,
        [
            "srt-sim-baseline",
            "--truth=/p/sim",
            "--method=per-donor",
            "--k=5",
            "--seed=2",
            "--out=/p/out.tsv",
        ]
    );
    checked(&cli(), &argv).unwrap();
    let bare = baseline_argv(
        &args::<SimBaselineToolArgs>(json!({"truth": "/p/sim"})),
        "/p/sim",
        "/p/out.tsv",
    );
    assert_eq!(
        bare,
        ["srt-sim-baseline", "--truth=/p/sim", "--out=/p/out.tsv"]
    );
    checked(&cli(), &bare).unwrap();
    let scvi = args::<SimBaselineToolArgs>(json!({"truth": "t", "method": "scvi"}));
    let e = checked(&cli(), &baseline_argv(&scvi, "t", "o.tsv")).unwrap_err();
    assert!(format!("{e}").contains("scvi"), "{e}");
}
