use super::*;
use crate::lr_activity::ligand_effect::output::SimRun;
use crate::lr_activity::ligand_effect::simulate::Scenario;
use legume_numeric::matrix::parquet::{write_named_table, Column};

fn cli() -> clap::Command {
    use clap::CommandFactory;
    let mut cli = crate::Cli::command();
    cli.build();
    cli
}

fn lra_args(v: Value) -> LraToolArgs {
    crate::mcp::parse(v).unwrap()
}

// * Command lines

#[test]
fn run_lra_builds_pinto_lras_command_line() {
    let a = lra_args(json!({
        "run": "r.pinto.json",
        "method": "contact-effect",
        "lr_pairs": "pairs.tsv",
        "targets": ["a.gmt", "b.tsv"],
        "fdr": 0.1,
        "species": "mouse",
    }));
    let argv = lra_argv(&a, "/p/r.pinto.json", "/p/out");
    let pairs = shown_path(Path::new("pairs.tsv"));
    let targets = format!(
        "{},{}",
        shown_path(Path::new("a.gmt")),
        shown_path(Path::new("b.tsv"))
    );
    let want: Vec<String> = [
        "lra",
        "--from",
        "/p/r.pinto.json",
        "-o",
        "/p/out",
        "--method",
        "contact-effect",
        "--lr-pairs",
        &pairs,
        "--targets",
        &targets,
        "--fdr",
        "0.1",
        "--species",
        "mouse",
    ]
    .map(String::from)
    .to_vec();
    assert_eq!(argv, want);
    checked(&cli(), &argv).unwrap();
    // Only what was given: the CLI's defaults stay its own.
    let bare = lra_argv(&lra_args(json!({"run": "r"})), "r.pinto.json", "o");
    assert_eq!(bare, ["lra", "--from", "r.pinto.json", "-o", "o"]);
    // clap's complaint on a value pinto does not take.
    let bad = lra_argv(&lra_args(json!({"run": "r", "species": "fish"})), "r", "o");
    assert!(checked(&cli(), &bad)
        .unwrap_err()
        .to_string()
        .contains("fish"));
}

#[test]
fn each_method_declares_what_it_writes() {
    let names = |m| -> Vec<String> {
        lra_outputs(lra_method(m).unwrap(), "o")
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect()
    };
    assert_eq!(names(None), ["o.lra.parquet", "o.lra.curve.parquet"]);
    assert_eq!(
        names(Some("coexpression-test")),
        ["o.lr_activity.parquet", "o.lr_activity.json"]
    );
    assert_eq!(names(Some("coexpression-scores")), ["o.lr_scores.parquet"]);
    let e = lra_method(Some("both")).unwrap_err().to_string();
    assert!(e.contains("contact-effect, coexpression-test"), "{e}");
    assert_eq!(named_out(Some("le/")).as_deref(), Some("le/lra"));
    assert_eq!(named_out(Some("le/x")).as_deref(), Some("le/x"));
}

#[test]
fn a_study_becomes_the_file_study_reads() {
    let a: LraSimulateToolArgs =
        crate::mcp::parse(json!({"run": "r", "study": {"reps": 2}, "species": "human"})).unwrap();
    let text = study_text(a.study.as_ref().unwrap()).unwrap();
    assert_eq!(Study::from_json(&text).unwrap().reps, 2);
    let path = study_path("/p/out");
    assert_eq!(path, Path::new("/p/out.study.json"));
    let argv = lra_simulate_argv(&a, "/p/r.pinto.json", "/p/out", Some(&path));
    assert_eq!(
        argv,
        [
            "lra-simulate",
            "--from",
            "/p/r.pinto.json",
            "-o",
            "/p/out",
            "--study",
            "/p/out.study.json",
            "--species",
            "human"
        ]
    );
    checked(&cli(), &argv).unwrap();
    // A study pinto would refuse is refused before anything is written.
    let e = study_text(&json!({"replicates": 2}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("replicates"), "{e}");
    assert!(study_text(&json!([1])).is_err());
    assert!(study_text(&json!({"reps": 0})).is_err());
}

#[test]
fn the_example_study_parses_as_the_built_in_study() {
    let v = example_study().unwrap();
    let study = Study::from_json(&v["study"].to_string()).unwrap();
    assert_eq!(study, Study::from_json(&Study::example_json()).unwrap());
    assert_eq!(study.reps, Study::default().reps);
    assert!(v["note"].as_str().unwrap().contains("Nothing was run"));
}

#[test]
fn unknown_arguments_are_refused() {
    let e = crate::mcp::parse::<LraToolArgs>(json!({"run": "r", "fdr_level": 0.1}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("unknown field `fdr_level`"), "{e}");
    let e = crate::mcp::parse::<LraSimulateToolArgs>(json!({"reps": 3}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("unknown field `reps`"), "{e}");
    assert!(
        crate::mcp::parse::<LraToolArgs>(json!({})).is_err(),
        "run is required"
    );
}

// * Replies

/// A small `{out}.lra.parquet`: three pairs pooled and one section row.
fn write_effects(path: &str) {
    let s = |v: &[&str]| -> Vec<Box<str>> { v.iter().map(|&x| x.into()).collect() };
    let section = s(&["all", "all", "all", "s1"]);
    let ligand = s(&["TGFB1", "WNT5A", "CCL2", "TGFB1"]);
    let receptor = s(&["TGFBR1", "FZD5", "CCR2", "TGFBR1"]);
    let verdict = s(&["supported", "no_evidence", "insufficient_data", "supported"]);
    let effect = [0.3f32, 0.01, f32::NAN, 0.4];
    let p = [1e-6f32, 0.4, f32::NAN, 1e-4];
    let q = [3.21e-5f32, 0.61, f32::NAN, 1e-3];
    let pct = [35.0f32, 1.0, f32::NAN, 49.0];
    let lo = [20.0f32, -5.0, f32::NAN, 10.0];
    let hi = [51.0f32, 7.0, f32::NAN, 90.0];
    let names = s(&["0", "1", "2", "3"]);
    write_named_table(
        path,
        "row",
        &names,
        &[
            ("section".into(), Column::Str(&section)),
            ("ligand".into(), Column::Str(&ligand)),
            ("receptor".into(), Column::Str(&receptor)),
            ("verdict".into(), Column::Str(&verdict)),
            ("effect_competent".into(), Column::F32(&effect)),
            ("effect_p".into(), Column::F32(&p)),
            ("effect_q".into(), Column::F32(&q)),
            ("effect_pct".into(), Column::F32(&pct)),
            ("effect_pct_lo".into(), Column::F32(&lo)),
            ("effect_pct_hi".into(), Column::F32(&hi)),
        ],
    )
    .unwrap();
}

#[test]
fn the_lra_reply_counts_verdicts_and_ranks_pairs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o.lra.parquet");
    write_effects(path.to_str().unwrap());
    let rows = read_pooled_effects(path.to_str().unwrap()).unwrap();
    assert_eq!(rows.len(), 3, "pooled rows only");
    let v = effects_summary(&rows);
    assert_eq!(v["pairs"], 3);
    assert_eq!(v["verdicts"]["supported"], 1);
    assert_eq!(v["verdicts"]["no_evidence"], 1);
    assert_eq!(v["verdicts"]["insufficient_data"], 1);
    assert_eq!(v["verdicts"]["control_failed"], 0);
    assert_eq!(
        v["verdicts"].as_object().unwrap().len(),
        Verdict::ALL.len(),
        "every verdict, none or not"
    );
    // Estimable pairs by p; the one without an effect is left out.
    let top = v["top"].as_array().unwrap();
    assert_eq!(top.len(), 2);
    assert_eq!(top[0]["ligand"], "TGFB1");
    assert_eq!(top[0]["receptor"], "TGFBR1");
    assert_eq!(top[0]["verdict"], "supported");
    assert_eq!(top[0]["change"], "+35% (95% CI +20% to +51%)");
    assert_eq!(top[0]["q"], 3.21e-5);
    assert_eq!(top[1]["ligand"], "WNT5A");
    assert_eq!(top[1]["q"], 0.61);
}

fn run(effect: f64, alpha: f64, rejects: [bool; 6], estimable: bool) -> SimRun {
    SimRun {
        scenario: Scenario {
            effect,
            confound_ligand: alpha,
            confound_response: alpha,
            ..Scenario::default()
        },
        estimable,
        rejects,
    }
}

#[test]
fn the_simulate_reply_is_the_comparison_table() {
    let (t, f) = (true, false);
    let runs = [
        run(0.0, 0.5, [t, t, f, f, f, f], t),
        run(0.0, 0.5, [t, f, f, f, f, f], f),
        run(0.3, 0.0, [t, t, t, t, t, t], t),
        run(0.3, 0.0, [t, t, t, t, t, f], t),
    ];
    let v = comparison_summary(&runs);
    assert_eq!(v["runs"], 4);
    assert_eq!(v["methods"], json!(COMPARED));
    let s = v["scenarios"].as_array().unwrap();
    assert_eq!(s.len(), 2);
    assert_eq!(s[0]["rate"], "type I");
    assert_eq!(s[0]["estimable"], 0.5);
    assert_eq!(s[0]["reject"]["crude"], 1.0);
    assert_eq!(s[0]["reject"]["celltype"], 0.5);
    assert_eq!(s[0]["reject"]["estimate"], 0.0);
    assert_eq!(s[1]["rate"], "power");
    assert_eq!(s[1]["reject"]["estimate"], 0.5);
    let means = v["means"].as_array().unwrap();
    assert_eq!(means[0]["over"], "mean type I, confounded nulls");
    assert_eq!(means[1]["over"], "mean power, planted effects");
    assert_eq!(means[1]["reject"]["moran"], 1.0);
}

#[test]
fn simulate_runs_read_back_as_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o.lra-simulate.parquet");
    let n = 2;
    let s = |x: &str| -> Vec<Box<str>> { vec![x.into(); n] };
    let (confounder, shape) = (s("realgene"), s("interaction"));
    let f = |v: [f32; 2]| v.to_vec();
    let floats: Vec<(String, Vec<f32>)> = vec![
        ("planted_effect".into(), f([0.3, 0.3])),
        ("planted_ring_effect".into(), f([0.0, 0.0])),
        ("confound_ligand".into(), f([0.3, 0.3])),
        ("confound_response".into(), f([0.3, 0.3])),
        ("confound_scale".into(), f([16.0, 16.0])),
        ("spillover".into(), f([0.1, 0.1])),
        ("shift".into(), f([0.0, 0.0])),
    ];
    let mut ints: Vec<(String, Vec<i32>)> = vec![
        ("heterogeneous".into(), vec![0, 0]),
        ("marker".into(), vec![1, 1]),
        ("confound_receptor".into(), vec![1, 1]),
        ("rep".into(), vec![0, 1]),
        ("estimable".into(), vec![1, 0]),
    ];
    for m in COMPARED {
        ints.push((format!("{m}_reject"), vec![1, (m == "crude") as i32]));
    }
    let names: Vec<Box<str>> = vec!["0".into(), "1".into()];
    let mut table: Vec<(Box<str>, Column)> = vec![
        ("confounder".into(), Column::Str(&confounder)),
        ("confound_shape".into(), Column::Str(&shape)),
    ];
    table.extend(
        floats
            .iter()
            .map(|(k, v)| (k.as_str().into(), Column::F32(v))),
    );
    table.extend(
        ints.iter()
            .map(|(k, v)| (k.as_str().into(), Column::I32(v))),
    );
    write_named_table(path.to_str().unwrap(), "row", &names, &table).unwrap();

    let runs = read_sim_runs(path.to_str().unwrap()).unwrap();
    assert_eq!(runs.len(), 2);
    let sc = runs[0].scenario;
    assert_eq!(sc.confound_ligand, 0.3, "the value as written, not f32's");
    assert!(sc.marker && sc.confound_receptor && !sc.heterogeneous);
    assert_eq!(runs[1].scenario.rep, 1);
    assert!(runs[0].estimable && !runs[1].estimable);
    assert_eq!(runs[1].rejects, [true, false, false, false, false, false]);
    // Both replicates are one scenario, worded as the run's summary words it.
    let v = comparison_summary(&runs);
    let s = v["scenarios"].as_array().unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!(
        s[0]["scenario"],
        "θ=0.30 α=0.3 β=0.3 realgene/interaction@16 κ=0.1 marker"
    );
    assert_eq!(s[0]["reject"]["crude"], 1.0);
    assert_eq!(s[0]["reject"]["estimate"], 0.5);
}
