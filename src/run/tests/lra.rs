use super::*;
use crate::built_cli as cli;
use crate::lr_activity::ligand_effect::output::SimRun;
use crate::lr_activity::ligand_effect::simulate::Scenario;
use legume_numeric::matrix::parquet::{write_named_table, Column};

fn lra_args(v: Value) -> LraToolArgs {
    crate::mcp::parse(v).unwrap()
}

// * Command lines

#[test]
fn run_lra_builds_pinto_lras_command_line() {
    let a = lra_args(json!({
        "run": "r.pinto.json",
        "method": "coexpression-test",
        "lr_pairs": "pairs.tsv",
        "targets": ["a.gmt", "b.tsv"],
        "species": "mouse",
    }));
    let argv = lra_argv(
        &a,
        LraMethod::CoexpressionTest,
        Some("/p/r.pinto.json"),
        "/p/out",
    );
    let pairs = shown_path(Path::new("pairs.tsv"));
    let targets = format!(
        "{},{}",
        shown_path(Path::new("a.gmt")),
        shown_path(Path::new("b.tsv"))
    );
    let want: Vec<String> = [
        "lra",
        "--from=/p/r.pinto.json",
        "--out=/p/out",
        "--method=coexpression-test",
        &format!("--lr-pairs={pairs}"),
        &format!("--targets={targets}"),
        "--species=mouse",
    ]
    .map(String::from)
    .to_vec();
    assert_eq!(argv, want);
    checked(&cli(), &argv).unwrap();
    // Only what was given: the CLI's defaults stay its own.
    let bare = lra_argv(
        &lra_args(json!({"run": "r"})),
        LraMethod::ContactEffect,
        Some("r.pinto.json"),
        "o",
    );
    assert_eq!(bare, ["lra", "--from=r.pinto.json", "--out=o"]);
    // clap's complaint on a value pinto does not take.
    let bad = lra_argv(
        &lra_args(json!({"run": "r", "species": "fish"})),
        LraMethod::ContactEffect,
        Some("r"),
        "o",
    );
    assert!(checked(&cli(), &bad)
        .unwrap_err()
        .to_string()
        .contains("fish"));
}

#[test]
fn run_lra_passes_only_and_writes_the_subset_table() {
    let a = lra_args(json!({"run": "r", "only": ["LIGA1", "LIGB1:RECE1_RECE2"]}));
    let argv = lra_argv(
        &a,
        LraMethod::ContactEffect,
        Some("/p/r.pinto.json"),
        "/p/out",
    );
    assert_eq!(argv[argv.len() - 1], "--only=LIGA1,LIGB1:RECE1_RECE2");
    checked(&cli(), &argv).unwrap();
    let made = lra_outputs(LraMethod::ContactEffect, "/p/out", &a.only());
    let tag = crate::lr_activity::ligand_effect::only::tag(&a.only());
    assert_eq!(
        made[0],
        Path::new(&format!("/p/out.lra.only-{tag}.parquet"))
    );
    assert!(crate::mcp::parse::<LraToolArgs>(json!({"run": "r", "only": "LIGA1"})).is_err());
}

#[test]
fn each_method_declares_what_it_writes() {
    let names = |m| -> Vec<String> {
        lra_outputs(lra_method(m).unwrap(), "o", &[])
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
fn a_simulate_call_becomes_its_command_line() {
    let a: LraSimulateToolArgs =
        crate::mcp::parse(json!({"run": "r", "species": "human"})).unwrap();
    let argv = lra_simulate_argv(&a, Some("/p/r.pinto.json"), "/p/out");
    assert_eq!(
        argv,
        [
            "lra-simulate",
            "--from=/p/r.pinto.json",
            "--out=/p/out",
            "--species=human"
        ]
    );
    checked(&cli(), &argv).unwrap();
    // The jackknife's kind passes through as given.
    let a: LraSimulateToolArgs = crate::mcp::parse(json!({
        "run": "r", "region": "/p/region.tsv", "region_jackknife": "no-refit"
    }))
    .unwrap();
    let argv = lra_simulate_argv(&a, Some("/p/r.pinto.json"), "/p/out");
    assert!(
        argv.iter().any(|w| w == "--region-jackknife=no-refit"),
        "{argv:?}"
    );
    checked(&cli(), &argv).unwrap();
    // A region benchmark on data no reference names.
    let a: LraSimulateToolArgs = crate::mcp::parse(json!({
        "run": "r", "lr_pairs": "/p/pairs.tsv", "targets": ["/p/t.tsv"],
        "region": "/p/region.tsv", "plant_region": "D3_specific_domain_1"
    }))
    .unwrap();
    let argv = lra_simulate_argv(&a, Some("/p/r.pinto.json"), "/p/out");
    let tail: Vec<&str> = argv[3..].iter().map(String::as_str).collect();
    assert_eq!(
        tail,
        [
            "--region=/p/region.tsv",
            "--lr-pairs=/p/pairs.tsv",
            "--targets=/p/t.tsv",
            "--plant-region=D3_specific_domain_1"
        ]
    );
    checked(&cli(), &argv).unwrap();
    // The study and its example are gone from the tool as from the CLI.
    for gone in ["study", "example_study"] {
        let call = json!({"run": "r", gone: true});
        assert!(
            crate::mcp::parse::<LraSimulateToolArgs>(call).is_err(),
            "{gone}"
        );
    }
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
    let none = crate::mcp::parse::<LraToolArgs>(json!({})).unwrap();
    assert!(
        none.input().run().is_err(),
        "run or data files are required"
    );
    let only_data: LraToolArgs = crate::mcp::parse(json!({"data_files": ["d.h5"]})).unwrap();
    assert!(
        only_data.input().run().is_err(),
        "data files need coordinates"
    );
}

#[test]
fn run_lra_refuses_what_a_run_or_data_files_would_ignore() {
    let dir = tempfile::tempdir().unwrap();
    let run = dir.path().join("r.pinto.json");
    std::fs::write(&run, "{}").unwrap();
    let run = run.to_str().unwrap();
    for ignored in [
        json!({"run": run, "coord_files": ["c.csv"]}),
        json!({"run": run, "batch_files": ["b.txt"]}),
        json!({"run": run, "knn_spatial": 7}),
    ] {
        let e = lra_args(ignored).input().run().unwrap_err().to_string();
        assert!(e.contains("read directly"), "{e}");
    }
    // The co-expression methods need a run's link communities.
    assert!(method_can_read(LraMethod::ContactEffect, None).is_ok());
    assert!(method_can_read(LraMethod::CoexpressionTest, Some(run)).is_ok());
    let e = method_can_read(LraMethod::CoexpressionScores, None).unwrap_err();
    assert!(e.to_string().contains("give run"), "{e}");
    // The call a reply names reads what this one read.
    let a = lra_args(json!({"run": run, "data_files": ["new.h5"]}));
    let args = a.input().args(Some(run));
    assert_eq!(args["run"], run);
    assert_eq!(args["data_files"], json!(["new.h5"]));
}

#[test]
fn run_lra_reads_data_files_and_passes_the_rest_of_the_flags() {
    let a = lra_args(json!({
        "data_files": ["d1.h5", "d2.h5"],
        "coord_files": ["c1.csv", "c2.csv"],
        "batch_files": ["b1.txt", "b2.txt"],
        "knn_spatial": 7,
        "seed": 3,
        "n_permutations": 20,
        "json_fwer_threshold": 0.1,
        "min_gene_count": 5.0,
    }));
    assert_eq!(a.input().run().unwrap(), None);
    let argv = lra_argv(&a, LraMethod::ContactEffect, None, "o");
    assert!(!argv.iter().any(|w| w.starts_with("--from")), "{argv:?}");
    for flag in [
        "--coord",
        "--batch-files",
        "--knn-spatial",
        "--seed",
        "--n-permutations",
        "--json-fwer-threshold",
        "--min-gene-count",
    ] {
        let given = format!("{flag}=");
        assert!(
            argv.iter().any(|w| w.starts_with(&given)),
            "{flag}: {argv:?}"
        );
    }
    checked(&cli(), &argv).unwrap();
    let s: LraSimulateToolArgs = crate::mcp::parse(json!({
        "data_files": ["d.h5"],
        "coord_files": ["c.csv"],
        "seed": 3,
    }))
    .unwrap();
    let argv = lra_simulate_argv(&s, None, "o");
    assert!(argv.iter().any(|w| w == "--seed=3"), "{argv:?}");
    checked(&cli(), &argv).unwrap();
}

/// `marginal_baselines` and `marginal_permutations` reach `pinto lra` as
/// its flags, which it takes, and the per-type-pair table is among what the
/// run writes; without them neither flag nor table.
#[test]
fn run_lra_passes_the_marginal_baselines() {
    let a = lra_args(json!({
        "run": "r",
        "marginal_baselines": true,
        "marginal_permutations": 500,
    }));
    let argv = lra_argv(&a, LraMethod::ContactEffect, Some("r.pinto.json"), "o");
    let tail = &argv[argv.len() - 2..];
    assert_eq!(
        tail,
        ["--marginal-baselines", "--marginal-permutations=500"]
    );
    checked(&cli(), &argv).unwrap();
    let made = a.outputs(LraMethod::ContactEffect, "o");
    assert_eq!(made.last().unwrap(), Path::new("o.lra.celltype.parquet"));
    let bare = lra_args(json!({"run": "r"}));
    let argv = lra_argv(&bare, LraMethod::ContactEffect, Some("r.pinto.json"), "o");
    assert!(
        !argv.iter().any(|w| w.starts_with("--marginal")),
        "{argv:?}"
    );
    // Permutations without the baselines are not passed on (run_lra
    // refuses them, as pinto lra does).
    let alone = lra_args(json!({"run": "r", "marginal_permutations": 500}));
    let argv = lra_argv(&alone, LraMethod::ContactEffect, Some("r.pinto.json"), "o");
    assert!(
        !argv.iter().any(|w| w.starts_with("--marginal")),
        "{argv:?}"
    );
    assert_eq!(
        bare.outputs(LraMethod::ContactEffect, "o"),
        lra_outputs(LraMethod::ContactEffect, "o", &[])
    );
    // A subset's table sits beside the subset.
    let only = lra_args(json!({"run": "r", "only": ["L1"], "marginal_baselines": true}));
    let made = only.outputs(LraMethod::ContactEffect, "o");
    let table = made[0].to_string_lossy().into_owned();
    assert_eq!(
        made.last().unwrap().to_string_lossy(),
        format!(
            "{}.celltype.parquet",
            table.strip_suffix(".parquet").unwrap()
        )
    );
}

// * Reading finished tables

/// A run's manifest at `dir` recording the full contact-effect table at
/// `{dir}/o` and a subset of it, each with `options`; the manifest's path
/// and the full table's outputs.
fn recorded_run(dir: &Path, options: LraOptions) -> (String, Vec<PathBuf>) {
    let run = dir.join("r.pinto.json");
    let out = dir.join("o").to_string_lossy().into_owned();
    let made = lra_outputs(LraMethod::ContactEffect, &out, &[]);
    let only: Vec<Box<str>> = vec!["LIGA1".into()];
    let subset = lra_outputs(LraMethod::ContactEffect, &out, &only);
    let mut m = PintoMetadata::default();
    let lra = &mut m.outputs.lra;
    lra.table = Some(crate::lr_activity::recorded(&made[0]));
    lra.options = Some(options.clone());
    lra.record_subset(crate::util::metadata::LraSubset {
        only: vec!["LIGA1".into()],
        table: crate::lr_activity::recorded(&subset[0]),
        curve: crate::lr_activity::recorded(&subset[1]),
        options: Some(LraOptions { seed: 9, ..options }),
        ..Default::default()
    });
    m.write(&run).unwrap();
    (run.to_string_lossy().into_owned(), made)
}

/// A finished table is read back only for a call asking what made it, as
/// the manifest records it: a call asking otherwise is refused, naming
/// what differs, rather than answered with flags that never made the
/// table. A table the manifest records nothing for is read back unchecked.
#[test]
fn a_finished_table_answers_only_the_options_that_made_it() {
    let dir = tempfile::tempdir().unwrap();
    let pairs = dir.path().join("pairs.tsv").to_string_lossy().into_owned();
    let options = LraOptions {
        lr_pairs: Some(shown_path(Path::new(&pairs))),
        ..LraOptions::default()
    };
    let (run, made) = recorded_run(dir.path(), options);
    let made_as = |extra: Value| {
        let mut v = json!({"run": run, "lr_pairs": pairs});
        v.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let a = lra_args(v);
        let method = lra_method(a.method.as_deref()).unwrap();
        let made = if a.only.is_empty() {
            made.clone()
        } else {
            lra_outputs(
                method,
                &made[0].to_string_lossy().replace(".lra.parquet", ""),
                &a.only(),
            )
        };
        made_as_asked(&a, method, Path::new(&run), &made)
    };
    assert!(made_as(json!({})).unwrap(), "the call that made it");
    assert!(
        made_as(json!({"species": "auto", "seed": 42})).unwrap(),
        "its defaults spelled out"
    );
    for (extra, named) in [
        (json!({"species": "mouse"}), "species"),
        (json!({"region": "regions.tsv"}), "region"),
        (json!({"seed": 3}), "seed"),
        (json!({"marginal_baselines": true}), "marginal_baselines"),
    ] {
        let e = made_as(extra.clone()).unwrap_err().to_string();
        assert!(
            e.contains(named) && e.contains("give another out"),
            "{extra}: {e}"
        );
    }
    let e = made_as(json!({"lr_pairs": null})).unwrap_err().to_string();
    assert!(e.contains("lr_pairs"), "{e}");
    // A subset answers with its own options.
    assert!(made_as(json!({"only": ["LIGA1"], "seed": 9})).unwrap());
    assert!(made_as(json!({"only": ["LIGA1"]})).is_err());
    // Nothing recorded for the co-expression methods, nor for a table the
    // manifest does not name.
    assert!(!made_as(json!({"method": "coexpression-scores", "species": "mouse"})).unwrap());
    // Nor for the data files a table read.
    assert!(!made_as(json!({"data_files": ["d.zarr"]})).unwrap());
    let elsewhere = lra_outputs(LraMethod::ContactEffect, "elsewhere", &[]);
    let a = lra_args(json!({"run": run, "species": "mouse"}));
    assert!(!made_as_asked(&a, LraMethod::ContactEffect, Path::new(&run), &elsewhere).unwrap());
}

/// A table made with the baselines answers a call asking for them, with
/// their permutations, and none other.
#[test]
fn a_table_with_the_baselines_answers_a_call_for_them() {
    let dir = tempfile::tempdir().unwrap();
    let (run, made) = recorded_run(
        dir.path(),
        LraOptions {
            marginal_baselines: Some("cell-types".into()),
            marginal_permutations: Some(crate::lr_activity::ligand_effect::PERMUTATIONS),
            ..LraOptions::default()
        },
    );
    let made_as = |v: Value| {
        made_as_asked(
            &lra_args(v),
            LraMethod::ContactEffect,
            Path::new(&run),
            &made,
        )
    };
    assert!(made_as(json!({"run": run, "marginal_baselines": true})).unwrap());
    for v in [
        json!({"run": run}),
        json!({"run": run, "marginal_baselines": true, "marginal_permutations": 500}),
    ] {
        let e = made_as(v.clone()).unwrap_err().to_string();
        assert!(e.contains("marginal"), "{v}: {e}");
    }
}

/// The power check a reply suggests reads the pairs and programmes the
/// analysis read: its lr_pairs, targets and species go with it.
#[test]
fn the_suggested_power_check_reads_the_same_pairs() {
    let a = lra_args(json!({
        "run": "r",
        "lr_pairs": "pairs.tsv",
        "targets": ["t.tsv"],
        "species": "human",
        "region": "regions.tsv",
        "only": ["LIGA1"],
    }));
    let args = a.simulate_args(Some("r.pinto.json"));
    assert_eq!(args["run"], "r.pinto.json");
    assert_eq!(args["lr_pairs"], "pairs.tsv");
    assert_eq!(args["targets"], json!(["t.tsv"]));
    assert_eq!(args["species"], "human");
    assert_eq!(args["region"], "regions.tsv");
    let s: LraSimulateToolArgs = crate::mcp::parse(args).unwrap();
    let argv = lra_simulate_argv(&s, Some("r.pinto.json"), "o");
    checked(&cli(), &argv).unwrap();
}

/// A reply read back unchecked shows no command: the call's flags may not
/// be those that made the table.
#[test]
fn an_unchecked_read_back_shows_no_command() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o.lra.parquet");
    write_effects(path.to_str().unwrap());
    let a = lra_args(json!({"run": "r"}));
    let made = [path.clone()];
    let argv: Vec<String> = vec!["lra".into()];
    let reply = lra_reply(&a, LraMethod::ContactEffect, Some(&argv), None, &made).unwrap();
    let Content::Json(v) = &reply[0] else {
        panic!("a JSON reply")
    };
    assert_eq!(v["command"], "pinto lra");
    let reply = lra_reply(&a, LraMethod::ContactEffect, None, None, &made).unwrap();
    let Content::Json(v) = &reply[0] else {
        panic!("a JSON reply")
    };
    assert!(v.get("command").is_none(), "{v}");
}

// * Replies

/// A small `{out}.lra.parquet`: three pairs pooled and one section row.
fn write_effects(path: &str) {
    let s = |v: &[&str]| -> Vec<Box<str>> { v.iter().map(|&x| x.into()).collect() };
    let section = s(&["all", "all", "all", "s1"]);
    let ligand = s(&["LIGB1", "LIGE1", "LIGF1", "LIGB1"]);
    let receptor = s(&["RECE1", "RECG1", "RECH1", "RECE1"]);
    let verdict = s(&["supported", "no_evidence", "insufficient_data", "supported"]);
    let effect = [0.3f32, 0.01, f32::NAN, 0.4];
    let p = [1e-6f32, 0.4, f32::NAN, 1e-4];
    let q = [3.21e-5f32, 0.61, f32::NAN, 1e-3];
    let pct = [35.0f32, 1.0, f32::NAN, 49.0];
    let lo = [20.0f32, -5.0, f32::NAN, 10.0];
    let hi = [51.0f32, 7.0, f32::NAN, 90.0];
    let names = s(&["0", "1", "2", "3"]);
    let blank = s(&["", "", "", ""]);
    let zero = [0f32; 4];
    write_named_table(
        path,
        "row",
        &names,
        &[
            ("section".into(), Column::Str(&section)),
            ("ligand".into(), Column::Str(&ligand)),
            ("receptor".into(), Column::Str(&receptor)),
            ("verdict".into(), Column::Str(&verdict)),
            ("verdict_note".into(), Column::Str(&blank)),
            ("warnings".into(), Column::Str(&blank)),
            ("effect_competent_se".into(), Column::F32(&zero)),
            ("coexpr_q".into(), Column::F32(&zero)),
            ("exposure_threshold".into(), Column::F32(&zero)),
            ("n_receivers".into(), Column::F32(&zero)),
            ("n_competent".into(), Column::F32(&zero)),
            ("n_exposed".into(), Column::F32(&zero)),
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
    // Only pairs whose effect survived the controls.
    let top = v["top"].as_array().unwrap();
    assert_eq!(top.len(), 1);
    assert_eq!(top[0]["ligand"], "LIGB1");
    assert_eq!(top[0]["receptor"], "RECE1");
    assert_eq!(top[0]["verdict"], "supported");
    assert_eq!(top[0]["change"], "+35% (95% CI +20% to +51%)");
    assert_eq!(top[0]["q"], 3.21e-5);
}

/// `rejects` in the order of [`COMPARED`].
fn run(effect: f64, alpha: f64, rejects: [bool; COMPARED.len()], estimable: bool) -> SimRun {
    SimRun {
        scenario: Scenario {
            effect_pve: effect,
            ligand_field_pve: alpha,
            response_field_pve: alpha,
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
        run(0.0, 0.5, [f, t, f, t, f, f, f, f], t),
        run(0.0, 0.5, [f, t, f, f, f, f, f, f], f),
        run(0.3, 0.0, [f, t, f, t, t, t, t, t], t),
        run(0.3, 0.0, [f, t, f, t, t, t, t, f], t),
    ];
    let v = comparison_summary(&runs, None);
    assert_eq!(v["runs"], 4);
    assert_eq!(v["methods"], json!(COMPARED));
    let s = v["scenarios"].as_array().unwrap();
    assert_eq!(s.len(), 2);
    assert_eq!(s[0]["rate"], "type I");
    assert_eq!(s[0]["estimable"], 0.5);
    assert_eq!(s[0]["reject"]["unadjusted_diff"], 1.0);
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
    let (confounder, shape) = (s("realgene"), s("quadratic"));
    let f = |x: f32| vec![x; n];
    let (code, planted) = (vec![1f32; n], s("R2"));
    // The table, with the unadjusted difference's rejections under its old
    // name when `old_name`, without the columns `without`, and with the
    // effect planted in region R2 when `in_region`.
    let write = |path: &std::path::Path, old_name: bool, without: &[&str], in_region: bool| {
        let floats: Vec<(String, Vec<f32>)> = vec![
            ("effect_pve".into(), f(0.3)),
            ("ligand_field_pve".into(), f(0.3)),
            ("response_field_pve".into(), f(0.3)),
            ("confound_scale".into(), f(16.0)),
            ("dose_midpoint".into(), f(0.75)),
        ];
        let mut ints: Vec<(String, Vec<i32>)> = vec![
            ("heterogeneous".into(), vec![0, 0]),
            ("marker".into(), vec![1, 1]),
            ("rep".into(), vec![0, 1]),
            ("estimable".into(), vec![1, 0]),
            ("dose_response".into(), vec![1, 1]),
            ("dose_order".into(), vec![8, 8]),
        ];
        for m in COMPARED {
            let m = if old_name && m == "unadjusted_diff" {
                "crude"
            } else {
                m
            };
            ints.push((
                format!("{m}_reject"),
                vec![1, (m == "unadjusted_diff") as i32],
            ));
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
        if in_region {
            table.push(("region".into(), Column::F32(&code)));
            table.push(("planted_region".into(), Column::Str(&planted)));
        }
        table.retain(|(k, _)| !without.contains(&k.as_ref()));
        write_named_table(path.to_str().unwrap(), "row", &names, &table).unwrap();
    };
    // A table from before the rename, as every earlier table is, is refused,
    // not read with that method missing.
    let old = dir.path().join("old.lra-simulate.parquet");
    write(&old, true, &[], false);
    let e = read_sim_runs(old.to_str().unwrap())
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a table this version"), "{e}");
    write(&path, false, &[], false);
    let runs = read_sim_runs(path.to_str().unwrap()).unwrap();
    assert_eq!(runs.len(), 2);
    let sc = runs[0].scenario;
    assert_eq!(sc.ligand_field_pve, 0.3, "the value as written, not f32's");
    assert!(sc.dose_response);
    assert_eq!((sc.dose_midpoint, sc.dose_order), (0.75, 8));
    assert!(sc.marker && !sc.heterogeneous);
    assert_eq!(runs[1].scenario.rep, 1);
    assert!(runs[0].estimable && !runs[1].estimable);
    assert_eq!(runs[1].rejects, COMPARED.map(|m| m == "unadjusted_diff"));
    // Both replicates are one scenario, worded as the run's summary words it.
    let v = comparison_summary(&runs, None);
    let s = v["scenarios"].as_array().unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!(
        s[0]["scenario"],
        "effect PVE=0.3 field PVE ligand=0.3 response=0.3 realgene/quadratic@16 dose marker"
    );
    assert_eq!(s[0]["reject"]["unadjusted_diff"], 1.0);
    assert_eq!(s[0]["reject"]["estimate"], 0.5);

    // A table without a column it is read for is an error, not read with a
    // default in its place.
    for gone in ["ipw_reject", "dose_order"] {
        write(&old, false, &[gone], false);
        assert!(read_sim_runs(old.to_str().unwrap()).is_err(), "{gone}");
    }

    // The read-back names the region the effect was planted in.
    write(&path, false, &[], true);
    let reply = simulate_reply(None, std::slice::from_ref(&path)).unwrap();
    let Content::Json(v) = &reply[0] else {
        panic!("a JSON reply")
    };
    let scenario = v["scenarios"][0]["scenario"].as_str().unwrap();
    assert!(scenario.ends_with("planted in region R2"), "{scenario}");
    assert!(v.get("command").is_none(), "read back unchecked: {v}");
}
