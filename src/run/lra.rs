//! `run_lra` and `run_lra_simulate` for `pinto mcp`: `pinto lra` and
//! `pinto lra-simulate` on a run, set up flag for flag as the command line
//! takes them, checked as pinto parses them, and run as a child pinto in
//! the project's records (a background job unless asked otherwise). The
//! reply is read from the tables the child wrote.

use super::form::check;
use super::mcp::{as_run, in_background, out_or_free, spawn};
use super::script;
use crate::lr_activity::args::LraMethod;
use crate::lr_activity::ligand_effect::lra_simulate_path;
use crate::lr_activity::ligand_effect::output::{
    self, compare_runs, percent_text, rank_pooled, read_pooled_effects, read_sim_runs, Verdict,
};
use crate::lr_activity::ligand_effect::simulate::{Study, COMPARED};
use crate::mcp::shown_path;
use crate::util::metadata::{run_stem, PintoMetadata};
use legume_mcp::{Call, Content, Next};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// Pairs the reply lists, strongest evidence first (as the run summary).
const TOP_PAIRS: usize = 5;

// * Arguments

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct LraToolArgs {
    pub run: String,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub lr_pairs: Option<String>,
    #[serde(default)]
    pub targets: Vec<String>,
    #[serde(default)]
    pub fdr: Option<f64>,
    #[serde(default)]
    pub species: Option<String>,
    #[serde(default = "in_background")]
    pub background: bool,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct LraSimulateToolArgs {
    #[serde(default)]
    pub run: Option<String>,
    #[serde(default)]
    pub study: Option<Value>,
    #[serde(default)]
    pub example_study: bool,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub species: Option<String>,
    #[serde(default = "in_background")]
    pub background: bool,
}

// * Command lines

/// The method `method` names, `contact-effect` when none, as `pinto lra`
/// reads it.
pub(super) fn lra_method(method: Option<&str>) -> anyhow::Result<LraMethod> {
    use clap::ValueEnum;
    match method {
        None => Ok(LraMethod::default()),
        Some(m) => LraMethod::from_str(m, false).map_err(|_| {
            let names: Vec<String> = LraMethod::value_variants()
                .iter()
                .filter_map(|v| v.to_possible_value())
                .map(|v| v.get_name().to_string())
                .collect();
            anyhow::anyhow!("method {m:?}: one of {}", names.join(", "))
        }),
    }
}

pub(super) use crate::lr_activity::lra_outputs;

/// `out` as `run_fit` names it: a folder or another header
/// ([`script::is_header`]) gets the output named `lra` under it.
fn named_out(out: Option<&str>) -> Option<String> {
    let o = out?.trim();
    Some(if script::is_header(o) {
        script::under(o, "lra")
    } else {
        o.to_string()
    })
}

/// `pinto lra` for `a`, reading `run` and writing under `out`: each flag the
/// tool was given, as the command line takes it.
pub(super) fn lra_argv(a: &LraToolArgs, run: &str, out: &str) -> Vec<String> {
    let mut argv: Vec<String> = vec![
        "lra".into(),
        "--from".into(),
        run.into(),
        "-o".into(),
        out.into(),
    ];
    if let Some(m) = &a.method {
        argv.extend(["--method".into(), m.clone()]);
    }
    if let Some(p) = &a.lr_pairs {
        argv.extend(["--lr-pairs".into(), shown_path(Path::new(p))]);
    }
    if !a.targets.is_empty() {
        let files: Vec<String> = a.targets.iter().map(|t| shown_path(Path::new(t))).collect();
        argv.extend(["--targets".into(), files.join(",")]);
    }
    if let Some(q) = a.fdr {
        argv.extend(["--fdr".into(), q.to_string()]);
    }
    if let Some(s) = &a.species {
        argv.extend(["--species".into(), s.clone()]);
    }
    argv
}

/// `pinto lra-simulate` reading `run`, writing under `out`, with the study
/// file when one was given.
pub(super) fn lra_simulate_argv(
    a: &LraSimulateToolArgs,
    run: &str,
    out: &str,
    study: Option<&Path>,
) -> Vec<String> {
    let mut argv: Vec<String> = vec![
        "lra-simulate".into(),
        "--from".into(),
        run.into(),
        "-o".into(),
        out.into(),
    ];
    if let Some(p) = study {
        argv.extend(["--study".into(), shown_path(p)]);
    }
    if let Some(s) = &a.species {
        argv.extend(["--species".into(), s.clone()]);
    }
    argv
}

/// The study file written beside `{out}`'s outputs.
pub(super) fn study_path(out: &str) -> PathBuf {
    format!("{out}.study.json").into()
}

/// `study` as the file `--study` reads, once checked as pinto reads it.
pub(super) fn study_text(study: &Value) -> anyhow::Result<String> {
    anyhow::ensure!(
        study.is_object(),
        "study: a JSON object (example_study: true shows one)"
    );
    let text = serde_json::to_string_pretty(study)?;
    Study::from_json(&text).map_err(|e| anyhow::anyhow!("study: {e}"))?;
    Ok(text + "\n")
}

/// The command line checked as pinto parses it.
fn checked(cli: &clap::Command, argv: &[String]) -> anyhow::Result<()> {
    check(cli, argv).map_err(|e| anyhow::anyhow!("pinto {}: {e}", argv[0]))
}

/// Whether every output of a step is there and no job holds it: a finished
/// step, whose reply is read rather than run again.
fn finished(cx: &Call, made: &[PathBuf]) -> bool {
    made.iter().all(|p| p.is_file()) && cx.ensure_free(made).is_ok()
}

/// Run `argv` making `made`: in the background (the reply is the job), or
/// to its end (`None`, and the caller reads the tables).
fn start(
    cx: &Call,
    argv: &[String],
    made: &[PathBuf],
    summary: &str,
    background: bool,
    tool: &str,
    again: &Value,
) -> anyhow::Result<Option<Value>> {
    let Some(job) = spawn(cx, argv, made, summary, background)? else {
        return Ok(None);
    };
    Ok(Some(json!({
        "command": format!("pinto {}", script::line(argv)),
        "job": job,
        "writes": made.iter().map(|p| shown_path(p)).collect::<Vec<_>>(),
        "note": format!(
            "job_wait (id {job}) or job_status until it ends: its output is the run's summary. \
             Then {tool} {again} reads the finished tables into the full reply."
        ),
    })))
}

// * run_lra

/// `run_lra`.
pub(super) fn lra(cli: &clap::Command, a: LraToolArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
    anyhow::ensure!(!a.run.trim().is_empty(), "give run, the run's .pinto.json");
    let method = lra_method(a.method.as_deref())?;
    let run = as_run(&a.run);
    anyhow::ensure!(
        Path::new(&run).is_file(),
        "{run} not found: give run, a run's .pinto.json"
    );
    let out = named_out(a.out.as_deref());
    // A finished `out` is read, not run again.
    if let Some(o) = out.as_deref().filter(|o| !o.is_empty()) {
        let o = shown_path(Path::new(o));
        let made = lra_outputs(method, &o);
        if finished(cx, &made) {
            let argv = lra_argv(&a, &run, &o);
            let mut reply = lra_reply(method, &argv, &run, &made)?;
            push_note(
                &mut reply,
                "read from the tables already there; nothing was run",
            );
            return Ok(reply);
        }
    }
    let (out, made) = out_or_free(cx, out.as_deref(), &run_stem(&run), |o| {
        Ok(lra_outputs(method, o))
    })?;
    let argv = lra_argv(&a, &run, &out);
    checked(cli, &argv)?;
    let summary = format!("lra {} of {run}", method_name(method));
    // The same call, by its out: what reads the tables once the job is done.
    let mut again = json!({"run": run, "out": out});
    for (k, v) in [
        ("method", a.method.clone().map(Value::from)),
        ("lr_pairs", a.lr_pairs.clone().map(Value::from)),
        (
            "targets",
            (!a.targets.is_empty()).then(|| a.targets.clone().into()),
        ),
        ("fdr", a.fdr.map(Value::from)),
        ("species", a.species.clone().map(Value::from)),
    ] {
        if let Some(v) = v {
            again[k] = v;
        }
    }
    if let Some(job) = start(cx, &argv, &made, &summary, a.background, "run_lra", &again)? {
        return Ok(vec![Content::Json(job)]);
    }
    lra_reply(method, &argv, &run, &made)
}

fn method_name(method: LraMethod) -> String {
    use clap::ValueEnum;
    method
        .to_possible_value()
        .map_or_else(String::new, |v| v.get_name().to_string())
}

/// Add `note` to the reply's JSON piece.
fn push_note(reply: &mut [Content], note: &str) {
    if let Some(Content::Json(v)) = reply.first_mut() {
        v["note"] = note.into();
    }
}

/// The reply to a finished `pinto lra`: for the contact effect, the verdict
/// counts over the pooled pairs and the strongest pairs, then the files and
/// what to do next.
fn lra_reply(
    method: LraMethod,
    argv: &[String],
    run: &str,
    made: &[PathBuf],
) -> anyhow::Result<Vec<Content>> {
    let written: Vec<String> = made.iter().map(|p| shown_path(p)).collect();
    let mut reply = if method == LraMethod::ContactEffect {
        let mut r = effects_summary(&read_pooled_effects(&made[0].to_string_lossy())?);
        r["reading"] =
            "one row per pair and section (`all` pooled); verdict_note says why in words".into();
        r
    } else {
        json!({})
    };
    reply["command"] = format!("pinto {}", script::line(argv)).into();
    reply["written"] = written.into();
    let mut pieces = vec![Content::Json(reply)];
    let checked = PintoMetadata::read(Path::new(run))
        .ok()
        .and_then(|m| m.outputs.lra.simulate);
    if checked.is_none() {
        pieces.push(Content::Next(vec![Next::new(
            "run_lra_simulate",
            json!({ "run": run }),
            "no power check for this tissue yet: it shows which effect sizes lra can detect \
             here and whether it stays calibrated under confounding",
        )]));
    }
    Ok(pieces)
}

/// The pooled pairs in a few lines: how many, each verdict's count, and the
/// strongest evidence (percent change with its 95% interval, and q).
pub(super) fn effects_summary(rows: &[output::PooledEffect]) -> Value {
    let mut counts = Map::new();
    for v in Verdict::ALL {
        let n = rows.iter().filter(|r| r.verdict == v).count();
        counts.insert(v.code().into(), n.into());
    }
    let top: Vec<Value> = rank_pooled(rows)
        .into_iter()
        .take(TOP_PAIRS)
        .map(|r| {
            json!({
                "ligand": r.ligand,
                "receptor": r.receptor,
                "verdict": r.verdict.code(),
                "change": percent_text(r.pct, r.pct_lo, r.pct_hi),
                "q": sig(r.effect_q),
            })
        })
        .collect();
    json!({ "pairs": rows.len(), "verdicts": counts, "top": top })
}

/// A q-value to three significant digits; null when there is none.
fn sig(x: f64) -> Value {
    format!("{x:.2e}")
        .parse::<f64>()
        .ok()
        .filter(|x| x.is_finite())
        .map_or(Value::Null, Value::from)
}

/// A share to two decimals, as the run's summary prints it.
fn share(x: f64) -> Value {
    ((x * 100.0).round() / 100.0).into()
}

// * run_lra_simulate

/// `run_lra_simulate`; with `example_study`, the built-in study instead.
pub(super) fn lra_simulate(
    cli: &clap::Command,
    a: LraSimulateToolArgs,
    cx: &Call,
) -> anyhow::Result<Vec<Content>> {
    if a.example_study {
        return Ok(vec![Content::Json(example_study()?)]);
    }
    let run = a
        .run
        .as_deref()
        .filter(|r| !r.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("give run, the run's .pinto.json (or example_study: true)")
        })?;
    let run = as_run(run);
    anyhow::ensure!(
        Path::new(&run).is_file(),
        "{run} not found: give run, a run's .pinto.json"
    );
    let study = a.study.as_ref().map(study_text).transpose()?;
    let out = named_out(a.out.as_deref());
    if let Some(o) = out.as_deref().filter(|o| !o.is_empty()) {
        let o = shown_path(Path::new(o));
        let made = vec![PathBuf::from(lra_simulate_path(&o))];
        if finished(cx, &made) {
            let file = study.is_some().then(|| study_path(&o));
            let argv = lra_simulate_argv(&a, &run, &o, file.as_deref());
            let mut reply = simulate_reply(&argv, &made)?;
            push_note(
                &mut reply,
                "read from the table already there; nothing was run",
            );
            return Ok(reply);
        }
    }
    let with_study = study.is_some();
    let (out, mut made) = out_or_free(cx, out.as_deref(), &run_stem(&run), |o| {
        let mut m = vec![PathBuf::from(lra_simulate_path(o))];
        if with_study {
            m.push(study_path(o));
        }
        Ok(m)
    })?;
    // The study is the step's input, not its output: written before it
    // runs, and kept in the records as the file it is.
    let file = made.get(1).cloned();
    made.truncate(1);
    let argv = lra_simulate_argv(&a, &run, &out, file.as_deref());
    checked(cli, &argv)?;
    if let (Some(path), Some(text)) = (&file, &study) {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        cx.write_file(path, text)?;
    }
    let summary = format!("lra-simulate power and calibration check of {run}");
    let mut again = json!({"run": run, "out": out});
    for (k, v) in [
        ("study", a.study.clone()),
        ("species", a.species.clone().map(Value::from)),
    ] {
        if let Some(v) = v {
            again[k] = v;
        }
    }
    if let Some(job) = start(
        cx,
        &argv,
        &made,
        &summary,
        a.background,
        "run_lra_simulate",
        &again,
    )? {
        return Ok(vec![Content::Json(job)]);
    }
    simulate_reply(&argv, &made)
}

/// The built-in study, each section explained in its `_doc`.
pub(super) fn example_study() -> anyhow::Result<Value> {
    let study: Value = serde_json::from_str(&Study::example_json())?;
    Ok(json!({
        "study": study,
        "note": "edit it and pass it as study to run_lra_simulate; fields left out keep these \
                 values, _doc fields are ignored. Nothing was run.",
    }))
}

/// The reply to a finished `pinto lra-simulate`: the comparison table.
fn simulate_reply(argv: &[String], made: &[PathBuf]) -> anyhow::Result<Vec<Content>> {
    let runs = read_sim_runs(&made[0].to_string_lossy())?;
    let mut reply = comparison_summary(&runs);
    reply["command"] = format!("pinto {}", script::line(argv)).into();
    reply["written"] = json!([shown_path(&made[0])]);
    Ok(vec![Content::Json(reply)])
}

/// Each method's share of runs rejecting at 5%, per scenario (type I error
/// where nothing is planted or the field is shifted, power elsewhere), the
/// estimable share, and the means over the confounded nulls and the
/// planted effects.
pub(super) fn comparison_summary(runs: &[output::SimRun]) -> Value {
    let c = compare_runs(runs);
    let rates = |r: &[f64; 6]| -> Value {
        COMPARED
            .iter()
            .zip(r)
            .map(|(m, x)| ((*m).to_string(), share(*x)))
            .collect::<Map<_, _>>()
            .into()
    };
    let scenarios: Vec<Value> = c
        .scenarios
        .iter()
        .map(|s| {
            json!({
                "scenario": s.scenario,
                "rate": if s.null { "type I" } else { "power" },
                "runs": s.runs,
                "estimable": share(s.estimable),
                "reject": rates(&s.rates),
            })
        })
        .collect();
    let means: Vec<Value> = c
        .means
        .iter()
        .map(|(label, n, m)| json!({ "over": label, "scenarios": n, "reject": rates(m) }))
        .collect();
    json!({
        "runs": runs.len(),
        "methods": COMPARED,
        "scenarios": scenarios,
        "means": means,
        "reading": "reject: share of runs significant at 5% per method. estimate is lra's \
                    adjusted effect; crude the unadjusted contrast; celltype, moran, contact \
                    and coexpr the co-expression baselines (CellPhoneDB/CellChat, SpatialDM, \
                    NICHES/COMMOT, lra's log odds). Trust lra here when estimate keeps type I \
                    near 0.05 under the confounded nulls and has power at the effect sizes \
                    that matter.",
    })
}

#[cfg(test)]
#[path = "tests/lra.rs"]
mod tests;
