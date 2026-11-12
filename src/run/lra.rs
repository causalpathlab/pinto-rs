//! `run_lra` and `run_lra_simulate` for `pinto mcp`: `pinto lra` and
//! `pinto lra-simulate` on a run or on data files, set up flag for flag as the command line
//! takes them, checked as pinto parses them, and run as a child pinto in
//! the project's records (a background job unless asked otherwise). The
//! reply is read from the tables the child wrote.

use super::form::check;
use super::mcp::{as_run, in_background, out_or_free, rooted, spawn, NOT_WRITTEN_OVER};
use super::script;
use crate::lr_activity::args::LraMethod;
use crate::lr_activity::ligand_effect::output::{
    self, compare_runs, lra_simulate_path, percent_text, rank_pooled, read_pooled_effects,
    read_sim_runs,
};
use crate::lr_activity::ligand_effect::simulate::COMPARED;
use crate::lr_activity::ligand_effect::verdict::Verdict;
use crate::mcp::shown_path;
use crate::util::metadata::{run_stem, LraOptions, PintoMetadata};
use legume_mcp::{Call, Content, Next};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// Pairs the reply lists, of those that survived the controls, strongest first
/// (as the run summary).
const TOP_PAIRS: usize = 5;

// * Arguments

#[derive(Deserialize, Serialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct LraToolArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub coord_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knn_spatial: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_permutations: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_fwer_threshold: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_gene_count: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr_pairs: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub species: Option<String>,
    /// `--only`: the pairs to estimate, `LIGAND:RECEPTOR` or a bare ligand.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub only: Vec<String>,
    #[serde(default = "in_background", skip_serializing)]
    pub background: bool,
}

impl LraToolArgs {
    /// `only` as `pinto lra` takes it.
    fn only(&self) -> Vec<Box<str>> {
        self.only.iter().map(|s| s.as_str().into()).collect()
    }

    fn input(&self) -> Input<'_> {
        Input {
            run: self.run.as_deref(),
            data_files: &self.data_files,
            coord_files: &self.coord_files,
            batch_files: &self.batch_files,
            knn_spatial: self.knn_spatial,
        }
    }
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct LraSimulateToolArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub data_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub coord_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knn_spatial: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub species: Option<String>,
    #[serde(default = "in_background", skip_serializing)]
    pub background: bool,
}

impl LraSimulateToolArgs {
    fn input(&self) -> Input<'_> {
        Input {
            run: self.run.as_deref(),
            data_files: &self.data_files,
            coord_files: &self.coord_files,
            batch_files: &self.batch_files,
            knn_spatial: self.knn_spatial,
        }
    }
}

/// What a tool reads, as `LraInputArgs` takes it: a run's manifest, or data
/// files with their coordinates.
struct Input<'a> {
    run: Option<&'a str>,
    data_files: &'a [String],
    coord_files: &'a [String],
    batch_files: &'a [String],
    knn_spatial: Option<usize>,
}

impl Input<'_> {
    /// The run's manifest when one was given, checked as `pinto lra` checks
    /// its input; `None` reads the data files.
    fn run(&self) -> anyhow::Result<Option<String>> {
        match self.run.filter(|r| !r.trim().is_empty()) {
            Some(r) => {
                let run = as_run(r);
                anyhow::ensure!(
                    Path::new(&run).is_file(),
                    "{run} not found: give run, a run's .pinto.json"
                );
                anyhow::ensure!(
                    self.coord_files.is_empty()
                        && self.batch_files.is_empty()
                        && self.knn_spatial.is_none(),
                    "coord_files, batch_files and knn_spatial apply to data files read \
                     directly; a run's coordinates, sections and contacts are used"
                );
                Ok(Some(run))
            }
            None => {
                anyhow::ensure!(
                    !self.data_files.is_empty() && !self.coord_files.is_empty(),
                    "give run, a run's .pinto.json, or data_files with coord_files"
                );
                Ok(None)
            }
        }
    }

    /// What it reads, by one name: the run, or the first data file.
    fn source<'s>(&'s self, run: Option<&'s str>) -> &'s str {
        run.unwrap_or_else(|| self.data_files.first().map_or("lra", String::as_str))
    }

    /// The name a free `out` goes by.
    fn stem(&self, run: Option<&str>) -> String {
        run_stem(self.source(run))
    }

    /// The flags beside `--from`: only what was given.
    fn flags(&self) -> Vec<String> {
        let shown = |files: &[String]| rooted(files).join(",");
        let mut argv = Vec::new();
        if !self.data_files.is_empty() {
            argv.push(shown(self.data_files));
        }
        if !self.coord_files.is_empty() {
            argv.extend(["--coord".into(), shown(self.coord_files)]);
        }
        if !self.batch_files.is_empty() {
            argv.extend(["--batch-files".into(), shown(self.batch_files)]);
        }
        if let Some(k) = self.knn_spatial {
            argv.extend(["--knn-spatial".into(), k.to_string()]);
        }
        argv
    }

    /// The same input as the tool's arguments, for the calls a reply names.
    fn args(&self, run: Option<&str>) -> Value {
        let mut args = json!({});
        if let Some(run) = run {
            args["run"] = run.into();
        }
        if !self.data_files.is_empty() {
            args["data_files"] = self.data_files.into();
        }
        if !self.coord_files.is_empty() {
            args["coord_files"] = self.coord_files.into();
        }
        if !self.batch_files.is_empty() {
            args["batch_files"] = self.batch_files.into();
        }
        if let Some(k) = self.knn_spatial {
            args["knn_spatial"] = k.into();
        }
        args
    }
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

/// The co-expression methods read link communities, which only a run has.
fn method_can_read(method: LraMethod, run: Option<&str>) -> anyhow::Result<()> {
    anyhow::ensure!(
        run.is_some() || method == LraMethod::ContactEffect,
        "method {} reads a run's link communities: give run",
        method_name(method)
    );
    Ok(())
}

/// `pinto lra` for `a` with `method`, reading `run` (or its data files) and
/// writing under `out`: each choice the tool was given, as the command line
/// takes it ([`LraOptions::argv`]).
pub(super) fn lra_argv(
    a: &LraToolArgs,
    method: LraMethod,
    run: Option<&str>,
    out: &str,
) -> Vec<String> {
    let d = LraOptions::default();
    let options = LraOptions {
        lr_pairs: a.lr_pairs.as_deref().map(|p| shown_path(Path::new(p))),
        targets: a.targets.iter().map(|t| shown_path(Path::new(t))).collect(),
        species: a.species.clone().unwrap_or(d.species),
        seed: a.seed.unwrap_or(d.seed),
        identity_excluded: Vec::new(),
    };
    let mut argv = options.argv(run, Some(out), method, &a.only);
    argv.extend(a.input().flags());
    if let Some(n) = a.n_permutations {
        argv.extend(["--n-permutations".into(), n.to_string()]);
    }
    if let Some(x) = a.json_fwer_threshold {
        argv.extend(["--json-fwer-threshold".into(), x.to_string()]);
    }
    if let Some(x) = a.min_gene_count {
        argv.extend(["--min-gene-count".into(), x.to_string()]);
    }
    argv
}

/// `pinto lra-simulate` reading `run` (or its data files), writing under
/// `out`.
pub(super) fn lra_simulate_argv(
    a: &LraSimulateToolArgs,
    run: Option<&str>,
    out: &str,
) -> Vec<String> {
    let mut argv: Vec<String> = vec!["lra-simulate".into()];
    if let Some(run) = run {
        argv.extend(["--from".into(), run.into()]);
    }
    argv.extend(["-o".into(), out.into()]);
    argv.extend(a.input().flags());
    if let Some(seed) = a.seed {
        argv.extend(["--seed".into(), seed.to_string()]);
    }
    if let Some(s) = &a.species {
        argv.extend(["--species".into(), s.clone()]);
    }
    argv
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

/// The reply from the tables a finished run left at `made`, read, not run
/// again; `None` when there are none. Tables that cannot be read are not
/// written over either.
fn read_finished(
    cx: &Call,
    made: &[PathBuf],
    reply: impl FnOnce() -> anyhow::Result<Vec<Content>>,
) -> Option<anyhow::Result<Vec<Content>>> {
    finished(cx, made).then(|| {
        let mut reply = reply().map_err(|e| anyhow::anyhow!("{e:#}; {NOT_WRITTEN_OVER}"))?;
        push_note(
            &mut reply,
            "read from the tables already there; nothing was run",
        );
        Ok(reply)
    })
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
    let method = lra_method(a.method.as_deref())?;
    let input = a.input();
    let run = input.run()?;
    let run = run.as_deref();
    method_can_read(method, run)?;
    let out = named_out(a.out.as_deref());
    let only = a.only();
    // A finished `out` is read, not run again.
    if let Some(o) = out.as_deref().filter(|o| !o.is_empty()) {
        let o = shown_path(Path::new(o));
        let made = lra_outputs(method, &o, &only);
        let argv = lra_argv(&a, method, run, &o);
        if let Some(reply) =
            read_finished(cx, &made, || lra_reply(method, &argv, &input, run, &made))
        {
            return reply;
        }
    }
    let (out, made) = out_or_free(cx, out.as_deref(), &input.stem(run), |o| {
        Ok(lra_outputs(method, o, &only))
    })?;
    let argv = lra_argv(&a, method, run, &out);
    checked(cli, &argv)?;
    let summary = format!("lra {} of {}", method_name(method), input.source(run));
    // The same call, by its out: what reads the tables once the job is done.
    let mut again = serde_json::to_value(&a)?;
    if let Some(run) = run {
        again["run"] = run.into();
    }
    again["out"] = out.clone().into();
    if let Some(job) = start(cx, &argv, &made, &summary, a.background, "run_lra", &again)? {
        return Ok(vec![Content::Json(job)]);
    }
    lra_reply(method, &argv, &input, run, &made)
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
/// counts over the pooled pairs and those that survived the controls, then the files and
/// what to do next.
fn lra_reply(
    method: LraMethod,
    argv: &[String],
    input: &Input<'_>,
    run: Option<&str>,
    made: &[PathBuf],
) -> anyhow::Result<Vec<Content>> {
    let written: Vec<String> = made.iter().map(|p| shown_path(p)).collect();
    let mut reply = if method == LraMethod::ContactEffect {
        let mut r = effects_summary(&read_pooled_effects(&made[0].to_string_lossy())?);
        r["reading"] =
            "one row per pair and section (`all` pooled); verdict_note says why in words".into();
        if argv.iter().any(|w| w == "--only") {
            r["q_scope"] = "q corrected within this --only subset, not over every pair; \
                            recorded under lra.subsets, the full table unchanged"
                .into();
        }
        r
    } else {
        json!({})
    };
    reply["command"] = format!("pinto {}", script::line(argv)).into();
    reply["written"] = written.into();
    let mut pieces = vec![Content::Json(reply)];
    let checked = run
        .and_then(|run| PintoMetadata::read(Path::new(run)).ok())
        .and_then(|m| m.outputs.lra.simulate);
    if checked.is_none() {
        pieces.push(Content::Next(vec![Next::new(
            "run_lra_simulate",
            input.args(run),
            "no power check for this tissue yet: it shows which effect sizes lra can detect \
             here and whether it stays calibrated under confounding",
        )]));
    }
    Ok(pieces)
}

/// The pooled pairs in a few lines: how many, each verdict's count, and the
/// strongest that survived the controls (percent change with its 95% interval, and q).
pub(super) fn effects_summary(rows: &[output::PooledEffect]) -> Value {
    let counts: Map<String, Value> = Verdict::counts(rows.iter().map(|r| r.verdict))
        .iter()
        .map(|(v, n)| (v.code().into(), (*n).into()))
        .collect();
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

/// `run_lra_simulate`.
pub(super) fn lra_simulate(
    cli: &clap::Command,
    a: LraSimulateToolArgs,
    cx: &Call,
) -> anyhow::Result<Vec<Content>> {
    let input = a.input();
    let run = input.run()?;
    let run = run.as_deref();
    let out = named_out(a.out.as_deref());
    if let Some(o) = out.as_deref().filter(|o| !o.is_empty()) {
        let o = shown_path(Path::new(o));
        let made = vec![PathBuf::from(lra_simulate_path(&o))];
        let argv = lra_simulate_argv(&a, run, &o);
        if let Some(reply) = read_finished(cx, &made, || simulate_reply(&argv, &made)) {
            return reply;
        }
    }
    let (out, made) = out_or_free(cx, out.as_deref(), &input.stem(run), |o| {
        Ok(vec![PathBuf::from(lra_simulate_path(o))])
    })?;
    let argv = lra_simulate_argv(&a, run, &out);
    checked(cli, &argv)?;
    let summary = format!(
        "lra-simulate power and calibration check of {}",
        input.source(run)
    );
    let mut again = serde_json::to_value(&a)?;
    if let Some(run) = run {
        again["run"] = run.into();
    }
    again["out"] = out.clone().into();
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
    let rates = |r: &[f64; COMPARED.len()]| -> Value {
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
                    adjusted effect; marginal the correlation of response and contact ligand; \
                    unadjusted_diff the difference in mean response, exposed minus unexposed, with no adjustment; ipw the effect by overlap weights alone; \
                    celltype, moran, contact and coexpr the co-expression baselines (CellPhoneDB/CellChat, SpatialDM, \
                    NICHES/COMMOT, lra's log odds). Trust lra here when estimate keeps type I \
                    near 0.05 under the confounded nulls and has power at the effect sizes \
                    that matter.",
    })
}

#[cfg(test)]
#[path = "tests/lra.rs"]
mod tests;
