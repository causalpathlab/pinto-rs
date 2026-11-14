//! `run_lra` and `run_lra_simulate` for `pinto mcp`: `pinto lra` and
//! `pinto lra-simulate` on a run or on data files, set up flag for flag as the command line
//! takes them, checked as pinto parses them, and run as a child pinto in
//! the project's records (a background job unless asked otherwise). The
//! reply is read from the tables the child wrote.

use super::mcp::{as_run, checked, in_background, out_or_free, rooted, spawn, NOT_WRITTEN_OVER};
use super::script;
use crate::lr_activity::args::LraMethod;
use crate::lr_activity::ligand_effect::output::{
    self, compare_runs, lra_simulate_path, percent_text, rank_pooled, read_planted_region,
    read_pooled_effects, read_sim_runs,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_types: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_jackknife: Option<String>,
    /// `--marginal-baselines`, and `--marginal-permutations`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub marginal_baselines: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marginal_permutations: Option<usize>,
    #[serde(default = "in_background", skip_serializing)]
    pub background: bool,
}

impl LraToolArgs {
    /// `only` as `pinto lra` takes it.
    fn only(&self) -> Vec<Box<str>> {
        self.only.iter().map(|s| s.as_str().into()).collect()
    }

    /// What `pinto lra` writes for these arguments: [`lra_outputs`], and
    /// with `marginal_baselines` the per-type-pair table beside the
    /// contact-effect table.
    fn outputs(&self, method: LraMethod, out: &str) -> Vec<PathBuf> {
        let mut made = lra_outputs(method, out, &self.only());
        if self.marginal_baselines && method == LraMethod::ContactEffect {
            let table = made[0].to_string_lossy().into_owned();
            made.push(output::celltype_beside(&table).into());
        }
        made
    }

    fn input(&self) -> Input<'_> {
        Input {
            run: self.run.as_deref(),
            data_files: &self.data_files,
            coord_files: &self.coord_files,
            batch_files: &self.batch_files,
            knn_spatial: self.knn_spatial,
            cell_types: self.cell_types.as_deref(),
            region: self.region.as_deref(),
            region_jackknife: self.region_jackknife.as_deref(),
        }
    }

    /// The `run_lra_simulate` call checking this analysis's power: the same
    /// input, pairs, programmes and species.
    fn simulate_args(&self, run: Option<&str>) -> Value {
        let mut args = self.input().args(run);
        if let Some(p) = &self.lr_pairs {
            args["lr_pairs"] = p.as_str().into();
        }
        if !self.targets.is_empty() {
            args["targets"] = self.targets.clone().into();
        }
        if let Some(s) = &self.species {
            args["species"] = s.as_str().into();
        }
        args
    }

    /// What `pinto lra` records in the manifest that it was asked, for this
    /// call: [`LraOptions`] but the identity genes and where the cell-type
    /// baseline's types came from, which the run decides.
    fn asked_options(&self) -> LraOptions {
        let d = LraOptions::default();
        let shown = |p: &String| shown_path(Path::new(p));
        LraOptions {
            lr_pairs: self.lr_pairs.as_ref().map(shown),
            targets: self.targets.iter().map(shown).collect(),
            species: self.species.clone().unwrap_or(d.species),
            seed: self.seed.unwrap_or(d.seed),
            cell_types: self.cell_types.as_ref().map(shown),
            region: self.region.as_ref().map(shown),
            region_jackknife: self.region_jackknife.clone().unwrap_or(d.region_jackknife),
            marginal_permutations: self.marginal_baselines.then(|| {
                self.marginal_permutations
                    .unwrap_or(crate::lr_activity::ligand_effect::PERMUTATIONS)
            }),
            ..d
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr_pairs: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plant_region: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_types: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_jackknife: Option<String>,
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
            cell_types: self.cell_types.as_deref(),
            region: self.region.as_deref(),
            region_jackknife: self.region_jackknife.as_deref(),
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
    cell_types: Option<&'a str>,
    region: Option<&'a str>,
    region_jackknife: Option<&'a str>,
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
        if let Some(c) = self.cell_types {
            argv.extend(["--cell-types".into(), shown_path(Path::new(c))]);
        }
        if let Some(r) = self.region {
            argv.extend(["--region".into(), shown_path(Path::new(r))]);
        }
        if let Some(j) = self.region_jackknife {
            argv.extend(["--region-jackknife".into(), j.into()]);
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
        if let Some(c) = self.cell_types {
            args["cell_types"] = c.into();
        }
        if let Some(r) = self.region {
            args["region"] = r.into();
        }
        if let Some(j) = self.region_jackknife {
            args["region_jackknife"] = j.into();
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
    // The labels and the baselines go below, as the input and flags give them.
    let d = LraOptions::default();
    let options = LraOptions {
        cell_types: None,
        region: None,
        region_jackknife: d.region_jackknife,
        marginal_permutations: None,
        ..a.asked_options()
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
    if a.marginal_baselines {
        argv.push("--marginal-baselines".into());
        if let Some(n) = a.marginal_permutations {
            argv.extend(["--marginal-permutations".into(), n.to_string()]);
        }
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
    if let Some(p) = &a.lr_pairs {
        argv.extend(["--lr-pairs".into(), shown_path(Path::new(p))]);
    }
    if !a.targets.is_empty() {
        let t: Vec<String> = a.targets.iter().map(|t| shown_path(Path::new(t))).collect();
        argv.extend(["--targets".into(), t.join(",")]);
    }
    if let Some(r) = &a.plant_region {
        argv.extend(["--plant-region".into(), r.clone()]);
    }
    argv
}

/// Whether every output of a step is there and no job holds it: a finished
/// step, whose reply is read rather than run again.
fn finished(cx: &Call, made: &[PathBuf]) -> bool {
    made.iter().all(|p| p.is_file()) && cx.ensure_free(made).is_ok()
}

/// The reply from the tables a finished run left at `made`, read, not run
/// again; `None` when there are none. `made_as` says whether the manifest
/// records them made as this call asks (an error when it records
/// otherwise, `false` when it records nothing), and `reply(made_as)` shows
/// the call's command only when it does. Tables that cannot be read are
/// not written over either.
fn read_finished(
    cx: &Call,
    made: &[PathBuf],
    made_as: impl FnOnce() -> anyhow::Result<bool>,
    reply: impl FnOnce(bool) -> anyhow::Result<Vec<Content>>,
) -> Option<anyhow::Result<Vec<Content>>> {
    finished(cx, made).then(|| {
        let checked = made_as()?;
        let mut reply = reply(checked).map_err(|e| anyhow::anyhow!("{e:#}; {NOT_WRITTEN_OVER}"))?;
        push_note(
            &mut reply,
            if checked {
                "read from the tables already there, made with this call's options; nothing \
                 was run"
            } else {
                "read from the tables already there; nothing was run. The manifest does not \
                 record all that made them (data files, the co-expression methods' options), \
                 so this call was not checked against them in full and its command is not shown"
            },
        );
        Ok(reply)
    })
}

/// The options the manifest at `manifest` records for the contact-effect
/// table `table`: the full table's or a subset's; `None` when it names
/// neither.
fn recorded_options(manifest: &Path, table: &Path) -> Option<LraOptions> {
    let lra = PintoMetadata::read(manifest).ok()?.outputs.lra;
    let table = crate::lr_activity::recorded(table);
    if lra.table.as_deref() == Some(table.as_str()) {
        return lra.options;
    }
    let subset = lra.subsets.iter().find(|s| s.table == table)?;
    lra.subset_options(subset)
}

/// Whether the finished tables `made` were made as `a` asks, by the options
/// the manifest at `manifest` records for them: `true` when they were, an
/// error naming each option that differs when they were not, `false` when
/// what made them is not all recorded (the co-expression methods record no
/// options, and no table records the data files it read).
fn made_as_asked(
    a: &LraToolArgs,
    method: LraMethod,
    manifest: &Path,
    made: &[PathBuf],
) -> anyhow::Result<bool> {
    if method != LraMethod::ContactEffect {
        return Ok(false);
    }
    let Some(recorded) = recorded_options(manifest, &made[0]) else {
        return Ok(false);
    };
    let fields = |o: &LraOptions| match serde_json::to_value(o) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    };
    let (was, asked) = (fields(&recorded), fields(&a.asked_options()));
    let mut differ: Vec<String> = was
        .keys()
        .chain(asked.keys())
        .filter(|&k| k != "identity_excluded" && k != "marginal_baselines")
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|&k| was.get(k) != asked.get(k))
        .map(|k| {
            let show = |m: &Map<String, Value>| m.get(k).map_or("none".into(), Value::to_string);
            format!("{k} {}, asked {}", show(&was), show(&asked))
        })
        .collect();
    if recorded.marginal_baselines.is_some() != a.marginal_baselines {
        differ.push(format!(
            "marginal_baselines {}, asked {}",
            recorded.marginal_baselines.is_some(),
            a.marginal_baselines
        ));
    }
    anyhow::ensure!(
        differ.is_empty(),
        "{} was made with other options ({}); {NOT_WRITTEN_OVER}, or call with the options it \
         was made with",
        shown_path(&made[0]),
        differ.join("; ")
    );
    Ok(a.data_files.is_empty())
}

/// `argv` checked as pinto parses it, and for the flags only the contact
/// effect takes, as `pinto lra` checks them before reading any data.
fn checked_lra(cli: &clap::Command, argv: &[String]) -> anyhow::Result<()> {
    let args: crate::lr_activity::SrtLrActivityArgs = super::mcp::parsed(cli, argv)?;
    crate::lr_activity::fit::check_method_flags(&args)
        .map_err(|e| anyhow::anyhow!("pinto {}: {e}", argv[0]))
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
    anyhow::ensure!(
        a.marginal_baselines || a.marginal_permutations.is_none(),
        "marginal_permutations applies with marginal_baselines, the tests it counts"
    );
    let out = named_out(a.out.as_deref());
    // A finished `out` is read, not run again, once the call is checked as
    // it would be run.
    if let Some(o) = out.as_deref().filter(|o| !o.is_empty()) {
        let o = shown_path(Path::new(o));
        let made = a.outputs(method, &o);
        let argv = lra_argv(&a, method, run, &o);
        checked_lra(cli, &argv)?;
        let manifest = run.map_or_else(|| PathBuf::from(format!("{o}.pinto.json")), PathBuf::from);
        if let Some(reply) = read_finished(
            cx,
            &made,
            || made_as_asked(&a, method, &manifest, &made),
            |checked| lra_reply(&a, method, checked.then_some(&argv[..]), run, &made),
        ) {
            return reply;
        }
    }
    let (out, made) = out_or_free(cx, out.as_deref(), &input.stem(run), |o| {
        Ok(a.outputs(method, o))
    })?;
    let argv = lra_argv(&a, method, run, &out);
    checked_lra(cli, &argv)?;
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
    lra_reply(&a, method, Some(&argv), run, &made)
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

/// The reply to a finished `pinto lra` for `a`: for the contact effect, the
/// verdict counts over the pooled pairs and those that survived the
/// controls, then the command that made the tables (when known), the files
/// and what to do next.
fn lra_reply(
    a: &LraToolArgs,
    method: LraMethod,
    command: Option<&[String]>,
    run: Option<&str>,
    made: &[PathBuf],
) -> anyhow::Result<Vec<Content>> {
    let written: Vec<String> = made.iter().map(|p| shown_path(p)).collect();
    let mut reply = if method == LraMethod::ContactEffect {
        let mut r = effects_summary(&read_pooled_effects(&made[0].to_string_lossy())?);
        r["reading"] =
            "one row per pair and section (`all` pooled); verdict_note says why in words".into();
        if !a.only.is_empty() {
            r["q_scope"] = "q corrected within this --only subset, not over every pair; \
                            recorded under lra.subsets, the full table unchanged"
                .into();
        }
        r
    } else {
        json!({})
    };
    if let Some(argv) = command {
        reply["command"] = format!("pinto {}", script::line(argv)).into();
    }
    reply["written"] = written.into();
    let mut pieces = vec![Content::Json(reply)];
    let checked = run
        .and_then(|run| PintoMetadata::read(Path::new(run)).ok())
        .and_then(|m| m.outputs.lra.simulate);
    if checked.is_none() {
        pieces.push(Content::Next(vec![Next::new(
            "run_lra_simulate",
            a.simulate_args(run),
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
        checked(cli, &lra_simulate_argv(&a, run, &o))?;
        // lra-simulate records no options for its table.
        if let Some(reply) = read_finished(cx, &made, || Ok(false), |_| simulate_reply(None, &made))
        {
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
    simulate_reply(Some(&argv), &made)
}

/// The reply to a finished `pinto lra-simulate`: the comparison table, and
/// the command that made it when known.
fn simulate_reply(command: Option<&[String]>, made: &[PathBuf]) -> anyhow::Result<Vec<Content>> {
    let table = made[0].to_string_lossy();
    let runs = read_sim_runs(&table)?;
    let mut reply = comparison_summary(&runs, read_planted_region(&table).as_deref());
    if let Some(argv) = command {
        reply["command"] = format!("pinto {}", script::line(argv)).into();
    }
    reply["written"] = json!([shown_path(&made[0])]);
    Ok(vec![Content::Json(reply)])
}

/// Each method's share of runs rejecting at 5%, per scenario (type I error
/// where nothing is planted or the field is shifted, power elsewhere), the
/// estimable share, and the means over the confounded nulls and the
/// planted effects.
pub(super) fn comparison_summary(runs: &[output::SimRun], planted: Option<&str>) -> Value {
    let c = compare_runs(runs, planted);
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
