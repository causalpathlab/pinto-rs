//! `pinto run`'s tools for `pinto mcp`: the methods and their flags, and
//! a fit set up as `pinto run` sets one up: the command built and checked
//! as pinto parses it, never over a run, run as a child `pinto` in the
//! project's records, as a background job unless asked otherwise (a fit
//! takes minutes to hours, and a call that waited would hold the server).

use super::form::{check, Method};
use super::script::{self, is_header, outputs, under};
use super::{BASELINES, METHODS, MODELS};
use crate::lr_activity::args::{LraMethod, ReferenceSpecies, RegionJackknife};
use crate::lr_activity::ligand_effect::simulate::PairAssignment;
use crate::mcp::{parse, shown_path};
use crate::tui::flag_value;
use crate::util::cli::choices;
use anyhow::Context as _;
use legume_mcp::{Call, Content, Handler, ToolSpec};
use legume_tui::form::{blamed, Kind};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::Path;

/// The run tools, and pinto's command line to read the methods from.
pub struct Tools {
    cli: clap::Command,
    /// The tools, described once from `cli`.
    specs: Vec<ToolSpec>,
}

impl Default for Tools {
    fn default() -> Self {
        let cli = crate::built_cli();
        let mut specs = specs(&cli);
        specs.extend(super::sim::specs(&cli));
        Tools { cli, specs }
    }
}

impl Handler for Tools {
    fn tools(&self) -> Vec<ToolSpec> {
        self.specs.clone()
    }

    fn call(&mut self, name: &str, args: Value, cx: &Call) -> anyhow::Result<Vec<Content>> {
        match name {
            "run_methods" => self.methods(parse(args)?),
            "run_fit" => self.fit(parse(args)?, cx),
            "run_compare" => compare(&self.cli, parse(args)?, cx),
            "run_predict" => self.predict(parse(args)?, cx),
            "run_region" => region(parse(args)?, cx),
            "run_lra" => super::lra::lra(&self.cli, parse(args)?, cx),
            "run_lra_simulate" => super::lra::lra_simulate(&self.cli, parse(args)?, cx),
            "run_lra_merge" => super::lra::lra_merge(&self.cli, parse(args)?, cx),
            "run_simulate" => super::sim::simulate(&self.cli, parse(args)?, cx),
            "run_sim_score_by_set" => super::sim::sim_score(parse(args)?),
            "run_sim_score_by_pair" => super::sim::sim_pairs(parse(args)?),
            "run_sim_baseline" => super::sim::baseline(&self.cli, parse(args)?, cx),
            _ => unreachable!("the server answers unknown tools itself"),
        }
    }
}

/// A method in a line: what it is, and the files it takes.
fn summary(name: &str, method: &Method) -> Value {
    json!({
        "method": name,
        "about": method.about,
        "coords": method.takes_coords(),
        "batches": method.takes_batches(),
    })
}

impl Tools {
    fn methods(&self, a: MethodsArgs) -> anyhow::Result<Vec<Content>> {
        let Some(name) = a.method else {
            let methods = METHODS
                .iter()
                .map(|&m| Ok(summary(m, &Method::new(&self.cli, m)?)))
                .collect::<anyhow::Result<Vec<_>>>()?;
            return Ok(vec![Content::Json(json!({ "methods": methods }))]);
        };
        let method = self.method(&name)?;
        // Hidden flags are tuning knobs: left out, as `--help` leaves them.
        let flags: Vec<Value> = method
            .fields
            .iter()
            .filter(|f| !f.advanced && f.long != MAX_THREADS)
            .map(|f| {
                let mut flag = json!({
                    "flag": format!("--{}", f.long),
                    "help": f.help,
                    "default": f.default,
                });
                match &f.kind {
                    Kind::Flag { .. } => flag["switch"] = true.into(),
                    Kind::Choice(values) => {
                        flag["choices"] = values.iter().filter(|v| !v.is_empty()).cloned().collect()
                    }
                    Kind::Text => {}
                }
                if f.required {
                    flag["required"] = true.into();
                }
                flag
            })
            .collect();
        let mut out = summary(&name, &method);
        out["flags"] = flags.into();
        Ok(vec![Content::Json(out)])
    }

    fn method(&self, name: &str) -> anyhow::Result<Method> {
        anyhow::ensure!(
            METHODS.contains(&name),
            "method {name:?}: one of {}",
            METHODS.join(", ")
        );
        Method::new(&self.cli, name)
    }

    fn fit(&self, a: FitArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
        let method = self.method(&a.method)?;
        let summary = ablation_summary(&a.flags)
            .unwrap_or_else(|| format!("{} fit of {}", a.method, a.data.join(", ")));
        let (reply, _) = self.launch(method, &a, "run_methods", &summary, cx)?;
        Ok(vec![Content::Json(reply)])
    }

    /// `run_predict`: `pinto apply predict` set up as a fit is, with the model given
    /// by its manifest so the records link the scoring to the fit it scores.
    fn predict(&self, a: PredictToolArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
        let method = Method::new(&self.cli, "predict")?;
        anyhow::ensure!(!a.model.trim().is_empty(), "give model, the fitted run");
        let model = as_run(&a.model);
        crate::util::metadata::Run::open(&model)?;
        let mut flags = a.flags;
        anyhow::ensure!(
            !flags.keys().any(|k| k.trim_start_matches('-') == "model"),
            "give the model as `model`, not as a flag"
        );
        flags.insert("model".into(), Value::String(model.clone()));
        let summary = format!("held-out scoring of {} against {model}", a.data.join(", "));
        let fit = FitArgs {
            method: "predict".into(),
            data: a.data,
            coords: a.coords,
            batches: a.batches,
            out: a.out,
            flags,
            max_threads: a.max_threads,
            atomic_outputs: a.atomic_outputs,
            background: a.background,
        };
        let (mut reply, manifest) =
            self.launch(method, &fit, "pinto apply predict --help", &summary, cx)?;
        // Run to its end: the scoring's totals, so the agent need not open the
        // manifest.
        if !fit.background {
            if let Some(h) = crate::util::metadata::PintoMetadata::read(&manifest)?.holdout {
                reply["holdout"] = serde_json::to_value(h)?;
            }
        }
        Ok(vec![Content::Json(reply)])
    }

    /// Set `method` up from `a`, check it as pinto parses it, and run it as a
    /// child pinto in the records: a background job, or to its end. `lister`
    /// is where an unknown flag's error points. Returns the reply and the
    /// run's manifest.
    fn launch(
        &self,
        mut method: Method,
        a: &FitArgs,
        lister: &str,
        summary: &str,
        cx: &Call,
    ) -> anyhow::Result<(Value, std::path::PathBuf)> {
        anyhow::ensure!(!a.data.is_empty(), "give the data files");
        anyhow::ensure!(!a.out.trim().is_empty(), "give out, the output prefix");
        for (what, takes, files) in [
            ("coordinate", method.takes_coords(), &a.coords),
            ("batch", method.takes_batches(), &a.batches),
        ] {
            anyhow::ensure!(
                files.is_empty() || takes,
                "{} takes no {what} files",
                a.method
            );
            anyhow::ensure!(
                files.is_empty() || files.len() == a.data.len(),
                "{what} files: one per data file ({} for {})",
                files.len(),
                a.data.len()
            );
        }
        for (flag, value) in &a.flags {
            let flag = flag.trim_start_matches('-');
            anyhow::ensure!(
                flag != MAX_THREADS,
                "give --{MAX_THREADS} as max_threads, not as a flag"
            );
            anyhow::ensure!(
                flag != ATOMIC_OUTPUTS,
                "give --{ATOMIC_OUTPUTS} as atomic_outputs, not as a flag"
            );
            let field = method.field_mut(flag).ok_or_else(|| {
                anyhow::anyhow!("{} has no flag --{flag}: {lister} lists them", a.method)
            })?;
            let long = field.long.clone();
            field.value = match (&field.kind, value) {
                (Kind::Flag { .. }, Value::Bool(on)) => on.to_string(),
                (Kind::Flag { .. }, v) => {
                    anyhow::bail!("--{long} is a switch: true or false, not {v}")
                }
                (_, Value::Array(vs)) => vs.iter().map(text).collect::<Vec<_>>().join(","),
                (_, v) => text(v),
            };
        }
        // An `out` naming a folder, or a header, is named for the method,
        // as `pinto run` names it; `.` is this folder.
        let out = if a.out == "." {
            "./".into()
        } else {
            a.out.clone()
        };
        let out = if is_header(&out) {
            under(&out, &a.method)
        } else {
            out
        };
        // Paths from the root: the fit, and the records' replay of it from
        // the project, read the same files wherever the server started.
        let (data, coords, batches) = (rooted(&a.data), rooted(&a.coords), rooted(&a.batches));
        let out = shown_path(Path::new(&out));
        let made = outputs(Path::new(""), &out);
        if let Some(there) = made.iter().find(|p| p.exists()) {
            anyhow::bail!("{} exists: {NOT_WRITTEN_OVER}", shown_path(there));
        }
        let mut argv = method.argv(&data, &coords, &batches, &out);
        argv.extend(shared_flags(a.max_threads, a.atomic_outputs));
        check(&self.cli, &argv).map_err(|e| match blamed(&e, &method.fields) {
            Some(flag) => anyhow::anyhow!("pinto {} --{flag}: {e}", named(&argv)),
            None => anyhow::anyhow!("pinto {}: {e}", named(&argv)),
        })?;

        let manifest = made[0].clone();
        let reply = json!({ "command": script::pinto_line(&argv) });
        let mut reply = launch(
            cx,
            &argv,
            std::slice::from_ref(&manifest),
            summary,
            a.background,
            reply,
            &format!("view_open {out}"),
        )?;
        let key = if reply.get("job").is_some() {
            "writes"
        } else {
            "written"
        };
        reply[key] = vec![shown_path(&manifest)].into();
        Ok((reply, manifest))
    }
}

/// The command line checked as pinto parses it.
pub(super) fn checked(cli: &clap::Command, argv: &[String]) -> anyhow::Result<()> {
    check(cli, argv).map_err(|e| anyhow::anyhow!("pinto {}: {e}", named(argv)))
}

/// The words of `argv` naming its command: `fit lc`, `lra simulate`.
pub(super) fn named(argv: &[String]) -> String {
    argv[..crate::cli_path::path_len(argv)].join(" ")
}

/// The command line checked, then parsed into its command's arguments, as
/// the child pinto will parse them.
pub(super) fn parsed<T: clap::FromArgMatches>(
    cli: &clap::Command,
    argv: &[String],
) -> anyhow::Result<T> {
    let matches = super::form::matches(cli, argv)
        .map_err(|e| anyhow::anyhow!("pinto {}: {e}", named(argv)))?;
    Ok(T::from_arg_matches(&matches)?)
}

/// `argv` run as a child pinto making `made` (a job when `background`),
/// and `reply` with the job and a note on what to do `then`.
pub(super) fn launch(
    cx: &Call,
    argv: &[String],
    made: &[std::path::PathBuf],
    summary: &str,
    background: bool,
    mut reply: Value,
    then: &str,
) -> anyhow::Result<Value> {
    match spawn(cx, argv, made, summary, background)? {
        Some(job) => {
            reply["note"] =
                format!("job_wait (id {job}) or job_status until it ends, then {then}").into();
            reply["job"] = job.into();
        }
        None => reply["note"] = then.into(),
    }
    Ok(reply)
}

/// Run a child pinto for `argv` making `made`, at the log level `pinto run`
/// starts a step with (so its progress reaches job_status as it does the
/// run screen): in the background, giving the job's id, or to its end.
pub(super) fn spawn(
    cx: &Call,
    argv: &[String],
    made: &[std::path::PathBuf],
    summary: &str,
    background: bool,
) -> anyhow::Result<Option<String>> {
    let mut cmd = crate::tui::own_command(argv)?;
    if std::env::var_os("RUST_LOG").is_none() {
        cmd.env("RUST_LOG", script::LOG_LEVEL);
    }
    if background {
        Ok(Some(cx.spawn_job(cmd, made, summary)?))
    } else {
        cx.run_making(cmd, made, summary)?;
        Ok(None)
    }
}

/// The records' line for a fit that is an ablation: `{kind} ablation of
/// {parent}`, so INDEX.tsv reads as what the run is for.
fn ablation_summary(flags: &Map<String, Value>) -> Option<String> {
    let flag = |name: &str| {
        flags
            .iter()
            .find(|(k, _)| k.trim_start_matches('-') == name)
            .map(|(_, v)| v)
            .filter(|v| !v.is_null() && **v != Value::Bool(false))
    };
    let kind = crate::util::ablation::AblationKind::ALL
        .into_iter()
        .find(|k| flag(k.flag()).is_some())?
        .name();
    let parent = flag("ablation-parent").map_or_else(|| "an unnamed parent".into(), text);
    Some(format!("{kind} ablation of {parent}"))
}

/// `paths`, each as a command line takes it.
pub(super) fn rooted(paths: &[String]) -> Vec<String> {
    paths.iter().map(|p| shown_path(Path::new(p))).collect()
}

/// A run as the records name it: a prefix whose manifest exists is given as
/// that `.pinto.json`, so the step's inputs link to the fit that made it.
pub(super) fn as_run(input: &str) -> String {
    let manifest = crate::util::metadata::run_manifest(input);
    let named = if !Path::new(input).is_file() && Path::new(&manifest).is_file() {
        manifest
    } else {
        input.to_string()
    };
    shown_path(Path::new(&named))
}

/// What a tool says when its outputs are already there: no run's outputs
/// are written over.
pub(super) const NOT_WRITTEN_OVER: &str = "nothing is written over; give another out";

/// The output prefix: `out` when given (from the root), else a free name
/// after `base` in today's results folder (passing over a name a file
/// holds); refused when one of the outputs `plan` names exists or a running
/// job holds it.
pub(super) fn out_or_free(
    cx: &Call,
    out: Option<&str>,
    base: &str,
    plan: impl Fn(&str) -> anyhow::Result<Vec<std::path::PathBuf>>,
) -> anyhow::Result<(String, Vec<std::path::PathBuf>)> {
    out_or_free_by(cx, out, base, plan, |_| false)
}

/// [`out_or_free`], a free name also passing over those `taken` holds.
pub(super) fn out_or_free_by(
    cx: &Call,
    out: Option<&str>,
    base: &str,
    plan: impl Fn(&str) -> anyhow::Result<Vec<std::path::PathBuf>>,
    taken: impl Fn(&str) -> bool,
) -> anyhow::Result<(String, Vec<std::path::PathBuf>)> {
    let (out, made) = match out.map(str::trim) {
        Some(o) if !o.is_empty() => {
            let o = shown_path(Path::new(o));
            let made = plan(&o)?;
            (o, made)
        }
        // a name a file holds is passed over: nothing is written under it
        _ => cx.free_prefix(base, &plan, |o| Path::new(o).is_file() || taken(o))?,
    };
    if let Some(there) = made.iter().find(|p| p.exists()) {
        anyhow::bail!("{} exists: {NOT_WRITTEN_OVER}", shown_path(there));
    }
    cx.ensure_free(&made)?;
    Ok((out, made))
}

/// A report a child wrote, as the reply: every field but `drop`, with the
/// command that made it.
fn reply_from(path: &Path, drop: &[&str], argv: &[String]) -> anyhow::Result<Map<String, Value>> {
    let Value::Object(mut reply) = serde_json::from_str(&std::fs::read_to_string(path)?)? else {
        anyhow::bail!("{} is not a JSON object", shown_path(path));
    };
    for key in drop {
        reply.remove(*key);
    }
    reply.insert("command".into(), script::pinto_line(argv).into());
    Ok(reply)
}

/// A comparison's name when the agent gives none: `{A}_vs_{B}` from the
/// runs' file names, with the levels when they are not the final ones.
fn default_name(a: &CompareArgs) -> String {
    let stem = |input: &str, level: Option<&str>| {
        let name = crate::util::metadata::run_stem(input);
        match level.map(str::trim) {
            Some(l) if l != crate::compare::DEFAULT_LEVEL => format!("{name}.{l}"),
            _ => name,
        }
    };
    // one level for both, or A,B, as `--level` reads it
    let level = a.level.as_deref();
    let (la, lb) = level.map_or((None, None), |l| match l.split_once(',') {
        Some((x, y)) => (Some(x), Some(y)),
        None => (Some(l), Some(l)),
    });
    format!("{}_vs_{}", stem(&a.a.0[0], la), stem(&a.b.0[0], lb))
}

/// `run_compare` with `paired`: two models' held-out scores, each side one or
/// more `pinto apply predict` runs (folds), tested over donors.
fn compare_paired(cli: &clap::Command, a: CompareArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
    use crate::compare::paired_path;
    anyhow::ensure!(
        a.level.is_none() && a.tau.is_none(),
        "level and tau are for a community comparison, not paired"
    );
    let (out, made) = out_or_free(
        cx,
        a.out.as_deref(),
        &format!("{}_paired", default_name(&a)),
        |out| Ok(vec![paired_path(out).into()]),
    )?;
    let side = |runs: &[String]| runs.iter().map(|r| as_run(r)).collect::<Vec<_>>().join(",");
    let (run_a, run_b) = (side(&a.a.0), side(&a.b.0));
    let mut argv: Vec<String> = vec![
        "compare".into(),
        "--paired".into(),
        run_a.clone(),
        run_b.clone(),
        flag_value("--out", out),
    ];
    if let Some(m) = &a.donor_map {
        argv.push(flag_value("--donor-map", shown_path(Path::new(m))));
    }
    if let Some(n) = a.bootstrap {
        argv.push(flag_value("--bootstrap", n));
    }
    argv.extend(shared_flags(a.max_threads, a.atomic_outputs));
    checked(cli, &argv)?;
    let summary = format!("paired held-out comparison of {run_a} vs {run_b}");
    cx.run_making(crate::tui::own_command(&argv)?, &made, &summary)?;

    let mut reply = reply_from(&made[0], &["a", "b"], &argv)?;
    reply.insert("written".into(), json!([shown_path(&made[0])]));
    Ok(vec![Content::Json(reply.into())])
}

/// One run, or several (the folds of one side of a paired comparison).
#[derive(Deserialize)]
#[serde(from = "OneOrMany")]
struct Runs(Vec<String>);

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl From<OneOrMany> for Runs {
    fn from(v: OneOrMany) -> Self {
        match v {
            OneOrMany::One(s) => Runs(vec![s]),
            OneOrMany::Many(v) => Runs(v),
        }
    }
}

/// `run_compare`: `pinto compare` as a child pinto, recorded with what it
/// writes. It reads two propensity tables, so it runs in the call.
fn compare(cli: &clap::Command, a: CompareArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
    anyhow::ensure!(
        !a.a.0.is_empty() && !a.b.0.is_empty(),
        "give at least one run on each side"
    );
    if a.paired {
        return compare_paired(cli, a, cx);
    }
    use crate::compare::{contingency_path, json_path};
    anyhow::ensure!(
        a.a.0.len() == 1 && a.b.0.len() == 1,
        "a community comparison takes one run on each side; several are for paired: true"
    );
    anyhow::ensure!(
        a.donor_map.is_none() && a.bootstrap.is_none(),
        "donor_map and bootstrap are for paired: true"
    );
    let (out, made) = out_or_free(cx, a.out.as_deref(), &default_name(&a), |out| {
        Ok(vec![json_path(out).into(), contingency_path(out).into()])
    })?;
    let (run_a, run_b) = (as_run(&a.a.0[0]), as_run(&a.b.0[0]));
    let mut argv: Vec<String> = vec![
        "compare".into(),
        run_a.clone(),
        run_b.clone(),
        flag_value("--out", out),
    ];
    if let Some(level) = &a.level {
        argv.push(flag_value("--level", level));
    }
    if let Some(tau) = a.tau {
        argv.push(flag_value("--tau", tau));
    }
    argv.extend(shared_flags(a.max_threads, a.atomic_outputs));
    // checked as the child parses it, so a value it refuses says why here
    checked(cli, &argv)?;
    let summary = format!("community comparison of {run_a} vs {run_b}");
    let cmd = crate::tui::own_command(&argv)?;
    cx.run_making(cmd, &made, &summary)?;

    let mut reply = reply_from(
        &made[0],
        &["matching", "communities_a", "communities_b", "contingency"],
        &argv,
    )?;
    reply.insert(
        "written".into(),
        made.iter()
            .map(|p| shown_path(p))
            .collect::<Vec<_>>()
            .into(),
    );
    reply.insert(
        "note".into(),
        format!(
            "{} lists every community's partners, its match and Jaccard; {} has the full table",
            shown_path(&made[0]),
            shown_path(&made[1])
        )
        .into(),
    );
    Ok(vec![Content::Json(reply.into())])
}

/// `run_region`: the cells of a run inside a polygon, written as a cell list
/// for `run_fit`'s `mask-cells`. The polygon is drawn in the view's world
/// coordinates (what view_open and view_map report), or read from a GeoJSON
/// file in each section's own coordinates.
fn region(a: RegionArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
    use crate::util::ablation::{cells_in_regions, inside, read_regions, section_of};
    let run = crate::view::data::Run::open(&a.run)?;
    let (cells, of, how): (Vec<Box<str>>, usize, String) = match (&a.polygon, &a.geojson) {
        (Some(points), None) => {
            anyhow::ensure!(points.len() >= 3, "a polygon needs at least three points");
            let geom = run.load_geometry()?;
            let ring: Vec<(f64, f64)> = points.iter().map(|p| (p[0], p[1])).collect();
            let rings = vec![ring];
            let cells = (0..geom.n())
                .filter(|&i| inside(&rings, f64::from(geom.x[i]), f64::from(geom.y[i])))
                .map(|i| geom.names[i].clone())
                .collect();
            (cells, geom.n(), "a polygon in world coordinates".into())
        }
        (None, Some(path)) => {
            // Section coordinates: the world positions less each tile's offset
            // are the cells' own, so read them from the run's cells table.
            let table = crate::util::parquet_io::read_cells_table(
                Path::new(
                    &run.resolve(
                        run.meta
                            .outputs
                            .cells
                            .as_deref()
                            .ok_or_else(|| anyhow::anyhow!("{}: no cells table", a.run))?,
                    ),
                ),
                run.meta.outputs.coord_columns.as_deref(),
            )?;
            let x: Vec<f32> = table.coords.iter().map(|c| c.0).collect();
            let y: Vec<f32> = table.coords.iter().map(|c| c.1).collect();
            let regions = read_regions(&shown_path(Path::new(path)))
                .map_err(|e| anyhow::anyhow!("geojson {e:#}"))?;
            let (picked, _) = cells_in_regions(&table.names, &x, &y, &regions)?;
            let cells = table
                .names
                .iter()
                .filter(|n| picked.contains(*n))
                .cloned()
                .collect();
            (
                cells,
                table.names.len(),
                format!("{} polygon(s) of {path}", regions.len()),
            )
        }
        _ => anyhow::bail!("give one of polygon (world coordinates) or geojson (a file)"),
    };
    anyhow::ensure!(!cells.is_empty(), "no cell of {} is inside {how}", a.run);
    let mut per_section: Vec<(&str, usize)> = Vec::new();
    for c in &cells {
        let s = section_of(c);
        match per_section.iter_mut().find(|(n, _)| *n == s) {
            Some((_, k)) => *k += 1,
            None => per_section.push((s, 1)),
        }
    }
    let (_, made) = out_or_free(cx, a.out.as_deref(), "region", |out| {
        Ok(vec![format!("{out}.cells.txt").into()])
    })?;
    let mut text = String::from("cell\n");
    for c in &cells {
        text.push_str(c);
        text.push('\n');
    }
    legume_numeric::matrix::common_io::mkdir_parent(&made[0].to_string_lossy())
        .with_context(|| format!("making the folder of {}", shown_path(&made[0])))?;
    cx.write_file(&made[0], &text)?;
    let path = shown_path(&made[0]);
    Ok(vec![Content::Json(json!({
        "cells": cells.len(),
        "of": of,
        "sections": per_section.iter().map(|(s, n)| json!({"section": s, "cells": n})).collect::<Vec<_>>(),
        "region": how,
        "written": [path.clone()],
        "note": format!("mask it in a refit: run_fit with flags {{\"mask-cells\": \"{path}\", \"ablation-parent\": \"{}\"}}", as_run(&a.run)),
    }))])
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegionArgs {
    run: String,
    #[serde(default)]
    polygon: Option<Vec<[f64; 2]>>,
    #[serde(default)]
    geojson: Option<String>,
    #[serde(default)]
    out: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictToolArgs {
    model: String,
    data: Vec<String>,
    #[serde(default)]
    coords: Vec<String>,
    #[serde(default)]
    batches: Vec<String>,
    out: String,
    #[serde(default)]
    flags: Map<String, Value>,
    #[serde(default)]
    max_threads: Option<usize>,
    #[serde(default)]
    atomic_outputs: Option<bool>,
    #[serde(default = "in_background")]
    background: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompareArgs {
    a: Runs,
    b: Runs,
    #[serde(default)]
    out: Option<String>,
    #[serde(default)]
    level: Option<String>,
    #[serde(default)]
    tau: Option<f64>,
    #[serde(default)]
    paired: bool,
    #[serde(default)]
    donor_map: Option<String>,
    #[serde(default)]
    bootstrap: Option<usize>,
    #[serde(default)]
    max_threads: Option<usize>,
    #[serde(default)]
    atomic_outputs: Option<bool>,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct MethodsArgs {
    method: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FitArgs {
    method: String,
    data: Vec<String>,
    #[serde(default)]
    coords: Vec<String>,
    #[serde(default)]
    batches: Vec<String>,
    out: String,
    #[serde(default)]
    flags: Map<String, Value>,
    #[serde(default)]
    max_threads: Option<usize>,
    #[serde(default)]
    atomic_outputs: Option<bool>,
    #[serde(default = "in_background")]
    background: bool,
}

pub(super) fn in_background() -> bool {
    true
}

/// pinto's `--max-threads`, without its dashes.
const MAX_THREADS: &str = "max-threads";

/// The `max_threads` field of each tool that starts a pinto command:
/// pinto's `--max-threads`, described as its help has it.
pub(super) fn max_threads_arg(cli: &clap::Command) -> Value {
    let mut arg = flag_arg(
        subcommand(cli, "lc"),
        "max_threads",
        json!({"type": "integer", "minimum": 1}),
    );
    if let Some(said) = arg["description"].as_str() {
        let default = format!("[default: {}]", crate::util::common::DEFAULT_MAX_THREADS);
        arg["description"] = said.replace(&default, SERVER_MAX_THREADS).into();
    }
    arg
}

/// The default of a tool's `max_threads`.
const SERVER_MAX_THREADS: &str = "[default: the server's own --max-threads]";

/// `--max-threads` for the command line: the tool's, else the server's
/// own when it was started with one.
fn max_threads_flag(n: Option<usize>) -> Option<String> {
    n.or_else(crate::tui::parent_max_threads)
        .map(|n| flag_value(&format!("--{MAX_THREADS}"), n))
}

/// The `atomic_outputs` field of each tool that starts a run:
/// pinto's `--atomic-outputs`, described as its help has it.
pub(super) fn atomic_outputs_arg(cli: &clap::Command) -> Value {
    let mut arg = flag_arg(
        subcommand(cli, "lc"),
        "atomic_outputs",
        json!({"type": "boolean"}),
    );
    if let Some(said) = arg["description"].as_str() {
        arg["description"] = said
            .replace("[default: false]", SERVER_ATOMIC_OUTPUTS)
            .into();
    }
    arg
}

/// pinto's `--atomic-outputs`, without its dashes.
const ATOMIC_OUTPUTS: &str = "atomic-outputs";

/// The default of a tool's `atomic_outputs`.
const SERVER_ATOMIC_OUTPUTS: &str = "[default: the server's own --atomic-outputs]";

/// `--atomic-outputs` for the command line: as the tool says, true or
/// false, else as the server was started.
fn atomic_outputs_flag(on: Option<bool>) -> Option<String> {
    on.unwrap_or_else(crate::util::staging::enabled)
        .then(|| format!("--{ATOMIC_OUTPUTS}"))
}

/// The flags every tool that starts a pinto command gives it:
/// `--max-threads` ([`max_threads_flag`]) and `--atomic-outputs`
/// ([`atomic_outputs_flag`]).
pub(super) fn shared_flags(
    max_threads: Option<usize>,
    atomic_outputs: Option<bool>,
) -> impl Iterator<Item = String> {
    max_threads_flag(max_threads)
        .into_iter()
        .chain(atomic_outputs_flag(atomic_outputs))
}

/// The `background` field every tool that starts a job takes.
pub(super) fn background() -> Value {
    json!({"type": "boolean", "description": "Run as a background job and return at once [default: true]; false waits"})
}

/// The `out` field of a tool that names a free output when none is given:
/// a `kind` (prefix or folder) named after `free` in today's results folder.
pub(super) fn out_arg(kind: &str, free: &str) -> Value {
    json!({"type": "string", "description": format!("Output {kind} (-o); left out, a free {free} name in today's results folder")})
}

/// A flag's value as typed on the command line.
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

fn specs(cli: &clap::Command) -> Vec<ToolSpec> {
    let (lra_cmd, sim_cmd) = (subcommand(cli, "lra"), subcommand(cli, "lra simulate"));
    let merge_cmd = subcommand(cli, "lra merge");
    let lra = |id: &str, schema: Value| flag_arg(lra_cmd, id, schema);
    let sim = |id: &str, schema: Value| flag_arg(sim_cmd, id, schema);
    let methods = json!({
        "enum": METHODS,
        "description": format!(
            "{} (lc and dsvd are recommended); or a spatial baseline: {}",
            MODELS.join(", "),
            BASELINES.join(", ")
        ),
    });
    vec![
        ToolSpec::new(
            "run_methods",
            &format!(
                "pinto's fitting methods (what pinto run sets up): {}, then the spatial \
                 baselines {}; with a method, its flags: name, help, default, choices, and \
                 whether it is a switch (hidden tuning flags are left out).",
                MODELS.join(", "),
                BASELINES.join(", ")
            ),
            json!({
                "type": "object",
                "properties": { "method": methods },
                "additionalProperties": false,
            }),
        )
        .read_only(),
        ToolSpec::new(
            "run_fit",
            "Fit one method on data files, with the command pinto run would build: checked as \
             pinto parses it, then run as a child pinto and kept in the project's records. A fit \
             takes minutes to hours, so it runs as a background job: the call returns the job at \
             once, to follow with job_status / job_wait / job_cancel (background: false waits for \
             it, for a small data set). A run is never written over. When it ends, view_open its \
             out.",
            json!({
                "type": "object",
                "properties": {
                    "method": methods,
                    "data": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Expression data files (.zarr, .zarr.zip, .h5)"},
                    "coords": {"type": "array", "items": {"type": "string"}, "description": "Coordinate files, one per data file (barcode, x, y, …)"},
                    "batches": {"type": "array", "items": {"type": "string"}, "description": "Batch label files, one per data file"},
                    "out": {"type": "string", "description": "Output prefix (-o); ending in /, _, - or . names the run for its method (res/ → res/lc)"},
                    "flags": {"type": "object", "description": "The method's flags by name, as run_methods lists them (a switch takes true or false): {\"n-communities\": 12}"},
                    "max_threads": max_threads_arg(cli),
                    "atomic_outputs": atomic_outputs_arg(cli),
                    "background": background(),
                },
                "required": ["method", "data", "out"],
                "additionalProperties": false,
            }),
        ),
        ToolSpec::new(
            "run_predict",
            "Score held-out cells on a fitted run (pinto apply predict). On an lc or dsvd model: each \
             cell pair of the held-out sample is split (count thinning, or flags.ablate-features \
             to hide a feature list from the fit and score exactly it, as senna does), its \
             community mix is fitted on one part and the other part is scored, beside the \
             training-composition null, a pooled K=1 profile and the ceiling. Writes \
             {out}.predictive.parquet (eval_llik_per_count, eval_null_llik_per_count, \
             eval_pooled_k1_llik_per_count, …) and {out}.pinto.json with a holdout block. Fit \
             the model without the held-out donor first (run_fit on the other files). A cage \
             model is applied to the sample instead. Runs as a background job unless asked; \
             never over a run.",
            json!({
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "The fitted run: its prefix or .pinto.json"},
                    "data": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "The held-out sample's data files"},
                    "coords": {"type": "array", "items": {"type": "string"}, "description": "Its coordinate files, one per data file (pairs from its spatial graph)"},
                    "batches": {"type": "array", "items": {"type": "string"}, "description": "Batch label files, one per data file (one δ per batch)"},
                    "out": {"type": "string", "description": "Output prefix (-o)"},
                    "flags": {"type": "object", "description": "pinto apply predict's flags by name: held-out (whole-donor|cells), holdout-unit (pair|cell), ablate-features, eval-features, null-from, seed (pinto apply predict --help lists them)"},
                    "max_threads": max_threads_arg(cli),
                    "atomic_outputs": atomic_outputs_arg(cli),
                    "background": background(),
                },
                "required": ["model", "data", "out"],
                "additionalProperties": false,
            }),
        ),
        ToolSpec::new(
            "run_region",
            "The cells of a run inside a region, written as a cell list ({out}.cells.txt) to mask \
             in a refit: run_fit with flags mask-cells (cut-edges by default: the masked cells' \
             edges are cut, so the graph does not join cells across the hole) and \
             ablation-parent. Give the region as polygon, points in the world coordinates \
             view_open and view_map report (sections tiled side by side, y down), or as geojson, \
             a file of polygons in each section's own coordinates with a section property (as \
             QuPath exports). Returns the count per section.",
            json!({
                "type": "object",
                "properties": {
                    "run": {"type": "string", "description": "The run whose cells are picked: its prefix or .pinto.json"},
                    "polygon": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, "minItems": 3, "description": "[[x, y], …] in world coordinates"},
                    "geojson": {"type": "string", "description": "A GeoJSON file of polygons in section coordinates"},
                    "out": out_arg("prefix", "region"),
                },
                "required": ["run"],
                "additionalProperties": false,
            }),
        ),
        ToolSpec::new(
            "run_compare",
            "Compare two runs' communities over the cells they share (pinto compare): ARI, NMI, \
             a one-to-one matching with each pair's Jaccard, and which communities of A vanish, \
             split or merge in B (a community's partners hold at least tau of its cells). Use it \
             to set an ablation or refit against its parent. Writes {out}.compare.json and \
             {out}.contingency.parquet, kept in the project's records; returns the summary. \
             With paired: true, a and b are run_predict outputs of two models on the same \
             held-out cells (a list per side for leave-one-donor-out folds): units are joined on \
             their cell pair, and the donor is the replicate (per-donor Δ nats/count, exact sign \
             and Wilcoxon tests over donors, a donor bootstrap interval); writes \
             {out}.paired.json.",
            json!({
                "type": "object",
                "properties": {
                    "a": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}], "description": "Reference run: its prefix or .pinto.json (or a propensity .parquet); with paired, pinto apply predict runs, a list for several folds"},
                    "b": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}], "description": "Run(s) to set against it, named the same ways"},
                    "out": out_arg("prefix", "compare"),
                    "level": {"type": "string", "description": "Levels to read: final, draft or a level tag (L1, L2, …); one for both, or A,B [default: final]"},
                    "tau": {"type": "number", "description": "Share of a community's cells that makes another its partner [default: 0.2]"},
                    "paired": {"type": "boolean", "description": "Compare held-out scores instead: a and b are run_predict outputs of two models on the same held-out cells (same seed and split flags); the donor is the replicate"},
                    "donor_map": {"type": "string", "description": "paired: file of section → donor (two columns); an unlisted section is its own donor"},
                    "bootstrap": {"type": "integer", "description": "paired: donor bootstrap replicates [default: 2000]"},
                    "max_threads": max_threads_arg(cli),
                    "atomic_outputs": atomic_outputs_arg(cli),
                },
                "required": ["a", "b"],
                "additionalProperties": false,
            }),
        ),
        ToolSpec::new(
            "run_lra",
            "Ligand-receptor analysis on a run's spatial contacts (pinto lra --from run), or on \
             data files with coordinates: per \
             ligand-receptor pair, do receivers touching ligand-carrying neighbours change \
             their response program, compared with similar cells that do not (method \
             contact-effect, the default)? Writes {out}.lra.parquet and {out}.lra.curve.parquet \
             and records the table in the run's manifest (slot lra.table); with \
             marginal_baselines also {out}.lra.celltype.parquet (slot lra.celltype); with only, \
             a few pairs to {out}.lra.only-<tag>.parquet (and .curve.parquet, and \
             .celltype.parquet with marginal_baselines; slot lra.subsets); with shard I/N, only \
             shard I of the pairs to {out}.lra.shard-I-of-N.cbor (no table, nothing in the \
             manifest), each shard its own job, then run_lra_merge writes the tables. The \
             co-expression methods \
             need an lc run and lr_pairs. Runs as a background job unless asked (tens of minutes \
             on a large whole tissue; job_status shows how many pairs are done); the reply when done holds how many pairs have an \
             estimate, those with the smallest p (percent change with its 95% CI, p) and the files. Called again with the \
             same run and out once done, it reads the tables instead of running; a call asking \
             for other options than those the manifest records for them is refused. Never \
             written over.",
            with_lra_input(cli, "lra", json!({
                "type": "object",
                "properties": {
                    "run": {"type": "string", "description": "The run whose cells, expression and contacts are read: its .pinto.json (--from); or give data_files and coord_files"},
                    "method": lra("method", json!({"enum": choices::<LraMethod>()})),
                    "out": {"type": "string", "description": "Output prefix (-o); ending in / names it lra in that folder; left out, a free name after the run in today's results folder"},
                    "lr_pairs": lra("lr_pairs", json!({"type": "string"})),
                    "targets": long_flag_arg(lra_cmd, "targets", json!({"type": "array", "items": {"type": "string"}})),
                    "species": lra("species", json!({"enum": choices::<ReferenceSpecies>()})),
                    "only": long_flag_arg(lra_cmd, "only", json!({"type": "array", "items": {"type": "string"}})),
                    "shard": long_flag_arg(lra_cmd, "shard", json!({"type": "string", "pattern": "^[0-9]+/[0-9]+$"})),
                    "link_community": long_flag_arg(lra_cmd, "link_community", json!({"type": "integer"})),
                    "n_permutations": lra("n_permutations", json!({"type": "integer", "minimum": 1})),
                    "min_gene_count": lra("min_gene_count", json!({"type": "number"})),
                    "marginal_baselines": lra("marginal_baselines", json!({"type": "boolean"})),
                    "marginal_permutations": lra("marginal_permutations", json!({"type": "integer", "minimum": 1})),
                    "null_permutations": lra("null_permutations", json!({"type": "integer", "minimum": 0})),
                    "background": background(),
                },
                "additionalProperties": false,
            })),
        ),
        ToolSpec::new(
            "run_lra_simulate",
            "Power and calibration check for run_lra on this tissue (pinto lra simulate --from \
             run): plants ligand effects of known size on the real tissue, with and without \
             hidden spatial confounders, and reports per scenario each method's share of runs \
             rejecting at 5% (type I error where nothing is planted, power elsewhere) beside \
             the co-expression baselines, the estimable share, and the means over the \
             confounded nulls and the planted effects. Writes {out}.lra-simulate.parquet and \
             records it in the run's manifest (slot lra.simulate). With one, set, pairs or \
             null_pairs it runs one scenario instead of the built-in grid: its effect planted on pairs \
             pairs and withheld on null_pairs others under the same hidden field, writing \
             {out}.lra-simulate.parquet (columns planted and, for each estimator, effect_dr, effect_ipw, effect_outcome and \
             effect_unadj with _se, _df and _p, each SE a delete-one-tile jackknife; for AUROC \
             and AUPRC) and then {out}.lra-simulate.json (the resolved scenario, the pairs, \
             their labels and seeds, the reference), recording nothing in the manifest; with \
             no out it takes today's free {run}-scenario name, never the run's own prefix. Runs \
             as a background job unless asked; called again with the same run and out once \
             done, it reads the table instead of running. With shard I/N it runs only shard I \
             of the runs, each shard its own job (a Slurm array task), and run_lra_merge on \
             the N shard files writes the call's tables.",
            with_lra_input(cli, "lra simulate", json!({
                "type": "object",
                "properties": {
                    "run": {"type": "string", "description": "The run to check on: its .pinto.json (--from); or give data_files and coord_files"},
                    "out": {"type": "string", "description": "Output prefix (-o); ending in / names it lra in that folder; left out, a free name after the run in today's results folder"},
                    "species": sim("species", json!({"enum": choices::<ReferenceSpecies>()})),
                    "lr_pairs": sim("lr_pairs", json!({"type": "string"})),
                    "targets": long_flag_arg(sim_cmd, "targets", json!({"type": "array", "items": {"type": "string"}})),
                    "plant_region": sim("plant_region", json!({"type": "string"})),
                    "reference_data": long_flag_arg(sim_cmd, "reference_data", json!({"type": "string"})),
                    "reference_data_labels": sim("reference_data_labels", json!({"type": "string"})),
                    "one": sim("one", json!({"type": "boolean"})),
                    "set": long_flag_arg(sim_cmd, "set", json!({"type": "array", "items": {"type": "string"}})),
                    "pairs": sim("pairs", json!({"type": "integer", "minimum": 1})),
                    "null_pairs": sim("null_pairs", json!({"type": "integer", "minimum": 1})),
                    "pair_assignment": sim("pair_assignment", json!({"enum": choices::<PairAssignment>()})),
                    "shard": long_flag_arg(sim_cmd, "shard", json!({"type": "string", "pattern": "^[0-9]+/[0-9]+$"})),
                    "background": background(),
                },
                "additionalProperties": false,
            })),
        ),
        ToolSpec::new(
            "run_lra_merge",
            "Put the shards of one run_lra run with shard together (pinto lra merge): reads every \
             shard file ({out}.lra.shard-I-of-N.cbor, in any order), checks they are the run's N \
             shards (the same settings and pairs, each shard once, every pair once), and writes \
             the tables a run_lra without shard writes, {out}.lra.parquet and \
             {out}.lra.curve.parquet (and {out}.lra.celltype.parquet when the shards ran \
             marginal_baselines), the p-values and co-expression's local fdr over every pair, recorded in the \
             run's manifest (else {out}.pinto.json). Runs as a background job unless asked; the \
             reply when done is run_lra's: how many pairs have an estimate, those with the \
             smallest p and the files. Called again with the same out once done, it reads the \
             tables instead of running. Never written over.",
            json!({
                "type": "object",
                "properties": {
                    "shards": flag_arg(merge_cmd, "shards", json!({"type": "array", "items": {"type": "string"}, "minItems": 1})),
                    "out": flag_arg(merge_cmd, "out", json!({"type": "string"})),
                    "max_threads": max_threads_arg(cli),
                    "atomic_outputs": atomic_outputs_arg(cli),
                    "background": background(),
                },
                "required": ["shards", "out"],
                "additionalProperties": false,
            }),
        ),
    ]
}

/// `schema` with the arguments of what lra reads beside a run, which `run_lra`
/// and `run_lra_simulate` share, described by `pinto {command}`'s flags.
fn with_lra_input(cli: &clap::Command, command: &str, mut schema: Value) -> Value {
    let sub = subcommand(cli, command);
    let flag = |id: &str, schema: Value| flag_arg(sub, id, schema);
    let files = || json!({"type": "array", "items": {"type": "string"}});
    let shared = json!({
        "data_files": flag("data_files", files()),
        "coord_files": flag("coord_files", files()),
        "batch_files": flag("batch_files", files()),
        "knn_spatial": flag("knn_spatial", json!({"type": "integer", "minimum": 1})),
        "seed": flag("seed", json!({"type": "integer", "minimum": 0})),
        "cell_types": flag("cell_types", json!({"type": "string"})),
        "region": flag("region", json!({"type": "string"})),
        "region_jackknife": flag("region_jackknife", json!({"enum": choices::<RegionJackknife>()})),
        "abf_prior_scale": flag("abf_prior_scale", json!({"type": "number", "exclusiveMinimum": 0})),
        "max_threads": max_threads_arg(cli),
        "atomic_outputs": atomic_outputs_arg(cli),
    });
    if let (Some(props), Value::Object(shared)) = (schema["properties"].as_object_mut(), shared) {
        props.extend(shared);
    }
    schema
}

/// The command `name`, at the top or under its verb.
pub(super) fn subcommand<'c>(cli: &'c clap::Command, name: &str) -> &'c clap::Command {
    crate::cli_path::find(cli, name)
        .unwrap_or_else(|| panic!("pinto has no command {name}"))
        .1
}

/// A tool argument that passes the argument `id` of `sub` (a pinto
/// command): `schema` keeps its type (and any enum), and its description
/// is the flag's own short help, as `--help` shows it, the flag, and
/// clap's default, so the two cannot drift.
pub(super) fn flag_arg(sub: &clap::Command, id: &str, schema: Value) -> Value {
    described(sub, id, schema, false)
}

/// [`flag_arg`] described by the flag's long help, for an argument whose
/// long help carries a vocabulary or a format an agent needs to fill it in.
pub(super) fn long_flag_arg(sub: &clap::Command, id: &str, schema: Value) -> Value {
    described(sub, id, schema, true)
}

fn described(sub: &clap::Command, id: &str, mut schema: Value, long: bool) -> Value {
    let arg = sub
        .get_arguments()
        .find(|a| a.get_id() == id)
        .unwrap_or_else(|| panic!("pinto {} has no argument {id}", sub.get_name()));
    let (short, long_help) = (arg.get_help(), arg.get_long_help());
    let help = if long {
        long_help.or(short)
    } else {
        short.or(long_help)
    };
    let help = help.map(ToString::to_string).unwrap_or_default();
    let mut said = help.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(flag) = arg.get_long() {
        said.push_str(&format!(" (--{flag})"));
    }
    let default: Vec<String> = arg
        .get_default_values()
        .iter()
        .map(|v| v.to_string_lossy().into_owned())
        .collect();
    if !default.is_empty() {
        said.push_str(&format!(" [default: {}]", default.join(",")));
    }
    schema["description"] = said.into();
    schema
}

#[cfg(test)]
#[path = "tests/mcp.rs"]
mod tests;
