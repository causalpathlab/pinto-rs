//! `pinto run`'s tools for `pinto mcp`: the methods and their flags, and
//! a fit set up as `pinto run` sets one up: the command built and checked
//! as pinto parses it, never over a run, run as a child `pinto` in the
//! project's records, as a background job unless asked otherwise (a fit
//! takes minutes to hours, and a call that waited would hold the server).

use super::form::{blamed, check, Kind, Method};
use super::script::{self, is_header, outputs, under};
use super::METHODS;
use crate::mcp::{parse, shown_path};
use legume_mcp::{Call, Content, Handler, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::Path;

/// The run tools, and pinto's command line to read the methods from.
pub struct Tools {
    cli: clap::Command,
}

impl Default for Tools {
    fn default() -> Self {
        use clap::CommandFactory;
        let mut cli = crate::Cli::command();
        cli.build();
        Tools { cli }
    }
}

impl Handler for Tools {
    fn tools(&self) -> Vec<ToolSpec> {
        specs()
    }

    fn call(&mut self, name: &str, args: Value, cx: &Call) -> anyhow::Result<Vec<Content>> {
        match name {
            "run_methods" => self.methods(parse(args)?),
            "run_fit" => self.fit(parse(args)?, cx),
            "run_compare" => compare(parse(args)?, cx),
            "run_predict" => self.predict(parse(args)?, cx),
            "run_region" => region(parse(args)?, cx),
            "run_lra" => super::lra::lra(&self.cli, parse(args)?, cx),
            "run_lra_simulate" => super::lra::lra_simulate(&self.cli, parse(args)?, cx),
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
        let flags: Vec<Value> = method
            .fields
            .iter()
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
                if f.advanced {
                    flag["advanced"] = true.into();
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

    /// `run_predict`: `pinto predict` set up as a fit is, with the model given
    /// by its manifest so the records link the scoring to the fit it scores.
    fn predict(&self, a: PredictToolArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
        let method = Method::new(&self.cli, "predict")?;
        anyhow::ensure!(!a.model.trim().is_empty(), "give model, the fitted run");
        let model = as_run(&a.model);
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
            background: a.background,
        };
        let (mut reply, manifest) =
            self.launch(method, &fit, "pinto predict --help", &summary, cx)?;
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
            let long = flag.trim_start_matches('-');
            let field = method
                .fields
                .iter_mut()
                .find(|f| f.long == long)
                .ok_or_else(|| {
                    anyhow::anyhow!("{} has no flag --{long}: {lister} lists them", a.method)
                })?;
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
            anyhow::bail!(
                "{} exists: a run is not written over; give another out",
                shown_path(there)
            );
        }
        let argv = method.argv(&data, &coords, &batches, &out);
        check(&self.cli, &argv).map_err(|e| match blamed(&e, &method.fields) {
            Some(flag) => anyhow::anyhow!("pinto {} --{flag}: {e}", a.method),
            None => anyhow::anyhow!("pinto {}: {e}", a.method),
        })?;

        let manifest = made[0].clone();
        let mut reply = json!({ "command": format!("pinto {}", script::line(&argv)) });
        let paths = vec![shown_path(&manifest)];
        match spawn(
            cx,
            &argv,
            std::slice::from_ref(&manifest),
            summary,
            a.background,
        )? {
            Some(job) => {
                reply["writes"] = paths.into();
                reply["note"] = format!(
                    "job_wait (id {job}) or job_status until it ends, then view_open {out}"
                )
                .into();
                reply["job"] = job.into();
            }
            None => {
                reply["written"] = paths.into();
                reply["note"] = format!("view_open {out} to look at it").into();
            }
        }
        Ok((reply, manifest))
    }
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
    let mut cmd = legume_mcp::own_command(argv)?;
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

/// A run as the records name it: a prefix whose manifest exists is given as
/// that `.pinto.json`, so the step's inputs link to the fit that made it.
/// `paths`, each as a command line takes it.
pub(super) fn rooted(paths: &[String]) -> Vec<String> {
    paths.iter().map(|p| shown_path(Path::new(p))).collect()
}

pub(super) fn as_run(input: &str) -> String {
    let manifest = crate::util::metadata::run_manifest(input);
    let named = if !Path::new(input).is_file() && Path::new(&manifest).is_file() {
        manifest
    } else {
        input.to_string()
    };
    shown_path(Path::new(&named))
}

/// The output prefix: `out` when given (from the root), else a free name
/// after `base` in today's results folder; refused when one of the outputs
/// `plan` names exists or a running job holds it.
pub(super) fn out_or_free(
    cx: &Call,
    out: Option<&str>,
    base: &str,
    plan: impl Fn(&str) -> anyhow::Result<Vec<std::path::PathBuf>>,
) -> anyhow::Result<(String, Vec<std::path::PathBuf>)> {
    let (out, made) = match out.map(str::trim) {
        Some(o) if !o.is_empty() => {
            let o = shown_path(Path::new(o));
            let made = plan(&o)?;
            (o, made)
        }
        _ => cx.free_prefix(base, &plan, |_| false)?,
    };
    if let Some(there) = made.iter().find(|p| p.exists()) {
        anyhow::bail!(
            "{} exists: nothing is written over; give another out",
            shown_path(there)
        );
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
    reply.insert(
        "command".into(),
        format!("pinto {}", script::line(argv)).into(),
    );
    Ok(reply)
}

/// A comparison's name when the agent gives none: `{A}_vs_{B}` from the
/// runs' file names, with the levels when they are not the final ones.
fn default_name(a: &CompareArgs) -> String {
    let stem = |input: &str, level: &Option<String>| {
        let name = crate::util::metadata::run_stem(input);
        match level.as_deref() {
            Some(l) if l != "final" => format!("{name}.{l}"),
            _ => name,
        }
    };
    format!(
        "{}_vs_{}",
        stem(&a.a.0[0], &a.level_a),
        stem(&a.b.0[0], &a.level_b)
    )
}

/// `run_compare` with `paired`: two models' held-out scores, each side one or
/// more `pinto predict` runs (folds), tested over donors.
fn compare_paired(a: CompareArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
    use crate::compare::paired_path;
    anyhow::ensure!(
        a.level_a.is_none() && a.level_b.is_none() && a.tau.is_none(),
        "level_a, level_b and tau are for a community comparison, not paired"
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
        "-o".into(),
        out,
    ];
    if let Some(m) = &a.donor_map {
        argv.extend(["--donor-map".into(), shown_path(Path::new(m))]);
    }
    if let Some(n) = a.bootstrap {
        argv.extend(["--bootstrap".into(), n.to_string()]);
    }
    let summary = format!("paired held-out comparison of {run_a} vs {run_b}");
    cx.run_making(legume_mcp::own_command(&argv)?, &made, &summary)?;

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
fn compare(a: CompareArgs, cx: &Call) -> anyhow::Result<Vec<Content>> {
    anyhow::ensure!(
        !a.a.0.is_empty() && !a.b.0.is_empty(),
        "give at least one run on each side"
    );
    if a.paired {
        return compare_paired(a, cx);
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
    let mut argv: Vec<String> = vec!["compare".into(), run_a.clone(), run_b.clone()];
    argv.extend(["-o".into(), out.clone()]);
    for (flag, value) in [("--level-a", &a.level_a), ("--level-b", &a.level_b)] {
        if let Some(v) = value {
            argv.extend([flag.into(), v.clone()]);
        }
    }
    if let Some(tau) = a.tau {
        argv.extend(["--tau".into(), tau.to_string()]);
    }
    let summary = format!("community comparison of {run_a} vs {run_b}");
    let cmd = legume_mcp::own_command(&argv)?;
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
            let regions = read_regions(path)?;
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
    level_a: Option<String>,
    #[serde(default)]
    level_b: Option<String>,
    #[serde(default)]
    tau: Option<f64>,
    #[serde(default)]
    paired: bool,
    #[serde(default)]
    donor_map: Option<String>,
    #[serde(default)]
    bootstrap: Option<usize>,
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
    #[serde(default = "in_background")]
    background: bool,
}

pub(super) fn in_background() -> bool {
    true
}

/// A flag's value as typed on the command line.
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

fn specs() -> Vec<ToolSpec> {
    let methods =
        json!({"enum": METHODS, "description": "lc (link community, recommended), cage, or dsvd"});
    vec![
        ToolSpec::new(
            "run_methods",
            "pinto's fitting methods (what pinto run sets up); with a method, its flags: \
             name, help, default, choices, and whether it is a switch.",
            json!({
                "type": "object",
                "properties": { "method": methods },
                "additionalProperties": false,
            }),
        ),
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
                    "background": {"type": "boolean", "description": "Run as a background job and return at once [default: true]; false waits for the fit"},
                },
                "required": ["method", "data", "out"],
                "additionalProperties": false,
            }),
        ),
        ToolSpec::new(
            "run_predict",
            "Score held-out cells on a fitted run (pinto predict). On an lc or dsvd model: each \
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
                    "flags": {"type": "object", "description": "pinto predict's flags by name: held-out (whole-donor|cells), holdout-unit (pair|cell), holdout-fit-fraction, ablate-features, eval-features, null-from, holdout-delta-iters, seed, …"},
                    "background": {"type": "boolean", "description": "Run as a background job and return at once [default: true]"},
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
                    "out": {"type": "string", "description": "Output prefix; left out, a free region name in today's results folder"},
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
                    "a": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}], "description": "Reference run: its prefix or .pinto.json (or a propensity .parquet); with paired, pinto predict runs, a list for several folds"},
                    "b": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}], "description": "Run(s) to set against it, named the same ways"},
                    "out": {"type": "string", "description": "Output prefix; left out, a free compare name in today's results folder"},
                    "level_a": {"type": "string", "description": "Level of A: final, draft, or L1, L2, … [default: final]"},
                    "level_b": {"type": "string", "description": "Level of B, the same way [default: final]"},
                    "tau": {"type": "number", "description": "Share of a community's cells that makes another its partner [default: 0.2]"},
                    "paired": {"type": "boolean", "description": "Compare held-out scores instead: a and b are run_predict outputs of two models on the same held-out cells (same seed and split flags); the donor is the replicate"},
                    "donor_map": {"type": "string", "description": "paired: file of section → donor (two columns); an unlisted section is its own donor"},
                    "bootstrap": {"type": "integer", "description": "paired: donor bootstrap replicates [default: 2000]"},
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
             their response programme, compared with similar cells that do not (method \
             contact-effect, the default)? Writes {out}.lra.parquet and {out}.lra.curve.parquet \
             and records the table in the run's manifest (slot lra.table); with only, a few \
             pairs to {out}.lra.only-<tag>.parquet (slot lra.subsets). The co-expression methods \
             need an lc run and lr_pairs. Runs as a background job unless asked (tens of minutes \
             on a large whole tissue; job_status shows how many pairs are done); the reply when done holds the verdict counts, the pairs \
             whose effect survived the controls (percent change with its 95% CI, q) and the files. Called again with the \
             same run and out once done, it reads the tables instead of running. Never \
             written over.",
            with_lra_input(json!({
                "type": "object",
                "properties": {
                    "run": {"type": "string", "description": "The run whose cells, expression and contacts are read: its .pinto.json (--from); or give data_files and coord_files"},
                    "method": {"enum": ["contact-effect", "coexpression-test", "coexpression-scores"], "description": "What to compute [default: contact-effect]"},
                    "out": {"type": "string", "description": "Output prefix (-o); ending in / names it lra in that folder; left out, a free name after the run in today's results folder"},
                    "lr_pairs": {"type": "string", "description": "Ligand-receptor pairs (two-column TSV/CSV) added to the reference (--lr-pairs); the co-expression methods test only these"},
                    "targets": {"type": "array", "items": {"type": "string"}, "description": "Response programme files (TSV or GMT) added to the reference (--targets)"},
                    "species": {"enum": ["auto", "human", "mouse"], "description": "Species of the reference; auto reads it from the gene-name case [default: auto]"},
                    "only": {"type": "array", "items": {"type": "string"}, "description": "Estimate only these pairs (--only): LIGAND:RECEPTOR (a complex joins subunits with _, TGFB1:TGFBR1_TGFBR2) or a bare LIGAND for all its pairs. Writes {out}.lra.only-<tag>.parquet, recorded under lra.subsets; q is corrected within the subset only"},
                    "n_permutations": {"type": "integer", "minimum": 1, "description": "coexpression-test: sample permutations for the null (--n-permutations) [default: 1000]"},
                    "json_fwer_threshold": {"type": "number", "description": "coexpression-test: FWER cutoff for the JSON sidecar of significant pairs (--json-fwer-threshold) [default: 0.05]"},
                    "min_gene_count": {"type": "number", "description": "Co-expression methods: minimum total count of a ligand or receptor gene for its pair to be scored (--min-gene-count) [default: 50]"},
                    "background": {"type": "boolean", "description": "Run as a background job and return at once [default: true]; false waits"},
                },
                "additionalProperties": false,
            })),
        ),
        ToolSpec::new(
            "run_lra_simulate",
            "Power and calibration check for run_lra on this tissue (pinto lra-simulate --from \
             run): plants ligand effects of known size on the real tissue, with and without \
             hidden spatial confounders, and reports per scenario each method's share of runs \
             rejecting at 5% (type I error where nothing is planted, power elsewhere) beside \
             the co-expression baselines, the estimable share, and the means over the \
             confounded nulls and the planted effects. Writes {out}.lra-simulate.parquet and \
             records it in the run's manifest (slot lra.simulate). study is the design as a \
             JSON object (written beside the output as {out}.study.json and passed as \
             --study); example_study: true returns the built-in study to start from and runs \
             nothing. Runs as a background job unless asked; called again with the same run \
             and out once done, it reads the table instead of running.",
            with_lra_input(json!({
                "type": "object",
                "properties": {
                    "run": {"type": "string", "description": "The run to check on: its .pinto.json (--from); or give data_files and coord_files"},
                    "study": {"type": "object", "description": "Study design: reps, pairs, scenarios (fields left out keep the built-in values; see example_study)"},
                    "example_study": {"type": "boolean", "description": "Return the built-in study as JSON and run nothing"},
                    "out": {"type": "string", "description": "Output prefix (-o); ending in / names it lra in that folder; left out, a free name after the run in today's results folder"},
                    "species": {"enum": ["auto", "human", "mouse"], "description": "Species of the reference [default: auto]"},
                    "background": {"type": "boolean", "description": "Run as a background job and return at once [default: true]; false waits"},
                },
                "additionalProperties": false,
            })),
        ),
    ]
}

/// `schema` with the arguments of what lra reads beside a run, which `run_lra`
/// and `run_lra_simulate` share.
fn with_lra_input(mut schema: Value) -> Value {
    let shared = json!({
                    "data_files": {"type": "array", "items": {"type": "string"}, "description": "Expression data files (.zarr or .h5) to read directly instead of run; need coord_files. With run, they replace the run's"},
                    "coord_files": {"type": "array", "items": {"type": "string"}, "description": "Coordinate files, one per data file (--coord), without run; contacts are the knn_spatial nearest neighbours within a batch"},
                    "batch_files": {"type": "array", "items": {"type": "string"}, "description": "Batch (section) label files, one per data file (--batch-files), without run"},
                    "knn_spatial": {"type": "integer", "minimum": 1, "description": "Spatial contacts per cell when building them from coordinates (--knn-spatial) [default: 5]"},
                    "seed": {"type": "integer", "minimum": 0, "description": "Random seed (--seed) [default: 42]"}
    });
    if let (Some(props), Value::Object(shared)) = (schema["properties"].as_object_mut(), shared) {
        props.extend(shared);
    }
    schema
}

#[cfg(test)]
#[path = "tests/mcp.rs"]
mod tests;
