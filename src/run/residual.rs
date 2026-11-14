//! `run_residual` for `pinto mcp`: `pinto residual` set up as the command
//! line takes it, its run checked as the command checks it before anything
//! runs (an lc, cage or dsvd run on raw counts, its data files found, `-o`
//! a folder, nothing written over), and run as a child pinto in the
//! project's records, a background job unless asked otherwise. The reply lists the residual's files for
//! `run_fit`, the second stage of the two-stage fit.

use super::mcp::{as_run, in_background, launch, out_or_free, parsed, rooted};
use super::script;
use crate::cell_activity_graph_embedding::args::FeatureNameMode;
use crate::residual::{Adapt, Excess, Plan, ResidualArgs};
use crate::util::metadata::Run;
use legume_mcp::{Call, Content, ToolSpec};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct ResidualToolArgs {
    pub run: String,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub data_files: Vec<String>,
    #[serde(default)]
    pub excess: Option<String>,
    #[serde(default)]
    pub adapt: Option<String>,
    #[serde(default)]
    pub adapt_cells: Option<f64>,
    #[serde(default)]
    pub feature_names: Option<String>,
    #[serde(default = "in_background")]
    pub background: bool,
}

/// `pinto residual` for `a` writing to `out`: only what was given. A data
/// file with a comma is refused: the command line would split it.
pub(super) fn residual_argv(a: &ResidualToolArgs, out: &str) -> anyhow::Result<Vec<String>> {
    if let Some(f) = a.data_files.iter().find(|f| f.contains(',')) {
        anyhow::bail!("{f:?} has a comma: give each data file as its own entry");
    }
    let mut argv: Vec<String> = vec!["residual".into(), "--from".into(), as_run(&a.run)];
    if !a.data_files.is_empty() {
        argv.extend(["--data-files".into(), rooted(&a.data_files).join(",")]);
    }
    // each `--flag=value` one token, so a negative number reaches the
    // command's own check rather than reading as a flag
    for (flag, value) in [
        ("--excess", a.excess.clone()),
        ("--adapt", a.adapt.clone()),
        ("--adapt-cells", a.adapt_cells.map(|x| x.to_string())),
        ("--feature-names", a.feature_names.clone()),
    ] {
        if let Some(v) = value {
            argv.push(format!("{flag}={v}"));
        }
    }
    argv.extend(["-o".into(), out.into()]);
    Ok(argv)
}

/// `a` as `pinto residual` will run it into `out`: its command line,
/// parsed as the child parses it, and what it writes, checked as the
/// command checks its run before reading any count.
pub(super) fn residual_plan(
    cli: &clap::Command,
    a: &ResidualToolArgs,
    out: &str,
) -> anyhow::Result<(Vec<String>, Plan)> {
    let argv = residual_argv(a, out)?;
    let args: ResidualArgs = parsed(cli, &argv)?;
    let run = Run::open(&args.run)?;
    let plan = Plan::new(&run, &args)?;
    Ok((argv, plan))
}

/// The residual's files as `run_fit` takes them: data files and their
/// coordinates, in the run's order, and the coordinate columns they carry.
pub(super) fn fit_lists(plan: &Plan) -> serde_json::Value {
    let flags = json!({"coord-column-names": plan.coord_names.join(",")});
    json!({"data": plan.zarrs, "coords": plan.coords, "flags": flags})
}

/// `run_residual`.
pub(super) fn residual(
    cli: &clap::Command,
    a: ResidualToolArgs,
    cx: &Call,
) -> anyhow::Result<Vec<Content>> {
    let (out, made) = out_or_free(cx, a.out.as_deref(), "residual", |o| {
        Ok(residual_plan(cli, &a, o)?.1.made())
    })?;
    let (argv, plan) = residual_plan(cli, &a, &out)?;
    let summary = format!(
        "residual of {} into {out}: {} data files",
        as_run(&a.run),
        plan.files.len()
    );
    let reply = json!({
        "command": format!("pinto {}", script::line(&argv)),
        "out": out,
        "record": plan.summary,
        "fit": fit_lists(&plan),
    });
    let then = "pass fit.data, fit.coords and fit.flags to run_fit as its data, coords and \
                flags (for example method dsvd), the second stage; run_lra on that fit then \
                reads the raw counts of the run's own data files, not the excess";
    Ok(vec![Content::Json(launch(
        cx,
        &argv,
        &made,
        &summary,
        a.background,
        reply,
        then,
    )?)])
}

/// The values a flag of type `T` takes, as the command line names them.
fn choices<T: clap::ValueEnum>() -> Vec<String> {
    T::value_variants()
        .iter()
        .filter_map(|v| v.to_possible_value())
        .map(|p| p.get_name().to_string())
        .collect()
}

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![ToolSpec::new(
        "run_residual",
        "The counts a fitted lc, cage or dsvd run leaves unexplained, as new data files \
         (pinto residual), for a second fit to pick up what the first absorbed, such as a \
         domain only one donor has. Each community's raw-count profile is refitted from the \
         run's propensities (by default re-estimated per batch of the run and shrunk toward \
         the pooled profile, so a shift of a whole community in a batch is absorbed and a \
         local patch stays), each cell's mix by EM, and the excess max(y - mu, 0) is kept as \
         counts. Writes {out}/{stem}.zarr with {stem}.coords.parquet per data file (the \
         run's coordinates under its column names) and {out}/residual.json. The reply's \
         fit.data and fit.coords list them for run_fit, in order, and fit.flags the \
         coordinate columns to read; run_lra on that fit reads the raw counts. The run is \
         checked before it starts; it runs as a background job unless asked, and never over \
         an existing file.",
        json!({
            "type": "object",
            "properties": {
                "run": {"type": "string", "description": "The stage-1 run: an lc, cage or dsvd run's prefix or .pinto.json (--from; --run also works)"},
                "out": {"type": "string", "description": "Output folder (-o); left out, a free residual name in today's results folder"},
                "data_files": {"type": "array", "items": {"type": "string"}, "description": "The run's own data files, all of them in its order, when they have moved from where its manifest records them (--data-files)"},
                "excess": {"type": "string", "enum": choices::<Excess>(), "description": "How y - mu becomes a count (--excess) [default: round]"},
                "adapt": {"type": "string", "enum": choices::<Adapt>(), "description": "batch: profiles re-estimated per batch of the run and shrunk toward the pool; none: the pooled profiles, so donor nuisance stays too (--adapt) [default: batch]"},
                "adapt_cells": {"type": "number", "minimum": 0, "description": "Prior weight of the pooled profile under adapt batch, in cells of the community (--adapt-cells) [default: 50]"},
                "feature_names": {"type": "string", "enum": choices::<FeatureNameMode>(), "description": "How the run's fit read feature names: exact (lc and dsvd), or the feature-names a cage run was fitted with (--feature-names) [default: exact]"},
                "background": {"type": "boolean", "description": "Run as a background job and return at once [default: true]; false waits"},
            },
            "required": ["run"],
            "additionalProperties": false,
        }),
    )]
}

#[cfg(test)]
#[path = "tests/residual.rs"]
mod tests;
