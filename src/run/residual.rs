//! `run_residual` for `pinto mcp`: `pinto residual` set up as the command
//! line takes it, its run checked as the command checks it before anything
//! runs (an lc, cage or dsvd run on raw counts, its data files found, `-o`
//! a folder, nothing written over), and run as a child pinto in the
//! project's records, a background job unless asked otherwise. The reply lists the residual's files for
//! `run_fit`, the second stage of the two-stage fit.

use super::mcp::{as_run, background, in_background, launch, out_arg, out_or_free, parsed, rooted};
use super::script;
use crate::cell_activity_graph_embedding::args::FeatureNameMode;
use crate::residual::{Adapt, Excess, Plan, ResidualArgs};
use crate::tui::flag_value;
use crate::util::cli::choices;
use crate::util::metadata::Run;
use legume_mcp::{Call, Content, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Value};

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
    #[serde(default, alias = "feature_names")]
    pub feature_name_mode: Option<String>,
    #[serde(default = "in_background")]
    pub background: bool,
}

/// `pinto residual` for `a` writing to `out`: only what was given. A data
/// file with a comma is refused: the command line would split it.
pub(super) fn residual_argv(a: &ResidualToolArgs, out: &str) -> anyhow::Result<Vec<String>> {
    if let Some(f) = a.data_files.iter().find(|f| f.contains(',')) {
        anyhow::bail!("{f:?} has a comma: give each data file as its own entry");
    }
    let mut argv: Vec<String> = vec!["residual".into(), flag_value("--from", as_run(&a.run))];
    if !a.data_files.is_empty() {
        argv.push(flag_value("--data-files", rooted(&a.data_files).join(",")));
    }
    for (flag, value) in [
        ("--excess", a.excess.clone()),
        ("--adapt", a.adapt.clone()),
        ("--adapt-cells", a.adapt_cells.map(|x| x.to_string())),
        ("--feature-name-mode", a.feature_name_mode.clone()),
    ] {
        if let Some(v) = value {
            argv.push(flag_value(flag, v));
        }
    }
    argv.push(flag_value("--out", out));
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
        "command": script::pinto_line(&argv),
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

pub(super) fn specs(cli: &clap::Command) -> Vec<ToolSpec> {
    let sub = super::mcp::subcommand(cli, "residual");
    let flag = |id: &str, schema: Value| super::mcp::flag_arg(sub, id, schema);
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
         coordinate columns to read; run_lra on that fit reads the raw counts. It removes no \
         donor effect and cannot correct a stage-1 run already split by donor (as happens \
         once the donor effect is about as large as the cell-type effect): before reading a \
         residual community confined to one donor as biology, check the stage-1 \
         communities' donor make-up (on a simulation, run_sim_pairs' donor_gap), and tell \
         the user. The run is checked before it starts; it runs as a background job unless \
         asked, and never over an existing file.",
        json!({
            "type": "object",
            "properties": {
                "run": {"type": "string", "description": "The stage-1 run: an lc, cage or dsvd run's prefix or .pinto.json (--from; --run also works)"},
                "out": out_arg("folder", "residual"),
                "data_files": flag("data_files", json!({"type": "array", "items": {"type": "string"}})),
                "excess": flag("excess", json!({"type": "string", "enum": choices::<Excess>()})),
                "adapt": flag("adapt", json!({"type": "string", "enum": choices::<Adapt>()})),
                "adapt_cells": flag("adapt_cells", json!({"type": "number", "minimum": 0})),
                "feature_name_mode": flag("feature_names", json!({"type": "string", "enum": choices::<FeatureNameMode>()})),
                "background": background(),
            },
            "required": ["run"],
            "additionalProperties": false,
        }),
    )]
}

#[cfg(test)]
#[path = "tests/residual.rs"]
mod tests;
