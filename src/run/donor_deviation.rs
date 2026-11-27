//! `run_donor_deviation` for `pinto mcp`: stage 2 (`pinto apply
//! donor-deviation`) on a finished stage-1 run, set up as the command line
//! takes it, checked before anything runs, and run as a child pinto in the
//! project's records (a background job unless asked otherwise), never over
//! a run.

use super::form::Method;
use super::mcp::{
    as_run, atomic_outputs_arg, background, checked, fill_flags, in_background, launch,
    max_threads_arg, out_arg, out_or_free, rooted, shared_flags,
};
use super::script;
use crate::donor_deviation::outputs::output_paths;
use crate::mcp::shown_path;
use legume_mcp::{Call, Content, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// The command, as pinto names it.
const COMMAND: &str = "donor-deviation";

/// Where an unknown flag's error points.
const LISTER: &str = "pinto apply donor-deviation --help";

/// The command's flags the tool takes as its own fields, by flag and field.
const OWN_FIELDS: [(&str, &str); 3] = [
    ("model", "model"),
    ("donor-of-batch", "donor_of_batch"),
    ("data", "data"),
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DonorDeviationToolArgs {
    model: String,
    donor_of_batch: String,
    #[serde(default)]
    out: Option<String>,
    #[serde(default)]
    data: Vec<String>,
    #[serde(default)]
    flags: Map<String, Value>,
    #[serde(default)]
    max_threads: Option<usize>,
    #[serde(default)]
    atomic_outputs: Option<bool>,
    #[serde(default = "in_background")]
    background: bool,
}

/// The command line of `a` on the stage-1 run `model` (as the records name
/// it, [`as_run`]), writing under `out`, as the child pinto parses it.
pub(super) fn donor_deviation_argv(
    cli: &clap::Command,
    a: &DonorDeviationToolArgs,
    model: &str,
    out: &str,
) -> anyhow::Result<Vec<String>> {
    for (flag, field) in OWN_FIELDS {
        anyhow::ensure!(
            !a.flags.keys().any(|k| k.trim_start_matches('-') == flag),
            "give --{flag} as {field}, not as a flag"
        );
    }
    let mut flags = a.flags.clone();
    flags.insert("model".into(), model.into());
    flags.insert(
        "donor-of-batch".into(),
        shown_path(Path::new(&a.donor_of_batch)).into(),
    );
    if !a.data.is_empty() {
        flags.insert("data".into(), rooted(&a.data).into());
    }
    let mut method = Method::new(cli, COMMAND)?;
    fill_flags(&mut method, &flags, LISTER)?;
    let mut argv = method.argv(&[], &[], &[], out);
    argv.extend(shared_flags(a.max_threads, a.atomic_outputs));
    checked(cli, &argv)?;
    Ok(argv)
}

/// `run_donor_deviation`.
pub(super) fn donor_deviation(
    cli: &clap::Command,
    a: DonorDeviationToolArgs,
    cx: &Call,
) -> anyhow::Result<Vec<Content>> {
    anyhow::ensure!(!a.model.trim().is_empty(), "give model, the stage-1 run");
    anyhow::ensure!(
        !a.donor_of_batch.trim().is_empty(),
        "give donor_of_batch, each batch's donor"
    );
    let model = as_run(&a.model);
    crate::util::metadata::Run::open(&model)?;
    let (out, made) = out_or_free(cx, a.out.as_deref(), COMMAND, |o| {
        Ok(output_paths(o).map(PathBuf::from).to_vec())
    })?;
    let argv = donor_deviation_argv(cli, &a, &model, &out)?;
    let summary = format!("stage-2 donor deviation of {model} into {out}");
    let [settings, genes, without_section, pair_states] = output_paths(&out);
    let reply = json!({
        "command": script::pinto_line(&argv),
        "writes": [&settings, &genes, &without_section, &pair_states],
    });
    let then = format!(
        "read {settings} (the settings and donors); the per (gene, donor) deviations are \
         {genes}, their section jackknife {without_section}, and each pair's state \
         {pair_states}"
    );
    Ok(vec![Content::Json(launch(
        cx,
        &argv,
        &made,
        &summary,
        a.background,
        reply,
        &then,
    )?)])
}

pub(super) fn specs(cli: &clap::Command) -> Vec<ToolSpec> {
    vec![ToolSpec::new(
        "run_donor_deviation",
        "Stage 2 on a finished stage-1 run (pinto apply donor-deviation): with the run's \
         cell-pair embedding held fixed, each gene's pairs of a donor are contrasted with \
         their nearest pairs of other donors (the same shared state elsewhere), and a donor \
         whose pairs carry the gene more than their matches gets a deviation, linear in the \
         pair's state (group lasso: 0 unless the data need it). Needs a cage or dsvd run \
         and at least three donors. Writes {out}.donor_gene_deviation.parquet (per gene and \
         donor: intercept, state_axis_1..), {out}.donor_gene_deviation_without_section.parquet \
         (each section left out: jackknife replicates), {out}.donor_pair_states.parquet (per \
         pair: section, donor, community, state axes, distances to its matches) and \
         {out}.donor_deviation.json; no cutoffs. Runs as a background job unless asked; \
         never over a run.",
        json!({
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "The stage-1 run (cage or dsvd): its prefix or .pinto.json"},
                "donor_of_batch": {"type": "string", "description": "Two-column file: each batch (section) as the run names it, and its donor (tab, comma or spaces; a `batch donor` header is skipped)"},
                "out": out_arg("prefix", COMMAND),
                "data": {"type": "array", "items": {"type": "string"}, "description": "Data files (default: those the run's manifest records)"},
                "flags": {"type": "object", "description": "pinto apply donor-deviation's other flags by name: pairs (cell|pb), matched-pairs, state-axes, group-lasso-penalty, max-anchor-pairs, max-iterations, seed, device (cpu|cuda|metal) (pinto apply donor-deviation --help lists them)"},
                "max_threads": max_threads_arg(cli),
                "atomic_outputs": atomic_outputs_arg(cli),
                "background": background(),
            },
            "required": ["model", "donor_of_batch"],
            "additionalProperties": false,
        }),
    )]
}

#[cfg(test)]
#[path = "tests/donor_deviation.rs"]
mod tests;
