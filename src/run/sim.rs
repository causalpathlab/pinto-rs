//! `run_simulate`, `run_sim_score`, `run_sim_pairs` and `run_sim_baseline`
//! for `pinto mcp`:
//! `pinto srt-simulate` and `pinto srt-sim-baseline` set up as the command
//! line takes them, checked before anything runs, and run as a child pinto
//! in the project's records (a background job unless asked otherwise); and
//! `pinto srt-sim-score`'s two tests and `pinto srt-sim-pairs`'s pair sets,
//! computed in the call, as they take a moment.

use super::mcp::{in_background, launch, out_or_free, parsed, rooted};
use super::script;
use crate::mcp::shown_path;
use crate::simulate::baseline::{check_inputs, BaselineArgs};
use crate::simulate::pairs::{
    pairs_header, score_input, PairOptions, SpotLayout, DEFAULT_KNN, DEFAULT_PAIRS, DEFAULT_SEED,
};
use crate::simulate::panel::{read_coords, section_name};
use crate::simulate::params::{SimParams, SCENARIOS, TERMS};
use crate::simulate::run::{
    backend_path, coords_path, manifest_path, resolve_params, truth_run_path, SimulateArgs,
};
use crate::simulate::score::{read_labels, score, table_header, table_rows, TruthSpots};
use crate::simulate::write::COORD_COLUMNS;
use crate::tui::flag_value;
use legume_mcp::{Call, Content, ToolSpec};
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};

// * Arguments

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct SimulateToolArgs {
    #[serde(default)]
    pub scenario: Option<String>,
    #[serde(default)]
    pub set: Vec<String>,
    #[serde(default)]
    pub coords: Vec<String>,
    #[serde(default)]
    pub coord_column_names: Vec<String>,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default = "in_background")]
    pub background: bool,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct SimScoreToolArgs {
    pub truth: String,
    pub labels: Vec<String>,
    #[serde(default = "final_level")]
    pub level: String,
    #[serde(default = "label_column")]
    pub label_column: String,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct SimPairsToolArgs {
    pub truth: String,
    pub vectors: Vec<String>,
    #[serde(default = "final_level")]
    pub level: String,
    #[serde(default)]
    pub pairs: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub knn: Option<usize>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct SimBaselineToolArgs {
    pub truth: String,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default = "in_background")]
    pub background: bool,
}

fn final_level() -> String {
    "final".into()
}

fn label_column() -> String {
    "label".into()
}

// * run_simulate

/// `a` as `pinto srt-simulate` will run it into `out`: its command line,
/// parsed as the child parses it, and its parameters and coordinate files
/// checked as it checks them. An entry with a comma is refused: the command
/// line would split it.
pub(super) fn simulation(
    cli: &clap::Command,
    a: &SimulateToolArgs,
    out: &str,
) -> anyhow::Result<(Vec<String>, SimulateArgs, SimParams)> {
    if let Some(c) = a
        .coords
        .iter()
        .chain(&a.coord_column_names)
        .find(|c| c.contains(','))
    {
        anyhow::bail!("{c:?} has a comma: give each file or column as its own entry");
    }
    let argv = simulate_argv(a, out);
    let args: SimulateArgs = parsed(cli, &argv)?;
    let p = resolve_params(&args)?;
    for f in &args.coords {
        read_coords(f, &args.coord_column_names)
            .map_err(|e| anyhow::anyhow!("coords {f}: {e:#}"))?;
    }
    Ok((argv, args, p))
}

/// The variance shares the run has, as shares_note spells them out, to six
/// places.
fn shares_json(p: &SimParams) -> serde_json::Value {
    let shares = TERMS
        .iter()
        .zip(p.effective_shares())
        .map(|(&k, x)| (k.to_string(), json!((x * 1e6).round() / 1e6)));
    serde_json::Value::Object(shares.collect())
}

/// The sections a simulation with `p` writes at `out`, as `run_fit` takes
/// them: data files and their coordinates, in the same order.
pub(super) fn fit_lists(out: &str, p: &SimParams) -> serde_json::Value {
    let sections = |path: fn(&str, &str) -> String| -> Vec<String> {
        (0..p.n_sections())
            .map(|s| path(out, &section_name(s)))
            .collect()
    };
    // run_fit's default coordinate names do not include the written ones
    let flags = json!({"coord-column-names": COORD_COLUMNS.join(",")});
    json!({"data": sections(backend_path), "coords": sections(coords_path), "flags": flags})
}

/// `pinto srt-simulate` for `a` writing to `out`: only what was given, and
/// column names only with the coordinate files they name.
pub(super) fn simulate_argv(a: &SimulateToolArgs, out: &str) -> Vec<String> {
    let mut argv: Vec<String> = vec!["srt-simulate".into()];
    if let Some(s) = &a.scenario {
        argv.push(flag_value("--scenario", s));
    }
    for s in &a.set {
        argv.push(flag_value("--set", s));
    }
    if !a.coords.is_empty() {
        argv.push(flag_value("--coords", rooted(&a.coords).join(",")));
        if !a.coord_column_names.is_empty() {
            argv.push(flag_value(
                "--coord-column-names",
                a.coord_column_names.join(","),
            ));
        }
    }
    argv.push(flag_value("--out", out));
    argv
}

/// What a simulation at `out` makes: the folder, its manifest, and the
/// truth run the viewer opens.
pub(super) fn sim_outputs(out: &str) -> Vec<PathBuf> {
    vec![
        PathBuf::from(out),
        manifest_path(out).into(),
        truth_run_path(out).into(),
    ]
}

/// `run_simulate`.
pub(super) fn simulate(
    cli: &clap::Command,
    a: SimulateToolArgs,
    cx: &Call,
) -> anyhow::Result<Vec<Content>> {
    let (out, made) = out_or_free(cx, a.out.as_deref(), "sim", |o| Ok(sim_outputs(o)))?;
    let (argv, args, p) = simulation(cli, &a, &out)?;
    let truth = truth_run_path(&out);
    let fit = fit_lists(&out, &p);
    let summary = format!(
        "srt-simulate {} into {out}: {} sections; truth run {truth}",
        args.scenario,
        p.n_sections()
    );
    let reply = json!({
        "command": script::pinto_line(&argv),
        "out": out,
        "truth": truth,
        "fit": fit,
        "shares": shares_json(&p),
        "shares_note": p.shares_summary(),
    });
    let open = format!(
        "view_open {truth} to see the planted domains; pass fit.data, fit.coords and fit.flags \
         to run_fit as its data, coords and flags; score the fits with \
         run_sim_score, truth {truth}"
    );
    Ok(vec![Content::Json(launch(
        cx,
        &argv,
        &made,
        &summary,
        a.background,
        reply,
        &open,
    )?)])
}

/// The error for a truth that cannot be read, with what to give instead.
fn bad_truth(truth: &str, e: anyhow::Error) -> anyhow::Error {
    anyhow::anyhow!("truth {truth}: {e:#}; give run_simulate's truth or out folder")
}

// * run_sim_score

/// `run_sim_score`: each labelling scored against the truth, as a table
/// (one row per labelling and donor-specific domain) and as JSON. One that
/// cannot be scored is listed with its error and the rest are scored, as
/// `pinto srt-sim-score` does; the call fails only when none can be.
pub(super) fn sim_score(a: SimScoreToolArgs) -> anyhow::Result<Vec<Content>> {
    anyhow::ensure!(!a.labels.is_empty(), "give labels: runs or label tables");
    let t = TruthSpots::read(&a.truth).map_err(|e| bad_truth(&a.truth, e))?;
    let mut lines = vec![table_header()];
    let (mut scores, mut failed, mut warnings) = (Vec::new(), Vec::new(), Vec::new());
    for input in &a.labels {
        match read_labels(input, &a.level, &a.label_column).and_then(|l| score(&t, &l)) {
            Ok(s) => {
                if let Some(w) = s.unmatched_warning() {
                    warnings.push(json!({ "labels": input, "warning": w }));
                }
                lines.extend(table_rows(input, &s));
                scores.push(json!({ "labels": input, "level": a.level, "score": s }));
            }
            Err(e) => failed.push(json!({ "labels": input, "error": format!("{e:#}") })),
        }
    }
    anyhow::ensure!(
        !scores.is_empty(),
        "no labelling could be scored: {failed:?}"
    );
    let mut reply = json!({ "truth": shown_path(Path::new(&a.truth)), "scores": scores });
    if !failed.is_empty() {
        reply["failed"] = failed.into();
    }
    if !warnings.is_empty() {
        reply["warnings"] = warnings.into();
    }
    Ok(vec![Content::Text(lines.join("\n")), Content::Json(reply)])
}

// * run_sim_pairs

/// `run_sim_pairs`: each input's vectors scored on the pair sets, as a
/// table (one row per input and donor-specific domain) and as JSON, with
/// `pinto srt-sim-pairs`'s defaults for what is not given. One that cannot
/// be scored is listed with its error and the rest are scored; the call
/// fails only when none can be.
pub(super) fn sim_pairs(a: SimPairsToolArgs) -> anyhow::Result<Vec<Content>> {
    anyhow::ensure!(!a.vectors.is_empty(), "give vectors: runs or vector tables");
    let o = PairOptions::new(
        a.pairs.unwrap_or(DEFAULT_PAIRS),
        a.seed.unwrap_or(DEFAULT_SEED),
        a.knn.unwrap_or(DEFAULT_KNN),
    )?;
    let t = TruthSpots::read(&a.truth).map_err(|e| bad_truth(&a.truth, e))?;
    let layout = SpotLayout::read(&a.truth, &t).map_err(|e| bad_truth(&a.truth, e))?;
    let mut lines = vec![pairs_header()];
    let (mut scores, mut failed) = (Vec::new(), Vec::new());
    for input in &a.vectors {
        match score_input(&t, &layout, input, &a.level, &o) {
            Ok((rows, record)) => {
                lines.extend(rows);
                scores.push(record);
            }
            Err(e) => failed.push(json!({ "vectors": input, "error": format!("{e:#}") })),
        }
    }
    anyhow::ensure!(!scores.is_empty(), "no vectors could be scored: {failed:?}");
    let mut reply = json!({ "truth": shown_path(Path::new(&a.truth)), "scores": scores });
    if !failed.is_empty() {
        reply["failed"] = failed.into();
    }
    Ok(vec![Content::Text(lines.join("\n")), Content::Json(reply)])
}

// * run_sim_baseline

/// `pinto srt-sim-baseline` for `a` on the simulation `truth`, writing the
/// table `table`: only what was given.
pub(super) fn baseline_argv(a: &SimBaselineToolArgs, truth: &str, table: &str) -> Vec<String> {
    let mut argv: Vec<String> = vec!["srt-sim-baseline".into(), flag_value("--truth", truth)];
    if let Some(m) = &a.method {
        argv.push(flag_value("--method", m));
    }
    if let Some(k) = a.k {
        argv.push(flag_value("--k", k));
    }
    if let Some(s) = a.seed {
        argv.push(flag_value("--seed", s));
    }
    argv.push(flag_value("--out", table));
    argv
}

/// `run_sim_baseline`: its command line parsed and checked as the child
/// will check it (its settings, and the simulation's files there) before
/// anything runs.
pub(super) fn baseline(
    cli: &clap::Command,
    a: SimBaselineToolArgs,
    cx: &Call,
) -> anyhow::Result<Vec<Content>> {
    let truth = shown_path(Path::new(&a.truth));
    let base = a
        .method
        .as_deref()
        .map_or_else(|| "baseline".into(), |m| format!("baseline-{m}"));
    let (out, made) = out_or_free(cx, a.out.as_deref(), &base, |o| {
        Ok(vec![PathBuf::from(format!("{o}.tsv"))])
    })?;
    let labels = format!("{out}.tsv");
    let argv = baseline_argv(&a, &truth, &labels);
    let args: BaselineArgs = parsed(cli, &argv)?;
    args.validate()?;
    check_inputs(&truth).map_err(|e| bad_truth(&a.truth, e))?;
    let summary = format!("srt-sim-baseline into {labels}");
    let reply = json!({
        "command": script::pinto_line(&argv),
        "labels": labels,
    });
    let then = format!("run_sim_score with truth {truth} and labels [{labels}]");
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

// * Tools

pub(super) fn specs() -> Vec<ToolSpec> {
    let scenarios: Vec<String> = SCENARIOS
        .iter()
        .map(|s| format!("{} ({})", s.name, s.about))
        .collect();
    vec![
        ToolSpec::new(
            "run_simulate",
            "Simulate several donors' spatial sections with a planted truth (pinto \
             srt-simulate): location sets a spot's domain, the domain its likely cell states, \
             the states its expression; donor and section nuisance (to be removed) and a domain \
             one donor alone has (to be kept) are set as shares of the log rate's variance, and \
             spot noise is what the shares leave: raising one share with the others fixed \
             shrinks the noise. The reply's shares and shares_note spell out what a setting \
             implies; relay shares_note to the user whenever a share is set. \
             Writes {out}/sections/S*.zarr with S*.coords.parquet (row_um, col_um), the truth \
             tables and run in {out}/truth (view_open {out}/truth/truth.pinto.json), and \
             {out}/manifest.json. The reply's fit.data and fit.coords list the sections for \
             run_fit, in order, and fit.flags the coordinate columns to read; its truth is \
             run_sim_score's truth. The settings and coordinate files are checked before it starts; it runs as a background job unless asked, and \
             never over an existing out.",
            json!({
                "type": "object",
                "properties": {
                    "scenario": {"type": "string", "enum": SCENARIOS.iter().map(|s| s.name).collect::<Vec<_>>(), "description": format!("Named scenario to start from [default: {}]: {}", SCENARIOS[0].name, scenarios.join("; "))},
                    "set": {"type": "array", "items": {"type": "string"}, "description": "Change any simulation parameter, key=value (pinto srt-simulate --set; the manifest records them all). The main dials: grid_side and grid_spacing_um (the default square grid, 40 x 40 spots 100 um apart), n_donors, sections_per_donor, n_domains, patch_kind (none, new-state, new-mix), patch_share, patch_length_um, domain_length_um, pve_type, pve_donor, pve_inter, pve_section, clean_donors (a list, [0]), copula_strength, nuisance_off (true with the same seed draws the exact counterfactual), seed"},
                    "coords": {"type": "array", "items": {"type": "string"}, "description": "Spot coordinate files, used by the sections in turn, positions only (--coords); the square grid when none"},
                    "coord_column_names": {"type": "array", "items": {"type": "string"}, "description": "Coordinate columns in those files [default: row_um, col_um]"},
                    "out": {"type": "string", "description": "Output folder (-o); left out, a free sim name in today's results folder"},
                    "background": {"type": "boolean", "description": "Run as a background job and return at once [default: true]; false waits"},
                },
                "additionalProperties": false,
            }),
        ),
        ToolSpec::new(
            "run_sim_score",
            "Score labellings against a run_simulate truth (pinto srt-sim-score), as two tests. \
             1, donor-invariant domains, on spots of shared domains: shared_ari (labels against \
             domains), donor_dependence (how much the labels depend on the donor within a \
             domain, 0 to 1, pooled and for each shared domain as donor_dependence_by_domain). 2, each donor-specific domain against its \
             best-matching label: f1, precision (low: shared spots absorbed), recall, \
             owner_share (1: the label is confined to the owning donor). Returns a table, one \
             row per labelling and donor-specific domain, and the scores as JSON. Fast: runs in \
             the call.",
            json!({
                "type": "object",
                "properties": {
                    "truth": {"type": "string", "description": "The simulation: run_simulate's truth (its truth run) or out folder"},
                    "labels": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Labellings: pinto runs (prefix or .pinto.json, scored at level), or tables (TSV or parquet) with a cell column and label_column"},
                    "level": {"type": "string", "description": "Level of a run to score: final, draft, or L1, L2, ... (the truth run itself: domain) [default: final]"},
                    "label_column": {"type": "string", "description": "Label column of a table (lupin's rounds: cell_type) [default: label]"},
                },
                "required": ["truth", "labels"],
                "additionalProperties": false,
            }),
        )
        .read_only(),
        ToolSpec::new(
            "run_sim_pairs",
            "Score per-cell vectors (a latent mapping) against a run_simulate truth on pairs of \
             spots, with no clustering (pinto srt-sim-pairs): the cosine of two spots' vectors, \
             by AUROC and AUPRC, each set with its prevalence (the AUPRC of chance). \
             cross_donor: shared-domain spots of two donors, positive when they share a domain; \
             within_donor: the same within one donor, the reference; donor_gap: within less \
             cross, what the donor costs the mapping. patches, one per donor-specific domain: \
             pairs of its spots (positive) against one of its spots with one outside it \
             (negative); closest_domain and closest_auroc: the domain it is told apart from \
             worst (0.5: folded into it). spatial: each spot's knn nearest neighbours in its \
             section, positive when they share a domain. Returns a table of the AUROCs, one row \
             per input and donor-specific domain, and every number as JSON. Runs in the call.",
            json!({
                "type": "object",
                "properties": {
                    "truth": {"type": "string", "description": "The simulation: run_simulate's truth (its truth run) or out folder"},
                    "vectors": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Per-cell vectors: pinto runs (prefix or .pinto.json, their propensity at level), or tables (TSV or parquet, a cage run's {prefix}.cell_embedding.parquet among them) with a cell column and numeric columns"},
                    "level": {"type": "string", "description": "Level of a run's propensity: final, draft, or L1, L2, ... (the truth run itself: domain, donor or state) [default: final]"},
                    "pairs": {"type": "integer", "minimum": 2, "description": format!("Pairs drawn for each sampled set [default: {DEFAULT_PAIRS}]")},
                    "seed": {"type": "integer", "minimum": 0, "description": format!("Seed of the pair draws [default: {DEFAULT_SEED}]")},
                    "knn": {"type": "integer", "minimum": 1, "description": format!("Spatial neighbours of each spot, within its section [default: {DEFAULT_KNN}]")},
                },
                "required": ["truth", "vectors"],
                "additionalProperties": false,
            }),
        )
        .read_only(),
        ToolSpec::new(
            "run_sim_baseline",
            "Label a run_simulate simulation with a standard single-cell pipeline (pinto \
             srt-sim-baseline), for run_sim_score to score next to the fits: pooled (the scanpy \
             recipe on all sections, no batch handling), harmony (the same with Harmony on the \
             PCs, batch = donor) or per-donor (each donor alone, its clusters matched to the \
             largest donor's by marker correlation). Leiden is tuned toward the true number of \
             domains unless k is given. Writes a cell/label table at {out}.tsv and names it as \
             labels; runs as a background job unless asked, never over an existing table.",
            json!({
                "type": "object",
                "properties": {
                    "truth": {"type": "string", "description": "The simulation: run_simulate's truth (its truth run) or out folder"},
                    "method": {"type": "string", "enum": ["pooled", "harmony", "per-donor"], "description": "Pipeline [default: pooled]"},
                    "k": {"type": "integer", "minimum": 1, "description": "Clusters to tune Leiden toward [default: the domains among the spots clustered]"},
                    "seed": {"type": "integer", "minimum": 0, "description": "Leiden and Harmony seed [default: 1]"},
                    "out": {"type": "string", "description": "Output prefix, the table at {out}.tsv; left out, a free baseline-<method> name in today's results folder"},
                    "background": {"type": "boolean", "description": "Run as a background job and return at once [default: true]; false waits"},
                },
                "required": ["truth"],
                "additionalProperties": false,
            }),
        ),
    ]
}

#[cfg(test)]
#[path = "tests/sim.rs"]
mod tests;
