//! `run_simulate`, `run_sim_score_by_set`, `run_sim_score_by_pair` and `run_sim_baseline`
//! for `pinto mcp`:
//! `pinto sim new` and `pinto sim baseline` set up as the command
//! line takes them, checked before anything runs, and run as a child pinto
//! in the project's records (a background job unless asked otherwise); and
//! `pinto sim score-by-set`'s two tests and `pinto sim score-by-pair`'s pair sets,
//! computed in the call, as they take a moment.

use super::mcp::{
    atomic_outputs_arg, background, in_background, launch, max_threads_arg, out_arg, out_or_free,
    parsed, rooted, shared_flags,
};
use super::script;
use crate::baselines::single_cell::{check_inputs, BaselineArgs};
use crate::mcp::shown_path;
use crate::sim_eval::pairs::{
    pairs_header, score_input, PairOptions, SpotLayout, DEFAULT_KNN, DEFAULT_PAIRS, DEFAULT_SEED,
};
use crate::sim_eval::score::{
    read_labels, score, scores_document, table_header, table_rows, TruthSpots,
};
use crate::simulate::panel::{read_coords, section_name};
use crate::simulate::params::{SimParams, SCENARIOS, TERMS};
use crate::simulate::run::{
    backend_path, coords_path, manifest_path, resolve_params, truth_run_path, SimulateArgs,
};
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
    pub reference_data: Option<String>,
    #[serde(default)]
    pub reference_data_labels: Option<String>,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub max_threads: Option<usize>,
    #[serde(default)]
    pub atomic_outputs: Option<bool>,
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
    pub n_clusters: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub out: Option<String>,
    #[serde(default)]
    pub max_threads: Option<usize>,
    #[serde(default)]
    pub atomic_outputs: Option<bool>,
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

/// `a` as `pinto sim new` will run it into `out`: its command line,
/// parsed as the child parses it, and its parameters and coordinate files
/// checked as it checks them. An entry with a comma is refused: the command
/// line would split it.
pub(super) fn simulation(
    cli: &clap::Command,
    a: &SimulateToolArgs,
    out: &str,
) -> anyhow::Result<(Vec<String>, SimulateArgs, &'static str, SimParams)> {
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
    let (scenario, p) = resolve_params(&args)?;
    for f in &args.coords {
        read_coords(f, &args.coord_column_names)
            .map_err(|e| anyhow::anyhow!("coords {f}: {e:#}"))?;
    }
    for (what, f) in [
        ("reference_data", args.reference_data.as_deref()),
        (
            "reference_data_labels",
            args.reference_data_labels.as_deref(),
        ),
    ] {
        if let Some(f) = f {
            anyhow::ensure!(std::path::Path::new(f).exists(), "{what} {f}: no such file");
        }
    }
    Ok((argv, args, scenario, p))
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

/// `pinto sim new` for `a` writing to `out`: only what was given, and
/// column names only with the coordinate files they name.
pub(super) fn simulate_argv(a: &SimulateToolArgs, out: &str) -> Vec<String> {
    let mut argv: Vec<String> = vec!["sim".into(), "new".into()];
    if let Some(s) = &a.scenario {
        argv.push(flag_value("--scenario", s));
    }
    for s in &a.set {
        argv.push(flag_value("--set", s));
    }
    if !a.coords.is_empty() {
        argv.push(flag_value("--coord", rooted(&a.coords).join(",")));
        if !a.coord_column_names.is_empty() {
            argv.push(flag_value(
                "--coord-column-names",
                a.coord_column_names.join(","),
            ));
        }
    }
    // the labels go in whether or not the data does, so the command's own
    // check refuses labels with no data
    if let Some(r) = &a.reference_data {
        argv.push(flag_value(
            "--reference-data",
            &rooted(std::slice::from_ref(r))[0],
        ));
    }
    if let Some(l) = &a.reference_data_labels {
        argv.push(flag_value(
            "--reference-data-labels",
            &rooted(std::slice::from_ref(l))[0],
        ));
    }
    argv.extend(shared_flags(a.max_threads, a.atomic_outputs));
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
    let (argv, _, scenario, p) = simulation(cli, &a, &out)?;
    let truth = truth_run_path(&out);
    let fit = fit_lists(&out, &p);
    let (lr_pairs, targets) = crate::simulate::write::lr_pairs_paths(&out);
    let summary = format!(
        "sim new {scenario} into {out}: {} sections; truth run {truth}",
        p.n_sections()
    );
    let reply = json!({
        "command": script::pinto_line(&argv),
        "out": out,
        "truth": truth,
        "fit": fit,
        "lra": {
            "lr_pairs": lr_pairs,
            "targets": [targets],
        },
        "score": {
            "run_sim_score_by_set": {"truth": truth, "labels": ["<the run_fit out prefix>"]},
            "run_sim_score_by_pair": {"truth": truth, "vectors": ["<the run_fit out prefix>"]},
        },
        "shares": shares_json(&p),
        "shares_note": p.shares_summary(),
    });
    let open = format!(
        "view_open {truth} to see the planted domains; run_fit with fit.data (the \
         .zarr.zip sections), fit.coords and fit.flags as its data, coords and flags; then \
         score each fit: run_sim_score_by_set with truth {truth} and the fit's out prefix in labels, \
         run_sim_score_by_pair with truth {truth} and the prefix in vectors; run_lra and run_lra_simulate on a fit take lra.lr_pairs \
         and lra.targets as lr_pairs and targets (the simulation's genes are no reference's)"
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

// * run_sim_score_by_set

/// `run_sim_score_by_set`: each labelling scored against the truth, as a table
/// (one row per labelling and donor-specific domain) and as JSON. One that
/// cannot be scored is listed with its error and the rest are scored, as
/// `pinto sim score-by-set` does; the call fails only when none can be.
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
    let mut reply = scores_document(&shown_path(Path::new(&a.truth)), scores, failed);
    if !warnings.is_empty() {
        reply["warnings"] = warnings.into();
    }
    Ok(vec![Content::Text(lines.join("\n")), Content::Json(reply)])
}

// * run_sim_score_by_pair

/// `run_sim_score_by_pair`: each input's vectors scored on the pair sets, as a
/// table (one row per input and donor-specific domain) and as JSON, with
/// `pinto sim score-by-pair`'s defaults for what is not given. One that cannot
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
    let reply = scores_document(&shown_path(Path::new(&a.truth)), scores, failed);
    Ok(vec![Content::Text(lines.join("\n")), Content::Json(reply)])
}

// * run_sim_baseline

/// `pinto sim baseline` for `a` on the simulation `truth`, writing the
/// table `table`: only what was given.
pub(super) fn baseline_argv(a: &SimBaselineToolArgs, truth: &str, table: &str) -> Vec<String> {
    let mut argv: Vec<String> = vec![
        "sim".into(),
        "baseline".into(),
        flag_value("--truth", truth),
    ];
    if let Some(m) = &a.method {
        argv.push(flag_value("--method", m));
    }
    if let Some(k) = a.n_clusters {
        argv.push(flag_value("--n-clusters", k));
    }
    if let Some(s) = a.seed {
        argv.push(flag_value("--seed", s));
    }
    argv.extend(shared_flags(a.max_threads, a.atomic_outputs));
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
    let summary = format!("sim baseline into {labels}");
    let reply = json!({
        "command": script::pinto_line(&argv),
        "labels": labels,
    });
    let then = format!("run_sim_score_by_set with truth {truth} and labels [{labels}]");
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

pub(super) fn specs(cli: &clap::Command) -> Vec<ToolSpec> {
    let scenarios: Vec<String> = SCENARIOS
        .iter()
        .map(|s| format!("{}: {}", s.name, s.about))
        .collect();
    vec![
        ToolSpec::new(
            "run_simulate",
            "Simulate several donors' spatial sections with a planted truth (pinto \
             sim new): location sets a spot's domain, the domain its likely cell states, \
             the states its expression; donor and section nuisance (to be removed) and a domain \
             one donor alone has (to be kept) are set as shares of the log rate's variance, and \
             spot noise is what the shares leave: raising one share with the others fixed \
             shrinks the noise. With reference_data (a real data set, .zarr.zip, .zarr or .h5) \
             the expression is its: each state takes one of its cell groups and draws its counts, \
             and the nuisance shares are of its variance between states. \
             The reply's shares and shares_note spell out what a setting \
             implies; relay shares_note to the user whenever a share is set. \
             Writes each section as {out}/sections/S<n>.zarr.zip (counts, genes by spots) \
             with {out}/sections/S<n>.coords.parquet (row_um, col_um), the truth tables and \
             run in {out}/truth (view_open {out}/truth/truth.pinto.json), and \
             {out}/manifest.json. To score: (1) run_fit with the reply's fit.data (the \
             .zarr.zip sections, in order), fit.coords and fit.flags as its data, coords and \
             flags; (2) run_sim_score_by_set with the reply's truth and the fit's out prefix in \
             labels, and run_sim_score_by_pair with the same truth and prefix in vectors. The reply's \
             score spells out both calls. The settings and coordinate files are checked before it starts; it runs as a background job unless asked, and \
             never over an existing out.",
            json!({
                "type": "object",
                "properties": {
                    "scenario": {"type": "string", "enum": SCENARIOS.iter().map(|s| s.name).collect::<Vec<_>>(), "description": format!("Named scenario to start from [default: {}]; a cell type is a cell state of the model. {}", SCENARIOS[0].name, scenarios.join(". "))},
                    "set": {"type": "array", "items": {"type": "string"}, "description": "Change any simulation parameter, key=value (pinto sim new --set; the manifest records them all). The main dials: grid_side and grid_spacing_um (the default square grid, 40 x 40 spots 100 um apart), n_donors, sections_per_donor, n_domains, patch_kind (none, new-state, new-mix), patch_share, patch_length_um, domain_length_um, pve_type, pve_donor, pve_inter, pve_section, unconfounded_sections (a list, [0]), n_unconfounded_donors (that many donors drawn from the seed, never the donor-only domain's owner, with no donor-wide, donor-by-cell-type or section shift; 1 gives one shift-free donor) or unconfounded_donors (a list naming them instead, not with n_unconfounded_donors), copula_strength, nuisance_off (true with the same seed draws the exact counterfactual), seed"},
                    "coords": {"type": "array", "items": {"type": "string"}, "description": "Spot coordinate files, used by the sections in turn, positions only (--coord); the square grid when none"},
                    "coord_column_names": {"type": "array", "items": {"type": "string"}, "description": "Coordinate columns in those files [default: row_um, col_um]"},
                    "reference_data": super::mcp::long_flag_arg(super::mcp::subcommand(cli, "sim new"), "reference_data", json!({"type": "string"})),
                    "reference_data_labels": super::mcp::flag_arg(super::mcp::subcommand(cli, "sim new"), "reference_data_labels", json!({"type": "string"})),
                    "out": out_arg("folder", "sim"),
                    "max_threads": max_threads_arg(cli),
                    "atomic_outputs": atomic_outputs_arg(cli),
                    "background": background(),
                },
                "additionalProperties": false,
            }),
        ),
        ToolSpec::new(
            "run_sim_score_by_set",
            "Score labellings against a run_simulate truth (pinto sim score-by-set), as two tests. \
             1, donor-invariant domains, on spots of shared domains, labels against domains: \
             shared_ari (adjusted Rand index), shared_ami (adjusted mutual information, exact \
             expected MI, arithmetic-mean normalization; 0 chance, 1 the same partition), \
             shared_nmi (normalized mutual information, arithmetic mean; not corrected for \
             chance), donor_conditional_entropy (H(donor | label, domain) / H(donor | domain), \
             0 to 1: 1 labels mix donors fully, 0 the label sets the donor; pooled by spots, \
             null when no shared domain spans two donors, and for each shared domain as \
             donor_conditional_entropy_by_domain, a domain in one donor skipped with a note). 2, each donor-specific domain within its \
             owning donor's spots only (within: owner_donor): its best-matching label there, \
             and that label's f1, precision (low: the owner's shared spots absorbed) and recall \
             over the owner's spots; owner_share over every spot, the share of the label's \
             spots in the owner (1: the label is confined to the owning donor). Returns a table, one \
             row per labelling and donor-specific domain, and the scores as one JSON object, \
             {truth, scores: [{labels, level, score}], failed: [{labels, error}]} (failed only \
             when a labelling could not be scored); pinto sim score-by-set -o writes that same object. \
             Fast: runs in the call.",
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
            "run_sim_score_by_pair",
            "Score per-cell vectors (a latent mapping) against a run_simulate truth on pairs of \
             spots, with no clustering (pinto sim score-by-pair): the cosine of two spots' vectors, \
             by AUROC and AUPRC, each set with its prevalence (the AUPRC of chance). \
             shared_domain_across_donors: shared-domain spots of two donors, positive when \
             they share a domain; shared_domain_within_donor: the same within one donor, the \
             reference; donor_gap: within less across, what the donor costs the mapping. \
             patches, one per donor-specific domain: donor_domain_vs_own_donor, pairs of its \
             spots (positive) against one of its spots with a spot of the same donor outside it \
             (negative; negatives: same_donor_outside_domain), so a mapping that only tells \
             donors apart scores 0.5. neighbour_same_domain: each spot's knn nearest \
             neighbours in its \
             section, positive when they share a domain. Which vectors to give: an lc or dsvd \
             run, its out prefix (scored on its propensity); a cage or spatial baseline run, \
             its {prefix}.cell_embedding.parquet (or its prefix, for its propensity); not \
             dsvd's {prefix}.latent.parquet, which is keyed by cell pair, not cell. Returns a \
             table of the AUROCs, one row per input and donor-specific domain, and the scores \
             as one JSON object, {truth, scores: [{vectors, level, pairs, seed, knn, score}], \
             failed: [{vectors, error}]} (failed only when an input could not be scored); \
             pinto sim score-by-pair -o writes that same object. Runs in the call.",
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
             sim baseline), for run_sim_score_by_set to score next to the fits: pooled (the scanpy \
             recipe on all sections, no batch handling), harmony (the same with Harmony on the \
             PCs, batch = donor) or per-donor (each donor alone, its clusters matched to the \
             largest donor's by marker correlation). Leiden is tuned toward the true number of \
             domains unless n_clusters is given. Writes a cell/label table at {out}.tsv and names it as \
             labels; runs as a background job unless asked, never over an existing table.",
            json!({
                "type": "object",
                "properties": {
                    "truth": {"type": "string", "description": "The simulation: run_simulate's truth (its truth run) or out folder"},
                    "method": {"type": "string", "enum": ["pooled", "harmony", "per-donor"], "description": "Pipeline [default: pooled]"},
                    "n_clusters": {"type": "integer", "minimum": 1, "description": "Clusters to tune Leiden toward [default: the domains among the spots clustered]"},
                    "seed": {"type": "integer", "minimum": 0, "description": "Leiden and Harmony seed [default: 1]"},
                    "out": {"type": "string", "description": "Output prefix, the table at {out}.tsv; left out, a free baseline-<method> name in today's results folder"},
                    "max_threads": max_threads_arg(cli),
                    "atomic_outputs": atomic_outputs_arg(cli),
                    "background": background(),
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
