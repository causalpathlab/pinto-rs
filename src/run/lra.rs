//! `run_lra`, `run_lra_simulate` and `run_lra_merge` for `pinto mcp`:
//! `pinto lra` and `pinto lra simulate` on a run or on data files, and
//! `pinto lra merge` on a sharded run's shard files, set up flag for flag
//! as the command line takes them, checked as pinto parses them, and run as
//! a child pinto in the project's records (a background job unless asked
//! otherwise). The reply is read from the tables the child wrote.

use super::mcp::{
    as_run, checked, in_background, out_or_free, out_or_free_by, rooted, shared_flags, spawn,
    NOT_WRITTEN_OVER,
};
use super::script;
use crate::lr_activity::args::LraMethod;
use crate::lr_activity::ligand_effect::output::{
    self, compare_runs, lra_simulate_path, percent_text, rank_pooled, read_planted_region,
    read_pooled_effects, read_sim_runs,
};
use crate::lr_activity::ligand_effect::shard::{self, Shard};
use crate::lr_activity::ligand_effect::simulate::COMPARED;
use crate::lr_activity::ligand_effect::verdict::Verdict;
use crate::mcp::shown_path;
use crate::tui::flag_value;
use crate::util::cli::{choices, value_name};
use crate::util::metadata::{manifest_path, run_stem, LraOptions, PintoMetadata};
use legume_mcp::{Call, Content, Next};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// Pairs the reply lists, of those that survived the controls, strongest first
/// (as the run summary).
const TOP_PAIRS: usize = 5;

// * Arguments

/// A tool's arguments: what it reads ([`Input`]) and, in the order given,
/// its own fields after `seed` and after `region_jackknife`. Both lra tools
/// declare the input fields in one place, in one order, and map them to
/// [`Input`] alike.
macro_rules! lra_tool_args {
    ($name:ident { $($mid:tt)* } { $($tail:tt)* }) => {
        #[derive(Deserialize, Serialize, Debug)]
        #[serde(deny_unknown_fields)]
        pub(super) struct $name {
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
            $($mid)*
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub cell_types: Option<String>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub region: Option<String>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub region_jackknife: Option<String>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub abf_prior_scale: Option<f64>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub max_threads: Option<usize>,
            #[serde(default, skip_serializing)]
            pub atomic_outputs: Option<bool>,
            $($tail)*
            #[serde(default = "in_background", skip_serializing)]
            pub background: bool,
        }

        impl $name {
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
                    abf_prior_scale: self.abf_prior_scale,
                    max_threads: self.max_threads,
                    atomic_outputs: self.atomic_outputs,
                }
            }
        }
    };
}

lra_tool_args!(LraToolArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_permutations: Option<usize>,
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
    /// `--shard`: shard `I` (0-based) of `N` of the pairs, `I/N`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
    /// `--link-community`: estimate through the contacts of this link
    /// community of the run (`pinto lra link-communities` lists them).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link_community: Option<i64>,
} {
    /// `--marginal-baselines`, and `--marginal-permutations`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub marginal_baselines: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marginal_permutations: Option<usize>,
});

impl LraToolArgs {
    /// `only` as `pinto lra` takes it.
    fn only(&self) -> Vec<Box<str>> {
        self.only.iter().map(|s| s.as_str().into()).collect()
    }

    /// `shard` as `pinto lra` reads it.
    fn shard(&self) -> anyhow::Result<Option<Shard>> {
        self.shard
            .as_deref()
            .map(|s| s.parse().map_err(|e| anyhow::anyhow!("shard {e}")))
            .transpose()
    }

    /// What `pinto lra` writes for these arguments: with `shard`, its shard
    /// file alone; else [`lra_outputs`], and with `marginal_baselines` the
    /// per-type-pair table beside the contact-effect table.
    fn outputs(&self, method: LraMethod, out: &str) -> Vec<PathBuf> {
        use crate::lr_activity::ligand_effect::link_community_scope;
        if let Ok(Some(s)) = self.shard() {
            return vec![link_community_scope::shard_path(out, s, self.link_community).into()];
        }
        let mut made = crate::lr_activity::lra_outputs_of_link_community(
            method,
            out,
            &self.only(),
            self.link_community,
        );
        if self.marginal_baselines && method == LraMethod::ContactEffect {
            let table = made[0].to_string_lossy().into_owned();
            made.push(output::celltype_beside(&table).into());
        }
        made
    }

    /// The `run_lra_simulate` call checking this analysis's power: the same
    /// input, pairs, programs and species.
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
            abf_prior_scale: self.abf_prior_scale.unwrap_or(d.abf_prior_scale),
            link_community: self.link_community,
            marginal_permutations: self.marginal_baselines.then(|| {
                self.marginal_permutations
                    .unwrap_or(crate::lr_activity::ligand_effect::PERMUTATIONS)
            }),
            ..d
        }
    }
}

lra_tool_args!(LraSimulateToolArgs {
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
    /// `--reference-data` and `--reference-data-labels`: the data set the
    /// program genes are drawn with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_data_labels: Option<String>,
} {
    /// `--one`: the single scenario at its defaults.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub one: bool,
    /// `--set`: the single scenario's keys, `KEY=VALUE`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub set: Vec<String>,
    /// `--pairs` and `--null-pairs`: the single scenario's planted and null
    /// pairs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairs: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub null_pairs: Option<usize>,
    /// `--pair-assignment`: how the single scenario's pairs are chosen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pair_assignment: Option<String>,
    /// `--shard`: shard `I` (0-based) of `N` of the runs, `I/N`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
});

impl LraSimulateToolArgs {
    /// Whether the call asks for one scenario rather than the built-in grid.
    fn single(&self) -> bool {
        crate::lr_activity::args::selects_single(self.one, &self.set, self.pairs, self.null_pairs)
    }

    /// `pair_assignment` as `pinto lra simulate` reads it, its default when
    /// not given.
    fn pair_assignment(
        &self,
    ) -> anyhow::Result<crate::lr_activity::ligand_effect::simulate::PairAssignment> {
        self.pair_assignment
            .as_deref()
            .map(|r| {
                clap::ValueEnum::from_str(r, false)
                    .map_err(|e| anyhow::anyhow!("pair_assignment {r:?}: {e}"))
            })
            .transpose()
            .map(Option::unwrap_or_default)
    }

    /// `shard` as `pinto lra simulate` reads it.
    fn shard(&self) -> anyhow::Result<Option<Shard>> {
        self.shard
            .as_deref()
            .map(|s| s.parse().map_err(|e| anyhow::anyhow!("shard {e}")))
            .transpose()
    }

    /// What the run writes under `out`: with `shard`, its shard file; else
    /// the table, and for a single scenario the sidecar that marks the call
    /// complete.
    pub(super) fn outputs(&self, out: &str) -> Vec<PathBuf> {
        if let Ok(Some(s)) = self.shard() {
            return vec![shard::sim_path(out, s).into()];
        }
        let mut made = vec![PathBuf::from(lra_simulate_path(out))];
        if self.single() {
            made.push(output::lra_simulate_sidecar_path(out).into());
        }
        made
    }

    /// Whether the finished outputs under `out` can be this call's: a single
    /// scenario's sidecar records this call's point, pair counts, seed and
    /// planted region, and a grid's table is no single scenario's. An error
    /// says what differs.
    pub(super) fn check_made(&self, out: &str) -> anyhow::Result<()> {
        use crate::lr_activity::ligand_effect::simulate::{ScenarioKeys, SingleScenario};
        let sidecar = PathBuf::from(output::lra_simulate_sidecar_path(out));
        if !self.single() {
            anyhow::ensure!(
                !sidecar.is_file(),
                "{} is a single scenario's table (beside {}), not the built-in grid's; \
                 {NOT_WRITTEN_OVER}",
                lra_simulate_path(out),
                shown_path(&sidecar)
            );
            return Ok(());
        }
        let text = std::fs::read_to_string(&sidecar)?;
        let made: Value = serde_json::from_str(&text)?;
        let want = json!({
            "scenario": ScenarioKeys::default().with_overrides(&self.set)?,
            "asked": {
                "planted": self.pairs.unwrap_or(SingleScenario::PAIRS),
                "null": self.null_pairs.unwrap_or(SingleScenario::PAIRS),
            },
            "seed": self.seed.unwrap_or(crate::lr_activity::args::DEFAULT_SEED),
            "plant_region": self.plant_region,
            "assignment": crate::util::cli::value_name(&self.pair_assignment()?),
        });
        let differs: Vec<&str> = ["scenario", "asked", "seed", "plant_region", "assignment"]
            .into_iter()
            .filter(|k| made[*k] != want[*k])
            .collect();
        anyhow::ensure!(
            differs.is_empty(),
            "{} was made with another {} ({}); {NOT_WRITTEN_OVER}",
            shown_path(&sidecar),
            differs.join(", "),
            differs
                .iter()
                .map(|k| format!("{k} {}", made[*k]))
                .collect::<Vec<_>>()
                .join("; ")
        );
        Ok(())
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
    abf_prior_scale: Option<f64>,
    max_threads: Option<usize>,
    atomic_outputs: Option<bool>,
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
            argv.push(flag_value("--coord", shown(self.coord_files)));
        }
        if !self.batch_files.is_empty() {
            argv.push(flag_value("--batch-files", shown(self.batch_files)));
        }
        if let Some(k) = self.knn_spatial {
            argv.push(flag_value("--knn-spatial", k));
        }
        if let Some(c) = self.cell_types {
            argv.push(flag_value("--cell-types", shown_path(Path::new(c))));
        }
        if let Some(r) = self.region {
            argv.push(flag_value("--region", shown_path(Path::new(r))));
        }
        if let Some(j) = self.region_jackknife {
            argv.push(flag_value("--region-jackknife", j));
        }
        if let Some(s) = self.abf_prior_scale {
            argv.push(flag_value("--abf-prior-scale", s));
        }
        argv.extend(shared_flags(self.max_threads, self.atomic_outputs));
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
        if let Some(s) = self.abf_prior_scale {
            args["abf_prior_scale"] = s.into();
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
            let names = choices::<LraMethod>().join(", ");
            anyhow::anyhow!("method {m:?}: one of {names}")
        }),
    }
}

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
        value_name(&method)
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
        abf_prior_scale: d.abf_prior_scale,
        marginal_permutations: None,
        ..a.asked_options()
    };
    let mut argv = options.argv(run, Some(out), method, &a.only);
    if let Some(s) = &a.shard {
        argv.push(flag_value("--shard", s));
    }
    if let Some(k) = a.link_community {
        argv.push(flag_value("--link-community", k));
    }
    argv.extend(a.input().flags());
    if let Some(n) = a.n_permutations {
        argv.push(flag_value("--n-permutations", n));
    }
    if let Some(x) = a.min_gene_count {
        argv.push(flag_value("--min-gene-count", x));
    }
    if a.marginal_baselines {
        argv.push("--marginal-baselines".into());
        if let Some(n) = a.marginal_permutations {
            argv.push(flag_value("--marginal-permutations", n));
        }
    }
    argv
}

/// `pinto lra simulate` reading `run` (or its data files), writing under
/// `out`.
pub(super) fn lra_simulate_argv(
    a: &LraSimulateToolArgs,
    run: Option<&str>,
    out: &str,
) -> Vec<String> {
    let mut argv: Vec<String> = vec!["lra".into(), "simulate".into()];
    if let Some(run) = run {
        argv.push(flag_value("--from", run));
    }
    argv.push(flag_value("--out", out));
    argv.extend(a.input().flags());
    if let Some(seed) = a.seed {
        argv.push(flag_value("--seed", seed));
    }
    if let Some(s) = &a.species {
        argv.push(flag_value("--species", s));
    }
    if let Some(p) = &a.lr_pairs {
        argv.push(flag_value("--lr-pairs", shown_path(Path::new(p))));
    }
    if !a.targets.is_empty() {
        let t: Vec<String> = a.targets.iter().map(|t| shown_path(Path::new(t))).collect();
        argv.push(flag_value("--targets", t.join(",")));
    }
    if let Some(r) = &a.plant_region {
        argv.push(flag_value("--plant-region", r));
    }
    // the labels go in whether or not the data does, so the command's own
    // check refuses labels with no data
    if let Some(r) = &a.reference_data {
        argv.push(flag_value("--reference-data", shown_path(Path::new(r))));
    }
    if let Some(l) = &a.reference_data_labels {
        argv.push(flag_value(
            "--reference-data-labels",
            shown_path(Path::new(l)),
        ));
    }
    if a.one {
        argv.push("--one".into());
    }
    // One token each, so a value cannot be read as a flag.
    argv.extend(a.set.iter().map(|s| format!("--set={s}")));
    argv.extend(a.pairs.map(|n| format!("--pairs={n}")));
    argv.extend(a.null_pairs.map(|n| format!("--null-pairs={n}")));
    if let Some(r) = &a.pair_assignment {
        argv.push(flag_value("--pair-assignment", r));
    }
    if let Some(s) = &a.shard {
        argv.push(flag_value("--shard", s));
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
        .map_err(|e| anyhow::anyhow!("pinto {}: {e}", super::mcp::named(argv)))
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
        "command": script::pinto_line(argv),
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
    let shard = a.shard()?;
    let out = named_out(a.out.as_deref());
    anyhow::ensure!(
        shard.is_none() || out.as_deref().is_some_and(|o| !o.is_empty()),
        "shard needs out: every shard of a run writes under the one prefix the merge reads"
    );
    // A finished `out` is read, not run again, once the call is checked as
    // it would be run: its tables, or a shard's file and the merge to follow.
    if let Some(o) = out.as_deref().filter(|o| !o.is_empty()) {
        let o = shown_path(Path::new(o));
        let made = a.outputs(method, &o);
        let argv = lra_argv(&a, method, run, &o);
        checked_lra(cli, &argv)?;
        if let Some(s) = shard {
            if finished(cx, &made) {
                let mut reply = shard_reply(s, &argv, &o, &made, a.link_community);
                push_note(
                    &mut reply,
                    &format!(
                        "read back: the shard file is already there, not checked against \
                         these arguments; nothing was run. {}",
                        shard_note(s, &o, a.link_community)
                    ),
                );
                return Ok(reply);
            }
        } else {
            let manifest = PathBuf::from(run.map_or_else(|| manifest_path(&o), str::to_string));
            if let Some(reply) = read_finished(
                cx,
                &made,
                || made_as_asked(&a, method, &manifest, &made),
                |checked| lra_reply(&a, method, checked.then_some(&argv[..]), run, &made),
            ) {
                return reply;
            }
        }
    }
    let (out, made) = out_or_free(cx, out.as_deref(), &input.stem(run), |o| {
        Ok(a.outputs(method, o))
    })?;
    let argv = lra_argv(&a, method, run, &out);
    checked_lra(cli, &argv)?;
    let summary = format!("lra {} of {}", value_name(&method), input.source(run));
    // The same call, by its out: what reads the tables once the job is done.
    let mut again = serde_json::to_value(&a)?;
    if let Some(run) = run {
        again["run"] = run.into();
    }
    again["out"] = out.clone().into();
    if let Some(mut job) = start(cx, &argv, &made, &summary, a.background, "run_lra", &again)? {
        if let Some(s) = shard {
            job["note"] = format!(
                "job_wait (id {}) or job_status until it ends. {}",
                job["job"].as_str().unwrap_or_default(),
                shard_note(s, &out, a.link_community)
            )
            .into();
        }
        return Ok(vec![Content::Json(job)]);
    }
    if let Some(s) = shard {
        return Ok(shard_reply(s, &argv, &out, &made, a.link_community));
    }
    lra_reply(&a, method, Some(&argv), run, &made)
}

/// The `run_lra_merge` call putting together the `N` shards of `shard`'s
/// run, each written under `out`.
fn merge_args(shard: Shard, out: &str, link_community: Option<i64>) -> Value {
    use crate::lr_activity::ligand_effect::link_community_scope::shard_path;
    let shards: Vec<String> = (0..shard.count)
        .map(|index| shard_path(out, Shard { index, ..shard }, link_community))
        .collect();
    json!({ "shards": shards, "out": out })
}

/// What follows a shard: the other shards, each its own job, then
/// `run_lra_merge`.
fn shard_note(shard: Shard, out: &str, link_community: Option<i64>) -> String {
    format!(
        "A shard writes no table and records nothing in the manifest. Run every shard \
         I/{n} (I from 0 to {last}) with the same arguments, each as its own job (a Slurm \
         array task, I = $SLURM_ARRAY_TASK_ID); once all {n} are done, run_lra_merge {} \
         writes the tables of the whole run, its local fdrs over every pair.",
        merge_args(shard, out, link_community),
        n = shard.count,
        last = shard.count - 1,
    )
}

/// The reply to a finished `pinto lra --shard`: the shard file it wrote and
/// the `run_lra_merge` call that puts the shards together.
fn shard_reply(
    shard: Shard,
    argv: &[String],
    out: &str,
    made: &[PathBuf],
    link_community: Option<i64>,
) -> Vec<Content> {
    let written: Vec<String> = made.iter().map(|p| shown_path(p)).collect();
    let reply = json!({
        "shard": shard.to_string(),
        "command": script::pinto_line(argv),
        "written": written,
        "note": shard_note(shard, out, link_community),
    });
    vec![
        Content::Json(reply),
        Content::Next(vec![Next::new(
            "run_lra_merge",
            merge_args(shard, out, link_community),
            "once every shard is done: the tables of the whole run",
        )]),
    ]
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
        let mut r = table_summary(&made[0])?;
        if !a.only.is_empty() {
            r["q_scope"] = "judged within this --only subset, not over every pair; \
                            recorded under lra.subsets, the full table unchanged"
                .into();
        }
        r
    } else {
        json!({})
    };
    if let Some(argv) = command {
        reply["command"] = script::pinto_line(argv).into();
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

/// A contact-effect table's pooled pairs in a few lines
/// ([`effects_summary`]), and how to read it.
fn table_summary(table: &Path) -> anyhow::Result<Value> {
    let rows = read_pooled_effects(&table.to_string_lossy())
        .map_err(|e| anyhow::anyhow!("{}: {e}", table.display()))?;
    let mut r = effects_summary(&rows);
    r["reading"] =
        "one row per pair and section (`all` pooled); verdict_note says why in words".into();
    Ok(r)
}

/// The pooled pairs in a few lines: how many, each verdict's count, and the
/// strongest that survived the controls (percent change with its 95% interval, and
/// local fdr).
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
                "lfdr": sig(r.effect_dr_lfdr),
            })
        })
        .collect();
    json!({ "pairs": rows.len(), "verdicts": counts, "top": top })
}

/// A local fdr to three significant digits; null when there is none.
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

// * run_lra_merge

/// `run_lra_merge`'s arguments: the shard files of one `run_lra` run with
/// `shard`, and where the tables go.
#[derive(Deserialize, Serialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct LraMergeToolArgs {
    pub shards: Vec<String>,
    pub out: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_threads: Option<usize>,
    #[serde(default, skip_serializing)]
    pub atomic_outputs: Option<bool>,
    #[serde(default = "in_background", skip_serializing)]
    pub background: bool,
}

/// `pinto lra merge` on `a`'s shard files, writing under `out`.
pub(super) fn lra_merge_argv(a: &LraMergeToolArgs, out: &str) -> Vec<String> {
    let mut argv: Vec<String> = vec!["lra".into(), "merge".into()];
    argv.extend(rooted(&a.shards));
    argv.push(flag_value("--out", out));
    argv.extend(shared_flags(a.max_threads, a.atomic_outputs));
    argv
}

/// `run_lra_merge`.
pub(super) fn lra_merge(
    cli: &clap::Command,
    a: LraMergeToolArgs,
    cx: &Call,
) -> anyhow::Result<Vec<Content>> {
    anyhow::ensure!(
        !a.shards.is_empty(),
        "give shards: every shard file of one run_lra run with shard"
    );
    if let Some(gone) = a.shards.iter().find(|s| !Path::new(s).is_file()) {
        anyhow::bail!("shard {gone} not found: give the files run_lra with shard wrote");
    }
    let out = named_out(Some(&a.out))
        .filter(|o| !o.is_empty())
        .ok_or_else(|| anyhow::anyhow!("give out, the prefix of the merged tables"))?;
    let out = shown_path(Path::new(&out));
    // effect tables are collected, not merged: quick, so not a job
    if crate::lr_activity::ligand_effect::output::collect::are_tables(&a.shards) {
        let written =
            crate::lr_activity::ligand_effect::output::collect::collect_link_community_tables(
                &a.shards, &out,
            )?;
        return Ok(vec![Content::Json(json!({
            "collected": a.shards,
            "written": [shown_path(Path::new(&written))],
        }))]);
    }
    let (run_key, kind) = shard::peek(&a.shards[0])?;
    let simulate = matches!(kind, shard::Kind::Simulate { .. });
    // what a merge of this kind writes: the per-type-pair table too when the
    // shards carry it, a single scenario's sidecar
    let outputs = |o: &str| match kind {
        shard::Kind::Lra {
            celltype,
            link_community,
        } => {
            let (table, curve) =
                crate::lr_activity::ligand_effect::link_community_scope::table_and_curve_paths(
                    o,
                    &[],
                    link_community,
                );
            let mut made = vec![PathBuf::from(table), PathBuf::from(curve)];
            if celltype {
                let table = made[0].to_string_lossy().into_owned();
                made.push(output::celltype_beside(&table).into());
            }
            made
        }
        shard::Kind::Simulate { one } => {
            let mut made = vec![PathBuf::from(lra_simulate_path(o))];
            if one {
                made.push(output::lra_simulate_sidecar_path(o).into());
            }
            made
        }
    };
    let made = outputs(&out);
    let argv = lra_merge_argv(&a, &out);
    checked(cli, &argv)?;
    // Finished tables are read, not merged again, when these shards made them.
    if finished(cx, &made) {
        let link_community = match kind {
            shard::Kind::Lra { link_community, .. } => link_community,
            shard::Kind::Simulate { .. } => None,
        };
        let record = crate::lr_activity::ligand_effect::link_community_scope::merge_record_path(
            &out,
            link_community,
        );
        let made_by = std::fs::read_to_string(&record)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v["run_key"].as_str().map(str::to_string));
        anyhow::ensure!(
            made_by.as_deref() == Some(run_key.as_str()),
            "{} is there but not a merge of these shards ({} says otherwise, or is \
             missing); {NOT_WRITTEN_OVER}",
            shown_path(&made[0]),
            record
        );
        let reply = if simulate {
            simulate_reply(None, &made)
        } else {
            merge_reply(None, &made)
        };
        let mut reply = reply.map_err(|e| anyhow::anyhow!("{e:#}; {NOT_WRITTEN_OVER}"))?;
        push_note(
            &mut reply,
            "read from the tables these shards' merge left; nothing was run",
        );
        return Ok(reply);
    }
    let (out, made) = out_or_free(cx, Some(&out), "lra", |o| Ok(outputs(o)))?;
    let summary = format!("lra merge of {} shards into {out}", a.shards.len());
    let mut again = serde_json::to_value(&a)?;
    again["out"] = out.clone().into();
    if let Some(job) = start(
        cx,
        &argv,
        &made,
        &summary,
        a.background,
        "run_lra_merge",
        &again,
    )? {
        return Ok(vec![Content::Json(job)]);
    }
    if simulate {
        return simulate_reply(Some(&argv), &made);
    }
    merge_reply(Some(&argv), &made)
}

/// The reply to a finished `pinto lra merge`: a full run's ([`table_summary`]),
/// with the command when known and the files, the per-type-pair table among
/// them when the shards carried the baselines.
fn merge_reply(command: Option<&[String]>, made: &[PathBuf]) -> anyhow::Result<Vec<Content>> {
    let mut reply = table_summary(&made[0])?;
    if let Some(argv) = command {
        reply["command"] = script::pinto_line(argv).into();
    }
    let written: Vec<String> = made.iter().map(|p| shown_path(p)).collect();
    reply["written"] = written.into();
    Ok(vec![Content::Json(reply)])
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
    let shard = a.shard()?;
    let out = named_out(a.out.as_deref());
    anyhow::ensure!(
        shard.is_none() || out.as_deref().is_some_and(|o| !o.is_empty()),
        "shard needs out: every shard of a call writes under the one prefix the merge reads"
    );
    if let Some(o) = out
        .as_deref()
        .filter(|o| !o.is_empty())
        .filter(|_| shard.is_none())
    {
        let o = shown_path(Path::new(o));
        let made = a.outputs(&o);
        checked(cli, &lra_simulate_argv(&a, run, &o))?;
        // lra-simulate records no options for its table; a single scenario's
        // sidecar records its point, which must be this call's.
        if let Some(reply) = read_finished(
            cx,
            &made,
            || a.check_made(&o).map(|()| false),
            |_| simulate_reply(None, &made),
        ) {
            return reply;
        }
    }
    // A single scenario's free name is no run's prefix (a run in today's
    // folder has one there), whose table is that run's grid's.
    let (out, made) = match a.single() {
        true => out_or_free_by(
            cx,
            out.as_deref(),
            &format!("{}-scenario", input.stem(run)),
            |o| Ok(a.outputs(o)),
            |o| Path::new(&crate::util::metadata::run_manifest(o)).is_file(),
        )?,
        false => out_or_free(cx, out.as_deref(), &input.stem(run), |o| Ok(a.outputs(o)))?,
    };
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
    if let Some(s) = shard {
        return Ok(sim_shard_reply(s, &argv, &out, &made));
    }
    simulate_reply(Some(&argv), &made)
}

/// The reply to a finished `pinto lra simulate --shard`: the shard file and
/// the `run_lra_merge` call that puts the shards together.
fn sim_shard_reply(shard: Shard, argv: &[String], out: &str, made: &[PathBuf]) -> Vec<Content> {
    let shards: Vec<String> = (0..shard.count)
        .map(|index| shard::sim_path(out, Shard { index, ..shard }))
        .collect();
    let merge = json!({ "shards": shards, "out": out });
    let reply = json!({
        "shard": shard.to_string(),
        "command": script::pinto_line(argv),
        "written": made.iter().map(|p| shown_path(p)).collect::<Vec<_>>(),
        "note": format!(
            "A shard writes no table. Run every shard I/{n} (I from 0 to {last}) with the \
             same arguments, each as its own job (a Slurm array task); once all {n} are done, \
             run_lra_merge {merge} writes the call's tables.",
            n = shard.count,
            last = shard.count - 1,
        ),
    });
    vec![
        Content::Json(reply),
        Content::Next(vec![Next::new(
            "run_lra_merge",
            merge,
            "once every shard is done: the call's tables",
        )]),
    ]
}

/// The reply to a finished `pinto lra simulate`: the comparison table, and
/// the command that made it when known.
fn simulate_reply(command: Option<&[String]>, made: &[PathBuf]) -> anyhow::Result<Vec<Content>> {
    let table = made[0].to_string_lossy();
    let runs = read_sim_runs(&table)?;
    let mut reply = comparison_summary(&runs, read_planted_region(&table).as_deref());
    if let Some(argv) = command {
        reply["command"] = script::pinto_line(argv).into();
    }
    let written: Vec<String> = made.iter().map(|p| shown_path(p)).collect();
    reply["written"] = json!(written);
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
        "reading": "reject: share of runs significant at 5% per method. dr is lra's \
                    adjusted effect (doubly robust); marginal the correlation of response and contact ligand; \
                    unadj the difference in mean response, exposed minus unexposed, with no adjustment; ipw the effect by overlap weights alone; outcome by the outcome regression alone (each with its delete-one-tile jackknife SE); \
                    celltype, moran, contact and coexpr the co-expression baselines (CellPhoneDB/CellChat, SpatialDM, \
                    NICHES/COMMOT, lra's log odds). Trust lra here when dr keeps type I \
                    near 0.05 under the confounded nulls and has power at the effect sizes \
                    that matter.",
    })
}

#[cfg(test)]
#[path = "tests/lra.rs"]
mod tests;
