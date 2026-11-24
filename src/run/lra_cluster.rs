//! `run_lra` with `cluster_shards: N`: the call as `N` shards on a cluster,
//! written as a legume benchmark spec and run by legume-mcp's benchmark
//! tools, not here. One sharded stage runs `pinto lra --shard {shard}/N`
//! with the call's own arguments (its `--null-permutations` draws too, each
//! shard drawing its own pairs'), then one combine stage runs `pinto lra
//! merge` on every shard file and writes the tables of the whole run.

use super::lra::{checked_lra, lra_argv, LraToolArgs};
use crate::lr_activity::args::LraMethod;
use crate::lr_activity::ligand_effect::link_community_scope::shard_path_named;
use crate::mcp::shown_path;
use crate::tui::flag_value;
use legume_mcp::Content;
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::path::Path;

/// The prefix each task writes under, inside its own folder.
const TASK_OUT: &str = "lra";
/// What a task books. pinto keeps to it with no flag of its own: legume sets
/// `RAYON_NUM_THREADS` to the threads booked, which caps pinto's
/// `--max-threads`, and pinto sizes the pairs at once by its cgroup's memory
/// ([`crate::util::memory`]), which Slurm sets to the memory booked.
const SHARD_THREADS: u32 = 16;
const SHARD_MEM_GB: u32 = 64;
const SHARD_TIMEOUT: &str = "4h";
const MERGE_THREADS: u32 = 4;
const MERGE_MEM_GB: u32 = 16;
const MERGE_TIMEOUT: &str = "1h";

/// `{out}.lra.cluster.json`: where the spec for `out` goes.
pub(super) fn cluster_spec_path(out: &str) -> String {
    format!("{out}.lra.cluster.json")
}

/// The files and folders the call reads, as its command line names them:
/// its run (or data files), coordinates, batches, labels, region, pairs and
/// programs.
fn read_paths(a: &LraToolArgs, run: Option<&str>) -> HashSet<String> {
    let single = [&a.lr_pairs, &a.cell_types, &a.region];
    let lists = [&a.data_files, &a.coord_files, &a.batch_files, &a.targets];
    let shown = |p: &String| shown_path(Path::new(p));
    run.map(str::to_string)
        .into_iter()
        .chain(single.into_iter().flatten().map(shown))
        .chain(lists.into_iter().flatten().map(shown))
        .collect()
}

/// `argv` with every word naming one of `paths` (alone, as a flag's
/// `--flag=value`, or within a comma list) replaced by an `{inputs.NAME}`
/// placeholder, and those inputs, so legume keys the tasks by the inputs'
/// bytes and a spec can be pointed at the cluster's copies.
fn declared_inputs(argv: &[String], paths: &HashSet<String>) -> (Vec<String>, Map<String, Value>) {
    let mut inputs = Map::new();
    let words = argv
        .iter()
        .map(|word| {
            let (flag, value) = match word.split_once('=') {
                Some((f, v)) if f.starts_with("--") => (Some(f), v),
                _ => (None, word.as_str()),
            };
            let parts: Vec<&str> = value.split(',').collect();
            if !parts.iter().all(|p| paths.contains(*p)) {
                return word.clone();
            }
            let base = flag.map_or("data".to_string(), |f| {
                f.trim_start_matches('-').replace('-', "_")
            });
            let named: Vec<String> = parts
                .iter()
                .enumerate()
                .map(|(j, p)| {
                    let name = if parts.len() == 1 {
                        base.clone()
                    } else {
                        format!("{base}_{j}")
                    };
                    let dir = Path::new(p).is_dir();
                    inputs.insert(name.clone(), json!({"path": p, "dir": dir}));
                    format!("{{inputs.{name}}}")
                })
                .collect();
            let value = named.join(",");
            match flag {
                Some(f) => format!("{f}={value}"),
                None => value,
            }
        })
        .collect();
    (words, inputs)
}

/// A benchmark name legume takes (a letter, then letters, digits and `_`,
/// at most 40) for the run under `out`: its stem and a hash of the whole
/// path, so runs of one stem in two folders never share a benchmark.
fn benchmark_name(out: &str) -> String {
    use sha2::Digest;
    let stem: String = Path::new(out)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .take(27)
        .collect();
    let hash = sha2::Sha256::digest(out.as_bytes());
    let hash: String = hash[..4].iter().map(|b| format!("{b:02x}")).collect();
    format!("lra_{stem}_{hash}")
}

/// A cluster spec, with its benchmark's name and the effect table its merge
/// writes.
struct ClusterSpec {
    spec: Value,
    name: String,
    table: String,
}

/// The legume benchmark spec running `a` (with `method`, reading `run`) as
/// `shards` shards and merging them.
fn cluster_spec(
    a: &LraToolArgs,
    method: LraMethod,
    run: Option<&str>,
    out: &str,
    shards: usize,
) -> ClusterSpec {
    let mut argv = lra_argv(a, method, run, &format!("{{out}}/{TASK_OUT}"));
    argv.push(flag_value("--shard", "{shard}/{shards}"));
    let (argv, inputs) = declared_inputs(&argv, &read_paths(a, run));
    let shard_file = shard_path_named(TASK_OUT, "{shard}", "{shards}", a.link_community);
    let every_shard = shard_path_named(TASK_OUT, "*", shards, a.link_community);
    let merged: Vec<String> = a
        .outputs(method, TASK_OUT)
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let mut shard_run = vec!["pinto".to_string()];
    shard_run.extend(argv);
    let (name, table) = (benchmark_name(out), merged[0].clone());
    let spec = json!({
        "name": name,
        "seeds": {"n": 1},
        "inputs": inputs,
        "stages": [
            {"name": "pairs", "shards": shards, "threads": SHARD_THREADS,
             "mem_gb": SHARD_MEM_GB, "timeout": SHARD_TIMEOUT,
             "run": shard_run, "makes": [shard_file]},
            {"name": "merge", "combine": "shards", "threads": MERGE_THREADS,
             "mem_gb": MERGE_MEM_GB, "timeout": MERGE_TIMEOUT,
             "run": ["pinto", "lra", "merge", format!("{{glob:pairs/{every_shard}}}"),
                     "-o", format!("{{out}}/{TASK_OUT}")],
             "makes": merged},
        ],
    });
    ClusterSpec { spec, name, table }
}

/// `run_lra` with `cluster_shards`: the spec written beside `out`, and the
/// legume-mcp calls that run it. Nothing runs here.
pub(super) fn cluster_reply(
    cli: &clap::Command,
    a: LraToolArgs,
    method: LraMethod,
    run: Option<&str>,
    out: &str,
    shards: usize,
) -> anyhow::Result<Vec<Content>> {
    anyhow::ensure!(shards >= 1, "cluster_shards is at least 1");
    anyhow::ensure!(
        a.shard.is_none(),
        "cluster_shards writes every shard's task itself: give no shard"
    );
    let out = shown_path(Path::new(out));
    // the call as one shard of them, checked as pinto lra reads it
    let mut argv = lra_argv(&a, method, run, &out);
    argv.push(flag_value("--shard", format!("0/{shards}")));
    checked_lra(cli, &argv)?;
    let ClusterSpec { spec, name, table } = cluster_spec(&a, method, run, &out, shards);
    let path = cluster_spec_path(&out);
    legume_numeric::matrix::common_io::mkdir_parent(&path)?;
    std::fs::write(&path, serde_json::to_string_pretty(&spec)?)?;
    let merged = format!("results/benchmarks/{name}/tasks/<id>.merge/{table}");
    let reply = json!({
        "spec": path,
        "shards": shards,
        "steps": [
            format!("legume-mcp benchmark_plan with spec {path:?} and on REMOTE (a remote of \
                     legume's remotes.toml): checks it against that cluster"),
            format!("legume-mcp benchmark_start with the same spec and on REMOTE: the {shards} \
                     shards as one Slurm array, then the merge behind them"),
            format!("legume-mcp benchmark_status with name {name:?} until it has ended; a \
                     failed task is named there, with its log"),
            format!("once done: the tables of the whole run are {merged} on the cluster \
                     (and the rest beside it), recorded in the manifest the tasks read: the \
                     cluster's, not this machine's"),
        ],
        "note": "Nothing was run here. The spec's inputs are this machine's paths, and a \
                 run's manifest names its data files by this machine's paths too: the \
                 cluster must see them at the same paths (a shared file system), or each \
                 inputs path and the manifest's own must be changed to the cluster's before \
                 benchmark_plan. threads, mem_gb and timeout are a start; raise mem_gb and \
                 timeout for a large tissue.",
    });
    Ok(vec![Content::Json(reply)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::built_cli as cli;

    /// The spec runs the call's own flags as `N` shards, declares the run it
    /// reads as an input, and merges every shard's file into the run's
    /// tables, null table included; each shard's command is one pinto lra
    /// takes. Under a link community, the reply names that community's table.
    #[test]
    fn cluster_shards_writes_a_spec_of_the_shards_and_their_merge() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("r.pinto.json");
        std::fs::write(&run, "{}").unwrap();
        let run = run.to_string_lossy().into_owned();
        let out = dir.path().join("o").to_string_lossy().into_owned();
        let a: LraToolArgs = crate::mcp::parse(json!({
            "run": run, "null_permutations": 2, "cluster_shards": 8
        }))
        .unwrap();
        let reply =
            cluster_reply(&cli(), a, LraMethod::ContactEffect, Some(&run), &out, 8).unwrap();
        let Content::Json(reply) = &reply[0] else {
            panic!("a JSON reply")
        };
        assert_eq!(reply["steps"].as_array().map(Vec::len), Some(4));
        let text = std::fs::read_to_string(cluster_spec_path(&out)).unwrap();
        let spec: Value = serde_json::from_str(&text).unwrap();
        assert!(spec["name"].as_str().unwrap().starts_with("lra_o_"));
        assert_eq!(spec["inputs"]["from"]["path"], run.as_str());
        let (pairs, merge) = (&spec["stages"][0], &spec["stages"][1]);
        assert_eq!(pairs["shards"], 8);
        let words: Vec<&str> = pairs["run"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap())
            .collect();
        for w in [
            "--from={inputs.from}",
            "--shard={shard}/{shards}",
            "--null-permutations=2",
        ] {
            assert!(words.contains(&w), "{w} in {words:?}");
        }
        assert!(!text.contains(&out), "the call's out is not the tasks'");
        assert_eq!(
            pairs["makes"],
            json!(["lra.lra.shard-{shard}-of-{shards}.cbor"])
        );
        assert_eq!(merge["combine"], "shards");
        assert_eq!(merge["run"][3], "{glob:pairs/lra.lra.shard-*-of-8.cbor}");
        assert!(merge["makes"]
            .as_array()
            .unwrap()
            .contains(&json!("lra.lra.null.parquet")));

        let a: LraToolArgs = crate::mcp::parse(json!({"run": run, "link_community": 3})).unwrap();
        let ClusterSpec { table, .. } =
            cluster_spec(&a, LraMethod::ContactEffect, Some(&run), &out, 2);
        assert_eq!(table, "lra.lra.link-community-3.parquet");
    }
}
