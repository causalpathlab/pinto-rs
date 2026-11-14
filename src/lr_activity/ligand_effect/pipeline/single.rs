//! `lra-simulate`'s single-scenario mode: one scenario, its effect planted on
//! some pairs and withheld on as many others under the same hidden field, in
//! one call that a sweep scores by AUROC and AUPRC (one call per point and
//! seed, many calls at once).

use super::{benchmark_pairs, planted_region, prepare, real_gene_confounder, PairSim, Progress};
use crate::lr_activity::ligand_effect::memory::{pair_memory, pair_workers, run_bounded};
use crate::lr_activity::ligand_effect::options::{self, EffectOpts};
use crate::lr_activity::ligand_effect::simulate::{
    self, ConfounderKind, Scenario, ScenarioKeys, SimRow, SimShared, SingleScenario,
};
use crate::lr_activity::ligand_effect::{output, reference, RunInputs, LOG_TARGET};
use crate::util::common::*;
use anyhow::Context;
use legume_numeric::matrix::rand_util::mix_seed;
use serde_json::json;
use std::path::Path;

/// The rule that labels pairs, as the sidecar names it
/// ([`interleaved_ranks`]).
const ASSIGNMENT: &str = "interleaved-rank";

/// Mixed into `--seed` for the call's hidden field: no abundance rank
/// reaches it, so the field's stream is apart from every pair's.
const FIELD_SALT: u64 = u64::MAX;

/// Planted and null pairs by abundance rank: the top `planted + null` of the
/// `available` ranks, the fewer label (null when the counts are equal)
/// spread evenly among the more, at rank `⌊(k + ½) m / few⌋` for its `k`-th
/// pair of `few` over `m` ranks, as Bresenham's line algorithm spreads its
/// steps (Bresenham 1965, IBM Systems Journal 4(1):25-30). Both labels then
/// have the same mean abundance rank to within one, whatever the counts:
/// ligand abundance sorts them no further. Equal counts alternate, even
/// ranks planted. With fewer ranks than asked, each label keeps its asked
/// share, at least one of each while two ranks are left.
pub fn interleaved_ranks(available: usize, planted: usize, null: usize) -> Vec<(usize, bool)> {
    let asked = planted + null;
    let m = asked.min(available);
    let n1 = match m {
        0 | 1 => m,
        _ if m == asked => planted,
        _ => ((2 * m * planted + asked) / (2 * asked)).clamp(1, m - 1),
    };
    let n0 = m - n1;
    let (few, label) = if n1 < n0 { (n1, true) } else { (n0, false) };
    let mut planted_at = vec![!label; m];
    for k in 0..few {
        planted_at[(2 * k + 1) * m / (2 * few)] = label;
    }
    planted_at.into_iter().enumerate().collect()
}

/// Remove an earlier single-scenario call's table and sidecar under `out`:
/// its sidecar would mark the next call complete before it is, and its
/// table would pass for the next call's should that one fail.
pub fn clear_single_outputs(out: &str) -> anyhow::Result<()> {
    let earlier = [
        output::lra_simulate_sidecar_path(out),
        output::lra_simulate_path(out),
    ];
    for path in &earlier {
        match std::fs::remove_file(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(e).with_context(|| format!("removing the earlier {path}"));
            }
            _ => {}
        }
    }
    Ok(())
}

/// A benchmarked pair of the call.
struct Labelled {
    /// Its abundance rank among the pairs fit to benchmark
    /// ([`benchmark_pairs`]).
    rank: usize,
    /// Its index in the run's pairs.
    pair: usize,
    planted: bool,
    /// `mix_seed(--seed, rank)`: what its draws start from, the scenario's
    /// `rep` mixed in as the built-in grid does ([`simulate::simulate`]).
    seed: u64,
}

/// One scenario ([`SingleScenario`]) on the run: the `planted` pairs carry
/// its effect, the `null` pairs none, every pair the same hidden field, drawn
/// once from `--seed` and `rep`, the labels interleaved by abundance rank
/// ([`interleaved_ranks`]). Pairs run at once as memory allows
/// ([`run_bounded`]), each from its own seed, so a pair's rows, given its
/// rank and label, are the same whatever else the call holds and however
/// many threads run it; the one exception is `realgene`, whose gene is the
/// most abundant one outside the call's pairs and so can change with how
/// many pairs it asks for. Earlier outputs under `out` are removed first
/// ([`clear_single_outputs`]), as the command does before reading data. It
/// writes `{out}.lra-simulate.parquet` ([`output::write_sim_scored`]), then,
/// last and atomically, `{out}.lra-simulate.json`, and records nothing in
/// the run's manifest.
pub fn run_simulate_one(inp: &RunInputs<'_>, one: &SingleScenario) -> anyhow::Result<()> {
    let a = &inp.settings;
    let table = output::lra_simulate_path(a.out);
    let sidecar = output::lra_simulate_sidecar_path(a.out);
    clear_single_outputs(a.out)?;
    let prep = prepare(inp)?;
    let tissue = &prep.tissue;
    // The benchmark plants on one competence threshold.
    let eopts = &EffectOpts {
        competences: &options::BENCHMARK_COMPETENCE,
        ..prep.eopts
    };
    let region = planted_region(tissue, a.plant_region)?;
    let scen = Scenario {
        region,
        ..one.scenario
    };
    let order = benchmark_pairs(&prep, eopts, one.planted + one.null, region)?;
    let labelled: Vec<Labelled> = interleaved_ranks(order.len(), one.planted, one.null)
        .into_iter()
        .map(|(rank, planted)| Labelled {
            rank,
            pair: order[rank],
            planted,
            seed: mix_seed(a.seed, rank as u64),
        })
        .collect();
    let n1 = labelled.iter().filter(|l| l.planted).count();
    let n0 = labelled.len() - n1;
    anyhow::ensure!(
        n0 >= 1,
        "only {} pair can be benchmarked; a call needs a planted and a null pair",
        order.len()
    );
    if (n1, n0) != (one.planted, one.null) {
        warn!(
            target: LOG_TARGET,
            "{n1} planted and {n0} null pairs of the {} and {} asked: no more pairs have the \
             receivers to benchmark on",
            one.planted,
            one.null
        );
    }
    if n1 + n0 < 20 {
        warn!(
            target: LOG_TARGET,
            "{n1} planted and {n0} null pairs: AUROC and AUPRC over fewer than 20 rows are coarse"
        );
    }
    let benchmarked: Vec<usize> = labelled.iter().map(|l| l.pair).collect();
    // The real gene is needed only where a field is drawn from it.
    let field_used = scen.ligand_field_pve != 0.0 || scen.response_field_pve != 0.0;
    let confounder = if field_used && scen.confounder == ConfounderKind::RealGene {
        let (g, counts) = real_gene_confounder(&prep, &benchmarked).ok_or_else(|| {
            anyhow::anyhow!(
                "confounder=realgene: no gene outside the call's {} pairs is left to play the \
                 real-gene confounder; ask for fewer pairs (--pairs, --null-pairs) or another \
                 confounder",
                benchmarked.len()
            )
        })?;
        let name = inp.axis.feature_names()[g].clone();
        info!(target: LOG_TARGET, "Real-gene confounder: {name}");
        Some((name, counts))
    } else {
        None
    };
    let gene = confounder.as_ref().map(|(_, c)| c.as_slice());
    let field_seed = mix_seed(mix_seed(a.seed, FIELD_SALT), scen.rep as u64);
    let field = simulate::confounding(tissue, gene, &[], &scen, field_seed)?;

    info!(
        target: LOG_TARGET,
        "Simulating one scenario on {n1} planted and {n0} null pairs..."
    );
    let pb = Progress::new(labelled.len(), "benchmark runs");
    // Each run holds about a pair's worth of vectors, and its pair's counts.
    let per_run = pair_memory(50 + 2, tissue, &eopts.nuisance);
    let workers = pair_workers(per_run, tissue, &eopts.nuisance);
    let runs: Vec<anyhow::Result<SimRow>> = run_bounded(&labelled, workers, |l| {
        let p = &prep.pairs[l.pair];
        let pair = PairSim::new(&prep, eopts, p);
        let base = pair.base(p, gene);
        let shared = SimShared {
            producers: simulate::calibrate_producers(tissue, base.ligand_counts),
            gene_fields: &[],
            field: Some(&field),
        };
        let s = if l.planted {
            scen
        } else {
            Scenario {
                effect_pve: 0.0,
                ..scen
            }
        };
        let r = simulate::simulate(tissue, &base, &shared, &s, eopts, l.seed)
            .with_context(|| format!("{}-{}", p.ligand, p.receptor));
        pb.inc();
        vec![r]
    });
    pb.finish();
    let mut rows = runs.into_iter().collect::<anyhow::Result<Vec<_>>>()?;
    // Wall times differ from call to call: the sidecar keeps them, so the
    // table is the same bytes for the same call.
    let seconds: Vec<(f64, f64)> = rows
        .iter_mut()
        .map(|r| {
            let s = (r.seconds, r.baseline_seconds);
            (r.seconds, r.baseline_seconds) = (f64::NAN, f64::NAN);
            s
        })
        .collect();
    let planted: Vec<bool> = labelled.iter().map(|l| l.planted).collect();
    output::write_sim_scored(&table, &rows, a.plant_region, &planted)?;

    let reference = prep
        .reference_record
        .as_deref()
        .map(|p| -> anyhow::Result<_> {
            Ok(json!({
                "path": crate::lr_activity::recorded(p),
                "sha256": reference::sha256_file(p)?,
            }))
        })
        .transpose()?;
    let pairs: Vec<serde_json::Value> = labelled
        .iter()
        .zip(&rows)
        .zip(&seconds)
        .map(|((l, r), (s, b))| {
            json!({
                "ligand": &*r.ligand,
                "receptor": &*r.receptor,
                "rank": l.rank,
                "planted": l.planted,
                "seed": l.seed,
                "estimable": r.estimable(),
                "field_sha256": r.field_sha256.as_deref(),
                "seconds": s,
                "baseline_seconds": b,
            })
        })
        .collect();
    let record = json!({
        "command": "lra-simulate",
        "mode": "single-scenario",
        "version": env!("CARGO_PKG_VERSION"),
        "seed": a.seed,
        "rep": scen.rep,
        "scenario": ScenarioKeys::from(&scen),
        "plant_region": a.plant_region,
        "planted": n1,
        "null": n0,
        "asked": {"planted": one.planted, "null": one.null},
        "assignment": ASSIGNMENT,
        "assignment_rule": "pairs ranked by ligand abundance among those with the receivers \
                            to benchmark on, the top planted + null of them labelled: the fewer \
                            label (null when equal) at rank floor((k + 1/2) m / few) for its \
                            k-th of few pairs over m ranks, the other label at the rest; planted \
                            pairs take effect_pve as set, null pairs effect_pve 0",
        "seeds": "pair: mix_seed(seed, rank), rep mixed in per run; hidden field: \
                  mix_seed(mix_seed(seed, 2^64 - 1), rep), one for every pair",
        "field_sha256": simulate::field_sha256(&field),
        "pairs": pairs,
        "real_gene_confounder": confounder.as_ref().map(|(n, _)| &**n),
        "reference": reference,
        "options": crate::util::metadata::LraOptions {
            identity_excluded: prep.identity_excluded.clone(),
            ..a.options()
        },
        "table": crate::lr_activity::recorded(Path::new(&table)),
        // Read back from the written table: the sidecar can only follow it.
        "table_sha256": reference::sha256_file(Path::new(&table))?,
    });
    let mut text = serde_json::to_string_pretty(&record)?;
    text.push('\n');
    reference::write_atomic(Path::new(&sidecar), text.as_bytes())?;
    println!("lra-simulate: wrote {n1} planted and {n0} null runs to {table}, then {sidecar}");
    Ok(())
}
