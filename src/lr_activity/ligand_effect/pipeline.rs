//! The commands from input to tables: `pinto lra` and `pinto lra-simulate`.

use super::estimate::{estimate_pair_with, PairData};
use super::genes::GeneStore;
use super::memory::{pair_memory, pair_workers, run_bounded};
use super::options::EffectOpts;
use super::overlap::NuisanceOpts;
use super::pairs::{pair_genes, reference_pairs, scarcest_subunit, signalling_genes, Pair};
use super::simulate::{self, default_grid, simulate, SimBase, SimShared, Study};
use super::tissue::{Tissue, TissueInputs, TissueOpts};
use super::{
    background, estimate, exposure, lra_simulate_path, only, options, output, receivers, report,
    rows, RunInputs, INTERNALS,
};
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use data_beans::alg::random_projection::{binary_sort_columns, RandProjOps};
use data_beans::utilities::name_matching::GeneIndex;
use std::path::Path;

/// The single competence threshold `lra-simulate` plants on
/// ([`super::Internals::benchmark_competence`]).
pub(super) const BENCHMARK_COMPETENCE: [f64; 1] = [0.1];

/// The shared progress bar over the long stages. Where no bar can draw (the
/// log of a background programme job, a pipe) each twentieth of the way is logged too.
struct Progress {
    bar: indicatif::ProgressBar,
    done: std::sync::atomic::AtomicU64,
}

impl Progress {
    fn new(total: usize, what: &'static str) -> Self {
        Self {
            bar: new_progress_bar(total as u64).with_message(what),
            done: 0.into(),
        }
    }

    fn inc(&self) {
        self.bar.inc(1);
        let done = 1 + self.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let total = self.bar.length().unwrap_or(done).max(1);
        if self.bar.is_hidden() && done * 20 / total > (done - 1) * 20 / total {
            // Not through the log, whose default level would hide it.
            eprintln!(
                "{done}/{total} {} ({:.0}s)",
                self.bar.message(),
                self.bar.elapsed().as_secs_f64()
            );
        }
    }

    fn finish(&self) {
        self.bar.finish_and_clear();
    }
}

/// Everything both the analysis and the benchmark start from.
struct Prepared {
    tissue: Tissue,
    pairs: Vec<Pair>,
    /// Pairs the full analysis would estimate, of which `pairs` are kept.
    n_all: usize,
    store: GeneStore,
    local: HashMap<usize, usize>,
    eopts: EffectOpts,
    reference_record: Option<std::path::PathBuf>,
    /// Genes of the background pool ([`background::BackgroundPool`]), in `local`.
    background_genes: Vec<usize>,
    /// The genes the receivers' identity leaves out, by name, sorted.
    identity_excluded: Vec<String>,
}

impl Prepared {
    fn ligand_counts(&self, p: &Pair) -> Vec<f64> {
        self.complex_counts(&p.ligand_genes)
    }

    fn receptor_counts(&self, p: &Pair) -> Vec<f64> {
        self.complex_counts(&p.receptor_genes)
    }

    /// A complex is present to the extent its scarcest subunit is.
    fn complex_counts(&self, genes: &[usize]) -> Vec<f64> {
        let n = self.tissue.n();
        if let [g] = genes {
            return self.store.dense(self.local[g], n);
        }
        let cols: Vec<Vec<f64>> = genes
            .iter()
            .map(|g| self.store.dense(self.local[g], n))
            .collect();
        scarcest_subunit(&cols)
    }
}

/// The receivers' identity projection and its bins (`identity_bits` of
/// it), every row of an `excluded` gene weighted out, so a receiver's
/// identity is not partly its own outcome.
pub(super) fn identity_bins(
    data: &SparseIoVec,
    axis: &FeatureAxis,
    excluded: &HashSet<usize>,
    cell_section: &[Box<str>],
    block_size: Option<usize>,
) -> anyhow::Result<(Mat, Vec<usize>)> {
    let k = INTERNALS;
    let row_weights: Vec<f32> = (0..data.num_rows())
        .map(|r| {
            if excluded.contains(&axis.feature_of_row(r)) {
                0.0
            } else {
                1.0
            }
        })
        .collect();
    let n_excluded = row_weights.iter().filter(|&&w| w == 0.0).count();
    if 2 * n_excluded > row_weights.len() {
        warn!(
            "Receiver identity leaves out {n_excluded} of {} rows (signalling and \
             programme genes); on a small panel it rests on few genes",
            row_weights.len()
        );
    }
    let identity = data.project_columns_weighted(
        k.identity_dim,
        block_size,
        Some(cell_section),
        &row_weights,
    )?;
    let bits = k.identity_bits.min(k.identity_dim);
    let bins = binary_sort_columns(&identity.proj, bits)?;
    Ok((identity.proj, bins))
}

fn prepare(inp: &RunInputs<'_>) -> anyhow::Result<Prepared> {
    let a = &inp.settings;
    let k = INTERNALS;
    let gene_names = inp.axis.feature_names().to_vec();
    let index = GeneIndex::build(&gene_names);
    let (all, record) = reference_pairs(&a.choices(), &gene_names, &index)?;
    let (excluded, background_genes) = signalling_genes(&all, gene_names.len(), a.seed);
    let mut identity_excluded: Vec<String> = excluded
        .iter()
        .map(|&g| gene_names[g].to_string())
        .collect();
    identity_excluded.sort_unstable();
    let n_all = all.len();
    let pairs: Vec<Pair> = if a.only.is_empty() {
        all
    } else {
        let wanted = only::parse(a.only)?;
        let keep = only::select(&wanted, &all, &gene_names, &index)?;
        info!(
            "--only: {} of {n_all} pairs ({})",
            keep.iter().filter(|&&k| k).count(),
            wanted
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        all.into_iter()
            .zip(keep)
            .filter_map(|(p, k)| k.then_some(p))
            .collect()
    };

    let (receivers, row_of_cell) = receivers::receiver_rows(inp.coords, inp.pairs);
    // The genes estimated pairs read, and the background programmes.
    let mut needed = pair_genes(&pairs);
    needed.extend(&background_genes);
    needed.sort_unstable();
    needed.dedup();
    let local: HashMap<usize, usize> = needed.iter().enumerate().map(|(k, &g)| (g, k)).collect();
    info!(
        "Reading {} genes (ligands, receptors, targets) sparsely over {} receivers...",
        needed.len(),
        receivers.len()
    );
    let (store, log1p_depth) =
        GeneStore::read(inp.data, inp.axis, &needed, &row_of_cell, a.block_size)?;

    let (identity, identity_bin) = identity_bins(
        inp.data,
        inp.axis,
        &excluded,
        inp.cell_section,
        a.block_size,
    )?;

    let topts = TissueOpts {
        scales: k.scales.to_vec(),
        tile: k.tile,
        folds: k.folds,
        ring: k.ring,
        seed: a.seed,
        ..TissueOpts::default()
    };
    let tissue = Tissue::build(
        &TissueInputs {
            coords: inp.coords,
            section: inp.cell_section,
            pairs: inp.pairs,
            identity: &identity,
            identity_bin: &identity_bin,
            log1p_depth: &log1p_depth,
            cell_type: inp.cell_type.map(|(codes, _)| codes),
        },
        &topts,
    )?
    .with_cell_type_names(inp.cell_type.map_or_else(Vec::new, |(_, n)| n.to_vec()));
    debug_assert_eq!(tissue.cell, receivers);
    let eopts = EffectOpts {
        threshold: None,
        min_overlap: k.min_overlap,
        abf_prior_sd: k.abf_prior_sd,
        background_draws: k.background_draws,
        seed: a.seed,
        competences: &options::COMPETENCE_GRID,
        curve: true,
        nuisance: NuisanceOpts {
            seed: a.seed,
            ..NuisanceOpts::default()
        },
        min_arm: k.min_arm,
        min_ess: k.min_ess,
        fit_thin: false,
    };
    Ok(Prepared {
        tissue,
        pairs,
        n_all,
        store,
        local,
        eopts,
        reference_record: Some(record),
        background_genes,
        identity_excluded,
    })
}

/// Record lra's outputs and the fixed settings in the run's manifest, each
/// path absolute ([`crate::lr_activity::recorded`]).
pub(super) fn record(
    manifest: &Path,
    set: impl FnOnce(&mut crate::util::metadata::LraFiles),
) -> anyhow::Result<()> {
    crate::util::metadata::PintoMetadata::backfill(manifest, |m| {
        set(&mut m.outputs.lra);
        m.outputs.lra.settings = serde_json::to_value(INTERNALS).ok();
    });
    Ok(())
}

pub fn run(inp: &RunInputs<'_>) -> anyhow::Result<()> {
    let a = &inp.settings;
    let prep = prepare(inp)?;
    let (tissue, store, local, eopts) = (&prep.tissue, &prep.store, &prep.local, &prep.eopts);
    let pool = background::BackgroundPool::new(
        &tissue.size_factor,
        prep.background_genes
            .iter()
            .map(|g| store.nonzeros(local[g]))
            .collect(),
    );

    let pb = Progress::new(prep.pairs.len(), "ligand-receptor pairs");
    // Each live pair holds a few dozen receiver-length vectors and its
    // propensity design, which bounds how many run at once
    // ([`super::memory::pair_workers`]). Beside the pair's own vectors, a
    // threshold's fits ([`fits::ThresholdFits`]) hold per competence threshold
    // an effect modifier and a pair column, and own ligand, niche, the exposure
    // column and the main, ring and three step propensities.
    let nc = eopts.competences.len();
    let threshold_fits = 2 * nc + 2 + 1 + 1 + 1 + 3;
    let per_pair = pair_memory(
        34 + threshold_fits + 3 * eopts.background_draws,
        tissue,
        &eopts.nuisance,
    );
    let concurrent = pair_workers(per_pair, tissue, &eopts.nuisance).min(prep.pairs.len());
    if concurrent > 1 {
        info!("Estimating up to {concurrent} pairs at a time");
    }
    let one = |p: &Pair| -> Vec<rows::EffectRow> {
        let lig = prep.ligand_counts(p);
        let rec = prep.receptor_counts(p);
        // Two weightings of the programme, as given and with hub targets
        // down-weighted by specificity, averaged by Rubin's rules: the
        // spread between them widens the interval instead of a switch
        // picking one.
        let weighting = |spec: bool| -> Vec<(f64, &[(u32, f32)])> {
            p.targets
                .iter()
                .map(|(g, w, s)| (if spec { w * s } else { *w }, store.nonzeros(local[g])))
                .filter(|(w, _)| *w != 0.0)
                .collect()
        };
        let (plain, specific) = (weighting(false), weighting(true));
        let data = PairData {
            ligand: &p.ligand,
            receptor: &p.receptor,
            ligand_counts: &lig,
            receptor_counts: &rec,
            targets: &plain,
            reweighted: (specific.len() >= INTERNALS.min_targets).then_some(specific.as_slice()),
            n_removed: p.n_removed,
        };
        let out = estimate_pair_with(tissue, &data, &pool, eopts).unwrap_or_else(|e| {
            warn!("{}-{}: {e}", p.ligand, p.receptor);
            Vec::new()
        });
        pb.inc();
        out
    };
    let t_pairs = std::time::Instant::now();
    let mut rows: Vec<rows::EffectRow> = run_bounded(&prep.pairs, concurrent, one);
    pb.finish();
    info!(
        "Estimated {} pairs in {:.0}s ({concurrent} at a time). Thread time: {}",
        prep.pairs.len(),
        t_pairs.elapsed().as_secs_f64(),
        estimate::timing::summary()
    );
    report::report(&mut rows, &INTERNALS.report_opts());
    let subset = !a.only.is_empty();
    let made = crate::lr_activity::lra_outputs(
        crate::lr_activity::args::LraMethod::ContactEffect,
        a.out,
        a.only,
    );
    let [path, curve_path] = [&made[0], &made[1]].map(|p| p.to_string_lossy().into_owned());
    let q_scope = if subset {
        only::q_scope(prep.pairs.len(), prep.n_all)
    } else {
        output::Q_SCOPE_ALL.into()
    };
    output::write_effects(&path, &rows, &q_scope)?;
    output::write_curve(&curve_path, &rows)?;
    let recorded = |p: &str| crate::lr_activity::recorded(Path::new(p));
    // The genes left out of the identity, so the viewer draws the
    // receivers as they were without the files that decided them.
    let options = crate::util::metadata::LraOptions {
        identity_excluded: prep.identity_excluded.clone(),
        ..a.options()
    };
    if subset {
        let entry = crate::util::metadata::LraSubset {
            only: only::canonical(a.only),
            pairs: prep.pairs.len(),
            of: prep.n_all,
            table: recorded(&path),
            curve: recorded(&curve_path),
            options: Some(options),
        };
        record(a.manifest, |o| o.record_subset(entry))?;
    } else {
        record(a.manifest, |o| {
            o.table = Some(recorded(&path));
            o.curve = Some(recorded(&curve_path));
            o.record_options(options);
            o.reference = prep
                .reference_record
                .as_deref()
                .map(crate::lr_activity::recorded);
        })?;
    }
    let power_check = crate::util::metadata::PintoMetadata::read(a.manifest)
        .ok()
        .and_then(|m| m.outputs.lra.simulate);
    println!(
        "{}",
        output::console_summary(&rows, &path, power_check.as_deref())
    );
    if subset {
        println!("  {q_scope}; the full table, when there is one, is unchanged");
    }
    Ok(())
}

pub fn run_simulate(inp: &RunInputs<'_>, study: &Study) -> anyhow::Result<()> {
    let a = &inp.settings;
    let prep = prepare(inp)?;
    let (tissue, store, local) = (&prep.tissue, &prep.store, &prep.local);
    // The benchmark plants on one competence threshold.
    let eopts = &EffectOpts {
        competences: &BENCHMARK_COMPETENCE,
        ..prep.eopts
    };
    let pairs = &prep.pairs;
    let n = tissue.n();
    // The most abundant distinct ligands among pairs with receivers to estimate
    // on: a benchmark on a ligand seen in a handful of cells would measure
    // sparsity, not the estimator, one without competent receivers would measure
    // nothing, and three receptors of one ligand would benchmark one ligand three
    // times.
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    let abundance = |p: &Pair| {
        p.ligand_genes
            .iter()
            .map(|g| store.total(local[g]))
            .fold(f64::INFINITY, f64::min)
    };
    order.sort_by(|&x, &y| abundance(&pairs[y]).total_cmp(&abundance(&pairs[x])));
    // Both arms at `min_arm` receivers, were the competent receivers split evenly.
    let enough = 2 * eopts.min_arm;
    let has_receivers = |k: usize| {
        let detected: Vec<bool> = prep
            .receptor_counts(&pairs[k])
            .iter()
            .map(|&r| r > 0.0)
            .collect();
        let competent = exposure::competence(&tissue.identity_bin, &detected, eopts.competence());
        competent.iter().filter(|&&c| c).count() >= enough
    };
    let mut seen_ligand: HashSet<&[usize]> = HashSet::default();
    let order: Vec<usize> = order
        .into_iter()
        .filter(|&k| {
            let ligand = pairs[k].ligand_genes.as_slice();
            !seen_ligand.contains(ligand) && has_receivers(k) && seen_ligand.insert(ligand)
        })
        .take(study.pairs)
        .collect();
    anyhow::ensure!(
        !order.is_empty(),
        "no pair has {enough} receptor-competent receivers to benchmark on"
    );
    // The real-gene confounder: the most abundant gene read for the run
    // that no benchmarked pair uses. It is left out of the identity
    // projection with the other signalling and programme genes, so it lies
    // outside the adjustment's basis.
    let mut used: HashSet<usize> = HashSet::default();
    for &k in &order {
        let p = &pairs[k];
        used.extend(&p.ligand_genes);
        used.extend(&p.receptor_genes);
        used.extend(p.targets.iter().map(|(g, _, _)| *g));
    }
    let confounder_gene = local
        .iter()
        .filter(|(g, _)| !used.contains(*g))
        .max_by(|x, y| {
            store
                .total(*x.1)
                .total_cmp(&store.total(*y.1))
                .then(y.0.cmp(x.0))
        })
        .map(|(&g, &l)| (g, store.dense(l, n)));
    let mut gopts = study.grid();
    match &confounder_gene {
        Some((g, _)) => {
            let name = pairs
                .iter()
                .find(|p| p.ligand_genes == [*g])
                .map_or_else(|| format!("gene #{g}"), |p| p.ligand.to_string());
            info!("Real-gene confounder: {name}");
        }
        None if gopts
            .confounders
            .contains(&simulate::ConfounderKind::RealGene) =>
        {
            warn!("No gene outside the benchmarked pairs for the real-gene confounder; dropped");
            gopts
                .confounders
                .retain(|&c| c != simulate::ConfounderKind::RealGene);
        }
        None => {}
    }
    let grid = default_grid(&gopts);
    info!(
        "Simulating {} scenarios × {} pairs on the real tissue...",
        grid.len(),
        order.len()
    );
    // The real-gene field at each scale, shared by every pair.
    let gene_fields = simulate::gene_fields(
        tissue,
        confounder_gene.as_ref().map(|(_, v)| v.as_slice()),
        &gopts.scales,
    );
    let pb = Progress::new(grid.len() * order.len(), "benchmark runs");
    let mut rows = Vec::new();
    for &k in &order {
        let p = &pairs[k];
        let lig = prep.ligand_counts(p);
        let rec = prep.receptor_counts(p);
        let detected: Vec<bool> = rec.iter().map(|&r| r > 0.0).collect();
        let competent = exposure::competence(&tissue.identity_bin, &detected, eopts.competence());
        let weights: Vec<f64> = p.targets.iter().map(|(_, w, _)| *w).collect();
        let genes: Vec<&[(u32, f32)]> = p
            .targets
            .iter()
            .map(|(g, _, _)| store.nonzeros(local[g]))
            .collect();
        let bin_means = simulate::target_bin_means(tissue, &genes);
        let base = SimBase {
            ligand: &p.ligand,
            receptor: &p.receptor,
            ligand_counts: &lig,
            target_weights: &weights,
            target_bin_means: &bin_means,
            competent: &competent,
            receptor_counts: &rec,
            confounder_gene: confounder_gene.as_ref().map(|(_, v)| v.as_slice()),
        };
        let shared = SimShared {
            producers: simulate::calibrate_producers(tissue, base.ligand_counts),
            gene_fields: &gene_fields,
        };
        // Scenarios are independent: run as many at once as the memory
        // budget allows (each holds about a pair's worth of vectors).
        let per_run = pair_memory(50 + 3 * eopts.background_draws, tissue, &eopts.nuisance);
        let concurrent = pair_workers(per_run, tissue, &eopts.nuisance);
        let jobs: Vec<(usize, &simulate::Scenario)> = grid.iter().enumerate().collect();
        rows.extend(run_bounded(&jobs, concurrent, |&(si, scen)| {
            let seed =
                legume_numeric::matrix::rand_util::mix_seed(a.seed, (k * grid.len() + si) as u64);
            let r = match simulate(tissue, &base, &shared, scen, eopts, seed) {
                Ok(r) => vec![r],
                Err(e) => {
                    warn!("{}-{} scenario {si}: {e}", p.ligand, p.receptor);
                    Vec::new()
                }
            };
            pb.inc();
            r
        }));
    }
    pb.finish();
    let path = lra_simulate_path(a.out);
    output::write_sim(&path, &rows)?;
    record(a.manifest, |o| {
        o.simulate = Some(crate::lr_activity::recorded(Path::new(&path)))
    })?;
    println!("{}", output::sim_summary(&rows));
    println!("lra-simulate: wrote {} runs to {path}", rows.len());
    Ok(())
}

/// Coordinates per data column from the `lc` run: `{prefix}.cells.parquet`
/// when the run wrote one, else the endpoints in `{prefix}.coord_pairs.parquet`.
pub fn read_coords(
    lc_prefix: &str,
    cell_to_col: &HashMap<Box<str>, usize>,
    n_cells: usize,
) -> anyhow::Result<Vec<Option<(f32, f32)>>> {
    use crate::util::parquet_io::{read_cells_from_coord_pairs, read_cells_table};
    let cells = std::path::PathBuf::from(format!("{lc_prefix}.cells.parquet"));
    let table = if cells.exists() {
        read_cells_table(&cells, None)?
    } else {
        read_cells_from_coord_pairs(
            std::path::Path::new(&format!("{lc_prefix}.coord_pairs.parquet")),
            None,
        )?
    };
    let mut out = vec![None; n_cells];
    let mut found = 0usize;
    for (name, &xy) in table.names.iter().zip(&table.coords) {
        if let Some(&c) = cell_to_col.get(name) {
            out[c] = Some(xy);
            found += 1;
        }
    }
    anyhow::ensure!(
        found > 0,
        "no cell of the data has coordinates in {}",
        if cells.exists() {
            cells.display().to_string()
        } else {
            format!("{lc_prefix}.coord_pairs.parquet")
        }
    );
    Ok(out)
}
