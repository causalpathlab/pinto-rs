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
    background, exposure, lra_simulate_path, only, options, output, receivers, report, rows,
    timing, RunInputs, Settings, INTERNALS,
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

/// The pairs to estimate: all of them, or those `--only` names.
fn select_only(
    only_names: &[Box<str>],
    all: Vec<Pair>,
    gene_names: &[Box<str>],
    index: &GeneIndex,
) -> anyhow::Result<Vec<Pair>> {
    if only_names.is_empty() {
        return Ok(all);
    }
    let wanted = only::parse(only_names)?;
    let keep = only::select(&wanted, &all, gene_names, index)?;
    info!(
        "--only: {} of {} pairs ({})",
        keep.iter().filter(|&&k| k).count(),
        all.len(),
        wanted
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(all
        .into_iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(p))
        .collect())
}

/// The genes estimated pairs read, and the background programmes, sorted
/// and distinct.
fn genes_to_read(pairs: &[Pair], background_genes: &[usize]) -> Vec<usize> {
    let mut needed = pair_genes(pairs);
    needed.extend(background_genes);
    needed.sort_unstable();
    needed.dedup();
    needed
}

/// The tissue the receivers live in, at the fixed scales and folds.
fn build_tissue(
    inp: &RunInputs<'_>,
    identity: &Mat,
    identity_bin: &[usize],
    log1p_depth: &[f32],
) -> anyhow::Result<Tissue> {
    let k = INTERNALS;
    let topts = TissueOpts {
        scales: k.scales.to_vec(),
        tile: k.tile,
        folds: k.folds,
        ring: k.ring,
        seed: inp.settings.seed,
        ..TissueOpts::default()
    };
    Ok(Tissue::build(
        &TissueInputs {
            coords: inp.coords,
            section: inp.cell_section,
            pairs: inp.pairs,
            identity,
            identity_bin,
            log1p_depth,
            cell_type: inp.cell_type.map(|(codes, _)| codes),
        },
        &topts,
    )?
    .with_cell_type_names(inp.cell_type.map_or_else(Vec::new, |(_, n)| n.to_vec())))
}

/// The estimator's options: the fixed settings and the run's seed.
fn effect_opts(seed: u64) -> EffectOpts {
    let k = INTERNALS;
    EffectOpts {
        threshold: None,
        min_overlap: k.min_overlap,
        abf_prior_sd: k.abf_prior_sd,
        background_draws: k.background_draws,
        seed,
        competences: &options::COMPETENCE_GRID,
        curve: true,
        nuisance: NuisanceOpts {
            seed,
            ..NuisanceOpts::default()
        },
        min_arm: k.min_arm,
        min_ess: k.min_ess,
        fit_thin: false,
    }
}

fn prepare(inp: &RunInputs<'_>) -> anyhow::Result<Prepared> {
    let a = &inp.settings;
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
    let pairs = select_only(a.only, all, &gene_names, &index)?;

    let (receivers, row_of_cell) = receivers::receiver_rows(inp.coords, inp.pairs);
    let needed = genes_to_read(&pairs, &background_genes);
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
    let tissue = build_tissue(inp, &identity, &identity_bin, &log1p_depth)?;
    debug_assert_eq!(tissue.cell, receivers);
    Ok(Prepared {
        tissue,
        pairs,
        n_all,
        store,
        local,
        eopts: effect_opts(a.seed),
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

/// How many pairs to estimate at once. Each live pair holds a few dozen
/// receiver-length vectors and its propensity design, which bounds how many
/// run at once ([`super::memory::pair_workers`]). Beside the pair's own
/// vectors, a threshold's fits ([`super::fits::ThresholdFits`]) hold per competence
/// threshold an effect modifier and a pair column, and own ligand, niche, the
/// exposure column and the main, ring and three step propensities.
fn pairs_at_once(prep: &Prepared) -> usize {
    let (tissue, eopts) = (&prep.tissue, &prep.eopts);
    let nc = eopts.competences.len();
    let threshold_fits = 2 * nc + 2 + 1 + 1 + 1 + 3;
    let per_pair = pair_memory(
        34 + threshold_fits + 3 * eopts.background_draws,
        tissue,
        &eopts.nuisance,
    );
    pair_workers(per_pair, tissue, &eopts.nuisance).min(prep.pairs.len())
}

/// One pair's rows; a pair that fails is warned about and has none.
fn estimate_one(
    prep: &Prepared,
    pool: &background::BackgroundPool<'_>,
    p: &Pair,
) -> Vec<rows::EffectRow> {
    let (store, local) = (&prep.store, &prep.local);
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
    estimate_pair_with(&prep.tissue, &data, pool, &prep.eopts).unwrap_or_else(|e| {
        warn!("{}-{}: {e}", p.ligand, p.receptor);
        Vec::new()
    })
}

/// Every pair's rows, in pair order.
fn estimate_pairs(prep: &Prepared) -> Vec<rows::EffectRow> {
    let pool = background::BackgroundPool::new(
        &prep.tissue.size_factor,
        prep.background_genes
            .iter()
            .map(|g| prep.store.nonzeros(prep.local[g]))
            .collect(),
    );
    let pb = Progress::new(prep.pairs.len(), "ligand-receptor pairs");
    let concurrent = pairs_at_once(prep);
    if concurrent > 1 {
        info!("Estimating up to {concurrent} pairs at a time");
    }
    let t_pairs = std::time::Instant::now();
    let rows = run_bounded(&prep.pairs, concurrent, |p: &Pair| {
        let out = estimate_one(prep, &pool, p);
        pb.inc();
        out
    });
    pb.finish();
    info!(
        "Estimated {} pairs in {:.0}s ({concurrent} at a time). Thread time: {}",
        prep.pairs.len(),
        t_pairs.elapsed().as_secs_f64(),
        timing::summary()
    );
    rows
}

/// Record the tables in the manifest: a `--only` run as a subset beside the
/// full table, any other as the full table with its reference.
fn record_tables(
    a: &Settings<'_>,
    prep: &Prepared,
    path: &str,
    curve_path: &str,
) -> anyhow::Result<()> {
    let recorded = |p: &str| crate::lr_activity::recorded(Path::new(p));
    // The genes left out of the identity, so the viewer draws the
    // receivers as they were without the files that decided them.
    let options = crate::util::metadata::LraOptions {
        identity_excluded: prep.identity_excluded.clone(),
        ..a.options()
    };
    if a.only.is_empty() {
        record(a.manifest, |o| {
            o.table = Some(recorded(path));
            o.curve = Some(recorded(curve_path));
            o.record_options(options);
            o.reference = prep
                .reference_record
                .as_deref()
                .map(crate::lr_activity::recorded);
        })
    } else {
        let entry = crate::util::metadata::LraSubset {
            only: only::canonical(a.only),
            pairs: prep.pairs.len(),
            of: prep.n_all,
            table: recorded(path),
            curve: recorded(curve_path),
            options: Some(options),
        };
        record(a.manifest, |o| o.record_subset(entry))
    }
}

/// Write the effect and curve tables and record them; the effect table's
/// path and the q-values' scope.
fn write_tables(
    a: &Settings<'_>,
    prep: &Prepared,
    rows: &[rows::EffectRow],
) -> anyhow::Result<(String, String)> {
    let made = crate::lr_activity::lra_outputs(
        crate::lr_activity::args::LraMethod::ContactEffect,
        a.out,
        a.only,
    );
    let [path, curve_path] = [&made[0], &made[1]].map(|p| p.to_string_lossy().into_owned());
    let q_scope = if a.only.is_empty() {
        output::Q_SCOPE_ALL.into()
    } else {
        only::q_scope(prep.pairs.len(), prep.n_all)
    };
    output::write_effects(&path, rows, &q_scope)?;
    output::write_curve(&curve_path, rows)?;
    record_tables(a, prep, &path, &curve_path)?;
    Ok((path, q_scope))
}

/// The lines a run ends with.
fn print_run_summary(a: &Settings<'_>, rows: &[rows::EffectRow], path: &str, q_scope: &str) {
    let power_check = crate::util::metadata::PintoMetadata::read(a.manifest)
        .ok()
        .and_then(|m| m.outputs.lra.simulate);
    println!(
        "{}",
        output::console_summary(rows, path, power_check.as_deref())
    );
    if !a.only.is_empty() {
        println!("  {q_scope}; the full table, when there is one, is unchanged");
    }
}

pub fn run(inp: &RunInputs<'_>) -> anyhow::Result<()> {
    let a = &inp.settings;
    let prep = prepare(inp)?;
    let mut rows = estimate_pairs(&prep);
    report::report(&mut rows, &INTERNALS.report_opts());
    let (path, q_scope) = write_tables(a, &prep, &rows)?;
    print_run_summary(a, &rows, &path, &q_scope);
    Ok(())
}

/// Who is competent for this receptor, by the estimator's own rule.
fn competent_receivers(tissue: &Tissue, receptor_counts: &[f64], eopts: &EffectOpts) -> Vec<bool> {
    let detected: Vec<bool> = receptor_counts.iter().map(|&r| r > 0.0).collect();
    exposure::competence(&tissue.identity_bin, &detected, eopts.competence())
}

/// The most abundant distinct ligands among pairs with receivers to estimate
/// on: a benchmark on a ligand seen in a handful of cells would measure
/// sparsity, not the estimator, one without competent receivers would measure
/// nothing, and three receptors of one ligand would benchmark one ligand three
/// times.
fn benchmark_pairs(
    prep: &Prepared,
    eopts: &EffectOpts,
    study: &Study,
) -> anyhow::Result<Vec<usize>> {
    let pairs = &prep.pairs;
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    let abundance = |p: &Pair| {
        p.ligand_genes
            .iter()
            .map(|g| prep.store.total(prep.local[g]))
            .fold(f64::INFINITY, f64::min)
    };
    order.sort_by(|&x, &y| abundance(&pairs[y]).total_cmp(&abundance(&pairs[x])));
    // Both arms at `min_arm` receivers, were the competent receivers split evenly.
    let enough = 2 * eopts.min_arm;
    let has_receivers = |k: usize| {
        let competent = competent_receivers(&prep.tissue, &prep.receptor_counts(&pairs[k]), eopts);
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
    Ok(order)
}

/// The real-gene confounder and its counts: the most abundant gene read for
/// the run that no benchmarked pair uses. It is left out of the identity
/// projection with the other signalling and programme genes, so it lies
/// outside the adjustment's basis.
fn real_gene_confounder(prep: &Prepared, order: &[usize]) -> Option<(usize, Vec<f64>)> {
    let store = &prep.store;
    let mut used: HashSet<usize> = HashSet::default();
    for &k in order {
        let p = &prep.pairs[k];
        used.extend(&p.ligand_genes);
        used.extend(&p.receptor_genes);
        used.extend(p.targets.iter().map(|(g, _, _)| *g));
    }
    prep.local
        .iter()
        .filter(|(g, _)| !used.contains(*g))
        .max_by(|x, y| {
            store
                .total(*x.1)
                .total_cmp(&store.total(*y.1))
                .then(y.0.cmp(x.0))
        })
        .map(|(&g, &l)| (g, store.dense(l, prep.tissue.n())))
}

/// Name the real-gene confounder, or drop that kind from the grid when no
/// gene is left to play it.
fn settle_real_gene(
    pairs: &[Pair],
    confounder_gene: Option<&(usize, Vec<f64>)>,
    gopts: &mut simulate::GridOpts,
) {
    match confounder_gene {
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
}

/// What every benchmarked pair's runs share.
struct Benchmark<'a> {
    prep: &'a Prepared,
    eopts: &'a EffectOpts,
    grid: &'a [simulate::Scenario],
    confounder_gene: Option<&'a [f64]>,
    /// The real-gene field at each scale, shared by every pair.
    gene_fields: &'a [(f64, Vec<f64>)],
    seed: u64,
    pb: &'a Progress,
}

impl Benchmark<'_> {
    /// The runs of pair `k`, one per scenario that ran; a scenario that fails
    /// is warned about and has none.
    fn pair_runs(&self, k: usize) -> Vec<simulate::SimRow> {
        let (prep, eopts, grid) = (self.prep, self.eopts, self.grid);
        let (tissue, store, local) = (&prep.tissue, &prep.store, &prep.local);
        let p = &prep.pairs[k];
        let lig = prep.ligand_counts(p);
        let rec = prep.receptor_counts(p);
        let competent = competent_receivers(tissue, &rec, eopts);
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
            confounder_gene: self.confounder_gene,
        };
        let shared = SimShared {
            producers: simulate::calibrate_producers(tissue, base.ligand_counts),
            gene_fields: self.gene_fields,
        };
        // Scenarios are independent: run as many at once as the memory
        // budget allows (each holds about a pair's worth of vectors).
        let per_run = pair_memory(50 + 3 * eopts.background_draws, tissue, &eopts.nuisance);
        let concurrent = pair_workers(per_run, tissue, &eopts.nuisance);
        let jobs: Vec<(usize, &simulate::Scenario)> = grid.iter().enumerate().collect();
        run_bounded(&jobs, concurrent, |&(si, scen)| {
            let seed = legume_numeric::matrix::rand_util::mix_seed(
                self.seed,
                (k * grid.len() + si) as u64,
            );
            let r = match simulate(tissue, &base, &shared, scen, eopts, seed) {
                Ok(r) => vec![r],
                Err(e) => {
                    warn!("{}-{} scenario {si}: {e}", p.ligand, p.receptor);
                    Vec::new()
                }
            };
            self.pb.inc();
            r
        })
    }
}

pub fn run_simulate(inp: &RunInputs<'_>, study: &Study) -> anyhow::Result<()> {
    let a = &inp.settings;
    let prep = prepare(inp)?;
    // The benchmark plants on one competence threshold.
    let eopts = &EffectOpts {
        competences: &BENCHMARK_COMPETENCE,
        ..prep.eopts
    };
    let order = benchmark_pairs(&prep, eopts, study)?;
    let confounder_gene = real_gene_confounder(&prep, &order);
    let mut gopts = study.grid();
    settle_real_gene(&prep.pairs, confounder_gene.as_ref(), &mut gopts);
    let grid = default_grid(&gopts);
    info!(
        "Simulating {} scenarios × {} pairs on the real tissue...",
        grid.len(),
        order.len()
    );
    let confounder_counts = confounder_gene.as_ref().map(|(_, v)| v.as_slice());
    let gene_fields = simulate::gene_fields(&prep.tissue, confounder_counts, &gopts.scales);
    let pb = Progress::new(grid.len() * order.len(), "benchmark runs");
    let benchmark = Benchmark {
        prep: &prep,
        eopts,
        grid: &grid,
        confounder_gene: confounder_counts,
        gene_fields: &gene_fields,
        seed: a.seed,
        pb: &pb,
    };
    let mut rows = Vec::new();
    for &k in &order {
        rows.extend(benchmark.pair_runs(k));
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
