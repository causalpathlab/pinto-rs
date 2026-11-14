//! The commands from input to tables: `pinto lra` and `pinto lra-simulate`.

use super::baselines::{marginals, Baselines, CellTypeTest, TypeLabels};
use super::estimate::{estimate_pair_shared, PairData};
use super::fits::LigandShare;
use super::genes::GeneStore;
use super::memory::{ligand_memory, pair_memory, pair_workers, run_bounded};
use super::options::EffectOpts;
use super::overlap::NuisanceOpts;
use super::pairs::{pair_genes, reference_pairs, scarcest_subunit, signalling_genes, Pair};
use super::simulate::{self, default_grid, simulate, SimBase, SimShared};
use super::tissue::{identity_bins, Tissue, TissueInputs, TissueOpts, ALL_SECTIONS};
use super::{
    contacts, only, options, output, receivers, report, rows, timing, RunInputs, Settings,
    INTERNALS, LOG_TARGET,
};
use crate::lr_activity::args::RegionJackknife;
use crate::util::common::*;
use data_beans::utilities::name_matching::GeneIndex;
use std::path::Path;

mod single;

#[cfg(test)]
pub(super) use single::alternate_ranks;
pub use single::run_simulate_one;

/// The shared progress bar over the long stages. Where no bar can draw (the
/// log of a background job, a pipe) each twentieth of the way is logged too.
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
            let elapsed = self.bar.elapsed().as_secs_f64();
            // The rate so far, which for runs of like cost is the honest guess.
            let left = elapsed * total.saturating_sub(done) as f64 / done as f64;
            eprintln!(
                "{done}/{total} {} ({elapsed:.0}s, about {left:.0}s left)",
                self.bar.message(),
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
        target: LOG_TARGET,
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
            region: inp.region,
        },
        &topts,
    )?
    .with_cell_type_names(inp.cell_type.map_or_else(Vec::new, |(_, n)| n.to_vec())))
}

/// The estimator's options: the fixed settings and the run's seed.
fn effect_opts(seed: u64, region_jackknife: RegionJackknife) -> EffectOpts {
    let k = INTERNALS;
    EffectOpts {
        keep_arm_weights: false,
        threshold: None,
        min_overlap: k.min_overlap,
        abf_prior_scale: k.abf_prior_scale,
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
        region_jackknife,
    }
}

fn prepare(inp: &RunInputs<'_>) -> anyhow::Result<Prepared> {
    let a = &inp.settings;
    let gene_names = inp.axis.feature_names().to_vec();
    let index = GeneIndex::build(&gene_names);
    let (all, record) = reference_pairs(&a.choices(), &gene_names, &index)?;
    let excluded = signalling_genes(&all);
    let mut identity_excluded: Vec<String> = excluded
        .iter()
        .map(|&g| gene_names[g].to_string())
        .collect();
    identity_excluded.sort_unstable();
    let n_all = all.len();
    let pairs = select_only(a.only, all, &gene_names, &index)?;

    let (receivers, row_of_cell) = receivers::receiver_rows(inp.coords, inp.pairs);
    let needed = pair_genes(&pairs);
    let local: HashMap<usize, usize> = needed.iter().enumerate().map(|(k, &g)| (g, k)).collect();
    info!(
        target: LOG_TARGET,
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
        eopts: effect_opts(a.seed, a.region_jackknife),
        reference_record: Some(record),
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

/// How many ligands to estimate at once, each its pairs one after another.
/// A live pair holds a few dozen receiver-length vectors and its propensity
/// design, and its ligand the shared base propensities, which bounds how many
/// run at once ([`super::memory::pair_workers`]). Beside the pair's own
/// vectors, a threshold's fits ([`super::fits::ThresholdFits`]) hold per competence
/// threshold an effect modifier and a pair column, and own ligand, niche and
/// the main and three step propensities.
fn ligands_at_once(prep: &Prepared, ligands: usize) -> usize {
    let (tissue, eopts) = (&prep.tissue, &prep.eopts);
    let nc = eopts.competences.len();
    let threshold_fits = 2 * nc + 2 + 1 + 3;
    let per_ligand = ligand_memory(34 + threshold_fits, tissue, &eopts.nuisance);
    pair_workers(per_ligand, tissue, &eopts.nuisance).min(ligands)
}

/// The pairs grouped by ligand (its genes, so a complex is one ligand), each
/// group in pair order.
fn ligand_groups(pairs: &[Pair]) -> Vec<Vec<usize>> {
    use crate::util::ablation::{dense_ids, group};
    group(&dense_ids(pairs.iter().map(|p| p.ligand_genes.as_slice())))
}

/// One pair's rows, its propensities bordered from its ligand's `share`; a
/// pair that fails is warned about and has none.
fn estimate_one(prep: &Prepared, p: &Pair, share: &LigandShare) -> Vec<rows::EffectRow> {
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
    estimate_pair_shared(&prep.tissue, &data, &prep.eopts, share).unwrap_or_else(|e| {
        warn!(target: LOG_TARGET, "{}-{}: {e}", p.ligand, p.receptor);
        Vec::new()
    })
}

/// Every pair's rows, in pair order. The pairs of one ligand run one after
/// another on one worker and share its base propensities
/// ([`LigandShare`]); ligands run side by side.
fn estimate_pairs(prep: &Prepared) -> Vec<rows::EffectRow> {
    let pb = Progress::new(prep.pairs.len(), "ligand-receptor pairs");
    // Largest first, so a ligand with many pairs does not start last and
    // set the finishing time; the rows are put back in pair order below.
    let mut groups = ligand_groups(&prep.pairs);
    groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
    let concurrent = ligands_at_once(prep, groups.len());
    if concurrent > 1 {
        info!(target: LOG_TARGET, "Estimating up to {concurrent} ligands at a time");
    }
    let t_pairs = std::time::Instant::now();
    let mut by_pair = run_bounded(&groups, concurrent, |group: &Vec<usize>| {
        let share = LigandShare::default();
        group
            .iter()
            .map(|&k| {
                let out = estimate_one(prep, &prep.pairs[k], &share);
                pb.inc();
                (k, out)
            })
            .collect()
    });
    by_pair.sort_by_key(|(k, _)| *k);
    let rows: Vec<rows::EffectRow> = by_pair.into_iter().flat_map(|(_, r)| r).collect();
    pb.finish();
    info!(
        target: LOG_TARGET,
        "Estimated {} pairs in {:.0}s ({concurrent} at a time). Thread time: {}",
        prep.pairs.len(),
        t_pairs.elapsed().as_secs_f64(),
        timing::summary()
    );
    rows
}

/// `--marginal-baselines`: what the run's pairs were scored against.
struct Marginal {
    labels: TypeLabels,
    /// Each pair's baselines and cell-type test, in pair order.
    tests: Vec<(Baselines, CellTypeTest)>,
}

/// A pair's own seed, from its names, so that a `--only` subset scores a
/// pair as the full run does. FNV-1a, which no Rust release changes.
fn pair_seed(seed: u64, p: &Pair) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in p.ligand.bytes().chain([0]).chain(p.receptor.bytes()) {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    legume_numeric::matrix::rand_util::mix_seed(seed, h)
}

/// Every pair's co-expression baselines over the whole tissue
/// ([`marginals`]), on the run's cell types when it has any, else on the
/// identity bins, each permutation test on up to `permutations`.
fn marginal_baselines(prep: &Prepared, permutations: usize, seed: u64) -> Marginal {
    let labels = TypeLabels::of(&prep.tissue);
    info!(
        target: LOG_TARGET,
        "Marginal baselines on {}, up to {permutations} permutations",
        labels.note(&prep.tissue)
    );
    let pb = Progress::new(prep.pairs.len(), "marginal baselines");
    let t0 = std::time::Instant::now();
    let tests = prep
        .pairs
        .par_iter()
        .map(|p| {
            let (l, r) = (prep.ligand_counts(p), prep.receptor_counts(p));
            let out = marginals(
                &prep.tissue,
                &l,
                &r,
                pair_seed(seed, p),
                &labels,
                permutations,
                // the per-type-pair table's p and q need every permutation
                false,
            );
            pb.inc();
            out
        })
        .collect();
    pb.finish();
    info!(
        target: LOG_TARGET,
        "Marginal baselines of {} pairs in {:.0}s",
        prep.pairs.len(),
        t0.elapsed().as_secs_f64()
    );
    Marginal { labels, tests }
}

/// Put each pair's baselines on its pooled row.
fn attach_marginals(rows: &mut [rows::EffectRow], prep: &Prepared, m: &Marginal) {
    let by_pair: HashMap<(&str, &str), &Baselines> = prep
        .pairs
        .iter()
        .zip(&m.tests)
        .map(|(p, (b, _))| ((&*p.ligand, &*p.receptor), b))
        .collect();
    for r in rows.iter_mut().filter(|r| &*r.section == ALL_SECTIONS) {
        r.marginal = by_pair.get(&(&*r.ligand, &*r.receptor)).map(|&&b| b);
    }
}

/// Record the tables in the manifest: a `--only` run as a subset beside the
/// full table, any other as the full table with its reference.
fn record_tables(
    a: &Settings<'_>,
    prep: &Prepared,
    path: &str,
    curve_path: &str,
    celltype: Option<(&str, &Marginal)>,
) -> anyhow::Result<()> {
    let recorded = |p: &str| crate::lr_activity::recorded(Path::new(p));
    // The genes left out of the identity, so the viewer draws the
    // receivers as they were without the files that decided them.
    let options = crate::util::metadata::LraOptions {
        identity_excluded: prep.identity_excluded.clone(),
        marginal_baselines: celltype.map(|(_, m)| m.labels.source.name().into()),
        ..a.options()
    };
    if a.only.is_empty() {
        record(a.manifest, |o| {
            o.table = Some(recorded(path));
            o.curve = Some(recorded(curve_path));
            o.celltype = celltype.map(|(p, _)| recorded(p));
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
            celltype: celltype.map(|(p, _)| recorded(p)),
            options: Some(options),
        };
        record(a.manifest, |o| o.record_subset(entry))
    }
}

/// Write the effect and curve tables (and with `marginal` the per-type-pair
/// table) and record them; the effect table's path and the q-values' scope.
fn write_tables(
    a: &Settings<'_>,
    prep: &Prepared,
    rows: &[rows::EffectRow],
    marginal: Option<&Marginal>,
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
    output::write_effects(&path, rows, &q_scope, marginal.is_some())?;
    output::write_curve(&curve_path, rows)?;
    let celltype_path = output::celltype_beside(&path);
    if let Some(m) = marginal {
        let table: Vec<output::CelltypeRow> = prep
            .pairs
            .iter()
            .zip(&m.tests)
            .flat_map(|(p, (_, t))| output::celltype_rows(&p.ligand, &p.receptor, t, &m.labels))
            .collect();
        output::write_celltype(&celltype_path, &table)?;
        info!(
            target: LOG_TARGET,
            "Wrote {} tested type pairs to {celltype_path}",
            table.len()
        );
    }
    let celltype = marginal.map(|m| (celltype_path.as_str(), m));
    record_tables(a, prep, &path, &curve_path, celltype)?;
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
    report::report(&mut rows, &report::ReportOpts::default());
    let marginal = a.marginal.map(|n| marginal_baselines(&prep, n, a.seed));
    if let Some(m) = &marginal {
        attach_marginals(&mut rows, &prep, m);
    }
    let (path, q_scope) = write_tables(a, &prep, &rows, marginal.as_ref())?;
    print_run_summary(a, &rows, &path, &q_scope);
    Ok(())
}

/// Who is competent for this receptor, by the estimator's own rule.
fn competent_receivers(tissue: &Tissue, receptor_counts: &[f64], eopts: &EffectOpts) -> Vec<bool> {
    let detected: Vec<bool> = receptor_counts.iter().map(|&r| r > 0.0).collect();
    contacts::competence(&tissue.identity_bin, &detected, eopts.competence())
}

/// The most abundant distinct ligands among pairs with receivers to estimate
/// on: a benchmark on a ligand seen in a handful of cells would measure
/// sparsity, not the estimator, one without competent receivers would measure
/// nothing, and three receptors of one ligand would benchmark one ligand three
/// times.
///
/// With a planted region, the competent receivers counted are those inside it.
fn benchmark_pairs(
    prep: &Prepared,
    eopts: &EffectOpts,
    n_pairs: usize,
    region: Option<u32>,
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
    let inside = region
        .map(|code| prep.tissue.region_rows(code).map(|r| prep.tissue.mask(r)))
        .transpose()?;
    let has_receivers = |k: usize| {
        let competent = competent_receivers(&prep.tissue, &prep.receptor_counts(&pairs[k]), eopts);
        (0..competent.len())
            .filter(|&i| competent[i] && inside.as_ref().is_none_or(|m| m[i]))
            .count()
            >= enough
    };
    let mut seen_ligand: HashSet<&[usize]> = HashSet::default();
    let order: Vec<usize> = order
        .into_iter()
        .filter(|&k| {
            let ligand = pairs[k].ligand_genes.as_slice();
            !seen_ligand.contains(ligand) && has_receivers(k) && seen_ligand.insert(ligand)
        })
        .take(n_pairs)
        .collect();
    anyhow::ensure!(
        !order.is_empty(),
        "no pair has {enough} receptor-competent receivers{} to benchmark on",
        if region.is_some() {
            " in the planted region"
        } else {
            ""
        }
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
            info!(target: LOG_TARGET, "Real-gene confounder: {name}");
        }
        None if gopts
            .confounders
            .contains(&simulate::ConfounderKind::RealGene) =>
        {
            warn!(
                target: LOG_TARGET,
                "No gene outside the benchmarked pairs for the real-gene confounder; dropped"
            );
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

/// What a benchmarked pair's runs draw on, read once: its ligand and receptor
/// counts, its competent receivers, and its programme's weights and bin
/// means.
struct PairSim {
    ligand: Vec<f64>,
    receptor: Vec<f64>,
    competent: Vec<bool>,
    weights: Vec<f64>,
    bin_means: Vec<f64>,
}

impl PairSim {
    fn new(prep: &Prepared, eopts: &EffectOpts, p: &Pair) -> Self {
        let (tissue, store, local) = (&prep.tissue, &prep.store, &prep.local);
        let ligand = prep.ligand_counts(p);
        let receptor = prep.receptor_counts(p);
        let competent = competent_receivers(tissue, &receptor, eopts);
        let weights = p.targets.iter().map(|(_, w, _)| *w).collect();
        let genes: Vec<&[(u32, f32)]> = p
            .targets
            .iter()
            .map(|(g, _, _)| store.nonzeros(local[g]))
            .collect();
        Self {
            ligand,
            receptor,
            competent,
            weights,
            bin_means: simulate::target_bin_means(tissue, &genes),
        }
    }

    fn base<'a>(&'a self, p: &'a Pair, confounder_gene: Option<&'a [f64]>) -> SimBase<'a> {
        SimBase {
            ligand: &p.ligand,
            receptor: &p.receptor,
            ligand_counts: &self.ligand,
            target_weights: &self.weights,
            target_bin_means: &self.bin_means,
            competent: &self.competent,
            receptor_counts: &self.receptor,
            confounder_gene,
        }
    }
}

impl Benchmark<'_> {
    /// The runs of pair `k`, one per scenario that ran; a scenario that fails
    /// is warned about and has none.
    fn pair_runs(&self, k: usize) -> Vec<simulate::SimRow> {
        let (prep, eopts, grid) = (self.prep, self.eopts, self.grid);
        let tissue = &prep.tissue;
        let p = &prep.pairs[k];
        let pair = PairSim::new(prep, eopts, p);
        let base = pair.base(p, self.confounder_gene);
        let shared = SimShared {
            producers: simulate::calibrate_producers(tissue, base.ligand_counts),
            gene_fields: self.gene_fields,
            field: None,
        };
        // Scenarios are independent: run as many at once as the memory
        // budget allows (each holds about a pair's worth of vectors).
        let per_run = pair_memory(50, tissue, &eopts.nuisance);
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
                    warn!(target: LOG_TARGET, "{}-{} scenario {si}: {e}", p.ligand, p.receptor);
                    Vec::new()
                }
            };
            self.pb.inc();
            r
        })
    }
}

/// The region code `--plant-region` names among the tissue's `--region`
/// labels, `unassigned` not one of them; `None` without it.
pub(super) fn planted_region(tissue: &Tissue, name: Option<&str>) -> anyhow::Result<Option<u32>> {
    let Some(name) = name else {
        return Ok(None);
    };
    let (_, names) = tissue
        .region
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--plant-region {name} needs --region"))?;
    let unassigned = crate::lr_activity::input::UNASSIGNED;
    let code = names
        .iter()
        .position(|n| &**n == name)
        .filter(|_| name != unassigned)
        .ok_or_else(|| {
            let choices: Vec<&str> = names
                .iter()
                .map(|n| &**n)
                .filter(|&n| n != unassigned)
                .collect();
            anyhow::anyhow!(
                "--plant-region {name}: not a region of --region (has {})",
                choices.join(", ")
            )
        })?;
    info!(target: LOG_TARGET, "Planting every effect in region {name} alone");
    Ok(Some(code as u32))
}

pub fn run_simulate(inp: &RunInputs<'_>) -> anyhow::Result<()> {
    run_grid(inp, simulate::GridOpts::default())
}

/// The built-in benchmark on the grid `gopts` builds ([`default_grid`]).
pub(super) fn run_grid(inp: &RunInputs<'_>, mut gopts: simulate::GridOpts) -> anyhow::Result<()> {
    let a = &inp.settings;
    let prep = prepare(inp)?;
    // The benchmark plants on one competence threshold.
    let eopts = &EffectOpts {
        competences: &options::BENCHMARK_COMPETENCE,
        ..prep.eopts
    };
    let region = planted_region(&prep.tissue, a.plant_region)?;
    let order = benchmark_pairs(&prep, eopts, gopts.pairs, region)?;
    let confounder_gene = real_gene_confounder(&prep, &order);
    settle_real_gene(&prep.pairs, confounder_gene.as_ref(), &mut gopts);
    let grid: Vec<simulate::Scenario> = default_grid(&gopts)
        .into_iter()
        .map(|s| simulate::Scenario { region, ..s })
        .collect();
    info!(
        target: LOG_TARGET,
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
    let path = output::lra_simulate_path(a.out);
    output::write_sim(&path, &rows, a.plant_region)?;
    record(a.manifest, |o| {
        o.simulate = Some(crate::lr_activity::recorded(Path::new(&path)))
    })?;
    println!("{}", output::sim_summary(&rows, a.plant_region));
    println!("lra-simulate: wrote {} runs to {path}", rows.len());
    Ok(())
}
