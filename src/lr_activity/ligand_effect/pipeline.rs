//! The commands from input to tables: `pinto lra` and `pinto lra simulate`.

use super::baselines::{marginals, Baselines, CellTypeTest, TypeLabels};
use super::estimate::{estimate_pair_shared, PairData};
use super::fits::LigandShare;
use super::genes::GeneStore;
use super::memory::{ligand_memory, pair_memory, pair_workers, run_bounded};
use super::options::EffectOpts;
use super::overlap::NuisanceOpts;
use super::pairs::{pair_genes, reference_pairs, scarcest_subunit, signalling_genes, Pair};
use super::shard::{self, TableRecord};
use super::simulate::{self, default_grid, simulate, SimBase, SimShared};
use super::tissue::{identity_bins, Tissue, TissueInputs, TissueOpts, ALL_SECTIONS};
use super::{
    contacts, only, options, output, receivers, report, rows, timing, Internals, RunInputs,
    Settings, INTERNALS, LOG_TARGET,
};
use crate::lr_activity::args::RegionJackknife;
use crate::util::common::*;
use data_beans::utilities::name_matching::GeneIndex;
use std::path::Path;

mod single;

#[cfg(test)]
pub(super) use single::interleaved_ranks;
pub use single::{clear_single_outputs, run_simulate_one};

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
    /// Each kept pair's place among those `n_all`, and with `--shard` the
    /// names of all `n_all` (ligand, receptor), for the run key.
    index_in_all: Vec<usize>,
    all_names: Vec<(Box<str>, Box<str>)>,
    store: GeneStore,
    local: HashMap<usize, usize>,
    eopts: EffectOpts,
    reference_record: Option<std::path::PathBuf>,
    /// The genes the receivers' identity leaves out, by name, sorted.
    identity_excluded: Vec<String>,
    /// [`Settings::memory_budget`].
    memory_budget: Option<usize>,
    /// Every gene's name, by the index the pairs hold.
    gene_names: Vec<Box<str>>,
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

/// Which pairs to estimate: all of them, those `--only` names, or the
/// ligands `--shard` is dealt ([`shard::deal`]).
fn pairs_kept(
    a: &Settings<'_>,
    all: &[Pair],
    gene_names: &[Box<str>],
    index: &GeneIndex,
) -> anyhow::Result<Vec<bool>> {
    if let Some(s) = a.shard {
        let of = shard::deal(&ligand_groups(all), all.len(), s.count);
        let keep: Vec<bool> = of.iter().map(|&k| k == s.index).collect();
        info!(
            target: LOG_TARGET,
            "--shard {s}: {} of {} pairs",
            keep.iter().filter(|&&k| k).count(),
            all.len()
        );
        return Ok(keep);
    }
    if a.only.is_empty() {
        return Ok(vec![true; all.len()]);
    }
    let wanted = only::parse(a.only)?;
    let keep = only::select(&wanted, all, gene_names, index)?;
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
    Ok(keep)
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
fn effect_opts(seed: u64, region_jackknife: RegionJackknife, abf_prior_scale: f64) -> EffectOpts {
    let k = INTERNALS;
    EffectOpts {
        keep_arm_weights: false,
        threshold: None,
        min_overlap: k.min_overlap,
        abf_prior_scale,
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
    let keep = pairs_kept(a, &all, &gene_names, &index)?;
    let index_in_all: Vec<usize> = (0..n_all).filter(|&k| keep[k]).collect();
    let all_names = match a.shard {
        Some(_) => all
            .iter()
            .map(|p| (p.ligand.clone(), p.receptor.clone()))
            .collect(),
        None => Vec::new(),
    };
    let pairs: Vec<Pair> = all
        .into_iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(p))
        .collect();

    let (receivers, row_of_cell) = receivers::receiver_rows(inp.coords, inp.pairs);
    let needed = pair_genes(&pairs);
    let local: HashMap<usize, usize> = needed.iter().enumerate().map(|(k, &g)| (g, k)).collect();
    info!(
        target: LOG_TARGET,
        "Reading {} genes (ligands, receptors, targets) sparsely over {} receivers...",
        needed.len(),
        receivers.len()
    );
    let (store, log1p_depth) = GeneStore::read(inp.data, inp.axis, &needed, &row_of_cell)?;

    let (identity, identity_bin) = identity_bins(inp.data, inp.axis, &excluded, inp.cell_section)?;
    let tissue = build_tissue(inp, &identity, &identity_bin, &log1p_depth)?;
    debug_assert_eq!(tissue.cell, receivers);
    Ok(Prepared {
        tissue,
        pairs,
        n_all,
        index_in_all,
        all_names,
        store,
        local,
        eopts: effect_opts(a.seed, a.region_jackknife, a.abf_prior_scale),
        reference_record: Some(record),
        identity_excluded,
        memory_budget: a.memory_budget,
        gene_names,
    })
}

/// Record lra's outputs and the settings in the run's manifest, each path
/// absolute ([`crate::lr_activity::recorded`]): the fixed ones, with the
/// `--abf-prior-scale` the run used.
pub(super) fn record(
    manifest: &Path,
    abf_prior_scale: f64,
    set: impl FnOnce(&mut crate::util::metadata::LraFiles) + Send + 'static,
) -> anyhow::Result<()> {
    let settings = Internals {
        abf_prior_scale,
        ..INTERNALS
    };
    crate::util::metadata::PintoMetadata::backfill(manifest, move |m| {
        set(&mut m.outputs.lra);
        m.outputs.lra.settings = serde_json::to_value(settings).ok();
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
    pair_workers(per_ligand, tissue, &eopts.nuisance, prep.memory_budget).min(ligands)
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
    // Two weightings of the program, as given and with hub targets
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

/// Each pair's rows, in pair order. The pairs of one ligand run one after
/// another on one worker and share its base propensities
/// ([`LigandShare`]); ligands run side by side.
fn estimate_pairs(prep: &Prepared) -> Vec<Vec<rows::EffectRow>> {
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
    let rows: Vec<Vec<rows::EffectRow>> = by_pair.into_iter().map(|(_, r)| r).collect();
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

/// What the run's tables are recorded with ([`shard::TableRecord`]).
fn table_record(a: &Settings<'_>, prep: &Prepared, marginal: Option<&Marginal>) -> TableRecord {
    TableRecord {
        manifest: a.manifest.to_path_buf(),
        from_run: a.from_run,
        abf_prior_scale: a.abf_prior_scale,
        // The genes left out of the identity, so the viewer draws the
        // receivers as they were without the files that decided them.
        options: crate::util::metadata::LraOptions {
            identity_excluded: prep.identity_excluded.clone(),
            marginal_baselines: marginal.map(|m| m.labels.source.name().into()),
            ..a.options()
        },
        reference_record: prep.reference_record.clone(),
        n_all: prep.n_all,
    }
}

/// Each pair's cell-type rows (`--marginal-baselines`), in pair order.
fn celltype_rows(prep: &Prepared, m: &Marginal) -> Vec<Vec<output::CelltypeRow>> {
    prep.pairs
        .iter()
        .zip(&m.tests)
        .map(|(p, (_, t))| output::celltype_rows(&p.ligand, &p.receptor, t, &m.labels))
        .collect()
}

/// Record the tables in the manifest: a `--only` run (`only`, of `n_pairs`
/// pairs) as a subset beside the full table, any other as the full table
/// with its reference.
fn record_tables(
    record: &TableRecord,
    only: &[Box<str>],
    n_pairs: usize,
    path: &str,
    curve_path: &str,
    celltype: Option<&str>,
) -> anyhow::Result<()> {
    let recorded = |p: &str| crate::lr_activity::recorded(Path::new(p));
    let options = record.options.clone();
    let (table, curve) = (recorded(path), recorded(curve_path));
    let celltype = celltype.map(recorded);
    if only.is_empty() {
        let reference = record
            .reference_record
            .as_deref()
            .map(crate::lr_activity::recorded);
        self::record(&record.manifest, record.abf_prior_scale, |o| {
            o.table = Some(table);
            o.curve = Some(curve);
            o.celltype = celltype;
            o.record_options(options);
            o.reference = reference;
        })
    } else {
        let entry = crate::util::metadata::LraSubset {
            only: only::canonical(only),
            pairs: n_pairs,
            of: record.n_all,
            table,
            curve,
            celltype,
            options: Some(options),
        };
        self::record(&record.manifest, record.abf_prior_scale, |o| {
            o.record_subset(entry)
        })
    }
}

/// Write the effect and curve tables of `rows` (`n_pairs` pairs) under
/// `out` (with `celltype` the per-type-pair table) and record them; the
/// effect table's path and the q-values' scope.
fn write_tables(
    out: &str,
    only: &[Box<str>],
    record: &TableRecord,
    n_pairs: usize,
    rows: &[rows::EffectRow],
    celltype: Option<&[output::CelltypeRow]>,
) -> anyhow::Result<(String, String)> {
    let made = crate::lr_activity::lra_outputs(
        crate::lr_activity::args::LraMethod::ContactEffect,
        out,
        only,
    );
    let [path, curve_path] = [&made[0], &made[1]].map(|p| p.to_string_lossy().into_owned());
    let q_scope = if only.is_empty() {
        output::Q_SCOPE_ALL.into()
    } else {
        only::q_scope(n_pairs, record.n_all)
    };
    output::write_effects(&path, rows, &q_scope, celltype.is_some())?;
    output::write_curve(&curve_path, rows)?;
    let celltype_path = output::celltype_beside(&path);
    if let Some(table) = celltype {
        output::write_celltype(&celltype_path, table)?;
        info!(
            target: LOG_TARGET,
            "Wrote {} tested type pairs to {}",
            table.len(),
            crate::util::staging::final_str(&celltype_path)
        );
    }
    let celltype = celltype.map(|_| celltype_path.as_str());
    record_tables(record, only, n_pairs, &path, &curve_path, celltype)?;
    Ok((path, q_scope))
}

/// The lines a run ends with.
fn print_run_summary(
    manifest: &Path,
    only: &[Box<str>],
    rows: &[rows::EffectRow],
    path: &str,
    q_scope: &str,
) {
    let power_check = crate::util::metadata::PintoMetadata::read(manifest)
        .ok()
        .and_then(|m| m.outputs.lra.simulate);
    let path = crate::util::staging::final_str(path);
    crate::util::staging::say(output::console_summary(rows, &path, power_check.as_deref()));
    if !only.is_empty() {
        crate::util::staging::say(format!(
            "  {q_scope}; the full table, when there is one, is unchanged"
        ));
    }
}

/// `--shard`: the shard's pairs' rows, before the q-values, to
/// [`shard::path`]; no table, nothing recorded.
fn write_shard(
    a: &Settings<'_>,
    s: shard::Shard,
    prep: &Prepared,
    by_pair: Vec<Vec<rows::EffectRow>>,
    marginal: Option<&Marginal>,
) -> anyhow::Result<()> {
    let record = table_record(a, prep, marginal);
    let names = prep.all_names.iter().map(|(l, r)| (&**l, &**r));
    let run_key = shard::run_key(&record, names);
    let mut celltype = marginal.map(|m| celltype_rows(prep, m));
    let pairs = by_pair
        .into_iter()
        .zip(&prep.index_in_all)
        .enumerate()
        .map(|(k, (rows, &index))| shard::ShardPair {
            index,
            rows,
            celltype: celltype
                .as_mut()
                .map(|c| std::mem::take(&mut c[k]))
                .unwrap_or_default(),
        })
        .collect();
    let file = shard::ShardFile {
        format: shard::FORMAT,
        pinto_version: env!("CARGO_PKG_VERSION").into(),
        shard: s,
        run_key,
        record,
        pairs,
        celltype_source: marginal.map(|m| m.labels.source),
    };
    let path = shard::path(a.out, s);
    shard::write(&path, &file)?;
    crate::util::staging::say(format!(
        "Shard {s}: {} pairs to {}; `pinto lra merge` puts the {} shards together",
        prep.pairs.len(),
        crate::util::staging::final_str(&path),
        s.count
    ));
    Ok(())
}

pub fn run(inp: &RunInputs<'_>) -> anyhow::Result<()> {
    let a = &inp.settings;
    let prep = prepare(inp)?;
    let by_pair = estimate_pairs(&prep);
    let marginal = a.marginal.map(|n| marginal_baselines(&prep, n, a.seed));
    if let Some(s) = a.shard {
        let mut by_pair = by_pair;
        if let Some(m) = &marginal {
            for rows in &mut by_pair {
                attach_marginals(rows, &prep, m);
            }
        }
        return write_shard(a, s, &prep, by_pair, marginal.as_ref());
    }
    let mut rows: Vec<rows::EffectRow> = by_pair.into_iter().flatten().collect();
    report::report(&mut rows, &report::ReportOpts::default());
    if let Some(m) = &marginal {
        attach_marginals(&mut rows, &prep, m);
    }
    let record = table_record(a, &prep, marginal.as_ref());
    let celltype: Option<Vec<output::CelltypeRow>> = marginal
        .as_ref()
        .map(|m| celltype_rows(&prep, m).into_iter().flatten().collect());
    let (path, q_scope) = write_tables(
        a.out,
        a.only,
        &record,
        prep.pairs.len(),
        &rows,
        celltype.as_deref(),
    )?;
    print_run_summary(a.manifest, a.only, &rows, &path, &q_scope);
    Ok(())
}

/// `{out}.pinto.json`, when it does not exist yet, as the shard's manifest
/// at `shard_manifest` reads but for its prefix and time.
fn own_manifest(shard_manifest: &Path, out: &str) -> anyhow::Result<std::path::PathBuf> {
    use crate::util::metadata::{manifest_path, now_secs, PintoMetadata};
    let out = crate::util::staging::final_str(out);
    let path = std::path::PathBuf::from(manifest_path(&out));
    let shard = PintoMetadata::read(shard_manifest)
        .map_err(|e| anyhow::anyhow!("{}: {e}", shard_manifest.display()))?;
    PintoMetadata {
        timestamp: now_secs(),
        prefix: out,
        outputs: Default::default(),
        ..shard
    }
    .write_if_missing(&path)?;
    Ok(path)
}

/// `pinto lra merge`: every shard of a run at `shards`, its tables under
/// `out` as a run without `--shard` writes them, recorded in its manifest.
pub fn merge(shards: &[String], out: &str) -> anyhow::Result<()> {
    let shard::Merged {
        mut record,
        count,
        mut rows,
        celltype,
    } = shard::read_all(shards)?;
    if !record.from_run {
        // a run on data files: the merge's own manifest, as the shards' own
        record.manifest = own_manifest(&record.manifest, out)?;
    }
    info!(
        target: LOG_TARGET,
        "Merging {count} shards: {} pairs, {} rows",
        record.n_all,
        rows.len()
    );
    report::report(&mut rows, &report::ReportOpts::default());
    let celltype = celltype.map(|(_, rows)| rows);
    let (path, q_scope) =
        write_tables(out, &[], &record, record.n_all, &rows, celltype.as_deref())?;
    print_run_summary(&record.manifest, &[], &rows, &path, &q_scope);
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
/// projection with the other signalling and program genes, so it lies
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
    /// `--reference-data`, fitted ([`fit_reference_program_genes`]).
    reference_data: Option<&'a simulate::reference_data::ReferenceProgramGenes>,
}

/// `--reference-data` fitted over the program genes of the pairs `order`, each
/// identity bin matched to a group by those genes' bin means; `None` without
/// it.
fn fit_reference_program_genes(
    prep: &Prepared,
    a: &Settings<'_>,
    order: &[usize],
) -> anyhow::Result<Option<simulate::reference_data::ReferenceProgramGenes>> {
    let Some(path) = a.reference_data else {
        return Ok(None);
    };
    let mut genes: Vec<usize> = order
        .iter()
        .flat_map(|&k| prep.pairs[k].targets.iter().map(|(g, _, _)| *g))
        .collect();
    genes.sort_unstable();
    genes.dedup();
    let names: Vec<Box<str>> = genes.iter().map(|&g| prep.gene_names[g].clone()).collect();
    let nonzeros: Vec<&[(u32, f32)]> = genes
        .iter()
        .map(|g| prep.store.nonzeros(prep.local[g]))
        .collect();
    let bin_means = simulate::target_bin_means(&prep.tissue, &nonzeros);
    info!(target: LOG_TARGET, "Fitting the reference data {path}...");
    let r = simulate::reference_data::ReferenceProgramGenes::fit(
        path,
        a.reference_data_labels,
        &names,
        &bin_means,
        simulate::n_bins(&prep.tissue),
    )?;
    let mut bins_of_group = vec![0usize; r.cell_groups.len()];
    for &k in &r.cell_group_of_bin {
        bins_of_group[k] += 1;
    }
    let per_group: Vec<String> = r
        .cell_groups
        .iter()
        .zip(&bins_of_group)
        .filter(|(_, &n)| n > 0)
        .map(|((name, _), n)| format!("{name} {n}"))
        .collect();
    info!(
        target: LOG_TARGET,
        "Reference data: {} of {} program genes drawn with it; identity bins per cell group: {}",
        names.iter().filter(|n| r.gene_position(n).is_some()).count(),
        names.len(),
        per_group.join(", ")
    );
    Ok(Some(r))
}

/// What a benchmarked pair's runs draw on, read once: its ligand and receptor
/// counts, its competent receivers, and its program's weights and bin
/// means.
struct PairSim {
    ligand: Vec<f64>,
    receptor: Vec<f64>,
    competent: Vec<bool>,
    weights: Vec<f64>,
    bin_means: Vec<f64>,
    /// Each program gene's position in the reference data, under it.
    reference_position_of_target: Option<Vec<Option<usize>>>,
}

impl PairSim {
    fn new(
        prep: &Prepared,
        eopts: &EffectOpts,
        p: &Pair,
        reference_data: Option<&simulate::reference_data::ReferenceProgramGenes>,
    ) -> Self {
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
            reference_position_of_target: reference_data.map(|r| {
                p.targets
                    .iter()
                    .map(|(g, _, _)| r.gene_position(&prep.gene_names[*g]))
                    .collect()
            }),
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
            reference_position_of_target: self.reference_position_of_target.as_deref(),
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
        let pair = PairSim::new(prep, eopts, p, self.reference_data);
        let base = pair.base(p, self.confounder_gene);
        let shared = SimShared {
            producers: simulate::calibrate_producers(tissue, base.ligand_counts),
            gene_fields: self.gene_fields,
            field: None,
            reference_data: self.reference_data,
        };
        // Scenarios are independent: run as many at once as the memory
        // budget allows (each holds about a pair's worth of vectors).
        // and the receivers' copula factors under reference data
        let factors = self.reference_data.map_or(0, |r| r.copula_rank());
        let per_run = pair_memory(50 + factors, tissue, &eopts.nuisance);
        let concurrent = pair_workers(per_run, tissue, &eopts.nuisance, prep.memory_budget);
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
    let reference_data = fit_reference_program_genes(&prep, a, &order)?;
    let pb = Progress::new(grid.len() * order.len(), "benchmark runs");
    let benchmark = Benchmark {
        prep: &prep,
        eopts,
        grid: &grid,
        confounder_gene: confounder_counts,
        gene_fields: &gene_fields,
        seed: a.seed,
        pb: &pb,
        reference_data: reference_data.as_ref(),
    };
    let mut rows = Vec::new();
    for &k in &order {
        rows.extend(benchmark.pair_runs(k));
    }
    pb.finish();
    let path = output::lra_simulate_path(a.out);
    output::write_sim(&path, &rows, a.plant_region)?;
    // the reference data the program genes were drawn with, beside
    // the table
    // (an earlier run's is removed when this one used none)
    let side = format!("{}.reference-data.json", path.trim_end_matches(".parquet"));
    match &reference_data {
        Some(r) => std::fs::write(&side, serde_json::to_string_pretty(&r.provenance())? + "\n")?,
        None if Path::new(&side).exists() => std::fs::remove_file(&side)?,
        None => {}
    }
    let table = crate::lr_activity::recorded(Path::new(&path));
    record(a.manifest, a.abf_prior_scale, |o| o.simulate = Some(table))?;
    crate::util::staging::say(output::sim_summary(&rows, a.plant_region));
    crate::util::staging::say(format!(
        "lra-simulate: wrote {} runs to {}",
        rows.len(),
        crate::util::staging::final_str(&path)
    ));
    Ok(())
}
