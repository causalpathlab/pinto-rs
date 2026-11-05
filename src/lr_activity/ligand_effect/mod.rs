//! Effect of neighbour ligand on receivers (`pinto lra`, the default
//! `--method contact-effect`; `pinto lra-simulate` benchmarks it).
//!
//! Co-expression scores ask whether a ligand and its receptor are detected
//! together across contacts. This asks what happens to a receiver's
//! response programme when its contact neighbours carry the ligand, among
//! receivers comparable in everything else that could drive both: region,
//! section, cell identity, neighbour composition, depth and segmentation
//! bleed. The two answers disagree when co-location is confounded, and the
//! output puts them side by side.
//!
//! **Estimands** (per pair; pooled and per section). With `T` = contact
//! neighbours carry ligand above a threshold, `Y` = the receiver's signed
//! target programme, `C` = the receiver's identity expresses the receptor:
//!
//! - E2 `effect_competent`, **the headline**: overlap-weighted effect of
//!   `T` on `Y` in receptor-competent receivers;
//! - E3 `effect_incompetent`: the same in incompetent receivers (≈ 0 is a
//!   negative control);
//! - E1 `effect_contrast`: E2 − E3. Competence is read from identity bins,
//!   so competent and incompetent receivers are different cell types by
//!   construction and E1 conflates the receptor with cell type; it is kept
//!   as a column, not as the headline;
//! - E5 `ring_effect`: effect of ligand on distant non-touching cells given
//!   the real exposure (≈ 0 is a negative control for spatial confounding);
//! - E0 / E0′: the co-expression log odds ratio and the crude contrast,
//!   for comparison.
//!
//! **Inference across pairs** ([`estimate::report`]): Benjamini–Hochberg
//! q-values within the pooled rows and within each section, for E2
//! (two-sided), co-expression (one-sided, positive) and both controls, and
//! a per-ligand q (Simes over the ligand's pairs, BH over ligands); the
//! verdict ([`output::Verdict`]) reads q < `--fdr`. When the ligand marks a cell type, exposure
//! is nearly "has a neighbour of that type" and cannot be told from
//! neighbour identity: `exposure_eta2` measures how much of the exposure
//! the dominant neighbour's identity bin explains, and at
//! [`INTERNALS`]`.max_exposure_eta2` the pair is `celltype_inseparable`
//! rather than a finding. A significant E2 beside a failed control is
//! `control_failed`.
//!
//! "Effect" is a contrast of two stochastic interventions: each receiver's
//! dose drawn from its own plausible doses above versus below the
//! threshold. Identification rests on (1) a receiver depending on its
//! neighbours only through `T`, (2) no unmeasured confounding given the
//! confounders, (3) overlap, which the overlap weights make a matter of
//! target population rather than of variance, and (4) consistency in that
//! stochastic sense. The negative controls probe (2).
//!
//! Layout: [`targets`] reads response programmes, [`genes`] reads the
//! needed genes sparsely, [`exposure`] and [`folds`] and [`confounders`]
//! build [`tissue::Tissue`], [`overlap`] and [`regression`] estimate,
//! [`estimate`] assembles one pair, [`simulate`] benchmarks the estimator on
//! this tissue with a planted effect, and [`output`] writes the tables.

pub mod baselines;
pub mod confounders;
pub mod estimate;
pub mod exposure;
pub mod folds;
pub mod genes;
pub mod only;
pub mod output;
pub mod overlap;
pub mod pseudobulk;
pub mod receivers;
pub mod reference;
pub mod regression;
pub mod simulate;
pub mod targets;
pub mod tissue;

#[cfg(test)]
mod tests;

use crate::lr_activity::args::ReferenceSpecies;
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use data_beans::alg::random_projection::{binary_sort_columns, RandProjOps};
use data_beans::utilities::name_matching::GeneIndex;
use estimate::{estimate_pair_with, EffectOpts, PairData};
use genes::GeneStore;
use overlap::NuisanceOpts;
use simulate::{default_grid, simulate, SimBase, SimShared, Study};
use std::path::Path;
use targets::TargetSets;
use tissue::{Tissue, TissueInputs, TissueOpts};

/// Memory allowed for pairs estimated at once when the machine does not say
/// what it has, and the least allowed when it does.
const MIN_PAIR_MEMORY_BUDGET: usize = 2 << 30;

/// Memory allowed for pairs estimated at once: half of what is available now
/// ([`available_memory`]), the rest left to the tissue and the machine's
/// other work.
fn pair_memory_budget() -> usize {
    available_memory().map_or(MIN_PAIR_MEMORY_BUDGET, |a| {
        (a / 2).max(MIN_PAIR_MEMORY_BUDGET)
    })
}

/// Bytes this process could still allocate: the system's available memory,
/// or less when a cgroup (a batch job, a container) limits it. `None` where
/// the system does not say.
fn available_memory() -> Option<usize> {
    let read = |p: &str| std::fs::read_to_string(p).ok();
    let meminfo = read("/proc/meminfo")?;
    let system = parse_mem_available(&meminfo)?;
    // cgroup v2 names the limit `memory.max` ("max" when unlimited), v1
    // `memory.limit_in_bytes` (a huge number when unlimited).
    let left = |limit: &str, usage: &str| -> Option<usize> {
        let limit: usize = read(limit)?.trim().parse().ok()?;
        let usage: usize = read(usage)?.trim().parse().ok()?;
        Some(limit.saturating_sub(usage))
    };
    let group = left("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current").or_else(|| {
        left(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        )
    });
    Some(group.map_or(system, |g| g.min(system)))
}

/// `MemAvailable` of `/proc/meminfo`, in bytes.
fn parse_mem_available(meminfo: &str) -> Option<usize> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Memory one live pair (or benchmark scenario) holds on a tissue of `n`
/// receivers: `vectors` receiver-length vectors, a draw's confounders (at
/// most a row per receiver, in f32, with the unit's two depth columns), and
/// the propensity's rows ([`regression::logistic_ridge`]).
fn pair_memory(n: usize, vectors: usize, tissue: &Tissue, nuisance: &NuisanceOpts) -> usize {
    let p = tissue.design.p();
    n * 8 * vectors + n * (p + 2) * 4 + nuisance.max_propensity_rows * (p + 8) * 8
}

/// The single competence threshold `lra-simulate` plants on
/// ([`Internals::benchmark_competence`]).
const BENCHMARK_COMPETENCE: [f64; 1] = [0.1];

/// The shared progress bar over the long stages. Where no bar can draw (the
/// log of a background job, a pipe) each twentieth of the way is logged too.
struct Progress {
    bar: indicatif::ProgressBar,
    what: &'static str,
    total: usize,
    done: std::sync::atomic::AtomicUsize,
    logs: bool,
    start: std::time::Instant,
}

impl Progress {
    fn new(total: usize, what: &'static str) -> Self {
        use std::io::IsTerminal;
        Self {
            bar: new_progress_bar(total as u64).with_message(what),
            what,
            total,
            done: 0.into(),
            logs: !std::io::stderr().is_terminal(),
            start: std::time::Instant::now(),
        }
    }

    fn inc(&self) {
        self.bar.inc(1);
        let done = 1 + self.done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.logs && (done * 20 / self.total.max(1)) > ((done - 1) * 20 / self.total.max(1)) {
            info!(
                "{done}/{} {} ({:.0}s)",
                self.total,
                self.what,
                self.start.elapsed().as_secs_f64()
            );
        }
    }

    fn finish(&self) {
        self.bar.finish_and_clear();
    }
}

/// A propensity fit's rows, in bytes, that stay in a core's share of the
/// cache; past this, fits side by side compete for memory.
const FITS_IN_CACHE: usize = 4 << 20;

/// Pairs (or benchmark scenarios) to run at once, each holding `per_pair`
/// bytes: what the memory budget allows, and on a tissue whose fits outgrow
/// the cache a quarter of the threads, the rest serving each pair's parallel
/// steps ([`run_bounded`]). Fits slow each other down past that: more of them
/// at once take longer in all.
fn pair_workers(per_pair: usize, n: usize, tissue: &Tissue, nuisance: &NuisanceOpts) -> usize {
    let threads = rayon::current_num_threads();
    let rows = n.min(nuisance.max_propensity_rows);
    let fit_bytes = rows * (tissue.design.p() + 8) * 8;
    let by_cache = if fit_bytes < FITS_IN_CACHE {
        threads
    } else {
        (threads / 4).max(1)
    };
    (pair_memory_budget() / per_pair.max(1))
        .min(by_cache)
        .clamp(1, threads)
}

// * Settings

/// The fixed choices of the contact-effect analysis. None is a flag: each
/// is recorded in the manifest (`outputs.lra.settings`) beside the table it
/// made.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Internals {
    /// Shares of a receiver identity bin detecting the receptor for its
    /// receivers to count as competent; the estimate averages over them.
    pub competence_grid: [f64; 3],
    /// The single competence threshold `lra-simulate` plants on.
    pub benchmark_competence: f64,
    /// Receivers per random pseudobulk unit (1 = single cells).
    pub pseudobulk: usize,
    /// Random pseudobulk partitions per pair, combined by Rubin's rules:
    /// chosen automatically between 3 and 20 (until the Monte Carlo error
    /// is small beside the sampling error), and 1 when no seed can change
    /// the draw (every stratum under two units' worth of receivers, or
    /// every fit skipped for a thin arm); the curve's thresholds by the
    /// same rule on their own partitions.
    pub draws: std::borrow::Cow<'static, str>,
    /// Fewest exposed and fewest unexposed receivers of a reported subgroup.
    pub min_arm: usize,
    /// Smallest effective sample size of a subgroup's overlap weights.
    pub min_ess: f64,
    /// Smallest mean overlap weight `e(1 − e)` of a subgroup.
    pub min_overlap: f64,
    /// Share of exposure variance explained by neighbour identity at which
    /// a pair is `celltype_inseparable`.
    pub max_exposure_eta2: f64,
    /// Dimension of the receiver identity projection, and its bins (2^bits).
    pub identity_dim: usize,
    pub identity_bits: usize,
    /// Fewest measured genes in a pair's response programme.
    pub min_targets: usize,
    /// Negative-control ring, inner and outer, in median contact lengths.
    pub ring: (f64, f64),
    /// Length scales of the regional confounder smooth, in contact lengths.
    pub scales: [f64; 2],
    /// Tile width for cross-fitting folds and variance clusters.
    pub tile: f64,
    pub folds: usize,
}

pub const INTERNALS: Internals = Internals {
    competence_grid: estimate::COMPETENCE_GRID,
    benchmark_competence: BENCHMARK_COMPETENCE[0],
    pseudobulk: pseudobulk::UNIT_SIZE,
    draws: std::borrow::Cow::Borrowed(
        "automatic, 3 to 20; 1 when the pseudobulk partition cannot change",
    ),
    min_arm: 20,
    min_ess: 30.0,
    min_overlap: 0.01,
    max_exposure_eta2: 0.8,
    identity_dim: 10,
    identity_bits: 6,
    min_targets: 5,
    ring: (15.0, 30.0),
    scales: [8.0, 32.0],
    tile: 25.0,
    folds: 5,
};

/// What the user chose, and where the run records itself.
pub struct Settings<'a> {
    pub out: &'a str,
    /// The manifest the outputs are recorded in.
    pub manifest: &'a Path,
    pub lr_pairs: Option<&'a str>,
    pub targets: &'a [Box<str>],
    pub species: ReferenceSpecies,
    pub fdr: f64,
    pub seed: u64,
    pub block_size: Option<usize>,
    /// `--only`: the pairs to estimate; empty for every pair.
    pub only: &'a [Box<str>],
}

impl Settings<'_> {
    /// What decides the pairs and the receivers' identity, as the manifest
    /// records it beside the table ([`crate::util::metadata::LraOptions`]).
    pub fn options(&self) -> crate::util::metadata::LraOptions {
        let abs = |p: &str| crate::lr_activity::recorded(Path::new(p));
        crate::util::metadata::LraOptions {
            lr_pairs: self.lr_pairs.map(abs),
            targets: self.targets.iter().map(|t| abs(t)).collect(),
            species: self.species.name(),
            seed: self.seed,
            fdr: self.fdr,
            identity_excluded: Vec::new(),
        }
    }

    fn choices(&self) -> PairChoices<'_> {
        PairChoices {
            species: self.species,
            lr_pairs: self.lr_pairs,
            targets: self.targets,
        }
    }
}

/// What decides which pairs there are: the reference's species and the
/// user's pair and programme files on top of it.
pub struct PairChoices<'a> {
    pub species: ReferenceSpecies,
    pub lr_pairs: Option<&'a str>,
    pub targets: &'a [Box<str>],
}

/// What the lra front end hands over.
pub struct RunInputs<'a> {
    pub settings: Settings<'a>,
    pub data: &'a SparseIoVec,
    pub axis: &'a FeatureAxis,
    /// Spatial contacts as data columns, and each cell's section label.
    pub pairs: &'a [(usize, usize)],
    pub cell_section: &'a [Box<str>],
    pub coords: &'a [Option<(f32, f32)>],
    /// Cell-type code per data column and the types' names, when known.
    pub cell_type: Option<(&'a [u32], &'a [Box<str>])>,
}

/// `{out}.lra.parquet`, the contact-effect table.
pub fn lra_path(out: &str) -> String {
    format!("{out}.lra.parquet")
}

/// `{out}.lra.curve.parquet`, each pair's threshold-response curve.
pub fn lra_curve_path(out: &str) -> String {
    curve_beside(&lra_path(out))
}

/// The curves written beside the contact-effect table `table`, the full
/// one or a subset's: `{table without .parquet}.curve.parquet`.
pub fn curve_beside(table: &str) -> String {
    format!(
        "{}.curve.parquet",
        table.strip_suffix(".parquet").unwrap_or(table)
    )
}

/// `{out}.lra-simulate.parquet`, the power and calibration check.
pub fn lra_simulate_path(out: &str) -> String {
    format!("{out}.lra-simulate.parquet")
}

/// A resolved pair: names, gene-axis indices (every subunit of a complex
/// ligand or receptor required), and its programme as `(gene index, weight)`.
struct Pair {
    ligand: Box<str>,
    receptor: Box<str>,
    ligand_genes: Vec<usize>,
    receptor_genes: Vec<usize>,
    /// `(gene, weight as given, specificity)`.
    targets: Vec<(usize, f64, f64)>,
    n_removed: usize,
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
    /// Genes of the decoy pool ([`estimate::DecoyPool`]), in `local`.
    decoy_genes: Vec<usize>,
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

/// Per cell, the count of a complex's scarcest subunit.
fn scarcest_subunit(cols: &[Vec<f64>]) -> Vec<f64> {
    let n = cols.first().map_or(0, Vec::len);
    (0..n)
        .map(|i| cols.iter().map(|c| c[i]).fold(f64::INFINITY, f64::min))
        .collect()
}

// * Reference

/// The reference's pair and programme files for this data, built into the
/// cache on first use. The first build prints where it came from, how big
/// it is, where it lives and how to move it or stay offline.
fn reference(
    species: ReferenceSpecies,
    gene_names: &[Box<str>],
) -> anyhow::Result<reference::ReferenceFiles> {
    let species = match species {
        ReferenceSpecies::Human => reference::Species::Human,
        ReferenceSpecies::Mouse => reference::Species::Mouse,
        ReferenceSpecies::Auto => species_of(gene_names),
    };
    let refresh =
        std::env::var_os("PINTO_REFERENCE_REFRESH").is_some_and(|v| !v.is_empty() && v != "0");
    let root = reference::cache_root(None)?;
    let downloads = root.join("downloads");
    let first = refresh || !downloads.is_dir();
    let files = reference::ensure(
        species,
        &reference::ReferenceOpts {
            dir: None,
            refresh,
            // Built locally for this user, so every source counts:
            // licences limit redistribution, not use.
            licence: reference::LicenceMode::All,
        },
    )?;
    let size = dir_size(&root);
    if first {
        let names: Vec<&str> = reference::sources::downloads()
            .iter()
            .map(|s| s.name)
            .collect();
        eprintln!(
            "Reference ({species:?}) built from LIANA+ consensus pairs, OmniPath signalling \
             and intercell, CollecTRI regulons and HCOP orthologs:\n  {}\n  \
             cache: {} ({}), reused by later runs\n  \
             offline machines: copy that folder and set PINTO_OFFLINE=1; \
             to keep it elsewhere set PINTO_REFERENCE_DIR=<dir>",
            names.join(", "),
            root.display(),
            human_bytes(size)
        );
    } else {
        info!(
            "Reference ({species:?}): {} ({})",
            root.display(),
            human_bytes(size)
        );
    }
    Ok(files)
}

fn dir_size(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    rd.filter_map(|e| e.ok())
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            _ => e.metadata().map(|m| m.len()).unwrap_or(0),
        })
        .sum()
}

pub fn human_bytes(b: u64) -> String {
    let mb = b as f64 / (1u64 << 20) as f64;
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

// * Preparation

/// The pairs this data can estimate: the reference's for its species,
/// then `--lr-pairs` and `--targets` on top, resolved against the data's
/// genes ([`resolve_pairs`]); with the reference's record.
fn reference_pairs(
    c: &PairChoices<'_>,
    gene_names: &[Box<str>],
    index: &GeneIndex,
) -> anyhow::Result<(Vec<Pair>, std::path::PathBuf)> {
    // The reference first, then the user's files on top of it.
    let files = reference(c.species, gene_names)?;
    let mut sets = TargetSets::default();
    sets.add_file(&files.receptor_targets.to_string_lossy())?;
    let mut pair_files = vec![files.lr_pairs];
    if let Some(f) = c.lr_pairs {
        pair_files.push(f.into());
    }
    for f in c.targets {
        sets.add_file(f)?;
    }
    let mut raw = Vec::new();
    for f in &pair_files {
        raw.extend(crate::lr_activity::io::read_lr_pairs(&f.to_string_lossy())?);
    }
    sets.settle_gmt_names(&raw);
    let specificity = sets.specificity();
    let pairs = resolve_pairs(raw, gene_names, index, &sets, &specificity);
    anyhow::ensure!(
        !pairs.is_empty(),
        "no ligand-receptor pair has a usable response programme"
    );
    Ok((pairs, files.record))
}

/// Every gene `pairs` read (ligands, receptors, targets), sorted.
fn pair_genes(pairs: &[Pair]) -> Vec<usize> {
    let mut needed: Vec<usize> = Vec::new();
    for p in pairs {
        needed.extend(&p.ligand_genes);
        needed.extend(&p.receptor_genes);
        needed.extend(p.targets.iter().map(|(g, _, _)| *g));
    }
    needed.sort_unstable();
    needed.dedup();
    needed
}

/// The genes the receivers' identity leaves out, for every pair of `all`
/// whichever are estimated, so a subset (`--only`) sees the receivers the
/// full analysis sees; and the decoy pool ([`estimate::DecoyPool`]), genes
/// outside every pair and programme, read with them and left out like
/// them.
fn signalling_genes(all: &[Pair], n_genes: usize, seed: u64) -> (HashSet<usize>, Vec<usize>) {
    let needed = pair_genes(all);
    let decoys = estimate::decoy_pool_genes(n_genes, &needed, seed);
    let excluded = needed.iter().chain(&decoys).copied().collect();
    (excluded, decoys)
}

/// The receivers' identity projection and its bins (`identity_bits` of
/// it), every row of an `excluded` gene weighted out, so a receiver's
/// identity is not partly its own outcome.
fn identity_bins(
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
    let (excluded, decoy_genes) = signalling_genes(&all, gene_names.len(), a.seed);
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
    // The genes estimated pairs read, and the decoys.
    let mut needed = pair_genes(&pairs);
    needed.extend(&decoy_genes);
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
        pseudobulk: k.pseudobulk,
        draws: estimate::AUTO_DRAWS,
        seed: a.seed,
        competences: &estimate::COMPETENCE_GRID,
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
        decoy_genes,
        identity_excluded,
    })
}

/// The contact-effect table's mean pseudobulk unit size, as [`record`]
/// writes it to the manifest.
enum UnitSize {
    /// `lra`'s own: the mean of its partitions, `None` when it drew none
    /// (single cells), which removes the key.
    Own(Option<f64>),
    /// Keep what `lra` recorded: `lra-simulate`'s units are not the
    /// table's.
    Kept,
}

/// Record lra's outputs and the fixed settings in the run's manifest, each
/// path absolute ([`crate::lr_activity::recorded`]), with the mean unit
/// size `unit_size` says.
fn record(
    manifest: &Path,
    unit_size: UnitSize,
    set: impl FnOnce(&mut crate::util::metadata::LraFiles),
) -> anyhow::Result<()> {
    const KEY: &str = "pseudobulk_mean_unit_size";
    crate::util::metadata::PintoMetadata::backfill(manifest, |m| {
        let size = match unit_size {
            UnitSize::Own(s) => s.map(|s| serde_json::json!((s * 100.0).round() / 100.0)),
            UnitSize::Kept => m
                .outputs
                .lra
                .settings
                .as_ref()
                .and_then(|v| v.get(KEY))
                .cloned(),
        };
        let mut settings = serde_json::to_value(INTERNALS).ok();
        if let (Some(serde_json::Value::Object(o)), Some(v)) = (&mut settings, size) {
            o.insert(KEY.into(), v);
        }
        set(&mut m.outputs.lra);
        m.outputs.lra.settings = settings;
    });
    Ok(())
}

/// Log the run's achieved mean pseudobulk unit size (recorded in the
/// manifest as `pseudobulk_mean_unit_size`, [`pseudobulk::UnitTally`]).
fn log_unit_size(units: pseudobulk::UnitTally) {
    if let Some(size) = units.mean() {
        info!(
            "Pseudobulk units hold {size:.1} receivers on average (target {})",
            pseudobulk::UNIT_SIZE
        );
    }
}

// * Contact effect

pub fn run(inp: &RunInputs<'_>) -> anyhow::Result<()> {
    let a = &inp.settings;
    let prep = prepare(inp)?;
    let (tissue, store, local, eopts) = (&prep.tissue, &prep.store, &prep.local, &prep.eopts);
    let n = tissue.n();
    let pool = estimate::DecoyPool::new(
        &tissue.size_factor,
        prep.decoy_genes
            .iter()
            .map(|g| store.nonzeros(local[g]))
            .collect(),
    );

    let pb = Progress::new(prep.pairs.len(), "ligand-receptor pairs");
    // Each live pair holds a few dozen receiver-length vectors and its
    // propensity design, which bounds how many run at once ([`pair_workers`]).
    // Beside the pair's own vectors, a draw's fits ([`estimate::DrawFits`])
    // hold per competence threshold an effect modifier and a pair column,
    // and own ligand, niche, the exposure column and the main, ring and
    // three step propensities.
    let nc = eopts.competences.len();
    let draw_fits = 2 * nc + 2 + 1 + 1 + 1 + 3;
    let per_pair = pair_memory(n, 32 + draw_fits, tissue, &eopts.nuisance);
    let concurrent = pair_workers(per_pair, n, tissue, &eopts.nuisance);
    if concurrent > 1 {
        info!("Estimating up to {concurrent} pairs at a time");
    }
    let one = |p: &Pair| -> Vec<(Vec<estimate::EffectRow>, pseudobulk::UnitTally)> {
        let lig = prep.ligand_counts(p);
        let rec = prep.receptor_counts(p);
        // Two weightings of the programme, as given and with hub targets
        // down-weighted by specificity, averaged like pseudobulk draws: the
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
        let out = match estimate_pair_with(tissue, &data, &pool, eopts) {
            Ok(r) => vec![r],
            Err(e) => {
                warn!("{}-{}: {e}", p.ligand, p.receptor);
                Vec::new()
            }
        };
        pb.inc();
        out
    };
    let t_pairs = std::time::Instant::now();
    let mut units = pseudobulk::UnitTally::default();
    let mut rows: Vec<estimate::EffectRow> = Vec::new();
    for (r, u) in run_bounded(&prep.pairs, concurrent, one) {
        rows.extend(r);
        units.merge(u);
    }
    pb.finish();
    log_unit_size(units);
    info!(
        "Estimated {} pairs in {:.0}s ({concurrent} at a time). Thread time: {}",
        prep.pairs.len(),
        t_pairs.elapsed().as_secs_f64(),
        estimate::timing::summary()
    );
    let ropts = estimate::ReportOpts {
        fdr: a.fdr,
        max_exposure_eta2: INTERNALS.max_exposure_eta2,
        min_arm: INTERNALS.min_arm,
        units: eopts.pseudobulk > 1,
    };
    estimate::report(&mut rows, &ropts);
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
        // The full table's unit size stays the manifest's.
        record(a.manifest, UnitSize::Kept, |o| o.record_subset(entry))?;
    } else {
        record(a.manifest, UnitSize::Own(units.mean()), |o| {
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

/// `one` over every item, at most `workers` at a time, in item order. Each
/// worker takes the next item as soon as it is free, so one slow item holds
/// up only its own worker (no barrier between batches). The threads are
/// shared out: each worker's item runs its parallel steps on a rayon pool of
/// its own, so no item waits inside another item's work, which a shared pool
/// would have it steal and finish first.
fn run_bounded<T: Sync, R: Send>(
    items: &[T],
    workers: usize,
    one: impl Fn(&T) -> Vec<R> + Sync,
) -> Vec<R> {
    let workers = workers.min(items.len());
    if workers <= 1 {
        return items.iter().flat_map(&one).collect();
    }
    let threads = (rayon::current_num_threads() / workers).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut done: Vec<(usize, Vec<R>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let work = || {
                        let mut out = Vec::new();
                        loop {
                            let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(item) = items.get(k) else { break };
                            out.push((k, one(item)));
                        }
                        out
                    };
                    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                        Ok(pool) => pool.install(work),
                        Err(_) => work(),
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("pair worker panicked"))
            .collect()
    });
    done.sort_unstable_by_key(|(k, _)| *k);
    done.into_iter().flat_map(|(_, r)| r).collect()
}

// * Power and calibration check

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
    // The most abundant distinct ligands: a benchmark on a ligand seen in a
    // handful of cells would measure sparsity, not the estimator, and three
    // receptors of one ligand would benchmark one ligand three times.
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    let abundance = |p: &Pair| {
        p.ligand_genes
            .iter()
            .map(|g| store.total(local[g]))
            .fold(f64::INFINITY, f64::min)
    };
    order.sort_by(|&x, &y| abundance(&pairs[y]).total_cmp(&abundance(&pairs[x])));
    let mut seen_ligand: HashSet<&[usize]> = HashSet::default();
    order.retain(|&k| seen_ligand.insert(&pairs[k].ligand_genes));
    order.truncate(study.pairs);
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
        // budget allows (each holds about a pair's worth of vectors, and
        // under pseudobulk three copies of the programme's non-zeros).
        let nnz: usize = genes.iter().map(|g| g.len()).sum();
        let per_run = pair_memory(n, 48, tissue, &eopts.nuisance)
            + if eopts.pseudobulk > 1 { nnz * 8 * 3 } else { 0 };
        let concurrent = pair_workers(per_run, n, tissue, &eopts.nuisance);
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
    let mut units = pseudobulk::UnitTally::default();
    for r in &rows {
        units.merge(r.units);
    }
    log_unit_size(units);
    let path = lra_simulate_path(a.out);
    output::write_sim(&path, &rows)?;
    record(a.manifest, UnitSize::Kept, |o| {
        o.simulate = Some(crate::lr_activity::recorded(Path::new(&path)))
    })?;
    println!("{}", output::sim_summary(&rows));
    println!("lra-simulate: wrote {} runs to {path}", rows.len());
    Ok(())
}

/// Separators of a complex's subunits in a ligand or receptor name
/// (`ITGA4_ITGB1`, `ITGA4+ITGB1`, `ITGA4:ITGB1`).
const SUBUNIT_SEPARATORS: [char; 3] = ['_', '+', ':'];

/// Gene-axis rows of a ligand or receptor name. A name that is itself a gene
/// of the panel is that gene; any other name with a subunit separator is a
/// complex, one row per subunit, and `None` when any subunit is missing.
/// The split comes first: the gene index, asked for `ITGA4_ITGB1`, would
/// match its trailing symbol `ITGB1` and lose the complex.
fn subunit_genes(name: &str, gene_names: &[Box<str>], index: &GeneIndex) -> Option<Vec<usize>> {
    let is_complex = name.contains(SUBUNIT_SEPARATORS) && !is_panel_gene(name, gene_names, index);
    if !is_complex {
        return index.match_gene(name).map(|g| vec![g]);
    }
    let mut genes: Vec<usize> = name
        .split(SUBUNIT_SEPARATORS)
        .filter(|s| !s.is_empty())
        .map(|s| index.match_gene(s))
        .collect::<Option<_>>()?;
    genes.sort_unstable();
    genes.dedup();
    (!genes.is_empty()).then_some(genes)
}

/// Whether `name` is a whole gene row of the panel (case aside), as written
/// or after an id prefix (`ENSG…_NAME`), rather than a part of one.
fn is_panel_gene(name: &str, gene_names: &[Box<str>], index: &GeneIndex) -> bool {
    let Some(g) = index.match_gene(name) else {
        return false;
    };
    let q = name.to_lowercase();
    let row = gene_names[g].to_lowercase();
    let core = row.split('/').next().unwrap_or(&row);
    core == q || core.ends_with(&format!("_{q}"))
}

/// Resolve each listed pair's genes and programme. Pairs whose ligand or
/// receptor, or any subunit of a complex one, is missing, or whose
/// programme keeps fewer than `INTERNALS.min_targets` measured genes, are
/// dropped with a count. Pairs that resolve to the same genes as an earlier
/// one (spelling or case aside) are kept once.
fn resolve_pairs(
    raw: Vec<(Box<str>, Box<str>)>,
    gene_names: &[Box<str>],
    index: &GeneIndex,
    sets: &TargetSets,
    specificity: &HashMap<Box<str>, f32>,
) -> Vec<Pair> {
    let mut receptors_of: HashMap<Box<str>, Vec<Box<str>>> = HashMap::default();
    for (l, r) in &raw {
        receptors_of
            .entry(l.to_ascii_uppercase().into())
            .or_default()
            .push(r.clone());
    }
    let (mut no_gene, mut no_set, mut thin) = (0usize, 0usize, 0usize);
    let mut seen: HashSet<(Vec<usize>, Vec<usize>)> = HashSet::default();
    let mut out = Vec::new();
    for (l, r) in raw {
        let (Some(lg), Some(rg)) = (
            subunit_genes(&l, gene_names, index),
            subunit_genes(&r, gene_names, index),
        ) else {
            no_gene += 1;
            continue;
        };
        if !seen.insert((lg.clone(), rg.clone())) {
            continue;
        }
        let others: Vec<&str> = receptors_of
            .get(&*l.to_ascii_uppercase())
            .map(|v| v.iter().map(|s| s.as_ref()).collect())
            .unwrap_or_default();
        // The programme is purged of every receptor of the ligand and of
        // every subunit of the ligand and those receptors.
        let subunits: Vec<&str> = std::iter::once(l.as_ref())
            .chain(others.iter().copied())
            .flat_map(|x| x.split(SUBUNIT_SEPARATORS))
            .filter(|s| !s.is_empty())
            .collect();
        let banned: Vec<&str> = others.iter().copied().chain(subunits).collect();
        let Some((set, removed)) = sets.for_pair(&l, &r, &banned) else {
            no_set += 1;
            continue;
        };
        let signalling: HashSet<usize> = lg.iter().chain(&rg).copied().collect();
        let targets: Vec<(usize, f64, f64)> = set
            .iter()
            .filter_map(|(g, w)| {
                let s = specificity
                    .get(&*g.to_ascii_uppercase())
                    .copied()
                    .unwrap_or(0.0) as f64;
                index.match_gene(g).map(|gi| (gi, *w as f64, s))
            })
            .filter(|(gi, _, _)| !signalling.contains(gi))
            .collect();
        if targets.len() < INTERNALS.min_targets {
            thin += 1;
            continue;
        }
        out.push(Pair {
            ligand: l,
            receptor: r,
            ligand_genes: lg,
            receptor_genes: rg,
            targets,
            n_removed: removed,
        });
    }
    if no_gene + no_set + thin > 0 {
        warn!(
            "Contact effect: kept {} pairs; dropped {no_gene} with a ligand or receptor (or a \
             subunit of one) not in the data, {no_set} with no response programme, {thin} \
             whose programme has fewer than {} measured genes",
            out.len(),
            INTERNALS.min_targets
        );
    }
    out
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

/// Human or mouse from the gene names' case: human symbols are upper case
/// (`TGFB1`), mouse symbols capitalised (`Tgfb1`). Ensembl ids and names
/// without letters vote for neither.
fn species_of(gene_names: &[Box<str>]) -> reference::Species {
    let (mut upper, mut title) = (0usize, 0usize);
    for g in gene_names {
        let sym = g.rsplit(['_', ':']).next().unwrap_or(g);
        if sym.starts_with("ENS") {
            continue;
        }
        let letters: Vec<char> = sym.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        if letters.len() < 2 {
            continue;
        }
        if letters.iter().all(|c| c.is_ascii_uppercase()) {
            upper += 1;
        } else if letters[0].is_ascii_uppercase()
            && letters[1..].iter().any(|c| c.is_ascii_lowercase())
        {
            title += 1;
        }
    }
    let species = if title > upper {
        reference::Species::Mouse
    } else {
        reference::Species::Human
    };
    info!("Species from gene names: {species:?} ({upper} upper-case, {title} capitalised symbols)");
    species
}
