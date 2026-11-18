//! `pinto apply residual`: the counts a fitted run leaves unexplained, as new data
//! files that a fit reads as it reads data.
//!
//! Stage 1 is any run that published what a residual reads, whatever
//! command wrote it (a fit or a spatial baseline). Its manifest records
//! ([`Plan::new`] checks each, and names the one that is missing):
//!
//! - `data_files`, the raw counts it was fitted on (or `--data-files`);
//! - `outputs.propensity`, its cell × community table `π_ik`: a soft
//!   posterior, or one-hot rows for a method that only labels its cells;
//! - `outputs.cells`, its cells table: each cell's coordinates and, when
//!   the run had more than one batch, its `batch`;
//! - `outputs.coord_columns`, the names of those coordinates.
//!
//! Nothing else of the run is read: no profile, loading or batch effect of
//! its own, so no command is a special case. What is conditioned on is its
//! communities, through `π_ik`. From it, raw-count community
//! profiles are refitted on the run's data, read as a fit reads it (the rows
//! every file shares, cells named as the run named them),
//!
//! `p_k(g) ∝ (1 + Σ_i π_ik y_gi) / (1 + Σ_i π_ik)`, normalized over `g`,
//!
//! lc's own feature-community statistic and Gamma(1, 1) posterior mean
//! (`link_community::profiles::fit_feature_community_param`) with unit
//! feature weights, the `1 + Σ_i π_ik` cancelling in the normalization: the NB
//! Fisher-information weights `{run}.feature_community` carries shrink
//! high-mean features, so that table is not a composition. Each cell then
//! gets its proportions `θ_i` by the per-cell EM of `pinto apply impute` and
//! held-out scoring ([`em_proportions`]) and the expectation
//! `μ_gi = n_i Σ_k θ_ik p_k(g)`, `n_i` the cell's total, the mean of that
//! multinomial mixture. No batch fold `δ` from the stage-1 run enters
//! `p` or `μ`: `δ` is the donor (batch) effect, the very thing the residual
//! is meant to keep.
//!
//! By default (`--adapt batch`) each community's profile is then
//! re-estimated per batch as the run knew it (the `batch` column of its
//! cells table, written when the run had more than one: its
//! `--batch-files`, else its data files), here and from the
//! stage-1 communities only:
//!
//! `p_{k,b}(g) = (α_k p_k(g) + S_{gk,b}) / (α_k + T_{kb})`,
//! `S_{gk,b} = Σ_{i∈b} π_ik y_gi`, `T_{kb} = Σ_g S_{gk,b}`,
//!
//! a Dirichlet prior centred on the pooled profile with the mass of
//! `--adapt-cells` cells of community k (the hierarchical Dirichlet
//! smoothing of MacKay and Peto 1995, Natural Language Engineering 1:289-308,
//! doi:10.1017/S1351324900000218), as Harmony corrects per cluster and per
//! batch (Korsunsky et al. 2019, Nat Methods 16:1289-1296,
//! doi:10.1038/s41592-019-0619-0). A shift that moves a whole community in a
//! batch (donor, donor-by-type or section nuisance) is then absorbed, while
//! a local patch, a minority of the community's weight there, stays in the
//! residual. The identifying assumption: donor-specific biology is local, and
//! a whole community shifted in one donor is read as nuisance.
//! `--adapt none` keeps the pooled profiles.
//!
//! What is left of the counts once `μ` is accounted for is taken one of two
//! ways (`--residual`), both from the one `μ` above ([`mu_at`]):
//!
//! **divide** (alias multiplicative; the default): stage 2 fits the raw
//! counts `y` with `μ` as a multiplicative offset, so it models `y / μ`.
//! Nothing per entry is written, only the low-rank pieces `μ` is rebuilt
//! from wherever a fit reads a count:
//!
//! ```text
//! μ_gi   = n_i Σ_k θ_ik p_{k,b(i)}(g)
//! p̄_g    = Σ_i μ_gi / Σ_i n_i
//! d_gi   = μ_gi / (n_i p̄_g) = Σ_k θ_ik p_{k,b(i)}(g) / p̄_g
//! φ_g    = max(0, Σ_i [(y_gi − μ_gi)² − μ_gi]) / Σ_i μ_gi²
//! σ²_gb  = (1 / N_b) Σ_{i∈b} (μ_gi + φ_g μ_gi²)
//! r_gi   = (y_gi − μ_gi) / σ_{g,b(i)}
//! ```
//!
//! `d` is the divisor dsvd, cage and cage-cell divide a count by where they
//! would divide by a batch effect, as numeric's adjust-by-division does: the
//! self-normalizing Poisson ratio `y / (λ_i d)`, `λ_i = Σ y / Σ d` over the
//! cell's nonzeros, so each cell's overall scale is absorbed and `d` sets
//! only the shape; lc scores its edge profiles against `μ` itself
//! (rates `D_kg / M_kg`, about 1 where stage 1 explains the counts), and
//! every stage-2 fit writes its feature × community dictionary as
//! `λ̂_gk = Σ_i π_ik y_gi / Σ_i π_ik d_gi`: a mean count per cell, as
//! without an offset, with each count over its `d` (so not on lc's scale);
//! `r` is the Pearson residual `--feature-inclusion moran` reads, with one
//! scale per row and batch, so its zeros have closed forms. `φ` is the
//! negative-binomial dispersion by the method of moments,
//! `E[(y − μ)²] = μ + φ μ²` pooled over the cells: the sums over every cell
//! split into the nonzeros' `Σ (y² − 2yμ)` and the zeros' closed forms,
//! `Σ_{i∈b} μ_gi = p_b(g)ᵀ Σ_{i∈b} n_i θ_i` and
//! `Σ_{i∈b} μ_gi² = p_b(g)ᵀ (Σ_{i∈b} n_i² θ_i θ_iᵀ) p_b(g)`, `K × K` per
//! batch, so nothing dense over cells × rows is formed. The tables
//! (`crate::util::stage1_divisor`): `{out}/stage1.cells.parquet` (per cell
//! `n_i`, its batch and `θ_i`), `{out}/stage1.profiles.parquet` (per row
//! `p_{k,b}`) and `{out}/stage1.features.parquet` (per row `p̄`, `φ` and
//! `σ²_b`). A stage-2 fit reads the stage-1 run's own data files and takes
//! the folder as `--offset {out}`.
//!
//! **subtract** (alias additive): per data file `{out}/{stem}.zarr.zip`, the
//! excess counts `round(max(y_gi − μ_gi, 0))` (or the floor, `--excess`),
//! which the count models fit as they fit data.
//!
//! Either way the excess files, if any, are on the rows every file shares
//! (as an exact-name fit reads them; with `--feature-name-mode` as the
//! fit's) and each file's own cells, and `{out}/{stem}.coords.parquet`
//! holds the run's first two coordinates for those cells. `{out}/residual.json`
//! records the operation (`divide` or `subtract`), per file the counts in
//! (and under subtract the excess out), how `μ` was made (`adapt`,
//! `adapt_cells`, `formula`), and the stage-1 run and raw data files: under
//! subtract the block `pinto lra` reads to take a fit on the excess files on
//! the raw counts; under divide the stage-1 run lra takes cell types from.

use crate::cell_activity_graph_embedding::args::FeatureNameMode;
use crate::util::common::*;
use crate::util::lupin_rounds::same_file;
use crate::util::metadata::Run;
use crate::util::profile_match::em_proportions;
use clap::Args;
use data_beans::aux::feature_names::FeatureNameKind;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use std::path::Path;

#[derive(Args, Debug)]
pub struct ResidualArgs {
    #[arg(
        long = "from",
        short = 'f',
        visible_alias = "run",
        value_hint = clap::ValueHint::FilePath,
        help = "Stage-1 run prefix (or its .pinto.json)"
    )]
    pub run: Box<str>,

    #[arg(short, long, help = "Output directory")]
    pub out: Box<str>,

    #[arg(
        long,
        value_delimiter = ',',
        value_hint = clap::ValueHint::FilePath,
        help = "The run's data files where they are now (default: its manifest's)",
        long_help = "The run's own data files, in its order, when they have moved\n\
                     from where its manifest records them. As many files as the run\n\
                     had: a fit names its cells by its files, so a subset would not\n\
                     find the run's cells."
    )]
    pub data_files: Option<Vec<Box<str>>>,

    #[arg(
        id = "residual",
        long = "residual",
        value_enum,
        default_value_t = Operation::Divide,
        help = "What is left once μ is accounted for: divide (stage 2 models y/μ) or subtract (max(y - μ, 0) as counts)",
        long_help = "How stage 2 sees the counts once the stage-1 expectation μ is\n\
                     accounted for.\n\
                     divide (alias multiplicative): stage 2 fits the raw counts with μ\n\
                     as a multiplicative offset, i.e. models y/μ. Writes the tables μ\n\
                     is rebuilt from (stage1.*.parquet) and no new data files; fit the\n\
                     run's own data files with --offset <this -o>.\n\
                     subtract (alias additive): round(max(y - μ, 0)) written as new\n\
                     count files, which stage 2 fits as data."
    )]
    pub operation: Operation,

    #[arg(
        long,
        value_enum,
        default_value_t = Excess::Round,
        help = "Under --residual subtract, how y - μ becomes a count: round, or floor (a count of 1 over a mean below 1 is dropped)"
    )]
    pub excess: Excess,

    #[arg(
        long,
        default_value_t = crate::util::profile_match::EM_ITERS,
        help = "EM iterations per cell for θ",
        hide = true
    )]
    pub em_iters: usize,

    #[arg(long, help = "Columns read per block", hide = true)]
    pub block_size: Option<usize>,

    #[arg(
        long = "feature-name-mode",
        value_enum,
        help = "How feature names are read (default: as the run's manifest records, else auto)",
        long_help = "How the data's feature names are read: auto (the default: gene\n\
                     names keyed by their symbol, ENSG..._SYMBOL and SYMBOL alike; loci\n\
                     by their coordinates), exact (the rows every file shares, by name\n\
                     as written), gene, locus or mixed. Without it, the mode the run's\n\
                     manifest records it read them with, else auto. A stage-2 fit with\n\
                     --offset finds its rows in the tables by name or by the same loose\n\
                     key, whichever way it reads the names itself."
    )]
    pub feature_names: Option<FeatureNameMode>,

    #[arg(
        long,
        value_enum,
        default_value_t = Adapt::Batch,
        help = "Profiles μ is computed from: adapted per batch of the run, or pooled",
        long_help = "batch: each community's profile re-estimated per batch as the run\n\
                     knew it (its cells table's batch column, else its data files),\n\
                     shrunk toward the pooled profile, so a shift of a whole community\n\
                     in a batch is absorbed and a local patch stays in the residual.\n\
                     none: the pooled profiles, so donor nuisance stays too."
    )]
    pub adapt: Adapt,

    #[arg(
        long,
        default_value_t = 50.0,
        value_parser = crate::util::common::non_negative_f64,
        help = "Prior weight of the pooled profile under --adapt batch, in cells of the community"
    )]
    pub adapt_cells: f64,
}

/// The profiles `μ` is computed from.
#[derive(clap::ValueEnum, serde::Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Adapt {
    /// The run's communities, pooled over every batch.
    None,
    /// Each community re-estimated per batch of the run, shrunk toward the pool.
    Batch,
}

/// How what is left of the counts is taken (`--residual`).
#[derive(clap::ValueEnum, serde::Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Operation {
    /// Stage 2 fits the raw counts with μ as a multiplicative offset.
    #[value(alias = "multiplicative")]
    Divide,
    /// `round(max(y − μ, 0))` as new count files.
    #[value(alias = "additive")]
    Subtract,
}

/// How `y − μ` becomes a count.
#[derive(clap::ValueEnum, serde::Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Excess {
    Round,
    Floor,
}

/// One cell's excess counts: `(row, round(max(y − μ, 0)))` (or the floor)
/// over its nonzero entries `col` (data row, count), with `μ` from `θ`
/// (`pi`, length `K`), the profiles `p_t` (`[K, G]`) and the cell's total
/// `n`.
fn excess_counts(
    col: &[(usize, f64)],
    n: f64,
    p_t: &nalgebra::DMatrix<f64>,
    pi: &[f64],
    how: Excess,
) -> Vec<(usize, f32)> {
    col.iter()
        .filter_map(|&(r, y)| {
            let d = (y - mu_at(n, p_t, pi, r)).max(0.0);
            let e = match how {
                Excess::Round => d.round(),
                Excess::Floor => d.floor(),
            };
            (e > 0.0).then_some((r, e as f32))
        })
        .collect()
}

/// `μ_gi = n_i Σ_k θ_ik p_k(g)` of row `r`, from the cell's total `n`, its
/// `θ` (`pi`) and its batch's profiles `p_t` (`[K, G]`): the one expectation
/// both operations take their residual from.
fn mu_at(n: f64, p_t: &nalgebra::DMatrix<f64>, pi: &[f64], r: usize) -> f64 {
    n * pi
        .iter()
        .zip(p_t.column(r).iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
}

/// `θ` for one cell by EM over the profiles `p_t`; zero when the cell has no
/// counts (then it has no excess either).
fn cell_theta(col: &[(usize, f64)], p_t: &nalgebra::DMatrix<f64>, iters: usize) -> Vec<f64> {
    let mut pi = vec![0.0; p_t.nrows()];
    em_proportions(col, p_t, iters, &mut pi);
    pi
}

/// `(row, count)` of column `j` of a block.
fn column_entries(csc: &nalgebra_sparse::CscMatrix<f32>, j: usize) -> Vec<(usize, f64)> {
    let c = csc.col(j);
    c.row_indices()
        .iter()
        .copied()
        .zip(c.values().iter().map(|&v| f64::from(v)))
        .collect()
}

/// What the run's files hold, read as the run read them.
struct Inputs {
    data: SparseIoVec,
    /// `data`'s cell names, as the run named them.
    names: Vec<Box<str>>,
    /// `[N, K]` on `data`'s columns; zero for a cell the run did not place.
    prop: Mat,
    /// Run cell name → coordinates, from its cells table.
    coords: HashMap<Box<str>, (f32, f32)>,
    /// Each cell's adaptation group (`--adapt`): 0 for all under `none`.
    group_of: Vec<usize>,
    /// Each group's name: the run's batch, `unplaced` for cells its table
    /// lacks, `all` under `--adapt none`.
    group_names: Vec<Box<str>>,
    /// How `data`'s rows were read ([`Plan::feature_names`], resolved).
    feature_kind: FeatureNameKind,
}

/// What `pinto apply residual` writes for a run, and what it reads to make it,
/// checked as far as it can be before any count is read: `-o` is not a
/// file, the run is on raw counts, its data files are found, all of them,
/// with distinct names, its manifest records a propensity and a cells
/// table, both found, and names its coordinate columns, and no output is
/// an input.
#[derive(Clone, Debug)]
pub struct Plan {
    /// The run's data files, in its order.
    pub files: Vec<Box<str>>,
    /// How what is left is taken (`--residual`).
    pub operation: Operation,
    /// Per data file, its residual zarr (`{out}/{stem}.zarr.zip`), under
    /// `--residual subtract`; none under divide.
    pub zarrs: Vec<String>,
    /// The stage-1 tables under `--residual divide`
    /// ([`crate::util::stage1_divisor::Stage1Tables::paths`]); none under
    /// subtract.
    pub tables: Vec<String>,
    /// Per data file, its residual's coordinates, beside the zarr.
    pub coords: Vec<String>,
    /// `{out}/residual.json`.
    pub summary: String,
    /// The run's first two coordinate columns, which the coordinate files
    /// carry.
    pub coord_names: [Box<str>; 2],
    /// The run's propensity (`outputs.propensity`): its cells' community
    /// memberships.
    pub propensity: String,
    /// The run's cells table (`outputs.cells`): its cells' coordinates and
    /// batches.
    pub cells: String,
    /// Whether the run was fitted on feature modules (its data files are
    /// module files), so the residual is at module level.
    pub module_level: bool,
    /// How the run's rows are read: `--feature-name-mode`, else as the
    /// run's manifest records it read them, else auto. Exact only when
    /// asked for (here, or by the run).
    pub feature_names: FeatureNameMode,
}

impl Plan {
    pub fn new(run: &Run, args: &ResidualArgs) -> anyhow::Result<Self> {
        // a file at -o or above it would make every path under it read as
        // free, and the run would fail only when it writes
        if let Some(file) = crate::util::staging::final_path(Path::new(args.out.as_ref()))
            .ancestors()
            .find(|p| p.exists())
            .filter(|p| !p.is_dir())
        {
            anyhow::bail!("{} is a file; -o is a folder", file.display());
        }
        let files = stage_one_files(run, args)?;
        let outputs = &run.meta.outputs;
        let propensity = recorded(
            run,
            outputs.propensity.as_deref(),
            "propensity.parquet",
            "propensity output (outputs.propensity): a residual is taken from a run that \
             wrote its cells' community memberships",
        )?;
        let coord_names = coord_names(run)?;
        let cells = recorded(
            run,
            outputs.cells.as_deref(),
            "cells.parquet",
            "cells table (outputs.cells): a residual takes each cell's coordinates and \
             batch from it",
        )?;
        let named = output_paths(&files, &args.out)?;
        let coords = named.iter().map(|z| coords_path(z)).collect();
        let (zarrs, tables) = match args.operation {
            Operation::Subtract => (named, Vec::new()),
            Operation::Divide => (
                Vec::new(),
                crate::util::stage1_divisor::Stage1Tables::paths(&args.out).to_vec(),
            ),
        };
        Ok(Self {
            files,
            operation: args.operation,
            zarrs,
            tables,
            coords,
            summary: in_dir(&args.out, MARKER),
            coord_names,
            propensity,
            cells,
            module_level: run
                .meta
                .feature_modules
                .as_ref()
                .is_some_and(|m| m.module_level()),
            feature_names: args
                .feature_names
                .or(run.meta.feature_names)
                .unwrap_or(FeatureNameMode::Auto),
        })
    }

    /// Every path it writes: under subtract each file's zarr and
    /// coordinates, under divide each file's coordinates and the stage-1
    /// tables; then `residual.json`.
    pub fn made(&self) -> Vec<std::path::PathBuf> {
        let files: Vec<&String> = match self.operation {
            Operation::Subtract => self
                .zarrs
                .iter()
                .zip(&self.coords)
                .flat_map(|(z, c)| [z, c])
                .collect(),
            Operation::Divide => self.coords.iter().chain(&self.tables).collect(),
        };
        files
            .into_iter()
            .chain([&self.summary])
            .map(std::path::PathBuf::from)
            .collect()
    }

    /// Refused when anything it writes is there already (at its final
    /// name, under `--atomic-outputs`).
    fn ensure_unwritten(&self) -> anyhow::Result<()> {
        for path in self.made() {
            let path = crate::util::staging::final_path(&path);
            anyhow::ensure!(
                !path.exists(),
                "{} exists: nothing is written over; give another -o",
                path.display()
            );
        }
        Ok(())
    }
}

/// The run's data files (its manifest's, or `--data-files`), checked: a
/// run on raw counts (not a fit on a residual), every file found, as many
/// as the run had, each with its own name.
fn stage_one_files(run: &Run, args: &ResidualArgs) -> anyhow::Result<Vec<Box<str>>> {
    let files: Vec<Box<str>> = match &args.data_files {
        Some(f) => {
            for file in f {
                anyhow::ensure!(
                    Path::new(file.as_ref()).exists(),
                    "--data-files {file}: not found"
                );
            }
            f.clone()
        }
        None => run
            .meta
            .data_files
            .iter()
            .flatten()
            .map(|f| resolve(run, f).map(Into::into))
            .collect::<anyhow::Result<_>>()?,
    };
    anyhow::ensure!(
        !files.is_empty(),
        "{}: the manifest records no data files (data_files); pass the run's as --data-files",
        run.manifest.display()
    );
    // its residual.json would name the excess files as raw, and lra on a
    // fit of it would read excess counts as raw counts
    let on_residual = crate::lr_activity::input::residual_of(run)?.is_some()
        || run.meta.offset.is_some()
        || files
            .iter()
            .any(|f| crate::lr_activity::input::is_excess(Path::new(f.as_ref())));
    anyhow::ensure!(
        !on_residual,
        "{}: a fit on a residual; a residual is taken from a fit on raw counts, not \
         from a fit on another residual",
        run.manifest.display()
    );
    // a residual file is named by its data file, and a fit names cells by it
    let mut seen: HashSet<Box<str>> = HashSet::default();
    for f in &files {
        let name = basename(f.trim_end_matches('/'))?;
        anyhow::ensure!(
            seen.insert(name.clone()),
            "{f} and an earlier data file have the same name {name}"
        );
    }
    // a manifest that records none (no list, or an empty one) says nothing
    // of how many the run had
    let n_run = run.meta.data_files.as_ref().map_or(0, Vec::len);
    anyhow::ensure!(
        n_run == 0 || files.len() == n_run,
        "{} data files given for a run fitted on {n_run}: pass all of the run's files",
        files.len()
    );
    Ok(files)
}

/// The run's first two coordinate columns, as its manifest names them.
fn coord_names(run: &Run) -> anyhow::Result<[Box<str>; 2]> {
    match &run.meta.outputs.coord_columns {
        Some(c) if c.len() >= 2 => Ok([c[0].as_str().into(), c[1].as_str().into()]),
        _ => anyhow::bail!(
            "{}: the manifest names no coordinate columns, so the residual's coordinate \
             files could not carry the run's names",
            run.manifest.display()
        ),
    }
}

/// An output of the run a residual reads, as its manifest records it
/// ([`Run::recorded_output`]): refused when the manifest records none
/// (`what` names the record and what is taken from it) or the file is gone.
fn recorded(run: &Run, record: Option<&str>, suffix: &str, what: &str) -> anyhow::Result<String> {
    let found = run.recorded_output(record, suffix)?.ok_or_else(|| {
        anyhow::anyhow!(
            "{}: the manifest of this {} run records no {what}",
            run.manifest.display(),
            run.meta.command
        )
    })?;
    Ok(found.to_string_lossy().into_owned())
}

/// A file the run names, beside its manifest.
fn resolve(run: &Run, f: &str) -> anyhow::Result<String> {
    run.resolve(f)
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| anyhow::anyhow!("{f}: not found beside {}", run.manifest.display()))
}

impl Inputs {
    fn load(args: &ResidualArgs, plan: &Plan) -> anyhow::Result<Self> {
        // the rows every file shares, by name (exact) or by their key
        let feature_kind = match plan.feature_names {
            FeatureNameMode::Auto => {
                let first = crate::util::input::open_sparse(&plan.files[0])?;
                FeatureNameKind::auto_detect(&first.row_names()?)
            }
            mode => mode.resolve_kind(&[]),
        };
        info!("Reading the data's feature names as {feature_kind:?}");
        let data = crate::util::input::open_fit_data(&plan.files, &feature_kind)?;
        let names = data.column_names()?;

        let cells = crate::util::parquet_io::read_cells_table(
            Path::new(&plan.cells),
            Some(&plan.coord_names.each_ref().map(|c| c.to_string())),
        )?;
        let group_of = match args.adapt {
            Adapt::None => vec![0; names.len()],
            Adapt::Batch => batch_groups(&cells.names, cells.batches.as_deref(), &names),
        };
        let group_names = match args.adapt {
            Adapt::None => vec!["all".into()],
            Adapt::Batch => group_names(&cells.names, cells.batches.as_deref(), &names, &group_of),
        };
        let coords = cells.names.into_iter().zip(cells.coords).collect();

        let prop_path = &plan.propensity;
        let exclude: HashSet<Box<str>> = plan.coord_names.iter().cloned().collect();
        let (prop_run, _, _, prop_cells) =
            crate::util::parquet_io::read_propensity(Path::new(prop_path), &exclude)?;
        // every run cell found in the data by a unique name, as impute pairs them
        let prop = crate::impute::align_to_columns(prop_path, prop_run, &prop_cells, &names)
            .map_err(|e| {
                anyhow::anyhow!(
                    "{e:#}; pass the run's own data files, in its order, as --data-files"
                )
            })?;
        let k = prop.ncols();
        let prop = weighted_communities(prop);
        anyhow::ensure!(
            prop.ncols() > 0,
            "{prop_path}: no community has weight in the data"
        );
        if prop.ncols() < k {
            info!(
                "{} of {k} communities have no weight in the data and are left out",
                k - prop.ncols()
            );
        }
        Ok(Self {
            data,
            names,
            prop,
            coords,
            group_of,
            group_names,
            feature_kind,
        })
    }
}

/// Each group of [`batch_groups`]'s name: its cells' batch in the run's
/// cells table, `unplaced` for cells the table lacks, `all` with no batches.
fn group_names(
    table_names: &[Box<str>],
    batches: Option<&[Box<str>]>,
    names: &[Box<str>],
    group_of: &[usize],
) -> Vec<Box<str>> {
    let n_groups = group_of.iter().max().map_or(1, |&g| g + 1);
    let Some(batches) = batches else {
        return vec!["all".into(); n_groups];
    };
    let batch_of: HashMap<&str, &str> = table_names
        .iter()
        .zip(batches)
        .map(|(c, b)| (c.as_ref(), b.as_ref()))
        .collect();
    // cells the table lacks group apart, under a name no batch has
    let mut unplaced = String::from("unplaced");
    while batches.iter().any(|b| b.as_ref() == unplaced) {
        unplaced.push('_');
    }
    let mut out: Vec<Option<Box<str>>> = vec![None; n_groups];
    for (c, &g) in names.iter().zip(group_of) {
        if out[g].is_none() {
            out[g] = Some(
                batch_of
                    .get(c.as_ref())
                    .map_or(unplaced.as_str().into(), |&b| b.into()),
            );
        }
    }
    out.into_iter()
        .map(|n| n.unwrap_or_else(|| unplaced.as_str().into()))
        .collect()
}

/// Group ids over the data's columns `names`: each cell's batch in the
/// run's cells table (`table_names`, `batches`; none means one batch). A
/// cell the table lacks was not placed, carries no weight, and groups apart.
fn batch_groups(
    table_names: &[Box<str>],
    batches: Option<&[Box<str>]>,
    names: &[Box<str>],
) -> Vec<usize> {
    let Some(batches) = batches else {
        return vec![0; names.len()];
    };
    let batch_of: HashMap<&str, &str> = table_names
        .iter()
        .zip(batches)
        .map(|(c, b)| (c.as_ref(), b.as_ref()))
        .collect();
    crate::util::ablation::dense_ids(names.iter().map(|c| batch_of.get(c.as_ref())))
        .into_iter()
        .map(|g| g as usize)
        .collect()
}

/// The propensity's communities that have weight in the data: one with
/// none would get a flat profile from the prior alone, which the per-cell EM
/// could still give weight, raising μ and shrinking the excess.
fn weighted_communities(prop: Mat) -> Mat {
    let kept: Vec<usize> = (0..prop.ncols())
        .filter(|&k| prop.column(k).sum() > 0.0)
        .collect();
    prop.select_columns(&kept)
}

/// The pooled profiles `[K, G]` from the community sums over every cell
/// (`[G, K]`): lc's Gamma(1, 1) posterior mean, `1 + S_gk` normalized over
/// the data's rows.
fn pooled_profiles(total: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
    normalized_profiles(&total.add_scalar(1.0))
}

/// Each community's share of every data row, `[K, G]`, from rates `[G, K]`
/// on the data's own rows, so a feature reads its own entry whatever its
/// name.
fn normalized_profiles(rates: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
    let mut p_t = rates.transpose();
    for mut row in p_t.row_iter_mut() {
        let total = row.sum();
        if total > 0.0 {
            row /= total;
        }
    }
    p_t
}

/// One group's profiles `[K, G]` from its sums `S` (`[G, K]`):
/// `(α_k p_k + S_k) / (α_k + T_k)` per community, the pooled `p` where the
/// group has no counts of it.
fn adapt_profiles(
    pooled: &nalgebra::DMatrix<f64>,
    sums: &nalgebra::DMatrix<f64>,
    alpha: &[f64],
) -> nalgebra::DMatrix<f64> {
    let mut p = pooled.clone();
    for (k, &a) in alpha.iter().enumerate() {
        let t = sums.column(k).sum();
        if t > 0.0 {
            let row = (pooled.row(k) * a + sums.column(k).transpose()) / (a + t);
            p.set_row(k, &row);
        }
    }
    p
}

/// Adds block `(lb, ub)`'s community sums `Σ_i π_ik y_gi` (`[G, K]`) into
/// its cells' groups, one sparse-dense product per group in the block.
fn sums_visitor(
    (lb, ub): (usize, usize),
    data: &SparseIoVec,
    inp: &Inputs,
    out: Arc<Mutex<&mut Vec<nalgebra::DMatrix<f64>>>>,
) -> anyhow::Result<()> {
    let csc = data.read_columns_csc(lb..ub)?;
    let groups = &inp.group_of[lb..ub];
    let mut local = Vec::new();
    for b in groups.iter().copied().collect::<HashSet<usize>>() {
        let p_nk = Mat::from_fn(ub - lb, inp.prop.ncols(), |j, k| {
            if groups[j] == b {
                inp.prop[(lb + j, k)]
            } else {
                0.0
            }
        });
        local.push((b, (&csc * &p_nk).map(f64::from)));
    }
    let mut out = out.lock().expect("lock residual sums");
    for (b, m) in local {
        out[b] += m;
    }
    Ok(())
}

/// Each group's profiles `[K, G]` from one pass over the data: the pooled
/// profiles from every group's sums (see the module doc), alone with one
/// group, else each group's adapted with `α_k` = `n0` cells' worth of
/// community k's counts.
fn group_profiles(
    inp: &Inputs,
    n0: f64,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<nalgebra::DMatrix<f64>>> {
    let n_groups = inp.group_of.iter().max().map_or(1, |&g| g + 1);
    let (g, k) = (inp.data.num_rows(), inp.prop.ncols());
    let mut sums = vec![nalgebra::DMatrix::<f64>::zeros(g, k); n_groups];
    // one merge per block under the lock: blocks of a few thousand cells
    // keep the merges few
    let block = Some(block_size.unwrap_or(4096).max(1024));
    inp.data
        .visit_columns_by_block(&sums_visitor, inp, &mut sums, block)?;
    let total = sums
        .iter()
        .fold(nalgebra::DMatrix::zeros(g, k), |a, s| a + s);
    let pooled = pooled_profiles(&total);
    if n_groups == 1 {
        return Ok(vec![pooled]);
    }
    // community k's mean counts per unit of propensity, Σ S / Σ_i π_ik
    let alpha: Vec<f64> = (0..k)
        .map(|c| {
            let mass: f64 = inp.prop.column(c).iter().map(|&v| f64::from(v)).sum();
            n0 * total.column(c).sum() / mass
        })
        .collect();
    Ok(sums
        .iter()
        .map(|s| adapt_profiles(&pooled, s, &alpha))
        .collect())
}

struct ExcessParam<'a> {
    profiles: &'a [nalgebra::DMatrix<f64>],
    group_of: &'a [usize],
    iters: usize,
    how: Excess,
    operation: Operation,
}

/// Per cell (on `data`'s columns): its total and its excess entries.
type CellExcess = Vec<(f64, Vec<(usize, f32)>)>;

/// What `--residual divide` sums over the cells, per group `b` (rows `g`):
/// `Σ_{i∈b} n_i θ_i` (`K`), `Σ_{i∈b} n_i² θ_i θ_iᵀ` (`K × K` row-major), the
/// cells, and over the nonzeros `Σ (y² − 2yμ)` (`[G × groups]` row-major);
/// with every cell's `θ` (`[cells × K]`).
struct Moments {
    k: usize,
    n_groups: usize,
    a: Vec<f64>,
    c: Vec<f64>,
    cells: Vec<f64>,
    nz: Vec<f64>,
    theta: Vec<f32>,
}

impl Moments {
    fn new(k: usize, n_groups: usize, n_rows: usize, n_cells: usize) -> Self {
        Self {
            k,
            n_groups,
            a: vec![0.0; n_groups * k],
            c: vec![0.0; n_groups * k * k],
            cells: vec![0.0; n_groups],
            nz: vec![0.0; n_rows * n_groups],
            theta: vec![0.0; n_cells * k],
        }
    }

    /// One cell `i` of group `b`: total `n`, nonzeros `col`, `θ` and its
    /// group's profiles.
    fn add(
        &mut self,
        b: usize,
        n: f64,
        col: &[(usize, f64)],
        pi: &[f64],
        p_t: &nalgebra::DMatrix<f64>,
    ) {
        let k = self.k;
        self.cells[b] += 1.0;
        for (t, &x) in pi.iter().enumerate() {
            self.a[b * k + t] += n * x;
            for (u, &z) in pi.iter().enumerate() {
                self.c[(b * k + t) * k + u] += n * n * x * z;
            }
        }
        for &(r, y) in col {
            let mu = mu_at(n, p_t, pi, r);
            self.nz[r * self.n_groups + b] += y * y - 2.0 * y * mu;
        }
    }

    fn merge(&mut self, other: &Moments) {
        let add = |x: &mut [f64], y: &[f64]| x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
        add(&mut self.a, &other.a);
        add(&mut self.c, &other.c);
        add(&mut self.cells, &other.cells);
        add(&mut self.nz, &other.nz);
    }
}

/// One pass over the cells: each cell's `θ` and total, and under subtract
/// its excess entries, under divide the [`Moments`].
struct CellPass {
    cells: CellExcess,
    moments: Option<Moments>,
}

fn excess_visitor(
    (lb, ub): (usize, usize),
    data: &SparseIoVec,
    param: &ExcessParam,
    out: Arc<Mutex<&mut CellPass>>,
) -> anyhow::Result<()> {
    let csc = data.read_columns_csc(lb..ub)?;
    let k = param.profiles[0].nrows();
    let n_groups = param.profiles.len();
    let mut moments =
        (param.operation == Operation::Divide).then(|| Moments::new(k, n_groups, csc.nrows(), 0));
    let mut thetas: Vec<f32> = Vec::new();
    let block: CellExcess = (0..csc.ncols())
        .map(|j| {
            let b = param.group_of[lb + j];
            let p_t = &param.profiles[b];
            let col = column_entries(&csc, j);
            let pi = cell_theta(&col, p_t, param.iters);
            let n = col.iter().map(|&(_, y)| y).sum();
            match moments.as_mut() {
                Some(m) => {
                    m.add(b, n, &col, &pi, p_t);
                    thetas.extend(pi.iter().map(|&x| x as f32));
                    (n, Vec::new())
                }
                None => (n, excess_counts(&col, n, p_t, &pi, param.how)),
            }
        })
        .collect();
    let mut out = out.lock().expect("lock residual pass");
    for (j, e) in block.into_iter().enumerate() {
        out.cells[lb + j] = e;
    }
    if let (Some(all), Some(m)) = (out.moments.as_mut(), moments) {
        all.merge(&m);
        all.theta[lb * k..ub * k].copy_from_slice(&thetas);
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct FileSummary {
    data_file: Box<str>,
    /// The excess file, under subtract.
    #[serde(skip_serializing_if = "Option::is_none")]
    residual_file: Option<String>,
    coords_file: String,
    n_cells: usize,
    counts_in: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    excess_out: Option<f64>,
}

pub fn run_residual(args: &ResidualArgs) -> anyhow::Result<()> {
    let run = Run::open(&args.run)?;
    let plan = Plan::new(&run, args)?;
    plan.ensure_unwritten()?;
    let inp = Inputs::load(args, &plan)?;
    let sections = file_sections(&plan.files, &inp)?;
    let profiles = group_profiles(&inp, args.adapt_cells, args.block_size)?;
    info!(
        "Refitted {} raw-count community profiles in {} group(s) ({:?})",
        profiles[0].nrows(),
        profiles.len(),
        args.adapt
    );

    let n_cells = inp.data.num_columns();
    let k = profiles[0].nrows();
    if args.operation == Operation::Divide {
        // θ, then the per-batch K × K sums and the nonzeros' per-row sums
        let bytes =
            n_cells * k * 4 + profiles.len() * (k * k + inp.data.num_rows()) * 8 * 2 + n_cells * 8;
        crate::util::memory::ensure_fits(
            "pinto apply residual --residual divide",
            bytes,
            &crate::util::memory::Budget::now()?,
            "take fewer cells",
        )?;
    }
    let mut pass = CellPass {
        cells: vec![(0.0, Vec::new()); n_cells],
        moments: (args.operation == Operation::Divide)
            .then(|| Moments::new(k, profiles.len(), inp.data.num_rows(), n_cells)),
    };
    let param = ExcessParam {
        profiles: &profiles,
        group_of: &inp.group_of,
        iters: args.em_iters,
        how: args.excess,
        operation: args.operation,
    };
    inp.data
        .visit_columns_by_block(&excess_visitor, &param, &mut pass, args.block_size)?;

    std::fs::create_dir_all(args.out.as_ref())?;
    let rows = inp.data.row_names()?;
    let cells = &pass.cells;
    let summary = match args.operation {
        Operation::Subtract => plan
            .files
            .iter()
            .zip(&plan.zarrs)
            .zip(sections)
            .map(|((f, out_file), section)| {
                write_file(f, out_file, &rows, section, cells, &plan.coord_names)
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
        Operation::Divide => {
            let moments = pass.moments.as_ref().expect("divide sums its moments");
            let tables = stage1_tables(&inp, &profiles, moments, cells, &rows);
            tables.write(&args.out)?;
            let phi_mean = tables.phi.iter().map(|&x| f64::from(x)).sum::<f64>()
                / tables.phi.len().max(1) as f64;
            info!(
                "Divide: wrote the stage-1 tables ({} cells, {} rows, {} communities, {} \
                 batch(es); mean φ {phi_mean:.3}) under {}; fit the run's own data files \
                 with --offset {}",
                tables.cells.len(),
                tables.features.len(),
                tables.k,
                tables.batches.len(),
                crate::util::staging::final_str(&args.out),
                crate::util::staging::final_str(&args.out),
            );
            plan.files
                .iter()
                .zip(&plan.coords)
                .zip(sections)
                .map(|((f, coords_file), (range, barcodes, coords))| {
                    write_section_coords(coords_file, &coords, &plan.coord_names)?;
                    Ok(FileSummary {
                        data_file: f.clone(),
                        residual_file: None,
                        coords_file: coords_file.clone(),
                        n_cells: barcodes.len(),
                        counts_in: cells[range].iter().map(|c| c.0).sum(),
                        excess_out: None,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        }
    };
    std::fs::write(
        &plan.summary,
        serde_json::to_string_pretty(&residual_marker(
            &run,
            &plan.files,
            args,
            &inp.feature_kind,
            summary,
        )?)?,
    )?;
    Ok(())
}

/// `residual.json`: the block lra reads to take a fit on the residual's
/// files as a residual fit (the stage-1 run and the raw data files, index
/// for index with the residual's, as canonical paths that resolve from
/// anywhere), with how the excess was made and each file's counts.
fn residual_marker(
    run: &Run,
    files: &[Box<str>],
    args: &ResidualArgs,
    feature_kind: &FeatureNameKind,
    summary: Vec<FileSummary>,
) -> anyhow::Result<serde_json::Value> {
    let canonical = |p: &Path| -> anyhow::Result<String> {
        Ok(std::fs::canonicalize(p)
            .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?
            .to_string_lossy()
            .into_owned())
    };
    let block = crate::lr_activity::input::ResidualBlock {
        // under divide stage 2 reads the raw counts themselves
        counts: match args.operation {
            Operation::Subtract => "excess".into(),
            Operation::Divide => "raw".into(),
        },
        stage1: Some(canonical(&run.manifest)?),
        raw_data_files: files
            .iter()
            .map(|f| canonical(Path::new(f.as_ref())))
            .collect::<anyhow::Result<_>>()?,
    };
    let mut marker = serde_json::to_value(block)?;
    marker["operation"] = serde_json::to_value(args.operation)?;
    marker["adapt"] = serde_json::to_value(args.adapt)?;
    marker["adapt_cells"] = serde_json::to_value(args.adapt_cells)?;
    // the rows' names as read: a stage-2 fit reads the data the same way
    marker["feature_names"] = serde_json::to_value(FeatureNameMode::of_kind(feature_kind))?;
    // a run fitted on feature modules read module files as its data: the
    // expectation is taken on those modules, and stage 2 fits them
    let modules = run
        .meta
        .feature_modules
        .as_ref()
        .filter(|m| m.module_level());
    marker["feature_level"] = if modules.is_some() {
        "module"
    } else {
        "feature"
    }
    .into();
    if let Some(m) = modules.and_then(|m| m.n_modules) {
        marker["n_modules"] = m.into();
    }
    match args.operation {
        Operation::Subtract => {
            marker["excess"] = serde_json::to_value(args.excess)?;
            marker["formula"] = "max(y - mu, 0) as a count; mu = n_i sum_k theta_ik p_kb(g); \
                                 no batch delta from stage 1"
                .into();
        }
        Operation::Divide => {
            marker["formula"] = "stage 2 fits y with mu as a multiplicative offset; \
                                 mu = n_i sum_k theta_ik p_kb(g); d = mu / (n_i pbar_g), \
                                 pbar_g = sum_i mu_gi / sum_i n_i; \
                                 phi_g = max(0, sum_i [(y - mu)^2 - mu]) / sum_i mu^2; \
                                 sigma2_gb = mean over b's cells of mu + phi_g mu^2; \
                                 r = (y - mu) / sigma_gb; no batch delta from stage 1"
                .into();
            marker["tables"] =
                serde_json::json!(crate::util::stage1_divisor::Stage1Tables::paths(".")
                    .iter()
                    .map(|p| p.trim_start_matches("./").to_string())
                    .collect::<Vec<_>>());
        }
    }
    marker["files"] = serde_json::to_value(summary)?;
    Ok(marker)
}

/// The marker's file name: written last, it marks the residual whole.
pub const MARKER: &str = "residual.json";

/// Each data file's residual zarr under `out`, checked not to be its input.
fn output_paths(files: &[Box<str>], out: &str) -> anyhow::Result<Vec<String>> {
    let outputs: Vec<String> = files
        .iter()
        .map(|f| {
            let stem = basename(f.trim_end_matches('/'))?;
            Ok(in_dir(out, &format!("{stem}.zarr.zip")))
        })
        .collect::<anyhow::Result<_>>()?;
    for (o, f) in outputs.iter().zip(files) {
        // its final name, under --atomic-outputs
        let o = crate::util::staging::final_str(o);
        let (o, f) = (Path::new(&o), Path::new(f.as_ref()));
        // the input's own folder, whatever the input's extension
        let same_folder = matches!(
            (o.parent(), f.parent()),
            (Some(a), Some(b)) if same_file(a, b)
        );
        anyhow::ensure!(
            !same_file(o, f) && !same_folder,
            "{} is beside the input {}; give -o a directory apart from the data",
            o.display(),
            f.display()
        );
    }
    Ok(outputs)
}

/// One data file's span of `data`'s columns, its barcodes and its cells'
/// coordinates.
type Section = (std::ops::Range<usize>, Vec<Box<str>>, SectionCoords);

/// Every file's section, found before anything is written, so a file with
/// no placed cell stops the run before it writes.
fn file_sections(files: &[Box<str>], inp: &Inputs) -> anyhow::Result<Vec<Section>> {
    let mut lb = 0;
    files
        .iter()
        .enumerate()
        .map(|(j, file)| {
            let barcodes = inp.data[j].column_names()?;
            let ub = lb + barcodes.len();
            let coords = section_coords(&inp.names[lb..ub], &barcodes, &inp.coords)
                .map_err(|e| anyhow::anyhow!("{file}: {e}"))?;
            let range = lb..ub;
            lb = ub;
            Ok((range, barcodes, coords))
        })
        .collect()
}

/// One file's residual zarr and coordinates, and its summary.
fn write_file(
    f: &str,
    out_file: &str,
    rows: &[Box<str>],
    (range, barcodes, coords): Section,
    cells: &CellExcess,
    coord_names: &[Box<str>; 2],
) -> anyhow::Result<FileSummary> {
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    let (mut counts_in, mut excess_out) = (0.0, 0.0);
    for (c, (n, ex)) in cells[range].iter().enumerate() {
        counts_in += n;
        for &(r, e) in ex {
            excess_out += f64::from(e);
            triplets.push((r as u64, c as u64, e));
        }
    }
    crate::simulate::write::write_backend(out_file, rows, &barcodes, triplets)?;
    write_section_coords(&coords_path(out_file), &coords, coord_names)?;
    info!(
        "{f}: {} cells, {counts_in:.0} counts, {excess_out:.0} ({:.1}%) left as excess in {}",
        barcodes.len(),
        100.0 * excess_out / counts_in.max(1.0),
        crate::util::staging::final_str(out_file)
    );
    Ok(FileSummary {
        data_file: f.into(),
        residual_file: Some(out_file.into()),
        coords_file: coords_path(out_file),
        n_cells: barcodes.len(),
        counts_in,
        excess_out: Some(excess_out),
    })
}

/// The stage-1 tables from one pass's [`Moments`] (see the module doc):
/// `θ`, `n` and each cell's group, each group's profiles, and per row the
/// pooled share, the dispersion and each group's Pearson scale, the zeros'
/// sums in closed form through the `K × K` accumulators.
fn stage1_tables(
    inp: &Inputs,
    profiles: &[nalgebra::DMatrix<f64>],
    m: &Moments,
    cells: &CellExcess,
    rows: &[Box<str>],
) -> crate::util::stage1_divisor::Stage1Tables {
    let (k, nb, g) = (m.k, m.n_groups, rows.len());
    // Σ_{i∈b} μ_gb and Σ_{i∈b} μ²_gb
    let mut s1 = vec![0f64; g * nb];
    let mut s2 = vec![0f64; g * nb];
    for (b, p_t) in profiles.iter().enumerate() {
        let a = &m.a[b * k..(b + 1) * k];
        let c = &m.c[b * k * k..(b + 1) * k * k];
        for r in 0..g {
            let p = p_t.column(r);
            s1[r * nb + b] = (0..k).map(|t| p[t] * a[t]).sum();
            s2[r * nb + b] = (0..k)
                .map(|t| p[t] * (0..k).map(|u| c[t * k + u] * p[u]).sum::<f64>())
                .sum();
        }
    }
    let total_n: f64 = cells.iter().map(|c| c.0).sum();
    let mut pbar = Vec::with_capacity(g);
    let mut phi = Vec::with_capacity(g);
    let mut sigma2 = Vec::with_capacity(g * nb);
    for r in 0..g {
        let (mu, mu2, nz): (f64, f64, f64) = (0..nb).fold((0.0, 0.0, 0.0), |acc, b| {
            (
                acc.0 + s1[r * nb + b],
                acc.1 + s2[r * nb + b],
                acc.2 + m.nz[r * nb + b],
            )
        });
        pbar.push(if total_n > 0.0 {
            (mu / total_n) as f32
        } else {
            0.0
        });
        let f = if mu2 > 0.0 {
            ((nz + mu2 - mu) / mu2).max(0.0)
        } else {
            0.0
        };
        phi.push(f as f32);
        for b in 0..nb {
            let n_b = m.cells[b];
            sigma2.push(if n_b > 0.0 {
                ((s1[r * nb + b] + f * s2[r * nb + b]) / n_b) as f32
            } else {
                0.0
            });
        }
    }
    let mut flat = vec![0f32; nb * g * k];
    for (b, p_t) in profiles.iter().enumerate() {
        for r in 0..g {
            for t in 0..k {
                flat[(b * g + r) * k + t] = p_t[(t, r)] as f32;
            }
        }
    }
    crate::util::stage1_divisor::Stage1Tables {
        cells: inp.names.clone(),
        n: cells.iter().map(|c| c.0 as f32).collect(),
        batch: inp.group_of.iter().map(|&b| b as u32).collect(),
        theta: m.theta.clone(),
        k,
        batches: inp.group_names.clone(),
        features: rows.to_vec(),
        profiles: flat,
        pbar,
        phi,
        sigma2,
    }
}

/// A residual zarr's coordinate file, beside it.
fn coords_path(zarr: &str) -> String {
    format!(
        "{}.coords.parquet",
        data_beans::hdf5_io::strip_backend_suffix(zarr)
    )
}

/// One file's coordinates: its cells' barcodes and the run's coordinates of
/// them (`names`, as the run named them). A cell the run did not place has
/// none; a file with no placed cell is an error.
type SectionCoords = (Vec<Box<str>>, Vec<f32>, Vec<f32>);

fn section_coords(
    names: &[Box<str>],
    barcodes: &[Box<str>],
    coords: &HashMap<Box<str>, (f32, f32)>,
) -> anyhow::Result<SectionCoords> {
    let mut kept = Vec::new();
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (name, b) in names.iter().zip(barcodes) {
        if let Some(&(cx, cy)) = coords.get(name) {
            kept.push(b.clone());
            x.push(cx);
            y.push(cy);
        }
    }
    anyhow::ensure!(!kept.is_empty(), "the run placed no cells of this file");
    Ok((kept, x, y))
}

fn write_section_coords(
    path: &str,
    (kept, x, y): &SectionCoords,
    coord_names: &[Box<str>; 2],
) -> anyhow::Result<()> {
    write_named_table(
        path,
        "spot",
        kept,
        &[
            (coord_names[0].clone(), Column::F32(x)),
            (coord_names[1].clone(), Column::F32(y)),
        ],
    )
}

#[cfg(test)]
#[path = "residual/tests.rs"]
mod tests;
