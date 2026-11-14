//! `pinto residual`: the counts a fitted run leaves unexplained, as new data
//! files that `lc`, `cage` and `dsvd` read unchanged.
//!
//! Stage 1 is any lc, cage or dsvd run; what is conditioned on is its
//! communities, through the per-cell propensity `π_ik` each of them
//! publishes (`{run}.propensity.parquet`). From it, raw-count community
//! profiles are refitted on the run's data, read as a fit reads it (the rows
//! every file shares, cells named as the run named them),
//!
//! `p_k(g) ∝ (1 + Σ_i π_ik y_gi) / (1 + Σ_i π_ik)`, normalised over `g`,
//!
//! lc's own feature-community statistic and Gamma(1, 1) posterior mean
//! (`link_community::profiles::fit_feature_community_param`) with unit
//! feature weights, the `1 + Σ_i π_ik` cancelling in the normalisation: the NB
//! Fisher-information weights `{run}.feature_community` carries shrink
//! high-mean features, so that table is not a composition. Each cell then
//! gets its proportions `θ_i` by the per-cell EM of `pinto impute` and
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
//! Per data file it writes `{out}/{stem}.zarr`, the excess counts
//! `round(max(y_gi − μ_gi, 0))` (or the floor, `--excess`) on the rows every
//! file shares (as an exact-name fit reads them; with `--feature-names` as
//! the fit's) and the file's own cells, and `{out}/{stem}.coords.parquet`,
//! the run's first two coordinates for those cells. The excess keeps counts,
//! so the count models fit it as they fit data. `{out}/residual.json`
//! records, per file, the counts in and the excess out, how `μ` was made
//! (`excess`, `adapt`, `adapt_cells`, `formula`), and carries the block
//! `pinto lra` reads to take a fit on these files on its raw counts: the
//! stage-1 run and the raw data files.

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
        long,
        value_enum,
        default_value_t = Excess::Round,
        help = "How y - μ becomes a count: round, or floor (a count of 1 over a mean below 1 is dropped)"
    )]
    pub excess: Excess,

    #[arg(long, default_value_t = 100, help = "EM iterations per cell for θ")]
    pub em_iters: usize,

    #[arg(long, help = "Columns read per block")]
    pub block_size: Option<usize>,

    #[arg(
        long,
        value_enum,
        default_value = "exact",
        help = "How the run's fit read feature names (cage's --feature-names)",
        long_help = "How the run's fit read feature names: exact (lc and dsvd, which\n\
                     read the rows every file shares), or the --feature-names a cage\n\
                     run was fitted with, so its rows are read as it read them."
    )]
    pub feature_names: FeatureNameMode,

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

/// The commands whose runs a residual is taken from.
const STAGE_ONE: [&str; 3] = ["lc", "cage", "dsvd"];

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
            let mu = n * pi
                .iter()
                .zip(p_t.column(r).iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            let d = (y - mu).max(0.0);
            let e = match how {
                Excess::Round => d.round(),
                Excess::Floor => d.floor(),
            };
            (e > 0.0).then_some((r, e as f32))
        })
        .collect()
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
    files: Vec<Box<str>>,
    data: SparseIoVec,
    /// `data`'s cell names, as the run named them.
    names: Vec<Box<str>>,
    /// `[N, K]` on `data`'s columns; zero for a cell the run did not place.
    prop: Mat,
    /// Run cell name → coordinates, from its cells table.
    coords: HashMap<Box<str>, (f32, f32)>,
    coord_names: [Box<str>; 2],
    /// Each cell's adaptation group (`--adapt`): 0 for all under `none`.
    group_of: Vec<usize>,
}

/// What `pinto residual` writes for a run, and what it reads to make it,
/// checked as far as it can be before any count is read: `-o` is not a
/// file, the run is an lc, cage or dsvd run on raw counts, its data files
/// are found, all of them, with distinct names, it names its coordinate
/// columns, and no output is an input.
#[derive(Clone, Debug)]
pub struct Plan {
    /// The run's data files, in its order.
    pub files: Vec<Box<str>>,
    /// Per data file, its residual zarr (`{out}/{stem}.zarr`).
    pub zarrs: Vec<String>,
    /// Per data file, its residual's coordinates, beside the zarr.
    pub coords: Vec<String>,
    /// `{out}/residual.json`.
    pub summary: String,
    /// The run's first two coordinate columns, which the coordinate files
    /// carry.
    pub coord_names: [Box<str>; 2],
}

impl Plan {
    pub fn new(run: &Run, args: &ResidualArgs) -> anyhow::Result<Self> {
        // a file at -o or above it would make every path under it read as
        // free, and the run would fail only when it writes
        if let Some(file) = Path::new(args.out.as_ref())
            .ancestors()
            .find(|p| p.exists())
            .filter(|p| !p.is_dir())
        {
            anyhow::bail!("{} is a file; -o is a folder", file.display());
        }
        let files = stage_one_files(run, args)?;
        let coord_names = coord_names(run)?;
        let zarrs = output_paths(&files, &args.out)?;
        let coords = zarrs.iter().map(|z| coords_path(z)).collect();
        Ok(Self {
            files,
            zarrs,
            coords,
            summary: summary_path(&args.out),
            coord_names,
        })
    }

    /// Every path it writes: each file's zarr and coordinates, then
    /// `residual.json`.
    pub fn made(&self) -> Vec<std::path::PathBuf> {
        self.zarrs
            .iter()
            .zip(&self.coords)
            .flat_map(|(z, c)| [z, c])
            .chain([&self.summary])
            .map(std::path::PathBuf::from)
            .collect()
    }

    /// Refused when anything it writes is there already.
    fn ensure_unwritten(&self) -> anyhow::Result<()> {
        for path in self.made() {
            anyhow::ensure!(
                !path.exists(),
                "{} exists: nothing is written over; give another -o",
                path.display()
            );
        }
        Ok(())
    }
}

/// The run's data files (its manifest's, or `--data-files`), checked: an
/// lc, cage or dsvd run on raw counts (not a fit on a residual), every file
/// found, as many as the run had, each with its own name.
fn stage_one_files(run: &Run, args: &ResidualArgs) -> anyhow::Result<Vec<Box<str>>> {
    anyhow::ensure!(
        STAGE_ONE.contains(&run.meta.command.as_str()),
        "{}: a {} run; a residual is taken from an lc, cage or dsvd run",
        run.manifest.display(),
        run.meta.command
    );
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
    anyhow::ensure!(!files.is_empty(), "{}: no data files", args.run);
    // its residual.json would name the excess files as raw, and lra on a
    // fit of it would read excess counts as raw counts
    let on_residual = crate::lr_activity::input::residual_of(run)?.is_some()
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
    let n_run = run.meta.data_files.as_ref().map_or(files.len(), Vec::len);
    anyhow::ensure!(
        files.len() == n_run,
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

/// A file the run names, beside its manifest.
fn resolve(run: &Run, f: &str) -> anyhow::Result<String> {
    run.resolve(f)
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| anyhow::anyhow!("{f}: not found beside {}", run.manifest.display()))
}

impl Inputs {
    fn load(run: &Run, args: &ResidualArgs, plan: &Plan) -> anyhow::Result<Self> {
        let output = |f: Option<&str>, suffix: &str| match f {
            Some(f) => resolve(run, f),
            None => Ok(format!("{}.{suffix}", run.prefix)),
        };
        let files = plan.files.clone();
        // read as the fit read them: lc and dsvd on the rows every file
        // shares, cage by its feature-name mode
        let kind = match args.feature_names {
            FeatureNameMode::Auto => {
                let first = crate::util::input::open_sparse(&files[0], false)?;
                FeatureNameKind::auto_detect(&first.row_names()?)
            }
            mode => mode.resolve_kind(&[]),
        };
        let data = crate::util::input::open_fit_data(&files, &kind, false)?;

        let coord_names = plan.coord_names.clone();
        let cells = crate::util::parquet_io::read_cells_table(
            Path::new(&output(run.meta.outputs.cells.as_deref(), "cells.parquet")?),
            Some(
                &coord_names
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>(),
            ),
        )?;
        let group_of = match args.adapt {
            Adapt::None => vec![0; data.num_columns()],
            Adapt::Batch => batch_groups(
                &cells.names,
                cells.batches.as_deref(),
                &data.column_names()?,
            ),
        };
        let coords = cells.names.into_iter().zip(cells.coords).collect();

        let prop_path = output(run.meta.outputs.propensity.as_deref(), "propensity.parquet")?;
        let exclude: HashSet<Box<str>> = coord_names.iter().cloned().collect();
        let (prop_run, _, _, prop_cells) =
            crate::util::parquet_io::read_propensity(Path::new(&prop_path), &exclude)?;
        // every run cell found in the data by a unique name, as impute pairs them
        let names = data.column_names()?;
        let prop = crate::impute::align_to_columns(&prop_path, prop_run, &prop_cells, &names)
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
            files,
            data,
            names,
            prop,
            coords,
            coord_names,
            group_of,
        })
    }
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
/// (`[G, K]`): lc's Gamma(1, 1) posterior mean, `1 + S_gk` normalised over
/// the data's rows.
fn pooled_profiles(total: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
    normalised_profiles(&total.add_scalar(1.0))
}

/// Each community's share of every data row, `[K, G]`, from rates `[G, K]`
/// on the data's own rows, so a feature reads its own entry whatever its
/// name.
fn normalised_profiles(rates: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
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
}

/// Per cell (on `data`'s columns): its total and its excess entries.
type CellExcess = Vec<(f64, Vec<(usize, f32)>)>;

fn excess_visitor(
    (lb, ub): (usize, usize),
    data: &SparseIoVec,
    param: &ExcessParam,
    out: Arc<Mutex<&mut CellExcess>>,
) -> anyhow::Result<()> {
    let csc = data.read_columns_csc(lb..ub)?;
    let block: CellExcess = (0..csc.ncols())
        .map(|j| {
            let p_t = &param.profiles[param.group_of[lb + j]];
            let col = column_entries(&csc, j);
            let pi = cell_theta(&col, p_t, param.iters);
            let n = col.iter().map(|&(_, y)| y).sum();
            (n, excess_counts(&col, n, p_t, &pi, param.how))
        })
        .collect();
    let mut out = out.lock().expect("lock residual excess");
    for (j, e) in block.into_iter().enumerate() {
        out[lb + j] = e;
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct FileSummary {
    data_file: Box<str>,
    residual_file: String,
    n_cells: usize,
    counts_in: f64,
    excess_out: f64,
}

pub fn run_residual(args: &ResidualArgs) -> anyhow::Result<()> {
    let run = Run::open(&args.run)?;
    let plan = Plan::new(&run, args)?;
    plan.ensure_unwritten()?;
    let inp = Inputs::load(&run, args, &plan)?;
    let sections = file_sections(&inp)?;
    let profiles = group_profiles(&inp, args.adapt_cells, args.block_size)?;
    info!(
        "Refitted {} raw-count community profiles in {} group(s) ({:?})",
        profiles[0].nrows(),
        profiles.len(),
        args.adapt
    );

    let mut cells: CellExcess = vec![(0.0, Vec::new()); inp.data.num_columns()];
    let param = ExcessParam {
        profiles: &profiles,
        group_of: &inp.group_of,
        iters: args.em_iters,
        how: args.excess,
    };
    inp.data
        .visit_columns_by_block(&excess_visitor, &param, &mut cells, args.block_size)?;

    std::fs::create_dir_all(args.out.as_ref())?;
    let rows = inp.data.row_names()?;
    let summary = inp
        .files
        .iter()
        .zip(&plan.zarrs)
        .zip(sections)
        .map(|((f, out_file), section)| {
            write_file(f, out_file, &rows, section, &cells, &inp.coord_names)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    std::fs::write(
        &plan.summary,
        serde_json::to_string_pretty(&residual_marker(&run, &inp.files, args, summary)?)?,
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
    summary: Vec<FileSummary>,
) -> anyhow::Result<serde_json::Value> {
    let canonical = |p: &Path| -> anyhow::Result<String> {
        Ok(std::fs::canonicalize(p)
            .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?
            .to_string_lossy()
            .into_owned())
    };
    let block = crate::lr_activity::input::ResidualBlock {
        counts: "excess".into(),
        stage1: Some(canonical(&run.manifest)?),
        raw_data_files: files
            .iter()
            .map(|f| canonical(Path::new(f.as_ref())))
            .collect::<anyhow::Result<_>>()?,
    };
    let mut marker = serde_json::to_value(block)?;
    marker["excess"] = serde_json::to_value(args.excess)?;
    marker["adapt"] = serde_json::to_value(args.adapt)?;
    marker["adapt_cells"] = serde_json::to_value(args.adapt_cells)?;
    marker["formula"] =
        "max(y - mu, 0) as a count; mu = n_i sum_k theta_ik p_k(g); no batch delta from stage 1"
            .into();
    marker["files"] = serde_json::to_value(summary)?;
    Ok(marker)
}

fn summary_path(out: &str) -> String {
    format!("{out}/residual.json")
}

/// Each data file's residual zarr under `out`, checked not to be its input.
fn output_paths(files: &[Box<str>], out: &str) -> anyhow::Result<Vec<String>> {
    let outputs: Vec<String> = files
        .iter()
        .map(|f| Ok(format!("{out}/{}.zarr", basename(f.trim_end_matches('/'))?)))
        .collect::<anyhow::Result<_>>()?;
    for (o, f) in outputs.iter().zip(files) {
        anyhow::ensure!(
            !same_file(Path::new(o), Path::new(f.as_ref())),
            "{o} is the input {f}; give -o a directory apart from the data"
        );
    }
    Ok(outputs)
}

/// One data file's span of `data`'s columns, its barcodes and its cells'
/// coordinates.
type Section = (std::ops::Range<usize>, Vec<Box<str>>, SectionCoords);

/// Every file's section, found before anything is written, so a file with
/// no placed cell stops the run before it writes.
fn file_sections(inp: &Inputs) -> anyhow::Result<Vec<Section>> {
    let mut lb = 0;
    (0..inp.files.len())
        .map(|j| {
            let barcodes = inp.data[j].column_names()?;
            let ub = lb + barcodes.len();
            let coords = section_coords(&inp.names[lb..ub], &barcodes, &inp.coords)
                .map_err(|e| anyhow::anyhow!("{}: {e}", inp.files[j]))?;
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
        "{f}: {} cells, {counts_in:.0} counts, {excess_out:.0} ({:.1}%) left as excess in {out_file}",
        barcodes.len(),
        100.0 * excess_out / counts_in.max(1.0)
    );
    Ok(FileSummary {
        data_file: f.into(),
        residual_file: out_file.into(),
        n_cells: barcodes.len(),
        counts_in,
        excess_out,
    })
}

/// A residual zarr's coordinate file, beside it.
fn coords_path(zarr: &str) -> String {
    format!("{}.coords.parquet", zarr.trim_end_matches(".zarr"))
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
