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
//! feature weights: the NB Fisher-information weights
//! `{run}.feature_community` carries shrink high-mean features, so that
//! table is not a composition. Each cell then gets its proportions `θ_i` by
//! the per-cell EM of `pinto impute` and held-out scoring ([`em_proportions`]) and the
//! expectation `μ_gi = n_i Σ_k θ_ik p_k(g)`, `n_i` the cell's total, the mean of
//! that multinomial mixture. No batch fold `δ` from the stage-1 run enters
//! `p` or `μ`: `δ` is the donor (batch) effect, the very thing the residual
//! is meant to keep.
//!
//! By default (`--adapt batch`) each community's profile is then
//! re-estimated per batch as the run knew it (the `batch` column of its
//! cells table: its `--batch-files`, else its data files), here and from the
//! stage-1 communities only:
//!
//! `p_{k,b}(g) = (α_k p_k(g) + S_{gk,b}) / (α_k + T_{kb})`,
//! `S_{gk,b} = Σ_{i∈b} π_ik y_gi`, `T_{kb} = Σ_g S_{gk,b}`,
//!
//! a Dirichlet prior centred on the pooled profile with the mass of
//! `--adapt-cells` cells of community k (the hierarchical Dirichlet
//! smoothing of MacKay and Peto 1995, Natural Language Engineering 1:289-308,
//! doi:10.1017/S1351324900000218). A shift that moves a whole community in a
//! batch (donor, donor-by-type, section nuisance) is then absorbed, while a
//! local patch, a minority of the community's weight there, stays in the
//! residual. The identifying assumption is that donor-specific biology is
//! local: a shift of a whole community in one donor is taken for nuisance.
//! This is the per-cluster, per-batch correction of Harmony (Korsunsky et
//! al. 2019, Nat Methods 16:1289-1296, doi:10.1038/s41592-019-0619-0) applied
//! to count profiles. `--adapt none` keeps the pooled profiles.
//!
//! Per data file it writes `{out}/{stem}.zarr`, the excess counts
//! `round(max(y_gi − μ_gi, 0))` (or the floor, `--excess`) on the file's own
//! rows and cells, and `{out}/{stem}.coords.parquet`, the run's coordinates
//! for those cells. The excess keeps counts, so the count models fit it as
//! they fit data. `{out}/residual.json` records, per file, the counts in and
//! the excess out, and the block lra reads to know a run on these files was
//! fitted on excess counts: `counts: "excess"`, `stage1` (the stage-1
//! manifest), `raw_data_files` (the raw files, in the residual files'
//! order), `formula` and `excess`, with absolute paths.

use crate::lr_activity::recorded;
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
    #[arg(long, help = "Stage-1 run prefix (or its .pinto.json)")]
    pub run: Box<str>,

    #[arg(short, long, help = "Output directory")]
    pub out: Box<str>,

    #[arg(
        long,
        value_delimiter = ',',
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

    #[arg(
        long,
        value_enum,
        default_value_t = Adapt::Batch,
        help = "Profiles μ is computed from: the run's pooled ones, or adapted per batch"
    )]
    pub adapt: Adapt,

    #[arg(
        long,
        default_value_t = 50.0,
        value_parser = crate::link_community::args::parse_alpha,
        help = "Prior weight of the pooled profile, in cells of the community (--adapt batch)"
    )]
    pub adapt_cells: f64,

    #[arg(long, default_value_t = 100, help = "EM iterations per cell for θ")]
    pub em_iters: usize,

    #[arg(long, help = "Columns read per block")]
    pub block_size: Option<usize>,
}

/// The profiles `μ` is computed from.
#[derive(clap::ValueEnum, serde::Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Adapt {
    /// The run's communities, pooled over every section.
    None,
    /// Each community re-estimated per batch of the run, shrunk toward the pool.
    Batch,
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
/// (`pi`, length `K`) and the profiles `p_t` (`[K, G]`). `n` is the cell's
/// total.
pub fn excess_counts(
    col: &[(usize, f64)],
    p_t: &nalgebra::DMatrix<f64>,
    pi: &[f64],
    how: Excess,
) -> Vec<(usize, f32)> {
    let n: f64 = col.iter().map(|&(_, y)| y).sum();
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
    /// `[N, K]` on `data`'s columns; zero for a cell the run did not place.
    prop: Mat,
    /// Run cell name → coordinates, from its cells table.
    coords: HashMap<Box<str>, (f32, f32)>,
    coord_names: [Box<str>; 2],
    /// Cells per data file, in order.
    sizes: Vec<usize>,
    /// Each cell's adaptation group (`--adapt`), and how many there are.
    group_of: Vec<usize>,
    n_groups: usize,
}

impl Inputs {
    fn load(run: &Run, args: &ResidualArgs) -> anyhow::Result<Self> {
        let resolve = |f: &str| -> anyhow::Result<String> {
            run.resolve(f)
                .map(|p| p.to_string_lossy().into_owned())
                .ok_or_else(|| anyhow::anyhow!("{f}: not found beside {}", run.manifest.display()))
        };
        let output = |f: Option<&str>, suffix: &str| match f {
            Some(f) => resolve(f),
            None => Ok(format!("{}.{suffix}", run.prefix)),
        };
        let files: Vec<Box<str>> = match &args.data_files {
            Some(f) => f.clone(),
            None => run
                .meta
                .data_files
                .iter()
                .flatten()
                .map(|f| resolve(f).map(Into::into))
                .collect::<anyhow::Result<_>>()?,
        };
        anyhow::ensure!(!files.is_empty(), "{}: no data files", args.run);
        let n_run = run.meta.data_files.as_ref().map_or(files.len(), Vec::len);
        anyhow::ensure!(
            files.len() == n_run,
            "{} data files given for a run fitted on {n_run}: pass all of the run's files",
            files.len()
        );
        // lc and dsvd read their files on the rows all of them share.
        let data = crate::util::input::open_fit_data(&files, &FeatureNameKind::Exact, false)?;

        let coord_names: [Box<str>; 2] = match &run.meta.outputs.coord_columns {
            Some(c) if c.len() >= 2 => [c[0].as_str().into(), c[1].as_str().into()],
            _ => anyhow::bail!(
                "{}: the manifest names no coordinate columns, so the residual's coordinate \
                 files could not carry the run's names",
                run.manifest.display()
            ),
        };
        let cells_path = output(run.meta.outputs.cells.as_deref(), "cells.parquet")?;
        let cells = crate::util::parquet_io::read_cells_table(
            Path::new(&cells_path),
            run.meta.outputs.coord_columns.as_deref(),
        )?;
        let coords = cells.names.into_iter().zip(cells.coords).collect();
        let sizes = data.num_columns_by_data()?;
        let (group_of, n_groups) = match args.adapt {
            Adapt::None => (vec![0; data.num_columns()], 1),
            Adapt::Batch => batch_groups(&cells_path, &data.column_names()?, &sizes)?,
        };

        let prop_path = output(run.meta.outputs.propensity.as_deref(), "propensity.parquet")?;
        let exclude: HashSet<Box<str>> = coord_names.iter().cloned().collect();
        let (prop_run, _, _, prop_cells) =
            crate::util::parquet_io::read_propensity(Path::new(&prop_path), &exclude)?;
        let row_of: HashMap<&str, usize> = prop_cells
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_ref(), i))
            .collect();
        let mut prop = Mat::zeros(data.num_columns(), prop_run.ncols());
        let mut placed = 0usize;
        for (j, c) in data.column_names()?.iter().enumerate() {
            if let Some(&i) = row_of.get(c.as_ref()) {
                prop.set_row(j, &prop_run.row(i));
                placed += 1;
            }
        }
        anyhow::ensure!(
            placed > 0,
            "{prop_path}: none of its cells are in the data files; pass the run's own --data-files"
        );
        info!(
            "{placed} of {} cells are placed by the run",
            data.num_columns()
        );
        Ok(Self {
            files,
            data,
            prop,
            coords,
            coord_names,
            sizes,
            group_of,
            n_groups,
        })
    }
}

/// Each data column's batch as the run knew it (the cells table's `batch`
/// column, written when the run had more than one), as group ids; a cell the
/// table lacks, or a run with one batch, groups by its data file.
fn batch_groups(
    cells_path: &str,
    names: &[Box<str>],
    sizes: &[usize],
) -> anyhow::Result<(Vec<usize>, usize)> {
    use legume_numeric::matrix::parquet::{
        peek_parquet_field_names, read_parquet_string_columns_by_name,
    };
    let fields = peek_parquet_field_names(cells_path)?;
    let batch_of: HashMap<Box<str>, Box<str>> = if fields.iter().any(|f| f.as_ref() == "batch") {
        let cols = read_parquet_string_columns_by_name(cells_path, &[fields[0].as_ref(), "batch"])?;
        cols[0]
            .iter()
            .cloned()
            .zip(cols[1].iter().cloned())
            .collect()
    } else {
        HashMap::default()
    };
    let file_of = sizes
        .iter()
        .enumerate()
        .flat_map(|(f, &n)| std::iter::repeat_n(f, n));
    let mut id: HashMap<String, usize> = HashMap::default();
    let group_of = names
        .iter()
        .zip(file_of)
        .map(|(c, f)| {
            let key = batch_of
                .get(c)
                .map_or_else(|| format!("file {f}"), |b| format!("batch {b}"));
            let next = id.len();
            *id.entry(key).or_insert(next)
        })
        .collect();
    Ok((group_of, id.len()))
}

/// Pooled raw-count profiles `[K, G]` from the community sums over every
/// group (`[K, G]`): `p_k(g) = (1 + S_gk) / Σ_g' (1 + S_g'k)`, lc's
/// feature-community Gamma(1, 1) posterior mean with unit weights,
/// normalised.
fn pooled_profiles(total: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
    let mut p = total.add_scalar(1.0);
    for mut row in p.row_iter_mut() {
        let t: f64 = row.sum();
        row /= t;
    }
    p
}

/// One group's profiles, written over its sums `[K, G]`:
/// `(α_k p_k + S_k) / (α_k + T_k)` per community, the pool where the group
/// has no counts of it.
fn adapt_profiles(
    pooled: &nalgebra::DMatrix<f64>,
    sums: &mut nalgebra::DMatrix<f64>,
    alpha: &[f64],
) {
    for (k, &a) in alpha.iter().enumerate() {
        let t: f64 = sums.row(k).sum();
        let row = if t > 0.0 {
            (pooled.row(k) * a + sums.row(k)) / (a + t)
        } else {
            pooled.row(k).into_owned()
        };
        sums.set_row(k, &row);
    }
}

/// Per-group community sums `S_{gk,b} = Σ_{i∈b} π_ik y_gi`, each `[K, G]`.
type GroupSums = Vec<nalgebra::DMatrix<f64>>;

fn sums_visitor(
    (lb, ub): (usize, usize),
    data: &SparseIoVec,
    inp: &Inputs,
    out: Arc<Mutex<&mut GroupSums>>,
) -> anyhow::Result<()> {
    let csc = data.read_columns_csc(lb..ub)?;
    let k = inp.prop.ncols();
    let mut local: HashMap<usize, nalgebra::DMatrix<f64>> = HashMap::default();
    let mut pi = vec![0.0f64; k];
    for j in 0..csc.ncols() {
        let i = lb + j;
        for (c, x) in pi.iter_mut().enumerate() {
            *x = f64::from(inp.prop[(i, c)]);
        }
        let acc = local
            .entry(inp.group_of[i])
            .or_insert_with(|| nalgebra::DMatrix::zeros(k, csc.nrows()));
        for (r, y) in column_entries(&csc, j) {
            let mut col = acc.column_mut(r);
            for (c, &x) in pi.iter().enumerate() {
                col[c] += x * y;
            }
        }
    }
    let mut out = out.lock().expect("lock residual sums");
    for (b, m) in local {
        out[b] += m;
    }
    Ok(())
}

/// Each group's profiles `[K, G]`: under `--adapt none` the one group's
/// pool, under `batch` each batch's adapted profiles, with `α_k` = `n0`
/// cells' worth of community k's counts.
fn group_profiles(
    inp: &Inputs,
    n0: f64,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<nalgebra::DMatrix<f64>>> {
    let (k, g) = (inp.prop.ncols(), inp.data.num_rows());
    let mut sums: GroupSums = vec![nalgebra::DMatrix::zeros(k, g); inp.n_groups];
    inp.data
        .visit_columns_by_block(&sums_visitor, inp, &mut sums, block_size)?;
    let total = sums
        .iter()
        .fold(nalgebra::DMatrix::zeros(k, g), |a, s| a + s);
    let pooled = pooled_profiles(&total);
    if inp.n_groups == 1 {
        return Ok(vec![pooled]);
    }
    // Community k's mean counts per unit of propensity, Σ S / Σ_i π_ik.
    let alpha: Vec<f64> = (0..k)
        .map(|c| {
            let mass: f64 = inp.prop.column(c).iter().map(|&v| f64::from(v)).sum();
            if mass > 0.0 {
                n0 * total.row(c).sum() / mass
            } else {
                0.0
            }
        })
        .collect();
    for s in &mut sums {
        adapt_profiles(&pooled, s, &alpha);
    }
    Ok(sums)
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
            (n, excess_counts(&col, p_t, &pi, param.how))
        })
        .collect();
    let mut out = out.lock().expect("lock residual excess");
    for (j, e) in block.into_iter().enumerate() {
        out[lb + j] = e;
    }
    Ok(())
}

fn cell_excess(
    inp: &Inputs,
    profiles: &[nalgebra::DMatrix<f64>],
    args: &ResidualArgs,
) -> anyhow::Result<CellExcess> {
    let mut cells: CellExcess = vec![(0.0, Vec::new()); inp.data.num_columns()];
    let param = ExcessParam {
        profiles,
        group_of: &inp.group_of,
        iters: args.em_iters,
        how: args.excess,
    };
    inp.data
        .visit_columns_by_block(&excess_visitor, &param, &mut cells, args.block_size)?;
    Ok(cells)
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
    let inp = Inputs::load(&run, args)?;
    let outputs: Vec<String> = inp
        .files
        .iter()
        .map(|f| {
            Ok(format!(
                "{}/{}.zarr",
                args.out,
                basename(f.trim_end_matches('/'))?
            ))
        })
        .collect::<anyhow::Result<_>>()?;
    for (o, f) in outputs.iter().zip(&inp.files) {
        anyhow::ensure!(
            !same_file(Path::new(o), Path::new(f.as_ref())),
            "{o} is the input {f}; give -o a directory apart from the data"
        );
    }

    let profiles = group_profiles(&inp, args.adapt_cells, args.block_size)?;
    info!(
        "Refitted {} raw-count community profiles in {} group(s) ({:?})",
        profiles[0].nrows(),
        profiles.len(),
        args.adapt
    );
    let cells = cell_excess(&inp, &profiles, args)?;

    std::fs::create_dir_all(args.out.as_ref())?;
    let rows = inp.data.row_names()?;
    let names = inp.data.column_names()?;
    let mut summary = Vec::with_capacity(inp.files.len());
    let mut lb = 0;
    for (j, (f, out_file)) in inp.files.iter().zip(&outputs).enumerate() {
        let barcodes = inp.data[j].column_names()?;
        let ub = lb + inp.sizes[j];
        let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
        let (mut counts_in, mut excess_out) = (0.0, 0.0);
        for (c, (n, ex)) in cells[lb..ub].iter().enumerate() {
            counts_in += n;
            for &(r, e) in ex {
                excess_out += f64::from(e);
                triplets.push((r as u64, c as u64, e));
            }
        }
        if Path::new(out_file).exists() {
            std::fs::remove_dir_all(out_file)?;
        }
        crate::simulate::write::write_backend(out_file, &rows, &barcodes, triplets)?;
        let coords_file = format!("{}.coords.parquet", out_file.trim_end_matches(".zarr"));
        write_section_coords(
            &coords_file,
            &names[lb..ub],
            &barcodes,
            &inp.coords,
            &inp.coord_names,
        )?;
        info!(
            "{f}: {} cells, {counts_in:.0} counts, {excess_out:.0} ({:.1}%) left as excess in {out_file}",
            barcodes.len(),
            100.0 * excess_out / counts_in.max(1.0)
        );
        summary.push(FileSummary {
            data_file: f.clone(),
            residual_file: out_file.clone(),
            n_cells: barcodes.len(),
            counts_in,
            excess_out,
        });
        lb = ub;
    }
    std::fs::write(
        format!("{}/residual.json", args.out),
        serde_json::to_string_pretty(&serde_json::json!({
            "run": run.prefix,
            "counts": "excess",
            "stage1": recorded(&run.manifest),
            "raw_data_files": inp.files.iter().map(|f| recorded(Path::new(f.as_ref()))).collect::<Vec<_>>(),
            "formula": "max(y - mu, 0) as a count (see excess), mu = n_i sum_k theta_ik p_k(g) \
                        on the run's communities (see adapt), no batch delta from the run",
            "excess": args.excess,
            "adapt": args.adapt,
            "adapt_cells": args.adapt_cells,
            "files": summary,
        }))?,
    )?;
    Ok(())
}

/// One file's coordinates: the run's coordinates of its cells (`names`, as
/// the run named them), keyed by the file's own `barcodes`. A cell the run
/// did not place has none.
fn write_section_coords(
    path: &str,
    names: &[Box<str>],
    barcodes: &[Box<str>],
    coords: &HashMap<Box<str>, (f32, f32)>,
    coord_names: &[Box<str>; 2],
) -> anyhow::Result<()> {
    let mut kept = Vec::new();
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (name, b) in names.iter().zip(barcodes) {
        if let Some(&(cx, cy)) = coords.get(name) {
            kept.push(b.clone());
            x.push(cx);
            y.push(cy);
        }
    }
    anyhow::ensure!(
        !kept.is_empty(),
        "{path}: the run placed no cells of this file"
    );
    write_named_table(
        path,
        "spot",
        &kept,
        &[
            (coord_names[0].clone(), Column::F32(&x)),
            (coord_names[1].clone(), Column::F32(&y)),
        ],
    )
}

#[cfg(test)]
#[path = "residual/tests.rs"]
mod tests;
