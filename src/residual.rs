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
//! ([`fit_feature_community_param`]) with unit feature weights: the NB
//! Fisher-information weights `{run}.feature_community` carries shrink
//! high-mean features, so that table is not a composition. Each cell then gets its proportions `θ_i` by the per-cell EM
//! of `pinto impute` and held-out scoring ([`em_proportions`]) and the
//! expectation `μ_gi = n_i Σ_k θ_ik p_k(g)`, `n_i` the cell's total, the mean of
//! that multinomial mixture. No batch fold `δ` from the stage-1 run enters
//! `p` or `μ`: `δ` is the donor (batch) effect, the very thing the residual
//! is meant to keep.
//!
//! Per data file it writes `{out}/{stem}.zarr`, the excess counts
//! `round(max(y_gi − μ_gi, 0))` (or the floor, `--excess`) on the file's own
//! rows and cells, and `{out}/{stem}.coords.parquet`, the run's coordinates
//! for those cells. The excess keeps counts, so the count models fit it as
//! they fit data. `{out}/residual.json` records, per file, the counts in and
//! the excess out.

use crate::link_community::profiles::fit_feature_community_param;
use crate::util::common::*;
use crate::util::lupin_rounds::same_file;
use crate::util::metadata::Run;
use crate::util::profile_match::{em_proportions, match_profiles};
use clap::Args;
use data_beans::aux::feature_names::FeatureNameKind;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use legume_numeric::param::traits::Inference;
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

    #[arg(long, default_value_t = 100, help = "EM iterations per cell for θ")]
    pub em_iters: usize,

    #[arg(long, help = "Columns read per block")]
    pub block_size: Option<usize>,
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
        let cells = crate::util::parquet_io::read_cells_table(
            Path::new(&output(run.meta.outputs.cells.as_deref(), "cells.parquet")?),
            run.meta.outputs.coord_columns.as_deref(),
        )?;
        let coords = cells.names.into_iter().zip(cells.coords).collect();

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
        })
    }
}

/// The run's communities as raw-count profiles `p_t` (`[K, G]`), from lc's
/// feature-community statistic with unit weights (see the module doc).
fn community_profiles(
    inp: &Inputs,
    block_size: Option<usize>,
) -> anyhow::Result<nalgebra::DMatrix<f64>> {
    let rows = inp.data.row_names()?;
    let ones = vec![1.0f32; rows.len()];
    let gamma = fit_feature_community_param(&inp.prop, &inp.data, Some(&ones), None, block_size)?;
    let kind = FeatureNameKind::Exact;
    let m = match_profiles(
        &rows,
        gamma.posterior_mean(),
        &rows,
        &kind,
        "run",
        inp.data.num_columns(),
    )?;
    Ok(m.p_t)
}

struct ExcessParam<'a> {
    p_t: &'a nalgebra::DMatrix<f64>,
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
            let col = column_entries(&csc, j);
            let pi = cell_theta(&col, param.p_t, param.iters);
            let n = col.iter().map(|&(_, y)| y).sum();
            (n, excess_counts(&col, param.p_t, &pi, param.how))
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
    let inp = Inputs::load(&run, args)?;
    let p_t = community_profiles(&inp, args.block_size)?;
    info!("Refitted {} raw-count community profiles", p_t.nrows());

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

    let mut cells: CellExcess = vec![(0.0, Vec::new()); inp.data.num_columns()];
    let param = ExcessParam {
        p_t: &p_t,
        iters: args.em_iters,
        how: args.excess,
    };
    inp.data
        .visit_columns_by_block(&excess_visitor, &param, &mut cells, args.block_size)?;

    std::fs::create_dir_all(args.out.as_ref())?;
    let rows = inp.data.row_names()?;
    let names = inp.data.column_names()?;
    let mut summary = Vec::with_capacity(inp.files.len());
    let mut lb = 0;
    for (j, (f, out_file)) in inp.files.iter().zip(&outputs).enumerate() {
        let barcodes = inp.data[j].column_names()?;
        let ub = lb + barcodes.len();
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
            "excess": args.excess,
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
