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
//! high-mean features, so that table is not a composition. Each cell then
//! gets its proportions `θ_i` by the per-cell EM of `pinto impute` and
//! held-out scoring ([`em_proportions`]) and the expectation
//! `μ_gi = n_i Σ_k θ_ik p_k(g)`, `n_i` the cell's total, the mean of that
//! multinomial mixture. No batch fold `δ` from the stage-1 run enters
//! `p` or `μ`: `δ` is the donor (batch) effect, the very thing the residual
//! is meant to keep.
//!
//! Per data file it writes `{out}/{stem}.zarr`, the excess counts
//! `round(max(y_gi − μ_gi, 0))` (or the floor, `--excess`) on the rows every
//! file shares (as an exact-name fit reads them; with `--feature-names` as
//! the fit's) and the file's own cells, and `{out}/{stem}.coords.parquet`,
//! the run's first two coordinates for those cells. The excess keeps counts,
//! so the count models fit it as they fit data. `{out}/residual.json` records, per file, the counts in and
//! the excess out.

use crate::cell_activity_graph_embedding::args::FeatureNameMode;
use crate::link_community::profiles::fit_feature_community_param;
use crate::util::common::*;
use crate::util::lupin_rounds::same_file;
use crate::util::metadata::Run;
use crate::util::profile_match::em_proportions;
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
}

impl Inputs {
    fn load(run: &Run, args: &ResidualArgs) -> anyhow::Result<Self> {
        anyhow::ensure!(
            STAGE_ONE.contains(&run.meta.command.as_str()),
            "{}: a {} run; a residual is taken from an lc, cage or dsvd run",
            run.manifest.display(),
            run.meta.command
        );
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
            Some(
                &coord_names
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>(),
            ),
        )?;
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
        })
    }
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

/// The run's communities as raw-count profiles `p_t` (`[K, G]`), from lc's
/// feature-community statistic with unit weights (see the module doc).
fn community_profiles(
    inp: &Inputs,
    block_size: Option<usize>,
) -> anyhow::Result<nalgebra::DMatrix<f64>> {
    let ones = vec![1.0f32; inp.data.num_rows()];
    let gamma = fit_feature_community_param(&inp.prop, &inp.data, Some(&ones), None, block_size)?;
    Ok(normalised_profiles(&gamma.posterior_mean().map(f64::from)))
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
            (n, excess_counts(&col, n, param.p_t, &pi, param.how))
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
    let outputs = output_paths(&inp.files, &args.out)?;
    let sections = file_sections(&inp)?;
    let p_t = community_profiles(&inp, args.block_size)?;
    info!("Refitted {} raw-count community profiles", p_t.nrows());

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
    let summary = inp
        .files
        .iter()
        .zip(&outputs)
        .zip(sections)
        .map(|((f, out_file), section)| {
            write_file(f, out_file, &rows, section, &cells, &inp.coord_names)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    std::fs::write(
        summary_path(&args.out),
        serde_json::to_string_pretty(&serde_json::json!({
            "run": run.prefix,
            "excess": args.excess,
            "files": summary,
        }))?,
    )?;
    Ok(())
}

fn summary_path(out: &str) -> String {
    format!("{out}/residual.json")
}

/// Each data file's residual zarr under `out`, checked to write over
/// nothing: not over its input, nor over any file already there.
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
    let targets = outputs
        .iter()
        .flat_map(|o| [o.clone(), coords_path(o)])
        .chain([summary_path(out)]);
    for path in targets {
        anyhow::ensure!(
            !Path::new(&path).exists(),
            "{path} exists: nothing is written over; give another -o"
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
