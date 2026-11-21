//! `--feature-modules` for the fits that read counts as data: lc, dsvd and
//! the spatial baselines.
//!
//! The modules are cage's: [`resolve_modules`] partitions the features by
//! their profiles over pseudobulks (`auto` lets Leiden find the count, a
//! count fixes it by k-means), the one builder every method uses. cage
//! trains a learned module layer from that partition; the other fits are
//! fitted on the modules themselves. Before the fit reads its data, each
//! data file's counts are summed within modules and written as
//! `{out}.feature_modules/{name}.zarr.zip` (rows `M0`, `M1`, ..., the
//! file's own cells), and the fit reads those files in place of the
//! features. Its manifest's `data_files` are those module files, so
//! everything that reads the run back (propensity, the viewer) reads the
//! rows the fit read. `{out}.feature_modules.parquet` names each feature's
//! module.
//!
//! The pseudobulks are the cells binned by the sign codes of a random
//! projection's leading directions. Cell QC runs here, on the features (a
//! module has no mitochondrial fraction): the module files hold only the
//! cells it keeps (and `--batch-files`, if given, are cut to them), and the
//! fit takes them as they are.
//!
//! `--feature-inclusion moran` is the second choice ([`prepare_moran`]): the
//! fit reads the features whose posterior probability of spatial signal is
//! 0.5 or more. Given alone it turns the default modules off; beside an
//! explicit `--feature-modules` it is refused, as in cage.
//!
//! A default (unset) `--feature-modules` yields to lc's `--feature-network`
//! (the user's own modules); an explicit count or `auto` beside it is
//! refused.

use crate::cell_activity_graph_embedding::args::{FeatureInclusionMode, ModulePlan};
use crate::cell_activity_graph_embedding::fit::modules::{pseudobulk_sizes, resolve_modules};
use crate::link_community::profiles::coarsen_cell_expression_dense;
use crate::util::common::*;
use crate::util::input::{data_file_names, open_fit_data, read_batch_file, InputArgs};
use data_beans::alg::random_projection::{binary_sort_columns, RandProjOps};
use data_beans::aux::feature_names::FeatureNameKind;
use serde::{Deserialize, Serialize};

/// Sign bits of the projection the pseudobulks are binned by: at most
/// `2^10` pseudobulks.
const PSEUDOBULK_BITS: usize = 10;

/// Columns read at a time while the module sums are taken.
const BLOCK: usize = 4096;

/// `--feature-modules`, as every fit but cage takes it (cage has its own,
/// with the same meaning and value).
#[derive(clap::Args, Debug, Clone, Default)]
pub struct FeatureModulesArgs {
    #[arg(
        long,
        value_name = "M|auto",
        help = "Fit on M feature modules: auto (default) takes M from the data, 0 fits the features",
        long_help = FEATURE_MODULES_HELP
    )]
    pub feature_modules: Option<ModulePlan>,

    #[arg(
        long,
        value_enum,
        default_value_t = FeatureInclusionMode::All,
        help = "Which features the fit reads: all, or those with spatial signal (moran)",
        long_help = FEATURE_INCLUSION_HELP,
        hide_possible_values = true
    )]
    pub feature_inclusion: FeatureInclusionMode,

    /// Whether the run's rows are modules, set once the flag is resolved
    /// (from the module files written):
    /// a fit with its own feature screen (spatialpca's SPARK-X) skips it
    /// on modules.
    #[arg(skip)]
    pub on: bool,
}

const FEATURE_MODULES_HELP: &str = "\
Fit on feature modules rather than on the features themselves.
The features are partitioned as cage partitions them: by their profiles over
pseudobulks of the cells, the features with no grouping evidence in one
background module and the rest in the Leiden communities of the profiles'
nearest-neighbour graph. Each module's counts are the sum of its features'.

M. auto (the default) takes the count from the data; a number of 2 or more
fixes it (k-means); 0 fits the features. If auto finds fewer than two modules
the run fits the features and says so.

Cell QC is taken on the features before they are summed. The module counts
are written as {out}.feature_modules/{name}.zarr.zip, which the run records
as its data files. {out}.feature_modules.parquet names each feature's module.

Left at its default it yields to --feature-inclusion moran and to lc's
--feature-network; given beside either it is refused.";

const FEATURE_INCLUSION_HELP: &str = "\
Which features the fit reads, the second choice to --feature-modules.
all (the default): every feature (on modules, unless --feature-modules 0).
moran: each feature's Moran's I over the spatial graph, its jackknife standard
error and a two-group empirical-Bayes model on the t statistics give pi_g, its
posterior probability of spatial signal (as cage's --feature-inclusion moran
computes it, on cells). The fit reads the features with pi_g >= 0.5, their
counts unchanged, written as {out}.feature_inclusion/{name}.zarr.zip; each
feature's statistics go to {out}.feature_inclusion.parquet. Needs
coordinates (-c), and its own feature screen (SPARK-X, highly variable genes)
is skipped, as on modules.

An alternative to --feature-modules, not a companion: moran given alone turns
the default modules off, and given beside an explicit --feature-modules it is
refused.";

/// How a run treated its features, as its manifest records it.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FeatureModulesInfo {
    /// What was asked: `auto`, a count, `off`, `moran` (cage's
    /// `--feature-inclusion moran`) or `feature-network` (lc's own modules).
    pub setting: String,
    /// The modules the run has, when it has any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_modules: Option<usize>,
    /// What the fit's rows are: `module` (summed counts) or `feature`.
    pub level: String,
    /// The feature-level data files the modules were summed from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_data_files: Option<Vec<String>>,
    /// `{out}.feature_modules.parquet`: each feature's module.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub membership: Option<String>,
    /// Cells cell QC dropped before the modules were summed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qc_dropped: Option<usize>,
}

impl FeatureModulesInfo {
    /// A run fitted on its features, for the reason `setting`.
    pub fn features(setting: &str) -> Self {
        Self {
            setting: setting.into(),
            n_modules: None,
            level: "feature".into(),
            feature_data_files: None,
            membership: None,
            qc_dropped: None,
        }
    }

    /// Whether the run's rows are modules.
    pub fn module_level(&self) -> bool {
        self.level == "module"
    }
}

/// `--feature-modules`' value as the manifest spells it.
pub fn setting(plan: ModulePlan) -> String {
    match plan {
        ModulePlan::Off => "off".into(),
        ModulePlan::Auto => "auto".into(),
        ModulePlan::Fixed(m) => m.to_string(),
    }
}

/// The plan of `--feature-modules` (`flag`; unset is `auto`) beside
/// `--feature-inclusion moran` and `--feature-network`, which a default
/// yields to and an explicit request is refused beside; with the setting
/// to record when it yields.
pub fn plan(
    flag: Option<ModulePlan>,
    moran: bool,
    network: bool,
) -> anyhow::Result<(ModulePlan, &'static str)> {
    anyhow::ensure!(
        !(moran && network),
        "--feature-inclusion moran chooses the features a fit reads, and \
         --feature-network gives it its rows already; drop one"
    );
    match flag {
        None if moran => Ok((ModulePlan::Off, "moran")),
        None if network => Ok((ModulePlan::Off, "feature-network")),
        None => Ok((ModulePlan::Auto, "")),
        Some(ModulePlan::Off) => Ok((ModulePlan::Off, "")),
        Some(p) => {
            anyhow::ensure!(
                !moran,
                "--feature-inclusion moran and --feature-modules are alternatives; pick one"
            );
            anyhow::ensure!(
                !network,
                "--feature-network gives lc its own feature modules, and --feature-modules \
                 builds pinto's; pick one"
            );
            Ok((p, ""))
        }
    }
}

/// Puts `input` on feature modules under `plan`: the module files written
/// and made its data files (QC taken and turned off, batch files cut to
/// the kept cells). Returns what the manifest records.
pub fn prepare(input: &mut InputArgs, plan: ModulePlan) -> anyhow::Result<FeatureModulesInfo> {
    let asked = setting(plan);
    if plan == ModulePlan::Off {
        return Ok(FeatureModulesInfo::features(&asked));
    }
    let files = input.data_files.clone();
    let data = open_fit_data(&files, &FeatureNameKind::Exact)?;
    let n_all = data.num_columns();
    let keep = match input.qc.to_config() {
        Some(cfg) => data_beans::qc_lib::compute_qc(&data, &cfg, None)?.train_keep,
        None => vec![true; n_all],
    };
    let n_kept = keep.iter().filter(|&&k| k).count();
    anyhow::ensure!(
        n_kept > 1,
        "cell QC keeps {n_kept} cells: no modules to find"
    );

    // the pseudobulks, on the kept cells
    let mut kept = open_fit_data(&files, &FeatureNameKind::Exact)?;
    kept.mask_columns(&keep)?;
    let proj = kept
        .project_columns_with_batch_correction_seeded::<Box<str>>(
            input.collapse.proj_dim,
            None,
            None,
            input.seed,
        )?
        .proj;
    let bits = PSEUDOBULK_BITS.min(proj.nrows()).min(n_kept - 1).max(1);
    let codes = binary_sort_columns(&proj, bits)?;
    let mut index: HashMap<usize, usize> = HashMap::default();
    let labels: Vec<usize> = codes
        .iter()
        .map(|&c| {
            let n = index.len();
            *index.entry(c).or_insert(n)
        })
        .collect();
    let n_pb = index.len();
    let profiles = coarsen_cell_expression_dense(&kept, &labels, n_pb, None)?;
    let sizes = pseudobulk_sizes(&labels, n_pb);
    drop(kept);

    let Some(start) = resolve_modules(plan, &profiles, &sizes, input.seed)? else {
        return Ok(FeatureModulesInfo::features(&asked));
    };
    let n_modules = start.cfg.n_modules;
    let module_of: Vec<Option<usize>> = start.labels.iter().map(|&m| Some(m as usize)).collect();
    let module_names: Vec<Box<str>> = (0..n_modules).map(|m| format!("M{m}").into()).collect();
    info!(
        "--feature-modules {asked}: fitting on {n_modules} modules of {} features",
        data.num_rows()
    );

    let out = input.out.to_string();
    let dir = format!("{out}.feature_modules");
    let membership = format!("{out}.feature_modules.parquet");
    let module_ids: Vec<f32> = start.labels.iter().map(|&m| m as f32).collect();
    legume_numeric::matrix::parquet::write_named_table(
        &membership,
        "feature",
        &data.row_names()?,
        &[(
            "module".into(),
            legume_numeric::matrix::parquet::Column::F32(&module_ids),
        )],
    )?;

    write_rows(input, &data, &keep, &module_of, &module_names, &dir)?;
    // QC was taken on the features above; a module has no mito fraction
    input.qc.no_qc = true;
    Ok(FeatureModulesInfo {
        setting: asked,
        n_modules: Some(n_modules),
        level: "module".into(),
        feature_data_files: Some(crate::util::metadata::recorded_files(&files)),
        membership: Some(membership),
        qc_dropped: Some(n_all - n_kept),
    })
}

/// Writes `data`'s kept cells (`keep`) with row `r` summed into row
/// `row_of[r]` of `names` (`None`: left out), one `{dir}/{name}.zarr.zip`
/// per data file, and makes them `input`'s data files (batch files cut to
/// the kept cells).
fn write_rows(
    input: &mut InputArgs,
    data: &data_beans::sparse_io_vector::SparseIoVec,
    keep: &[bool],
    row_of: &[Option<usize>],
    names_out: &[Box<str>],
    dir: &str,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let files = input.data_files.clone();
    let names = data_file_names(&files)?;
    let per_file = data.num_columns_by_data()?;
    let mut new_files = Vec::with_capacity(files.len());
    let mut new_batches = Vec::new();
    let mut lb = 0usize;
    for (j, (name, &n_j)) in names.iter().zip(&per_file).enumerate() {
        let barcodes = data[j].column_names()?;
        let mut cells: Vec<Box<str>> = Vec::new();
        let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
        let mut start_col = lb;
        while start_col < lb + n_j {
            let end = (start_col + BLOCK).min(lb + n_j);
            let csc = data.read_columns_csc(start_col..end)?;
            for c in 0..csc.ncols() {
                let i = start_col + c;
                if !keep[i] {
                    continue;
                }
                let mut sums = vec![0f32; names_out.len()];
                let col = csc.col(c);
                for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                    if let Some(m) = row_of[r] {
                        sums[m] += v;
                    }
                }
                let cell = cells.len() as u64;
                for (m, &s) in sums.iter().enumerate() {
                    if s != 0.0 {
                        triplets.push((m as u64, cell, s));
                    }
                }
                cells.push(barcodes[i - lb].clone());
            }
            start_col = end;
        }
        let path = format!("{dir}/{name}.zarr.zip");
        crate::simulate::write::write_backend(&path, names_out, &cells, triplets)?;
        if let Some(batch_files) = input.batch.batch_files.as_ref() {
            let labels = read_batch_file(&batch_files[j], n_j)?;
            let cut: Vec<&str> = labels
                .iter()
                .zip(&keep[lb..lb + n_j])
                .filter(|(_, &k)| k)
                .map(|(l, _)| l.as_ref())
                .collect();
            let bpath = format!("{dir}/{name}.batch.txt");
            std::fs::write(&bpath, cut.join("\n") + "\n")?;
            new_batches.push(bpath.into_boxed_str());
        }
        new_files.push(path.into_boxed_str());
        lb += n_j;
    }
    input.data_files = new_files;
    if input.batch.batch_files.is_some() {
        input.batch.batch_files = Some(new_batches);
    }
    Ok(())
}

/// Puts `input` on the features `--feature-inclusion moran` keeps: each
/// feature's posterior probability of spatial signal π (cage's
/// [`cell_inclusion`](crate::cell_activity_graph_embedding::feature_inclusion::cell_inclusion),
/// on cells), the features with π ≥ 0.5 written as data files with their
/// counts unchanged, every cell kept (the fit's own QC runs as usual).
pub fn prepare_moran(input: &mut InputArgs) -> anyhow::Result<FeatureModulesInfo> {
    use crate::cell_activity_graph_embedding::feature_inclusion::{
        cell_inclusion, CELL_COLUMN_BLOCK,
    };
    use crate::util::srt_pipeline::{
        preprocess_srt, topology_graph, BatchEstimate, FeatureAxisMode, SrtPreprocessConfig,
    };
    anyhow::ensure!(
        !input.coord_files.is_empty(),
        "--feature-inclusion moran needs coordinates (-c): it scores spatial signal"
    );
    let out = input.out.to_string();
    let dir = format!("{out}.feature_inclusion");
    // the score's own read, its side outputs kept apart from the fit's
    #[derive(clap::Parser)]
    struct Srt {
        #[command(flatten)]
        srt: crate::util::input::SrtInputArgs,
    }
    let mut srt = <Srt as clap::Parser>::try_parse_from(["pinto", "x", "-o", "x"])?.srt;
    srt.input = input.clone();
    srt.input.out = format!("{dir}/score").into();
    let pre = preprocess_srt(SrtPreprocessConfig {
        common: &srt,
        ablation: None,
        batch_effects: BatchEstimate::Skip,
        feature_axis: FeatureAxisMode::Strict,
        cell_projection: false,
        auto_batch_memory_check: None,
        feature_kind: None,
    })?;
    let axis = pre
        .feature_axis
        .as_ref()
        .expect("FeatureAxisMode::Strict yields an axis");
    let inc = cell_inclusion(
        &pre.data_vec,
        axis,
        (&pre.coordinates, &pre.coordinate_names),
        &pre.section_ids,
        &topology_graph(&pre.graph, &pre.spatial_graph).edges,
        INCLUSION_BLOCKS,
        CELL_COLUMN_BLOCK,
    )?;
    drop(pre);

    let data = open_fit_data(&input.data_files, &FeatureNameKind::Exact)?;
    let rows = data.row_names()?;
    let row_axis = crate::util::feature_axis::FeatureAxis::resolve(&rows)?;
    let kept: Vec<usize> = (0..rows.len())
        .filter(|&r| inc.pi[row_axis.feature_of_row(r)] >= 0.5)
        .collect();
    anyhow::ensure!(
        kept.len() >= 2,
        "--feature-inclusion moran: {} of {} features have a posterior probability of \
         spatial signal of 0.5 or more; a fit needs at least 2. Use --feature-modules instead",
        kept.len(),
        rows.len()
    );
    info!(
        "--feature-inclusion moran: fitting {} of {} features (pi >= 0.5)",
        kept.len(),
        rows.len()
    );
    let membership = format!("{out}.feature_inclusion.parquet");
    let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let (moran, se, t) = (f32s(&inc.moran), f32s(&inc.se), f32s(&inc.t));
    use legume_numeric::matrix::parquet::{write_named_table, Column};
    write_named_table(
        &membership,
        "feature",
        row_axis.feature_names(),
        &[
            ("moran_i".into(), Column::F32(&moran)),
            ("se".into(), Column::F32(&se)),
            ("t".into(), Column::F32(&t)),
            ("pi".into(), Column::F32(&inc.pi)),
        ],
    )?;
    let mut row_of = vec![None; rows.len()];
    for (new, &r) in kept.iter().enumerate() {
        row_of[r] = Some(new);
    }
    let names: Vec<Box<str>> = kept.iter().map(|&r| rows[r].clone()).collect();
    let files = input.data_files.clone();
    write_rows(
        input,
        &data,
        &vec![true; data.num_columns()],
        &row_of,
        &names,
        &dir,
    )?;
    Ok(FeatureModulesInfo {
        setting: "moran".into(),
        n_modules: None,
        level: "feature".into(),
        feature_data_files: Some(crate::util::metadata::recorded_files(&files)),
        membership: Some(membership),
        qc_dropped: None,
    })
}

/// Jackknife blocks per section for `--feature-inclusion moran`, cage's
/// default.
const INCLUSION_BLOCKS: usize = 16;

/// What a cage or cage-cell run records: its module plan (`moran` when
/// `--feature-inclusion moran` turned the default modules off), and the
/// modules its layer trained, counted from `{out}.module_membership.parquet`.
/// cage's modules are a layer of the model, so its rows stay features.
pub fn cage_info(
    args: &crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs,
) -> anyhow::Result<FeatureModulesInfo> {
    use crate::cell_activity_graph_embedding::args::FeatureInclusionMode;
    let plan = args.module_plan()?;
    let moran = args.feature_inclusion == FeatureInclusionMode::Moran;
    let mut info = FeatureModulesInfo::features(&if moran && plan == ModulePlan::Off {
        "moran".to_string()
    } else {
        setting(plan)
    });
    let membership = format!("{}.module_membership.parquet", args.common.out);
    if std::path::Path::new(&membership).exists() {
        let table = Mat::from_parquet(&membership)?;
        info.n_modules = Some(table.cols.len());
        info.membership = Some(membership);
    }
    Ok(info)
}

/// Writes `info` into the manifest at `{out}.pinto.json`, and under
/// modules the module files as its data files (as the fit read them).
pub fn record(out: &str, info: &FeatureModulesInfo) -> anyhow::Result<()> {
    let path = std::path::PathBuf::from(format!("{out}.pinto.json"));
    let mut meta = crate::util::metadata::PintoMetadata::read(&path)?;
    meta.feature_modules = Some(info.clone());
    meta.write(&path)
}

#[cfg(test)]
#[path = "tests/feature_modules.rs"]
mod tests;
