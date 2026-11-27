//! The outputs every spatial baseline writes, named from its `--out`
//! prefix, as a pinto run: `pinto view`, `sim score-by-set --labels`,
//! `sim score-by-pair --vectors`, `compare`, `lra` and `pinto run`/MCP
//! read it as they read an `lc`, `dsvd` or `cage` run.
//!
//! - `{out}.pinto.json`: the manifest;
//! - `{out}.propensity.parquet`: cell × K membership (the method's soft
//!   posterior when it has one, else one-hot labels), `cluster`, `entropy`;
//! - `{out}.cells.parquet`: every cell with its coordinates;
//! - `{out}.cell_embedding.parquet`: `cell` and the method's per-cell vectors;
//! - `{out}.labels.tsv`: `cell<TAB>label`, labels `C0`, `C1`, ..., as the
//!   propensity columns name them;
//! - `{out}.baseline.json`: the method's settings, as run.

use super::input::{SpatialData, SpatialInputArgs};
use crate::link_community::outputs::write_propensity_matrix;
use crate::util::common::Mat;
use crate::util::metadata::{
    delta_files, manifest_path, now_secs, BaselineInput, BaselineRun, BatchLabels, InputLayout,
    OutputFiles, PintoMetadata, PseudobulkInput,
};
use crate::util::parquet_io::{cell_pb_path, cells_table_path, write_cell_pb};
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::matrix::parquet::Column;
use legume_numeric::matrix::traits::IoOps;

fn labels_path(prefix: &str) -> String {
    format!("{prefix}.labels.tsv")
}

fn embedding_path(prefix: &str) -> String {
    format!("{prefix}.cell_embedding.parquet")
}

fn propensity_path(prefix: &str) -> String {
    format!("{prefix}.propensity.parquet")
}

fn settings_path(prefix: &str) -> String {
    format!("{prefix}.baseline.json")
}

/// What a baseline fit produced, for [`write_run`].
pub struct Fitted<'a> {
    /// The subcommand, e.g. `stagate`.
    pub method: &'a str,
    /// 0-based labels, one per cell.
    pub labels: &'a [usize],
    /// Soft membership, cells × K, when the method has one.
    pub membership: Option<&'a Mat>,
    /// Per-cell vectors, cells × d, and the name their columns take.
    pub embedding: &'a Mat,
    pub embedding_name: &'a str,
    /// The settings as run, recorded in `{out}.baseline.json`.
    pub settings: serde_json::Value,
    /// How the input was held (`baseline.input_layout`), for a method
    /// whose layout depends on the device; `None` otherwise.
    pub input_layout: Option<InputLayout>,
    /// What a `-pb` run fitted on and its cells' super-cells; `None`
    /// on cells.
    pub pseudobulk: Option<PbFit<'a>>,
}

/// The super-cells a `-pb` run fitted on (`baseline.input`), and
/// each cell's super-cell and that super-cell's label, the broadcast
/// diagnostic, for `{out}.cell_pb.parquet`.
pub struct PbFit<'a> {
    pub input: PseudobulkInput,
    pub super_cell: &'a [usize],
    pub broadcast: &'a [usize],
}

/// `{out}.cell_pb.parquet`: `cell`, its super-cell `pb` (as cage writes
/// it) and the super-cell's label `pb_label` broadcast to it.
fn write_pb_labels(prefix: &str, cells: &[Box<str>], pb: &PbFit<'_>) -> anyhow::Result<()> {
    let label: Vec<Box<str>> = pb
        .broadcast
        .iter()
        .map(|l| format!("C{l}").into())
        .collect();
    write_cell_pb(
        prefix,
        cells,
        pb.super_cell,
        vec![("pb_label".into(), Column::Str(&label))],
    )
}

/// The run's input layout and what it was fitted on.
fn baseline_run(fit: &Fitted<'_>) -> BaselineRun {
    BaselineRun {
        input_layout: fit.input_layout,
        input: match &fit.pseudobulk {
            Some(pb) => BaselineInput::Pseudobulk(pb.input.clone()),
            None => BaselineInput::Cells,
        },
    }
}

/// One-hot rows of 0-based `labels` over `k` labels.
fn one_hot(labels: &[usize], k: usize) -> Mat {
    let mut m = Mat::zeros(labels.len(), k);
    for (i, &l) in labels.iter().enumerate() {
        m[(i, l)] = 1.0;
    }
    m
}

fn write_labels(prefix: &str, cells: &[Box<str>], labels: &[usize]) -> anyhow::Result<()> {
    let mut text = String::from("cell\tlabel\n");
    for (c, l) in cells.iter().zip(labels) {
        text.push_str(&format!("{c}\tC{l}\n"));
    }
    std::fs::write(labels_path(prefix), text)?;
    Ok(())
}

/// The column names `{name}_1..{name}_d`.
fn numbered(name: &str, d: usize) -> Vec<Box<str>> {
    (1..=d).map(|j| format!("{name}_{j}").into()).collect()
}

fn write_embedding(prefix: &str, cells: &[Box<str>], x: &Mat, name: &str) -> anyhow::Result<()> {
    let names = numbered(name, x.ncols());
    x.to_parquet_with_names(
        &embedding_path(prefix),
        (Some(cells), Some("cell")),
        Some(&names),
    )
}

/// `{prefix}.{stem}.parquet`: a method's per-gene loadings, a `feature`
/// column (the genes, as read) and `{name}_1..{name}_d`.
pub fn write_loadings(
    prefix: &str,
    stem: &str,
    genes: &[Box<str>],
    w: &Mat,
    name: &str,
) -> anyhow::Result<()> {
    mkdir_parent(prefix)?;
    let names = numbered(name, w.ncols());
    w.to_parquet_with_names(
        &format!("{prefix}.{stem}.parquet"),
        (Some(genes), Some("feature")),
        Some(&names),
    )
}

/// What was done about batch effects: δ divided out (the one deviation
/// from the original method), or nothing, under `--ignore-batch` or with a
/// single batch.
fn batch_correction(input: &SpatialInputArgs, data: &SpatialData) -> serde_json::Value {
    match &data.delta {
        Some(d) => serde_json::json!({"delta": true, "batches": d.batches}),
        None => serde_json::json!({
            "delta": false,
            "why": if input.batch.ignore_batch { "--ignore-batch" } else { "one batch" },
        }),
    }
}

/// Write every output of a fitted baseline under `input.out`.
pub fn write_run(
    input: &SpatialInputArgs,
    data: &SpatialData,
    fit: Fitted<'_>,
) -> anyhow::Result<()> {
    let prefix = input.out.as_ref();
    let n = data.cells.len();
    anyhow::ensure!(fit.labels.len() == n, "one label per cell");
    anyhow::ensure!(fit.embedding.nrows() == n, "one embedding row per cell");
    // a fit gone non-finite would write a domain per cell and an embedding
    // nothing can score
    anyhow::ensure!(
        fit.embedding.iter().all(|v| v.is_finite())
            && fit
                .membership
                .is_none_or(|m| m.iter().all(|v| v.is_finite())),
        "{}: the fit's embedding is not finite; nothing is written",
        fit.method
    );
    let k = fit.labels.iter().max().map_or(0, |&m| m + 1);
    let run = baseline_run(&fit);
    log::info!("{}: {k} clusters", fit.method);
    let one_hot_labels;
    let membership = match fit.membership {
        Some(m) => {
            anyhow::ensure!(m.nrows() == n, "one membership row per cell");
            m
        }
        None => {
            one_hot_labels = one_hot(fit.labels, k);
            &one_hot_labels
        }
    };
    mkdir_parent(prefix)?;
    write_labels(prefix, &data.cells, fit.labels)?;
    if let Some(pb) = &fit.pseudobulk {
        write_pb_labels(prefix, &data.cells, pb)?;
    }
    write_embedding(prefix, &data.cells, fit.embedding, fit.embedding_name)?;
    // the cells table is the read's (every cell, those QC dropped too)
    write_propensity_matrix(prefix, membership, &data.cells)?;
    let settings = serde_json::json!({
        "method": fit.method,
        "pinto_version": env!("CARGO_PKG_VERSION"),
        "data_files": input.data_files,
        "coord_files": input.coord_files,
        "batch_files": input.batch.batch_files,
        "auto_batch": input.batch.auto_batch,
        "cnv_clones": input.collapse.cnv_clones,
        "seed": input.seed,
        "run": run,
        "batch_correction": batch_correction(input, data),
        "n_clusters_found": k,
        "settings": fit.settings,
    });
    std::fs::write(
        settings_path(prefix),
        serde_json::to_string_pretty(&settings)? + "\n",
    )?;
    let (batch_effects, delta_cells) = data.delta.as_ref().map(|_| delta_files(prefix)).unzip();
    let meta = PintoMetadata {
        command: fit.method.into(),
        version: env!("CARGO_PKG_VERSION").into(),
        timestamp: now_secs(),
        prefix: prefix.into(),
        data_files: Some(crate::util::metadata::recorded_files(&input.data_files)),
        coord_file: Some(input.coord_files.join(",")),
        n_cells: n,
        n_features: data.genes.len(),
        n_communities: Some(membership.ncols()),
        outputs: OutputFiles {
            cells: Some(cells_table_path(prefix)),
            coord_columns: Some(data.coord_names.iter().map(|c| c.to_string()).collect()),
            propensity: Some(propensity_path(prefix)),
            cell_embedding: Some(embedding_path(prefix)),
            batch_effects,
            delta_cells,
            cell_pb: fit.pseudobulk.as_ref().map(|_| cell_pb_path(prefix)),
            covariates: data
                .delta
                .as_ref()
                .filter(|d| d.covariates)
                .map(|_| crate::util::metadata::covariate_files(prefix)),
            ..Default::default()
        },
        batch_labels: data.delta.as_ref().map(|_| BatchLabels::of(&input.batch)),
        file_batches: crate::util::metadata::file_batches(
            data.delta.as_ref().map(|_| BatchLabels::of(&input.batch)),
            &input.data_files,
        ),
        baseline: Some(run),
        ablation: data.ablation.clone(),
        ..Default::default()
    };
    meta.write(std::path::Path::new(&manifest_path(prefix)))?;
    log::info!(
        "wrote {}",
        crate::util::staging::final_str(&manifest_path(prefix))
    );
    Ok(())
}
