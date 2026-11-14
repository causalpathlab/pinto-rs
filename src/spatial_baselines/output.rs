//! The outputs every spatial baseline writes, named from its `--out`
//! prefix, as a pinto run: `pinto view`, `srt-sim-score --labels`,
//! `srt-sim-pairs --vectors`, `compare`, `lra` and `pinto run`/MCP read it
//! as they read an `lc`, `dsvd` or `cage` run.
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
use crate::util::metadata::{now_secs, OutputFiles, PintoMetadata};
use crate::util::parquet_io::{cells_table_path, write_cells_table};
use legume_numeric::matrix::parquet::{write_named_table, Column};

pub fn labels_path(prefix: &str) -> String {
    format!("{prefix}.labels.tsv")
}

pub fn embedding_path(prefix: &str) -> String {
    format!("{prefix}.cell_embedding.parquet")
}

pub fn propensity_path(prefix: &str) -> String {
    format!("{prefix}.propensity.parquet")
}

pub fn settings_path(prefix: &str) -> String {
    format!("{prefix}.baseline.json")
}

pub fn manifest_path(prefix: &str) -> String {
    format!("{prefix}.pinto.json")
}

/// What a baseline fit produced, for [`write_run`].
pub struct Fitted<'a> {
    /// The subcommand, e.g. `banksy`.
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
}

/// One-hot rows of 0-based `labels` over `k` labels.
pub fn one_hot(labels: &[usize], k: usize) -> Mat {
    let mut m = Mat::zeros(labels.len(), k);
    for (i, &l) in labels.iter().enumerate() {
        m[(i, l)] = 1.0;
    }
    m
}

fn ensure_parent(prefix: &str) -> anyhow::Result<()> {
    if let Some(dir) = std::path::Path::new(prefix).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    Ok(())
}

fn write_labels(prefix: &str, cells: &[Box<str>], labels: &[usize]) -> anyhow::Result<()> {
    let mut text = String::from("cell\tlabel\n");
    for (c, l) in cells.iter().zip(labels) {
        text.push_str(&format!("{c}\tC{l}\n"));
    }
    std::fs::write(labels_path(prefix), text)?;
    Ok(())
}

fn write_embedding(prefix: &str, cells: &[Box<str>], x: &Mat, name: &str) -> anyhow::Result<()> {
    let names: Vec<Box<str>> = (1..=x.ncols())
        .map(|j| format!("{name}_{j}").into())
        .collect();
    let cols: Vec<Vec<f32>> = (0..x.ncols())
        .map(|j| x.column(j).iter().copied().collect())
        .collect();
    let columns: Vec<(Box<str>, Column<'_>)> = names
        .iter()
        .zip(&cols)
        .map(|(n, c)| (n.clone(), Column::F32(c)))
        .collect();
    write_named_table(&embedding_path(prefix), "cell", cells, &columns)
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
    let k = fit.labels.iter().max().map_or(0, |&m| m + 1);
    let membership = match fit.membership {
        Some(m) => {
            anyhow::ensure!(m.nrows() == n, "one membership row per cell");
            m.clone()
        }
        None => one_hot(fit.labels, k),
    };
    ensure_parent(prefix)?;
    write_labels(prefix, &data.cells, fit.labels)?;
    write_embedding(prefix, &data.cells, fit.embedding, fit.embedding_name)?;
    write_propensity_matrix(prefix, &membership, &data.cells)?;
    write_cells_table(
        prefix,
        &data.cells,
        &data.xy,
        &data.coord_names,
        &data.batches,
        &vec![true; n],
    )?;
    let settings = serde_json::json!({
        "method": fit.method,
        "pinto_version": env!("CARGO_PKG_VERSION"),
        "data_files": input.data_files,
        "coord_files": input.coord_files,
        "batch_files": input.batch_files,
        "seed": input.seed,
        "settings": fit.settings,
    });
    std::fs::write(
        settings_path(prefix),
        serde_json::to_string_pretty(&settings)? + "\n",
    )?;
    let meta = PintoMetadata {
        command: fit.method.into(),
        version: env!("CARGO_PKG_VERSION").into(),
        timestamp: now_secs(),
        prefix: prefix.into(),
        data_files: Some(input.data_files.iter().map(|f| f.to_string()).collect()),
        n_cells: n,
        n_features: data.genes.len(),
        n_communities: Some(membership.ncols()),
        outputs: OutputFiles {
            cells: Some(cells_table_path(prefix)),
            coord_columns: Some(data.coord_names.iter().map(|c| c.to_string()).collect()),
            propensity: Some(propensity_path(prefix)),
            cell_embedding: Some(embedding_path(prefix)),
            ..Default::default()
        },
        ..Default::default()
    };
    meta.write(std::path::Path::new(&manifest_path(prefix)))
}
