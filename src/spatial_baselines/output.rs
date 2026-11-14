//! The outputs every spatial baseline writes, named from its `--out` prefix.

use crate::util::common::Mat;
use legume_numeric::matrix::parquet::{write_named_table, Column};

pub fn labels_path(prefix: &str) -> String {
    format!("{prefix}.labels.tsv")
}

pub fn embedding_path(prefix: &str) -> String {
    format!("{prefix}.cell_embedding.parquet")
}

pub fn manifest_path(prefix: &str) -> String {
    format!("{prefix}.baseline.json")
}

fn ensure_parent(prefix: &str) -> anyhow::Result<()> {
    if let Some(dir) = std::path::Path::new(prefix).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    Ok(())
}

/// `{prefix}.labels.tsv`: `cell<TAB>label`, labels `C1`, `C2`, ... from the
/// 0-based `labels`.
pub fn write_labels(prefix: &str, cells: &[Box<str>], labels: &[usize]) -> anyhow::Result<()> {
    anyhow::ensure!(cells.len() == labels.len(), "one label per cell");
    ensure_parent(prefix)?;
    let mut text = String::from("cell\tlabel\n");
    for (c, l) in cells.iter().zip(labels) {
        text.push_str(&format!("{c}\tC{}\n", l + 1));
    }
    std::fs::write(labels_path(prefix), text)?;
    Ok(())
}

/// `{prefix}.cell_embedding.parquet`: a `cell` column and one column per
/// embedding dimension, `{name}_1`, `{name}_2`, ...
pub fn write_embedding(
    prefix: &str,
    cells: &[Box<str>],
    x: &Mat,
    name: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(cells.len() == x.nrows(), "one embedding row per cell");
    ensure_parent(prefix)?;
    let names: Vec<Box<str>> = (1..=x.ncols()).map(|j| format!("{name}_{j}").into()).collect();
    let cols: Vec<Vec<f32>> = (0..x.ncols()).map(|j| x.column(j).iter().copied().collect()).collect();
    let columns: Vec<(Box<str>, Column<'_>)> = names
        .iter()
        .zip(&cols)
        .map(|(n, c)| (n.clone(), Column::F32(c)))
        .collect();
    write_named_table(&embedding_path(prefix), "cell", cells, &columns)
}

/// `{prefix}.baseline.json`: the method, its settings and its inputs.
pub fn write_manifest(
    prefix: &str,
    method: &str,
    settings: serde_json::Value,
    input: &super::input::SpatialInputArgs,
) -> anyhow::Result<()> {
    ensure_parent(prefix)?;
    let v = serde_json::json!({
        "method": method,
        "pinto_version": env!("CARGO_PKG_VERSION"),
        "data_files": input.data_files,
        "coord_files": input.coord_files,
        "seed": input.seed,
        "settings": settings,
        "outputs": {
            "labels": labels_path(prefix),
            "cell_embedding": embedding_path(prefix),
        },
    });
    std::fs::write(manifest_path(prefix), serde_json::to_string_pretty(&v)? + "\n")?;
    Ok(())
}
