//! What the integration tests share.

use std::path::Path;

/// A run of `n` cells on a grid, two communities split down the middle,
/// written as pinto writes them: its cells table, its propensity and its
/// manifest, at `{dir}/run`. The prefix.
pub fn write_run(dir: &Path, n: usize) -> String {
    use legume_numeric::matrix::traits::IoOps;
    let prefix = dir.join("run").to_string_lossy().into_owned();
    let side = (n as f64).sqrt().ceil() as usize;
    let names: Vec<Box<str>> = (0..n).map(|i| format!("c{i}").into()).collect();
    let xy = nalgebra::DMatrix::<f32>::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i % side) as f32
        } else {
            (i / side) as f32
        }
    });
    let cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    xy.to_parquet_with_names(
        &format!("{prefix}.cells.parquet"),
        (Some(&names), Some("cell")),
        Some(&cols),
    )
    .unwrap();
    let prop = nalgebra::DMatrix::<f32>::from_fn(n, 2, |i, j| {
        let left = i % side < side / 2;
        if left == (j == 0) {
            0.9
        } else {
            0.1
        }
    });
    let cols: Vec<Box<str>> = vec!["C0".into(), "C1".into()];
    prop.to_parquet_with_names(
        &format!("{prefix}.propensity.parquet"),
        (Some(&names), Some("cell")),
        Some(&cols),
    )
    .unwrap();
    let manifest = serde_json::json!({
        "command": "lc",
        "version": "0",
        "timestamp": "",
        "prefix": prefix,
        "n_cells": n,
        "n_features": 0,
        "outputs": {
            "cells": format!("{prefix}.cells.parquet"),
            "propensity": format!("{prefix}.propensity.parquet"),
        },
    });
    std::fs::write(format!("{prefix}.pinto.json"), manifest.to_string()).unwrap();
    prefix
}
