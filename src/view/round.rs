//! A lupin round loaded for drawing: its clusters and its cell types as
//! two groupings of the run's cells, each with marker rates.
//!
//! Only lupin's files are read: the round's `clusters.parquet` (cell →
//! cluster id), `argmax.tsv` (cell → cell type) and `cluster_expression`
//! (genes × the first round's clusters). A merged cluster's expression is
//! the mix of its first-round clusters, weighted by their cells in it.

use super::data::{Communities, Geometry, NO_CLUSTER};
use super::lupin::{self, Panel, Reviewed};
use super::markers::FeatureRates;
use super::{Base, Level};
use crate::util::common::*;
use crate::util::parquet_io::{read_keyed_column, read_labelled_matrix};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct Round {
    pub path: PathBuf,
    /// Position of the annotated level in the run's levels.
    pub level: usize,
    /// Per cluster id: size, label, calls and history.
    pub reviewed: BTreeMap<i64, Reviewed>,
    /// Group index → cluster id, for [`Round::clusters`].
    pub ids: Vec<i64>,
    /// Cells grouped by cluster, named `K{id} label`.
    pub clusters: Level,
    /// Cells grouped by cell type.
    pub types: Level,
    /// The marker panel the next round starts from.
    pub panel: Option<Panel>,
    /// FDR level a call must pass.
    pub alpha: f32,
}

impl Round {
    pub fn load(path: &Path, level: usize, base: &Base) -> anyhow::Result<Round> {
        let geom = &base.geom;
        let m = lupin::read_json(path)
            .ok_or_else(|| anyhow::anyhow!("cannot read {}", path.display()))?;
        let a = m
            .get("annotate")
            .ok_or_else(|| anyhow::anyhow!("{}: not an annotated round", path.display()))?;
        let field = |v: &serde_json::Value, key: &str| {
            v.get(key)
                .and_then(|p| p.as_str())
                .map(|p| lupin::resolve(path, p))
        };
        let reviewed = lupin::review(path)?;

        // Cell → cluster id.
        let clusters_file = m
            .get("cluster")
            .and_then(|c| field(c, "clusters"))
            .ok_or_else(|| anyhow::anyhow!("{}: no cluster.clusters", path.display()))?;
        let cluster_of = cell_ids(&clusters_file, geom)?;
        let mut ids: Vec<i64> = reviewed.keys().copied().collect();
        for &c in cluster_of.iter().flatten() {
            if let Err(at) = ids.binary_search(&c) {
                ids.insert(at, c);
            }
        }
        let group_of = |c: Option<i64>| {
            c.and_then(|c| ids.binary_search(&c).ok())
                .map_or(NO_CLUSTER, |g| g as u16)
        };
        let cluster_group: Vec<u16> = cluster_of.iter().map(|&c| group_of(c)).collect();
        let cluster_names: Vec<Box<str>> = ids
            .iter()
            .map(|id| {
                let label = reviewed
                    .get(id)
                    .and_then(|r| r.digest.label.as_deref())
                    .unwrap_or("–");
                format!("K{id} {label}").into()
            })
            .collect();

        // Cell → cell type.
        let argmax = field(a, "argmax")
            .ok_or_else(|| anyhow::anyhow!("{}: no annotate.argmax", path.display()))?;
        let (type_group, type_names) = cell_types(&argmax, geom)?;

        // Marker rates, when the round carries the expression tables.
        let rates = match (
            field(a, "cluster_expression"),
            field(a, "expression_clusters"),
        ) {
            (Some(expr), Some(first)) => {
                match mixed_rates(
                    &expr,
                    &first,
                    geom,
                    &[&cluster_group, &type_group],
                    &[ids.len(), type_names.len()],
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        warn!("{}: no marker rates: {e}", path.display());
                        vec![None, None]
                    }
                }
            }
            _ => vec![None, None],
        };
        let mut rates = rates.into_iter();
        let panel = field(a, "markers").and_then(|p| Panel::read(&p).ok());
        let alpha = a
            .pointer("/settings/enrichment/fdr_alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1) as f32;

        let cluster_ids = ids.iter().map(|id| format!("K{id}").into()).collect();
        Ok(Round {
            path: path.to_path_buf(),
            level,
            reviewed,
            ids,
            clusters: base.grouping(
                level,
                Communities::from_groups("clusters", cluster_group, cluster_names)
                    .with_ids(cluster_ids),
                rates.next().flatten(),
            ),
            types: base.grouping(
                level,
                Communities::from_groups("cell types", type_group, type_names),
                rates.next().flatten(),
            ),
            panel,
            alpha,
        })
    }

    pub fn name(&self) -> String {
        round_name(&self.path)
    }

    /// The group index of cluster `id` in [`Round::clusters`].
    pub fn group(&self, id: i64) -> Option<usize> {
        self.ids.binary_search(&id).ok()
    }

    /// Cluster `id`'s current label.
    pub fn label(&self, id: i64) -> Option<&str> {
        self.reviewed.get(&id)?.digest.label.as_deref()
    }

    /// Every label a decision might use: the panel's types, then the
    /// round's labels, each once.
    pub fn known_labels(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let panel = self
            .panel
            .iter()
            .flat_map(|p| p.types.iter().map(|(t, _)| t.as_str()));
        let labels = self
            .reviewed
            .values()
            .filter_map(|r| r.digest.label.as_deref());
        for l in panel.chain(labels) {
            if !out
                .iter()
                .any(|o| lupin::label_key(o) == lupin::label_key(l))
            {
                out.push(l.to_string());
            }
        }
        out
    }
}

/// A round's name: its manifest's file name without `.lupin.json`.
pub fn round_name(path: &Path) -> String {
    lupin::file_name(path)
        .trim_end_matches(".lupin.json")
        .to_string()
}

/// Per geometry cell, its cluster id in `path` (`None` when absent or NaN).
fn cell_ids(path: &Path, geom: &Geometry) -> anyhow::Result<Vec<Option<i64>>> {
    let (cells, values) = read_keyed_column(path, "cluster")?;
    let mut out = vec![None; geom.n()];
    for (name, v) in cells.iter().zip(values) {
        if let (Some(&i), true) = (geom.index.get(name), v.is_finite()) {
            out[i] = Some(v as i64);
        }
    }
    Ok(out)
}

/// Per geometry cell, its cell type's index in the returned names, read
/// from an `argmax.tsv` (`cell<TAB>cell_type<TAB>probability`).
fn cell_types(path: &Path, geom: &Geometry) -> anyhow::Result<(Vec<u16>, Vec<Box<str>>)> {
    let text =
        std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let mut names: Vec<Box<str>> = Vec::new();
    let mut at: HashMap<Box<str>, u16> = HashMap::default();
    let mut group = vec![NO_CLUSTER; geom.n()];
    for line in text.lines().skip(1) {
        let mut f = line.split('\t');
        let (Some(cell), Some(label)) = (f.next(), f.next()) else {
            continue;
        };
        let Some(&i) = geom.index.get(cell) else {
            continue;
        };
        // lupin's word for a cell no type claimed: drawn as ungrouped.
        if label.eq_ignore_ascii_case("unassigned") {
            continue;
        }
        let g = *at.entry(label.into()).or_insert_with(|| {
            names.push(label.into());
            (names.len() - 1) as u16
        });
        group[i] = g;
    }
    // Alphabetical, so a type keeps its colour from round to round.
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by(|&a, &b| names[a].cmp(&names[b]));
    let mut rank = vec![0u16; names.len()];
    for (r, &o) in order.iter().enumerate() {
        rank[o] = r as u16;
    }
    for g in group.iter_mut().filter(|g| **g != NO_CLUSTER) {
        *g = rank[*g as usize];
    }
    let names = order.into_iter().map(|o| names[o].clone()).collect();
    Ok((group, names))
}

/// Marker rates of each grouping in `groups` (per cell, `k` groups each):
/// the first round's per-cluster expression (`expr`: genes × `K{id}`),
/// mixed by how many of a group's cells each first-round cluster holds.
fn mixed_rates(
    expr: &Path,
    first: &Path,
    geom: &Geometry,
    groups: &[&[u16]],
    k: &[usize],
) -> anyhow::Result<Vec<Option<FeatureRates>>> {
    let table = read_labelled_matrix(expr)?;
    let column: HashMap<i64, usize> = table
        .cols
        .iter()
        .enumerate()
        .filter_map(|(j, c)| Some((c.strip_prefix('K')?.parse().ok()?, j)))
        .collect();
    let first = cell_ids(first, geom)?;
    let n_first = table.mat.ncols();
    let g = table.mat.nrows();
    Ok(groups
        .iter()
        .zip(k)
        .map(|(group, &k)| {
            // Cells of first-round cluster j in group c.
            let mut count = Mat::zeros(n_first, k);
            for (i, &c) in group.iter().enumerate() {
                if let (Some(&j), true) = (first[i].and_then(|f| column.get(&f)), (c as usize) < k)
                {
                    count[(j, c as usize)] += 1.;
                }
            }
            for c in 0..k {
                let total: f32 = count.column(c).sum();
                if total > 0. {
                    count.column_mut(c).scale_mut(1. / total);
                }
            }
            let rates = &table.mat * &count;
            debug_assert_eq!(rates.nrows(), g);
            Some(FeatureRates {
                names: table.rows.clone(),
                rates,
            })
        })
        .collect())
}
