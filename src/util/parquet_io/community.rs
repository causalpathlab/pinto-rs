//! Link-community and feature-community tables.

use super::{field_names, ints, label_column, parse_community_col_name, read_named_columns};
use crate::util::common::*;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;
use std::path::Path;

/// A cell-pair edge: (left_cell_name, right_cell_name).
type EdgePair = (Box<str>, Box<str>);

/// Read a link-community parquet: E rows, each `(left_cell, right_cell,
/// community)`. Returns `(pairs, community, total_counts)`: `pairs` and
/// `community` are parallel and hold only the ADJACENT pairs (see below),
/// while `total_counts[c]` counts community `c`'s edges over EVERY pair,
/// expression-similar ones included — a community's SIZE is a statement
/// about the fit, not about what is drawable.
/// Every pair's `community`, in file order, with **no filtering**.
///
/// [`read_link_community`] exists for plotting and keeps only spatial edges; anything
/// that zips this table against `{prefix}.latent.parquet` positionally needs all of
/// them, because that file has one row per pair regardless of edge kind.
pub fn read_link_community_labels(path: &Path) -> anyhow::Result<Vec<i64>> {
    ints(&read_named_columns(path, &["community"])?[0])
}

pub fn read_link_community(path: &Path) -> anyhow::Result<(Vec<EdgePair>, Vec<i64>, Vec<usize>)> {
    let mut pairs: Vec<EdgePair> = Vec::new();
    let mut community: Vec<i64> = Vec::new();
    let total_counts = visit_link_community(path, |l, r, c| {
        pairs.push((l.into(), r.into()));
        community.push(c);
    })?;
    Ok((pairs, community, total_counts))
}

/// Call `visit(left, right, community)` for every ADJACENT pair of a
/// link_community table, the cell names borrowed from the file's columns;
/// returns each community's edge count over every pair (see
/// [`read_link_community`]).
pub(crate) fn visit_link_community(
    path: &Path,
    mut visit: impl FnMut(&str, &str, i64),
) -> anyhow::Result<Vec<usize>> {
    let fields = field_names(path)?;
    let has = |n: &str| fields.iter().any(|f| f.as_ref() == n);
    for n in ["left_cell", "right_cell", "community"] {
        anyhow::ensure!(has(n), "{path:?}: missing {n}");
    }
    // Expression-similar pairs are dropped here, once, rather than at each
    // consumer. Every plot that reads this list is asking about ADJACENCY: the
    // mesh draws the pair as a line between two cells, the interface mode
    // walks 1- and 2-hop neighbourhoods, and the ligand-receptor overlay draws
    // an arrow along it. A pair whose cells sit at opposite ends of the
    // section would render as a chord across the whole image and would make
    // "1-hop" mean nothing.
    //
    // Absent on a run that did not augment, where every pair is adjacent.
    let mut wanted = vec!["left_cell", "right_cell", "community"];
    if has("edge_kind") {
        wanted.push("edge_kind");
    }
    let cols = read_named_columns(path, &wanted)?;
    let (left, right) = (label_column(&cols[0])?, label_column(&cols[1])?);
    let community = ints(&cols[2])?;
    let kind = if has("edge_kind") {
        Some(ints(&cols[3])?)
    } else {
        None
    };

    let mut total_counts: Vec<usize> = Vec::new();
    let mut n_dropped = 0usize;
    for (e, &c) in community.iter().enumerate() {
        if c >= 0 {
            let cu = c as usize;
            if cu >= total_counts.len() {
                total_counts.resize(cu + 1, 0);
            }
            total_counts[cu] += 1;
        }
        if kind
            .as_ref()
            .is_some_and(|k| k[e] != crate::util::cell_pairs::EDGE_KIND_SPATIAL as i64)
        {
            n_dropped += 1;
            continue;
        }
        visit(left.get(e), right.get(e), c);
    }
    if n_dropped > 0 {
        log::info!(
            "{}: showing {} adjacent pairs, hiding {} expression-similar ones",
            path.display(),
            community.len() - n_dropped,
            n_dropped
        );
    }
    Ok(total_counts)
}

/// Feature-community column names, current first, then the names older runs
/// wrote (`{prefix}.gene_topic.parquet` had `gene` and `topic`).
const FEATURE_NAME_COLS: [&str; 2] = ["feature", "gene"];
const COMMUNITY_COLS: [&str; 2] = ["community", "topic"];

/// The first of `names` the schema has.
fn first_present(
    names: &[&'static str],
    schema: &HashMap<Box<str>, usize>,
) -> Option<&'static str> {
    names.iter().copied().find(|c| schema.contains_key(*c))
}

/// Read a feature_community parquet: G × K. Returns (mat, feature_names).
///
/// `pinto lc` writes this file in *melted* form (one row per
/// feature-community pair, with columns `feature`, `community`, `mean`, `sd`,
/// `log_mean`, `log_sd`). We pivot the `mean` column back to a wide
/// G × K matrix here so downstream code can index `gt[(g, k)]` as
/// "posterior mean for feature g in community k". Communities are sorted
/// numerically by their string label (the writer emits `"0".."K-1"`).
pub fn read_feature_community(path: &Path) -> anyhow::Result<(Mat, Vec<Box<str>>)> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {path:?}"))?;
    let file = File::open(path_str)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let name_to_idx: HashMap<Box<str>, usize> = schema
        .get_fields()
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name().to_string().into_boxed_str(), i))
        .collect();

    // lc writes long format: (feature, community, mean) triples. cage /
    // cage-mcmc write `feature_dictionary.parquet` in wide format:
    // one row per feature, one column per cluster. Dispatch on schema.
    let name_col = first_present(&FEATURE_NAME_COLS, &name_to_idx);
    let community_col = first_present(&COMMUNITY_COLS, &name_to_idx);
    let (Some(name_col), Some(community_col), true) =
        (name_col, community_col, name_to_idx.contains_key("mean"))
    else {
        return read_feature_community_wide(path, &reader, &name_to_idx);
    };
    let feature_idx = name_to_idx[&Box::<str>::from(name_col)];
    let community_idx = name_to_idx[&Box::<str>::from(community_col)];
    let mean_idx = name_to_idx[&Box::<str>::from("mean")];

    let mut feature_pos: HashMap<Box<str>, usize> = HashMap::default();
    let mut feature_names: Vec<Box<str>> = Vec::new();
    let mut community_pos: HashMap<Box<str>, usize> = HashMap::default();
    let mut community_labels: Vec<Box<str>> = Vec::new();
    let mut triples: Vec<(usize, usize, f32)> = Vec::new();

    for record in reader.get_row_iter(None)? {
        let row = record?;
        let feature = row.get_string(feature_idx)?.clone().into_boxed_str();
        let community = row.get_string(community_idx)?.clone().into_boxed_str();
        let mean = row
            .get_float(mean_idx)
            .or_else(|_| row.get_double(mean_idx).map(|v| v as f32))?;
        let g_pos = *feature_pos.entry(feature.clone()).or_insert_with(|| {
            feature_names.push(feature.clone());
            feature_names.len() - 1
        });
        let c_pos = *community_pos.entry(community.clone()).or_insert_with(|| {
            community_labels.push(community.clone());
            community_labels.len() - 1
        });
        triples.push((g_pos, c_pos, mean));
    }

    // Map each label to its parsed community ID, then place column j of
    // the returned matrix at community j (matching `read_propensity`'s
    // invariant). Missing intermediate IDs are zero-filled with a warn,
    // not a hard error, so a clustering that drops a community between
    // levels still produces a usable plot.
    let mut parsed: Vec<(i64, usize)> = community_labels
        .iter()
        .enumerate()
        .map(|(i, lab)| {
            let c = parse_community_col_name(lab).ok_or_else(|| {
                anyhow::anyhow!(
                    "{path_str}: feature_community `community` label {lab:?} is not a community ID \
                     (expected \"C{{c}}\" or bare integer)."
                )
            })?;
            Ok::<_, anyhow::Error>((c, i))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    parsed.sort_by_key(|&(c, _)| c);
    if let Some(&(c0, _)) = parsed.first() {
        if c0 < 0 {
            anyhow::bail!(
                "{path_str}: feature_community has negative community ID {c0}; expected non-negative integers."
            );
        }
    }
    let max_id = parsed.last().map(|&(c, _)| c).unwrap_or(-1);
    let n_communities = (max_id + 1).max(0) as usize;
    let present: HashSet<i64> = parsed.iter().map(|&(c, _)| c).collect();
    let missing: Vec<i64> = (0..n_communities as i64)
        .filter(|c| !present.contains(c))
        .collect();
    if !missing.is_empty() {
        log::warn!(
            "{path_str}: feature_community is missing communities {missing:?} \
             (have 0..{} with gaps); zero-filling so plot indices stay aligned.",
            n_communities.saturating_sub(1),
        );
    }
    // old c_pos (insertion order) → new column = community ID
    let new_index: HashMap<usize, usize> = parsed
        .iter()
        .map(|&(c, old_i)| (old_i, c as usize))
        .collect();

    let n_features = feature_names.len();
    let mut mat = Mat::zeros(n_features, n_communities);
    for (g, c_old, v) in triples {
        let c_new = new_index[&c_old];
        mat[(g, c_new)] = v;
    }
    Ok((mat, feature_names))
}

/// Wide-format `feature_dictionary.parquet` (cage / cage-mcmc):
/// one row per feature, row-name column `feature`/`feature`, plus
/// per-cluster columns (`cluster_<k>` / `C<k>` / bare ints). Returns
/// `[G × K]` matrix and feature names; matrix column j corresponds to
/// the cluster ID parsed from the column name (zero-filled gaps).
fn read_feature_community_wide(
    path: &Path,
    reader: &SerializedFileReader<File>,
    name_to_idx: &HashMap<Box<str>, usize>,
) -> anyhow::Result<(Mat, Vec<Box<str>>)> {
    let path_str = path.to_str().unwrap_or("<non-utf8>");

    let Some(name_col) = first_present(&FEATURE_NAME_COLS, name_to_idx) else {
        anyhow::bail!(
            "{path_str}: feature dictionary is missing the row-name column \
             (expected `feature` or `gene`)"
        );
    };
    let name_col_label: Box<str> = name_col.into();
    let name_idx = name_to_idx[&name_col_label];

    let mut col_to_community: Vec<(usize, i64)> = Vec::new();
    for (name, &idx) in name_to_idx {
        if name.as_ref() == name_col_label.as_ref() {
            continue;
        }
        if let Some(c) = parse_community_col_name(name) {
            col_to_community.push((idx, c));
        }
    }
    anyhow::ensure!(
        !col_to_community.is_empty(),
        "{path_str}: feature dictionary has no community-named columns \
         (expected `cluster_<k>` / `C<k>` / `propensity_<k>`)"
    );
    let max_c = col_to_community.iter().map(|&(_, c)| c).max().unwrap();
    anyhow::ensure!(
        max_c >= 0,
        "{path_str}: feature dictionary has negative community ID {max_c}"
    );
    let n_communities = (max_c + 1) as usize;

    let mut feature_names: Vec<Box<str>> = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for record in reader.get_row_iter(None)? {
        let row = record?;
        let g = row.get_string(name_idx)?.clone().into_boxed_str();
        let mut r = vec![0.0f32; n_communities];
        for &(idx, c) in &col_to_community {
            let v = row
                .get_float(idx)
                .or_else(|_| row.get_double(idx).map(|v| v as f32))
                .unwrap_or(0.0);
            r[c as usize] = v;
        }
        feature_names.push(g);
        rows.push(r);
    }

    let n_features = feature_names.len();
    let mut mat = Mat::zeros(n_features, n_communities);
    for (i, r) in rows.into_iter().enumerate() {
        for (j, v) in r.into_iter().enumerate() {
            mat[(i, j)] = v;
        }
    }
    Ok((mat, feature_names))
}
