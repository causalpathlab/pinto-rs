//! Parquet + TSV readers for `pinto lr-activity`.
//!
//! Reads:
//! - `{lc_prefix}.link_community.parquet` — per-edge (left_cell, right_cell, community).
//! - `{lc_prefix}.coord_pairs.parquet` — per-edge (left_batch, right_batch) when multi-batch.
//! - `{lr_pairs}` — two-column TSV/CSV of directional (ligand, receptor) gene names.

use crate::util::parquet_io::{field_names, ints, labels, read_named_columns};
use legume_numeric::matrix::common_io::{read_lines_of_words_delim, ReadLinesOut};
use legume_numeric::matrix::membership::detect_delimiter;
use std::path::Path;

/// Per-edge record joined across `link_community.parquet` and `coord_pairs.parquet`.
pub struct EdgeRecord {
    pub left_cell: Box<str>,
    pub right_cell: Box<str>,
    pub community: u32,
    /// If either `left_batch` or `right_batch` is absent the whole fit is
    /// treated as single-batch and this is `None`.
    pub batch: Option<Box<str>>,
    /// Each end's own batch, `(left, right)`, when the table has batch
    /// columns: a cell's batch even when the edge straddles two.
    pub side_batches: Option<(Box<str>, Box<str>)>,
    /// Whether the two cells are physically adjacent. `true` for every edge of
    /// a run whose pair graph was not augmented, and for the shared pairs of
    /// one that was, since those really are adjacent.
    pub is_spatial: bool,
}

/// Read `{prefix}.link_community.parquet` (columns: left_cell, right_cell, community).
pub fn read_link_community(file_path: &str) -> anyhow::Result<Vec<EdgeRecord>> {
    read_pair_table(file_path, true)
}

/// Read a table of cell pairs (`left_cell`, `right_cell`): a link-community
/// table, or `coord_pairs.parquet` (`community: false`, every community 0).
/// Each pair's kind comes from `edge_kind` and its batch from `left_batch`
/// / `right_batch` (shared by both ends) when the table has those columns.
/// Read column-wise, each column at whatever width its writer chose.
pub fn read_pair_table(file_path: &str, community: bool) -> anyhow::Result<Vec<EdgeRecord>> {
    let path = Path::new(file_path);
    let fields = field_names(path)?;
    let has = |n: &str| fields.iter().any(|f| f.as_ref() == n);
    // `edge_kind` is absent on a run that predates pair-graph augmentation,
    // and on any run that did not augment, which, with `--knn-expr`
    // defaulting off, is the common case rather than the exception. Both
    // mean every pair is spatial.
    //
    // On an expression-mode run there is no adjacency at all, and every pair
    // still reads as spatial here. That is deliberate: the coordinates such a
    // run carries ARE the layout its pairs are adjacent in, and treating them
    // as expression pairs would drop all of them from the test.
    let with_kind = has("edge_kind");
    let with_batch = has("left_batch") && has("right_batch");
    let mut names = vec!["left_cell", "right_cell"];
    if community {
        names.push("community");
    }
    if with_kind {
        names.push("edge_kind");
    }
    if with_batch {
        names.extend(["left_batch", "right_batch"]);
    }
    // The Arrow columns go once each is converted, before the records are
    // built.
    let (left, right, communities, spatial, sides) = {
        let cols = read_named_columns(path, &names)?;
        let mut cols = cols.iter();
        let mut next = || cols.next().expect("one column per name");
        let left = labels(next())?;
        let right = labels(next())?;
        let n = left.len();
        let communities: Vec<u32> = if community {
            // A negative id is no community; it reads as 0, as it always has.
            ints(next())?
                .into_iter()
                .map(|c| c.clamp(0, i64::from(u32::MAX)) as u32)
                .collect()
        } else {
            vec![0; n]
        };
        let spatial: Vec<bool> = if with_kind {
            ints(next())?
                .into_iter()
                .map(|k| k == i64::from(crate::util::cell_pairs::EDGE_KIND_SPATIAL))
                .collect()
        } else {
            vec![true; n]
        };
        let sides: Vec<Option<(Box<str>, Box<str>)>> = if with_batch {
            let (l, r) = (labels(next())?, labels(next())?);
            l.into_iter().zip(r).map(Some).collect()
        } else {
            vec![None; n]
        };
        (left, right, communities, spatial, sides)
    };
    Ok(left
        .into_iter()
        .zip(right)
        .zip(communities)
        .zip(spatial)
        .zip(sides)
        .map(
            |((((left_cell, right_cell), community), is_spatial), side_batches)| {
                let batch = side_batches
                    .as_ref()
                    .and_then(|(l, r)| (l == r).then(|| l.clone()));
                EdgeRecord {
                    left_cell,
                    right_cell,
                    community,
                    batch,
                    side_batches,
                    is_spatial,
                }
            },
        )
        .collect())
}

/// Attach per-edge batch labels from `{prefix}.coord_pairs.parquet` when the fit
/// was multi-batch. Matches edges by position (row order). If either
/// `left_batch` or `right_batch` is missing, leaves all `batch` as `None`.
pub fn attach_batch_from_coord_pairs(
    edges: &mut [EdgeRecord],
    coord_pairs_path: &str,
) -> anyhow::Result<()> {
    let path = Path::new(coord_pairs_path);
    let fields = field_names(path)?;
    let has = |n: &str| fields.iter().any(|f| f.as_ref() == n);
    if !(has("left_batch") && has("right_batch")) {
        return Ok(());
    }
    let cols = read_named_columns(path, &["left_batch", "right_batch"])?;
    let (left, right) = (labels(&cols[0])?, labels(&cols[1])?);
    anyhow::ensure!(
        left.len() <= edges.len(),
        "coord_pairs.parquet has more rows than link_community.parquet"
    );
    // Edge is single-batch iff both endpoints share a label. We only test
    // within-batch activity; cross-batch edges are dropped downstream by
    // assigning the joint label `None`.
    for (edge, (l, r)) in edges.iter_mut().zip(left.into_iter().zip(right)) {
        edge.batch = (l == r).then_some(l);
    }
    Ok(())
}

/// Header keywords used to detect and skip a title row on the LR pairs file.
const LR_HEADER_KEYWORDS: &[&str] = &["ligand", "source", "receptor", "target"];

/// Parse a directional LR pairs file (two columns: ligand, receptor).
/// Delimiter is auto-detected from extension. Skips empty lines and lines with
/// fewer than two non-empty tokens. Trims whitespace.
pub fn read_lr_pairs(file_path: &str) -> anyhow::Result<Vec<(Box<str>, Box<str>)>> {
    let delim = detect_delimiter(file_path);
    let ReadLinesOut { lines, .. } = read_lines_of_words_delim(file_path, delim, -1)?;
    let mut out = Vec::with_capacity(lines.len());
    for row in lines {
        let cleaned: Vec<Box<str>> = row
            .into_iter()
            .map(|s| s.trim().to_string().into_boxed_str())
            .filter(|s| !s.is_empty())
            .collect();
        if cleaned.len() < 2 {
            continue;
        }
        let looks_header = out.is_empty()
            && [cleaned[0].as_ref(), cleaned[1].as_ref()]
                .iter()
                .any(|cell| {
                    LR_HEADER_KEYWORDS
                        .iter()
                        .any(|&kw| cell.eq_ignore_ascii_case(kw))
                });
        if looks_header {
            continue;
        }
        out.push((cleaned[0].clone(), cleaned[1].clone()));
    }
    Ok(out)
}
