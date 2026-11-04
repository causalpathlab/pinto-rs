//! Parquet + TSV readers for `pinto lr-activity`.
//!
//! Reads:
//! - `{lc_prefix}.link_community.parquet` — per-edge (left_cell, right_cell, community).
//! - `{lc_prefix}.coord_pairs.parquet` — per-edge (left_batch, right_batch) when multi-batch.
//! - `{lr_pairs}` — two-column TSV/CSV of directional (ligand, receptor) gene names.

use crate::util::common::*;
use crate::util::parquet_io::row_label;
use legume_numeric::matrix::common_io::{read_lines_of_words_delim, ReadLinesOut};
use legume_numeric::matrix::membership::detect_delimiter;
use legume_numeric::matrix::parquet::peek_parquet_field_names;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;

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
pub fn read_pair_table(file_path: &str, community: bool) -> anyhow::Result<Vec<EdgeRecord>> {
    let file = File::open(file_path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let name_to_idx: HashMap<Box<str>, usize> = schema
        .get_fields()
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name().to_string().into_boxed_str(), i))
        .collect();
    let col = |name: &str| {
        name_to_idx
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("{name} missing in {file_path}"))
    };
    let (li, ri) = (col("left_cell")?, col("right_cell")?);
    let ci = if community {
        Some(col("community")?)
    } else {
        None
    };
    // Absent on a run that predates pair-graph augmentation, and on any run
    // that did not augment — which, with `--knn-expr` defaulting off, is the
    // common case rather than the exception. Both mean every pair is spatial.
    //
    // On an expression-mode run there is no adjacency at all, and every pair
    // still reads as spatial here. That is deliberate: the coordinates such a
    // run carries ARE the layout its pairs are adjacent in, and treating them
    // as expression pairs would drop all of them from the test.
    let ki = name_to_idx.get("edge_kind").copied();
    let batch = name_to_idx
        .get("left_batch")
        .copied()
        .zip(name_to_idx.get("right_batch").copied());

    let mut out = Vec::new();
    for record in reader.get_row_iter(None)? {
        let row = record?;
        let community = match ci {
            Some(c) => row.get_float(c)? as u32,
            None => 0,
        };
        let is_spatial = match ki {
            Some(k) => {
                crate::util::parquet_io::row_int_like(&row, k)?
                    == crate::util::cell_pairs::EDGE_KIND_SPATIAL as i64
            }
            None => true,
        };
        let side_batches = match batch {
            Some((lb, rb)) => Some((row_label(&row, lb)?, row_label(&row, rb)?)),
            None => None,
        };
        let batch = side_batches
            .as_ref()
            .and_then(|(l, r)| (l == r).then(|| l.clone()));
        out.push(EdgeRecord {
            left_cell: row_label(&row, li)?,
            right_cell: row_label(&row, ri)?,
            community,
            batch,
            side_batches,
            is_spatial,
        });
    }
    Ok(out)
}

/// Attach per-edge batch labels from `{prefix}.coord_pairs.parquet` when the fit
/// was multi-batch. Matches edges by position (row order). If either
/// `left_batch` or `right_batch` is missing, leaves all `batch` as `None`.
pub fn attach_batch_from_coord_pairs(
    edges: &mut [EdgeRecord],
    coord_pairs_path: &str,
) -> anyhow::Result<()> {
    let fields = peek_parquet_field_names(coord_pairs_path)?;
    let has_batch = fields.iter().any(|f| f.as_ref() == "left_batch")
        && fields.iter().any(|f| f.as_ref() == "right_batch");
    if !has_batch {
        return Ok(());
    }

    let file = File::open(coord_pairs_path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let name_to_idx: HashMap<Box<str>, usize> = schema
        .get_fields()
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name().to_string().into_boxed_str(), i))
        .collect();

    let li = *name_to_idx.get("left_batch").unwrap();
    let ri = *name_to_idx.get("right_batch").unwrap();

    for (pos, record) in reader.get_row_iter(None)?.enumerate() {
        if pos >= edges.len() {
            return Err(anyhow::anyhow!(
                "coord_pairs.parquet has more rows than link_community.parquet"
            ));
        }
        let row = record?;
        let left_b = row_label(&row, li)?;
        let right_b = row_label(&row, ri)?;
        // Edge is single-batch iff both endpoints share a label. We only test
        // within-batch activity; cross-batch edges are dropped downstream by
        // assigning the joint label `None`.
        if left_b == right_b {
            edges[pos].batch = Some(left_b);
        } else {
            edges[pos].batch = None;
        }
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
