//! Parquet readers shared across pinto subcommands.
//!
//! Thin wrappers over `legume_numeric::matrix` primitives. The
//! pinto-specific piece is [`read_cells_from_coord_pairs`], which
//! dedupes `coord_pairs.parquet` into a per-cell table `(name, x, y)`,
//! with the batch and the rest of what the viewer draws in its build.

use crate::util::common::*;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use legume_numeric::matrix::parquet::peek_parquet_field_names;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use parquet::record::{Row, RowAccessor};
use std::fs::File;
use std::path::Path;

mod community;
#[cfg(feature = "view")]
pub(crate) use community::visit_link_community;
pub use community::{read_feature_community, read_link_community, read_link_community_labels};

/// Read a column as a label string, regardless of physical type.
///
/// `pinto svd --coord` writes whatever coordinate columns the user
/// supplied as FLOAT, including a "batch" column that's actually a
/// numeric batch index. The plot pipeline still wants a `Box<str>`
/// label, so accept any numeric/string type and stringify.
pub(crate) fn row_label(row: &Row, idx: usize) -> anyhow::Result<Box<str>> {
    if let Ok(s) = row.get_string(idx) {
        return Ok(s.clone().into_boxed_str());
    }
    if let Ok(v) = row.get_float(idx) {
        return Ok(stringify_numeric(v as f64));
    }
    if let Ok(v) = row.get_double(idx) {
        return Ok(stringify_numeric(v));
    }
    if let Ok(v) = row.get_long(idx) {
        return Ok(v.to_string().into_boxed_str());
    }
    if let Ok(v) = row.get_int(idx) {
        return Ok(v.to_string().into_boxed_str());
    }
    anyhow::bail!("column {idx} has no string/numeric value");
}

fn stringify_numeric(v: f64) -> Box<str> {
    if v.is_finite() && v.fract() == 0.0 {
        format!("{}", v as i64).into_boxed_str()
    } else {
        format!("{v}").into_boxed_str()
    }
}

/// Read a column that holds a small integer code, whatever width the writer
/// chose for it.
///
/// pinto writes these from more than one place and the parquet type has not
/// always agreed: `community` is a float in one table, `edge_kind` an int in
/// another. Rather than have each reader guess the order to try, they all come
/// through here.
pub(crate) fn row_int_like(row: &Row, idx: usize) -> anyhow::Result<i64> {
    if let Ok(v) = row.get_int(idx) {
        return Ok(v as i64);
    }
    if let Ok(v) = row.get_long(idx) {
        return Ok(v);
    }
    if let Ok(v) = row.get_float(idx) {
        return Ok(v as i64);
    }
    if let Ok(v) = row.get_double(idx) {
        return Ok(v as i64);
    }
    anyhow::bail!("column {idx} is not an integer-like type")
}

/// Named columns of a parquet file, read column-wise through Arrow: each as
/// the chunks (record batches) it came in, in the order of `names`.
fn read_named_columns(path: &Path, names: &[&str]) -> anyhow::Result<Vec<Vec<ArrayRef>>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let schema = builder.parquet_schema();
    let leaves = names
        .iter()
        .map(|n| {
            (0..schema.num_columns())
                .find(|&i| schema.column(i).name() == *n)
                .ok_or_else(|| anyhow::anyhow!("column {n} missing in {path:?}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mask = ProjectionMask::leaves(schema, leaves);
    let reader = builder
        .with_projection(mask)
        .with_batch_size(1 << 16)
        .build()?;
    let mut out = vec![Vec::new(); names.len()];
    for batch in reader {
        let batch = batch?;
        for (k, n) in names.iter().enumerate() {
            let col = batch
                .column_by_name(n)
                .ok_or_else(|| anyhow::anyhow!("column {n} missing in {path:?}"))?;
            // The row readers these replace refused missing values; so do
            // these, rather than read them as 0 or "".
            anyhow::ensure!(
                col.null_count() == 0,
                "column {n} in {path:?} has {} missing values",
                col.null_count()
            );
            out[k].push(col.clone());
        }
    }
    Ok(out)
}

/// A column's values as labels, borrowed from the Arrow arrays when they
/// are strings (no string per value), written as [`row_label`] writes them
/// when they are numbers.
enum LabelColumn<'a> {
    Borrowed(Vec<&'a str>),
    Owned(Vec<Box<str>>),
}

impl LabelColumn<'_> {
    fn get(&self, i: usize) -> &str {
        match self {
            LabelColumn::Borrowed(v) => v[i],
            LabelColumn::Owned(v) => &v[i],
        }
    }

    fn len(&self) -> usize {
        match self {
            LabelColumn::Borrowed(v) => v.len(),
            LabelColumn::Owned(v) => v.len(),
        }
    }
}

fn label_column(chunks: &[ArrayRef]) -> anyhow::Result<LabelColumn<'_>> {
    let mut borrowed = Vec::new();
    for a in chunks {
        if let Some(s) = a.as_string_opt::<i32>() {
            borrowed.extend(s.iter().map(Option::unwrap_or_default));
        } else if let Some(s) = a.as_string_opt::<i64>() {
            borrowed.extend(s.iter().map(Option::unwrap_or_default));
        } else if let Some(s) = a.as_string_view_opt() {
            borrowed.extend(s.iter().map(Option::unwrap_or_default));
        } else {
            let owned = numeric(chunks, stringify_numeric)?;
            return Ok(LabelColumn::Owned(owned));
        }
    }
    Ok(LabelColumn::Borrowed(borrowed))
}

/// A column as owned labels.
fn labels(chunks: &[ArrayRef]) -> anyhow::Result<Vec<Box<str>>> {
    let col = label_column(chunks)?;
    Ok((0..col.len()).map(|i| Box::from(col.get(i))).collect())
}

/// A numeric column, whichever width it was stored in, as `T`.
fn numeric<T>(chunks: &[ArrayRef], cast: impl Fn(f64) -> T) -> anyhow::Result<Vec<T>> {
    use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
    let mut out = Vec::new();
    for a in chunks {
        if let Some(v) = a.as_primitive_opt::<Float32Type>() {
            out.extend(v.values().iter().map(|&x| cast(x as f64)));
        } else if let Some(v) = a.as_primitive_opt::<Float64Type>() {
            out.extend(v.values().iter().map(|&x| cast(x)));
        } else if let Some(v) = a.as_primitive_opt::<Int32Type>() {
            out.extend(v.values().iter().map(|&x| cast(x as f64)));
        } else if let Some(v) = a.as_primitive_opt::<Int64Type>() {
            out.extend(v.values().iter().map(|&x| cast(x as f64)));
        } else {
            anyhow::bail!("column of type {} is not numeric", a.data_type())
        }
    }
    Ok(out)
}

fn floats(chunks: &[ArrayRef]) -> anyhow::Result<Vec<f32>> {
    numeric(chunks, |v| v as f32)
}

fn ints(chunks: &[ArrayRef]) -> anyhow::Result<Vec<i64>> {
    numeric(chunks, |v| v as i64)
}

/// Column names of a parquet file.
fn field_names(path: &Path) -> anyhow::Result<Vec<Box<str>>> {
    peek_parquet_field_names(
        path.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {path:?}"))?,
    )
}

/// A table whose first column labels the rows and whose other columns are
/// numbers, read column-wise: what `Mat::from_parquet` reads, faster.
pub(crate) fn read_labelled_matrix(path: &Path) -> anyhow::Result<MatWithNames<Mat>> {
    let fields = field_names(path)?;
    anyhow::ensure!(!fields.is_empty(), "{path:?}: no columns");
    let names: Vec<&str> = fields.iter().map(|f| f.as_ref()).collect();
    let cols = read_named_columns(path, &names)?;
    let rows = labels(&cols[0])?;
    // Columns one after another are the column-major layout `Mat` keeps.
    let mut flat = Vec::with_capacity(rows.len() * (cols.len() - 1));
    for chunks in &cols[1..] {
        let column = floats(chunks)?;
        anyhow::ensure!(column.len() == rows.len(), "{path:?}: ragged columns");
        flat.extend(column);
    }
    let mat = Mat::from_vec(rows.len(), cols.len() - 1, flat);
    Ok(MatWithNames {
        rows,
        cols: fields[1..].to_vec(),
        mat,
    })
}

/// Row keys, and each requested float column.
pub type KeyedColumns = (Vec<Box<str>>, Vec<Vec<f32>>);

/// Several float columns of a table keyed by its first column, in one pass:
/// the keys and each named column, in order (e.g. a lupin round's `cell`
/// and `cluster`, with NaN for none).
pub fn read_keyed_columns(path: &Path, columns: &[&str]) -> anyhow::Result<KeyedColumns> {
    let fields = field_names(path)?;
    let key = fields
        .first()
        .ok_or_else(|| anyhow::anyhow!("{path:?}: no columns"))?;
    let mut names = vec![key.as_ref()];
    names.extend_from_slice(columns);
    let cols = read_named_columns(path, &names)?;
    let rows = labels(&cols[0])?;
    let values = cols[1..]
        .iter()
        .map(|c| floats(c))
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(
        values.iter().all(|v| v.len() == rows.len()),
        "{path:?}: ragged columns"
    );
    Ok((rows, values))
}

/// Write `{prefix}.cells.parquet`: every cell the run read, before QC, with
/// its coordinates (the internal `batch` offset column left out, as in
/// `coord_pairs`), its batch label when there is more than one, and
/// `in_graph` (1 if the cell became a graph node, 0 if QC dropped it).
///
/// `coord_pairs` only names cells that have an edge; this table is the
/// run's own record of where every cell is.
pub fn write_cells_table(
    prefix: &str,
    names: &[Box<str>],
    coords: &Mat,
    coord_names: &[Box<str>],
    batches: &[Box<str>],
    in_graph: &[bool],
) -> anyhow::Result<()> {
    let n = names.len();
    anyhow::ensure!(
        coords.nrows() == n && batches.len() == n && in_graph.len() == n,
        "cells table: {n} names, {} coordinate rows, {} batch labels, {} flags",
        coords.nrows(),
        batches.len(),
        in_graph.len()
    );
    let coord_cols: Vec<(usize, Box<str>)> = coord_names
        .iter()
        .enumerate()
        .filter(|(_, name)| name.as_ref() != "batch")
        .map(|(j, name)| (j, name.clone()))
        .collect();
    let values: Vec<Vec<f32>> = coord_cols
        .iter()
        .map(|&(j, _)| coords.column(j).iter().copied().collect())
        .collect();
    let flags: Vec<i32> = in_graph.iter().map(|&k| i32::from(k)).collect();

    let mut columns: Vec<(Box<str>, Column<'_>)> = coord_cols
        .iter()
        .zip(&values)
        .map(|((_, name), v)| (name.clone(), Column::F32(v)))
        .collect();
    let distinct: HashSet<&str> = batches.iter().map(|b| b.as_ref()).collect();
    if distinct.len() > 1 {
        columns.push(("batch".into(), Column::Str(batches)));
    }
    columns.push(("in_graph".into(), Column::I32(&flags)));
    write_named_table(&cells_table_path(prefix), "cell", names, &columns)
}

/// Where [`write_cells_table`] writes for `prefix`.
pub fn cells_table_path(prefix: &str) -> String {
    format!("{prefix}.cells.parquet")
}

/// Read a table written by [`write_cells_table`]. `coord_columns` names the
/// `[x, y]` columns (from the manifest); without it, the first two numeric
/// columns other than `in_graph`.
pub fn read_cells_table(
    path: &Path,
    coord_columns: Option<&[String]>,
) -> anyhow::Result<CellTable> {
    let fields = field_names(path)?;
    anyhow::ensure!(
        fields.len() >= 3,
        "{path:?}: expected a cell column and two coordinates"
    );
    let (x, y) = match coord_columns {
        Some(cols) if cols.len() >= 2 => (cols[0].clone(), cols[1].clone()),
        _ => {
            let coords: Vec<&str> = fields[1..]
                .iter()
                .map(|f| f.as_ref())
                .filter(|f| !matches!(*f, "batch" | "in_graph"))
                .collect();
            anyhow::ensure!(
                coords.len() >= 2,
                "{path:?}: fewer than two coordinate columns"
            );
            (coords[0].to_string(), coords[1].to_string())
        }
    };
    let has = |n: &str| fields.iter().any(|f| f.as_ref() == n);
    let mut wanted = vec![fields[0].as_ref(), x.as_str(), y.as_str()];
    for extra in ["batch", "in_graph"] {
        // Only the viewer reads these.
        if cfg!(feature = "view") && has(extra) {
            wanted.push(extra);
        }
    }
    let cols = read_named_columns(path, &wanted)?;
    let names = labels(&cols[0])?;
    let (xs, ys) = (floats(&cols[1])?, floats(&cols[2])?);
    #[cfg(feature = "view")]
    let (batches, in_graph, index) = {
        let column = |n: &str| wanted.iter().position(|w| *w == n).map(|k| &cols[k]);
        let batches = column("batch").map(|c| labels(c)).transpose()?;
        let in_graph = column("in_graph")
            .map(|c| ints(c).map(|v| v.into_iter().map(|k| k != 0).collect::<Vec<_>>()))
            .transpose()?
            .filter(|v| v.iter().any(|&k| !k));
        let index = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        (batches, in_graph, index)
    };
    Ok(CellTable {
        names,
        coords: xs.into_iter().zip(ys).collect(),
        #[cfg(feature = "view")]
        batches,
        #[cfg(feature = "view")]
        index,
        #[cfg(feature = "view")]
        in_graph,
        #[cfg(feature = "view")]
        coord_col_names: vec![x.into_boxed_str(), y.into_boxed_str()],
    })
}

/// One row per cell. `batch` is `None` if the fit was single-batch
/// (i.e. `coord_pairs.parquet` lacks a `left_batch` column). Everything past
/// the names and coordinates is for the viewer, and read only in its build.
pub struct CellTable {
    /// Cell barcode, unique and stable-ordered.
    pub names: Vec<Box<str>>,
    /// (x, y) coordinate per cell.
    pub coords: Vec<(f32, f32)>,
    /// Optional batch label per cell.
    #[cfg(feature = "view")]
    pub batches: Option<Vec<Box<str>>>,
    /// `name → index into names/coords/batches`
    #[cfg(feature = "view")]
    pub index: HashMap<Box<str>, usize>,
    /// Whether each cell became a graph node (`false`: QC dropped it, or it
    /// has no neighbours); `None` when every cell did.
    #[cfg(feature = "view")]
    pub in_graph: Option<Vec<bool>>,
    /// Bare coordinate column names (without `left_`/`right_` prefix).
    /// Pass-through from `coord_pairs.parquet` so downstream readers
    /// (e.g. `read_propensity`) can exclude coord trailers that
    /// `pinto prop` may append to `.propensity.parquet`.
    #[cfg(feature = "view")]
    pub coord_col_names: Vec<Box<str>>,
}

/// Read `{prefix}.coord_pairs.parquet`, union left+right, dedupe.
///
/// `coord_columns`, when supplied (typically from `.pinto.json`'s
/// `outputs.coord_columns`), names the bare coord basenames in `(x, y)`
/// order — e.g. `["pxl_row_in_fullres", "pxl_col_in_fullres"]`. Pass
/// `None` to fall back to the legacy auto-discovery (first two paired
/// `left_*` / `right_*` columns by schema order), which is correct for
/// pinto-written files but brittle when users splice extra columns in.
pub fn read_cells_from_coord_pairs(
    path: &Path,
    coord_columns: Option<&[String]>,
) -> anyhow::Result<CellTable> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {path:?}"))?;
    let fields = peek_parquet_field_names(path_str)?;

    let (x_bare, y_bare) = match coord_columns {
        Some(cols) if cols.len() >= 2 => (cols[0].clone(), cols[1].clone()),
        Some(_) | None => {
            // Legacy fallback: scan for paired left_* columns and pick
            // the first two as (x, y). Excludes the `left_cell` /
            // `left_batch` non-coord columns.
            let left_coords: Vec<Box<str>> = fields
                .iter()
                .filter(|f| {
                    f.starts_with("left_")
                        && f.as_ref() != "left_cell"
                        && f.as_ref() != "left_batch"
                })
                .cloned()
                .collect();
            if left_coords.len() < 2 {
                anyhow::bail!(
                    "coord_pairs.parquet {path:?} has fewer than 2 coordinate columns \
                     (needs left_x + left_y pair) and .pinto.json carried no \
                     coord_columns hint. Was this run fit without --coord?"
                );
            }
            (
                strip_left(&left_coords[0]).to_string(),
                strip_left(&left_coords[1]).to_string(),
            )
        }
    };

    let x_col_left: Box<str> = format!("left_{x_bare}").into_boxed_str();
    let y_col_left: Box<str> = format!("left_{y_bare}").into_boxed_str();
    let x_col_right: Box<str> = format!("right_{x_bare}").into_boxed_str();
    let y_col_right: Box<str> = format!("right_{y_bare}").into_boxed_str();

    // Only the viewer reads the batches.
    let has_batch = cfg!(feature = "view")
        && fields.iter().any(|f| f.as_ref() == "left_batch")
        && fields.iter().any(|f| f.as_ref() == "right_batch");

    // Column-wise: millions of pairs read in a fraction of the time the
    // row iterator takes.
    let mut wanted = vec![
        "left_cell",
        "right_cell",
        &*x_col_left,
        &*y_col_left,
        &*x_col_right,
        &*y_col_right,
    ];
    if has_batch {
        wanted.extend(["left_batch", "right_batch"]);
    }
    let cols = read_named_columns(path, &wanted)?;
    // Borrowed from the Arrow arrays: a string is made only per new cell,
    // not per edge endpoint.
    let (left, right) = (label_column(&cols[0])?, label_column(&cols[1])?);
    let (lx, ly, rx, ry) = (
        floats(&cols[2])?,
        floats(&cols[3])?,
        floats(&cols[4])?,
        floats(&cols[5])?,
    );
    let (lb, rb) = if has_batch {
        (Some(label_column(&cols[6])?), Some(label_column(&cols[7])?))
    } else {
        (None, None)
    };

    let mut index: HashMap<Box<str>, usize> = HashMap::default();
    let mut names: Vec<Box<str>> = Vec::new();
    let mut coords: Vec<(f32, f32)> = Vec::new();
    let mut batches: Vec<Box<str>> = Vec::new();

    for e in 0..left.len() {
        for (col, x, y, b) in [(&left, lx[e], ly[e], &lb), (&right, rx[e], ry[e], &rb)] {
            let name = col.get(e);
            if index.contains_key(name) {
                continue;
            }
            let name: Box<str> = name.into();
            index.insert(name.clone(), names.len());
            names.push(name);
            coords.push((x, y));
            if let Some(b) = b {
                batches.push(b.get(e).into());
            }
        }
    }

    Ok(CellTable {
        names,
        coords,
        #[cfg(feature = "view")]
        batches: has_batch.then_some(batches),
        #[cfg(feature = "view")]
        index,
        #[cfg(feature = "view")]
        in_graph: None,
        #[cfg(feature = "view")]
        coord_col_names: vec![x_bare.into_boxed_str(), y_bare.into_boxed_str()],
    })
}

fn strip_left(col: &str) -> &str {
    col.strip_prefix("left_").unwrap_or(col)
}

/// Read a propensity parquet. Schemas differ across pinto variants:
///
/// - `pinto lc`: columns named `"0"`, `"1"`, …, `"{K-1}"` plus optional
///   `entropy`. No `cluster`, no coord trailer.
/// - `pinto prop`: columns `propensity_0 … propensity_{K-1}, cluster,
///   entropy`, plus an optional coord trailer (e.g. `pxl_row_in_fullres`,
///   `pxl_col_in_fullres`).
/// - `pinto dsvd`: columns `0 … K-1, cluster, entropy`.
///
/// This reader keeps every float column as a propensity slot *except*
/// names that appear in `exclude_cols`, the explicit `cluster` column,
/// and the optional `entropy` column (each consumed separately).
/// Returns `(propensity[N×K], cluster[N], entropy[N] when present, cell_names[N])`.
pub type PropensityRead = (Mat, Vec<i64>, Option<Vec<f32>>, Vec<Box<str>>);

pub fn read_propensity(
    path: &Path,
    exclude_cols: &HashSet<Box<str>>,
) -> anyhow::Result<PropensityRead> {
    let MatWithNames { rows, cols, mat } = read_labelled_matrix(path)?;

    // Pull propensity columns by NAME. The current writer emits
    // `C{c}` (e.g. "C0","C1",…,"C{K-1}") so the column name itself
    // declares "this is community c". Older parquets that wrote bare
    // integer names ("0","1",…) are still accepted for backwards-compat.
    // This makes "column j ↔ community j" an explicit, checked invariant
    // rather than a positional convention.
    let mut cluster_idx: Option<usize> = None;
    let mut entropy_idx: Option<usize> = None;
    let mut prop_named: Vec<(i64, usize)> = Vec::new();
    for (j, name) in cols.iter().enumerate() {
        match name.as_ref() {
            "cluster" => cluster_idx = Some(j),
            "entropy" => entropy_idx = Some(j),
            _ if exclude_cols.contains(name) => {}
            n => match parse_community_col_name(n) {
                Some(c) => prop_named.push((c, j)),
                None => anyhow::bail!(
                    "{path:?}: propensity column {n:?} is not a community ID. \
                     Expected names \"C0\",\"C1\",…,\"C{{K-1}}\" (or `cluster_<k>` / \
                     `propensity_<k>` / bare ints) plus optional \"entropy\"/coord cols."
                ),
            },
        }
    }

    if prop_named.is_empty() {
        anyhow::bail!("{path:?}: no propensity columns found (all columns excluded)");
    }

    // Pin "matrix column j ↔ community j" by mapping each named column
    // to its parsed ID. Missing intermediate IDs (e.g. labels {0,1,3,5}
    // → 2 and 4 absent) are tolerated by zero-filling those columns and
    // warning, so K stays = max_id+1 and every other plot's
    // `colors.color(c)` keeps working for the present communities. Negative
    // IDs are rejected — the writer never emits them, so seeing one means
    // genuine schema corruption.
    prop_named.sort_by_key(|&(c, _)| c);
    if let Some(&(c0, _)) = prop_named.first() {
        if c0 < 0 {
            anyhow::bail!(
                "{path:?}: propensity has negative community ID {c0}; expected non-negative integers."
            );
        }
    }
    let max_id = prop_named.last().map(|&(c, _)| c).unwrap_or(-1);
    let k = (max_id + 1).max(0) as usize;
    let present: HashSet<i64> = prop_named.iter().map(|&(c, _)| c).collect();
    let missing: Vec<i64> = (0..k as i64).filter(|c| !present.contains(c)).collect();
    if !missing.is_empty() {
        log::warn!(
            "{path:?}: propensity is missing community columns {missing:?} \
             (have 0..{} with gaps); zero-filling so plot indices stay aligned.",
            k - 1,
        );
    }

    let n = mat.nrows();
    let mut prop = Mat::zeros(n, k);
    // Source-column index per *present* community ID. Index by community
    // ID so the loop below can populate non-contiguous IDs directly.
    let mut src_for: Vec<Option<usize>> = vec![None; k];
    for &(c, j) in &prop_named {
        src_for[c as usize] = Some(j);
    }
    for (out_j, slot) in src_for.iter().enumerate() {
        if let Some(src_j) = *slot {
            for i in 0..n {
                prop[(i, out_j)] = mat[(i, src_j)];
            }
        }
    }
    // For argmax fallback below, only consider present columns so that
    // an all-zero (missing) community can't accidentally become the
    // argmax when a row has no signal.
    let prop_idx: Vec<usize> = prop_named.iter().map(|&(_, j)| j).collect();

    let cluster = match cluster_idx {
        Some(j) => (0..n).map(|i| mat[(i, j)] as i64).collect::<Vec<_>>(),
        None => (0..n)
            .map(|i| {
                if prop_idx.is_empty() {
                    -1
                } else {
                    let rank = argmax_row(&mat, i, &prop_idx) as usize;
                    prop_named[rank].0
                }
            })
            .collect(),
    };

    let entropy = entropy_idx.map(|j| (0..n).map(|i| mat[(i, j)]).collect::<Vec<_>>());

    Ok((prop, cluster, entropy, rows))
}

/// Parse a community-column / community-label string into its integer ID.
///
/// Accepts the current `C{c}` schema and two legacy forms — bare integer
/// (`"5"`, older `pinto lc`) and `propensity_{c}` (older `pinto propensity`)
/// — so existing parquets on disk still load. Returns `None` for anything
/// else (e.g. `"entropy"`, coord names, malformed labels).
pub(crate) fn parse_community_col_name(name: &str) -> Option<i64> {
    if let Some(rest) = name.strip_prefix('C') {
        return rest.parse::<i64>().ok();
    }
    if let Some(rest) = name.strip_prefix("propensity_") {
        return rest.parse::<i64>().ok();
    }
    if let Some(rest) = name.strip_prefix("cluster_") {
        return rest.parse::<i64>().ok();
    }
    name.parse::<i64>().ok()
}

fn argmax_row(mat: &Mat, row: usize, cols: &[usize]) -> i64 {
    let mut best_j = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (rank, &j) in cols.iter().enumerate() {
        let v = mat[(row, j)];
        if v > best_v {
            best_v = v;
            best_j = rank;
        }
    }
    best_j as i64
}
