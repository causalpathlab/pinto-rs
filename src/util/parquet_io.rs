//! Parquet readers shared across pinto subcommands.
//!
//! Thin wrappers over `legume_numeric::matrix` primitives. The
//! pinto-specific piece is [`read_cells_from_coord_pairs`], which
//! dedupes `coord_pairs.parquet` into a per-cell table `(name, x, y,
//! optional batch)`.

use crate::util::common::*;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use data_beans::hdf5_io::strip_backend_suffix;
use legume_numeric::matrix::common_io::basename;
use legume_numeric::matrix::parquet::peek_parquet_field_names;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Row, RowAccessor};
use std::fs::File;
use std::path::Path;

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

/// One numeric column of a table keyed by its first column: row names and
/// the values (e.g. a lupin round's `cell`, `cluster`, with NaN for none).
pub fn read_keyed_column(path: &Path, column: &str) -> anyhow::Result<(Vec<Box<str>>, Vec<f32>)> {
    let fields = field_names(path)?;
    let key = fields
        .first()
        .ok_or_else(|| anyhow::anyhow!("{path:?}: no columns"))?;
    let cols = read_named_columns(path, &[key.as_ref(), column])?;
    let rows = labels(&cols[0])?;
    let values = floats(&cols[1])?;
    anyhow::ensure!(rows.len() == values.len(), "{path:?}: ragged columns");
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
        if has(extra) {
            wanted.push(extra);
        }
    }
    let cols = read_named_columns(path, &wanted)?;
    let column = |n: &str| wanted.iter().position(|w| *w == n).map(|k| &cols[k]);
    let names = labels(&cols[0])?;
    let (xs, ys) = (floats(&cols[1])?, floats(&cols[2])?);
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
    Ok(CellTable {
        names,
        coords: xs.into_iter().zip(ys).collect(),
        batches,
        index,
        in_graph,
        coord_col_names: vec![x.into_boxed_str(), y.into_boxed_str()],
    })
}

/// One row per cell. `batch` is `None` if the fit was single-batch
/// (i.e. `coord_pairs.parquet` lacks a `left_batch` column).
pub struct CellTable {
    /// Cell barcode, unique and stable-ordered.
    pub names: Vec<Box<str>>,
    /// (x, y) coordinate per cell.
    pub coords: Vec<(f32, f32)>,
    /// Optional batch label per cell.
    pub batches: Option<Vec<Box<str>>>,
    /// `name → index into names/coords/batches`
    pub index: HashMap<Box<str>, usize>,
    /// Whether each cell became a graph node (`false`: QC dropped it, or it
    /// has no neighbours); `None` when every cell did.
    pub in_graph: Option<Vec<bool>>,
    /// Bare coordinate column names (without `left_`/`right_` prefix).
    /// Pass-through from `coord_pairs.parquet` so downstream readers
    /// (e.g. `read_propensity`) can exclude coord trailers that
    /// `pinto prop` may append to `.propensity.parquet`.
    pub coord_col_names: Vec<Box<str>>,
}

impl CellTable {
    pub fn n(&self) -> usize {
        self.names.len()
    }
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

    let has_batch = fields.iter().any(|f| f.as_ref() == "left_batch")
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

    let batches = if has_batch { Some(batches) } else { None };
    let coord_col_names = vec![x_bare.into_boxed_str(), y_bare.into_boxed_str()];
    Ok(CellTable {
        names,
        coords,
        batches,
        index,
        in_graph: None,
        coord_col_names,
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

/// A cell-pair edge: (left_cell_name, right_cell_name).
pub type EdgePair = (Box<str>, Box<str>);

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

/// Map "0","1",… numeric batch labels to the friendly basenames of the
/// upstream input files (`.pinto.json::data_files`).
///
/// `pinto svd` assigns string-of-integer batch labels (`"0"`, `"1"`, …)
/// when the user passes multiple data files without an explicit
/// `--batch-files`, and these labels propagate through
/// `coord_pairs.parquet` into the plot. The user-facing batch name is
/// the input file's basename (without the `.zarr` / `.h5` / `.zarr.zip`
/// extension), which is what the data-beans merger uses internally —
/// reusing that here keeps the plot dirs consistent with how the
/// upstream code names batches.
///
/// Returns the resolved name table when all current labels parse as
/// non-negative integers `< data_files.len()`. `None` means the labels
/// are already strings (user supplied real batch names) or the
/// metadata is missing — no remap is applied in either case.
pub fn resolve_batch_name_map(
    cells_batches: &[Box<str>],
    data_files: &[String],
) -> Option<Vec<Box<str>>> {
    if data_files.is_empty() {
        return None;
    }
    // Every existing label must be a non-negative integer that indexes
    // into `data_files`; one non-numeric label is enough to bail (the
    // user already supplied human-readable batch names).
    for b in cells_batches {
        let parsed: Result<usize, _> = b.parse();
        match parsed {
            Ok(i) if i < data_files.len() => {}
            _ => return None,
        }
    }
    Some(unique_batch_names_from_data_files(data_files))
}

/// Apply the index→name table from `resolve_batch_name_map` to a slice
/// of batch labels. Caller owns the slice; labels that don't parse are
/// left as-is (defensive — `resolve_batch_name_map` already guards).
pub fn remap_batch_labels(labels: &mut [Box<str>], name_map: &[Box<str>]) {
    for b in labels.iter_mut() {
        if let Ok(i) = b.parse::<usize>() {
            if let Some(friendly) = name_map.get(i) {
                *b = friendly.clone();
            }
        }
    }
}

/// Mirror of `data_beans::handlers::merging::generate_unique_batch_names`,
/// reimplemented here because that helper sits inside a non-public
/// `handlers` module. Strips backend suffixes from each file's
/// basename and disambiguates duplicates with a `_{n}` counter so the
/// returned vector is index-aligned with `data_files`.
fn unique_batch_names_from_data_files(data_files: &[String]) -> Vec<Box<str>> {
    let bare: Vec<Box<str>> = data_files
        .iter()
        .map(|f| {
            basename(f)
                .map(|b| {
                    let stripped = strip_backend_suffix(&b);
                    if stripped.len() == b.len() {
                        b
                    } else {
                        stripped.into()
                    }
                })
                .unwrap_or_else(|_| f.clone().into_boxed_str())
        })
        .collect();
    let mut counts: HashMap<Box<str>, usize> = HashMap::default();
    for n in &bare {
        *counts.entry(n.clone()).or_insert(0) += 1;
    }
    let mut counters: HashMap<Box<str>, usize> = HashMap::default();
    bare.iter()
        .map(|n| match counts.get(n).copied().unwrap_or(0) {
            0 | 1 => n.clone(),
            _ => {
                let c = counters.entry(n.clone()).or_insert(0);
                let out = format!("{n}_{c}").into_boxed_str();
                *c += 1;
                out
            }
        })
        .collect()
}
