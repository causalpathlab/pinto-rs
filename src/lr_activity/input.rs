//! What `pinto lra` and `pinto lra-simulate` read: a prior run named by its
//! manifest (`--from`), or data files with coordinates, resolved into
//! expression, coordinates and spatial contacts.

use crate::lr_activity::args::LraInputArgs;
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use crate::util::lupin_rounds::{latest_rounds, read_json, same_file};
use crate::util::metadata::{resolve_beside, GraphParams, PintoMetadata, Run};
use std::path::{Path, PathBuf};

/// The label of a cell no cell-type or region table names.
pub const UNASSIGNED: &str = "unassigned";

/// Resolved inputs.
#[derive(Debug, Clone)]
pub struct LraInput {
    pub data_files: Vec<Box<str>>,
    pub run: Option<Run>,
    pub coord_files: Vec<Box<str>>,
    pub coord_column_names: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub knn_spatial: usize,
    pub preload_data: bool,
    pub block_size: Option<usize>,
    /// `--cell-types` and `--region`: cell-label tables.
    pub cell_types: Option<Box<str>>,
    pub region: Option<Box<str>>,
    /// The stage-1 run of a residual fit, when `--from` names one: its
    /// newest lupin round gives the cell types ([`residual_of`]).
    pub stage1: Option<Run>,
}

impl LraInputArgs {
    pub fn resolve(&self) -> anyhow::Result<LraInput> {
        let run = self
            .from
            .as_deref()
            .map(|m| Run::open(m).map_err(|e| anyhow::anyhow!("--from {e}")))
            .transpose()?;
        for f in &self.data_files {
            if let Some((marker, _)) = excess_marker(Path::new(&**f)) {
                anyhow::bail!(
                    "{f} holds excess counts (see {}): lra estimates on the raw counts. Pass the raw \
                     data files, or --from the residual run alone, which names them",
                    marker.display()
                );
            }
        }
        let residual = run.as_ref().map(residual_of).transpose()?.flatten();
        if let (Some(r), Some(res)) = (&run, &residual) {
            info!(
                "--from {} is a residual fit ({}): reading its raw counts, not its excess counts",
                r.manifest.display(),
                res.holder.display()
            );
            if self.region.is_none() {
                warn!(
                    "a residual fit's communities are its donor-specific regions; pass them as \
                     --region (a cell, region table), best from a fit with the pairs' genes \
                     dropped, to adjust for them and read each one's effect"
                );
            }
        }
        let data_files: Vec<Box<str>> = match (&run, &residual, self.data_files.is_empty()) {
            (_, _, false) => self.data_files.clone(),
            (Some(r), Some(res), true) => raw_data_files(r, res)?,
            (Some(r), None, true) => {
                let listed = r.meta.data_files.clone().unwrap_or_default();
                anyhow::ensure!(
                    !listed.is_empty(),
                    "{} lists no data_files; pass the data files on the command line",
                    r.manifest.display()
                );
                listed
                    .iter()
                    .map(|f| {
                        r.resolve(f)
                            .map(|p| p.to_string_lossy().into_owned().into_boxed_str())
                            .ok_or_else(|| missing(&r.manifest, f, "data file"))
                    })
                    .collect::<anyhow::Result<_>>()?
            }
            (None, _, true) => anyhow::bail!(
                "no input: pass a prior run's manifest (--from run.pinto.json), \
                 or data files with --coord"
            ),
        };
        Ok(LraInput {
            data_files,
            run,
            coord_files: self.coord_files.clone(),
            coord_column_names: self.coord_column_names.clone(),
            batch_files: self.batch_files.clone(),
            knn_spatial: self.knn_spatial,
            preload_data: self.preload_data,
            block_size: self.block_size,
            cell_types: self.cell_types.clone(),
            region: self.region.clone(),
            stage1: residual.and_then(|r| r.stage1),
        })
    }
}

// * Residual fits

/// The `residual` block a residual fit's manifest carries, and that `pinto
/// residual` writes into `{out}/residual.json` beside its excess-count
/// backends: one schema for both sides. lra reads only what decides its
/// input. Paths are absolute, or relative to the file holding the block.
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub(crate) struct ResidualBlock {
    /// `excess` for counts a fitted run left unexplained.
    pub counts: String,
    /// The stage-1 run's manifest.
    #[serde(default)]
    pub stage1: Option<String>,
    /// Aligned with the run's `data_files`, index for index.
    #[serde(default)]
    pub raw_data_files: Vec<String>,
}

/// Whether two data files give their cells the same name, as a fit names
/// them by [`basename`] (a second extension such as `.zarr` in `.zarr.zip`
/// dropped): a residual's file and the raw file it came from must.
pub(crate) fn same_data_name(a: &str, b: &str) -> bool {
    let name = |p: &str| basename(p.trim_end_matches('/')).ok();
    name(a).is_some() && name(a) == name(b)
}

/// What lra takes from a residual fit: the raw data files, the stage-1 run,
/// and where the block was read (its paths are relative to that file).
#[derive(Debug)]
pub struct ResidualInput {
    pub raw_data_files: Vec<std::path::PathBuf>,
    pub stage1: Option<Run>,
    pub holder: PathBuf,
}

/// The excess-count marker beside data file `path`: `residual.json` in its
/// directory saying `counts: excess`, with the marker's block. A raw file
/// the marker names is not excess.
fn excess_marker(path: &Path) -> Option<(PathBuf, serde_json::Value)> {
    let marker = path.parent()?.join("residual.json");
    let value = read_json(&marker)?;
    let block: ResidualBlock = serde_json::from_value(value.clone()).ok()?;
    let raw = block
        .raw_data_files
        .iter()
        .filter_map(|f| resolve_beside(&marker, f))
        .any(|f| same_file(&f, path));
    (block.counts == "excess" && !raw).then_some((marker, value))
}

/// Whether data file `path` holds excess counts, by the marker beside it.
pub(crate) fn is_excess(path: &Path) -> bool {
    excess_marker(path).is_some()
}

/// `run` as a residual fit: its manifest's `residual` block, else the
/// marker beside its first data file; `None` for a run on raw counts.
pub fn residual_of(run: &Run) -> anyhow::Result<Option<ResidualInput>> {
    let in_manifest = read_json(&run.manifest)
        .and_then(|m| m.get("residual").cloned())
        .map(|v| (v, run.manifest.clone()));
    let beside = || {
        let first = run.meta.data_files.as_ref()?.first()?;
        let (marker, value) = excess_marker(&run.resolve(first)?)?;
        Some((value, marker))
    };
    let Some((value, holder)) = in_manifest.or_else(beside) else {
        return Ok(None);
    };
    let block: ResidualBlock = serde_json::from_value(value).map_err(|e| {
        anyhow::anyhow!(
            "{}: a residual block lra cannot read: {e}",
            holder.display()
        )
    })?;
    if block.counts != "excess" {
        return Ok(None);
    }
    let raw_data_files = block
        .raw_data_files
        .iter()
        .map(|f| resolve_beside(&holder, f).ok_or_else(|| missing(&holder, f, "raw data file")))
        .collect::<anyhow::Result<_>>()?;
    let stage1 = block
        .stage1
        .as_deref()
        .map(|s| {
            let path =
                resolve_beside(&holder, s).ok_or_else(|| missing(&holder, s, "stage-1 run"))?;
            Run::open(&path.to_string_lossy())
        })
        .transpose()?;
    Ok(Some(ResidualInput {
        raw_data_files,
        stage1,
        holder,
    }))
}

/// The raw counts of residual fit `run`: the block's `raw_data_files`, else
/// the stage-1 run's data files, one per data file of `run` and with the
/// same stems, so the cells keep their names.
fn raw_data_files(run: &Run, res: &ResidualInput) -> anyhow::Result<Vec<Box<str>>> {
    let raw: Vec<PathBuf> = if !res.raw_data_files.is_empty() {
        res.raw_data_files.clone()
    } else if let Some(s1) = &res.stage1 {
        s1.meta
            .data_files
            .iter()
            .flatten()
            .map(|f| {
                s1.resolve(f)
                    .ok_or_else(|| missing(&s1.manifest, f, "data file"))
            })
            .collect::<anyhow::Result<_>>()?
    } else {
        anyhow::bail!(
            "{} names neither raw_data_files nor a stage1 run: pass the raw data files",
            res.holder.display()
        );
    };
    let own = run.meta.data_files.clone().unwrap_or_default();
    anyhow::ensure!(
        raw.len() == own.len(),
        "{} gives {} raw data files for the run's {}",
        res.holder.display(),
        raw.len(),
        own.len()
    );
    for (r, o) in raw.iter().zip(&own) {
        anyhow::ensure!(
            same_data_name(&r.to_string_lossy(), o),
            "raw data file {} does not match the run's {o}: they must share a name, so \
             the cells keep their names",
            r.display()
        );
    }
    Ok(raw
        .iter()
        .map(|p| p.to_string_lossy().into_owned().into_boxed_str())
        .collect())
}

fn missing(manifest: &Path, file: &str, what: &str) -> anyhow::Error {
    let dir = manifest.parent().unwrap_or(Path::new(""));
    anyhow::anyhow!(
        "{what} {file} named in {} not found (looked under {} and in the working directory)",
        manifest.display(),
        if dir.as_os_str().is_empty() {
            ".".into()
        } else {
            dir.display().to_string()
        }
    )
}

impl LraInput {
    /// The output prefix: `--out`, else the run's prefix.
    pub fn out(&self, out: Option<&str>) -> anyhow::Result<String> {
        match (out, &self.run) {
            (Some(o), _) => Ok(o.to_string()),
            (None, Some(r)) => Ok(r.prefix.clone()),
            (None, None) => anyhow::bail!("--out is required without --from"),
        }
    }

    /// A file of the `--from` run, `{prefix}.{suffix}`; the error names it.
    pub fn run_file(&self, suffix: &str, what: &str) -> anyhow::Result<PathBuf> {
        let r = self
            .run
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("{what} comes from a prior run: pass --from"))?;
        let path = PathBuf::from(format!("{}.{suffix}", r.prefix));
        anyhow::ensure!(
            path.is_file(),
            "{what}: {} not found (the run in {} did not write it)",
            path.display(),
            r.manifest.display()
        );
        Ok(path)
    }

    /// The prefix of the `--from` run, which must have link communities.
    pub fn lc_prefix(&self) -> anyhow::Result<String> {
        self.run_file(
            "link_community.parquet",
            "the co-expression methods' link communities (a `pinto lc` run)",
        )?;
        Ok(self.run.as_ref().expect("run_file checked").prefix.clone())
    }
}

// * Contacts

/// Expression, coordinates and spatial contacts on the data's columns.
pub struct Contacts {
    pub data: SparseIoVec,
    pub axis: FeatureAxis,
    /// Spatial contacts as data columns.
    pub pairs: Vec<(usize, usize)>,
    /// Each cell's section (batch) label.
    pub cell_section: Vec<Box<str>>,
    pub coords: Vec<Option<(f32, f32)>>,
    /// Cell types of `--cell-types`, else of the run's newest lupin round,
    /// when it has one: a code per data column into the names (unlabelled
    /// cells get `unassigned`).
    pub cell_type: Option<Labels>,
    /// Regions of `--region`, coded the same way.
    pub region: Option<Labels>,
}

/// A label per data column, as a code into the names; `unassigned`, the
/// last name, for a cell the labels leave out.
pub type Labels = (Vec<u32>, Vec<Box<str>>);

/// Read the contacts of `input` ([`read_contacts`]), with the cell types
/// and regions of `--cell-types` and `--region` when given.
pub fn load_contacts(input: &LraInput) -> anyhow::Result<Contacts> {
    let mut c = read_contacts(input)?;
    if input.cell_types.is_none() && input.region.is_none() {
        return Ok(c);
    }
    let cell_to_col: HashMap<Box<str>, usize> = c
        .data
        .column_names()?
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let n_cells = c.data.num_columns();
    if let Some(f) = &input.cell_types {
        let t = read_labels(
            Path::new(&**f),
            &["cell_type", "label"],
            &cell_to_col,
            n_cells,
        )
        .map_err(|e| e.context(format!("--cell-types {f}")))?;
        info!("Cell types from {f}: {} types", t.1.len());
        warn!("{}", cell_type_caveat(&format!("--cell-types {f}")));
        c.cell_type = Some(t);
    }
    if let Some(f) = &input.region {
        let r = read_labels(Path::new(&**f), &["region", "label"], &cell_to_col, n_cells)
            .map_err(|e| e.context(format!("--region {f}")))?;
        info!("Regions from {f}: {} regions", r.1.len());
        c.region = Some(r);
    }
    Ok(c)
}

/// Read the contacts of `input`: from the `--from` run's cell pairs, or by
/// building a spatial KNN graph on the coordinates given.
fn read_contacts(input: &LraInput) -> anyhow::Result<Contacts> {
    if input.run.is_some() {
        anyhow::ensure!(
            input.coord_files.is_empty(),
            "--coord applies to data files read directly; with --from the run's \
             coordinates are used"
        );
        return contacts_from_run(input);
    }
    anyhow::ensure!(
        !input.coord_files.is_empty(),
        "spatial contacts need coordinates: pass --coord (one file per data file), \
         or --from a prior run's manifest"
    );
    let crate::util::input::SRTData {
        data,
        coordinates,
        batches,
        ..
    } = crate::util::input::read_data_with_coordinates(crate::util::input::SRTReadArgs {
        data_files: input.data_files.clone(),
        coord_files: input.coord_files.clone(),
        preload_data: input.preload_data,
        coord_columns: Vec::new(),
        coord_column_names: input.coord_column_names.clone(),
        batch_files: input.batch_files.clone(),
        header_in_coord: None,
        feature_kind: data_beans::aux::feature_names::FeatureNameKind::Exact,
    })?;
    anyhow::ensure!(
        coordinates.ncols() >= 2,
        "the coordinate files give fewer than two coordinates"
    );
    // The batch pseudo-coordinate, when present, keeps sections apart.
    let graph = crate::util::cell_pairs::build_spatial_graph(
        &coordinates,
        crate::util::cell_pairs::SrtCellPairsArgs {
            knn: input.knn_spatial,
            block_size: input.block_size,
            reciprocal: false,
        },
    )?;
    let coords = (0..coordinates.nrows())
        .map(|i| Some((coordinates[(i, 0)], coordinates[(i, 1)])))
        .collect();
    let axis = FeatureAxis::resolve_or_identity(&data.row_names()?)?;
    info!(
        "Built {} spatial contacts ({} per cell) over {} cells",
        graph.edges.len(),
        input.knn_spatial,
        data.num_columns()
    );
    Ok(Contacts {
        data,
        axis,
        pairs: graph.edges,
        cell_section: batches,
        coords,
        cell_type: None,
        region: None,
    })
}

fn contacts_from_run(input: &LraInput) -> anyhow::Result<Contacts> {
    let data = crate::lr_activity::fit::load_expr_data(&input.data_files, input.preload_data)?;
    let axis = FeatureAxis::resolve_or_identity(&data.row_names()?)?;
    let n_cells = data.num_columns();
    let cell_to_col: HashMap<Box<str>, usize> = data
        .column_names()?
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let coord_pairs = input.run_file("coord_pairs.parquet", "the run's cell pairs")?;
    let (pairs, cell_batch) = read_spatial_contacts(&coord_pairs, &cell_to_col, n_cells)?;
    // Each cell's own batch, whether or not it has a contact inside it.
    let cell_section = cell_batch
        .into_iter()
        .map(|b| b.unwrap_or_else(|| crate::lr_activity::fit::BATCH_LABEL_ALL.into()))
        .collect();
    let prefix = &input.run.as_ref().expect("from a run").prefix;
    let coords = read_coords(prefix, &cell_to_col, n_cells)?;
    // A residual fit's cell types are its stage-1 run's: donor-invariant.
    let manifest = match &input.stage1 {
        Some(s1) => &s1.manifest,
        None => &input.run.as_ref().expect("from a run").manifest,
    };
    let cell_type = match newest_round(manifest) {
        Some(round) => match round_cell_types(&round, &cell_to_col, n_cells) {
            Ok(t) => {
                info!(
                    "Cell types from lupin round {}: {} types",
                    round.display(),
                    t.1.len()
                );
                warn!(
                    "{}",
                    cell_type_caveat(&format!("lupin round {}", round.display()))
                );
                Some(t)
            }
            Err(e) => {
                warn!("lupin round {}: no cell types read: {e}", round.display());
                None
            }
        },
        None => None,
    };
    Ok(Contacts {
        data,
        axis,
        pairs,
        cell_section,
        coords,
        cell_type,
        region: None,
    })
}

/// Coordinates per data column from the `lc` run: `{prefix}.cells.parquet`
/// when the run wrote one, else the endpoints in `{prefix}.coord_pairs.parquet`.
fn read_coords(
    lc_prefix: &str,
    cell_to_col: &HashMap<Box<str>, usize>,
    n_cells: usize,
) -> anyhow::Result<Vec<Option<(f32, f32)>>> {
    use crate::util::parquet_io::{read_cells_from_coord_pairs, read_cells_table};
    let cells = std::path::PathBuf::from(format!("{lc_prefix}.cells.parquet"));
    let table = if cells.exists() {
        read_cells_table(&cells, None)?
    } else {
        read_cells_from_coord_pairs(
            std::path::Path::new(&format!("{lc_prefix}.coord_pairs.parquet")),
            None,
        )?
    };
    let mut out = vec![None; n_cells];
    let mut found = 0usize;
    for (name, &xy) in table.names.iter().zip(&table.coords) {
        if let Some(&c) = cell_to_col.get(name) {
            out[c] = Some(xy);
            found += 1;
        }
    }
    anyhow::ensure!(
        found > 0,
        "no cell of the data has coordinates in {}",
        if cells.exists() {
            cells.display().to_string()
        } else {
            format!("{lc_prefix}.coord_pairs.parquet")
        }
    );
    Ok(out)
}

// * Cell types

/// What cell types read from `source` (a lupin round, a `--cell-types`
/// file) mean for the estimate. They are optional: a run without them has
/// no cell-type confounders, strata or rows.
pub fn cell_type_caveat(source: &str) -> String {
    format!(
        "cell types from {source} adjust the estimate (confounder columns, pseudobulk \
         strata, per-type rows); they come from expression-derived clusters, and while the \
         receiver identity projection leaves the ligand, receptor and target genes out, the \
         labels may have been called on them"
    )
}

/// The newest lupin round (`*.lupin.json`) made from the run at `manifest`:
/// among the rounds whose `annotate.source` chain leads back to it, the
/// newest that no other round was made from.
pub fn newest_round(manifest: &Path) -> Option<PathBuf> {
    latest_rounds(manifest).into_iter().next()
}

/// A round's cell types (its `annotate.argmax` table: `cell`, `cell_type`)
/// as a code per data column; cells it leaves out or calls `unassigned`
/// share the last code, `unassigned`.
pub fn round_cell_types(
    round: &Path,
    cell_to_col: &HashMap<Box<str>, usize>,
    n_cells: usize,
) -> anyhow::Result<(Vec<u32>, Vec<Box<str>>)> {
    use legume_numeric::matrix::table::TableReader;
    let m = read_json(round).ok_or_else(|| anyhow::anyhow!("not readable JSON"))?;
    let argmax = m
        .pointer("/annotate/argmax")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("no annotate.argmax"))?;
    let table =
        resolve_beside(round, argmax).ok_or_else(|| anyhow::anyhow!("{argmax} not found"))?;
    let t = TableReader::open(&table.to_string_lossy())?;
    let cols = t.select(&["cell", "cell_type"])?;
    let (label, _) = labels_by_column(&t, &cols, cell_to_col, n_cells)?;
    code_labels(label)
}

/// A cell-label table (`--cell-types`, `--region`): its `cell` column (else
/// the first) and the first of `label_names` it has (else the second), as a
/// code per data column ([`Labels`]). Errors when no cell of the data is
/// named; warns when fewer than half are.
pub fn read_labels(
    path: &Path,
    label_names: &[&str],
    cell_to_col: &HashMap<Box<str>, usize>,
    n_cells: usize,
) -> anyhow::Result<Labels> {
    use legume_numeric::matrix::table::TableReader;
    let t = TableReader::open(&path.to_string_lossy())?;
    anyhow::ensure!(
        t.header().len() >= 2,
        "needs a cell column and a label column, has {:?}",
        t.header()
    );
    let cell = t.find_column(&["cell"]).unwrap_or(0);
    let label = t
        .find_column(label_names)
        .unwrap_or(if cell == 1 { 0 } else { 1 });
    anyhow::ensure!(cell != label, "the cell and label columns are the same");
    let (label, matched) = labels_by_column(&t, &[cell, label], cell_to_col, n_cells)?;
    anyhow::ensure!(
        matched > 0,
        "no cell it names is a cell of the data (first data cells: {})",
        first_names(cell_to_col)
    );
    if 2 * matched < n_cells {
        warn!(
            "{}: labels {matched} of {n_cells} cells; the rest are unassigned",
            path.display()
        );
    }
    code_labels(label)
}

/// A few of the data's cell names, for an error that says how to name them.
fn first_names(cell_to_col: &HashMap<Box<str>, usize>) -> String {
    let mut v: Vec<(&usize, &Box<str>)> = cell_to_col.iter().map(|(n, c)| (c, n)).collect();
    v.sort_unstable();
    v.iter()
        .take(3)
        .map(|(_, n)| n.as_ref())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Each data column's label from the table's `cols` (cell, label), and how
/// many cells got one; empty and `unassigned` labels are left out.
fn labels_by_column(
    t: &legume_numeric::matrix::table::TableReader,
    cols: &[usize],
    cell_to_col: &HashMap<Box<str>, usize>,
    n_cells: usize,
) -> anyhow::Result<(Vec<Option<Box<str>>>, usize)> {
    let mut label = vec![None::<Box<str>>; n_cells];
    let mut matched = 0usize;
    for row in t.rows(cols)? {
        let row = row?;
        let (cell, ty) = (row[0].as_ref(), row[1].as_ref());
        if ty.is_empty() || ty.eq_ignore_ascii_case(UNASSIGNED) {
            continue;
        }
        if let Some(&c) = cell_to_col.get(cell) {
            matched += label[c].is_none() as usize;
            label[c] = Some(ty.into());
        }
    }
    Ok((label, matched))
}

/// Labels as codes into their sorted names, `unassigned` last.
fn code_labels(label: Vec<Option<Box<str>>>) -> anyhow::Result<Labels> {
    let mut names: Vec<Box<str>> = label.iter().flatten().cloned().collect();
    names.sort_unstable();
    names.dedup();
    anyhow::ensure!(!names.is_empty(), "no cell of the data has a label");
    let unassigned = names.len() as u32;
    let codes = label
        .iter()
        .map(|l| {
            l.as_ref().map_or(unassigned, |l| {
                names.binary_search(l).expect("listed") as u32
            })
        })
        .collect();
    names.push(UNASSIGNED.into());
    Ok((codes, names))
}

/// Each data column's own batch, when known.
pub type CellBatches = Vec<Option<Box<str>>>;

/// The physically adjacent pairs of `{prefix}.coord_pairs.parquet` (every
/// pair, unless an `edge_kind` column marks expression pairs) as data
/// columns, and each data column's own batch, read from the ends of every
/// pair it is in (`None` without batch columns or pairs).
pub fn read_spatial_contacts(
    path: &Path,
    cell_to_col: &HashMap<Box<str>, usize>,
    n_cells: usize,
) -> anyhow::Result<(Vec<(usize, usize)>, CellBatches)> {
    let records = crate::lr_activity::io::read_pair_table(&path.to_string_lossy(), false)?;
    let (mut out, mut unresolved) = (Vec::new(), 0usize);
    let mut batch: Vec<Option<Box<str>>> = vec![None; n_cells];
    for e in records.into_iter().filter(|e| e.is_spatial) {
        let (Some(&i), Some(&j)) = (
            cell_to_col.get(&e.left_cell),
            cell_to_col.get(&e.right_cell),
        ) else {
            unresolved += 1;
            continue;
        };
        if let Some((l, r)) = e.side_batches {
            batch[i].get_or_insert(l);
            batch[j].get_or_insert(r);
        }
        out.push((i, j));
    }
    if unresolved > 0 {
        info!("Dropped {unresolved} cell pairs whose cells are not in the expression data");
    }
    anyhow::ensure!(
        !out.is_empty(),
        "no spatial contact of {} names two cells of the data",
        path.display()
    );
    Ok((out, batch))
}

/// The manifest lra records into: the `--from` run's, else
/// `{out}.pinto.json`, written now when it does not exist yet.
pub fn record_manifest(
    input: &LraInput,
    out: &str,
    contacts: &Contacts,
) -> anyhow::Result<PathBuf> {
    if let Some(r) = &input.run {
        return Ok(r.manifest.clone());
    }
    let path = PathBuf::from(format!("{out}.pinto.json"));
    PintoMetadata {
        command: "lra".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        timestamp: crate::util::metadata::now_secs(),
        prefix: out.into(),
        data_files: Some(input.data_files.iter().map(|s| s.to_string()).collect()),
        coord_file: Some(input.coord_files.join(",")),
        n_cells: contacts.data.num_columns(),
        n_features: contacts.axis.n_features(),
        n_edges: Some(contacts.pairs.len()),
        graph: Some(GraphParams {
            knn_base: input.knn_spatial,
            ..Default::default()
        }),
        ..Default::default()
    }
    .write_if_missing(&path)?;
    Ok(path)
}
