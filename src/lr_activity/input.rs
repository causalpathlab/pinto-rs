//! What `pinto lra` and `pinto lra-simulate` read: a prior run named by its
//! manifest (`--from`), or data files with coordinates, resolved into
//! expression, coordinates and spatial contacts.

use crate::lr_activity::args::LraInputArgs;
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use crate::util::lupin_rounds::{latest_rounds, read_json};
use crate::util::metadata::{resolve_beside, GraphParams, OutputFiles, PintoMetadata, Run};
use std::path::{Path, PathBuf};

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
}

impl LraInputArgs {
    pub fn resolve(&self) -> anyhow::Result<LraInput> {
        let run = self
            .from
            .as_deref()
            .map(|m| Run::open(m).map_err(|e| anyhow::anyhow!("--from {e}")))
            .transpose()?;
        let data_files: Vec<Box<str>> = match (&run, self.data_files.is_empty()) {
            (_, false) => self.data_files.clone(),
            (Some(r), true) => {
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
            (None, true) => anyhow::bail!(
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
        })
    }
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
    /// Cell types of the run's newest lupin round, when it has one: a code
    /// per data column into the names (unlabelled cells get `unassigned`).
    pub cell_type: Option<(Vec<u32>, Vec<Box<str>>)>,
}

/// Read the contacts of `input`: from the `--from` run's cell pairs, or by
/// building a spatial KNN graph on the coordinates given.
pub fn load_contacts(input: &LraInput) -> anyhow::Result<Contacts> {
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
    let coords =
        crate::lr_activity::ligand_effect::pipeline::read_coords(prefix, &cell_to_col, n_cells)?;
    let manifest = &input.run.as_ref().expect("from a run").manifest;
    let cell_type = match newest_round(manifest) {
        Some(round) => match round_cell_types(&round, &cell_to_col, n_cells) {
            Ok(t) => {
                info!(
                    "Cell types from lupin round {}: {} types",
                    round.display(),
                    t.1.len()
                );
                warn!("{CELL_TYPE_CAVEAT}");
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
    })
}

// * Cell types

/// What cell types read from a lupin round mean for the estimate. They are
/// optional: a run without a round has no cell-type confounders, strata or
/// rows.
pub const CELL_TYPE_CAVEAT: &str =
    "cell types from the lupin round adjust the estimate (confounder \
     columns, pseudobulk strata, per-type rows); they come from expression-derived clusters, \
     and while the receiver identity projection leaves the ligand, receptor and target genes \
     out, the labels may have been called on them";

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
    let mut label = vec![None::<Box<str>>; n_cells];
    for row in t.rows(&cols)? {
        let row = row?;
        let (cell, ty) = (row[0].as_ref(), row[1].as_ref());
        if ty.is_empty() || ty.eq_ignore_ascii_case("unassigned") {
            continue;
        }
        if let Some(&c) = cell_to_col.get(cell) {
            label[c] = Some(ty.into());
        }
    }
    let mut names: Vec<Box<str>> = label.iter().flatten().cloned().collect();
    names.sort_unstable();
    names.dedup();
    anyhow::ensure!(!names.is_empty(), "no cell of the data has a cell type");
    let unassigned = names.len() as u32;
    let codes = label
        .iter()
        .map(|l| {
            l.as_ref().map_or(unassigned, |l| {
                names.binary_search(l).expect("listed") as u32
            })
        })
        .collect();
    names.push("unassigned".into());
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
    if !path.exists() {
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
            outputs: OutputFiles::default(),
            ..Default::default()
        }
        .write(&path)?;
    }
    Ok(path)
}
