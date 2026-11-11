use crate::util::cell_pairs::connected_components;
use crate::util::common::*;
use crate::util::knn_graph::KnnGraph;
use data_beans::aux::feature_names::FeatureNameKind;
use data_beans::convert::try_open_or_convert;

mod args;
pub use args::{ResolvedKnn, SrtInputArgs};

pub struct SRTReadArgs {
    pub data_files: Vec<Box<str>>,
    pub coord_files: Vec<Box<str>>,
    pub preload_data: bool,
    pub coord_columns: Vec<usize>,
    pub coord_column_names: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub header_in_coord: Option<usize>,
    /// Optional row-name canonicalizer for fuzzy cross-file feature/locus
    /// alignment. `Exact` (used by `pinto lc` and friends) keeps the rows
    /// every file shares; other kinds canonicalize through
    /// [`SparseIoVec::with_row_canonicalizer`] and keep the union.
    pub feature_kind: FeatureNameKind,
}

pub struct SRTData {
    pub data: SparseIoVec,
    pub coordinates: Mat,
    pub coordinate_names: Vec<Box<str>>,
    pub batches: Vec<Box<str>>,
}

/// Help for `--preload-data`, shared by every command that takes it.
pub const PRELOAD_HELP: &str = "Preload sparse data even with LEGUME_AUTO_PRELOAD=off";
pub const PRELOAD_LONG_HELP: &str = "Sparse column data is preloaded into memory by default\n\
     whenever it fits in half the memory free at the first preload.\n\
     That half is shared by every file the command preloads.\n\
     Data over it, or files without a stored nonzero count, are read from disk.\n\
     LEGUME_AUTO_PRELOAD=off reads from disk unless this flag is given.\n\
     The flag also preloads files without a stored count.\n\
     LEGUME_PRELOAD_BUDGET_BYTES sets the limit for each file instead.";

/// Open `data_file`, its columns preloaded when asked or, by default,
/// whenever data-beans' memory budget allows; the budget is shared by the
/// whole process and given back when the data is dropped. data-beans checks
/// it only for a backend that knows its nonzero count, so one that does not
/// is preloaded only when asked.
pub fn open_sparse(
    data_file: &str,
    preload: bool,
) -> anyhow::Result<Box<dyn data_beans::sparse_io::SparseIo<IndexIter = Vec<usize>>>> {
    let mut data = try_open_or_convert(data_file)?;
    let auto = data_beans::sparse_io::auto_preload_enabled() && data.num_non_zeros().is_some();
    if preload || auto {
        data.preload_columns()?;
    }
    Ok(data)
}

/// Open `data_files` on the union of their rows, for reading expression
/// back (propensity, the viewer's gene maps): a feature some file never
/// measured is kept, and [`crate::util::coverage::Coverage`] tells its
/// unmeasured cells from zeros. The fits read the shared rows instead
/// ([`read_data_with_coordinates`]).
pub fn read_expr_data(data_files: &[Box<str>]) -> anyhow::Result<SparseIoVec> {
    if data_files.is_empty() {
        return Err(anyhow::anyhow!("empty data files"));
    }

    let attach_data_name = data_files.len() > 1;
    let mut data_vec = SparseIoVec::new();

    for data_file in data_files.iter() {
        info!("Importing data file: {}", data_file);
        let data = try_open_or_convert(data_file)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;
        data_vec.push(Arc::from(data), data_name)?;
    }

    Ok(data_vec)
}

/// Open every data file into one [`SparseIoVec`]. Under `Exact` names the
/// files are read on the rows all of them share, in whatever order each
/// lists them: a feature one sample never measured is not a zero there.
/// Other kinds canonicalize names first and keep their union, as before.
fn open_data_files(args: &SRTReadArgs) -> anyhow::Result<SparseIoVec> {
    let attach_data_name = args.data_files.len() > 1;
    let mut data_vec = SparseIoVec::new();
    if let Some(canon) = args.feature_kind.clone().into_canonicalizer() {
        data_vec = data_vec.with_row_canonicalizer(move |name| canon(name))?;
    } else {
        data_vec = data_vec.with_row_alignment(RowAlignment::Intersect)?;
    }

    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        let data = open_sparse(data_file, args.preload_data)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;

        data_vec.push(Arc::from(data), data_name)?;
    }

    anyhow::ensure!(data_vec.num_rows() > 0, "the data files share no row name");
    // Rows spelled differently across files (`ENSG..._GENE1` against
    // `GENE1`) leave few shared: say so, rather than fit on what is left.
    if args.feature_kind.is_exact() {
        let shared = data_vec.num_rows();
        for (j, file) in args.data_files.iter().enumerate() {
            let own = data_vec[j].num_rows().unwrap_or(0);
            if 2 * shared < own {
                log::warn!("{file}: only {shared} of its {own} rows are in every data file");
            }
        }
    }
    Ok(data_vec)
}

/// Each cell's batch label, with each data file's cell count: from the
/// batch files (one label per line, one file per data file), else one batch
/// per data file.
fn read_batch_membership(
    args: &SRTReadArgs,
    data_vec: &SparseIoVec,
) -> anyhow::Result<(Vec<Box<str>>, Vec<usize>)> {
    let cells_per_file = data_vec.num_columns_by_data()?;
    let mut batch_membership: Vec<Box<str>> = Vec::with_capacity(data_vec.num_columns());
    if let Some(batch_files) = &args.batch_files {
        anyhow::ensure!(
            batch_files.len() == args.data_files.len(),
            "# batch files != # of data files"
        );
        for (batch_file, &nn) in batch_files.iter().zip(&cells_per_file) {
            info!("Reading batch file: {}", batch_file);
            let lines = read_lines(batch_file)?;
            anyhow::ensure!(
                lines.len() == nn,
                "{batch_file}: {} labels for its data file's {nn} cells",
                lines.len()
            );
            batch_membership.extend(lines);
        }
    } else {
        if cells_per_file.len() > 1 {
            info!("Each data file will be considered a different batch.");
        }
        for (id, &nn) in cells_per_file.iter().enumerate() {
            batch_membership.extend(vec![id.to_string().into_boxed_str(); nn]);
        }
    }
    Ok((batch_membership, cells_per_file))
}

pub fn read_data_with_coordinates(args: SRTReadArgs) -> anyhow::Result<SRTData> {
    anyhow::ensure!(
        args.coord_files.len() == args.data_files.len(),
        "Number of coordinate files ({}) must match number of data files ({})",
        args.coord_files.len(),
        args.data_files.len()
    );

    let data_vec = open_data_files(&args)?;

    let mut coord_vec = Vec::with_capacity(args.coord_files.len());

    let mut coord_column_names = vec![];

    for (i, coord_file) in args.coord_files.iter().enumerate() {
        info!("Reading coordinate file: {}", coord_file);

        let MatWithNames {
            rows: coord_cell_names,
            cols: column_names,
            mat: data,
        } = read_one_coord_file(
            coord_file,
            &args.coord_columns,
            &args.coord_column_names,
            args.header_in_coord,
        )?;

        let data_cell_names = data_vec[i].column_names()?;

        if coord_column_names.is_empty() {
            coord_column_names.extend(column_names);
        } else if coord_column_names != column_names {
            return Err(anyhow::anyhow!(
                "coordinate column names do not match with each other"
            ));
        }

        if data_cell_names == coord_cell_names {
            coord_vec.push(data);
        } else {
            info!("reordering coordinate information");

            let coord_index_map: HashMap<&Box<str>, usize> = coord_cell_names
                .iter()
                .enumerate()
                .map(|(index, name)| (name, index))
                .collect();

            let reordered_indices: Vec<usize> = data_cell_names
                .iter()
                .map(|name| {
                    coord_index_map
                        .get(name)
                        .ok_or_else(|| {
                            anyhow::anyhow!("cell '{}' not found in the file {}", name, coord_file)
                        })
                        .copied()
                })
                .collect::<anyhow::Result<_>>()?;

            coord_vec.push(concatenate_vertical(
                &reordered_indices
                    .iter()
                    .map(|&index| data.row(index))
                    .collect::<Vec<_>>(),
            )?);
        }
    }

    let coord_nk = concatenate_vertical(&coord_vec)?;

    let (batch_membership, cells_per_file) = read_batch_membership(&args, &data_vec)?;

    // Push every (data file, batch label) group apart in the spatial KNN:
    // files can share one (x, y) frame, and a batch file that gives them all
    // one label (to skip batch correction) must not fold them into one.
    let keys = crate::util::ablation::sections(&cells_per_file, &batch_membership);
    let n_keys = keys.iter().max().map_or(0, |&m| m as usize + 1);
    let coord_nk = if n_keys > 1 {
        info!("attaching a batch index coordinate ({n_keys} groups)");
        coord_column_names.push("batch".to_string().into_boxed_str());
        append_batch_coordinate(&coord_nk, &keys)?
    } else {
        coord_nk
    };

    info!(
        "Read {} x {} coordinates",
        coord_nk.nrows(),
        coord_nk.ncols()
    );

    Ok(SRTData {
        data: data_vec,
        coordinates: coord_nk,
        coordinate_names: coord_column_names,
        batches: batch_membership,
    })
}

/// Load expression data and batch labels without spatial coordinates.
///
/// Use this when no coordinate files are provided (expression mode).
/// Coordinates will be filled in later from expression embeddings.
pub fn read_data_without_coordinates(args: SRTReadArgs) -> anyhow::Result<SRTData> {
    let data_vec = open_data_files(&args)?;

    let (batch_membership, _) = read_batch_membership(&args, &data_vec)?;

    let n_cells = data_vec.num_columns();

    // Placeholder coordinates (0×0) — will be replaced by expression embeddings
    let coordinates = Mat::zeros(n_cells, 0);
    let coordinate_names = vec![];

    info!("Read {} cells (expression mode, no coordinates)", n_cells);

    Ok(SRTData {
        data: data_vec,
        coordinates,
        coordinate_names,
        batches: batch_membership,
    })
}

/// Where `--knn-expr` looks for expression neighbours.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnnExprScope {
    /// Inside each disconnected piece of the spatial graph.
    Within,
    /// Anywhere on the slide.
    Global,
}

/// Label each disconnected piece of the spatial graph as its own batch.
///
/// Pieces that are fragments of a larger one are folded into it first, so the
/// count returned is the number of BATCHES, which is at most the number of
/// components. Leaves the labels untouched when the graph is connected.
pub fn auto_batch_from_components(
    graph: &KnnGraph,
    coordinates: &Mat,
    batch_membership: &mut Vec<Box<str>>,
    fold_fragments: bool,
) -> usize {
    let (labels, n_components) = connected_components(graph);
    if n_components <= 1 {
        return n_components;
    }
    // Folding reasons about physical geometry: box containment in the
    // coordinate frame, with a slack measured from the graph's own edge
    // lengths. Without real coordinates both halves of that argument are in
    // different, arbitrary units (a 2-D layout vs projection-space
    // distances), so a caller in expression mode asks for no folding and
    // every component stays its own batch.
    if !fold_fragments {
        for (cell, &l) in labels.iter().enumerate() {
            batch_membership[cell] = format!("component_{l}").into_boxed_str();
        }
        return n_components;
    }

    // A connected component stands in for a sample, but it is not one. The
    // spatial graph breaks wherever the tissue does, so a single physical piece
    // arrives as one large component plus a scatter of small ones. Treating each
    // as a batch estimates an effect per fragment, on far too few cells.
    //
    // A fragment is recognised by WHERE it is, not by how small it is: it lies
    // inside the piece it broke off from, while two genuinely different pieces
    // sit far apart. So a component is folded into another when its centre falls
    // within that component's extent, widened by a few cell diameters to cover
    // the gap that split them.
    //
    // The widening is measured, not chosen: the graph's own edge lengths are the
    // cell-to-cell spacing. It only has to be small against the distance between
    // pieces, and on real tissue those differ by more than an order of
    // magnitude, so the exact multiple does not change the answer.
    let spacing = median_edge_length(graph).unwrap_or(0.0);
    let slack = 3.0 * spacing;

    // Not a clamp: `0.clamp(1, 2)` is 1, and the coordinate lookup below would
    // then index a column that is not there.
    if coordinates.ncols() == 0 {
        return n_components;
    }
    // Every coordinate column, deliberately including the per-file batch
    // pseudo-coordinate a multi-file load appends: it is what separates two
    // files sharing one (x, y) frame, and judging containment without it
    // folds them into a single batch and silently skips batch correction.
    let dims = coordinates.ncols();
    let mut lo = vec![f64::INFINITY; n_components * dims];
    let mut hi = vec![f64::NEG_INFINITY; n_components * dims];
    let mut sum = vec![0.0f64; n_components * dims];
    let mut size = vec![0usize; n_components];
    let mut rep = vec![usize::MAX; n_components];
    for (cell, &l) in labels.iter().enumerate() {
        rep[l] = rep[l].min(cell);
        size[l] += 1;
        for d in 0..dims {
            let v = coordinates[(cell, d)] as f64;
            lo[l * dims + d] = lo[l * dims + d].min(v);
            hi[l * dims + d] = hi[l * dims + d].max(v);
            sum[l * dims + d] += v;
        }
    }

    // Fold into the SMALLEST enclosing piece, so a fragment joins the core it
    // sits in rather than any larger box that happens to span it.
    let mut into: Vec<usize> = (0..n_components).collect();
    for c in 0..n_components {
        let mut best: Option<(f64, f64, usize, usize)> = None;
        for other in 0..n_components {
            if other == c || size[other] <= size[c] {
                continue;
            }
            let centre = |d: usize| sum[c * dims + d] / size[c].max(1) as f64;
            let inside = (0..dims).all(|d| {
                centre(d) >= lo[other * dims + d] - slack
                    && centre(d) <= hi[other * dims + d] + slack
            });
            if !inside {
                continue;
            }
            let extent: f64 = (0..dims)
                .map(|d| hi[other * dims + d] - lo[other * dims + d])
                .sum();
            // Ties break on the candidate's own corner, then on its smallest
            // member cell — a stable identity — never on `other`, whose
            // numbering comes from a parallel union-find and varies per run.
            let corner = lo[other * dims];
            let key = (extent, corner, rep[other]);
            if best.is_none_or(|(e, c, r, _)| (extent, corner, rep[other]) < (e, c, r)) {
                best = Some((key.0, key.1, key.2, other));
            }
        }
        if let Some((_, _, _, target)) = best {
            into[c] = target;
        }
    }
    // A fragment may point at a piece that is itself a fragment; follow through.
    for c in 0..n_components {
        let mut t = into[c];
        let mut hops = 0;
        while into[t] != t && hops < n_components {
            t = into[t];
            hops += 1;
        }
        into[c] = t;
    }

    let kept: std::collections::BTreeSet<usize> = into.iter().copied().collect();
    *batch_membership = labels
        .iter()
        .map(|&l| format!("cc_{}", into[l]).into_boxed_str())
        .collect();
    let n_batches = kept.len();
    if n_batches < n_components {
        // Sizes, not just a count: a run that folded a 4000-cell "fragment"
        // into its neighbour has merged two samples, and a count alone hides it.
        let mut folded: Vec<usize> = (0..n_components)
            .filter(|&c| into[c] != c)
            .map(|c| size[c])
            .collect();
        folded.sort_unstable_by(|a, b| b.cmp(a));
        info!(
            "Auto-detected {} spatial components; {} lie inside another and were \
             folded into it, giving {} batches. Folded piece sizes, largest first: {:?}",
            n_components,
            n_components - n_batches,
            n_batches,
            &folded[..folded.len().min(10)]
        );
    } else {
        info!(
            "Auto-detected {} spatial components as batches",
            n_components
        );
    }
    n_batches
}

/// Median length of the graph's edges, which on a spatial graph is the
/// cell-to-cell spacing. `None` when the graph carries no usable lengths.
fn median_edge_length(graph: &KnnGraph) -> Option<f64> {
    let mut d: Vec<f32> = graph
        .distances
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if d.is_empty() {
        return None;
    }
    let mid = d.len() / 2;
    d.select_nth_unstable_by(mid, f32::total_cmp);
    Some(d[mid] as f64)
}

/// Read a single coordinate file with a name-then-index fallback.
///
/// Coordinates are the columns the caller named or numbered. Nothing is
/// inferred: reading by position is a guess about the file, and a wrong guess
/// does not fail, because any two numeric columns pass as coordinates.
///
/// When neither selector resolves, this errors and quotes the file's first
/// line back, since the caller cannot fix it without seeing what the file
/// holds.
pub fn read_one_coord_file(
    coord_file: &str,
    user_indices: &[usize],
    user_names: &[Box<str>],
    header_in_coord: Option<usize>,
) -> anyhow::Result<MatWithNames<Mat>> {
    let ext = file_ext(coord_file)?;
    let is_zarr = coord_file.contains(".zarr");

    // One index is a typo for two, not a request for 1-D geometry: it also
    // bypasses the too-few-columns fallback below (which only arms itself when
    // NO index was given), so a spatial graph would be built on a line with no
    // error anywhere downstream.
    anyhow::ensure!(
        user_indices.is_empty() || user_indices.len() >= 2,
        "--coord-column-indices selects {} column; coordinates need at least two",
        user_indices.len()
    );

    let read_coord = |indices: &[usize], names: &[Box<str>]| -> anyhow::Result<MatWithNames<Mat>> {
        if is_zarr {
            data_beans::zarr_io::read_zarr_coordinates(coord_file, indices, names)
        } else {
            match ext.as_ref() {
                "parquet" => Mat::from_parquet_with_indices_names(
                    coord_file,
                    Some(0),
                    Some(indices),
                    Some(names),
                ),
                _ => {
                    let header_row = header_in_coord
                        .or_else(|| detect_header_row(coord_file, &['\t', ',', ' '], names))
                        .or_else(|| {
                            legume_numeric::matrix::common_io::detect_header_row_numeric(
                                coord_file,
                                &['\t', ',', ' '],
                            )
                        });
                    Mat::read_data(
                        coord_file,
                        &['\t', ',', ' '],
                        header_row,
                        Some(0),
                        Some(indices),
                        Some(names),
                    )
                }
            }
        }
    };

    let initial = read_coord(user_indices, user_names);
    let needs_fallback = user_indices.is_empty()
        && match &initial {
            Err(_) => true,
            Ok(r) => r.mat.ncols() < 2,
        };

    if needs_fallback {
        // No guessing. Coordinates are whichever columns the caller named or
        // numbered; if neither resolves, say so and stop.
        //
        // This used to fall back to columns 4 and 5, the positions they occupy
        // in one particular vendor layout. That is a guess about the file, and
        // when it is wrong it does not fail: any two numeric columns are
        // accepted as coordinates, so a table with per-cell counts in those
        // positions puts every cell in the same place and the run looks
        // healthy. A wrong answer that announces nothing is worse than an
        // error, and the caller already has an exact way to say what it meant.
        let mut hint = String::new();
        if !is_zarr {
            if let Ok(first) =
                legume_numeric::matrix::common_io::first_line_fields(coord_file, &['\t', ',', ' '])
            {
                hint = format!(" The file's first line reads {first:?}.");
            }
        }
        let matched = initial.as_ref().map_or(0, |r| r.mat.ncols());
        return Err(anyhow::anyhow!(
            "coord file '{}' resolved {matched} of the coordinate columns asked for \
             {:?}, and two are needed.{} \
             Pass --coord-column-names with the names it does have, or \
             --coord-column-indices with their 0-based positions. For the classic \
             spot layout that is --coord-column-indices 4,5.",
            coord_file,
            user_names,
            hint,
        ));
    }

    if let Ok(r) = initial.as_ref() {
        if r.mat.ncols() > 2 {
            warn!(
                "coord file '{}' resolved {} coordinate columns {:?}; the graph will be \
                 built in that many dimensions. If two naming conventions both matched, \
                 pass --coord-column-names with just the pair you mean",
                coord_file,
                r.mat.ncols(),
                r.cols,
            );
        }
    }
    initial
}

/// Auto-detect whether the first line of a delimited file is a header row
/// by checking if it contains any of the requested column names.
fn detect_header_row(
    file_path: &str,
    delimiters: &[char],
    column_names: &[Box<str>],
) -> Option<usize> {
    if column_names.is_empty() {
        return None;
    }
    let first_line = std::io::BufRead::lines(std::io::BufReader::new(
        std::fs::File::open(file_path).ok()?,
    ))
    .next()?
    .ok()?;
    let tokens: HashSet<&str> = first_line.split(delimiters.as_ref()).collect();
    if column_names
        .iter()
        .any(|name| tokens.contains(name.as_ref()))
    {
        info!("Auto-detected header row in {}", file_path);
        Some(0)
    } else {
        None
    }
}

/// `coords` with one more column: `keys[i]` (dense ids from
/// `ablation::sections`) times the coordinates' full span, so groups sit
/// far apart in the spatial KNN.
fn append_batch_coordinate(coords: &Mat, keys: &[u32]) -> anyhow::Result<Mat> {
    anyhow::ensure!(
        coords.nrows() == keys.len(),
        "incompatible batch membership"
    );
    let width = (coords.max() - coords.min()).max(1.);
    let batch_coord = keys.iter().map(|&k| width * k as f32).collect();
    let bb = Mat::from_vec(coords.nrows(), 1, batch_coord);
    concatenate_horizontal(&[coords.clone(), bb])
}

#[cfg(test)]
mod tests;
