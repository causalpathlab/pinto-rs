//! `pinto apply impute` — impute full-feature expression for a new sample by
//! kNN over community propensities against a trained run's cells.
//!
//! The matching space is the per-cell community propensity — the one
//! partition-level cell representation every pinto model (`cage`, `lc`,
//! `dsvd`) publishes — so the retrieval step is uniform while the
//! query-side projection dispatches on the manifest's `command`:
//!
//! - **cage**: the full `pinto apply predict` pipeline runs first, writing its
//!   usual outputs under `{out}`; the query propensity is its result, and
//!   the reference propensity is the training run's own
//!   `{model}.propensity.parquet`. Both sides are incident-edge fractions
//!   over the same trained communities.
//! - **lc / dsvd** (no frozen feature dictionary): each cell's propensity is
//!   estimated from `{model}.feature_community.parquet` by a per-cell EM fit
//!   of a multinomial mixture over the community expression profiles —
//!   and it is estimated the SAME way for the reference cells, so the two
//!   sides come from one map rather than comparing an edge-based
//!   propensity against an expression-based one. The query's profile
//!   columns are renormalized over its matched features (a panel observes a
//!   window of each profile; conditioning on that window is what makes
//!   the per-cell likelihood proper). Profiles fitted on δ-divided counts
//!   (the manifest's `feature_community_batch_adjusted`) have both sides
//!   divided onto their scale first: the reference by the run's δ, the
//!   query by the model's for a batch it knows and by an estimate for a
//!   new one. An older model's raw-count profiles meet raw counts.
//!
//! The retrieval itself (kNN over propensity rows, softmax distance
//! weights, streamed weighted average of reference counts) lives in
//! [`data_beans::alg::retrieval_impute`]. A cell with no pairs (cage) or
//! no counts on any model feature (profiles) has a zero propensity row and
//! is skipped rather than matched arbitrarily.
//!
//! Writes `{out}.imputed.parquet` (`N_query` × `n_ref_features`), plus
//! `{out}.propensity.parquet` for the profile-projected query.

use crate::link_community::outputs::write_propensity_matrix;
use crate::predict::{predict_cage, PredictCoreArgs, ScoringArgs};
use crate::util::common::*;
use crate::util::metadata::{PintoMetadata, Run};
use crate::util::profile_match::{em_proportions, match_profiles, MatchedProfiles, EM_ITERS};
use clap::Args;
use data_beans::alg::retrieval_impute::RetrievalImputeConfig;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans::sparse_io_vector::SparseIoVec;
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::matrix::traits::IoOps;
use log::info;
use retrieval::{retrieval_impute, RefDivide};
use std::path::Path;

/// Query cells per chunk of the retrieval.
const RETRIEVAL_CHUNK: usize = 512;

#[derive(Args, Debug, Clone)]
pub struct ImputeArgs {
    #[command(flatten)]
    pub predict: PredictCoreArgs,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Reference data files (default: the model manifest's data files)",
        long_help = "Sparse backends holding the reference cells' counts — the\n\
                     training data of the --model run, in the same order.\n\
                     Defaults to the data files recorded in {model}.pinto.json;\n\
                     pass this explicitly when the data has moved."
    )]
    pub reference_data: Option<Vec<Box<str>>>,

    #[arg(
        long,
        help = "Reference propensity parquet (default {model}.propensity.parquet)",
        long_help = "Cage models only. The training run's cell propensity;\n\
                     its rows must be the cells of --reference-data, in order.\n\
                     For lc / dsvd models the reference propensity is\n\
                     re-estimated from the reference data instead, so both\n\
                     sides come from the same profile projection."
    )]
    pub reference_propensity: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 25,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Reference nearest neighbours pooled per query cell (at least 1)",
        long_help = "Named --impute-knn (not -k): the SrtInput -k sizes the\n\
                     spatial pair graph, this sizes the retrieval pool."
    )]
    pub impute_knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Softmax temperature on kNN distances (lower = sharper neighbour weights)",
        hide = true
    )]
    pub impute_temperature: f32,
}

/// `qc_given` names the cell QC flags given on the command line (see
/// [`SrtInputArgs::qc_flags_given`](crate::util::input::SrtInputArgs::qc_flags_given)),
/// which an lc / dsvd model refuses.
pub fn run_impute(args: &ImputeArgs, qc_given: &[String]) -> anyhow::Result<()> {
    // A model named by its manifest is the same model, as `pinto apply predict`
    // takes it.
    let mut args = args.clone();
    let run = open_impute_model(&mut args.predict.model)?;
    let args = &args;
    let c = &args.predict.common;
    let model = args.predict.model.as_ref();
    mkdir_parent(&c.out)?;

    // The manifest is the authority on what the model IS; artifact probing
    // (e.g. for feature_embedding.parquet) would silently reroute a cage
    // model with a misplaced file — or a `pinto apply predict` output prefix —
    // onto the profile path with different semantics. The probe survives
    // only as a fallback for pre-manifest runs.
    let meta = run.as_ref().map(|r| &r.meta);
    let is_cage = match meta.map(|m| m.command.as_str()) {
        Some("cage") => true,
        Some("lc" | "dsvd") => false,
        Some(other) => anyhow::bail!(
            "{model} is a `{other}` run; impute needs a trained cage, lc, or dsvd model"
        ),
        None => {
            let probe = Path::new(&format!("{model}.feature_embedding.parquet")).is_file();
            log::warn!(
                "{model}.pinto.json not found; dispatching by artifact probe \
                 ({} path)",
                if probe { "cage" } else { "lc/dsvd profile" }
            );
            probe
        }
    };

    // The lc / dsvd projection reads every cell as given: no QC runs there,
    // so a QC flag would be ignored. Refuse it instead.
    if !is_cage {
        anyhow::ensure!(
            qc_given.is_empty(),
            "{model} is an lc / dsvd model, whose impute runs no cell QC: {} \
             would do nothing. Filter the cells before the run.",
            qc_given.join(", ")
        );
    }

    let reference_files = resolve_reference_data(args, run.as_ref())?;
    // A cage model's reference cells are divided by the run's δ, as predict
    // divides the query. So are an lc / dsvd model's, when its manifest says
    // its profiles are batch-adjusted, its query divided alike; an older one
    // matches raw counts to raw profiles.
    let (query_prop, query_cells, ref_prop, ref_data, divide) = if is_cage {
        let (query_prop, query_cells, ref_prop, ref_data) =
            cage_propensities(args, run.as_ref(), &reference_files)?;
        let divide = if c.batch.ignore_batch {
            None
        } else {
            reference_divide(model, run.as_ref(), &ref_data.row_names()?, &ref_data)?
        };
        (query_prop, query_cells, ref_prop, ref_data, divide)
    } else {
        profile_propensities(args, run.as_ref(), &reference_files)?
    };
    anyhow::ensure!(
        query_prop.ncols() == ref_prop.ncols(),
        "query propensity has {} communities but the reference has {}; the two \
         files are from different runs",
        query_prop.ncols(),
        ref_prop.ncols()
    );

    let ref_feature_names = ref_data.row_names()?;
    let imputed = retrieval_impute(
        &query_prop,
        &ref_prop,
        &ref_data,
        &RetrievalImputeConfig {
            knn: args.impute_knn,
            temperature: args.impute_temperature,
            chunk: RETRIEVAL_CHUNK,
        },
        divide.as_ref(),
    )?;

    let imputed_path = format!("{}.imputed.parquet", c.out);
    imputed.to_parquet_with_names(
        &imputed_path,
        (Some(&query_cells), Some("cell")),
        Some(&ref_feature_names),
    )?;
    info!(
        "Wrote imputed {} × {} matrix to {}",
        imputed.nrows(),
        imputed.ncols(),
        crate::util::staging::final_str(&imputed_path)
    );
    Ok(())
}

/// The model `model` names, opened as `pinto apply predict` opens it
/// ([`crate::predict::open_model`]), with `model` set to its prefix; else,
/// for a pre-unification lc / dsvd run, its `{prefix}.metadata.json` (the
/// same schema). `None` when neither manifest is there; one that is there
/// but cannot be read is an error.
fn open_impute_model(model: &mut Box<str>) -> anyhow::Result<Option<Run>> {
    if let Some(run) = crate::predict::open_model(model)? {
        return Ok(Some(run));
    }
    let older = format!("{model}.metadata.json");
    if !Path::new(&older).is_file() {
        return Ok(None);
    }
    let meta = PintoMetadata::read(Path::new(&older))
        .map_err(|e| anyhow::anyhow!("{older}: not a pinto manifest: {e}"))?;
    Ok(Some(Run {
        manifest: older.into(),
        prefix: model.to_string(),
        meta,
    }))
}

/// The run's δ on the reference cells, when the run (manifest `run`) has
/// one: its mean on the reference's rows (`ref_rows`), matched by name, and
/// each cell's batch as the run estimated it, from the table the manifest
/// records beside its δ (an error when it records one that is nowhere),
/// else (a run fitted before it wrote one) from
/// `{model}.coord_pairs.parquet`. A cell without a batch there (with the
/// coord_pairs fallback, one the run placed in no pair) is left as is.
fn reference_divide(
    model: &str,
    run: Option<&Run>,
    ref_rows: &[Box<str>],
    ref_data: &SparseIoVec,
) -> anyhow::Result<Option<RefDivide>> {
    use crate::util::batch_effects::{
        delta_cells_path, read_cell_batches, read_delta, run_delta_cells_path, run_delta_path,
    };
    let Some(path) = run_delta_path(model, run)? else {
        return Ok(None);
    };
    let delta = read_delta(&path)?;
    let recorded = run.map(run_delta_cells_path).transpose()?.flatten();
    let batch_of = read_cell_batches(&recorded.unwrap_or_else(|| delta_cells_path(model)))?;
    let col_of: HashMap<&str, usize> = delta
        .batches
        .iter()
        .enumerate()
        .map(|(j, b)| (b.as_ref(), j))
        .collect();
    let batch_of_cell: Vec<Option<usize>> = ref_data
        .column_names()?
        .iter()
        .map(|cell| {
            batch_of
                .get(cell)
                .and_then(|b| col_of.get(b.as_ref()).copied())
        })
        .collect();
    let n = batch_of_cell.iter().flatten().count();
    info!(
        "Reference counts divided by {path}: {n} of {} cells by their batch's δ",
        batch_of_cell.len()
    );
    Ok(Some(RefDivide {
        delta: delta.mean_on_rows(ref_rows),
        batch_of_cell,
    }))
}

/// Reference data files: the explicit flag, else the model's manifest, each
/// recorded file resolved beside it as predict resolves the model's files.
fn resolve_reference_data(args: &ImputeArgs, run: Option<&Run>) -> anyhow::Result<Vec<Box<str>>> {
    if let Some(files) = args.reference_data.clone() {
        return Ok(files);
    }
    let model = args.predict.model.as_ref();
    let run = run.ok_or_else(|| {
        anyhow::anyhow!("cannot read {model}.pinto.json. Pass --reference-data explicitly.")
    })?;
    let meta = &run.meta;
    let files: Vec<Box<str>> = meta
        .data_files
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(Into::into)
        .collect();
    anyhow::ensure!(
        !files.is_empty(),
        "{model}.pinto.json records no data files; pass --reference-data explicitly"
    );
    files
        .iter()
        .map(|f| {
            run.resolve(f)
                .map(|p| p.to_string_lossy().into())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "reference data file {f} (recorded in {}) does not exist; pass \
                         --reference-data with the current location",
                        run.manifest.display()
                    )
                })
        })
        .collect()
}

/// Open a set of sparse backends. The multi-file `@basename` cell-name
/// suffixing matches what training wrote into its propensity table, so the
/// cage path's name check compares like with like.
fn open_backends(what: &str, files: &[Box<str>]) -> anyhow::Result<SparseIoVec> {
    info!("Opening {what} data ({} file(s))", files.len());
    let loaded = data_beans::aux::data_loading::read_data_on_shared_rows(
        data_beans::aux::data_loading::ReadSharedRowsArgs {
            data_files: files.to_vec(),
            // Inference: every reference and query column comes back.
            keep_empty_barcodes: true,
            ..Default::default()
        },
    )?;
    Ok(loaded.data)
}

//////////////////////////////////
// cage: predict-backed side    //
//////////////////////////////////

/// Run `pinto apply predict` for the query propensity; read the training run's
/// propensity for the reference side. The reference opens only after
/// predict has succeeded, so a preloaded atlas is not held in memory
/// through the whole predict pass.
fn cage_propensities(
    args: &ImputeArgs,
    run: Option<&Run>,
    reference_files: &[Box<str>],
) -> anyhow::Result<(Mat, Vec<Box<str>>, Mat, SparseIoVec)> {
    let model = args.predict.model.as_ref();

    info!("cage model: projecting the query sample through `pinto apply predict`");
    let (query_prop, query_cells) = predict_cage(&args.predict, &ScoringArgs::default(), run)?;

    let ref_data = open_backends("reference", reference_files)?;
    let ref_prop_path = args
        .reference_propensity
        .clone()
        .unwrap_or_else(|| format!("{model}.propensity.parquet").into());
    let none = HashSet::default();
    let (ref_prop, _, _, ref_cells) =
        crate::util::parquet_io::read_propensity(Path::new(ref_prop_path.as_ref()), &none)?;

    // The retrieval reads reference cells by column, so the propensity's
    // rows must sit on the backend's column axis: paired by name, since a
    // run keeps only the cells it could place (those with coordinates, for
    // one) and the files hold every cell.
    let data_cells = ref_data.column_names()?;
    let ref_prop =
        align_to_columns(&ref_prop_path, ref_prop, &ref_cells, &data_cells).map_err(|e| {
            anyhow::anyhow!(
            "{e:#}; pass the training run's own data files (in training order) as --reference-data"
        )
        })?;
    Ok((query_prop, query_cells, ref_prop, ref_data))
}

/// `prop`'s rows, named `prop_cells`, moved onto the axis `data_cells`: a
/// data cell the run never placed gets a zero row, which the retrieval never
/// returns; a propensity cell absent from the data means the files are not
/// the run's. The identity when the two axes already agree. Callers add
/// which files to pass instead.
pub(crate) fn align_to_columns(
    path: &str,
    prop: Mat,
    prop_cells: &[Box<str>],
    data_cells: &[Box<str>],
) -> anyhow::Result<Mat> {
    if prop_cells == data_cells {
        return Ok(prop);
    }
    let col_of: HashMap<&str, usize> = data_cells
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_ref(), i))
        .collect();
    anyhow::ensure!(
        col_of.len() == data_cells.len(),
        "{path}: the data's cell names are not unique, so its cells cannot be paired \
         with the propensity's by name"
    );
    let mut aligned = Mat::zeros(data_cells.len(), prop.ncols());
    let mut missing = 0usize;
    for (r, cell) in prop_cells.iter().enumerate() {
        match col_of.get(cell.as_ref()) {
            Some(&c) => aligned.set_row(c, &prop.row(r)),
            None => missing += 1,
        }
    }
    anyhow::ensure!(
        missing == 0,
        "{path}: {missing} of its {} cells are not in the data",
        prop_cells.len()
    );
    let unplaced = data_cells.len() - prop_cells.len();
    if unplaced > 0 {
        info!(
            "{unplaced} of the {} reference cells were not placed by the run (no coordinates, \
             or filtered out) and sit out the retrieval",
            data_cells.len()
        );
    }
    Ok(aligned)
}

///////////////////////////////////////
// lc / dsvd: profile projection    //
///////////////////////////////////////

/// What [`profile_propensities`] gives: the query's propensity and cell
/// names, the reference's propensity and data, and the divide its counts
/// take.
type ProfileSides = (Mat, Vec<Box<str>>, Mat, SparseIoVec, Option<RefDivide>);

/// Estimate BOTH sides' propensities from the model's feature-community
/// profiles (manifest `run`), one EM per cell, then write the query's under
/// `{out}`. A model whose manifest says its profiles are batch-adjusted
/// has both sides put on their scale first: the reference divided by the
/// run's δ ([`reference_divide`]), the query by the model's δ for each
/// batch it knows and an estimate for a new one
/// ([`crate::predict::delta::adjusted_query_delta`]).
fn profile_propensities(
    args: &ImputeArgs,
    run: Option<&Run>,
    reference_files: &[Box<str>],
) -> anyhow::Result<ProfileSides> {
    let c = &args.predict.common;
    let model = args.predict.model.as_ref();
    let adjusted = crate::predict::delta::batch_adjusted(
        model,
        run,
        c.batch.ignore_batch,
        crate::predict::delta::QueryReader::Impute,
    )?;
    if args.reference_propensity.is_some() {
        log::warn!(
            "--reference-propensity is ignored for an lc / dsvd model: the reference \
             propensity is re-estimated from the reference data so both sides come \
             from the same profile projection"
        );
    }

    let gc_path = format!("{model}.feature_community.parquet");
    anyhow::ensure!(
        Path::new(&gc_path).is_file(),
        "{model}: feature_community.parquet does not exist — not an lc / dsvd model prefix"
    );
    let (profiles, model_features) =
        crate::util::parquet_io::read_feature_community(Path::new(&gc_path))?;
    info!(
        "lc/dsvd model: projecting propensities through {} feature × {} community profiles",
        profiles.nrows(),
        profiles.ncols()
    );

    anyhow::ensure!(!c.data_files.is_empty(), "impute: no data files given");
    let mut query_data = open_backends("query", &c.data_files)?;
    let feature_kind = args
        .predict
        .feature_name_mode
        .resolve_kind(&query_data.row_names()?);
    let query_cells = query_data.column_names()?;
    let projection = ProfileProjection {
        profiles: &profiles,
        model_features: &model_features,
        feature_kind: &feature_kind,
        iters: EM_ITERS,
    };
    let query_divide = match run.filter(|_| adjusted) {
        Some(run) => {
            let membership = crate::util::input::read_batch_labels(
                &c.to_read_args_with_kind(feature_kind.clone()),
                &query_data,
            )?;
            crate::predict::delta::adjusted_query_delta(
                model,
                run,
                c,
                &mut query_data,
                &membership,
            )?
            .map(|(delta, batch)| RefDivide {
                delta,
                batch_of_cell: batch.into_iter().map(|b| Some(b as usize)).collect(),
            })
        }
        None => None,
    };
    let query_prop =
        project_profile_propensity(&query_data, &projection, query_divide.as_ref(), "query")?;
    write_propensity_matrix(&c.out, &query_prop, &query_cells)?;
    info!(
        "Wrote {}.propensity.parquet (profile-projected)",
        crate::util::staging::final_str(&c.out)
    );
    // Everything needed from the query is extracted; release it before the
    // reference opens so two (possibly preloaded) datasets never coexist;
    // dropping it also gives its preload budget back to the reference.
    drop(query_data);

    // The reference goes through the SAME map — not through the training
    // run's edge-based propensity — so the kNN compares like with like.
    let ref_data = open_backends("reference", reference_files)?;
    let ref_divide = if adjusted {
        reference_divide(model, run, &ref_data.row_names()?, &ref_data)?
    } else {
        None
    };
    let ref_prop =
        project_profile_propensity(&ref_data, &projection, ref_divide.as_ref(), "reference")?;
    Ok((query_prop, query_cells, ref_prop, ref_data, ref_divide))
}

/// The model's profiles a dataset's cells are projected through.
struct ProfileProjection<'a> {
    profiles: &'a Mat,
    model_features: &'a [Box<str>],
    feature_kind: &'a data_beans::aux::feature_names::FeatureNameKind,
    iters: usize,
}

struct ProfileEmParam<'a> {
    profiles: MatchedProfiles,
    /// Already clamped to ≥ 1.
    iters: usize,
    divide: Option<&'a RefDivide>,
}

/// `[N, K]` per-cell community proportions by EM over the profile mixture
/// ([`em_proportions`] per cell). `p̃_k` is the profile conditioned on this
/// dataset's matched features. A cell with no counts on any matched feature
/// keeps a zero row (skipped by the retrieval core). With `divide`, each
/// count is divided by its row's δ in its cell's batch first.
fn project_profile_propensity(
    data: &SparseIoVec,
    p: &ProfileProjection<'_>,
    divide: Option<&RefDivide>,
    what: &str,
) -> anyhow::Result<Mat> {
    let k = p.profiles.ncols();
    let matched = match_profiles(
        &data.row_names()?,
        p.profiles,
        p.model_features,
        p.feature_kind,
        what,
        data.num_columns(),
    )?;
    let param = ProfileEmParam {
        profiles: matched,
        iters: p.iters.max(1),
        divide,
    };
    let n = data.num_columns();
    let mut prop_kn = Mat::zeros(k, n);
    data.visit_columns_by_block(&profile_em_visitor, &param, &mut prop_kn, None)?;
    Ok(prop_kn.transpose())
}

fn profile_em_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    param: &ProfileEmParam<'_>,
    arc_prop_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let mut csc = data.read_columns_csc(lb..ub)?;
    if let Some(d) = param.divide {
        crate::util::batch_effects::divide_columns(&mut csc, &d.delta, |j| d.batch_of_cell[lb + j]);
    }
    let k = param.profiles.p_t.nrows();
    let mut chunk = Mat::zeros(k, ub - lb);

    let mut obs: Vec<(usize, f64)> = Vec::new();
    let mut pi = vec![0.0f64; k];
    for c_local in 0..csc.ncols() {
        let col = csc.col(c_local);
        // The cell's counts on the matched window.
        obs.clear();
        obs.extend(
            col.row_indices()
                .iter()
                .zip(col.values().iter())
                .filter_map(|(&row, &v)| {
                    param.profiles.row_to_matched[row].map(|slot| (slot, f64::from(v)))
                }),
        );
        if !em_proportions(&obs, &param.profiles.p_t, param.iters, &mut pi) {
            continue; // zero row → skipped downstream
        }
        for (c, &pi_c) in pi.iter().enumerate() {
            chunk[(c, c_local)] = pi_c as f32;
        }
    }

    let mut prop_kn = arc_prop_kn.lock().expect("lock propensity in impute");
    prop_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
}

#[path = "impute/retrieval.rs"]
mod retrieval;

#[cfg(test)]
#[path = "impute/tests.rs"]
mod tests;
