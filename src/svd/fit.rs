use crate::link_community::profiles::{
    compute_propensity_and_feature_community_stat, PropensityReportConfig,
};
use crate::util::cell_pairs::*;
use crate::util::common::*;
use crate::util::graph_coarsen::*;
use crate::util::srt_pipeline::{
    preprocess_srt, topology_graph, BatchEstimate, FeatureAxisMode, Seeding, SrtPreprocessConfig,
    SrtPreprocessed,
};
use data_beans::alg::cell_pairs::CellPairs;
use data_beans::alg::random_projection::*;

use clap::Parser;
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::param::dmatrix_gamma::GammaMatrix;
use legume_numeric::param::traits::*;

#[derive(Parser, Debug, Clone)]
pub struct SrtDeltaSvdArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[command(flatten)]
    pub modules: crate::util::feature_modules::FeatureModulesArgs,

    #[command(flatten)]
    pub ablation: crate::util::ablation::AblationArgs,

    #[command(flatten)]
    pub pair_fold: crate::util::batch_fold::PairFoldArgs,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of SVD components for latent pair representation"
    )]
    n_latent_topics: usize,

    /// How the pair latent becomes link communities. Shared verbatim with
    /// `cage` — the k-means fallback width is `--n-latent-topics` here.
    #[command(flatten)]
    edge_clustering: crate::util::edge_clustering::EdgeClusterArgs,
}

/// Input for the pair-delta visitor (finest coarsening level only).
pub(super) struct FusedDeltaInput<'a> {
    pub(super) batch_effect: Option<&'a Mat>,
    /// Each pair's own batch fold, in pair order, in place of `batch_effect`.
    pub(super) pair_fold: Option<&'a crate::util::batch_fold::PairFold>,
    pub(super) pair_to_sample: &'a [usize],
}

/// Accumulated shared/difference statistics per feature per sample.
///
/// For each cell pair (left, right) and each feature g, with
/// `y = ln(1 + 1e4 x / Σ x)` each cell's log-normalized counts
/// ([`read_log_normalized`]):
///   shared_g = y_left_g + y_right_g
///   diff_g   = |y_left_g - y_right_g|
pub(crate) struct PairDeltaCollapsedStat {
    pub(super) shared_ds: Mat,
    pub(super) diff_ds: Mat,
    size_s: DVec,
    n_features: usize,
    n_samples: usize,
}

impl PairDeltaCollapsedStat {
    /// Both accumulator columns of one sample, borrowed together so the
    /// scatter writes shared and diff in a single pass.
    fn columns_mut(
        &mut self,
        sample: usize,
    ) -> (
        nalgebra::DVectorViewMut<'_, f32>,
        nalgebra::DVectorViewMut<'_, f32>,
    ) {
        (
            self.shared_ds.column_mut(sample),
            self.diff_ds.column_mut(sample),
        )
    }

    pub(crate) fn new(n_features: usize, n_samples: usize) -> Self {
        Self {
            shared_ds: Mat::zeros(n_features, n_samples),
            diff_ds: Mat::zeros(n_features, n_samples),
            size_s: DVec::zeros(n_samples),
            n_features,
            n_samples,
        }
    }

    pub(crate) fn optimize(
        &self,
        hyper_param: Option<(f32, f32)>,
    ) -> anyhow::Result<PairDeltaParameters> {
        let (a0, b0) = hyper_param.unwrap_or((1_f32, 1_f32));
        let shape = (self.n_features, self.n_samples);

        let mut shared = GammaMatrix::new(shape, a0, b0);
        let mut diff = GammaMatrix::new(shape, a0, b0);

        let size_s = &self.size_s.transpose();
        let sample_size_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        info!("Calibrating pair delta statistics");

        shared.update_stat(&self.shared_ds, &sample_size_ds);
        shared.calibrate();
        diff.update_stat(&self.diff_ds, &sample_size_ds);
        diff.calibrate();

        info!("Resolved pair delta collapsed statistics");

        Ok(PairDeltaParameters { shared, diff })
    }
}

pub(crate) struct PairDeltaParameters {
    pub(crate) shared: GammaMatrix,
    pub(crate) diff: GammaMatrix,
}

/// dsvd's per-cell projection: corrected by the cells' batches
/// (`batch_membership`) unless `--ignore-batch`, at inputs `c`'s `-p` and
/// block size. Preprocessing's (`shared`) is that projection with a δ
/// (`has_delta`: the one its estimate collapsed on), under
/// `--ignore-batch` (uncorrected), and is taken as it is; otherwise (one
/// batch, or an estimate without a δ) preprocessing's is uncorrected and
/// dsvd's is taken here.
pub(crate) fn dsvd_cell_projection(
    shared: Option<RandColProjOut>,
    has_delta: bool,
    c: &crate::util::input::SrtInputArgs,
    data_vec: &SparseIoVec,
    batch_membership: &[Box<str>],
) -> anyhow::Result<RandColProjOut> {
    match shared {
        Some(proj) if has_delta || c.batch.ignore_batch => Ok(proj),
        _ => {
            info!("Per-cell random projection...");
            data_vec.project_columns_with_batch_correction(
                c.collapse.proj_dim,
                None,
                (!c.batch.ignore_batch).then_some(batch_membership),
            )
        }
    }
}

/// Cell-pair SVD pipeline with shared/difference channels:
///
/// 1. Load data + coordinates
/// 2. Estimate batch effects
/// 3. Build spatial cell-cell KNN graph
/// 4. Random projection → assign pairs to samples
/// 5. Collapse: compute shared/diff per feature per sample, on counts
///    divided by batch effects, scaled per cell to 1e4 and logged
/// 6. Fit Poisson-Gamma on each channel
/// 7. SVD on vertically stacked [shared; diff] posterior log means
/// 8. Nystrom projection → per-pair latent codes
/// 9. Export dictionary + pair latents
pub fn fit_srt_delta_svd(args: &SrtDeltaSvdArgs) -> anyhow::Result<()> {
    let c = &args.common;
    mkdir_parent(&c.out)?;

    anyhow::ensure!(c.n_pseudobulk > 0, "n_pseudobulk must be > 0");
    anyhow::ensure!(args.n_latent_topics > 0, "n_latent_topics must be > 0");

    // 1-3. Load + KNN + batch effects (no Fisher weights).
    let SrtPreprocessed {
        fold_written,
        cell_fold,
        pair_fold,
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        delta_membership,
        covariates_binned,
        section_ids: _,
        seeding,
        batch_labels,
        batch_effects: batch_db,
        graph,
        knn,
        base_batch_corrected,
        spatial_graph,
        edge_source,
        cell_proj: shared_cell_proj,
        feature_axis: _,
        n_cells,
        n_rows: n_features,
        ablation,
        permuted_coordinates: _,
    } = preprocess_srt(SrtPreprocessConfig {
        covariate_model: None,
        common: c,
        ablation: Some(&args.ablation),
        batch_effects: BatchEstimate::Fit,
        feature_axis: FeatureAxisMode::Rows,
        // the per-cell projection below, when it is preprocessing's
        // ([`dsvd_cell_projection`])
        cell_projection: true,
        auto_batch_memory_check: None,
        batch_fold: crate::util::srt_pipeline::BatchFoldKind::Pairs(
            args.pair_fold.pair_fold_states,
        ),
        feature_kind: None,
    })?;
    // Only the mean divides; the full posterior is the file preprocessing wrote.
    let batch_db = batch_db
        .map(crate::util::batch_effects::BatchPosterior::into_mean)
        .transpose()?;
    let feature_names = data_vec.row_names()?;

    // Wrap graph with data for pair-level operations
    let srt_cell_pairs = SrtCellPairs::with_graph(
        &data_vec,
        &coordinates,
        &graph,
        edge_source.as_deref(),
        Some(&batch_membership),
    );

    srt_cell_pairs.write_coord_pairs(&c.out, &coordinate_names)?;

    // 4. Per-cell random projection
    let mut cell_proj = dsvd_cell_projection(
        shared_cell_proj,
        batch_db.is_some(),
        c,
        &data_vec,
        &delta_membership,
    )?;

    // 5. Graph-constrained coarsening + multi-level assignment
    info!(
        "Graph coarsening + multi-level assignment ({} levels, n_clusters={})...",
        c.collapse.num_levels, c.n_pseudobulk
    );

    let batch_ref = batch_db.as_ref();
    let has_batch_effects = batch_db.is_some();

    let topology = topology_graph(&graph, &spatial_graph);
    let ml = graph_coarsen_multilevel(
        topology,
        &mut cell_proj.proj,
        srt_cell_pairs.inner.pairs(),
        CoarsenConfig {
            n_clusters: c.n_pseudobulk,
            num_levels: c.collapse.num_levels,
            seeding: seeding.as_ref().map(Seeding::params),
            modularity_veto: None,
        },
    );

    // Only the FINEST level is ever read (the Poisson-Gamma fit below),
    // so only that one is accumulated. The coarser levels cost
    // `n_features x n_samples` dense accumulators EACH, and every one of
    // them also multiplied the work done while the pass below holds its
    // lock; they were built and dropped unused.
    let finest_samples = *ml
        .all_num_samples
        .last()
        .ok_or(anyhow::anyhow!("no levels"))?;
    let finest_pair_to_sample = ml
        .all_pair_to_sample
        .last()
        .ok_or(anyhow::anyhow!("no levels"))?;
    info!(
        "Accumulating pair-delta statistics at the finest level ({} samples of {:?})",
        finest_samples, ml.all_num_samples
    );
    let mut collapsed = PairDeltaCollapsedStat::new(n_features, finest_samples);

    let fused_input = FusedDeltaInput {
        batch_effect: batch_ref,
        pair_fold: pair_fold.as_ref(),
        pair_to_sample: finest_pair_to_sample,
    };

    srt_cell_pairs.inner.visit_pairs_by_block(
        &fused_pair_delta_visitor,
        &fused_input,
        &mut collapsed,
        None,
    )?;

    // 6. Fit Poisson-Gamma (finest level)
    info!("Fitting Poisson-Gamma model...");
    let collapsed_stat = &collapsed;
    let params = collapsed_stat.optimize(None)?;

    // 7. SVD on [shared; diff] posterior log means
    info!("Randomized SVD ({} components)...", args.n_latent_topics);

    let training_dm = concatenate_vertical(&[
        params.shared.posterior_log_mean().scale_columns(),
        params.diff.posterior_log_mean().scale_columns(),
    ])?;

    let (u_dk, s_k, _) = training_dm.rsvd(args.n_latent_topics)?;
    let basis_dk = nystrom_basis(&u_dk, &s_k);

    // Write dictionary
    let dict_row_names: Vec<Box<str>> = feature_names
        .iter()
        .map(|g| format!("{}@shared", g).into_boxed_str())
        .chain(
            feature_names
                .iter()
                .map(|g| format!("{}@diff", g).into_boxed_str()),
        )
        .collect();

    u_dk.to_parquet_with_names(
        &(c.out.to_string() + ".basis.parquet"),
        (Some(&dict_row_names), Some("feature")),
        None,
    )?;

    // 8. Nystrom projection
    info!("Nystrom projection...");

    let mut pair_latent_kn = Mat::zeros(args.n_latent_topics, srt_cell_pairs.inner.num_pairs());

    let nystrom_input = NystromPairInput {
        basis_shared: basis_dk.rows(0, n_features).clone_owned(),
        basis_diff: basis_dk.rows(n_features, n_features).clone_owned(),
        batch_effect: batch_db,
        pair_fold: pair_fold.as_ref(),
    };

    srt_cell_pairs.inner.visit_pairs_by_block(
        &nystrom_pair_delta_visitor,
        &nystrom_input,
        &mut pair_latent_kn,
        None,
    )?;

    // 9. Export
    // L2-normalize each pair's latent vector so downstream clustering
    // is driven by direction rather than magnitude.
    pair_latent_kn.normalize_columns_inplace();

    // One `[E × T]` copy, written out and then clustered — the shared routine
    // takes pairs as rows, so this is the same buffer both times.
    let pair_latent_nk = pair_latent_kn.transpose();
    pair_latent_nk.to_parquet_with_names(
        &(c.out.to_string() + ".latent.parquet"),
        (None, Some("cell_pair")),
        None,
    )?;

    // 10. Propensity + dictionary
    let edges = srt_cell_pairs.inner.pairs();
    let batch_of_cell = nystrom_input
        .batch_effect
        .as_ref()
        .map(|_| crate::util::batch_effects::batch_ids(&delta_membership, &batch_labels));

    // Leiden by default, same as `cage`: the SVD width is chosen for how much
    // variance to keep, which is no reason for the pairs to fall into exactly
    // that many interaction regimes. `pinto apply propensity` still re-cuts the same latent
    // at a fixed K when you want one.
    let n_clusters = compute_propensity_and_feature_community_stat(
        &pair_latent_nk,
        edges,
        &data_vec,
        n_cells,
        &PropensityReportConfig {
            clustering: args.edge_clustering.resolve(c.seed),
            // `dsvd` never resolves a feature axis (it stacks its two channels on
            // the row axis), so its feature-community table stays row-keyed.
            feature_axis: None,
            edge_kind: srt_cell_pairs.edge_kind.as_deref(),
            // batch-adjusted as the pairs are
            // each cell's own fold, else δ
            divisor: match cell_fold.as_ref() {
                Some(f) => Some(f.table.divisor().into()),
                None => crate::util::batch_effects::BatchDivisor::of(
                    nystrom_input.batch_effect.as_ref(),
                    batch_of_cell.as_deref(),
                )
                .map(Into::into),
            },
        },
        &c.out,
    )?
    .n_clusters;

    {
        use crate::util::metadata::{create_dsvd_metadata, RunInputs};
        let coord_file_str = c.coord_files_joined();
        let meta = create_dsvd_metadata(&RunInputs {
            prefix: &c.out,
            data_files: &c.data_files,
            coord_file: coord_file_str.as_deref(),
            coord_columns: &coordinate_names,
            n_cells,
            n_features: data_vec.num_rows(),
            n_edges: edges.len(),
            graph: crate::util::metadata::GraphParams::of(&knn, base_batch_corrected),
            ablation,
            k: n_clusters,
            delta: has_batch_effects.then(|| {
                crate::util::metadata::DeltaWritten::fit(c, fold_written, covariates_binned)
            }),
        });
        let meta_path = std::path::PathBuf::from(crate::util::metadata::manifest_path(&c.out));
        meta.write(&meta_path)?;
        info!(
            "Wrote {}",
            crate::util::staging::final_path(&meta_path).display()
        );
    }

    info!("Done");
    Ok(())
}

/// Each cell (column) of `y` scaled to [`LIBRARY_SCALE`] and logged, in
/// place on its stored entries: the crate's one transform on sparse columns
/// ([`normalize_log`], scanpy's `normalize_total(target_sum=1e4)` then
/// `log1p`; Wolf, Angerer and Theis 2018, Genome Biology 19:15). The total
/// is the cell's own, over every row it has.
///
/// A cell whose total is not a positive finite number is set to zero first,
/// so it reads as an empty cell and not as NaN. That is a cell of stored
/// zeros after a batch divide (`0 / 0`, the reader keeps stored zeros), and
/// also any cell holding a count that is not finite.
pub(super) fn log_normalize_cells(y: &mut nalgebra_sparse::CscMatrix<f32>) {
    for mut cell in y.col_iter_mut() {
        let values = cell.values_mut();
        let total: f32 = values.iter().sum();
        if !(total > 0.0 && total.is_finite()) {
            values.fill(0.0);
        }
    }
    normalize_log(y);
}

/// The columns of `cells` as both pair visitors take them: divided by the
/// batch effects of each cell's batch when there are any, then
/// log-normalized ([`log_normalize_cells`]) on the total after the divide.
/// Sparse throughout.
///
/// The divide scales a cell by a rate of its own as it divides. Where every
/// effect the cell's counts meet is positive (the fitted effects are a Gamma
/// posterior mean), that rate is a constant of the cell: it cancels in the
/// rescaling, and a cell reads the same with a unit batch effect as with
/// none. A count whose effect is not positive is left as it is, by the rate
/// too, so for such a cell the rate does not cancel.
///
/// With `pair_fold` (the run's per-pair batch fold and the block's first
/// pair, the cells one endpoint of each pair in order), each endpoint is
/// divided by its pair's own fold instead, scale-matched, then `log1p`
/// ([`crate::util::batch_fold::divide_by_pair_folds_log1p`]): the library
/// rescale would cancel the fold.
fn read_log_normalized(
    data: &CellPairs,
    cells: impl Iterator<Item = usize> + Clone,
    batch_effect: Option<&Mat>,
    pair_fold: Option<(&crate::util::batch_fold::PairFold, usize, bool)>,
) -> anyhow::Result<nalgebra_sparse::CscMatrix<f32>> {
    let mut y = data.data.read_columns_csc(cells.clone())?;
    if let Some((table, first, second)) = pair_fold {
        crate::util::batch_fold::divide_by_pair_folds_log1p(&mut y, table, first, second);
        return Ok(y);
    }
    if let Some(delta_db) = batch_effect {
        let batches = data.data.get_batch_membership(cells);
        y.adjust_by_division_of_selected_inplace(delta_db, &batches);
    }
    log_normalize_cells(&mut y);
    Ok(y)
}

/// Block visitor: read each pair's two columns once, merge their sparse
/// feature sets, and scatter the touched features into the finest level's stats.
pub(super) fn fused_pair_delta_visitor(
    bound: (usize, usize),
    data: &CellPairs,
    input: &FusedDeltaInput,
    arc_stats: Arc<Mutex<&mut PairDeltaCollapsedStat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs()[lb..ub];
    let n_pairs = ub - lb;

    // Two sequential reads, deliberately NOT deduplicated. Endpoints do
    // repeat within a block, but a deduplicated read forces a per-pair
    // index lookup and scattered column access afterwards, and measured
    // slower on a large pair graph than simply reading both sides in
    // order and zipping them.
    let left = pairs.iter().map(|x| x.0);
    let right = pairs.iter().map(|x| x.1);
    let y_left = read_log_normalized(
        data,
        left,
        input.batch_effect,
        input.pair_fold.map(|t| (t, lb, false)),
    )?;
    let y_right = read_log_normalized(
        data,
        right,
        input.batch_effect,
        input.pair_fold.map(|t| (t, lb, true)),
    )?;

    // Per-pair SPARSE contributions: a pair touches its nonzero features, not
    // all of them, and the lock below only pays for what it touches.
    // The two CSC columns arrive with sorted, unique row indices, so the
    // union is a two-pointer merge with no map and no per-pair allocation
    // (the union is at most nnz(left) + nnz(right) entries, reserved once).
    let mut entries: Vec<(u32, f32, f32)> = Vec::with_capacity(y_left.nnz() + y_right.nnz());
    let mut offsets: Vec<usize> = Vec::with_capacity(n_pairs + 1);
    offsets.push(0);

    for (lc, rc) in y_left.col_iter().zip(y_right.col_iter()) {
        let (li, lv) = (lc.row_indices(), lc.values());
        let (ri, rv) = (rc.row_indices(), rc.values());
        let (mut a, mut b) = (0usize, 0usize);
        loop {
            match (li.get(a), ri.get(b)) {
                (Some(&lg), Some(&rg)) => match lg.cmp(&rg) {
                    std::cmp::Ordering::Less => {
                        let l = lv[a];
                        entries.push((lg as u32, l, l));
                        a += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        let r = rv[b];
                        entries.push((rg as u32, r, r));
                        b += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        let l = lv[a];
                        let r = rv[b];
                        entries.push((lg as u32, l + r, (l - r).abs()));
                        a += 1;
                        b += 1;
                    }
                },
                (Some(&lg), None) => {
                    let l = lv[a];
                    entries.push((lg as u32, l, l));
                    a += 1;
                }
                (None, Some(&rg)) => {
                    let r = rv[b];
                    entries.push((rg as u32, r, r));
                    b += 1;
                }
                (None, None) => break,
            }
        }
        offsets.push(entries.len());
    }

    // Scatter under the lock. Held only for the touched features of one
    // level, where it used to cover every feature row of every level.
    let mut stats = arc_stats.lock().expect("lock fused delta stats");
    for local_idx in 0..n_pairs {
        let sample = input.pair_to_sample[lb + local_idx];
        let (mut col_shared, mut col_diff) = stats.columns_mut(sample);
        for &(g, sh, df) in &entries[offsets[local_idx]..offsets[local_idx + 1]] {
            col_shared[g as usize] += sh;
            col_diff[g as usize] += df;
        }
        stats.size_s[sample] += 1.0;
    }

    Ok(())
}

/// Shared input for Nystrom pair-delta projection.
pub(super) struct NystromPairInput<'a> {
    pub(super) basis_shared: Mat,
    pub(super) basis_diff: Mat,
    pub(super) batch_effect: Option<Mat>,
    /// Each pair's own batch fold, in pair order, in place of `batch_effect`.
    pub(super) pair_fold: Option<&'a crate::util::batch_fold::PairFold>,
}

/// Nystrom projection visitor: project each pair onto the split basis.
///
/// For each pair and each feature present in either cell, with `y` the
/// cells' log-normalized counts ([`read_log_normalized`]), as collapsed:
///
/// ```text
/// shared = y_left + y_right
/// diff   = |y_left - y_right|
/// proj  += shared * basis_shared[feature] + diff * basis_diff[feature]
/// ```
pub(super) fn nystrom_pair_delta_visitor(
    bound: (usize, usize),
    data: &CellPairs,
    shared_in: &NystromPairInput<'_>,
    arc_proj: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs()[lb..ub];
    let left = pairs.iter().map(|pp| pp.0);
    let right = pairs.iter().map(|pp| pp.1);

    let y_left = read_log_normalized(
        data,
        left,
        shared_in.batch_effect.as_ref(),
        shared_in.pair_fold.map(|t| (t, lb, false)),
    )?;
    let y_right = read_log_normalized(
        data,
        right,
        shared_in.batch_effect.as_ref(),
        shared_in.pair_fold.map(|t| (t, lb, true)),
    )?;

    let n_topics = shared_in.basis_shared.ncols();
    let n_pairs_block = ub - lb;
    let mut local_proj = Mat::zeros(n_topics, n_pairs_block);

    for (pair_idx, (left_col, right_col)) in y_left.col_iter().zip(y_right.col_iter()).enumerate() {
        let right_log: HashMap<usize, f32> = right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
            .map(|(&g, &v)| (g, v))
            .collect();

        let mut proj_k = DVec::zeros(n_topics);
        let mut left_visited: HashSet<usize> = Default::default();

        for (&feature, &log_left) in left_col.row_indices().iter().zip(left_col.values().iter()) {
            let log_right = right_log.get(&feature).copied().unwrap_or(0.0);
            let sigma = log_left + log_right;
            let delta = (log_left - log_right).abs();
            proj_k += sigma * &shared_in.basis_shared.row(feature).transpose();
            proj_k += delta * &shared_in.basis_diff.row(feature).transpose();
            left_visited.insert(feature);
        }

        // Right-only features: log_left = 0 → sigma = log_right, delta = log_right
        for (&feature, _) in right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
        {
            if !left_visited.contains(&feature) {
                let log_right = right_log[&feature];
                proj_k += log_right * &shared_in.basis_shared.row(feature).transpose();
                proj_k += log_right * &shared_in.basis_diff.row(feature).transpose();
            }
        }

        local_proj.column_mut(pair_idx).copy_from(&proj_k);
    }

    let mut proj = arc_proj.lock().expect("lock nystrom pair delta proj");
    proj.columns_range_mut(lb..ub).copy_from(&local_proj);

    Ok(())
}
