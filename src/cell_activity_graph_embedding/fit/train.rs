//! The training loop, and the probe that sizes its chunks.

use crate::cell_activity_graph_embedding::args::{
    CellActivityGraphEmbeddingArgs, CONVERGENCE_TOL, GRAD_CLIP,
};
use crate::cell_activity_graph_embedding::feature_chain_sampler::{
    EdgeChainBatch, FeatureGatedChainSampler,
};
use crate::cell_activity_graph_embedding::loss::{
    cage_cell_edge_nce_loss, cage_nce_loss_per_feature_level, localize_edges, CageLossOut,
    EdgeChunk, EdgeRows,
};
use crate::cell_activity_graph_embedding::pretrained;
use crate::cell_activity_graph_embedding::row_adam::RowAdam;
use crate::util::common::*;
use crate::util::score_trace::ScoreEntry;

use super::module_cells::CellCounts;
use graph_embedding_util::loss::{
    draw_gene_keep_mask, embedding_ridge, log_membership_diagnostics, masked_membership,
    module_priors, module_softmax_loss,
};
use graph_embedding_util::loss::{NceObjective, UnitChainBatch};
use graph_embedding_util::model::JointEmbedModel;
use graph_embedding_util::stop::setup_stop_handler;
use graph_embedding_util::FeatureModuleConfig;
use legume_numeric::candle::candle_core::backprop::GradStore;
use legume_numeric::candle::candle_core::{Device, Tensor, Var};
use legume_numeric::candle::candle_nn::AdamW;
use legume_numeric::candle::fast_index::gather_rows;
use legume_numeric::candle::grad_clip::clip_grad_global_norm;
use legume_numeric::candle::lora::LoraPlus;
use legume_numeric::candle::vae::{clip_and_step_dense_all, PhaseTimers};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::RngExt;
use rand::SeedableRng;
use std::sync::atomic::Ordering;

/// Classic SGD chunk over features: the validated fixed default on CPU and
/// the CAP of the CUDA auto-probe (auto-sizing only ever shrinks it).
/// The args long_help describes this value; keep the two in step.
const FEATURE_BATCH_DEFAULT: usize = 2048;
/// The probe cycles this many times the requested count of features before
/// giving up on filling a pilot minibatch: `sample` can miss per
/// (feature, batch), so some slack is needed, and the bound keeps a sparse
/// sampler from turning the probe into a long scan.
const PROBE_OVERSAMPLE: usize = 4;

/// One forward-only pilot minibatch of `n` FEATURES for the GPU memory
/// probe, each feature expanded to every batch label that samples, exactly
/// as a real training chunk expands (`perm.chunks(feature_batch_size)`
/// then per-feature, per-batch sampling). The unit measured must be the
/// unit the chunk size is spent in: counting (feature, batch) pairs here
/// would under-measure a multi-batch run by the mean batches per feature.
/// Samples through the REAL sampler, scores with the REAL loss, returns
/// the un-backwarded loss tensor. Never steps the optimizer, so no
/// parameter moves before epoch 0.
fn probe_forward(
    inputs: &ProbeInputs<'_>,
    n_probe_features: usize,
    probe_rng: &mut SmallRng,
) -> legume_numeric::candle::candle_core::Result<Tensor> {
    let ProbeInputs {
        args,
        active_features,
        n_exp_batches,
        sampler,
        model,
        pairs,
        dev,
    } = *inputs;
    let mut mini: Vec<(u32, u32, Draw)> = Vec::with_capacity(n_probe_features);
    let mut features_sampled = 0usize;
    'fill: for &g in active_features
        .iter()
        .cycle()
        .take(PROBE_OVERSAMPLE * n_probe_features)
    {
        let before = mini.len();
        for b in 0..n_exp_batches {
            if let Some(d) = Draw::sample(sampler, pairs.is_some(), g, b, probe_rng) {
                mini.push((g as u32, b as u32, d));
            }
        }
        if mini.len() > before {
            features_sampled += 1;
            if features_sampled >= n_probe_features {
                break 'fill;
            }
        }
    }
    if mini.is_empty() {
        return Err(legume_numeric::candle::candle_core::Error::Msg(
            "probe sampled nothing".into(),
        ));
    }
    let (feature_ids_u32, (row_blocks, draws)): (Vec<u32>, (Vec<u32>, Vec<Draw>)) =
        mini.into_iter().map(|(g, b, d)| (g, (b, d))).unzip();
    let out = chunk_loss(
        model,
        pairs.map(|p| (p.z, p.edges)),
        Chunk {
            draws,
            feature_ids: &feature_ids_u32,
            row_blocks: &row_blocks,
        },
        &[],
        args.nce_objective.to_ge(),
        dev,
    )?;
    out.loss.per_feature_level.sum_all()
}

/// One `(feature, batch)` draw: on units (pseudobulks), or on cell pairs
/// under cage-cell.
enum Draw {
    Units(UnitChainBatch),
    Pairs(EdgeChainBatch),
}

impl Draw {
    fn sample<R: rand::Rng>(
        sampler: &FeatureGatedChainSampler<'_>,
        on_pairs: bool,
        g: usize,
        b: usize,
        rng: &mut R,
    ) -> Option<Self> {
        if on_pairs {
            sampler.sample_edges(g, b, rng).map(Draw::Pairs)
        } else {
            sampler.sample(g, b, rng).map(|(cb, _)| Draw::Units(cb))
        }
    }
}

/// A chunk's loss, and under cage-cell the pair rows it read.
struct ChunkLoss {
    loss: CageLossOut,
    pairs: Option<PairStep>,
}

/// The cell pairs one cage-cell step reads: their ids (ascending), the
/// leaf their rows were copied into, and the rows of the module term's
/// draws.
struct PairStep {
    ids: Vec<u32>,
    leaf: Var,
    module_rows: Vec<u32>,
}

/// A chunk's draws (all of one kind), each one's feature, and the block
/// (batch) its negatives came from.
struct Chunk<'a> {
    draws: Vec<Draw>,
    feature_ids: &'a [u32],
    row_blocks: &'a [u32],
}

/// The NCE of one chunk; `module_picks` are the pairs the module term will
/// read, gathered into the same leaf.
fn chunk_loss(
    model: &JointEmbedModel,
    pairs: Option<(&RowAdam, &[(u32, u32)])>,
    chunk: Chunk<'_>,
    module_picks: &[u32],
    objective: NceObjective,
    dev: &Device,
) -> legume_numeric::candle::candle_core::Result<ChunkLoss> {
    let Chunk {
        draws,
        feature_ids,
        row_blocks,
    } = chunk;
    let Some((table, edges)) = pairs else {
        let cbs = draws
            .into_iter()
            .map(|d| match d {
                Draw::Units(cb) => cb,
                Draw::Pairs(_) => unreachable!("a pseudobulk fit draws units"),
            })
            .collect();
        return Ok(ChunkLoss {
            loss: cage_nce_loss_per_feature_level(
                model,
                cbs,
                feature_ids,
                row_blocks,
                objective,
                dev,
            )?,
            pairs: None,
        });
    };
    let mut ebs: Vec<EdgeChainBatch> = draws
        .into_iter()
        .map(|d| match d {
            Draw::Pairs(eb) => eb,
            Draw::Units(_) => unreachable!("cage-cell draws pairs"),
        })
        .collect();
    let (ids, module_rows) = localize_edges(&mut ebs, module_picks);
    let leaf = Var::from_tensor(&table.rows(&ids)?)?;
    let (left, right): (Vec<u32>, Vec<u32>) = ids.iter().map(|&e| edges[e as usize]).unzip();
    let rows = EdgeRows {
        z: leaf.as_tensor(),
        left: &left,
        right: &right,
    };
    let loss = cage_cell_edge_nce_loss(
        model,
        EdgeChunk {
            rows,
            batches: ebs,
            feature_ids,
            row_blocks,
        },
        objective,
        dev,
    )?;
    Ok(ChunkLoss {
        loss,
        pairs: Some(PairStep {
            ids,
            leaf,
            module_rows,
        }),
    })
}

/// The global-norm clip and step with the step's pair rows inside the
/// norm: `leaf` is outside every optimiser, so its gradient is clipped with
/// the rest (legume's `clip_grad_global_norm`), taken out of the store and
/// returned for the row optimiser. `None`: the norm was not finite and
/// nothing stepped.
fn clip_and_step_with_rows(
    adams: &mut [AdamW],
    mut grads: GradStore,
    max_norm: f64,
    leaf: &Var,
) -> anyhow::Result<Option<Tensor>> {
    let norm = clip_grad_global_norm(&mut grads, max_norm)?;
    if !norm.is_finite() {
        return Ok(None);
    }
    let row_grad = match grads.remove(leaf.as_tensor()) {
        Some(g) => g,
        None => leaf.as_tensor().zeros_like()?,
    };
    // clipped above; this only steps
    clip_and_step_dense_all(adams, grads, 0.0)?;
    Ok(Some(row_grad))
}

pub(super) fn resolve_feature_batch_size(inputs: ProbeInputs<'_>) -> usize {
    let ProbeInputs {
        args,
        active_features,
        dev,
        ..
    } = inputs;
    let c = &args.common;
    // Chunk ("SGD minibatch over features") resolution. An explicit
    // --feature-batch-size always wins. Otherwise CPU keeps the classic
    // 2048, and CUDA probes: one forward per candidate size through the
    // REAL sampler and loss, measuring what the retained graph costs,
    // growing while the next doubling fits --gpu-mem-fraction of free
    // memory (half held back for backward). The probe never calls
    // backward, so no parameter moves before epoch 0.
    match args.feature_batch_size {
        Some(explicit) => explicit,
        None => {
            let mut probe_rng = SmallRng::seed_from_u64(c.seed ^ 0x9e37_79b9);
            legume_numeric::candle::device::auto_chunk_size(
                dev,
                FEATURE_BATCH_DEFAULT,
                16.min(active_features.len().max(1)),
                args.gpu_mem_fraction,
                |n_probe_features| probe_forward(&inputs, n_probe_features, &mut probe_rng),
            )
            .unwrap_or(FEATURE_BATCH_DEFAULT)
        }
    }
}

/// What sizing a chunk needs.
pub(super) struct ProbeInputs<'a> {
    pub args: &'a CellActivityGraphEmbeddingArgs,
    /// Every feature with an active batch, before `--feature-inclusion`
    /// drops those with π = 0, so the probe draws the same either way.
    pub active_features: &'a [usize],
    pub n_exp_batches: usize,
    pub sampler: &'a FeatureGatedChainSampler<'a>,
    pub model: &'a JointEmbedModel,
    pub pairs: Option<PairRows<'a>>,
    pub dev: &'a Device,
}

/// cage-cell's pair table, read by the probe.
#[derive(Clone, Copy)]
pub(super) struct PairRows<'a> {
    pub z: &'a RowAdam,
    /// Each row's two cells.
    pub edges: &'a [(u32, u32)],
}

/// cage-cell's pair table, trained by the loop.
pub(super) struct PairTable<'a> {
    pub z: &'a mut RowAdam,
    /// Each row's two cells.
    pub edges: &'a [(u32, u32)],
}

/// The units whose counts the exact unit-module term pools.
pub(super) enum ModuleUnits<'a> {
    /// cage: the finest pseudobulks' profiles on the device as
    /// `[n_pb × G]`, so a step is one row gather.
    Pseudobulks(Tensor),
    /// cage-cell: the cells, whose sparse columns a step reads as it draws
    /// them (never a features × cells table); a drawn pair pools its two.
    Cells(CellCounts<'a>),
}

/// What the training loop runs on.
pub(super) struct TrainInputs<'a> {
    pub args: &'a CellActivityGraphEmbeddingArgs,
    pub model: &'a mut JointEmbedModel,
    pub frozen_feature: Option<&'a pretrained::FrozenGene>,
    pub adams: &'a mut [AdamW],
    pub lora_plus: Option<&'a LoraPlus<'a>>,
    pub sampler: &'a FeatureGatedChainSampler<'a>,
    pub trainable_features: &'a [usize],
    /// Features an epoch visits: the trainable ones, or Σπ under
    /// `--feature-inclusion`, capped by `--features-per-epoch`.
    pub features_visited: usize,
    pub feature_batch_size: usize,
    pub feature_fisher_weights: &'a [f32],
    /// Each feature's probability of being visited in an epoch
    /// (`--feature-inclusion`); `None` visits every one.
    pub inclusion: Option<&'a [f32]>,
    pub module_cfg: Option<&'a FeatureModuleConfig>,
    pub module_units: Option<ModuleUnits<'a>>,
    /// cage-cell: the pair table the loop trains in place of the unit table.
    pub pairs: Option<PairTable<'a>>,
    pub n_exp_batches: usize,
    pub n_pb: usize,
    pub n_features: usize,
    pub n_chain_levels: usize,
    pub dev: &'a Device,
}

/// cage-cell's training length: `per_pair` updates of each of `pairs`
/// rows on average, the updates of a step being the distinct pairs it
/// touched (a row moves at most once a step).
#[derive(Debug)]
pub(super) struct PairBudget {
    target: u64,
    pairs: usize,
    touched: u64,
    steps: usize,
}

impl PairBudget {
    pub(super) fn new(per_pair: usize, pairs: usize) -> Self {
        Self {
            target: per_pair as u64 * pairs as u64,
            pairs,
            touched: 0,
            steps: 0,
        }
    }

    /// One step that moved `touched` distinct pairs.
    pub(super) fn record(&mut self, touched: usize) {
        self.touched += touched as u64;
        self.steps += 1;
    }

    pub(super) fn reached(&self) -> bool {
        self.touched >= self.target
    }

    pub(super) fn mean_touched(&self) -> f64 {
        self.touched as f64 / self.steps.max(1) as f64
    }

    pub(super) fn mean_updates(&self) -> f64 {
        self.touched as f64 / self.pairs.max(1) as f64
    }

    /// The steps the budget takes at the pairs a step has touched so far.
    pub(super) fn steps_needed(&self) -> usize {
        (self.target as f64 / self.mean_touched().max(1.0)).ceil() as usize
    }
}

/// An epoch's mean loss and per-level losses from its sums over the
/// `chunks` that stepped, `features` features in all. The per-level sum is
/// of chunk means, so it averages per chunk. The loss sums every feature's
/// rows, so it averages per chunk too, except under `--feature-inclusion`
/// (`full_chunk` the feature batch size), where an epoch's feature count
/// is a random draw: there it is per feature, scaled to a full chunk, so
/// the loss the convergence test reads does not move with the draw.
pub(super) fn epoch_means(
    loss_sum: f64,
    per_level_sum: &[f32],
    chunks: usize,
    features: usize,
    full_chunk: Option<usize>,
) -> (f64, Vec<f64>) {
    let chunks = chunks.max(1) as f64;
    let loss = match full_chunk {
        Some(b) => loss_sum * b as f64 / features.max(1) as f64,
        None => loss_sum / chunks,
    };
    let per_level = per_level_sum
        .iter()
        .map(|&v| f64::from(v) / chunks)
        .collect();
    (loss, per_level)
}

/// Runs the epochs; the score trace is what comes back.
pub(super) fn train(inputs: TrainInputs<'_>) -> anyhow::Result<Vec<ScoreEntry>> {
    let TrainInputs {
        args,
        model,
        frozen_feature,
        adams,
        lora_plus,
        sampler,
        trainable_features,
        features_visited,
        feature_batch_size,
        feature_fisher_weights,
        inclusion,
        module_cfg,
        module_units,
        mut pairs,
        n_exp_batches,
        n_pb,
        n_features,
        n_chain_levels,
        dev,
    } = inputs;
    // The units the module term draws: the pseudobulks, or the pairs.
    let n_units = pairs.as_ref().map_or(n_pb, |p| p.z.n_rows());
    let on_pairs = pairs.is_some();
    let c = &args.common;
    let mut score_trace: Vec<ScoreEntry> = Vec::new();
    let mut rng_master = SmallRng::seed_from_u64(c.seed);
    // Wall-clock split of the SGD loop, to rank optimization work by evidence
    // rather than guess. Accumulated across all epochs; logged once at the end.
    let mut timers = PhaseTimers::default();
    // Steps the optimizer refused because the global gradient norm was not
    // finite. Reported at the end: a high count means the fit did not train.
    let mut skipped_steps: usize = 0;

    // One bar across the WHOLE training phase, not one per epoch: the useful
    // question while waiting is "how long until training is done", and a bar
    // that restarts five times cannot answer it. The total is the real step
    // count — chunks per epoch is `ceil(features_visited / --feature-batch-size)` —
    // so the ETA is derived from observed step rate rather than guessed.
    // `--features-per-epoch` shortens every epoch equally, so this stays exact.
    // Early exit (convergence or Ctrl-C) simply finishes the bar short.
    // Under --feature-inclusion an epoch visits each feature with its
    // probability, so its length is that expectation (for the bar and the
    // per-step ridge share only).
    let steps_per_epoch = features_visited.div_ceil(feature_batch_size);
    // The LoRA ridge is a per-epoch weight; every step takes its share.
    if let (Some(l), Some(lp)) = (model.lora.as_mut(), lora_plus) {
        l.ridge_step = f64::from(lp.ridge) / steps_per_epoch.max(1) as f64;
    }
    let total_steps = args.epochs * steps_per_epoch.max(1);
    // cage-cell's length: updates per pair, counted as the steps go.
    let mut budget = pairs
        .as_ref()
        .map(|p| PairBudget::new(args.pair_updates, p.z.n_rows()));
    let mut steps_taken = 0usize;
    // The epochs the run expects to take: --epochs, or under cage-cell the
    // pair budget's, known after the first epoch (and capped by --epochs).
    let mut planned_epochs = args.epochs;
    let model = &*model;
    let train_bar = new_progress_bar(total_steps as u64).with_message("training steps");

    // First ^C = graceful stop after current chunk, finalize outputs;
    // second ^C = hard abort. See graph_embedding_util::stop.
    let stop = setup_stop_handler();

    // Module warm-up: hold the starting membership for the first epochs, release
    // it afterwards (geu's schedule).
    // Under cage-cell the schedule is set again after the first epoch,
    // against the epochs the pair budget takes.
    let mut module_run: Option<(usize, ModuleUnits)> = match (module_cfg, module_units) {
        (Some(gm), Some(units)) => {
            let wu = gm.warmup_epochs_for(args.epochs);
            if let Some(m) = &model.modules {
                m.set_frozen(wu > 0);
            }
            if budget.is_none() {
                info!(
                    "feature modules: membership held for {wu} of {} epochs, then trained",
                    args.epochs
                );
            }
            Some((wu, units))
        }
        _ => None,
    };

    'epochs: for epoch in 0..args.epochs {
        if let (Some(m), Some((wu, _))) = (&model.modules, &module_run) {
            if epoch == *wu && m.is_frozen() {
                m.set_frozen(false);
                info!("epoch {epoch}: module membership released — π now trains");
            }
        }
        // Soft inclusion: feature g joins this epoch with probability π_g,
        // so the expected objective is Σ π_g L_g. Its draws and order are
        // per feature, so they take nothing from the master stream.
        let mut perm: Vec<usize> = match inclusion {
            Some(pi) => crate::cell_activity_graph_embedding::feature_inclusion::sample_included(
                trainable_features,
                pi,
                c.seed,
                epoch,
            ),
            None => {
                let mut perm = trainable_features.to_vec();
                perm.shuffle(&mut rng_master);
                perm
            }
        };
        // Optional cost lever: visit a random subset of the feature axis this
        // epoch. Stochastic coverage, NOT axis selection — every feature stays
        // in the model, keeps its gate, and appears in the output tables; it
        // just waits for a later epoch.
        if args.features_per_epoch > 0 && args.features_per_epoch < perm.len() {
            perm.truncate(args.features_per_epoch);
        }

        let mut skip_count: usize = 0;
        let mut sample_count: usize = 0;

        // Tensor accumulators — avoid per-step CUDA → CPU sync. The
        // `.detach()` strips autograd so we can keep adding into them
        // without retaining the backward graph across chunks.
        let mut epoch_loss_acc: Option<Tensor> = None;
        let mut per_level_acc: Option<Tensor> = None;
        let mut pair_acc: Option<Tensor> = None;
        let mut chunk_count: usize = 0;
        // Features in the chunks that reached a step (not a chunk that drew
        // nothing, nor one after a stop).
        let mut processed: usize = 0;
        debug_assert!(inclusion.is_some() || perm.len() == features_visited);
        for chunk in perm.chunks(feature_batch_size) {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            // (a) Parallel sampling — pure CPU, no candle.
            let t_sample = std::time::Instant::now();
            let mini: Vec<(usize, u32, Draw)> = chunk
                .par_iter()
                .flat_map_iter(|&g| {
                    let seed = c
                        .seed
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(g as u64)
                        .wrapping_add((epoch as u64).wrapping_mul(1_000_003));
                    let mut rng = SmallRng::seed_from_u64(seed);
                    (0..n_exp_batches)
                        .filter_map(move |b| {
                            Draw::sample(sampler, on_pairs, g, b, &mut rng)
                                .map(|d| (g, b as u32, d))
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            // Attribute sampling BEFORE the early-out: a chunk whose features all
            // failed to sample still paid for the rayon draw, and dropping it
            // under-reports the sampler exactly where it did the most futile
            // work. Advance the bar too, or its total (which assumes every
            // chunk steps) leaves it short and the ETA running long.
            timers.precompute += t_sample.elapsed();
            if mini.is_empty() {
                skip_count += chunk.len() * n_exp_batches;
                train_bar.inc(1);
                continue;
            }

            let (feature_ids, (row_blocks, cb_batches)): (Vec<usize>, (Vec<u32>, Vec<_>)) =
                mini.into_iter().map(|(g, b, cb)| (g, (b, cb))).unzip();
            sample_count += feature_ids.len();
            let t_fwd = std::time::Instant::now();

            // Feature identity enters the score function via the gated
            // loss. The chunk's feature ids parallel-replicate to align
            // with the [G*B] cell-side gathers.
            let feature_ids_u32: Vec<u32> = feature_ids.iter().map(|&g| g as u32).collect();

            // The module term's draws come first: under cage-cell its pairs
            // are read through the same leaf as the chunk's.
            let module_picks: Option<Vec<u32>> = match (&model.modules, module_cfg, &module_run) {
                (Some(_), Some(gm), Some(_)) => {
                    let u = gm.units_per_step.min(n_units);
                    (u > 0 && gm.lambda_module > 0.0).then(|| {
                        (0..u)
                            .map(|_| rng_master.random_range(0..n_units) as u32)
                            .collect()
                    })
                }
                _ => None,
            };

            // (b) ONE forward / backward over the whole chunk.
            let ChunkLoss {
                loss:
                    CageLossOut {
                        per_feature_level: per_level_gl,
                        mean_abs_pair,
                    },
                pairs: pair_step,
            } = chunk_loss(
                model,
                pairs.as_ref().map(|p| (&*p.z, p.edges)),
                Chunk {
                    draws: cb_batches,
                    feature_ids: &feature_ids_u32,
                    row_blocks: &row_blocks,
                },
                module_picks.as_deref().unwrap_or(&[]),
                args.nce_objective.to_ge(),
                dev,
            )?; // [G, L]
                // NB-Fisher per-feature precision: scale each feature's row of the
                // per-level loss by w_g ∈ (0,1] before summing, so high-mean /
                // high-dispersion housekeeping features contribute less gradient
                // (the loss-side analog of bge's count·fisher positive draw and
                // lc's `apply_feature_weights` on the feature basis). Coverage stays
                // uniform — every feature is still visited once per epoch.
            let w_chunk: Vec<f32> = feature_ids
                .iter()
                .map(|&g| feature_fisher_weights[g])
                .collect();
            let w_g1 = Tensor::from_vec(w_chunk, (feature_ids.len(), 1), dev)?;
            let per_level_gl = per_level_gl.broadcast_mul(&w_g1)?;
            let loss = per_level_gl.sum_all()?;
            let mut total = loss.clone();

            if args.embedding_l2 > 0.0 {
                // geu's ridge, not a local copy: the reduction is the whole
                // content of this penalty and it was wrong here in the same way.
                // On pairs, over the step's rows: the mean over a uniform
                // share of the table has the whole table's expected gradient.
                let table = match &pair_step {
                    Some(step) => step.leaf.as_tensor(),
                    None => &model.e_cell,
                };
                total = (total + embedding_ridge(table, args.embedding_l2 as f64)?)?;
            }
            // Which feature-side shrinkage applies is a model property
            // (`feature_ridge`: the free table or the adapter's residual at
            // the table ridge, the anchored model's own residual ridge, or
            // nothing). Pinned rows are the one cage-local exception: the
            // table ridge would only push gradient at rows the restore below
            // reverts anyway, so it is off under freeze and lora.
            let table_lam = if frozen_feature.is_some() {
                0.0
            } else {
                args.embedding_l2 as f64
            };
            if let Some(ridge) = model.feature_ridge(table_lam)? {
                total = (total + ridge)?;
            }
            // Exact unit–module term + membership priors, once per optimizer
            // step, as geu's `module_step_loss` makes it: draw `units_per_step`
            // trained units (pseudobulks, or cells under cage-cell) uniformly,
            // pool their count rows through the (dropout-masked) membership,
            // and score every module against the trained unit table. The
            // pooling carries no gradient (geu detaches the masked membership
            // and the pooled target), so cells pool their sparse counts.
            if let (Some(m), Some(gm), Some((_, units)), Some(picks)) = (
                &model.modules,
                module_cfg,
                module_run.as_ref(),
                &module_picks,
            ) {
                let pi = m.membership()?;
                let pi_masked = if gm.feature_dropout > 0.0 {
                    let keep =
                        draw_gene_keep_mask(n_features, gm.feature_dropout, &mut rng_master, dev)?;
                    masked_membership(&pi, &keep)?
                } else {
                    pi.detach()
                };
                let u = picks.len();
                let e_units = match &pair_step {
                    Some(step) => gather_rows(
                        step.leaf.as_tensor(),
                        &Tensor::from_slice(&step.module_rows, u, dev)?,
                    )?,
                    None => gather_rows(&model.e_cell, &Tensor::from_slice(picks, u, dev)?)?,
                };
                let x_cm = match (units, &pairs) {
                    (ModuleUnits::Pseudobulks(profiles), _) => {
                        gather_rows(profiles, &Tensor::from_slice(picks, u, dev)?)?
                            .matmul(&pi_masked)?
                    }
                    // a pair pools its two cells' counts
                    (ModuleUnits::Cells(cells), Some(p)) => {
                        // each distinct cell read once, then summed per pair
                        let mut ends: Vec<u32> = picks
                            .iter()
                            .flat_map(|&e| {
                                let (a, b) = p.edges[e as usize];
                                [a, b]
                            })
                            .collect();
                        let slot =
                            |c: &u32, ends: &[u32]| ends.binary_search(c).expect("listed") as u32;
                        let pairs_ends: Vec<u32> = ends.clone();
                        ends.sort_unstable();
                        ends.dedup();
                        let (left, right): (Vec<u32>, Vec<u32>) = pairs_ends
                            .chunks(2)
                            .map(|ab| (slot(&ab[0], &ends), slot(&ab[1], &ends)))
                            .unzip();
                        let pooled = cells.pool(&ends, &pi_masked)?;
                        let at = |i: Vec<u32>| Tensor::from_vec(i, u, dev);
                        (gather_rows(&pooled, &at(left)?)? + gather_rows(&pooled, &at(right)?)?)?
                    }
                    (ModuleUnits::Cells(cells), None) => cells.pool(picks, &pi_masked)?,
                };
                let module_loss = module_softmax_loss(&e_units, &m.mu, &m.b_module, &x_cm)?
                    .affine(f64::from(gm.lambda_module), 0.0)?;
                total = (total + module_loss)?;
                if let Some(prior) = module_priors(m, &pi, gm.lambda_balance)? {
                    total = (total + prior)?;
                }
            }
            // Global-norm clip before the step. The NCE loss spikes when a
            // chunk draws a feature whose active edges are nearly all positives,
            // and an unbounded step there inflates the embedding norms every
            // later epoch has to work against. A step whose global norm is
            // non-finite is skipped rather than laundered into the parameters.
            timers.decoder_fwd += t_fwd.elapsed();
            let t_bwd = std::time::Instant::now();
            // Backward and the step run separately so the phase timers can
            // attribute them apart.
            let grads = total.backward()?;
            timers.backward += t_bwd.elapsed();
            let t_opt = std::time::Instant::now();
            // A step is SKIPPED when the global gradient norm is not finite.
            // Count those: a run where every step is skipped moves no parameter
            // at all, and without this the bar would still reach 100% and the
            // phase timings would still look normal — the instrumentation would
            // make the failure less visible rather than more.
            let stepped = match (&pair_step, pairs.as_mut()) {
                (Some(step), Some(table)) => {
                    match clip_and_step_with_rows(adams, grads, f64::from(GRAD_CLIP), &step.leaf)? {
                        Some(row_grad) => {
                            table
                                .z
                                .step(&step.ids, Some(step.leaf.as_tensor()), &row_grad)?;
                            true
                        }
                        None => false,
                    }
                }
                _ => clip_and_step_dense_all(adams, grads, f64::from(GRAD_CLIP))?,
            };
            if !stepped {
                skipped_steps += 1;
            }
            steps_taken += 1;
            if let (Some(b), Some(step), true) = (budget.as_mut(), &pair_step, stepped) {
                b.record(step.ids.len());
            }
            // Freezing is a post-step restore, not a gradient mask: AdamW's
            // moment state moves a row even at zero gradient, so the frozen
            // rows are put back from the fixed copy after every step.
            if stepped {
                if let Some(frozen) = frozen_feature {
                    frozen.restore()?;
                }
            }
            timers.optimize += t_opt.elapsed();
            train_bar.inc(1);

            // Diagnostics (no host sync) — accumulate detached tensors.
            // Per-feature mean of per_level → [L] for this chunk; running sum.
            let per_level_chunk_mean = per_level_gl.mean(0)?.detach();
            let loss_chunk = loss.detach();
            pair_acc = Some(match pair_acc {
                Some(prev) => (prev + &mean_abs_pair)?,
                None => mean_abs_pair,
            });
            epoch_loss_acc = Some(match epoch_loss_acc {
                Some(prev) => (prev + loss_chunk)?,
                None => loss_chunk,
            });
            per_level_acc = Some(match per_level_acc {
                Some(prev) => (prev + per_level_chunk_mean)?,
                None => per_level_chunk_mean,
            });
            chunk_count += 1;
            processed += chunk.len();
            if budget.as_ref().is_some_and(PairBudget::reached) {
                break;
            }
        }
        if epoch == 0 {
            if let Some(b) = &budget {
                let steps = b.steps_needed();
                info!(
                    "cage-cell: a step touches {:.0} of {} pairs on average, so {} pair \
                     updates each take about {} steps ({} epochs of {}; --epochs caps at {})",
                    b.mean_touched(),
                    b.pairs,
                    args.pair_updates,
                    steps,
                    steps.div_ceil(steps_per_epoch.max(1)),
                    steps_per_epoch.max(1),
                    args.epochs
                );
                train_bar.set_length(steps.min(total_steps) as u64);
                planned_epochs = steps.div_ceil(steps_per_epoch.max(1)).min(args.epochs);
                if let (Some(gm), Some((wu, _))) = (module_cfg, module_run.as_mut()) {
                    *wu = gm.warmup_epochs_for(planned_epochs);
                    info!(
                        "feature modules: membership held for {wu} of {planned_epochs} \
                         expected epochs, then trained",
                        wu = *wu
                    );
                }
            }
        }

        // ONE host sync per epoch — pull accumulated tensors only now.
        let (mean_loss, mean_per_level) = if chunk_count > 0 {
            let loss_sum: f32 = epoch_loss_acc
                .as_ref()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap_or(f32::NAN);
            let pl_sum: Vec<f32> = per_level_acc
                .as_ref()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap_or_default();
            epoch_means(
                f64::from(loss_sum),
                &pl_sum,
                chunk_count,
                processed,
                inclusion.map(|_| feature_batch_size),
            )
        } else {
            (f64::NAN, vec![0.0; n_chain_levels])
        };
        info!(
            "epoch {} of {planned_epochs}{}: mean loss = {:.4e} (per-level: {:?}), samples = {}, skipped pairs = {}",
            epoch,
            budget.as_ref().map_or(String::new(), |b| format!(
                " ({:.1} of {} updates per pair)",
                b.mean_updates(),
                args.pair_updates
            )),
            mean_loss,
            mean_per_level,
            sample_count,
            skip_count
        );
        if let Some(m) = &model.modules {
            let pi_host: Vec<f32> = m.membership()?.detach().flatten_all()?.to_vec1()?;
            // cage draws its negatives over pseudobulks, so no within-module fallback.
            log_membership_diagnostics(m, &pi_host, epoch, planned_epochs, 0)?;
        }

        // Pair-term magnitude: the collapse detector. If this decays toward
        // zero while the loss still falls, the ungated cell biases have taken
        // over the objective and the feature direction is doing nothing.
        if chunk_count > 0 {
            let pair_mean = pair_acc
                .as_ref()
                .and_then(|t| t.to_scalar::<f32>().ok())
                .map_or(f64::NAN, |v| v as f64 / chunk_count as f64);
            info!("epoch {}: mean |pair| = {:.4e}", epoch, pair_mean);
            // The loss is the bar's one `key=value`: the score `pinto run`
            // draws, a NaN loss included.
            let updates = budget.as_ref().map_or(String::new(), |b| {
                format!(
                    " updates/pair {:.0}/{}",
                    b.mean_updates(),
                    args.pair_updates
                )
            });
            train_bar.set_message(format!(
                "training epoch {}/{planned_epochs}{updates} loss={:.4e} |pair| {:.2e}",
                epoch + 1,
                mean_loss,
                pair_mean
            ));
        }

        // Push one summary row per epoch. `level = epoch`, `sweep = 0`.
        // `total_mass = #samples`, `mutual_information` = mean of per-level loss
        // (informally summarizing chain effectiveness).
        let mi = mean_per_level.iter().sum::<f64>() / (mean_per_level.len().max(1) as f64);
        score_trace.push(ScoreEntry {
            level: epoch as i32,
            sweep: 0,
            score: mean_loss,
            n_edges: sample_count,
            total_mass: sample_count as f64,
            mutual_information: mi,
        });

        // Convergence check: if the last `convergence_window` epochs'
        // mean losses are within [`CONVERGENCE_TOL`] relative range,
        // exit. `convergence_window == 0` disables.
        if args.convergence_window > 0 && score_trace.len() >= args.convergence_window {
            let window = &score_trace[score_trace.len() - args.convergence_window..];
            let losses: Vec<f64> = window.iter().map(|e| e.score).collect();
            let mean = losses.iter().sum::<f64>() / (losses.len() as f64);
            let (lo, hi) = losses
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            let rel = if mean.abs() > 1e-30 {
                (hi - lo) / mean.abs()
            } else {
                f64::INFINITY
            };
            if rel.is_finite() && rel < f64::from(CONVERGENCE_TOL) {
                info!(
                    "converged at epoch {}: last {} losses within rel-range {:.4} < tol {:.4}",
                    epoch, args.convergence_window, rel, CONVERGENCE_TOL
                );
                break;
            }
        }

        if let Some(b) = budget.as_ref().filter(|b| b.reached()) {
            info!(
                "cage-cell: {:.1} updates per pair reached at epoch {epoch} ({} steps)",
                b.mean_updates(),
                b.steps
            );
            break 'epochs;
        }

        if stop.load(Ordering::SeqCst) {
            info!(
                "Stopping early at epoch {}/{} — finalizing outputs",
                epoch + 1,
                args.epochs
            );
            break 'epochs;
        }
    }

    train_bar.finish_and_clear();
    match &budget {
        Some(b) if !b.reached() => warn!(
            "cage-cell: stopped at {:.1} of {} updates per pair after {} steps \
             (--epochs or --convergence-window ended it first)",
            b.mean_updates(),
            args.pair_updates,
            steps_taken
        ),
        _ => info!("training: {steps_taken} optimizer steps"),
    }
    if skipped_steps > 0 {
        warn!(
            "{}/{} optimizer steps were SKIPPED for non-finite gradients — those \
             steps moved no parameters. Lower --lr if this is a \
             large fraction.",
            skipped_steps, steps_taken
        );
    }

    // Where the SGD loop's PER-STEP work went: `precompute` is the rayon
    // feature/edge sampling, `decoder_fwd` the loss forward, then backward and the
    // AdamW step. Reported so optimization work is ranked on measurement.
    //
    // These four do NOT sum to the training wall clock. Per-EPOCH work sits
    // outside them: the `--selection-refresh-epochs` Gibbs refresh (which on the
    // default sampled arm fires every 5th epoch and copies `e_cell` to the host),
    // the per-epoch diagnostic syncs, and the permutation shuffle. Read the
    // percentages as a split of per-step cost, not of the whole phase.
    timers.log_summary();
    Ok(score_trace)
}

#[cfg(test)]
mod tests {
    use super::{epoch_means, PairBudget};

    /// The budget is updates per pair: steps touching 30 of 100 pairs reach
    /// 3 updates per pair after 10 of them, and say so after the first.
    #[test]
    fn the_pair_budget_counts_updates_per_pair() {
        let mut b = PairBudget::new(3, 100);
        b.record(30);
        assert_eq!(b.steps_needed(), 10);
        for _ in 1..9 {
            b.record(30);
        }
        assert!(!b.reached());
        b.record(30);
        assert!(b.reached());
        assert_eq!(b.mean_updates(), 3.0);
    }

    /// The per-level losses average per chunk that stepped, with or
    /// without inclusion; under inclusion the loss is per feature processed,
    /// as a full chunk, however few features the epoch drew.
    #[test]
    fn epoch_means_divide_by_what_was_processed() {
        // one chunk of one feature, batch size 4
        let (loss, per_level) = epoch_means(3.0, &[2.0, 6.0], 1, 1, Some(4));
        assert_eq!(per_level, vec![2.0, 6.0]);
        assert_eq!(loss, 12.0);
        // without inclusion both average per chunk
        let (loss, per_level) = epoch_means(3.0, &[2.0, 6.0], 2, 7, None);
        assert_eq!(per_level, vec![1.0, 3.0]);
        assert_eq!(loss, 1.5);
    }
}
