//! The training loop, and the probe that sizes the pieces a step is
//! computed in.
//!
//! A step covers `--feature-batch-size` features (2048 by default); its
//! draws are scored in pieces, each backwarded on its own and the gradients
//! summed, so a step's update is the same however many pieces it takes and
//! only one piece's graph is alive at a time. The memory probe sizes the
//! piece.

use crate::cell_activity_graph_embedding::args::{
    CellActivityGraphEmbeddingArgs, NCE_OBJECTIVE, PER_FEATURE_BATCH,
};
use crate::cell_activity_graph_embedding::feature_chain_sampler::FeatureGatedChainSampler;
use crate::cell_activity_graph_embedding::pretrained;
use crate::util::common::*;
use crate::util::score_trace::ScoreEntry;

use super::units::{Drawn, Units};
use graph_embedding_util::loss::{
    draw_gene_keep_mask, embedding_ridge, log_membership_diagnostics, masked_membership,
    module_priors, module_softmax_loss,
};
use graph_embedding_util::model::JointEmbedModel;
use graph_embedding_util::stop::setup_stop_handler;
use graph_embedding_util::FeatureModuleConfig;
use legume_numeric::candle::candle_core::backprop::GradStore;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::candle::candle_nn::AdamW;
use legume_numeric::candle::lora::LoraPlus;
use legume_numeric::candle::vae::PhaseTimers;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::RngExt;
use rand::SeedableRng;
use std::sync::atomic::Ordering;

/// Classic SGD chunk over features: a step's features unless
/// `--feature-batch-size` sets it. The args long_help describes this value;
/// keep the two in step.
const FEATURE_BATCH_DEFAULT: usize = 2048;
/// The probe cycles this many times the requested count of features before
/// giving up on filling a pilot minibatch: `sample` can miss per
/// (feature, batch), so some slack is needed, and the bound keeps a sparse
/// sampler from turning the probe into a long scan.
const PROBE_OVERSAMPLE: usize = 4;

/// What the loop runs on, besides the units.
pub(super) struct TrainSetup<'a> {
    pub args: &'a CellActivityGraphEmbeddingArgs,
    pub model: &'a mut JointEmbedModel,
    pub frozen_feature: Option<&'a pretrained::FrozenGene>,
    pub adams: &'a mut [AdamW],
    pub lora_plus: Option<&'a LoraPlus<'a>>,
    pub sampler: &'a FeatureGatedChainSampler<'a>,
    /// Every feature with an active batch, before `--feature-inclusion`
    /// drops those with π = 0, so the probe draws the same either way.
    pub active_features: &'a [usize],
    pub trainable_features: &'a [usize],
    /// Features an epoch visits: the trainable ones, or Σπ under
    /// `--feature-inclusion`, capped by `--features-per-epoch`.
    pub features_visited: usize,
    pub feature_fisher_weights: &'a [f32],
    /// Each feature's probability of being visited in an epoch
    /// (`--feature-inclusion`); `None` visits every one.
    pub inclusion: Option<&'a [f32]>,
    pub module_cfg: Option<&'a FeatureModuleConfig>,
    /// Positive draws per (feature, batch).
    pub positives_per_draw: usize,
    pub n_exp_batches: usize,
    pub n_features: usize,
    pub n_chain_levels: usize,
    pub dev: &'a Device,
}

/// One forward-only pilot piece of `n` FEATURES for the GPU memory probe,
/// each feature expanded to every batch label that samples, exactly as a
/// real piece expands. Samples through the REAL sampler, scores with the
/// REAL loss, returns the un-backwarded loss tensor. Never steps the
/// optimizer, so no parameter moves before epoch 0.
fn probe_forward<U: Units>(
    setup: &TrainSetup<'_>,
    units: &U,
    n_probe_features: usize,
    probe_rng: &mut SmallRng,
) -> legume_numeric::candle::candle_core::Result<Tensor> {
    let mut mini: Vec<(usize, u32, U::Draw)> = Vec::with_capacity(n_probe_features);
    let mut features_sampled = 0usize;
    'fill: for &g in setup
        .active_features
        .iter()
        .cycle()
        .take(PROBE_OVERSAMPLE * n_probe_features)
    {
        let before = mini.len();
        for b in 0..setup.n_exp_batches {
            if let Some(d) = U::sample(setup.sampler, g, b, probe_rng) {
                mini.push((g, b as u32, d));
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
    let (feature_ids, row_blocks, mut draws) = split_draws(mini);
    let feature_ids: Vec<u32> = feature_ids.into_iter().map(|g| g as u32).collect();
    let read = units.read(&mut draws, &[])?;
    let out = units.nce(
        setup.model,
        &read,
        Drawn {
            draws,
            feature_ids: &feature_ids,
            row_blocks: &row_blocks,
        },
        NCE_OBJECTIVE,
        setup.dev,
    )?;
    out.per_feature_level.sum_all()
}

/// A step's features, and the features one piece of it holds.
///
/// A step is `--feature-batch-size` features, or [`FEATURE_BATCH_DEFAULT`].
/// A piece holds at most as many positive draws as cage's default step
/// (that many features at [`PER_FEATURE_BATCH`] each), so a run that draws
/// more per feature splits its steps rather than growing them; on CUDA the
/// probe then shrinks the piece while it does not fit `--gpu-mem-fraction`
/// of the free memory (half held back for backward).
fn resolve_chunks<U: Units>(setup: &TrainSetup<'_>, units: &U) -> (usize, usize) {
    let args = setup.args;
    let step = args
        .feature_batch_size
        .unwrap_or(FEATURE_BATCH_DEFAULT)
        .max(1);
    let cap = (FEATURE_BATCH_DEFAULT * PER_FEATURE_BATCH / setup.positives_per_draw.max(1))
        .clamp(1, step);
    let mut probe_rng = SmallRng::seed_from_u64(args.common.seed ^ 0x9e37_79b9);
    let piece = legume_numeric::candle::device::auto_chunk_size(
        setup.dev,
        cap,
        16.min(setup.active_features.len().max(1)).min(cap),
        args.gpu_mem_fraction,
        |n| probe_forward(setup, units, n, &mut probe_rng),
    )
    .unwrap_or(cap);
    if piece < step {
        info!("each step of {step} features is computed in pieces of {piece}");
    }
    (step, piece)
}

/// Draws `(feature, batch, draw)` as their features, batches and draws.
fn split_draws<D>(mini: Vec<(usize, u32, D)>) -> (Vec<usize>, Vec<u32>, Vec<D>) {
    let mut features = Vec::with_capacity(mini.len());
    let mut blocks = Vec::with_capacity(mini.len());
    let mut draws = Vec::with_capacity(mini.len());
    for (g, b, d) in mini {
        features.push(g);
        blocks.push(b);
        draws.push(d);
    }
    (features, blocks, draws)
}

/// `grads` added into `acc`, variable by variable.
fn add_grads(acc: &mut Option<GradStore>, grads: GradStore) -> anyhow::Result<()> {
    let Some(acc) = acc else {
        *acc = Some(grads);
        return Ok(());
    };
    for &id in grads.get_ids() {
        let g = grads.get_id(id).expect("listed");
        let sum = match acc.get_id(id) {
            Some(prev) => (prev + g)?,
            None => g.clone(),
        };
        acc.insert_id(id, sum);
    }
    Ok(())
}

/// `x` added into the running sum `acc`.
fn accumulate(acc: &mut Option<Tensor>, x: Tensor) -> anyhow::Result<()> {
    *acc = Some(match acc.take() {
        Some(prev) => (prev + x)?,
        None => x,
    });
    Ok(())
}

/// An epoch's mean loss and per-level losses from its sums over the
/// `chunks` that stepped, `features` features in all. The per-level sum is
/// of chunk means, so it averages per chunk. The loss sums every feature's
/// rows, so it averages per chunk too, except under `--feature-inclusion`
/// (`full_chunk` the feature batch size), where an epoch's feature count
/// is a random draw: there it is per feature, scaled to a full chunk, so
/// the loss the score trace and the training bar show does not move with
/// the draw.
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

/// The epochs a run expects after `epochs_done`, at `progress` of its
/// planned length, never past the `cap`.
fn planned_epochs(epochs_done: usize, progress: f64, cap: usize) -> usize {
    if progress > 0.0 {
        ((epochs_done as f64 / progress).round() as usize).clamp(epochs_done.max(1), cap)
    } else {
        cap
    }
}

/// Sizes the step pieces, then runs the epochs; the score trace is what
/// comes back.
pub(super) fn train<U: Units>(
    setup: TrainSetup<'_>,
    mut units: U,
) -> anyhow::Result<Vec<ScoreEntry>> {
    let (feature_batch_size, piece) = resolve_chunks(&setup, &units);
    let TrainSetup {
        args,
        model,
        frozen_feature,
        adams,
        lora_plus,
        sampler,
        active_features: _,
        trainable_features,
        features_visited,
        feature_fisher_weights,
        inclusion,
        module_cfg,
        positives_per_draw: _,
        n_exp_batches,
        n_features,
        n_chain_levels,
        dev,
    } = setup;
    let n_units = units.n_units();
    let c = &args.common;
    let mut score_trace: Vec<ScoreEntry> = Vec::new();
    let mut rng_master = SmallRng::seed_from_u64(c.seed);
    // Wall-clock split of the SGD loop, to rank optimization work by evidence
    // rather than guess. Accumulated across all epochs; logged once at the end.
    let mut timers = PhaseTimers::default();
    // Steps the optimizer refused because the global gradient norm was not
    // finite. Reported at the end: a high count means the fit did not train.
    let mut skipped_steps: usize = 0;

    // Under --feature-inclusion an epoch visits each feature with its
    // probability, so its length is that expectation (for the schedule and
    // the per-step ridge share only).
    let steps_per_epoch = features_visited.div_ceil(feature_batch_size).max(1);
    // The LoRA ridge is a per-epoch weight; every step takes its share.
    if let (Some(l), Some(lp)) = (model.lora.as_mut(), lora_plus) {
        l.ridge_step = f64::from(lp.ridge) / steps_per_epoch as f64;
    }
    // The schedule is one measure, the share of the planned training done:
    // steps of `--epochs` worth for pseudobulks, pair updates of the budget
    // for pairs. The bar, the module warm-up and the expected epoch count
    // all read it.
    let planned_steps = args.epochs * steps_per_epoch;
    let mut steps_taken = 0usize;
    let model = &*model;
    // One bar across the whole training phase, in thousandths of it.
    let train_bar = new_progress_bar(1000).with_message("training");

    // First ^C = graceful stop after current chunk, finalize outputs;
    // second ^C = hard abort. See graph_embedding_util::stop.
    let stop = setup_stop_handler();

    // Module warm-up: hold the starting membership for the first share of
    // the schedule (geu's, as a share of --epochs), release it afterwards.
    let warmup = match (module_cfg, &model.modules) {
        (Some(gm), Some(m)) => {
            let share = gm.warmup_epochs_for(args.epochs) as f64 / args.epochs.max(1) as f64;
            m.set_frozen(share > 0.0);
            info!(
                "feature modules: membership held for the first {:.0}% of training, then trained",
                share * 100.0
            );
            Some(share)
        }
        _ => None,
    };

    'epochs: for epoch in 0..args.epochs {
        if let (Some(m), Some(share)) = (&model.modules, warmup) {
            if units.progress(steps_taken, planned_steps) >= share && m.is_frozen() {
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
            let mini: Vec<(usize, u32, U::Draw)> = chunk
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
                            U::sample(sampler, g, b, &mut rng).map(|d| (g, b as u32, d))
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            // Attribute sampling BEFORE the early-out: a chunk whose features all
            // failed to sample still paid for the rayon draw.
            timers.precompute += t_sample.elapsed();
            if mini.is_empty() {
                skip_count += chunk.len() * n_exp_batches;
                continue;
            }

            let (feature_ids, row_blocks, mut draws) = split_draws(mini);
            let n_draws = feature_ids.len();
            sample_count += n_draws;
            let t_fwd = std::time::Instant::now();
            let mut t_backward = std::time::Duration::ZERO;
            let feature_ids_u32: Vec<u32> = feature_ids.iter().map(|&g| g as u32).collect();

            // The module term's draws come first: under cage-cell its pairs
            // are read with the chunk's.
            let module_picks: Option<Vec<u32>> = match (&model.modules, module_cfg) {
                (Some(_), Some(gm)) => {
                    let u = gm.units_per_step.min(n_units);
                    (u > 0 && gm.lambda_module > 0.0).then(|| {
                        (0..u)
                            .map(|_| rng_master.random_range(0..n_units) as u32)
                            .collect()
                    })
                }
                _ => None,
            };
            let read = units.read(&mut draws, module_picks.as_deref().unwrap_or(&[]))?;

            // (b) The chunk's NCE in pieces, each backwarded on its own and
            // the gradients summed: one step's update, one piece's graph.
            // NB-Fisher per-feature precision scales each feature's row of the
            // per-level loss by w_g ∈ (0,1] before summing, so high-mean /
            // high-dispersion housekeeping features contribute less gradient.
            // Coverage stays uniform — every feature is still visited once per
            // epoch.
            let piece_draws = (piece * n_exp_batches).max(1);
            let mut grads: Option<GradStore> = None;
            let mut loss_sum: Option<Tensor> = None;
            let mut level_sum: Option<Tensor> = None;
            let mut pair_sum: Option<Tensor> = None;
            let mut draws = draws.into_iter();
            for start in (0..n_draws).step_by(piece_draws) {
                let end = (start + piece_draws).min(n_draws);
                let out = units.nce(
                    model,
                    &read,
                    Drawn {
                        draws: draws.by_ref().take(end - start).collect(),
                        feature_ids: &feature_ids_u32[start..end],
                        row_blocks: &row_blocks[start..end],
                    },
                    NCE_OBJECTIVE,
                    dev,
                )?; // [G, L]
                let w: Vec<f32> = feature_ids[start..end]
                    .iter()
                    .map(|&g| feature_fisher_weights[g])
                    .collect();
                let w = Tensor::from_vec(w, (end - start, 1), dev)?;
                let per_level = out.per_feature_level.broadcast_mul(&w)?;
                let loss = per_level.sum_all()?;
                let t_bwd = std::time::Instant::now();
                add_grads(&mut grads, loss.backward()?)?;
                t_backward += t_bwd.elapsed();
                let share = (end - start) as f64 / n_draws as f64;
                accumulate(&mut loss_sum, loss.detach())?;
                accumulate(&mut level_sum, per_level.detach().sum(0)?)?;
                accumulate(&mut pair_sum, out.mean_abs_pair.affine(share, 0.0)?)?;
            }

            // The step's other terms, once, on the whole step.
            let mut extra: Option<Tensor> = None;
            if args.embedding_l2 > 0.0 {
                // geu's ridge, not a local copy: the reduction is the whole
                // content of this penalty.
                accumulate(
                    &mut extra,
                    embedding_ridge(units.ridge_rows(model, &read), args.embedding_l2 as f64)?,
                )?;
            }
            // Which feature-side shrinkage applies is a model property
            // (`feature_ridge`). Pinned rows are the one cage-local exception:
            // the table ridge would only push gradient at rows the restore
            // below reverts anyway, so it is off under freeze and lora.
            let table_lam = if frozen_feature.is_some() {
                0.0
            } else {
                args.embedding_l2 as f64
            };
            if let Some(ridge) = model.feature_ridge(table_lam)? {
                accumulate(&mut extra, ridge)?;
            }
            // Exact unit–module term + membership priors, once per optimizer
            // step, as geu's `module_step_loss` makes it: the drawn units'
            // counts pooled through the (dropout-masked) membership, every
            // module scored against the trained unit rows. The pooling carries
            // no gradient (the masked membership and the pooled target are
            // detached), so cells pool their sparse counts.
            if let (Some(m), Some(gm), Some(picks)) = (&model.modules, module_cfg, &module_picks) {
                let pi = m.membership()?;
                let pi_masked = if gm.feature_dropout > 0.0 {
                    let keep =
                        draw_gene_keep_mask(n_features, gm.feature_dropout, &mut rng_master, dev)?;
                    masked_membership(&pi, &keep)?
                } else {
                    pi.detach()
                };
                if let Some((e_units, x_cm)) =
                    units.module_inputs(model, &read, picks, &pi_masked, dev)?
                {
                    accumulate(
                        &mut extra,
                        module_softmax_loss(&e_units, &m.mu, &m.b_module, &x_cm)?
                            .affine(f64::from(gm.lambda_module), 0.0)?,
                    )?;
                    if let Some(prior) = module_priors(m, &pi, gm.lambda_balance)? {
                        accumulate(&mut extra, prior)?;
                    }
                }
            }
            if let Some(extra) = extra {
                let t_bwd = std::time::Instant::now();
                add_grads(&mut grads, extra.backward()?)?;
                t_backward += t_bwd.elapsed();
            }
            timers.decoder_fwd += t_fwd.elapsed().saturating_sub(t_backward);
            timers.backward += t_backward;
            let t_opt = std::time::Instant::now();
            // Global-norm clip before the step. The NCE loss spikes when a
            // chunk draws a feature whose active edges are nearly all positives,
            // and an unbounded step there inflates the embedding norms every
            // later epoch has to work against. A step whose global norm is
            // non-finite is skipped rather than laundered into the parameters,
            // and counted: a run where every step is skipped moves nothing.
            let stepped = units.step(adams, grads.expect("a chunk backwarded"), read)?;
            steps_taken += 1;
            if !stepped {
                skipped_steps += 1;
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
            train_bar
                .set_position((units.progress(steps_taken, planned_steps) * 1000.0).round() as u64);

            // Diagnostics (no host sync) — accumulate detached tensors.
            let per_level_chunk_mean = level_sum
                .expect("a chunk has draws")
                .affine(1.0 / n_draws as f64, 0.0)?;
            accumulate(&mut pair_acc, pair_sum.expect("a chunk has draws"))?;
            accumulate(&mut epoch_loss_acc, loss_sum.expect("a chunk has draws"))?;
            accumulate(&mut per_level_acc, per_level_chunk_mean)?;
            chunk_count += 1;
            processed += chunk.len();
            if units.done() {
                break;
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
        let progress = units.progress(steps_taken, planned_steps);
        let planned = planned_epochs(epoch + 1, progress, args.epochs);
        if epoch == 0 {
            if let Some(note) = units.first_epoch_note(steps_per_epoch, args.epochs) {
                info!("{note}");
            }
        }
        info!(
            "epoch {} of {planned} ({:.0}% of training): mean loss = {:.4e} (per-level: {:?}), \
             samples = {}, skipped pairs = {}",
            epoch,
            progress * 100.0,
            mean_loss,
            mean_per_level,
            sample_count,
            skip_count
        );
        if let Some(m) = &model.modules {
            let pi_host: Vec<f32> = m.membership()?.detach().flatten_all()?.to_vec1()?;
            // cage draws its negatives over pseudobulks, so no within-module fallback.
            log_membership_diagnostics(m, &pi_host, epoch, planned, 0)?;
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
            train_bar.set_message(format!(
                "training epoch {}/{planned} loss={:.4e} |pair| {:.2e}",
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

        if units.done() {
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
    match units.summary(steps_taken) {
        Some((note, true)) => warn!("{note}"),
        Some((note, false)) => info!("{note}"),
        None => info!("training: {steps_taken} optimizer steps"),
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
    // feature/edge sampling, `decoder_fwd` the loss forward, then backward
    // (every piece's) and the step. Per-epoch work (diagnostic syncs, the
    // permutation shuffle) sits outside them.
    timers.log_summary();
    Ok(score_trace)
}

#[cfg(test)]
mod tests {
    use super::{epoch_means, planned_epochs};

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

    /// The expected epochs follow the share of training done, capped.
    #[test]
    fn planned_epochs_extrapolate_the_schedule() {
        assert_eq!(planned_epochs(1, 0.005, 1000), 200);
        assert_eq!(planned_epochs(10, 0.1, 1000), 100);
        assert_eq!(planned_epochs(1, 0.0001, 1000), 1000, "capped");
        assert_eq!(planned_epochs(3, 0.0, 50), 50, "nothing done yet");
    }
}
