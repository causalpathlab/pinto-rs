//! The training loop, and the probe that sizes its chunks.

use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
use crate::cell_activity_graph_embedding::feature_chain_sampler::FeatureGatedChainSampler;
use crate::cell_activity_graph_embedding::loss::{cage_nce_loss_per_feature_level, CageLossOut};
use crate::cell_activity_graph_embedding::pretrained;
use crate::util::common::*;
use crate::util::score_trace::ScoreEntry;

use graph_embedding_util::loss::{
    draw_gene_keep_mask, embedding_ridge, log_membership_diagnostics, masked_membership,
    module_priors, module_step_loss,
};
use graph_embedding_util::model::JointEmbedModel;
use graph_embedding_util::stop::setup_stop_handler;
use graph_embedding_util::FeatureModuleConfig;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::candle::candle_core::Tensor;
use legume_numeric::candle::candle_nn::AdamW;
use legume_numeric::candle::fast_index::gather_rows;
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
        trainable_features,
        n_exp_batches,
        sampler,
        model,
        dev,
    } = *inputs;
    let mut mini: Vec<(usize, graph_embedding_util::loss::UnitChainBatch)> =
        Vec::with_capacity(n_probe_features);
    let mut features_sampled = 0usize;
    'fill: for &g in trainable_features
        .iter()
        .cycle()
        .take(PROBE_OVERSAMPLE * n_probe_features)
    {
        let before = mini.len();
        for b in 0..n_exp_batches {
            if let Some((cb, _)) = sampler.sample(g, b, probe_rng) {
                mini.push((g, cb));
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
    let (feature_ids, cbs): (Vec<usize>, Vec<_>) = mini.into_iter().unzip();
    let feature_ids_u32: Vec<u32> = feature_ids.iter().map(|&g| g as u32).collect();
    let out = cage_nce_loss_per_feature_level(
        model,
        cbs,
        &feature_ids_u32,
        args.nce_objective.to_ge(),
        dev,
    )?;
    out.per_feature_level.sum_all()
}

pub(super) fn resolve_feature_batch_size(inputs: ProbeInputs<'_>) -> usize {
    let ProbeInputs {
        args,
        trainable_features,
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
                16.min(trainable_features.len().max(1)),
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
    pub trainable_features: &'a [usize],
    pub n_exp_batches: usize,
    pub sampler: &'a FeatureGatedChainSampler<'a>,
    pub model: &'a JointEmbedModel,
    pub dev: &'a Device,
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
    pub feature_batch_size: usize,
    pub feature_fisher_weights: Option<&'a [f32]>,
    pub module_cfg: Option<&'a FeatureModuleConfig>,
    pub module_profiles: Option<&'a Mat>,
    pub n_exp_batches: usize,
    pub n_pb: usize,
    pub n_features: usize,
    pub n_chain_levels: usize,
    pub dev: &'a Device,
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
        feature_batch_size,
        feature_fisher_weights,
        module_cfg,
        module_profiles,
        n_exp_batches,
        n_pb,
        n_features,
        n_chain_levels,
        dev,
    } = inputs;
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
    let features_per_epoch_actual = if args.features_per_epoch > 0 {
        args.features_per_epoch.min(trainable_features.len())
    } else {
        trainable_features.len()
    };
    let steps_per_epoch = features_per_epoch_actual.div_ceil(feature_batch_size);
    // The LoRA ridge is a per-epoch weight; every step takes its share.
    if let (Some(l), Some(lp)) = (model.lora.as_mut(), lora_plus) {
        l.ridge_step = f64::from(lp.ridge) / steps_per_epoch.max(1) as f64;
    }
    let total_steps = args.epochs * steps_per_epoch.max(1);
    let model = &*model;
    let train_bar = new_progress_bar(total_steps as u64).with_message("training steps");

    // First ^C = graceful stop after current chunk, finalize outputs;
    // second ^C = hard abort. See graph_embedding_util::stop.
    let stop = setup_stop_handler();

    // Module warm-up: hold the k-means membership for the first epochs, release
    // it afterwards (geu's schedule). The profile matrix the exact term pools sits
    // on the device as `[n_pb × G]`, so a step is one row gather.
    let module_run: Option<(usize, Tensor)> = match (module_cfg, module_profiles) {
        (Some(gm), Some(p)) => {
            let wu = gm.warmup_epochs_for(args.epochs);
            if let Some(m) = &model.modules {
                m.set_frozen(wu > 0);
            }
            info!(
                "feature modules: membership held for {wu} of {} epochs, then trained",
                args.epochs
            );
            let rows: Vec<f32> = (0..p.ncols())
                .flat_map(|j| p.column(j).iter().copied().collect::<Vec<f32>>())
                .collect();
            Some((wu, Tensor::from_vec(rows, (p.ncols(), p.nrows()), dev)?))
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
        let mut perm: Vec<usize> = trainable_features.to_vec();
        perm.shuffle(&mut rng_master);
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
        // Features this epoch will actually visit.
        debug_assert_eq!(perm.len(), features_per_epoch_actual);
        for chunk in perm.chunks(feature_batch_size) {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            // (a) Parallel sampling — pure CPU, no candle.
            let t_sample = std::time::Instant::now();
            let mini: Vec<(usize, _)> = chunk
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
                            sampler.sample(g, b, &mut rng).map(|(cb, _st)| (g, cb))
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

            let (feature_ids, cb_batches): (Vec<usize>, Vec<_>) = mini.into_iter().unzip();
            sample_count += feature_ids.len();
            let t_fwd = std::time::Instant::now();

            // Feature identity enters the score function via the gated
            // loss. The chunk's feature ids parallel-replicate to align
            // with the [G*B] cell-side gathers.
            let feature_ids_u32: Vec<u32> = feature_ids.iter().map(|&g| g as u32).collect();

            // (b) ONE forward / backward over the whole chunk.
            let CageLossOut {
                per_feature_level: per_level_gl,
                mean_abs_pair,
            } = cage_nce_loss_per_feature_level(
                model,
                cb_batches,
                &feature_ids_u32,
                args.nce_objective.to_ge(),
                dev,
            )?; // [G, L]
                // NB-Fisher per-feature precision: scale each feature's row of the
                // per-level loss by w_g ∈ (0,1] before summing, so high-mean /
                // high-dispersion housekeeping features contribute less gradient
                // (the loss-side analog of bge's count·fisher positive draw and
                // lc's `apply_feature_weights` on the feature basis). Coverage stays
                // uniform — every feature is still visited once per epoch.
            let per_level_gl = match feature_fisher_weights {
                Some(w) => {
                    let w_chunk: Vec<f32> = feature_ids.iter().map(|&g| w[g]).collect();
                    let w_g1 = Tensor::from_vec(w_chunk, (feature_ids.len(), 1), dev)?;
                    per_level_gl.broadcast_mul(&w_g1)?
                }
                None => per_level_gl,
            };
            let loss = per_level_gl.sum_all()?;
            let mut total = loss.clone();

            if args.embedding_l2 > 0.0 {
                // geu's ridge, not a local copy: the reduction is the whole
                // content of this penalty and it was wrong here in the same way.
                total = (total + embedding_ridge(&model.e_cell, args.embedding_l2 as f64)?)?;
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
            // Exact pseudobulk–module term + membership priors, once per optimizer
            // step, through the same functions geu's composite trainer uses: draw
            // `units_per_step` pseudobulks uniformly, pool their count rows through
            // the (dropout-masked) membership, and score every module against the
            // trained PB table.
            if let (Some(m), Some(gm), Some((_, profile_t))) =
                (&model.modules, module_cfg, module_run.as_ref())
            {
                let u = gm.units_per_step.min(n_pb);
                if u > 0 && gm.lambda_module > 0.0 {
                    let picks: Vec<u32> = (0..u)
                        .map(|_| rng_master.random_range(0..n_pb) as u32)
                        .collect();
                    let idx = Tensor::from_vec(picks, u, dev)?;
                    let x = gather_rows(profile_t, &idx)?;
                    let pi = m.membership()?;
                    let pi_masked = if gm.feature_dropout > 0.0 {
                        let keep = draw_gene_keep_mask(
                            n_features,
                            gm.feature_dropout,
                            &mut rng_master,
                            dev,
                        )?;
                        masked_membership(&pi, &keep)?
                    } else {
                        pi.detach()
                    };
                    let e_units = gather_rows(&model.e_cell, &idx)?;
                    total =
                        (total + module_step_loss(m, &pi_masked, &x, &e_units, gm.lambda_module)?)?;
                    if let Some(prior) = module_priors(m, &pi, gm.lambda_balance)? {
                        total = (total + prior)?;
                    }
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
            let stepped = clip_and_step_dense_all(adams, grads, f64::from(args.grad_clip))?;
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
            let scale = chunk_count as f64;
            (
                (loss_sum as f64) / scale,
                pl_sum.iter().map(|v| *v as f64 / scale).collect::<Vec<_>>(),
            )
        } else {
            (f64::NAN, vec![0.0; n_chain_levels])
        };
        info!(
            "epoch {}: mean loss = {:.4e} (per-level: {:?}), samples = {}, skipped pairs = {}",
            epoch, mean_loss, mean_per_level, sample_count, skip_count
        );
        if let Some(m) = &model.modules {
            let pi_host: Vec<f32> = m.membership()?.detach().flatten_all()?.to_vec1()?;
            // cage draws its negatives over pseudobulks, so no within-module fallback.
            log_membership_diagnostics(m, &pi_host, epoch, args.epochs, 0)?;
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
                "training epoch {}/{} loss={:.4e} |pair| {:.2e}",
                epoch + 1,
                args.epochs,
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
        // mean losses are within `convergence_tol` relative range,
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
            if rel.is_finite() && rel < args.convergence_tol as f64 {
                info!(
                    "converged at epoch {}: last {} losses within rel-range {:.4} < tol {:.4}",
                    epoch, args.convergence_window, rel, args.convergence_tol
                );
                break;
            }
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
    if skipped_steps > 0 {
        warn!(
            "{}/{} optimizer steps were SKIPPED for non-finite gradients — those \
             steps moved no parameters. Lower --lr or --grad-clip if this is a \
             large fraction.",
            skipped_steps, total_steps
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
