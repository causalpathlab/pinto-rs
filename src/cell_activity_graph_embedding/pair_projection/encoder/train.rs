//! Training the pair encoder, and encoding every pair with it.

use super::super::{CellLatent, PairLatent, SCORE_CLAMP};
use super::{rows_within_budget, CellRow, CorpusStats, FullStats, PairEncoder, PairEncoderSpec};
use crate::util::common::*;
use legume_numeric::candle::candle_core::Tensor;
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW};
use legume_numeric::candle::fast_index::gather_rows;
use legume_numeric::candle::vae::{clip_and_step_dense, PhaseTimers};
use legume_numeric::matrix::rand_util::mix_seed;
use legume_numeric::matrix::utils::generate_minibatch_intervals;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f64 = 1e-4;
const GRAD_CLIP: f64 = 5.0;
/// A step with fewer pairs than this is skipped.
const MIN_STEP_PAIRS: usize = 16;
/// Pairs held out for the evaluation line.
const CHECK_PAIRS: usize = 2048;
/// Evaluations over a run, about, besides the one before training and the
/// one after the last epoch.
const CHECKS_PER_RUN: usize = 8;

/// What training reports back.
pub(crate) struct TrainStats {
    pub steps: usize,
    pub nll_per_count: f32,
}

impl PairEncoder {
    /// The statistics of cells `ids`, gathered on the device.
    fn cell_stats(&self, corpus: &CorpusStats, ids: &[u32]) -> anyhow::Result<FullStats> {
        corpus.gather_full(&Tensor::from_slice(ids, ids.len(), self.dict.device())?)
    }

    /// The endpoint statistics of a chunk of pairs, the `u` rows then the
    /// `v` rows, gathered on the device.
    fn endpoint_stats(
        &self,
        corpus: &CorpusStats,
        chunk: &[(u32, u32)],
    ) -> anyhow::Result<FullStats> {
        let ids: Vec<u32> = chunk
            .iter()
            .map(|&(u, _)| u)
            .chain(chunk.iter().map(|&(_, v)| v))
            .collect();
        self.cell_stats(corpus, &ids)
    }

    /// NLL per pooled count over `pairs`, and the mean expert weights: the
    /// checkpoint reading, and whether the gate still routes.
    fn evaluate(
        &self,
        corpus: &CorpusStats,
        pairs: &[(u32, u32)],
        batch: usize,
    ) -> anyhow::Result<(f32, Vec<f32>)> {
        let (mut nll, mut count) = (0f64, 0f64);
        let mut pi_sum = vec![0f64; self.head.n_experts()];
        for chunk in pairs.chunks(batch.max(1)) {
            let fwd = self.pair_forward(self.endpoint_stats(corpus, chunk)?)?;
            let pi: Vec<f32> = self.gate_weights(&fwd)?.sum(0)?.to_vec1()?;
            for (s, p) in pi_sum.iter_mut().zip(&pi) {
                *s += f64::from(*p);
            }
            nll += f64::from(
                self.nll(&fwd.z_uv, &self.scores(&fwd.z_uv)?, &fwd.pairs)?
                    .sum_all()?
                    .to_scalar::<f32>()?,
            );
            count += f64::from(fwd.pairs.stats.totals.sum_all()?.to_scalar::<f32>()?);
        }
        let n_rows = pairs.len().max(1) as f64;
        let pi_mean = pi_sum.iter().map(|&s| (s / n_rows) as f32).collect();
        Ok(((nll / count.max(1.0)) as f32, pi_mean))
    }

    /// Train on the pairs' likelihood (and the endpoints' own), from the
    /// current vars; `stats` is [`Self::corpus_stats`] of `corpus`.
    pub(crate) fn train(
        &self,
        corpus: &[CellRow],
        stats: &CorpusStats,
        edges: &[(u32, u32)],
        spec: &PairEncoderSpec,
        seed: u64,
    ) -> anyhow::Result<TrainStats> {
        let n_pairs = edges.len();
        // How many pair endpoints each cell is: its weight in the self-pair
        // term, which visits each endpoint cell once per epoch.
        anyhow::ensure!(
            edges
                .iter()
                .all(|&(u, v)| (u.max(v) as usize) < corpus.len()),
            "pair encoder: an edge names a cell beyond the corpus's {}",
            corpus.len()
        );
        let mut degree = vec![0f32; corpus.len()];
        for &(u, v) in edges {
            degree[u as usize] += 1.0;
            degree[v as usize] += 1.0;
        }
        let mut cell_order: Vec<u32> = (0..corpus.len() as u32)
            .filter(|&c| degree[c as usize] > 0.0)
            .collect();
        // The batch is sized for a pair and its two endpoints; a step scores
        // its pairs and only its share of the endpoint cells.
        let batch = rows_within_budget(3 * self.dict.g, spec.batch).max(MIN_STEP_PAIRS);
        let steps_per_epoch = n_pairs.div_ceil(batch).max(1);
        let degree = Tensor::from_vec(degree, corpus.len(), self.dict.device())?;
        let epochs = spec.steps.div_ceil(steps_per_epoch);
        let check_every = epochs.div_ceil(CHECKS_PER_RUN).max(1);
        let mut adam = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: LEARNING_RATE,
                weight_decay: WEIGHT_DECAY,
                ..Default::default()
            },
        )?;

        // A fixed, seeded subset scored in evaluation mode every
        // `check_every` epochs and after the last.
        let check: Vec<(u32, u32)> = {
            let mut rng = StdRng::seed_from_u64(mix_seed(seed, 0x0043_4845_434b));
            (0..CHECK_PAIRS.min(n_pairs))
                .map(|_| edges[rng.random_range(0..n_pairs)])
                .collect()
        };
        let (nll0, _) = self.evaluate(stats, &check, batch)?;
        info!(
            "Pair encoder: {n_pairs} pairs, L={}, K={}, {epochs} epochs of {batch}-pair steps, \
             ridge λ={}; NLL/count before {nll0:.4}",
            self.trunk_width(),
            self.n_experts(),
            self.ridge
        );

        let mut order: Vec<usize> = (0..n_pairs).collect();
        let mut timers = PhaseTimers::default();
        let (mut steps, mut skipped) = (0usize, 0usize);
        let mut nll_per_count = nll0;
        let bar = new_progress_bar((epochs * steps_per_epoch) as u64).with_message("pair encoder");
        // The mean step loss since the last evaluation line.
        let (mut loss_sum, mut n_steps) = (0f64, 0usize);
        for epoch in 0..epochs {
            let mut rng = StdRng::seed_from_u64(mix_seed(seed, epoch as u64));
            order.shuffle(&mut rng);
            cell_order.shuffle(&mut rng);
            // The steps that train (a last chunk under MIN_STEP_PAIRS is
            // skipped), and the endpoint cells shared out over exactly those.
            let chunks: Vec<&[usize]> = order
                .chunks(batch)
                .filter(|c| c.len() >= MIN_STEP_PAIRS)
                .collect();
            bar.inc((steps_per_epoch - chunks.len()) as u64);
            let n_chunks = chunks.len().max(1);
            let cell_share = |k: usize| {
                &cell_order[k * cell_order.len() / n_chunks..(k + 1) * cell_order.len() / n_chunks]
            };
            for (k, chunk) in chunks.iter().enumerate() {
                let t = std::time::Instant::now();
                let pairs: Vec<(u32, u32)> = chunk.iter().map(|&i| edges[i]).collect();
                let endpoints = self.endpoint_stats(stats, &pairs)?;
                let cells = cell_share(k);
                let selves = if cells.is_empty() {
                    None
                } else {
                    let ids = Tensor::from_slice(cells, cells.len(), self.dict.device())?;
                    Some((stats.gather_full(&ids)?, gather_rows(&degree, &ids)?))
                };
                timers.precompute += t.elapsed();

                let t = std::time::Instant::now();
                let loss = self.step_loss(endpoints, selves, batch)?;
                timers.encoder_fwd += t.elapsed();

                let t = std::time::Instant::now();
                let grads = loss.backward()?;
                timers.backward += t.elapsed();

                let t = std::time::Instant::now();
                if !clip_and_step_dense(&mut adam, grads, GRAD_CLIP)? {
                    skipped += 1;
                }
                timers.optimize += t.elapsed();

                loss_sum += f64::from(loss.to_scalar::<f32>()?);
                n_steps += 1;
                steps += 1;
                bar.inc(1);
            }
            if (epoch + 1) % check_every != 0 && epoch + 1 != epochs {
                continue;
            }
            let (nll, pi) = self.evaluate(stats, &check, batch)?;
            nll_per_count = nll;
            // The held-out NLL first: it is the score `pinto run` draws.
            bar.set_message(format!(
                "pair encoder epoch {}/{epochs} nll={nll:.4}",
                epoch + 1
            ));
            info!(
                "Pair encoder epoch {}/{epochs}: mean step loss {:.2}, held-out NLL/count {:.4}, expert usage {}",
                epoch + 1,
                loss_sum / n_steps.max(1) as f64,
                nll,
                pi.iter()
                    .map(|p| format!("{p:.2}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            (loss_sum, n_steps) = (0.0, 0);
        }
        bar.finish_and_clear();
        timers.log_summary();
        if skipped > 0 {
            warn!("Pair encoder: {skipped}/{steps} steps were skipped for non-finite gradients");
        }
        Ok(TrainStats {
            steps,
            nll_per_count,
        })
    }

    /// Place every cell, then every pair, in evaluation mode; `stats` is
    /// [`Self::corpus_stats`] of `corpus`, and the placement's `scores` come
    /// back empty. `cell_block` cells share one pass and `pair_block` pairs
    /// one, at most: both are held to [`super::BLOCK_ELEMENTS`] on the
    /// feature axis.
    pub(crate) fn encode_all(
        &self,
        corpus: &[CellRow],
        stats: &CorpusStats,
        edges: &[(u32, u32)],
        pair_block: usize,
        cell_block: usize,
    ) -> anyhow::Result<PairLatent> {
        let dev = self.dict.device();
        let (n_cells, g, d) = (corpus.len(), self.dict.g, self.dict.d);
        let cell_block = rows_within_budget(g, cell_block);
        let pair_block = rows_within_budget(g, pair_block);

        // Cells: the codes stay on the device for the pair pass; the
        // self-pairs are written out.
        let mut codes = Vec::new();
        let mut cell_latent = Mat::zeros(n_cells, d);
        let mut cell_bias = vec![0f32; n_cells];
        let bar = new_progress_bar(n_cells as u64).with_message("encoding cells");
        for (lb, ub) in generate_minibatch_intervals(n_cells, 0, Some(cell_block)) {
            let ids = Tensor::from_vec((lb as u32..ub as u32).collect::<Vec<u32>>(), ub - lb, dev)?;
            let own = stats.gather(&ids)?;
            let h = self.cell_codes(&own)?;
            // The self-pair's objective is over the doubled profile.
            let doubled = own.doubled()?;
            let z = self.self_pair_code(&h, &doubled)?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            let z_host: Vec<f32> = z.flatten_all()?.to_vec1()?;
            for (i, c) in (lb..ub).enumerate() {
                let total = corpus[c].total;
                if total > 0.0 {
                    for j in 0..d {
                        cell_latent[(c, j)] = z_host[i * d + j];
                    }
                    cell_bias[c] = (total.ln() - lse[i]).clamp(-SCORE_CLAMP, SCORE_CLAMP);
                }
            }
            codes.push(h);
            bar.inc((ub - lb) as u64);
        }
        bar.finish_and_clear();
        let codes = Tensor::cat(&codes, 0)?;

        // Pairs: the two endpoints' codes and statistics gathered on the
        // device, through the head.
        let n_pairs = edges.len();
        let mut pair_latent = Mat::zeros(n_pairs, d);
        let mut pair_bias = vec![0f32; n_pairs];
        let bar = new_progress_bar(n_pairs as u64).with_message("encoding pairs");
        for (lb, ub) in generate_minibatch_intervals(n_pairs, 0, Some(pair_block)) {
            let chunk = &edges[lb..ub];
            let b = chunk.len();
            let ids = |pick: fn(&(u32, u32)) -> u32| -> anyhow::Result<Tensor> {
                Ok(Tensor::from_vec(
                    chunk.iter().map(pick).collect::<Vec<u32>>(),
                    b,
                    dev,
                )?)
            };
            let (u, v) = (ids(|e| e.0)?, ids(|e| e.1)?);
            let (h_u, h_v) = (gather_rows(&codes, &u)?, gather_rows(&codes, &v)?);
            let pooled = stats.gather(&u)?.pooled(&stats.gather(&v)?)?;
            let z = self.pair_code(&h_u, &h_v, &pooled)?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            let z_host: Vec<f32> = z.flatten_all()?.to_vec1()?;
            for (i, &(u, v)) in chunk.iter().enumerate() {
                let total = corpus[u as usize].total + corpus[v as usize].total;
                if total > 0.0 {
                    for j in 0..d {
                        pair_latent[(lb + i, j)] = z_host[i * d + j];
                    }
                    pair_bias[lb + i] = (total.ln() - lse[i]).clamp(-SCORE_CLAMP, SCORE_CLAMP);
                }
            }
            bar.inc(b as u64);
        }
        bar.finish_and_clear();

        Ok(PairLatent {
            latent: pair_latent,
            bias: pair_bias,
            scores: Vec::new(),
            cells: CellLatent {
                latent: cell_latent,
                bias: cell_bias,
            },
        })
    }
}
