//! Training the pair encoder, encoding every pair with it, and checking the
//! amortized answer against the exact one.

use super::super::{PairDictionary, SCORE_CLAMP};
use super::{
    rows_within_budget, CellRow, CorpusStats, Encoded, PairEncoder, PairEncoderSpec, Placement,
    Stats,
};
use crate::util::common::*;
use legume_numeric::candle::candle_core::Tensor;
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW};
use legume_numeric::candle::fast_index::gather_rows;
use legume_numeric::candle::vae::{clip_and_step_dense, PhaseTimers};
use legume_numeric::matrix::rand_util::mix_seed;
use legume_numeric::matrix::utils::{cosine, quantiles};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f64 = 1e-4;
const GRAD_CLIP: f64 = 5.0;
/// A step with fewer pairs than this is skipped.
const MIN_STEP_PAIRS: usize = 16;
/// Optimizer steps a run makes at least, whatever its pair count: a small
/// sample at the batch cap would otherwise train for a handful of steps.
const MIN_STEPS: usize = 512;
/// Pairs held out for the per-epoch evaluation line and the closing check
/// against the exact solver.
const CHECK_PAIRS: usize = 2048;

/// What training reports back.
pub(crate) struct TrainStats {
    pub steps: usize,
    pub nll_per_count: f32,
}

impl PairEncoder {
    /// The statistics of cells `ids`, gathered on the device.
    fn cell_stats(&self, corpus: &CorpusStats, ids: &[u32]) -> anyhow::Result<Stats> {
        corpus.gather(&Tensor::from_slice(ids, ids.len(), self.dict.device())?)
    }

    /// The endpoint statistics of a chunk of pairs, the `u` rows then the
    /// `v` rows, gathered on the device.
    fn endpoint_stats(&self, corpus: &CorpusStats, chunk: &[(u32, u32)]) -> anyhow::Result<Stats> {
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
            let pi: Vec<f32> = self.head.gate(&fwd.features)?.sum(0)?.to_vec1()?;
            for (s, p) in pi_sum.iter_mut().zip(&pi) {
                *s += f64::from(*p);
            }
            nll += f64::from(
                self.nll(&fwd.z_uv, &self.scores(&fwd.z_uv)?, &fwd.pairs)?
                    .sum_all()?
                    .to_scalar::<f32>()?,
            );
            count += f64::from(fwd.pairs.totals.sum_all()?.to_scalar::<f32>()?);
        }
        let n_rows = pairs.len().max(1) as f64;
        let pi_mean = pi_sum.iter().map(|&s| (s / n_rows) as f32).collect();
        Ok(((nll / count.max(1.0)) as f32, pi_mean))
    }

    /// Train on the pairs' likelihood (and the endpoints' own), from the
    /// current vars.
    pub(crate) fn train(
        &self,
        corpus: &[CellRow],
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
        // The batch is sized as it always was, for a pair and its two
        // endpoints, so the optimizer takes the same steps per epoch; a step
        // now scores its pairs and only its share of the endpoint cells, and
        // costs that much less.
        let batch = rows_within_budget(3 * self.dict.g, spec.batch).max(MIN_STEP_PAIRS);
        let steps_per_epoch = n_pairs.div_ceil(batch).max(1);
        let degree = Tensor::from_vec(degree, corpus.len(), self.dict.device())?;
        let epochs = if spec.epochs == 0 {
            0
        } else {
            spec.epochs.max(MIN_STEPS.div_ceil(steps_per_epoch))
        };
        let stats = CorpusStats::new(corpus, self.dict.d, self.dict.device())?;
        let mut adam = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: LEARNING_RATE,
                weight_decay: WEIGHT_DECAY,
                ..Default::default()
            },
        )?;

        // A fixed, seeded subset scored in evaluation mode after every epoch.
        let check: Vec<(u32, u32)> = {
            let mut rng = StdRng::seed_from_u64(mix_seed(seed, 0x0043_4845_434b));
            (0..CHECK_PAIRS.min(n_pairs))
                .map(|_| edges[rng.random_range(0..n_pairs)])
                .collect()
        };
        let (nll0, _) = self.evaluate(&stats, &check, batch)?;
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
            let mut loss_sum = 0f64;
            let mut n_steps = 0usize;
            for (k, chunk) in chunks.iter().enumerate() {
                let t = std::time::Instant::now();
                let pairs: Vec<(u32, u32)> = chunk.iter().map(|&i| edges[i]).collect();
                let endpoints = self.endpoint_stats(&stats, &pairs)?;
                let cells = cell_share(k);
                let selves = if cells.is_empty() {
                    None
                } else {
                    let ids = Tensor::from_slice(cells, cells.len(), self.dict.device())?;
                    Some((stats.gather(&ids)?, gather_rows(&degree, &ids)?))
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
            let (nll, pi) = self.evaluate(&stats, &check, batch)?;
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

    /// Place every cell, then every pair, in evaluation mode. `cell_block`
    /// cells share one pass and `pair_block` pairs one, at most: both are
    /// held to [`super::BLOCK_ELEMENTS`] on the feature axis.
    pub(crate) fn encode_all(
        &self,
        corpus: &[CellRow],
        edges: &[(u32, u32)],
        pair_block: usize,
        cell_block: usize,
    ) -> anyhow::Result<Encoded> {
        let dev = self.dict.device();
        let (n_cells, g, d) = (corpus.len(), self.dict.g, self.dict.d);
        let cell_block = rows_within_budget(g, cell_block);
        let pair_block = rows_within_budget(g, pair_block);
        let stats = CorpusStats::new(corpus, d, dev)?;
        // The certificate's outer-product chunks, built once for the pass
        // when they fit their budget, else per block.
        let ee = self.dict.outer_chunks_cached()?;

        // Cells: the codes stay on the device for the pair pass; the
        // self-pairs are written out.
        let mut codes = Vec::new();
        let mut cell_latent = Mat::zeros(n_cells, d);
        let mut cell_bias = vec![0f32; n_cells];
        let mut cell_gap = vec![0f32; n_cells];
        let bar = new_progress_bar(n_cells as u64).with_message("encoding cells");
        for (lb, ub) in legume_numeric::matrix::utils::generate_minibatch_intervals(
            n_cells,
            0,
            Some(cell_block),
        ) {
            let ids = Tensor::from_vec((lb as u32..ub as u32).collect::<Vec<u32>>(), ub - lb, dev)?;
            let own = stats.gather(&ids)?;
            let h = self.cell_codes(&own)?;
            // The self-pair's objective is over the doubled profile.
            let doubled = own.doubled()?;
            let z = self.self_pair_code(&h, &doubled)?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            let gap = self.decrement(&z, &scores, &doubled, ee.as_deref())?;
            let z_host: Vec<f32> = z.flatten_all()?.to_vec1()?;
            for (i, c) in (lb..ub).enumerate() {
                let total = corpus[c].total;
                if total > 0.0 {
                    for j in 0..d {
                        cell_latent[(c, j)] = z_host[i * d + j];
                    }
                    cell_bias[c] = (total.ln() - lse[i]).clamp(-SCORE_CLAMP, SCORE_CLAMP);
                    cell_gap[c] = gap[i];
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
        let mut pair_gap = vec![0f32; n_pairs];
        let bar = new_progress_bar(n_pairs as u64).with_message("encoding pairs");
        for (lb, ub) in legume_numeric::matrix::utils::generate_minibatch_intervals(
            n_pairs,
            0,
            Some(pair_block),
        ) {
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
            let (z, _) = self.pair_code(&h_u, &h_v, &pooled)?;
            let scores = self.scores(&z)?;
            let lse: Vec<f32> = scores.log_sum_exp(1)?.to_vec1()?;
            let gap = self.decrement(&z, &scores, &pooled, ee.as_deref())?;
            let z_host: Vec<f32> = z.flatten_all()?.to_vec1()?;
            for (i, &(u, v)) in chunk.iter().enumerate() {
                let total = corpus[u as usize].total + corpus[v as usize].total;
                if total > 0.0 {
                    for j in 0..d {
                        pair_latent[(lb + i, j)] = z_host[i * d + j];
                    }
                    pair_bias[lb + i] = (total.ln() - lse[i]).clamp(-SCORE_CLAMP, SCORE_CLAMP);
                    pair_gap[lb + i] = gap[i];
                }
            }
            bar.inc(b as u64);
        }
        bar.finish_and_clear();

        Ok(Encoded {
            pairs: Placement {
                latent: pair_latent,
                bias: pair_bias,
                gap: pair_gap,
            },
            cells: Placement {
                latent: cell_latent,
                bias: cell_bias,
                gap: cell_gap,
            },
        })
    }
}

///////////////////////////////
// The check against the MAP //
///////////////////////////////

/// A seeded sample of pairs solved exactly once, so any placement of those
/// pairs can be compared with the optimum.
pub(crate) struct ExactCheck {
    /// Pair index, its optimum `θ`, the NLL there, and its norm.
    solved: Vec<(usize, Vec<f32>, f32, f32)>,
    /// Each sampled pair's local profile.
    profiles: Vec<Vec<(u32, f32)>>,
}

/// How far a placement sits from the exact per-pair optimum on the sample.
pub(crate) struct AmortizationGap {
    pub mean_cosine: f32,
    /// Placement NLL over exact NLL, summed over the sample.
    pub nll_ratio: f32,
    /// Median and largest `‖z‖` of the placement and of the solver.
    pub norm_encoder: (f32, f32),
    pub norm_exact: (f32, f32),
}

impl ExactCheck {
    pub(crate) fn new(
        dict: &PairDictionary,
        corpus: &[CellRow],
        edges: &[(u32, u32)],
        ridge: f32,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(mix_seed(seed, 0x0047_4150));
        let sample: Vec<usize> = (0..CHECK_PAIRS.min(edges.len()))
            .map(|_| rng.random_range(0..edges.len()))
            .filter(|&e| {
                let (u, v) = edges[e];
                corpus[u as usize].total + corpus[v as usize].total > 0.0
            })
            .collect();
        let profiles: Vec<Vec<(u32, f32)>> = sample
            .par_iter()
            .map(|&e| {
                let (u, v) = edges[e];
                corpus[u as usize].pooled(&corpus[v as usize])
            })
            .collect();
        let solved = sample
            .par_iter()
            .zip(&profiles)
            .map(|(&e, obs)| {
                let (theta, _, _) = super::super::solve_exact(obs, dict, ridge);
                let norm = theta.iter().map(|v| v * v).sum::<f32>().sqrt();
                let nll = dict.nll(obs, &theta);
                (e, theta, nll, norm)
            })
            .collect();
        Self { solved, profiles }
    }

    pub(crate) fn n_pairs(&self) -> usize {
        self.solved.len()
    }

    /// Compare a placement of every pair (rows in `edges` order) with the
    /// sample's optima. The dictionary is the one the check was built on.
    pub(crate) fn compare(&self, dict: &PairDictionary, latent: &Mat) -> AmortizationGap {
        let per_pair: Vec<(f32, f32, f32)> = self
            .solved
            .par_iter()
            .zip(&self.profiles)
            .map(|((e, theta, _, _), obs)| {
                let z: Vec<f32> = latent.row(*e).iter().copied().collect();
                (
                    cosine(&z, theta),
                    dict.nll(obs, &z),
                    z.iter().map(|v| v * v).sum::<f32>().sqrt(),
                )
            })
            .collect();
        let n = per_pair.len().max(1) as f32;
        let (enc, exact) = per_pair
            .iter()
            .zip(&self.solved)
            .fold((0f64, 0f64), |(a, b), (p, s)| {
                (a + f64::from(p.1), b + f64::from(s.2))
            });
        let median_max = |v: Vec<f32>| {
            let q = quantiles(&v, &[0.5, 1.0]);
            (q[0], q[1])
        };
        AmortizationGap {
            mean_cosine: per_pair.iter().map(|p| p.0).sum::<f32>() / n,
            nll_ratio: if exact != 0.0 {
                (enc / exact) as f32
            } else {
                f32::NAN
            },
            norm_encoder: median_max(per_pair.iter().map(|p| p.2).collect()),
            norm_exact: median_max(self.solved.iter().map(|s| s.3).collect()),
        }
    }
}
