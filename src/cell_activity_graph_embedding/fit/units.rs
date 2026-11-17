//! What a run trains besides the feature side, decided once per run:
//! cage's pseudobulk table, a part of the model, or cage-cell's pair table,
//! outside it. Each owns its draws, what a step reads, the rows its ridge
//! and module term take, its optimiser step and its schedule.

use crate::cell_activity_graph_embedding::args::GRAD_CLIP;
use crate::cell_activity_graph_embedding::feature_chain_sampler::{
    EdgeChainBatch, FeatureGatedChainSampler,
};
use crate::cell_activity_graph_embedding::loss::{
    cage_cell_edge_nce_loss, cage_nce_loss_per_feature_level, localize_edges, CageLossOut,
    EdgeChunk, EdgeRows,
};
use crate::cell_activity_graph_embedding::row_adam::RowAdam;
use crate::util::rows::uniq_with_inverse;

use super::module_cells::CellCounts;
use graph_embedding_util::loss::{NceObjective, UnitChainBatch};
use graph_embedding_util::model::JointEmbedModel;
use legume_numeric::candle::candle_core::backprop::GradStore;
use legume_numeric::candle::candle_core::{Device, Result as CResult, Tensor, Var};
use legume_numeric::candle::candle_nn::AdamW;
use legume_numeric::candle::fast_index::gather_rows;
use legume_numeric::candle::grad_clip::clip_grad_global_norm;
use legume_numeric::candle::vae::clip_and_step_dense_all;
use rand::rngs::SmallRng;

/// A run's trained units.
pub(super) trait Units {
    /// One `(feature, batch)` draw.
    type Draw: Send;
    /// What one step reads beyond the model.
    type Read;

    /// How many units the module term draws from.
    fn n_units(&self) -> usize;

    /// One draw for feature `g` in batch `b`: a function of the sampler
    /// alone, so the draws of a chunk run in parallel.
    fn sample(
        sampler: &FeatureGatedChainSampler<'_>,
        g: usize,
        b: usize,
        rng: &mut SmallRng,
    ) -> Option<Self::Draw>;

    /// What a step's `draws` and the module term's `picks` name, read once
    /// for the whole step; the draws are rewritten to it.
    fn read(&self, draws: &mut [Self::Draw], picks: &[u32]) -> CResult<Self::Read>;

    /// The per-(feature, level) NCE `[G, L]` of draws already read.
    fn nce(
        &self,
        model: &JointEmbedModel,
        read: &Self::Read,
        drawn: Drawn<'_, Self::Draw>,
        objective: NceObjective,
        dev: &Device,
    ) -> CResult<CageLossOut>;

    /// The trained rows the embedding ridge penalises this step.
    fn ridge_rows<'r>(&'r self, model: &'r JointEmbedModel, read: &'r Self::Read) -> &'r Tensor;

    /// The module term's unit embeddings and their counts pooled through
    /// the membership `pi`, for the drawn units `picks`; `None` without
    /// counts to pool.
    fn module_inputs(
        &self,
        model: &JointEmbedModel,
        read: &Self::Read,
        picks: &[u32],
        pi: &Tensor,
        dev: &Device,
    ) -> anyhow::Result<Option<(Tensor, Tensor)>>;

    /// Clip the step's summed gradient and step every parameter; `false`:
    /// the norm was not finite and nothing moved.
    fn step(
        &mut self,
        adams: &mut [AdamW],
        grads: GradStore,
        read: Self::Read,
    ) -> anyhow::Result<bool>;

    /// The share of the planned training done, after `steps` of the
    /// `planned_steps` that `--epochs` allows.
    fn progress(&self, steps: usize, planned_steps: usize) -> f64;

    /// Whether the units' own budget has ended training.
    fn done(&self) -> bool {
        false
    }

    /// What the first epoch showed about the length of training, for the
    /// log.
    fn first_epoch_note(&self, _steps_per_epoch: usize, _epochs_cap: usize) -> Option<String> {
        None
    }

    /// How the units' own budget ended, after `steps`: the message, and
    /// whether it fell short (a cap ended training first).
    fn summary(&self, _steps: usize) -> Option<(String, bool)> {
        None
    }
}

/// Some draws of a step, each one's feature and the block (batch) its
/// negatives came from.
pub(super) struct Drawn<'a, D> {
    pub draws: Vec<D>,
    pub feature_ids: &'a [u32],
    pub row_blocks: &'a [u32],
}

/// cage's units: the finest pseudobulks, whose table is the model's.
pub(super) struct Pseudobulks {
    pub n_pb: usize,
    /// The pseudobulks' count profiles `[n_pb × G]` on the device, when the
    /// module term pools them.
    pub profiles: Option<Tensor>,
}

impl Units for Pseudobulks {
    type Draw = UnitChainBatch;
    type Read = ();

    fn n_units(&self) -> usize {
        self.n_pb
    }

    fn sample(
        sampler: &FeatureGatedChainSampler<'_>,
        g: usize,
        b: usize,
        rng: &mut SmallRng,
    ) -> Option<UnitChainBatch> {
        sampler.sample(g, b, rng).map(|(cb, _)| cb)
    }

    fn read(&self, _draws: &mut [UnitChainBatch], _picks: &[u32]) -> CResult<()> {
        Ok(())
    }

    fn nce(
        &self,
        model: &JointEmbedModel,
        _read: &(),
        drawn: Drawn<'_, UnitChainBatch>,
        objective: NceObjective,
        dev: &Device,
    ) -> CResult<CageLossOut> {
        cage_nce_loss_per_feature_level(
            model,
            drawn.draws,
            drawn.feature_ids,
            drawn.row_blocks,
            objective,
            dev,
        )
    }

    fn ridge_rows<'r>(&'r self, model: &'r JointEmbedModel, _read: &'r ()) -> &'r Tensor {
        &model.e_cell
    }

    fn module_inputs(
        &self,
        model: &JointEmbedModel,
        _read: &(),
        picks: &[u32],
        pi: &Tensor,
        dev: &Device,
    ) -> anyhow::Result<Option<(Tensor, Tensor)>> {
        let Some(profiles) = &self.profiles else {
            return Ok(None);
        };
        let idx = Tensor::from_slice(picks, picks.len(), dev)?;
        Ok(Some((
            gather_rows(&model.e_cell, &idx)?,
            gather_rows(profiles, &idx)?.matmul(pi)?,
        )))
    }

    fn step(&mut self, adams: &mut [AdamW], grads: GradStore, _read: ()) -> anyhow::Result<bool> {
        clip_and_step_dense_all(adams, grads, f64::from(GRAD_CLIP))
    }

    fn progress(&self, steps: usize, planned_steps: usize) -> f64 {
        steps as f64 / planned_steps.max(1) as f64
    }
}

/// cage-cell's units: the cell pairs, a row each in a table outside the
/// model, trained to an updates-per-pair budget.
pub(super) struct Pairs<'a> {
    z: &'a mut RowAdam,
    /// Each row's two cells.
    edges: &'a [(u32, u32)],
    /// The cells' counts, when the module term pools them.
    counts: Option<CellCounts<'a>>,
    budget: PairBudget,
}

impl<'a> Pairs<'a> {
    pub(super) fn new(
        z: &'a mut RowAdam,
        edges: &'a [(u32, u32)],
        counts: Option<CellCounts<'a>>,
        updates_per_pair: usize,
    ) -> Self {
        let budget = PairBudget::new(updates_per_pair, z.n_rows());
        Self {
            z,
            edges,
            counts,
            budget,
        }
    }
}

/// The pairs one cage-cell step reads: their ids (ascending), the leaf
/// their rows were copied into, their cells, and the leaf rows of the
/// module term's draws.
pub(super) struct PairRead {
    ids: Vec<u32>,
    leaf: Var,
    left: Vec<u32>,
    right: Vec<u32>,
    module_rows: Vec<u32>,
}

impl Units for Pairs<'_> {
    type Draw = EdgeChainBatch;
    type Read = PairRead;

    fn n_units(&self) -> usize {
        self.z.n_rows()
    }

    fn sample(
        sampler: &FeatureGatedChainSampler<'_>,
        g: usize,
        b: usize,
        rng: &mut SmallRng,
    ) -> Option<EdgeChainBatch> {
        sampler.sample_edges(g, b, rng)
    }

    fn read(&self, draws: &mut [EdgeChainBatch], picks: &[u32]) -> CResult<PairRead> {
        let (ids, module_rows) = localize_edges(draws, picks);
        let leaf = Var::from_tensor(&self.z.rows(&ids)?)?;
        let (left, right) = ids.iter().map(|&e| self.edges[e as usize]).unzip();
        Ok(PairRead {
            ids,
            leaf,
            left,
            right,
            module_rows,
        })
    }

    fn nce(
        &self,
        model: &JointEmbedModel,
        read: &PairRead,
        drawn: Drawn<'_, EdgeChainBatch>,
        objective: NceObjective,
        dev: &Device,
    ) -> CResult<CageLossOut> {
        cage_cell_edge_nce_loss(
            model,
            EdgeChunk {
                rows: EdgeRows {
                    z: read.leaf.as_tensor(),
                    left: &read.left,
                    right: &read.right,
                },
                batches: drawn.draws,
                feature_ids: drawn.feature_ids,
                row_blocks: drawn.row_blocks,
            },
            objective,
            dev,
        )
    }

    /// The step's rows: the mean over a uniform share of the table has the
    /// whole table's expected gradient.
    fn ridge_rows<'r>(&'r self, _model: &'r JointEmbedModel, read: &'r PairRead) -> &'r Tensor {
        read.leaf.as_tensor()
    }

    /// A pair's counts are its two cells', each distinct cell read once.
    fn module_inputs(
        &self,
        _model: &JointEmbedModel,
        read: &PairRead,
        picks: &[u32],
        pi: &Tensor,
        dev: &Device,
    ) -> anyhow::Result<Option<(Tensor, Tensor)>> {
        let Some(cells) = &self.counts else {
            return Ok(None);
        };
        let u = picks.len();
        let ends: Vec<u32> = picks
            .iter()
            .flat_map(|&e| {
                let (a, b) = self.edges[e as usize];
                [a, b]
            })
            .collect();
        let (distinct, slot) = uniq_with_inverse(&ends);
        let pooled = cells.pool(&distinct, pi)?;
        let (left, right): (Vec<u32>, Vec<u32>) = slot.chunks(2).map(|ab| (ab[0], ab[1])).unzip();
        let at = |i: Vec<u32>| Tensor::from_vec(i, u, dev);
        let x = (gather_rows(&pooled, &at(left)?)? + gather_rows(&pooled, &at(right)?)?)?;
        let e = gather_rows(
            read.leaf.as_tensor(),
            &Tensor::from_slice(&read.module_rows, u, dev)?,
        )?;
        Ok(Some((e, x)))
    }

    /// The global-norm clip (legume's `clip_grad_global_norm`) with the
    /// pair rows' gradient inside the norm, then the dense optimisers and
    /// the row optimiser.
    fn step(
        &mut self,
        adams: &mut [AdamW],
        mut grads: GradStore,
        read: PairRead,
    ) -> anyhow::Result<bool> {
        let norm = clip_grad_global_norm(&mut grads, f64::from(GRAD_CLIP))?;
        if !norm.is_finite() {
            return Ok(false);
        }
        let row_grad = match grads.remove(read.leaf.as_tensor()) {
            Some(g) => g,
            None => read.leaf.as_tensor().zeros_like()?,
        };
        // clipped above; this only steps
        clip_and_step_dense_all(adams, grads, 0.0)?;
        self.z.step(&read.ids, read.leaf.as_tensor(), &row_grad)?;
        self.budget.record(read.ids.len());
        Ok(true)
    }

    fn progress(&self, _steps: usize, _planned_steps: usize) -> f64 {
        self.budget.fraction()
    }

    fn done(&self) -> bool {
        self.budget.reached()
    }

    fn first_epoch_note(&self, steps_per_epoch: usize, epochs_cap: usize) -> Option<String> {
        let b = &self.budget;
        let steps = b.steps_needed();
        Some(format!(
            "cage-cell: a step touches {:.0} of {} pairs on average, so {} pair updates each \
             take about {steps} steps ({} epochs of {steps_per_epoch}; --epochs caps at {epochs_cap})",
            b.mean_touched(),
            b.pairs,
            b.per_pair,
            steps.div_ceil(steps_per_epoch.max(1)),
        ))
    }

    fn summary(&self, steps: usize) -> Option<(String, bool)> {
        let b = &self.budget;
        Some(if b.reached() {
            (
                format!(
                    "cage-cell: {:.1} updates per pair reached after {steps} steps",
                    b.mean_updates()
                ),
                false,
            )
        } else {
            (
                format!(
                    "cage-cell: stopped at {:.1} of {} updates per pair after {steps} steps \
                     (--epochs or --convergence-window ended it first)",
                    b.mean_updates(),
                    b.per_pair
                ),
                true,
            )
        })
    }
}

/// cage-cell's training length: `per_pair` updates of each of `pairs`
/// rows on average, the updates of a step being the distinct pairs it
/// touched (a row moves at most once a step).
#[derive(Debug)]
pub(super) struct PairBudget {
    per_pair: usize,
    pairs: usize,
    touched: u64,
    steps: usize,
}

impl PairBudget {
    pub(super) fn new(per_pair: usize, pairs: usize) -> Self {
        Self {
            per_pair,
            pairs,
            touched: 0,
            steps: 0,
        }
    }

    fn target(&self) -> u64 {
        self.per_pair as u64 * self.pairs as u64
    }

    /// One step that moved `touched` distinct pairs.
    pub(super) fn record(&mut self, touched: usize) {
        self.touched += touched as u64;
        self.steps += 1;
    }

    pub(super) fn reached(&self) -> bool {
        self.touched >= self.target()
    }

    pub(super) fn fraction(&self) -> f64 {
        (self.touched as f64 / self.target().max(1) as f64).min(1.0)
    }

    pub(super) fn mean_touched(&self) -> f64 {
        self.touched as f64 / self.steps.max(1) as f64
    }

    pub(super) fn mean_updates(&self) -> f64 {
        self.touched as f64 / self.pairs.max(1) as f64
    }

    /// The steps the budget takes at the pairs a step has touched so far.
    pub(super) fn steps_needed(&self) -> usize {
        (self.target() as f64 / self.mean_touched().max(1.0)).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::PairBudget;

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
        assert!((b.fraction() - 0.9).abs() < 1e-12);
        b.record(30);
        assert!(b.reached());
        assert_eq!(b.mean_updates(), 3.0);
    }
}
