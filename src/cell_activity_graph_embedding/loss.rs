//! cage's batched feature-modulated super-edge NCE: a row `z_e` per super
//! edge (cage's pseudobulk pairs, cage-cell's cell pairs), scored
//!
//! ```text
//! s(e) = θ_g · z_e + b_u + b_v        e = (u, v)
//! ```
//!
//! against negatives that are other super edges of the batch the row was
//! drawn in, one feature per draw ([`cage_cell_edge_nce_loss`]). The per-edge
//! NCE itself is geu's; cage trains with its softmax objective
//! ([`NceObjective`]).
//!
//! When blocks are small, the negative scores come from a table of every draw
//! against every pair of its block, formed by one batched product, and each
//! draw reads its entry: the very scores the per-draw gather forms, so the loss
//! and its gradient are unchanged. See [`BlockScoring`].

use crate::cell_activity_graph_embedding::feature_chain_sampler::EdgeChainBatch;
use crate::util::rows::uniq_with_inverse;
use graph_embedding_util::loss::{gather_feature_rows, logistic_nce, softmax_nce, NceObjective};
use graph_embedding_util::model::JointEmbedModel;
use legume_numeric::candle::candle_core::{DType, Device, Result as CResult, Tensor};
// Not `index_select`, whose CUDA backward is serial over ids: see fast_index.
use legume_numeric::candle::fast_index::gather_rows;

pub struct CageLossOut {
    /// Per-(feature, level) NCE loss, `[G, L]`.
    pub per_feature_level: Tensor,
    /// Mean `|θ_g · z_e|` over the positives, detached, the collapse
    /// detector. Watch it against the loss: if this decays toward zero while
    /// the loss still falls, the unit biases have taken over the objective
    /// and the feature direction is no longer identified.
    pub mean_abs_pair: Tensor,
}

/// How a chunk's negative scores are formed. Every choice gives the same
/// loss and gradient (`loss_block_tests`); they differ only in cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockScoring {
    /// The block table when it costs no more than the per-draw gather,
    /// the gather otherwise.
    Auto,
    /// Read every draw from the block score table.
    #[cfg(test)]
    Table,
    /// Gather each drawn negative's embedding row.
    #[cfg(test)]
    Gathered,
}

/// The rows of a step: each distinct super edge the step names (a
/// pseudobulk pair, or under cage-cell a cell pair), its embedding row `z`
/// (`[U, D]`, the step's leaf) and its two ends.
pub struct EdgeRows<'a> {
    pub z: &'a Tensor,
    pub left: &'a [u32],
    pub right: &'a [u32],
}

/// One chunk: its pair rows, one draw per `(feature, batch)`
/// naming pairs by their row in `rows` (see [`localize_edges`]), each
/// draw's feature and the block (batch) its negatives came from.
pub struct EdgeChunk<'a> {
    pub rows: EdgeRows<'a>,
    pub batches: Vec<EdgeChainBatch>,
    pub feature_ids: &'a [u32],
    pub row_blocks: &'a [u32],
}

/// The per-feature NCE over super edges (cage's pseudobulk pairs, cage-cell's
/// cell pairs), `[G, 1]`: the trained unit is the pair itself, `z_e` a row of
/// its own.
///
/// ```text
/// s(e) = θ_g · z_e + b_u + b_v        e = (u, v)
/// ```
///
/// A positive is a pair drawn by the feature's pooled count; its negatives
/// are other pairs of its batch (two units that are not adjacent have no
/// row to score), each scored by its own `z` and its own two ends' biases.
/// A draw's positives and negatives share its feature row, so each is one
/// batched product against it; the negatives come from a per-block table
/// of draws against pairs when that is smaller than their gathered rows.
pub fn cage_cell_edge_nce_loss(
    model: &JointEmbedModel,
    chunk: EdgeChunk<'_>,
    objective: NceObjective,
    dev: &Device,
) -> CResult<CageLossOut> {
    cage_cell_edge_nce_loss_scored(model, chunk, objective, BlockScoring::Auto, dev)
}

/// [`cage_cell_edge_nce_loss`] with the negative scoring chosen.
pub(crate) fn cage_cell_edge_nce_loss_scored(
    model: &JointEmbedModel,
    chunk: EdgeChunk<'_>,
    objective: NceObjective,
    scoring: BlockScoring,
    dev: &Device,
) -> CResult<CageLossOut> {
    let EdgeChunk {
        rows,
        batches,
        feature_ids,
        row_blocks,
    } = chunk;
    let g = batches.len();
    assert!(g > 0, "non-empty batches required");
    assert_eq!(
        feature_ids.len(),
        g,
        "feature_ids and batches length mismatch"
    );
    assert_eq!(
        row_blocks.len(),
        g,
        "row_blocks and batches length mismatch"
    );
    let b = batches[0].pos_edges.len();
    let k = batches[0].n_negatives;
    for eb in &batches {
        assert_eq!(eb.pos_edges.len(), b, "super-edge loss: B mismatch");
        assert_eq!(eb.n_negatives, k, "super-edge loss: K mismatch");
    }
    if b == 0 {
        return Ok(CageLossOut {
            per_feature_level: Tensor::zeros((g, 1), DType::F32, dev)?,
            mean_abs_pair: Tensor::zeros((), DType::F32, dev)?,
        });
    }
    let d = rows.z.dim(1)?;
    let mut pos: Vec<u32> = Vec::with_capacity(g * b);
    let mut negs: Vec<u32> = Vec::with_capacity(g * b * k);
    for eb in &batches {
        pos.extend(&eb.pos_edges);
        negs.extend(&eb.neg_edges);
    }
    // each pair row's bias, `b_u + b_v`
    let ends = |cells: &[u32]| Tensor::from_slice(cells, cells.len(), dev);
    let b_rows = (gather_rows(&model.b_cell, &ends(rows.left)?)?
        + gather_rows(&model.b_cell, &ends(rows.right)?)?)?;
    // one feature row per draw: its B positives and B·K negatives share it
    let theta = gather_feature_rows(model, &Tensor::from_slice(feature_ids, g, dev)?)?; // [G, D]
    let theta_col = theta.unsqueeze(2)?.contiguous()?; // [G, D, 1]
    let pos_idx = Tensor::from_vec(pos, g * b, dev)?;
    let pair_pos = gather_rows(rows.z, &pos_idx)?
        .reshape((g, b, d))?
        .matmul(&theta_col)?
        .reshape(g * b)?;
    let pos_score = (&pair_pos + gather_rows(&b_rows, &pos_idx)?)?;
    let mean_abs_pair = pair_pos.detach().abs()?.mean_all()?;

    let table = PairTable::new(row_blocks, &negs, b * k);
    let use_table = match scoring {
        BlockScoring::Auto => table.cost(d) <= g * b * k * d,
        #[cfg(test)]
        BlockScoring::Table => true,
        #[cfg(test)]
        BlockScoring::Gathered => false,
    };
    let neg_score = (if use_table {
        table.scores(&theta, rows.z, &b_rows, dev)?
    } else {
        // each negative's row, scored against its draw's feature row
        let neg_idx = Tensor::from_slice(&negs, g * b * k, dev)?;
        let pair_neg = gather_rows(rows.z, &neg_idx)?
            .reshape((g, b * k, d))?
            .matmul(&theta_col)?
            .reshape(g * b * k)?;
        (pair_neg + gather_rows(&b_rows, &neg_idx)?)?
    })
    .reshape((g * b, k))?;
    let negs = std::slice::from_ref(&neg_score);
    let per_edge = match objective {
        NceObjective::Logistic => logistic_nce(&pos_score, negs)?,
        NceObjective::Softmax => softmax_nce(&pos_score, negs)?,
    };
    Ok(CageLossOut {
        per_feature_level: per_edge.reshape((g, b))?.mean(1)?.unsqueeze(1)?,
        mean_abs_pair,
    })
}

/// The negative scores of a chunk as a table per block (batch):
/// every draw of the block against every distinct pair its negatives name
/// there, `θ_g · z_w + b_w`, one batched product, each negative then read
/// from it. A pair's negative score depends only on the draw's feature and
/// the pair, so the table holds each once.
struct PairTable {
    /// Per block, its draws, in caller order.
    draws: Vec<Vec<u32>>,
    /// Per block, the distinct pair rows its negatives name.
    units: Vec<Vec<u32>>,
    /// Per negative, in caller order: its entry in the flattened table.
    entry: Vec<u32>,
    /// The most draws, and the most pairs, of any block.
    width_draws: usize,
    width_units: usize,
}

impl PairTable {
    /// `negs` holds `per_draw` negatives (pair rows) per draw, the draws
    /// in `row_blocks`' order.
    fn new(row_blocks: &[u32], negs: &[u32], per_draw: usize) -> Self {
        let (labels, block_of_draw) = uniq_with_inverse(row_blocks);
        let n_blocks = labels.len();
        let mut draws: Vec<Vec<u32>> = vec![Vec::new(); n_blocks];
        for (i, &blk) in block_of_draw.iter().enumerate() {
            draws[blk as usize].push(i as u32);
        }
        // each block's negatives, its draws in order, deduplicated
        let mut units = Vec::with_capacity(n_blocks);
        let mut rank_of = vec![(0u32, 0u32); negs.len()]; // (block, rank) per negative
        for (blk, mine) in draws.iter().enumerate() {
            let listed: Vec<u32> = mine
                .iter()
                .flat_map(|&i| &negs[i as usize * per_draw..(i as usize + 1) * per_draw])
                .copied()
                .collect();
            let (uniq, inverse) = uniq_with_inverse(&listed);
            for (slot, &i) in mine.iter().enumerate() {
                for j in 0..per_draw {
                    rank_of[i as usize * per_draw + j] = (blk as u32, inverse[slot * per_draw + j]);
                }
            }
            units.push(uniq);
        }
        let width_draws = draws.iter().map(Vec::len).max().unwrap_or(0);
        let width_units = units.iter().map(Vec::len).max().unwrap_or(0);
        let mut position = vec![0u32; row_blocks.len()];
        for mine in &draws {
            for (p, &i) in mine.iter().enumerate() {
                position[i as usize] = p as u32;
            }
        }
        let entry = rank_of
            .iter()
            .enumerate()
            .map(|(n, &(blk, rank))| {
                let p = position[n / per_draw.max(1)] as usize;
                ((blk as usize * width_draws + p) * width_units + rank as usize) as u32
            })
            .collect();
        Self {
            draws,
            units,
            entry,
            width_draws,
            width_units,
        }
    }

    /// Floats the table holds, with the pair rows it reads at width `d`.
    fn cost(&self, d: usize) -> usize {
        let n_blocks = self.draws.len();
        n_blocks * self.width_units * (self.width_draws + d)
    }

    /// Every negative's score, in caller order, from feature rows `theta`
    /// `[G, D]`, pair rows `z` and their biases `b_rows`.
    fn scores(&self, theta: &Tensor, z: &Tensor, b_rows: &Tensor, dev: &Device) -> CResult<Tensor> {
        let (n_blocks, wd, wu) = (self.draws.len(), self.width_draws, self.width_units);
        let d = theta.dim(1)?;
        // padding repeats a block's first draw or pair, and is never read
        let padded = |lists: &[Vec<u32>], w: usize| -> Vec<u32> {
            lists
                .iter()
                .flat_map(|l| (0..w).map(move |j| l.get(j).copied().unwrap_or(l[0])))
                .collect()
        };
        let draw_idx = Tensor::from_vec(padded(&self.draws, wd), n_blocks * wd, dev)?;
        let unit_idx = Tensor::from_vec(padded(&self.units, wu), n_blocks * wu, dev)?;
        let th = gather_rows(theta, &draw_idx)?.reshape((n_blocks, wd, d))?;
        let zz = gather_rows(z, &unit_idx)?.reshape((n_blocks, wu, d))?;
        let bb = gather_rows(b_rows, &unit_idx)?.reshape((n_blocks, 1, wu))?;
        let table = th
            .matmul(&zz.transpose(1, 2)?.contiguous()?)?
            .broadcast_add(&bb)?
            .reshape((n_blocks * wd * wu, 1))?;
        let entry = Tensor::from_slice(&self.entry, self.entry.len(), dev)?;
        gather_rows(&table, &entry)?.reshape(self.entry.len())
    }
}

/// The distinct pairs `batches` and `extra` name, ascending (global
/// pair ids), with every pair id in `batches` rewritten as its index in
/// that list, the step's row. `extra`'s rows are returned beside it.
pub fn localize_edges(batches: &mut [EdgeChainBatch], extra: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let listed: Vec<u32> = batches
        .iter()
        .flat_map(|eb| eb.pos_edges.iter().chain(&eb.neg_edges))
        .chain(extra)
        .copied()
        .collect();
    let (uniq, inverse) = uniq_with_inverse(&listed);
    let mut local = inverse.into_iter();
    for eb in batches.iter_mut() {
        for e in eb.pos_edges.iter_mut().chain(eb.neg_edges.iter_mut()) {
            *e = local.next().expect("listed");
        }
    }
    (uniq, local.collect())
}
