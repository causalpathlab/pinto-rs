//! cage's batched feature-modulated PB-PB chain NCE (no cell is ever
//! scored here: `u`, `v` are finest-level super-cells).
//!
//! Scores `G` independent [`UnitChainBatch`]es, one per feature in the current
//! chunk, in a single forward pass, returning `[G, L]` per-(feature, level)
//! losses. Collapses ~18k tiny device forwards per epoch into ~600 big ones.
//!
//! Per-row score, for feature `g` at chain level `ℓ`:
//!
//! ```text
//! s(u, v) = θ_g · (e_u ⊙ e_v) + b_u + b_v
//! θ_g     = z_g ⊙ e_feature[g]     z ~ Bern(pip), redrawn once per EPOCH
//! ```
//!
//! `e_u ⊙ e_v` is the EDGE embedding: the pair's participation in each latent
//! community, high only when both endpoints load on it. So the score asks
//! whether `u` and `v` agree under a per-feature diagonal metric, an adjacency
//! statistic, where the rank-1 form it replaced scored any two cells lying
//! along `θ_g`, adjacent or not.
//!
//! cage's own loss: geu's chain module scores a rank-1 quadratic form and
//! has no feature-modulated variant left. The per-edge NCE itself is geu's, and
//! selectable, see [`NceObjective`] and `--nce-objective`.
//!
//! Every row's negatives are drawn inside one block of units (the super-cells
//! of the batch the row was sampled in). When blocks are small, the negative
//! scores come from a table of every anchor against every unit of its block,
//! formed by one batched matrix product, and each draw reads its entry: the
//! very scores the per-draw gather forms, so the loss and its gradient are
//! unchanged, while the `K · L` scattered `D`-wide row gathers per anchor
//! (and their scattered backward) become dense products. See [`BlockScoring`].

use graph_embedding_util::loss::{
    gather_feature_rows, logistic_nce, softmax_nce, NceObjective, UnitChainBatch,
};
use graph_embedding_util::model::JointEmbedModel;
use legume_numeric::candle::candle_core::{DType, Device, Result as CResult, Tensor};
// Not `index_select`, whose CUDA backward is serial over ids: see fast_index.
use legume_numeric::candle::fast_index::gather_rows;

/// Most anchors per tile of the block score table. A block's anchors are cut
/// into equal tiles, the last padded, so every tile is one equal-shaped
/// product in a single batched matmul. The tile is also at most the mean
/// anchors per block, so padding (under one tile per block) at most doubles
/// the anchor-side work.
const MAX_TILE: usize = 1024;

pub struct CageLossOut {
    /// Per-(feature, level) NCE loss, `[G, L]`.
    pub per_feature_level: Tensor,
    /// Mean `|θ_g · (e_u ⊙ e_v)|`, detached, the collapse detector.
    /// Watch it against the loss: if this decays toward zero while the loss
    /// still falls, the ungated cell biases have taken over the objective and
    /// the feature direction is no longer identified.
    pub mean_abs_pair: Tensor,
}

/// How a chunk's negative scores are formed. Every choice gives the same
/// loss and gradient (`loss_block_tests`); they differ only in cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockScoring {
    /// The block table when blocks are small enough to pay for it
    /// ([`table_pays`]), the per-draw gather otherwise.
    Auto,
    /// Read every draw from the block score table.
    #[cfg(test)]
    Table,
    /// Gather each drawn negative's embedding row.
    #[cfg(test)]
    Gathered,
}

/// The widest block for which the table beats the per-draw gather. Per
/// anchor the table costs `width` scores of `d + 2` multiply-adds, in a
/// dense product, where the gather moves `l · k` rows of `d + 1` floats to
/// scattered addresses and scatters them again backward. The table won on
/// CPU at a width about `1.5 · l · k`; the bound stays near that measured
/// range.
fn max_table_width(l: usize, k: usize) -> usize {
    2 * l * k
}

/// Batched feature-modulated per-level loss. `feature_ids[i]` is the feature for
/// `batches[i]`, and `row_blocks[i]` the block (batch) whose units its
/// negatives were drawn from; all batches must share the same `B`, `L` and `K`.
///
/// Levels differ only in their negative pools, the feature direction is the same
/// at every scale. The only modulation is the epoch's spike draw `z`, applied
/// through geu's `gathered_gate_weights`; there is no per-level gate.
///
/// `objective` picks the per-edge NCE, exactly as geu's feature-side loss does
/// (`graph_embedding_util::loss::feat`). Prefer `Softmax` under a sampled mask:
/// the sampler's profiled-Poisson normalizer is the sampled-softmax estimand,
/// so the gate and the loss then optimize the same thing.
pub fn cage_nce_loss_per_feature_level(
    model: &JointEmbedModel,
    batches: Vec<UnitChainBatch>,
    feature_ids: &[u32],
    row_blocks: &[u32],
    objective: NceObjective,
    dev: &Device,
) -> CResult<CageLossOut> {
    cage_nce_loss_scored(
        model,
        batches,
        feature_ids,
        row_blocks,
        objective,
        BlockScoring::Auto,
        dev,
    )
}

/// [`cage_nce_loss_per_feature_level`] with the negative scoring chosen.
pub(crate) fn cage_nce_loss_scored(
    model: &JointEmbedModel,
    batches: Vec<UnitChainBatch>,
    feature_ids: &[u32],
    row_blocks: &[u32],
    objective: NceObjective,
    scoring: BlockScoring,
    dev: &Device,
) -> CResult<CageLossOut> {
    let g = batches.len();
    assert!(g > 0, "non-empty batches required");
    assert_eq!(
        feature_ids.len(),
        g,
        "feature_ids ({}) and batches ({}) length mismatch",
        feature_ids.len(),
        g
    );
    assert_eq!(
        row_blocks.len(),
        g,
        "row_blocks and batches length mismatch"
    );
    let b = batches[0].left_units.len();
    let l = batches[0].per_level_neg.len();
    let k = batches[0].n_negatives;
    for cb in &batches {
        assert_eq!(cb.left_units.len(), b, "cage loss: B mismatch");
        assert_eq!(cb.per_level_neg.len(), l, "cage loss: L mismatch");
        assert_eq!(cb.n_negatives, k, "cage loss: K mismatch");
    }
    if b == 0 {
        return Ok(CageLossOut {
            per_feature_level: Tensor::zeros((g, l), DType::F32, dev)?,
            mean_abs_pair: Tensor::zeros((), DType::F32, dev)?,
        });
    }

    let total_b = g * b;
    let total_neg = total_b * k;

    let mut all_left: Vec<u32> = Vec::with_capacity(total_b);
    let mut all_right: Vec<u32> = Vec::with_capacity(total_b);
    let mut all_neg_per_level: Vec<Vec<u32>> =
        (0..l).map(|_| Vec::with_capacity(total_neg)).collect();
    // Replicate each feature id B times so the feature-side gather lines up
    // row-for-row with the [G*B] cell-side gather.
    let mut feature_repeat: Vec<u32> = Vec::with_capacity(total_b);
    for (cb, &gid) in batches.into_iter().zip(feature_ids.iter()) {
        all_left.extend(cb.left_units);
        all_right.extend(cb.right_units);
        feature_repeat.extend(std::iter::repeat_n(gid, b));
        for (lvl_neg, negs) in all_neg_per_level.iter_mut().zip(cb.per_level_neg) {
            lvl_neg.extend(negs);
        }
    }

    // Choose the scoring, and with it the order anchors are scored in: tile
    // order for the block table, caller order for the per-draw gather.
    let max_width = match scoring {
        BlockScoring::Auto => Some(max_table_width(l, k)),
        #[cfg(test)]
        BlockScoring::Table => Some(usize::MAX),
        #[cfg(test)]
        BlockScoring::Gathered => None,
    };
    let n_units = model.e_cell.dim(0)?;
    let layout = max_width
        .and_then(|cap| BlockLayout::new(row_blocks, b, k, n_units, &all_neg_per_level, cap));
    let lay_out = |v: Vec<u32>| match &layout {
        Some(layout) => layout.lay_out(&v, 1),
        None => v,
    };
    let n = layout.as_ref().map_or(total_b, |t| t.anchor_of_slot.len());

    let left_idx = Tensor::from_vec(lay_out(all_left), n, dev)?;
    let right_idx = Tensor::from_vec(lay_out(all_right), n, dev)?;
    let feature_idx = Tensor::from_vec(lay_out(feature_repeat), n, dev)?;
    let e_left = gather_rows(&model.e_cell, &left_idx)?;
    let b_left = gather_rows(&model.b_cell, &left_idx)?;
    let e_right = gather_rows(&model.e_cell, &right_idx)?;
    let b_right = gather_rows(&model.b_cell, &right_idx)?;

    // Feature rows with whatever gates them applied, geu's own helper, so cage,
    // `senna bge` and `senna gem` share ONE definition of what multiplies a
    // feature loading. It dispatches on what the model actually carries:
    //
    //   - sampled mask installed: this epoch's `z` rows (or the frozen `pip`
    //     between draws), applied to the raw effect;
    //   - learned gate: `σ(S/τ)` from `s_feat`, plus the reparameterization
    //     draw off `e_feat_logstd`, which is what makes the gate variational;
    //   - neither: a plain gather, ungated.
    //
    // It also reads `e_feat_raw` once a gate is on, so a post-training
    // `materialize_e_feat` cannot gate an already-gated table.
    //
    // Re-deriving this locally is what silently broke the learned arm: passing
    // `logits: None` to `gathered_gate_weights` makes it return `None` unless a
    // pip is installed, so the gate Vars never entered the graph, never got a
    // gradient, and `α` stayed pinned at its init (sd = 0.0000 across 18k
    // features) while the KL alone nudged the mean.
    let e_feature_l = gather_feature_rows(model, &feature_idx)?; // [n, D]

    // Left factor is shared across levels and negatives: the anchor's
    // projection along the feature direction. Only the right factor changes.
    // LINEAR in the feature direction: score the feature against the elementwise
    // product of the two endpoints, `theta_g . (e_a * e_b)`, rather than the
    // product of two separate projections.
    //
    // The rank-1 quadratic form it replaces, `e_a^T (theta theta^T) e_b`, is
    // high for ANY two cells lying along `theta`, including distant ones, so
    // it never tested adjacency at all. `diag(theta)` asks whether a and b
    // AGREE, under a per-feature diagonal metric, which is an adjacency statistic.
    // It also lets a feature say dims should ANTI-align (theta_h < 0), which a
    // rank-1 PSD form cannot express.
    // `θ_g ⊙ e_left` does not change across chain levels or between the
    // positive and its negatives, so it is formed ONCE, a 3x saving on the
    // left factor.
    let theta_left = (&e_feature_l * &e_left)?; // [n, D]
    let pair_pos = (&theta_left * &e_right)?.sum(1)?; // [n]
    let pos_score = ((&pair_pos + &b_left)? + &b_right)?;

    // Back to caller order, `[G*B]`; `None` when already in it. Padding
    // slots are never read, so they carry no gradient.
    let back = layout
        .as_ref()
        .map(|t| Tensor::from_slice(&t.slot_of_anchor, total_b, dev))
        .transpose()?;
    let to_anchors = |x: Tensor| match &back {
        Some(back) => gather_rows(&x, back),
        None => Ok(x),
    };

    // Collapse detector: the bias-free pair term, so its magnitude is not
    // masked by `b_u + b_v`, which is exactly what would be absorbing the
    // objective if the gate were collapsing.
    let mean_abs_pair = to_anchors(pair_pos.detach())?.abs()?.mean_all()?;

    let table = match &layout {
        Some(t) => Some((t, t.score_table(model, &theta_left, &b_left, dev)?)),
        None => None,
    };

    let mut per_feature_per_level: Vec<Tensor> = Vec::with_capacity(l);
    for (level, lvl_neg) in all_neg_per_level.iter().enumerate() {
        let neg_score = match &table {
            Some((layout, table)) => {
                let ranks = Tensor::from_slice(&layout.ranks[level], (n, k), dev)?;
                table.gather(&ranks, 1)?
            }
            None => {
                let neg_idx = Tensor::from_slice(lvl_neg, total_neg, dev)?;
                let e_neg_flat = gather_rows(&model.e_cell, &neg_idx)?;
                let b_neg_flat = gather_rows(&model.b_cell, &neg_idx)?;
                let e_neg = e_neg_flat.reshape((total_b, k, e_neg_flat.dim(1)?))?;
                let b_neg = b_neg_flat.reshape((total_b, k))?;
                // Negatives replace the RIGHT endpoint: the edge embedding becomes
                // `e_u * e_w`, scored against the same feature direction, which is
                // exactly geu's `score_negatives` with the feature-projected anchor
                // as the cell side. Call it rather than re-derive it: cage's own
                // copy of this expression was the slow broadcast form long after
                // the shape of the fix was known, and one definition is what stops
                // that recurring. (The bias terms associate in the other order
                // here; same value, and float associativity is well inside the
                // run-to-run spread.)
                JointEmbedModel::score_negatives(&e_neg, &theta_left, &b_neg, &b_left)?
            }
        };
        let negs = std::slice::from_ref(&neg_score);
        let per_edge = match objective {
            NceObjective::Logistic => logistic_nce(&pos_score, negs)?,
            NceObjective::Softmax => softmax_nce(&pos_score, negs)?,
        };
        let per_edge_gb = to_anchors(per_edge)?.reshape((g, b))?;
        per_feature_per_level.push(per_edge_gb.mean(1)?); // [G]
    }
    Ok(CageLossOut {
        per_feature_level: Tensor::stack(&per_feature_per_level, 1)?,
        mean_abs_pair,
    })
}

/// A slot that holds no anchor: the padding at the end of a block's last
/// tile. It scores a copy of anchor 0 and is never read.
const PAD: u32 = u32::MAX;

/// The block score table's shape. It is `[tiles * tile, width]`: each tile
/// holds up to `tile` anchors of one block against that block's units
/// (padded to `width`), and an anchor's score against unit `w` sits in its
/// slot's row at the rank of `w` in the block. Tiles run block by block.
struct BlockLayout {
    /// Per block, its distinct negative units, in first-drawn order.
    units: Vec<Vec<u32>>,
    /// Per level, each slot's `k` draws as ranks in its block, `[slots * k]`.
    ranks: Vec<Vec<u32>>,
    /// Per anchor, its slot: tile `t` holds slots `t * tile .. (t + 1) * tile`.
    slot_of_anchor: Vec<u32>,
    /// Per slot, the anchor it scores, or [`PAD`].
    anchor_of_slot: Vec<u32>,
    /// Per tile, its block.
    block_of_tile: Vec<u32>,
    tile: usize,
    /// The widest block, which every tile is padded to.
    width: usize,
}

impl BlockLayout {
    /// `negs_per_level[ℓ]` holds `k` draws per anchor, `b` anchors per row,
    /// each a unit id below `n_units`. `None` when no draws are made or a
    /// block holds more than `max_width` distinct units, which is as soon
    /// as the table stops paying.
    fn new(
        row_blocks: &[u32],
        b: usize,
        k: usize,
        n_units: usize,
        negs_per_level: &[Vec<u32>],
        max_width: usize,
    ) -> Option<Self> {
        if k == 0 {
            return None;
        }
        let mut labels: Vec<u32> = row_blocks.to_vec();
        labels.sort_unstable();
        labels.dedup();
        let n_blocks = labels.len();
        let block_of_row: Vec<usize> = row_blocks
            .iter()
            .map(|r| labels.binary_search(r).expect("label is listed"))
            .collect();

        // Tiles block by block, each block's anchors in caller order.
        let mut per_block = vec![0usize; n_blocks];
        for &blk in &block_of_row {
            per_block[blk] += b;
        }
        let tile = (block_of_row.len() * b)
            .div_ceil(n_blocks)
            .clamp(1, MAX_TILE);
        let mut first_tile = Vec::with_capacity(n_blocks);
        let mut block_of_tile = Vec::new();
        for (blk, &count) in per_block.iter().enumerate() {
            first_tile.push(block_of_tile.len());
            block_of_tile.extend(std::iter::repeat_n(blk as u32, count.div_ceil(tile)));
        }
        let mut filled = vec![0usize; n_blocks];
        let slot_of_anchor: Vec<u32> = block_of_row
            .iter()
            .flat_map(|&blk| std::iter::repeat_n(blk, b))
            .map(|blk| {
                let slot = first_tile[blk] * tile + filled[blk];
                filled[blk] += 1;
                slot as u32
            })
            .collect();
        let mut anchor_of_slot = vec![PAD; block_of_tile.len() * tile];
        for (a, &s) in slot_of_anchor.iter().enumerate() {
            anchor_of_slot[s as usize] = a as u32;
        }

        // Rank each unit within its block as it is first drawn. Slots visit
        // one block at a time, so a unit's rank is current exactly when it
        // was stamped by the block being visited.
        let mut stamp = vec![u32::MAX; n_units];
        let mut rank_of = vec![0u32; n_units];
        let mut units: Vec<Vec<u32>> = vec![Vec::new(); n_blocks];
        let mut ranks: Vec<Vec<u32>> = negs_per_level
            .iter()
            .map(|_| Vec::with_capacity(anchor_of_slot.len() * k))
            .collect();
        for (s, &a) in anchor_of_slot.iter().enumerate() {
            let blk = block_of_tile[s / tile];
            for (lvl_ranks, negs) in ranks.iter_mut().zip(negs_per_level) {
                if a == PAD {
                    lvl_ranks.extend(std::iter::repeat_n(0, k));
                    continue;
                }
                let a = a as usize;
                for &u in &negs[a * k..(a + 1) * k] {
                    let u = u as usize;
                    if stamp[u] != blk {
                        stamp[u] = blk;
                        rank_of[u] = units[blk as usize].len() as u32;
                        units[blk as usize].push(u as u32);
                        if units[blk as usize].len() > max_width {
                            return None;
                        }
                    }
                    lvl_ranks.push(rank_of[u]);
                }
            }
        }
        let width = units.iter().map(Vec::len).max().unwrap_or(0);
        Some(Self {
            units,
            ranks,
            slot_of_anchor,
            anchor_of_slot,
            block_of_tile,
            tile,
            width,
        })
    }

    /// `k` per-anchor values, in slot order, `[slots * k]`; a padding slot
    /// repeats anchor 0's.
    fn lay_out(&self, per_anchor: &[u32], k: usize) -> Vec<u32> {
        self.anchor_of_slot
            .iter()
            .flat_map(|&a| {
                let a = if a == PAD { 0 } else { a as usize };
                &per_anchor[a * k..(a + 1) * k]
            })
            .copied()
            .collect()
    }

    /// Every slot's score against every unit of its tile's block,
    /// `[tiles * tile, width]`: `θ_g · (e_u ⊙ e_w) + b_u + b_w`, the same
    /// expression the gathered path forms per draw. Both biases ride in the
    /// product as two extra columns, `[θ, 1, b_u] · [e_w, b_w, 1]`, so the
    /// table is written once. Padding columns score a block's first unit and
    /// are never read.
    fn score_table(
        &self,
        model: &JointEmbedModel,
        theta: &Tensor,
        b_anchor: &Tensor,
        dev: &Device,
    ) -> CResult<Tensor> {
        let (n_blocks, n_tiles) = (self.units.len(), self.block_of_tile.len());
        let (w, d) = (self.width, theta.dim(1)?);
        // Each block's units once, padded with its first unit.
        let block_units: Vec<u32> = self
            .units
            .iter()
            .flat_map(|u| (0..w).map(move |j| u.get(j).copied().unwrap_or(u[0])))
            .collect();
        let block_units = Tensor::from_vec(block_units, n_blocks * w, dev)?;
        let unit_side = Tensor::cat(
            &[
                &gather_rows(&model.e_cell, &block_units)?,
                &gather_rows(&model.b_cell, &block_units)?.unsqueeze(1)?,
                &Tensor::ones((n_blocks * w, 1), DType::F32, dev)?,
            ],
            1,
        )?
        .reshape((n_blocks, w * (d + 2)))?;
        // ...then each tile's copy of its block's side.
        let tile_blocks = Tensor::from_slice(&self.block_of_tile, n_tiles, dev)?;
        let unit_side = gather_rows(&unit_side, &tile_blocks)?.reshape((n_tiles, w, d + 2))?;
        let n = n_tiles * self.tile;
        let anchor_side = Tensor::cat(
            &[
                theta,
                &Tensor::ones((n, 1), DType::F32, dev)?,
                &b_anchor.unsqueeze(1)?,
            ],
            1,
        )?
        .reshape((n_tiles, self.tile, d + 2))?;
        anchor_side
            .matmul(&unit_side.transpose(1, 2)?)?
            .reshape((n, w))
    }
}
