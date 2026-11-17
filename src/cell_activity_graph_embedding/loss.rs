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
//! has no feature-modulated variant left. The per-edge NCE itself is geu's;
//! cage trains with its softmax objective ([`NceObjective`]).
//!
//! cage-cell trains a row per cell pair instead ([`cage_cell_edge_nce_loss`]):
//! `s(e) = θ_g · z_e + b_u + b_v`, its negatives other pairs of the batch,
//! through the same NCE and the same negative scoring.
//!
//! Every row's negatives are drawn inside one block of units (the super-cells
//! of the batch the row was sampled in). When blocks are small, the negative
//! scores come from a table of every anchor against every unit of its block,
//! formed by one batched matrix product, and each draw reads its entry: the
//! very scores the per-draw gather forms, so the loss and its gradient are
//! unchanged, while the `K · L` scattered `D`-wide row gathers per anchor
//! (and their scattered backward) become dense products. See [`BlockScoring`].

use crate::cell_activity_graph_embedding::feature_chain_sampler::EdgeChainBatch;
use crate::util::rows::uniq_with_inverse;
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
    let layout = block_layout(scoring, row_blocks, b, l, k, &all_neg_per_level);
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

    score_slots(
        Slots {
            layout: layout.as_ref(),
            anchor: &theta_left,
            anchor_bias: &b_left,
            pos_score: &pos_score,
            pair_pos: &pair_pos,
            unit_e: &model.e_cell,
            unit_b: &model.b_cell,
            negs: &all_neg_per_level,
            g,
            b,
            k,
        },
        objective,
        dev,
    )
}

/// The block layout `scoring` asks for over `negs_per_level` (`l` levels of
/// `k` draws per anchor, `b` anchors per row), or `None` for the per-draw
/// gather.
fn block_layout(
    scoring: BlockScoring,
    row_blocks: &[u32],
    b: usize,
    l: usize,
    k: usize,
    negs_per_level: &[Vec<u32>],
) -> Option<BlockLayout> {
    let max_width = match scoring {
        BlockScoring::Auto => Some(max_table_width(l, k)),
        #[cfg(test)]
        BlockScoring::Table => Some(usize::MAX),
        #[cfg(test)]
        BlockScoring::Gathered => None,
    };
    max_width.and_then(|cap| BlockLayout::new(row_blocks, b, k, negs_per_level, cap))
}

/// A chunk's scored slots, in the layout's order (caller order without
/// one): what the unit front end hands the NCE.
struct Slots<'a> {
    layout: Option<&'a BlockLayout>,
    /// `[n, D]`, the factor every negative's unit row is scored against.
    anchor: &'a Tensor,
    /// `[n]`, the anchor's part of every negative's score.
    anchor_bias: &'a Tensor,
    /// `[n]`, each positive's score.
    pos_score: &'a Tensor,
    /// `[n]`, each positive's bias-free pair term.
    pair_pos: &'a Tensor,
    /// The unit table the negatives are rows of, `[units, D]`, and its
    /// per-unit bias `[units]`.
    unit_e: &'a Tensor,
    unit_b: &'a Tensor,
    /// Per level, `k` negative units per anchor, in caller order.
    negs: &'a [Vec<u32>],
    g: usize,
    b: usize,
    k: usize,
}

/// Each negative's score `anchor · e_w + b_w + anchor_bias`, from the block
/// table or the per-draw gather, then the NCE per level, back in caller
/// order, averaged per feature.
fn score_slots(s: Slots<'_>, objective: NceObjective, dev: &Device) -> CResult<CageLossOut> {
    let Slots {
        layout,
        anchor,
        anchor_bias,
        pos_score,
        pair_pos,
        unit_e,
        unit_b,
        negs,
        g,
        b,
        k,
    } = s;
    let total_b = g * b;
    let n = anchor.dim(0)?;
    // Back to caller order, `[G*B]`; `None` when already in it. Padding
    // slots are never read, so they carry no gradient.
    let back = layout
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

    let table = match layout {
        Some(t) => Some((t, t.score_table(unit_e, unit_b, anchor, anchor_bias, dev)?)),
        None => None,
    };

    let mut per_feature_per_level: Vec<Tensor> = Vec::with_capacity(negs.len());
    for (level, lvl_neg) in negs.iter().enumerate() {
        let neg_score = match &table {
            Some((layout, table)) => {
                let ranks = Tensor::from_slice(&layout.ranks[level], (n, k), dev)?;
                table.gather(&ranks, 1)?
            }
            None => {
                let neg_idx = Tensor::from_slice(lvl_neg, total_b * k, dev)?;
                let e_neg_flat = gather_rows(unit_e, &neg_idx)?;
                let b_neg_flat = gather_rows(unit_b, &neg_idx)?;
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
                JointEmbedModel::score_negatives(&e_neg, anchor, &b_neg, anchor_bias)?
            }
        };
        let negs = std::slice::from_ref(&neg_score);
        let per_edge = match objective {
            NceObjective::Logistic => logistic_nce(pos_score, negs)?,
            NceObjective::Softmax => softmax_nce(pos_score, negs)?,
        };
        let per_edge_gb = to_anchors(per_edge)?.reshape((g, b))?;
        per_feature_per_level.push(per_edge_gb.mean(1)?); // [G]
    }
    Ok(CageLossOut {
        per_feature_level: Tensor::stack(&per_feature_per_level, 1)?,
        mean_abs_pair,
    })
}

/// The rows of a cage-cell step: each distinct cell pair the step names,
/// its embedding row `z` (`[U, D]`, the step's leaf) and its two cells.
pub struct EdgeRows<'a> {
    pub z: &'a Tensor,
    pub left: &'a [u32],
    pub right: &'a [u32],
}

/// One cage-cell chunk: its pair rows, one draw per `(feature, batch)`
/// naming pairs by their row in `rows` (see [`localize_edges`]), each
/// draw's feature and the block (batch) its negatives came from.
pub struct EdgeChunk<'a> {
    pub rows: EdgeRows<'a>,
    pub batches: Vec<EdgeChainBatch>,
    pub feature_ids: &'a [u32],
    pub row_blocks: &'a [u32],
}

/// cage-cell's per-(feature, level) NCE over cell pairs, `[G, 1]`: the
/// trained unit is the pair itself, `z_e` a row of its own.
///
/// ```text
/// s(e) = θ_g · z_e + b_u + b_v        e = (u, v)
/// ```
///
/// A positive is a feature-active pair, as in cage; its negatives are
/// other pairs of its batch (a cell pair that is not an edge has no row
/// to score), each scored by its own `z` and its own two cells' biases.
/// The same NCE and weights as [`cage_nce_loss_per_feature_level`]. A
/// draw's positives and negatives share its feature row, so each is one
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
        assert_eq!(eb.pos_edges.len(), b, "cage-cell loss: B mismatch");
        assert_eq!(eb.n_negatives, k, "cage-cell loss: K mismatch");
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

/// The negative scores of a cage-cell chunk as a table per block (batch):
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

/// The distinct cell pairs `batches` and `extra` name, ascending (global
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
    /// each a unit id. `None` when no draws are made or a
    /// block holds more than `max_width` distinct units, which is as soon
    /// as the table stops paying.
    fn new(
        row_blocks: &[u32],
        b: usize,
        k: usize,
        negs_per_level: &[Vec<u32>],
        max_width: usize,
    ) -> Option<Self> {
        if k == 0 {
            return None;
        }
        let (labels, block_of_row) = uniq_with_inverse(row_blocks);
        let n_blocks = labels.len();
        let block_of_row: Vec<usize> = block_of_row.into_iter().map(|b| b as usize).collect();

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
        // one block at a time, so the ranks are kept for the block being
        // visited only: a map of at most `max_width` units, never a table
        // over every unit (on cell pairs, millions).
        let mut rank_of: rustc_hash::FxHashMap<u32, u32> = Default::default();
        let mut visiting = u32::MAX;
        let mut units: Vec<Vec<u32>> = vec![Vec::new(); n_blocks];
        let mut ranks: Vec<Vec<u32>> = negs_per_level
            .iter()
            .map(|_| Vec::with_capacity(anchor_of_slot.len() * k))
            .collect();
        for (s, &a) in anchor_of_slot.iter().enumerate() {
            let blk = block_of_tile[s / tile];
            if blk != visiting {
                rank_of.clear();
                visiting = blk;
            }
            let block_units = &mut units[blk as usize];
            for (lvl_ranks, negs) in ranks.iter_mut().zip(negs_per_level) {
                if a == PAD {
                    lvl_ranks.extend(std::iter::repeat_n(0, k));
                    continue;
                }
                let a = a as usize;
                for &u in &negs[a * k..(a + 1) * k] {
                    let rank = *rank_of.entry(u).or_insert_with(|| {
                        block_units.push(u);
                        block_units.len() as u32 - 1
                    });
                    if block_units.len() > max_width {
                        return None;
                    }
                    lvl_ranks.push(rank);
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
    /// `[tiles * tile, width]`: `a · e_w + b_w + b_a` for the slot's anchor
    /// factor `a` (`θ_g ⊙ e_u` on cells and pseudobulks, `θ_g` on cell
    /// pairs) and bias `b_a`, unit rows `e_w` of `unit_e` and biases of
    /// `unit_b`: the same expression the gathered path forms per draw. Both
    /// biases ride in the product as two extra columns, `[a, 1, b_a] ·
    /// [e_w, b_w, 1]`, so the table is written once. Padding columns score a
    /// block's first unit and are never read.
    fn score_table(
        &self,
        unit_e: &Tensor,
        unit_b: &Tensor,
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
                &gather_rows(unit_e, &block_units)?,
                &gather_rows(unit_b, &block_units)?.unsqueeze(1)?,
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
