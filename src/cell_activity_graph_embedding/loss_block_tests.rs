//! Reading each negative's score out of a table of every anchor against
//! every unit of its block must be the SAME loss as gathering each drawn
//! negative's embedding row: the table entry is the very score the gather
//! forms. So both paths must agree on the loss and on every parameter
//! gradient, for both objectives, with repeated draws, with rows arriving
//! out of block order, and with a block wider than one tile.

use super::loss::{cage_nce_loss_scored, BlockScoring};
use graph_embedding_util::loss::{NceObjective, UnitChainBatch};
use graph_embedding_util::model::{JointEmbedModel, ModelArgs, ModelInit};
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::candle::candle_nn::VarMap;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const N_UNITS: usize = 30;
const N_FEATURES: usize = 5;
const DIM: usize = 4;
const B: usize = 3;
const K: usize = 6;
const L: usize = 2;
/// Units `10 * b .. 10 * b + 10` form block `b`. Rows are listed out of block
/// order, so the per-block regrouping must put the result back in place.
const ROW_BLOCKS: [u32; 8] = [2, 0, 1, 0, 2, 1, 0, 2];
const BLOCK_SIZE: u32 = 10;

fn model(varmap: &VarMap, dev: &Device) -> JointEmbedModel {
    // Non-zero biases, so a dropped or doubled bias term shows.
    let mut r = SmallRng::seed_from_u64(5);
    let b_feat: Vec<f32> = (0..N_FEATURES).map(|_| r.random_range(-0.5..0.5)).collect();
    let b_cell: Vec<f32> = (0..N_UNITS).map(|_| r.random_range(-0.5..0.5)).collect();
    JointEmbedModel::new_with_init(
        ModelArgs {
            n_features: N_FEATURES,
            n_cells: N_UNITS,
            embedding_dim: DIM,
            seed: 13,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &b_feat,
            b_cell: &b_cell,
        },
        varmap,
        dev,
    )
    .unwrap()
}

/// One row per entry of `row_blocks`; every unit a row names lies in its
/// block, and the first two negatives of each anchor repeat, so counts of 2
/// and more occur at every level.
fn rows(row_blocks: &[u32]) -> (Vec<UnitChainBatch>, Vec<u32>) {
    let mut r = SmallRng::seed_from_u64(99);
    let mut out = Vec::new();
    let mut features = Vec::new();
    for &blk in row_blocks {
        let base = blk * BLOCK_SIZE;
        let unit = |r: &mut SmallRng| base + r.random_range(0..BLOCK_SIZE);
        let left_units: Vec<u32> = (0..B).map(|_| unit(&mut r)).collect();
        let right_units: Vec<u32> = (0..B).map(|_| unit(&mut r)).collect();
        let per_level_neg = (0..L)
            .map(|_| {
                let mut negs = Vec::with_capacity(B * K);
                for _ in 0..B {
                    let first = unit(&mut r);
                    negs.push(first);
                    negs.push(first);
                    negs.extend((2..K).map(|_| unit(&mut r)));
                }
                negs
            })
            .collect();
        out.push(UnitChainBatch {
            left_units,
            right_units,
            per_level_neg,
            n_negatives: K,
        });
        features.push(r.random_range(0..N_FEATURES as u32));
    }
    (out, features)
}

/// The `[G, L]` loss and, after a backward through a weighted sum of it,
/// every parameter's gradient.
fn loss_and_grads(
    row_blocks: &[u32],
    labels: &[u32],
    scoring: BlockScoring,
    objective: NceObjective,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let m = model(&varmap, &dev);
    let (cbs, features) = rows(row_blocks);
    let out = cage_nce_loss_scored(&m, cbs, &features, labels, objective, scoring, &dev)
        .unwrap()
        .per_feature_level;
    // Distinct weights per entry, so a row landing in the wrong place moves
    // the gradient as well as the loss.
    let n = row_blocks.len() * L;
    let w: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 / n as f32).collect();
    let w = Tensor::from_vec(w, (row_blocks.len(), L), &dev).unwrap();
    let grads = (&out * &w).unwrap().sum_all().unwrap().backward().unwrap();
    let mut vars = varmap.all_vars();
    vars.sort_by_key(|v| v.as_tensor().elem_count());
    let grads = vars
        .iter()
        .map(|v| match grads.get(v.as_tensor()) {
            Some(g) => g.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            None => Vec::new(),
        })
        .collect();
    (out.flatten_all().unwrap().to_vec1().unwrap(), grads)
}

/// Equal up to float rounding: the two paths sum the same terms in a
/// different order, so the bound scales with the largest entry compared.
fn assert_close(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let scale = a.iter().fold(1f32, |s, x| s.max(x.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-5 * scale,
            "{what}[{i}]: table {x} vs gathered {y}"
        );
    }
}

/// Rows drawn from the units of `row_blocks` and labelled `labels`, scored
/// by `scoring`, give the gathered loss and gradients.
fn matches_gathered(
    row_blocks: &[u32],
    labels: &[u32],
    scoring: BlockScoring,
    objective: NceObjective,
) {
    let (loss, grads) = loss_and_grads(row_blocks, labels, scoring, objective);
    let (gath_loss, gath_grads) =
        loss_and_grads(row_blocks, labels, BlockScoring::Gathered, objective);
    assert_close("loss", &loss, &gath_loss);
    assert!(
        grads.iter().any(|g| !g.is_empty()),
        "no parameter received a gradient"
    );
    for (i, (d, g)) in grads.iter().zip(&gath_grads).enumerate() {
        assert_close(&format!("grad {i}"), d, g);
    }
}

#[test]
fn softmax_block_scoring_is_the_gathered_loss() {
    matches_gathered(
        &ROW_BLOCKS,
        &ROW_BLOCKS,
        BlockScoring::Table,
        NceObjective::Softmax,
    );
}

#[test]
fn logistic_block_scoring_is_the_gathered_loss() {
    matches_gathered(
        &ROW_BLOCKS,
        &ROW_BLOCKS,
        BlockScoring::Table,
        NceObjective::Logistic,
    );
}

/// The automatic choice must be one of the two, so it agrees as well: here
/// the table, on blocks of `BLOCK_SIZE` units.
#[test]
fn automatic_block_scoring_is_the_gathered_loss() {
    matches_gathered(
        &ROW_BLOCKS,
        &ROW_BLOCKS,
        BlockScoring::Auto,
        NceObjective::Softmax,
    );
}

/// One label over all `N_UNITS` units is wider than the `2 · L · K` units the
/// table is chosen for, so the automatic choice stops building the table
/// partway and falls back to the gather.
#[test]
fn a_wide_block_falls_back_to_the_gathered_loss() {
    const { assert!(N_UNITS > 2 * L * K) };
    let labels = [0; ROW_BLOCKS.len()];
    matches_gathered(
        &ROW_BLOCKS,
        &labels,
        BlockScoring::Auto,
        NceObjective::Softmax,
    );
}

/// Regrouping rows by block must hand each caller row back its own loss:
/// every row of the chunk equals that row scored alone.
#[test]
fn each_row_keeps_its_own_loss() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let m = model(&varmap, &dev);
    let score = |cbs: Vec<UnitChainBatch>, features: &[u32], blocks: &[u32]| {
        cage_nce_loss_scored(
            &m,
            cbs,
            features,
            blocks,
            NceObjective::Softmax,
            BlockScoring::Table,
            &dev,
        )
        .unwrap()
        .per_feature_level
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
    };
    let (cbs, features) = rows(&ROW_BLOCKS);
    let chunk = score(cbs, &features, &ROW_BLOCKS);
    let (cbs, _) = rows(&ROW_BLOCKS);
    for (i, cb) in cbs.into_iter().enumerate() {
        let alone = score(vec![cb], &features[i..=i], &ROW_BLOCKS[i..=i]);
        assert_close(&format!("row {i}"), &chunk[i * L..(i + 1) * L], &alone);
    }
}

/// Blocks of unequal size, interleaved. The tile is the mean anchors per
/// block (900 / 3 = 300), so block 0's 540 anchors fill one tile and spill
/// into a second, and every block's last tile is part-filled.
#[test]
fn blocks_spanning_tiles_are_the_gathered_loss() {
    let row_blocks: Vec<u32> = (0..300).map(|i| [0, 2, 0, 1, 0][i % 5]).collect();
    matches_gathered(
        &row_blocks,
        &row_blocks,
        BlockScoring::Table,
        NceObjective::Softmax,
    );
}

/// Blocks need not be disjoint: two labels whose rows draw from the same
/// units each get their own ranks.
#[test]
fn blocks_sharing_units_are_the_gathered_loss() {
    let units = [0, 0, 0, 0, 1, 1];
    let labels = [0, 1, 0, 1, 2, 3];
    matches_gathered(&units, &labels, BlockScoring::Table, NceObjective::Softmax);
}
