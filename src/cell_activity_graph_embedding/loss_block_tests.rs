//! Reading each negative's score out of a table of every draw against every
//! pair of its block must be the SAME loss as gathering each drawn negative's
//! row: the table entry is the very score the gather forms. So both paths must
//! agree on the loss and on every parameter gradient, for both objectives,
//! with repeated draws.

use super::loss::BlockScoring;
use graph_embedding_util::loss::NceObjective;
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

/// cage-cell's pair loss and, after a backward through a weighted sum of
/// it, every parameter's gradient and the pair rows' (the leaf's). Pairs
/// `10 * b .. 10 * b + 10` form block `b`, their cells drawn from that
/// block's; the first two negatives of each positive repeat.
fn pair_loss_and_grads(
    scoring: BlockScoring,
    objective: NceObjective,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    use super::feature_chain_sampler::EdgeChainBatch;
    use super::loss::{cage_cell_edge_nce_loss_scored, EdgeChunk, EdgeRows};
    use legume_numeric::candle::candle_core::Var;
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let m = model(&varmap, &dev);
    let n_pairs = 3 * BLOCK_SIZE as usize;
    let mut r = SmallRng::seed_from_u64(7);
    let z: Vec<f32> = (0..n_pairs * DIM)
        .map(|_| r.random_range(-1.0..1.0))
        .collect();
    let z = Var::from_tensor(&Tensor::from_vec(z, (n_pairs, DIM), &dev).unwrap()).unwrap();
    let (left, right): (Vec<u32>, Vec<u32>) = (0..n_pairs as u32)
        .map(|e| {
            let base = (e / BLOCK_SIZE) * BLOCK_SIZE;
            (
                base + r.random_range(0..BLOCK_SIZE),
                base + r.random_range(0..BLOCK_SIZE),
            )
        })
        .unzip();
    let mut batches = Vec::new();
    let mut features = Vec::new();
    for &blk in &ROW_BLOCKS {
        let pair = |r: &mut SmallRng| blk * BLOCK_SIZE + r.random_range(0..BLOCK_SIZE);
        let pos_edges: Vec<u32> = (0..B).map(|_| pair(&mut r)).collect();
        let mut neg_edges = Vec::with_capacity(B * K);
        for _ in 0..B {
            let first = pair(&mut r);
            neg_edges.extend([first, first]);
            neg_edges.extend((2..K).map(|_| pair(&mut r)));
        }
        batches.push(EdgeChainBatch {
            pos_edges,
            neg_edges,
            n_negatives: K,
        });
        features.push(r.random_range(0..N_FEATURES as u32));
    }
    let out = cage_cell_edge_nce_loss_scored(
        &m,
        EdgeChunk {
            rows: EdgeRows {
                z: z.as_tensor(),
                left: &left,
                right: &right,
            },
            batches,
            feature_ids: &features,
            row_blocks: &ROW_BLOCKS,
        },
        objective,
        scoring,
        &dev,
    )
    .unwrap()
    .per_feature_level;
    assert_eq!(out.dims(), [ROW_BLOCKS.len(), 1]);
    let n = ROW_BLOCKS.len();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 / n as f32).collect();
    let w = Tensor::from_vec(w, (n, 1), &dev).unwrap();
    let grads = (&out * &w).unwrap().sum_all().unwrap().backward().unwrap();
    let mut vars = varmap.all_vars();
    vars.sort_by_key(|v| v.as_tensor().elem_count());
    vars.push(z);
    let grads = vars
        .iter()
        .map(|v| match grads.get(v.as_tensor()) {
            Some(g) => g.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            None => Vec::new(),
        })
        .collect();
    (out.flatten_all().unwrap().to_vec1().unwrap(), grads)
}

/// On cell pairs, the block table over each batch's negative pairs is the
/// gathered loss too, and the pair rows receive the same gradient.
#[test]
fn pair_block_scoring_is_the_gathered_loss() {
    for objective in [NceObjective::Logistic, NceObjective::Softmax] {
        let (loss, grads) = pair_loss_and_grads(BlockScoring::Table, objective);
        let (gath_loss, gath_grads) = pair_loss_and_grads(BlockScoring::Gathered, objective);
        assert_close("pair loss", &loss, &gath_loss);
        assert!(
            grads.last().is_some_and(|g| g.iter().any(|&x| x != 0.0)),
            "the pair rows received no gradient"
        );
        for (i, (d, g)) in grads.iter().zip(&gath_grads).enumerate() {
            assert_close(&format!("pair grad {i}"), d, g);
        }
    }
}
