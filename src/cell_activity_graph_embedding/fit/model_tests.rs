//! The trained unit table's starting value.

use super::{build_model, ModelInputs};
use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
use crate::test_support::{numbered, sparse_vec};
use crate::util::feature_axis::FeatureAxis;
use clap::Parser;
use graph_embedding_util::model::{JointEmbedModel, ModelArgs, ModelInit};
use legume_numeric::candle::candle_core::Device;
use legume_numeric::candle::candle_nn::VarMap;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    cage: CellActivityGraphEmbeddingArgs,
}

const N_FEATURES: usize = 6;
const N_CELLS: usize = 8;
const N_PB: usize = 4;
const DIM: usize = 3;

/// cage's pseudobulk table as [`build_model`] starts it.
fn starting_table(args: &CellActivityGraphEmbeddingArgs) -> Vec<Vec<f32>> {
    let dir = tempfile::tempdir().unwrap();
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for c in 0..N_CELLS {
        for g in 0..N_FEATURES {
            let x = ((g * 5 + c * 3) % 7) as f32;
            if x > 0.0 {
                triplets.push((g as u64, c as u64, x));
            }
        }
    }
    let data = sparse_vec(
        dir.path(),
        "start",
        &numbered("GENE", N_FEATURES),
        &numbered("c", N_CELLS),
        &triplets,
    );
    let names = data.row_names().unwrap();
    let labels: Vec<usize> = (0..N_CELLS).map(|c| c / 2).collect();
    let built = build_model(ModelInputs {
        args,
        profile_labels: &labels,
        data_vec: &data,
        feature_axis: &FeatureAxis::resolve(&names).unwrap(),
        feature_kind: &args.feature_name_mode.resolve_kind(&names),
        module_cfg: None,
        lora_spec: None,
        n_pb: N_PB,
        embedding_dim: DIM,
        varmap: &VarMap::new(),
        dev: &Device::Cpu,
    })
    .unwrap();
    built.model.e_cell.to_vec2::<f32>().unwrap()
}

/// The pseudobulk table starts from the seeded random draw of its shape,
/// the one cage-cell's cell table starts from, not from a fit to the counts.
#[test]
fn the_pseudobulk_table_starts_from_the_seeded_random_draw() {
    let args = Cli::try_parse_from(["x", "data.zarr", "-o", "out"])
        .unwrap()
        .cage;
    let drawn = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features: N_FEATURES,
            n_cells: N_PB,
            embedding_dim: DIM,
            seed: args.common.seed,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &[0.0; N_FEATURES],
            b_cell: &[0.0; N_PB],
        },
        &VarMap::new(),
        &Device::Cpu,
    )
    .unwrap()
    .e_cell
    .to_vec2::<f32>()
    .unwrap();
    assert_eq!(drawn.len(), N_PB);
    assert!(drawn.iter().flatten().any(|&v| v != 0.0));
    assert_eq!(starting_table(&args), drawn);
}
