//! The encoder on planted data: it is symmetric end to end, a cell does not
//! move with the block size, a saved encoder reproduces itself, and empty
//! rows stay at the origin. And what it trains on: cage's pseudobulk rows
//! are their cells' sums, and the thinning keeps that many counts.

use super::fixture::*;
use crate::cell_activity_graph_embedding::pair_projection::encoder::training_set::{
    thin_stats, thin_to, Scratch, SparseRow,
};
use crate::cell_activity_graph_embedding::pair_projection::encoder::{
    CellRow, CorpusStats, PairEncoder, PairEncoderSpec, Placed, TrainingSet,
};
use crate::cell_activity_graph_embedding::pair_projection::{
    project_pairs, PairDictionary, PairProjectionArgs, PairSolver, SuperEdgeTable, TrainOn,
};
use crate::test_support::{numbered, sparse_vec};
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use legume_numeric::candle::candle_core::Device;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N_TRAIN_CELLS: usize = 150;
const RIDGE: f32 = 1.0;

/// A planted cell: its exact expected counts and the latent they came from.
type PlantedCell = (Vec<(u32, f32)>, Vec<f32>);

/// Cells whose latents sit near one of three centres, with their exact
/// expected counts, as `(global profile, planted latent)`.
fn planted_cells(n: usize, seed: u64) -> Vec<PlantedCell> {
    let e = dictionary_matrix();
    let (b, _) = abundances();
    let centres = [
        [0.7f32, -0.3, 0.2, 0.0],
        [-0.4, 0.5, 0.0, 0.3],
        [0.1, 0.1, -0.6, -0.4],
    ];
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|c| {
            let centre = centres[c % 3];
            let theta: Vec<f32> = centre
                .iter()
                .map(|&v| v + 0.15 * (rng.random::<f32>() - 0.5))
                .collect();
            let beta = 0.5 * (rng.random::<f32>() - 0.5);
            (counts_from(&e, &b, &theta, beta), theta)
        })
        .collect()
}

fn dictionary() -> PairDictionary {
    let e = dictionary_matrix();
    let (_, totals) = abundances();
    PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary")
}

fn corpus_of(dict: &PairDictionary, cells: &[PlantedCell]) -> Vec<CellRow> {
    cells
        .iter()
        .map(|(profile, _)| CellRow::from_profile(dict, profile))
        .collect()
}

fn random_pairs(n_cells: usize, n_pairs: usize, seed: u64) -> Vec<(u32, u32)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_pairs)
        .map(|_| {
            let u = rng.random_range(0..n_cells as u32);
            let mut v = rng.random_range(0..n_cells as u32);
            while v == u {
                v = rng.random_range(0..n_cells as u32);
            }
            (u.min(v), u.max(v))
        })
        .collect()
}

/// Small steps: `steps` optimizer steps, rounded up to whole passes over
/// the pairs, so on this fixture the step's size sets the cost.
fn spec(steps: usize) -> PairEncoderSpec {
    PairEncoderSpec {
        trunk_width: 32,
        n_experts: 4,
        steps,
        batch: 16,
        ridge: RIDGE,
    }
}

/// A trained encoder on the planted cells, with the training pairs.
struct Trained {
    dict: PairDictionary,
    corpus: Vec<CellRow>,
    stats: CorpusStats,
    edges: Vec<(u32, u32)>,
    enc: PairEncoder,
}

/// Trained once for every test that reads it; the encoder is never mutated
/// after training.
fn trained() -> &'static Trained {
    static TRAINED: std::sync::OnceLock<Trained> = std::sync::OnceLock::new();
    TRAINED.get_or_init(|| {
        let dict = dictionary();
        let cells = planted_cells(N_TRAIN_CELLS, 1);
        let corpus = corpus_of(&dict, &cells);
        let edges = random_pairs(N_TRAIN_CELLS, 1500, 2);
        let train_spec = spec(1);
        let set = TrainingSet::cells(corpus.len(), &edges);
        let moments = set.moments(DIM, None, &corpus);
        let enc = PairEncoder::build(&dict, moments, &train_spec, 7, &Device::Cpu).unwrap();
        let stats = enc.corpus_stats(&corpus).unwrap();
        let placed = Placed {
            corpus: &corpus,
            edges: &edges,
            stats: Some(&stats),
        };
        enc.train(&set, placed, None, &train_spec, 7).unwrap();
        Trained {
            dict,
            corpus,
            stats,
            edges,
            enc,
        }
    })
}

#[test]
fn pair_code_is_symmetric_end_to_end() {
    let Trained {
        corpus,
        stats,
        edges,
        enc,
        ..
    } = trained();
    let flipped: Vec<(u32, u32)> = edges.iter().map(|&(u, v)| (v, u)).collect();
    let a = enc.encode_all(corpus, stats, edges, 100, 64).unwrap();
    let b = enc.encode_all(corpus, stats, &flipped, 100, 64).unwrap();
    assert_eq!(a.latent.as_slice(), b.latent.as_slice());
    assert_eq!(a.bias, b.bias);
}

/// The cell block size moves neither a pair nor, through its pairs, a cell.
#[test]
fn blocks_do_not_move_a_cell() {
    let Trained {
        corpus,
        stats,
        edges,
        enc,
        ..
    } = trained();
    let whole = enc
        .encode_all(corpus, stats, edges, 64, corpus.len())
        .unwrap();
    let thirds = enc
        .encode_all(corpus, stats, edges, 64, corpus.len() / 3)
        .unwrap();
    for (a, b, what) in [
        (&whole.latent, &thirds.latent, "pair"),
        (&whole.cells.latent, &thirds.cells.latent, "cell"),
    ] {
        for r in 0..a.nrows() {
            let x: Vec<f32> = a.row(r).iter().copied().collect();
            let y: Vec<f32> = b.row(r).iter().copied().collect();
            if x.iter().any(|&v| v != 0.0) {
                assert!(
                    cosine(&x, &y) > 0.999,
                    "{what} {r} moved with the block size"
                );
            }
        }
    }
}

#[test]
fn empty_endpoints_land_on_the_origin() {
    let Trained {
        dict,
        corpus,
        edges,
        enc,
        ..
    } = trained();
    // Two cells with no counts at all.
    let mut corpus = corpus.clone();
    corpus.push(CellRow::from_profile(dict, &[]));
    corpus.push(CellRow::from_profile(dict, &[]));
    let n = corpus.len() as u32;
    let probe = vec![(n - 2, n - 1), edges[0]];
    let stats = enc.corpus_stats(&corpus).unwrap();
    let out = enc.encode_all(&corpus, &stats, &probe, 64, 64).unwrap();
    assert!(out.latent.row(0).iter().all(|&v| v == 0.0));
    assert_eq!(out.bias[0], 0.0);
    assert!(out.latent.row(1).iter().any(|&v| v != 0.0));
    for c in [n - 2, n - 1] {
        assert!(out.cells.latent.row(c as usize).iter().all(|&v| v == 0.0));
    }
}

#[test]
fn save_load_round_trip_is_byte_identical() {
    let Trained {
        dict,
        corpus,
        stats,
        edges,
        enc,
    } = trained();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("enc.safetensors");
    let path = path.to_str().unwrap();
    enc.save(path).unwrap();
    let back = PairEncoder::load(dict, path, &Device::Cpu).unwrap();
    assert_eq!(back.trunk_width(), 32);
    assert_eq!(back.n_experts(), 4);
    let a = enc.encode_all(corpus, stats, edges, 100, 64).unwrap();
    let b = back.encode_all(corpus, stats, edges, 100, 64).unwrap();
    assert_eq!(a.latent.as_slice(), b.latent.as_slice());
    assert_eq!(a.cells.latent.as_slice(), b.cells.latent.as_slice());
    assert_eq!(a.bias, b.bias);
}

////////////////////////////////////////
// Through `project_pairs`, from disk //
////////////////////////////////////////

/// A sparse matrix of the planted cells, one row per feature.
fn planted_data(dir: &tempfile::TempDir, cells: &[PlantedCell]) -> anyhow::Result<SparseIoVec> {
    let triplets: Vec<(u64, u64, f32)> = cells
        .iter()
        .enumerate()
        .flat_map(|(c, (profile, _))| profile.iter().map(move |&(g, n)| (g as u64, c as u64, n)))
        .collect();
    let rows = numbered("G", N_FEATURES);
    Ok(sparse_vec(
        dir.path(),
        "planted",
        &rows,
        &numbered("c", cells.len()),
        &triplets,
    ))
}

#[test]
fn the_saved_encoder_reproduces_the_runs_pairs_and_cells() {
    let dir = tempfile::tempdir().unwrap();
    let cells = planted_cells(60, 5);
    let data = planted_data(&dir, &cells).unwrap();
    let e = dictionary_matrix();
    let (_, totals) = abundances();
    let rows: Vec<Box<str>> = (0..N_FEATURES).map(|g| format!("G{g}").into()).collect();
    let axis = FeatureAxis::resolve_or_identity(&rows).unwrap();
    let edges = random_pairs(60, 300, 3);
    let saved = dir.path().join("run.pair_encoder.safetensors");
    let saved = saved.to_str().unwrap();
    let spec = spec(1);
    let cell_to_pb: Vec<usize> = (0..60).map(|c| c % N_PB).collect();
    let pb = pb_edges();
    let table = Table::new(&pb, &edges);
    let fitted = project_pairs(
        &data,
        &edges,
        &e,
        None,
        &PairProjectionArgs {
            solver: PairSolver::TrainEncoder {
                spec: &spec,
                on: TrainOn {
                    cell_to_pb: &cell_to_pb,
                    n_pb: N_PB,
                    edges: &pb,
                    table: table.view(),
                },
                dev: &Device::Cpu,
                save_to: saved,
            },
            seed: 11,
            pair_block: 64,
            eval_features: None,
            score_pairs: true,
        },
        &axis,
        &totals,
    )
    .unwrap();
    assert_eq!(fitted.cells.latent.nrows(), 60);
    assert_eq!(fitted.scores.len(), edges.len());
    assert!(std::path::Path::new(saved).is_file());

    let loaded = project_pairs(
        &data,
        &edges,
        &e,
        None,
        &PairProjectionArgs {
            solver: PairSolver::LoadEncoder {
                path: saved,
                dev: &Device::Cpu,
            },
            seed: 11,
            pair_block: 64,
            eval_features: None,
            score_pairs: false,
        },
        &axis,
        &totals,
    )
    .unwrap();
    assert_eq!(fitted.latent.as_slice(), loaded.latent.as_slice());
    assert_eq!(
        fitted.cells.latent.as_slice(),
        loaded.cells.latent.as_slice()
    );
    assert!(loaded.scores.is_empty());
}

/////////////////////////////////////////
// What the encoder trains on under cage //
/////////////////////////////////////////

/// Pseudobulk `c % N_PB` for cell `c`, and pseudobulk pairs over them:
/// every pair of the first seven, so the eighth is in none.
const N_PB: usize = 8;

fn pb_edges() -> Vec<(u32, u32)> {
    (0..7u32)
        .flat_map(|u| (u + 1..7).map(move |v| (u, v)))
        .collect()
}

/// A super-edge table for `pb` (row `e` = `[e + 1, -(e as f32), 0.5, 2]`,
/// not unit) and each cell pair's super edge: the one at its position
/// modulo `pb`'s length, none for every fifth.
struct Table {
    rows: Vec<f32>,
    of: Vec<Option<usize>>,
    trained: Vec<bool>,
}

impl Table {
    fn new(pb: &[(u32, u32)], cell_edges: &[(u32, u32)]) -> Self {
        let rows = (0..pb.len())
            .flat_map(|e| [e as f32 + 1.0, -(e as f32), 0.5, 2.0])
            .collect();
        let of = (0..cell_edges.len())
            .map(|k| (k % 5 != 4).then_some(k % pb.len()))
            .collect();
        let trained = vec![true; pb.len()];
        Self { rows, of, trained }
    }

    fn view(&self) -> SuperEdgeTable<'_> {
        SuperEdgeTable {
            rows: &self.rows,
            d: DIM,
            of_cell_pair: &self.of,
            trained: &self.trained,
        }
    }

    /// Row `e`, at unit length.
    fn unit(&self, e: usize) -> Vec<f32> {
        let r = &self.rows[e * DIM..(e + 1) * DIM];
        let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        r.iter().map(|x| x / n).collect()
    }
}

/// Planted cells, their counts scaled as a batch divisor would, so they are
/// fractional.
fn divided_cells(dict: &PairDictionary, n: usize) -> Vec<CellRow> {
    planted_cells(n, 9)
        .iter()
        .map(|(profile, _)| {
            let divided: Vec<(u32, f32)> = profile.iter().map(|&(g, x)| (g, x / 1.3)).collect();
            CellRow::from_profile(dict, &divided)
        })
        .collect()
}

#[test]
fn cage_trains_on_pseudobulk_rows_summed_from_their_cells() {
    let dict = dictionary();
    let cells = divided_cells(&dict, 40);
    let cell_to_pb: Vec<usize> = (0..cells.len()).map(|c| c % N_PB).collect();
    let cell_edges = random_pairs(cells.len(), 120, 4);
    let pb = pb_edges();
    let table = Table::new(&pb, &cell_edges);
    let set = TrainingSet::pseudobulks(
        &dict,
        &cells,
        &cell_to_pb,
        N_PB,
        &pb,
        &cell_edges,
        &table.view(),
    )
    .unwrap();
    // The training rows are the pseudobulks and the pairs theirs, not the cells'.
    assert_eq!(set.n_units, N_PB);
    assert_eq!(set.edges, &pb[..]);
    let rows = &set.thinning.as_ref().expect("thinned").rows;
    assert_eq!(rows.len(), N_PB);
    for (p, row) in rows.iter().enumerate() {
        let mut want: std::collections::BTreeMap<u32, f32> = Default::default();
        for c in (0..cells.len()).filter(|&c| cell_to_pb[c] == p) {
            for (&g, &n) in cells[c].features.iter().zip(&cells[c].counts) {
                *want.entry(g).or_default() += n;
            }
        }
        assert_eq!(row.features, want.keys().copied().collect::<Vec<_>>());
        for (&n, (_, &w)) in row.counts.iter().zip(&want) {
            assert!(
                (n - w).abs() <= 1e-4 * w.max(1.0),
                "pseudobulk {p}: {n} vs {w}"
            );
        }
    }

    // Training reads the pseudobulk rows and is scored on the cell pairs;
    // an unthinned set must be the placed cells.
    let spec = spec(1);
    let first = set.draw(3, 0);
    let moments = set.moments(DIM, first.as_ref(), &cells);
    let enc = PairEncoder::build(&dict, moments, &spec, 3, &Device::Cpu).unwrap();
    let placed = Placed {
        corpus: &cells,
        edges: &cell_edges,
        stats: None,
    };
    enc.train(&set, placed, first, &spec, 3).unwrap();
    let stats = enc.corpus_stats(&cells).unwrap();
    let wrong = TrainingSet {
        n_units: N_PB,
        edges: &cell_edges,
        visited: Vec::new(),
        thinning: None,
        targets: None,
    };
    let placed = Placed {
        corpus: &cells,
        edges: &cell_edges,
        stats: Some(&stats),
    };
    assert!(enc.train(&wrong, placed, None, &spec, 3).is_err());
}

#[test]
fn too_few_pseudobulk_pairs_train_on_the_cell_pairs() {
    let dict = dictionary();
    let cells = divided_cells(&dict, 20);
    let cell_to_pb: Vec<usize> = (0..cells.len()).map(|c| c % N_PB).collect();
    let cell_edges = random_pairs(cells.len(), 60, 5);
    let few = [(0, 1), (1, 2)];
    let table = Table::new(&few, &cell_edges);
    let set = TrainingSet::pseudobulks(
        &dict,
        &cells,
        &cell_to_pb,
        N_PB,
        &few,
        &cell_edges,
        &table.view(),
    )
    .unwrap();
    assert!(set.thinning.is_none());
    assert_eq!(set.n_units, cells.len());
    assert_eq!(set.edges, &cell_edges[..]);
    // each cell pair's target is its super edge's row at unit length; one
    // with no super edge has weight 0
    let targets = set.targets.as_ref().expect("targets");
    let ids: Vec<usize> = (0..cell_edges.len()).collect();
    let (rows, weight) = targets.gather(&ids);
    for k in ids {
        let got = &rows[k * DIM..(k + 1) * DIM];
        match table.of[k] {
            Some(e) => {
                assert_eq!(weight[k], 1.0, "cell pair {k}");
                assert_eq!(got, &table.unit(e)[..], "cell pair {k}");
            }
            None => assert_eq!((weight[k], got), (0.0, &[0.0; DIM][..]), "cell pair {k}"),
        }
    }
}

/// On enough pseudobulk pairs each one's target is its own row of the
/// table, at unit length; a row phase 1 never trained (it kept its random
/// start) gets weight 0.
#[test]
fn each_pseudobulk_pair_targets_its_own_row() {
    let dict = dictionary();
    let cells = divided_cells(&dict, 40);
    let cell_to_pb: Vec<usize> = (0..cells.len()).map(|c| c % N_PB).collect();
    let cell_edges = random_pairs(cells.len(), 120, 4);
    let pb = pb_edges();
    let mut table = Table::new(&pb, &cell_edges);
    table.trained[3] = false;
    let set = TrainingSet::pseudobulks(
        &dict,
        &cells,
        &cell_to_pb,
        N_PB,
        &pb,
        &cell_edges,
        &table.view(),
    )
    .unwrap();
    let all: Vec<usize> = (0..pb.len()).collect();
    let (_, weight) = set.targets.as_ref().expect("targets").gather(&all);
    for e in all {
        assert_eq!(weight[e], if e == 3 { 0.0 } else { 1.0 }, "pair {e}");
    }
    let ids: Vec<usize> = (0..pb.len()).rev().filter(|&e| e != 3).collect();
    let (rows, _) = set.targets.as_ref().expect("targets").gather(&ids);
    for (i, &e) in ids.iter().enumerate() {
        assert_eq!(
            &rows[i * DIM..(i + 1) * DIM],
            &table.unit(e)[..],
            "pair {e}"
        );
    }
}

#[test]
fn a_thinned_epoch_is_at_cell_pair_depth_with_even_endpoints() {
    let dict = dictionary();
    // Whole counts, so every cell pair's pooled depth is a whole number;
    // pseudobulks 0 and 1 hold two cells (10 counts), the rest one (5).
    let cells: Vec<CellRow> = (0..10u32)
        .map(|c| CellRow::from_profile(&dict, &[(c, 3.0), (c + 20, 2.0)]))
        .collect();
    let cell_to_pb: Vec<usize> = (0..cells.len()).map(|c| c % N_PB).collect();
    let cell_edges = vec![(0, 1), (2, 3), (4, 5)]; // each pools 10 counts
    let pb = pb_edges();
    let table = Table::new(&pb, &cell_edges);
    let set = TrainingSet::pseudobulks(
        &dict,
        &cells,
        &cell_to_pb,
        N_PB,
        &pb,
        &cell_edges,
        &table.view(),
    )
    .unwrap();
    let t = set.draw(17, 0).unwrap();
    // Each endpoint at half the drawn depth, whatever its own size; only
    // the seven visited pseudobulks have their own row.
    assert_eq!(t.n_rows(), 2 * pb.len() + 7);
    assert_eq!(set.visited, (0..7).collect::<Vec<u32>>());
    for i in 0..pb.len() {
        let (a, b) = set.pair_rows(i);
        assert_eq!((t.row(a as usize).1, t.row(b as usize).1), (5.0, 5.0));
    }
    for c in 0..7 {
        assert_eq!(t.row(set.self_row(c) as usize).1, 5.0);
    }
    assert_eq!(t.packed, set.draw(17, 0).unwrap().packed);
    assert_ne!(t.packed, set.draw(17, 1).unwrap().packed);
}

fn sparse(r: &CellRow) -> SparseRow {
    SparseRow {
        features: r.features.clone(),
        counts: r.counts.clone(),
    }
}

#[test]
fn thinning_keeps_the_drawn_depth_sparse_and_within_the_counts() {
    let dict = dictionary();
    let u = sparse(&CellRow::from_profile(
        &dict,
        &[(1, 40.0), (5, 7.0), (9, 1.0), (30, 120.0)],
    ));
    let thin = |target: u64, seed: u64| {
        let mut rng = StdRng::seed_from_u64(seed);
        thin_to(&u, target, &mut rng)
    };
    for target in [0u64, 1, 17, 100, 167] {
        let (f, n) = thin(target, 5);
        assert_eq!(n.iter().sum::<f32>(), target as f32);
        assert!(n.iter().all(|&x| x > 0.0), "a zero is stored");
        for (g, x) in f.iter().zip(&n) {
            let at = u
                .features
                .iter()
                .position(|h| h == g)
                .expect("a kept feature");
            assert!(*x <= u.counts[at]);
        }
        assert_eq!((f, n), thin(target, 5));
    }
    // At or beyond the row's own depth it comes back whole.
    assert_eq!(thin(173, 5), (u.features.clone(), u.counts.clone()));

    // The statistic written straight from the kept units is the thinned
    // row's own.
    for target in [17u64, 160, 1000] {
        let mut out = vec![0f32; DIM + 2];
        let mut rng = rand::rngs::SmallRng::seed_from_u64(9);
        let mut want_rng = rand::rngs::SmallRng::seed_from_u64(9);
        let (f, n) = thin_to(&u, target, &mut want_rng);
        thin_stats(
            &dict,
            &u,
            target,
            &mut rng,
            &mut Scratch::default(),
            &mut out,
        );
        let row = CellRow::from_local(&dict, f, n);
        assert_eq!(&out[..DIM], &row.sums[..]);
        assert_eq!((out[DIM], out[DIM + 1]), (row.total, row.offset));
    }

    // Fractional counts become whole units whether or not the row is
    // thinned, never above the count rounded up.
    let f = sparse(&CellRow::from_profile(
        &dict,
        &[(2, 0.4), (3, 5.5), (4, 12.25)],
    ));
    for target in [6u64, 1000] {
        let mut rng = StdRng::seed_from_u64(1);
        let (g, x) = thin_to(&f, target, &mut rng);
        for (g, x) in g.iter().zip(&x) {
            let at = f.features.iter().position(|h| h == g).unwrap();
            assert_eq!(x.fract(), 0.0);
            assert!(*x <= f.counts[at].ceil());
        }
    }
}

#[test]
fn cage_projection_trains_on_pseudobulks_and_places_every_cell_pair() {
    let dir = tempfile::tempdir().unwrap();
    let cells = planted_cells(30, 6);
    let data = planted_data(&dir, &cells).unwrap();
    let e = dictionary_matrix();
    let (_, totals) = abundances();
    let rows: Vec<Box<str>> = (0..N_FEATURES).map(|g| format!("G{g}").into()).collect();
    let axis = FeatureAxis::resolve_or_identity(&rows).unwrap();
    let edges = random_pairs(30, 90, 8);
    let cell_to_pb: Vec<usize> = (0..30).map(|c| c % N_PB).collect();
    let pb = pb_edges();
    let table = Table::new(&pb, &edges);
    let saved = dir.path().join("pb.pair_encoder.safetensors");
    let spec = spec(1);
    let out = project_pairs(
        &data,
        &edges,
        &e,
        None,
        &PairProjectionArgs {
            solver: PairSolver::TrainEncoder {
                spec: &spec,
                on: TrainOn {
                    cell_to_pb: &cell_to_pb,
                    n_pb: N_PB,
                    edges: &pb,
                    table: table.view(),
                },
                dev: &Device::Cpu,
                save_to: saved.to_str().unwrap(),
            },
            seed: 2,
            pair_block: 64,
            eval_features: None,
            score_pairs: false,
        },
        &axis,
        &totals,
    )
    .unwrap();
    assert_eq!(out.latent.nrows(), edges.len());
    assert_eq!(out.cells.latent.nrows(), 30);
}

/// The mean cosine between each pair's code and its target row.
fn mean_cosine(latent: &Mat, targets: &[f32]) -> f32 {
    let n = latent.nrows();
    (0..n)
        .map(|k| {
            let z: Vec<f32> = latent.row(k).iter().copied().collect();
            let t = &targets[k * DIM..(k + 1) * DIM];
            let dot: f32 = z.iter().zip(t).map(|(a, b)| a * b).sum();
            dot / z.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12)
        })
        .sum::<f32>()
        / n as f32
}

/// Given phase 1's rows as targets, training moves the encoder's codes
/// toward them: the codes' mean cosine to the targets rises. The targets
/// are each pair's planted latents summed, at unit length.
#[test]
fn training_on_targets_moves_the_codes_toward_them() {
    use crate::cell_activity_graph_embedding::pair_projection::encoder::training_set::EdgeTargets;
    let dict = dictionary();
    let cells = planted_cells(N_TRAIN_CELLS, 1);
    let corpus = corpus_of(&dict, &cells);
    let edges = random_pairs(N_TRAIN_CELLS, 600, 3);
    let mut targets = Vec::with_capacity(edges.len() * DIM);
    for &(u, v) in &edges {
        let t: Vec<f32> = (0..DIM)
            .map(|j| cells[u as usize].1[j] + cells[v as usize].1[j])
            .collect();
        let n = t.iter().map(|x| x * x).sum::<f32>().sqrt();
        targets.extend(t.iter().map(|x| x / n));
    }
    let mut set = TrainingSet::cells(corpus.len(), &edges);
    set.targets = Some(EdgeTargets {
        rows: targets.clone(),
        weight: vec![1.0; edges.len()],
        d: DIM,
    });
    let train_spec = spec(64);
    let moments = set.moments(DIM, None, &corpus);
    let enc = PairEncoder::build(&dict, moments, &train_spec, 5, &Device::Cpu).unwrap();
    let stats = enc.corpus_stats(&corpus).unwrap();
    let before = mean_cosine(
        &enc.encode_all(&corpus, &stats, &edges, 100, 64)
            .unwrap()
            .latent,
        &targets,
    );
    let placed = Placed {
        corpus: &corpus,
        edges: &edges,
        stats: Some(&stats),
    };
    enc.train(&set, placed, None, &train_spec, 5).unwrap();
    let after = mean_cosine(
        &enc.encode_all(&corpus, &stats, &edges, 100, 64)
            .unwrap()
            .latent,
        &targets,
    );
    assert!(after > before, "mean cosine {before} -> {after}");
}

/// A cell is the mean of its pairs' latents at unit length, each pair once,
/// and a cell in no pair is zero.
#[test]
fn a_cell_is_the_mean_of_its_unit_pair_latents() {
    use crate::cell_activity_graph_embedding::pair_projection::mean_of_pairs;
    let latent = Mat::from_row_slice(2, 2, &[3.0, 4.0, 0.0, 2.0]);
    let cells = mean_of_pairs(&latent, &[(0, 1), (1, 2)], 4);
    let row = |c: usize| vec![cells[(c, 0)], cells[(c, 1)]];
    assert_eq!(row(0), [0.6, 0.8]);
    assert!(
        (row(1)[0] - 0.3).abs() < 1e-6 && (row(1)[1] - 0.9).abs() < 1e-6,
        "{:?}",
        row(1)
    );
    assert_eq!(row(2), [0.0, 1.0]);
    assert_eq!(row(3), [0.0, 0.0], "a cell in no pair");
}
