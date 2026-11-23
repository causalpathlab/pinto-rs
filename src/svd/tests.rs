//! dsvd's pair channels: each cell is scaled to a common total before the
//! log, so depth alone moves neither channel and a run reads the same with
//! or without batches.

use super::fit::*;
use crate::test_support::make_test_graph;
use crate::util::cell_pairs::SrtCellPairs;
use crate::util::common::*;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIo};

const TOL: f32 = 1e-3;

/// Four cells over five features. Cells 0 and 1 share one composition at
/// different depths; cells 2 and 3 differ in composition, depth and support.
fn counts() -> Mat {
    Mat::from_column_slice(
        5,
        4,
        &[
            1.0, 2.0, 0.0, 4.0, 3.0, //
            3.0, 6.0, 0.0, 12.0, 9.0, //
            0.0, 5.0, 7.0, 1.0, 0.0, //
            20.0, 0.0, 2.0, 8.0, 40.0, //
        ],
    )
}

/// `counts` as one in-memory file, its cells in the batches `batch_of_cell`.
fn data(counts: &Mat, batch_of_cell: &[usize]) -> SparseIoVec {
    let sp = create_sparse_from_dmatrix(counts, None, None).unwrap();
    data_of(sp, counts.shape(), batch_of_cell)
}

/// The backend `sp` (features x cells of `shape`) as one in-memory file, its
/// cells in the batches `batch_of_cell`.
fn data_of(
    mut sp: Box<dyn SparseIo<IndexIter = Vec<usize>>>,
    shape: (usize, usize),
    batch_of_cell: &[usize],
) -> SparseIoVec {
    let names = |tag: &str, n: usize| -> Vec<Box<str>> {
        (0..n).map(|i| format!("{tag}{i}").into()).collect()
    };
    sp.register_row_names_vec(&names("GENE", shape.0));
    sp.register_column_names_vec(&names("cell", shape.1));
    sp.preload_columns().unwrap();
    let mut data = SparseIoVec::new();
    data.push(Arc::from(sp), None).unwrap();
    data.register_batch_membership(batch_of_cell);
    data
}

/// Both visitors over `pairs` of `data`, every pair its own sample: the
/// collapsed shared and difference channels (features x pairs), and each
/// pair's projection on `basis` (shared rows, then difference rows).
fn channels(
    data: &SparseIoVec,
    pairs: &[(usize, usize)],
    delta: Option<&Mat>,
    basis: &Mat,
) -> (Mat, Mat, Mat) {
    let n_cells = data.num_columns();
    let n_features = data.num_rows();
    let coords = Mat::from_fn(n_cells, 2, |i, j| (i * (j + 1)) as f32);
    let graph = make_test_graph(n_cells, pairs.to_vec());
    let cell_pairs = SrtCellPairs::with_graph(data, &coords, &graph, None, None);
    let n_pairs = cell_pairs.inner.num_pairs();
    assert_eq!(n_pairs, pairs.len());

    let pair_to_sample: Vec<usize> = (0..n_pairs).collect();
    let mut collapsed = PairDeltaCollapsedStat::new(n_features, n_pairs);
    cell_pairs
        .inner
        .visit_pairs_by_block(
            &fused_pair_delta_visitor,
            &FusedDeltaInput {
                batch_effect: delta,
                pair_fold: None,
                pair_to_sample: &pair_to_sample,
            },
            &mut collapsed,
            None,
        )
        .unwrap();

    let mut proj = Mat::zeros(basis.ncols(), n_pairs);
    cell_pairs
        .inner
        .visit_pairs_by_block(
            &nystrom_pair_delta_visitor,
            &NystromPairInput {
                basis_shared: basis.rows(0, n_features).clone_owned(),
                basis_diff: basis.rows(n_features, n_features).clone_owned(),
                batch_effect: delta.cloned(),
                pair_fold: None,
            },
            &mut proj,
            None,
        )
        .unwrap();

    (collapsed.shared_ds, collapsed.diff_ds, proj)
}

/// A basis of three components with no two rows alike.
fn basis(n_features: usize) -> Mat {
    Mat::from_fn(2 * n_features, 3, |i, k| {
        ((i + 1) * (k + 2)) as f32 * 0.01 - 0.1
    })
}

/// A basis reading the difference channel only, one component per feature.
fn difference_only_basis(n_features: usize) -> Mat {
    Mat::from_fn(2 * n_features, n_features, |i, k| {
        if i == n_features + k {
            1.0
        } else {
            0.0
        }
    })
}

/// Batch effects that differ by feature and by batch.
fn uneven_delta(n_features: usize) -> Mat {
    Mat::from_fn(n_features, 2, |g, b| 0.5 + (g + 2 * b) as f32 * 0.75)
}

fn assert_close(got: &Mat, want: &Mat, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}");
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() <= TOL * (1.0 + w.abs()), "{what}: {g} vs {w}");
    }
}

/// Two cells of one composition differ in depth only, so their pair has no
/// difference in any feature, in the collapsed statistics and in the
/// projection alike, without batch effects and with them.
#[test]
fn depth_alone_leaves_the_difference_channel_empty() {
    let x = counts();
    let g = x.nrows();
    let data = data(&x, &[0, 0, 1, 1]);
    let delta = uneven_delta(g);
    let no_effect = Mat::from_element(g, 2, 1.0);
    for delta in [None, Some(&delta)] {
        let (shared, diff, proj) = channels(&data, &[(0, 1)], delta, &difference_only_basis(g));
        let what = if delta.is_some() { "with" } else { "without" };
        // each cell alone, on its own total after a plain division by its
        // batch's effects: what a pair of them shares
        let plain = plainly_divided(&x, delta.unwrap_or(&no_effect), &[0, 0, 1, 1]);
        let want_shared = Mat::from_columns(&[2.0 * plain.column(0)]);
        assert_close(&shared, &want_shared, what);
        assert!(diff.iter().all(|v| v.abs() < TOL), "{what} delta: {diff}");
        assert!(proj.iter().all(|v| v.abs() < TOL), "{what} delta: {proj}");
    }
}

/// A cell with nothing to scale stays all zero: no stored entry, stored
/// zeros, or a total that is not a positive finite number (the `0 / 0` a
/// batch divide makes of stored zeros, or a count that is not finite). Its
/// neighbour is untouched by it.
#[test]
fn a_cell_without_counts_stays_empty() {
    let mut y = nalgebra_sparse::CscMatrix::try_from_csc_data(
        3,
        5,
        vec![0, 0, 2, 4, 6, 8],
        vec![0, 2, 0, 1, 1, 2, 0, 1],
        vec![0.0, 0.0, f32::NAN, f32::NAN, 1.0, 3.0, 2.0, f32::INFINITY],
    )
    .unwrap();
    log_normalize_cells(&mut y);
    assert_eq!(y.values()[..4], [0.0; 4]);
    assert_eq!(y.values()[6..], [0.0; 2]);
    let want = [
        (LIBRARY_SCALE * 0.25).ln_1p(),
        (LIBRARY_SCALE * 0.75).ln_1p(),
    ];
    for (got, want) in y.values()[4..6].iter().zip(want) {
        assert!((got - want).abs() < TOL, "{got} vs {want}");
    }
}

/// A cell whose stored counts are all zero goes through the batch divide
/// and both visitors as an empty cell: every channel is finite, and a pair
/// of it and a neighbour reads as the neighbour alone.
#[test]
fn a_cell_of_stored_zeros_is_empty_after_the_batch_divide() {
    let x = counts();
    let (g, n) = x.shape();
    let batch_of_cell = [0, 0, 1, 1];
    let hollow = 2;
    let mut triplets = Vec::new();
    for j in 0..n {
        for i in 0..g {
            if j == hollow {
                triplets.push((i as u64, j as u64, 0.0));
            } else if x[(i, j)] > 0.0 {
                triplets.push((i as u64, j as u64, x[(i, j)]));
            }
        }
    }
    let sp = create_sparse_from_triplets(&triplets, (g, n, triplets.len()), None, None).unwrap();
    let data = data_of(sp, (g, n), &batch_of_cell);
    let stored = data.read_columns_csc(std::iter::once(hollow)).unwrap();
    assert_eq!(stored.nnz(), g, "the reader keeps stored zeros");

    let delta = uneven_delta(g);
    let basis = basis(g);
    let pairs = [(0, hollow), (hollow, 3)];
    let (shared, diff, proj) = channels(&data, &pairs, Some(&delta), &basis);

    let plain = plainly_divided(&x, &delta, &batch_of_cell);
    let alone = Mat::from_columns(&[plain.column(0), plain.column(3)]);
    let want_proj = (basis.rows(0, g) + basis.rows(g, g)).transpose() * &alone;
    assert_close(&shared, &alone, "shared");
    assert_close(&diff, &alone, "difference");
    assert_close(&proj, &want_proj, "projection");
}

/// The same counts read as one batch (no batch effects) and as two batches
/// whose effects are all one give the same pair features.
#[test]
fn one_batch_and_several_batches_of_unit_effect_agree() {
    let x = counts();
    let g = x.nrows();
    let pairs = [(0, 1), (1, 2), (2, 3), (0, 3)];
    let basis = basis(g);

    let single = channels(&data(&x, &[0, 0, 0, 0]), &pairs, None, &basis);
    let ones = Mat::from_element(g, 2, 1.0);
    let multi = channels(&data(&x, &[0, 0, 1, 1]), &pairs, Some(&ones), &basis);

    assert!(single.1.iter().any(|&v| v > 0.1), "pairs differ somewhere");
    assert_close(&multi.0, &single.0, "shared");
    assert_close(&multi.1, &single.1, "difference");
    assert_close(&multi.2, &single.2, "projection");
}

/// `ln(1 + 1e4 x / Σ x)` of each cell of `x`, features x cells.
fn log_normalized(x: &Mat) -> Mat {
    let mut out = x.clone();
    for mut col in out.column_iter_mut() {
        let total = col.sum();
        col.iter_mut()
            .for_each(|v| *v = (LIBRARY_SCALE * *v / total).ln_1p());
    }
    out
}

/// Each count of `x` divided by the effect of its feature in its cell's
/// batch and nothing else, then each cell log-normalized.
fn plainly_divided(x: &Mat, delta: &Mat, batch_of_cell: &[usize]) -> Mat {
    log_normalized(&Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
        x[(i, j)] / delta[(i, batch_of_cell[j])]
    }))
}

/// The pair channels of features x cells `y`: the sum and the absolute
/// difference of each pair's two cells.
fn pair_channels(y: &Mat, pairs: &[(usize, usize)]) -> (Mat, Mat) {
    let mut shared = Mat::zeros(y.nrows(), pairs.len());
    let mut diff = Mat::zeros(y.nrows(), pairs.len());
    for (e, &(i, j)) in pairs.iter().enumerate() {
        shared.set_column(e, &(y.column(i) + y.column(j)));
        diff.set_column(e, &(y.column(i) - y.column(j)).abs());
    }
    (shared, diff)
}

/// The divide the visitors use estimates a per-cell rate as it divides. A
/// per-cell constant is gone once the cell is rescaled, so the channels are
/// those of a plain division of each count by its batch effect.
#[test]
fn the_self_normalizing_divide_reads_as_a_plain_divide_once_rescaled() {
    let x = counts();
    let g = x.nrows();
    let batch_of_cell = [0, 0, 1, 1];
    let delta = uneven_delta(g);
    let pairs = [(0, 1), (1, 2), (2, 3), (0, 3)];
    let basis = basis(g);

    let plain = plainly_divided(&x, &delta, &batch_of_cell);
    let (want_shared, want_diff) = pair_channels(&plain, &pairs);
    let want_proj =
        basis.rows(0, g).transpose() * &want_shared + basis.rows(g, g).transpose() * &want_diff;

    let (shared, diff, proj) = channels(&data(&x, &batch_of_cell), &pairs, Some(&delta), &basis);
    assert_close(&shared, &want_shared, "shared");
    assert_close(&diff, &want_diff, "difference");
    assert_close(&proj, &want_proj, "projection");
}

/// The plain-divide reading holds for positive batch effects only. The
/// divide leaves a count whose effect is not positive as it is, divided by
/// neither the effect nor the cell's rate, so the rate no longer cancels:
/// the cell is the divide's own output, rescaled, and two cells of one
/// composition at different depths then differ.
#[test]
fn a_count_without_a_positive_effect_keeps_the_cell_rate() {
    let x = counts();
    let g = x.nrows();
    let batch_of_cell = [0, 0, 1, 1];
    let mut delta = uneven_delta(g);
    delta[(1, 0)] = 0.0;
    let pairs = [(0, 1), (1, 2), (2, 3), (0, 3)];
    let basis = basis(g);

    // the divide as documented: rate = Σ counts / Σ effects over the cell's
    // stored rows, and a count is divided by effect x rate where the effect
    // is positive
    let mut divided = x.clone();
    for (j, mut cell) in divided.column_iter_mut().enumerate() {
        let d = delta.column(batch_of_cell[j]);
        let stored = |i: &usize| x[(*i, j)] > 0.0;
        let rate = cell.sum() / (0..g).filter(stored).map(|i| d[i]).sum::<f32>();
        for i in (0..g).filter(|&i| d[i] > 0.0) {
            cell[i] /= d[i] * rate;
        }
    }
    let (want_shared, want_diff) = pair_channels(&log_normalized(&divided), &pairs);
    let want_proj =
        basis.rows(0, g).transpose() * &want_shared + basis.rows(g, g).transpose() * &want_diff;

    let (shared, diff, proj) = channels(&data(&x, &batch_of_cell), &pairs, Some(&delta), &basis);
    assert_close(&shared, &want_shared, "shared");
    assert_close(&diff, &want_diff, "difference");
    assert_close(&proj, &want_proj, "projection");
    assert!(
        diff.column(0).iter().any(|&v| v > 0.1),
        "depth enters the pair of one composition: {diff}"
    );
}
