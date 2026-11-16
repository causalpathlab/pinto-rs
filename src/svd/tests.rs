//! dsvd's pair channels: each cell is scaled to a common total before the
//! log, so depth alone moves neither channel and a run reads the same with
//! or without batches.

use super::fit::*;
use crate::test_support::make_test_graph;
use crate::util::cell_pairs::SrtCellPairs;
use crate::util::common::*;

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
    let mut sp = create_sparse_from_dmatrix(counts, None, None).unwrap();
    let names = |tag: &str, n: usize| -> Vec<Box<str>> {
        (0..n).map(|i| format!("{tag}{i}").into()).collect()
    };
    sp.register_row_names_vec(&names("GENE", counts.nrows()));
    sp.register_column_names_vec(&names("cell", counts.ncols()));
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
    // each cell alone, on its own total: what a pair of them shares
    let want_shared = Mat::from_columns(&[2.0 * log_normalised(&x).column(0)]);
    for delta in [None, Some(&delta)] {
        let (shared, diff, proj) = channels(&data, &[(0, 1)], delta, &difference_only_basis(g));
        let what = if delta.is_some() { "with" } else { "without" };
        if delta.is_none() {
            assert_close(&shared, &want_shared, what);
        }
        assert!(shared.iter().any(|&v| v > 1.0), "{what} delta: {shared}");
        assert!(diff.iter().all(|v| v.abs() < TOL), "{what} delta: {diff}");
        assert!(proj.iter().all(|v| v.abs() < TOL), "{what} delta: {proj}");
    }
}

/// A cell with nothing to scale stays all zero: no stored entry, stored
/// zeros, or the `0 / 0` a batch divide makes of stored zeros. Its
/// neighbour is untouched by it.
#[test]
fn a_cell_without_counts_stays_empty() {
    let mut y = nalgebra_sparse::CscMatrix::try_from_csc_data(
        3,
        4,
        vec![0, 0, 2, 4, 6],
        vec![0, 2, 0, 1, 1, 2],
        vec![0.0, 0.0, f32::NAN, f32::NAN, 1.0, 3.0],
    )
    .unwrap();
    log_normalise_cells(&mut y);
    assert_eq!(y.values()[..4], [0.0; 4]);
    let want = [
        (LIBRARY_SCALE * 0.25).ln_1p(),
        (LIBRARY_SCALE * 0.75).ln_1p(),
    ];
    for (got, want) in y.values()[4..].iter().zip(want) {
        assert!((got - want).abs() < TOL, "{got} vs {want}");
    }
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
fn log_normalised(x: &Mat) -> Mat {
    let mut out = x.clone();
    for mut col in out.column_iter_mut() {
        let total = col.sum();
        col.iter_mut()
            .for_each(|v| *v = (LIBRARY_SCALE * *v / total).ln_1p());
    }
    out
}

/// The divide the visitors use estimates a per-cell rate as it divides. A
/// per-cell constant is gone once the cell is rescaled, so the channels are
/// those of a plain division of each count by its batch effect.
#[test]
fn the_self_normalising_divide_reads_as_a_plain_divide_once_rescaled() {
    let x = counts();
    let g = x.nrows();
    let batch_of_cell = [0, 0, 1, 1];
    let delta = uneven_delta(g);
    let pairs = [(0, 1), (1, 2), (2, 3), (0, 3)];
    let basis = basis(g);

    let plain = log_normalised(&Mat::from_fn(g, x.ncols(), |i, j| {
        x[(i, j)] / delta[(i, batch_of_cell[j])]
    }));
    let mut want_shared = Mat::zeros(g, pairs.len());
    let mut want_diff = Mat::zeros(g, pairs.len());
    for (e, &(i, j)) in pairs.iter().enumerate() {
        want_shared.set_column(e, &(plain.column(i) + plain.column(j)));
        want_diff.set_column(e, &(plain.column(i) - plain.column(j)).abs());
    }
    let want_proj =
        basis.rows(0, g).transpose() * &want_shared + basis.rows(g, g).transpose() * &want_diff;

    let (shared, diff, proj) = channels(&data(&x, &batch_of_cell), &pairs, Some(&delta), &basis);
    assert_close(&shared, &want_shared, "shared");
    assert_close(&diff, &want_diff, "difference");
    assert_close(&proj, &want_proj, "projection");
}
