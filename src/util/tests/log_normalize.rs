//! The one depth normalization: each unit scaled to [`LIBRARY_SCALE`] in
//! total, then `ln(1 + x)`, in its scalar, sparse-slice and CSC forms.

use crate::util::common::{log_cp10k, log_normalize, normalize_log, LIBRARY_SCALE};
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;

/// features × units; unit 2 is empty, unit 3 is unit 0 at seven times the
/// depth.
fn counts() -> DMatrix<f32> {
    DMatrix::from_row_slice(
        4,
        4,
        &[
            1.0, 0.0, 0.0, 7.0, //
            3.0, 2.0, 0.0, 21.0, //
            0.0, 5.0, 0.0, 0.0, //
            6.0, 1.0, 0.0, 42.0,
        ],
    )
}

/// One unit as a sparse `(feature, count)` list.
fn sparse_unit(x: &DMatrix<f32>, unit: usize) -> Vec<(u32, f32)> {
    x.column(unit)
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > 0.0)
        .map(|(g, &v)| (g as u32, v))
        .collect()
}

fn normalized(x: &DMatrix<f32>) -> DMatrix<f32> {
    let mut csc = CscMatrix::from(x);
    normalize_log(&mut csc);
    DMatrix::from(&csc)
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5 * (1.0 + a.abs().max(b.abs()))
}

/// Multiplying a unit's counts by a positive constant changes nothing: the
/// output depends on the unit's composition only.
#[test]
fn depth_does_not_change_the_output() {
    let x = counts();
    let base = normalized(&x);
    for scale in [0.01f32, 0.5, 3.0, 1e3] {
        let got = normalized(&(&x * scale));
        for (a, b) in got.iter().zip(base.iter()) {
            assert!(close(*a, *b), "x{scale}: {a} vs {b}");
        }
        let mut p = sparse_unit(&(&x * scale), 1);
        log_normalize(&mut p);
        let mut q = sparse_unit(&x, 1);
        log_normalize(&mut q);
        for (a, b) in p.iter().zip(&q) {
            assert!(close(a.1, b.1), "x{scale}: {a:?} vs {b:?}");
        }
    }
    // unit 3 is unit 0 at another depth
    for g in 0..4 {
        assert!(close(base[(g, 0)], base[(g, 3)]), "feature {g}");
    }
    // and the output is not the plain log1p of the counts
    assert!((base[(1, 0)] - 3f32.ln_1p()).abs() > 1.0);
}

/// Back on the counts scale a non-empty unit totals [`LIBRARY_SCALE`]; an
/// empty unit stays empty.
#[test]
fn a_unit_totals_the_library_scale_and_an_empty_one_stays_empty() {
    let y = normalized(&counts());
    for unit in [0, 1, 3] {
        let total: f32 = y.column(unit).iter().map(|v| v.exp_m1()).sum();
        assert!(
            (total - LIBRARY_SCALE).abs() < 1e-3 * LIBRARY_SCALE,
            "unit {unit}: {total}"
        );
    }
    assert!(y.column(2).iter().all(|&v| v == 0.0));
}

/// An empty list, zero counts and a zero total give zeros; a NaN total is
/// bad input, not an empty unit, and stays NaN.
#[test]
fn empty_zero_and_nan_inputs() {
    let mut none: Vec<(u32, f32)> = Vec::new();
    log_normalize(&mut none);
    assert!(none.is_empty());
    let mut zeros = vec![(0u32, 0f32), (3, 0.0)];
    log_normalize(&mut zeros);
    assert_eq!(zeros, vec![(0, 0.0), (3, 0.0)]);
    assert_eq!(log_cp10k(0f32, 0.0), 0.0);
    assert_eq!(log_cp10k(2f64, 0.0), 0.0);
    assert!(log_cp10k(2f32, f32::NAN).is_nan());
    let mut bad = vec![(0u32, 1f32), (1, f32::NAN)];
    log_normalize(&mut bad);
    assert!(bad.iter().all(|&(_, v)| v.is_nan()));
}

/// The slice and CSC forms are the scalar form applied with the unit's own
/// total, value for value.
#[test]
fn the_slice_csc_and_scalar_forms_agree() {
    let x = counts();
    let y = normalized(&x);
    for unit in 0..x.ncols() {
        let total: f32 = x.column(unit).sum();
        let mut p = sparse_unit(&x, unit);
        log_normalize(&mut p);
        for &(g, v) in &p {
            let g = g as usize;
            let scalar = log_cp10k(x[(g, unit)], total);
            assert_eq!(v, scalar, "slice, unit {unit} feature {g}");
            assert_eq!(y[(g, unit)], scalar, "csc, unit {unit} feature {g}");
        }
        assert_eq!(p.len(), x.column(unit).iter().filter(|&&v| v > 0.0).count());
    }
}

/// The arithmetic order is `(s * x) / total` in every form, to the bit, and
/// it is not `(x / total) * s`, which rounds differently.
#[test]
fn the_arithmetic_order_is_scale_times_count_over_total() {
    let mut differs = false;
    for total in (7u16..2000).step_by(13) {
        let total = f32::from(total);
        for x in 1u16..=40 {
            let x = f32::from(x);
            let pinned = (LIBRARY_SCALE * x / total).ln_1p();
            assert_eq!(log_cp10k(x, total).to_bits(), pinned.to_bits());
            let wide = (1e4 * f64::from(x) / f64::from(total)).ln_1p();
            assert_eq!(
                log_cp10k(f64::from(x), f64::from(total)).to_bits(),
                wide.to_bits()
            );
            differs |= (x / total * LIBRARY_SCALE).ln_1p().to_bits() != pinned.to_bits();
        }
    }
    assert!(differs, "the fixture does not tell the two orders apart");
}

/// The normalization is local to a unit, so a pass over blocks of columns is
/// the pass over the whole matrix.
#[test]
fn a_block_by_block_pass_equals_the_whole_matrix() {
    let x = counts();
    let whole = normalized(&x);
    for block in [1, 2, 3] {
        let mut lb = 0;
        while lb < x.ncols() {
            let ub = (lb + block).min(x.ncols());
            let part = normalized(&x.columns(lb, ub - lb).into_owned());
            assert_eq!(part, whole.columns(lb, ub - lb).into_owned(), "{lb}..{ub}");
            lb = ub;
        }
    }
}
