use super::{residency, set_rows, Residency, RowAdam};
use legume_numeric::candle::candle_core::{Device, Tensor, Var};
use legume_numeric::candle::candle_nn::{AdamW, Optimizer, ParamsAdamW};
use legume_numeric::candle::fast_index::gather_rows;

const D: usize = 3;

/// Deterministic values spread over `[-1, 1)`.
fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7919 + salt * 104_729) % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn rows_of(t: &Tensor, rows: &[u32]) -> Tensor {
    let idx = Tensor::from_slice(rows, rows.len(), t.device()).unwrap();
    gather_rows(t, &idx).unwrap()
}

fn params() -> ParamsAdamW {
    // weight decay on, so a row the step skips would move under the dense rule
    ParamsAdamW {
        lr: 0.05,
        weight_decay: 0.1,
        ..Default::default()
    }
}

/// Rows the lazy Adam steps on every step end exactly where candle's dense
/// AdamW puts them, bit for bit; a row stepped on some steps only moves
/// then; rows no step names keep their value and their moments, bit for bit.
#[test]
fn lazy_adam_matches_dense_adamw_on_the_rows_it_steps_and_leaves_the_rest() {
    let dev = Device::Cpu;
    let n_rows = 6;
    let init = Tensor::from_vec(values(n_rows * D, 1), (n_rows, D), &dev).unwrap();
    let mut lazy = RowAdam::new(values(n_rows * D, 1), (n_rows, D), &dev, params(), &dev).unwrap();
    let every = [1u32, 4];
    let dense = Var::from_tensor(&rows_of(&init, &every)).unwrap();
    let mut adam = AdamW::new(vec![dense.clone()], params()).unwrap();

    for step in 0..7 {
        let g =
            Tensor::from_vec(values(every.len() * D, 10 + step), (every.len(), D), &dev).unwrap();
        // the dense optimizer sees exactly `g` as its gradient
        let grads = (dense.as_tensor() * &g)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        adam.step(&grads).unwrap();
        // the leaf's values passed in, as training passes them
        if step % 3 == 0 {
            // row 2 joins this step
            let ids = [1u32, 2, 4];
            let extra = Tensor::from_vec(values(D, 50 + step), (1, D), &dev).unwrap();
            let rows = Tensor::cat(
                &[
                    &g.narrow(0, 0, 1).unwrap(),
                    &extra,
                    &g.narrow(0, 1, 1).unwrap(),
                ],
                0,
            )
            .unwrap();
            let current = lazy.rows(&ids).unwrap();
            lazy.step(&ids, &current, &rows).unwrap();
        } else {
            let current = lazy.rows(&every).unwrap();
            lazy.step(&every, &current, &g).unwrap();
        }
    }
    assert_eq!(
        bits(&rows_of(lazy.table(), &every)),
        bits(dense.as_tensor()),
        "the rows stepped every step left the dense AdamW path"
    );
    let untouched = [0u32, 3, 5];
    assert_eq!(
        bits(&rows_of(lazy.table(), &untouched)),
        bits(&rows_of(&init, &untouched)),
        "an untouched row moved"
    );
    let (m, v) = lazy.moments();
    for state in [m, v] {
        assert!(
            bits(&rows_of(&state, &untouched)).iter().all(|&b| b == 0),
            "an untouched row's moments moved"
        );
    }
    assert_ne!(
        bits(&rows_of(lazy.table(), &[2])),
        bits(&rows_of(&init, &[2])),
        "the row stepped on some steps did not move"
    );
}

/// A step's rows must be distinct and ascending, as a leaf lists them: a
/// repeated or unordered id is refused, and nothing moves.
#[test]
fn a_step_refuses_repeated_or_unordered_rows() {
    let dev = Device::Cpu;
    let n_rows = 4;
    let mut z = RowAdam::new(values(n_rows * D, 2), (n_rows, D), &dev, params(), &dev).unwrap();
    let before = bits(z.table());
    let g = Tensor::from_vec(values(2 * D, 3), (2, D), &dev).unwrap();
    for ids in [[3u32, 3], [3, 1]] {
        let current = z.rows(&ids).unwrap();
        assert!(z.step(&ids, &current, &g).is_err(), "{ids:?}");
    }
    assert_eq!(bits(z.table()), before);
}

/// `set_rows` writes the named rows in place and nothing else; a row past
/// the table is refused.
#[test]
fn set_rows_writes_only_the_named_rows() {
    let dev = Device::Cpu;
    let table = Tensor::zeros(
        (4, 2),
        legume_numeric::candle::candle_core::DType::F32,
        &dev,
    )
    .unwrap();
    let ids = Tensor::from_slice(&[2u32, 0], 2, &dev).unwrap();
    let src = Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &dev).unwrap();
    set_rows(&table, &ids, &src).unwrap();
    assert_eq!(
        table.to_vec2::<f32>().unwrap(),
        vec![vec![3., 4.], vec![0., 0.], vec![1., 2.], vec![0., 0.]]
    );
    let past = Tensor::from_slice(&[4u32], 1, &dev).unwrap();
    let row = Tensor::from_slice(&[1f32, 1.], (1, 2), &dev).unwrap();
    assert!(set_rows(&table, &past, &row).is_err());
}

/// A table stays on the device when it fits half the free fraction, and
/// goes to the host otherwise; a device that cannot say keeps it.
#[test]
fn a_table_too_big_for_the_device_goes_to_the_host() {
    assert_eq!(residency(100, None, 0.8), Residency::Device);
    assert_eq!(residency(400, Some(1000), 0.8), Residency::Device);
    assert_eq!(residency(401, Some(1000), 0.8), Residency::Host);
    // the fraction is clamped as --gpu-mem-fraction is
    assert_eq!(residency(474, Some(1000), 5.0), Residency::Device);
    assert_eq!(residency(476, Some(1000), 5.0), Residency::Host);
}
