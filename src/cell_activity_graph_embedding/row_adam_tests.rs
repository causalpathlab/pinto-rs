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
        // the dense optimiser sees exactly `g` as its gradient
        let grads = (dense.as_tensor() * &g)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        adam.step(&grads).unwrap();
        if step % 3 == 0 {
            // row 2 joins this step, listed first
            let extra = Tensor::from_vec(values(D, 50 + step), (1, D), &dev).unwrap();
            let both = Tensor::cat(&[&extra, &g], 0).unwrap();
            lazy.step(&[2, 1, 4], &both).unwrap();
        } else {
            lazy.step(&every, &g).unwrap();
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
            bits(&rows_of(state, &untouched)).iter().all(|&b| b == 0),
            "an untouched row's moments moved"
        );
    }
    assert_ne!(
        bits(&rows_of(lazy.table(), &[2])),
        bits(&rows_of(&init, &[2])),
        "the row stepped on some steps did not move"
    );
}

/// A row named twice in one step is stepped once, by the sum of its
/// gradients, and counts one step: two runs, one with the duplicate and
/// one with the summed gradient, stay identical over the next step too.
#[test]
fn duplicate_ids_in_a_step_coalesce() {
    let dev = Device::Cpu;
    let n_rows = 4;
    let new = || RowAdam::new(values(n_rows * D, 2), (n_rows, D), &dev, params(), &dev).unwrap();
    let (mut dup, mut summed) = (new(), new());
    let (a, b, c) = (values(D, 3), values(D, 4), values(D, 5));
    let ac: Vec<f32> = a.iter().zip(&c).map(|(x, y)| x + y).collect();
    let g = |rows: Vec<f32>, n: usize| Tensor::from_vec(rows, (n, D), &dev).unwrap();
    dup.step(&[3, 1, 3], &g([a, b.clone(), c].concat(), 3))
        .unwrap();
    summed.step(&[1, 3], &g([b, ac].concat(), 2)).unwrap();
    assert_eq!(bits(dup.table()), bits(summed.table()));
    let next = g(values(2 * D, 6), 2);
    dup.step(&[1, 3], &next).unwrap();
    summed.step(&[1, 3], &next).unwrap();
    assert_eq!(
        bits(dup.table()),
        bits(summed.table()),
        "the duplicate counted as two steps"
    );
    let (dm, dv) = dup.moments();
    let (sm, sv) = summed.moments();
    assert_eq!((bits(dm), bits(dv)), (bits(sm), bits(sv)));
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
