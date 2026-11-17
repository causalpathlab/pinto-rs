//! Row helpers shared by the pair paths: deduplicating row ids, and
//! scaling a table's rows to unit length.

use crate::util::common::Mat;

/// The distinct values of `ids`, ascending, and each entry's index in
/// them, so `uniq[inverse[i]] == ids[i]`.
pub fn uniq_with_inverse(ids: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut uniq = ids.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    let inverse = ids
        .iter()
        .map(|id| uniq.binary_search(id).expect("listed") as u32)
        .collect();
    (uniq, inverse)
}

/// Every row of `m` scaled to unit L2 norm (a norm below 1e-8 divides by
/// 1e-8), in place.
pub fn normalize_rows(m: &mut Mat) {
    let mut sq = vec![0f32; m.nrows()];
    for col in m.column_iter() {
        for (s, &x) in sq.iter_mut().zip(col.iter()) {
            *s += x * x;
        }
    }
    let inv: Vec<f32> = sq.iter().map(|&s| 1.0 / s.sqrt().max(1e-8)).collect();
    for mut col in m.column_iter_mut() {
        for (x, &w) in col.iter_mut().zip(&inv) {
            *x *= w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_rows, uniq_with_inverse};
    use crate::util::common::Mat;

    #[test]
    fn uniq_with_inverse_maps_back() {
        let ids = [5u32, 2, 5, 9, 2];
        let (uniq, inv) = uniq_with_inverse(&ids);
        assert_eq!(uniq, [2, 5, 9]);
        assert!(ids.iter().zip(&inv).all(|(&id, &i)| uniq[i as usize] == id));
    }

    #[test]
    fn normalized_rows_have_unit_length() {
        let mut m = Mat::from_row_slice(3, 2, &[3.0, 4.0, 0.0, 0.0, -1.0, 0.0]);
        normalize_rows(&mut m);
        assert_eq!(m.row(0).iter().copied().collect::<Vec<_>>(), [0.6, 0.8]);
        assert_eq!(m.row(1).iter().copied().collect::<Vec<_>>(), [0.0, 0.0]);
        assert_eq!(m.row(2).iter().copied().collect::<Vec<_>>(), [-1.0, 0.0]);
    }
}
