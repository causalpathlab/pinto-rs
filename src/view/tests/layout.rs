use super::*;

fn lay_out(table: &Mat, method: Method, seed: u64) -> anyhow::Result<Mat> {
    let names = (0..table.nrows()).map(|i| i.to_string().into()).collect();
    Coarse::new(names, table, seed)?.lay_out(method, seed)
}

/// Three tight groups of `per` rows each, far apart in four dimensions.
fn groups(per: usize) -> Mat {
    let n = 3 * per;
    Mat::from_fn(n, 4, |i, j| {
        let g = i / per;
        let centre = if j == g { 10.0 } else { 0.0 };
        // a small, fixed spread inside each group
        centre + ((i * 7 + j * 3) % 11) as f32 * 0.01
    })
}

/// Every layout places every row, in order, at a finite point.
#[test]
fn every_layout_places_every_row_at_a_finite_point() {
    let table = groups(20);
    for method in Method::ALL {
        let xy = lay_out(&table, method, 7).unwrap();
        assert_eq!(xy.shape(), (table.nrows(), 2), "{}", method.name());
        assert!(xy.iter().all(|v| v.is_finite()), "{}", method.name());
    }
}

/// Past [`PSEUDOBULKS`] rows, a layout of the pseudobulks still places
/// every row.
#[test]
fn a_layout_past_its_pseudobulks_places_every_row() {
    let per = PSEUDOBULKS / 3 + 50;
    let table = groups(per);
    let xy = lay_out(&table, Method::Stress, 7).unwrap();
    assert_eq!(xy.nrows(), table.nrows());
    assert!(xy.iter().all(|v| v.is_finite()));
}

/// The same seed lays the cells out the same way.
#[test]
fn a_layout_is_the_same_from_the_same_seed() {
    let table = groups(20);
    assert_eq!(
        lay_out(&table, Method::Stress, 3).unwrap(),
        lay_out(&table, Method::Stress, 3).unwrap()
    );
}

/// `next` steps through every layout, in order, and ends after the last.
#[test]
fn next_steps_through_every_layout() {
    let mut seen = vec![Method::ALL[0]];
    while let Some(m) = seen.last().unwrap().next() {
        seen.push(m);
    }
    assert_eq!(seen, Method::ALL);
}
