//! cage's phase 1 trains a row per super edge (pseudobulk pair), and the
//! pair encoder trained on that table places the cell pairs: the table is
//! written with each row's two pseudobulks, no per-pseudobulk table is, and
//! a cell's embedding is the mean of its pairs' latents.

use super::cell_level::fit;
use crate::test_support::two_sections;
use crate::util::common::Mat;
use legume_numeric::matrix::traits::IoOps;

#[test]
fn cage_trains_a_row_per_super_edge_and_a_cell_averages_its_pairs() {
    let dir = tempfile::tempdir().unwrap();
    let (data, coords) = two_sections(dir.path(), 8);
    let out = fit(dir.path(), &data, &coords, "pb", &[]);

    // the table: a row per pseudobulk pair, keyed "P-Q" by two distinct
    // pseudobulks of pb_bias, each pair once, at the embedding width
    let table = Mat::from_parquet(&crate::util::parquet_io::pb_pair_embedding_path(&out)).unwrap();
    let n_pb = Mat::from_parquet(&format!("{out}.pb_bias.parquet"))
        .unwrap()
        .mat
        .nrows();
    assert!(table.mat.nrows() > 0);
    assert_eq!(table.mat.ncols(), 4, "the embedding width");
    let mut seen = std::collections::HashSet::new();
    for key in &table.rows {
        let (p, q) = key.split_once('-').expect("a P-Q key");
        let (p, q): (usize, usize) = (p.parse().unwrap(), q.parse().unwrap());
        assert!(p < n_pb && q < n_pb && p != q, "pair {key} of {n_pb}");
        assert!(seen.insert((p.min(q), p.max(q))), "pair {key} twice");
    }
    assert!(table.mat.iter().all(|x| x.is_finite()));
    assert!(!std::path::Path::new(&format!("{out}.pb_embedding.parquet")).exists());

    // a cell's embedding is the mean of the (unit) latents of the cell pairs
    // it is in, in coord_pairs order
    let pairs =
        crate::lr_activity::io::read_pair_table(&format!("{out}.coord_pairs.parquet"), false)
            .unwrap();
    let latent = Mat::from_parquet(&format!("{out}.latent.parquet"))
        .unwrap()
        .mat;
    let cells = Mat::from_parquet(&format!("{out}.cell_embedding.parquet")).unwrap();
    assert_eq!(latent.nrows(), pairs.len());
    let index: std::collections::HashMap<&str, usize> = cells
        .rows
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_ref(), i))
        .collect();
    let d = latent.ncols();
    let mut sum = vec![vec![0f32; d]; cells.rows.len()];
    let mut count = vec![0usize; cells.rows.len()];
    for (k, p) in pairs.iter().enumerate() {
        for c in [&p.left_cell, &p.right_cell] {
            let i = index[c.as_ref()];
            for (s, &x) in sum[i].iter_mut().zip(latent.row(k).iter()) {
                *s += x;
            }
            count[i] += 1;
        }
    }
    for (i, name) in cells.rows.iter().enumerate() {
        for (j, &total) in sum[i].iter().enumerate() {
            let want = if count[i] > 0 {
                total / count[i] as f32
            } else {
                0.0
            };
            let got = cells.mat[(i, j)];
            assert!(
                (got - want).abs() < 1e-4,
                "cell {name} dim {j}: {got} vs {want}"
            );
        }
    }
}
