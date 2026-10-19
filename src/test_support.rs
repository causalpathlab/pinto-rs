use crate::util::knn_graph::KnnGraph;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// Build a simple undirected `KnnGraph` from an edge list, with unit distances.
/// Used by tests across modules.
pub(crate) fn make_test_graph(n_nodes: usize, edges: Vec<(usize, usize)>) -> KnnGraph {
    let distances = vec![1.0; edges.len()];

    let mut coo = CooMatrix::new(n_nodes, n_nodes);
    for &(i, j) in &edges {
        coo.push(i, j, 1.0f32);
        coo.push(j, i, 1.0f32);
    }
    let adjacency = CscMatrix::from(&coo);

    KnnGraph {
        adjacency,
        edges,
        distances,
        n_nodes,
    }
}

/// A zarr file `{tag}.zarr` in `dir` of `rows` by `n_cols` columns named
/// `{tag}_0`, `{tag}_1`, …, holding `triplets` (row, column, value).
pub(crate) fn write_zarr(
    dir: &std::path::Path,
    tag: &str,
    rows: &[&str],
    n_cols: usize,
    triplets: &[(u64, u64, f32)],
) -> Box<str> {
    use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
    let path = dir.join(format!("{tag}.zarr"));
    let mut backend = create_sparse_from_triplets(
        triplets,
        (rows.len(), n_cols, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    let rows: Vec<Box<str>> = rows.iter().map(|&s| s.into()).collect();
    let cols: Vec<Box<str>> = (0..n_cols).map(|j| format!("{tag}_{j}").into()).collect();
    backend.register_row_names_vec(&rows);
    backend.register_column_names_vec(&cols);
    path.to_str().unwrap().into()
}
