use legume_numeric::matrix::knn_graph::KnnGraph;
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

/// Points in the plane, and edges between them by index.
pub(crate) type PlaneGraph = (Vec<(f32, f32)>, Vec<(usize, usize)>);

/// A `side × side` grid's coordinates, cell `r * side + c` at `(c, r)`,
/// and its rook edges (each once).
pub(crate) fn rook_grid(side: usize) -> PlaneGraph {
    let xy = (0..side * side)
        .map(|c| ((c % side) as f32, (c / side) as f32))
        .collect();
    let mut e = Vec::new();
    for r in 0..side {
        for c in 0..side {
            let i = r * side + c;
            if c + 1 < side {
                e.push((i, i + 1));
            }
            if r + 1 < side {
                e.push((i, i + side));
            }
        }
    }
    (xy, e)
}

/// Owned copies of `names`.
pub(crate) fn boxed(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|&s| s.into()).collect()
}

/// `{prefix}0`, `{prefix}1`, …, `n` of them.
pub(crate) fn numbered(prefix: &str, n: usize) -> Vec<Box<str>> {
    (0..n).map(|i| format!("{prefix}{i}").into()).collect()
}

/// A zarr file `{tag}.zarr` in `dir` with rows `rows` and columns `cols`,
/// holding `triplets` (row, column, value), and its path.
fn zarr_backend(
    dir: &std::path::Path,
    tag: &str,
    rows: &[Box<str>],
    cols: &[Box<str>],
    triplets: &[(u64, u64, f32)],
) -> (
    Box<dyn data_beans::sparse_io::SparseIo<IndexIter = Vec<usize>>>,
    Box<str>,
) {
    use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
    let path = dir.join(format!("{tag}.zarr"));
    let mut backend = create_sparse_from_triplets(
        triplets,
        (rows.len(), cols.len(), triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )
    .unwrap();
    backend.register_row_names_vec(rows);
    backend.register_column_names_vec(cols);
    (backend, path.to_str().unwrap().into())
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
    let cols = numbered(&format!("{tag}_"), n_cols);
    zarr_backend(dir, tag, &boxed(rows), &cols, triplets).1
}

/// [`write_zarr`]'s file with columns `cols`, opened as the only file of a
/// [`SparseIoVec`](data_beans::sparse_io_vector::SparseIoVec).
pub(crate) fn sparse_vec(
    dir: &std::path::Path,
    tag: &str,
    rows: &[Box<str>],
    cols: &[Box<str>],
    triplets: &[(u64, u64, f32)],
) -> data_beans::sparse_io_vector::SparseIoVec {
    let (backend, _) = zarr_backend(dir, tag, rows, cols, triplets);
    let mut v = data_beans::sparse_io_vector::SparseIoVec::new();
    v.push(std::sync::Arc::from(backend), None).unwrap();
    v
}

/// Two sections of a `side × side` grid, the left and right halves
/// expressing different features, and the second section's every third feature
/// four times the first's (a batch effect); returns the data and
/// coordinate files.
pub(crate) fn two_sections(dir: &std::path::Path, side: usize) -> (String, String) {
    let genes: Vec<String> = (0..12).map(|g| format!("GENE{g}")).collect();
    let genes: Vec<&str> = genes.iter().map(String::as_str).collect();
    let (mut data, mut coords) = (Vec::new(), Vec::new());
    for tag in ["s1", "s2"] {
        let n = side * side;
        let mut triplets = Vec::new();
        let mut csv = String::from("cell,x,y\n");
        for c in 0..n {
            let (r, col) = (c / side, c % side);
            let left = col < side / 2;
            for g in 0..12u64 {
                let marker = if left { g < 6 } else { g >= 6 };
                let v = if marker {
                    4.0 + ((c + g as usize) % 3) as f32
                } else {
                    ((c * 7 + g as usize) % 2) as f32
                };
                let v = if tag == "s2" && g % 3 == 0 {
                    4.0 * v
                } else {
                    v
                };
                if v > 0.0 {
                    triplets.push((g, c as u64, v));
                }
            }
            csv.push_str(&format!("{tag}_{c},{col},{r}\n"));
        }
        data.push(write_zarr(dir, tag, &genes, n, &triplets).to_string());
        let path = dir.join(format!("{tag}.coords.csv"));
        std::fs::write(&path, csv).unwrap();
        coords.push(path.to_str().unwrap().to_string());
    }
    (data.join(","), coords.join(","))
}

/// `p` (absolute) as a path relative to where the tests run.
#[cfg(feature = "view")]
pub(crate) fn from_cwd(p: &std::path::Path) -> std::path::PathBuf {
    use std::path::Component;
    let up: std::path::PathBuf = std::env::current_dir()
        .unwrap()
        .components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .map(|_| Component::ParentDir)
        .collect();
    up.join(p.strip_prefix("/").unwrap())
}
