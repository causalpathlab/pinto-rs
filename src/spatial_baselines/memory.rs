//! Peak memory of a `stagate` or `spatialde2` run, estimated before its
//! large allocations, so a run that would not fit stops with an error
//! ([`ensure_fits`]) instead of swapping the machine
//! to a halt. Every estimate is in bytes.

use super::input::InputShape;
use crate::util::memory::{ensure_fits, Budget};

/// The binary, its libraries and the allocator's own arenas: what a run of
/// a few hundred cells already holds, above its counted memory (candle's
/// CPU backend and its thread pool for STAGATE).
pub const STAGATE_PROCESS_BYTES: usize = 256 << 20;
pub const SPATIALDE2_PROCESS_BYTES: usize = 64 << 20;

/// Gram block bytes one kNN task holds (legume-numeric's
/// `knn::all_pairs::BLOCK_BYTES`), which sets its row block.
const GRAM_BLOCK_BYTES: usize = 64 << 20;

/// STAGATE's peak over its counted training memory: candle keeps buffers
/// the count misses (the zeroed buffer each first gradient is added to,
/// allocator retention). Calibrated on full runs on the CPU at 2000 genes,
/// hidden width 512 and 6 neighbours: the largest ratio of a run's
/// resident memory above [`STAGATE_PROCESS_BYTES`] to its count, over
/// every measured run (not the fitted slope: single runs scatter above
/// it), with a small margin.
pub const STAGATE_SAFETY: f64 = 1.65;

/// Bytes a stored nonzero holds while the counts are read
/// (`SparseIoVec::read_columns_csc`): preloaded, the preloaded arrays (12)
/// beside the `(row, value)` buckets of each column (8) and the CSC
/// assembled from them (12); streamed (data over data-beans' preload
/// budget), a file's `(row, column, value)` triplets (24) beside the
/// buckets they fill, then the buckets beside the CSC. The read's buffers
/// of fixed size (a fused retrieve of at most 48 MB, a file's cache of
/// decoded chunks) are within the process's own.
const READ_NONZERO_BYTES: usize = 32;

/// Bytes a nonzero of the counts holds once read (CSC: an f32 value, a row
/// index).
const HELD_NONZERO_BYTES: usize = 12;

/// Peak bytes of reading counts of `nonzeros` stored nonzeros.
pub fn read_bytes(nonzeros: usize) -> usize {
    READ_NONZERO_BYTES * nonzeros
}

/// A run's peak: the `process`'s own bytes, then the read of `nonzeros`
/// ([`read_bytes`]) or the most it `held` at once after it.
fn peak(process: usize, nonzeros: usize, held: usize) -> usize {
    process + read_bytes(nonzeros).max(held)
}

/// The first memory check, from the input's stored shape before its
/// counts are read: `what`'s `estimate` from it, `None` when the files do
/// not store their shape, which leaves the read unchecked.
pub fn check_stored_shape(
    what: &str,
    estimate: Option<usize>,
    budget: &Budget,
    change: &str,
) -> anyhow::Result<()> {
    let Some(estimate) = estimate else {
        log::info!(
            "{what}: the data files do not store their column and nonzero counts; \
             their read is not checked"
        );
        return Ok(());
    };
    ensure_fits(
        &format!("{what} (from the input's stored shape)"),
        estimate,
        budget,
        change,
    )
}

/// Gram block bytes of the all-pairs kNN in flight at once: each task holds
/// one `cells × block` f32 block of a section, and at most `threads` run at
/// once.
pub fn knn_gram_bytes(section_sizes: &[usize], threads: usize) -> usize {
    let mut tasks: Vec<usize> = section_sizes
        .iter()
        .filter(|&&n| n > 1)
        .flat_map(|&n| {
            let block = (GRAM_BLOCK_BYTES / (4 * n)).clamp(16, 1024).min(n);
            std::iter::repeat_n(4 * n * block, n.div_ceil(block))
        })
        .collect();
    tasks.sort_unstable_by(|a, b| b.cmp(a));
    tasks.iter().take(threads.max(1)).sum()
}

/// Directed edges of each cell's `k` nearest within its section, at most
/// (fewer when a distance is 0 or the section is small).
pub fn knn_edges(section_sizes: &[usize], k: usize) -> usize {
    section_sizes
        .iter()
        .map(|&n| n * k.min(n.saturating_sub(1)))
        .sum()
}

/// What `spatialde2-pb` holds above its cell-level fit: the cells'
/// projection (cage's `-p` f32 a cell), which the coarsening takes and
/// normalises in place, and the seeds' features (two), the coarsening
/// graph (cage's `-k` neighbours a cell, as lists, edges and adjacency: 64
/// bytes a neighbour), the
/// super-cells' counts (at most every nonzero: 12 as built, 12 held), and
/// the fold of the method's `edges` listed edges onto the super-cells (an
/// undirected copy, 24 bytes a pair, and the fold's map and fine-to-super
/// index, 48 a pair: 36 a listed edge, 40 with a margin).
pub fn pseudobulk_bytes(cells: usize, nonzeros: usize, edges: usize) -> usize {
    let proj = 2 * 4 * crate::util::input::DEFAULT_PROJ_DIM * cells;
    let graph = 64 * crate::spatial_baselines::pseudobulk::COARSEN_K * cells;
    proj + graph + 24 * nonzeros + 40 * edges
}

/// What a STAGATE run holds at once.
#[derive(Debug, Clone)]
pub struct StagateShape {
    pub cells: usize,
    /// Edges of the spatial graph, self loops included.
    pub edges: usize,
    /// Genes trained on.
    pub genes: usize,
    pub hidden: usize,
    /// Nonzeros of the counts: read ([`read_bytes`]), then held until the
    /// dense input is built.
    pub nonzeros: usize,
    /// [`knn_gram_bytes`] of the graph's kNN.
    pub gram: usize,
    /// Neighbours the kNN searches, [`knn_edges`] at `--k-cutoff` (or at
    /// the radius search's cap), each an index and a distance.
    pub searched: usize,
    /// Whether training runs in host memory (the CPU, and Metal's unified
    /// memory); on CUDA the device holds it.
    pub train_on_host: bool,
    /// Whether the input is sparse (the default) or dense (`--reference`).
    pub sparse: bool,
    /// Nonzeros of the genes trained on, which the sparse input holds; at
    /// most [`Self::nonzeros`].
    pub selected_nonzeros: usize,
}

/// Training memory, f32, counted from what candle keeps for autograd and
/// what its backward adds, per cell at `E = (k + 1) N` edges:
///
/// - forward kept: `4G + 4(k+1)H + 8H` (the input, reconstruction,
///   residual and its square; the gathered and the attention-weighted
///   messages of the two attention layers; the layers' node outputs);
/// - backward adds: `G` (the input's gradient, which candle computes
///   though nothing reads it), `2H` (the zeroed aggregation buffers'),
///   `6(k+1)H` (a message product's gradients, live at once) and `4H`
///   (the ELU's masks);
/// - the weights, their gradients and Adam's moments: `6GH`, whatever `N`.
pub fn stagate_training_bytes(s: &StagateShape) -> usize {
    let (n, e, g, h) = (s.cells, s.edges, s.genes, s.hidden);
    let floats = 5 * g * n + 10 * e * h + 14 * h * n + 6 * g * h;
    (4.0 * floats as f64 * STAGATE_SAFETY) as usize
}

/// Bytes a nonzero of the sparse input holds: a gene index and a value by
/// cell, a cell index and a value by gene.
const SPARSE_INPUT_NONZERO_BYTES: usize = 16;

/// The sparse formulation's peak over its counted training memory, as
/// [`STAGATE_SAFETY`] is the dense one's. Calibrated on sparse CPU runs at
/// the measured runs' settings (1024 to 16384 cells, 4 threads): there the
/// ratio of a run's peak, above the process and the sparse input, to its
/// count rises with the cells toward the ratio of their per-cell slopes,
/// about 1.37, so it is that slope's ratio with a margin rather than the
/// largest ratio measured.
pub const STAGATE_SPARSE_SAFETY: f64 = 1.45;

/// The sparse formulation's training memory, f32, per cell at `E` edges
/// as [`stagate_training_bytes`] counts the dense one's, less the input
/// and reconstruction it never holds whole (`5G` a cell): the first
/// layer's product as a leaf, its gradient and their host copies (`4H`); a
/// loss block of `B` cells (its dense rows, reconstruction, residual and
/// its square, and the residual's gradient, `5BG`, with the block's
/// gradient to `h3`, `BH`, and the gradient to `h3` gathered, `NH`); the
/// weights, gradients and moments (`6GH`) and a host copy of `W1`
/// (`GH`); beside the sparse input itself.
pub fn stagate_sparse_training_bytes(s: &StagateShape) -> usize {
    let (n, e, g, h) = (s.cells, s.edges, s.genes, s.hidden);
    let b = crate::spatial_baselines::stagate::loss_block_cells(g).min(n);
    let floats = 10 * e * h + 19 * h * n + 5 * b * g + b * h + 7 * g * h;
    (4.0 * floats as f64 * STAGATE_SPARSE_SAFETY) as usize
        + SPARSE_INPUT_NONZERO_BYTES * s.selected_nonzeros
}

/// STAGATE's peak, in the order of a run: the largest of the read
/// ([`read_bytes`]; the gene filter after it holds the counts beside their
/// filtered copy, 24 bytes a nonzero, less), the input (dense, or sparse)
/// built beside the counts, the graph (the dense input, the counts freed; the Gram
/// blocks; 28 bytes a neighbour searched: 12 as found, beside the 16 of the
/// edge list made from them) and training ([`stagate_training_bytes`],
/// [`stagate_sparse_training_bytes`] for the sparse input, or on CUDA the
/// input copied to the device).
pub fn stagate_peak_bytes(s: &StagateShape) -> usize {
    let counts = HELD_NONZERO_BYTES * s.nonzeros;
    // the input: dense, or sparse (on the host, wherever training runs)
    let dense = match s.sparse {
        true => SPARSE_INPUT_NONZERO_BYTES * s.selected_nonzeros,
        false => 4 * s.cells * s.genes,
    };
    let graph = dense + s.gram + 28 * s.searched;
    let training = match (s.train_on_host, s.sparse) {
        (true, false) => stagate_training_bytes(s),
        (true, true) => stagate_sparse_training_bytes(s),
        (false, _) => dense,
    };
    peak(
        STAGATE_PROCESS_BYTES,
        s.nonzeros,
        (counts + dense).max(graph).max(training),
    )
}

/// What a SpatialDE2 run holds at once.
#[derive(Debug, Clone)]
pub struct Spatialde2Shape {
    pub cells: usize,
    /// Nonzeros of all the counts, as read ([`read_bytes`]).
    pub nonzeros: usize,
    /// Nonzeros of the genes segmented on.
    pub selected_nonzeros: usize,
    /// Genes segmented on.
    pub genes: usize,
    /// Edges of the symmetrised spatial graph, at most (twice the kNN's).
    pub edges: usize,
    /// Regions at the start (`--n-clusters`, else ceil(sqrt(cells))).
    pub regions: usize,
    /// [`knn_gram_bytes`] of the graph's kNN.
    pub gram: usize,
    /// Cells an all-pairs graph (`--reference`) holds, each its 2-D
    /// coordinates, its section and its place in the section's list.
    pub pair_cells: usize,
}

/// Bytes a cell holds in an all-pairs graph: two f64 coordinates, its
/// section and its entry in its section's list (8 each), with as much again
/// for the lists' growth.
const PAIR_CELL_BYTES: usize = 48;

/// Bytes a selected nonzero holds at once in SpatialDE2's copy of the
/// counts: 16 while copied, 24 as kept (by cell and by gene), and 16 of
/// growth slack (the copied and the by-cell arrays grow by push, so their
/// capacity runs up to twice their length). The slack is calibrated so
/// that every measured peak is covered at the counts' real nonzeros.
const SELECTED_NONZERO_BYTES: usize = 16 + 24 + 16;

/// Bytes an edge of the symmetrised graph holds while it is built
/// (`Neighbours::symmetrised`), at its bound of twice the kNN's `E` edges:
/// the kNN's lists (12 `E`), their sorted copy (8 `E`), the reverse edges
/// missing from them (24 each, `E` at most) and the merged lists (16 each,
/// `2E` at most), live at once, 76 `E`. The weighted graph built after it,
/// beside the symmetric lists, holds 28 an edge, and keeps 16.
const GRAPH_EDGE_BYTES: usize = 38;

/// SpatialDE2's peak: the read ([`read_bytes`]), or after it, f64
/// throughout and summed as if all were live at once, the counts as read
/// ([`HELD_NONZERO_BYTES`] a nonzero) beside the selected counts
/// ([`SELECTED_NONZERO_BYTES`]); the graph as built
/// ([`GRAPH_EDGE_BYTES`]); the posteriors, cells × regions, twice while
/// pruning; an update's genes × regions arrays (E[λ], E[log λ] in both
/// layouts, π̂ Y, the old and the new gammahat_1: six); and the kNN's Gram
/// blocks.
pub fn spatialde2_peak_bytes(s: &Spatialde2Shape) -> usize {
    let held = HELD_NONZERO_BYTES * s.nonzeros
        + SELECTED_NONZERO_BYTES * s.selected_nonzeros
        + GRAPH_EDGE_BYTES * s.edges
        + 16 * s.cells * s.regions
        + 6 * 8 * s.genes * s.regions
        + PAIR_CELL_BYTES * s.pair_cells
        + s.gram;
    peak(SPATIALDE2_PROCESS_BYTES, s.nonzeros, held)
}

/// The binary, its libraries and the allocator's arenas, for `spatialpca`
/// and `mefisto`.
pub const FACTOR_PROCESS_BYTES: usize = 64 << 20;

/// What a `spatialpca` or `mefisto` run holds at once.
#[derive(Debug, Clone)]
pub struct FactorShape {
    pub input: InputShape,
    /// Genes modelled.
    pub genes: usize,
    /// Neighbours of the prior's kNN graph.
    pub knn: usize,
    /// Sections up to this size take a dense eigendecomposition of their
    /// Laplacian.
    pub exact_eig_max: usize,
    /// f64 vectors over the cells the fit holds at once.
    pub cell_vectors: usize,
    /// f64 vectors over the genes the fit holds at once.
    pub gene_vectors: usize,
    /// A `-pb` run's target number of super-cells (0 on cells): the
    /// coarsening's memory too.
    pub super_cells: usize,
    /// What `--reference` holds beyond the sparse path.
    pub reference: usize,
}

/// Bytes a stored nonzero takes over a factor baseline's run, summed
/// rather than overlapped since the allocator keeps what each step frees
/// for the next (measured: the resident memory after each step only
/// grows): the read ([`read_bytes`]), the selected genes' copy (12) and
/// the implicit expression (16, by gene and by cell), every nonzero
/// counted as selected.
const FACTOR_NONZERO_BYTES: usize = 32 + 12 + 16;

/// A factor baseline's peak: the process, every nonzero's
/// [`FACTOR_NONZERO_BYTES`], the graph as built (the kNN's Gram blocks,
/// [`GRAPH_EDGE_BYTES`] an edge of the symmetrised kNN, then its lists and
/// its Laplacian, 12 bytes an entry each), the dense blocks of the exactly
/// decomposed sections (three `n_s^2` copies, a thread each), the fit's
/// vectors, what `--reference` adds, and with `-pb` the coarsening and
/// the super-cells' own expression.
pub fn factor_peak_bytes(s: &FactorShape) -> usize {
    let sizes = &s.input.section_sizes;
    let n: usize = sizes.iter().sum();
    let nnz = s.input.nonzeros;
    let edges = 2 * knn_edges(sizes, s.knn);
    let threads = rayon::current_num_threads().max(1);
    let mut exact: Vec<usize> = sizes
        .iter()
        .filter(|&&k| k <= s.exact_eig_max)
        .map(|&k| 3 * 8 * k * k)
        .collect();
    exact.sort_unstable_by(|a, b| b.cmp(a));
    let exact: usize = exact.iter().take(threads).sum();
    let graph = knn_gram_bytes(sizes, threads) + (GRAPH_EDGE_BYTES + 24) * edges;
    // the super-cells' counts, their selected copy and their expression,
    // at most every nonzero and every gene of every super-cell, beside the
    // coarsening's
    let pb = if s.super_cells > 0 {
        let super_nnz = nnz.min(s.super_cells * s.input.genes);
        pseudobulk_bytes(n, nnz, edges) + 40 * super_nnz
    } else {
        0
    };
    FACTOR_PROCESS_BYTES
        + FACTOR_NONZERO_BYTES * nnz
        + graph
        + exact
        + 8 * n * s.cell_vectors
        + 8 * s.genes * s.gene_vectors
        + s.reference
        + pb
}
