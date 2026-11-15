//! Peak memory of a `stagate` or `spatialde2` run, estimated before its
//! large allocations, so a run that would not fit stops with an error
//! ([`ensure_fits`]) instead of swapping the machine
//! to a halt. Every estimate is in bytes.

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

/// Bytes a stored nonzero holds while δ is estimated (cage's estimator,
/// data-beans' multilevel collapse): the preloaded counts, the cells'
/// projection and the collapse's per-cell buffers. Calibrated on
/// srt-simulate runs of two sections (500 and 2000 genes, 512 to 16200
/// cells): the largest measured residual per nonzero, with a margin.
const DELTA_NONZERO_BYTES: usize = 52;

/// Bytes a gene holds per pseudobulk sample while δ is estimated: the
/// collapse's dense gene × sample statistics over its levels. Calibrated
/// with [`DELTA_NONZERO_BYTES`].
const DELTA_GENE_SAMPLE_BYTES: usize = 128;

/// Pseudobulk samples the δ estimate's finest level holds per batch at
/// most: `2^sort_dim` groups, `sort_dim` 10 as cage sets it.
pub const DELTA_SAMPLES_PER_BATCH: usize = 1 << 10;

/// What the δ estimate holds at once, above the process's own, for
/// `genes` genes, `nonzeros` stored nonzeros and at most `samples`
/// pseudobulk samples (each batch's cells, at most
/// [`DELTA_SAMPLES_PER_BATCH`] of them).
pub fn delta_estimate_bytes(genes: usize, nonzeros: usize, samples: usize) -> usize {
    DELTA_NONZERO_BYTES * nonzeros + DELTA_GENE_SAMPLE_BYTES * genes * samples
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
