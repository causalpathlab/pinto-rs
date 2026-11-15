//! Peak memory of a `stagate` or `spatialde2` run, estimated before its
//! large allocations, so a run that would not fit stops with an error
//! ([`crate::util::memory::ensure_fits`]) instead of swapping the machine
//! to a halt. Every estimate is in bytes.

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
/// (`SparseIoVec::read_columns_csc`): streamed, a file's `(row, column,
/// value)` triplets (24) beside the `(row, value)` buckets of each column
/// they fill (8), then the buckets beside the CSC assembled from them (12);
/// with `--preload-data`, the preloaded arrays (12) beside the buckets and
/// the CSC. The read's buffers of fixed size (a fused retrieve of at most
/// 48 MB, a file's cache of decoded chunks) are within the process's own.
const READ_NONZERO_BYTES: usize = 32;

/// Peak bytes of reading counts of `nonzeros` stored nonzeros.
pub fn read_bytes(nonzeros: usize) -> usize {
    READ_NONZERO_BYTES * nonzeros
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

/// STAGATE's peak, in the order of a run: the largest of the read
/// ([`read_bytes`]; the gene filter after it holds the counts beside their
/// filtered copy, 24 bytes a nonzero, less), the dense input built beside
/// the counts, the graph (the dense input, the counts freed; the Gram
/// blocks; 28 bytes a neighbour searched: 12 as found, beside the 16 of the
/// edge list made from them) and training ([`stagate_training_bytes`], or
/// on CUDA the input copied to the device).
pub fn stagate_peak_bytes(s: &StagateShape) -> usize {
    let counts = 12 * s.nonzeros;
    let dense = 4 * s.cells * s.genes;
    let graph = dense + s.gram + 28 * s.searched;
    let training = if s.train_on_host {
        stagate_training_bytes(s)
    } else {
        dense
    };
    STAGATE_PROCESS_BYTES
        + read_bytes(s.nonzeros)
            .max(counts + dense)
            .max(graph)
            .max(training)
}

/// What a SpatialDE2 run holds at once.
#[derive(Debug, Clone)]
pub struct Spatialde2Shape {
    pub cells: usize,
    /// Nonzeros of all the counts, as read ([`read_bytes`]).
    pub nonzeros: usize,
    /// Nonzeros of the genes segmented on.
    pub selected_nonzeros: usize,
    /// Edges of the symmetrised spatial graph, at most.
    pub edges: usize,
    /// Regions at the start (`--n-clusters`, else ceil(sqrt(cells))).
    pub regions: usize,
    /// [`knn_gram_bytes`] of the graph's kNN.
    pub gram: usize,
}

/// Bytes a selected nonzero holds at once in SpatialDE2's copy of the
/// counts: 16 while copied, 24 as kept (by cell and by gene), and 16 of
/// growth slack (the copied and the by-cell arrays grow by push, so their
/// capacity runs up to twice their length). The slack is calibrated so
/// that every measured peak is covered at the counts' real nonzeros.
const SELECTED_NONZERO_BYTES: usize = 16 + 24 + 16;

/// SpatialDE2's peak: the read ([`read_bytes`]), or after it, f64
/// throughout and summed as if all were live at once, the counts as read
/// (12 per nonzero) beside the selected counts
/// ([`SELECTED_NONZERO_BYTES`]); the weighted graph (16 per edge); the
/// posteriors, cells × regions, twice while pruning; and the kNN's Gram
/// blocks.
pub fn spatialde2_peak_bytes(s: &Spatialde2Shape) -> usize {
    let held = 12 * s.nonzeros
        + SELECTED_NONZERO_BYTES * s.selected_nonzeros
        + 16 * s.edges
        + 16 * s.cells * s.regions
        + s.gram;
    SPATIALDE2_PROCESS_BYTES + read_bytes(s.nonzeros).max(held)
}
