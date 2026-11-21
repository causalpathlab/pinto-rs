//! ==========================================================================
//! !!!  cage (AND cage-cell) IS ABOUT PAIRS, NOT CELLS.  !!!
//! ==========================================================================
//!
//! pinto's first-line methods (lc, dsvd, cage) model CELL PAIRS: spatially
//! adjacent cells (or super-cells) in contact. The unit every one of them
//! fits, embeds and clusters is the PAIR (link, edge, contact), and a cell's
//! outputs (propensity, embedding) are DERIVED from the pairs it is in.
//!
//! Never let a pair reduce to its two cells: no pair latent that is only a
//! function of two separately estimated cell vectors (a product, sum or mean
//! of node rows), no score whose pair term cancels, no clustering of cells
//! standing in for clustering of pairs. A change that turns a pair quantity
//! into a per-cell one breaks what these methods are for.
//! ==========================================================================
//!
//! `pinto fit cage` — activity-gated cell-graph embedding.
//!
//! Learns a row per super edge (cage: a pair of adjacent finest super-cells;
//! cage-cell: a cell pair) by visiting each feature: a feature's pooled count
//! at a pair's two ends draws it as a positive, against other pairs of the
//! same batch. Embedding-only — no count decoder. cage's pair encoder then
//! learns to place a pair where phase 1 put it, and places every cell pair.

pub mod args;
pub mod feature_chain_sampler;
pub mod feature_gating;
pub mod feature_inclusion;
pub mod fit;
pub mod loss;
pub mod pair_projection;
pub mod pb_frame;
pub mod pretrained;
pub mod row_adam;

#[cfg(test)]
mod feature_gating_tests;
#[cfg(test)]
mod pb_frame_tests;

#[cfg(test)]
mod loss_block_tests;

#[cfg(test)]
mod pretrained_tests;

#[cfg(test)]
mod tests;

pub use args::CellActivityGraphEmbeddingArgs;
pub use fit::fit_cell_activity_graph_embedding;
