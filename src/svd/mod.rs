//! ==========================================================================
//! !!!  dsvd (DELTA-SVD) IS ABOUT PAIRS, NOT CELLS.  !!!
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
pub mod fit;

#[cfg(test)]
mod tests;
