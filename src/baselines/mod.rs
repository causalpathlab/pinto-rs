//! Published methods pinto is compared against, in one place.
//!
//! - [`single_cell`]: `pinto sim baseline`, the scanpy recipe (pooled,
//!   Harmony-corrected, per donor) run on a simulation.
//! - [`spatial`]: the spatial embedding baselines (SpatialDE2, STAGATE,
//!   SpatialPCA, MEFISTO and their `-pb` forms).

pub mod single_cell;
pub mod spatial;

#[cfg(test)]
mod notices_tests;
