//! Published spatial embedding and clustering methods, ported to run on
//! pinto's data-beans input beside `lc`, `dsvd` and `cage`, as baselines.
//!
//! One method per strategy:
//!
//! - [`spatialde2`]: SpatialDE2's tissue segmentation, a Poisson hidden
//!   Markov random field with a Dirichlet-process prior over regions (Kats,
//!   Vento-Tormo and Stegle 2021, bioRxiv doi:10.1101/2021.10.27.466045).
//! - [`stagate`]: a graph-attention autoencoder over the spatial graph
//!   (STAGATE, Dong and Zhang 2022, Nature Communications 13:1739,
//!   doi:10.1038/s41467-022-29439-6).
//!
//! Each is faithful at the model level: the preprocessing, model, loss or
//! updates, defaults and clustering step of its reference code, cited per
//! module. Each runs on the cells, one section at a time where the method
//! builds a spatial graph, and writes the same outputs ([`output`]):
//!
//! - `{out}.labels.tsv`: `cell<TAB>label`, for `pinto srt-sim-score --labels`;
//! - `{out}.cell_embedding.parquet`: a `cell` column and the method's
//!   per-cell vectors, for `pinto srt-sim-pairs --vectors`;
//! - `{out}.baseline.json`: the method, its settings and its inputs.
//!
//! Batch effects are the one deviation every baseline shares: with two or
//! more batches, the counts are divided by the input's own δ estimate
//! (cage's estimator at cage's defaults, written to `{out}.delta.parquet`
//! with each cell's batch) before the method's own preprocessing
//! ([`input::SpatialInputArgs::read`]); `--ignore-batch` gives the original
//! method its counts as stored. The manifest records the δ files and
//! `baseline.json` records `batch_correction`.

pub mod graph;
pub mod input;
pub mod memory;
pub mod output;
pub mod spatialde2;
pub mod stagate;

#[cfg(test)]
mod tests;
