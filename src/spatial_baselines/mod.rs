//! Published spatial embedding and clustering methods, ported to run on
//! pinto's data-beans input beside `lc`, `dsvd` and `cage`, as baselines.
//!
//! One method per strategy:
//!
//! - [`banksy`]: neighbour-augmented features, then PCA and Leiden (BANKSY,
//!   Singhal et al. 2024, Nature Genetics 56:431, doi:10.1038/s41588-024-01664-3).
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

pub mod banksy;
pub mod graph;
pub mod input;
pub mod output;
pub mod spatialde2;
pub mod stagate;

#[cfg(test)]
mod tests;
