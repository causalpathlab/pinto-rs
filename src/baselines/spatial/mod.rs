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
//! - [`spatialpca`]: spatially aware probabilistic PCA (SpatialPCA, Shang
//!   and Zhou 2022, Nature Communications 13:7203,
//!   doi:10.1038/s41467-022-34879-1).
//! - [`mefisto`]: multi-group factor analysis with spatially smooth factors
//!   (MEFISTO, Velten et al. 2022, Nature Methods 19:179,
//!   doi:10.1038/s41592-021-01343-9).
//!
//! The last two share one sparse spatial factor core
//! ([`spatial_factor`]) and are written from their papers alone; every
//! baseline's domains come from [`cluster`] where its original clusters
//! with a GPL package.
//!
//! Each also runs on super-cells (its `-pb` command): the cells coarsened
//! as cage coarsens them ([`pseudobulk`]), the fit then unravelled to
//! cells by conditional re-inference (`stagate-pb`: the trained encoder on
//! each cell).
//!
//! Each is faithful at the model level: the preprocessing, model, loss or
//! updates, defaults and clustering step of its reference code or paper,
//! cited per module. Each runs pinto's sparse formulation of the method
//! only, because the original dense formulations cost too much time and
//! memory at real cell counts; there is no `--reference` mode. Each runs on
//! the cells, one section at a time where the method builds a spatial
//! graph, and writes the same outputs ([`output`]):
//!
//! - `{out}.labels.tsv`: `cell<TAB>label`, for `pinto sim score-by-set --labels`;
//! - `{out}.cell_embedding.parquet`: a `cell` column and the method's
//!   per-cell vectors, for `pinto sim score-by-pair --vectors`;
//! - `{out}.baseline.json`: the method, its settings and its inputs.
//!
//! Batch effects are the one deviation every baseline shares: with two or
//! more batches, the counts are divided by the input's own δ estimate
//! (cage's estimator under cage's batch options, the same flags and
//! defaults, written to `{out}.delta.parquet` with each cell's batch)
//! before the method's own preprocessing
//! ([`input::SpatialInputArgs::read`]); `--ignore-batch` gives the original
//! method its counts as stored. The manifest records the δ files and
//! `baseline.json` records `batch_correction`.

pub mod cluster;
pub mod graph;
pub mod input;
pub mod mefisto;
pub mod memory;
pub mod output;
pub mod pseudobulk;
pub mod spatial_factor;
pub mod spatialde2;
pub mod spatialpca;
pub mod stagate;
