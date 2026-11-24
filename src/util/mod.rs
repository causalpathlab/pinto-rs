pub mod ablation;
pub mod batch_effects;
pub mod batch_fold;
pub mod cell_pairs;
pub mod cli;
pub mod common;
pub mod coverage;
pub mod device;
pub mod dictionary_rows;
pub mod edge_clustering;
pub mod feature_axis;
pub mod feature_modules;
pub mod graph_coarsen;
pub mod input;
pub mod local_fdr;
pub mod lupin_rounds;
pub mod memory;
pub mod metadata;
pub mod parquet_io;
pub mod paths;
pub mod profile_match;
pub mod rows;
pub mod score_trace;
pub mod srt_pipeline;
pub mod staging;

#[cfg(test)]
pub(crate) mod tests;
