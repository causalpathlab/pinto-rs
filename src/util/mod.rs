pub mod ablation;
pub mod batch_effects;
pub mod cell_pairs;
pub mod common;
pub mod coverage;
pub mod device;
pub mod dictionary_rows;
pub mod edge_clustering;
pub mod feature_axis;
pub mod graph_coarsen;
pub mod input;
pub mod lupin_rounds;
pub mod metadata;
pub mod parquet_io;
pub mod profile_match;
pub mod score_trace;
pub mod srt_pipeline;

#[cfg(test)]
mod tests;
