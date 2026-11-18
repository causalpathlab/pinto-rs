//! Scoring labellings and per-cell vectors against an `srt-simulate` run's
//! planted truth: `pinto sim score-by-set` ([`score`]) and `pinto sim score-by-pair`
//! ([`pairs`]). The simulator itself is in [`crate::simulate`]; the baseline
//! methods scored here are in [`crate::baselines`].

pub mod pairs;
pub mod score;

#[cfg(test)]
mod tests;

pub use pairs::{run_pairs, SimPairsArgs};
pub use score::{run_score, SimScoreArgs};
