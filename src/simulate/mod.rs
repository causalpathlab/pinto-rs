//! `pinto srt-simulate`: a multi-donor spatial transcriptomics simulator
//! with a planted truth.
//!
//! Location sets a spot's domain, the domain sets its likely cell states,
//! and the states set expression (`truth`, `expression`). Donors carry
//! nuisance a method should remove, donor-wide and donor by type, at set
//! proportions of variance; a domain one donor alone has (a new state, or a
//! new mix of shared states) is biology a method should keep, and the truth
//! tables mark it with its owner. Counts are negative binomial through a
//! low-rank Gaussian copula (`draw`), from keyed random streams (`rng`), so
//! a run with `--set nuisance_off=true` and the same seed is the exact
//! counterfactual. `sim::Sim` holds one simulation.
//!
//! With one donor, no donor or section term and one state per domain, this
//! is the reference-free model of SRTsim (Zhu, Shang and Zhou 2023,
//! doi:10.1186/s13059-023-02879-z): domains whose genes differ by a fold
//! change. The cell-state layer and the donor terms are what it adds.

pub mod draw;
pub mod expression;
pub mod field;
pub mod panel;
pub mod params;
pub mod rng;
pub mod run;
pub mod score;
pub mod sim;
pub mod truth;
pub mod write;

#[cfg(test)]
mod tests;

pub use run::{run_simulate, SimulateArgs};
pub use score::{run_score, SimScoreArgs};
