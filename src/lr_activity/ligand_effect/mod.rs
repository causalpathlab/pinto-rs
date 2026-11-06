//! Effect of neighbour ligand on receivers (`pinto lra`, the default
//! `--method contact-effect`; `pinto lra-simulate` benchmarks it).
//!
//! Co-expression scores ask whether a ligand and its receptor are detected
//! together across contacts. This asks what happens to a receiver's
//! response programme when its contact neighbours carry the ligand, among
//! receivers comparable in everything else that could drive both: region,
//! section, cell identity, neighbour composition, depth and segmentation
//! bleed. The two answers disagree when co-location is confounded, and the
//! output puts them side by side.
//!
//! **Estimands** (per pair; pooled and per section). With `T` = contact
//! neighbours carry ligand above a threshold, `Y` = the receiver's signed
//! target programme, `C` = the receiver's identity expresses the receptor:
//!
//! - `effect_competent`, the competent effect, **the headline**:
//!   overlap-weighted effect of `T` on `Y` in receptor-competent receivers;
//! - `effect_incompetent`, the incompetent effect: the same in incompetent
//!   receivers (≈ 0 is a negative control);
//! - `effect_contrast`, the contrast: competent effect − incompetent effect.
//!   Competence is read from identity bins, so competent and incompetent
//!   receivers are different cell types by construction and the contrast
//!   conflates the receptor with cell type; it is no headline, but a
//!   significant effect must exceed the incompetent one;
//! - `ring_effect`, the ring effect: effect of ligand on distant non-touching cells given
//!   the real exposure (≈ 0 is a negative control for spatial confounding);
//! - for comparison, and in no verdict: `coexpr_log_or`, the co-expression log odds
//!   ratio; `marginal_cor`, the correlation of response and contact ligand; `crude`, the
//!   unadjusted contrast; `effect_ipw` and `effect_outcome`, the headline's two halves,
//!   overlap weights alone and the outcome regression alone.
//!
//! **Inference across pairs** ([`report::report`]): Benjamini–Hochberg q-values
//! within the pooled rows and within each section, for the competent effect
//! (two-sided), co-expression (one-sided, positive) and the controls, and a
//! per-ligand q (Simes over the ligand's pairs, BH over ligands); the verdict
//! ([`verdict::Verdict`]) reads q at [`Internals::fdr`]. When the ligand marks
//! a cell type, exposure is nearly "has a neighbour of that type" and cannot be
//! told from neighbour identity: `exposure_eta2` measures how much of the
//! exposure the dominant neighbour's identity bin explains, and at
//! [`INTERNALS`]`.max_exposure_eta2` the pair is `celltype_inseparable` rather
//! than a finding. A significant competent effect beside a failed control is
//! `control_failed`.
//!
//! "Effect" is a contrast of two stochastic interventions: each receiver's
//! dose drawn from its own plausible doses above versus below the
//! threshold. Identification rests on (1) a receiver depending on its
//! neighbours only through `T`, (2) no unmeasured confounding given the
//! confounders, (3) overlap, which the overlap weights make a matter of
//! target population rather than of variance, and (4) consistency in that
//! stochastic sense. The negative controls probe (2).
//!
//! Layout:
//!
//! - the commands: [`pipeline`] runs them from input to tables, [`pairs`]
//!   and [`only`] pick the pairs from the [`mod@reference`] and [`targets`] reads
//!   the programmes, [`genes`] reads the needed genes sparsely, [`memory`]
//!   says how many pairs run at once;
//! - the tissue: [`exposure`], [`folds`], [`confounders`] and [`receivers`]
//!   build [`tissue::Tissue`];
//! - one pair: [`estimate`] assembles it from [`dose`] and [`response`],
//!   [`thresholds`] fits each exposure threshold, [`fits`] holds the nuisance
//!   fits there, [`effects`] reads every estimand from them, [`curve`] and
//!   [`background`] add the threshold-response curve and the matched
//!   programmes, under the [`options`] asked for, into [`rows`];
//! - the estimator: [`overlap`], [`crossfit`] and [`regression`] estimate,
//!   [`inference`] and [`averaging`] read and combine the estimates;
//! - across pairs: [`report`] corrects for multiplicity and [`verdict`] words
//!   the answer;
//! - around them: [`baselines`] are the co-expression scores it is compared
//!   with, [`simulate`] benchmarks the estimator on this tissue with a planted
//!   effect, [`output`] writes and reads the tables, and [`timing`] sums the
//!   time per stage for the run log; [`frame`] puts one pair on a run's
//!   receivers for `pinto view`.

pub mod averaging;
pub mod background;
pub mod baselines;
pub mod confounders;
pub mod crossfit;
pub mod curve;
pub mod dose;
pub mod effects;
pub mod estimate;
pub mod exposure;
pub mod fits;
pub mod folds;
#[cfg(feature = "view")]
pub mod frame;
pub mod genes;
pub mod inference;
pub mod memory;
pub mod only;
pub mod options;
pub mod output;
pub mod overlap;
pub mod pairs;
pub mod pipeline;
pub mod receivers;
pub mod reference;
pub mod regression;
pub mod report;
pub mod response;
pub mod rows;
pub mod simulate;
pub mod targets;
pub mod thresholds;
pub mod timing;
pub mod tissue;
pub mod verdict;

#[cfg(test)]
mod tests;

use crate::lr_activity::args::ReferenceSpecies;
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use std::path::Path;

/// The fixed choices of the contact-effect analysis. None is a flag: each
/// is recorded in the manifest (`outputs.lra.settings`) beside the table it
/// made.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Internals {
    /// Shares of a receiver identity bin detecting the receptor for its
    /// receivers to count as competent; the estimate averages over them.
    pub competence_grid: [f64; 3],
    /// The single competence threshold `lra-simulate` plants on.
    pub benchmark_competence: f64,
    /// Prior SD of an effect at one exposure threshold, for the average over
    /// thresholds.
    pub abf_prior_sd: f64,
    /// Background programmes per pair.
    pub background_draws: usize,
    /// Fewest exposed and fewest unexposed receivers of a reported subgroup.
    pub min_arm: usize,
    /// Smallest effective sample size of a subgroup's overlap weights.
    pub min_ess: f64,
    /// Smallest mean overlap weight `e(1 − e)` of a subgroup.
    pub min_overlap: f64,
    /// Share of exposure variance explained by neighbour identity at which
    /// a pair is `celltype_inseparable`.
    pub max_exposure_eta2: f64,
    /// The false discovery rate a test counts at in the verdicts.
    pub fdr: f64,
    /// Dimension of the receiver identity projection, and its bins (2^bits).
    pub identity_dim: usize,
    pub identity_bits: usize,
    /// Fewest measured genes in a pair's response programme.
    pub min_targets: usize,
    /// Negative-control ring, inner and outer, in median contact lengths.
    pub ring: (f64, f64),
    /// Length scales of the regional confounder smooth, in contact lengths.
    pub scales: [f64; 2],
    /// Tile width for cross-fitting folds and variance clusters.
    pub tile: f64,
    pub folds: usize,
}

pub const INTERNALS: Internals = Internals {
    competence_grid: options::COMPETENCE_GRID,
    benchmark_competence: options::BENCHMARK_COMPETENCE[0],
    abf_prior_sd: options::ABF_PRIOR_SD,
    background_draws: options::BACKGROUND_DRAWS,
    min_arm: 20,
    min_ess: 30.0,
    min_overlap: 0.01,
    max_exposure_eta2: 0.8,
    fdr: 0.05,
    identity_dim: 10,
    identity_bits: 6,
    min_targets: 5,
    ring: (15.0, 30.0),
    scales: [8.0, 32.0],
    tile: 25.0,
    folds: 5,
};

/// What the user chose, and where the run records itself.
pub struct Settings<'a> {
    pub out: &'a str,
    /// The manifest the outputs are recorded in.
    pub manifest: &'a Path,
    pub lr_pairs: Option<&'a str>,
    pub targets: &'a [Box<str>],
    pub species: ReferenceSpecies,
    pub seed: u64,
    pub block_size: Option<usize>,
    /// `--only`: the pairs to estimate; empty for every pair.
    pub only: &'a [Box<str>],
}

impl Settings<'_> {
    /// What decides the pairs and the receivers' identity, as the manifest
    /// records it beside the table ([`crate::util::metadata::LraOptions`]).
    pub fn options(&self) -> crate::util::metadata::LraOptions {
        let abs = |p: &str| crate::lr_activity::recorded(Path::new(p));
        crate::util::metadata::LraOptions {
            lr_pairs: self.lr_pairs.map(abs),
            targets: self.targets.iter().map(|t| abs(t)).collect(),
            species: self.species.name(),
            seed: self.seed,
            identity_excluded: Vec::new(),
        }
    }

    fn choices(&self) -> PairChoices<'_> {
        PairChoices {
            species: self.species,
            lr_pairs: self.lr_pairs,
            targets: self.targets,
        }
    }
}

/// What decides which pairs there are: the reference's species and the
/// user's pair and programme files on top of it.
pub struct PairChoices<'a> {
    pub species: ReferenceSpecies,
    pub lr_pairs: Option<&'a str>,
    pub targets: &'a [Box<str>],
}

/// What the lra front end hands over.
pub struct RunInputs<'a> {
    pub settings: Settings<'a>,
    pub data: &'a SparseIoVec,
    pub axis: &'a FeatureAxis,
    /// Spatial contacts as data columns, and each cell's section label.
    pub pairs: &'a [(usize, usize)],
    pub cell_section: &'a [Box<str>],
    pub coords: &'a [Option<(f32, f32)>],
    /// Cell-type code per data column and the types' names, when known.
    pub cell_type: Option<(&'a [u32], &'a [Box<str>])>,
}
