//! Fitting every gene, with its section jackknife, in one of two ways:
//!
//! * `fit_genes_on_cpu_threads` (`--device cpu`): one gene per rayon task,
//!   `contrast_fit` and `section_jackknife` on the CPU;
//! * `fit_genes_in_blocks` (`--device cuda` or `metal`): blocks of genes on
//!   a candle device, `batched_contrast_fit`, the block size taken from the
//!   device's free memory.
//!
//! Both read a gene's activity the same way (`write_activity_of_gene`): its
//! pooled counts on each pair, divided by its same-section factor.

use super::batched_contrast_fit::{fit_block_of_genes, SharedContrastTensors};
use super::contrast_fit::{
    contrast_sets_of_gene, fit_deviation_of_gene, ContrastFitSettings, PairStates,
};
use super::matched_other_donor_pairs::MatchedOtherDonorPairs;
use super::outputs::FittedGene;
use super::{same_section_gene_factor, section_jackknife, IndexedSections};
use crate::util::common::*;

/// What a gene's activity is read with.
pub(super) struct ActivityInputs<'a> {
    pub matched: &'a MatchedOtherDonorPairs,
    /// For every pair, the pairs that list it among their matches.
    pub pairs_matching_pair: &'a [Vec<u32>],
    pub sections: &'a IndexedSections,
}

/// One thread's scratch over all pairs, all 0 / false between genes.
struct GeneScratch {
    activity_of_pair: Vec<f32>,
    pair_has_signal: Vec<bool>,
}

impl ActivityInputs<'_> {
    fn number_of_pairs(&self) -> usize {
        self.sections.section_index_of_pair.len()
    }

    fn new_scratch(&self) -> GeneScratch {
        GeneScratch {
            activity_of_pair: vec![0f32; self.number_of_pairs()],
            pair_has_signal: vec![false; self.number_of_pairs()],
        }
    }

    /// Write a gene's activity into the scratch at its `nonzero_pairs`: the
    /// pooled count divided by the exp of its section's log factor.
    /// `clear_activity_of_gene` puts the scratch back to 0.
    fn write_activity_of_gene(&self, nonzero_pairs: &[(u32, f32)], scratch: &mut GeneScratch) {
        for &(pair, count) in nonzero_pairs {
            scratch.activity_of_pair[pair as usize] = count;
        }
        let pairs_with_signal = self.pairs_with_own_or_matched_counts(nonzero_pairs, scratch);
        let log_factor_of_section = same_section_gene_factor::log_factor_of_section_for_gene(
            &scratch.activity_of_pair,
            &pairs_with_signal,
            self.matched,
            &self.sections.section_index_of_pair,
            &self.sections.number_of_pairs_in_section,
        );
        for &(pair, _) in nonzero_pairs {
            let pair = pair as usize;
            scratch.activity_of_pair[pair] /=
                log_factor_of_section[self.sections.section_index_of_pair[pair]].exp();
        }
    }

    /// The pairs whose log ratio to their matches is not exactly 0 for a
    /// gene: those with a count of their own, and those with one among their
    /// matches; each once (marked in the scratch, then unmarked).
    fn pairs_with_own_or_matched_counts(
        &self,
        nonzero_pairs: &[(u32, f32)],
        scratch: &mut GeneScratch,
    ) -> Vec<u32> {
        let mut pairs = Vec::with_capacity(nonzero_pairs.len());
        for &(pair, _) in nonzero_pairs {
            let own_and_matching = std::iter::once(pair)
                .chain(self.pairs_matching_pair[pair as usize].iter().copied());
            for p in own_and_matching {
                if !scratch.pair_has_signal[p as usize] {
                    scratch.pair_has_signal[p as usize] = true;
                    pairs.push(p);
                }
            }
        }
        for &p in &pairs {
            scratch.pair_has_signal[p as usize] = false;
        }
        pairs
    }
}

fn clear_activity_of_gene(nonzero_pairs: &[(u32, f32)], scratch: &mut GeneScratch) {
    for &(pair, _) in nonzero_pairs {
        scratch.activity_of_pair[pair as usize] = 0.0;
    }
}

/// `--device cpu`: each gene a rayon task, its full fit then its refits
/// without each section, started from it.
pub(super) fn fit_genes_on_cpu_threads(
    nonzero_pairs_of_gene: &[Vec<(u32, f32)>],
    inputs: &ActivityInputs,
    anchor_pairs: &[u32],
    states: &PairStates,
    settings: &ContrastFitSettings,
) -> Vec<FittedGene> {
    nonzero_pairs_of_gene
        .par_iter()
        .progress_count(nonzero_pairs_of_gene.len() as u64)
        .map_init(
            || inputs.new_scratch(),
            |scratch, nonzero_pairs| {
                inputs.write_activity_of_gene(nonzero_pairs, scratch);
                let sets =
                    contrast_sets_of_gene(&scratch.activity_of_pair, anchor_pairs, inputs.matched);
                clear_activity_of_gene(nonzero_pairs, scratch);
                let full_fit = fit_deviation_of_gene(&sets, states, None, None, settings);
                let refits = section_jackknife::deviation_without_each_section(
                    &full_fit.deviation,
                    &sets,
                    states,
                    &inputs.sections.section_index_of_pair,
                    inputs.sections.section_names.len(),
                    settings,
                );
                let fits_at_iteration_cap = std::iter::once(&full_fit)
                    .chain(&refits)
                    .filter(|fit| fit.reached_iteration_cap)
                    .count();
                FittedGene {
                    deviation: full_fit.deviation,
                    deviation_without_section: refits
                        .into_iter()
                        .map(|fit| fit.deviation)
                        .collect(),
                    fits_at_iteration_cap,
                }
            },
        )
        .collect()
}

/// Genes per block on a device without a memory query (CPU, Metal).
const GENES_PER_BLOCK_WITHOUT_MEMORY_QUERY: usize = 16;

/// How many genes a block fits in: on CUDA, as many as fit `memory_fraction`
/// of the device's free memory (each gene its activity and its refits'
/// working tensors, the refit phase being the larger), else a fixed count.
pub(super) fn genes_per_block(shared: &SharedContrastTensors, memory_fraction: f32) -> usize {
    let refits_per_gene = (shared.number_of_fits - 1).max(1);
    let bytes_per_gene = shared.bytes_per_item() * refits_per_gene
        + shared.member_pair.len() * std::mem::size_of::<f32>();
    match legume_numeric::candle::device::gpu_mem_info(shared.device()) {
        Some((free, _total)) => {
            let room = f64::from(memory_fraction.clamp(0.05, 0.95)) * free as f64;
            ((room / bytes_per_gene as f64) as usize).max(1)
        }
        None => GENES_PER_BLOCK_WITHOUT_MEMORY_QUERY,
    }
}

/// `--device cuda` or `metal`: blocks of `genes_per_block` genes, each
/// block's activities read on CPU threads and fitted on the device.
pub(super) fn fit_genes_in_blocks(
    nonzero_pairs_of_gene: &[Vec<(u32, f32)>],
    inputs: &ActivityInputs,
    shared: &SharedContrastTensors,
    settings: &ContrastFitSettings,
    genes_per_block: usize,
) -> anyhow::Result<Vec<FittedGene>> {
    let progress = indicatif::ProgressBar::new(nonzero_pairs_of_gene.len() as u64);
    let mut fitted = Vec::with_capacity(nonzero_pairs_of_gene.len());
    for block in nonzero_pairs_of_gene.chunks(genes_per_block.max(1)) {
        // each gene's activity on every member of every set
        let member_activity_of_gene: Vec<Vec<f32>> = block
            .par_iter()
            .map_init(
                || inputs.new_scratch(),
                |scratch, nonzero_pairs| {
                    inputs.write_activity_of_gene(nonzero_pairs, scratch);
                    let member_activity = shared
                        .member_pair
                        .iter()
                        .map(|&pair| scratch.activity_of_pair[pair as usize])
                        .collect();
                    clear_activity_of_gene(nonzero_pairs, scratch);
                    member_activity
                },
            )
            .collect();
        fitted.extend(fit_block_of_genes(
            shared,
            &member_activity_of_gene,
            settings,
        )?);
        progress.inc(block.len() as u64);
    }
    progress.finish_and_clear();
    Ok(fitted)
}
