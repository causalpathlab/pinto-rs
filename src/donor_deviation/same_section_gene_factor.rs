//! The second kind of negative: what is constant within a section.
//!
//! Matched other-donor pairs alone would flag a gene that is simply higher
//! everywhere in one section or donor (depth, an unremoved batch effect, a
//! donor-wide shift): every one of its pairs beats its matches. Same-section
//! negatives cancel such a constant. Their robust form is used here: each
//! (gene, section) factor is the MEDIAN, over the section's pairs, of the log
//! ratio of the pair's count to the mean count of its matched other-donor
//! pairs. A shift shared by the whole section moves that median; a domain
//! covering a minority of the section's pairs does not, so its gain is left
//! for the contrast. (The exact form, each gene's share of the section's
//! reads, also cancels the constant but turns a gene concentrated in one
//! donor's island into a depletion everywhere else in that donor.)

use super::matched_other_donor_pairs::MatchedOtherDonorPairs;

/// Pseudo-count added to both sides of each pair's log ratio, so pairs with
/// no reads on either side give a ratio of 1 instead of an undefined one.
pub const LOG_RATIO_PSEUDO_COUNT: f32 = 0.5;

/// Each section's log factor for one gene: the median over the section's
/// pairs of log((count + 0.5) / (mean matched count + 0.5)).
/// `count_of_pair`: the gene's pooled count on every pair (dense);
/// `pairs_with_signal`: the pairs whose own or matched count is nonzero (a
/// pair outside it has a log ratio of exactly 0); `matched`: each pair's
/// matched other-donor pairs.
pub fn log_factor_of_section_for_gene(
    count_of_pair: &[f32],
    pairs_with_signal: &[u32],
    matched: &MatchedOtherDonorPairs,
    section_index_of_pair: &[usize],
    number_of_pairs_in_section: &[usize],
) -> Vec<f32> {
    let mut log_ratios_of_section: Vec<Vec<f32>> =
        vec![Vec::new(); number_of_pairs_in_section.len()];
    for &pair in pairs_with_signal {
        let pair = pair as usize;
        let matched_of_pair = matched.matched_of(pair);
        let mean_matched_count = matched_of_pair
            .iter()
            .map(|&m| count_of_pair[m as usize])
            .sum::<f32>()
            / matched_of_pair.len() as f32;
        log_ratios_of_section[section_index_of_pair[pair]].push(
            ((count_of_pair[pair] + LOG_RATIO_PSEUDO_COUNT)
                / (mean_matched_count + LOG_RATIO_PSEUDO_COUNT))
                .ln(),
        );
    }
    log_ratios_of_section
        .into_iter()
        .zip(number_of_pairs_in_section)
        .map(|(ratios, &n_pairs)| median_with_zeros(ratios, n_pairs))
        .collect()
}

/// The median of `values` together with zeros up to `total_count` entries
/// (the section's pairs without signal, whose log ratio is 0), in linear
/// time: in rank order the negative values come first, then the zeros, then
/// the positive values, so only a nonzero rank needs a selection.
pub fn median_with_zeros(values: Vec<f32>, total_count: usize) -> f32 {
    if total_count == 0 {
        return 0.0;
    }
    let (mut negative, mut positive): (Vec<f32>, Vec<f32>) = values
        .into_iter()
        .filter(|&v| v != 0.0)
        .partition(|&v| v < 0.0);
    let zeros = total_count - negative.len() - positive.len();
    let mut value_at_rank = |rank: usize| -> f32 {
        if rank < negative.len() {
            *negative.select_nth_unstable_by(rank, f32::total_cmp).1
        } else if rank < negative.len() + zeros {
            0.0
        } else {
            let rank = rank - negative.len() - zeros;
            *positive.select_nth_unstable_by(rank, f32::total_cmp).1
        }
    };
    let mid = total_count / 2;
    if total_count % 2 == 1 {
        value_at_rank(mid)
    } else {
        0.5 * (value_at_rank(mid - 1) + value_at_rank(mid))
    }
}
