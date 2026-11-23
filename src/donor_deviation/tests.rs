//! Small checks of the algorithm's parts: the gradient, the proximal step,
//! the section factor, the matching, and what the contrast recovers on a
//! planted three-donor set (no full fits, no performance thresholds).

use super::contrast_fit::*;
use super::matched_other_donor_pairs::*;
use super::same_section_gene_factor::*;
use crate::util::common::*;

/// Pairs on a line: `pairs_per_donor` per donor at states 0..1, each pair's
/// state coordinates (intercept, state) and its donor.
fn pairs_on_a_line(
    number_of_donors: usize,
    pairs_per_donor: usize,
) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
    let mut state_of_pair = Vec::new();
    let mut donor_of_pair = Vec::new();
    for donor in 0..number_of_donors {
        for i in 0..pairs_per_donor {
            state_of_pair.push(i as f32 / (pairs_per_donor - 1) as f32);
            donor_of_pair.push(donor);
        }
    }
    let coordinates = state_of_pair.iter().flat_map(|&s| [1.0, s]).collect();
    (coordinates, donor_of_pair, state_of_pair)
}

/// Each pair's matched pairs: the pairs of the other donors at the same
/// position on the line (exact matches of state).
/// (distances all 0; one match per other donor).
fn matched_at_same_state(
    number_of_donors: usize,
    pairs_per_donor: usize,
) -> MatchedOtherDonorPairs {
    let mut matched = Vec::new();
    for donor in 0..number_of_donors {
        for i in 0..pairs_per_donor {
            for other in (0..number_of_donors).filter(|&o| o != donor) {
                matched.push((other * pairs_per_donor + i) as u32);
            }
        }
    }
    matched_pairs_from(matched, number_of_donors - 1)
}

/// Matched pairs given as a flat list, `per_pair` each, at distance 0.
fn matched_pairs_from(matched_pair_of_pair: Vec<u32>, per_pair: usize) -> MatchedOtherDonorPairs {
    MatchedOtherDonorPairs {
        distance_to_matched_pair: vec![0.0; matched_pair_of_pair.len()],
        matched_pair_of_pair,
        matched_pairs_per_pair: per_pair,
    }
}

/// Two sections per donor on the line: its even and its odd pairs.
fn two_sections_per_donor(number_of_donors: usize, pairs_per_donor: usize) -> Vec<usize> {
    (0..number_of_donors * pairs_per_donor)
        .map(|p| 2 * (p / pairs_per_donor) + p % 2)
        .collect()
}

#[test]
fn contrast_gradient_matches_finite_differences_with_and_without_a_section() {
    let (coordinates, donors, _) = pairs_on_a_line(3, 5);
    let matched = matched_at_same_state(3, 5);
    let sections = two_sections_per_donor(3, 5);
    let activity: Vec<f32> = (0..15).map(|p| 1.0 + (p % 4) as f32).collect();
    let anchors: Vec<u32> = (0..15).collect();
    let sets = contrast_sets_of_gene(&activity, &anchors, &matched);
    let states = PairStates {
        state_coordinates: &coordinates,
        coordinates_per_pair: 2,
        donor_index_of_pair: &donors,
        number_of_donors: 3,
    };
    let deviation = vec![0.3, -0.2, 0.1, 0.5, -0.4, 0.2];
    let left_out_section_3 = LeftOutSection {
        section_index_of_pair: &sections,
        left_out_section: 3,
    };
    for left_out in [None, Some(&left_out_section_3)] {
        let mut gradient = vec![0f32; 6];
        contrast_loss_and_gradient(&deviation, &sets, &states, left_out, &mut gradient);
        let mut scratch = vec![0f32; 6];
        for k in 0..6 {
            let h = 1e-3;
            let (mut up, mut down) = (deviation.clone(), deviation.clone());
            up[k] += h;
            down[k] -= h;
            let numeric = (contrast_loss_and_gradient(&up, &sets, &states, left_out, &mut scratch)
                - contrast_loss_and_gradient(&down, &sets, &states, left_out, &mut scratch))
                / (2.0 * h);
            assert!(
                (numeric - gradient[k]).abs() < 1e-3,
                "coefficient {k}: {numeric} vs {}",
                gradient[k]
            );
        }
    }
}

#[test]
fn leaving_out_a_section_with_no_pairs_changes_nothing() {
    let (coordinates, donors, _) = pairs_on_a_line(3, 5);
    let matched = matched_at_same_state(3, 5);
    let sections = two_sections_per_donor(3, 5);
    let activity: Vec<f32> = (0..15).map(|p| 1.0 + (p % 3) as f32).collect();
    let anchors: Vec<u32> = (0..15).collect();
    let sets = contrast_sets_of_gene(&activity, &anchors, &matched);
    let states = PairStates {
        state_coordinates: &coordinates,
        coordinates_per_pair: 2,
        donor_index_of_pair: &donors,
        number_of_donors: 3,
    };
    let deviation = vec![0.2, 0.1, -0.3, 0.4, 0.0, -0.1];
    let mut gradient = vec![0f32; 6];
    let mut gradient_without_empty = vec![0f32; 6];
    let loss = contrast_loss_and_gradient(&deviation, &sets, &states, None, &mut gradient);
    let empty_section = LeftOutSection {
        section_index_of_pair: &sections,
        left_out_section: 6,
    };
    let loss_without_empty = contrast_loss_and_gradient(
        &deviation,
        &sets,
        &states,
        Some(&empty_section),
        &mut gradient_without_empty,
    );
    assert_eq!(loss, loss_without_empty);
    assert_eq!(gradient, gradient_without_empty);
}

#[test]
fn a_refit_without_a_section_from_the_full_fit_lands_where_one_from_zero_does() {
    let (coordinates, donors, state) = pairs_on_a_line(3, 20);
    let matched = matched_at_same_state(3, 20);
    let sections = two_sections_per_donor(3, 20);
    // donor 2 gains the gene at the high end
    let activity: Vec<f32> = state
        .iter()
        .zip(&donors)
        .map(|(&s, &d)| if d == 2 && s > 0.5 { 40.0 } else { 5.0 })
        .collect();
    let anchors: Vec<u32> = (0..60).collect();
    let sets = contrast_sets_of_gene(&activity, &anchors, &matched);
    let states = PairStates {
        state_coordinates: &coordinates,
        coordinates_per_pair: 2,
        donor_index_of_pair: &donors,
        number_of_donors: 3,
    };
    let settings = ContrastFitSettings {
        group_lasso_penalty: 0.01,
        max_iterations: 20000,
        gradient_mapping_tolerance: 1e-7,
    };
    let full = fit_deviation_of_gene(&sets, &states, None, None, &settings).deviation;
    let left_out = LeftOutSection {
        section_index_of_pair: &sections,
        left_out_section: 4,
    };
    let from_zero =
        fit_deviation_of_gene(&sets, &states, None, Some(&left_out), &settings).deviation;
    let from_full = super::section_jackknife::deviation_without_each_section(
        &full, &sets, &states, &sections, 6, &settings,
    );
    for (w, z) in from_full[4].deviation.iter().zip(&from_zero) {
        assert!(
            (w - z).abs() < 1e-3,
            "{:?} vs {from_zero:?}",
            from_full[4].deviation
        );
        assert!(!from_full[4].reached_iteration_cap);
    }
    // leaving out pairs that carry the gain changes the fit
    assert_ne!(from_zero, full);
}

#[test]
fn group_lasso_step_shrinks_each_donor_row_or_zeroes_it() {
    let mut deviation = vec![3.0, 4.0, 0.3, 0.4];
    shrink_each_donor_row_by_group_lasso(&mut deviation, 2, 1.0);
    // row 1: norm 5 -> 4; row 2: norm 0.5 < 1 -> 0
    assert!((deviation[0] - 2.4).abs() < 1e-6 && (deviation[1] - 3.2).abs() < 1e-6);
    assert_eq!(&deviation[2..], &[0.0, 0.0]);
}

#[test]
fn median_with_zeros_counts_the_pairs_without_signal() {
    assert_eq!(median_with_zeros(vec![5.0, 6.0], 5), 0.0);
    assert_eq!(median_with_zeros(vec![5.0, 6.0, 7.0], 5), 5.0);
    assert_eq!(median_with_zeros(vec![2.0], 2), 1.0);
}

#[test]
fn section_factor_takes_a_section_wide_shift_and_leaves_a_minority_domain() {
    // two sections of 10 pairs, each pair matched to the same position in the other
    let matched = matched_pairs_from((0..20u32).map(|p| (p + 10) % 20).collect(), 1);
    let section: Vec<usize> = (0..20).map(|p| p / 10).collect();
    // section 0: three times section 1 everywhere, plus a 3-pair domain x10
    let mut count: Vec<f32> = (0..20).map(|p| if p < 10 { 30.0 } else { 10.0 }).collect();
    for c in &mut count[0..3] {
        *c *= 10.0;
    }
    let all: Vec<u32> = (0..20).collect();
    let factor = log_factor_of_section_for_gene(&count, &all, &matched, &section, &[10, 10]);
    let expected = ((30.0 + LOG_RATIO_PSEUDO_COUNT) / (10.0 + LOG_RATIO_PSEUDO_COUNT)).ln();
    assert!(
        (factor[0] - expected).abs() < 1e-5,
        "section-wide shift: {}",
        factor[0]
    );
    assert!(factor[1] < 0.0, "the other section is the mirror image");
}

#[test]
fn matched_pairs_are_never_of_the_pairs_own_donor() {
    let embedding = Mat::from_fn(12, 2, |pair, dim| ((pair * 7 + dim * 3) % 5) as f32);
    let donors: Vec<usize> = (0..12).map(|p| p % 3).collect();
    let matched = find_matched_other_donor_pairs(&embedding, &donors, 3, 3).unwrap();
    for pair in 0..12 {
        for &m in &matched.matched_pair_of_pair[pair * 3..pair * 3 + 3] {
            assert_ne!(
                donors[m as usize], donors[pair],
                "pair {pair} matched to {m}"
            );
        }
    }
}

#[test]
fn matched_distances_are_euclidean_and_nearest_first() {
    let embedding = Mat::from_fn(12, 2, |pair, dim| ((pair * 7 + dim * 3) % 5) as f32);
    let donors: Vec<usize> = (0..12).map(|p| p % 3).collect();
    let matched = find_matched_other_donor_pairs(&embedding, &donors, 3, 3).unwrap();
    for pair in 0..12 {
        let slots = pair * 3..pair * 3 + 3;
        let distances = &matched.distance_to_matched_pair[slots.clone()];
        assert!(distances.windows(2).all(|w| w[0] <= w[1]), "{distances:?}");
        for (&m, &distance) in matched.matched_pair_of_pair[slots].iter().zip(distances) {
            let euclidean = (embedding.row(pair) - embedding.row(m as usize)).norm();
            assert!(
                (euclidean - distance).abs() < 1e-4,
                "{euclidean} vs {distance}"
            );
        }
    }
}

#[test]
fn state_coordinates_start_with_an_intercept_and_have_unit_variance() {
    let embedding = Mat::from_fn(50, 4, |pair, dim| ((pair * (dim + 1)) % 11) as f32);
    let (coordinates, width) = state_coordinates_of_pairs(&embedding, 2);
    assert_eq!(width, 3);
    let axis_1: Vec<f32> = (0..50).map(|p| coordinates[p * 3 + 1]).collect();
    assert!((0..50).all(|p| coordinates[p * 3] == 1.0));
    let mean = axis_1.iter().sum::<f32>() / 50.0;
    let variance = axis_1.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 49.0;
    assert!(mean.abs() < 1e-4 && (variance - 1.0).abs() < 1e-3);
}

/// Fit one gene on the line: three donors, each pair matched to the same
/// state in the other two donors.
fn fit_on_the_line(activity: &[f32]) -> Vec<f32> {
    let (coordinates, donors, _) = pairs_on_a_line(3, 20);
    let matched = matched_at_same_state(3, 20);
    let anchors: Vec<u32> = (0..60).collect();
    let sets = contrast_sets_of_gene(activity, &anchors, &matched);
    let states = PairStates {
        state_coordinates: &coordinates,
        coordinates_per_pair: 2,
        donor_index_of_pair: &donors,
        number_of_donors: 3,
    };
    fit_deviation_of_gene(
        &sets,
        &states,
        None,
        None,
        &ContrastFitSettings {
            group_lasso_penalty: 0.01,
            max_iterations: 5000,
            gradient_mapping_tolerance: 1e-7,
        },
    )
    .deviation
}

fn row_norm(deviation: &[f32], donor: usize) -> f32 {
    deviation[donor * 2..donor * 2 + 2]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
}

#[test]
fn a_gene_shared_alike_by_every_donor_has_no_deviation() {
    let (_, _, state) = pairs_on_a_line(3, 20);
    let activity: Vec<f32> = state.iter().map(|s| 2.0 + 10.0 * s).collect();
    let deviation = fit_on_the_line(&activity);
    assert!(deviation.iter().all(|&c| c == 0.0), "{deviation:?}");
}

#[test]
fn a_gene_gained_by_one_donor_in_one_state_deviates_in_that_donor_only() {
    let (_, donors, state) = pairs_on_a_line(3, 20);
    // donor 2 carries the gene 8 times more at the high end of the line
    let activity: Vec<f32> = state
        .iter()
        .zip(&donors)
        .map(|(&s, &d)| if d == 2 && s > 0.5 { 40.0 } else { 5.0 })
        .collect();
    let deviation = fit_on_the_line(&activity);
    assert!(row_norm(&deviation, 2) > 1.0, "{deviation:?}");
    assert_eq!(row_norm(&deviation, 0), 0.0, "{deviation:?}");
    assert_eq!(row_norm(&deviation, 1), 0.0, "{deviation:?}");
    // the gain is in the direction of the high states
    assert!(deviation[2 * 2 + 1] > 0.0);
}

/// The two planted genes on the line, fitted with their section jackknife
/// by the rayon path and by the batched candle path on the CPU device.
#[test]
fn batched_candle_fit_agrees_with_the_cpu_fit_and_its_jackknife() {
    use super::batched_contrast_fit::{fit_block_of_genes, SharedContrastTensors};
    use legume_numeric::candle::candle_core::Device;
    let (coordinates, donors, state) = pairs_on_a_line(3, 20);
    let matched = matched_at_same_state(3, 20);
    let sections = two_sections_per_donor(3, 20);
    let states = PairStates {
        state_coordinates: &coordinates,
        coordinates_per_pair: 2,
        donor_index_of_pair: &donors,
        number_of_donors: 3,
    };
    let settings = ContrastFitSettings {
        group_lasso_penalty: 0.01,
        max_iterations: 20000,
        gradient_mapping_tolerance: 1e-7,
    };
    let shared_alike: Vec<f32> = state.iter().map(|s| 2.0 + 10.0 * s).collect();
    let gained_by_donor_2: Vec<f32> = state
        .iter()
        .zip(&donors)
        .map(|(&s, &d)| if d == 2 && s > 0.5 { 40.0 } else { 5.0 })
        .collect();
    let anchors: Vec<u32> = (0..60).collect();
    let shared =
        SharedContrastTensors::new(&anchors, &matched, &states, &sections, 6, &Device::Cpu)
            .unwrap();
    let genes = [shared_alike, gained_by_donor_2];
    let member_activity: Vec<Vec<f32>> = genes
        .iter()
        .map(|activity| {
            shared
                .member_pair
                .iter()
                .map(|&p| activity[p as usize])
                .collect()
        })
        .collect();
    let batched = fit_block_of_genes(&shared, &member_activity, &settings).unwrap();
    assert!(
        batched[0].deviation.iter().all(|&c| c == 0.0),
        "shared alike: {:?}",
        batched[0].deviation
    );
    assert!(row_norm(&batched[1].deviation, 2) > 1.0);
    assert_eq!(row_norm(&batched[1].deviation, 0), 0.0);
    for (activity, fit) in genes.iter().zip(&batched) {
        let sets = contrast_sets_of_gene(activity, &anchors, &matched);
        let deviation = fit_deviation_of_gene(&sets, &states, None, None, &settings).deviation;
        let refits = super::section_jackknife::deviation_without_each_section(
            &deviation, &sets, &states, &sections, 6, &settings,
        );
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3);
        assert!(
            close(&fit.deviation, &deviation),
            "{:?} vs {deviation:?}",
            fit.deviation
        );
        assert_eq!(fit.deviation_without_section.len(), 6);
        for (b, c) in fit.deviation_without_section.iter().zip(&refits) {
            assert!(close(b, &c.deviation), "refit {b:?} vs {:?}", c.deviation);
        }
    }
}

#[test]
fn pb_pairs_keep_self_loops_and_average_counts_and_coordinates() {
    use super::fit_units::{mean_counts_over_units, mean_flat_rows_over_units};
    use super::stage1_inputs::PairCountsByGene;
    // four cells in two super-cells {0, 1} and {2, 3}; pairs 0-1 and 2-3 lie
    // inside one, pairs 1-2 and 0-3 cross
    let cell_pairs = [(0, 1), (2, 3), (1, 2), (0, 3)];
    let super_cell_of_cell = [7, 7, 9, 9];
    let (pb_pairs, unit_of_cell_pair) =
        data_beans::alg::cell_pairs::collapse_pairs(&cell_pairs, &super_cell_of_cell);
    assert_eq!(pb_pairs.pairs, vec![(7, 7), (9, 9), (7, 9)]);
    assert_eq!(unit_of_cell_pair, vec![0, 1, 2, 2]);
    let counts = PairCountsByGene {
        gene_names: vec!["g".into()],
        nonzero_pairs_of_gene: vec![vec![(0, 1.0), (2, 3.0), (3, 5.0)]],
    };
    let averaged = mean_counts_over_units(&counts, &unit_of_cell_pair, &[1, 1, 2]);
    assert_eq!(averaged.nonzero_pairs_of_gene[0], vec![(0, 1.0), (2, 4.0)]);
    let coordinates = [1.0, 0.0, 1.0, 2.0, 1.0, 4.0, 1.0, 6.0];
    let means = mean_flat_rows_over_units(&coordinates, 2, &unit_of_cell_pair, &[1, 1, 2]);
    assert_eq!(means, vec![1.0, 0.0, 1.0, 2.0, 1.0, 5.0]);
}
