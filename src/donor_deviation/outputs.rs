//! What stage 2 writes: only what cannot be recomputed cheaply from the rest.
//! Every gain, domain summary or ranking is derived downstream from the two
//! tables, with no cutoff decided here.
//!
//! A pair's LOG GAIN for a gene is its donor's deviation at the pair's state
//! coordinates, the fitted log ratio of the gene's activity on the pair to
//! that on its matched other-donor pairs:
//!
//! ```text
//! log_gain(gene, pair) = intercept[gene, donor(pair)]
//!                      + Σ_k state_axis_k[gene, donor(pair)] · state_axis_k[pair]
//! ```
//!
//! It is linear in the pair's coordinates, so its mean over any set of a
//! donor's pairs (a stage-1 community, say) is the deviation at their mean
//! coordinates.
//!
//! * `{out}.donor_gene_deviation.parquet`: a row per (gene, donor), the
//!   deviation's coefficients (`intercept`, `state_axis_1..`).
//! * `{out}.donor_pair_states.parquet`: a row per cell pair, its section,
//!   donor and stage-1 community, its state coordinates (`state_axis_1..`;
//!   the intercept's coordinate is 1), and its distances to its matched
//!   other-donor pairs in the stage-1 embedding (nearest and mean: large
//!   where no other donor has the pair's state).
//! * `{out}.donor_gene_deviation_without_section.parquet`: a row per (gene,
//!   donor, left-out section), the deviation's coefficients estimated
//!   without that section's pairs (refitted from the full fit,
//!   `section_jackknife`): the jackknife replicates downstream takes any
//!   quantity's uncertainty from.
//! * `{out}.donor_deviation.json`: the settings and donors of the run.

use super::contrast_fit::{DeviationOfGene, PairStates};
use super::fit_units::FitUnits;
use super::matched_other_donor_pairs::MatchedOtherDonorPairs;
use super::stage1_inputs::Stage1CellPairs;
use super::DonorDeviationArgs;
use crate::util::common::*;
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::matrix::parquet::{write_named_table, Column};

/// `state_axis_1`, `state_axis_2`, ...: the names of the state axes, shared
/// by the gene table's coefficients and the pair table's coordinates.
fn state_axis_names(number_of_state_axes: usize) -> Vec<Box<str>> {
    (1..=number_of_state_axes)
        .map(|a| format!("state_axis_{a}").into())
        .collect()
}

/// One gene's fit: its deviation, and its deviation without each section
/// (in section index order).
pub(crate) struct FittedGene {
    pub deviation: DeviationOfGene,
    pub deviation_without_section: Vec<DeviationOfGene>,
    /// Of its full fit and refits, how many reached `--max-iterations`.
    pub fits_at_iteration_cap: usize,
}

/// Everything the tables are written from.
pub(crate) struct DonorDeviationResults<'a> {
    pub gene_names: &'a [Box<str>],
    pub donor_names: &'a [Box<str>],
    pub section_names: &'a [Box<str>],
    pub pairs: &'a Stage1CellPairs,
    /// The cell pairs' own state coordinates and donors (the read-out).
    pub cell_pair_states: &'a PairStates<'a>,
    /// The units the fit was on, and each cell pair's unit.
    pub units: &'a FitUnits<'a>,
    pub matched: &'a MatchedOtherDonorPairs,
    pub fitted_genes: &'a [FittedGene],
}

pub(crate) fn write_donor_deviation_outputs(
    out: &str,
    results: &DonorDeviationResults,
) -> anyhow::Result<()> {
    mkdir_parent(out)?;
    write_gene_table(
        &format!("{out}.donor_gene_deviation.parquet"),
        results,
        WhichFits::FullFits,
    )?;
    write_gene_table(
        &format!("{out}.donor_gene_deviation_without_section.parquet"),
        results,
        WhichFits::RefitsWithoutEachSection,
    )?;
    write_pair_state_table(out, results)?;
    Ok(())
}

/// Which of each gene's fits a gene table holds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WhichFits {
    /// The full fit: a row per (gene, donor).
    FullFits,
    /// The jackknife refits: a row per (gene, donor, left-out section).
    RefitsWithoutEachSection,
}

/// A gene table: per row the gene, the donor, the left-out section (refits
/// only) and the deviation's coefficients (`intercept`, `state_axis_1..`).
fn write_gene_table(
    path: &str,
    results: &DonorDeviationResults,
    which: WhichFits,
) -> anyhow::Result<()> {
    let width = results.cell_pair_states.coordinates_per_pair;
    let mut gene = Vec::new();
    let mut donor = Vec::new();
    let mut left_out_section = Vec::new();
    let mut coefficient_of_column: Vec<Vec<f32>> = vec![Vec::new(); width];
    for (gene_name, fit) in results.gene_names.iter().zip(results.fitted_genes) {
        let deviations: Vec<(Option<&Box<str>>, &DeviationOfGene)> = match which {
            WhichFits::FullFits => vec![(None, &fit.deviation)],
            WhichFits::RefitsWithoutEachSection => results
                .section_names
                .iter()
                .map(Some)
                .zip(&fit.deviation_without_section)
                .collect(),
        };
        for (section_name, deviation) in deviations {
            for (donor_name, row) in results.donor_names.iter().zip(deviation.chunks(width)) {
                gene.push(gene_name.clone());
                donor.push(donor_name.clone());
                if let Some(section_name) = section_name {
                    left_out_section.push(section_name.clone());
                }
                for (column, &c) in coefficient_of_column.iter_mut().zip(row) {
                    column.push(c);
                }
            }
        }
    }
    let mut columns: Vec<(Box<str>, Column)> = vec![("donor".into(), Column::Str(&donor))];
    if which == WhichFits::RefitsWithoutEachSection {
        columns.push(("left_out_section".into(), Column::Str(&left_out_section)));
    }
    let coefficient_names = std::iter::once("intercept".into()).chain(state_axis_names(width - 1));
    for (name, column) in coefficient_names.zip(&coefficient_of_column) {
        columns.push((name, Column::F32(column)));
    }
    write_named_table(path, "gene", &gene, &columns)?;
    info!("Wrote {path}");
    Ok(())
}

fn write_pair_state_table(out: &str, results: &DonorDeviationResults) -> anyhow::Result<()> {
    let DonorDeviationResults {
        donor_names,
        pairs,
        cell_pair_states,
        units,
        matched,
        ..
    } = *results;
    let width = cell_pair_states.coordinates_per_pair;
    let donor: Vec<Box<str>> = cell_pair_states
        .donor_index_of_pair
        .iter()
        .map(|&d| donor_names[d].clone())
        .collect();
    // the coordinates past the intercept, one column per state axis
    let coordinate_of_axis: Vec<Vec<f32>> = (1..width)
        .map(|axis| {
            cell_pair_states
                .state_coordinates
                .chunks(width)
                .map(|coordinates| coordinates[axis])
                .collect()
        })
        .collect();
    // each cell pair's unit's distances to its matches, nearest first
    let distances_of_unit: Vec<&[f32]> = matched
        .distance_to_matched_pair
        .chunks(matched.matched_pairs_per_pair)
        .collect();
    let distances_of_cell_pair = || {
        units
            .unit_of_cell_pair
            .iter()
            .map(|&unit| distances_of_unit[unit])
    };
    let nearest_matched_distance: Vec<f32> = distances_of_cell_pair().map(|d| d[0]).collect();
    let mean_matched_distance: Vec<f32> = distances_of_cell_pair()
        .map(|d| d.iter().sum::<f32>() / d.len() as f32)
        .collect();
    let pb_pair_of_cell_pair: Option<Vec<Box<str>>> =
        units.pb_pair_name_of_unit.as_ref().map(|names| {
            units
                .unit_of_cell_pair
                .iter()
                .map(|&unit| names[unit].clone())
                .collect()
        });
    let axis_names = state_axis_names(width - 1);
    let mut columns: Vec<(Box<str>, Column)> = vec![
        ("right_cell".into(), Column::Str(&pairs.right_cell_of_pair)),
        ("section".into(), Column::Str(&pairs.section_of_pair)),
        ("donor".into(), Column::Str(&donor)),
        ("community".into(), Column::Str(&pairs.community_of_pair)),
        (
            "nearest_matched_distance".into(),
            Column::F32(&nearest_matched_distance),
        ),
        (
            "mean_matched_distance".into(),
            Column::F32(&mean_matched_distance),
        ),
    ];
    if let Some(pb_pair_of_cell_pair) = &pb_pair_of_cell_pair {
        columns.push(("pb_pair".into(), Column::Str(pb_pair_of_cell_pair)));
    }
    for (name, column) in axis_names.iter().zip(&coordinate_of_axis) {
        columns.push((name.clone(), Column::F32(column)));
    }
    let path = format!("{out}.donor_pair_states.parquet");
    write_named_table(&path, "left_cell", &pairs.left_cell_of_pair, &columns)?;
    info!("Wrote {path}");
    Ok(())
}

pub(crate) fn write_settings_record(
    args: &DonorDeviationArgs,
    donor_names: &[Box<str>],
    settings: &super::contrast_fit::ContrastFitSettings,
    number_of_anchor_pairs: usize,
) -> anyhow::Result<()> {
    let record = serde_json::json!({
        "command": "donor-deviation",
        "version": env!("CARGO_PKG_VERSION"),
        "stage1_run": args.model,
        "donor_of_batch": args.donor_of_batch,
        "donors": donor_names,
        "matched_pairs": args.matched_pairs,
        "state_axes": args.state_axes,
        "anchor_pairs": number_of_anchor_pairs,
        "fit": settings,
        "seed": args.seed,
        "pairs": args.pairs,
        "section_jackknife": "each section left out in turn, refitted from the full fit",
    });
    let path = format!("{}.donor_deviation.json", args.out);
    std::fs::write(&path, serde_json::to_string_pretty(&record)?)?;
    info!("Wrote {path}");
    Ok(())
}
