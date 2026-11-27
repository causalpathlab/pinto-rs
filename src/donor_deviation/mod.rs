//! Stage 2: which stage-1 domains are specific to a donor, and which genes
//! drive them (`pinto apply donor-deviation`).
//!
//! Stage 1 (a finished cage or dsvd run) embedded the cell pairs in a
//! space shared by every donor. Stage 2 keeps that embedding frozen and asks,
//! gene by gene, whether a donor's pairs carry the gene more than the pairs of
//! OTHER donors that stage 1 puts in the same state. It is a contrast with
//! two kinds of negatives:
//!
//! 1. matched other-donor pairs (`matched_other_donor_pairs`): the anchor's
//!    nearest pairs of other donors in the stage-1 embedding, so the shared
//!    state cancels and only the donor's own departure can win;
//! 2. same-section pairs, in their robust form (`same_section_gene_factor`):
//!    a gene's per-section factor, so anything constant within a section
//!    (depth, an unremoved batch effect, a donor-wide shift) cancels.
//!
//! The counts are pooled raw counts, NOT divided by stage 1's state-aware
//! fold: that fold places each donor's odd cells in one node and divides
//! away their markers, which is exactly what stage 2 looks for.
//!
//! The per-gene fit (`contrast_fit`) gives `deviation[gene, donor]`, a linear
//! function of the pair's state coordinates (group lasso per donor row, so
//! it is 0 unless the data need it; the penalty only says which donor a gain
//! belongs to, it is not a selection). Stage 2 decides nothing: `outputs`
//! writes the deviations and each pair's state coordinates and matched
//! distances, from which every gain, domain summary or ranking follows.

mod batched_contrast_fit;
mod contrast_fit;
mod fit_genes;
mod fit_units;
mod matched_other_donor_pairs;
pub(crate) mod outputs;
mod same_section_gene_factor;
mod section_jackknife;
mod stage1_inputs;
#[cfg(test)]
mod tests;

use crate::util::common::*;
use crate::util::device::ComputeDevice;
use contrast_fit::{ContrastFitSettings, PairStates};
use fit_units::{FitUnitKind, FitUnits};
use rand::SeedableRng;

/// The fit stops when its gradient mapping is below this in norm (the loss
/// is normalized by the gene's total activity, so this is scale-free).
const GRADIENT_MAPPING_TOLERANCE: f32 = 1e-4;

/// `pinto apply donor-deviation`'s long help.
pub const DONOR_DEVIATION_LONG_ABOUT: &str =
    "Stage 2 on a finished stage-1 run (cage or dsvd): which of its domains\n\
     are specific to a donor, and which genes drive them.\n\n\
     The run's cell-pair embedding is held fixed. For each gene, every anchor\n\
     pair is contrasted with its nearest pairs of OTHER donors in that\n\
     embedding (the same shared state elsewhere), on pooled raw counts with\n\
     each section's robust gene factor taken out (what is constant within a\n\
     section: depth, batch, a donor-wide shift). A donor whose pairs carry the\n\
     gene more than their matches gets a deviation, a linear function of the\n\
     pair's state (group lasso: 0 unless the data need it). Needs at least\n\
     three donors.\n\n\
     TYPICAL USE:\n\
     \x20 pinto fit cage s1.zarr.zip,...,s6.zarr.zip -c ... -o stage1\n\
     \x20 pinto apply donor-deviation -m stage1 --donor-of-batch donors.tsv -o stage2\n\n\
     Outputs (no cutoffs; everything else is derived from them):\n\
     \x20 {out}.donor_gene_deviation.parquet  per (gene, donor): the deviation's\n\
     \x20     coefficients, intercept and state_axis_1..\n\
     \x20 {out}.donor_pair_states.parquet     per pair: section, donor, stage-1\n\
     \x20     community, state_axis_1.., distances to its matched other-donor\n\
     \x20     pairs (nearest, mean)\n\
     \x20 {out}.donor_gene_deviation_without_section.parquet  the same, per\n\
     \x20     left-out section (refitted from the full fit): jackknife\n\
     \x20     replicates; a quantity's variance is (S - 1) / S · Σ_s (q_s - mean)²\n\
     \x20 {out}.donor_deviation.json          settings and donors\n\n\
     A pair's log gain for a gene (the fitted log ratio of the gene on the\n\
     pair over its matched other-donor pairs) is its donor's\n\
     intercept + Σ_k state_axis_k(gene, donor) · state_axis_k(pair);\n\
     being linear, its mean over a set of pairs (a domain) is the same\n\
     expression at their mean coordinates.";

#[derive(clap::Args, Debug, Clone)]
pub struct DonorDeviationArgs {
    #[arg(
        long,
        short = 'm',
        help = "Stage-1 run (prefix or its .pinto.json) whose pair embedding is held fixed"
    )]
    pub model: Box<str>,

    #[arg(
        long,
        help = "Two-column file: each batch (section) of the run and its donor",
        long_help = "Two-column text file, one batch per line: the batch label as\n\
                     the run names it (coord_pairs' left_batch) and its donor,\n\
                     separated by a tab, a comma or spaces. A `batch donor` header\n\
                     is skipped. At least three donors are needed."
    )]
    pub donor_of_batch: Box<str>,

    #[arg(long, short = 'o', help = "Output prefix")]
    pub out: Box<str>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Data files (default: those the run's manifest records)"
    )]
    pub data: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = FitUnitKind::Cell,
        value_enum,
        help = "What the contrast is fitted on: cell pairs, or pb pairs",
        long_help = "What the contrast is fitted on. cell: every cell pair of the\n\
                     run. pb: the cell pairs collapsed by their cells' super-cells\n\
                     (the run's cell_pb; a cell pair inside one super-cell is its\n\
                     self-loop), counts and coordinates averaged per cell pair: far fewer\n\
                     units to fit. Either way the deviations are read out on every\n\
                     cell pair at its own state coordinates."
    )]
    pub pairs: FitUnitKind,

    #[arg(
        long,
        default_value_t = 10,
        help = "Matched other-donor pairs per anchor pair (the contrast's negatives)"
    )]
    pub matched_pairs: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "Leading axes of the stage-1 embedding a deviation may vary along"
    )]
    pub state_axes: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Group-lasso penalty on each (gene, donor) deviation"
    )]
    pub group_lasso_penalty: f32,

    #[arg(
        long,
        default_value_t = 50_000,
        help = "Anchor pairs the contrast uses at most (sampled; every pair is read out)"
    )]
    pub max_anchor_pairs: usize,

    #[arg(
        long,
        default_value_t = 500,
        help = "FISTA iterations per gene at most"
    )]
    pub max_iterations: usize,

    #[arg(long, default_value_t = DEFAULT_SEED, help = SEED_HELP)]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = ComputeDevice::Cpu,
        value_enum,
        help = "Compute device for the per-gene fits",
        long_help = "Compute device for the per-gene fits and their section\n\
                     jackknife. cpu: one gene per thread. cuda or metal: blocks of\n\
                     genes fitted together on the device (one run per GPU at a\n\
                     time); results agree with cpu to the fit's tolerance, not\n\
                     bit for bit."
    )]
    pub device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "Device ordinal (for cuda/metal)",
        hide = true
    )]
    pub device_no: usize,

    #[arg(
        long,
        default_value_t = 0.8,
        help = "Fraction of free GPU memory a block of genes may take",
        hide = true
    )]
    pub gpu_mem_fraction: f32,
}

pub fn run_donor_deviation(args: &DonorDeviationArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.matched_pairs >= 1 && args.max_anchor_pairs >= 1,
        "--matched-pairs and --max-anchor-pairs must be at least 1"
    );
    let run = crate::util::metadata::Run::open(&args.model)?;
    run.refuse_residual_fit()?;
    info!("Stage-1 run {} ({})", run.prefix, run.meta.command);
    let pairs = stage1_inputs::read_stage1_cell_pairs(&run)?;
    let donor_of_section = stage1_inputs::read_donor_of_section(&args.donor_of_batch)?;
    let donors = stage1_inputs::donors_of_pairs(&pairs.section_of_pair, &donor_of_section)?;
    let number_of_pairs = pairs.number_of_pairs();
    let number_of_donors = donors.donor_names.len();
    info!(
        "{number_of_pairs} cell pairs in {number_of_donors} donors; stage-1 embedding of {} dims",
        pairs.stage1_pair_embedding.ncols()
    );

    let data_files = match &args.data {
        Some(files) => files.clone(),
        None => recorded_data_files(&run)?,
    };
    let data = data_beans::aux::data_loading::read_data_on_shared_rows(
        data_beans::aux::data_loading::ReadSharedRowsArgs {
            data_files,
            keep_empty_barcodes: true,
            ..Default::default()
        },
    )?
    .data;
    let cell_pair_counts = stage1_inputs::pool_pair_counts_by_gene(&data, &pairs)?;
    drop(data);

    // state coordinates: the cell-pair embedding's leading axes; a pb pair
    // sits at its cell pairs' mean, so both levels share them
    let (cell_pair_state_coordinates, coordinates_per_pair) =
        matched_other_donor_pairs::state_coordinates_of_pairs(
            &pairs.stage1_pair_embedding,
            args.state_axes,
        );
    let units = match args.pairs {
        FitUnitKind::Cell => {
            FitUnits::cell_pairs(&pairs, cell_pair_counts, &cell_pair_state_coordinates)
        }
        FitUnitKind::Pb => {
            let pb_units = FitUnits::pb_pairs(
                &run,
                &pairs,
                &cell_pair_counts,
                &cell_pair_state_coordinates,
                coordinates_per_pair,
            )?;
            drop(cell_pair_counts);
            pb_units
        }
    };
    let number_of_units = units.number_of_units();
    let unit_donors = stage1_inputs::donors_of_pairs(&units.section_of_unit, &donor_of_section)?;

    let matched = matched_other_donor_pairs::find_matched_other_donor_pairs(
        &units.embedding_of_unit,
        &unit_donors.donor_index_of_pair,
        number_of_donors,
        args.matched_pairs,
    )?;
    let pairs_matching_pair = matched.pairs_matching_each_pair(number_of_units);
    let unit_states = PairStates {
        state_coordinates: &units.state_coordinates_of_unit,
        coordinates_per_pair,
        donor_index_of_pair: &unit_donors.donor_index_of_pair,
        number_of_donors,
    };

    let sections = index_sections(&units.section_of_unit);
    let anchor_pairs = sample_anchor_pairs(number_of_units, args.max_anchor_pairs, args.seed);
    let settings = ContrastFitSettings {
        group_lasso_penalty: args.group_lasso_penalty,
        max_iterations: args.max_iterations,
        gradient_mapping_tolerance: GRADIENT_MAPPING_TOLERANCE,
    };
    info!(
        "Fitting {} genes on {number_of_units} {} pairs: {} anchors, {} matched each, {} state coordinates",
        units.counts.gene_names.len(),
        args.pairs,
        anchor_pairs.len(),
        args.matched_pairs,
        coordinates_per_pair
    );
    let activity_inputs = fit_genes::ActivityInputs {
        matched: &matched,
        pairs_matching_pair: &pairs_matching_pair,
        sections: &sections,
    };
    let fitted_genes: Vec<outputs::FittedGene> = match args.device {
        ComputeDevice::Cpu => fit_genes::fit_genes_on_cpu_threads(
            &units.counts.nonzero_pairs_of_gene,
            &activity_inputs,
            &anchor_pairs,
            &unit_states,
            &settings,
        ),
        ComputeDevice::Cuda | ComputeDevice::Metal => {
            let device = args.device.to_device(args.device_no)?;
            let shared = batched_contrast_fit::SharedContrastTensors::new(
                &anchor_pairs,
                &matched,
                &unit_states,
                &sections.section_index_of_pair,
                sections.section_names.len(),
                &device,
            )?;
            let genes_per_block = fit_genes::genes_per_block(&shared, args.gpu_mem_fraction);
            info!("On {}: {genes_per_block} genes per block", args.device);
            fit_genes::fit_genes_in_blocks(
                &units.counts.nonzero_pairs_of_gene,
                &activity_inputs,
                &shared,
                &settings,
                genes_per_block,
            )?
        }
    };

    let fits_at_cap: usize = fitted_genes
        .iter()
        .map(|fit| fit.fits_at_iteration_cap)
        .sum();
    if fits_at_cap > 0 {
        warn!(
            "{fits_at_cap} fits (full or jackknife) reached --max-iterations {} before \
             converging; raise it for their deviations to be at the optimum",
            args.max_iterations
        );
    }

    // read out on every cell pair, at its own coordinates
    let cell_pair_states = PairStates {
        state_coordinates: &cell_pair_state_coordinates,
        coordinates_per_pair,
        donor_index_of_pair: &donors.donor_index_of_pair,
        number_of_donors,
    };
    outputs::write_donor_deviation_outputs(
        &args.out,
        &outputs::DonorDeviationResults {
            gene_names: &units.counts.gene_names,
            donor_names: &donors.donor_names,
            section_names: &sections.section_names,
            pairs: &pairs,
            cell_pair_states: &cell_pair_states,
            units: &units,
            matched: &matched,
            fitted_genes: &fitted_genes,
        },
    )?;
    outputs::write_settings_record(args, &donors.donor_names, &settings, anchor_pairs.len())?;
    Ok(())
}

/// The data files the run's manifest records, each found beside it.
fn recorded_data_files(run: &crate::util::metadata::Run) -> anyhow::Result<Vec<Box<str>>> {
    let recorded = run.meta.data_files.clone().unwrap_or_default();
    anyhow::ensure!(
        !recorded.is_empty(),
        "{} records no data files; pass --data",
        run.manifest.display()
    );
    recorded
        .iter()
        .map(|f| {
            run.resolve(f)
                .map(|p| p.to_string_lossy().into())
                .ok_or_else(|| {
                    anyhow::anyhow!("data file {f} (recorded in the run) not found; pass --data")
                })
        })
        .collect()
}

/// The run's sections, indexed in order of first appearance.
struct IndexedSections {
    section_index_of_pair: Vec<usize>,
    number_of_pairs_in_section: Vec<usize>,
    section_names: Vec<Box<str>>,
}

fn index_sections(section_of_pair: &[Box<str>]) -> IndexedSections {
    use crate::util::ablation::{dense_ids, section_sizes};
    let section_id_of_pair = dense_ids(section_of_pair.iter());
    let number_of_pairs_in_section = section_sizes(&section_id_of_pair);
    let mut section_names: Vec<Box<str>> = vec!["".into(); number_of_pairs_in_section.len()];
    for (name, &id) in section_of_pair.iter().zip(&section_id_of_pair) {
        section_names[id as usize] = name.clone();
    }
    IndexedSections {
        section_index_of_pair: section_id_of_pair
            .into_iter()
            .map(|id| id as usize)
            .collect(),
        number_of_pairs_in_section,
        section_names,
    }
}

fn sample_anchor_pairs(number_of_pairs: usize, max_anchor_pairs: usize, seed: u64) -> Vec<u32> {
    if number_of_pairs <= max_anchor_pairs {
        return (0..number_of_pairs as u32).collect();
    }
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut anchors: Vec<u32> =
        rand::seq::index::sample(&mut rng, number_of_pairs, max_anchor_pairs)
            .into_iter()
            .map(|p| p as u32)
            .collect();
    anchors.sort_unstable();
    anchors
}
