//! CLI arguments for `pinto lra` and `pinto lra-simulate`.

use clap::{Args, Parser, ValueEnum, ValueHint};

/// `--seed`'s default.
pub const DEFAULT_SEED: u64 = 42;

// * Input

/// Where lra reads its cells, expression and spatial contacts: a prior run's
/// manifest (`--from`), or data files with coordinates. Shared by `lra` and
/// `lra-simulate`.
//
// Deliberately NOT the shared `SrtInputArgs`: lra fits nothing, so the QC,
// coarsening and batch-correction options that struct carries would all
// parse and do nothing here. lra declares exactly what it consumes.
#[derive(Args, Debug, Clone, Default)]
pub struct LraInputArgs {
    #[arg(
        value_delimiter(','),
        help_heading = "Input",
        help = "Expression data files (.zarr or .h5); read from --from when left out",
        long_help = "Spatial gene expression data files, comma separated (.zarr or .h5).\n\
                     Each file is a genes-by-cells sparse matrix; several files are\n\
                     concatenated over cells. With --from they are read from the\n\
                     manifest when left out; given here, they win."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long,
        short = 'f',
        value_name = "PINTO_JSON",
        value_hint = ValueHint::FilePath,
        help_heading = "Input",
        help = "A prior pinto run's .pinto.json (lc, cage, dsvd, ...)",
        long_help = "A prior pinto run's manifest ({prefix}.pinto.json) from lc, cage,\n\
                     dsvd or any run that wrote cell pairs. Its data files, coordinates\n\
                     and spatial contacts are reused; relative paths in it are read\n\
                     against the manifest's own directory. The co-expression methods\n\
                     also need its link communities (a `pinto lc` run).\n\
                     Without --out the results go next to the run: {prefix}.lra.*."
    )]
    pub from: Option<Box<str>>,

    #[arg(
        long = "coord",
        short = 'c',
        value_delimiter(','),
        help_heading = "Input",
        help = "Coordinate files, one per data file (without --from)",
        long_help = "Spatial coordinate files, one per data file, comma separated, when\n\
                     reading data files directly instead of --from. The first column\n\
                     names the cells; the next two are x and y. Spatial contacts are\n\
                     the -k nearest neighbours of each cell, within its batch."
    )]
    pub coord_files: Vec<Box<str>>,

    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        help_heading = "Input",
        hide = true,
        help = "Column names of the coordinates in the coordinate files"
    )]
    pub coord_column_names: Vec<Box<str>>,

    #[arg(
        long,
        short = 'b',
        value_delimiter(','),
        help_heading = "Input",
        help = "Batch (section) label files, one per data file (without --from)"
    )]
    pub batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long = "cell-types",
        value_name = "FILE",
        value_hint = ValueHint::FilePath,
        help_heading = "Input",
        help = "Cell types per cell (cell, cell_type), in place of the run's lupin round",
        long_help = "Cell-type labels, one row per cell: a table (TSV, CSV or parquet) with a\n\
                     `cell` column naming the data's cells and a `cell_type` (or `label`)\n\
                     column; without those names, the first two columns. They stand in for\n\
                     the --from run's newest lupin round: confounder columns for the\n\
                     receiver's type and its neighbours' types, and a row per type. For a\n\
                     residual fit, the donor-invariant stage-1 types. Cells it leaves out\n\
                     share the type `unassigned`."
    )]
    pub cell_types: Option<Box<str>>,

    #[arg(
        long,
        value_name = "FILE",
        value_hint = ValueHint::FilePath,
        help_heading = "Input",
        help = "Region per cell (cell, region): adjusted for within each section, and a row per region",
        long_help = "Region labels, one row per cell: a table (TSV, CSV or parquet) with a\n\
                     `cell` column and a `region` (or `label`) column; without those names,\n\
                     the first two columns. Each region gets an indicator per section among\n\
                     the confounders (the smooth fields cannot follow a sharp region edge),\n\
                     and each pair a row per region, `section` = `region:<name>`, its q\n\
                     corrected within that region. For a residual fit, the stage-3\n\
                     communities, best fitted with the pairs' genes dropped so the region is\n\
                     not read off the response. Cells it leaves out are `unassigned`, which\n\
                     gets no row."
    )]
    pub region: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = RegionJackknife::NoRefit,
        help_heading = "Input",
        help = "How a region row's SE is widened near significance",
        long_help = "How a --region row whose effect, or contrast with the rest, has p < 0.2\n\
                     is checked by a delete-one-tile jackknife over the tiles of its\n\
                     sections; its SEs are widened by the jackknife over the clustered SE,\n\
                     never narrowed.\n\
                     no-refit (default): the tile leaves the averages alone, the nuisance\n\
                     fits kept, all replicates read from one pass of them.\n\
                     refit: each replicate refits the nuisance models without the tile,\n\
                     so their own error counts. Conservative, at a large cost in power and\n\
                     one refit per tile.\n\
                     Neither removes confounding finer than the spatial smooth follows: a\n\
                     hidden field at that scale, driving both ligand and response inside a\n\
                     region, biases its row as it biases the pooled one."
    )]
    pub region_jackknife: RegionJackknife,

    #[arg(
        long = "knn-spatial",
        short = 'k',
        default_value_t = 5,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help_heading = "Input",
        help = "Spatial contacts per cell when building them from coordinates"
    )]
    pub knn_spatial: usize,

    #[arg(long, default_value_t = DEFAULT_SEED, help = "Random seed for reproducibility")]
    pub seed: u64,

    #[arg(
        long,
        help = "Cells per parallel block (omit for auto-scaling by feature count)",
        hide = true
    )]
    pub block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        hide = true,
        help = crate::util::input::PRELOAD_HELP,
        long_help = crate::util::input::PRELOAD_LONG_HELP
    )]
    pub preload_data: bool,
}

// * lra

/// What `pinto lra` computes.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LraMethod {
    /// Effect of neighbouring ligand on the receiver's response programme
    #[default]
    ContactEffect,
    /// Permutation test of ligand-receptor co-expression per link community
    CoexpressionTest,
    /// Descriptive co-expression scores per batch and link community, no test
    CoexpressionScores,
}

#[derive(Parser, Debug, Clone)]
pub struct SrtLrActivityArgs {
    #[command(flatten)]
    pub input: LraInputArgs,

    #[arg(
        long,
        short,
        help = "Output prefix [default with --from: the run's prefix]",
        long_help = "Output prefix (e.g. results/my_run). With --from it defaults to the\n\
                     run's own prefix, so the table lands beside the run as\n\
                     {prefix}.lra.parquet."
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = LraMethod::ContactEffect,
        help = "What to compute"
    )]
    pub method: LraMethod,

    #[arg(
        long,
        value_hint = ValueHint::FilePath,
        help = "Ligand-receptor pairs (two-column TSV/CSV), added to the reference",
        long_help = "Ligand-receptor pair file, one pair per line: ligand gene, receptor\n\
                     gene, separated by tab, comma or whitespace. contact-effect adds\n\
                     them to the reference pairs; the co-expression methods test only\n\
                     these and require the file. Pairs naming a gene missing from the\n\
                     data are dropped with a warning."
    )]
    pub lr_pairs: Option<Box<str>>,

    #[arg(
        long,
        value_delimiter(','),
        value_name = "PAIRS",
        help = "Estimate only these pairs: LIGAND:RECEPTOR or a bare LIGAND, comma separated",
        long_help = "Estimate only these pairs (contact-effect), comma separated: LIGAND:RECEPTOR,\n\
                     a complex joining its subunits with `_` (L2:R1_R2), or a\n\
                     bare LIGAND for all of its pairs. Matched on genes, case aside, among\n\
                     the reference pairs and --lr-pairs; an entry no pair answers is an\n\
                     error naming the nearest. The subset goes to\n\
                     {out}.lra.only-<tag>.parquet (and .curve.parquet, and .celltype.parquet\n\
                     with --marginal-baselines), never over the full table, and is recorded\n\
                     apart in the manifest (outputs.lra.subsets).\n\
                     Its q-values are corrected within the subset only (column q_scope).\n\
                     The receivers and each pair's estimate are those of the full run."
    )]
    pub only: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter(','),
        value_hint = ValueHint::FilePath,
        help = "Response programmes (TSV or GMT), added to the reference",
        long_help = "Response programme files, comma separated, added to the reference.\n\
                     TSV/CSV: a header with `receptor` and/or `ligand`, `target`, optional\n\
                     `weight` (signed). GMT: set names are a receptor, a ligand, or\n\
                     `ligand_receptor`. A name with `_` that the pair list uses whole for\n\
                     a ligand or receptor (a complex, R1_R2) keys that gene;\n\
                     otherwise it splits at the `_` that names a listed pair, else at\n\
                     the first `_`."
    )]
    pub targets: Vec<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = ReferenceSpecies::Auto,
        help = "Species of the reference; auto reads it from the gene-name case"
    )]
    pub species: ReferenceSpecies,

    #[arg(
        long,
        default_value_t = false,
        help = "Also report the unadjusted co-expression baselines per pair (permutations: slow)",
        long_help = "Also report, per pair (contact-effect), the co-expression statistics\n\
                     lra-simulate benchmarks the estimate against, none of which adjusts for\n\
                     confounding: the cell-type label permutation test (CellPhoneDB v2,\n\
                     CellChat's product score) on the run's cell types (--cell-types or its\n\
                     lupin round; the identity bins, coarsened to at most 8, when it has\n\
                     none), bivariate Moran's I over contacts (SpatialDM) and the contact\n\
                     co-expression score (NICHES, COMMOT). The pooled rows gain\n\
                     celltype_score, celltype_p, celltype_tested, moran_i, moran_p,\n\
                     contact_score and contact_p, and {out}.lra.celltype.parquet holds\n\
                     every tested sender type x receiver type (recorded in the manifest as\n\
                     outputs.lra.celltype). Off by default: the permutations cost more than\n\
                     the estimate on a full reference."
    )]
    pub marginal_baselines: bool,

    #[arg(
        long,
        value_name = "N",
        default_value_t = crate::lr_activity::ligand_effect::PERMUTATIONS,
        requires = "marginal_baselines",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Permutations of each --marginal-baselines test",
        long_help = "Permutations of each --marginal-baselines permutation test (needs\n\
                     --marginal-baselines). Moran's I and the contact score stop early once\n\
                     their verdict at 5% is settled; the cell-type test runs all N, since\n\
                     its per-type-pair table's p and q need them. Its summary is\n\
                     Bonferroni-corrected over the type pairs, so with m tested type pairs\n\
                     it can reject only when m / (1 + N) < 0.05: many types need more."
    )]
    pub marginal_permutations: usize,

    // * Co-expression methods
    #[arg(
        long,
        default_value_t = 1000,
        help_heading = "Co-expression methods",
        help = "Sample permutations for the null (coexpression-test)"
    )]
    pub n_permutations: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        help_heading = "Co-expression methods",
        help = "FWER cutoff for the JSON sidecar of significant pairs (coexpression-test)",
        long_help = "Westfall-Young FWER cutoff for the JSON sidecar (coexpression-test).\n\
                     A row enters the sidecar when its FWER is below this AND its\n\
                     restandardized z is positive. The parquet always carries the\n\
                     full table."
    )]
    pub json_fwer_threshold: f32,

    #[arg(
        long,
        default_value_t = 50.0,
        help_heading = "Co-expression methods",
        help = "Minimum total count of a ligand or receptor gene for its pair to be scored"
    )]
    pub min_gene_count: f32,

    #[arg(
        long,
        default_value_t = 10,
        help_heading = "Co-expression methods",
        hide = true,
        help = "Random-projection axes for binning cells by propensity"
    )]
    pub propensity_dim: usize,

    #[arg(
        long,
        default_value_t = 4,
        help_heading = "Co-expression methods",
        hide = true,
        help = "Top propensity bits the shuffle is stratified by (0 = unstratified)"
    )]
    pub shuffle_stratify_dim: usize,

    #[arg(
        long,
        default_value_t = 100,
        help_heading = "Co-expression methods",
        hide = true,
        help = "Skip communities with fewer than this many edges"
    )]
    pub min_edges_per_community: usize,

    #[arg(
        long,
        default_value = "_",
        help_heading = "Co-expression methods",
        hide = true,
        help = "Delimiter used to alias compound gene row names (ENSG..._SYMBOL)"
    )]
    pub gene_delimiter: Option<char>,

    #[arg(
        long,
        default_value_t = false,
        help_heading = "Co-expression methods",
        hide = true,
        help = "Allow prefix matching of gene names as a fallback"
    )]
    pub gene_allow_prefix: bool,

    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        help_heading = "Co-expression methods",
        hide = true,
        help = "Also write the JSON sidecar of significant pairs (true/false)"
    )]
    pub emit_json: bool,
}

/// The delete-one-tile jackknife of region rows ([`SrtLrActivityArgs`]'s
/// `--region-jackknife`).
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RegionJackknife {
    /// Each replicate refits the nuisance models without its tile
    Refit,
    /// The tile leaves the average only; the nuisance fits are the full ones
    #[default]
    NoRefit,
}

impl RegionJackknife {
    /// As `--region-jackknife` spells it.
    pub fn name(self) -> String {
        self.to_possible_value()
            .map(|v| v.get_name().to_string())
            .unwrap_or_default()
    }
}

/// Species of the contact-effect reference.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceSpecies {
    Auto,
    Human,
    Mouse,
}

impl ReferenceSpecies {
    /// As `--species` spells it.
    pub fn name(self) -> String {
        self.to_possible_value()
            .map(|v| v.get_name().to_string())
            .unwrap_or_default()
    }
}

// * lra-simulate

#[derive(Parser, Debug, Clone)]
pub struct LraSimulateArgs {
    #[command(flatten)]
    pub input: LraInputArgs,

    #[arg(
        long,
        short,
        help = "Output prefix [default with --from: the run's prefix]"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        value_hint = ValueHint::FilePath,
        help = "Ligand-receptor pairs (two-column TSV/CSV), added to the reference",
        long_help = "Ligand-receptor pair file, as `pinto lra --lr-pairs` takes it, added\n\
                     to the reference pairs the benchmark picks from. Needed for data\n\
                     whose gene names no reference knows (a simulation's GENE1, ...)."
    )]
    pub lr_pairs: Option<Box<str>>,

    #[arg(
        long,
        value_delimiter(','),
        value_hint = ValueHint::FilePath,
        help = "Response programmes (TSV or GMT), added to the reference",
        long_help = "Response programme files, as `pinto lra --targets` takes them: the\n\
                     programmes the benchmark's responses are drawn on."
    )]
    pub targets: Vec<Box<str>>,

    #[arg(
        long,
        value_name = "REGION",
        help = "Plant the effect only on receivers in this --region label",
        long_help = "Plant every scenario's effect only on receivers in this region (a\n\
                     label of the --region table), and read the estimate, its truth and\n\
                     its verdict on that region's row as well as the pooled one. The\n\
                     hidden confounders still act everywhere. Needs --region."
    )]
    pub plant_region: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = ReferenceSpecies::Auto,
        help = "Species of the reference; auto reads it from the gene-name case"
    )]
    pub species: ReferenceSpecies,

    #[arg(
        long,
        help_heading = "Single scenario",
        help = "Run one scenario on planted and null pairs, not the built-in grid",
        long_help = "Run one scenario, not the built-in grid: its effect planted on\n\
                     --pairs pairs and withheld (effect 0) on --null-pairs others, all\n\
                     under the same hidden field, so the one call can be scored by\n\
                     AUROC and AUPRC. --set, --pairs or --null-pairs select it too.\n\
                     Needs -o. Writes {out}.lra-simulate.parquet with a planted label\n\
                     and each method's p, then {out}.lra-simulate.json (the resolved\n\
                     scenario, the pairs and their seeds), whose presence marks the\n\
                     call complete. The run's manifest is not written to."
    )]
    pub one: bool,

    #[arg(
        long = "set",
        value_name = "KEY=VALUE",
        help_heading = "Single scenario",
        help = "Set a key of the single scenario (repeatable); keys are the table's columns",
        long_help = "Set a key of the single scenario, KEY=VALUE, repeatable; the keys are\n\
                     the {out}.lra-simulate.parquet columns that record them:\n\
                     effect_pve, ligand_field_pve, response_field_pve (signed),\n\
                     confound_scale, confounder (fourier, bumps or realgene),\n\
                     confound_shape (linear or quadratic), heterogeneous, marker,\n\
                     dose_response (true/false or 1/0), dose_midpoint, dose_order and\n\
                     rep. Unset keys are no effect and no confounding. A point out of\n\
                     range is refused before any data is read."
    )]
    pub set: Vec<String>,

    #[arg(
        long,
        value_name = "N",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help_heading = "Single scenario",
        help = "Pairs the single scenario's effect is planted on [default: 10]"
    )]
    pub pairs: Option<usize>,

    #[arg(
        long,
        value_name = "N",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help_heading = "Single scenario",
        help = "Further pairs the single scenario withholds its effect from [default: 10]"
    )]
    pub null_pairs: Option<usize>,
}

impl LraSimulateArgs {
    /// The single scenario these flags ask for, checked; `None` for the
    /// built-in grid.
    pub fn single(
        &self,
    ) -> anyhow::Result<Option<crate::lr_activity::ligand_effect::simulate::SingleScenario>> {
        use crate::lr_activity::ligand_effect::simulate::SingleScenario;
        if !(self.one || !self.set.is_empty() || self.pairs.is_some() || self.null_pairs.is_some())
        {
            return Ok(None);
        }
        SingleScenario::new(
            &self.set,
            self.pairs.unwrap_or(SingleScenario::PAIRS),
            self.null_pairs.unwrap_or(SingleScenario::PAIRS),
        )
        .map(Some)
    }
}

// * Removed flags

/// Flags of the released `pinto lr-activity` that `pinto lra` no longer
/// takes, with what replaced each.
pub const REMOVED_LRA_FLAGS: &[(&str, &str)] = &[
    ("--edge-scores-only", "use `--method coexpression-scores`"),
    (
        "--fdr",
        "the level is fixed (outputs.lra.settings in the manifest); the table has every q",
    ),
    (
        "--lc-prefix",
        "pass the run's manifest: `--from {prefix}.pinto.json`",
    ),
];

/// The first removed flag in `tail` (the arguments after `lra`), with its
/// replacement, as an error message.
pub fn removed_lra_flag(tail: &[String]) -> Option<String> {
    tail.iter().find_map(|a| {
        let name = a.split('=').next().unwrap_or(a);
        REMOVED_LRA_FLAGS
            .iter()
            .find(|&&(flag, _)| name == flag)
            .map(|(_, hint)| format!("`{name}` was removed from `pinto lra`: {hint}"))
    })
}
