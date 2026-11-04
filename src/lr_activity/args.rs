//! CLI arguments for `pinto lra` and `pinto lra-simulate`.

use clap::{Args, Parser, ValueEnum, ValueHint};

/// `--seed`'s default.
pub const DEFAULT_SEED: u64 = 42;

/// `--fdr`'s default.
pub const DEFAULT_FDR: f64 = 0.05;

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
        long = "knn-spatial",
        short = 'k',
        default_value_t = 5,
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
                     a complex joining its subunits with `_` (TGFB1:TGFBR1_TGFBR2), or a\n\
                     bare LIGAND for all of its pairs. Matched on genes, case aside, among\n\
                     the reference pairs and --lr-pairs; an entry no pair answers is an\n\
                     error naming the nearest. The subset goes to\n\
                     {out}.lra.only-<tag>.parquet (and .curve.parquet), never over the full\n\
                     table, and is recorded apart in the manifest (outputs.lra.subsets).\n\
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
                     a ligand or receptor (a complex, TGFBR1_TGFBR2) keys that gene;\n\
                     otherwise it splits at the `_` that names a listed pair, else at\n\
                     the first `_`."
    )]
    pub targets: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = DEFAULT_FDR,
        help = "False discovery rate for the verdicts",
        long_help = "False discovery rate for the contact-effect verdicts.\n\
                     The effect, the co-expression score and the negative controls each\n\
                     get Benjamini-Hochberg q-values across pairs, pooled and within\n\
                     each section; a test counts when its q is below this level."
    )]
    pub fdr: f64,

    #[arg(
        long,
        value_enum,
        default_value_t = ReferenceSpecies::Auto,
        help = "Species of the reference; auto reads it from the gene-name case"
    )]
    pub species: ReferenceSpecies,

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
        value_name = "JSON",
        value_hint = ValueHint::FilePath,
        help = "Study design: scenarios, replicates, pairs [default: the built-in study]",
        long_help = "Study design as JSON: the scenario grid (confounder kinds, length\n\
                     scales, shapes, effect sizes, spillover, field shift, marker,\n\
                     heterogeneous and ring scenarios), replicates and pairs. Fields left\n\
                     out keep the built-in study's values. See --example-study."
    )]
    pub study: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Print the built-in study as commented JSON and exit"
    )]
    pub example_study: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = ReferenceSpecies::Auto,
        help = "Species of the reference; auto reads it from the gene-name case"
    )]
    pub species: ReferenceSpecies,
}

// * Removed flags

/// Flags of the released `pinto lr-activity` that `pinto lra` no longer
/// takes, with what replaced each.
pub const REMOVED_LRA_FLAGS: &[(&str, &str)] = &[
    ("--edge-scores-only", "use `--method coexpression-scores`"),
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
