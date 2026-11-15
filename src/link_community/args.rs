//! CLI arguments for the link community model.
//!
//! Split out of `fit.rs` so the orchestrator stays focused on pipeline flow.

use clap::Parser;

#[derive(Parser, Debug, Clone)]
pub struct SrtLinkCommunityArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[command(flatten)]
    pub ablation: crate::util::ablation::AblationArgs,

    #[arg(
        long,
        default_value_t = 50,
        help = "Number of spatial link communities to discover",
        long_help = "Number of link communities (K).\n\
                     Collapsed Gibbs sampling assigns each cell pair to one.\n\
                     Communities capture distinct spatial expression patterns.\n\
                     Cell propensity is the fraction of edges per community.\n\
                     \n\
                     Prefer over-shooting K to under-shooting it.\n\
                     The default of 50 leans on the cosine dictionary-merge pass.\n\
                     That pass collapses redundant feature programs."
    )]
    pub n_communities: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Gibbs iterations at the coarsest level",
        long_help = "Gibbs iterations at the coarsest coarsening level.\n\
                     Later V-cycle levels, and the full-resolution EM, run a fraction\n\
                     of these. They can afford fewer: each warm-starts from the level above.",
        hide = true
    )]
    pub num_gibbs: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Max greedy refinement sweeps after Gibbs",
        long_help = "Maximum greedy (argmax) sweeps after Gibbs sampling.\n\
                     Each sweep moves every edge to its best community.\n\
                     Sweeping stops early once no edge moves.\n\
                     This typically converges in 2-5 sweeps.",
        hide = true
    )]
    pub num_greedy: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Min total count to include a feature in the projection basis",
        long_help = "Features under this total count are zeroed in the basis.\n\
                     That removes them from every profile dimension.\n\
                     Feature-pair mode ignores this flag. Pass 0 to include all features.",
        hide = true
    )]
    pub min_feature_count: f32,

    #[arg(
        long,
        help = "Feature network file (two columns of feature pairs); profiles cell pairs by feature modules",
        long_help = "A feature-feature network you supply: a two-column text file, one pair of\n\
                     features per line, named as the data's features (tab-separated, or\n\
                     comma-separated when it ends in .csv). A line naming a feature the data\n\
                     lacks, such as a header, is skipped. pinto ships no network of its own.\n\n\
                     With it, lc groups the features into modules: the network, densified by\n\
                     shared neighbours and k-core trimmed, is clustered by Leiden (Traag,\n\
                     Waltman and van Eck 2019). Each cell pair's profile is then sparse over module pairs (a, b): how far the two cells co-express a\n\
                     and b beyond what their totals predict, in place of the random-projection\n\
                     profile. A feature trimmed away joins no module.\n\
                     Writes {out}.feature_graph.parquet, the feature pairs kept."
    )]
    #[arg(alias = "gene-network")]
    pub feature_network: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Allow prefix matching for feature names in external network",
        hide = true
    )]
    #[arg(alias = "gene-network-allow-prefix")]
    pub feature_network_allow_prefix: bool,

    #[arg(
        long,
        default_value = "_",
        help = "Delimiter for splitting compound feature names",
        hide = true
    )]
    #[arg(alias = "gene-network-delimiter")]
    pub feature_network_delimiter: Option<char>,

    #[arg(
        long,
        help = "Cosine cutoff for cutting the dictionary-merge tree (unset = no cut)",
        long_help = "Merges at or above this cosine collapse into one community.\n\
                     Merges below the cutoff stay separate.\n\
                     \n\
                     Unset (the default), the tree is not cut: the merge tree is\n\
                     written to <out>.dict_merges.parquet and the draft partition\n\
                     is the final result, so no community is lost to a threshold.\n\
                     The tree says which communities are alike, and a cut can be\n\
                     taken from it later.\n\
                     \n\
                     Cosine runs on per-feature-centred log-rates of the\n\
                     feature-community posterior, restricted to detected features\n\
                     (a detection cutoff read off the data). With several sections (data files,\n\
                     crossed with --batch-files labels), each section's mean\n\
                     profile is removed first, so communities are compared by\n\
                     how they differ from their own section. This holds under\n\
                     --ignore-batch too: it is how communities are compared, not\n\
                     a batch effect.\n\
                     \n\
                     If a cut collapses far more than expected, check the\n\
                     feature filter: undetected features inflate every pairwise\n\
                     cosine.\n\
                     \n\
                     A cut lands in <out>.dict_merges.cut.parquet.\n\
                     Its columns are (community, consensus).\n\
                     Empty communities get consensus = −1."
    )]
    pub merge_cut: Option<f64>,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable per-level cascade outputs (still runs the V-cycle)",
        long_help = "By default the V-cycle emits per-level outputs:\n\
                     \n  \
                     <out>.L{l}.link_community.parquet\n  \
                     <out>.L{l}.propensity.parquet\n  \
                     <out>.L{l}.feature_community.parquet\n\
                     They let you inspect the clustering at every resolution.\n\
                     \n\
                     Pass this flag to skip those writes.\n\
                     Only the final fine-resolution outputs are then emitted,\n\
                     matching the pre-V-cycle behaviour. The cascade still runs internally."
    )]
    pub no_level_outputs: bool,
}

/// The Dirichlet prior on community sizes: one pseudo-edge per community,
/// washed out by the thousands of edges any tissue has.
pub const DIRICHLET_ALPHA: f64 = 1.0;

/// Shared neighbours a feature pair needs for the SNN augmentation of a
/// feature network (0 would disable it).
pub const SNN_MIN_SHARED: usize = 3;

/// k-core degree a feature keeps in the feature network; a feature below it
/// joins no module and no module-pair basis entry.
pub const FEATURE_TRIM_MIN_DEGREE: usize = 3;

/// Leiden resolution of the feature modules cut from a feature network.
pub const FEATURE_MODULES_RESOLUTION: f64 = 1.0;

/// Modularity resolution of the coarsening's merge veto (0 would disable
/// the veto).
pub const MODULARITY_GAMMA: f32 = 1.0;

/// The Gamma(a, b) prior on the community incidence B.
pub const INCIDENCE_A: f64 = 1.0;
pub const INCIDENCE_B: f64 = 1.0;

/// Full-resolution EM sweeps after the cascade, from its Gibbs budget.
pub fn num_em_sweeps(num_gibbs: usize) -> usize {
    (num_gibbs / 4).max(5)
}
