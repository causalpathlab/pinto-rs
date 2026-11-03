//! Shared CLI surface for cutting a pair latent into link communities.
//!
//! `cage`, `dsvd` and `prop` all end in the same place: an
//! `[N_pairs × K_latent]` projection that has to become per-edge community
//! labels. They therefore offer the same flags, with the same meanings and the
//! same defaults, and resolve them through the same [`EdgeClustering`] enum
//! that `compute_propensity_and_feature_community_stat` consumes.

use crate::link_community::profiles::EdgeClustering;
use clap::{Args, ValueEnum};

/// Clusterer for the per-pair latent. `kmeans` (spherical, seeded) fixes the
/// count; `leiden` discovers it from the graph.
///
/// No `Display` impl: clap renders `default_value_t` for a `value_enum` field
/// through `ValueEnum::to_possible_value`, so a hand-written one would be a
/// second copy of the names `rename_all` already derives.
#[derive(ValueEnum, Clone, Copy, Debug)]
#[clap(rename_all = "lowercase")]
pub enum EdgeClusterMethod {
    Kmeans,
    Leiden,
}

/// The flags every pair-latent subcommand shares. Flatten this rather than
/// re-declaring them, so the three commands cannot drift apart.
#[derive(Args, Debug, Clone)]
pub struct EdgeClusterArgs {
    #[arg(
        long,
        default_value_t = EdgeClusterMethod::Leiden,
        value_enum,
        help = "How to cut the pair latent into link communities",
        long_help = "leiden is the default.\n\
                     It builds a cosine kNN graph over the pair latent,\n\
                     and --leiden-resolution then decides how many communities exist.\n\
                     Under leiden, --n-edge-clusters is only a target:\n\
                     the resolution is steered toward it, not fixed at it.\n\
                     \n\
                     kmeans instead fixes the community count at --n-edge-clusters.\n\
                     It is spherical k-means on the L2-normalized pair latent,\n\
                     seeded by --seed, so the cut is reproducible.\n\
                     Pick it when you need a specific K, or a run comparable to an older one.\n\
                     Nothing else about the run changes:\n\
                     both consume the same pair latent,\n\
                     and both write the same three propensity tables."
    )]
    pub edge_cluster_method: EdgeClusterMethod,

    #[arg(
        long,
        default_value_t = 30,
        help = "Neighbours per pair in the Leiden kNN graph over the pair latent"
    )]
    pub leiden_knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution; higher gives more, finer communities"
    )]
    pub leiden_resolution: f64,

    #[arg(
        long,
        default_value_t = 10,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Leiden runs combined into one consensus partition (1 = a single run)",
        long_help = "Leiden's partition moves with its seed: on one cage fit's pair latent,\n\
                     five seeds agreed only at ARI 0.58-0.79, with 12 to 14 communities.\n\
                     So Leiden runs this many times (seeds derived from --seed), each\n\
                     kNN-graph edge is weighted by the share of runs that keep its two\n\
                     pairs together, and one final Leiden on that graph gives the\n\
                     partition. On that latent, 10 runs agreed at ARI 0.80-0.91 across\n\
                     seeds; 30 runs at 0.89-0.95, for more stability at three times the\n\
                     work. The runs share the cores. 1 runs Leiden once, as before.\n\
                     Ignored under kmeans."
    )]
    pub leiden_runs: usize,

    #[arg(
        long,
        help = "Link communities to cut from the pair latent [default: let leiden decide]",
        long_help = "How many edge clusters to cut from the pair latent.\n\
                     Under the default --edge-cluster-method leiden this is a TARGET:\n\
                     the resolution is steered toward it, and omitting it lets\n\
                     --leiden-resolution alone decide.\n\
                     Under kmeans it is the exact count, defaulting to the latent width.\n\
                     \n\
                     A cell's propensity is its incident-edge fraction, taken per community.\n\
                     This is the definition `pinto lc` and `pinto dsvd` use.\n\
                     See this subcommand's own --help for the files it writes."
    )]
    pub n_edge_clusters: Option<usize>,

    #[arg(
        long,
        alias = "maxiter-clustering",
        default_value_t = 100,
        help = "Lloyd iterations for kmeans; ignored under leiden"
    )]
    pub kmeans_max_iter: usize,
}

impl EdgeClusterArgs {
    /// Resolve to the algorithm-side enum with the run's seed. An unset
    /// `--n-edge-clusters` stays `None` in both arms: k-means falls back to
    /// the latent width, which [`EdgeClustering::cluster`] reads off the
    /// matrix it is handed, and for Leiden an absent target is exactly what
    /// lets the graph decide.
    pub fn resolve(&self, seed: u64) -> EdgeClustering {
        match self.edge_cluster_method {
            EdgeClusterMethod::Kmeans => EdgeClustering::Kmeans {
                n_clusters: self.n_edge_clusters,
                max_iter: self.kmeans_max_iter,
                seed,
            },
            EdgeClusterMethod::Leiden => EdgeClustering::Leiden {
                knn: self.leiden_knn,
                resolution: self.leiden_resolution,
                target: self.n_edge_clusters,
                runs: self.leiden_runs,
                seed,
            },
        }
    }
}
