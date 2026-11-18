//! CLI arguments for `pinto fit cage`.

use clap::{Parser, ValueEnum};
use data_beans::alg::hvg::HvgCliArgs;
use data_beans::aux::feature_names::FeatureNameKind;

use super::pair_projection::encoder::{DEFAULT_PAIR_ENCODER_BATCH, DEFAULT_PAIR_ENCODER_STEPS};
use crate::util::device::ComputeDevice;

/// Which features each epoch trains (`--feature-inclusion`).
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum FeatureInclusionMode {
    /// Every trainable feature, every epoch.
    All,
    /// Each feature with probability π_g, its posterior probability of
    /// spatial signal (jackknifed Moran's I, two-group empirical Bayes).
    Moran,
}

/// Row-name canonicalization strategy for matching the data's feature
/// names against external resources (PPI networks, marker lists,
/// pretrained feature embeddings). `Auto` sniffs the first data file's
/// row names and dispatches to [`FeatureNameKind::auto_detect`] —
/// gene-symbol-style names (`ENSG..._SYMBOL`) get the `Gene` rule
/// applied automatically.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[clap(rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum FeatureNameMode {
    /// Peek the first file's row names and pick `Exact` / `Gene` /
    /// `Locus` / `Mixed` via [`FeatureNameKind::auto_detect`].
    Auto,
    /// Strict string match — no canonicalization. The historical pinto
    /// default for `lc` / `svd`.
    Exact,
    /// `Gene { delim: '_' }`: gene-symbol rows — register every `_`-split
    /// component as an alias of the full row name.
    Gene,
    /// `Locus { merge_overlapping: true }`: normalize chrom-coord names
    /// and collapse overlapping intervals.
    Locus,
    /// Heterogeneous axis: dispatch per row name.
    Mixed,
}

impl FeatureNameMode {
    /// Resolve to a concrete [`FeatureNameKind`]. `peek_names` is only
    /// consulted under `Auto`; other modes ignore it.
    pub fn resolve_kind(self, peek_names: &[Box<str>]) -> FeatureNameKind {
        match self {
            FeatureNameMode::Auto => FeatureNameKind::auto_detect(peek_names),
            FeatureNameMode::Exact => FeatureNameKind::Exact,
            FeatureNameMode::Gene => FeatureNameKind::Gene { delim: '_' },
            FeatureNameMode::Locus => FeatureNameKind::Locus {
                merge_overlapping: true,
            },
            FeatureNameMode::Mixed => FeatureNameKind::Mixed,
        }
    }

    /// The mode that resolves to `kind` (never `Auto`), what a run's
    /// manifest records of how it read its rows' names; `None` for a kind
    /// no mode resolves to.
    pub fn of_kind(kind: &FeatureNameKind) -> Option<Self> {
        let mode = match kind {
            FeatureNameKind::Exact => Self::Exact,
            FeatureNameKind::Gene { delim: '_' } => Self::Gene,
            FeatureNameKind::Locus {
                merge_overlapping: true,
            } => Self::Locus,
            FeatureNameKind::Mixed => Self::Mixed,
            _ => return None,
        };
        Some(mode)
    }
}

/// Positive super-edge draws per (feature, batch) each epoch. A batch's
/// per-feature pool is tens of super edges, so repeated draws are by design;
/// over-sampling can degrade the fit, so `--epochs` carries the budget.
pub const PER_FEATURE_BATCH: usize = 12;

/// cage-cell's positive draws per cell pair per epoch, unless
/// `--positives-per-epoch` sets the total: an epoch draws this many times
/// the pair count, split over the features and batches as that flag's total
/// is (never below [`PER_FEATURE_BATCH`] per feature and batch), so each
/// update of a pair's row averages several draws at any data size. Measured
/// on 9,216 cells at 200 updates per pair: about 1 draw per pair per epoch
/// gave ARI 0.37, 3.5 gave 0.90; more steps of the smaller draw (400 updates)
/// gave 0.72 in about the same time, and splitting an epoch into more steps
/// (a smaller --feature-batch-size) 0.20.
pub const PAIR_DRAWS_PER_EPOCH: usize = 4;

/// Sibling-pseudobulk negatives drawn per positive super edge per chain level.
pub const N_NEGATIVES: usize = 8;

/// Exponent of the power-of-degree negative sampling.
pub const ALPHA_NEG: f32 = 0.75;

/// The NCE objective cage trains, and probes, with: InfoNCE, its negatives
/// competing with the positive in one softmax.
pub const NCE_OBJECTIVE: graph_embedding_util::loss::NceObjective =
    graph_embedding_util::loss::NceObjective::Softmax;

/// Gradient-norm clip of each optimizer step.
pub const GRAD_CLIP: f32 = 1.0;

/// Prior precision the pair encoder is fitted under.
pub const PAIR_RIDGE: f32 = 1.0;

/// Cell pairs placed on the feature embedding per block.
pub const PAIR_BLOCK: usize = 8192;

#[derive(Parser, Debug, Clone)]
// legume-graph-embedding's LoRA tuning flags, hidden here: the rank is a
// size knob, and the learning-rate ratio and ridge stay at upstream's
// defaults until it drops them.
#[command(
    mut_arg("lora_rank", |a| a.hide(true)),
    mut_arg("lora_lr_ratio", |a| a.hide(true)),
    mut_arg("lora_ridge", |a| a.hide(true))
)]
pub struct CellActivityGraphEmbeddingArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[command(flatten)]
    pub stage2: crate::util::input::OffsetArgs,

    #[command(flatten)]
    pub ablation: crate::util::ablation::AblationArgs,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
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
        default_value_t = FeatureNameMode::Auto,
        value_enum,
        help = "Feature-name canonicalization for matching external resources",
        long_help = "Row-name canonicalization strategy:\n\
                     auto:  peek row names and auto-detect (default)\n\
                     exact: strict string equality (pinto fit lc / svd default)\n\
                     gene:  gene symbols; split on '_' so 'ENSG…_SYMBOL' and 'SYMBOL' alias\n\
                     locus: normalize chrom-coord names; collapse overlaps\n\
                     mixed: per-row dispatch (RNA+ATAC paired axes)",
        hide = true
    )]
    pub feature_name_mode: FeatureNameMode,

    #[arg(
        long,
        value_name = "H",
        default_value_t = graph_embedding_util::EmbeddingDim::Auto,
        value_name = "H|auto",
        help = "Cell embedding dimensionality (auto = a pinned dictionary's width, else 64)",
        long_help = "Cell embedding dimensionality. auto is the default.\n\
                     With --feature-embedding under freeze, free or lora the rows are\n\
                     installed verbatim, so auto is the dictionary's width and a\n\
                     given width must agree with it.\n\
                     Otherwise auto is 64; the adapt mode maps the dictionary into\n\
                     this run's own width.\n\
                     Under --edge-cluster-method kmeans with no --n-communities,\n\
                     the width is also the number of communities."
    )]
    pub embedding_dim: graph_embedding_util::EmbeddingDim,

    #[arg(
        long,
        default_value_t = 100,
        help = "Training epochs over the feature axis",
        long_help = "Passes over the feature axis.\n\
                     \n\
                     Pair with --features-per-epoch to cap per-epoch cost."
    )]
    pub epochs: usize,

    #[arg(
        long,
        default_value_t = 200,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "cage-cell: train until each pair's embedding is updated this many times, on average",
        long_help = "cage-cell's training length, in updates per cell pair.\n\
                     \n\
                     Each pair has an embedding row of its own, moved only by the\n\
                     optimizer steps whose draws name it (at most once a step).\n\
                     Training stops after the step at which the updates summed over\n\
                     all steps (the distinct pairs each step touched) reach this\n\
                     number times the pair count, so the budget means the same at\n\
                     any data size: steps = N x pairs / pairs touched per step.\n\
                     The log reports the pairs a step touches, the steps this\n\
                     implies and the steps taken. --epochs caps it.\n\
                     cage ignores it.",
        hide = true
    )]
    pub pair_updates: usize,

    #[arg(
        long,
        default_value_t = DEFAULT_PAIR_ENCODER_STEPS,
        help = "Optimizer steps for the pair encoder, at least",
        long_help = format!(
            "Optimizer steps the pair encoder trains for, at least: rounded up to\n\
             whole passes over the training pairs, since a pass is always made\n\
             whole, so many pairs take more. The training pairs are the finest\n\
             super-cells' pairs under cage, and the cell pairs under cage-cell.\n\
             A pass is the pairs split into steps of up to {DEFAULT_PAIR_ENCODER_BATCH}.\n\
             Fewer is faster; too few leaves the encoder near its random start.\n\
             0 skips training."
        )
    )]
    pub pair_encoder_steps: usize,

    #[arg(
        long,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Features per outer parallel sampling chunk",
        long_help = "The outer loop samples this many features in parallel via rayon.\n\
                     Forward and backward then run serially. candle Var is not parallel-safe.\n\
                     \n\
                     This is ALSO the SGD minibatch, not just a parallel width.\n\
                     An epoch takes trainable-features / this many optimizer steps,\n\
                     each paying a forward, a backward, a gradient clip and an\n\
                     AdamW update over the whole feature table.\n\
                     So it sets both wall-clock and how many parameter updates\n\
                     a run performs, and it is not fit-neutral:\n\
                     raising 64 to 2048 cut wall-clock several fold AND raised\n\
                     spatial coherence, at an unchanged sampling budget.\n\
                     \n\
                     Below ~this many trainable features an epoch is ONE step,\n\
                     so a small panel may want a smaller value here\n\
                     or more --epochs.\n\
                     \n\
                     Unset, the default is 2048 on CPU.\n\
                     On CUDA the size is chosen automatically:\n\
                     a short probe measures the memory one step retains,\n\
                     and grows the chunk\n\
                     while it fits --gpu-mem-fraction of free device memory,\n\
                     never past 2048.\n\
                     The coherence result above was validated at 2048;\n\
                     a memory-constrained device that resolves lower\n\
                     trades some of that benefit for fitting at all.\n\
                     Passing a value disables the probe and always wins.",
        hide = true
    )]
    pub feature_batch_size: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.6,
        help = "Fraction of free GPU memory the training chunk may target",
        long_help = "Ceiling for the automatic chunk sizing on CUDA.\n\
                     The probe grows the chunk\n\
                     while one step's retained memory,\n\
                     with half reserved for the backward pass,\n\
                     fits this fraction of the device memory free at start.\n\
                     Fractions outside 0.05 to 0.95 are clamped to that range.\n\
                     Ignored on CPU and when --feature-batch-size is set.",
        hide = true
    )]
    pub gpu_mem_fraction: f32,

    #[arg(
        long,
        value_name = "N",
        help = "Total positive edges drawn per epoch, across all features (unset = auto)",
        long_help = "The epoch's total SUPER-EDGE sampling budget.\n\
                     \n\
                     Divided evenly: each visited feature draws\n\
                     N / (expected visits x batches) positives per batch,\n\
                     floored, at least one. The expected visits are the\n\
                     trainable features, or the sum of their inclusion\n\
                     probabilities pi under --feature-inclusion.\n\
                     Unset, each feature draws a fixed number per batch.\n\
                     \n\
                     --features-per-epoch does not enter the split: it caps\n\
                     the features visited, each keeps its draws, and the\n\
                     epoch draws fewer than N in all.\n\
                     --epochs is the usual way to train longer.",
        hide = true
    )]
    pub positives_per_epoch: Option<usize>,

    #[arg(
        long,
        default_value_t = 5e-3,
        help = "AdamW learning rate",
        hide = true
    )]
    pub lr: f32,

    #[arg(
        long,
        default_value_t = 0,
        help = "Features visited per epoch; 0 = the whole axis",
        long_help = "Cost lever. cage walks the feature axis once per epoch.\n\
                     Runtime is therefore linear in the feature count.\n\
                     This caps how many are VISITED per epoch.\n\
                     A fresh random subset is drawn each time.\n\
                     \n\
                     That is stochastic coverage, NOT feature selection.\n\
                     Every feature stays on the trained axis. It keeps its sampled loading.\n\
                     It appears in every output table.\n\
                     A feature left out simply waits for a later epoch. Contrast --n-hvg,\n\
                     which weights the projection. That likewise drops nobody.",
        hide = true
    )]
    pub features_per_epoch: usize,

    #[arg(
        long,
        default_value_t = 0.0625,
        help = "L2 penalty λ on the shared cell and feature embeddings; 0 = off",
        long_help = "L2 penalty λ on E_pb ∈ ℝ^{P×D} and E_feature ∈ ℝ^{G×D}.\n\
                     It adds λ · (mean_n ‖e_n‖² + mean_g ‖e_g‖²) to the loss:\n\
                     a sum over the D latent dims, averaged over rows.\n\
                     The row-mean keeps λ scale-invariant across N and G, and\n\
                     summing over D rather than averaging keeps it invariant to\n\
                     --embedding-dim too — see loss::embedding_ridge.\n\
                     The default 0.0625 is the shrinkage cage was tuned at\n\
                     (~40% off the free feature-embedding norm), and it means\n\
                     the same thing at every D.\n\
                     Useful range 0.01-0.25. Do NOT reach for 1.0: measured on\n\
                     a 10.9k-cell 18k-feature sample it drives both embeddings to\n\
                     zero. A dense every-row penalty competes with a SPARSE data\n\
                     gradient under Adam's per-parameter normalization, so it\n\
                     bites far harder than its size against the loss suggests.",
        hide = true
    )]
    pub embedding_l2: f32,

    #[arg(
        long,
        value_delimiter(','),
        help = "Chain levels from the coarsening hierarchy.\nUnset: every level coarser than the finest, up to three.\nExplicit levels must all be coarser than the finest,\nwhich is the trained unit itself.",
        long_help = "Chain levels from the coarsening hierarchy,\n\
                     coarsest first.\n\
                     Unset, the default takes every level coarser than\n\
                     the finest, up to three of them,\n\
                     so it adapts to --num-levels.\n\
                     Explicit levels must all be coarser than the finest\n\
                     level: the finest IS the trained unit,\n\
                     so no positive super edge can share one.",
        hide = true
    )]
    pub chain_levels: Option<Vec<usize>>,

    #[arg(
        long,
        value_enum,
        default_value_t = FeatureInclusionMode::All,
        help = "Which features each epoch trains: all, or each with its probability of spatial signal",
        long_help = "Which features each epoch trains.\n\
                     all: every trainable feature, every epoch.\n\
                     moran: feature g with probability π_g, its posterior\n\
                     probability of spatial signal. Each feature's Moran's I\n\
                     (Moran 1950) over the spatial graph (log-normalized, centred\n\
                     per section, pooled; on the PB tree's finest super-cells and\n\
                     super edges, or on cells for cage-cell) gets a\n\
                     delete-one-block jackknife standard error; a two-group\n\
                     empirical-Bayes model on the t statistics gives\n\
                     π_g = 1 − local fdr (Efron 2004), with no threshold to\n\
                     choose. Features are weighted, not cut: the expected\n\
                     objective is Σ π_g L_g, and the likely-noise features\n\
                     mostly cost nothing. Needs coordinates (-c).\n\
                     An alternative to --feature-modules, not a companion: a\n\
                     run takes one, and moran given alone turns the default\n\
                     modules off. Writes {out}.feature_inclusion.parquet."
    )]
    pub feature_inclusion: FeatureInclusionMode,

    #[arg(
        long,
        default_value_t = 16,
        hide = true,
        help = "Jackknife blocks per section for --feature-inclusion",
        long_help = "About this many blocks per section, which the jackknife\n\
                     leaves out one at a time: coarse super-cells of the PB tree\n\
                     (merged to this many per section), or\n\
                     on cells (cage-cell) a grid of ceil(sqrt(N)) tiles per side.\n\
                     Blocks within sections, never whole sections, so a domain\n\
                     that only one section holds is not penalized."
    )]
    pub inclusion_blocks: usize,

    /// Cells and cell pairs are the trained unit, with no pseudobulks:
    /// set by `pinto fit cage-cell`, not a flag.
    #[arg(skip)]
    pub cell_level: bool,

    /// cage-cell only: the flags given that shape nothing but the
    /// pseudobulks seeding `--feature-modules auto`, which did nothing if
    /// no module layer is found (see `fit::check_cell_level`).
    #[arg(skip)]
    pub seeding_flags: Vec<String>,

    /// HVG selection: senna-style shared CLI (`--n-hvg`,
    /// `--feature-list-file`). cage **weights the random projection** with it,
    /// exactly as `senna bge` and `senna gem` do — non-selected features get
    /// projection weight 0 and so sit out the basis the coarsening hierarchy is
    /// built from, but they stay on the trained axis. The selection shapes
    /// *where the pseudobulks land*, not *which features the model may use*.
    /// `--n-hvg 0` disables. Use `--features-per-epoch` for the cost lever a hard
    /// subset used to provide.
    #[command(flatten)]
    pub hvg: HvgCliArgs,

    /// How the pair latent becomes link communities. Shared verbatim with
    /// `dsvd` — the k-means fallback width is `--embedding-dim` here.
    #[command(flatten)]
    pub edge_clustering: crate::util::edge_clustering::EdgeClusterArgs,

    #[arg(
        long,
        help = "Pre-trained feature x H embedding parquet to start the feature side from",
        long_help = "Path to a pre-trained feature x H embedding parquet.\n\
                     Row column 0 holds the feature name; value columns are the dimensions.\n\
                     Feed a RAW dictionary: a run's feature_embedding.parquet.\n\
                     Co-embedding outputs (feature_coembedding.parquet) are not\n\
                     dictionaries and are rejected.\n\
                     Features are matched under --feature-name-mode.\n\
                     A feature with no dictionary row is seeded from the matched feature\n\
                     with the most similar count profile and listed in\n\
                     {out}.feature_embedding_init.parquet. Under freeze, free and lora such\n\
                     rows train; under adapt a seeded feature follows its seed through\n\
                     the shared map until --feature-adapter-residual gives it its own\n\
                     correction."
    )]
    pub feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        requires = "feature_embedding",
        help = "Optional per-feature bias parquet ([D, 1]) paired with --feature-embedding;\n\
                features without a row get bias 0"
    )]
    pub feature_embedding_bias: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = FeatureEmbeddingMode::Adapt,
        requires = "feature_embedding",
        help = "What training may do to the pre-trained feature embedding",
        long_help = "What training may do to the pre-trained feature embedding.\n\
                     \n\
                     adapt keeps the dictionary fixed and trains one\n\
                     shared linear map on top of it, so every feature's gradient\n\
                     updates the same few parameters.\n\
                     The dictionary width and --embedding-dim may differ.\n\
                     Add --feature-adapter-residual for a per-feature correction\n\
                     where the shared map is not enough.\n\
                     \n\
                     freeze keeps every dictionary-matched row fixed at its loaded value.\n\
                     Neighbor-seeded rows still train.\n\
                     Requires the dictionary width to equal --embedding-dim.\n\
                     The e_feat ridge is skipped: a fixed table needs no shrinkage.\n\
                     \n\
                     free initializes from the dictionary and then trains every row.\n\
                     Also requires the widths to match. This is the fallback\n\
                     when the shared map underfits.\n\
                     \n\
                     lora keeps every dictionary-matched row fixed, as freeze does,\n\
                     and trains a low-rank residual on top of those rows:\n\
                     row_g = dictionary_g + u_g · V, with u_g per feature (--lora-rank numbers)\n\
                     and V shared by every matched feature, at the LoRA+ rate\n\
                     (--lora-lr-ratio) and under --lora-ridge.\n\
                     Neighbor-seeded rows still train. Requires the widths to match.\n\
                     The written feature embedding carries the residual folded in."
    )]
    pub feature_embedding_mode: FeatureEmbeddingMode,

    /// `--lora-rank`, `--lora-lr-ratio`, `--lora-ridge`; read under
    /// `--feature-embedding-mode lora` only.
    #[command(flatten)]
    pub lora: graph_embedding_util::LoraArgs,

    #[arg(
        long,
        requires = "feature_embedding",
        help = "adapt only: add a ridge-shrunk per-feature correction\n\
                on top of the shared map",
        hide = true
    )]
    pub feature_adapter_residual: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = FeatureInitMode::Membership,
        requires = "feature_embedding",
        help = "How a feature with no dictionary row starts",
        long_help = "How a feature with no row in --feature-embedding starts.\n\
                     \n\
                     membership places it through the dictionary's learned modules:\n\
                     its membership is the similarity-weighted mean of the closest\n\
                     matched features' memberships (by count profile), and its row is\n\
                     that membership times the module dictionary, with no residual.\n\
                     Needs {stem}.module_membership.parquet and\n\
                     {stem}.module_dictionary.parquet beside the dictionary; without\n\
                     them it falls back to neighbor and says so in\n\
                     {out}.feature_embedding_init.parquet.\n\
                     \n\
                     neighbor copies the row of the single closest matched feature."
    )]
    pub feature_init_mode: FeatureInitMode,

    /// `--feature-modules`: unset is `auto` that yields to an explicit
    /// `--feature-inclusion moran` or a `--feature-embedding`
    /// ([`Self::module_plan`]). The flag's id and alias are those of
    /// `graph_embedding_util::FeatureModuleArgs`, whose knob defaults a run
    /// takes; that group's count cannot say `auto` or 0, so the flag is
    /// cage's own.
    /// Cage has no feature-negative NCE, so the within-module negatives do not apply
    /// here; the composition, the exact pseudobulk–module term, the feature dropout
    /// and the warm start do. The residual takes `--embedding-l2` like every other
    /// feature-side table in cage.
    #[arg(
        long,
        value_name = "M|auto",
        help = "Embed features through M learned modules: auto (default) takes M from the data, 0 turns them off",
        long_help = FEATURE_MODULES_LONG_HELP
    )]
    pub feature_modules: Option<ModulePlan>,
}

/// How a feature with no dictionary row is initialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FeatureInitMode {
    /// Through the dictionary's learned modules (falls back to `Neighbor`
    /// without module tables).
    Membership,
    /// The closest matched feature's row.
    Neighbor,
}

/// What training may do to a pre-trained feature embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FeatureEmbeddingMode {
    /// Fixed dictionary + one shared trainable map.
    Adapt,
    /// Dictionary-matched rows stay fixed; neighbor-seeded rows train.
    Freeze,
    /// Initialize from the dictionary, then train every row.
    Free,
    /// Dictionary-matched rows stay fixed under a low-rank residual;
    /// neighbor-seeded rows train.
    Lora,
}

/// What a run does about feature modules: the value of `--feature-modules`
/// (`auto`, a count of 2 or more, or 0 for none), and what the run's flags
/// resolve to before any data is read ([`module_plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModulePlan {
    /// No module layer.
    Off,
    /// A module layer whose count the profiles decide; fewer than two found
    /// trains without the layer.
    Auto,
    /// A module layer of this many modules.
    Fixed(usize),
}

impl std::str::FromStr for ModulePlan {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        match s.parse::<usize>() {
            Ok(0) => Ok(Self::Off),
            Ok(1) => Err("one module is no layer; pass 2 or more, `auto`, or 0 for none".into()),
            Ok(m) => Ok(Self::Fixed(m)),
            Err(_) => Err(format!("expected a module count or `auto`, got `{s}`")),
        }
    }
}

/// The [`ModulePlan`] of `--feature-modules` (`flag`; unset is the default)
/// against the two other ways to shape the feature side: `moran`
/// (`--feature-inclusion moran`) and `pretrained` (a `--feature-embedding`).
/// The default yields to either; modules asked for by the flag beside
/// either are refused.
pub fn module_plan(
    flag: Option<ModulePlan>,
    moran: bool,
    pretrained: bool,
) -> anyhow::Result<ModulePlan> {
    match flag {
        None if moran || pretrained => Ok(ModulePlan::Off),
        None => Ok(ModulePlan::Auto),
        Some(ModulePlan::Off) => Ok(ModulePlan::Off),
        Some(plan) => {
            // Two ways to handle uninformative features; on the simulations
            // they did worse together than either alone, so a run takes one.
            anyhow::ensure!(
                !moran,
                "--feature-inclusion moran and --feature-modules are alternatives; pick one"
            );
            anyhow::ensure!(
                !pretrained,
                "--feature-modules learns the feature side through a module layer and \
                 --feature-embedding installs a pre-trained one; the two parameterizations \
                 are exclusive. Drop one."
            );
            Ok(plan)
        }
    }
}

/// cage's long help for `--feature-modules`.
pub const FEATURE_MODULES_LONG_HELP: &str = "\
Put a learned mixed-membership module layer in front of the feature embedding.
Each feature's row becomes a sparse mixture of M shared module vectors, plus a
residual of its own (ridge-shrunk by --embedding-l2).

M. auto (the default) takes the count from the data; a number of 2 or more
fixes it; 0 trains without the layer. Under auto, the features whose counts
carry no grouping evidence share one background module, and the rest are
grouped by their profiles over the finest pseudobulks: the profiles are
reduced by a truncated SVD, a nearest-neighbour graph is built on them, and
Leiden (Traag, Waltman and van Eck 2019) cuts it into communities. M is the
communities plus the background module, and is logged. If fewer than two
modules are found the run trains without the layer and says so. A fixed M
partitions the same profiles by k-means instead.

WHY. A free row receives gradient only on the steps that draw its feature. A
rare feature, or one absent from a later dataset, has nothing standing in for
it. Through a module, every edge on any member trains the shared vector, so a
feature that is never drawn still inherits a trained row from its siblings.

HOW. Membership is sparsemax (Martins and Astudillo 2016) over learned logits:
a feature sits on a few modules with exact zeros elsewhere. It starts from the
partition above, held fixed for the first quarter of the epochs, then trains.
An exact softmax over all M modules scores each trained unit's module-pooled
counts, so every module gets gradient on every step, and a load-balance prior
keeps every module in use. cage's NCE is unchanged: its negatives are other
units of the batch, never within a module.

cage-cell trains on cells, so it builds the pseudobulks only to start the
partition, and each step pools the sparse counts of the cells it draws.

A run takes this or --feature-inclusion moran, not both, and not with
--feature-embedding. Left at its default, this flag yields: a run given
--feature-inclusion moran or --feature-embedding trains without modules.
Given beside either (auto or a count), it is refused.

OUTPUTS. {out}.module_membership.parquet (feature x M),
{out}.module_dictionary.parquet (M x H), {out}.module_residual.parquet and
{out}.module_bias.parquet. {out}.feature_embedding.parquet holds the composed
row, so nothing that reads it has to know modules exist; `pinto apply predict` places
features the model never saw through these tables.";

/// The width when neither the flag nor a pinned dictionary decides it.
pub const DEFAULT_EMBEDDING_DIM: usize = 64;

impl CellActivityGraphEmbeddingArgs {
    /// This run's [`ModulePlan`] (see [`module_plan`]).
    pub fn module_plan(&self) -> anyhow::Result<ModulePlan> {
        module_plan(
            self.feature_modules,
            self.feature_inclusion == FeatureInclusionMode::Moran,
            self.feature_embedding.is_some(),
        )
    }

    /// The rules between `--feature-embedding-mode` and the flags only one mode
    /// reads, checked before any data is opened.
    pub fn validate_feature_embedding(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !(self.feature_adapter_residual
                && self.feature_embedding_mode != FeatureEmbeddingMode::Adapt),
            "--feature-adapter-residual is the adapter's per-feature correction and only \
             --feature-embedding-mode adapt trains one; under another mode the flag \
             would be read and ignored. Drop it, or use the adapt mode."
        );
        self.lora.refuse_unless_selected(
            self.feature_embedding_mode == FeatureEmbeddingMode::Lora,
            "--feature-embedding-mode lora",
        )
    }

    /// The embedding width this run trains at: the flag, else a pinned
    /// dictionary's width (`dictionary_width`, when `--feature-embedding` is
    /// given), else the default. A pinned dictionary (every mode but adapt)
    /// installs its rows verbatim, so a flag that disagrees with it is
    /// refused; the adapter maps into its own width. The LoRA settings are
    /// checked here, with the resolved width (the rank is not capped at it).
    pub fn resolve_embedding_dim(&self, dictionary_width: Option<usize>) -> anyhow::Result<usize> {
        self.validate_feature_embedding()?;
        let pinned = dictionary_width.filter(|_| {
            self.feature_embedding.is_some()
                && self.feature_embedding_mode != FeatureEmbeddingMode::Adapt
        });
        let dim = self
            .embedding_dim
            .resolve(pinned)?
            .unwrap_or(DEFAULT_EMBEDDING_DIM);
        if self.feature_embedding_mode == FeatureEmbeddingMode::Lora {
            graph_embedding_util::PresetMode::Lora(self.lora.spec()).validate(dim)?;
        }
        Ok(dim)
    }
}
