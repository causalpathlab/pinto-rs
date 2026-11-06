//! The command-line input arguments shared by the pinto commands.

use super::{KnnExprScope, SRTReadArgs, PRELOAD_HELP, PRELOAD_LONG_HELP};
use clap::Parser;
use data_beans::aux::feature_names::FeatureNameKind;

/// Shared CLI arguments for spatial data input across all pinto subcommands.
///
/// Flatten this into each subcommand's args struct with `#[command(flatten)]`.
#[derive(Parser, Debug, Clone)]
pub struct SrtInputArgs {
    #[arg(
        required = true,
        value_delimiter(','),
        help = "Spatial feature expression data files (.zarr or .h5)",
        long_help = "Spatial feature expression data files, comma separated.\n\
                     Accepted formats are .zarr and .h5.\n\
                     Each file is a features-by-cells sparse matrix.\n\
                     Multiple files are concatenated column-wise, over cells.\n\
                     Each file is then its own batch unless --batch-files says otherwise."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long = "coord",
        short = 'c',
        value_delimiter(','),
        help = "Spatial coordinate files, one per data file (recommended)",
        long_help = "Spatial coordinate files, one per data file, comma separated.\n\
                     Recommended for spatial transcriptomics data.\n\
                     Omit them to run in expression mode.\n\
                     The KNN graph then comes from feature expression.\n\
                     \n\
                     Accepted formats: CSV, TSV, space-delimited text, .parquet,\n\
                     or .zarr/.zarr.zip (Xenium cells.zarr.zip).\n\
                     \n\
                     In CSV, TSV and parquet, the first column is cell names.\n\
                     Those names must match the data file's column names.\n\
                     Every later column is a spatial coordinate.\n\
                     \n\
                     In zarr, cell IDs are read from /cell_id.\n\
                     Coordinates come from /cell_summary.\n\
                     Its attributes supply the column names.\n\
                     \n\
                     Pick columns with --coord-column-names or -indices."
    )]
    pub coord_files: Vec<Box<str>>,

    #[arg(
        long = "coord-column-indices",
        value_delimiter(','),
        help = "0-based column indices for coordinates in coord files",
        long_help = "0-based column indices for coordinate columns, comma separated.\n\
                     Use this when the coord file has columns beyond barcode,x,y.\n\
                     It overrides --coord-column-names when both are given. Example:\n\
                     --coord-column-indices 1,2 picks columns 2 and 3."
    )]
    pub coord_columns: Option<Vec<usize>>,

    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres,cell_centroid_x,cell_centroid_y,x_centroid,y_centroid",
        help = "Column names for spatial coordinates in coord files",
        long_help = "Column names to select as spatial coordinates, comma separated.\n\
                     Names absent from a given file are skipped, so the default\n\
                     covers the common single-cell and spot layouts at once.\n\
                     \n\
                     Default: pxl_row_in_fullres, pxl_col_in_fullres,\n\
                     cell_centroid_x, cell_centroid_y, x_centroid, y_centroid.\n\
                     \n\
                     Quoting does not matter. A file whose header reads \"x_centroid\"\n\
                     matches the name x_centroid, and the same holds for the cell\n\
                     names in the first column.\n\
                     \n\
                     If none of these names is present, the run stops and prints the\n\
                     file's first line, so you can see what it does have. Nothing is\n\
                     read by position on a guess: any two numeric columns would pass\n\
                     as coordinates, and a table with counts where the coordinates\n\
                     were expected would put every cell in one spot without\n\
                     complaining.\n\
                     \n\
                     Say which columns instead. Use this flag for named ones, or\n\
                     --coord-column-indices for positions. The classic spot layout\n\
                     has no header, and its coordinates are --coord-column-indices 4,5."
    )]
    pub coord_column_names: Vec<Box<str>>,

    #[arg(
        long,
        help = "Header row index in coord files (0 = first line)",
        long_help = "0-based row index of the header in coord files. If omitted,\n\
                     it is auto-detected. Detection checks whether the first row looks numeric.\n\
                     Set it to 0 when the first line holds column names.",
        hide = true
    )]
    pub coord_header_row: Option<usize>,

    #[arg(
        long,
        short = 'b',
        value_delimiter(','),
        help = "Batch label files, one per data file",
        long_help = "Batch membership files, one per data file, comma separated.\n\
                     Each is plain text with one batch label per cell, per line.\n\
                     Cells sharing a label share batch effects. If omitted,\n\
                     each data file is treated as one batch."
    )]
    pub batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Auto-detect batches from disconnected spatial graph components"
    )]
    pub auto_batch: bool,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix (e.g., results/my_run)"
    )]
    pub out: Box<str>,

    #[arg(
        long,
        help = "Cells per parallel block (omit for auto-scaling by feature count)",
        hide = true
    )]
    pub block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = PRELOAD_HELP,
        long_help = PRELOAD_LONG_HELP
    )]
    pub preload_data: bool,

    #[arg(long, default_value_t = 42, help = "Random seed for reproducibility")]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of multi-level coarsening levels",
        long_help = "Number of hierarchical coarsening levels, coarse to fine.\n\
                     Each level merges cells by graph-constrained matching.\n\
                     That halves the number of groups per level.\n\
                     More levels initialize the finest level better. They also take longer.\n\
                     Typical range: 2-10.",
        hide = true
    )]
    pub num_levels: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Refinement sweeps per coarsening level (0 to disable)",
        long_help = "Leiden-style local-moving sweeps per coarsening level.\n\
                     They run after the dendrogram cut.\n\
                     A sweep moves nodes to graph-adjacent clusters.\n\
                     A move must improve cosine similarity to the centroid.\n\
                     Moves that would disconnect the source cluster are rejected.\n\
                     Sweeping stops early once no node moves. 5 is usually enough;\n\
                     0 skips refinement.",
        hide = true
    )]
    pub refine_iterations: usize,

    #[arg(
        long,
        short = 'p',
        default_value_t = 200,
        help = "Random projection dimension for cell embeddings",
        long_help = "Dimension of the random projection for cell embeddings.\n\
                     Cells are projected from G feature dimensions down to this one.\n\
                     That projection feeds KNN construction and coarsening,\n\
                     and in `lc` it also sets the width of every edge profile.\n\
                     \n\
                     This is the main quality knob, and it is easy to set too low.\n\
                     A sketch too narrow to separate two cell programs merges them,\n\
                     so the rare profiles that sit between programs get absorbed\n\
                     into whichever neighbouring community is largest.\n\
                     That shows up as speckle in a propensity map\n\
                     rather than as coherent spatial domains.\n\
                     \n\
                     Cost is linear in this value:\n\
                     runtime, peak memory, and the size of the edge profile store.\n\
                     Raise it when domains look speckled or over-merged.\n\
                     Lower it when a large dataset will not fit."
    )]
    pub proj_dim: usize,

    #[arg(
        long = "batch-knn",
        default_value_t = 10,
        help = "KNN for cross-batch matching during batch correction",
        long_help = "Neighbours per pb-sample for cross-batch matching.\n\
                     Batch-effect estimation first coarsens cells into pb-samples.\n\
                     Each pb-sample then finds K neighbours in other batches.\n\
                     The search is HNSW over centroids.\n\
                     Those matches give counterfactual expression estimates.\n\
                     Batch-effect decomposition needs them.\n\
                     This is used only when multiple batches are present.",
        hide = true
    )]
    pub batch_knn: usize,

    #[arg(
        long = "cnv-clones",
        help = "CNV clone table from `mung clones`; batch δ cannot mix across strata",
        long_help = "Path to `{out}.clones.parquet` written by `mung clones`.\n\
                     Each cell's stratum is a hard parent cut on the multilevel\n\
                     collapse used for batch-effect estimation (cage / lc / dsvd):\n\
                     finest codes are crossed with the stratum, BBKNN matches only\n\
                     within stratum, and unmatched (clone-only) mass is excluded\n\
                     from the δ update / pin vote while the δ learned on mixable\n\
                     cells is still applied to clones. Missing cells default to\n\
                     stratum 0. A batch with no matched mass gets δ ≈ 1 (warned).\n\
                     \n\
                     Does not gate spatial graph coarsening — only the MultilevelParams\n\
                     δ path shared by cage / lc / dsvd."
    )]
    pub cnv_clones: Option<Box<str>>,

    #[arg(
        long,
        short = 'd',
        default_value_t = 1024,
        help = "Target number of pseudobulk samples at coarsest level",
        long_help = "Target number of pseudobulk cell groups at the coarsest level.\n\
                     Coarsening merges cells until roughly this many remain.\n\
                     They feed Poisson-Gamma estimation and refinement.\n\
                     Larger values give finer granularity. They also coarsen more slowly."
    )]
    pub n_pseudobulk: usize,

    #[arg(
        short = 'k',
        long,
        help = "KNN: neighbours per cell for the spatial graph [default: 5]",
        long_help = "Neighbours per cell when building the SPATIAL cell-pair graph.\n\
                     Each cell connects to its K closest neighbours in coordinate\n\
                     space. Search runs over an HNSW index in Euclidean distance.\n\
                     Defaults to 5.\n\
                     \n\
                     The resulting edges are the cell pairs used downstream.\n\
                     Typical range: 3-20.\n\
                     \n\
                     This flag is about coordinates. Without --coord the graph comes\n\
                     from expression instead, and --knn-expr sizes it. Passing -k\n\
                     there still works, and says so, but --knn-expr is the name for\n\
                     that job. Passing both without --coord is an error when\n\
                     --knn-expr is positive: they would be naming the same\n\
                     graph. --knn-expr 0 asks for no expression pairs, so it\n\
                     leaves -k to size the graph and stays legal."
    )]
    pub knn_spatial: Option<usize>,

    #[arg(
        long,
        help = "KNN: expression neighbours [default: 0 with --coord, else -k or 5]",
        long_help = "Neighbours per cell in a KNN graph built on random-projected\n\
                     expression. What this graph IS depends on --coord.\n\
                     \n\
                     WITH --coord it is a second graph, and its edges are added to\n\
                     the spatial ones, so the cell pairs downstream are the union of\n\
                     both. Defaults to 0, i.e. off: the run then pays only for the\n\
                     spatial pairs. This is also how to reproduce a run made before\n\
                     the union existed.\n\
                     \n\
                     WITHOUT --coord it is the ONLY graph — there is nothing to union\n\
                     into — and it defaults to 5, or to -k when only that was given.\n\
                     Typical range there is 10-30: expression neighbours agree with\n\
                     each other less often than spatial ones, so a small k fragments\n\
                     the graph, and components become batches in that mode.\n\
                     Note it also sets the k of the force-directed layout, so it\n\
                     decides pc_1/pc_2, the coordinates written to\n\
                     coord_pairs.parquet, and what `pinto plot` draws.\n\
                     \n\
                     The rest of this applies to the --coord case.\n\
                     \n\
                     Expression-similar pairs are the same cell type by construction.\n\
                     They act as a reference for what a same-type pair looks like,\n\
                     which sharpens those communities and leaves the pairs that bridge\n\
                     two types standing out as the residue.\n\
                     \n\
                     The expression graph is built after batch correction, so it\n\
                     matches cell type rather than batch.\n\
                     \n\
                     This is not symmetric with -k: expression neighbours agree with\n\
                     each other less often, so fewer of their edges collapse together\n\
                     and the same K adds more pairs than the spatial graph holds.\n\
                     Runtime and peak memory both rise with the resulting pair\n\
                     count, and at the same K that count more than doubles.\n\
                     \n\
                     lc models every pair. cage's chain training only samples pairs\n\
                     that share a super-cell at each chain level and a batch, so\n\
                     distant expression pairs reach its pair outputs and propensity\n\
                     but largely not its chain loss; the dropped counts are logged."
    )]
    pub knn_expr: Option<usize>,

    #[arg(
        long,
        value_enum,
        default_value_t = KnnExprScope::Global,
        help = "Whether expression neighbours may cross disconnected tissue",
        long_help = "Where --knn-expr searches for expression neighbours.\n\
                     \n\
                     Only `within` has preconditions, and it needs both: --coord,\n\
                     because it scopes by the SPATIAL graph's components, and a\n\
                     positive --knn-expr, because it scopes a search that would\n\
                     otherwise not run. Missing either is refused rather than\n\
                     silently ignored.\n\
                     \n\
                     global (default): anywhere on the slide. Asking for expression\n\
                     neighbours at all means wanting a reference for what a cell type\n\
                     looks like, and the strongest such reference is drawn from every\n\
                     piece of tissue rather than one. Section effects are handled by\n\
                     correction rather than by refusing to look: the projection these\n\
                     neighbours come from is batch-corrected when batches are known,\n\
                     so pair --auto-batch with this on a multi-section slide.\n\
                     \n\
                     within: inside each disconnected piece of the\n\
                     spatial graph. Separate sections, or the cores of a tissue\n\
                     microarray, are usually separate samples, and this keeps every\n\
                     pair inside one of them. Choose it when a pair spanning two\n\
                     samples is not something the model should be shown at all.\n\
                     \n\
                     On a microarray the two differ sharply: searching globally,\n\
                     nearly every neighbour found sits in a different core."
    )]
    pub knn_expr_scope: KnnExprScope,

    #[arg(
        long,
        default_value_t = false,
        help = "Use reciprocal (mutual) KNN matching for every graph",
        long_help = "Use reciprocal (mutual) KNN matching for every KNN graph this\n\
                     run builds — the spatial one, the expression one, and the\n\
                     --knn-expr augmentation alike.\n\
                     The default is union matching.\n\
                     There an edge (i,j) exists if i is in j's KNN list, or j is in i's.\n\
                     Reciprocal matching requires both.\n\
                     That yields a sparser graph of higher-confidence edges.",
        hide = true
    )]
    pub reciprocal: bool,

    #[command(flatten)]
    pub qc: data_beans::qc_lib::QcArgs,
}

/// Neighbours per cell when neither `-k` nor `--knn-expr` says otherwise.
///
/// Spelled out by hand in the `-k` and `--knn-expr` help text: dropping
/// `default_value_t` means clap no longer renders it, so those strings are the
/// only place a user learns this and the only place it can drift.
const DEFAULT_KNN: usize = 5;

/// The graph spec a run's flags resolve to.
///
/// Produced only by [`SrtInputArgs::resolve_knn`], which validates and logs on
/// the way, so holding one is evidence those ran. Carried on the preprocessing
/// output rather than re-derived, so the manifest cannot drift from the graph
/// that was actually built.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedKnn {
    /// Neighbours per cell in the base graph: spatial under `--coord`,
    /// expression without it.
    pub base: usize,
    /// Neighbours per cell in the expression graph unioned in; `0` for none.
    pub augment: usize,
    /// Where the expression search looked. Meaningful only when `augment > 0`.
    pub scope: KnnExprScope,
    /// Mutual-KNN instead of union-KNN, for every graph this run builds.
    pub reciprocal: bool,
}

impl SrtInputArgs {
    /// Whether spatial coordinate files were provided.
    pub fn has_coordinates(&self) -> bool {
        !self.coord_files.is_empty()
    }

    //////////////////////////////////////////////////////
    // -k and --knn-expr, resolved into the two k's the  //
    // pipeline uses. Two flags, two graphs, and which   //
    // flag sizes which depends on --coord.              //
    //////////////////////////////////////////////////////

    /// `--knn-expr` as a REQUEST for expression pairs.
    ///
    /// `Some(0)` is not a request for a 0-sized graph, it is the absence of a
    /// request, so it collapses to `None` here. Every question of the form
    /// "did the user ask for expression pairs?" goes through this, so the
    /// predicate has one spelling and one meaning.
    fn requested_expr_knn(&self) -> Option<usize> {
        self.knn_expr.filter(|&k| k > 0)
    }

    /// Neighbours per cell for the BASE graph — the spatial one under
    /// `--coord`, the expression one without it.
    ///
    /// `--knn-expr` wins in expression mode because that is the graph it names
    /// there. `-k` is still honoured when it is the only one given, so an
    /// existing command line keeps building the graph it always built: k moves
    /// the force-directed layout, so re-sizing it silently would move every
    /// plot. Setting both is refused rather than resolved.
    pub fn base_knn(&self) -> usize {
        if self.has_coordinates() {
            self.knn_spatial.unwrap_or(DEFAULT_KNN)
        } else {
            self.requested_expr_knn()
                .or(self.knn_spatial)
                .unwrap_or(DEFAULT_KNN)
        }
    }

    /// Neighbours per cell for the expression graph unioned into the base one.
    /// `0` means no augmentation.
    ///
    /// Always `0` without coordinates. That is not a shortcut: there the base
    /// graph already IS the expression graph, so a union would be a self-union.
    /// Every caller that asks whether augmentation runs must read THIS, never
    /// `knn_expr` directly, or the two questions drift apart.
    pub fn augment_knn(&self) -> usize {
        if self.has_coordinates() {
            self.knn_expr.unwrap_or(0)
        } else {
            0
        }
    }

    /// The flag that supplied [`Self::base_knn`], for error messages.
    ///
    /// One source of truth for the precedence. Re-deriving it at each use site
    /// lets the two drift: flip the precedence in `base_knn` and the messages
    /// keep naming the old flag with nothing failing.
    fn base_knn_flag(&self) -> &'static str {
        // Nothing typed points at `--knn-expr` too: in expression mode that is
        // the flag naming this graph, not the one the help calls spatial.
        if self.has_coordinates()
            || (self.requested_expr_knn().is_none() && self.knn_spatial.is_some())
        {
            "-k"
        } else {
            "--knn-expr"
        }
    }

    /// Reject the flag combinations that need no data to judge.
    ///
    /// Split from the bounds check so a typo fails before the run reads a
    /// single file: these depend only on argv, while the `< n_cells` bounds
    /// cannot be judged until QC has settled the cell count.
    pub fn validate_knn_flags(&self) -> anyhow::Result<()> {
        if !self.has_coordinates() {
            // The two flags name the same graph here, and the choice is not
            // cosmetic: component count is k-dependent, components become
            // batches, and batches decide whether batch-effect estimation runs
            // at all. Too load-bearing to settle with a warning.
            anyhow::ensure!(
                !(self.knn_spatial.is_some() && self.requested_expr_knn().is_some()),
                "-k and --knn-expr both set without --coord: they name the same \
                 graph there. Pass only --knn-expr."
            );
            // Same condition as the guard below, kept separate for its message
            // and ordered first. That ordering is load-bearing: the other
            // message's remedy (pass --knn-expr N) is wrong here, where
            // --knn-expr sizes the base graph and still runs no scoped search.
            anyhow::ensure!(
                self.knn_expr_scope == KnnExprScope::Global,
                "--knn-expr-scope needs --coord: it scopes the search by the \
                 spatial graph's components, and there is no spatial graph here."
            );
        }
        anyhow::ensure!(self.base_knn() > 0, "{} must be > 0", self.base_knn_flag());
        Ok(())
    }

    /// Resolve both k's, checking the data-dependent bounds and logging any
    /// flag whose role this run changed.
    ///
    /// The one way to obtain a k. Validation, the migration notice and the
    /// resolution are one call because they are one decision: a caller that
    /// could take the k's without them would silently skip both.
    ///
    /// The upper bound is a sanity check, not an OOM guard: the builder clamps
    /// to `n_cells - 1`, and that graph is already complete, so a `k` large
    /// enough to hurt is accepted. It catches the transposed-argument class of
    /// mistake, where `k` exceeds the cell count outright.
    pub fn resolve_knn(&self, n_cells: usize) -> anyhow::Result<ResolvedKnn> {
        self.validate_knn_flags()?;

        let base = self.base_knn();
        anyhow::ensure!(
            base < n_cells,
            "{} ({base}) must be < number of cells ({n_cells})",
            self.base_knn_flag()
        );

        // 0 is how augmentation is turned off, so only the upper bound applies.
        let augment = self.augment_knn();
        anyhow::ensure!(
            augment < n_cells,
            "--knn-expr ({augment}) must be < number of cells ({n_cells})"
        );

        // Last, because it is the least fatal: an unusable k makes the run
        // impossible, while a scope with nothing to scope only makes a flag
        // inert. Reporting the scope first would send the user round twice.
        // Reachable only WITH coordinates — see `validate_knn_flags`.
        anyhow::ensure!(
            self.knn_expr_scope == KnnExprScope::Global || augment > 0,
            "--knn-expr-scope scopes the expression search, and --knn-expr is 0 \
             so no expression search runs. Pass --knn-expr N to ask for one."
        );

        self.warn_knn_flag_use(base);
        Ok(ResolvedKnn {
            base,
            augment,
            scope: self.knn_expr_scope,
            reciprocal: self.reciprocal,
        })
    }

    /// Say so when this run gives a flag a role it did not have before.
    ///
    /// Private and called from [`Self::resolve_knn`] only: it must fire once
    /// per run, and a separately-callable version would be both forgettable
    /// and, at the three manifest sites, repeatable.
    fn warn_knn_flag_use(&self, k: usize) {
        if self.has_coordinates() {
            return;
        }
        if self.requested_expr_knn().is_some() {
            // `--knn-expr` used to be inert without --coord, so a command line
            // carrying it built a DEFAULT-sized graph. It sizes the graph now,
            // and k moves the component count, the batch partition and the
            // layout. `warn`, not `info`, because the default log filter is
            // `warn` and this genuinely changed.
            log::warn!(
                "--knn-expr sizes the expression graph on this run (k={k}); \
                 before it was ignored without --coord and -k sized it."
            );
        } else if self.knn_spatial.is_some() {
            // Unchanged behaviour: -k sized this graph before and still does.
            // An info, not a warning — CI that greps for WARN should not start
            // failing on runs that did not change.
            log::info!(
                "-k names the spatial graph, and this run has no --coord. \
                 Building the expression graph with k={k}; --knn-expr {k} is \
                 the name for that job (pass it INSTEAD of -k)."
            );
        }
    }

    /// Comma-joined string of coordinate file paths, or `None` when the
    /// run was without `--coord`. Convenient for .pinto.json fields.
    pub fn coord_files_joined(&self) -> Option<String> {
        if self.coord_files.is_empty() {
            None
        } else {
            Some(
                self.coord_files
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        }
    }

    /// Convert to the internal read args for data loading. The
    /// `feature_kind` selects how row names get canonicalized for
    /// cross-file / cross-resource matching. `pinto cage` uses
    /// `FeatureNameKind::Gene { delim: '_' }` (or `auto_detect`'d) so
    /// feature names like `ENSG…_SYMBOL` register both the full name and
    /// the `SYMBOL` suffix as aliases, which matching against external
    /// feature resources (PPI, marker sets, etc.) needs. `lc`
    /// and `svd` pass `FeatureNameKind::Exact`, which reads the files on
    /// the rows they all share.
    pub fn to_read_args_with_kind(&self, feature_kind: FeatureNameKind) -> SRTReadArgs {
        SRTReadArgs {
            data_files: self.data_files.clone(),
            coord_files: self.coord_files.clone(),
            preload_data: self.preload_data,
            coord_columns: self.coord_columns.clone().unwrap_or_default(),
            coord_column_names: self.coord_column_names.clone(),
            batch_files: self.batch_files.clone(),
            header_in_coord: self.coord_header_row,
            feature_kind,
        }
    }
}
