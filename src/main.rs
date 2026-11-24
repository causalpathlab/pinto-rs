mod baselines;
mod cell_activity_graph_embedding;
mod cli_path;
mod compare;
mod donor_deviation;
mod holdout;
mod impute;
mod link_community;
mod lr_activity;
#[cfg(feature = "view")]
mod mcp;
mod paired;
mod predict;
mod propensity;
#[cfg(feature = "view")]
mod run;
mod sim_eval;
mod simulate;
mod svd;
#[cfg(feature = "view")]
mod tui;
mod util;
#[cfg(feature = "view")]
mod view;

#[cfg(test)]
mod test_support;

use cell_activity_graph_embedding::{
    fit_cell_activity_graph_embedding, CellActivityGraphEmbeddingArgs,
};
use clap::{Parser, Subcommand};
use colored::Colorize;
use compare::{run_compare, CompareArgs};
use impute::{run_impute, ImputeArgs};
use link_community::fit::*;
use lr_activity::{fit_lra_simulate, fit_srt_lr_activity, LraSimulateArgs, SrtLrActivityArgs};
use predict::{predict, ModelOnlyFlags, PredictArgs};
use propensity::*;
use svd::fit::*;

const LOGO: &str = include_str!("../logo.txt");

fn colorize_logo_line(line: &str) -> String {
    line.replace('▄', &"▄".truecolor(190, 100, 70).to_string())
        .replace('▓', &"▓".truecolor(217, 119, 87).to_string())
        .replace('█', &"█".truecolor(180, 120, 60).to_string())
        .replace('▀', &"▀".truecolor(190, 100, 70).to_string())
        .replace('━', &"━".truecolor(0, 100, 0).to_string())
}

fn print_logo() {
    for line in LOGO.lines() {
        println!("  {}", colorize_logo_line(line));
    }
    println!(
        " {}",
        "Proximity-derived Interaction Network --> Tissue Organization".bold()
    );
    println!();
}

/// PINTO
#[derive(Parser, Debug)]
#[command(
    version,
    propagate_version = true,
    about = "PINTO - Proximity-derived Interaction Network for Tissue Organization",
    long_about = "PINTO discovers cell-cell interaction patterns.\n\
                  It reads spatial transcriptomics.\n\
                  It detects link communities on cell-pair graphs.\n\n\
                  QUICK START:\n\n\
                  \x20 # Prepare data (convert MTX to HDF5):\n\
                  \x20 data-beans from-mtx -r features.tsv.gz -c barcodes.tsv.gz \\\n\
                  \x20   matrix.mtx.gz --backend hdf5 -o data.h5\n\n\
                  \x20 # Link community (spatial, 10x Visium):\n\
                  \x20 pinto fit lc data.h5 -c tissue_positions.csv -o results\n\n\
                  \x20 # Link community (expression-only, no coordinates):\n\
                  \x20 pinto fit lc data.h5 -o results\n\n\
                  \x20 # Delta-SVD:\n\
                  \x20 pinto fit dsvd data.h5 -c coords.csv -o results\n\n\
                  INPUT FILES:\n\n\
                  \x20 Data:   .h5 or .zarr (features x cells, sparse). Multiple files\n\
                  \x20         comma-separated for multi-sample: s1.h5,s2.h5\n\
                  \x20 Coords: CSV/TSV/parquet, first column = barcode, rest = x,y,...\n\
                  \x20         Default columns: pxl_row_in_fullres,pxl_col_in_fullres\n\
                  \x20         Omit -c for expression-only mode.\n\
                  \x20 Batch:  -b labels.txt (one label per cell per line, optional)\n\n\
                  Each fitting command's --help lists the files it writes.",
    term_width = 80
)]
struct Cli {
    #[arg(
        short = 'v',
        long,
        global = true,
        help = "Log progress (info level; RUST_LOG overrides)"
    )]
    verbose: bool,

    #[arg(
        long,
        global = true,
        default_value_t = util::common::DEFAULT_MAX_THREADS,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..),
        help = "Most threads a run uses; lc's Gibbs splits a component into up to this many \
                blocks, so the same --seed and --max-threads give the same draws on any machine",
        long_help = "Most worker threads a run uses: rayon's own count\n\
                     (RAYON_NUM_THREADS, else the machine's cores), at most this.\n\
                     Raise it to use more cores.\n\
                     lc's Gibbs splits each large component into up to this many\n\
                     blocks, whatever threads run them, so the same --seed, data\n\
                     and --max-threads give the same draws on any machine."
    )]
    max_threads: usize,

    #[arg(
        long,
        global = true,
        help = "Write outputs under a temporary name; move them into place only if the run succeeds",
        long_help = "Write the outputs in a hidden folder beside them\n\
                     (.{name}.partial-{pid}-{random}) and rename them into place\n\
                     only when the run succeeds, the manifest ({out}.pinto.json)\n\
                     last: a manifest on disk means the run completed. A run\n\
                     that fails or panics removes the folder and leaves nothing\n\
                     under the final names; one that is killed leaves the folder\n\
                     behind. run and mcp pass it to the commands they start."
    )]
    atomic_outputs: bool,

    #[command(subcommand)]
    commands: Commands,
}

/// pinto's command line, built as clap builds it to parse.
#[cfg(any(feature = "view", test))]
pub(crate) fn built_cli() -> clap::Command {
    use clap::CommandFactory;
    let mut cli = Cli::command();
    cli.build();
    cli
}

/// The spatial baselines' memory guard, for their `long_about`.
macro_rules! baseline_memory {
    () => {
        "It stops before its large allocations when its estimated peak memory\n\
         exceeds PINTO_MEMORY_BUDGET (bytes; default half the memory available\n\
         at the start). It checks first from the files' stored shape, before\n\
         reading the counts; a file that does not store its nonzero count\n\
         leaves that read unchecked.\n\n"
    };
}

/// What the spatial baselines do about batch effects, for their
/// `long_about`.
macro_rules! baseline_batch {
    () => {
        "The input is read as lc, dsvd and cage read it: cells QC'd alike\n\
         (--no-qc keeps every cell), and the ablations taken alike.\n\
         With two or more batches (data files, or -b labels) the counts are\n\
         divided by the input's own batch effects (δ, estimated as cage does\n\
         and written to {out}.delta.parquet) before the method's own\n\
         preprocessing. That is a deviation from the original method;\n\
         --ignore-batch runs the original.\n\n"
    };
}

/// The spatial baselines' outputs, as `baselines/spatial/output.rs`
/// writes them, for their `long_about`.
macro_rules! baseline_outputs {
    () => {
        "The output is a pinto run, which `pinto view`, compare and lra read\n\
         as they read an lc run: the manifest {out}.pinto.json\n\
         (rerunning into the same --out overwrites it),\n\
         {out}.propensity.parquet (cells x K: the method's posteriors when it\n\
         has them, else one-hot labels),\n\
         {out}.cells.parquet (cells with coordinates), {out}.labels.tsv (for\n\
         sim score-by-set --labels), {out}.cell_embedding.parquet (the method's\n\
         per-cell vectors) and {out}.baseline.json (the settings as run)."
    };
}

/// `pinto fit`'s long help.
const FIT_ABOUT: &str = "Fit a model to data files and their coordinates.\n\n\
     pinto's models: lc (link communities on cell pairs), dsvd (shared and\n\
     difference channels by SVD), cage (cell-graph embedding on super-cells)\n\
     and cage-cell (cage trained on cells). The spatial baselines, ports of\n\
     published methods: spatialde2, stagate, spatialpca and mefisto, and each\n\
     fitted on super-cells as its -pb form. Every fit writes a pinto run\n\
     ({out}.pinto.json and the tables it lists), which view, compare, lra and\n\
     apply read.\n\n\
     Examples:\n\
     \x20 pinto fit lc data.h5 -c tissue_positions.csv -o results/\n\
     \x20 pinto fit stagate s1.h5,s2.h5 -c c1.csv,c2.csv -o results/\n\n\
     `pinto fit <method> --help` lists a method's flags and the files it writes.";

/// `pinto apply`'s long help.
const APPLY_ABOUT: &str = "Use a fitted run.\n\n\
     predict applies a run to a new sample or scores held-out cells; impute\n\
     fills in a new sample's counts from the run's cells; propensity re-cuts\n\
     a run's pair latent at a fixed number of communities.";

/// `pinto sim`'s long help.
const SIM_ABOUT: &str =
    "Simulate donors' sections with a planted truth, and score fits against it.\n\n\
     new writes the sections and the truth; score-by-set scores labellings\n\
     (sets of spots), score-by-pair scores per-cell vectors on pairs of\n\
     spots with no clustering, and baseline labels the spots with a\n\
     standard pipeline to score beside them.";

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        subcommand,
        about = "Fit a model: lc, dsvd, cage, cage-cell, or a spatial baseline",
        long_about = FIT_ABOUT
    )]
    Fit(Fit),

    #[command(
        subcommand,
        about = "Use a fitted run: held-out scoring, imputation, a re-cut",
        long_about = APPLY_ABOUT
    )]
    Apply(Apply),

    #[command(
        name = "lra",
        about = "Ligand-receptor analysis: does neighbouring ligand change the receiving cells?",
        long_about = lr_activity::lra_long_about(),
        args_conflicts_with_subcommands = true,
        subcommand_negates_reqs = true
    )]
    Lra(Box<Lra>),

    #[command(
        subcommand,
        about = "Simulate donors' sections with a planted truth, and score fits against it",
        long_about = SIM_ABOUT
    )]
    Sim(Sim),

    #[command(
        about = "Compare two runs' communities over the cells they share",
        long_about = "Compare two runs' hard community labels, joined on cell name.\n\n\
                      Over the cells both runs placed it reports the adjusted Rand\n\
                      index, normalized mutual information, a one-to-one community\n\
                      matching (Hungarian, on Jaccard) and each community's fate.\n\
                      A community's partners are the other run's communities holding\n\
                      at least --tau of its cells: two or more is a split (from A) or\n\
                      a merge (into B); none means the other run has no counterpart\n\
                      (vanished from A, new in B).\n\n\
                      Writes {out}.compare.json and {out}.contingency.parquet.\n\n\
                      Examples:\n\
                      \x20 pinto compare base.pinto.json perm.pinto.json -o base_vs_perm\n\
                      \x20 pinto compare base perm --level draft,final --tau 0.3 -o cmp"
    )]
    Compare(Box<CompareArgs>),

    #[cfg(feature = "view")]
    #[command(
        about = "View a run's communities on the tissue (multi-resolution)",
        long_about = format!(
            "View a pinto run's cells and communities on the tissue.\n\n\
             Reads {{prefix}}.pinto.json and the parquet files it lists:\n\
             \x20 cells            every cell's coordinates and batch label\n\
             \x20 propensity       per-cell community mixture, per level\n\
             \x20 link_community   per-edge community labels\n\n\
             Batches share one coordinate frame, so they are tiled\n\
             side by side rather than drawn on top of each other.\n\n\
             Examples:\n\
             \x20 pinto view results --summary\n\
             \x20 pinto view results.pinto.json --level L2 --summary\n\
             \x20 pinto view --follow     # what an agent shows on `pinto mcp`\n\n\
             {}",
            legume_tui::cli::session_help(
                &tui::TOOL,
                legume_tui::cli::SessionHelp::new()
                    .resume_hint(Some("`pinto view` with no run"))
                    .more("`?` shows the keys in effect.")
            )
        )
    )]
    View(Box<view::ViewArgs>),

    #[cfg(feature = "view")]
    #[command(
        about = "Serve pinto run and the viewer to agents (Model Context Protocol, stdio)",
        long_about = format!(
            "Serve pinto to an agent as a Model Context Protocol server on\n\
             stdin and stdout. It tells the agent, as it connects:\n\n\
             {}\n\n\
             For a client that does not pass that on, put it in the agent's\n\
             own instructions (an AGENTS.md, a system prompt).\n\n\
             Any MCP client runs it as the command `pinto mcp`. The project's\n\
             records go to its working directory, or to LEGUME_MCP_PROJECT;\n\
             LEGUME_MCP_RECORDS=off keeps none. `pinto view --follow` shows\n\
             a person what the agent does. The README has each client's setup.\n\n\
             Examples:\n\
             \x20 claude mcp add pinto -e LEGUME_MCP_PROJECT=\"$PWD\" -- pinto mcp\n\
             \x20 codex mcp add pinto -- pinto mcp",
            legume_tui::cli::wrap(&mcp::instructions(), 80)
        )
    )]
    Mcp,

    #[cfg(feature = "view")]
    #[command(
        about = "Set up fits in the terminal and run them",
        long_about = format!(
            "Set up pinto fits in the terminal and run them.\n\n\
             Pick the data files with the coordinate and batch files of\n\
             each, queue one or more of lc, dsvd, cage and cage-cell, and\n\
             change their flags. Every flag a method has is listed with its help;\n\
             hidden ones under `a`. On a fitted run, lra and lra simulate\n\
             can be queued too; a --from names a run already fitted.\n\n\
             Each data file is its own batch unless `n` names one batch for\n\
             all its cells (files given one name are one batch) or `b` takes\n\
             a label file, whose labels `e` renames. Label files the run\n\
             needs beyond those given are written to `{{out}}.batches/`.\n\n\
             ctrl+r shows the exact commands, checked as pinto would parse\n\
             them, and runs them in turn with their log on screen; the\n\
             methods a run finished are then unqueued, so ctrl+r again (or\n\
             `r` on the Run screen) runs only the rest, each at the --out\n\
             it had. Tab or shift-enter moves to the next screen.\n\
             Each is saved first as `{{out}}.cmd.sh`: run it again with\n\
             `bash {{out}}.cmd.sh`. The script refuses to run over an\n\
             existing `{{out}}.pinto.json`, and pinto run never writes over\n\
             a script. When the fits finish, `v` opens one in `pinto view`.\n\n\
             {}",
            legume_tui::cli::session_help(
                &tui::TOOL,
                legume_tui::cli::SessionHelp::new()
                    .what_goes_on("the fits go on")
                    .resume_hint(Some("`pinto run` with no folder"))
                    .more("The keys in effect are listed at the foot of the screen.")
            )
        )
    )]
    Run(run::RunArgs),

    #[cfg(feature = "view")]
    #[command(
        subcommand,
        about = "Return to, list, stop or label the `pinto run` and `pinto view` sessions"
    )]
    Session(Session),
}

/// `pinto fit <method>`: every fit, pinto's models and the spatial
/// baselines, on the same input flags.
#[derive(Subcommand, Debug)]
enum Fit {
    #[command(
        name = "lc",
        about = "Link community model via collapsed Gibbs sampling",
        long_about = "Link community detection for spatial transcriptomics.\n\n\
                      Each cell-cell edge is assigned to one of K communities.\n\
                      The assignment reads per-edge expression profiles.\n\
                      Per-cell soft membership follows from those labels.\n\n\
                      QUICK START:\n\n\
                      \x20 # Typical spatial run (10x Visium):\n\
                      \x20 pinto fit lc data.h5 -c tissue_positions.csv -o out\n\n\
                      \x20 # More communities:\n\
                      \x20 pinto fit lc data.h5 -c coords.csv -o out --n-communities 25\n\n\
                      \x20 # Expression-only (no coordinates):\n\
                      \x20 pinto fit lc data.h5 -o out\n\n\
                      \x20 # With external feature-feature network:\n\
                      \x20 pinto fit lc data.h5 -c coords.csv -o out \\\n\
                      \x20   --feature-network feature_pairs.tsv\n\n\
                      \x20 # Multi-sample with batch correction:\n\
                      \x20 pinto fit lc s1.h5,s2.h5 -c c1.csv,c2.csv -o out\n\n\
                      INPUT FILES:\n\n\
                      \x20 data.h5 / data.zarr   Features-by-cells sparse matrix.\n\
                      \x20                        Convert from MTX: data-beans from-mtx in.mtx -o out.zarr\n\
                      \x20 -c coords.csv          Cell coordinates (barcode,x,y).\n\
                      \x20                        Omit for expression-only mode.\n\n\
                      EDGE PROFILE MODES:\n\n\
                      \x20 Compressed all-feature profile (default):\n\
                      \x20   y_e = W^T(x_i + x_j), W = rows × --proj-dim Gaussian basis.\n\
                      \x20   Rows, not features: on a {feature}/count/{spliced,unspliced}\n\
                      \x20   matrix there are two rows per feature, and both carry the\n\
                      \x20   same feature-level count filter and NB weight so a feature is\n\
                      \x20   never split across the projection.\n\
                      \x20   Every profile dim is a full linear combination of ALL features\n\
                      \x20   (no features dropped); M = proj-dim just compresses the feature axis.\n\n\
                      \x20 Feature-network module-pair profile (--feature-network file.tsv):\n\
                      \x20   Feature-feature edges you supply (two-column TSV; pinto has\n\
                      \x20   no network of its own), optionally SNN-\n\
                      \x20   augmented, k-core-trimmed, Leiden-clustered into feature modules.\n\
                      \x20   Edge profile is SPARSE over module-pairs (a, b) with entries\n\
                      \x20   max(0, x_{i,a}·x_{j,b} + x_{i,b}·x_{j,a} − X_i·X_j · deg(a)·deg(b)/(2W)²).\n\n\
                      ALGORITHM:\n\n\
                      \x20 1. Build spatial KNN graph (or expression KNN if no coords)\n\
                      \x20 2. Batch effect estimation (multi-sample only;\n\
                      \x20    optional `--cnv-clones` from `mung clones` so private CN\n\
                      \x20    stays out of δ — does not gate spatial coarsening)\n\
                      \x20 3. Multi-level graph coarsening\n\
                      \x20 4. Resolve feature modules (projection or SNN + k-core + Leiden)\n\
                      \x20 5. Build sparse edge profiles (projection or module-pair residual),\n\
                      \x20    each count divided by its batch's δ, as dsvd and cage divide\n\
                      \x20    (the written feature_community too; --ignore-batch: no δ)\n\
                      \x20 6. V-cycle Gibbs + greedy across coarsening levels\n\
                      \x20 7. Component-EM + greedy on full fine-resolution edges\n\
                      \x20 8. Extract cell propensity + feature-community statistics (+ cosine dictionary merge)\n\n\
                      See `pinto fit lc --help` for individual flag docs.\n\n\
                      OUTPUT FILES:\n\n\
                      \x20 {out}.propensity.parquet      Cell community membership [N × K]\n\
                      \x20                                Columns: 0 .. K-1, plus `entropy`\n\
                      \x20                                (Shannon entropy of each row, nats).\n\
                      \x20 {out}.feature_community.parquet      Feature-community rates [G × K]\n\
                      \x20                                (rows scaled by the NB Fisher-info weight\n\
                      \x20                                 w_g = 1/(1 + π_g·s̄·φ(μ_g)); no flag)\n\
                      \x20                                Keyed by the bare FEATURE name: on a matrix of\n\
                      \x20                                {feature}/count/{spliced,unspliced} rows the two\n\
                      \x20                                tracks are pooled. `cage` keeps its own copy of\n\
                      \x20                                this table on the matrix rows instead.\n\
                      \x20 {out}.link_community.parquet  Edge community assignments [E × 3]\n\
                      \x20 {out}.coord_pairs.parquet     Cell pair coordinates\n\
                      \x20 {out}.scores.parquet          Per-sweep diagnostics (level, sweep,\n\
                      \x20                                score, n_edges, total_mass,\n\
                      \x20                                mutual_information). `score` is the\n\
                      \x20                                plug-in Poisson DC-SBM log-likelihood\n\
                      \x20                                Σ_kg f(D_kg) − Σ_k f(V_k) with\n\
                      \x20                                f(x)=x·ln x, where D_kg is the\n\
                      \x20                                edge-weighted feature degree in community k\n\
                      \x20                                and V_k = Σ_g D_kg is its volume\n\
                      \x20                                (equivalently −Σ_k V_k · H(p_k), nats).\n\
                      \x20                                Higher = better; `score/total_mass`\n\
                      \x20                                is the\n\
                      \x20                                mass-weighted mean per-community\n\
                      \x20                                log-likelihood per edge unit.\n\
                      \x20                                A report: the sweeps move on the collapsed\n\
                      \x20                                conditional, so it need not rise every sweep.\n\
                      \x20 {out}.delta.parquet           Batch effects δ (multi-sample only;\n\
                      \x20                                named in the manifest)\n\
                      \x20 {out}.delta_cells.parquet     Each cell's batch, as δ's columns name it\n\
                      \x20 {out}.feature_graph.parquet      Feature-feature pairs (feature-pair mode only)\n\
                      \x20 {out}.L{l}.*.parquet          Per-cascade-level outputs (unless --no-level-outputs)\n\
                      \x20 {out}.draft.*.parquet         Pre-merge fine partition (when a --merge-cut collapsed)\n\
                      \x20 {out}.dict_merges.parquet     Cosine merge tree over the feature-community dictionary\n\
                      \x20 {out}.dict_merges.cut.parquet Fine→super community remap (with --merge-cut)\n\
                      \x20 {out}.pinto.json           Information-flow manifest:\n\
                      \x20                                lists every parquet, level tags,\n\
                      \x20                                dict-merge presence, and (when set by\n\
                      \x20                                pinto lra) its tables.\n\
                      \x20                                Pass this path to `pinto lra --from`."
    )]
    LinkCommunity(Box<SrtLinkCommunityArgs>),

    #[command(
        name = "dsvd",
        about = "Feature-level shared/difference analysis by SVD",
        long_about = "Feature-level cell-cell interaction analysis by SVD.\n\
                      It uses shared and difference channels.\n\n\
                      Model:\n\
                      \x20 Each cell's counts x (divided by its batch's effects\n\
                      \x20 when there are batches) are scaled to a total of 1e4\n\
                      \x20 and logged, so cells are compared at one depth:\n\
                      \x20   y_ig = log1p(1e4 * x_ig / sum_g x_ig)\n\
                      \x20 For each cell pair e=(i,j) and feature g:\n\
                      \x20   sigma_e^g = y_ig + y_jg    shared\n\
                      \x20   delta_e^g = |y_ig - y_jg|  difference\n\
                      \x20 Pairs grouped into S pseudobulk samples via\n\
                      \x20 graph-constrained coarsening.\n\
                      \x20 Per sample s, feature g:\n\
                      \x20   Y_s^g = sum_{e in s} sigma_e^g  (or delta_e^g)\n\
                      \x20   Y_s^g | mu_g ~ Poisson(n_s * mu_g)\n\
                      \x20   mu_g ~ Gamma(a0, b0)   collapsed out\n\n\
                      Algorithm:\n\
                      \x20 1. Load data X [G x N] and coordinates [N x D]\n\
                      \x20    (if no coordinates, use expression embeddings)\n\
                      \x20 2. Estimate batch effects delta [G x B]\n\
                      \x20    (optional `--cnv-clones` from `mung clones` so private CN\n\
                      \x20     stays out of δ; does not gate spatial coarsening)\n\
                      \x20 3. Build KNN graph -> E cell pairs\n\
                      \x20    (spatial KNN from coordinates, or expression KNN\n\
                      \x20     from random-projected feature expression)\n\
                      \x20 4. Random projection of cells [N x P]\n\
                      \x20 5. Graph coarsening -> assign pairs to S samples\n\
                      \x20 6. Collapse: accumulate sigma/delta per feature per sample\n\
                      \x20    Sigma[g,s] += y_ig + y_jg\n\
                      \x20    Delta[g,s] += |y_ig - y_jg|\n\
                      \x20 7. Fit Poisson-Gamma -> posterior log means\n\
                      \x20    mu_hat[g,s] = E[ln mu_g | Y_s^g]\n\
                      \x20 8. Stack M = [mu_shared; mu_diff] [2G x S]\n\
                      \x20 9. Randomized SVD: M = U S V^T, keep top T cols\n\
                      \x20 10. Nystrom: for each pair e=(i,j):\n\
                      \x20     z_e = basis_shared^T * sigma_e + basis_diff^T * delta_e\n\
                      \x20     z_e <- z_e / ||z_e||   (L2 normalize)\n\
                      \x20 11. Cut z into link communities, then take each\n\
                      \x20     cell's propensity as its incident-edge fraction\n\n\
                      --edge-cluster-method picks the cut.\n\
                      leiden is the default,\n\
                      deciding the count from --leiden-resolution.\n\
                      kmeans instead uses a fixed --n-communities,\n\
                      spherical on the pair latent and seeded by --seed.\n\
                      `pinto apply propensity` re-cuts the same latent at a fixed K.\n\n\
                      Outputs:\n\
                      - {out}.delta.parquet: batch effects (when multi-batch)\n\
                      - {out}.delta_cells.parquet: each cell's batch, as δ's columns name it\n\
                      - {out}.coord_pairs.parquet: cell pair coordinates\n\
                      - {out}.basis.parquet: SVD basis (2G x T)\n\
                      - {out}.latent.parquet: per-pair latent codes (E x T)\n\
                      - {out}.propensity.parquet: cell propensity (N x K)\n\
                      \x20 Columns: 0 .. K-1, cluster (argmax), entropy (Shannon, nats).\n\
                      - {out}.link_community.parquet: per-edge community labels\n\
                      - {out}.feature_community.parquet: feature-community Poisson-Gamma statistics (G x K).\n\
                      \x20 Rows are scaled by the NB Fisher-info weight\n\
                      \x20 w_g = 1 / (1 + π_g · s̄ · φ(μ_g)), which attenuates\n\
                      \x20 high-mean high-dispersion features. There is no flag for it.\n\
                      - {out}.pinto.json: information-flow manifest used by\n\
                      \x20 `pinto lra --from` (lists every parquet)."
    )]
    DeltaSvd(Box<SrtDeltaSvdArgs>),

    #[command(
        about = "Activity-gated cell-graph embedding (cage)",
        long_about = "Learn SUPER-CELL PAIR embeddings on the coarsened spatial graph.\n\
                      The trained unit is a pair of adjacent finest-level super-cells\n\
                      (PBs), a super edge, with a row of its own.\n\
                      Cell-cell KNN edges fold into PB super edges up front;\n\
                      no cell and no cell pair is ever trained on.\n\
                      cage visits one feature at a time.\n\
                      Each feature defines a per-cell activity vector,\n\
                      folded onto the super edges it touches,\n\
                      which draws them as positives; negatives are other\n\
                      super edges of the same batch.\n\
                      This is embedding-only. There is no count decoder.\n\n\
                      Where most features are noise, a run takes one of two\n\
                      remedies, not both. By default features are embedded through\n\
                      learned modules, their number M taken from the data\n\
                      (--feature-modules auto; a count fixes M, 0 turns them off).\n\
                      --feature-inclusion moran instead trains each feature with\n\
                      its probability of spatial signal, and turns the default\n\
                      modules off.\n\n\
                      --n-hvg weights the random projection that builds the\n\
                      coarsening hierarchy, as in senna bge and senna gem;\n\
                      it does not subset the trained feature axis.\n\
                      Every feature is present in every output table; a feature\n\
                      TRAINS only if it is active on a super edge.\n\
                      A feature whose activity sits entirely inside super-cells\n\
                      keeps its initialization; the log counts them.\n\
                      Use --features-per-epoch to cap per-epoch cost instead.\n\n\
                      SPLICE CHANNELS are recognized on the feature axis.\n\
                      Rows named {feature}/count/spliced pair with their\n\
                      {feature}/count/unspliced counterpart.\n\
                      A feature's two rows are ONE feature everywhere the model fits.\n\
                      Their counts are summed before the log1p activity.\n\
                      Feature-side output tables are keyed by the bare feature name.\n\
                      {out}.feature_community.parquet stays on the matrix rows,\n\
                      so it still lists a feature's two channels separately.\n\
                      --n-hvg counts ROWS, then widens to whole features,\n\
                      so a feature is never half-weighted in the projection.\n\
                      A matrix mixing channel rows with plain rows is rejected.\n\
                      A {feature}/count/total row is the usual cause.\n\n\
                      With both tracks present, the manifest reports how many features\n\
                      carry counts on BOTH tracks -- the structural precondition for\n\
                      a nascent-minus-mature contrast -- and the base track that\n\
                      contrast is measured from (delta_base, `senna gem`'s sign).\n\n\
                      After training, cells return in EVALUATION only.\n\
                      Every CELL PAIR is placed on the frozen feature embedding.\n\
                      Its pooled counts x_gu + x_gv enter through one statistic,\n\
                      and a small encoder maps that statistic to the per-pair\n\
                      latent e_uv in one pass.\n\
                      The encoder trains on the finest super-cells' pairs, each\n\
                      thinned at random to a cell pair's depth every epoch,\n\
                      to reproduce the pair's trained row (its direction),\n\
                      and then only encodes the cell pairs.\n\n\
                      Clustering those pairs gives link communities.\n\
                      A cell's propensity is its incident-edge fraction.\n\
                      That is the same definition `lc` and `dsvd` use.\n\
                      --edge-cluster-method picks the cut.\n\
                      leiden is the default,\n\
                      deciding the count from --leiden-resolution.\n\
                      kmeans instead uses a fixed --n-communities\n\
                      (without it, the embedding width),\n\
                      spherical on the pair latent and seeded by --seed.\n\n\
                      A cell's embedding is the mean of its pairs' (L2-normalized) latents,\n\
                      written for `lupin annotate`.\n\
                      A cell in no pair gets a zero row.\n\n\
                      Outputs:\n\
                      \x20 {out}.pb_pair_embedding.parquet  super-cell pair × embedding_dim (trained)\n\
                      \x20 {out}.pb_bias.parquet         per-super-cell scalar (trained)\n\
                      \x20 {out}.cell_pb.parquet         cell → finest super-cell id\n\
                      \x20 {out}.cell_embedding.parquet  cell × embedding_dim (same map as the pairs)\n\
                      \x20 {out}.pair_encoder.safetensors  the pair encoder, for `pinto apply predict`\n\
                      \x20 {out}.feature_embedding.parquet  feature × embedding_dim\n\
                      \x20 {out}.feature_bias.parquet       per-feature scalar\n\
                      \x20 {out}.coord_pairs.parquet     cell pair list, tagged by kind\n\
                      \x20 {out}.latent.parquet          cell pair × embedding_dim\n\
                      \x20 {out}.propensity.parquet      cell × K, + cluster, entropy\n\
                      \x20 {out}.link_community.parquet  per-edge community\n\
                      \x20 {out}.feature_community.parquet  feature × K Poisson-Gamma rates\n\
                      \x20 {out}.scores.parquet          per-epoch loss trace\n\
                      \x20 {out}.fisher_weights.parquet  per-ROW NB precisions w_r\n\
                      \x20 {out}.delta.parquet           batch effects (multi-batch only;\n\
                      \x20                                optional `--cnv-clones` on the δ path)\n\
                      \x20 {out}.delta_cells.parquet     each cell's batch, as δ's columns name it\n\
                      \x20 {out}.pinto.json           manifest"
    )]
    Cage(Box<CellActivityGraphEmbeddingArgs>),

    #[command(
        name = "cage-cell",
        about = "cage trained on cells and cell pairs, not on pseudobulks",
        long_about = "cage trained on cells instead of super-cells, as cell-level\n\
                      methods such as STAGATE are, for a like-for-like comparison: the\n\
                      same model and outputs.\n\n\
                      The cell pair is the training unit: each within-batch spatial\n\
                      cell pair has an embedding row of its own, and a feature-active\n\
                      pair is a positive against other pairs of the same batch. Each\n\
                      cell keeps a bias. No coarsening hierarchy is trained on.\n\n\
                      Batch effects (δ) are estimated and removed as in cage. The δ\n\
                      estimate pools cells internally, only to estimate δ (so\n\
                      --num-levels and --cnv-clones shape it); training\n\
                      never sees those pools. --ignore-batch runs without δ, as STAGATE\n\
                      does.\n\n\
                      Where most features are noise, feature modules (the default,\n\
                      --feature-modules auto, a count M, or 0 for none) or\n\
                      --feature-inclusion moran apply as in cage, never both. The\n\
                      modules start from pseudobulks built for that alone; training\n\
                      pools the sparse counts of the pairs' cells each step draws.\n\n\
                      --n-pseudobulk and the HVG flags shape only those starting\n\
                      pseudobulks, and do nothing in a run without modules\n\
                      (--feature-modules 0, or --feature-inclusion moran; a warning\n\
                      says so); --feature-embedding is refused.\n\
                      --embedding-l2 defaults to 0 here, and the learning rate (--lr)\n\
                      to 0.02 (cage's is 0.005).\n\n\
                      Training runs until each pair's row has been updated\n\
                      --pair-updates times on average (default 200): the steps are\n\
                      that many times the pair count over the pairs a step touches,\n\
                      both logged. --epochs (default 1000 here) only caps it. An\n\
                      epoch draws about four positives per pair, so each update\n\
                      averages several draws at any size.\n\n\
                      The pairs' trained rows are the latent (latent.parquet, L2-normalized,\n\
                      one row per listed pair), and a cell's embedding is the mean of its\n\
                      pairs' rows. No pair encoder is fitted, so there is no\n\
                      pb_pair_embedding.parquet or pair_encoder.safetensors, and `pinto apply predict`\n\
                      and `pinto apply impute` refuse a cage-cell model for now.\n\
                      A pair table too big for the device is kept on the host.\n\n\
                      Writes what cage writes otherwise; its manifest says `cell_level`.",
        // cage-cell's own defaults, all here; the per-pair positive draws,
        // set by the data, are `args::PAIR_DRAWS_PER_EPOCH`.
        mut_arg("embedding_l2", |a| a.default_value("0")),
        mut_arg("lr", |a| a
            .default_value("0.02")
            .long_help("AdamW learning rate. cage-cell's default is 0.02 (cage's 0.005): \
                        a pair's row moves only on the steps that draw it.")),
        mut_arg("positives_per_epoch", |a| a.long_help(
            "The epoch's total positive draws, split over the features and batches \
             as in cage. Unset, 4 per cell pair (and at least 12 per feature and \
             batch), so each update of a pair's row averages several draws.")),
        mut_arg("pair_updates", |a| a.hide(false)),
        // What cage-cell refuses, or warns does nothing on cells, is off its
        // help.
        mut_args(cell_activity_graph_embedding::fit::hide_on_cells)
    )]
    CageCell(Box<CellActivityGraphEmbeddingArgs>),

    #[command(
        about = "SpatialDE2 tissue segmentation: Poisson DP-HMRF (baseline)",
        long_about = concat!(
            "SpatialDE2's tissue segmentation (Kats, Vento-Tormo and Stegle 2021),\n\
             faithful to its reference code at the model level, on pinto's inputs:\n\
             the data, -c, -b, -o and --seed flags lc takes.\n\n\
             Sparse: each cell's --neighbors nearest in its section, the\n\
             reference's own neighbors=k option. The reference's default graph,\n\
             every pair of cells connected, costs too much time and memory at\n\
             real cell counts and is not run.\n\n",
            baseline_batch!(),
            baseline_memory!(),
            baseline_outputs!()
        )
    )]
    Spatialde2(Box<baselines::spatial::spatialde2::Args>),

    #[command(
        about = "STAGATE: graph-attention autoencoder on the spatial graph (baseline)",
        long_about = concat!(
            "STAGATE (Dong and Zhang 2022), faithful to its reference code at the\n\
             model level, on pinto's inputs: the data, -c, -b, -o and --seed flags\n\
             lc takes. Its embedding is clustered by pinto's own methods: seeded\n\
             k-means into --n-clusters domains, or Leiden without it. With\n\
             --n-clusters this differs from the STAGATE tutorial, which fits a\n\
             Gaussian mixture in R at that step.\n\n\
             Sparse: on the CPU the input stays sparse and the loss is summed over\n\
             blocks of cells, the same numbers as the original's dense input up to\n\
             float rounding, without its time and memory at real cell counts. A\n\
             GPU (--device) holds the selected genes dense.\n\n",
            baseline_batch!(),
            baseline_memory!(),
            baseline_outputs!()
        )
    )]
    Stagate(Box<baselines::spatial::stagate::Args>),

    #[command(
        about = "SpatialPCA: spatially aware probabilistic PCA (baseline)",
        long_about = concat!(
            "SpatialPCA (Shang and Zhou 2022), written from the paper, on pinto's\n\
             inputs: the data, -c, -b, -o and --seed flags lc takes. Probabilistic\n\
             PCA whose spatial PCs carry a spatial prior, clustered into domains by\n\
             pinto's seeded k-means (--n-clusters) or Leiden in place of the\n\
             paper's Walktrap and Louvain, then refined by the paper's majority\n\
             vote over spatial neighbours.\n\n\
             Sparse: the paper's Gaussian kernel is replaced by a sparse precision\n\
             on the kNN graph within each section, its strength fitted with the\n\
             rest; no N x N matrix is held, as the paper's dense kernel costs too\n\
             much time and memory at real cell counts. spatialpca-pb fits on\n\
             super-cells for scale.\n\
             Genes: SPARK-X's spatially variable genes (FDR 0.05, at most\n\
             --n-genes); expression: SCTransform's Pearson residuals, as the\n\
             paper preprocesses.\n\n",
            baseline_batch!(),
            baseline_memory!(),
            baseline_outputs!()
        )
    )]
    Spatialpca(Box<baselines::spatial::spatialpca::Args>),

    #[command(
        about = "MEFISTO: factor analysis with spatially smooth factors (baseline)",
        long_about = concat!(
            "MEFISTO (Velten et al. 2022), written from the paper, on pinto's inputs:\n\
             the data, -c, -b, -o and --seed flags lc takes. Bayesian factor\n\
             analysis with sparse weights, the sections as groups, each factor\n\
             spatially smooth. The paper does not cluster; pinto's seeded k-means\n\
             (--n-clusters) or Leiden turns the factors into domains.\n\n\
             Sparse: each factor's Gaussian process is replaced by a sparse\n\
             precision on the kNN graph within each section, its strength fitted\n\
             per factor; no N x N matrix is held, as the paper's Gaussian\n\
             processes cost too much time and memory at real cell counts.\n\
             mefisto-pb fits on super-cells for scale.\n\n",
            baseline_batch!(),
            baseline_memory!(),
            baseline_outputs!()
        )
    )]
    Mefisto(Box<baselines::spatial::mefisto::Args>),

    #[command(
        name = "spatialde2-pb",
        about = "spatialde2 fitted on cage's super-cells, unravelled to cells (baseline)",
        long_about = concat!(
            "spatialde2 fitted on super-cells, then unravelled to cells.\n\n\
             The cells are coarsened as cage coarsens them, to about -d super-cells\n\
             within each section. A super-cell's counts are its cells' summed, and a\n\
             super-edge's weight sums the cell-level weights of the cell-cell edges\n\
             it collapses: the cells' model with each super-cell's cells sharing a\n\
             region. Each cell's region posterior is then re-inferred on its own\n\
             counts and the cell graph, the fit's region rates held fixed. Takes\n\
             spatialde2's flags.\n\n",
            baseline_batch!(),
            baseline_memory!(),
            baseline_outputs!()
        )
    )]
    Spatialde2Pb(Box<baselines::spatial::spatialde2::PbArgs>),

    #[command(
        name = "stagate-pb",
        about = "stagate trained on cage's super-cells, unravelled to cells (baseline)",
        long_about = concat!(
            "stagate trained on super-cells, then unravelled to cells.\n\n\
             The cells are coarsened as cage coarsens them, to about -d super-cells\n\
             within each section, each with its cells' counts summed, on the cell\n\
             graph folded onto them (unweighted, as STAGATE's attention takes it).\n\
             The trained encoder then embeds each cell from its own counts on the\n\
             cell graph, and each cell takes the nearest centroid, on the cells'\n\
             embedding, of the domains clustered on the super-cells. Sparse, as\n\
             stagate on the CPU. Takes stagate's flags.\n\n",
            baseline_batch!(),
            baseline_memory!(),
            baseline_outputs!()
        )
    )]
    StagatePb(Box<baselines::spatial::stagate::PbArgs>),

    #[command(
        name = "spatialpca-pb",
        about = "spatialpca fitted on cage's super-cells, unravelled to cells (baseline)",
        long_about = concat!(
            "spatialpca fitted on super-cells, then unravelled to cells.\n\n\
             The cells are coarsened as cage coarsens them, to about -d super-cells\n\
             within each section, each with its cells' counts summed, on the cell\n\
             graph collapsed onto them (a super-edge weighted by the cell-cell\n\
             edges it collapses). The sparse model is fitted there; its loadings\n\
             are then held fixed, the noise and prior scales refitted on the cells\n\
             and their graph, and each cell's spatial PCs inferred. Takes\n\
             spatialpca's flags.\n\n",
            baseline_batch!(),
            baseline_memory!(),
            baseline_outputs!()
        )
    )]
    SpatialpcaPb(Box<baselines::spatial::spatialpca::PbArgs>),

    #[command(
        name = "mefisto-pb",
        about = "mefisto fitted on cage's super-cells, unravelled to cells (baseline)",
        long_about = concat!(
            "mefisto fitted on super-cells, then unravelled to cells.\n\n\
             The cells are coarsened as cage coarsens them, to about -d super-cells\n\
             within each section, each with its cells' counts summed, on the cell\n\
             graph collapsed onto them. The sparse model is fitted there; its\n\
             weights are then held fixed while the noise, the factors and each\n\
             factor's smoothness are refitted on the cells and their graph. Takes\n\
             mefisto's flags.\n\n",
            baseline_batch!(),
            baseline_memory!(),
            baseline_outputs!()
        )
    )]
    MefistoPb(Box<baselines::spatial::mefisto::PbArgs>),
}

/// `pinto apply <op>`: what reads a fitted run and writes from it.
#[derive(Subcommand, Debug)]
enum Apply {
    #[command(
        about = "Apply a trained run to a new sample (cage), or score held-out cells (lc, dsvd)",
        long_about = "Apply a trained run to a new sample.\n\n\
                      LC / DSVD MODELS -- held-out scoring. Fit lc without a donor\n\
                      (leave out its files), then:\n\
                      \x20 pinto apply predict donor.zarr.zip -c donor.cells.parquet \\\n\
                      \x20     --model fit_without_donor -o held_out\n\
                      Each cell pair of the donor's graph (--holdout-unit cell for\n\
                      cells) is split in two: binomial thinning of its counts\n\
                      (--holdout-fit-fraction), or with --ablate-features the listed\n\
                      features hidden from the fit and exactly they scored, as in\n\
                      `senna predict`. Its community mix is fitted by EM on one part\n\
                      and the other part is scored. The donor's per-batch shift δ is\n\
                      estimated from the fitting parts only (--holdout-delta-iters).\n\
                      {out}.predictive.parquet has the columns cage and senna write\n\
                      (eval_llik, eval_count, eval_llik_per_count, spearman,\n\
                      pearson_log1p, eval_null_llik_per_count) plus\n\
                      eval_pooled_k1_llik_per_count. The null is the training data's\n\
                      composition: --null-from, else the model's own data files.\n\
                      {out}.pinto.json carries a `holdout` block with the totals and\n\
                      the ceiling. Score a non-spatial lc fit (no -c) the same way\n\
                      for the spatial baseline.\n\n\
                      CAGE MODELS -- the rest of this help.\n\
                      The feature side and the community dictionary transfer;\n\
                      the geometry is rebuilt from the new sample's own coordinates.\n\n\
                      TYPICAL USE -- score a trained run on a held-out half:\n\
                      \x20 data-beans split data.zarr -o cv --test-frac 0.2 \\\n\
                      \x20     --coord positions.csv --coord-columns 4,5 --grid 8\n\
                      \x20 pinto fit cage cv.train.zarr.zip -o model -c positions.csv\n\
                      \x20 pinto apply predict cv.test.zarr.zip --model model -o pred \\\n\
                      \x20     -c positions.csv --null-from cv.train.zarr.zip \\\n\
                      \x20     --eval-features panel.txt\n\n\
                      ALWAYS pass --null-from the TRAINING half. Those per-feature\n\
                      totals are not only the null: they are b_g, half of the\n\
                      pair log-rate b_g + <e_g, e_uv>. Taken from the query the\n\
                      prediction is anchored on the data being scored, and llik\n\
                      is not held out at all. Same flag, same meaning as\n\
                      `senna predict --null-from`; pass the same half to both.\n\n\
                      Split by REGION (--coord/--grid), not at random: adjacent\n\
                      cells are near-duplicates, so a random split leaves every\n\
                      test cell ringed by training cells.\n\n\
                      Pass the same --eval-features file to every arm, and to\n\
                      `senna predict`, or the arms are graded on different features.\n\n\
                      Steps:\n\
                      \x20 1. Preprocess the new data as cage does (graph, batches).\n\
                      \x20    Batch effects: a batch the model trained on is divided by the\n\
                      \x20    model's own {model}.delta.parquet; only a new batch is\n\
                      \x20    estimated on the sample, carried into the model's frame through\n\
                      \x20    the batches the two share. Batches match by --batch-files label;\n\
                      \x20    without one, by data file name (s1.zarr is batch s1), when the\n\
                      \x20    model labelled by file too. A model without a δ\n\
                      \x20    trained on undivided counts: its query is not divided.\n\
                      \x20    {out}.delta.parquet is the δ used. --ignore-batch: none,\n\
                      \x20    refused for a model fitted on δ-divided counts (its manifest's\n\
                      \x20    feature_community_batch_adjusted): its query would not be on\n\
                      \x20    the model's scale. An lc / dsvd model's held-out sample: a\n\
                      \x20    batch it knows takes its δ, a new one is estimated from the\n\
                      \x20    fitting parts (--holdout-delta-iters 0: left unshifted).\n\
                      \x20 2. Align {model}.feature_embedding.parquet to its feature axis by name.\n\
                      \x20    Features without a model row are dropped, never seeded.\n\
                      \x20 3. Place every cell pair, and every cell, on the frozen dictionary\n\
                      \x20    by the model's pair encoder ({model}.pair_encoder.safetensors),\n\
                      \x20    under the ridge it was fitted with. Nothing is optimized per pair.\n\
                      \x20 4. Assign each pair to the nearest trained link community.\n\
                      \x20    The centroids are recomputed from {model}.latent\n\
                      \x20    and {model}.link_community; a pair that matches none abstains.\n\
                      \x20 5. Propensity is the incident-edge fraction, per community.\n\
                      \x20    The cell embedding is the cell's own placement from step 3.\n\n\
                      Outputs:\n\
                      \x20 {out}.coord_pairs.parquet, {out}.latent.parquet,\n\
                      \x20 {out}.link_community.parquet, {out}.propensity.parquet,\n\
                      \x20 {out}.feature_community.parquet, {out}.cell_embedding.parquet,\n\
                      \x20 {out}.pinto.json (command = predict), readable by\n\
                      \x20 `lupin annotate` like a fitted run,\n\
                      \x20 {out}.predictive.parquet, one row per cell PAIR.\n\n\
                      SCORING. predictive.parquet holds eval_llik, eval_count,\n\
                      eval_llik_per_count, eval_null_llik_per_count and, with\n\
                      --eval-features, spearman and pearson_log1p -- the same\n\
                      column names `senna predict` uses for the same quantities,\n\
                      so one script reads both.\n\n\
                      The likelihood is a multinomial over the scored features, so\n\
                      eval_llik_per_count is nats per observed count. A pair pools\n\
                      two cells, but nats per count does not care how many cells\n\
                      went in. Rank on eval_llik_per_count MINUS\n\
                      eval_null_llik_per_count.\n\n\
                      Rows are PAIRS here and CELLS in senna, so the two tables\n\
                      compare in aggregate, not row by row -- do not join them.\n\
                      A pair with no counts carries NaN; filter total > 0 before\n\
                      averaging a *_per_count column.\n\n\
                      --eval-features restricts the likelihood AND the\n\
                      correlations to those features, which is what makes the number\n\
                      comparable with senna's. It is off by default because a\n\
                      pair-level correlation sorts the feature axis once per pair,\n\
                      and a sample has far more pairs than cells. Pass the same\n\
                      file to both commands."
    )]
    Predict(Box<PredictArgs>),

    #[command(
        about = "Impute full-feature counts for a new sample by kNN over community propensities.",
        long_about = "Retrieval-based imputation against a trained run's cells:\n  \
                      1. Place the query cells on the model's community propensity.\n  \
                      \x20  A cage model runs the full `pinto apply predict` pipeline\n  \
                      \x20  (its usual outputs land under {out}); lc / dsvd models\n  \
                      \x20  project each cell onto the feature_community profiles\n  \
                      \x20  by a per-cell EM fit — and project the reference\n  \
                      \x20  cells the same way, so both sides come from one map.\n  \
                      2. For each query cell, find its nearest reference cells\n  \
                      \x20  in propensity space, softmax-weight the distances,\n  \
                      \x20  and accumulate those cells' full-feature counts.\n\
                      \x20  For a cage model each reference cell's counts are first\n\
                      \x20  divided by the run's δ for its batch (the batches as\n\
                      \x20  estimated, from {model}.delta_cells.parquet, or the\n\
                      \x20  coord_pairs of a run fitted before it), as predict\n\
                      \x20  divides the query. An lc / dsvd model fitted on δ-divided\n\
                      \x20  counts (its manifest's feature_community_batch_adjusted)\n\
                      \x20  puts both sides on its profiles' scale: the reference cells\n\
                      \x20  divided so, in the EM and in the pooling, and the query by\n\
                      \x20  the model's δ for a batch it knows, by an estimate (moved\n\
                      \x20  into the model's frame) for a new one, as predict divides a\n\
                      \x20  cage query (--auto-batch: no graph is built here to find its\n\
                      \x20  components, so the query is one new batch, undivided). An\n\
                      \x20  older lc / dsvd model leaves both sides\n\
                      \x20  undivided, on its raw-count profiles' scale. --ignore-batch:\n\
                      \x20  no δ, refused for a model fitted on δ-divided counts.\n\
                      \n\
                      The reference data defaults to the files recorded in\n\
                      {model}.pinto.json; --reference-data overrides.\n\
                      Predict-stage flags (coordinates, -k, eval flags)\n\
                      apply to cage models only.\n\
                      \n\
                      Writes {out}.imputed.parquet (N_query × n_ref_features)."
    )]
    Impute(Box<ImputeArgs>),

    #[command(
        name = "donor-deviation",
        about = "Stage 2: which stage-1 domains are donor-specific, and which genes drive them",
        long_about = donor_deviation::DONOR_DEVIATION_LONG_ABOUT
    )]
    DonorDeviation(Box<donor_deviation::DonorDeviationArgs>),

    #[command(
        name = "propensity",
        about = "Estimate vertex propensity from edge clusters (standalone)",
        long_about = "Estimate vertex (cell) propensity scores from edge\n\
                      (cell-pair) cluster assignments.\n\n\
                      NOTE: cage and dsvd produce propensity and edge outputs inline.\n\
                      Use this subcommand to re-cut the same latent,\n\
                      or for separate expression data.\n\
                      It is the fixed-K path:\n\
                      --edge-cluster-method kmeans --n-communities K.\n\n\
                      Model:\n\
                      \x20 Given latent codes z_e [E x T] from cage or delta-svd:\n\
                      \x20   c_e = the pair's community, cut by leiden (default)\n\
                      \x20        or by spherical kmeans, argmax_k cos(z_e, centroid_k)\n\
                      \x20 For each vertex i:\n\
                      \x20   p_i[k] = |{e incident to i : c_e = k}| / degree(i)\n\
                      \x20 Optionally, cluster-specific feature expression:\n\
                      \x20   mu_{g,k} ~ Gamma(a0, b0) with pseudocount sums\n\n\
                      Algorithm:\n\
                      \x20 1. Load latent codes Z [E x T] from .latent.parquet\n\
                      \x20 2. Load cell pair names from .coord_pairs.parquet\n\
                      \x20 3. Cluster Z^T -> assignment c_e for each edge\n\
                      \x20 4. For each vertex i, count edges per cluster:\n\
                      \x20    p_i[k] = count(c_e=k for e incident to i) / deg(i)\n\
                      \x20 5. dominant_cluster[i] = argmax_k p_i[k]\n\
                      \x20 6. If expression data provided:\n\
                      \x20    weighted feature sums per cluster -> Poisson-Gamma\n\n\
                      Inputs (all passed by flag; there is no positional arg):\n\
                      - -z/--latent-data-file: .latent.parquet (from cage or delta-svd)\n\
                      - -e/--coord-pair-file: .coord_pairs.parquet (cell pair names)\n\
                      - --data-files: expression data (.zarr or .h5), optional\n\n\
                      Outputs:\n\
                      - {out}.propensity.parquet: per-vertex propensity (N x K)\n\
                      \x20 Columns: C0 .. C{K-1}, cluster (argmax),\n\
                      \x20 entropy (Shannon, nats), plus optional coord trailer.\n\
                      - {out}.link_community.parquet: per-edge community labels\n\
                      - {out}.features.parquet: cluster-specific feature expression (with --data-files).\n\
                      \x20 Rows are scaled by the NB Fisher-info weight\n\
                      \x20 w_g = 1 / (1 + π_g · s̄ · φ(μ_g)). There is no flag for it.\n\
                      - {out}.pinto.json: information-flow manifest used by\n\
                      \x20 downstream tools."
    )]
    Propensity(Box<SrtPropensityArgs>),
}

/// `pinto lra`, and `pinto lra simulate` under it.
#[derive(clap::Args, Debug)]
struct Lra {
    #[command(subcommand)]
    simulate: Option<LraSub>,

    #[command(flatten)]
    args: SrtLrActivityArgs,
}

#[derive(Subcommand, Debug)]
enum LraSub {
    #[command(
        name = "simulate",
        about = "Power and calibration check for lra on this tissue",
        long_about = lr_activity::lra_simulate_long_about()
    )]
    Simulate(Box<LraSimulateArgs>),
    #[command(
        name = "merge",
        about = "Put the shards of one `pinto lra --shard` run together",
        long_about = "Read every shard file of one `pinto lra --shard I/N` run\n\
                      ({out}.lra.shard-I-of-N.cbor, given in any order), check they are its\n\
                      N shards (the same settings and pairs, each shard once, every pair\n\
                      once), and write the tables a run without --shard writes: the\n\
                      local fdrs and verdicts over every pair, recorded in the run's manifest\n\
                      (the --from run's, else {out}.pinto.json). Given effect tables\n\
                      instead, collects the marginal and --link-community tables into\n\
                      {out}.lra.by-link-community.parquet."
    )]
    Merge(Box<lr_activity::args::LraMergeArgs>),
    #[command(
        name = "link-communities",
        about = "List the link communities `pinto lra --link-community` can run on",
        long_about = "List the --from run's link communities over its spatial contacts, one per\n\
                      line: the community, its links, its receivers (cells with a link in\n\
                      it; an upper bound, as a cell missing from the data or without\n\
                      coordinates is dropped when the community is fitted) and whether\n\
                      --min-receivers lets `pinto lra --link-community` fit it. In community order, so line I of the body can be task I of\n\
                      a job array."
    )]
    LinkCommunities(Box<lr_activity::args::LraLinkCommunitiesArgs>),
}

/// `pinto sim <op>`: simulation with a planted truth, and scoring against it.
#[derive(Subcommand, Debug)]
enum Sim {
    #[command(
        name = "new",
        about = "Simulate multi-donor spatial data with a planted truth",
        long_about = "Simulate several donors' sections with a planted truth.\n\n\
                      Location sets a spot's domain, the domain its likely cell states, and\n\
                      the states its expression. Donor and section nuisance (to be removed)\n\
                      and a domain one donor alone has (to be kept) are set as proportions\n\
                      of the log rate's variance; spot noise is what the shares leave, so\n\
                      raising one share with the others fixed shrinks the noise (the log\n\
                      says what a setting implies). Counts are negative binomial through a\n\
                      low-rank Gaussian copula, from keyed random numbers: --set\n\
                      nuisance_off=true with the same seed draws the exact counterfactual.\n\n\
                      --reference-data takes the expression from a real data set (.zarr.zip,\n\
                      .zarr or .h5): each planted state takes one of its cell groups, its\n\
                      counts are drawn from the group's own distributions through the\n\
                      reference's copula, and the nuisance shares are of its variance between\n\
                      states (the manifest's realized_pve_expressed_genes says how close).\n\
                      Convert an AnnData file first: data-beans from-h5ad x.h5ad -o x.\n\n\
                      Writes:\n\
                      \x20 {out}/sections/S*.zarr.zip       counts per section (genes × spots)\n\
                      \x20 {out}/sections/S*.coords.parquet spot coordinates (row_um, col_um)\n\
                      \x20 {out}/truth/*.parquet            spots, domains, states, genes\n\
                      \x20 {out}/truth/lr_pairs.tsv         ligand-receptor pairs of its genes, and\n\
                      \x20   targets.tsv their response programs, for lra --lr-pairs --targets\n\
                      \x20 {out}/manifest.json              every parameter and the realized read-outs"
    )]
    New(Box<simulate::SimulateArgs>),

    #[command(
        name = "score-by-set",
        alias = "score",
        about = "Score labellings against a simulation's planted truth",
        long_about = "Score how well labellings recover a `sim new` run's planted truth,\n\
                      as two tests.\n\n\
                      1. Donor-invariant domains, on spots of shared domains, labels\n\
                      \x20  against domains: shared_ari (adjusted Rand index); shared_ami\n\
                      \x20  (adjusted mutual information, Vinh et al. 2010: MI less its exact\n\
                      \x20  expectation under random labellings of the same sizes, over the\n\
                      \x20  arithmetic mean of the two entropies less it; 0 chance, 1 the same\n\
                      \x20  partition); shared_nmi (MI over the arithmetic mean of the two\n\
                      \x20  entropies; not corrected for chance, so many small labels raise it);\n\
                      \x20  donor_conditional_entropy, H(donor | label, domain) / H(donor |\n\
                      \x20  domain), 0 to 1 (1: labels mix donors fully, 0: the label sets the\n\
                      \x20  donor), pooled over the shared domains by their spots and for each;\n\
                      \x20  a domain in one donor is skipped and noted.\n\
                      2. Donor-specific domains, each within its owning donor's spots only\n\
                      \x20  (within: owner_donor in the JSON): its best-matching label there,\n\
                      \x20  and that label's f1, precision (low: the owner's shared spots\n\
                      \x20  absorbed) and recall, all over the owner's spots.\n\n\
                      Labellings are pinto runs at --level, or tables with a cell column\n\
                      (--label-column names the labels)."
    )]
    Score(Box<sim_eval::SimScoreArgs>),

    #[command(
        name = "score-by-pair",
        alias = "pairs",
        about = "Score per-cell vectors against a simulation's planted truth, on pairs",
        long_about = "Score how well per-cell vectors (a latent mapping) recover an\n\
                      `sim new` run's planted truth, on pairs of spots and with no\n\
                      clustering: the cosine of two spots' vectors, by AUROC and AUPRC, on\n\
                      these sets of pairs, each with its prevalence.\n\n\
                      shared_domain_across_donors: shared-domain spots of two donors,\n\
                      \x20 positive when they share a domain. shared_domain_within_donor: the\n\
                      \x20 same within one donor, the reference; donor_gap is within less\n\
                      \x20 across, what the donor costs the mapping.\n\
                      For each donor-specific domain: donor_domain_vs_own_donor, pairs of\n\
                      \x20 its spots (positive) against one of its spots with a spot of the\n\
                      \x20 same donor outside it (negative; negatives: same_donor_outside_domain\n\
                      \x20 in the JSON), so telling donors apart scores 0.5.\n\
                      neighbour_same_domain: each spot's --knn nearest neighbours in its\n\
                      \x20 section, positive when they share a domain.\n\n\
                      Vectors are pinto runs (their propensity at --level; lc, dsvd and\n\
                      cage alike), a spatial baseline's cell_embedding table, or any table\n\
                      with a cell column and numeric columns. Pairs are drawn from --seed, --pairs per sampled set."
    )]
    Pairs(Box<sim_eval::SimPairsArgs>),

    #[command(
        name = "baseline",
        about = "Label a simulation with a standard pipeline, as a baseline",
        long_about = "Label a `sim new` run's spots with the standard scanpy recipe, as a\n\
                      baseline for sim score-by-set: total-count normalization to 1e4, log1p,\n\
                      highly variable genes by normalized dispersion, scaling, PCA, a kNN\n\
                      graph and Leiden tuned toward --n-clusters clusters (default: the simulation's\n\
                      domains).\n\n\
                      --method pooled runs all sections together, with no batch handling;\n\
                      --method harmony corrects the PCs for the donor with Harmony\n\
                      (Korsunsky et al. 2019) before the kNN graph; --method per-donor\n\
                      runs each donor alone and gives each cluster the label of the most\n\
                      correlated cluster of the largest donor (marker-correlation matching).\n\
                      Writes a cell<TAB>label table."
    )]
    Baseline(Box<baselines::single_cell::BaselineArgs>),
}

#[cfg(feature = "view")]
/// `pinto session <op>`: the detachable `pinto run` and `pinto view`
/// sessions.
#[derive(Subcommand, Debug)]
enum Session {
    #[command(
        about = "Return to a detached `pinto run` or `pinto view`",
        long_about = legume_tui::cli::about(&tui::TOOL, "ctrl+c in it").attach
    )]
    Attach(legume_tui::cli::AttachArgs),

    #[command(
        name = "list",
        about = "List the live `pinto run` and `pinto view` sessions",
        long_about = legume_tui::cli::about(&tui::TOOL, "ctrl+c in it").sessions
    )]
    List,

    #[command(
        about = "Stop a `pinto run` or `pinto view` session as ctrl+c in it does",
        long_about = format!(
            "{}\n\n{}",
            legume_tui::cli::about(&tui::TOOL, "ctrl+c in it").kill,
            "`pinto run` stops the fits it is running first; with no terminal\n\
             attached, the lines it ends on are said here, and each fit's\n\
             `{out}.cmd.sh` is there to run it again. `pinto view` lets a\n\
             relabelling lupin is writing finish and stops any other lupin run."
        )
    )]
    Kill(legume_tui::cli::KillArgs),

    #[command(
        about = "Give a `pinto run` or `pinto view` session a label",
        long_about = legume_tui::cli::about(&tui::TOOL, "ctrl+c in it").rename
    )]
    Rename(legume_tui::cli::RenameArgs),
}

impl Commands {
    /// A fit's `--out`, and the name its files take in a folder `--out`.
    fn out_mut(&mut self) -> Option<(&mut Box<str>, &'static str)> {
        Some(match self {
            Commands::Fit(fit) => match fit {
                Fit::LinkCommunity(a) => (&mut a.common.out, "lc"),
                Fit::DeltaSvd(a) => (&mut a.common.out, "dsvd"),
                Fit::Cage(a) => (&mut a.common.out, "cage"),
                Fit::CageCell(a) => (&mut a.common.out, "cage-cell"),
                Fit::Spatialde2(a) => (&mut a.fit.input.out, "spatialde2"),
                Fit::Stagate(a) => (&mut a.input.out, "stagate"),
                Fit::Spatialpca(a) => (&mut a.fit.core.input.out, "spatialpca"),
                Fit::Mefisto(a) => (&mut a.fit.core.input.out, "mefisto"),
                Fit::Spatialde2Pb(a) => (&mut a.fit.input.out, "spatialde2-pb"),
                Fit::StagatePb(a) => (&mut a.fit.input.out, "stagate-pb"),
                Fit::SpatialpcaPb(a) => (&mut a.fit.core.input.out, "spatialpca-pb"),
                Fit::MefistoPb(a) => (&mut a.fit.core.input.out, "mefisto-pb"),
            },
            Commands::Apply(Apply::Propensity(a)) => (&mut a.out, "prop"),
            Commands::Apply(Apply::Predict(a)) => (&mut a.core.common.out, "predict"),
            Commands::Apply(Apply::Impute(a)) => (&mut a.predict.common.out, "impute"),
            Commands::Lra(a) => match &mut a.simulate {
                Some(LraSub::Simulate(s)) => (s.out.as_mut()?, "lra"),
                Some(LraSub::Merge(m)) => (&mut m.out, "lra"),
                Some(LraSub::LinkCommunities(_)) => return None,
                None => (a.args.out.as_mut()?, "lra"),
            },
            _ => return None,
        })
    }
}

/// An `--out` ending in `/` is a folder: the fit writes `{folder}/{name}.*`
/// (`--out res/` on `fit lc` gives `res/lc`), and makes the folder as it
/// makes any `--out`'s parent.
fn name_folder_out(commands: &mut Commands) {
    if let Some((out, name)) = commands.out_mut() {
        if out.ends_with('/') {
            *out = format!("{out}{name}").into();
        }
    }
}

/// Under `--atomic-outputs`, each command's out made its staged one
/// ([`util::staging`]): a fit's prefix, sim new's folder,
/// the file sim score-by-set, sim score-by-pair and sim baseline write, compare's
/// prefix. lra without `--out` stages its `--from` run's prefix when it
/// resolves it.
fn stage_outputs(commands: &mut Commands) -> anyhow::Result<()> {
    use util::staging::{stage_folder, stage_prefix, stage_run};
    let prefix = |o: &mut Box<str>| -> anyhow::Result<()> {
        *o = stage_prefix(o)?.into();
        Ok(())
    };
    // lra changes the manifest at its out and never replaces it; a fit
    // writes its own
    let changes = matches!(commands, Commands::Lra(_));
    if let Some((out, _)) = commands.out_mut() {
        if changes {
            return prefix(out);
        }
        *out = stage_run(out)?.into();
        return Ok(());
    }
    match commands {
        Commands::Sim(Sim::New(a)) => a.out = stage_folder(&a.out)?.into(),
        Commands::Sim(Sim::Score(a)) => a.out.as_mut().map_or(Ok(()), prefix)?,
        Commands::Sim(Sim::Pairs(a)) => a.out.as_mut().map_or(Ok(()), prefix)?,
        Commands::Sim(Sim::Baseline(a)) => prefix(&mut a.out)?,
        Commands::Compare(a) => a.out = stage_prefix(&a.out)?,
        // lra stages as it names its out; view writes its stills in
        // place, and the rest write no files of their own
        _ => {}
    }
    Ok(())
}

/// `--max-threads` handed to the commands whose results depend on it: lc's
/// Gibbs splits a component into up to that many blocks.
fn pass_max_threads(cli: &mut Cli) {
    if let Commands::Fit(Fit::LinkCommunity(a)) = &mut cli.commands {
        a.max_threads = cli.max_threads;
    }
}

/// The flags of a fit that `--feature-modules` reads: its input (made to
/// read module files), its `--feature-modules` and lc's
/// `--feature-network`. `None` for cage, which builds its own module layer,
/// and for any other command.
fn module_inputs(
    commands: &mut Commands,
) -> Option<(
    &mut util::input::InputArgs,
    &mut util::feature_modules::FeatureModulesArgs,
    bool,
)> {
    fn spatial(
        s: &mut baselines::spatial::input::SpatialInputArgs,
    ) -> (
        &mut util::input::InputArgs,
        &mut util::feature_modules::FeatureModulesArgs,
        bool,
    ) {
        (&mut s.input, &mut s.modules, false)
    }
    let Commands::Fit(fit) = commands else {
        return None;
    };
    Some(match fit {
        Fit::LinkCommunity(a) => (
            &mut a.common.input,
            &mut a.modules,
            a.feature_network.is_some(),
        ),
        Fit::DeltaSvd(a) => (&mut a.common.input, &mut a.modules, false),
        Fit::Cage(_) | Fit::CageCell(_) => return None,
        Fit::Spatialde2(a) => spatial(&mut a.fit.input),
        Fit::Spatialde2Pb(a) => spatial(&mut a.fit.input),
        Fit::Stagate(a) => spatial(&mut a.input),
        Fit::StagatePb(a) => spatial(&mut a.fit.input),
        Fit::Spatialpca(a) => spatial(&mut a.fit.core.input),
        Fit::SpatialpcaPb(a) => spatial(&mut a.fit.core.input),
        Fit::Mefisto(a) => spatial(&mut a.fit.core.input),
        Fit::MefistoPb(a) => spatial(&mut a.fit.core.input),
    })
}

/// A fit's `--feature-modules` resolved (see [`util::feature_modules`]):
/// under modules its data put on them, before it reads anything. With the
/// fit's out and what its manifest records; `None` for cage and for
/// commands that fit nothing.
fn prepare_feature_modules(
    commands: &mut Commands,
) -> anyhow::Result<Option<(String, util::feature_modules::FeatureModulesInfo)>> {
    use cell_activity_graph_embedding::args::FeatureInclusionMode;
    use util::feature_modules::{plan, prepare, prepare_moran, FeatureModulesInfo};
    let Some((input, modules, network)) = module_inputs(commands) else {
        return Ok(None);
    };
    // the modules' files sit beside the fit's, written before the fit makes
    // its out prefix's folder
    if let Some(dir) = std::path::Path::new(&*input.out)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
    {
        std::fs::create_dir_all(dir)?;
    }
    let moran = modules.feature_inclusion == FeatureInclusionMode::Moran;
    let (p, yielded) = plan(modules.feature_modules, moran, network)?;
    let info = if moran {
        prepare_moran(input)?
    } else if !yielded.is_empty() {
        FeatureModulesInfo::features(yielded)
    } else {
        prepare(input, p)?
    };
    // modules, or moran's features: either way the fit's own screen is off
    modules.on = info.module_level() || info.setting == "moran";
    Ok(Some((input.out.to_string(), info)))
}

fn main() -> anyhow::Result<()> {
    // removes what a run staged when it ends before its commit: on an
    // error return and on a panic
    let _abort = util::staging::AbortOnDrop;
    run_command().map_err(util::staging::final_error)?;
    util::staging::commit()?;
    // what the run held back to say once its outputs are in place
    util::staging::say_held();
    Ok(())
}

/// Parse the command line and run its command.
fn run_command() -> anyhow::Result<()> {
    // A session's host runs this binary again: it serves the session and
    // exits here, before any of pinto's own work.
    #[cfg(feature = "view")]
    legume_tui::serve_if_host();

    // Rust ignores SIGPIPE, so printing into a closed pipe (`pinto … | head`)
    // panics. Take the default back: the process just ends, as other
    // command-line tools do.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        print_logo();
    }

    let mut cmd = <Cli as clap::CommandFactory>::command();
    let argv: Vec<String> = std::env::args().collect();
    // `Cli::parse_from`, keeping the command and its matches: cage-cell,
    // predict and impute ask which flags were given rather than left at
    // their defaults.
    let matches = cmd
        .try_get_matches_from_mut(argv)
        .unwrap_or_else(|e| e.exit());
    let mut cli = <Cli as clap::FromArgMatches>::from_arg_matches(&matches).unwrap_or_else(|e| {
        e.format(&mut <Cli as clap::CommandFactory>::command())
            .exit()
    });
    name_folder_out(&mut cli.commands);
    if cli.atomic_outputs {
        util::staging::enable();
        stage_outputs(&mut cli.commands)?;
    }

    // Rayon's own count, at most --max-threads.
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let env = std::env::var("RAYON_NUM_THREADS").ok();
    rayon::ThreadPoolBuilder::new()
        .num_threads(util::common::pool_threads(
            cli.max_threads,
            env.as_deref(),
            cores,
        ))
        .build_global()?;
    pass_max_threads(&mut cli);
    #[cfg(feature = "view")]
    tui::keep_max_threads(cli.max_threads);
    if let Commands::Fit(Fit::CageCell(a)) = &mut cli.commands {
        a.cell_level = true;
    }

    crate::util::common::init_logger(cli.verbose);

    // The command run, as built, and its matches.
    let (c, m) = cli_path::ran(&cmd, &matches);

    let modules = prepare_feature_modules(&mut cli.commands)?;

    match &cli.commands {
        Commands::Fit(fit) => {
            match fit {
                Fit::LinkCommunity(args) => fit_srt_link_community(args)?,
                Fit::DeltaSvd(args) => fit_srt_delta_svd(args)?,
                Fit::Cage(args) => fit_cell_activity_graph_embedding(args)?,
                Fit::CageCell(args) => {
                    let mut args = args.clone();
                    args.seeding_flags =
                        cell_activity_graph_embedding::fit::check_cell_level(c, m)?;
                    fit_cell_activity_graph_embedding(&args)?;
                }
                Fit::Spatialde2(args) => baselines::spatial::spatialde2::run(args)?,
                Fit::Spatialde2Pb(args) => baselines::spatial::spatialde2::run_pb(args)?,
                Fit::Stagate(args) => baselines::spatial::stagate::run(args)?,
                Fit::StagatePb(args) => baselines::spatial::stagate::run_pb(args)?,
                Fit::Spatialpca(args) => baselines::spatial::spatialpca::run(args)?,
                Fit::SpatialpcaPb(args) => baselines::spatial::spatialpca::run_pb(args)?,
                Fit::Mefisto(args) => baselines::spatial::mefisto::run(args)?,
                Fit::MefistoPb(args) => baselines::spatial::mefisto::run_pb(args)?,
            }
            let recorded = match (fit, modules) {
                (Fit::Cage(a) | Fit::CageCell(a), _) => Some((
                    a.common.out.to_string(),
                    util::feature_modules::cage_info(a)?,
                )),
                (_, m) => m,
            };
            if let Some((out, info)) = recorded {
                util::feature_modules::record(&out, &info)?;
            }
        }
        Commands::Apply(op) => match op {
            Apply::Predict(args) => {
                // The return value serves `pinto apply impute`; the CLI path
                // only wants the files predict writes.
                let (_propensity, _cell_names) = predict(args, &ModelOnlyFlags::given(c, m))?;
            }
            Apply::Impute(args) => {
                run_impute(args, &util::input::SrtInputArgs::qc_flags_given(c, m))?;
            }
            Apply::Propensity(args) => fit_srt_propensity(args)?,
            Apply::DonorDeviation(args) => donor_deviation::run_donor_deviation(args)?,
        },
        Commands::Lra(lra) => match &lra.simulate {
            Some(LraSub::Simulate(args)) => fit_lra_simulate(args)?,
            Some(LraSub::Merge(args)) => {
                use lr_activity::ligand_effect::output::collect;
                if collect::are_tables(&args.shards) {
                    collect::collect_link_community_tables(&args.shards, &args.out)?;
                } else {
                    lr_activity::ligand_effect::pipeline::merge(&args.shards, &args.out)?
                }
            }
            Some(LraSub::LinkCommunities(args)) => {
                lr_activity::ligand_effect::link_community_scope::list_link_communities(args)?
            }
            None => fit_srt_lr_activity(&lra.args)?,
        },
        Commands::Sim(op) => match op {
            Sim::New(args) => simulate::run_simulate(args)?,
            Sim::Score(args) => sim_eval::run_score(args)?,
            Sim::Pairs(args) => sim_eval::run_pairs(args)?,
            Sim::Baseline(args) => baselines::single_cell::run_baseline(args)?,
        },
        Commands::Compare(args) => run_compare(args)?,
        #[cfg(feature = "view")]
        Commands::View(args) => {
            view::run_view(args)?;
        }
        #[cfg(feature = "view")]
        Commands::Mcp => legume_mcp::serve(
            legume_mcp::ServerInfo {
                name: view::mcp::SERVER.into(),
                version: env!("CARGO_PKG_VERSION").into(),
                instructions: Some(mcp::instructions()),
            },
            mcp::Server::default(),
        )?,
        #[cfg(feature = "view")]
        Commands::Run(args) => {
            run::tui::run(built_cli(), args)?;
        }
        #[cfg(feature = "view")]
        Commands::Session(op) => match op {
            Session::Attach(args) => legume_tui::cli::attach(&tui::TOOL, args)?,
            Session::List => legume_tui::cli::sessions(&tui::TOOL)?,
            Session::Kill(args) => legume_tui::cli::kill(&tui::TOOL, args)?,
            Session::Rename(args) => legume_tui::cli::rename(&tui::TOOL, args)?,
        },
    }

    Ok(())
}

#[cfg(test)]
#[path = "tests/main.rs"]
mod tests;

#[cfg(all(test, feature = "view"))]
#[path = "tests/session_cli.rs"]
mod session_cli_tests;

#[cfg(test)]
#[path = "tests/surface.rs"]
mod surface_tests;
