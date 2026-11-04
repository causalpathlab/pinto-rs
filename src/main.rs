mod annotate;
mod cell_activity_graph_embedding;
mod compare;
mod feature_network;
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
mod svd;
#[cfg(feature = "view")]
mod tui;
mod util;
#[cfg(feature = "view")]
mod view;

#[cfg(test)]
mod test_support;

use annotate::AnnotateArgs;
use cell_activity_graph_embedding::{
    fit_cell_activity_graph_embedding, CellActivityGraphEmbeddingArgs,
};
use clap::{Parser, Subcommand};
use colored::Colorize;
use compare::{run_compare, CompareArgs};
use impute::{run_impute, ImputeArgs};
use link_community::fit::*;
use lr_activity::{fit_lra_simulate, fit_srt_lr_activity, LraSimulateArgs, SrtLrActivityArgs};
use predict::{predict, PredictArgs};
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
        "Proximity-based Interaction Network --> Tissue Organization".bold()
    );
    println!();
}

/// PINTO
#[derive(Parser, Debug)]
#[command(
    version,
    about = "PINTO - Proximity-based Interaction Network for Tissue Organization",
    long_about = "PINTO discovers cell-cell interaction patterns.\n\
                  It reads spatial transcriptomics.\n\
                  It detects link communities on cell-pair graphs.\n\n\
                  SUBCOMMANDS:\n\n\
                  \x20 lc    Link community model (recommended)\n\
                  \x20       Assigns each cell-cell edge to a community via collapsed\n\
                  \x20       Gibbs sampling on compressed all-feature edge profiles.\n\n\
                  \x20 dsvd  Delta-SVD model\n\
                  \x20       Cell-pair shared/difference analysis via Poisson-Gamma\n\
                  \x20       SVD on pseudobulk co-expression.\n\n\
                  \x20 prop  Propensity (standalone)\n\
                  \x20       Re-cut a cage/dsvd edge latent at a different K.\n\n\
                  \x20 view  Viewer (multi-resolution)\n\
                  \x20       Zoomable map of a run's communities, in the terminal.\n\n\
                  \x20 run   Set up fits in the terminal\n\
                  \x20       Choose data, methods and flags, then run them.\n\n\
                  \x20 mcp   Server for agents (Model Context Protocol)\n\
                  \x20       pinto run's fits and the viewer's map as tools, on stdio.\n\n\
                  QUICK START:\n\n\
                  \x20 # Prepare data (convert MTX to HDF5):\n\
                  \x20 data-beans from-mtx -r features.tsv.gz -c barcodes.tsv.gz \\\n\
                  \x20   matrix.mtx.gz --backend hdf5 -o data.h5\n\n\
                  \x20 # Link community (spatial, 10x Visium):\n\
                  \x20 pinto lc data.h5 -c tissue_positions.csv -o results\n\n\
                  \x20 # Link community (expression-only, no coordinates):\n\
                  \x20 pinto lc data.h5 -o results\n\n\
                  \x20 # Delta-SVD:\n\
                  \x20 pinto dsvd data.h5 -c coords.csv -o results\n\n\
                  INPUT FILES:\n\n\
                  \x20 Data:   .h5 or .zarr (features x cells, sparse). Multiple files\n\
                  \x20         comma-separated for multi-sample: s1.h5,s2.h5\n\
                  \x20 Coords: CSV/TSV/parquet, first column = barcode, rest = x,y,...\n\
                  \x20         Default columns: pxl_row_in_fullres,pxl_col_in_fullres\n\
                  \x20         Omit -c for expression-only mode.\n\
                  \x20 Batch:  -b labels.txt (one label per cell per line, optional)\n\n\
                  OUTPUT: All outputs are .parquet files with {out} prefix.\n\
                  \x20 Use --help on each subcommand for output file details.",
    term_width = 80
)]
struct Cli {
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
// Measured, not assumed: this fires only since `cage` gained `--eval-block-file` — removing
// the attribute reproduces the warning, and it did not fire before that flag. The variants
// are clap arg structs, exactly one is constructed per process at startup, and clap's
// `Subcommand` derive cannot take a boxed payload, so the layout cost the lint is about
// does not exist here.
#[allow(clippy::large_enum_variant)]
enum Commands {
    #[command(
        alias = "dsvd",
        about = "Feature-level shared/difference analysis by SVD",
        long_about = "Feature-level cell-cell interaction analysis by SVD.\n\
                      It uses shared and difference channels.\n\n\
                      Model:\n\
                      \x20 For each cell pair e=(i,j) and feature g:\n\
                      \x20   sigma_e^g = log1p(x_ig) + log1p(x_jg)    shared\n\
                      \x20   delta_e^g = |log1p(x_ig) - log1p(x_jg)|  difference\n\
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
                      \x20    Sigma[g,s] += log1p(x_ig) + log1p(x_jg)\n\
                      \x20    Delta[g,s] += |log1p(x_ig) - log1p(x_jg)|\n\
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
                      kmeans instead uses a fixed --n-edge-clusters,\n\
                      spherical on the pair latent and seeded by --seed.\n\
                      `pinto prop` re-cuts the same latent at a fixed K.\n\n\
                      Outputs:\n\
                      - {out}.delta.parquet: batch effects (when multi-batch)\n\
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
    DeltaSvd(SrtDeltaSvdArgs),

    #[command(
        alias = "prop",
        about = "Estimate vertex propensity from edge clusters (standalone)",
        long_about = "Estimate vertex (cell) propensity scores from edge\n\
                      (cell-pair) cluster assignments.\n\n\
                      NOTE: cage and dsvd produce propensity and edge outputs inline.\n\
                      Use this subcommand to re-cut the same latent,\n\
                      or for separate expression data.\n\
                      It is the fixed-K path:\n\
                      --edge-cluster-method kmeans --n-edge-clusters K.\n\n\
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
                      - -d/--expr-data-files: expression data (.zarr or .h5), optional\n\n\
                      Outputs:\n\
                      - {out}.propensity.parquet: per-vertex propensity (N x K)\n\
                      \x20 Columns: C0 .. C{K-1}, cluster (argmax),\n\
                      \x20 entropy (Shannon, nats), plus optional coord trailer.\n\
                      - {out}.link_community.parquet: per-edge community labels\n\
                      - {out}.features.parquet: cluster-specific feature expression (with -d/--expr-data-files).\n\
                      \x20 Rows are scaled by the NB Fisher-info weight\n\
                      \x20 w_g = 1 / (1 + π_g · s̄ · φ(μ_g)). There is no flag for it.\n\
                      - {out}.pinto.json: information-flow manifest used by\n\
                      \x20 downstream tools."
    )]
    Propensity(SrtPropensityArgs),

    #[command(
        alias = "lc",
        about = "Link community model via collapsed Gibbs sampling",
        long_about = "Link community detection for spatial transcriptomics.\n\n\
                      Each cell-cell edge is assigned to one of K communities.\n\
                      The assignment reads per-edge expression profiles.\n\
                      Per-cell soft membership follows from those labels.\n\n\
                      QUICK START:\n\n\
                      \x20 # Typical spatial run (10x Visium):\n\
                      \x20 pinto lc data.h5 -c tissue_positions.csv -o out\n\n\
                      \x20 # More communities:\n\
                      \x20 pinto lc data.h5 -c coords.csv -o out --n-communities 25\n\n\
                      \x20 # Expression-only (no coordinates):\n\
                      \x20 pinto lc data.h5 -o out\n\n\
                      \x20 # With external feature-feature network:\n\
                      \x20 pinto lc data.h5 -c coords.csv -o out \\\n\
                      \x20   --feature-network biogrid_pairs.tsv\n\n\
                      \x20 # Multi-sample with batch correction:\n\
                      \x20 pinto lc s1.h5,s2.h5 -c c1.csv,c2.csv -o out\n\n\
                      INPUT FILES:\n\n\
                      \x20 data.h5 / data.zarr   Features-by-cells sparse matrix.\n\
                      \x20                        Convert from MTX: data-beans from-mtx in.mtx out.h5\n\
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
                      \x20   (no features dropped); M = proj-dim just compresses the feature axis.\n\
                      \x20   Optionally zero basis rows for features below --min-feature-count.\n\n\
                      \x20 Feature-network module-pair profile (--feature-network file.tsv):\n\
                      \x20   External feature-feature edges (two-column TSV), optionally SNN-\n\
                      \x20   augmented, k-core-trimmed, Leiden-clustered into feature modules.\n\
                      \x20   Edge profile is SPARSE over module-pairs (a, b) with entries\n\
                      \x20   max(0, x_{i,a}·x_{j,b} + x_{i,b}·x_{j,a} − X_i·X_j · deg(a)·deg(b)/(2W)²).\n\
                      \x20   Controls: --snn-min-shared, --feature-trim-min-degree,\n\
                      \x20   --feature-modules-resolution.\n\n\
                      ALGORITHM:\n\n\
                      \x20 1. Build spatial KNN graph (or expression KNN if no coords)\n\
                      \x20 2. Batch effect estimation (multi-sample only;\n\
                      \x20    optional `--cnv-clones` from `mung clones` so private CN\n\
                      \x20    stays out of δ — does not gate spatial coarsening)\n\
                      \x20 3. Multi-level graph coarsening\n\
                      \x20 4. Resolve feature modules (projection or SNN + k-core + Leiden)\n\
                      \x20 5. Build sparse edge profiles (projection or module-pair residual)\n\
                      \x20 6. V-cycle Gibbs + greedy across coarsening levels\n\
                      \x20 7. Component-EM + greedy on full fine-resolution edges\n\
                      \x20 8. Extract cell propensity + feature-community statistics (+ cosine dictionary merge)\n\n\
                      See `pinto lc --help` for individual flag docs.\n\n\
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
                      \x20 {out}.delta.parquet           Batch effects (multi-sample only)\n\
                      \x20 {out}.feature_graph.parquet      Feature-feature pairs (feature-pair mode only)\n\
                      \x20 {out}.L{l}.*.parquet          Per-cascade-level outputs (unless --no-level-outputs)\n\
                      \x20 {out}.draft.*.parquet         Pre-merge fine partition (when dictionary merge collapsed)\n\
                      \x20 {out}.dict_merges.parquet     Cosine merge tree over the feature-community dictionary\n\
                      \x20 {out}.dict_merges.cut.parquet Fine→super community remap from --merge-cut\n\
                      \x20 {out}.pinto.json           Information-flow manifest:\n\
                      \x20                                lists every parquet, level tags,\n\
                      \x20                                dict-merge presence, and (when set by\n\
                      \x20                                pinto lra) its tables.\n\
                      \x20                                Pass this path to `pinto lra --from`."
    )]
    LinkCommunity(SrtLinkCommunityArgs),

    #[command(
        alias = "cge",
        about = "Activity-gated cell-graph embedding (cage)",
        long_about = "Learn per-SUPER-CELL embeddings on the coarsened spatial graph.\n\
                      The trained unit is a finest-level super-cell (PB).\n\
                      Cell-cell KNN edges fold into PB super edges up front;\n\
                      no cell and no cell pair is ever trained on.\n\
                      cage visits one feature at a time.\n\
                      Each feature defines a per-cell activity vector,\n\
                      folded onto the super edges it touches.\n\
                      That gates a shared multi-scale PB hierarchy.\n\n\
                      Chain levels differ only in their negative pools.\n\
                      This is embedding-only. There is no count decoder.\n\n\
                      NOTE --n-hvg no longer subsets the trained feature axis.\n\
                      It weights the random projection instead.\n\
                      That projection builds the coarsening hierarchy.\n\
                      senna bge and senna gem do the same.\n\
                      Every feature is present in every output table; a feature\n\
                      TRAINS only if it is active on a super edge.\n\
                      A feature whose activity sits entirely inside super-cells\n\
                      keeps its initialization; the log counts them.\n\
                      Use --features-per-epoch to cap per-epoch cost instead.\n\n\
                      SPLICE CHANNELS are recognised on the feature axis.\n\
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
                      and a small encoder trained on this run's pairs\n\
                      maps that statistic to the per-pair latent e_uv in one pass.\n\
                      --pair-ridge sets the prior it is fitted under.\n\
                      A seeded sample of pairs is always re-solved exactly\n\
                      and the agreement is logged;\n\
                      the few placements the certificate puts far out\n\
                      are finished exactly.\n\n\
                      Clustering those pairs gives link communities.\n\
                      A cell's propensity is its incident-edge fraction.\n\
                      That is the same definition `lc` and `dsvd` use.\n\
                      --edge-cluster-method picks the cut.\n\
                      leiden is the default,\n\
                      deciding the count from --leiden-resolution.\n\
                      kmeans instead uses a fixed --n-edge-clusters,\n\
                      spherical on the pair latent and seeded by --seed.\n\n\
                      A cell's embedding is its own placement on the feature embedding,\n\
                      by the same encoder that places its pairs,\n\
                      written for `pinto annotate`.\n\
                      A cell with no counts gets a zero row.\n\n\
                      Outputs:\n\
                      \x20 {out}.pb_embedding.parquet    super-cell × embedding_dim (trained)\n\
                      \x20 {out}.pb_bias.parquet         per-super-cell scalar (trained)\n\
                      \x20 {out}.cell_pb.parquet         cell → finest super-cell id\n\
                      \x20 {out}.cell_embedding.parquet  cell × embedding_dim (same map as the pairs)\n\
                      \x20 {out}.pair_encoder.safetensors  the pair encoder, for `pinto predict`\n\
                      \x20 {out}.feature_embedding.parquet  feature × embedding_dim\n\
                      \x20 {out}.pseudobulk_cells.parquet  cell × (coords, super-cell, e_pb)\n\
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
                      \x20 {out}.pinto.json           manifest"
    )]
    Cage(CellActivityGraphEmbeddingArgs),

    #[command(
        visible_alias = "cage-annotate",
        about = "Marker-set cell-type annotation by projection (any embedding run)",
        long_about = "Moved to `lupin annotate`. Run `lupin annotate --help`."
    )]
    Annotate(AnnotateArgs),

    #[command(
        about = "Apply a trained run to a new sample (cage), or score held-out cells (lc, dsvd)",
        long_about = "Apply a trained run to a new sample.\n\n\
                      LC / DSVD MODELS -- held-out scoring. Fit lc without a donor\n\
                      (leave out its files), then:\n\
                      \x20 pinto predict donor.zarr.zip -c donor.cells.parquet \\\n\
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
                      \x20 pinto cage cv.train.zarr.zip -o model -c positions.csv\n\
                      \x20 pinto predict cv.test.zarr.zip --model model -o pred \\\n\
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
                      \x20 1. Preprocess the new data as cage does (graph, batches)\n\
                      \x20 2. Align {model}.feature_embedding.parquet to its feature axis by name.\n\
                      \x20    Features without a model row are dropped, never seeded.\n\
                      \x20 3. Place every cell pair, and every cell, on the frozen dictionary\n\
                      \x20    by the model's pair encoder ({model}.pair_encoder.safetensors),\n\
                      \x20    under the ridge it was fitted with. Nothing is optimised per pair.\n\
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
                      \x20 `pinto annotate` like a fitted run,\n\
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
    Predict(PredictArgs),

    #[command(
        about = "Impute full-feature counts for a new sample by kNN over community propensities.",
        long_about = "Retrieval-based imputation against a trained run's cells:\n  \
                      1. Place the query cells on the model's community propensity.\n  \
                      \x20  A cage model runs the full `pinto predict` pipeline\n  \
                      \x20  (its usual outputs land under {out}); lc / dsvd models\n  \
                      \x20  project each cell onto the feature_community profiles\n  \
                      \x20  by a per-cell EM fit — and project the reference\n  \
                      \x20  cells the same way, so both sides come from one map.\n  \
                      2. For each query cell, find its nearest reference cells\n  \
                      \x20  in propensity space, softmax-weight the distances,\n  \
                      \x20  and accumulate those cells' full-feature counts.\n\
                      \n\
                      The reference data defaults to the files recorded in\n\
                      {model}.pinto.json; --reference-data overrides.\n\
                      Predict-stage flags (coordinates, -k, --pair-block,\n\
                      eval flags) apply to cage models only.\n\
                      \n\
                      Writes {out}.imputed.parquet (N_query × n_ref_features)."
    )]
    Impute(ImputeArgs),

    #[command(
        name = "lra",
        aliases = ["lr-activity", "test-lr"],
        about = "Ligand-receptor analysis: does neighbouring ligand change the receiving cells?",
        long_about = lr_activity::lra_long_about()
    )]
    LrActivity(SrtLrActivityArgs),

    #[command(
        name = "lra-simulate",
        about = "Power and calibration check for lra on this tissue",
        long_about = lr_activity::lra_simulate_long_about()
    )]
    LraSimulate(LraSimulateArgs),

    #[command(
        about = "Compare two runs' communities over the cells they share",
        long_about = "Compare two runs' hard community labels, joined on cell name.\n\n\
                      Over the cells both runs placed it reports the adjusted Rand\n\
                      index, normalised mutual information, a one-to-one community\n\
                      matching (Hungarian, on Jaccard) and each community's fate.\n\
                      A community's partners are the other run's communities holding\n\
                      at least --tau of its cells: two or more is a split (from A) or\n\
                      a merge (into B); none means the other run has no counterpart\n\
                      (vanished from A, new in B).\n\n\
                      Writes {out}.compare.json and {out}.contingency.parquet.\n\n\
                      Examples:\n\
                      \x20 pinto compare base.pinto.json perm.pinto.json -o base_vs_perm\n\
                      \x20 pinto compare base perm --level-a draft --tau 0.3 -o cmp"
    )]
    Compare(CompareArgs),

    #[cfg(feature = "view")]
    #[command(
        about = "View a run's communities on the tissue (multi-resolution)",
        long_about = format!(
            "View a pinto run's cells and communities on the tissue.\n\n\
             Reads {{prefix}}.pinto.json and the parquet files it lists:\n\
             \x20 cells            every cell's coordinates and batch label\n\
             \x20                  (older runs: from coord_pairs)\n\
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
    View(view::ViewArgs),

    #[cfg(feature = "view")]
    #[command(
        about = "Serve pinto run and the viewer to agents (Model Context Protocol, stdio)",
        long_about = "Serve pinto to an agent as a Model Context Protocol server on\n\
                      stdin and stdout. run_methods and run_fit set up and run fits as\n\
                      `pinto run` does, as background jobs unless asked to wait. The\n\
                      viewer's tools: view_open opens a run, view_map draws it as a PNG\n\
                      with its legend, view_markers, view_cell and view_export ask\n\
                      about it and write figures.\n\
                      The view is kept between calls, as `pinto view` keeps it.\n\
                      lupin rounds are shown as they are on disk.\n\n\
                      Any MCP client runs it as the command `pinto mcp`. The project's\n\
                      records go to its working directory, or to LEGUME_MCP_PROJECT;\n\
                      LEGUME_MCP_RECORDS=off keeps none. `pinto view --follow` shows\n\
                      a person what the agent does. The README has each client's setup.\n\n\
                      Examples:\n\
                      \x20 claude mcp add pinto -e LEGUME_MCP_PROJECT=\"$PWD\" -- pinto mcp\n\
                      \x20 codex mcp add pinto -- pinto mcp"
    )]
    Mcp,

    #[cfg(feature = "view")]
    #[command(
        about = "Set up fits in the terminal and run them",
        long_about = format!(
            "Set up pinto fits in the terminal and run them.\n\n\
             Pick the data files with the coordinate and batch files of\n\
             each, queue one or more of lc, cage and dsvd, and change\n\
             their flags. Every flag a method has is listed with its help;\n\
             hidden ones under `a`.\n\n\
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
        about = "Return to a detached `pinto run` or `pinto view`",
        long_about = legume_tui::cli::about(&tui::TOOL, "ctrl+c in it").attach
    )]
    Attach(legume_tui::cli::AttachArgs),

    #[cfg(feature = "view")]
    #[command(
        about = "List the live `pinto run` and `pinto view` sessions",
        long_about = legume_tui::cli::about(&tui::TOOL, "ctrl+c in it").sessions
    )]
    Sessions,

    #[cfg(feature = "view")]
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

    #[cfg(feature = "view")]
    #[command(
        about = "Give a `pinto run` or `pinto view` session a label",
        long_about = legume_tui::cli::about(&tui::TOOL, "ctrl+c in it").rename
    )]
    Rename(legume_tui::cli::RenameArgs),
}

/// A helpful error for a flag `pinto lra` no longer takes, before clap's
/// plain "unexpected argument".
fn check_removed_lra_flags(args: &[String]) -> anyhow::Result<()> {
    const LRA_NAMES: &[&str] = &["lra", "lr-activity", "test-lr"];
    let Some(pos) = args.iter().position(|a| LRA_NAMES.contains(&a.as_str())) else {
        return Ok(());
    };
    match lr_activity::args::removed_lra_flag(&args[pos + 1..]) {
        Some(msg) => anyhow::bail!(msg),
        None => Ok(()),
    }
}

impl Commands {
    /// A fit's `--out`, and the name its files take in a folder `--out`.
    fn out_mut(&mut self) -> Option<(&mut Box<str>, &'static str)> {
        Some(match self {
            Commands::Propensity(a) => (&mut a.out, "prop"),
            Commands::DeltaSvd(a) => (&mut a.common.out, "dsvd"),
            Commands::LinkCommunity(a) => (&mut a.common.out, "lc"),
            Commands::Cage(a) => (&mut a.common.out, "cage"),
            Commands::Predict(a) => (&mut a.common.out, "predict"),
            Commands::Impute(a) => (&mut a.predict.common.out, "impute"),
            Commands::LrActivity(a) => (a.out.as_mut()?, "lra"),
            Commands::LraSimulate(a) => (a.out.as_mut()?, "lra"),
            _ => return None,
        })
    }
}

/// An `--out` ending in `/` is a folder: the fit writes `{folder}/{name}.*`
/// (`--out res/` on `lc` gives `res/lc`), and makes the folder as it makes
/// any `--out`'s parent.
fn name_folder_out(commands: &mut Commands) {
    if let Some((out, name)) = commands.out_mut() {
        if out.ends_with('/') {
            *out = format!("{out}{name}").into();
        }
    }
}

fn main() -> anyhow::Result<()> {
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

    let argv: Vec<String> = std::env::args().collect();
    check_removed_lra_flags(&argv)?;
    let mut cli = Cli::parse_from(argv);
    name_folder_out(&mut cli.commands);

    crate::util::common::init_logger(cli.verbose);

    match &cli.commands {
        Commands::Propensity(args) => {
            fit_srt_propensity(args)?;
        }
        Commands::DeltaSvd(args) => {
            fit_srt_delta_svd(args)?;
        }
        Commands::LinkCommunity(args) => {
            fit_srt_link_community(args)?;
        }
        Commands::Cage(args) => {
            fit_cell_activity_graph_embedding(args)?;
        }
        Commands::Annotate(_args) => {
            eprintln!(
                "The `pinto annotate` command moved to `lupin annotate`.\n\
                 Run `lupin annotate --help` for usage."
            );
            std::process::exit(1);
        }
        Commands::Predict(args) => {
            // The return value serves `pinto impute`; the CLI path only wants
            // the files predict writes.
            let (_propensity, _cell_names) = predict(args)?;
        }
        Commands::Impute(args) => {
            run_impute(args)?;
        }
        Commands::LrActivity(args) => {
            fit_srt_lr_activity(args)?;
        }
        Commands::LraSimulate(args) => {
            fit_lra_simulate(args)?;
        }
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
            use clap::CommandFactory;
            let mut cli = Cli::command();
            cli.build();
            run::tui::run(cli, args)?;
        }
        #[cfg(feature = "view")]
        Commands::Attach(args) => legume_tui::cli::attach(&tui::TOOL, args)?,
        #[cfg(feature = "view")]
        Commands::Sessions => legume_tui::cli::sessions(&tui::TOOL)?,
        #[cfg(feature = "view")]
        Commands::Kill(args) => legume_tui::cli::kill(&tui::TOOL, args)?,
        #[cfg(feature = "view")]
        Commands::Rename(args) => legume_tui::cli::rename(&tui::TOOL, args)?,
    }

    Ok(())
}

#[cfg(test)]
#[path = "tests/main.rs"]
mod tests;

#[cfg(all(test, feature = "view"))]
#[path = "tests/session_cli.rs"]
mod session_cli_tests;
