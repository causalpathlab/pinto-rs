//! `pinto mcp`: pinto as a Model Context Protocol server, the viewer's
//! tools ([`crate::view::mcp`]) and `pinto run`'s ([`crate::run::mcp`]) on
//! one stdio server.

use legume_mcp::{Call, Content, Handler, ToolSpec};
use serde_json::Value;

/// What the agent is told when it connects: what the tools are for, and in
/// which order to call them. `pinto mcp --help` prints the same, for clients
/// that do not pass it on.
pub fn instructions() -> String {
    format!("{FITS} {SIMULATE} {LRA} {LOOKING}")
}

const FITS: &str = "\
Use pinto's MCP tools to fit spatial link-community models and to look at \
them on the tissue. `run_methods` lists the methods and, with a method, its \
flags; `run_fit` runs one as a background job (follow it with `job_status` / \
`job_wait`) and never writes over a run. pinto's models are lc (link \
communities) and dsvd (shared and difference channels by SVD), the two \
recommended, and cage (a learned cell-graph embedding; cage-cell trains it \
on cells); spatialde2, stagate, spatialpca and mefisto, and the -pb forms \
spatialde2-pb, stagate-pb, spatialpca-pb and mefisto-pb (fitted on cage's \
super-cells) are spatial baselines, ports of published methods to compare pinto \
against, fit and scored as lc is. Where most \
features are noise, every method works on feature modules by default \
(feature-modules auto takes their number from the data, a count fixes it, 0 \
for none), all built the same way: cage and cage-cell embed features through \
a learned module layer started from them, the others are fitted on the \
modules' summed counts (written as {out}.feature_modules/, the run's data \
files). \
feature-inclusion moran, on every method, given alone replaces the default \
modules (cage weights features by their probability of spatial signal; the \
others fit the features whose probability is 0.5 or more), and given beside \
an explicit feature-modules value is refused; lc's feature-network (a network \
file the user supplies) likewise replaces the default modules, and beside an \
explicit feature-modules is refused. Ablations are fits (lc, dsvd, cage \
or a baseline) with run_fit's ablation flags (permute-coords, drop-features, \
mask-cells with ablation-parent); `run_region` turns a polygon into the cell \
list mask-cells takes; `run_compare` sets two runs' communities against each \
other; `run_predict` scores held-out cells on a fitted model.";

const SIMULATE: &str = "\
Simulation with a planted truth, step by step. 1. `run_simulate` (a background \
job) writes each section as `<out>/sections/S<n>.zarr.zip` (counts, genes by \
spots) with `<out>/sections/S<n>.coords.parquet` (row_um, col_um), and the \
truth in `<out>/truth`; `view_open` its `<out>/truth/truth.pinto.json` shows \
the planted domains. 2. Fit with `run_fit`, passing the reply's `fit.data` \
(the `.zarr.zip` sections, in order), `fit.coords` (their coordinate files, in \
the same order) and `fit.flags` exactly as its data, coords and flags; give \
no other data path. 3. Score each fit against the reply's `truth`: \
`run_sim_score_by_set` with the fit's out prefix in `labels` scores its labels; \
`run_sim_score_by_pair` scores per-cell vectors on pairs of spots, with no \
clustering: for lc, dsvd and cage give the fit's out prefix in `vectors` \
(its propensity; never cage's `{prefix}.cell_embedding.parquet` nor dsvd's \
`{prefix}.latent.parquet`, which are not the communities' propensity); `run_sim_baseline` writes a standard \
pipeline's labels table to put in `labels` beside them. A spatial baseline's \
run is scored as a fit is: its prefix in `labels`, its \
`{out}.cell_embedding.parquet` in `vectors`. Cells are named \
`barcode@S<n>`, as a fit of those sections names them, so the scores join \
on every spot. Both scorers reply with one JSON object, `{truth, scores: \
[...], failed: [...]}` (`failed` only when an input could not be scored), \
and `pinto sim score-by-set -o` / `pinto sim score-by-pair -o` write that same object.";

const LRA: &str = "\
Ligand-receptor analysis: `run_lra` asks, per ligand-receptor pair, whether \
receivers touching ligand-carrying neighbours change their response \
program; `run_lra_simulate` plants effects of known size on the same \
tissue and reports power and type I error. Run lra simulate to check power \
and calibration before trusting lra on a new tissue. A large tissue can run \
in shards: `run_lra` with `shard` I/N for each I, each shard a separate job, \
then `run_lra_merge` on the N shard files writes the tables of the whole run. \
`run_lra_link_communities` lists a run's link communities for `run_lra`'s \
`link_community`. \
On a cluster, `run_lra` with `cluster_shards` N runs nothing and writes a \
legume benchmark spec of the N shards and their merge (null draws too, with \
`null_permutations`); hand it to legume-mcp: `benchmark_plan` and \
`benchmark_start` with that spec and `on` the cluster, then `benchmark_status` \
until it ends. lra judges nothing: each pair's effect, its \
effect in receptor-negative receivers and their contrast come with SE, df and \
p, and co-expression with its local fdr; the reader decides. `view_lra` lists \
a run's pairs with their statistics and, given a pair, makes `view_map` draw \
it; `view_export` with `curve: true` writes its threshold-response curve.";

const LOOKING: &str = "\
Looking: start with `view_open` on a run prefix (the `out` of a \
fit, or its `.pinto.json`): it lists the run's levels, batches, groups and any \
lupin rounds. `view_map` draws the map as a PNG with its legend; its arguments \
stay set until changed, so pass only what changes. `view_markers` lists a \
group's top features, `view_cell` names the cell under a pixel of the last map \
(or at a world point, or by name), and `view_export` writes the current view as \
PNG/PDF files. World coordinates are those `view_open` reports, with y running \
down the map. lupin rounds are read here as they are on disk; annotating and \
relabelling them is lupin's own server.";

/// The server: the viewer's tools and the run tools.
#[derive(Default)]
pub struct Server {
    view: crate::view::mcp::Tools,
    run: crate::run::mcp::Tools,
}

impl Handler for Server {
    fn tools(&self) -> Vec<ToolSpec> {
        let mut tools = self.run.tools();
        tools.extend(self.view.tools());
        tools
    }

    fn call(&mut self, name: &str, args: Value, cx: &Call) -> anyhow::Result<Vec<Content>> {
        if name.starts_with("run_") {
            self.run.call(name, args, cx)
        } else {
            self.view.call(name, args, cx)
        }
    }
}

pub use crate::util::paths::shown_path;
pub use legume_mcp::parse_args as parse;
