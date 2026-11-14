//! `pinto mcp`: pinto as a Model Context Protocol server, the viewer's
//! tools ([`crate::view::mcp`]) and `pinto run`'s ([`crate::run::mcp`]) on
//! one stdio server.

use legume_mcp::{Call, Content, Handler, ToolSpec};
use serde_json::Value;

/// What the agent is told when it connects: what the tools are for, and in
/// which order to call them. The README gives the same, for clients that
/// do not pass it on. The verdicts are `pinto lra --help`'s, from the one
/// table ([`crate::lr_activity::ligand_effect::verdict::Verdict`]).
pub fn instructions() -> String {
    format!(
        "{FITS} {SIMULATE} {RESIDUAL} {LRA} Each pair's verdict is one of: {}. {LOOKING}",
        crate::lr_activity::ligand_effect::verdict::verdict_line()
    )
}

const FITS: &str = "\
Use pinto's MCP tools to fit spatial link-community models and to look at \
them on the tissue. A fit: `run_methods` lists the methods (lc is the \
recommended one) and, with a method, its flags; `run_fit` runs one on data \
files with their coordinates, in the background for anything but a small \
data set (follow its job with `job_status` / `job_wait`), and never writes \
over a run. Its methods also hold three spatial baselines, model-level \
ports of published methods to compare pinto against: banksy (BANKSY), \
spatialde2 (SpatialDE2's tissue segmentation) and stagate (STAGATE); \
run_fit runs them as it runs lc, and each writes a run with its labels and \
a cell embedding. Ablations are fits with run_fit's ablation flags \
(permute-coords, drop-features, mask-cells with ablation-parent); \
`run_region` turns a polygon into the cell list mask-cells takes; \
`run_compare` sets two runs' communities against each other (ARI, NMI, \
matched communities, and which vanish, split or merge); `run_predict` scores \
held-out cells on a fitted lc model.";

const SIMULATE: &str = "\
Simulation with a planted truth: `run_simulate` writes donors' sections \
(a background job); `view_open` its `<out>/truth/truth.pinto.json` shows the \
planted domains; fit with `run_fit`, passing the reply's `fit.data` and \
`fit.coords` (the sections and their coordinates, in order) as its data and \
coords, and `fit.flags` as its flags; `run_sim_score` scores the fits' labels against the reply's truth, \
`run_sim_pairs` scores their per-cell vectors on pairs of spots, with no clustering, and \
`run_sim_baseline` writes a standard pipeline's labels to score next to them; \
a spatial baseline's run is scored as a fit is, its labels by `run_sim_score` and \
its `{out}.cell_embedding.parquet` by `run_sim_pairs`.";

const RESIDUAL: &str = "\
Two-stage fit: `run_residual` writes the counts an lc, cage or dsvd run \
leaves unexplained as new data files (a background job); fit them with \
`run_fit`, passing the reply's `fit.data`, `fit.coords` and `fit.flags` as \
its data, coords and flags, and `run_lra` on that second fit reads the raw \
counts, not the excess.";

const LRA: &str = "\
Ligand-receptor analysis: `run_lra` on a run asks, per ligand-receptor pair, \
whether receivers touching ligand-carrying neighbours change their response \
programme compared with similar cells that do not (adjusted for region, \
section, cell identity, neighbour make-up, depth and segmentation bleed, with \
a negative control); `run_lra_simulate` plants effects of known size on the \
same tissue and reports, per scenario, power and type I error beside the \
co-expression baselines. \
Run lra-simulate to check power and calibration before trusting lra on a new \
tissue. `run_lra` with `only` estimates a few pairs into a subset table of its \
own (its q corrected within the subset). `view_lra` lists a run's pairs with \
their verdicts and, given a pair, makes `view_map` draw it: senders, and \
receivers exposed or not and receptor-competent or not; `view_export` with \
`curve: true` writes its threshold-response curve.";

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

/// The arguments of a call, or what is wrong with them.
pub fn parse<T: serde::de::DeserializeOwned>(args: Value) -> anyhow::Result<T> {
    serde_json::from_value(args).map_err(|e| anyhow::anyhow!("arguments: {e}"))
}

/// `p` as results give paths: from the root.
pub fn shown_path(p: &std::path::Path) -> String {
    std::path::absolute(p)
        .unwrap_or_else(|_| p.to_path_buf())
        .to_string_lossy()
        .into_owned()
}
