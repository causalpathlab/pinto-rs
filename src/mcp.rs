//! `pinto mcp`: pinto as a Model Context Protocol server, the viewer's
//! tools ([`crate::view::mcp`]) and `pinto run`'s ([`crate::run::mcp`]) on
//! one stdio server.

use legume_mcp::{Call, Content, Handler, ToolSpec};
use serde_json::Value;

/// What the agent is told when it connects: what the tools are for, and in
/// which order to call them. The README gives the same, for clients that
/// do not pass it on.
pub const INSTRUCTIONS: &str = "\
Use pinto's MCP tools to fit spatial link-community models and to look at \
them on the tissue. A fit: `run_methods` lists the methods (lc is the \
recommended one) and, with a method, its flags; `run_fit` runs one on data \
files with their coordinates, in the background for anything but a small \
data set (follow its job with `job_status` / `job_wait`), and never writes \
over a run. Looking: start with `view_open` on a run prefix (the `out` of a \
fit, or its `.pinto.json`): it lists the run's levels, batches, groups and any \
lupin rounds. `view_map` draws the map as a PNG with its legend; its arguments \
stay set until changed, so pass only what changes. `view_markers` lists a \
group's top features, `view_cell` names the cell under a pixel of the last map \
(or at a world point, or by name), and `view_export` writes the current view as \
PNG/PDF files. World coordinates are those `view_open` reports, with y running \
down the map. lupin rounds are read here as they are on disk; annotating and \
relabelling them is lupin's own server.";

/// The server: the viewer's tools and the run tools.
pub struct Server {
    view: crate::view::mcp::Tools,
    run: crate::run::mcp::Tools,
}

impl Server {
    /// The server, reading the run methods from `cli` (built).
    pub fn new(cli: clap::Command) -> Self {
        Server {
            view: Default::default(),
            run: crate::run::mcp::Tools::new(cli),
        }
    }
}

impl Handler for Server {
    fn tools(&self) -> Vec<ToolSpec> {
        let mut tools = crate::run::mcp::specs();
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
