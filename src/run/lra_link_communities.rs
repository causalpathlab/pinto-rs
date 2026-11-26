//! `run_lra_link_communities` for `pinto mcp`: `pinto lra link-communities`
//! read in this process, the run's link communities with their links and
//! receivers, and whether `run_lra` with `link_community` fits each.

use super::mcp::{as_run, flag_arg, subcommand};
use crate::lr_activity::ligand_effect::link_community_scope::{
    recorded_link_community_sizes, MIN_RECEIVERS,
};
use legume_mcp::{Content, ToolSpec};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct LinkCommunitiesToolArgs {
    /// The run, its `.pinto.json` or prefix (`--from`).
    pub run: String,
    /// `--min-receivers`: the fewest receivers to be fitted.
    #[serde(default)]
    pub min_receivers: Option<usize>,
}

pub(super) fn spec(cli: &clap::Command) -> ToolSpec {
    let sub = subcommand(cli, "lra link-communities");
    ToolSpec::new(
        "run_lra_link_communities",
        "The run's link communities (pinto lra link-communities), for run_lra's \
         link_community: each community's links (spatial pairs) and receivers (the \
         distinct cells of its links, an upper bound on what link_community fits), and \
         fitted, whether it has min_receivers receivers or more so run_lra with \
         link_community fits it. Reads the link-community table the run's manifest \
         records (lc, cage, dsvd); writes nothing.",
        json!({
            "type": "object",
            "properties": {
                "run": {"type": "string", "description": "The run: its .pinto.json or prefix (--from)"},
                "min_receivers": flag_arg(sub, "min_receivers", json!({"type": "integer", "minimum": 1})),
            },
            "required": ["run"],
            "additionalProperties": false,
        }),
    )
    .read_only()
}

pub(super) fn link_communities(a: LinkCommunitiesToolArgs) -> anyhow::Result<Vec<Content>> {
    let run = as_run(&a.run);
    let min_receivers = a.min_receivers.unwrap_or(MIN_RECEIVERS);
    let sizes = recorded_link_community_sizes(&run)?;
    let mut lines = vec!["link_community\tlinks\treceivers\tfitted".to_string()];
    let rows: Vec<Value> = sizes
        .iter()
        .map(|s| {
            let fitted = s.receivers >= min_receivers;
            lines.push(format!(
                "{}\t{}\t{}\t{}",
                s.link_community,
                s.links,
                s.receivers,
                if fitted { "yes" } else { "no" }
            ));
            json!({
                "link_community": s.link_community,
                "links": s.links,
                "receivers": s.receivers,
                "fitted": fitted,
            })
        })
        .collect();
    let reply = json!({
        "run": run,
        "min_receivers": min_receivers,
        "link_communities": rows,
    });
    Ok(vec![Content::Text(lines.join("\n")), Content::Json(reply)])
}

#[cfg(test)]
#[path = "tests/lra_link_communities.rs"]
mod tests;
