//! `view_lra`'s `start`: `pinto lra --only` on chosen pairs, with the
//! options a table of the run was made with, run as `run_lra` runs it, so
//! its checks, its refusal to write over a table, its reading of a finished
//! one and its job records are `run_lra`'s.

use crate::util::metadata::LraOptions;
use legume_mcp::{Call, Content};
use serde_json::{json, Value};
use std::path::Path;

/// `run_lra`'s arguments for `only` on the run at `manifest` (prefix
/// `prefix`), with `options` as the table shown was made: the flags
/// [`LraOptions::argv`] passes, beside the run, in the background.
fn subset_args(manifest: &Path, prefix: &str, options: &LraOptions, only: &[String]) -> Value {
    let d = LraOptions::default();
    let mut a = json!({
        "run": manifest.to_string_lossy(),
        "out": prefix,
        "only": only,
        "background": true,
    });
    if let Some(p) = &options.lr_pairs {
        a["lr_pairs"] = p.as_str().into();
    }
    if !options.targets.is_empty() {
        a["targets"] = options.targets.clone().into();
    }
    if let Some(c) = &options.cell_types {
        a["cell_types"] = c.as_str().into();
    }
    if let Some(r) = &options.region {
        a["region"] = r.as_str().into();
    }
    if options.region_jackknife != d.region_jackknife {
        a["region_jackknife"] = options.region_jackknife.as_str().into();
    }
    if options.abf_prior_scale != d.abf_prior_scale {
        a["abf_prior_scale"] = options.abf_prior_scale.into();
    }
    if options.species != d.species {
        a["species"] = options.species.as_str().into();
    }
    if options.seed != d.seed {
        a["seed"] = options.seed.into();
    }
    if options.marginal_baselines.is_some() {
        a["marginal_baselines"] = true.into();
        if let Some(n) = options.marginal_permutations {
            a["marginal_permutations"] = n.into();
        }
    }
    if let Some(k) = options.null_permutations {
        a["null_permutations"] = k.into();
    }
    a
}

/// Start `pinto lra --only only` on the run, through `run_lra`.
pub(crate) fn start_subset(
    manifest: &Path,
    prefix: &str,
    options: &LraOptions,
    only: &[String],
    cx: &Call,
) -> anyhow::Result<Vec<Content>> {
    let args = crate::mcp::parse(subset_args(manifest, prefix, options, only))?;
    super::lra::lra(&crate::built_cli(), args, cx)
}

#[cfg(test)]
#[path = "tests/lra_subset.rs"]
mod tests;
