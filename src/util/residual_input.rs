//! Recognising a residual's output among data files: `pinto residual` writes
//! `residual.json` beside its excess counts, and every reader that must
//! tell excess from raw counts goes through here. `pinto lra` takes a fit on
//! excess counts back to its raw counts, `pinto residual` refuses a second
//! residual, and a fit on excess counts estimates no batch effect
//! ([`no_batch_effect_on`]).

use crate::util::common::basename;
use crate::util::input::{BatchArgs, CollapseArgs};
use crate::util::lupin_rounds::{canonical, read_json, same_file};
use crate::util::metadata::resolve_beside;
use std::path::{Path, PathBuf};

/// The marker's file name: written last, it marks the residual whole.
pub const MARKER: &str = "residual.json";

/// The `residual` block a residual fit's manifest carries, and that `pinto
/// residual` writes into `{out}/residual.json` beside its excess-count
/// backends: one schema for both sides. lra reads only what decides its
/// input. Paths are absolute, or relative to the file holding the block.
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub(crate) struct ResidualBlock {
    /// `excess` for counts a fitted run left unexplained.
    pub counts: String,
    /// The stage-1 run's manifest.
    #[serde(default)]
    pub stage1: Option<String>,
    /// Aligned with the run's `data_files`, index for index.
    #[serde(default)]
    pub raw_data_files: Vec<String>,
}

/// Whether two data files give their cells the same name, as a fit names
/// them by [`basename`] (a second extension such as `.zarr` in `.zarr.zip`
/// dropped): a residual's file and the raw file it came from must.
pub(crate) fn same_data_name(a: &str, b: &str) -> bool {
    let name = |p: &str| basename(p.trim_end_matches('/')).ok();
    name(a).is_some() && name(a) == name(b)
}

/// The excess-count marker beside data file `path`: `residual.json` in its
/// directory saying `counts: excess`, with the marker's block. A raw file
/// the marker names is not excess.
pub(crate) fn excess_marker(path: &Path) -> Option<(PathBuf, serde_json::Value)> {
    let marker = path.parent()?.join(MARKER);
    let value = read_json(&marker)?;
    let block: ResidualBlock = serde_json::from_value(value.clone()).ok()?;
    let raw = block
        .raw_data_files
        .iter()
        .filter_map(|f| resolve_beside(&marker, f))
        .any(|f| same_file(&f, path));
    (block.counts == "excess" && !raw).then_some((marker, value))
}

/// Whether data file `path` holds excess counts, by the marker beside it.
pub(crate) fn is_excess(path: &Path) -> bool {
    excess_marker(path).is_some()
}

/// The marker that names data file `path` as its residual's output: the
/// excess-count marker beside it ([`excess_marker`]) listing a file of its
/// name, among the raw data files (a residual's file keeps its raw file's
/// name) or the `files` it wrote. A marker that lists no file names every
/// file beside it. `None` for any other file, one put in a residual's
/// folder included.
pub(crate) fn naming_marker(path: &Path) -> Option<PathBuf> {
    let (marker, value) = excess_marker(path)?;
    let written = value.get("files").and_then(|f| f.as_array());
    let mut listed = value
        .get("raw_data_files")
        .and_then(|f| f.as_array())
        .into_iter()
        .flatten()
        .filter_map(|f| f.as_str())
        .chain(
            written
                .into_iter()
                .flatten()
                .filter_map(|f| f.get("residual_file")?.as_str()),
        )
        .peekable();
    let own = path.to_string_lossy();
    (listed.peek().is_none() || listed.any(|f| same_data_name(f, &own))).then_some(marker)
}

/// The residual whose output `files` are: its marker, when one
/// `residual.json` names every file ([`naming_marker`]); `None` when it
/// names none of them. A list that mixes a residual's files with other
/// files, or with another residual's, is refused: no single rule on the
/// batch effect would hold for all of it.
pub(crate) fn residual_of_files(files: &[Box<str>]) -> anyhow::Result<Option<PathBuf>> {
    let markers: Vec<Option<PathBuf>> = files
        .iter()
        .map(|f| naming_marker(Path::new(f.as_ref())))
        .collect();
    let Some((at, first)) = markers
        .iter()
        .enumerate()
        .find_map(|(i, m)| Some((i, m.as_ref()?)))
    else {
        return Ok(None);
    };
    for (file, marker) in files.iter().zip(&markers) {
        match marker {
            None => anyhow::bail!(
                "{} holds a residual's excess counts ({}) and {file} does not: fit a \
                 residual's files on their own, or the raw data files on their own",
                files[at],
                first.display()
            ),
            Some(m) if canonical(m) != canonical(first) => anyhow::bail!(
                "{} and {file} are the excess counts of two residuals ({} and {}): fit \
                 one residual's files at a time",
                files[at],
                first.display(),
                m.display()
            ),
            Some(_) => {}
        }
    }
    Ok(Some(first.clone()))
}

/// Whether a fit on `files` estimates no batch effect because they are a
/// residual's excess counts: the residual's marker when they are
/// ([`residual_of_files`]). The residual's per-batch profiles already took
/// out every shift of a whole community in a batch, so what is left in a
/// batch is the local structure of that batch, which a second estimate
/// would take out. Options that only feed the estimate are refused with
/// that reason; `--ignore-batch` asks for what is done anyway.
pub(crate) fn no_batch_effect_on(
    files: &[Box<str>],
    batch: &BatchArgs,
    collapse: &CollapseArgs,
) -> anyhow::Result<Option<PathBuf>> {
    let Some(marker) = residual_of_files(files)? else {
        return Ok(None);
    };
    let given = [
        ("--batch-files", batch.batch_files.is_some()),
        ("--auto-batch", batch.auto_batch),
        ("--cnv-clones", collapse.cnv_clones.is_some()),
    ];
    if let Some((flag, _)) = given.iter().find(|(_, given)| *given) {
        anyhow::bail!(
            "{flag} sets up a batch-effect estimate, and none is made on a residual's \
             excess counts ({}): the residual already took out each batch's shift of a \
             whole community, and what is left in a batch is its own local structure. \
             Drop {flag}",
            marker.display()
        );
    }
    Ok(Some(marker))
}

/// Say, once, that a fit reads `marker`'s excess counts with no batch
/// effect.
pub(crate) fn log_no_batch_effect(marker: &Path) {
    log::info!(
        "The data files are a residual's excess counts ({}): no batch effect is estimated",
        marker.display()
    );
}
