//! lupin rounds beside a run: which `*.lupin.json` rounds were made from a
//! run manifest, and the small path and JSON helpers they are read with
//! (paths inside a round resolve by [`resolve_beside`]).
//! Shared by `pinto view` and `pinto lra` (which reads cell types from the
//! newest round), so it lives outside the feature-gated viewer.

use crate::util::metadata::resolve_beside;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// Every chain's newest round made from the run at `manifest`, newest
/// first by modification time.
pub fn latest_rounds(manifest: &Path) -> Vec<PathBuf> {
    let Some(dir) = manifest.parent().map(dir_or_cwd) else {
        return Vec::new();
    };
    let run = canonical(manifest);
    // Round → the file it was made from.
    let mut source: BTreeMap<PathBuf, PathBuf> = BTreeMap::new();
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if !path.to_string_lossy().ends_with(".lupin.json") {
            continue;
        }
        let from = read_json(&path)
            .and_then(|m| m.pointer("/annotate/source")?.as_str().map(String::from));
        if let Some(from) = from.and_then(|f| resolve_beside(&path, &f)) {
            source.insert(canonical(&path), canonical(&from));
        }
    }
    let of_run = |start: &PathBuf| {
        let mut r = start;
        for _ in 0..source.len() + 1 {
            match source.get(r) {
                Some(s) if *s == run => return true,
                Some(s) => r = s,
                None => return false,
            }
        }
        false
    };
    let parents: HashSet<&PathBuf> = source.values().collect();
    let mut tips: Vec<PathBuf> = source
        .keys()
        .filter(|r| !parents.contains(r) && of_run(r))
        .cloned()
        .collect();
    tips.sort_by_key(|p| std::cmp::Reverse(modified(p)));
    tips
}

fn modified(p: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// `p`, or `.` for the empty path (the parent of a bare file name).
pub fn dir_or_cwd(p: &Path) -> PathBuf {
    if p.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        p.to_path_buf()
    }
}

/// Whether `a` and `b` are one file, through links.
pub fn same_file(a: &Path, b: &Path) -> bool {
    canonical(a) == canonical(b)
}

/// `p` resolved through links, or as given when it does not resolve.
pub fn canonical(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

pub fn read_json(path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}
