//! What the terminal front ends, `pinto view` and `pinto run`, share: how
//! text and popups look, a file browser, and the pinto and lupin processes
//! they start and follow.

pub(crate) mod browse;
pub(crate) mod child;
pub(crate) mod style;

use std::path::Path;

/// The last component of `path`, or an empty string.
#[must_use]
pub fn name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// `p` relative to `base` when it is under it.
#[must_use]
pub fn relative_to(p: &Path, base: &Path) -> std::path::PathBuf {
    p.strip_prefix(base)
        .ok()
        .filter(|r| !r.as_os_str().is_empty())
        .unwrap_or(p)
        .to_path_buf()
}

/// `p` relative to the working directory when it is under it.
#[must_use]
pub fn relative(p: &Path) -> std::path::PathBuf {
    match std::env::current_dir() {
        Ok(cwd) => relative_to(p, &cwd),
        Err(_) => p.to_path_buf(),
    }
}

/// A path as shown: relative to the working directory when it is under it.
#[must_use]
pub fn shown(p: &Path) -> String {
    relative(p).to_string_lossy().into_owned()
}
