//! What the terminal front ends, `pinto view` and `pinto run`, share: how
//! text and popups look, a file browser, and the pinto and lupin processes
//! they start and follow.

pub(crate) mod browse;
pub(crate) mod child;
pub(crate) mod style;

use std::path::Path;

/// Run `f` on the whole terminal and give the terminal back however it
/// ends. Logging is off meanwhile: log lines would land over the screen.
pub fn with_terminal<T>(
    f: impl FnOnce(&mut ratatui::DefaultTerminal) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let logging = log::max_level();
    log::set_max_level(log::LevelFilter::Off);
    let mut terminal = ratatui::init();
    let out = f(&mut terminal);
    ratatui::restore();
    log::set_max_level(logging);
    out
}

/// The last component of `path`, or an empty string.
#[must_use]
pub fn name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// `p` relative to the working directory when it is under it.
#[must_use]
pub fn relative(p: &Path) -> std::path::PathBuf {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| p.strip_prefix(cwd).ok().map(Path::to_path_buf))
        .filter(|r| !r.as_os_str().is_empty())
        .unwrap_or_else(|| p.to_path_buf())
}

/// A path as shown: relative to the working directory when it is under it.
#[must_use]
pub fn shown(p: &Path) -> String {
    relative(p).to_string_lossy().into_owned()
}
