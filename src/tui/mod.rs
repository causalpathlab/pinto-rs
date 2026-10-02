//! What the terminal front ends, `pinto view` and `pinto run`, share: how
//! text and popups look, a file browser, and the pinto and lupin processes
//! they start and follow.

pub(crate) mod browse;
pub(crate) mod child;
pub(crate) mod style;

use std::path::Path;

/// The terminal reporting modifiers on enter, so shift-enter is told
/// apart from enter, until dropped.
pub struct EnhancedKeys;

impl EnhancedKeys {
    /// Ask for it; `None` where the terminal cannot. The terminal is asked
    /// and answers first, so call it once something is on screen.
    pub fn push() -> Option<Self> {
        use ratatui::crossterm::event::{KeyboardEnhancementFlags, PushKeyboardEnhancementFlags};
        use ratatui::crossterm::{execute, terminal::supports_keyboard_enhancement};
        let flags =
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES);
        (supports_keyboard_enhancement().unwrap_or(false)
            && execute!(std::io::stdout(), flags).is_ok())
        .then_some(EnhancedKeys)
    }
}

impl Drop for EnhancedKeys {
    fn drop(&mut self) {
        use ratatui::crossterm::{event::PopKeyboardEnhancementFlags, execute};
        let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    }
}

/// Whether `k` is shift-enter.
#[must_use]
pub fn shift_enter(k: &ratatui::crossterm::event::KeyEvent) -> bool {
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};
    k.code == KeyCode::Enter && k.modifiers.contains(KeyModifiers::SHIFT)
}

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

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
