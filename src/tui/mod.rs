//! What the terminal front ends, `pinto view` and `pinto run`, share: how
//! text and popups look, a file browser, and the pinto and lupin processes
//! they start and follow.

pub(crate) mod browse;
pub(crate) mod child;
pub(crate) mod field;
pub(crate) mod style;

use std::path::Path;

/// The tool `pinto run` and `pinto view` sessions belong to: they are
/// named `pinto-run-1`, `pinto-view-2`, and `pinto attach` finds them.
pub(crate) const TOOL: legume_tui::Tool = legume_tui::Tool::new("pinto");

/// The modes of a screen: modifiers reported on enter, so shift-enter is
/// told apart from enter where the terminal can, and the mouse as asked.
/// The session turns them on again after every attach.
pub(crate) fn modes(mouse: legume_tui::Mouse) -> legume_tui::Modes {
    use ratatui::crossterm::event::KeyboardEnhancementFlags;
    legume_tui::Modes {
        mouse,
        keyboard_enhancement: Some(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES),
        ..Default::default()
    }
}

/// Whether the terminal attached now tells shift-enter from enter. It is
/// asked, so call it with the screen up and before mouse reports are on:
/// once something is on screen, and again in the repaint after an attach.
pub(crate) fn shift_enter_works(session: &legume_tui::Session) -> bool {
    session.attached()
        && ratatui::crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false)
}

/// The detach and switch keys, each with what it does in a word or two,
/// in lower case as the help shows keys; none when the screen runs plain.
pub(crate) fn session_keys() -> Vec<(String, String)> {
    legume_tui::help_keys()
        .into_iter()
        .map(|(key, help)| {
            let what = help.split(':').next().unwrap_or(&help).to_string();
            (key.to_lowercase(), what)
        })
        .collect()
}

/// A session run plain by `--no-detach`, for tests.
#[cfg(test)]
pub(crate) fn plain_session() -> legume_tui::Session {
    let opts = legume_tui::Options {
        no_detach: true,
        ..Default::default()
    };
    TOOL.session("test", &opts).unwrap()
}

/// Whether `k` is shift-enter.
#[must_use]
pub fn shift_enter(k: &ratatui::crossterm::event::KeyEvent) -> bool {
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};
    k.code == KeyCode::Enter && k.modifiers.contains(KeyModifiers::SHIFT)
}

/// Run `f` on the whole terminal of `session`, its `modes` on, and give
/// the terminal back however it ends. Logging is off meanwhile: log lines
/// would land over the screen.
pub fn with_terminal<T>(
    session: &legume_tui::Session,
    modes: legume_tui::Modes,
    f: impl FnOnce(&mut ratatui::DefaultTerminal) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let _quiet = legume_tui::logs::quiet();
    let mut terminal = session.terminal_with(modes)?;
    let out = f(&mut terminal);
    session.restore();
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

/// `word` as one shell word: as it is when nothing in it is special, else
/// in single quotes.
#[must_use]
pub fn quote(word: &str) -> String {
    let plain = !word.is_empty()
        && word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if plain {
        word.to_string()
    } else {
        format!("'{}'", word.replace('\'', r"'\''"))
    }
}
