//! What the terminal front ends, `pinto view` and `pinto run`, share: how
//! text and popups look, a file browser, and how the programs they start
//! run in their session.

pub(crate) mod browse;
pub(crate) mod field;
pub(crate) mod style;

use std::path::Path;

/// The tool `pinto run` and `pinto view` sessions belong to: they are
/// named `pinto-run-1`, `pinto-view-2`, and `pinto attach` finds them.
pub(crate) const TOOL: legume_tui::Tool = legume_tui::Tool::new("pinto");

/// The modes of a screen: modifiers reported on enter where the terminal
/// has them, so shift-enter is told apart from enter
/// ([`legume_tui::Session::keyboard_enhanced`]), and the mouse as asked.
/// The session turns them on again after every attach.
pub(crate) fn modes(mouse: legume_tui::Mouse) -> legume_tui::Modes {
    use ratatui::crossterm::event::KeyboardEnhancementFlags;
    legume_tui::Modes::new()
        .mouse(mouse)
        .keyboard_enhancement(legume_tui::Keyboard::IfSupported(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES,
        ))
}

/// The detach and switch keys as the help lists them, the key in lower
/// case then what it does in a word; none when the screen runs plain.
pub(crate) fn session_keys() -> impl Iterator<Item = (String, String)> {
    legume_tui::help_keys()
        .into_iter()
        .map(|k| (k.key.to_lowercase(), k.short))
}

/// How the session started first in this process starts its programs,
/// for code far from it (a worker thread); as a plain program does before
/// there is one.
pub(crate) fn child_env() -> legume_tui::ChildEnv {
    legume_tui::current()
        .map(legume_tui::Session::child_env)
        .unwrap_or_default()
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

/// The key that runs what a screen set up, as in senna: ctrl+r, which
/// every terminal passes on as it is.
pub(crate) const RUN_KEY: &str = "ctrl+r";

/// Whether `k` is [`RUN_KEY`].
#[must_use]
pub(crate) fn is_run_key(k: &ratatui::crossterm::event::KeyEvent) -> bool {
    use ratatui::crossterm::event::{KeyCode, KeyModifiers};
    k.code == KeyCode::Char('r') && k.modifiers.contains(KeyModifiers::CONTROL)
}

/// Run `f` on the whole terminal of `session`, its `modes` on, and give
/// the terminal back however it ends. Logging is off meanwhile: log lines
/// would land over the screen. An error it ends with is held where it is
/// seen ([`legume_tui::Session::hold_error`]): drawn until a terminal
/// attaches, or left for whoever looks next.
pub fn with_terminal<T>(
    session: &legume_tui::Session,
    modes: legume_tui::Modes,
    f: impl FnOnce(&mut ratatui::DefaultTerminal) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let _quiet = legume_tui::logs::quiet();
    let mut terminal = session.terminal_with(modes)?;
    let out = f(&mut terminal);
    if let Err(e) = &out {
        let _ = session.hold_error(&mut terminal, e);
    }
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
