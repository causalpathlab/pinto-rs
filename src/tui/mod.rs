//! What the terminal front ends, `pinto view` and `pinto run`, share: how
//! text and popups look, a file browser, and how the programs they start
//! run in their session.

pub(crate) mod browse;
pub(crate) mod style;

use std::path::Path;
use std::sync::OnceLock;

/// The `--max-threads` this pinto was started with, for the pinto
/// commands it starts.
static MAX_THREADS: OnceLock<usize> = OnceLock::new();

/// Keep this pinto's `--max-threads`, once, for the commands it starts.
pub(crate) fn keep_max_threads(n: usize) {
    let _ = MAX_THREADS.set(n);
}

/// This pinto's `--max-threads`, when it was given other than the default.
pub(crate) fn parent_max_threads() -> Option<usize> {
    MAX_THREADS
        .get()
        .copied()
        .filter(|&n| n != crate::util::common::DEFAULT_MAX_THREADS)
}

/// `argv`, a pinto command line without the program, given this pinto's
/// `--max-threads` (see [`with_max_threads`]).
pub(crate) fn inherit_max_threads(argv: &mut Vec<String>) {
    if let Some(parent) = parent_max_threads() {
        with_max_threads(argv, parent);
    }
}

/// `argv` given `--max-threads={parent}` ahead of its command, so a pinto
/// started from this one uses no more threads (and lc no more Gibbs
/// blocks) than this one was told: unless `argv` gives its own count, or
/// `parent` is the default the child takes anyway.
fn with_max_threads(argv: &mut Vec<String>, parent: usize) {
    const FLAG: &str = "--max-threads";
    let own = argv
        .iter()
        .take_while(|a| *a != "--")
        .any(|a| a == FLAG || a.starts_with(&format!("{FLAG}=")));
    if !own && parent != crate::util::common::DEFAULT_MAX_THREADS {
        argv.insert(0, flag_value(FLAG, parent));
    }
}

/// [`legume_mcp::own_command`] for `argv`, given this pinto's
/// `--max-threads` as [`inherit_max_threads`] gives it.
pub(crate) fn own_command(argv: &[String]) -> anyhow::Result<std::process::Command> {
    let mut argv = argv.to_vec();
    inherit_max_threads(&mut argv);
    legume_mcp::own_command(&argv)
}

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

/// `pinto` and its arguments `argv` as one shell line.
#[must_use]
pub fn pinto_line(argv: &[String]) -> String {
    format!("pinto {}", legume_tui::form::script::line(argv))
}

/// A flag and its value as one word of a command line, `--flag=value`: a
/// value that starts with `-`, such as a negative number, is then read as
/// the flag's value, not as a flag of its own, and reaches the command's
/// own check of it.
#[must_use]
pub fn flag_value(flag: &str, value: impl std::fmt::Display) -> String {
    format!("{flag}={value}")
}
