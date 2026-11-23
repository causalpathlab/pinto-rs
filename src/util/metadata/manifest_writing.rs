//! Writing a manifest safely: hidden lock and temporary files, an atomic
//! install that replaces the old manifest, and the timestamp it records.

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// `.{name}` for the file `path` names: hidden, so a folder listing (the
/// viewer's run browser among them) passes by the lock and temporary files
/// named from it.
fn hidden_name(path: &Path) -> String {
    format!(
        ".{}",
        path.file_name().unwrap_or_default().to_string_lossy()
    )
}

/// Replace the manifest at `path` with `json`, whole: write a temporary
/// file beside it, flush it to disk and rename it over `path`, so a reader
/// sees the old manifest or the new one. A manifest [`why_in_place`] says to
/// keep is written in place instead, still under the lock; so is one in a
/// folder that takes no new file. Call it under the lock ([`locked`]).
pub(super) fn replace(path: &Path, json: &str) -> anyhow::Result<()> {
    use std::io::{ErrorKind, Write};
    let fail = |e: std::io::Error| anyhow::anyhow!("writing {}: {e}", path.display());
    let write_in_place = || std::fs::write(path, json).map_err(fail);
    let old = match std::fs::metadata(path) {
        Ok(m) => Some(m),
        Err(e) if e.kind() == ErrorKind::NotFound => None,
        Err(e) => return Err(fail(e)),
    };
    if old.as_ref().and_then(why_in_place).is_some() {
        return write_in_place();
    }
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let mut tmp = match tempfile::Builder::new()
        .prefix(&format!("{}.", hidden_name(path)))
        .suffix(TMP_SUFFIX)
        .tempfile_in(dir)
    {
        Ok(t) => t,
        Err(e) if e.kind() == ErrorKind::PermissionDenied => return write_in_place(),
        Err(e) => return Err(fail(e)),
    };
    // A manifest rewritten keeps its permissions, as one written in place
    // did.
    if let Some(m) = &old {
        tmp.as_file()
            .set_permissions(m.permissions())
            .map_err(fail)?;
    }
    tmp.write_all(json.as_bytes()).map_err(fail)?;
    tmp.as_file().sync_all().map_err(fail)?;
    tmp.persist(path).map_err(|e| fail(e.error))?;
    Ok(())
}

/// Put the staged manifest `staged` at `path`, whole and under its lock,
/// as [`PintoMetadata::write`] writes one.
pub(crate) fn install(staged: &Path, path: &Path) -> anyhow::Result<()> {
    let json = std::fs::read_to_string(staged)?;
    locked(path, |path| replace(path, &json))
}

/// Move the manifest at `path` to `to`, under its lock, so no manifest is
/// at `path` while the run it describes is replaced. One that could only
/// be written in place ([`kept_in_place`]) is not moved, and that is an
/// error.
pub(crate) fn retire(path: &Path, to: &Path) -> anyhow::Result<()> {
    locked(path, |real| {
        if let Some(why) = kept_in_place(path) {
            anyhow::bail!("it is {why}, and is written in place");
        }
        Ok(std::fs::rename(real, to)?)
    })
}

/// Why the manifest at `path` is written in place rather than replaced by
/// a rename, when it is ([`why_in_place`], and a link, which a rename would
/// replace): what `--atomic-outputs` cannot take away and put back whole.
pub(crate) fn kept_in_place(path: &Path) -> Option<&'static str> {
    if path.is_symlink() {
        return Some("a link");
    }
    why_in_place(&std::fs::metadata(path).ok()?)
}

/// Whether a file named `name` is the manifest writer's own, a lock or a
/// temporary file ([`locked`], [`replace`]), and no output.
pub(crate) fn is_scratch(name: &str) -> bool {
    name.starts_with('.') && (name.ends_with(LOCK_SUFFIX) || name.ends_with(TMP_SUFFIX))
}

/// What the lock and the temporary file of a manifest are named by, after
/// its [`hidden_name`].
const LOCK_SUFFIX: &str = ".lock";
const TMP_SUFFIX: &str = ".tmp";

/// The file `path` names: through links (to a file not made yet among
/// them), the name at the end of the chain, so a rename over it keeps
/// every link.
fn through_link(path: &Path) -> PathBuf {
    if let Ok(real) = std::fs::canonicalize(path) {
        return real;
    }
    let mut at = path.to_path_buf();
    // As many hops as the system follows before it gives up on a loop.
    for _ in 0..40 {
        match std::fs::read_link(&at) {
            Ok(target) => at = at.parent().unwrap_or(Path::new("")).join(target),
            Err(_) => break,
        }
    }
    at
}

/// Why the manifest `m` describes is written in place rather than replaced
/// by a rename, when it is: one another user owns (a rename would make it
/// this user's and shut its owner out), one with other hard links (a
/// rename would leave them on the old manifest), or a read-only one
/// (refused, as a write in place was).
fn why_in_place(m: &std::fs::Metadata) -> Option<&'static str> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // SAFETY: geteuid has no preconditions and cannot fail.
        if m.uid() != unsafe { libc::geteuid() } {
            return Some("another user's");
        }
        if m.nlink() > 1 {
            return Some("hard-linked");
        }
    }
    m.permissions().readonly().then_some("read-only")
}

/// Run `then` on the manifest at `path` ([`through_link`]) holding the
/// manifest's lock: an exclusive lock on `.{name}.lock` beside it, released
/// when `then` returns. The lock file is kept, since removing it would let
/// a run that opened it before the removal and one that made it anew both
/// hold "the" lock; one made by another user is locked read-only. A lock
/// that cannot be had (a file system without locks) is warned about once,
/// and `then` runs without it.
pub(super) fn locked<T>(
    path: &Path,
    then: impl FnOnce(&Path) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    static WARNED: AtomicBool = AtomicBool::new(false);
    let path = through_link(path);
    let lock_path = path.with_file_name(format!("{}{LOCK_SUFFIX}", hidden_name(&path)));
    let open = |write: bool| {
        std::fs::OpenOptions::new()
            .read(true)
            .write(write)
            .create(write)
            .truncate(false)
            .open(&lock_path)
    };
    // No lock file can be made where the folder is missing or takes no new
    // file: the read or write that follows reports that itself.
    let _lock = open(true)
        .or_else(|_| open(false))
        .ok()
        .and_then(|f| match f.lock() {
            Ok(()) => Some(f),
            Err(e) => {
                if !WARNED.swap(true, Ordering::Relaxed) {
                    log::warn!(
                        "{}: no lock ({e}); manifests are written without one, so runs \
                     updating one manifest at once may lose updates",
                        lock_path.display()
                    );
                }
                None
            }
        });
    then(&path)
}

pub(crate) fn now_secs() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}
