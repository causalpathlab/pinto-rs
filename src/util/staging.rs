//! `--atomic-outputs`: a run's outputs are complete or absent.
//!
//! Under the flag a run writes every output in a staging folder beside its
//! final place, hidden and its own (`{dir}/.{stem}.partial-{pid}-{random}`
//! for the prefix `{dir}/{stem}`), and puts the outputs in place only when
//! it succeeds. Each output moves by a rename on the same file system. The
//! files that mark a run complete ([`is_marker`]) come last: an earlier
//! run's are taken away before the first output moves and this run's are
//! put there after the last, a pinto manifest under its lock as any
//! manifest write is, so a manifest on disk means the run completed.
//! Where no rename can do that the run is refused before its first move:
//! an earlier manifest that could only be written in place, a link, or a
//! folder where a file lands (and the reverse).
//!
//! A run that fails, or panics, removes the staging folder, and nothing
//! appears under the final names. What it says on stdout at its end is
//! held until its outputs are in place ([`say`]). A run killed outright
//! leaves its staging folder behind, hidden. So does a run whose final
//! move fails part way: the markers and stores it had moved aside are put
//! back (a file an output was already renamed over is not), and the error
//! names the folder that holds the outputs not moved.
//!
//! Within the run every path is the staged one, so a command reading back
//! its own outputs finds them. Paths recorded in the run's JSON files are
//! made the final ones as the outputs move. A change to a manifest that is
//! not staged (`lra` records into its `--from` run's, or into the one at
//! its `--out`) waits for the move: it is made then, under the manifest's
//! lock, on the manifest as it is read then.

use crate::util::metadata::{self, PintoMetadata};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

/// How the staged outputs land.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Shape {
    /// Files under the prefix `{dir}/{stem}`, staged under
    /// `{dir}/{marker}/{stem}` and each moved into `{dir}`.
    Prefix,
    /// The folder `{dir}/{name}`, staged as `{dir}/{marker}`: moved there
    /// whole when it is missing or empty, else file by file into it.
    Folder { name: String },
}

/// One run's staged outputs.
pub struct Staging {
    shape: Shape,
    /// The staging folder, made by this run.
    dir: PathBuf,
    /// The staging folder's name, which no other path has: a path with it
    /// as a component is a staged one.
    marker: String,
    /// Where the outputs go: `{dir}` for a prefix, the folder for a folder.
    into: PathBuf,
    /// The out as the command was given it, and as it writes it.
    final_out: String,
    staged_out: String,
    /// The changes to manifests that are not staged, in the order made.
    pending: Vec<Pending>,
    /// The folders above the staging folder that this run made, the
    /// deepest first: removed again, while empty, by a run that lands
    /// nothing in them.
    made: Vec<PathBuf>,
    /// Whether this process stands in the output folder or below it: such
    /// a folder is filled file by file, never replaced by another.
    stood_in: bool,
}

/// A change to a manifest.
type Change = Box<dyn FnOnce(&mut PintoMetadata) + Send>;

/// The changes this run makes to one manifest that is not staged.
struct Pending {
    path: PathBuf,
    /// The manifest to start from when none is there: this run's own.
    fresh: Option<PintoMetadata>,
    changes: Vec<Change>,
}

/// The folder in a staging folder for what a commit replaces.
const REPLACED: &str = ".replaced";

/// `out` split after its last `/`: the folder part as written (with the
/// slash; empty for a bare name) and the name.
fn split(out: &str) -> (&str, &str) {
    match out.rfind('/') {
        Some(i) => (&out[..=i], &out[i + 1..]),
        None => ("", out),
    }
}

/// The folder a folder part names: itself, or `.` for none.
fn folder(part: &str) -> PathBuf {
    crate::util::lupin_rounds::dir_or_cwd(Path::new(part))
}

/// Remove the folder `dir` and all in it; a missing one is fine.
fn remove_all(dir: &Path) -> std::io::Result<()> {
    match std::fs::remove_dir_all(dir) {
        Err(e) if e.kind() != ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Whether a file named `name` marks a run or a call complete, by the
/// modules that write it: such a file is moved after every other output.
fn is_marker(name: &str) -> bool {
    name.ends_with(metadata::MANIFEST_SUFFIX)
        || name.ends_with(crate::lr_activity::ligand_effect::output::SIDECAR_SUFFIX)
        || name == crate::simulate::run::MANIFEST
        || name.ends_with(crate::compare::REPORT_SUFFIX)
}

/// A folder that is one output, replaced whole rather than merged.
fn is_store(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "zarr")
}

/// Whether something is at `path`, a link that leads nowhere included.
fn is_there(path: &Path) -> bool {
    path.symlink_metadata().is_ok()
}

/// A new, empty staging folder in `parent` for the output `name`, its
/// name, and the folders above it that had to be made, the deepest first.
/// One of that name already there is another run's and is left.
fn make_staging(parent: &Path, name: &str) -> anyhow::Result<(PathBuf, String, Vec<PathBuf>)> {
    let made: Vec<PathBuf> = parent
        .ancestors()
        .take_while(|p| !p.as_os_str().is_empty() && !is_there(p))
        .map(Path::to_path_buf)
        .collect();
    let make = || -> Result<(PathBuf, String), String> {
        for _ in 0..16 {
            // made anew each time round: a run that ended meanwhile may
            // have removed a folder both had made
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            let marker = format!(
                ".{name}.partial-{}-{:08x}",
                std::process::id(),
                rand::random::<u32>()
            );
            let dir = parent.join(&marker);
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok((dir, marker)),
                Err(e) if matches!(e.kind(), ErrorKind::AlreadyExists | ErrorKind::NotFound) => {}
                Err(e) => return Err(e.to_string()),
            }
        }
        Err("every name tried is taken".into())
    };
    match make() {
        Ok((dir, marker)) => Ok((dir, marker, made)),
        Err(e) => {
            remove_made(&made);
            anyhow::bail!(
                "--atomic-outputs: no staging folder can be made in {}: {e}",
                parent.display()
            )
        }
    }
}

/// Remove the folders of `made`, the deepest first, up to the first that
/// is not empty: those a run made for outputs it then did not land.
fn remove_made(made: &[PathBuf]) {
    for dir in made {
        if std::fs::remove_dir(dir).is_err() {
            break;
        }
    }
}

/// Whether this process stands in the folder `into` or below it.
fn stands_in(into: &Path) -> bool {
    let real = |p: std::io::Result<PathBuf>| p.and_then(std::fs::canonicalize);
    match (real(std::env::current_dir()), real(Ok(into.to_path_buf()))) {
        (Ok(cwd), Ok(into)) => cwd.starts_with(into),
        _ => false,
    }
}

/// The folder `out` names: the folder part it is in as written (with its
/// slash; empty for a bare name), its name and its path. A link, and an
/// `out` that ends in `.` or `..`, name the folder they lead to; `dir/`
/// and `dir/.` name `dir`, made or not.
fn folder_named(out: &str) -> anyhow::Result<(String, String, PathBuf)> {
    let mut written = out;
    loop {
        let cut = written.trim_end_matches('/');
        let cut = cut.strip_suffix("/.").unwrap_or(cut);
        if cut.len() == written.len() {
            break;
        }
        written = cut;
    }
    if written.is_empty() {
        written = if out.starts_with('/') { "/" } else { "." };
    }
    let (part, name) = split(written);
    let into = folder(part).join(name);
    let dots = matches!(name, "" | "." | "..");
    if !dots && !into.is_symlink() {
        return Ok((part.to_string(), name.to_string(), into));
    }
    let real = std::fs::canonicalize(&into).map_err(|e| {
        let what = if dots { "names" } else { "is a link to" };
        anyhow::anyhow!("--atomic-outputs: {out} {what} no folder: {e}")
    })?;
    let named = real.parent().zip(real.file_name()).map(|(p, n)| {
        (
            format!("{}/", p.to_string_lossy().trim_end_matches('/')),
            n.to_string_lossy().into_owned(),
        )
    });
    match named {
        Some((part, name)) => Ok((part, name, real)),
        None => anyhow::bail!("--atomic-outputs: {out} names no folder to stage beside"),
    }
}

/// What a commit moved out of the way, to put back should it fail.
struct Aside {
    dir: PathBuf,
    /// Where each is held, where it was, and whether it is a marker.
    held: Vec<(PathBuf, PathBuf, bool)>,
    /// Whether an output has reached its final name.
    landed: bool,
}

impl Aside {
    /// Move what is at `dst` out of the way with `mv` (from, to).
    fn take(
        &mut self,
        dst: &Path,
        marker: bool,
        mv: impl FnOnce(&Path, &Path) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let at = self.dir.join(self.held.len().to_string());
        mv(dst, &at)?;
        self.held.push((at, dst.to_path_buf(), marker));
        Ok(())
    }

    /// Where the marker that was at `dst` is held, when one was taken.
    fn marker_of(&self, dst: &Path) -> Option<&Path> {
        self.held
            .iter()
            .find(|(_, d, marker)| *marker && d == dst)
            .map(|(at, _, _)| at.as_path())
    }

    /// After a failure: what was replaced and has nothing in its place is
    /// put back. An earlier run's marker goes back only when no output
    /// landed, so it never sits beside another run's outputs.
    fn restore(&self) {
        for (at, dst, marker) in self.held.iter().rev() {
            if (*marker && self.landed) || is_there(dst) || !is_there(at) {
                continue;
            }
            if let Err(e) = std::fs::rename(at, dst) {
                log::warn!("could not put {} back: {e}", dst.display());
            }
        }
    }
}

impl Staging {
    /// Staging for the output prefix `out`: the staged prefix is
    /// [`Self::out`], in a staging folder made now beside `out`'s files.
    pub fn prefix(out: &str) -> anyhow::Result<Self> {
        let (part, stem) = split(out);
        let (_, marker, made) = make_staging(&folder(part), stem)?;
        Ok(Self {
            made,
            ..Self::prefix_named(out, marker)
        })
    }

    /// Staging for the output prefix `out` in the staging folder named
    /// `marker` beside its files, which is not made here.
    pub(crate) fn prefix_named(out: &str, marker: String) -> Self {
        let (part, stem) = split(out);
        let into = folder(part);
        Self {
            shape: Shape::Prefix,
            staged_out: format!("{part}{marker}/{stem}"),
            final_out: out.to_string(),
            dir: into.join(&marker),
            marker,
            into,
            pending: Vec::new(),
            made: Vec::new(),
            stood_in: false,
        }
    }

    /// Staging for the output folder `out`: the staged folder, made now
    /// beside it, is [`Self::out`]. A link to a folder, and an `out` that
    /// ends in `.` or `..`, are staged beside the folder they lead to.
    /// Refused, before the run does any work, where no rename could put
    /// the outputs: at a file, a link that leads nowhere, the root, or a
    /// folder on another file system than the one it is in.
    pub fn folder(out: &str) -> anyhow::Result<Self> {
        let (part, name, into) = folder_named(out)?;
        let parent = folder(&part);
        if is_there(&into) {
            anyhow::ensure!(into.is_dir(), "--atomic-outputs: {out} is not a folder");
            anyhow::ensure!(
                same_file_system(&into, &parent),
                "--atomic-outputs: {out} is on another file system than the folder it is in, \
                 so nothing can be staged beside it; give a folder inside it"
            );
        }
        let stood_in = stands_in(&into);
        let (dir, marker, made) = make_staging(&parent, &name)?;
        Ok(Self {
            shape: Shape::Folder { name },
            staged_out: format!("{part}{marker}"),
            final_out: out.to_string(),
            into,
            dir,
            marker,
            pending: Vec::new(),
            made,
            stood_in,
        })
    }

    /// The out the command writes.
    pub fn out(&self) -> &str {
        &self.staged_out
    }

    /// The final path `path` is staged for; `None` when it is not staged.
    pub fn final_path(&self, path: &str) -> Option<String> {
        let parts: Vec<&str> = path.split('/').collect();
        let at = parts.iter().position(|p| *p == self.marker)?;
        let mut out: Vec<&str> = parts[..at].to_vec();
        if let Shape::Folder { name } = &self.shape {
            out.push(name);
        }
        out.extend(&parts[at + 1..]);
        let joined = out.join("/");
        // the staging folder itself, of a prefix at the root or a bare one
        Some(match (joined.is_empty(), path.starts_with('/')) {
            (true, true) => "/".into(),
            (true, false) => ".".into(),
            _ => joined,
        })
    }

    /// Whether `path` is a staged one.
    fn stages(&self, path: &Path) -> bool {
        self.final_path(&path.to_string_lossy()).is_some()
    }

    /// `text`, a message, with each staged path in it made the final one.
    pub(crate) fn final_text(&self, text: &str) -> String {
        match &self.shape {
            Shape::Prefix => text.replace(&format!("{}/", self.marker), ""),
            Shape::Folder { name } => text.replace(&self.marker, name),
        }
    }

    /// The changes to the manifest at `path`, which is not staged.
    fn pending(&mut self, path: &Path) -> &mut Pending {
        let at = match self.pending.iter().position(|p| p.path == path) {
            Some(i) => i,
            None => {
                self.pending.push(Pending {
                    path: path.to_path_buf(),
                    fresh: None,
                    changes: Vec::new(),
                });
                self.pending.len() - 1
            }
        };
        &mut self.pending[at]
    }

    /// Change the manifest at `path`, not a staged one, when this run's
    /// outputs are in place: `set` sees it as read then, under its lock,
    /// with this run's earlier changes made.
    pub fn update_elsewhere(
        &mut self,
        path: &Path,
        set: impl FnOnce(&mut PintoMetadata) + Send + 'static,
    ) {
        self.pending(path).changes.push(Box::new(set));
    }

    /// Write `fresh` at `path`, not a staged one, when this run's outputs
    /// are in place and no manifest is there.
    pub fn write_elsewhere_if_missing(&mut self, path: &Path, fresh: PintoMetadata) {
        self.pending(path).fresh.get_or_insert(fresh);
    }

    /// `text`, a JSON document, with every string in it that names a
    /// staged path made the final path, and all else as it was; `None`
    /// when it names none. `marker` is the staging folder's name as JSON
    /// writes it: a string without it is copied as it is.
    fn finalize_text(&self, text: &str, marker: &str) -> Option<String> {
        let (mut out, mut changed) = (String::with_capacity(text.len()), false);
        let mut rest = text;
        while let Some(open) = rest.find('"') {
            out.push_str(&rest[..open]);
            let lit = &rest[open..];
            // the closing quote: the first one no backslash escapes
            let mut end = 1;
            let bytes = lit.as_bytes();
            while end < bytes.len() && bytes[end] != b'"' {
                end += if bytes[end] == b'\\' { 2 } else { 1 };
            }
            let raw = &lit[..(end + 1).min(lit.len())];
            let named = raw
                .contains(marker)
                .then(|| serde_json::from_str::<String>(raw).ok())
                .flatten()
                .and_then(|s| self.final_path(&s));
            match named {
                Some(f) => {
                    out.push_str(&serde_json::to_string(&f).expect("a string"));
                    changed = true;
                }
                None => out.push_str(raw),
            }
            rest = &lit[raw.len()..];
        }
        out.push_str(rest);
        changed.then_some(out)
    }

    /// The recorded paths in every staged JSON file under `dir` made
    /// final. `marker` is the staging folder's name as JSON writes it: a
    /// file without it names no staged path and is not read further.
    fn finalize_files(&self, dir: &Path, marker: &str) -> anyhow::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                // a store records no path
                if !is_store(&path) {
                    self.finalize_files(&path, marker)?;
                }
            } else if path.extension().is_some_and(|e| e == "json") {
                let text = std::fs::read_to_string(&path)?;
                if !text.contains(marker)
                    || serde_json::from_str::<serde::de::IgnoredAny>(&text).is_err()
                {
                    continue;
                }
                if let Some(text) = self.finalize_text(&text, marker) {
                    std::fs::write(&path, text)?;
                }
            }
        }
        Ok(())
    }

    /// The moves that put the staged entries of `from` into `to`: each
    /// entry renamed whole, but a plain folder already there merged into.
    fn moves(from: &Path, to: &Path, out: &mut Vec<(PathBuf, PathBuf)>) -> anyhow::Result<()> {
        for entry in std::fs::read_dir(from)? {
            let entry = entry?;
            let name = entry.file_name();
            if metadata::is_scratch(&name.to_string_lossy()) {
                continue;
            }
            let (src, dst) = (entry.path(), to.join(&name));
            if src.is_dir() && dst.is_dir() && !is_store(&src) {
                Self::moves(&src, &dst, out)?;
            } else {
                out.push((src, dst));
            }
        }
        Ok(())
    }

    /// Put the outputs in place and remove the staging folder. Should a
    /// step fail, the staging folder is kept with what was not moved, what
    /// the commit had moved aside is put back, and the error names the folder.
    pub fn commit(self) -> anyhow::Result<()> {
        self.commit_observed(|_| {})
    }

    /// [`Self::commit`]; `moved` sees each output as it lands.
    pub fn commit_observed(mut self, mut moved: impl FnMut(&Path)) -> anyhow::Result<()> {
        let pending = std::mem::take(&mut self.pending);
        self.land(pending, &mut moved).map_err(|e| {
            // the staging folder is gone once a folder has landed whole
            e.context(if is_there(&self.dir) {
                format!(
                    "--atomic-outputs: the outputs could not all be put in place; those not \
                     moved are kept in {}",
                    self.dir.display()
                )
            } else {
                format!(
                    "--atomic-outputs: the outputs are in place in {}, but the run is not \
                     recorded",
                    self.into.display()
                )
            })
        })?;
        // what is left: the emptied staging folder, and for a run that
        // wrote nothing the folders made above it
        self.abort();
        Ok(())
    }

    /// The commit's steps. All that can fail without a rename comes first:
    /// the recorded paths made final, the manifests to change read, the
    /// moves listed and each checked against what is in its place
    /// ([`refuse_unfit`]). Then the earlier run's markers are taken away,
    /// every other output moved, the manifests that are not staged
    /// changed, and this run's markers put in place, the outermost last.
    fn land(&self, pending: Vec<Pending>, moved: &mut dyn FnMut(&Path)) -> anyhow::Result<()> {
        let marker = serde_json::to_string(&self.marker)?;
        self.finalize_files(&self.dir, marker.trim_matches('"'))?;
        let mut all = Vec::new();
        Self::moves(&self.dir, &self.into, &mut all)?;
        all.sort();
        refuse_unfit(&all)?;

        let whole = matches!(self.shape, Shape::Folder { .. })
            && !self.stood_in
            && std::fs::read_dir(&self.into).map_or(true, |mut d| d.next().is_none());
        if whole {
            // nothing written: no folder is made for it
            if !all.is_empty() {
                let emptied = self.into.is_dir();
                if emptied {
                    std::fs::remove_dir(&self.into)
                        .map_err(|e| fail("replacing the empty", &self.into, &e))?;
                }
                std::fs::rename(&self.dir, &self.into).map_err(|e| {
                    if emptied {
                        let _ = std::fs::create_dir(&self.into);
                    }
                    fail("moving into", &self.into, &e)
                })?;
                moved(&self.into);
            }
            return land_pending(pending);
        }

        let (mut last, first): (Vec<_>, Vec<_>) =
            all.into_iter().partition(|(s, _)| is_marker(&name_of(s)));
        // the deepest marker first: the outermost names the run
        last.sort_by_key(|(s, _)| std::cmp::Reverse(s.components().count()));
        let mut aside = Aside {
            dir: self.dir.join(REPLACED),
            held: Vec::new(),
            landed: false,
        };
        std::fs::create_dir(&aside.dir).map_err(|e| fail("making", &aside.dir, &e))?;
        let landed = Self::land_each(&first, &last, pending, &mut aside, moved);
        if landed.is_err() {
            aside.restore();
        }
        landed
    }

    /// The renames of a commit file by file: `last`, the markers, after
    /// `first` and the changes to the manifests that are not staged.
    fn land_each(
        first: &[(PathBuf, PathBuf)],
        last: &[(PathBuf, PathBuf)],
        pending: Vec<Pending>,
        aside: &mut Aside,
        moved: &mut dyn FnMut(&Path),
    ) -> anyhow::Result<()> {
        let rename = |from: &Path, to: &Path| Ok(std::fs::rename(from, to)?);
        // an earlier run's markers first: none sits beside this run's outputs
        for (_, dst) in last.iter().filter(|(_, dst)| is_there(dst)) {
            let taken = if is_manifest(dst) {
                aside.take(dst, true, metadata::retire)
            } else {
                aside.take(dst, true, rename)
            };
            taken.map_err(|e| fail("replacing", dst, &e))?;
        }
        for (src, dst) in first {
            // a store over a store: the one there is moved aside first
            if src.is_dir() && is_there(dst) {
                aside
                    .take(dst, false, rename)
                    .map_err(|e| fail("replacing", dst, &e))?;
            }
            std::fs::rename(src, dst).map_err(|e| fail("moving into", dst, &e))?;
            aside.landed = true;
            moved(dst);
        }
        land_pending(pending)?;
        for (src, dst) in last {
            // a pinto manifest under its lock, as its writer writes it
            let put = if is_manifest(dst) {
                metadata::install(src, dst)
            } else {
                std::fs::rename(src, dst).map_err(Into::into)
            };
            put.map_err(|e| fail("moving into", dst, &e))?;
            // with the permissions of the one it replaces, as a manifest
            // written over another keeps them
            if let Some(old) = aside.marker_of(dst).and_then(|at| at.metadata().ok()) {
                let _ = std::fs::set_permissions(dst, old.permissions());
            }
            aside.landed = true;
            moved(dst);
        }
        Ok(())
    }

    /// Remove the staged outputs, and the folders made to stage them in:
    /// nothing reaches the final names.
    pub fn abort(self) {
        if let Err(e) = remove_all(&self.dir) {
            log::warn!("could not remove {}: {e}", self.dir.display());
        }
        remove_made(&self.made);
    }
}

/// The error of a commit's step `what` on `p`.
fn fail(what: &str, p: &Path, e: &dyn std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{what} {}: {e}", p.display())
}

fn name_of(p: &Path) -> String {
    p.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// Whether `p` is named as a pinto manifest is.
fn is_manifest(p: &Path) -> bool {
    name_of(p).ends_with(metadata::MANIFEST_SUFFIX)
}

/// The error of an earlier manifest at `path` that could only be written
/// in place, for the reason `why`: no rename can take it away and put the
/// new one there, so the run is refused.
fn kept_manifest(path: &Path, why: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "--atomic-outputs: the manifest {} is {why}, so it could only be written in place, not \
         replaced whole; move it away, or run without --atomic-outputs",
        path.display()
    )
}

/// Refuse the moves (staged, final) that would put an output over what
/// cannot take it: a manifest that is written in place
/// ([`metadata::kept_in_place`]), a link, a folder where a file lands or a
/// file where a folder lands. What is merged into (a plain folder over a
/// plain folder) is not among the moves.
fn refuse_unfit(moves: &[(PathBuf, PathBuf)]) -> anyhow::Result<()> {
    for (src, dst) in moves {
        let Ok(there) = dst.symlink_metadata() else {
            continue;
        };
        if is_manifest(dst) {
            if let Some(why) = metadata::kept_in_place(dst) {
                return Err(kept_manifest(dst, why));
            }
        }
        let why = if there.is_symlink() {
            "is a link"
        } else if there.is_dir() && !src.is_dir() {
            "is a folder where this run writes a file"
        } else if src.is_dir() && !there.is_dir() {
            "is a file where this run writes a folder"
        } else {
            continue;
        };
        anyhow::bail!(
            "{} {why}: nothing is moved over it; move it away or give another out",
            dst.display()
        );
    }
    Ok(())
}

/// Whether `a` and `b`, both there, are on one file system.
#[cfg(unix)]
fn same_file_system(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (a.metadata(), b.metadata()) {
        (Ok(a), Ok(b)) => a.dev() == b.dev(),
        _ => true,
    }
}

/// Whether `a` and `b` are on one file system: not known here, so taken
/// as so.
#[cfg(not(unix))]
fn same_file_system(_: &Path, _: &Path) -> bool {
    true
}

/// Make the changes to the manifests that are not staged, as they are
/// made without staging, each manifest read anew under its lock, so what
/// other runs wrote since is kept. A manifest this run must make
/// (`fresh`, when none is there) must be written; a failure to change one
/// that is there is warned about ([`PintoMetadata::backfill`]).
fn land_pending(pending: Vec<Pending>) -> anyhow::Result<()> {
    for Pending {
        path,
        fresh,
        changes,
    } in pending
    {
        if let Some(fresh) = fresh {
            fresh
                .write_if_missing(&path)
                .map_err(|e| e.context(format!("writing {}", path.display())))?;
        }
        PintoMetadata::backfill(&path, move |m| changes.into_iter().for_each(|set| set(m)));
    }
    Ok(())
}

// * The process's staging

/// Whether this pinto was started with `--atomic-outputs`.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// The run's staging, once begun.
static ACTIVE: Mutex<Option<Staging>> = Mutex::new(None);

fn active() -> MutexGuard<'static, Option<Staging>> {
    ACTIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// This pinto was started with `--atomic-outputs`.
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Whether this pinto was started with `--atomic-outputs`: it stages its
/// own outputs, and the pinto commands it starts stage theirs.
pub fn enabled() -> bool {
    #[cfg(test)]
    if ENABLED_HERE.get() {
        return true;
    }
    ENABLED.load(Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    /// A test's stand-in for a pinto started with `--atomic-outputs`: so
    /// on the thread that sets it, and on no other test's.
    pub(crate) static ENABLED_HERE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// The out a command writes for `out` (its prefix, or its folder with
/// `as_folder`), staged. Asked again for the same out, the same staged
/// one.
fn begin(out: &str, as_folder: bool) -> anyhow::Result<String> {
    let mut a = active();
    if let Some(s) = a.as_ref() {
        anyhow::ensure!(
            s.final_out == out,
            "--atomic-outputs stages one output per run: {} already, not {out}",
            s.final_out
        );
        return Ok(s.out().to_string());
    }
    let s = if as_folder {
        Staging::folder(out)?
    } else {
        Staging::prefix(out)?
    };
    let staged = s.out().to_string();
    *a = Some(s);
    Ok(staged)
}

/// The prefix a command writes for the output prefix `out`, staged.
pub fn stage_prefix(out: &str) -> anyhow::Result<String> {
    begin(out, false)
}

/// [`stage_prefix`] for a run that writes its own manifest at `out`:
/// refused at once, before the run does any work, when an earlier manifest
/// there could only be written in place ([`metadata::kept_in_place`]).
pub fn stage_run(out: &str) -> anyhow::Result<String> {
    let manifest = PathBuf::from(metadata::manifest_path(out));
    match metadata::kept_in_place(&manifest) {
        Some(why) => Err(kept_manifest(&manifest, why)),
        None => stage_prefix(out),
    }
}

/// The folder a command writes for the output folder `out`, staged.
pub fn stage_folder(out: &str) -> anyhow::Result<String> {
    begin(out, true)
}

/// The prefix a command writes for `out`, which it names only as it runs
/// (lra without `--out`): staged under `--atomic-outputs`, else `out`.
pub fn stage_later(out: &str) -> anyhow::Result<String> {
    if enabled() {
        stage_prefix(out)
    } else {
        Ok(out.to_string())
    }
}

/// The final path the run's `path`, a prefix or a path, is staged for;
/// `path` itself when it is not staged. What an existence check on a run's
/// outputs asks about, and what a message names.
pub fn final_str(path: &str) -> String {
    active()
        .as_ref()
        .and_then(|st| st.final_path(path))
        .unwrap_or_else(|| path.to_string())
}

/// [`final_str`] of a path.
pub fn final_path(path: &Path) -> PathBuf {
    PathBuf::from(final_str(&path.to_string_lossy()))
}

/// `error` with the staged paths its message names made the final ones:
/// what the run reports when it fails.
pub fn final_error(error: anyhow::Error) -> anyhow::Error {
    let a = active();
    let Some(st) = a.as_ref() else { return error };
    let said = format!("{error:?}");
    if said.contains(&st.marker) {
        anyhow::Error::msg(st.final_text(&said))
    } else {
        error
    }
}

/// Under staging, the manifest at `path`, when it is not a staged one,
/// changed by `set` once the outputs are in place, and `None`; else `set`
/// back, to change it now.
pub(crate) fn defer_update<F>(path: &Path, set: F) -> Option<F>
where
    F: FnOnce(&mut PintoMetadata) + Send + 'static,
{
    match active().as_mut() {
        Some(st) if !st.stages(path) => {
            st.update_elsewhere(path, set);
            None
        }
        _ => Some(set),
    }
}

/// Under staging, `fresh` written at `path`, when it is not a staged one
/// and no manifest is there once the outputs are in place; whether that
/// was arranged.
pub(crate) fn defer_write_if_missing(path: &Path, fresh: &PintoMetadata) -> bool {
    match active().as_mut() {
        Some(st) if !st.stages(path) => {
            st.write_elsewhere_if_missing(path, fresh.clone());
            true
        }
        _ => false,
    }
}

/// What a staged run says at its end, held until its outputs are in place.
static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Print `line`, a line of what a run says on stdout at its end. A staged
/// run holds it until its outputs are in place ([`said`]): it then names
/// nothing that is not there, and a reader that has gone (a closed pipe
/// ends the run) cannot end it with its outputs still staged.
pub fn say(line: impl std::fmt::Display) {
    if active().is_some() {
        SAID.lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(line.to_string());
    } else {
        println!("{line}");
    }
}

/// Print the lines a staged run held ([`say`]): once [`commit`] has put
/// its outputs in place, or by a command that fails with results to show
/// that name no output.
pub fn say_held() {
    let held = std::mem::take(&mut *SAID.lock().unwrap_or_else(|e| e.into_inner()));
    for line in held {
        println!("{line}");
    }
}

/// Put the run's staged outputs in place, if it staged any.
pub fn commit() -> anyhow::Result<()> {
    let mut a = active();
    let staged = a.take();
    drop(a);
    staged.map_or(Ok(()), Staging::commit)
}

/// Removes the run's staged outputs when dropped before [`commit`]: on an
/// error return, and on a panic.
pub struct AbortOnDrop;

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(s) = active().take() {
            s.abort();
        }
    }
}
