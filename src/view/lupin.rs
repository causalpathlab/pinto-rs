//! `pinto view` ↔ lupin: annotation rounds of a run's communities.
//!
//! The viewer never writes labels. `lupin annotate` makes a round from a
//! level's communities and a marker panel, `lupin relabel` makes the next
//! round from decisions sent on stdin, and `lupin review --json` reads a
//! round back. Every call is a subprocess run off the UI thread ([`Job`]).
//!
//! A round is `{out}.lupin.json`; its `annotate.source` names the round it
//! was made from, back to the run's `.pinto.json`. The newest round of a
//! chain is the one no other round names as its source.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// The lupin binary: `$PINTO_LUPIN`, else `lupin` on the `PATH`.
pub fn binary() -> String {
    std::env::var("PINTO_LUPIN").unwrap_or_else(|_| "lupin".into())
}

/// A lupin subprocess running in the background. Its stderr, one line at
/// a time, is the progress shown while it runs; on failure its last line
/// is the reason.
pub struct Job {
    pub what: JobKind,
    progress: Arc<Mutex<String>>,
    handle: Option<JoinHandle<Result<String, String>>>,
}

/// What a finished [`Job`] is for.
pub enum JobKind {
    /// A first round, written to this manifest.
    Annotate(PathBuf),
    /// `relabel --next` from this round.
    Next(PathBuf),
    /// `relabel --preview`.
    Preview,
}

impl Job {
    /// Run `lupin args…` in `dir` with `stdin` fed in; the result is its
    /// stdout.
    pub fn spawn(what: JobKind, dir: PathBuf, args: Vec<String>, stdin: Option<String>) -> Self {
        let progress = Arc::new(Mutex::new(String::from("starting lupin ...")));
        let shared = Arc::clone(&progress);
        let handle = std::thread::spawn(move || run(&dir, &args, stdin, &shared));
        Job {
            what,
            progress,
            handle: Some(handle),
        }
    }

    /// The last progress line.
    pub fn progress(&self) -> String {
        self.progress.lock().map(|p| p.clone()).unwrap_or_default()
    }

    /// The result once the subprocess has exited; `None` while it runs.
    pub fn poll(&mut self) -> Option<Result<String, String>> {
        if !self.handle.as_ref().is_some_and(JoinHandle::is_finished) {
            return None;
        }
        let handle = self.handle.take()?;
        Some(
            handle
                .join()
                .unwrap_or_else(|_| Err("lupin job panicked".into())),
        )
    }
}

fn run(
    dir: &Path,
    args: &[String],
    stdin: Option<String>,
    progress: &Mutex<String>,
) -> Result<String, String> {
    let bin = binary();
    let mut child = Command::new(&bin)
        .current_dir(dir)
        .args(args)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run {bin}: {e} (set PINTO_LUPIN to its path)"))?;
    if let (Some(text), Some(mut pipe)) = (stdin, child.stdin.take()) {
        // A write error shows up as lupin's own complaint on stderr.
        let _ = pipe.write_all(text.as_bytes());
    }
    // stdout on its own thread, so neither pipe fills and stalls lupin.
    let mut out_pipe = child.stdout.take().expect("piped");
    let out = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out_pipe.read_to_string(&mut s);
        s
    });
    let mut last = String::new();
    for line in BufReader::new(child.stderr.take().expect("piped"))
        .lines()
        .map_while(Result::ok)
    {
        let line = strip_log_prefix(&line).trim().to_string();
        if line.is_empty() {
            continue;
        }
        if let Ok(mut p) = progress.lock() {
            p.clone_from(&line);
        }
        last = line;
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    let stdout = out.join().unwrap_or_default();
    if status.success() {
        Ok(stdout)
    } else if last.is_empty() {
        Err(format!("lupin failed ({status})"))
    } else {
        Err(last)
    }
}

/// `[2026-09-30T19:03:26Z INFO  lupin::annotate] text` → `text`.
fn strip_log_prefix(line: &str) -> &str {
    match (line.starts_with('['), line.find("] ")) {
        (true, Some(i)) => &line[i + 2..],
        _ => line,
    }
}

/// `lupin annotate` of level `tag` of the run at `manifest`, into `out`.
///
/// lupin runs in the manifest's directory: a run records its data files
/// as typed when it was fit, usually from there.
pub fn annotate(manifest: &Path, tag: &str, markers: &Path, out: &str) -> Job {
    let manifest = absolute(manifest);
    let out = absolute(Path::new(out)).display().to_string();
    let args = vec![
        "annotate".into(),
        "-f".into(),
        manifest.display().to_string(),
        "--level".into(),
        tag.into(),
        "-m".into(),
        absolute(markers).display().to_string(),
        "-o".into(),
        out.clone(),
        // Only enrichment keeps the level's communities as the clusters and
        // writes the per-cluster gene evidence relabelling reads.
        "--method".into(),
        "enrichment".into(),
    ];
    Job::spawn(
        JobKind::Annotate(PathBuf::from(format!("{out}.lupin.json"))),
        manifest_dir(&manifest),
        args,
        None,
    )
}

/// `p` from the root, so it reads the same from any working directory.
fn absolute(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

fn manifest_dir(manifest: &Path) -> PathBuf {
    manifest
        .parent()
        .map_or_else(|| PathBuf::from("."), dir_or_cwd)
}

/// `lupin relabel` of `round` with `decisions` (JSON lines): the next
/// round, or with `preview` a JSON account of what it would change.
pub fn relabel(round: &Path, decisions: String, preview: bool) -> Job {
    let round = absolute(round);
    let args = vec![
        "relabel".into(),
        "-f".into(),
        round.display().to_string(),
        "-d".into(),
        "-".into(),
        if preview { "--preview" } else { "--next" }.into(),
    ];
    let what = if preview {
        JobKind::Preview
    } else {
        JobKind::Next(round.clone())
    };
    Job::spawn(what, manifest_dir(&round), args, Some(decisions))
}

/// The first `{prefix}.{tag}.a{k}` without a round manifest yet.
pub fn annotate_out(prefix: &str, tag: &str) -> String {
    (1..)
        .map(|k| format!("{prefix}.{tag}.a{k}"))
        .find(|out| !Path::new(&format!("{out}.lupin.json")).exists())
        .expect("unbounded")
}

/// The level a round annotated: as lupin records it, else from the name
/// pinto gave the chain's first round (`{prefix}.{tag}.a{k}`).
pub fn round_level(round: &Path, tags: &[&str]) -> Option<String> {
    let mut at = round.to_path_buf();
    for _ in 0..1000 {
        let m = read_json(&at)?;
        let a = m.get("annotate")?;
        let recorded = a
            .get("level")
            .or_else(|| a.pointer("/settings/enrichment/level"))
            .and_then(|v| v.as_str());
        if let Some(tag) = recorded {
            return Some(tag.to_string());
        }
        let source = resolve(&at, a.get("source")?.as_str()?);
        if !source.to_string_lossy().ends_with(".lupin.json") {
            // `at` is the chain's first round.
            let stem = file_name(&at);
            return tags
                .iter()
                .find(|t| stem.contains(&format!(".{t}.a")))
                .map(|t| t.to_string());
        }
        at = source;
    }
    None
}

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
        if !file_name(&path).ends_with(".lupin.json") {
            continue;
        }
        let from = read_json(&path)
            .and_then(|m| m.pointer("/annotate/source")?.as_str().map(String::from));
        if let Some(from) = from {
            source.insert(canonical(&path), canonical(&resolve(&path, &from)));
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
    let parents: std::collections::HashSet<&PathBuf> = source.values().collect();
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

fn dir_or_cwd(p: &Path) -> PathBuf {
    if p.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        p.to_path_buf()
    }
}

/// `p` resolved through links, or as given when it does not resolve.
pub fn canonical(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

pub fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

pub fn read_json(path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// `path` as written in `manifest`: relative to the manifest's directory.
pub fn resolve(manifest: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    dir_or_cwd(manifest.parent().unwrap_or(Path::new(""))).join(p)
}

// ── lupin review --json ─────────────────────────────────────────────────

/// One cluster of a round, as `lupin review --json` reports it.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Reviewed {
    #[serde(default)]
    pub digest: Digest,
    #[serde(default)]
    pub history: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Digest {
    #[serde(default)]
    pub size: usize,
    #[serde(default)]
    pub label: Option<String>,
    /// Candidate labels, most significant first.
    #[serde(default)]
    pub calls: Vec<Call>,
    #[serde(default)]
    pub evidence: Option<Evidence>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Call {
    pub label: String,
    #[serde(default)]
    pub q: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Evidence {
    #[serde(default)]
    pub agrees: Option<bool>,
}

/// `lupin review -f round --json`, keyed by cluster id. Blocks: a round's
/// review takes about a second.
pub fn review(round: &Path) -> anyhow::Result<BTreeMap<i64, Reviewed>> {
    let out = Command::new(binary())
        .args(["review", "-f"])
        .arg(round)
        .arg("--json")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| anyhow::anyhow!("cannot run {}: {e}", binary()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let last = err
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        anyhow::bail!("lupin review: {}", strip_log_prefix(last));
    }
    let map: BTreeMap<String, Reviewed> = serde_json::from_slice(&out.stdout)?;
    Ok(map
        .into_iter()
        .filter_map(|(k, v)| Some((k.parse().ok()?, v)))
        .collect())
}

/// The oldest lupin with everything the viewer uses: `annotate --level`,
/// `relabel --preview/--next`, `review --json`.
const MIN_LUPIN: (u32, u32, u32) = (0, 2, 1);

/// Whether the installed lupin is new enough; the reason when it is not.
/// A good answer is kept for the session; a bad one is asked again, so
/// installing lupin needs no restart.
pub fn check() -> Result<(), String> {
    static OK: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if OK.get().is_some() {
        return Ok(());
    }
    let out = Command::new(binary())
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run {}: {e} (set PINTO_LUPIN to its path)", binary()))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let version = text.split_whitespace().last().unwrap_or("");
    let mut parts = version
        .split(['.', '-'])
        .map(|p| p.parse::<u32>().unwrap_or(0));
    let v = (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    );
    if v < MIN_LUPIN {
        let (a, b, c) = MIN_LUPIN;
        return Err(format!(
            "lupin {version} is too old; need {a}.{b}.{c}+ (cargo install lupin-rs)"
        ));
    }
    let _ = OK.set(());
    Ok(())
}

// ── marker panels ───────────────────────────────────────────────────────

/// lupin's label key: split on whitespace, `,` and `_`, joined with `_`, so
/// "CT 1" and "CT_1" are one type. Compared case-insensitively.
pub fn label_key(label: &str) -> String {
    label
        .split(|c: char| c.is_whitespace() || c == ',' || c == '_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("_")
        .to_lowercase()
}

/// A marker panel: cell type → its genes, in file order.
pub struct Panel {
    pub types: Vec<(String, Vec<String>)>,
}

impl Panel {
    /// Read `gene<TAB>type` (or `gene,type`) lines, as lupin does: blank,
    /// `#` and header lines skipped, gzip read through.
    pub fn read(path: &Path) -> anyhow::Result<Panel> {
        let file = std::fs::File::open(path)?;
        let mut text = String::new();
        if file_name(path).ends_with(".gz") {
            flate2::read::MultiGzDecoder::new(file).read_to_string(&mut text)?;
        } else {
            BufReader::new(file).read_to_string(&mut text)?;
        }
        let mut types: Vec<(String, Vec<String>)> = Vec::new();
        let mut at: BTreeMap<String, usize> = BTreeMap::new();
        for line in text.lines() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let (gene, kind) = match line.split_once('\t').or_else(|| line.split_once(',')) {
                Some((g, t)) => (g.trim(), t.split('\t').next().unwrap_or("").trim()),
                None => continue,
            };
            let lower = gene.to_lowercase();
            if gene.is_empty() || kind.is_empty() || lower == "gene" || lower == "symbol" {
                continue;
            }
            let i = *at.entry(label_key(kind)).or_insert_with(|| {
                types.push((kind.to_string(), Vec::new()));
                types.len() - 1
            });
            types[i].1.push(gene.to_string());
        }
        Ok(Panel { types })
    }

    /// The genes listed for `label`, if the panel has that type.
    pub fn genes(&self, label: &str) -> Option<&[String]> {
        let key = label_key(label);
        self.types
            .iter()
            .find(|(t, _)| label_key(t) == key)
            .map(|(_, g)| &g[..])
    }

    /// Types listing `gene`.
    pub fn types_of(&self, gene: &str) -> Vec<&str> {
        self.types
            .iter()
            .filter(|(_, g)| g.iter().any(|x| x.eq_ignore_ascii_case(gene)))
            .map(|(t, _)| t.as_str())
            .collect()
    }
}
