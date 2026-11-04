//! Raw source files, fetched once into the cache and hashed.
//!
//! A download streams into a temporary file of its own beside the target
//! (`<name>.<pid>-<nanos>-<k>.part`) and is renamed once whole, so a
//! half-written file is never read, and two builds at once never write
//! into one temporary file. A source with a pinned checksum (the
//! LIANA+ table at a fixed commit) is checked when it is fetched; the
//! others have no release to pin, so their sha256 is recorded instead.

use anyhow::{bail, Context, Result};
use log::info;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// One file to fetch: its name in the cache and where it comes from.
#[derive(Clone, Debug)]
pub struct Source {
    pub name: &'static str,
    pub url: String,
    /// Expected sha256 of a fresh download, when the source is pinned.
    pub sha256: Option<&'static str>,
}

/// A source file present in the cache.
#[derive(Clone, Debug)]
pub struct Fetched {
    pub name: &'static str,
    pub url: String,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

/// Every source in `dir`, downloading what is missing (or everything,
/// with `refresh`). With `offline`, nothing is downloaded: a missing file
/// is an error that names it and the cache.
pub fn fetch_all(
    dir: &Path,
    sources: &[Source],
    refresh: bool,
    offline: bool,
) -> Result<Vec<Fetched>> {
    let needed: Vec<&Source> = sources
        .iter()
        .filter(|s| refresh || !dir.join(s.name).is_file())
        .collect();
    if offline && !needed.is_empty() {
        let list: Vec<String> = needed
            .iter()
            .map(|s| format!("  {} ({})", s.name, s.url))
            .collect();
        bail!(
            "PINTO_OFFLINE is set, but the ligand-effect reference needs to download{}:\n{}\n\
             into {}; fetch them there, or unset PINTO_OFFLINE",
            if refresh { " (refresh asked)" } else { "" },
            list.join("\n"),
            dir.display()
        );
    }
    if !needed.is_empty() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    sources
        .iter()
        .map(|s| {
            let path = dir.join(s.name);
            let (sha256, bytes) = if refresh || !path.is_file() {
                download(s, &path)?
            } else {
                (sha256_file(&path)?, fs::metadata(&path)?.len())
            };
            Ok(Fetched {
                name: s.name,
                url: s.url.clone(),
                path,
                sha256,
                bytes,
            })
        })
        .collect()
}

/// Stream `src` to `to`, hashing on the way; returns (sha256, bytes).
fn download(src: &Source, to: &Path) -> Result<(String, u64)> {
    info!("downloading {}", src.url);
    let response = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout(Duration::from_secs(900))
        .user_agent(concat!("pinto-rs/", env!("CARGO_PKG_VERSION")))
        .build()
        .get(&src.url)
        .call()
        .with_context(|| format!("downloading {}", src.url))?;
    let tmp = part_path(to);
    let result = stream_into(src, response, &tmp, to);
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// The body of [`download`]: the response into `tmp`, checked, renamed to
/// `to`.
fn stream_into(
    src: &Source,
    response: ureq::Response,
    tmp: &Path,
    to: &Path,
) -> Result<(String, u64)> {
    let mut file = BufWriter::new(
        fs::File::create(tmp).with_context(|| format!("creating {}", tmp.display()))?,
    );
    let mut reader = response.into_reader();
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut bytes = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .with_context(|| format!("reading {}", src.url))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n])?;
        bytes += n as u64;
    }
    file.flush()?;
    drop(file);
    let sha = hex(&hasher.finalize());
    if let Some(expected) = src.sha256 {
        if sha != expected {
            bail!(
                "{}: sha256 {sha}, expected {expected}; the pinned source changed",
                src.url
            );
        }
    }
    fs::rename(tmp, to).with_context(|| format!("renaming into {}", to.display()))?;
    info!("cached {} ({bytes} bytes, sha256 {sha})", to.display());
    Ok((sha, bytes))
}

/// A temporary file beside `path`, unique to this process and call: its
/// file name, then the process id, the time in nanoseconds and a counter,
/// then `.part`. In the same directory, so the rename into `path` is atomic.
pub fn part_path(path: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let k = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}-{nanos}-{k}.part", std::process::id()));
    path.with_file_name(name)
}

/// Write `data` to `path` through a temporary file ([`part_path`]) and a
/// rename.
pub fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = part_path(path);
    if let Err(e) = fs::write(&tmp, data) {
        let _ = fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("writing {}", tmp.display()));
    }
    fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))
}

/// An advisory lock on a cache directory, held while it lives: a lock file
/// made with `create_new`, so one process builds while the others wait.
/// A lock older than [`STALE_LOCK`] is taken to be left by a process that
/// died, and broken.
pub struct BuildLock {
    path: PathBuf,
}

/// Age past which a lock file is broken.
pub const STALE_LOCK: Duration = Duration::from_secs(2 * 3600);

impl BuildLock {
    /// Wait for and take the lock `path`.
    pub fn acquire(path: &Path) -> Result<Self> {
        let mut said = false;
        loop {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
            {
                Ok(mut f) => {
                    let _ = writeln!(f, "{}", std::process::id());
                    return Ok(Self {
                        path: path.to_path_buf(),
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let age = fs::metadata(path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok());
                    if age.is_some_and(|a| a > STALE_LOCK) {
                        log::warn!("breaking a stale lock {}", path.display());
                        let _ = fs::remove_file(path);
                        continue;
                    }
                    if !said {
                        info!("waiting for another build to release {}", path.display());
                        said = true;
                    }
                    std::thread::sleep(Duration::from_millis(500));
                }
                Err(e) => {
                    return Err(e).with_context(|| format!("creating lock {}", path.display()))
                }
            }
        }
    }
}

impl Drop for BuildLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Hex sha256 of a file's bytes.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Hex sha256 of `data`.
pub fn sha256_bytes(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
