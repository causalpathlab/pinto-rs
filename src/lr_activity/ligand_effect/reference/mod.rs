//! Default ligand–receptor pairs and receptor → TF → target programmes,
//! built from public resources on first use and cached.
//!
//! [`ensure`] returns the two tables for a species, building them when the
//! cache lacks them:
//!
//! - `lr_pairs.<species>.tsv` (`ligand receptor source ligand_subunits
//!   receptor_subunits`): LIANA+ `consensus` / `mouseconsensus` pairs, with
//!   the OmniPath resources that support each;
//! - `receptor_targets.<species>.tsv` (`receptor target weight n_tfs
//!   min_path`): each receptor's top 100 targets by signed search through
//!   OmniPath signalling to CollecTRI regulons (see [`programmes`]); mouse
//!   from the human networks through HCOP one-to-one orthologs.
//!
//! [`LicenceMode::Permissive`] keeps an interaction only when one of its
//! supporting resources is licensed for commercial use without share-alike
//! or no-derivatives terms (OmniPath's licence table);
//! [`LicenceMode::All`] keeps everything, for local use only.
//!
//! Cache root: [`ReferenceOpts::dir`], else `$PINTO_REFERENCE_DIR`, else
//! the user cache directory's `pinto/ligand_effect`. It holds `downloads/`
//! (the raw files, reused until a refresh) and `built/<licence>/` (both
//! species' tables and `reference.json`: URLs and sha256 of the downloads,
//! the LIANA+ commit, build time, licence mode, row counts and the pinto
//! version that built them). With `$PINTO_OFFLINE` set, nothing is
//! downloaded and a missing file is an error.
//!
//! The build ports `scripts/build_ligand_effect_data.py`: for the same
//! downloads it writes the same bytes (rows sorted, weights `%.6g`, sums
//! exactly rounded).
//!
//! Layout: [`download`] fetches and hashes the raw files, [`table`] reads
//! and writes delimited text, [`sources`] reads LIANA+, OmniPath and HCOP
//! and applies the licence filter, and [`programmes`] builds the networks,
//! the per-receptor targets and their mouse translation.

mod download;
mod programmes;
pub mod sources;
mod table;

#[cfg(test)]
mod tests;

use anyhow::{Context, Result};
use log::info;
use rustc_hash::FxHashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Edges from a receptor to its farthest TF.
const MAX_STEPS: usize = 3;
/// Targets kept per receptor.
pub const TOP: usize = 100;
/// HCOP resources an ortholog pair needs.
const HCOP_MIN_EVIDENCE: usize = 3;

/// Bumped whenever the build changes what it writes, so a cache built by
/// an older rule is rebuilt instead of reused. 2: ligands must be annotated
/// as transmitted (OmniPath intercell).
const BUILD_REVISION: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Species {
    Human,
    Mouse,
}

impl Species {
    /// The name in file names and the record.
    pub fn tag(self) -> &'static str {
        match self {
            Species::Human => "human",
            Species::Mouse => "mouse",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LicenceMode {
    /// Only interactions with a redistributable supporting resource.
    #[default]
    Permissive,
    /// Every interaction (not redistributable).
    All,
}

impl LicenceMode {
    /// The name of its directory under `built/`.
    pub fn tag(self) -> &'static str {
        match self {
            LicenceMode::Permissive => "permissive",
            LicenceMode::All => "all",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReferenceOpts {
    /// Cache root (default: `$PINTO_REFERENCE_DIR`, else the user cache).
    pub dir: Option<PathBuf>,
    /// Download and build again even when cached.
    pub refresh: bool,
    pub licence: LicenceMode,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceFiles {
    pub lr_pairs: PathBuf,
    pub receptor_targets: PathBuf,
    pub record: PathBuf,
}

impl ReferenceFiles {
    fn at(built: &Path, species: Species) -> Self {
        let tag = species.tag();
        ReferenceFiles {
            lr_pairs: built.join(format!("lr_pairs.{tag}.tsv")),
            receptor_targets: built.join(format!("receptor_targets.{tag}.tsv")),
            record: built.join("reference.json"),
        }
    }

    fn exist(&self) -> bool {
        [&self.lr_pairs, &self.receptor_targets, &self.record]
            .iter()
            .all(|p| p.is_file())
    }
}

/// Build (or reuse) the reference for `species`; returns its files.
// Live before the lra front end calls it (and with it the whole build).
pub fn ensure(species: Species, opts: &ReferenceOpts) -> Result<ReferenceFiles> {
    let offline = std::env::var_os("PINTO_OFFLINE").is_some_and(|v| !v.is_empty() && v != "0");
    ensure_with(species, opts, offline)
}

fn ensure_with(species: Species, opts: &ReferenceOpts, offline: bool) -> Result<ReferenceFiles> {
    let root = cache_root(opts.dir.as_deref())?;
    let built = root
        .join("built")
        .join(format!("r{BUILD_REVISION}-{}", opts.licence.tag()));
    let files = ReferenceFiles::at(&built, species);
    if !opts.refresh && files.exist() {
        info!("ligand-effect reference: reusing {}", built.display());
        return Ok(files);
    }
    // One build at a time per cache; a process that waited reuses what the
    // other built.
    fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
    let _lock = download::BuildLock::acquire(&root.join(".build.lock"))?;
    if !opts.refresh && files.exist() {
        info!("ligand-effect reference: reusing {}", built.display());
        return Ok(files);
    }
    build(
        &root.join("downloads"),
        &built,
        opts.licence,
        opts.refresh,
        offline,
    )?;
    Ok(files)
}

#[cfg(test)]
thread_local! {
    /// A test's cache root, in place of the environment's on this thread
    /// ([`with_test_root`]).
    static TEST_ROOT: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

/// Run `f` with `root` as the cache root on this thread, so a test reads
/// a reference it wrote without touching the user's cache or the process
/// environment.
#[cfg(test)]
pub fn with_test_root<T>(root: &Path, f: impl FnOnce() -> T) -> T {
    /// Clears the root when dropped, a panicking `f` too.
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_ROOT.with(|r| *r.borrow_mut() = None);
        }
    }
    TEST_ROOT.with(|r| *r.borrow_mut() = Some(root.to_path_buf()));
    let _reset = Reset;
    f()
}

/// An empty reference for `species` under the cache root `root`: its
/// tables hold only their headers, so a test's own pair and programme
/// files are all the pairs there are.
#[cfg(test)]
pub fn write_empty_reference(root: &Path, species: Species) -> Result<()> {
    let built = root
        .join("built")
        .join(format!("r{BUILD_REVISION}-{}", LicenceMode::All.tag()));
    fs::create_dir_all(&built)?;
    fs::create_dir_all(root.join("downloads"))?;
    let files = ReferenceFiles::at(&built, species);
    fs::write(&files.lr_pairs, "ligand\treceptor\n")?;
    fs::write(&files.receptor_targets, "receptor\ttarget\tweight\n")?;
    fs::write(&files.record, "{}")?;
    Ok(())
}

/// The cache root: `dir`, else `$PINTO_REFERENCE_DIR`, else the user cache.
pub fn cache_root(dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(d) = dir {
        return Ok(d.to_path_buf());
    }
    #[cfg(test)]
    if let Some(d) = TEST_ROOT.with(|r| r.borrow().clone()) {
        return Ok(d);
    }
    if let Some(d) = std::env::var_os("PINTO_REFERENCE_DIR").filter(|d| !d.is_empty()) {
        return Ok(d.into());
    }
    dirs::cache_dir()
        .map(|d| d.join("pinto").join("ligand_effect"))
        .context("no user cache directory: give a reference directory or set PINTO_REFERENCE_DIR")
}

/// Fetch the sources into `download_dir` and write both species' tables and
/// the record into `out`.
fn build(
    download_dir: &Path,
    out: &Path,
    mode: LicenceMode,
    refresh: bool,
    offline: bool,
) -> Result<()> {
    use programmes::*;
    use sources::*;

    info!(
        "ligand-effect reference: building ({} licences) in {}",
        mode.tag(),
        out.display()
    );
    let fetched = download::fetch_all(download_dir, &sources::downloads(), refresh, offline)?;
    let path_of = |name: &str| {
        fetched
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.path.as_path())
            .expect("every source is fetched")
    };
    let read = |name: &str| {
        let p = path_of(name);
        fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))
    };

    let lic = Licences::from_json(&read(RESOURCES)?)?;
    let signalling = parse_interactions(&read(SIGNALLING)?).context(SIGNALLING)?;
    let ligrec = parse_interactions(&read(LIGREC)?).context(LIGREC)?;
    let collectri = parse_interactions(&read(COLLECTRI)?).context(COLLECTRI)?;
    let liana = parse_liana(&read(LIANA)?).context(LIANA)?;

    let prov = Provenance::new(signalling.iter().chain(&ligrec), &lic, mode);
    let unknown = unknown_sources(signalling.iter().chain(&ligrec).chain(&collectri), &lic);
    let (human_lr, human_unsupported) = lr_pairs(&liana, &prov, "consensus");
    let (mouse_lr, mouse_unsupported) = lr_pairs(&liana, &prov, "mouseconsensus");
    let h2m = hcop_one_to_one(&read_maybe_gz(path_of(HCOP))?, HCOP_MIN_EVIDENCE).context(HCOP)?;
    // Only ligands that can leave the cell or sit on its surface.
    let ligands = parse_intercell_ligands(&read(INTERCELL)?).context(INTERCELL)?;
    let m2h: FxHashMap<String, String> = h2m
        .iter()
        .map(|(h, m)| (m.to_ascii_uppercase(), h.to_string()))
        .collect();
    let (human_lr, human_intracellular) = transmitted_pairs(human_lr, &ligands, &|s| s.to_string());
    let (mouse_lr, mouse_intracellular) = transmitted_pairs(mouse_lr, &ligands, &|s| {
        m2h.get(&s.to_ascii_uppercase())
            .cloned()
            .unwrap_or_else(|| s.to_string())
    });
    info!(
        "ligand-effect reference: dropped {human_intracellular} human and \
         {mouse_intracellular} mouse pairs whose ligand is not annotated as transmitted"
    );

    let (graph, signalling_stats) = build_signalling(&signalling, &lic, mode);
    let (regulons, collectri_stats) = build_regulons(&collectri, &lic, mode);
    let (human_targets, human_stats) =
        species_targets(&human_lr, &graph, &regulons, MAX_STEPS, TOP);

    let (mouse_targets, mouse_stats) = species_targets(
        &mouse_lr,
        &translate_graph(&graph, &h2m),
        &translate_regulons(&regulons, &h2m),
        MAX_STEPS,
        TOP,
    );

    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let lr_header = [
        "ligand",
        "receptor",
        "source",
        "ligand_subunits",
        "receptor_subunits",
    ];
    let target_header = ["receptor", "target", "weight", "n_tfs", "min_path"];
    let mut species_record = serde_json::Map::new();
    for (species, lr, unsupported, intracellular, targets, stats) in [
        (
            Species::Human,
            &human_lr,
            human_unsupported,
            human_intracellular,
            &human_targets,
            human_stats,
        ),
        (
            Species::Mouse,
            &mouse_lr,
            mouse_unsupported,
            mouse_intracellular,
            &mouse_targets,
            mouse_stats,
        ),
    ] {
        let files = ReferenceFiles::at(out, species);
        let lr_rows: Vec<[String; 5]> = lr.iter().map(LrPair::row).collect();
        let lr_text = table::format_tsv(&lr_header, &lr_rows);
        let target_text = table::format_tsv(&target_header, targets);
        download::write_atomic(&files.lr_pairs, lr_text.as_bytes())?;
        download::write_atomic(&files.receptor_targets, target_text.as_bytes())?;
        info!(
            "ligand-effect reference: {}: {} pairs, {} of {} receptors with targets, {} target rows",
            species.tag(),
            lr.len(),
            stats["receptors_with_targets"],
            stats["receptors"],
            targets.len()
        );
        let tag = species.tag();
        let liana_rows = liana
            .iter()
            .filter(|r| {
                &*r.resource
                    == match species {
                        Species::Human => "consensus",
                        Species::Mouse => "mouseconsensus",
                    }
            })
            .count();
        species_record.insert(
            tag.into(),
            serde_json::json!({
                "counts": stats,
                "liana_rows": liana_rows,
                "liana_rows_without_omnipath_support": unsupported,
                "pairs_dropped_ligand_not_transmitted": intracellular,
                "outputs": {
                    format!("lr_pairs.{tag}.tsv"): {
                        "rows": lr.len(),
                        "sha256": download::sha256_bytes(lr_text.as_bytes()),
                    },
                    format!("receptor_targets.{tag}.tsv"): {
                        "rows": targets.len(),
                        "sha256": download::sha256_bytes(target_text.as_bytes()),
                    },
                },
            }),
        );
    }

    let downloads_record: serde_json::Map<String, serde_json::Value> = fetched
        .iter()
        .map(|f| {
            (
                f.name.to_string(),
                serde_json::json!({ "url": f.url, "sha256": f.sha256, "bytes": f.bytes }),
            )
        })
        .collect();
    let record = serde_json::json!({
        "builder": env!("CARGO_PKG_NAME"),
        "builder_version": env!("CARGO_PKG_VERSION"),
        "built": utc_now(),
        "licence": mode.tag(),
        "liana_commit": LIANA_COMMIT,
        "max_steps": MAX_STEPS,
        "top": TOP,
        "hcop_min_evidence": HCOP_MIN_EVIDENCE,
        "hcop_one_to_one_pairs": h2m.len(),
        "downloads": downloads_record,
        "signalling_stats": signalling_stats,
        "collectri_stats": collectri_stats,
        "sources_without_licence_entry": unknown,
        "species": species_record,
    });
    let mut text = serde_json::to_string_pretty(&record)?;
    text.push('\n');
    download::write_atomic(&out.join("reference.json"), text.as_bytes())
}

/// A text file, gunzipped when it starts with the gzip magic.
fn read_maybe_gz(path: &Path) -> Result<String> {
    let raw = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut text = String::new();
        flate2::read::MultiGzDecoder::new(&raw[..])
            .read_to_string(&mut text)
            .with_context(|| format!("gunzipping {}", path.display()))?;
        Ok(text)
    } else {
        String::from_utf8(raw).with_context(|| format!("{} is not UTF-8", path.display()))
    }
}

/// The current UTC time, `YYYY-MM-DDTHH:MM:SSZ`.
fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let s = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        s / 3600,
        s / 60 % 60,
        s % 60
    )
}

/// Proleptic Gregorian (year, month, day) of a day count from 1970-01-01
/// (H. Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}
