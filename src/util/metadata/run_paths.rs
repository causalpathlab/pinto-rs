//! Where a run's manifest is and how a run is named: a run's prefix, its
//! manifest's path, and resolving the files it records beside it.

use super::*;

/// What a run's manifest is named by, after its prefix.
pub(crate) const MANIFEST_SUFFIX: &str = ".pinto.json";

/// A run named by its prefix or by its manifest: the prefix.
pub fn run_prefix(run: &str) -> &str {
    run.strip_suffix(MANIFEST_SUFFIX).unwrap_or(run)
}

/// A run's file name without its directory or the suffix of the file it was
/// named by (`.pinto.json`, `.predictive.parquet`, `.propensity.parquet`, …).
pub fn run_stem(run: &str) -> String {
    let name = Path::new(run)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    [
        MANIFEST_SUFFIX,
        ".predictive.parquet",
        ".propensity.parquet",
        ".parquet",
    ]
    .iter()
    .find_map(|s| name.strip_suffix(s))
    .unwrap_or(&name)
    .to_string()
}

/// A run named by its prefix or by its manifest: the manifest's path.
pub fn run_manifest(run: &str) -> String {
    manifest_path(run_prefix(run))
}

/// The manifest a run writes under the output prefix `prefix`, taken as
/// given.
pub fn manifest_path(prefix: &str) -> String {
    format!("{prefix}{MANIFEST_SUFFIX}")
}

/// A prior run, opened by its prefix or its manifest.
#[derive(Debug, Clone)]
pub struct Run {
    pub manifest: std::path::PathBuf,
    /// The run's prefix, from the manifest's own path: its directory and
    /// file name, so a run opened from elsewhere still finds its files.
    pub prefix: String,
    pub meta: PintoMetadata,
}

/// The `residual.json` beside data file `path` that says its counts are
/// excess counts, as the residual fit pinto had up to 0.17.10 wrote them.
pub fn excess_marker(path: &Path) -> Option<PathBuf> {
    let marker = path.parent()?.join("residual.json");
    let json = crate::util::lupin_rounds::read_json(&marker)?;
    (json["counts"] == "excess").then_some(marker)
}

impl Run {
    /// The run `run` names: its manifest (`{prefix}.pinto.json`, or any
    /// manifest file named in full), or its prefix. The prefix of
    /// `{prefix}.pinto.json` is `{prefix}`; of a manifest named otherwise
    /// (`run1.json`), its directory joined with the file name of the prefix
    /// the manifest records, so it still finds the run's files (its own
    /// name less `.json` when it records none).
    pub fn open(run: &str) -> anyhow::Result<Self> {
        let manifest = Self::manifest_of(run);
        anyhow::ensure!(
            manifest.is_file(),
            "{}: no such manifest (a prior run's {{prefix}}.pinto.json)",
            manifest.display()
        );
        let meta = PintoMetadata::read(&manifest)
            .map_err(|e| anyhow::anyhow!("{}: not a pinto manifest: {e}", manifest.display()))?;
        let m = manifest.to_string_lossy();
        let dir = manifest.parent().unwrap_or(Path::new(""));
        let prefix = match (
            m.strip_suffix(MANIFEST_SUFFIX),
            Path::new(&meta.prefix).file_name(),
        ) {
            (Some(p), _) => p.to_string(),
            (None, Some(stem)) => dir.join(stem).to_string_lossy().into_owned(),
            (None, None) => m.strip_suffix(".json").unwrap_or(&m).to_string(),
        };
        Ok(Self {
            prefix,
            manifest,
            meta,
        })
    }

    /// The manifest file `run` names: itself when it is a file, else the
    /// manifest of the prefix it is.
    fn manifest_of(run: &str) -> PathBuf {
        if Path::new(run).is_file() {
            PathBuf::from(run)
        } else {
            PathBuf::from(run_manifest(run))
        }
    }

    /// A path as the manifest writes it ([`resolve_beside`]).
    pub fn resolve(&self, path: &str) -> Option<std::path::PathBuf> {
        resolve_beside(&self.manifest, path)
    }

    /// Refused when the run is a stage-2 fit of the residual fit pinto had
    /// up to 0.17.10: its manifest records the stage-1 divisor (`offset`)
    /// or a `residual` block, or its first data file holds excess counts
    /// ([`excess_marker`]). Such a run was fitted on counts over the
    /// stage-1 expectation, or on the counts above it, which nothing now
    /// reads: taken as a run on raw counts, its scores would be wrong. The
    /// manifest is read again as plain JSON: [`PintoMetadata`] no longer
    /// has those fields, and they are only looked for here.
    pub fn refuse_residual_fit(&self) -> anyhow::Result<()> {
        let manifest = crate::util::lupin_rounds::read_json(&self.manifest);
        let recorded = manifest.as_ref().and_then(|m| {
            ["offset", "residual"]
                .into_iter()
                .find(|k| !m[*k].is_null())
        });
        let excess = self
            .meta
            .data_files
            .as_ref()
            .and_then(|f| f.first())
            .and_then(|f| self.resolve(f))
            .and_then(|f| excess_marker(&f));
        let why = match (recorded, excess) {
            (Some(key), _) => format!("its manifest records `{key}`"),
            (None, Some(marker)) => format!("its data are excess counts ({})", marker.display()),
            (None, None) => return Ok(()),
        };
        anyhow::bail!(
            "{} is a residual (two-stage) fit, which pinto no longer reads ({why}): refit it \
             on the raw data files",
            self.manifest.display()
        )
    }

    /// An output the manifest records as `recorded`, found without the
    /// working directory: the recorded path when absolute, beside the
    /// manifest when relative. A record that names the run's own
    /// `{prefix}.{suffix}` as the fit's prefix had it (a run moved since)
    /// is also found at the run's own file under its prefix now; any other
    /// record is not, so a stale file under the prefix is never taken for
    /// a recorded one that is gone. `None` when the manifest records none;
    /// an error when it records one that is not found.
    pub fn recorded_output(
        &self,
        recorded: Option<&str>,
        suffix: &str,
    ) -> anyhow::Result<Option<PathBuf>> {
        let Some(recorded) = recorded else {
            return Ok(None);
        };
        let dir = self.manifest.parent().unwrap_or(Path::new(""));
        // `join` keeps an absolute path as it is
        let beside = dir.join(recorded);
        if beside.is_file() {
            return Ok(Some(beside));
        }
        let own = PathBuf::from(format!("{}.{suffix}", self.prefix));
        let names_own = recorded == format!("{}.{suffix}", self.meta.prefix);
        anyhow::ensure!(
            names_own && own.is_file(),
            "{}: records {recorded}, which is not there{}",
            self.manifest.display(),
            if names_own {
                format!(" nor at {}", own.display())
            } else {
                String::new()
            }
        );
        Ok(Some(own))
    }

    /// [`Self::recorded_output`] for an output the run must have: an error
    /// naming it when the manifest records none.
    pub fn required_recorded_output(
        &self,
        recorded: Option<&str>,
        suffix: &str,
    ) -> anyhow::Result<PathBuf> {
        self.recorded_output(recorded, suffix)?
            .ok_or_else(|| anyhow::anyhow!("{}: records no {suffix}", self.manifest.display()))
    }
}

/// `path` as written in `manifest`: an absolute path as is; a relative one
/// under the manifest's directory when it exists there, else as given
/// (relative to the working directory) when it exists there. `None` when it
/// exists in neither place; the caller names both.
pub fn resolve_beside(manifest: &Path, path: &str) -> Option<std::path::PathBuf> {
    let p = Path::new(path);
    if p.is_absolute() {
        return p.exists().then(|| p.to_path_buf());
    }
    let dir = manifest.parent().unwrap_or(Path::new(""));
    let under = dir.join(p);
    if under.exists() {
        return Some(under);
    }
    p.exists().then(|| p.to_path_buf())
}
