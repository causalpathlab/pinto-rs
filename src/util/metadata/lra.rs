//! The record of a run's `pinto lra` outputs and options.

use serde::{Deserialize, Serialize};

/// What `pinto lra` (contact effect) and `pinto lra-simulate` recorded:
/// their tables, the reference behind them, and the fixed settings they
/// were made with (none of them is a flag). Paths are absolute.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(default)]
pub struct LraFiles {
    /// The contact-effect table, `{out}.lra.parquet`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table: Option<String>,
    /// Each pair's threshold–response curve, `{out}.lra.curve.parquet`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curve: Option<String>,
    /// The power and calibration check, `{out}.lra-simulate.parquet`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulate: Option<String>,
    /// The reference record (`reference.json`: sources, download
    /// checksums, build date) behind the table's pairs and programmes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// The fixed settings of the last analysis or check recorded here
    /// ([`crate::lr_activity::ligand_effect::Internals`]), kept as JSON: a
    /// record for the reader, never read back, so a settings block from
    /// another version can never make the manifest unreadable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settings: Option<serde_json::Value>,
    /// What the full table was made with (`--lr-pairs`, `--targets`, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<LraOptions>,
    /// Tables of a subset of the pairs (`pinto lra --only`), kept apart
    /// from [`LraFiles::table`]: their q-values are corrected within the
    /// subset only. One per list of pairs, the newest run of it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub subsets: Vec<LraSubset>,
}

impl LraFiles {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Record `subset`, in place of an earlier one at the same table. Its
    /// identity genes are left out when they are the full table's, which
    /// [`LraFiles::subset_options`] gives back.
    pub fn record_subset(&mut self, mut subset: LraSubset) {
        if let Some(o) = &mut subset.options {
            share_identity(o, self.options.as_ref());
        }
        match self.subsets.iter_mut().find(|s| s.table == subset.table) {
            Some(s) => *s = subset,
            None => self.subsets.push(subset),
        }
    }

    /// Record the full table's `options`. A subset inheriting the identity
    /// genes of the options replaced gets its own copy of them first, so it
    /// reads the same list as before; a subset whose own list is the new
    /// one's then drops its copy.
    pub fn record_options(&mut self, options: LraOptions) {
        let old = self.options.take();
        for o in self.subsets.iter_mut().filter_map(|s| s.options.as_mut()) {
            if let Some(full) = old.as_ref().filter(|f| inherits(o, f)) {
                o.identity_excluded = full.identity_excluded.clone();
            }
        }
        for o in self.subsets.iter_mut().filter_map(|s| s.options.as_mut()) {
            share_identity(o, Some(&options));
        }
        self.options = Some(options);
    }

    /// The options `subset` was made with, its identity genes the full
    /// table's when it records none of its own and decides its receivers
    /// as the full table does; `None` when it records no options.
    #[cfg(any(feature = "view", test))]
    pub fn subset_options(&self, subset: &LraSubset) -> Option<LraOptions> {
        let mut o = subset.options.clone()?;
        if let Some(full) = self.options.as_ref().filter(|f| inherits(&o, f)) {
            o.identity_excluded = full.identity_excluded.clone();
        }
        Some(o)
    }
}

/// Drop a subset's identity genes `o` records when they are those of the
/// full table's options `full` and it would inherit them.
fn share_identity(o: &mut LraOptions, full: Option<&LraOptions>) {
    if full.is_some_and(|f| o.identity_excluded == f.identity_excluded && o.same_receivers(f)) {
        o.identity_excluded.clear();
    }
}

/// Whether a subset's options `o` take the full table's identity genes:
/// they record none and decide the receivers as `full` does.
fn inherits(o: &LraOptions, full: &LraOptions) -> bool {
    o.identity_excluded.is_empty() && o.same_receivers(full)
}

impl LraOptions {
    /// Whether `self` and `other` decide the same receivers' identity: the
    /// same pairs, targets, species and seed (the FDR does not enter).
    pub fn same_receivers(&self, other: &Self) -> bool {
        self.lr_pairs == other.lr_pairs
            && self.targets == other.targets
            && self.species == other.species
            && self.seed == other.seed
    }
}

/// The choices a contact-effect table was made with, beyond the fixed
/// settings: what decides which pairs there are and which genes the
/// receivers' identity leaves out, so the viewer can draw a pair as the
/// table saw it. Paths are absolute.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct LraOptions {
    pub lr_pairs: Option<String>,
    pub targets: Vec<String>,
    /// `auto`, `human` or `mouse`.
    pub species: String,
    pub seed: u64,
    /// The genes the receivers' identity left out (every pair's genes),
    /// by name: the viewer reads the receivers with these
    /// rather than the files above, which may have moved. Thousands of
    /// names, kept on one line ([`comma_list`]). Empty in a manifest from
    /// before 0.16.4.
    #[serde(skip_serializing_if = "Vec::is_empty", with = "comma_list")]
    pub identity_excluded: Vec<String>,
    /// `--cell-types`: the cell-label file that stood in for the run's
    /// lupin round, when one was given.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_types: Option<String>,
    /// `--region`: the region-label file adjusted for and reported by.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// `--region-jackknife`: `refit` or `no-refit`.
    pub region_jackknife: String,
}

/// A list of names as one comma-separated string, so a long one is one
/// line of the manifest; a name holding a comma is kept whole by writing
/// the list as an array instead, which is read back too.
mod comma_list {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(names: &[String], s: S) -> Result<S::Ok, S::Error> {
        if names.iter().any(|n| n.contains(',')) {
            names.serialize(s)
        } else {
            s.serialize_str(&names.join(","))
        }
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Written {
        Line(String),
        List(Vec<String>),
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
        Ok(match Written::deserialize(d)? {
            Written::Line(line) => line
                .split(',')
                .filter(|n| !n.is_empty())
                .map(String::from)
                .collect(),
            Written::List(names) => names,
        })
    }
}

impl Default for LraOptions {
    /// `pinto lra`'s defaults.
    fn default() -> Self {
        Self {
            lr_pairs: None,
            targets: Vec::new(),
            species: crate::lr_activity::args::ReferenceSpecies::Auto.name(),
            seed: crate::lr_activity::args::DEFAULT_SEED,
            identity_excluded: Vec::new(),
            cell_types: None,
            region: None,
            region_jackknife: crate::lr_activity::args::RegionJackknife::default().name(),
        }
    }
}

/// A table of `pinto lra --only`: the pairs asked for, how many it holds,
/// and its files.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(default)]
pub struct LraSubset {
    /// The `--only` entries, as [`crate::lr_activity::ligand_effect::only`]
    /// writes them: trimmed, upper case, sorted.
    pub only: Vec<String>,
    /// Pairs estimated, out of the pairs the full analysis would have.
    pub pairs: usize,
    pub of: usize,
    pub table: String,
    pub curve: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<LraOptions>,
}
