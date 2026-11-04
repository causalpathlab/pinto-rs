//! The public resources behind the reference, and how each is read.
//!
//! - LIANA+ `omni_resource.csv` at a pinned commit: the `consensus`
//!   (human) and `mouseconsensus` (mouse) ligand–receptor pairs.
//! - OmniPath `/resources`: each resource's licence.
//! - OmniPath `/interactions` (signalling, ligrecextra, CollecTRI), human,
//!   every row with its supporting resources (`license=ignore`); the
//!   licence filter is applied here, row by row.
//! - HCOP human–mouse orthologs, kept one-to-one.

use super::download::Source;
use super::table::{field, Table};
use super::LicenceMode;
use anyhow::{Context, Result};
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};

// * Downloads

pub const LIANA_COMMIT: &str = "bf70ace969ffb0155ad075f6737bad791f01818b";
const LIANA_SHA256: &str = "65e46a428818c0600de66c1574c94ff2f5dc80859998321c03a956d330dd67fb";
const OMNIPATH: &str = "https://omnipathdb.org";
const IX_FIELDS: &str =
    "genesymbols=yes&fields=sources,type,entity_type&license=ignore&organisms=9606";

pub const LIANA: &str = "liana_omni_resource.csv";
pub const RESOURCES: &str = "omnipath_resources.json";
pub const SIGNALLING: &str = "omnipath_signalling.9606.tsv";
pub const LIGREC: &str = "omnipath_ligrecextra.9606.tsv";
pub const COLLECTRI: &str = "collectri.9606.tsv";
pub const HCOP: &str = "hcop_human_mouse.txt.gz";
pub const INTERCELL: &str = "omnipath_intercell_ligands.tsv";

/// The files the reference is built from.
pub fn downloads() -> Vec<Source> {
    let ix = |dataset: &str| format!("{OMNIPATH}/interactions?datasets={dataset}&{IX_FIELDS}");
    vec![
        Source {
            name: LIANA,
            url: format!(
                "https://raw.githubusercontent.com/scverse/liana/{LIANA_COMMIT}/src/liana/resource/omni_resource.csv"
            ),
            sha256: Some(LIANA_SHA256),
        },
        Source {
            name: RESOURCES,
            url: format!("{OMNIPATH}/resources"),
            sha256: None,
        },
        Source {
            name: SIGNALLING,
            url: ix("omnipath"),
            sha256: None,
        },
        Source {
            name: LIGREC,
            url: ix("ligrecextra"),
            sha256: None,
        },
        Source {
            name: COLLECTRI,
            url: ix("collectri"),
            sha256: None,
        },
        Source {
            name: INTERCELL,
            url: format!(
                "{OMNIPATH}/intercell?categories=ligand&scope=generic&aspect=functional\
                 &source=composite&format=tsv"
            ),
            sha256: None,
        },
        Source {
            name: HCOP,
            url: "https://storage.googleapis.com/public-download-files/hcop/human_mouse_hcop_fifteen_column.txt.gz"
                .into(),
            sha256: None,
        },
    ]
}

// * Licences

/// Source labels in OmniPath interaction rows that have no entry of their
/// own in `/resources`: the entry whose licence applies.
const LICENCE_ALIASES: &[(&str, &str)] = &[
    ("NTNUcuration_CollecTRI2", "NTNU.Curated_CollecTRI"),
    ("TfactS_CollecTRI2", "TFactS_CollecTRI"),
    ("Pavlidis_CollecTRI2", "Pavlidis2021_CollecTRI"),
    ("HTRIdb_CollecTRI2", "HTRI_CollecTRI"),
    ("ExTRI2_CollecTRI2", "ExTRI_CollecTRI"),
    ("hprd_ProtMapper", "HPRD"),
    ("CollecTRI2", "CollecTRI"),
];

/// OmniPath's licence table: resource → (purpose, sharing).
pub struct Licences(FxHashMap<String, (String, String)>);

impl Licences {
    /// From the `/resources` JSON.
    pub fn from_json(text: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(text).context("OmniPath /resources is not JSON")?;
        let obj = v
            .as_object()
            .context("OmniPath /resources is not a JSON object")?;
        let get = |lic: Option<&serde_json::Value>, key: &str| {
            lic.and_then(|l| l.get(key))
                .and_then(|x| x.as_str())
                .unwrap_or("?")
                .to_string()
        };
        Ok(Licences(
            obj.iter()
                .map(|(k, v)| {
                    let lic = v.get("license").filter(|l| l.is_object());
                    (k.clone(), (get(lic, "purpose"), get(lic, "sharing")))
                })
                .collect(),
        ))
    }

    /// The entry that applies to source label `name`, through the aliases
    /// and the `*_CollecTRI2` → `*_CollecTRI` rule.
    pub fn resolve(&self, name: &str) -> Option<&(String, String)> {
        if let Some(e) = self.0.get(name) {
            return Some(e);
        }
        let alias = LICENCE_ALIASES
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| (*v).to_string())
            .or_else(|| {
                name.strip_suffix('2')
                    .filter(|_| name.ends_with("_CollecTRI2"))
                    .map(String::from)
            })?;
        self.0.get(&alias)
    }

    /// Usable commercially, with no share-alike or no-derivatives terms.
    pub fn is_permissive(&self, name: &str) -> bool {
        self.resolve(name).is_some_and(|(purpose, sharing)| {
            (purpose == "commercial" || purpose == "for_profit") && sharing == "free"
        })
    }

    /// The supporting resources of a row (`;`-separated) that count under
    /// `mode`, sorted and unique, or `None` to drop the row.
    pub fn keep_sources<'a>(&self, sources: &'a str, mode: LicenceMode) -> Option<Vec<&'a str>> {
        let srcs: BTreeSet<&str> = sources.split(';').filter(|s| !s.is_empty()).collect();
        match mode {
            LicenceMode::All => Some(srcs.into_iter().collect()),
            LicenceMode::Permissive => {
                let ok: Vec<&str> = srcs.into_iter().filter(|s| self.is_permissive(s)).collect();
                (!ok.is_empty()).then_some(ok)
            }
        }
    }
}

// * OmniPath interactions

/// One OmniPath interaction row (the columns the build reads).
pub struct Interaction {
    pub source: Box<str>,
    pub target: Box<str>,
    pub source_genesymbol: Box<str>,
    pub target_genesymbol: Box<str>,
    pub is_directed: bool,
    pub consensus_direction: bool,
    pub consensus_stimulation: bool,
    pub consensus_inhibition: bool,
    pub sources: Box<str>,
    pub entity_type_source: Box<str>,
    pub entity_type_target: Box<str>,
}

impl Interaction {
    /// +1 stimulation only, −1 inhibition only, else 0.
    pub fn sign(&self) -> i8 {
        match (self.consensus_stimulation, self.consensus_inhibition) {
            (true, false) => 1,
            (false, true) => -1,
            _ => 0,
        }
    }
}

/// Rows of an OmniPath `/interactions` TSV.
pub fn parse_interactions(text: &str) -> Result<Vec<Interaction>> {
    let t = Table::parse(text, '\t')?;
    let c = |n: &str| t.col(n);
    let (src, tgt, sg, tg) = (
        c("source")?,
        c("target")?,
        c("source_genesymbol")?,
        c("target_genesymbol")?,
    );
    let (dir, cdir, cstim, cinh) = (
        c("is_directed")?,
        c("consensus_direction")?,
        c("consensus_stimulation")?,
        c("consensus_inhibition")?,
    );
    let (srcs, ets, ett) = (
        c("sources")?,
        c("entity_type_source")?,
        c("entity_type_target")?,
    );
    Ok(t.rows
        .iter()
        .map(|r| {
            let s = |i| field(r, i).into();
            let b = |i| field(r, i) == "True";
            Interaction {
                source: s(src),
                target: s(tgt),
                source_genesymbol: s(sg),
                target_genesymbol: s(tg),
                is_directed: b(dir),
                consensus_direction: b(cdir),
                consensus_stimulation: b(cstim),
                consensus_inhibition: b(cinh),
                sources: s(srcs),
                entity_type_source: s(ets),
                entity_type_target: s(ett),
            }
        })
        .collect())
}

/// Supporting-resource labels with no licence entry, with row counts.
pub fn unknown_sources<'a>(
    rows: impl Iterator<Item = &'a Interaction>,
    lic: &Licences,
) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for r in rows {
        for s in r.sources.split(';').filter(|s| !s.is_empty()) {
            if lic.resolve(s).is_none() {
                *out.entry(s.to_string()).or_insert(0) += 1;
            }
        }
    }
    out
}

// * LIANA+ ligand-receptor pairs

/// One row of LIANA+'s `omni_resource.csv`.
pub struct LianaRow {
    pub source: Box<str>,
    pub target: Box<str>,
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub resource: Box<str>,
}

/// Rows of LIANA+'s `omni_resource.csv`.
pub fn parse_liana(text: &str) -> Result<Vec<LianaRow>> {
    let t = Table::parse(text, ',')?;
    let (src, tgt) = (t.col("source")?, t.col("target")?);
    let (lig, rec, res) = (
        t.col("source_genesymbol")?,
        t.col("target_genesymbol")?,
        t.col("resource")?,
    );
    Ok(t.rows
        .iter()
        .map(|r| LianaRow {
            source: field(r, src).into(),
            target: field(r, tgt).into(),
            ligand: field(r, lig).into(),
            receptor: field(r, rec).into(),
            resource: field(r, res).into(),
        })
        .collect())
}

/// OmniPath support for LIANA pairs: (source id, target id) → resources
/// kept under the licence mode. LIANA's mouse resource keeps the human
/// ids, so one table serves both species.
pub struct Provenance {
    mode: LicenceMode,
    support: Support,
}

type Support = FxHashMap<(Box<str>, Box<str>), BTreeSet<Box<str>>>;

impl Provenance {
    pub fn new<'a>(
        rows: impl Iterator<Item = &'a Interaction>,
        lic: &Licences,
        mode: LicenceMode,
    ) -> Self {
        let mut support = Support::default();
        for r in rows {
            if let Some(keep) = lic.keep_sources(&r.sources, mode) {
                support
                    .entry((r.source.clone(), r.target.clone()))
                    .or_default()
                    .extend(keep.into_iter().map(Box::from));
            }
        }
        Provenance { mode, support }
    }
}

/// A ligand–receptor pair and the resources supporting it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LrPair {
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    pub sources: String,
}

impl LrPair {
    /// The output row: names, sources, then the names again as subunits.
    pub fn row(&self) -> [String; 5] {
        [
            self.ligand.to_string(),
            self.receptor.to_string(),
            self.sources.clone(),
            self.ligand.to_string(),
            self.receptor.to_string(),
        ]
    }
}

/// The pairs of LIANA resource `which`, sorted by (ligand, receptor), and
/// the number of its rows without OmniPath support (dropped under
/// `Permissive`, credited to `LIANA_consensus` under `All`).
pub fn lr_pairs(liana: &[LianaRow], prov: &Provenance, which: &str) -> (Vec<LrPair>, usize) {
    let mut out: BTreeMap<(&str, &str), BTreeSet<&str>> = BTreeMap::new();
    let mut unsupported = 0;
    for r in liana.iter().filter(|r| &*r.resource == which) {
        let key = (r.source.clone(), r.target.clone());
        let entry = match prov.support.get(&key) {
            Some(srcs) => srcs.iter().map(|s| &**s).collect::<Vec<_>>(),
            None => {
                unsupported += 1;
                if prov.mode == LicenceMode::Permissive {
                    continue;
                }
                vec!["LIANA_consensus"]
            }
        };
        out.entry((&r.ligand, &r.receptor))
            .or_default()
            .extend(entry);
    }
    let pairs = out
        .into_iter()
        .map(|((lig, rec), srcs)| LrPair {
            ligand: lig.into(),
            receptor: rec.into(),
            sources: srcs.into_iter().collect::<Vec<_>>().join(";"),
        })
        .collect();
    (pairs, unsupported)
}

// * Ligand roles

/// Human genes OmniPath's intercellular annotation calls ligands (generic,
/// functional, composite): secreted or membrane-bound transmitters. A
/// complex counts each of its subunits.
pub fn parse_intercell_ligands(text: &str) -> Result<BTreeSet<Box<str>>> {
    let t = Table::parse(text, '\t').context("intercell table")?;
    let g = t.col("genesymbol")?;
    let mut out = BTreeSet::new();
    for row in &t.rows {
        let name = field(row, g);
        let name = name.strip_prefix("COMPLEX:").unwrap_or(name);
        for part in name.split('_').filter(|s| !s.is_empty()) {
            out.insert(part.to_ascii_uppercase().into_boxed_str());
        }
    }
    anyhow::ensure!(!out.is_empty(), "the intercell table names no ligand");
    Ok(out)
}

/// Pairs whose every ligand subunit is an annotated ligand, and the number
/// dropped. Consensus resources carry intracellular proteins as "ligands"
/// (calmodulin, G-protein subunits, enzymes standing in for a metabolite);
/// they cannot reach a neighbour, so a pair on them has no contact
/// estimand. `to_human` maps a mouse symbol to its human ortholog (identity
/// for human pairs); a symbol without one is compared upper-cased.
pub fn transmitted_pairs(
    pairs: Vec<LrPair>,
    ligands: &BTreeSet<Box<str>>,
    to_human: &dyn Fn(&str) -> String,
) -> (Vec<LrPair>, usize) {
    let before = pairs.len();
    let kept: Vec<LrPair> = pairs
        .into_iter()
        .filter(|p| {
            p.ligand
                .split('_')
                .filter(|s| !s.is_empty())
                .all(|s| ligands.contains(to_human(s).to_ascii_uppercase().as_str()))
        })
        .collect();
    let dropped = before - kept.len();
    (kept, dropped)
}

// * HCOP orthologs

/// Human → mouse symbols from HCOP's fifteen-column table: pairs with at
/// least `min_evidence` supporting resources, kept where both genes have
/// exactly one partner.
pub fn hcop_one_to_one(text: &str, min_evidence: usize) -> Result<FxHashMap<Box<str>, Box<str>>> {
    let t = Table::parse(text, '\t')?;
    let (hs, ms, sup) = (
        t.col("human_symbol")?,
        t.col("mouse_symbol")?,
        t.col("support")?,
    );
    let mut pairs: BTreeSet<(&str, &str)> = BTreeSet::new();
    for r in &t.rows {
        let (h, m) = (field(r, hs), field(r, ms));
        if h.is_empty() || h == "-" || m.is_empty() || m == "-" {
            continue;
        }
        if field(r, sup).split(',').count() < min_evidence {
            continue;
        }
        pairs.insert((h, m));
    }
    let mut hc: FxHashMap<&str, usize> = FxHashMap::default();
    let mut mc: FxHashMap<&str, usize> = FxHashMap::default();
    for &(h, m) in &pairs {
        *hc.entry(h).or_default() += 1;
        *mc.entry(m).or_default() += 1;
    }
    Ok(pairs
        .into_iter()
        .filter(|(h, m)| hc[h] == 1 && mc[m] == 1)
        .map(|(h, m)| (h.into(), m.into()))
        .collect())
}
