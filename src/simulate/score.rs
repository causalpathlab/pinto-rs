//! `pinto srt-sim-score`: how well a labelling recovers a simulation's
//! planted truth, as two tests.
//!
//! **Test 1, donor-invariant domains** (on spots of shared domains):
//! - `shared_ari`: adjusted Rand index of the labels against the domains.
//! - `donor_dependence`: how much the labels depend on the donor within a
//!   domain, the conditional mutual information `I(label; donor | domain)`
//!   over `H(donor | domain)`. 0: labels donor-invariant; 1: labels set by
//!   the donor. No threshold; a domain that sits in one donor, or a
//!   single-donor panel, adds nothing. `donor_dependence_by_domain` gives
//!   the same ratio for each shared domain on its own (`None` for a domain
//!   in one donor).
//!
//! **Test 2, donor-specific domains**, each against the label that matches
//! it best (highest F1): that label's `precision` (low: shared spots
//! absorbed into it), `recall`, and `owner_share`, the share of its spots
//! from the owning donor (1: confined to its owner).
//!
//! A labelling is any pinto run at any level (its propensity's hard labels),
//! or a table (TSV or parquet) with a `cell` column and a label column, as
//! lupin writes them. Empty, `NA`, `-1` and `unassigned` labels are none;
//! spots left unlabelled are not scored and are counted. Label rows naming
//! cells the truth lacks are counted too (`n_unmatched`) and warned of.

use super::panel::{donor_label, parse_donor_label};
use crate::compare::{ari, propensity_path, Partition, Table};
use crate::util::common::HashMap;
use legume_numeric::matrix::parquet::read_parquet_string_columns_by_name;
use legume_numeric::matrix::table::TableReader;
use serde::Serialize;

/// The planted truth of each spot.
pub struct TruthSpots {
    pub cells: Vec<Box<str>>,
    /// Index into `domain_labels`.
    pub domain: Vec<usize>,
    pub donor: Vec<usize>,
    pub domain_labels: Vec<Box<str>>,
    /// `None` for a shared domain, the owner donor otherwise.
    pub domain_owner: Vec<Option<usize>>,
    pub n_donors: usize,
}

/// The simulation `truth` names: its out folder, its `truth` folder, or the
/// truth run in it, by its manifest `truth/truth.pinto.json` or its prefix
/// `truth/truth`.
pub(crate) fn sim_folder(truth: &str) -> String {
    use crate::util::metadata::{run_manifest, run_prefix};
    use std::path::Path;
    let prefix = Path::new(run_prefix(truth));
    let folder = if prefix.ends_with("truth/truth") && Path::new(&run_manifest(truth)).is_file() {
        prefix.parent().and_then(Path::parent)
    } else if prefix.ends_with("truth") && !prefix.join("truth").is_dir() {
        prefix.parent()
    } else {
        return truth.into();
    };
    match folder {
        Some(p) if p.as_os_str().is_empty() => ".".into(),
        Some(p) => p.to_string_lossy().into_owned(),
        None => truth.into(),
    }
}

impl TruthSpots {
    /// From a simulation's `{out}/truth/spots.parquet` and `domains.parquet`;
    /// `out` may also be given as its truth folder or truth run.
    pub fn read(out: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(!out.trim().is_empty(), "no truth given");
        let out = &sim_folder(out);
        let dom = read_parquet_string_columns_by_name(
            &super::run::domains_path(out),
            &["domain", "owner_donor"],
        )?;
        let donor_index = |d: &str| -> anyhow::Result<usize> {
            parse_donor_label(d)
                .ok_or_else(|| anyhow::anyhow!("{out}: donor {d:?} is not D1, D2, ..."))
        };
        let domain_labels = dom[0].clone();
        let domain_owner = dom[1]
            .iter()
            .map(|o| (!o.is_empty()).then(|| donor_index(o)).transpose())
            .collect::<anyhow::Result<Vec<_>>>()?;
        let spots = read_parquet_string_columns_by_name(
            &format!("{out}/truth/spots.parquet"),
            &["cell", "donor", "domain"],
        )?;
        let of_domain: HashMap<&str, usize> = domain_labels
            .iter()
            .enumerate()
            .map(|(k, l)| (l.as_ref(), k))
            .collect();
        let domain = spots[2]
            .iter()
            .map(|l| {
                of_domain
                    .get(l.as_ref())
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("{out}: domain {l} not in domains.parquet"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let donor = spots[1]
            .iter()
            .map(|d| donor_index(d))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let n_donors = donor.iter().max().map_or(0, |&m| m + 1);
        anyhow::ensure!(
            domain_owner.iter().flatten().all(|&o| o < n_donors),
            "{out}: domains.parquet names an owner donor with no spots in spots.parquet"
        );
        Ok(Self {
            cells: spots[0].clone(),
            domain,
            donor,
            domain_labels,
            domain_owner,
            n_donors,
        })
    }
}

#[derive(Serialize, Debug)]
pub struct PatchScore {
    pub domain: String,
    pub owner: String,
    pub f1: f64,
    pub precision: f64,
    pub recall: f64,
    pub owner_share: f64,
}

/// One shared domain's donor dependence, `I(label; donor)` over
/// `H(donor)` on its spots; `None` when its spots sit in one donor.
#[derive(Serialize, Debug)]
pub struct DomainDependence {
    pub domain: String,
    pub dependence: Option<f64>,
}

#[derive(Serialize, Debug)]
pub struct Score {
    pub n_spots: usize,
    pub n_scored: usize,
    /// Label rows naming a cell the truth lacks.
    pub n_unmatched: usize,
    pub shared_ari: f64,
    pub donor_dependence: f64,
    pub donor_dependence_by_domain: Vec<DomainDependence>,
    pub patches: Vec<PatchScore>,
}

impl Score {
    /// What to warn of when label rows name cells the truth lacks: every
    /// simulation names its cells alike, so a labelling of another one
    /// still matches some of them.
    pub fn unmatched_warning(&self) -> Option<String> {
        (self.n_unmatched > 0).then(|| {
            format!(
                "{} label rows name cells not in the truth: is it a labelling of \
                 another simulation?",
                self.n_unmatched
            )
        })
    }
}

/// `counts[k][d]`: label → spots of domain `k` and donor `d`.
type Counts = Vec<Vec<HashMap<i64, usize>>>;

/// Sum of `x ln x` over the positive counts.
fn xlogx<'a>(v: impl Iterator<Item = &'a usize>) -> f64 {
    v.filter(|&&x| x > 0)
        .map(|&x| x as f64 * (x as f64).ln())
        .sum()
}

/// Score `labels` (cell, label; a negative label is none) against `t`.
pub fn score(t: &TruthSpots, labels: &[(Box<str>, i64)]) -> anyhow::Result<Score> {
    let mut given: HashMap<&str, i64> = HashMap::default();
    for (c, l) in labels {
        anyhow::ensure!(
            given.insert(c.as_ref(), *l).is_none(),
            "the labelling names cell {c} twice"
        );
    }
    let mut counts: Counts = vec![vec![HashMap::default(); t.n_donors]; t.domain_labels.len()];
    let (mut n_scored, mut n_found) = (0, 0);
    for (i, c) in t.cells.iter().enumerate() {
        let Some(&l) = given.get(c.as_ref()) else {
            continue;
        };
        n_found += 1;
        if l >= 0 {
            *counts[t.domain[i]][t.donor[i]].entry(l).or_insert(0) += 1;
            n_scored += 1;
        }
    }
    let n_unmatched = given.len().saturating_sub(n_found);
    anyhow::ensure!(
        n_scored > 0,
        "no spot of the truth is labelled: are the cell names spelled alike?"
    );
    let (shared_ari, donor_dependence, donor_dependence_by_domain) = invariant(t, &counts);
    Ok(Score {
        n_spots: t.cells.len(),
        n_scored,
        n_unmatched,
        shared_ari,
        donor_dependence,
        donor_dependence_by_domain,
        patches: patches(t, &counts),
    })
}

/// Information over entropy, in [0, 1]; `None` without entropy.
fn ratio(info: f64, entropy: f64) -> Option<f64> {
    (entropy > 1e-12).then(|| (info / entropy).clamp(0.0, 1.0))
}

/// Test 1, over the shared domains: ARI, donor dependence pooled and by
/// domain.
fn invariant(t: &TruthSpots, counts: &Counts) -> (f64, f64, Vec<DomainDependence>) {
    let shared: Vec<usize> = (0..counts.len())
        .filter(|&k| t.domain_owner[k].is_none())
        .collect();
    // domain × label, summed over donors
    let mut labels: Vec<i64> = shared
        .iter()
        .flat_map(|&k| counts[k].iter().flat_map(|m| m.keys().copied()))
        .collect();
    labels.sort_unstable();
    labels.dedup();
    let at: HashMap<i64, usize> = labels.iter().enumerate().map(|(j, &l)| (l, j)).collect();
    let mut n = vec![vec![0usize; labels.len()]; shared.len()];
    for (r, &k) in shared.iter().enumerate() {
        for (&l, &x) in counts[k].iter().flatten() {
            n[r][at[&l]] += x;
        }
    }
    let total: usize = n.iter().flatten().sum();
    let ari_value = if total > 0 {
        ari(&Table {
            a: shared.iter().map(|&k| k as i64).collect(),
            b: labels,
            n,
            n_only_a: 0,
            n_only_b: 0,
        })
    } else {
        f64::NAN
    };

    // I(label; donor | domain) and H(donor | domain), in nats, summed over
    // shared domains weighted by their spots:
    // N_k I_k = Σ x ln x (label, donor) − Σ x ln x (donor) − Σ x ln x (label) + N_k ln N_k
    let (mut info, mut entropy) = (0.0, 0.0);
    let mut by_domain = Vec::with_capacity(shared.len());
    for &k in &shared {
        let per_donor: Vec<usize> = counts[k].iter().map(|m| m.values().sum()).collect();
        let mut per_label: HashMap<i64, usize> = HashMap::default();
        counts[k]
            .iter()
            .flatten()
            .for_each(|(&l, &x)| *per_label.entry(l).or_insert(0) += x);
        let nk: usize = per_donor.iter().sum();
        let (mut ik, mut hk) = (0.0, 0.0);
        if nk > 0 {
            let joint = xlogx(counts[k].iter().flat_map(|m| m.values()));
            let nk_ln = nk as f64 * (nk as f64).ln();
            ik = joint - xlogx(per_donor.iter()) - xlogx(per_label.values()) + nk_ln;
            hk = nk_ln - xlogx(per_donor.iter());
        }
        info += ik;
        entropy += hk;
        by_domain.push(DomainDependence {
            domain: t.domain_labels[k].to_string(),
            dependence: ratio(ik, hk),
        });
    }
    let dependence = ratio(info, entropy).unwrap_or(0.0);
    (ari_value, dependence, by_domain)
}

/// Test 2: each donor-specific domain against its best-matching label.
fn patches(t: &TruthSpots, counts: &Counts) -> Vec<PatchScore> {
    // label → spots per donor, over every domain
    let mut by_label: HashMap<i64, Vec<usize>> = HashMap::default();
    for (d, m) in counts
        .iter()
        .flat_map(|per_donor| per_donor.iter().enumerate())
    {
        for (&l, &x) in m {
            by_label.entry(l).or_insert_with(|| vec![0; t.n_donors])[d] += x;
        }
    }
    (0..counts.len())
        .filter_map(|k| t.domain_owner[k].map(|o| (k, o)))
        .map(|(k, owner)| {
            let mut overlap: HashMap<i64, usize> = HashMap::default();
            counts[k]
                .iter()
                .flatten()
                .for_each(|(&l, &x)| *overlap.entry(l).or_insert(0) += x);
            let n_patch: usize = overlap.values().sum();
            let size = |l: i64| by_label[&l].iter().sum::<usize>();
            // F1 = 2 tp / (|label| + |patch|); ties to the larger overlap, then the smaller label
            let best = overlap.iter().max_by(|a, b| {
                let f = |(&l, &tp): (&i64, &usize)| tp as f64 / (size(l) + n_patch) as f64;
                f(*a)
                    .total_cmp(&f(*b))
                    .then(a.1.cmp(b.1))
                    .then(b.0.cmp(a.0))
            });
            let (f1, precision, recall, owner_share) =
                best.map_or((0.0, 0.0, 0.0, 0.0), |(&l, &tp)| {
                    let s = size(l) as f64;
                    let (p, r) = (tp as f64 / s, tp as f64 / n_patch as f64);
                    (
                        2.0 * tp as f64 / (s + n_patch as f64),
                        p,
                        r,
                        by_label[&l][owner] as f64 / s,
                    )
                });
            PatchScore {
                domain: t.domain_labels[k].to_string(),
                owner: donor_label(owner).to_string(),
                f1,
                precision,
                recall,
                owner_share,
            }
        })
        .collect()
}

/// Labels a table names no cell type: none.
fn is_none_label(label: &str) -> bool {
    matches!(label, "" | "NA" | "-1") || label.eq_ignore_ascii_case("unassigned")
}

/// Whether `input` names a table (TSV or parquet, a propensity table
/// aside) rather than a run.
pub(crate) fn is_table(input: &str) -> bool {
    [".tsv", ".txt", ".tsv.gz"]
        .iter()
        .any(|e| input.ends_with(e))
        || (input.ends_with(".parquet") && !input.ends_with(".propensity.parquet"))
}

/// Hard labels of a run at `level`, or of a table's `cell` and
/// `label_column` columns (TSV or parquet).
pub fn read_labels(
    input: &str,
    level: &str,
    label_column: &str,
) -> anyhow::Result<Vec<(Box<str>, i64)>> {
    if is_table(input) {
        let t = TableReader::open(input)?;
        let cols = t.select(&["cell", label_column])?;
        let mut ids: HashMap<Box<str>, i64> = HashMap::default();
        let mut out = Vec::new();
        for row in t.rows(&cols)? {
            let row = row?;
            let label = row[1].trim();
            let id = if is_none_label(label) {
                -1
            } else {
                let n = ids.len() as i64;
                *ids.entry(label.into()).or_insert(n)
            };
            out.push((row[0].trim().into(), id));
        }
        return Ok(out);
    }
    let p = Partition::read(&propensity_path(input, level)?)?;
    Ok(p.cells.into_iter().zip(p.labels).collect())
}

#[derive(clap::Args, Debug)]
pub struct SimScoreArgs {
    /// The simulation's output directory (as given to srt-simulate -o)
    #[arg(long)]
    pub truth: Box<str>,

    /// Labellings to score: pinto runs (prefix or .pinto.json), or tables
    /// (TSV or parquet) with a `cell` column, comma-separated or repeated
    #[arg(long, value_delimiter = ',', required = true)]
    pub labels: Vec<Box<str>>,

    /// Level of a run to score
    #[arg(long, default_value = "final")]
    pub level: String,

    /// Label column of a table (lupin's rounds: cell_type)
    #[arg(long, default_value = "label")]
    pub label_column: String,

    /// Write the scores as JSON here
    #[arg(short, long)]
    pub out: Option<Box<str>>,
}

/// The table's header line, as `srt-sim-score` prints it.
pub fn table_header() -> String {
    format!(
        "{:<40} {:>7} {:>10} {:>10} {:>6} {:>6} {:>6} {:>6}  {}",
        "labels",
        "scored",
        "shared_ari",
        "donor_dep",
        "patch",
        "f1",
        "prec",
        "owner",
        "donor_dep_by_domain"
    )
}

/// An input as the tables show it: its file and folder.
pub(crate) fn shown_name(input: &str) -> String {
    let path = std::path::Path::new(input);
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|p| p.to_string_lossy());
    let file = path
        .file_name()
        .map_or_else(|| input.to_string(), |f| f.to_string_lossy().to_string());
    parent.map_or(file.clone(), |p| format!("{p}/{file}"))
}

/// Each shared domain's donor dependence, as `A:0.012 B:-` (`-`: the
/// domain sits in one donor).
fn by_domain_cell(s: &Score) -> String {
    s.donor_dependence_by_domain
        .iter()
        .map(|d| match d.dependence {
            Some(x) => format!("{}:{x:.3}", d.domain),
            None => format!("{}:-", d.domain),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A labelling's rows of the table: one per donor-specific domain, or one
/// when there is none. The labelling goes by its file and folder.
pub fn table_rows(input: &str, s: &Score) -> Vec<String> {
    let head = format!(
        "{:<40} {:>7} {:>10.3} {:>10.3}",
        shown_name(input),
        s.n_scored,
        s.shared_ari,
        s.donor_dependence
    );
    let by_domain = by_domain_cell(s);
    if s.patches.is_empty() {
        return vec![format!(
            "{head} {:>6} {:>6} {:>6} {:>6}  {by_domain}",
            "", "", "", ""
        )];
    }
    s.patches
        .iter()
        .map(|p| {
            format!(
                "{head} {:>6} {:>6.3} {:>6.3} {:>6.3}  {by_domain}",
                p.domain, p.f1, p.precision, p.owner_share
            )
        })
        .collect()
}

pub fn run_score(args: &SimScoreArgs) -> anyhow::Result<()> {
    let t = TruthSpots::read(&args.truth)?;
    let mut all = Vec::new();
    println!("{}", table_header());
    let mut failed = 0;
    for input in &args.labels {
        let s =
            match read_labels(input, &args.level, &args.label_column).and_then(|l| score(&t, &l)) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{input}: {e:#}");
                    failed += 1;
                    continue;
                }
            };
        if let Some(w) = s.unmatched_warning() {
            log::warn!("{input}: {w}");
        }
        for row in table_rows(input, &s) {
            println!("{row}");
        }
        all.push(serde_json::json!({ "labels": input.as_ref(), "level": args.level, "score": s }));
    }
    if let Some(out) = &args.out {
        std::fs::write(out.as_ref(), serde_json::to_string_pretty(&all)?)?;
    }
    anyhow::ensure!(failed == 0, "{failed} labelling(s) could not be scored");
    Ok(())
}
