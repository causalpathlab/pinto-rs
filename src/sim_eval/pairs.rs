//! `pinto sim score-by-pair`: how well per-cell vectors (a latent mapping)
//! recover a simulation's planted truth, scored on pairs of spots, with no
//! clustering step.
//!
//! Two spots' similarity is the cosine of their vectors. Each set of pairs
//! has a planted answer (positive or not), and the similarity is scored on
//! it by AUROC and AUPRC, reported with the set's prevalence (its share of
//! positives, the AUPRC of a random ranking):
//!
//! - `shared_domain_across_donors`: random pairs of shared-domain spots
//!   from two donors; positive when the two share a domain.
//! - `shared_domain_within_donor`: the same within one donor, the reference
//!   (pairs drawn from the pooled spots, so a donor weighs as its spot count
//!   squared); `donor_gap` is its AUROC less the across-donors one, what the
//!   donor costs the mapping.
//! - `patches`, one per donor-specific domain, `donor_domain_vs_own_donor`:
//!   pairs of its spots in its owner (positive) against pairs of one of its
//!   spots and a spot of the same donor outside it (negative, `negatives:
//!   "same_donor_outside_domain"`), half and half. The negatives stay in the
//!   owner, so a mapping that only tells the donors apart scores 0.5.
//! - `neighbour_same_domain`: every edge of the `knn` nearest neighbours of
//!   each spot in its own section (Euclidean on `row_um`, `col_um`);
//!   positive when the two share a domain.
//!
//! The sampled sets are drawn from a seeded generator, so a score is
//! reproducible. Vectors are a pinto run's propensity at any level, a cage
//! run's `cell_embedding` table, or any table (TSV or parquet) with a
//! `cell` column and numeric columns.

use super::score::{sim_folder, TruthSpots, NAME_WIDTH};
use crate::compare::propensity_path;
use crate::simulate::panel::donor_label;
use crate::util::common::{in_dir, HashMap, HashSet, Mat};
use crate::util::parquet_io::{read_keyed_columns, read_propensity};
use legume_numeric::matrix::knn_graph::{KnnGraph, KnnGraphArgs};
use legume_numeric::matrix::parquet::read_parquet_string_columns_by_name;
use legume_numeric::matrix::rand_util::mix_seed;
use legume_numeric::matrix::table::TableReader;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde::Serialize;
use std::path::Path;

/// Where each truth spot is: its section and its coordinates, in the
/// order of [`TruthSpots::cells`].
pub struct SpotLayout {
    pub section: Vec<usize>,
    pub xy: Vec<[f32; 2]>,
}

impl SpotLayout {
    /// From the simulation's `truth/spots.parquet` (`out` named as
    /// [`TruthSpots::read`] takes it), checked against `t`.
    pub fn read(out: &str, t: &TruthSpots) -> anyhow::Result<Self> {
        let path = in_dir(&sim_folder(out), "truth/spots.parquet");
        let (cells, xy) = read_keyed_columns(Path::new(&path), &["row_um", "col_um"])?;
        anyhow::ensure!(
            cells == t.cells,
            "{path}: its cells are not the truth's, in its order"
        );
        let names = read_parquet_string_columns_by_name(&path, &["section"])?;
        let mut ids: HashMap<&str, usize> = HashMap::default();
        let section = names[0]
            .iter()
            .map(|s| {
                let n = ids.len();
                *ids.entry(s.as_ref()).or_insert(n)
            })
            .collect();
        Ok(Self {
            section,
            xy: xy[0].iter().zip(&xy[1]).map(|(&r, &c)| [r, c]).collect(),
        })
    }
}

/// `srt-sim-pairs`'s defaults: pairs per sampled set, seed, spatial
/// neighbours.
pub const DEFAULT_PAIRS: usize = 20_000;
pub const DEFAULT_SEED: u64 = 1;
pub const DEFAULT_KNN: usize = 6;

/// How the pairs are drawn.
#[derive(Clone, Debug)]
pub struct PairOptions {
    /// Pairs drawn for each sampled set.
    pub pairs: usize,
    pub seed: u64,
    /// Spatial neighbours of each spot.
    pub knn: usize,
}

impl PairOptions {
    /// Checked: two pairs and one neighbour at least.
    pub fn new(pairs: usize, seed: u64, knn: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(pairs >= 2, "pairs must be at least 2");
        anyhow::ensure!(knn >= 1, "knn must be at least 1");
        Ok(Self { pairs, seed, knn })
    }
}

/// One set of pairs, scored.
#[derive(Serialize, Debug, Clone)]
pub struct PairSet {
    pub n_pairs: usize,
    pub n_positive: usize,
    pub prevalence: f64,
    pub auroc: f64,
    pub auprc: f64,
    /// How the negatives were drawn, where a set states it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negatives: Option<&'static str>,
}

/// The negatives of a patch's `donor_domain_vs_own_donor`: one of its
/// spots with a spot of its owner outside it.
pub const PATCH_NEGATIVES: &str = "same_donor_outside_domain";

#[derive(Serialize, Debug)]
pub struct PatchPairs {
    pub domain: String,
    pub owner: String,
    /// The domain's spots in its owner that have a vector.
    pub n_spots: usize,
    /// Its own pairs (positive) against one of its spots with a spot of its
    /// owner outside it ([`PATCH_NEGATIVES`]). Its AUROC and AUPRC are
    /// null, with a `note`, when either side has no pair.
    pub donor_domain_vs_own_donor: PairSet,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<&'static str>,
}

/// Why a patch has no score: fewer than two of its spots have a vector.
pub const NO_POSITIVES: &str = "fewer than two of its spots have a vector: no positive pairs";
/// Why a patch has no score: no spot of its owner outside it has one.
pub const NO_NEGATIVES: &str = "no spot of its owner outside it has a vector: no negative pairs";

#[derive(Serialize, Debug)]
pub struct PairScore {
    pub n_spots: usize,
    pub n_scored: usize,
    pub shared_domain_across_donors: PairSet,
    pub shared_domain_within_donor: PairSet,
    /// `shared_domain_within_donor` AUROC less the across-donors one.
    pub donor_gap: f64,
    pub patches: Vec<PatchPairs>,
    pub neighbour_same_domain: PairSet,
}

/// AUROC: the chance that a random positive scores above a random
/// negative, ties counting a half; the Mann-Whitney U over the product of
/// the two counts (Mann and Whitney 1947, Ann. Math. Stat. 18:50,
/// doi:10.1214/aoms/1177730491; as the area under the ROC curve, Hanley and
/// McNeil 1982, Radiology 143:29, doi:10.1148/radiology.143.1.7063747),
/// from midranks. NaN without a positive or a negative.
pub fn auroc(scores: &[f64], positive: &[bool]) -> f64 {
    let n_pos = positive.iter().filter(|&&p| p).count();
    let n_neg = positive.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut rank_sum = 0.0;
    let mut start = 0;
    while start < order.len() {
        let end = tie_end(&order, scores, start);
        // ranks start+1 ..= end share their mean
        let mid = (start + 1 + end) as f64 / 2.0;
        let pos = order[start..end].iter().filter(|&&i| positive[i]).count();
        rank_sum += mid * pos as f64;
        start = end;
    }
    let n_pos = n_pos as f64;
    (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg as f64)
}

/// Average precision, the step-wise area under the precision-recall
/// curve: the precision at each threshold weighted by the recall it adds,
/// tied scores taken as one threshold (Manning, Raghavan and Schütze 2008,
/// Introduction to Information Retrieval, section 8.4). NaN without a
/// positive.
pub fn average_precision(scores: &[f64], positive: &[bool]) -> f64 {
    let n_pos = positive.iter().filter(|&&p| p).count();
    if n_pos == 0 {
        return f64::NAN;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let (mut tp, mut ap) = (0usize, 0.0);
    let mut start = 0;
    while start < order.len() {
        let end = tie_end(&order, scores, start);
        let hits = order[start..end].iter().filter(|&&i| positive[i]).count();
        tp += hits;
        ap += hits as f64 / n_pos as f64 * tp as f64 / end as f64;
        start = end;
    }
    ap
}

/// The end of the run of tied scores in `order` that begins at `start`.
fn tie_end(order: &[usize], scores: &[f64], start: usize) -> usize {
    let s = scores[order[start]];
    start
        + order[start..]
            .iter()
            .take_while(|&&i| scores[i].total_cmp(&s).is_eq())
            .count()
}

/// Unit rows of `x`, row-major; a zero row stays zero (cosine 0 with all).
fn unit_rows(x: &Mat) -> Vec<f32> {
    let mut u = Vec::with_capacity(x.len());
    for r in x.row_iter() {
        let norm = r.norm();
        let scale = if norm > 0.0 { norm.recip() } else { 0.0 };
        u.extend(r.iter().map(|v| v * scale));
    }
    u
}

struct Joined<'a> {
    t: &'a TruthSpots,
    /// Row of each truth spot's vector in `unit`, when it has one.
    row: Vec<Option<usize>>,
    unit: Vec<f32>,
    dim: usize,
}

impl Joined<'_> {
    fn cosine(&self, i: usize, j: usize) -> f64 {
        let at = |s: usize| {
            let r = self.row[s].expect("a scored spot");
            &self.unit[r * self.dim..(r + 1) * self.dim]
        };
        at(i)
            .iter()
            .zip(at(j))
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum()
    }

    /// Each pair's cosine.
    fn cosines(&self, pairs: &[(usize, usize)]) -> Vec<f64> {
        pairs.iter().map(|&(i, j)| self.cosine(i, j)).collect()
    }

    /// Each pair's cosine, and whether its two spots share a domain.
    fn labelled(&self, pairs: &[(usize, usize)]) -> (Vec<f64>, Vec<bool>) {
        let scores = self.cosines(pairs);
        let pos = pairs
            .iter()
            .map(|&(i, j)| self.t.domain[i] == self.t.domain[j])
            .collect();
        (scores, pos)
    }

    fn score(&self, pairs: &[(usize, usize)]) -> PairSet {
        let (scores, pos) = self.labelled(pairs);
        set_of(&scores, &pos)
    }
}

fn set_of(scores: &[f64], positive: &[bool]) -> PairSet {
    let n_positive = positive.iter().filter(|&&p| p).count();
    PairSet {
        n_pairs: scores.len(),
        n_positive,
        prevalence: n_positive as f64 / scores.len() as f64,
        auroc: auroc(scores, positive),
        auprc: average_precision(scores, positive),
        negatives: None,
    }
}

/// The scores `pos` (positives) against each group of `negs` together.
fn versus(pos: &[f64], negs: &[&[f64]]) -> PairSet {
    let scores: Vec<f64> = pos
        .iter()
        .chain(negs.iter().copied().flatten())
        .copied()
        .collect();
    let label: Vec<bool> = (0..scores.len()).map(|i| i < pos.len()).collect();
    set_of(&scores, &label)
}

/// Up to `n` pairs of distinct spots, one from `left` and one from
/// `right`, that `keep` accepts, drawn uniformly with replacement. Gives
/// up after `100 n` draws, so a set with no such pair comes back empty.
fn draw(
    rng: &mut SmallRng,
    left: &[usize],
    right: &[usize],
    n: usize,
    keep: impl Fn(usize, usize) -> bool,
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(n);
    if left.is_empty() || right.is_empty() {
        return out;
    }
    for _ in 0..n.saturating_mul(100) {
        if out.len() == n {
            break;
        }
        let i = left[rng.random_range(0..left.len())];
        let j = right[rng.random_range(0..right.len())];
        if i != j && keep(i, j) {
            out.push((i, j));
        }
    }
    out
}

/// Score the vectors `x` (row `r` is cell `cells[r]`) against the truth.
pub fn score_pairs(
    t: &TruthSpots,
    layout: &SpotLayout,
    cells: &[Box<str>],
    x: &Mat,
    o: &PairOptions,
) -> anyhow::Result<PairScore> {
    anyhow::ensure!(cells.len() == x.nrows(), "one vector per cell");
    if let Some(r) = x.row_iter().position(|v| v.iter().any(|a| !a.is_finite())) {
        anyhow::bail!("the vector of cell {} is not finite", cells[r]);
    }
    anyhow::ensure!(
        layout.section.len() == t.cells.len() && layout.xy.len() == t.cells.len(),
        "the layout is not the truth's"
    );
    let mut given: HashMap<&str, usize> = HashMap::default();
    for (r, c) in cells.iter().enumerate() {
        anyhow::ensure!(
            given.insert(c.as_ref(), r).is_none(),
            "the vectors name cell {c} twice"
        );
    }
    let row: Vec<Option<usize>> = t
        .cells
        .iter()
        .map(|c| given.get(c.as_ref()).copied())
        .collect();
    let scored: Vec<usize> = (0..t.cells.len()).filter(|&i| row[i].is_some()).collect();
    anyhow::ensure!(
        !scored.is_empty(),
        "no spot of the truth has a vector: are the cell names spelled alike?"
    );
    let j = Joined {
        t,
        row,
        unit: unit_rows(x),
        dim: x.ncols(),
    };
    // each set its own stream, so one set's draws leave the others' alone
    let stream = |tag: u64| SmallRng::seed_from_u64(mix_seed(o.seed, tag));

    let shared: Vec<usize> = scored
        .iter()
        .copied()
        .filter(|&i| t.domain_owner[t.domain[i]].is_none())
        .collect();
    // draw only what can be drawn: two donors for cross, a donor with two
    // spots for within
    let mut per_donor = vec![0usize; t.n_donors];
    shared.iter().for_each(|&i| per_donor[t.donor[i]] += 1);
    let cross_n = if per_donor.iter().filter(|&&n| n > 0).count() >= 2 {
        o.pairs
    } else {
        0
    };
    let within_n = if per_donor.iter().any(|&n| n >= 2) {
        o.pairs
    } else {
        0
    };
    let cross = draw(&mut stream(1), &shared, &shared, cross_n, |a, b| {
        t.donor[a] != t.donor[b]
    });
    let within = draw(&mut stream(2), &shared, &shared, within_n, |a, b| {
        t.donor[a] == t.donor[b]
    });
    let across = j.score(&cross);
    let within = j.score(&within);

    let half = o.pairs / 2;
    let patches = (0..t.domain_labels.len())
        .filter_map(|k| t.domain_owner[k].map(|owner| (k, owner)))
        .map(|(k, owner)| {
            let patch: Vec<usize> = scored
                .iter()
                .copied()
                .filter(|&i| t.domain[i] == k && t.donor[i] == owner)
                .collect();
            // the negatives stay in the owner: its spots outside the domain
            let own_outside: Vec<usize> = scored
                .iter()
                .copied()
                .filter(|&i| t.domain[i] != k && t.donor[i] == owner)
                .collect();
            let mut r = stream(3 + k as u64);
            let n_pos = if patch.len() >= 2 { half } else { 0 };
            // the positives first, so they are the pairs drawn before
            let positives = j.cosines(&draw(&mut r, &patch, &patch, n_pos, |_, _| true));
            let negatives = j.cosines(&draw(
                &mut r,
                &patch,
                &own_outside,
                o.pairs - n_pos,
                |_, _| true,
            ));
            let note = if positives.is_empty() {
                Some(NO_POSITIVES)
            } else if negatives.is_empty() {
                Some(NO_NEGATIVES)
            } else {
                None
            };
            let mut set = PairSet {
                negatives: Some(PATCH_NEGATIVES),
                ..versus(&positives, &[&negatives])
            };
            if note.is_some() {
                // one side alone ranks nothing: no score, not a perfect one
                (set.auroc, set.auprc) = (f64::NAN, f64::NAN);
            }
            PatchPairs {
                domain: t.domain_labels[k].to_string(),
                owner: donor_label(owner).to_string(),
                n_spots: patch.len(),
                donor_domain_vs_own_donor: set,
                note,
            }
        })
        .collect();

    let edges = spatial_edges(layout, &scored, o.knn)?;
    Ok(PairScore {
        n_spots: t.cells.len(),
        n_scored: scored.len(),
        donor_gap: within.auroc - across.auroc,
        shared_domain_across_donors: across,
        shared_domain_within_donor: within,
        patches,
        neighbour_same_domain: j.score(&edges),
    })
}

/// Every edge of each spot's `knn` nearest neighbours among the `scored`
/// spots of its own section, each edge once.
fn spatial_edges(
    layout: &SpotLayout,
    scored: &[usize],
    knn: usize,
) -> anyhow::Result<Vec<(usize, usize)>> {
    let n_sections = crate::util::ablation::n_sections(&layout.section);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_sections];
    for &i in scored {
        members[layout.section[i]].push(i);
    }
    let mut edges = Vec::new();
    for m in members.iter().filter(|m| m.len() >= 2) {
        let xy = Mat::from_fn(m.len(), 2, |r, c| layout.xy[m[r]][c]);
        let g = KnnGraph::from_rows(
            &xy,
            KnnGraphArgs {
                knn,
                block_size: 1000,
                reciprocal: false,
            },
        )?;
        edges.extend(g.edges.iter().map(|&(a, b)| (m[a], m[b])));
    }
    Ok(edges)
}

/// Per-cell vectors: a run's propensity at `level`, or a table's `cell`
/// column and every other column, each a number (TSV or parquet, a cage
/// run's `cell_embedding` among them).
pub fn read_vectors(input: &str, level: &str) -> anyhow::Result<(Vec<Box<str>>, Mat)> {
    if !super::score::is_table(input) {
        let (prop, _, _, cells) =
            read_propensity(&propensity_path(input, level)?, &HashSet::default())?;
        return Ok((cells, prop));
    }
    let t = TableReader::open(input)?;
    let cell = t.select(&["cell"])?[0];
    let mut cols = vec![cell];
    cols.extend((0..t.header().len()).filter(|&c| c != cell));
    anyhow::ensure!(cols.len() > 1, "{input}: no column besides cell");
    let (mut cells, mut flat) = (Vec::new(), Vec::new());
    for row in t.rows(&cols)? {
        let row = row?;
        cells.push(row[0].trim().into());
        for (v, &c) in row[1..].iter().zip(&cols[1..]) {
            let x: f32 = v.trim().parse().map_err(|_| {
                anyhow::anyhow!("{input}: column {} is not a number ({v:?})", t.header()[c])
            })?;
            flat.push(x);
        }
    }
    let x = Mat::from_row_slice(cells.len(), cols.len() - 1, &flat);
    Ok((cells, x))
}

#[derive(clap::Args, Debug)]
pub struct SimPairsArgs {
    /// The simulation's output directory (as given to srt-simulate -o), its
    /// truth folder or its truth run
    #[arg(long)]
    pub truth: Box<str>,

    /// Vectors to score: pinto runs (prefix or .pinto.json, their
    /// propensity at --level), or tables (TSV or parquet, a cage run's
    /// cell_embedding among them) with a `cell` column and numeric columns,
    /// comma-separated or repeated
    #[arg(long, value_delimiter = ',', required = true)]
    pub vectors: Vec<Box<str>>,

    /// Level of a run's propensity
    #[arg(long, default_value = "final")]
    pub level: String,

    /// Pairs drawn for each sampled set
    #[arg(long, default_value_t = DEFAULT_PAIRS)]
    pub pairs: usize,

    /// Seed of the pair draws
    #[arg(long, default_value_t = DEFAULT_SEED)]
    pub seed: u64,

    /// Spatial neighbours of each spot, within its section
    #[arg(long, default_value_t = DEFAULT_KNN)]
    pub knn: usize,

    /// Write the scores as JSON here: one object, {"truth": ..., "scores":
    /// [{"vectors", "level", "pairs", "seed", "knn", "score"}, ...], "failed":
    /// [{"vectors", "error"}]} (failed only when an input could not be
    /// scored), the shape of the MCP's run_sim_score_by_pair reply
    #[arg(short, long)]
    pub out: Option<Box<str>>,
}

// the table's AUROC columns, named as the JSON's sets
const ACROSS: &str = "shared_domain_across_donors";
const WITHIN: &str = "shared_domain_within_donor";
const GAP: &str = "donor_gap";
const NEIGHBOUR: &str = "neighbour_same_domain";
const OWN: &str = "donor_domain_vs_own_donor";

/// The table's header line, as `srt-sim-pairs` prints it: AUROC of each
/// set, named as in the JSON (AUPRC and prevalence are in the JSON);
/// `donor_domain` names the donor-specific domain.
pub fn pairs_header() -> String {
    format!(
        "{:<40} {:>7} {ACROSS} {WITHIN} {GAP} {NEIGHBOUR} {:<NAME_WIDTH$} {OWN}",
        "vectors", "scored", "donor_domain",
    )
}

/// A scoring's rows of the table: one per donor-specific domain, or one
/// when there is none.
pub fn pairs_rows(input: &str, s: &PairScore) -> Vec<String> {
    let head = format!(
        "{:<40} {:>7} {:>a$.3} {:>w$.3} {:>g$.3} {:>n$.3}",
        super::score::shown_name(input),
        s.n_scored,
        s.shared_domain_across_donors.auroc,
        s.shared_domain_within_donor.auroc,
        s.donor_gap,
        s.neighbour_same_domain.auroc,
        a = ACROSS.len(),
        w = WITHIN.len(),
        g = GAP.len(),
        n = NEIGHBOUR.len(),
    );
    if s.patches.is_empty() {
        return vec![head];
    }
    s.patches
        .iter()
        .map(|p| {
            format!(
                "{head} {:<NAME_WIDTH$} {:>o$.3}",
                p.domain,
                p.donor_domain_vs_own_donor.auroc,
                o = OWN.len(),
            )
        })
        .collect()
}

/// One input's vectors read at `level` and scored: its table rows, and its
/// JSON record with how the pairs were drawn.
pub fn score_input(
    t: &TruthSpots,
    layout: &SpotLayout,
    input: &str,
    level: &str,
    o: &PairOptions,
) -> anyhow::Result<(Vec<String>, serde_json::Value)> {
    let (cells, x) = read_vectors(input, level)?;
    let s = score_pairs(t, layout, &cells, &x, o)?;
    let record = serde_json::json!({
        "vectors": input,
        "level": level,
        "pairs": o.pairs,
        "seed": o.seed,
        "knn": o.knn,
        "score": s,
    });
    Ok((pairs_rows(input, &s), record))
}

pub fn run_pairs(args: &SimPairsArgs) -> anyhow::Result<()> {
    let o = PairOptions::new(args.pairs, args.seed, args.knn)?;
    let t = TruthSpots::read(&args.truth)?;
    let layout = SpotLayout::read(&args.truth, &t)?;
    let mut all = Vec::new();
    crate::util::staging::say(pairs_header());
    let mut failed = Vec::new();
    for input in &args.vectors {
        match score_input(&t, &layout, input, &args.level, &o) {
            Ok((rows, record)) => {
                rows.iter().for_each(crate::util::staging::say);
                all.push(record);
            }
            Err(e) => {
                eprintln!("{input}: {e:#}");
                failed.push(
                    serde_json::json!({ "vectors": input.as_ref(), "error": format!("{e:#}") }),
                );
            }
        }
    }
    let n_failed = failed.len();
    if let Some(out) = &args.out {
        let doc = super::score::scores_document(&args.truth, all, failed);
        std::fs::write(out.as_ref(), serde_json::to_string_pretty(&doc)?)?;
    }
    if n_failed != 0 {
        // the rows that were scored are shown, as without staging
        crate::util::staging::say_held();
    }
    anyhow::ensure!(n_failed == 0, "{n_failed} input(s) could not be scored");
    Ok(())
}
