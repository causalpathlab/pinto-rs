//! `pinto compare A B -o out`: how far two runs' partitions of the same
//! cells agree.
//!
//! Each run's hard community labels (the `cluster` column of a propensity
//! table, or its argmax) are joined on the cell name. Over the cells both
//! runs placed it reports the adjusted Rand index, normalised mutual
//! information, a one-to-one community matching (Hungarian, on Jaccard),
//! and each community's fate: kept, split, merged, vanished or new. A
//! community's *partners* are the communities of the other run holding at
//! least τ of its cells; two or more partners is a split (seen from A) or a
//! merge (seen from B), none is a community the other run does not have.
//!
//! Writes `{out}.compare.json` and `{out}.contingency.parquet` (one row per
//! nonzero cell of the A × B table).

use crate::util::metadata::{run_manifest, run_prefix, PintoMetadata};
use crate::util::parquet_io::read_propensity;
use clap::Args;
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::matrix::parquet::{write_named_table, Column};
use rustc_hash::FxHashMap as HashMap;
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct CompareArgs {
    #[arg(
        help = "First run: a prefix, its .pinto.json, or a propensity .parquet",
        long_help = "First run (the reference): a run prefix, its {prefix}.pinto.json, or a\n\
                     propensity .parquet file itself. Communities called split, merged or\n\
                     vanished are named from A's side."
    )]
    pub a: String,

    #[arg(help = "Second run: a prefix, its .pinto.json, or a propensity .parquet")]
    pub b: String,

    #[arg(
        short,
        long,
        required = true,
        help = "Output prefix: writes {out}.compare.json and {out}.contingency.parquet"
    )]
    pub out: String,

    #[arg(
        long,
        default_value = "final",
        conflicts_with = "paired",
        help = "Level of A to read: final, draft, or a level tag (L1, L2, …)"
    )]
    pub level_a: String,

    #[arg(
        long,
        default_value = "final",
        conflicts_with = "paired",
        help = "Level of B to read: final, draft, or a level tag (L1, L2, …)"
    )]
    pub level_b: String,

    #[arg(
        long,
        default_value_t = 0.2,
        value_parser = fraction,
        conflicts_with = "paired",
        help = "Share of a community's cells that makes another community its partner",
        long_help = "A community's partners are the other run's communities holding at\n\
                     least this share of its cells. Two or more partners is a split (from\n\
                     A's side) or a merge (from B's side); none means the other run has no\n\
                     counterpart (vanished from A, new in B)."
    )]
    pub tau: f64,

    #[arg(
        long,
        help = "Compare held-out scores instead: A and B are `pinto predict` runs (comma lists)",
        long_help = "Compare two models' held-out scores on the same cells. A and B are\n\
                     `pinto predict` runs (prefix, .pinto.json or predictive.parquet),\n\
                     each a comma list for several folds (leave-one-donor-out). Units\n\
                     are joined on their cell pair; both sides must have split them\n\
                     alike (same data, -k, --seed and split flags). The donor is the\n\
                     replicate: per donor Δ = Σ(llik_A − llik_B) / Σ count, then an\n\
                     exact sign test, an exact Wilcoxon test and a donor bootstrap\n\
                     interval. Writes {out}.paired.json."
    )]
    pub paired: bool,

    #[arg(
        long,
        value_name = "FILE",
        requires = "paired",
        help = "--paired: section → donor (two columns); an unlisted section is its own donor"
    )]
    pub donor_map: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 2000,
        help = "--paired: donor bootstrap replicates"
    )]
    pub bootstrap: usize,

    #[arg(
        long,
        default_value_t = 42,
        help = "--paired: bootstrap seed",
        hide = true
    )]
    pub bootstrap_seed: u64,
}

fn fraction(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if v > 0.0 && v <= 1.0 {
        Ok(v)
    } else {
        Err(format!("{v} is not in (0, 1]"))
    }
}

/// Where `{out}.compare.json` goes.
pub fn json_path(out: &str) -> String {
    format!("{out}.compare.json")
}

/// Where `{out}.contingency.parquet` goes.
pub fn contingency_path(out: &str) -> String {
    format!("{out}.contingency.parquet")
}

/// The propensity table `input` names at `level`.
pub fn propensity_path(input: &str, level: &str) -> anyhow::Result<PathBuf> {
    if input.ends_with(".parquet") {
        anyhow::ensure!(
            level == "final",
            "{input} is a propensity table itself: it has no level {level:?}"
        );
        return Ok(PathBuf::from(input));
    }
    let prefix = run_prefix(input);
    let dir = Path::new(prefix)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let beside = |p: &str| -> PathBuf {
        let p = Path::new(p);
        if p.is_absolute() || p.exists() {
            p.to_path_buf()
        } else {
            dir.join(p.file_name().unwrap_or_default())
        }
    };
    let by_name = match level {
        "final" => format!("{prefix}.propensity.parquet"),
        tag => format!("{prefix}.{tag}.propensity.parquet"),
    };
    let listed = PintoMetadata::read(Path::new(&run_manifest(input)))
        .ok()
        .and_then(|meta| meta.level_list().into_iter().find(|l| l.tag == level))
        .map(|l| beside(&l.propensity));
    let path = listed.unwrap_or_else(|| PathBuf::from(by_name));
    anyhow::ensure!(
        path.exists(),
        "{input}: no propensity table for level {level:?} ({} does not exist)",
        path.display()
    );
    Ok(path)
}

/// One run's hard labels: community names and each cell's community.
pub struct Partition {
    pub cells: Vec<Box<str>>,
    pub labels: Vec<i64>,
}

impl Partition {
    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let (_, labels, _, cells) = read_propensity(path, &Default::default())?;
        Ok(Partition { cells, labels })
    }
}

/// A community's name as the propensity columns spell it.
fn name(c: i64) -> String {
    format!("C{c}")
}

#[derive(Serialize, Debug)]
pub struct Side {
    pub input: String,
    pub propensity: String,
    pub level: String,
    pub n_cells: usize,
    pub n_communities: usize,
}

#[derive(Serialize, Debug, Clone)]
pub struct Partner {
    pub community: String,
    /// Share of this community's shared cells the partner holds.
    pub fraction: f64,
}

#[derive(Serialize, Debug)]
pub struct Fate {
    pub community: String,
    /// Its cells both runs placed.
    pub n_cells: usize,
    pub status: &'static str,
    pub partners: Vec<Partner>,
    /// Its Hungarian match in the other run, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jaccard: Option<f64>,
}

#[derive(Serialize, Debug)]
pub struct Match {
    pub a: String,
    pub b: String,
    pub n_cells: usize,
    pub jaccard: f64,
}

#[derive(Serialize, Debug)]
pub struct Split {
    pub a: String,
    pub into: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct Merge {
    pub b: String,
    pub from: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct Report {
    pub a: Side,
    pub b: Side,
    /// Cells both runs placed: every statistic is over these.
    pub n_shared: usize,
    pub n_only_a: usize,
    pub n_only_b: usize,
    pub tau: f64,
    pub ari: f64,
    pub nmi: f64,
    /// Mean Jaccard over A's communities, 0 for an unmatched one.
    pub mean_jaccard: f64,
    pub matching: Vec<Match>,
    pub vanished: Vec<String>,
    pub split: Vec<Split>,
    pub merged: Vec<Merge>,
    pub new: Vec<String>,
    pub communities_a: Vec<Fate>,
    pub communities_b: Vec<Fate>,
    pub contingency: String,
}

/// The A × B table over the shared cells, communities in ascending order.
pub struct Table {
    pub a: Vec<i64>,
    pub b: Vec<i64>,
    /// `n[i][j]`: cells in `a[i]` and `b[j]`.
    pub n: Vec<Vec<usize>>,
    pub n_only_a: usize,
    pub n_only_b: usize,
}

impl Table {
    pub fn new(pa: &Partition, pb: &Partition) -> anyhow::Result<Self> {
        let mut in_b: HashMap<&str, i64> = HashMap::default();
        for (cell, &l) in pb.cells.iter().zip(&pb.labels) {
            anyhow::ensure!(
                in_b.insert(cell.as_ref(), l).is_none(),
                "B names cell {cell} twice"
            );
        }
        let mut pairs = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for (cell, &l) in pa.cells.iter().zip(&pa.labels) {
            anyhow::ensure!(seen.insert(cell.as_ref()), "A names cell {cell} twice");
            if l < 0 {
                continue;
            }
            match in_b.get(cell.as_ref()) {
                Some(&m) if m >= 0 => pairs.push((l, m)),
                _ => {}
            }
        }
        anyhow::ensure!(
            !pairs.is_empty(),
            "the runs share no cells: are the cell names spelled alike?"
        );
        let index = |ls: &mut Vec<i64>| -> HashMap<i64, usize> {
            ls.sort_unstable();
            ls.dedup();
            ls.iter().enumerate().map(|(i, &l)| (l, i)).collect()
        };
        let mut a: Vec<i64> = pairs.iter().map(|p| p.0).collect();
        let mut b: Vec<i64> = pairs.iter().map(|p| p.1).collect();
        let (ia, ib) = (index(&mut a), index(&mut b));
        let mut n = vec![vec![0usize; b.len()]; a.len()];
        for &(l, m) in &pairs {
            n[ia[&l]][ib[&m]] += 1;
        }
        let shared = pairs.len();
        let placed = |p: &Partition| p.labels.iter().filter(|&&l| l >= 0).count();
        Ok(Table {
            a,
            b,
            n,
            n_only_a: placed(pa) - shared,
            n_only_b: placed(pb) - shared,
        })
    }

    pub fn total(&self) -> usize {
        self.n.iter().flatten().sum()
    }

    pub fn row_sums(&self) -> Vec<usize> {
        self.n.iter().map(|r| r.iter().sum()).collect()
    }

    pub fn col_sums(&self) -> Vec<usize> {
        (0..self.b.len())
            .map(|j| self.n.iter().map(|r| r[j]).sum())
            .collect()
    }

    pub fn jaccard(&self, i: usize, j: usize, rows: &[usize], cols: &[usize]) -> f64 {
        let both = self.n[i][j];
        let either = rows[i] + cols[j] - both;
        if either == 0 {
            0.0
        } else {
            both as f64 / either as f64
        }
    }
}

fn choose2(n: usize) -> f64 {
    let n = n as f64;
    n * (n - 1.0) / 2.0
}

/// Adjusted Rand index (Hubert & Arabie). 1 when both runs put every cell in
/// one community, where the index is otherwise 0/0.
pub fn ari(t: &Table) -> f64 {
    let sum_ij: f64 = t.n.iter().flatten().map(|&x| choose2(x)).sum();
    let sum_a: f64 = t.row_sums().into_iter().map(choose2).sum();
    let sum_b: f64 = t.col_sums().into_iter().map(choose2).sum();
    let total = choose2(t.total());
    let expected = if total > 0.0 {
        sum_a * sum_b / total
    } else {
        0.0
    };
    let max = (sum_a + sum_b) / 2.0;
    if (max - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_ij - expected) / (max - expected)
}

/// Normalised mutual information, I / mean(H_A, H_B) (the arithmetic mean,
/// scikit-learn's default). 1 when both entropies are 0.
pub fn nmi(t: &Table) -> f64 {
    let n = t.total() as f64;
    let entropy = |sums: &[usize]| -> f64 {
        sums.iter()
            .filter(|&&s| s > 0)
            .map(|&s| {
                let p = s as f64 / n;
                -p * p.ln()
            })
            .sum()
    };
    let (rows, cols) = (t.row_sums(), t.col_sums());
    let (ha, hb) = (entropy(&rows), entropy(&cols));
    let mut mi = 0.0;
    for (i, row) in t.n.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            if x > 0 {
                let p = x as f64 / n;
                mi += p * (p / (rows[i] as f64 / n * cols[j] as f64 / n)).ln();
            }
        }
    }
    let mean = (ha + hb) / 2.0;
    if mean <= 0.0 {
        return 1.0;
    }
    (mi / mean).clamp(0.0, 1.0)
}

/// The assignment of rows to columns of `cost` (rows ≤ columns) with the
/// least total cost: the Hungarian method with potentials, O(n²m).
/// `result[i]` is row i's column.
pub fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return vec![];
    }
    let m = cost[0].len();
    assert!(n <= m, "hungarian: more rows ({n}) than columns ({m})");
    // 1-based, with row/column 0 as the sentinel.
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut assigned = vec![0usize; n];
    for j in 1..=m {
        if p[j] > 0 {
            assigned[p[j] - 1] = j - 1;
        }
    }
    assigned
}

/// The one-to-one matching of A's communities to B's with the largest total
/// Jaccard; pairs that share no cells are left unmatched.
pub fn matching(t: &Table) -> Vec<(usize, usize)> {
    let (rows, cols) = (t.row_sums(), t.col_sums());
    let (na, nb) = (t.a.len(), t.b.len());
    let jac = |i: usize, j: usize| t.jaccard(i, j, &rows, &cols);
    let pairs: Vec<(usize, usize)> = if na <= nb {
        let cost: Vec<Vec<f64>> = (0..na)
            .map(|i| (0..nb).map(|j| -jac(i, j)).collect())
            .collect();
        hungarian(&cost).into_iter().enumerate().collect()
    } else {
        let cost: Vec<Vec<f64>> = (0..nb)
            .map(|j| (0..na).map(|i| -jac(i, j)).collect())
            .collect();
        hungarian(&cost)
            .into_iter()
            .enumerate()
            .map(|(j, i)| (i, j))
            .collect()
    };
    let mut pairs: Vec<_> = pairs.into_iter().filter(|&(i, j)| t.n[i][j] > 0).collect();
    pairs.sort_unstable();
    pairs
}

/// Each community's partners: the other side's communities holding ≥ τ of
/// its cells, largest first.
fn partners(counts: &[usize], total: usize, names: &[i64], tau: f64) -> Vec<Partner> {
    let mut out: Vec<Partner> = counts
        .iter()
        .zip(names)
        .filter(|(&x, _)| total > 0 && x as f64 / total as f64 >= tau)
        .map(|(&x, &c)| Partner {
            community: name(c),
            fraction: x as f64 / total as f64,
        })
        .collect();
    out.sort_by(|p, q| q.fraction.total_cmp(&p.fraction));
    out
}

/// Compare two partitions; `a` and `b` describe where they came from.
pub fn compare(t: &Table, a: Side, b: Side, tau: f64, contingency: String) -> Report {
    let (rows, cols) = (t.row_sums(), t.col_sums());
    let (na, nb) = (t.a.len(), t.b.len());
    let pa: Vec<Vec<Partner>> = (0..na)
        .map(|i| partners(&t.n[i], rows[i], &t.b, tau))
        .collect();
    let pb: Vec<Vec<Partner>> = (0..nb)
        .map(|j| {
            let col: Vec<usize> = t.n.iter().map(|r| r[j]).collect();
            partners(&col, cols[j], &t.a, tau)
        })
        .collect();
    let a_pos: HashMap<String, usize> =
        t.a.iter().enumerate().map(|(i, &c)| (name(c), i)).collect();
    let b_pos: HashMap<String, usize> =
        t.b.iter().enumerate().map(|(j, &c)| (name(c), j)).collect();

    let matched = matching(t);
    let mut match_a: Vec<Option<usize>> = vec![None; na];
    let mut match_b: Vec<Option<usize>> = vec![None; nb];
    for &(i, j) in &matched {
        match_a[i] = Some(j);
        match_b[j] = Some(i);
    }

    let mut vanished = Vec::new();
    let mut split = Vec::new();
    let communities_a: Vec<Fate> = (0..na)
        .map(|i| {
            let ps = &pa[i];
            let status = match ps.len() {
                0 => {
                    vanished.push(name(t.a[i]));
                    "vanished"
                }
                1 if pb[b_pos[&ps[0].community]].len() > 1 => "merged",
                1 => "kept",
                _ => {
                    split.push(Split {
                        a: name(t.a[i]),
                        into: ps.iter().map(|p| p.community.clone()).collect(),
                    });
                    "split"
                }
            };
            Fate {
                community: name(t.a[i]),
                n_cells: rows[i],
                status,
                partners: ps.clone(),
                matched: match_a[i].map(|j| name(t.b[j])),
                jaccard: match_a[i].map(|j| t.jaccard(i, j, &rows, &cols)),
            }
        })
        .collect();

    let mut new = Vec::new();
    let mut merged = Vec::new();
    let communities_b: Vec<Fate> = (0..nb)
        .map(|j| {
            let ps = &pb[j];
            let status = match ps.len() {
                0 => {
                    new.push(name(t.b[j]));
                    "new"
                }
                1 if pa[a_pos[&ps[0].community]].len() > 1 => "split_part",
                1 => "kept",
                _ => {
                    merged.push(Merge {
                        b: name(t.b[j]),
                        from: ps.iter().map(|p| p.community.clone()).collect(),
                    });
                    "merge"
                }
            };
            Fate {
                community: name(t.b[j]),
                n_cells: cols[j],
                status,
                partners: ps.clone(),
                matched: match_b[j].map(|i| name(t.a[i])),
                jaccard: match_b[j].map(|i| t.jaccard(i, j, &rows, &cols)),
            }
        })
        .collect();

    let matching: Vec<Match> = matched
        .iter()
        .map(|&(i, j)| Match {
            a: name(t.a[i]),
            b: name(t.b[j]),
            n_cells: t.n[i][j],
            jaccard: t.jaccard(i, j, &rows, &cols),
        })
        .collect();
    let mean_jaccard = matching.iter().map(|m| m.jaccard).sum::<f64>() / na.max(1) as f64;

    Report {
        a,
        b,
        n_shared: t.total(),
        n_only_a: t.n_only_a,
        n_only_b: t.n_only_b,
        tau,
        ari: ari(t),
        nmi: nmi(t),
        mean_jaccard,
        matching,
        vanished,
        split,
        merged,
        new,
        communities_a,
        communities_b,
        contingency,
    }
}

/// Write the nonzero cells of the table: one row per (A, B) pair.
pub fn write_contingency(path: &str, t: &Table) -> anyhow::Result<()> {
    let (rows, cols) = (t.row_sums(), t.col_sums());
    let mut ka: Vec<Box<str>> = Vec::new();
    let mut kb: Vec<Box<str>> = Vec::new();
    let (mut n, mut of_a, mut of_b, mut jac) = (vec![], vec![], vec![], vec![]);
    for (i, row) in t.n.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            if x == 0 {
                continue;
            }
            ka.push(name(t.a[i]).into());
            kb.push(name(t.b[j]).into());
            n.push(x as i64);
            of_a.push(x as f32 / rows[i] as f32);
            of_b.push(x as f32 / cols[j] as f32);
            jac.push(t.jaccard(i, j, &rows, &cols) as f32);
        }
    }
    write_named_table(
        path,
        "community_a",
        &ka,
        &[
            ("community_b".into(), Column::Str(&kb)),
            ("n_cells".into(), Column::I64(&n)),
            ("fraction_of_a".into(), Column::F32(&of_a)),
            ("fraction_of_b".into(), Column::F32(&of_b)),
            ("jaccard".into(), Column::F32(&jac)),
        ],
    )
}

/// Where `{out}.paired.json` goes.
pub fn paired_path(out: &str) -> String {
    format!("{out}.paired.json")
}

fn run_paired(args: &CompareArgs) -> anyhow::Result<()> {
    use crate::paired::{paired, read_donor_map, read_side};
    let json = paired_path(&args.out);
    anyhow::ensure!(
        !Path::new(&json).exists(),
        "{json} exists: a comparison is not written over; give another --out"
    );
    let list = |s: &str| -> Vec<String> {
        s.split(',')
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .map(str::to_string)
            .collect()
    };
    let (a_runs, b_runs) = (list(&args.a), list(&args.b));
    let map = match args.donor_map.as_deref() {
        Some(p) => read_donor_map(p)?,
        None => Vec::new(),
    };
    let (a, b) = (read_side(&a_runs)?, read_side(&b_runs)?);
    let report = paired(
        &a_runs,
        &b_runs,
        &a,
        &b,
        &map,
        args.bootstrap,
        args.bootstrap_seed,
    )?;
    mkdir_parent(&args.out)?;
    std::fs::write(&json, serde_json::to_string_pretty(&report)?)?;
    println!(
        "{} shared units over {} donor(s): Δ {:+.5} nats/count (A {:.4}, B {:.4}), 95% donor \
         bootstrap [{:+.5}, {:+.5}]",
        report.n_units,
        report.n_donors,
        report.delta_per_count,
        report.a_llik_per_count,
        report.b_llik_per_count,
        report.bootstrap.ci95[0],
        report.bootstrap.ci95[1]
    );
    for d in &report.donors {
        println!(
            "  {:<24} {:>8} units  Δ {:+.5}",
            d.donor, d.n_units, d.delta_per_count
        );
    }
    println!(
        "donors A better {} / B better {}: sign test p {:.3}, Wilcoxon p {}",
        report.donors_a_better,
        report.donors_b_better,
        report.sign_test_p,
        report
            .wilcoxon_p
            .map_or("n/a".into(), |p| format!("{p:.3}"))
    );
    println!("wrote {json}");
    Ok(())
}

pub fn run_compare(args: &CompareArgs) -> anyhow::Result<()> {
    if args.paired {
        return run_paired(args);
    }
    let (json, table_path) = (json_path(&args.out), contingency_path(&args.out));
    for p in [&json, &table_path] {
        anyhow::ensure!(
            !Path::new(p).exists(),
            "{p} exists: a comparison is not written over; give another --out"
        );
    }
    let read = |input: &str, level: &str| -> anyhow::Result<(Partition, Side)> {
        let path = propensity_path(input, level)?;
        let part = Partition::read(&path)?;
        let n_communities = part
            .labels
            .iter()
            .filter(|&&l| l >= 0)
            .collect::<HashSet<_>>()
            .len();
        let side = Side {
            input: input.to_string(),
            propensity: path.display().to_string(),
            level: level.to_string(),
            n_cells: part.labels.iter().filter(|&&l| l >= 0).count(),
            n_communities,
        };
        Ok((part, side))
    };
    let (pa, sa) = read(&args.a, &args.level_a)?;
    let (pb, sb) = read(&args.b, &args.level_b)?;
    let table = Table::new(&pa, &pb)?;
    let report = compare(&table, sa, sb, args.tau, table_path.clone());

    mkdir_parent(&args.out)?;
    write_contingency(&table_path, &table)?;
    std::fs::write(&json, serde_json::to_string_pretty(&report)?)?;

    println!(
        "{} vs {}: {} shared cells ({} only in A, {} only in B)",
        report.a.input, report.b.input, report.n_shared, report.n_only_a, report.n_only_b
    );
    println!(
        "ARI {:.4}  NMI {:.4}  mean Jaccard {:.4}  ({} vs {} communities)",
        report.ari, report.nmi, report.mean_jaccard, report.a.n_communities, report.b.n_communities
    );
    println!(
        "vanished {}  split {}  merged into {}  new {}",
        report.vanished.len(),
        report.split.len(),
        report.merged.len(),
        report.new.len()
    );
    println!("wrote {json} and {table_path}");
    Ok(())
}

#[cfg(test)]
#[path = "tests/compare.rs"]
mod tests;
