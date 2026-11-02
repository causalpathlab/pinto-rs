//! `pinto compare --paired`: two models' held-out scores on the same cells.
//!
//! Each side is one or more `pinto predict` runs on held-out data (one per
//! fold of a leave-one-donor-out design, say). Their `predictive.parquet`
//! units are joined on the unit key: both models scored the same cell pairs
//! with the same split (same data, graph and seed), so a unit's difference
//! `llik_A − llik_B` is over identical counts. A unit whose scored counts
//! differ between the two is refused: the splits were not the same.
//!
//! Units of one donor are not independent (neighbouring spots, one batch,
//! one δ), so the DONOR is the replicate. A unit's donor is its section (the
//! `@sample` of its first cell, or the run it came from when names carry
//! none), mapped to a donor by `--donor-map` when sections share a donor. Per
//! donor: `Δ_d = Σ (llik_A − llik_B) / Σ count`, nats per count. Over donors:
//! an exact sign test, an exact Wilcoxon signed-rank test, and a donor
//! bootstrap interval of the pooled `Δ`. The share of units with `Δ > 0` is
//! reported as description only.

use crate::util::ablation::section_of;
use crate::util::common::*;
use crate::util::metadata::{run_prefix, run_stem};
use crate::util::parquet_io::read_keyed_columns;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde::Serialize;
use std::path::Path;

/// One unit's score under one model.
#[derive(Clone, Copy, Debug)]
pub struct UnitScore {
    pub llik: f64,
    pub count: f64,
}

/// A run's `predictive.parquet`: `{run}.predictive.parquet` for a prefix or
/// manifest, or the file itself.
pub fn predictive_path(run: &str) -> anyhow::Result<String> {
    let path = if run.ends_with(".parquet") {
        run.to_string()
    } else {
        format!("{}.predictive.parquet", run_prefix(run))
    };
    anyhow::ensure!(
        Path::new(&path).is_file(),
        "{run}: no {path}; a paired comparison takes `pinto predict` runs"
    );
    Ok(path)
}

/// Every scored unit of `runs`, keyed by unit, with its section.
pub fn read_side(runs: &[String]) -> anyhow::Result<HashMap<Box<str>, (String, UnitScore)>> {
    let mut out: HashMap<Box<str>, (String, UnitScore)> = HashMap::default();
    for run in runs {
        let path = predictive_path(run)?;
        let (keys, mut cols) = read_keyed_columns(Path::new(&path), &["eval_llik", "eval_count"])?;
        let (count, llik) = (
            cols.pop().unwrap_or_default(),
            cols.pop().unwrap_or_default(),
        );
        // A unit's cells carry their section after `@`; a run without that
        // (one sample) is its own section.
        let stem = run_stem(run);
        for ((k, l), c) in keys.into_iter().zip(llik).zip(count) {
            if c.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !l.is_finite() {
                continue;
            }
            let first = k.split('|').next().unwrap_or(&k);
            // Names without `@sample` repeat across samples (barcodes, small
            // ids): such a unit is keyed and grouped by its run, so folds of
            // single-sample runs do not collide.
            let (sec, k): (String, Box<str>) = match section_of(first) {
                "" => (stem.clone(), format!("{stem}:{k}").into()),
                s => (s.to_string(), k),
            };
            anyhow::ensure!(
                out.insert(
                    k.clone(),
                    (
                        sec,
                        UnitScore {
                            llik: f64::from(l),
                            count: f64::from(c)
                        }
                    )
                )
                .is_none(),
                "unit {k} is scored twice on one side: do two of its runs overlap?"
            );
        }
    }
    Ok(out)
}

/// Section → donor from a two-column file (section, donor); a section not
/// listed is its own donor. A section is matched as `--mask-cells` matches
/// one: the same, or an `_`-separated end of it.
pub fn read_donor_map(path: &str) -> anyhow::Result<Vec<(String, String)>> {
    let pairs: Vec<(String, String)> = read_lines(path)?
        .iter()
        .filter_map(|line| {
            let mut f = line
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty());
            let (s, d) = (f.next()?, f.next()?);
            (!s.starts_with('#') && s != "section").then(|| (s.to_string(), d.to_string()))
        })
        .collect();
    anyhow::ensure!(
        !pairs.is_empty(),
        "--donor-map {path}: no section/donor rows"
    );
    Ok(pairs)
}

fn donor_of(section: &str, map: &[(String, String)]) -> String {
    // The most specific row: an exact name, else the longest matching end.
    map.iter()
        .filter(|(s, _)| crate::util::ablation::section_matches(section, s))
        .max_by_key(|(s, _)| (s == section, s.len()))
        .map_or_else(|| section.to_string(), |(_, d)| d.clone())
}

#[derive(Serialize, Debug, Clone)]
pub struct DonorDiff {
    pub donor: String,
    pub sections: Vec<String>,
    pub n_units: usize,
    pub count: f64,
    /// `Σ (llik_A − llik_B) / Σ count`.
    pub delta_per_count: f64,
    pub a_llik_per_count: f64,
    pub b_llik_per_count: f64,
}

#[derive(Serialize, Debug)]
pub struct PairedReport {
    pub a: Vec<String>,
    pub b: Vec<String>,
    pub n_units: usize,
    pub n_only_a: usize,
    pub n_only_b: usize,
    pub n_donors: usize,
    /// Pooled over all shared units, nats per count.
    pub delta_per_count: f64,
    pub a_llik_per_count: f64,
    pub b_llik_per_count: f64,
    pub donors: Vec<DonorDiff>,
    pub donors_a_better: usize,
    pub donors_b_better: usize,
    pub sign_test_p: f64,
    /// `None` with fewer than two donors that differ.
    pub wilcoxon_p: Option<f64>,
    pub bootstrap: Bootstrap,
    /// Description only: units of one donor are not independent.
    pub unit_share_a_better: f64,
    pub notes: Vec<String>,
}

#[derive(Serialize, Debug)]
pub struct Bootstrap {
    pub reps: usize,
    pub seed: u64,
    /// 2.5% and 97.5% of the pooled Δ over donors resampled with replacement.
    pub ci95: [f64; 2],
}

/// Two-sided exact sign test of `k` successes in `n`.
pub fn sign_test(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let pmf = |i: usize| -> f64 {
        let ln_c = ln_choose(n, i);
        (ln_c - n as f64 * std::f64::consts::LN_2).exp()
    };
    let lo: f64 = (0..=k).map(pmf).sum();
    let hi: f64 = (k..=n).map(pmf).sum();
    (2.0 * lo.min(hi)).min(1.0)
}

fn ln_choose(n: usize, k: usize) -> f64 {
    (1..=k).map(|i| ((n - k + i) as f64 / i as f64).ln()).sum()
}

/// Two-sided exact Wilcoxon signed-rank test on nonzero differences, ties in
/// |d| given mid-ranks. Exact by counting rank-sum subsets (ranks doubled to
/// stay integral); `None` below two differences.
pub fn wilcoxon(diffs: &[f64]) -> Option<f64> {
    let mut d: Vec<f64> = diffs.iter().copied().filter(|x| *x != 0.0).collect();
    let n = d.len();
    if n < 2 {
        return None;
    }
    d.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    // Doubled mid-ranks.
    let mut rank2 = vec![0usize; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && d[j + 1].abs() == d[i].abs() {
            j += 1;
        }
        let r2 = i + j + 2; // 2 × mean of ranks i+1..j+1
        for r in &mut rank2[i..=j] {
            *r = r2;
        }
        i = j + 1;
    }
    let w_plus: usize = d
        .iter()
        .zip(&rank2)
        .filter(|(x, _)| **x > 0.0)
        .map(|(_, r)| r)
        .sum();
    let total: usize = rank2.iter().sum();
    // ways[s] = number of sign assignments with doubled positive rank sum s.
    let mut ways = vec![0f64; total + 1];
    ways[0] = 1.0;
    for &r in &rank2 {
        for s in (r..=total).rev() {
            ways[s] += ways[s - r];
        }
    }
    let all: f64 = ways.iter().sum();
    let lo: f64 = ways[..=w_plus].iter().sum::<f64>() / all;
    let hi: f64 = ways[w_plus..].iter().sum::<f64>() / all;
    Some((2.0 * lo.min(hi)).min(1.0))
}

/// One donor's running sums.
#[derive(Default)]
struct DonorSums {
    sections: Vec<String>,
    n_units: usize,
    count: f64,
    /// `Σ (llik_A − llik_B)`.
    diff: f64,
    llik_a: f64,
    llik_b: f64,
}

/// Pair the two sides' units and test over donors.
pub fn paired(
    a_runs: &[String],
    b_runs: &[String],
    a: &HashMap<Box<str>, (String, UnitScore)>,
    b: &HashMap<Box<str>, (String, UnitScore)>,
    donor_map: &[(String, String)],
    reps: usize,
    seed: u64,
) -> anyhow::Result<PairedReport> {
    // donor → (sections, n, count, diff, llik_a, llik_b)
    let mut by_donor: HashMap<String, DonorSums> = HashMap::default();
    let (mut n, mut a_better) = (0usize, 0usize);
    // A section's donor, looked up once per section rather than per unit.
    let mut donor_cache: HashMap<&str, String> = HashMap::default();
    for (key, (sec, sa)) in a {
        let Some((_, sb)) = b.get(key) else { continue };
        anyhow::ensure!(
            (sa.count - sb.count).abs() < 0.5,
            "unit {key} has {} scored counts under A and {} under B: the two were not split \
             alike (same data, coordinates, -k, --seed and split flags are needed)",
            sa.count,
            sb.count
        );
        n += 1;
        if sa.llik > sb.llik {
            a_better += 1;
        }
        let donor = donor_cache
            .entry(sec.as_str())
            .or_insert_with(|| donor_of(sec, donor_map))
            .clone();
        let e = by_donor.entry(donor).or_default();
        if !e.sections.contains(sec) {
            e.sections.push(sec.clone());
        }
        e.n_units += 1;
        e.count += sa.count;
        e.diff += sa.llik - sb.llik;
        e.llik_a += sa.llik;
        e.llik_b += sb.llik;
    }
    anyhow::ensure!(n > 0, "the two sides share no scored unit");
    let mut donors: Vec<DonorDiff> = by_donor
        .into_iter()
        .map(|(donor, mut d)| {
            d.sections.sort();
            DonorDiff {
                donor,
                sections: d.sections,
                n_units: d.n_units,
                count: d.count,
                delta_per_count: d.diff / d.count,
                a_llik_per_count: d.llik_a / d.count,
                b_llik_per_count: d.llik_b / d.count,
            }
        })
        .collect();
    donors.sort_by(|x, y| x.donor.cmp(&y.donor));

    let total_count: f64 = donors.iter().map(|d| d.count).sum();
    let pooled = |ds: &[&DonorDiff]| -> f64 {
        let c: f64 = ds.iter().map(|d| d.count).sum();
        ds.iter().map(|d| d.delta_per_count * d.count).sum::<f64>() / c
    };
    let all: Vec<&DonorDiff> = donors.iter().collect();
    let delta = pooled(&all);
    let diffs: Vec<f64> = donors.iter().map(|d| d.delta_per_count).collect();
    let (pos, neg) = (
        diffs.iter().filter(|&&x| x > 0.0).count(),
        diffs.iter().filter(|&&x| x < 0.0).count(),
    );

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut boot: Vec<f64> = (0..reps)
        .map(|_| {
            let pick: Vec<&DonorDiff> = (0..donors.len())
                .map(|_| &donors[rng.random_range(0..donors.len())])
                .collect();
            pooled(&pick)
        })
        .collect();
    boot.sort_by(f64::total_cmp);
    let q = |p: f64| -> f64 {
        if boot.is_empty() {
            f64::NAN
        } else {
            boot[((p * (boot.len() - 1) as f64).round() as usize).min(boot.len() - 1)]
        }
    };

    let mut notes = vec![
        "the donor is the replicate: units of one donor share tissue, batch and δ, so the \
         unit share is description, not a test"
            .to_string(),
    ];
    let nd = donors.len();
    if nd < 6 {
        notes.push(format!(
            "{nd} donor(s): the smallest two-sided sign-test p possible is {:.3}; hold out more \
             donors (leave-one-donor-out folds) for a test with power",
            sign_test(0, nd)
        ));
    }
    let la: f64 = donors
        .iter()
        .map(|d| d.a_llik_per_count * d.count)
        .sum::<f64>()
        / total_count;
    let lb: f64 = donors
        .iter()
        .map(|d| d.b_llik_per_count * d.count)
        .sum::<f64>()
        / total_count;
    Ok(PairedReport {
        a: a_runs.to_vec(),
        b: b_runs.to_vec(),
        n_units: n,
        n_only_a: a.len() - n,
        n_only_b: b.keys().filter(|k| !a.contains_key(*k)).count(),
        n_donors: nd,
        delta_per_count: delta,
        a_llik_per_count: la,
        b_llik_per_count: lb,
        donors,
        donors_a_better: pos,
        donors_b_better: neg,
        sign_test_p: sign_test(pos, pos + neg),
        wilcoxon_p: wilcoxon(&diffs),
        bootstrap: Bootstrap {
            reps,
            seed,
            ci95: [q(0.025), q(0.975)],
        },
        unit_share_a_better: a_better as f64 / n as f64,
        notes,
    })
}

#[cfg(test)]
#[path = "tests/paired.rs"]
mod tests;
