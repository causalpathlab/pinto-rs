//! Receptor → TF → target programmes.
//!
//! From each receptor (every subunit of a complex) a breadth-first search
//! over directed protein signalling, up to `max_steps` edges, reaches
//! transcription factors (sources of CollecTRI regulons). For a reached TF
//! `t`, `d_t` is its shortest distance and `S_t` the sum over all shortest
//! paths of the product of edge signs (unsigned edges count +1); its
//! weight is `w_t = S_t / max(d_t, 1)`, and a complex TF takes the mean
//! over its subunits. A target's weight is `Σ_t w_t · mor(t, g)`, summed
//! exactly rounded as Python's `math.fsum` does, and the top `top` targets
//! by |weight| are kept, ties by name.
//!
//! Mouse programmes run the same search on the human networks translated
//! through one-to-one orthologs.

use super::sources::{Interaction, Licences, LrPair};
use super::LicenceMode;
use crate::lr_activity::ligand_effect::output::strip_zeros;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;

/// Directed signed edges `u → v` (+1, −1, or 0 for unsigned).
pub type Graph = FxHashMap<Box<str>, FxHashMap<Box<str>, i8>>;
/// TF (a gene or an `A_B` complex) → target → mode of regulation (±1).
pub type Regulons = FxHashMap<Box<str>, FxHashMap<Box<str>, i8>>;
/// Named counts for the build record.
pub type Stats = BTreeMap<&'static str, usize>;

/// The subunits of a `_`-joined complex name (a gene is its own subunit).
pub fn split_entity(name: &str) -> Vec<&str> {
    name.split('_').filter(|g| !g.is_empty()).collect()
}

// * Networks

/// Gene-level signalling: directed rows (by consensus) between proteins
/// or complexes that pass the licence filter, complexes expanded to their
/// subunits, self-loops dropped. Per gene pair, the edge takes the one
/// non-zero sign seen, or 0 when neither or both were seen.
pub fn build_signalling(rows: &[Interaction], lic: &Licences, mode: LicenceMode) -> (Graph, Stats) {
    const ZERO: u8 = 1;
    const PLUS: u8 = 2;
    const MINUS: u8 = 4;
    let mut stats = Stats::new();
    let mut seen: FxHashMap<(&str, &str), u8> = FxHashMap::default();
    let protein = |t: &str| t == "protein" || t == "complex";
    for r in rows {
        *stats.entry("rows").or_default() += 1;
        if !r.is_directed || !r.consensus_direction {
            *stats.entry("undirected").or_default() += 1;
            continue;
        }
        if !protein(&r.entity_type_source) || !protein(&r.entity_type_target) {
            *stats.entry("non_protein").or_default() += 1;
            continue;
        }
        if lic.keep_sources(&r.sources, mode).is_none() {
            *stats.entry("licence_dropped").or_default() += 1;
            continue;
        }
        let bit = match r.sign() {
            1 => PLUS,
            -1 => MINUS,
            _ => ZERO,
        };
        let targets = split_entity(&r.target_genesymbol);
        for u in split_entity(&r.source_genesymbol) {
            for &v in targets.iter().filter(|&&v| v != u) {
                *seen.entry((u, v)).or_default() |= bit;
            }
        }
        *stats.entry("kept_rows").or_default() += 1;
    }
    let mut graph = Graph::default();
    for ((u, v), bits) in seen {
        let sign = match bits & (PLUS | MINUS) {
            PLUS => 1,
            MINUS => -1,
            _ => 0,
        };
        graph.entry(u.into()).or_default().insert(v.into(), sign);
        *stats.entry("edges").or_default() += 1;
        *stats
            .entry(if sign != 0 {
                "edges_signed"
            } else {
                "edges_unsigned"
            })
            .or_default() += 1;
    }
    (graph, stats)
}

/// CollecTRI regulons: rows with a protein target that pass the licence
/// filter; mode −1 for inhibition only, else +1. A TF–target pair seen
/// with both modes is dropped.
pub fn build_regulons(
    rows: &[Interaction],
    lic: &Licences,
    mode: LicenceMode,
) -> (Regulons, Stats) {
    let mut stats = Stats::new();
    let mut seen: FxHashMap<(&str, &str), u8> = FxHashMap::default();
    for r in rows {
        *stats.entry("rows").or_default() += 1;
        if &*r.entity_type_target != "protein" {
            *stats.entry("complex_target").or_default() += 1;
            continue;
        }
        if lic.keep_sources(&r.sources, mode).is_none() {
            *stats.entry("licence_dropped").or_default() += 1;
            continue;
        }
        let bit = if r.sign() == -1 { 2 } else { 1 };
        *seen
            .entry((&r.source_genesymbol, &r.target_genesymbol))
            .or_default() |= bit;
    }
    let mut reg = Regulons::default();
    for ((tf, g), bits) in seen {
        let mor = match bits {
            1 => 1,
            2 => -1,
            _ => {
                *stats.entry("conflicting_mor_dropped").or_default() += 1;
                continue;
            }
        };
        reg.entry(tf.into()).or_default().insert(g.into(), mor);
        *stats.entry("edges").or_default() += 1;
    }
    (reg, stats)
}

// * Search

/// Distance and signed shortest-path sum of every node within
/// `max_steps` edges of `starts` (which have distance 0 and sum 1).
pub fn signed_bfs<'a>(
    graph: &'a Graph,
    starts: &[&'a str],
    max_steps: usize,
) -> FxHashMap<&'a str, (usize, i64)> {
    let mut reach: FxHashMap<&str, (usize, i64)> = starts.iter().map(|&s| (s, (0, 1))).collect();
    let mut frontier: Vec<&str> = reach.keys().copied().collect();
    for step in 1..=max_steps {
        let mut next: FxHashMap<&str, i64> = FxHashMap::default();
        for u in &frontier {
            let su = reach[u].1;
            let Some(nb) = graph.get(*u) else { continue };
            for (v, &sign) in nb {
                if reach.contains_key(&**v) {
                    continue;
                }
                *next.entry(&**v).or_default() += su * if sign == 0 { 1 } else { sign as i64 };
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next.keys().copied().collect();
        reach.extend(next.into_iter().map(|(v, s)| (v, (step, s))));
    }
    reach
}

/// TFs of `regulons` with their subunits, sorted by name.
pub fn tf_parts(regulons: &Regulons) -> Vec<(&str, Vec<&str>)> {
    let mut out: Vec<(&str, Vec<&str>)> = regulons
        .keys()
        .map(|tf| (&**tf, split_entity(tf)))
        .collect();
    out.sort();
    out
}

/// One target of a receptor's programme.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRow {
    pub target: Box<str>,
    pub weight: f64,
    /// TFs with non-zero weight that regulate the target.
    pub n_tfs: usize,
    /// Edges from receptor to target through the nearest such TF.
    pub min_path: usize,
}

/// The top `top` targets of `receptor` by |weight|, ties by name.
pub fn receptor_targets(
    receptor: &str,
    graph: &Graph,
    regulons: &Regulons,
    tfs: &[(&str, Vec<&str>)],
    max_steps: usize,
    top: usize,
) -> Vec<TargetRow> {
    let starts = split_entity(receptor);
    let reach = signed_bfs(graph, &starts, max_steps);
    let mut contrib: FxHashMap<&str, Vec<(f64, usize)>> = FxHashMap::default();
    for (tf, parts) in tfs {
        let reached: Vec<(f64, usize)> = parts
            .iter()
            .filter_map(|p| reach.get(p))
            .map(|&(d, s)| (s as f64 / d.max(1) as f64, d))
            .collect();
        let Some(d) = reached.iter().map(|&(_, d)| d).min() else {
            continue;
        };
        let w = fsum(reached.iter().map(|&(w, _)| w)) / parts.len() as f64;
        if w == 0.0 {
            continue;
        }
        for (g, &mor) in &regulons[*tf] {
            contrib
                .entry(&**g)
                .or_default()
                .push((w * f64::from(mor), d));
        }
    }
    let mut rows: Vec<TargetRow> = contrib
        .into_iter()
        .filter_map(|(g, cs)| {
            let weight = fsum(cs.iter().map(|c| c.0));
            (weight != 0.0).then(|| TargetRow {
                target: g.into(),
                weight,
                n_tfs: cs.len(),
                min_path: cs.iter().map(|c| c.1).min().unwrap_or(0) + 1,
            })
        })
        .collect();
    rows.sort_by(|a, b| {
        b.weight
            .abs()
            .total_cmp(&a.weight.abs())
            .then_with(|| a.target.cmp(&b.target))
    });
    rows.truncate(top);
    rows
}

/// The `receptor_targets` rows for every receptor of `lr` (sorted, then
/// each receptor's targets in rank order), and coverage counts.
pub fn species_targets(
    lr: &[LrPair],
    graph: &Graph,
    regulons: &Regulons,
    max_steps: usize,
    top: usize,
) -> (Vec<[String; 5]>, Stats) {
    let mut receptors: Vec<&str> = lr.iter().map(|p| &*p.receptor).collect();
    receptors.sort_unstable();
    receptors.dedup();
    let tfs = tf_parts(regulons);
    let per_receptor: Vec<Vec<TargetRow>> = receptors
        .par_iter()
        .map(|rec| receptor_targets(rec, graph, regulons, &tfs, max_steps, top))
        .collect();
    let mut rows = Vec::new();
    let mut covered: FxHashSet<&str> = FxHashSet::default();
    for (rec, targets) in receptors.iter().zip(&per_receptor) {
        if !targets.is_empty() {
            covered.insert(rec);
        }
        rows.extend(targets.iter().map(|t| {
            [
                rec.to_string(),
                t.target.to_string(),
                fmt_g(t.weight),
                t.n_tfs.to_string(),
                t.min_path.to_string(),
            ]
        }));
    }
    let nodes: FxHashSet<&str> = graph
        .iter()
        .flat_map(|(u, nb)| std::iter::once(&**u).chain(nb.keys().map(|v| &**v)))
        .collect();
    let stats = Stats::from([
        ("lr_pairs", lr.len()),
        ("receptors", receptors.len()),
        ("receptors_with_targets", covered.len()),
        ("receptors_without_tf_path", receptors.len() - covered.len()),
        (
            "pairs_covered",
            lr.iter().filter(|p| covered.contains(&*p.receptor)).count(),
        ),
        ("target_rows", rows.len()),
        ("regulon_tfs", regulons.len()),
        ("signalling_nodes", nodes.len()),
        (
            "signalling_edges",
            graph.values().map(FxHashMap::len).sum::<usize>(),
        ),
    ]);
    (rows, stats)
}

// * Orthology

/// The edges whose two genes both have an ortholog.
pub fn translate_graph(graph: &Graph, h2m: &FxHashMap<Box<str>, Box<str>>) -> Graph {
    let mut out = Graph::default();
    for (u, nb) in graph {
        let Some(mu) = h2m.get(u) else { continue };
        for (v, &s) in nb {
            if let Some(mv) = h2m.get(v) {
                out.entry(mu.clone()).or_default().insert(mv.clone(), s);
            }
        }
    }
    out
}

/// The regulons whose TF subunits all have orthologs, restricted to
/// targets with one; a TF left with no target is dropped.
pub fn translate_regulons(reg: &Regulons, h2m: &FxHashMap<Box<str>, Box<str>>) -> Regulons {
    let mut out = Regulons::default();
    for (tf, targets) in reg {
        let Some(parts) = split_entity(tf)
            .into_iter()
            .map(|p| h2m.get(p).map(|m| &**m))
            .collect::<Option<Vec<&str>>>()
        else {
            continue;
        };
        let mt: FxHashMap<Box<str>, i8> = targets
            .iter()
            .filter_map(|(g, &m)| h2m.get(g).map(|mg| (mg.clone(), m)))
            .collect();
        if !mt.is_empty() {
            out.insert(parts.join("_").into(), mt);
        }
    }
    out
}

// * Numbers

/// The exactly rounded sum of `xs` (Shewchuk's algorithm as CPython's
/// `math.fsum` runs it), so sums do not depend on order.
pub fn fsum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for mut x in xs {
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        if x != 0.0 {
            partials.push(x);
        }
    }
    let Some(mut n) = partials.len().checked_sub(1) else {
        return 0.0;
    };
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    hi
}

/// `x` as Python's `"%.6g"` writes it.
pub fn fmt_g(x: f64) -> String {
    const P: i32 = 6;
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.into();
    }
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".into()
        } else if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let sci = format!("{:.*e}", (P - 1) as usize, x);
    let (mant, exp) = sci.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("exponent");
    if !(-4..P).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip_zeros(mant), exp.abs())
    } else {
        strip_zeros(&format!("{:.*}", (P - 1 - exp) as usize, x)).into()
    }
}
