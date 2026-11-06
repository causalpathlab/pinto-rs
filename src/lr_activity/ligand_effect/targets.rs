//! Receiver-side response programmes: the target genes whose expression
//! measures a receiver's response to one ligand-receptor pair.
//!
//! Sets are keyed by receptor (the default reference: receptor → directed
//! signalling → transcription factor → TF targets), by ligand, or by the
//! pair. A pair takes the most specific set available: pair, then receptor,
//! then ligand. Sets from several sources are merged, so a user file
//! augments the default rather than replacing it.
//!
//! Weights are signed: a target the pathway represses counts against the
//! response. Every set is purged of the ligand, of every receptor listed for
//! that ligand and of the pair's own receptor. A programme containing the
//! signalling genes themselves would measure co-expression again, not
//! response. No published resource applies that rule, so it is applied here
//! and the number removed is reported.
//!
//! Formats:
//! - **TSV/CSV** with a header naming its columns. The key column is one of
//!   `receptor`, `ligand` or both; then `target`, and optionally `weight`
//!   (default 1).
//! - **GMT**: `set_name<TAB>description<TAB>gene…`, unit weights. The set name
//!   is a receptor or ligand symbol, or `ligand_receptor` for a pair set. A
//!   name with `_` is read against the pair list
//!   ([`TargetSets::settle_gmt_names`]): a name the pair list uses whole for
//!   a ligand or receptor (a complex, `REC1_REC2`) keys that gene; else
//!   the `_` that splits it into a listed pair (`LIG1_REC1_REC2`); else
//!   the first `_`.

use crate::util::common::*;

/// Weighted target genes, by name.
pub type TargetSet = Vec<(Box<str>, f32)>;

#[derive(Default)]
pub struct TargetSets {
    by_gene: HashMap<Box<str>, TargetSet>,
    by_pair: HashMap<(Box<str>, Box<str>), TargetSet>,
    /// GMT sets whose name has `_`, until the pair list settles them.
    unsettled: Vec<(Box<str>, TargetSet)>,
}

fn key(s: &str) -> Box<str> {
    s.trim().to_ascii_uppercase().into_boxed_str()
}

impl TargetSets {
    /// Read one file and merge it in.
    pub fn add_file(&mut self, path: &str) -> anyhow::Result<()> {
        // `read_lines` opens `.gz` transparently.
        let text = read_lines(path)?.join("\n");
        if path
            .to_ascii_lowercase()
            .trim_end_matches(".gz")
            .ends_with(".gmt")
        {
            self.add_gmt(&text);
            Ok(())
        } else {
            let csv = path
                .to_ascii_lowercase()
                .trim_end_matches(".gz")
                .ends_with(".csv");
            self.add_table(&text, csv)
        }
    }

    pub fn add_gmt(&mut self, text: &str) {
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').map(str::trim).collect();
            if f.len() < 3 || f[0].is_empty() {
                continue;
            }
            let genes = f[2..]
                .iter()
                .filter(|g| !g.is_empty())
                .map(|g| ((*g).into(), 1.0));
            let name = f[0].trim_matches('_');
            if name.contains('_') {
                self.unsettled.push((key(name), genes.collect()));
            } else {
                self.by_gene.entry(key(name)).or_default().extend(genes);
            }
        }
    }

    /// Key the GMT sets whose name has `_` against the pair list: a name
    /// some pair uses whole as its ligand or receptor (a complex such as
    /// `REC1_REC2`) keys that gene; else a split at a `_` that names a
    /// listed pair; else the split at the first `_`.
    pub fn settle_gmt_names(&mut self, pairs: &[(Box<str>, Box<str>)]) {
        let genes: HashSet<Box<str>> = pairs.iter().flat_map(|(l, r)| [key(l), key(r)]).collect();
        let listed: HashSet<(Box<str>, Box<str>)> =
            pairs.iter().map(|(l, r)| (key(l), key(r))).collect();
        for (name, set) in std::mem::take(&mut self.unsettled) {
            if genes.contains(&name) {
                self.by_gene.entry(name).or_default().extend(set);
                continue;
            }
            let splits: Vec<(Box<str>, Box<str>)> = name
                .match_indices('_')
                .map(|(i, _)| (name[..i].into(), name[i + 1..].into()))
                .collect();
            let pair = splits
                .iter()
                .find(|p| listed.contains(*p))
                .or(splits.first())
                .cloned()
                .expect("an unsettled name has a `_` inside");
            self.by_pair.entry(pair).or_default().extend(set);
        }
    }

    pub fn add_table(&mut self, text: &str, csv: bool) -> anyhow::Result<()> {
        let split = |line: &str| -> Vec<String> {
            if csv {
                line.split(',').map(|t| t.trim().to_string()).collect()
            } else {
                line.split('\t').map(|t| t.trim().to_string()).collect()
            }
        };
        let mut lines = text
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
        let header = split(
            lines
                .next()
                .ok_or_else(|| anyhow::anyhow!("empty target file"))?,
        );
        let col = |name: &str| header.iter().position(|h| h.eq_ignore_ascii_case(name));
        let (lig, rec, tgt, wt) = (col("ligand"), col("receptor"), col("target"), col("weight"));
        let tgt = tgt.ok_or_else(|| anyhow::anyhow!("target file needs a `target` column"))?;
        anyhow::ensure!(
            lig.is_some() || rec.is_some(),
            "target file needs a `ligand` or `receptor` column"
        );
        for (ln, line) in lines.enumerate() {
            let f = split(line);
            let get = |i: Option<usize>| {
                i.and_then(|i| f.get(i))
                    .map(|s| s.as_str())
                    .filter(|s| !s.is_empty())
            };
            let Some(t) = get(Some(tgt)) else { continue };
            let w = match get(wt) {
                Some(s) => s.parse::<f32>().map_err(|_| {
                    anyhow::anyhow!("line {}: weight {s:?} is not a number", ln + 2)
                })?,
                None => 1.0,
            };
            let entry = (t.into(), w);
            match (get(lig), get(rec)) {
                (Some(l), Some(r)) => self
                    .by_pair
                    .entry((key(l), key(r)))
                    .or_default()
                    .push(entry),
                (Some(g), None) | (None, Some(g)) => {
                    self.by_gene.entry(key(g)).or_default().push(entry)
                }
                (None, None) => {}
            }
        }
        Ok(())
    }

    /// Each target's specificity, `ln((N + 1) / (n_g + 1))` over the `N`
    /// sets, `n_g` of which contain gene `g` (an inverse document
    /// frequency). Literature networks route most receptors to the same
    /// immediate-early and hub genes; a programme dominated
    /// by them responds to any stimulus. A gene in every set scores 0.
    pub fn specificity(&self) -> HashMap<Box<str>, f32> {
        debug_assert!(self.unsettled.is_empty(), "settle the GMT names first");
        let n_sets = (self.by_gene.len() + self.by_pair.len()) as f64;
        let mut df: HashMap<Box<str>, usize> = HashMap::default();
        for set in self.by_gene.values().chain(self.by_pair.values()) {
            let genes: HashSet<Box<str>> = set.iter().map(|(g, _)| key(g)).collect();
            for g in genes {
                *df.entry(g).or_insert(0) += 1;
            }
        }
        df.into_iter()
            .map(|(g, d)| (g, ((n_sets + 1.0) / (d as f64 + 1.0)).ln() as f32))
            .collect()
    }

    /// The pair's programme with the signalling genes removed, and how many
    /// entries were removed. `receptors_of_ligand` is every receptor the
    /// pair list names for this ligand. A target listed twice keeps the sum
    /// of its weights.
    pub fn for_pair(
        &self,
        ligand: &str,
        receptor: &str,
        receptors_of_ligand: &[&str],
    ) -> Option<(TargetSet, usize)> {
        let set = self
            .by_pair
            .get(&(key(ligand), key(receptor)))
            .or_else(|| self.by_gene.get(&key(receptor)))
            .or_else(|| self.by_gene.get(&key(ligand)))?;
        let banned = |g: &str| {
            g.eq_ignore_ascii_case(ligand)
                || g.eq_ignore_ascii_case(receptor)
                || receptors_of_ligand
                    .iter()
                    .any(|r| g.eq_ignore_ascii_case(r))
        };
        let mut order: Vec<Box<str>> = Vec::new();
        let mut sum: HashMap<Box<str>, f32> = HashMap::default();
        let mut removed = 0;
        for (g, w) in set {
            if banned(g) {
                removed += 1;
                continue;
            }
            let k = key(g);
            if !sum.contains_key(&k) {
                order.push(k.clone());
            }
            *sum.entry(k).or_insert(0.0) += w;
        }
        let kept = order
            .into_iter()
            .filter_map(|g| {
                let w = sum[&g];
                (w != 0.0).then_some((g, w))
            })
            .collect();
        Some((kept, removed))
    }
}
