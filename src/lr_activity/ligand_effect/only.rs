//! `pinto lra --only`: estimate a few pairs, named on the command line,
//! instead of every pair of the reference and `--lr-pairs`.
//!
//! An entry is `LIGAND:RECEPTOR` (a complex joins its subunits with `_`,
//! `TGFB1:TGFBR1_TGFBR2`) or a bare `LIGAND` for all of its pairs. Entries
//! are matched on the genes they resolve to, as the pairs themselves are
//! ([`super::pairs::subunit_genes`]): case aside, subunits in any order. An entry
//! no pair answers is an error that lists the nearest pairs.
//!
//! A subset never writes over the full table: it goes to
//! `{out}.lra.only-<tag>.parquet` ([`tag`]), and the manifest records it
//! apart (`outputs.lra.subsets`). Its q-values are corrected within the
//! subset only, which its `q_scope` column says.

use super::pairs::Pair;
use data_beans::utilities::name_matching::GeneIndex;
use sha2::Digest;

/// Longest readable tag; a longer list is named by its hash.
const MAX_SLUG: usize = 24;

/// Nearest pairs an unknown entry's error lists.
const NEAR: usize = 5;

/// One `--only` entry: a ligand, and a receptor unless it is bare.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wanted {
    pub ligand: Box<str>,
    pub receptor: Option<Box<str>>,
}

impl std::fmt::Display for Wanted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.receptor {
            Some(r) => write!(f, "{}:{r}", self.ligand),
            None => write!(f, "{}", self.ligand),
        }
    }
}

/// The entries of `--only`, each checked for its shape.
pub fn parse(list: &[Box<str>]) -> anyhow::Result<Vec<Wanted>> {
    let mut out = Vec::new();
    for entry in list.iter().map(|e| e.trim()).filter(|e| !e.is_empty()) {
        let mut parts = entry.split(':').map(str::trim);
        let ligand = parts.next().unwrap_or_default();
        let receptor = parts.next();
        anyhow::ensure!(
            parts.next().is_none(),
            "--only {entry:?}: one `:` between ligand and receptor (a complex joins its \
             subunits with `_`: TGFB1:TGFBR1_TGFBR2)"
        );
        anyhow::ensure!(
            !ligand.is_empty() && receptor.is_none_or(|r| !r.is_empty()),
            "--only {entry:?}: LIGAND:RECEPTOR, or a bare LIGAND for all its pairs"
        );
        out.push(Wanted {
            ligand: ligand.into(),
            receptor: receptor.map(Into::into),
        });
    }
    anyhow::ensure!(!out.is_empty(), "--only names no pair");
    Ok(out)
}

/// The entries as the manifest and the tag know them: trimmed, upper
/// case, sorted, each once.
pub fn canonical(list: &[Box<str>]) -> Vec<String> {
    let mut v: Vec<String> = list
        .iter()
        .map(|e| {
            e.split(':')
                .map(str::trim)
                .collect::<Vec<_>>()
                .join(":")
                .to_uppercase()
        })
        .filter(|e| !e.is_empty())
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// A short, stable name for the list: the entries themselves, lower case
/// and joined by `+` (`:` written `-`), when that is short and safe in a
/// file name; else the first 8 hex digits of the SHA-256 of the
/// [`canonical`] list.
pub fn tag(list: &[Box<str>]) -> String {
    let entries = canonical(list);
    let slug = entries.join("+").to_lowercase().replace(':', "-");
    let safe = slug
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '_' | '.'));
    if safe && !slug.is_empty() && slug.len() <= MAX_SLUG {
        return slug;
    }
    let digest = sha2::Sha256::digest(entries.join(",").as_bytes());
    digest[..4].iter().map(|b| format!("{b:02x}")).collect()
}

/// `{out}.lra.only-<tag>.parquet`, the subset's table.
pub fn table_path(out: &str, list: &[Box<str>]) -> String {
    format!("{out}.lra.only-{}.parquet", tag(list))
}

/// `{out}.lra.only-<tag>.curve.parquet`, its curves.
pub fn curve_path(out: &str, list: &[Box<str>]) -> String {
    super::curve_beside(&table_path(out, list))
}

/// What a subset table's `q_scope` column says.
pub fn q_scope(n: usize, of: usize) -> String {
    format!("q corrected within the --only subset ({n} of {of} pairs)")
}

/// Which of `pairs` `wanted` names; an entry none answers is an error
/// listing the nearest pairs. `gene_names` and
/// `index` are the data's, which the pairs were resolved against.
pub(super) fn select(
    wanted: &[Wanted],
    pairs: &[Pair],
    gene_names: &[Box<str>],
    index: &GeneIndex,
) -> anyhow::Result<Vec<bool>> {
    let genes = |name: &str| super::pairs::subunit_genes(name, gene_names, index);
    let mut keep = vec![false; pairs.len()];
    let mut unknown: Vec<String> = Vec::new();
    for w in wanted {
        let lg = genes(&w.ligand);
        let rg = w.receptor.as_deref().map(genes);
        let hits: Vec<usize> = (0..pairs.len())
            .filter(|&k| {
                let p = &pairs[k];
                lg.as_ref() == Some(&p.ligand_genes)
                    && rg
                        .as_ref()
                        .is_none_or(|rg| rg.as_ref() == Some(&p.receptor_genes))
            })
            .collect();
        if hits.is_empty() {
            unknown.push(explain(w, lg.is_some(), rg.flatten().is_some(), pairs));
        }
        for k in hits {
            keep[k] = true;
        }
    }
    anyhow::ensure!(unknown.is_empty(), "--only: {}", unknown.join("; "));
    Ok(keep)
}

/// Why `w` matched no pair, with the nearest pairs.
fn explain(w: &Wanted, ligand_known: bool, receptor_known: bool, pairs: &[Pair]) -> String {
    let why = if !ligand_known {
        format!("no gene {} in the data", w.ligand)
    } else if w.receptor.is_some() && !receptor_known {
        format!(
            "no gene {} in the data",
            w.receptor.as_deref().unwrap_or_default()
        )
    } else {
        "no such pair with a usable response programme".into()
    };
    let near = near_pairs(w, pairs);
    if near.is_empty() {
        format!("{w}: {why}")
    } else {
        format!("{w}: {why}; nearest: {}", near.join(", "))
    }
}

/// The pairs whose name is nearest `w`'s (edit distance, case aside): for
/// a bare ligand the nearest ligands, each once.
pub(super) fn near_pairs(w: &Wanted, pairs: &[Pair]) -> Vec<String> {
    let names = pairs.iter().map(|p| match w.receptor {
        Some(_) => format!("{}:{}", p.ligand, p.receptor),
        None => p.ligand.to_string(),
    });
    nearest(&w.to_string(), names)
}

/// The [`NEAR`] of `names` nearest `want` (edit distance, case aside),
/// nearest first, each once.
pub fn nearest(want: &str, names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut names: Vec<String> = names.into_iter().collect();
    names.sort_unstable();
    names.dedup();
    let want = want.to_uppercase();
    let mut scored: Vec<(usize, String)> = names
        .into_iter()
        .map(|n| (edit_distance(&want, &n.to_uppercase()), n))
        .collect();
    scored.sort();
    scored.into_iter().take(NEAR).map(|(_, n)| n).collect()
}

/// Levenshtein distance between `a` and `b`, by characters.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diag = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let next = (diag + usize::from(ca != cb))
                .min(row[j] + 1)
                .min(row[j + 1] + 1);
            diag = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}
