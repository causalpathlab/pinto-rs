//! Which ligand-receptor pairs to test: the reference, subunits, and `--only`.

use super::targets::TargetSets;
use super::{background, reference, PairChoices, INTERNALS};
use crate::lr_activity::args::ReferenceSpecies;
use crate::util::common::*;
use data_beans::utilities::name_matching::GeneIndex;
use std::path::Path;

/// A resolved pair: names, gene-axis indices (every subunit of a complex
/// ligand or receptor required), and its programme as `(gene index, weight)`.
pub(super) struct Pair {
    pub(super) ligand: Box<str>,
    pub(super) receptor: Box<str>,
    pub(super) ligand_genes: Vec<usize>,
    pub(super) receptor_genes: Vec<usize>,
    /// `(gene, weight as given, specificity)`.
    pub(super) targets: Vec<(usize, f64, f64)>,
    pub(super) n_removed: usize,
}

/// Per cell, the count of a complex's scarcest subunit.
pub(super) fn scarcest_subunit(cols: &[Vec<f64>]) -> Vec<f64> {
    let n = cols.first().map_or(0, Vec::len);
    (0..n)
        .map(|i| cols.iter().map(|c| c[i]).fold(f64::INFINITY, f64::min))
        .collect()
}

/// The reference's pair and programme files for this data, built into the
/// cache on first use. The first build prints where it came from, how big
/// it is, where it lives and how to move it or stay offline.
fn reference(
    species: ReferenceSpecies,
    gene_names: &[Box<str>],
) -> anyhow::Result<reference::ReferenceFiles> {
    let species = match species {
        ReferenceSpecies::Human => reference::Species::Human,
        ReferenceSpecies::Mouse => reference::Species::Mouse,
        ReferenceSpecies::Auto => species_of(gene_names),
    };
    let refresh =
        std::env::var_os("PINTO_REFERENCE_REFRESH").is_some_and(|v| !v.is_empty() && v != "0");
    let root = reference::cache_root(None)?;
    let downloads = root.join("downloads");
    let first = refresh || !downloads.is_dir();
    let files = reference::ensure(
        species,
        &reference::ReferenceOpts {
            dir: None,
            refresh,
            // Built locally for this user, so every source counts:
            // licences limit redistribution, not use.
            licence: reference::LicenceMode::All,
        },
    )?;
    let size = dir_size(&root);
    if first {
        let names: Vec<&str> = reference::sources::downloads()
            .iter()
            .map(|s| s.name)
            .collect();
        eprintln!(
            "Reference ({species:?}) built from LIANA+ consensus pairs, OmniPath signalling \
             and intercell, CollecTRI regulons and HCOP orthologs:\n  {}\n  \
             cache: {} ({}), reused by later runs\n  \
             offline machines: copy that folder and set PINTO_OFFLINE=1; \
             to keep it elsewhere set PINTO_REFERENCE_DIR=<dir>",
            names.join(", "),
            root.display(),
            human_bytes(size)
        );
    } else {
        info!(
            "Reference ({species:?}): {} ({})",
            root.display(),
            human_bytes(size)
        );
    }
    Ok(files)
}

fn dir_size(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    rd.filter_map(|e| e.ok())
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            _ => e.metadata().map(|m| m.len()).unwrap_or(0),
        })
        .sum()
}

pub fn human_bytes(b: u64) -> String {
    let mb = b as f64 / (1u64 << 20) as f64;
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

/// The pairs this data can estimate: the reference's for its species,
/// then `--lr-pairs` and `--targets` on top, resolved against the data's
/// genes ([`resolve_pairs`]); with the reference's record.
pub(super) fn reference_pairs(
    c: &PairChoices<'_>,
    gene_names: &[Box<str>],
    index: &GeneIndex,
) -> anyhow::Result<(Vec<Pair>, std::path::PathBuf)> {
    // The reference first, then the user's files on top of it.
    let files = reference(c.species, gene_names)?;
    let mut sets = TargetSets::default();
    sets.add_file(&files.receptor_targets.to_string_lossy())?;
    let mut pair_files = vec![files.lr_pairs];
    if let Some(f) = c.lr_pairs {
        pair_files.push(f.into());
    }
    for f in c.targets {
        sets.add_file(f)?;
    }
    let mut raw = Vec::new();
    for f in &pair_files {
        raw.extend(crate::lr_activity::io::read_lr_pairs(&f.to_string_lossy())?);
    }
    sets.settle_gmt_names(&raw);
    let specificity = sets.specificity();
    let pairs = resolve_pairs(raw, gene_names, index, &sets, &specificity);
    anyhow::ensure!(
        !pairs.is_empty(),
        "no ligand-receptor pair has a usable response programme"
    );
    Ok((pairs, files.record))
}

/// Every gene `pairs` read (ligands, receptors, targets), sorted.
pub(super) fn pair_genes(pairs: &[Pair]) -> Vec<usize> {
    let mut needed: Vec<usize> = Vec::new();
    for p in pairs {
        needed.extend(&p.ligand_genes);
        needed.extend(&p.receptor_genes);
        needed.extend(p.targets.iter().map(|(g, _, _)| *g));
    }
    needed.sort_unstable();
    needed.dedup();
    needed
}

/// The genes the receivers' identity leaves out, for every pair of `all`
/// whichever are estimated, so a subset (`--only`) sees the receivers the
/// full analysis sees; and the background pool ([`background::BackgroundPool`]), genes
/// outside every pair and programme, read with them and left out like
/// them.
pub(super) fn signalling_genes(
    all: &[Pair],
    n_genes: usize,
    seed: u64,
) -> (HashSet<usize>, Vec<usize>) {
    let needed = pair_genes(all);
    let backgrounds = background::background_pool_genes(n_genes, &needed, seed);
    let excluded = needed.iter().chain(&backgrounds).copied().collect();
    (excluded, backgrounds)
}

/// Separators of a complex's subunits in a ligand or receptor name
/// (`ITGA4_ITGB1`, `ITGA4+ITGB1`, `ITGA4:ITGB1`).
const SUBUNIT_SEPARATORS: [char; 3] = ['_', '+', ':'];

/// Gene-axis rows of a ligand or receptor name. A name that is itself a gene
/// of the panel is that gene; any other name with a subunit separator is a
/// complex, one row per subunit, and `None` when any subunit is missing.
/// The split comes first: the gene index, asked for `ITGA4_ITGB1`, would
/// match its trailing symbol `ITGB1` and lose the complex.
pub(super) fn subunit_genes(
    name: &str,
    gene_names: &[Box<str>],
    index: &GeneIndex,
) -> Option<Vec<usize>> {
    let is_complex = name.contains(SUBUNIT_SEPARATORS) && !is_panel_gene(name, gene_names, index);
    if !is_complex {
        return index.match_gene(name).map(|g| vec![g]);
    }
    let mut genes: Vec<usize> = name
        .split(SUBUNIT_SEPARATORS)
        .filter(|s| !s.is_empty())
        .map(|s| index.match_gene(s))
        .collect::<Option<_>>()?;
    genes.sort_unstable();
    genes.dedup();
    (!genes.is_empty()).then_some(genes)
}

/// Whether `name` is a whole gene row of the panel (case aside), as written
/// or after an id prefix (`ENSG…_NAME`), rather than a part of one.
fn is_panel_gene(name: &str, gene_names: &[Box<str>], index: &GeneIndex) -> bool {
    let Some(g) = index.match_gene(name) else {
        return false;
    };
    let q = name.to_lowercase();
    let row = gene_names[g].to_lowercase();
    let core = row.split('/').next().unwrap_or(&row);
    core == q || core.ends_with(&format!("_{q}"))
}

/// Resolve each listed pair's genes and programme. Pairs whose ligand or
/// receptor, or any subunit of a complex one, is missing, or whose
/// programme keeps fewer than `INTERNALS.min_targets` measured genes, are
/// dropped with a count. Pairs that resolve to the same genes as an earlier
/// one (spelling or case aside) are kept once.
pub(super) fn resolve_pairs(
    raw: Vec<(Box<str>, Box<str>)>,
    gene_names: &[Box<str>],
    index: &GeneIndex,
    sets: &TargetSets,
    specificity: &HashMap<Box<str>, f32>,
) -> Vec<Pair> {
    let mut receptors_of: HashMap<Box<str>, Vec<Box<str>>> = HashMap::default();
    for (l, r) in &raw {
        receptors_of
            .entry(l.to_ascii_uppercase().into())
            .or_default()
            .push(r.clone());
    }
    let (mut no_gene, mut no_set, mut thin) = (0usize, 0usize, 0usize);
    let mut seen: HashSet<(Vec<usize>, Vec<usize>)> = HashSet::default();
    let mut out = Vec::new();
    for (l, r) in raw {
        let (Some(lg), Some(rg)) = (
            subunit_genes(&l, gene_names, index),
            subunit_genes(&r, gene_names, index),
        ) else {
            no_gene += 1;
            continue;
        };
        if !seen.insert((lg.clone(), rg.clone())) {
            continue;
        }
        let others: Vec<&str> = receptors_of
            .get(&*l.to_ascii_uppercase())
            .map(|v| v.iter().map(|s| s.as_ref()).collect())
            .unwrap_or_default();
        // The programme is purged of every receptor of the ligand and of
        // every subunit of the ligand and those receptors.
        let subunits: Vec<&str> = std::iter::once(l.as_ref())
            .chain(others.iter().copied())
            .flat_map(|x| x.split(SUBUNIT_SEPARATORS))
            .filter(|s| !s.is_empty())
            .collect();
        let banned: Vec<&str> = others.iter().copied().chain(subunits).collect();
        let Some((set, removed)) = sets.for_pair(&l, &r, &banned) else {
            no_set += 1;
            continue;
        };
        let signalling: HashSet<usize> = lg.iter().chain(&rg).copied().collect();
        let targets: Vec<(usize, f64, f64)> = set
            .iter()
            .filter_map(|(g, w)| {
                let s = specificity
                    .get(&*g.to_ascii_uppercase())
                    .copied()
                    .unwrap_or(0.0) as f64;
                index.match_gene(g).map(|gi| (gi, *w as f64, s))
            })
            .filter(|(gi, _, _)| !signalling.contains(gi))
            .collect();
        if targets.len() < INTERNALS.min_targets {
            thin += 1;
            continue;
        }
        out.push(Pair {
            ligand: l,
            receptor: r,
            ligand_genes: lg,
            receptor_genes: rg,
            targets,
            n_removed: removed,
        });
    }
    if no_gene + no_set + thin > 0 {
        warn!(
            "Contact effect: kept {} pairs; dropped {no_gene} with a ligand or receptor (or a \
             subunit of one) not in the data, {no_set} with no response programme, {thin} \
             whose programme has fewer than {} measured genes",
            out.len(),
            INTERNALS.min_targets
        );
    }
    out
}

/// Human or mouse from the gene names' case: human symbols are upper case
/// (`TGFB1`), mouse symbols capitalised (`Tgfb1`). Ensembl ids and names
/// without letters vote for neither.
fn species_of(gene_names: &[Box<str>]) -> reference::Species {
    let (mut upper, mut title) = (0usize, 0usize);
    for g in gene_names {
        let sym = g.rsplit(['_', ':']).next().unwrap_or(g);
        if sym.starts_with("ENS") {
            continue;
        }
        let letters: Vec<char> = sym.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        if letters.len() < 2 {
            continue;
        }
        if letters.iter().all(|c| c.is_ascii_uppercase()) {
            upper += 1;
        } else if letters[0].is_ascii_uppercase()
            && letters[1..].iter().any(|c| c.is_ascii_lowercase())
        {
            title += 1;
        }
    }
    let species = if title > upper {
        reference::Species::Mouse
    } else {
        reference::Species::Human
    };
    info!("Species from gene names: {species:?} ({upper} upper-case, {title} capitalised symbols)");
    species
}
