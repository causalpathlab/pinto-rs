//! The tissue as every ligand-receptor pair sees it, built once per run:
//! which cells are receivers, who touches whom (and whom their contacts
//! touch), the confounder matrix with its cross-fitting folds, size
//! factors, identity bins and the distant ring partners of the negative
//! control.
//!
//! Besides the full design it keeps the nested designs of the sequential
//! adjustment, which show how much of an apparent effect each confounder
//! group explains: (a) section, (b) + spatial smooth, (c) + cell identity
//! (receiver and neighbour identity, and cell type when given); the full
//! design adds depth. Each is a column prefix of the one standardized
//! matrix and owns only its small per-fold Grams.

use super::confounders::{self, ConfounderInputs, Group};
use super::estimate::{Scope, ALL_SECTIONS};
use super::exposure::{median_contact_length, ring_partners, Contacts};
use super::folds::spatial_folds;
use super::overlap::{CrossFit, Design, DesignPrefix};
use super::receivers::{receiver_contacts, size_factors};
use crate::util::common::*;

/// Settings that shape the tissue, shared by estimation and simulation.
#[derive(Clone, Debug)]
pub struct TissueOpts {
    /// Smooth length scales, in contact lengths.
    pub scales: Vec<f64>,
    /// Fourier features per scale.
    pub n_freq: usize,
    /// Tile width for folds and clusters, in contact lengths.
    pub tile: f64,
    pub folds: usize,
    /// Ring of the negative control, in contact lengths.
    pub ring: (f64, f64),
    pub seed: u64,
}

impl Default for TissueOpts {
    fn default() -> Self {
        Self {
            scales: vec![8.0, 32.0],
            n_freq: 16,
            tile: 25.0,
            folds: 5,
            ring: (15.0, 30.0),
            seed: 42,
        }
    }
}

pub struct Tissue {
    /// Data column of each row.
    pub cell: Vec<usize>,
    pub coords: Vec<(f32, f32)>,
    pub section: Vec<u32>,
    pub section_names: Vec<Box<str>>,
    /// Median contact length per section.
    pub unit: Vec<f64>,
    pub contacts: Contacts,
    /// Second-order neighbours: contacts of contacts that are neither the
    /// row nor one of its own contacts.
    pub second: Vec<Vec<u32>>,
    pub ring: Vec<Vec<u32>>,
    pub cf: CrossFit,
    pub design: Design,
    /// Nested adjustment steps, coarsest first, all short of the full
    /// design: section, spatial, identity.
    pub steps: Vec<AdjustmentStep>,
    /// Depth over the median depth.
    pub size_factor: Vec<f64>,
    /// Identity bin of each row, for competence.
    pub identity_bin: Vec<usize>,
    /// Cell-type code of each row, when given ([`TissueInputs::cell_type`]).
    pub cell_type: Option<Vec<u32>>,
    /// Name of each cell-type code. Empty unless set after [`Tissue::build`]
    /// ([`Tissue::with_cell_type_names`]); with names, every pair also gets
    /// one row per receiver cell type (`section` = `type:<name>`).
    pub cell_type_names: Vec<Box<str>>,
    /// Where each pair's estimands are summed, with their row labels
    /// ([`Tissue::scopes`]), pooled first, and each scope's rows.
    pub scopes: Vec<(Box<str>, Scope)>,
    pub scope_rows: Vec<Vec<u32>>,
    /// Each row's dominant-neighbour identity bin
    /// ([`dominant_neighbour_bin`]).
    pub neighbour_bin: Vec<usize>,
}

/// One nested adjustment set: a column prefix of [`Tissue::design`].
pub struct AdjustmentStep {
    pub name: &'static str,
    pub design: DesignPrefix,
}

/// The step names, in the order of [`Tissue::steps`].
pub const STEP_NAMES: [&str; 3] = ["section", "spatial", "identity"];

/// Prefix width of each step: the end of the last group the step admits.
fn step_widths(groups: &[Group]) -> [usize; 3] {
    let end_of = |admit: &dyn Fn(&str) -> bool| {
        groups
            .iter()
            .filter(|g| admit(&g.name))
            .map(|g| g.cols.end)
            .max()
            .unwrap_or(1)
    };
    let section = end_of(&|g| g == "intercept" || g == "section");
    let spatial = end_of(&|g| g.starts_with("smooth@")).max(section);
    let identity =
        end_of(&|g| matches!(g, "identity" | "neighbour identity" | "cell type")).max(spatial);
    [section, spatial, identity]
}

pub struct TissueInputs<'a> {
    /// Coordinates per data column; `None` for a cell without coordinates.
    pub coords: &'a [Option<(f32, f32)>],
    /// Section label per data column.
    pub section: &'a [Box<str>],
    /// Spatial contacts between data columns.
    pub pairs: &'a [(usize, usize)],
    /// Identity projection, `K × n_cells`.
    pub identity: &'a Mat,
    /// Identity bin per data column.
    pub identity_bin: &'a [usize],
    /// `log1p(total count)` per data column.
    pub log1p_depth: &'a [f32],
    /// Optional cell-type code per data column; joins the identity group.
    pub cell_type: Option<&'a [u32]>,
}

/// Rows of the analysis: cells with coordinates and at least one contact
/// to another such cell. A sender is itself in that set, so contacts stay
/// inside it.
pub fn receiver_cells(coords: &[Option<(f32, f32)>], pairs: &[(usize, usize)]) -> Vec<usize> {
    let mut degree = vec![0usize; coords.len()];
    for &(i, j) in pairs {
        if i != j && coords[i].is_some() && coords[j].is_some() {
            degree[i] += 1;
            degree[j] += 1;
        }
    }
    (0..coords.len()).filter(|&c| degree[c] > 0).collect()
}

impl Tissue {
    pub fn build(inp: &TissueInputs<'_>, opts: &TissueOpts) -> anyhow::Result<Self> {
        let (cell, _, contacts) = receiver_contacts(inp.coords, inp.pairs);
        anyhow::ensure!(
            cell.len() >= 50,
            "too few cells with spatial contacts ({})",
            cell.len()
        );
        let second = second_neighbours(&contacts);

        let coords: Vec<(f32, f32)> = cell
            .iter()
            .map(|&c| inp.coords[c].expect("filtered"))
            .collect();
        let mut section_names: Vec<Box<str>> = Vec::new();
        let mut section_id: HashMap<Box<str>, u32> = HashMap::default();
        let section: Vec<u32> = cell
            .iter()
            .map(|&c| {
                let s = &inp.section[c];
                *section_id.entry(s.clone()).or_insert_with(|| {
                    section_names.push(s.clone());
                    section_names.len() as u32 - 1
                })
            })
            .collect();
        let n_sections = section_names.len();
        let unit = median_contact_length(&coords, &section, n_sections, &contacts);

        // A small tissue holds few tiles of the requested width. Narrow the
        // tiles until every fold gets at least `MIN_TILES_PER_FOLD`, down to
        // `MIN_TILE` contact lengths; below that the tiles stop separating
        // neighbourhoods and the variance clusters stop meaning anything.
        const MIN_TILES_PER_FOLD: usize = 8;
        const MIN_TILE: f64 = 3.0;
        let mut tile = opts.tile;
        let cf = loop {
            let side: Vec<f64> = unit.iter().map(|u| u * tile).collect();
            let cf = spatial_folds(&coords, &section, &side, opts.folds, opts.seed);
            let mut tiles_in = vec![HashSet::<u32>::default(); opts.folds];
            for (&f, &c) in cf.fold.iter().zip(&cf.cluster) {
                tiles_in[f as usize].insert(c);
            }
            if tiles_in.iter().all(|t| t.len() >= MIN_TILES_PER_FOLD) {
                break cf;
            }
            anyhow::ensure!(
                tile / 1.5 >= MIN_TILE,
                "the tissue is too small for {} cross-fitting folds of {MIN_TILES_PER_FOLD} \
                 tiles each, even at {tile:.1} contact lengths per tile (internal setting \
                 folds, see outputs.lra.settings in the manifest)",
                opts.folds
            );
            tile /= 1.5;
        };
        if tile < opts.tile {
            info!(
                "Narrowed cross-fitting tiles to {tile:.1} contact lengths (from {}) so each \
                 of {} folds holds at least {MIN_TILES_PER_FOLD} tiles",
                opts.tile, opts.folds
            );
        }

        let log_depth: Vec<f64> = cell.iter().map(|&c| inp.log1p_depth[c] as f64).collect();
        let size_factor = size_factors(&log_depth);

        let identity = inp.identity.select_columns(&cell);
        let identity_bin: Vec<usize> = cell.iter().map(|&c| inp.identity_bin[c]).collect();
        let cell_type: Option<Vec<u32>> = inp
            .cell_type
            .map(|ct| cell.iter().map(|&c| ct[c]).collect());

        let (x, groups) = confounders::build(&ConfounderInputs {
            coords: &coords,
            section: &section,
            n_sections,
            unit: &unit,
            contacts: &contacts,
            identity: &identity,
            log_depth: &log_depth,
            cell_type: cell_type.as_deref(),
            scales: &opts.scales,
            n_freq: opts.n_freq,
            seed: opts.seed,
        });
        info!(
            "Ligand effect: {} receivers over {} sections, {} tiles in {} folds; confounders {}",
            cell.len(),
            n_sections,
            cf.n_clusters(),
            opts.folds,
            groups
                .iter()
                .map(|g| format!("{}×{}", g.name, g.cols.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let design = Design::new(x, &cf);
        let steps: Vec<AdjustmentStep> = STEP_NAMES
            .iter()
            .zip(step_widths(&groups))
            .map(|(&name, p)| AdjustmentStep {
                name,
                design: design.prefix(p),
            })
            .collect();
        info!(
            "Sequential adjustment over the first {} of {} confounder columns",
            steps
                .iter()
                .map(|s| format!("{} ({})", s.design.p(), s.name))
                .collect::<Vec<_>>()
                .join(", "),
            design.p()
        );

        let ring = ring_partners(
            &coords,
            &section,
            &unit,
            &contacts,
            opts.ring.0,
            opts.ring.1,
            opts.seed,
        );
        let short = ring
            .iter()
            .enumerate()
            .filter(|(i, r)| r.len() < contacts.degree(*i))
            .count();
        if short > 0 {
            info!(
                "{short} of {} receivers have fewer ring partners than contacts \
                 (sparse tissue {}–{} contact lengths away)",
                cell.len(),
                opts.ring.0,
                opts.ring.1
            );
        }

        let neighbour_bin = dominant_neighbour_bin(&contacts, &identity_bin);
        let mut tissue = Self {
            cell,
            coords,
            section,
            section_names,
            unit,
            contacts,
            second,
            ring,
            cf,
            design,
            steps,
            size_factor,
            identity_bin,
            cell_type,
            cell_type_names: Vec::new(),
            scopes: Vec::new(),
            scope_rows: Vec::new(),
            neighbour_bin,
        };
        tissue.set_scopes();
        Ok(tissue)
    }

    /// The scopes of every pair with their row labels: pooled, each section
    /// when there are several, and each receiver cell type when types are
    /// named; and each scope's rows.
    fn set_scopes(&mut self) {
        let mut out: Vec<(Box<str>, Scope)> = vec![(ALL_SECTIONS.into(), Scope::All)];
        if self.section_names.len() > 1 {
            for (s, name) in self.section_names.iter().enumerate() {
                out.push((section_label(name), Scope::Section(s as u32)));
            }
        }
        if let (Some(ct), false) = (&self.cell_type, self.cell_type_names.is_empty()) {
            let mut codes: Vec<u32> = ct.clone();
            codes.sort_unstable();
            codes.dedup();
            for c in codes {
                let name = self
                    .cell_type_names
                    .get(c as usize)
                    .map_or_else(|| c.to_string(), |n| n.to_string());
                out.push((format!("type:{name}").into(), Scope::CellType(c)));
            }
        }
        let scopes: Vec<Scope> = out.iter().map(|(_, s)| *s).collect();
        self.scope_rows = scope_rows(&scopes, &self.section, self.cell_type.as_deref());
        self.scopes = out;
    }

    pub fn n(&self) -> usize {
        self.cell.len()
    }

    /// Name the cell-type codes given in [`TissueInputs::cell_type`]
    /// (`names[code]`), which turns on the per-type rows. Ignored without
    /// cell types.
    pub fn with_cell_type_names(mut self, names: Vec<Box<str>>) -> Self {
        if self.cell_type.is_some() {
            self.cell_type_names = names;
            self.set_scopes();
        }
        self
    }

    /// Mean of `v` over each row's second-order neighbours; 0 for a row
    /// without any.
    pub fn second_mean(&self, v: &[f64]) -> Vec<f64> {
        self.second
            .par_iter()
            .map(|r| {
                if r.is_empty() {
                    0.0
                } else {
                    r.iter().map(|&j| v[j as usize]).sum::<f64>() / r.len() as f64
                }
            })
            .collect()
    }

    /// Sum of `v` over each row's ring partners.
    pub fn ring_sum(&self, v: &[f64]) -> Vec<f64> {
        self.ring
            .par_iter()
            .map(|r| r.iter().map(|&j| v[j as usize]).sum())
            .collect()
    }
}

/// Contacts of contacts, without the row itself or its own contacts.
fn second_neighbours(contacts: &Contacts) -> Vec<Vec<u32>> {
    let adj = &contacts.adj;
    adj.par_iter()
        .enumerate()
        .map(|(i, own)| {
            let mut out: Vec<u32> = own
                .iter()
                .flat_map(|&j| adj[j as usize].iter().copied())
                .filter(|&k| k as usize != i && own.binary_search(&k).is_err())
                .collect();
            out.sort_unstable();
            out.dedup();
            out
        })
        .collect()
}

/// A section's row label: its name, unless the name reads as the pooled
/// row's ([`ALL_SECTIONS`]) or a cell type's (`type:…`), which gets a
/// `section:` prefix so no two kinds of row share a label.
pub fn section_label(name: &str) -> Box<str> {
    if name == ALL_SECTIONS || name.starts_with("type:") || name.starts_with("section:") {
        format!("section:{name}").into()
    } else {
        name.into()
    }
}

/// The rows (ascending) inside each scope, given each row's section and
/// cell type.
pub fn scope_rows(scopes: &[Scope], section: &[u32], cell_type: Option<&[u32]>) -> Vec<Vec<u32>> {
    let mut by_section: HashMap<u32, Vec<usize>> = HashMap::default();
    let mut by_type: HashMap<u32, Vec<usize>> = HashMap::default();
    for (k, s) in scopes.iter().enumerate() {
        match *s {
            Scope::All => {}
            Scope::Section(v) => by_section.entry(v).or_default().push(k),
            Scope::CellType(v) => by_type.entry(v).or_default().push(k),
        }
    }
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); scopes.len()];
    for i in 0..section.len() {
        for (k, s) in scopes.iter().enumerate() {
            if *s == Scope::All {
                out[k].push(i as u32);
            }
        }
        for &k in by_section.get(&section[i]).into_iter().flatten() {
            out[k].push(i as u32);
        }
        if let Some(ct) = cell_type {
            for &k in by_type.get(&ct[i]).into_iter().flatten() {
                out[k].push(i as u32);
            }
        }
    }
    out
}

/// Each row's dominant-neighbour identity bin: the most common bin among
/// its contacts, ties to the smaller bin; `usize::MAX` without contacts.
pub fn dominant_neighbour_bin(contacts: &Contacts, identity_bin: &[usize]) -> Vec<usize> {
    contacts
        .adj
        .par_iter()
        .map(|adj| {
            let mut b: Vec<usize> = adj.iter().map(|&j| identity_bin[j as usize]).collect();
            b.sort_unstable();
            let (mut best, mut best_len) = (usize::MAX, 0usize);
            for run in b.chunk_by(|x, y| x == y) {
                if run.len() > best_len {
                    (best, best_len) = (run[0], run.len());
                }
            }
            best
        })
        .collect()
}
