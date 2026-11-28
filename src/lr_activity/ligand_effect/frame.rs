//! A run's receivers and one pair on them, as `pinto view` draws them: each
//! receiver's contact exposure to the ligand, whether it counts as exposed,
//! whether its identity bin makes it receptor-competent, and whether it
//! carries the ligand itself.

use super::contacts::{competence, Contacts};
use super::dose::contact_dose;
use super::genes::GeneStore;
use super::link_community_scope::{community_inputs, community_scope, CommunityScope};
use super::options::{self, EffectOpts};
use super::pairs::{reference_pairs, scarcest_subunit, signalling_genes, subunit_genes};
use super::receivers::{receiver_contacts, size_factors};
use super::tissue::identity_bins;
use super::{dose, PairChoices, INTERNALS};
use crate::lr_activity::args::{LraInputArgs, ReferenceSpecies};
use crate::lr_activity::input::{
    link_community_of_contacts, load_contacts, Contacts as RunContacts,
};
use crate::util::common::HashSet;
use crate::util::metadata::LraOptions;
use data_beans::utilities::name_matching::GeneIndex;

/// One pair on the receivers, per row.
#[derive(Clone, Debug, PartialEq)]
pub struct PairMap {
    /// Contact dose: neighbours' ligand over their size factors, summed.
    pub exposure: Vec<f64>,
    /// The exposure threshold τ: a receiver is exposed above it.
    pub threshold: f64,
    pub exposed: Vec<bool>,
    /// The share of an identity bin detecting the receptor at which its
    /// receivers count as competent: the middle of the grid the estimate
    /// averages over, the one its counts use.
    pub competence: f64,
    pub competent: Vec<bool>,
    /// The receiver carries the ligand itself.
    pub sender: Vec<bool>,
    /// What each row gives each of its contacts: its ligand over its size
    /// factor, the term a neighbour's exposure sums.
    pub sent: Vec<f64>,
    /// Under a `--link-community` table, whether each row is a receiver of
    /// the community; every other row is out of scope, neither exposed nor
    /// a control. `None`: every row is a receiver.
    pub in_community: Option<Vec<bool>>,
}

/// `pair` on the receivers: `ligand_counts` and `receptor_counts` per row
/// (a complex as its scarcest subunit), exposed above `threshold` (a
/// table's, [`table_threshold`]), or above the threshold the estimate
/// would choose ([`dose::choose_threshold`]) when none is given. Every row
/// is in scope; the viewer draws through [`pair_map_in`].
#[cfg(test)]
pub fn pair_map(
    contacts: &Contacts,
    size_factor: &[f64],
    identity_bin: &[usize],
    ligand_counts: &[f64],
    receptor_counts: &[f64],
    threshold: Option<f64>,
) -> PairMap {
    pair_map_in(
        contacts,
        None,
        size_factor,
        identity_bin,
        (ligand_counts, receptor_counts),
        threshold,
    )
}

/// A pair on the receivers, its exposure summed over `contacts` (a link
/// community's, under `--link-community`) and, with `in_community`, the
/// rows outside it out of scope: the threshold read from the community's
/// receivers' exposures alone, as lra reads it ([`dose::Exposure::build`]).
pub fn pair_map_in(
    contacts: &Contacts,
    in_community: Option<&[bool]>,
    size_factor: &[f64],
    identity_bin: &[usize],
    (ligand_counts, receptor_counts): (&[f64], &[f64]),
    threshold: Option<f64>,
) -> PairMap {
    let (sent, a) = contact_dose(contacts, size_factor, ligand_counts);
    let tau = table_threshold(&dose::receiver_doses(&a, in_community), threshold);
    let detected: Vec<bool> = receptor_counts.iter().map(|&r| r > 0.0).collect();
    let grid = &options::COMPETENCE_GRID;
    let level = grid[grid.len() / 2];
    let mut exposed = dose::exposed(&a, tau);
    if let Some(inside) = in_community {
        for (e, &is_inside) in exposed.iter_mut().zip(inside) {
            *e &= is_inside;
        }
    }
    PairMap {
        exposed,
        exposure: a,
        threshold: tau,
        competence: level,
        competent: competence(identity_bin, &detected, level),
        sender: ligand_counts.iter().map(|&l| l > 0.0).collect(),
        sent,
        in_community: in_community.map(<[bool]>::to_vec),
    }
}

/// The threshold `given` as a table holds it, on the exposures `a`. The
/// table keeps τ as an `f32`, and τ is often an exposure itself (the
/// median, when the ligand reaches nearly every receiver): read back, it
/// can fall just below the receivers at τ and count them exposed. So
/// the estimate's own choice is taken when it is `given` to `f32`
/// precision, else the largest exposure that is; else `given` as it is.
/// With none given (or not a number), the estimate's choice.
pub fn table_threshold(a: &[f64], given: Option<f64>) -> f64 {
    let opts = EffectOpts {
        threshold: None,
        min_arm: INTERNALS.min_arm,
        ..EffectOpts::default()
    };
    let chosen = dose::choose_threshold(a, &opts);
    let Some(t) = given.filter(|t| t.is_finite()) else {
        return chosen;
    };
    let near = |x: f64| x == t || x as f32 == t as f32;
    if near(chosen) {
        return chosen;
    }
    a.iter()
        .copied()
        .filter(|&x| near(x))
        .max_by(f64::total_cmp)
        .unwrap_or(t)
}

/// What a receiver is on a pair's map, the order the classes are drawn and
/// listed in: [`Class::ALL`] lists them in this order, so `class as
/// usize` is its place there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Carries the ligand itself.
    Sender,
    ExposedCompetent,
    ExposedIncompetent,
    UnexposedCompetent,
    UnexposedIncompetent,
}

impl Class {
    pub const ALL: [Class; 5] = [
        Class::Sender,
        Class::ExposedCompetent,
        Class::ExposedIncompetent,
        Class::UnexposedCompetent,
        Class::UnexposedIncompetent,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Class::Sender => "sender (ligand+)",
            Class::ExposedCompetent => "exposed, receptor+",
            Class::ExposedIncompetent => "exposed, receptor-",
            Class::UnexposedCompetent => "unexposed, receptor+",
            Class::UnexposedIncompetent => "unexposed, receptor-",
        }
    }
}

impl PairMap {
    /// Row `i`'s class: a sender first, else by exposure and competence.
    pub fn class(&self, i: usize) -> Class {
        match (self.sender[i], self.exposed[i], self.competent[i]) {
            (true, _, _) => Class::Sender,
            (_, true, true) => Class::ExposedCompetent,
            (_, true, false) => Class::ExposedIncompetent,
            (_, false, true) => Class::UnexposedCompetent,
            (_, false, false) => Class::UnexposedIncompetent,
        }
    }

    pub fn n(&self) -> usize {
        self.exposure.len()
    }

    /// Whether row `i` is a receiver of the table: of its link community
    /// under `--link-community`, else every row.
    pub fn in_scope(&self, i: usize) -> bool {
        self.in_community.as_ref().is_none_or(|inside| inside[i])
    }

    /// The rows in scope where `flags` holds.
    pub fn count_in_scope(&self, flags: &[bool]) -> usize {
        (0..self.n())
            .filter(|&i| flags[i] && self.in_scope(i))
            .count()
    }

    /// Receivers in scope in each [`Class`], in [`Class::ALL`]'s order.
    pub fn counts(&self) -> [usize; 5] {
        let mut out = [0; 5];
        for i in (0..self.n()).filter(|&i| self.in_scope(i)) {
            out[self.class(i) as usize] += 1;
        }
        out
    }

    /// The share of receiver `to`'s exposure that its contact `from` gives
    /// when `to` is exposed (so in scope) and receptor-competent; 0
    /// otherwise. `from` must be one of `to`'s contacts the exposure is
    /// summed over.
    pub fn share(&self, from: usize, to: usize) -> f64 {
        let active = self.exposed[to] && self.competent[to];
        if active && self.exposure[to] > 0.0 {
            self.sent[from] / self.exposure[to]
        } else {
            0.0
        }
    }
}

/// The genes of `gene_names` named in `names`.
fn excluded_by_name(names: &[String], gene_names: &[Box<str>]) -> HashSet<usize> {
    let names: HashSet<&str> = names.iter().map(String::as_str).collect();
    (0..gene_names.len())
        .filter(|&g| names.contains(&*gene_names[g]))
        .collect()
}

/// A run's receivers as `pinto lra` saw them, read once: the data and
/// contacts of the run, each receiver's size factor and identity bin
/// (the identity leaving out the genes the options record, else every
/// pair's of the reference and of the options' `--lr-pairs` and
/// `--targets`, read again). [`RunFrame::pair`] then puts one pair on it.
pub struct RunFrame {
    run: RunContacts,
    gene_names: Vec<Box<str>>,
    index: GeneIndex,
    /// Data column of each receiver row, and each column's row.
    pub cell: Vec<usize>,
    row_of: Vec<usize>,
    contacts: Contacts,
    pub size_factor: Vec<f64>,
    pub identity_bin: Vec<usize>,
    /// Each data column's name.
    pub column_names: Vec<Box<str>>,
    /// Under a `--link-community` table, its community on the rows.
    community: Option<CommunityScope>,
}

impl RunFrame {
    /// The receivers of the run at `manifest` (its `.pinto.json`), with
    /// the identity the analysis made with `options`.
    pub fn open(manifest: &std::path::Path, options: &LraOptions) -> anyhow::Result<Self> {
        let by_name = !options.identity_excluded.is_empty();
        for f in options.lr_pairs.iter().chain(&options.targets) {
            anyhow::ensure!(
                by_name || std::path::Path::new(f).is_file(),
                "{f}: the file the lra table was made with (--lr-pairs or --targets) is \
                 not there; run pinto lra again to draw its pairs"
            );
        }
        let input = LraInputArgs {
            from: Some(manifest.to_string_lossy().into()),
            knn_spatial: 5,
            seed: options.seed,
            ..Default::default()
        }
        .resolve()?;
        let run = load_contacts(&input)?;
        let gene_names = run.axis.feature_names().to_vec();
        let index = GeneIndex::build(&gene_names);
        let excluded = if by_name {
            excluded_by_name(&options.identity_excluded, &gene_names)
        } else {
            Self::excluded_from_files(options, &gene_names, &index)?
        };
        // a --link-community table's exposure runs through its community's
        // links alone, read from the run as lra read them
        let community = options
            .link_community
            .map(|k| anyhow::Ok((k, link_community_of_contacts(&input, &run)?)))
            .transpose()?;
        let frame = Self::build(run, (gene_names, index), &excluded)?;
        match community {
            Some((k, of_pair)) => frame.in_link_community(k, &of_pair),
            None => Ok(frame),
        }
    }

    /// These receivers under `--link-community link_community`, each
    /// contact's community in `community_of_pair`: the exposure through
    /// the community's links, its receivers alone in scope
    /// ([`community_scope`], as lra builds it).
    pub fn in_link_community(
        mut self,
        link_community: i64,
        community_of_pair: &[i64],
    ) -> anyhow::Result<Self> {
        // the table was fitted, so its receivers were enough
        let inputs = community_inputs(&self.run.pairs, community_of_pair, link_community, 1)?;
        self.community = Some(community_scope(&inputs, &self.row_of, self.n())?);
        Ok(self)
    }

    /// The contacts a receiver's exposure is summed over: the community's
    /// links under `--link-community`, else every contact.
    pub fn exposure_contacts(&self) -> &Contacts {
        self.community
            .as_ref()
            .map_or(&self.contacts, |c| &c.exposure_contacts)
    }

    /// Whether rows `a` and `b` are contacts the exposure is summed over.
    pub fn joined(&self, a: usize, b: usize) -> bool {
        self.exposure_contacts().adj[a]
            .binary_search(&(b as u32))
            .is_ok()
    }

    /// The genes the identity left out, for a manifest that does not
    /// record them: every pair's of the reference and of the options'
    /// files, read again.
    fn excluded_from_files(
        options: &LraOptions,
        gene_names: &[Box<str>],
        index: &GeneIndex,
    ) -> anyhow::Result<HashSet<usize>> {
        let targets: Vec<Box<str>> = options.targets.iter().map(|t| t.as_str().into()).collect();
        let choices = PairChoices {
            species: clap::ValueEnum::from_str(&options.species, true)
                .unwrap_or(ReferenceSpecies::Auto),
            lr_pairs: options.lr_pairs.as_deref(),
            targets: &targets,
        };
        let (pairs, _) = reference_pairs(&choices, gene_names, index)
            .map_err(|e| e.context("reading the pairs the lra table was made with"))?;
        Ok(signalling_genes(&pairs))
    }

    /// The receivers of `run` (`genes`, its names and their index), the
    /// identity leaving out the genes `excluded`.
    fn build(
        run: RunContacts,
        (gene_names, index): (Vec<Box<str>>, GeneIndex),
        excluded: &HashSet<usize>,
    ) -> anyhow::Result<Self> {
        let (cell, row_of, contacts) = receiver_contacts(&run.coords, &run.pairs);
        anyhow::ensure!(!cell.is_empty(), "no cell has a spatial contact");
        // The depths, in one pass that keeps no gene.
        let (_, log1p_depth) = GeneStore::read(&run.data, &run.axis, &[], &row_of)?;
        let log_depth: Vec<f64> = cell.iter().map(|&c| log1p_depth[c] as f64).collect();
        let (_, bins) = identity_bins(&run.data, &run.axis, excluded, &run.cell_section)?;
        let identity_bin = cell.iter().map(|&c| bins[c]).collect();
        let column_names = run.data.column_names()?;
        Ok(Self {
            size_factor: size_factors(&log_depth),
            identity_bin,
            cell,
            row_of,
            contacts,
            column_names,
            gene_names,
            index,
            run,
            community: None,
        })
    }

    /// The receivers of `run`, the identity leaving out the genes named
    /// `excluded` ([`LraOptions::identity_excluded`]). For tests, which
    /// have no reference.
    #[cfg(test)]
    pub fn without_reference(run: RunContacts, excluded: &[String]) -> anyhow::Result<Self> {
        let gene_names = run.axis.feature_names().to_vec();
        let index = GeneIndex::build(&gene_names);
        let excluded = excluded_by_name(excluded, &gene_names);
        Self::build(run, (gene_names, index), &excluded)
    }

    /// Receivers.
    pub fn n(&self) -> usize {
        self.cell.len()
    }

    /// `ligand`–`receptor` (as a table names them) on the receivers,
    /// exposed above `threshold` (the table's), else above the one the
    /// estimate would choose.
    pub fn pair(
        &self,
        ligand: &str,
        receptor: &str,
        threshold: Option<f64>,
    ) -> anyhow::Result<PairMap> {
        // The names resolved on the data's genes as the analysis
        // resolved the reference's ([`subunit_genes`]).
        let genes = |name: &str| {
            subunit_genes(name, &self.gene_names, &self.index)
                .ok_or_else(|| anyhow::anyhow!("{name}: not in the run's data"))
        };
        let lig = self.complex_counts(&genes(ligand)?)?;
        let rec = self.complex_counts(&genes(receptor)?)?;
        Ok(pair_map_in(
            self.exposure_contacts(),
            self.community.as_ref().map(|c| c.in_community.as_slice()),
            &self.size_factor,
            &self.identity_bin,
            (&lig, &rec),
            threshold,
        ))
    }

    /// Per receiver row, the count of a complex's scarcest subunit;
    /// splice channels of one gene summed, as [`GeneStore`] reads them.
    fn complex_counts(&self, genes: &[usize]) -> anyhow::Result<Vec<f64>> {
        let n = self.n();
        let axis = &self.run.axis;
        let mut cols = Vec::with_capacity(genes.len());
        for &g in genes {
            let mut v = vec![0.0f64; n];
            // One row at a time: reading several rows of a `.zarr.zip`
            // at once has returned corrupt values (see view::gene).
            for r in (0..axis.n_rows()).filter(|&r| axis.feature_of_row(r) == g) {
                let row = self.run.data.read_rows_csr(std::iter::once(r))?;
                for (&col, &x) in row.col_indices().iter().zip(row.values()) {
                    if let Some(&i) = self.row_of.get(col).filter(|&&i| i != usize::MAX) {
                        v[i] += x as f64;
                    }
                }
            }
            cols.push(v);
        }
        Ok(match cols.len() {
            1 => cols.pop().expect("one"),
            _ => scarcest_subunit(&cols),
        })
    }
}
