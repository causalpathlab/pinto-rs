//! The receivers as the analysis sees them, and one pair on them: each
//! receiver's contact exposure to the ligand, whether it counts as exposed,
//! whether its identity bin makes it receptor-competent, and whether it
//! carries the ligand itself. [`super::tissue::Tissue::build`] takes its
//! receivers, contacts and size factors from here and the estimate its
//! exposure, so `pinto view` ([`RunFrame`], [`pair_map`]) draws a pair as
//! the table saw it.

use super::exposure::{upper_median, Contacts};

/// Receivers (cells with coordinates and a contact to another such cell)
/// by data column, and each data column's row (`usize::MAX` for none).
pub fn receiver_rows(
    coords: &[Option<(f32, f32)>],
    pairs: &[(usize, usize)],
) -> (Vec<usize>, Vec<usize>) {
    let cell = super::tissue::receiver_cells(coords, pairs);
    let mut row_of = vec![usize::MAX; coords.len()];
    for (r, &c) in cell.iter().enumerate() {
        row_of[c] = r;
    }
    (cell, row_of)
}

/// [`receiver_rows`], and the contacts between rows.
pub fn receiver_contacts(
    coords: &[Option<(f32, f32)>],
    pairs: &[(usize, usize)],
) -> (Vec<usize>, Vec<usize>, Contacts) {
    let (cell, row_of) = receiver_rows(coords, pairs);
    let row_pairs: Vec<(usize, usize)> = pairs
        .iter()
        .filter(|&&(i, j)| row_of[i] != usize::MAX && row_of[j] != usize::MAX)
        .map(|&(i, j)| (row_of[i], row_of[j]))
        .collect();
    let contacts = Contacts::from_pairs(cell.len(), &row_pairs);
    (cell, row_of, contacts)
}

/// Each receiver's depth over the median depth, from `log1p(total count)`.
pub fn size_factors(log_depth: &[f64]) -> Vec<f64> {
    let depth: Vec<f64> = log_depth.iter().map(|l| l.exp_m1().max(1.0)).collect();
    let med = upper_median(&mut depth.clone()).unwrap_or(1.0).max(1.0);
    depth.iter().map(|d| d / med).collect()
}

/// Each row's ligand over its size factor, and its contact dose: that
/// summed over its contacts, which the exposure threshold reads.
pub fn contact_dose(
    contacts: &Contacts,
    size_factor: &[f64],
    ligand_counts: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let lnorm: Vec<f64> = ligand_counts
        .iter()
        .zip(size_factor)
        .map(|(c, s)| c / s)
        .collect();
    let a = contacts.sum(&lnorm);
    (lnorm, a)
}

// * For the viewer

#[cfg(feature = "view")]
pub use frame::RunFrame;
#[cfg(all(feature = "view", test))]
pub use map::table_threshold;
#[cfg(feature = "view")]
pub use map::{pair_map, Class, PairMap};

#[cfg(feature = "view")]
mod map {
    use super::super::dose;
    use super::super::exposure::{competence, Contacts};
    use super::super::options::{self, EffectOpts};
    use super::super::INTERNALS;
    use super::contact_dose;

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
    }

    /// `pair` on the receivers: `ligand_counts` and `receptor_counts` per row
    /// (a complex as its scarcest subunit), exposed above `threshold` (a
    /// table's, [`table_threshold`]), or above the threshold the estimate
    /// would choose ([`dose::choose_threshold`]) when none is given.
    pub fn pair_map(
        contacts: &Contacts,
        size_factor: &[f64],
        identity_bin: &[usize],
        ligand_counts: &[f64],
        receptor_counts: &[f64],
        threshold: Option<f64>,
    ) -> PairMap {
        let (_, a) = contact_dose(contacts, size_factor, ligand_counts);
        let tau = table_threshold(&a, threshold);
        let detected: Vec<bool> = receptor_counts.iter().map(|&r| r > 0.0).collect();
        let grid = &options::COMPETENCE_GRID;
        let level = grid[grid.len() / 2];
        PairMap {
            exposed: dose::exposed(&a, tau),
            exposure: a,
            threshold: tau,
            competence: level,
            competent: competence(identity_bin, &detected, level),
            sender: ligand_counts.iter().map(|&l| l > 0.0).collect(),
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

        /// Receivers in each [`Class`], in [`Class::ALL`]'s order.
        pub fn counts(&self) -> [usize; 5] {
            let mut out = [0; 5];
            for i in 0..self.n() {
                out[self.class(i) as usize] += 1;
            }
            out
        }
    }
}

#[cfg(feature = "view")]
mod frame {
    use super::super::genes::GeneStore;
    use super::super::pairs::{reference_pairs, signalling_genes, subunit_genes};
    use super::super::pipeline::identity_bins;
    use super::super::PairChoices;
    use super::{pair_map, receiver_contacts, size_factors, PairMap};
    use crate::lr_activity::args::{LraInputArgs, ReferenceSpecies};
    use crate::lr_activity::input::{load_contacts, Contacts as RunContacts};
    use crate::util::common::HashSet;
    use crate::util::metadata::LraOptions;
    use data_beans::utilities::name_matching::GeneIndex;

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
        contacts: super::Contacts,
        pub size_factor: Vec<f64>,
        pub identity_bin: Vec<usize>,
        /// Each data column's name.
        pub column_names: Vec<Box<str>>,
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
            Self::build(run, (gene_names, index), &excluded)
        }

        /// The genes the identity left out, for a manifest that does not
        /// record them: every pair's of the reference and of the options'
        /// files, read again (the background programmes drawn again with the seed).
        fn excluded_from_files(
            options: &LraOptions,
            gene_names: &[Box<str>],
            index: &GeneIndex,
        ) -> anyhow::Result<HashSet<usize>> {
            let targets: Vec<Box<str>> =
                options.targets.iter().map(|t| t.as_str().into()).collect();
            let choices = PairChoices {
                species: clap::ValueEnum::from_str(&options.species, true)
                    .unwrap_or(ReferenceSpecies::Auto),
                lr_pairs: options.lr_pairs.as_deref(),
                targets: &targets,
            };
            let (pairs, _) = reference_pairs(&choices, gene_names, index)
                .map_err(|e| e.context("reading the pairs the lra table was made with"))?;
            Ok(signalling_genes(&pairs, gene_names.len(), options.seed).0)
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
            let (_, log1p_depth) = GeneStore::read(&run.data, &run.axis, &[], &row_of, None)?;
            let log_depth: Vec<f64> = cell.iter().map(|&c| log1p_depth[c] as f64).collect();
            let (_, bins) = identity_bins(&run.data, &run.axis, excluded, &run.cell_section, None)?;
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
            // resolved the reference's ([`super::pairs::subunit_genes`]).
            let genes = |name: &str| {
                subunit_genes(name, &self.gene_names, &self.index)
                    .ok_or_else(|| anyhow::anyhow!("{name}: not in the run's data"))
            };
            let lig = self.complex_counts(&genes(ligand)?)?;
            let rec = self.complex_counts(&genes(receptor)?)?;
            Ok(pair_map(
                &self.contacts,
                &self.size_factor,
                &self.identity_bin,
                &lig,
                &rec,
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
                _ => super::super::pairs::scarcest_subunit(&cols),
            })
        }
    }
}
