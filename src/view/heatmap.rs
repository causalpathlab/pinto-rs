//! Top genes × groups heatmap.
//!
//! A gene marks the group it leads by the widest margin: its level there
//! minus its highest level in any other group. Ranking by margin, not by
//! fold over the mean of the rest, makes each row peak in its own column, so
//! the map reads as a diagonal; a tie peaks nowhere and is left out.
//!
//! Genes are chosen in two stages. The model's levels, ln(1 + rate), rank
//! every gene cheaply and propose [`CANDIDATES`](chart::CANDIDATES) × N per
//! group; the groups' mean observed ln(1 + count) then ranks those
//! candidates again, and each group keeps its top N. Without readable counts the model's levels do
//! both. Rows are z-scored across the groups and clipped to ±[`CLIP`], so a
//! few outliers do not wash out the rest. The ranking, the z-scores and the
//! table are legume-tui's chart.

use super::data::Communities;
use super::markers::{self, FeatureRates};
use legume_tui::chart::heatmap::{self as chart, ByFn, Levels};

pub use chart::CLIP;

pub struct Heatmap {
    /// Columns: groups, largest first.
    pub groups: Vec<usize>,
    /// Rows: feature name, symbol, and the group it marks.
    pub genes: Vec<(Box<str>, String, usize)>,
    /// What is shown, rows × columns: mean ln(1 + count), or the model's
    /// ln(1 + rate).
    pub values: Vec<f32>,
    /// `values` z-scored per row and clipped to ±[`CLIP`].
    pub z: Vec<f32>,
    /// Whether `values` are observed counts.
    pub observed: bool,
}

/// Observed mean ln(1 + count) of each named feature over every group of
/// the grouping; `None` for a feature not in the data, or no data at all.
pub type Observe<'a> = dyn FnOnce(&[&str]) -> Option<Vec<Option<Vec<f32>>>> + 'a;

impl Heatmap {
    /// Up to `per_group` genes for each non-empty group of `comm`, ranked
    /// by `observe`'s counts when it has them.
    pub fn build(
        rates: &FeatureRates,
        comm: &Communities,
        per_group: usize,
        observe: Box<Observe>,
    ) -> Self {
        let groups: Vec<usize> = comm
            .by_size
            .iter()
            .copied()
            .filter(|&c| c < rates.rates.ncols())
            .collect();
        let cols = groups.len();
        let model = ByFn {
            rows: rates.names.len(),
            cols,
            f: |f: usize, j: usize| rates.rates[(f, groups[j])].max(0.).ln_1p(),
        };

        // Stage one: the model proposes candidates from every gene.
        let proposed = chart::candidates(&model, per_group);
        let names: Vec<&str> = proposed.iter().map(|&f| rates.names[f].as_ref()).collect();

        // Stage two: the candidates' own levels, observed when possible.
        let observed = observe(&names).filter(|m| m.iter().any(Option::is_some));
        let (rows, level): (Vec<usize>, Vec<Vec<f32>>) = match &observed {
            Some(means) => proposed
                .iter()
                .zip(means)
                .filter_map(|(&f, m)| {
                    let m = m.as_ref()?;
                    Some((
                        f,
                        groups
                            .iter()
                            .map(|&c| m.get(c).copied().unwrap_or(0.))
                            .collect(),
                    ))
                })
                .unzip(),
            None => proposed
                .iter()
                .map(|&f| (f, (0..cols).map(|j| model.at(f, j)).collect()))
                .unzip(),
        };
        let shown = ByFn {
            rows: rows.len(),
            cols,
            f: |r: usize, j: usize| level[r][j],
        };
        let chosen = chart::by_margin(&shown, per_group);

        let genes = chosen
            .iter()
            .map(|&(r, j)| {
                let name = rates.names[rows[r]].clone();
                let symbol = markers::symbol(&name).to_string();
                (name, symbol, groups[j])
            })
            .collect();
        let values: Vec<f32> = chosen
            .iter()
            .flat_map(|&(r, _)| level[r].iter().copied())
            .collect();
        let mut z = values.clone();
        chart::zscore_rows(&mut z, cols, CLIP);
        Heatmap {
            groups,
            genes,
            values,
            z,
            observed: observed.is_some(),
        }
    }

    /// Tab-separated: gene, the group it marks, then the value per group.
    pub fn tsv(&self, comm: &Communities) -> String {
        let what = if self.observed {
            "mean ln(1+count)"
        } else {
            "model ln(1+rate)"
        };
        let rows: Vec<String> = self.genes.iter().map(|(n, _, _)| n.to_string()).collect();
        let marks: Vec<String> = self.genes.iter().map(|&(_, _, c)| comm.name(c)).collect();
        let cols: Vec<String> = self.groups.iter().map(|&c| comm.name(c)).collect();
        chart::tsv(what, "gene", &rows, &marks, &cols, &self.values)
    }
}
