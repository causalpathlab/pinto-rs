//! Top genes × groups heatmap.
//!
//! A gene marks the group it leads by the widest margin: its level there
//! minus its highest level in any other group. Ranking by margin, not by
//! fold over the mean of the rest, makes each row peak in its own column, so
//! the map reads as a diagonal; a tie peaks nowhere and is left out.
//!
//! Genes are chosen in two stages. The model's levels, ln(1 + rate), rank
//! every gene cheaply and propose [`CANDIDATES`] × N per group; the groups'
//! mean observed ln(1 + count) then ranks those candidates again, and each
//! group keeps its top N. Without readable counts the model's levels do
//! both. Rows are z-scored across the groups and clipped to ±[`CLIP`], so a
//! few outliers do not wash out the rest.

use super::data::Communities;
use super::markers::{self, FeatureRates};

/// Largest |z| drawn; beyond it colours saturate.
pub const CLIP: f32 = 2.5;

/// Candidates the model proposes per gene finally kept.
pub const CANDIDATES: usize = 5;

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
        let model = |f: usize, j: usize| rates.rates[(f, groups[j])].max(0.).ln_1p();

        // Stage one: the model proposes candidates from every gene.
        let proposed = by_margin(rates.names.len(), cols, &model, per_group * CANDIDATES);
        let names: Vec<&str> = proposed
            .iter()
            .map(|&(f, _)| rates.names[f].as_ref())
            .collect();

        // Stage two: the candidates' own levels, observed when possible.
        let observed = observe(&names).filter(|m| m.iter().any(Option::is_some));
        let (rows, level): (Vec<usize>, Vec<Vec<f32>>) = match &observed {
            Some(means) => proposed
                .iter()
                .zip(means)
                .filter_map(|(&(f, _), m)| {
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
                .map(|&(f, _)| (f, (0..cols).map(|j| model(f, j)).collect()))
                .unzip(),
        };
        let chosen = by_margin(rows.len(), cols, &|r, j| level[r][j], per_group);

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
        let z = zscores(&values, cols);
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
        let cols = self.groups.len();
        let what = if self.observed {
            "mean ln(1+count)"
        } else {
            "model ln(1+rate)"
        };
        let mut out = format!("# {what}\ngene\tmarks");
        for &c in &self.groups {
            out.push('\t');
            out.push_str(&comm.name(c));
        }
        out.push('\n');
        for (r, (name, _, c)) in self.genes.iter().enumerate() {
            out.push_str(&format!("{name}\t{}", comm.name(*c)));
            for v in &self.values[r * cols..(r + 1) * cols] {
                out.push_str(&format!("\t{v:.4}"));
            }
            out.push('\n');
        }
        out
    }
}

/// Of `rows` × `cols` levels, each column's top `per_col` rows by margin
/// over their next-highest column, among the rows that peak there: as
/// (row, column), columns in order, best margin first.
pub fn by_margin(
    rows: usize,
    cols: usize,
    level: &dyn Fn(usize, usize) -> f32,
    per_col: usize,
) -> Vec<(usize, usize)> {
    // Each row's peak column and its lead over the runner-up.
    let mut lead: Vec<Vec<(usize, f32)>> = vec![Vec::new(); cols];
    for r in 0..rows {
        let (mut top, mut first, mut second) = (0, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for j in 0..cols {
            let v = level(r, j);
            if v > first {
                (top, second, first) = (j, first, v);
            } else if v > second {
                second = v;
            }
        }
        let margin = if cols > 1 { first - second } else { first };
        if margin > 0. && margin.is_finite() {
            lead[top].push((r, margin));
        }
    }
    let mut out = Vec::new();
    for (j, mut rows) in lead.into_iter().enumerate() {
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        out.extend(rows.into_iter().take(per_col).map(|(r, _)| (r, j)));
    }
    out
}

/// Each row of `values` (`cols` wide) as z-scores, clipped to ±[`CLIP`]; a
/// flat row is 0, NaN stays NaN.
pub fn zscores(values: &[f32], cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len());
    for row in values.chunks(cols.max(1)) {
        let finite: Vec<f32> = row.iter().copied().filter(|v| v.is_finite()).collect();
        let n = finite.len().max(1) as f32;
        let mean = finite.iter().sum::<f32>() / n;
        let sd = (finite.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
        out.extend(row.iter().map(|&v| {
            if !v.is_finite() {
                f32::NAN
            } else if sd > 0. {
                ((v - mean) / sd).clamp(-CLIP, CLIP)
            } else {
                0.
            }
        }));
    }
    out
}
