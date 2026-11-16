//! One feature's level per cell, drawn on the map instead of communities.
//!
//! Two sources:
//!
//! - **observed**: `ln(1 + 1e4 · count / total)` from the run's data files
//!   ([`log_cp10k`], `total` the cell's count over every row), so a cell's
//!   level is its composition and not its depth. A row is read at a time
//!   through the same reader the fits use;
//! - **expected**: what the community model predicts, `Σ_k p_ik · μ_gk`,
//!   from the level's propensity and `feature_community` rates. The rates
//!   carry a per-feature weight, so only the relative level means anything.
//!
//! Either way the values become a one-"community" table with its own
//! pyramid, so the renderer draws them like a single community's
//! propensity.

use super::data::{Communities, Geometry, NO_CLUSTER};
use super::index::{Grid, Pyramid};
use super::markers::{symbol, FeatureRates};
use crate::util::common::*;
use crate::util::coverage::{measured_mass, Coverage};
use crate::util::input::read_expr_data;
use legume_numeric::matrix::utils::generate_minibatch_intervals;

/// What an observed level is, as legends and exports name it: `ln(1 + x)`
/// of counts per ten thousand of the cell's total ([`log_cp10k`]).
pub const OBSERVED_SCALE: &str = "log1p CP10k";

#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Observed,
    Expected,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Observed => "observed",
            Source::Expected => "expected",
        }
    }

    pub fn other(self) -> Source {
        match self {
            Source::Observed => Source::Expected,
            Source::Expected => Source::Observed,
        }
    }
}

/// A feature's values on the map.
pub struct GeneMap {
    /// Feature name as the data or rates table spell it.
    pub feature: Box<str>,
    pub source: Source,
    /// The value drawn at the top of the colour ramp; higher values clip.
    pub top: f32,
    /// Percentile of the positive values `top` sits at.
    pub clip: f32,
    /// The values, scaled to `max`, as a single community.
    pub comm: Communities,
    pub pyramid: Pyramid,
    /// What the ramp measures, when it is not the feature's own level
    /// (an lra pair's exposure).
    pub label: Option<Box<str>>,
}

impl GeneMap {
    /// `values` per cell; with `measured`, a cell it marks false has no
    /// level (its file never measured the feature), drawn as such.
    pub(super) fn new(
        feature: &str,
        source: Source,
        values: &[f32],
        measured: Option<&[bool]>,
        clip: f32,
        grid: &Grid,
    ) -> Self {
        let top = top_of(values, clip);
        let scaled: Vec<u8> = values
            .iter()
            .map(|&v| {
                if top > 0. {
                    ((v / top).min(1.) * 255.).round() as u8
                } else {
                    0
                }
            })
            .collect();
        let mut comm = Communities::single(feature, scaled);
        if let Some(measured) = measured {
            for (c, _) in comm.cluster.iter_mut().zip(measured).filter(|(_, &m)| !m) {
                *c = NO_CLUSTER;
            }
            comm.sizes = vec![measured.iter().filter(|&&m| m).count()];
        }
        let pyramid = Pyramid::build(grid, &comm);
        GeneMap {
            feature: feature.into(),
            source,
            top,
            clip,
            comm,
            pyramid,
            label: None,
        }
    }

    /// The same map, its ramp titled `label`.
    pub(super) fn labelled(self, label: &str) -> Self {
        GeneMap {
            label: Some(label.into()),
            ..self
        }
    }

    /// Model-expected level of `feature` in every cell of `comm`.
    pub fn expected(
        feature: &str,
        rates: &FeatureRates,
        comm: &Communities,
        clip: f32,
        grid: &Grid,
    ) -> anyhow::Result<Self> {
        let g = find(&rates.names, feature)
            .ok_or_else(|| anyhow::anyhow!("no feature {feature:?} in feature_community"))?;
        let k = comm.k.min(rates.rates.ncols());
        let mu: Vec<f32> = (0..k).map(|c| rates.rates[(g, c)]).collect();
        let values: Vec<f32> = comm
            .prop
            .par_chunks(comm.k.max(1))
            .map(|row| {
                row.iter()
                    .zip(&mu)
                    .map(|(&q, &m)| q as f32 / 255. * m)
                    .sum()
            })
            .collect();
        Ok(Self::new(
            &rates.names[g],
            Source::Expected,
            &values,
            None,
            clip,
            grid,
        ))
    }

    /// Legend label of the ramp's top: its value, and the percentile it
    /// clips at.
    pub fn top_label(&self) -> String {
        let top = significant(self.top, 3);
        if self.clip >= 100. {
            top
        } else {
            format!("{top}+ (p{})", self.clip)
        }
    }

    /// Whether this map fell back to the model because the observed level was
    /// asked for and the run has no data file.
    pub fn fell_back(&self, asked: Source) -> bool {
        asked == Source::Observed && self.source == Source::Expected
    }

    /// Legend title: the feature's symbol and what the ramp measures.
    pub fn title(&self) -> String {
        if let Some(label) = &self.label {
            return label.to_string();
        }
        let what = match self.source {
            Source::Observed => OBSERVED_SCALE,
            Source::Expected => "expected, relative",
        };
        format!("{} {what}", symbol(&self.feature))
    }
}

/// `v` to `digits` significant digits, without an exponent: 1.61, 0.0123.
pub fn significant(v: f32, digits: i32) -> String {
    if v == 0. || !v.is_finite() {
        return format!("{v}");
    }
    let decimals = (digits - 1 - v.abs().log10().floor() as i32).max(0) as usize;
    format!("{v:.decimals$}")
}

/// The `clip` percentile of the positive values: a sparse feature's few
/// extreme cells should not push every other cell to the bottom of the
/// ramp. 100 is the maximum. A linear-time selection rather than
/// `legume_numeric`'s `quantiles` (a full sort of a copy): it runs on every
/// gene switch.
fn top_of(values: &[f32], clip: f32) -> f32 {
    let mut positive: Vec<f32> = values.iter().copied().filter(|&v| v > 0.).collect();
    if positive.is_empty() {
        return 0.;
    }
    let at = ((positive.len() - 1) as f32 * clip.clamp(0., 100.) / 100.).round() as usize;
    *positive.select_nth_unstable_by(at, f32::total_cmp).1
}

/// The run's expression data, matched to the map's cells.
pub struct Expression {
    data: SparseIoVec,
    features: Vec<Box<str>>,
    /// Map cell of each data column, if it is on the map.
    cell_of: Vec<Option<u32>>,
    /// Which file measured which feature, when some file lacks some.
    coverage: Option<Coverage>,
    /// Total count of each data column over every row it has: what
    /// [`log_cp10k`] scales its counts by.
    totals: Vec<f32>,
}

impl Expression {
    pub fn open(files: &[Box<str>], geom: &Geometry) -> anyhow::Result<Self> {
        let data = read_expr_data(files)?;
        let features = data.row_names()?;
        let cell_of = data
            .column_names()?
            .iter()
            .map(|name| geom.index.get(name).map(|&i| i as u32))
            .collect();
        let coverage = Coverage::of(&data);
        let totals = column_totals(&data)?;
        Ok(Expression {
            data,
            features,
            cell_of,
            coverage,
            totals,
        })
    }

    /// Mean observed level ([`log_cp10k`]) of each of `features` over each of `k`
    /// groups' cells (`group`, one per map cell; `NO_CLUSTER` for none):
    /// features × groups, row-major. `None` for a feature not in the data.
    pub fn group_means(
        &self,
        features: &[&str],
        group: &[u16],
        k: usize,
    ) -> anyhow::Result<Vec<Option<Vec<f32>>>> {
        // Cells per file and group, among the data's columns on the map; a
        // feature's means divide by the files that measured it.
        let cov = self.coverage.as_ref();
        let mut n_bk = Mat::zeros(cov.map_or(1, Coverage::n_files), k);
        for (col, cell) in self.cell_of.iter().enumerate() {
            let Some(&c) = cell.and_then(|i| group.get(i as usize)) else {
                continue;
            };
            if (c as usize) < k {
                n_bk[(cov.map_or(0, |v| v.file(col)), c as usize)] += 1.;
            }
        }
        // One row per read: data-beans 0.6.12 reading many rows of a
        // `.zarr.zip` at once sometimes returns corrupt indices or values;
        // single rows read back the same every time.
        features
            .iter()
            .map(|f| {
                let Some(g) = find(&self.features, f) else {
                    return Ok(None);
                };
                let row = self.data.read_rows_csr(std::iter::once(g))?;
                let mut sum = vec![0f32; k];
                for (_, col, &count) in row.triplet_iter() {
                    let cell = self.cell_of.get(col).copied().flatten();
                    let c = cell.and_then(|i| group.get(i as usize));
                    if let Some(s) = c.and_then(|&c| sum.get_mut(c as usize)) {
                        *s += log_cp10k(count, self.totals[col]);
                    }
                }
                let n = measured_mass(&n_bk, cov, g);
                Ok(Some(
                    sum.iter().zip(&n).map(|(s, n)| s / n.max(1.)).collect(),
                ))
            })
            .collect()
    }

    /// Observed level ([`log_cp10k`]) of data row `g` in each of the map's
    /// `n` cells; 0 where the cell has no count of it, or no count at all.
    pub(super) fn levels(&self, g: usize, n: usize) -> anyhow::Result<Vec<f32>> {
        let row = self.data.read_rows_csr(std::iter::once(g))?;
        let mut values = vec![0f32; n];
        for (&col, &count) in row.col_indices().iter().zip(row.values()) {
            if let Some(i) = self.cell_of[col] {
                values[i as usize] = log_cp10k(count, self.totals[col]);
            }
        }
        Ok(values)
    }

    /// Observed level ([`log_cp10k`]) of `feature` in every cell on the map.
    pub fn observed(
        &self,
        feature: &str,
        geom: &Geometry,
        clip: f32,
        grid: &Grid,
    ) -> anyhow::Result<GeneMap> {
        let g = find(&self.features, feature)
            .ok_or_else(|| anyhow::anyhow!("no feature {feature:?} in the data files"))?;
        let values = self.levels(g, geom.n())?;
        // A cell no data file holds, or whose file never measured the
        // feature, has no level.
        let cov = self.coverage.as_ref();
        let mut measured = vec![false; geom.n()];
        for (col, cell) in self.cell_of.iter().enumerate() {
            if let Some(i) = cell {
                measured[*i as usize] = cov.is_none_or(|c| c.measured(g, col));
            }
        }
        Ok(GeneMap::new(
            &self.features[g],
            Source::Observed,
            &values,
            Some(&measured),
            clip,
            grid,
        ))
    }
}

/// Each column's total count over every row of `data`, read a block of
/// columns at a time: one pass, and never more than a block in memory.
/// The total is over every row of the cell as the viewer reads it.
fn column_totals(data: &SparseIoVec) -> anyhow::Result<Vec<f32>> {
    let blocks = generate_minibatch_intervals(data.num_columns(), data.num_rows().max(1), None);
    let totals = blocks
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<Vec<f32>> {
            let x = data.read_columns_csc(lb..ub)?;
            Ok((0..x.ncols())
                .map(|j| x.col(j).values().iter().map(|&v| f64::from(v)).sum::<f64>() as f32)
                .collect())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(totals.into_iter().flatten().collect())
}

/// Position of `feature` in `names`: the exact name, or else the first
/// whose symbol (`ENSG…_GENE1` → `GENE1`) matches.
pub fn find(names: &[Box<str>], feature: &str) -> Option<usize> {
    names
        .iter()
        .position(|n| n.as_ref() == feature)
        .or_else(|| names.iter().position(|n| symbol(n) == feature))
}
