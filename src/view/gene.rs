//! One feature's level per cell, drawn on the map instead of communities.
//!
//! Two sources:
//!
//! - **observed**: `ln(1 + count)` from the run's data files, read one row
//!   at a time through the same reader the fits use;
//! - **expected**: what the community model predicts, `Σ_k p_ik · μ_gk`,
//!   from the level's propensity and `feature_community` rates. The rates
//!   carry a per-feature weight, so only the relative level means anything.
//!
//! Either way the values become a one-"community" table with its own
//! pyramid, so the renderer draws them like a single community's
//! propensity.

use super::data::{Communities, Geometry};
use super::index::{Grid, Pyramid};
use super::markers::{symbol, FeatureRates};
use crate::util::common::*;
use crate::util::input::read_expr_data;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    Observed,
    Expected,
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
}

impl GeneMap {
    fn new(feature: &str, source: Source, values: &[f32], clip: f32, grid: &Grid) -> Self {
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
        let comm = Communities::single(feature, scaled);
        let pyramid = Pyramid::build(grid, &comm);
        GeneMap {
            feature: feature.into(),
            source,
            top,
            clip,
            comm,
            pyramid,
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

    /// Legend title: the feature's symbol and what the ramp measures.
    pub fn title(&self) -> String {
        let what = match self.source {
            Source::Observed => "ln(1+count)",
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
/// ramp. 100 is the maximum.
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
        Ok(Expression {
            data,
            features,
            cell_of,
        })
    }

    /// Observed `ln(1 + count)` of `feature` in every cell on the map.
    pub fn observed(
        &self,
        feature: &str,
        geom: &Geometry,
        clip: f32,
        grid: &Grid,
    ) -> anyhow::Result<GeneMap> {
        let g = find(&self.features, feature)
            .ok_or_else(|| anyhow::anyhow!("no feature {feature:?} in the data files"))?;
        let row = self.data.read_rows_csr(std::iter::once(g))?;
        let mut values = vec![0f32; geom.n()];
        for (&col, &count) in row.col_indices().iter().zip(row.values()) {
            if let Some(i) = self.cell_of[col] {
                values[i as usize] = count.ln_1p();
            }
        }
        Ok(GeneMap::new(
            &self.features[g],
            Source::Observed,
            &values,
            clip,
            grid,
        ))
    }
}

/// Position of `feature` in `names`: the exact name, or else the first
/// whose symbol (`ENSG…_CD3E` → `CD3E`) matches.
pub fn find(names: &[Box<str>], feature: &str) -> Option<usize> {
    names
        .iter()
        .position(|n| n.as_ref() == feature)
        .or_else(|| names.iter().position(|n| symbol(n) == feature))
}
