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

use super::color::{to_linear, Rgb, Theme};
use super::data::Communities;
use super::markers::{self, FeatureRates};
use super::pdf::{chart_fits, chart_fonts, chart_page, CHART_PAGE};
use legume_tui::chart::heatmap::{self as chart, ByFn, Levels};
use legume_tui::chart::heatmap::{Heat, HeatStyle};
use legume_tui::export::Page;

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
        let cols = self.group_names(comm);
        chart::tsv(what, "gene", &rows, &marks, &cols, &self.values)
    }

    /// The heatmap as a page for a PDF: `title`, a cell per gene and group
    /// on the diverging scale with each group's colour over its column,
    /// and the genes, groups and key as text. The page is as large as the
    /// chart needs to name every gene and group. With nothing to draw, or too
    /// much for a page, why not.
    pub fn figure(
        &self,
        comm: &Communities,
        palette: &[Rgb],
        title: &str,
        theme: Theme,
    ) -> Result<Page, &'static str> {
        let rows: Vec<String> = self.genes.iter().map(|(_, s, _)| s.clone()).collect();
        let cols = self.group_names(comm);
        let swatch: Vec<_> = self.groups.iter().map(|&c| to_linear(palette[c])).collect();
        let style = HeatStyle {
            missing: to_linear(theme.dimmed()),
            ..HeatStyle::for_theme(theme.page())
        };
        let heat = Heat {
            title,
            rows: &rows,
            cols: &cols,
            col_swatch: Some(&swatch),
            z: &self.z,
            clip: CLIP,
            bold_row: None,
        };
        let (font, bold) = chart_fonts();
        let too_large = |_| "too large for a page: fewer genes per group (-) draw one";
        // A line a gene: too many for any page is refused before laying out.
        let w = CHART_PAGE.px(1.).0;
        chart_fits(w, (rows.len() + 3) * font.line_height()).map_err(too_large)?;
        let (w, h) = chart::heat_size(&heat, font, &style, w);
        chart_fits(w, h).map_err(too_large)?;
        let (page, drawn) = chart_page((w, h), theme, |frame| {
            chart::draw_heat(frame, &heat, font, bold, theme.page(), &style)
        });
        drawn.map(|_| page)
    }

    /// Each column's group, by name.
    fn group_names(&self, comm: &Communities) -> Vec<String> {
        self.groups.iter().map(|&c| comm.name(c)).collect()
    }
}
