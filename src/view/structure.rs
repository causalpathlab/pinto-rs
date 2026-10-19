//! Structure plot: every cell's community mixture as a stacked bar, as
//! admixture plots show ancestry and topic models show topic shares,
//! through legume-tui's chart.
//!
//! Communities stack in order of overall prevalence, most prevalent at the
//! bottom; cells are grouped into panels (batches, or a lupin round's cell
//! types or clusters), and within a panel ordered by their dominant
//! community's rank, then by its share, highest first. A pixel column
//! averages the cells that fall on it.

use super::color::{to_linear, Rgb, Theme};
use super::data::{Communities, NO_CLUSTER};
use super::pdf::{chart_fits, chart_page, CHART_PAGE};
use super::render::Frame;
use legume_tui::chart::structure::{
    self as chart, BarStyle, Figure, Grouping, Layout, Names, Span, Weights,
};
use legume_tui::export::Page;

/// The bars of one level's communities, in panels.
pub struct Structure(Layout);

/// A rendered plot: the picture, and where its panels and communities are.
pub struct Drawn {
    pub frame: Frame,
    bars: chart::Drawn,
}

impl Drawn {
    /// The community at pixel `(x, y)`.
    pub fn at(&self, x: usize, y: usize) -> Option<usize> {
        self.bars.at(x, y).map(|(_, c)| c)
    }

    /// Where each panel is drawn, with its name and cells.
    pub fn panels(&self) -> &[Span] {
        &self.bars.spans
    }
}

/// The cells that have a propensity row.
fn valid(comm: &Communities) -> Vec<bool> {
    comm.cluster.iter().map(|&c| c != NO_CLUSTER).collect()
}

fn weights<'a>(comm: &'a Communities, valid: Option<&'a [bool]>) -> Weights<'a, u8> {
    Weights {
        k: comm.k,
        w: &comm.prop,
        valid,
    }
}

impl Structure {
    /// Bars from `comm`'s propensities; panels from `panel_of` (one per
    /// cell, [`NO_CLUSTER`] for none), named `names`, in `order`. Cells
    /// without a propensity row are left out.
    pub fn build(comm: &Communities, panel_of: &[u16], names: &[String], order: &[usize]) -> Self {
        let valid = valid(comm);
        let grouping = Grouping {
            of: panel_of,
            names,
            order: Some(order),
        };
        Structure(Layout::build(&weights(comm, Some(&valid)), &grouping))
    }

    /// Each panel as a plot of its own, by name, on the same stack.
    pub fn into_panels(self) -> Vec<(String, Structure)> {
        self.0
            .into_panels()
            .into_iter()
            .map(|(name, l)| (name, Structure(l)))
            .collect()
    }

    /// What the plot shows: its panels, and the communities from the
    /// bottom of a bar to the top.
    pub fn layout(&self) -> &Layout {
        &self.0
    }

    /// The plot at `w × h` pixels. With `focus`, communities not in it
    /// are drawn in the dimmed tissue colour.
    pub fn render(
        &self,
        comm: &Communities,
        palette: &[Rgb],
        (w, h): (usize, usize),
        focus: Option<&[bool]>,
        theme: Theme,
    ) -> Drawn {
        let bg = theme.background();
        // Encoded exactly, so the bars come back as the palette's own bytes.
        let mut frame = legume_tui::map::raster::Frame::blank_exact(w, h, bg);
        let palette: Vec<_> = palette.iter().map(|&c| to_linear(c)).collect();
        let style = BarStyle {
            palette: &palette,
            focus: focus.map(|f| (f, to_linear(theme.dimmed()))),
            gap: chart::GAP,
            hits: true,
        };
        // The layout chose the rows; drawing only weighs them.
        let bars = chart::bars(
            &mut frame,
            [0, 0, w, h],
            &self.0,
            &weights(comm, None),
            &style,
        );
        Drawn {
            frame: Frame::from_raster(&frame, bg),
            bars,
        }
    }

    /// The plot as a page for a PDF: `title` over the bars, each panel's
    /// name under it and the communities' key at the right, as text. With
    /// `focus`, communities not in it are dimmed, as on screen. The page is
    /// as tall as its key and its panel names need; past what a page holds,
    /// or with no room left for the bars, there is no figure, and why.
    pub fn figure(
        &self,
        comm: &Communities,
        palette: &[Rgb],
        title: &str,
        focus: Option<&[bool]>,
        theme: Theme,
    ) -> Result<Page, &'static str> {
        if self.0.panels.is_empty() {
            return Err("nothing to draw: no cell is in a panel");
        }
        let (font, bold) = CHART_PAGE.fonts();
        let w = CHART_PAGE.px(1.).0;
        let palette: Vec<_> = palette.iter().map(|&c| to_linear(c)).collect();
        let key: Vec<String> = (0..comm.k).map(|c| comm.name(c)).collect();
        let style = BarStyle {
            palette: &palette,
            focus: focus.map(|f| (f, to_linear(theme.dimmed()))),
            gap: chart::GAP,
            hits: false,
        };
        let fig = Figure {
            title: Some(title),
            key: Some(&key),
            names: Names::Auto,
            font,
            bold,
            theme: theme.page(),
        };
        let no_room = "too small to draw: the key or the names leave no room for the bars";
        let h = chart::figure_size_checked(&self.0, &style, &fig, w).ok_or(no_room)?;
        let h = h.max(CHART_PAGE.px(0.4).1);
        chart_fits(w, h)?;
        let (page, drawn) = chart_page((w, h), theme, |frame| {
            chart::try_figure(frame, &self.0, &weights(comm, None), &style, &fig)
        });
        drawn.map_err(|_| no_room)?;
        Ok(page)
    }
}
