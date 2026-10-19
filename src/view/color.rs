//! Colours: pinto's two map themes, drawn with legume-tui's, and the
//! palette, ramps and sRGB tables they share with the other viewers.

pub use legume_tui::map::colour::{diverging, encode, to_linear, viridis, Ramp, Srgb as Rgb};
use legume_tui::map::theme::Theme as Page;

/// The map's colours on a dark or a light background.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Theme {
    Dark,
    Light,
}

impl Theme {
    /// legume-tui's page for this theme.
    fn page(self) -> &'static Page {
        match self {
            Theme::Dark => &Page::DARK,
            Theme::Light => &Page::LIGHT,
        }
    }

    /// The other one, for the theme key.
    pub fn other(self) -> Theme {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::Dark,
        }
    }

    pub fn background(self) -> Rgb {
        self.page().background
    }

    /// Cells outside the focused communities: visible as tissue, not as
    /// colour.
    pub fn dimmed(self) -> Rgb {
        self.page().dimmed
    }

    /// Cells with no community.
    pub fn no_community(self) -> Rgb {
        self.page().no_group
    }

    /// Scale bar fill and the outline that keeps it readable over cells.
    pub fn bar(self) -> (Rgb, Rgb) {
        self.page().bar
    }

    /// `k` distinct colours: golden-angle hues in OKLCH at a few lightness
    /// steps, so neighbouring ids never share a hue or a lightness.
    pub fn palette(self, k: usize) -> Vec<Rgb> {
        self.page().palette.srgb_all(k)
    }

    /// Ramp for one community's propensity: low values fade into the page.
    pub fn magma(self) -> &'static Ramp {
        self.page().magma()
    }

    /// Ramp for feature levels: from the dimmed tissue grey at zero, so
    /// cells without the feature still show the tissue.
    pub fn expression(self) -> &'static Ramp {
        self.page().levels()
    }

    /// The theme for a background colour: light when its luminance is above
    /// middle grey (18% in linear light).
    pub fn for_background(bg: Rgb) -> Theme {
        if Page::for_background(bg).is_dark() {
            Theme::Dark
        } else {
            Theme::Light
        }
    }
}
