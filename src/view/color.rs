//! Colours: a categorical palette for communities and 256-entry lookup
//! tables for continuous values.

use std::sync::LazyLock;

pub type Rgb = [u8; 3];

/// The map's colours on a dark or a light background.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Theme {
    Dark,
    Light,
}

impl Theme {
    pub fn background(self) -> Rgb {
        match self {
            Theme::Dark => [17, 17, 20],
            Theme::Light => [250, 250, 248],
        }
    }

    /// Cells outside the focused communities: visible as tissue, not as
    /// colour.
    pub fn dimmed(self) -> Rgb {
        match self {
            Theme::Dark => [40, 40, 46],
            Theme::Light => [222, 222, 226],
        }
    }

    /// Cells with no community.
    pub fn no_community(self) -> Rgb {
        match self {
            Theme::Dark => [70, 70, 76],
            Theme::Light => [175, 175, 181],
        }
    }

    /// Scale bar fill and the outline that keeps it readable over cells.
    pub fn bar(self) -> (Rgb, Rgb) {
        match self {
            Theme::Dark => ([255; 3], [0; 3]),
            Theme::Light => ([0; 3], [255; 3]),
        }
    }

    /// `k` distinct colours: golden-angle hues in OKLCH at a few lightness
    /// steps, so neighbouring ids never share a hue or a lightness. On a
    /// light background the steps are darker and a little more saturated.
    pub fn palette(self, k: usize) -> Vec<Rgb> {
        let (lightness, chroma) = match self {
            Theme::Dark => ([0.72, 0.60, 0.82], [0.13, 0.13, 0.10]),
            Theme::Light => ([0.58, 0.47, 0.68], [0.15, 0.14, 0.14]),
        };
        (0..k)
            .map(|i| {
                let hue = (i as f32 * 137.507_76 + 20.).to_radians();
                let step = i % lightness.len();
                let (l, c) = (lightness[step], chroma[step]);
                oklab_to_srgb(l, c * hue.cos(), c * hue.sin())
            })
            .collect()
    }

    /// Ramp for one community's propensity: from the background's side up
    /// to the strongest colour, so low values fade into the page.
    pub fn magma(self) -> &'static Ramp {
        // Magma run backwards from near-white, without its black and its
        // pale ends.
        static LIGHT: LazyLock<Ramp> = LazyLock::new(|| {
            let stops: Vec<u32> = std::iter::once(0xf4f4f0)
                .chain(MAGMA[1..MAGMA.len() - 1].iter().rev().copied())
                .collect();
            Ramp::from_stops(&stops)
        });
        match self {
            Theme::Dark => magma(),
            Theme::Light => &LIGHT,
        }
    }

    /// Ramp for feature levels: from the dimmed tissue grey at zero, so
    /// cells without the feature still show the tissue, to the colour
    /// furthest from the background.
    pub fn expression(self) -> &'static Ramp {
        fn hex([r, g, b]: Rgb) -> u32 {
            (r as u32) << 16 | (g as u32) << 8 | b as u32
        }
        static DARK: LazyLock<Ramp> = LazyLock::new(|| {
            Ramp::from_stops(&[
                hex(Theme::Dark.dimmed()),
                0x4f127b,
                0xb5367a,
                0xfb8761,
                0xfcfdbf,
            ])
        });
        static LIGHT: LazyLock<Ramp> = LazyLock::new(|| {
            Ramp::from_stops(&[
                hex(Theme::Light.dimmed()),
                0xfb8761,
                0xe55064,
                0x812581,
                0x1c1044,
            ])
        });
        match self {
            Theme::Dark => &DARK,
            Theme::Light => &LIGHT,
        }
    }

    /// The theme for a background colour: light when its luminance is above
    /// middle grey (18% in linear light).
    pub fn for_background([r, g, b]: Rgb) -> Theme {
        let luminance = 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
        if luminance > 0.18 {
            Theme::Light
        } else {
            Theme::Dark
        }
    }
}

fn oklab_to_srgb(l: f32, a: f32, b: f32) -> Rgb {
    let l_ = (l + 0.396_337_78 * a + 0.215_803_76 * b).powi(3);
    let m_ = (l - 0.105_561_346 * a - 0.063_854_17 * b).powi(3);
    let s_ = (l - 0.089_484_18 * a - 1.291_485_5 * b).powi(3);
    let r = 4.076_741_7 * l_ - 3.307_711_6 * m_ + 0.230_969_94 * s_;
    let g = -1.268_438 * l_ + 2.609_757_4 * m_ - 0.341_319_4 * s_;
    let bl = -0.004_196_086_3 * l_ - 0.703_418_6 * m_ + 1.707_614_7 * s_;
    [encode(r), encode(g), encode(bl)]
}

/// Linear light → sRGB byte.
pub fn encode(c: f32) -> u8 {
    let c = c.clamp(0., 1.);
    let v = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1. / 2.4) - 0.055
    };
    (v * 255.).round() as u8
}

/// Steps of the linear-light table behind [`encode_fast`]: finer than a byte
/// of sRGB needs anywhere but the darkest few codes.
const ENCODE_STEPS: usize = 4096;

/// [`encode`] by table lookup, for per-pixel use.
pub fn encode_fast(c: f32) -> u8 {
    static TABLE: LazyLock<Vec<u8>> = LazyLock::new(|| {
        (0..=ENCODE_STEPS)
            .map(|i| encode(i as f32 / ENCODE_STEPS as f32))
            .collect()
    });
    TABLE[(c.clamp(0., 1.) * ENCODE_STEPS as f32).round() as usize]
}

/// [`decode`] by table lookup, for per-pixel use.
pub fn linear(v: u8) -> f32 {
    static TABLE: LazyLock<[f32; 256]> = LazyLock::new(|| std::array::from_fn(|v| decode(v as u8)));
    TABLE[v as usize]
}

pub fn viridis() -> &'static Ramp {
    static RAMP: LazyLock<Ramp> = LazyLock::new(|| {
        Ramp::from_stops(&[
            0x440154, 0x472d7b, 0x3b528b, 0x2c728e, 0x21918c, 0x28ae80, 0x5ec962, 0xaddc30,
            0xfde725,
        ])
    });
    &RAMP
}

/// Magma's stops, black to pale yellow.
const MAGMA: [u32; 9] = [
    0x000004, 0x1c1044, 0x4f127b, 0x812581, 0xb5367a, 0xe55064, 0xfb8761, 0xfec287, 0xfcfdbf,
];

pub fn magma() -> &'static Ramp {
    static RAMP: LazyLock<Ramp> = LazyLock::new(|| Ramp::from_stops(&MAGMA));
    &RAMP
}

/// sRGB byte → linear light.
pub(super) fn decode(v: u8) -> f32 {
    let c = v as f32 / 255.;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// A 256-entry colour ramp.
pub struct Ramp(Vec<Rgb>);

impl Ramp {
    /// Evenly spaced stops, interpolated in sRGB.
    fn from_stops(stops: &[u32]) -> Self {
        let rgb = |h: u32| {
            [
                (h >> 16) as f32,
                ((h >> 8) & 0xff) as f32,
                (h & 0xff) as f32,
            ]
        };
        let segments = (stops.len() - 1) as f32;
        let lut = (0..256)
            .map(|i| {
                let t = i as f32 / 255. * segments;
                let j = (t.floor() as usize).min(stops.len() - 2);
                let f = t - j as f32;
                let (a, b) = (rgb(stops[j]), rgb(stops[j + 1]));
                [0, 1, 2].map(|c| (a[c] + f * (b[c] - a[c])).round() as u8)
            })
            .collect();
        Ramp(lut)
    }

    /// Colour for `t` in `[0, 1]`.
    pub fn at(&self, t: f32) -> Rgb {
        self.0[(t.clamp(0., 1.) * 255.).round() as usize]
    }

    pub fn at_u8(&self, q: u8) -> Rgb {
        self.0[q as usize]
    }
}
