//! Colours: a categorical palette for communities and 256-entry lookup
//! tables for continuous values.

pub type Rgb = [u8; 3];

/// Background and "no community" colours.
pub const BACKGROUND: Rgb = [17, 17, 20];
pub const NO_COMMUNITY: Rgb = [70, 70, 76];

/// `k` distinct colours: golden-angle hues in OKLCH at a few lightness
/// steps, so neighbouring ids never share a hue or a lightness.
pub fn palette(k: usize) -> Vec<Rgb> {
    const LIGHTNESS: [f32; 3] = [0.72, 0.60, 0.82];
    const CHROMA: [f32; 3] = [0.13, 0.13, 0.10];
    (0..k)
        .map(|i| {
            let hue = (i as f32 * 137.507_76 + 20.).to_radians();
            let step = i % LIGHTNESS.len();
            let (l, c) = (LIGHTNESS[step], CHROMA[step]);
            oklab_to_srgb(l, c * hue.cos(), c * hue.sin())
        })
        .collect()
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

/// sRGB byte → linear light.
pub fn decode(v: u8) -> f32 {
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
    pub fn viridis() -> Self {
        Self::from_stops(&[
            0x440154, 0x472d7b, 0x3b528b, 0x2c728e, 0x21918c, 0x28ae80, 0x5ec962, 0xaddc30,
            0xfde725,
        ])
    }

    pub fn magma() -> Self {
        Self::from_stops(&[
            0x000004, 0x1c1044, 0x4f127b, 0x812581, 0xb5367a, 0xe55064, 0xfb8761, 0xfec287,
            0xfcfdbf,
        ])
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_colours_are_distinct() {
        let p = palette(60);
        for i in 0..p.len() {
            for j in i + 1..p.len() {
                assert_ne!(p[i], p[j], "{i} and {j}");
            }
        }
    }

    #[test]
    fn srgb_round_trips() {
        for v in 0..=255u8 {
            assert_eq!(encode(decode(v)), v);
        }
    }

    #[test]
    fn ramp_ends_are_the_first_and_last_stops() {
        let v = Ramp::viridis();
        assert_eq!(v.at(0.), [0x44, 0x01, 0x54]);
        assert_eq!(v.at(1.), [0xfd, 0xe7, 0x25]);
    }
}
