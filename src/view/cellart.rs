//! A frame as coloured text characters, for terminals without graphics.
//!
//! A character cell has two colours, foreground and background, and a glyph
//! deciding which parts of the cell take which. Each cell is sampled as a
//! small grid of glyph pixels; every glyph of the set splits that grid into
//! a foreground and a background part, and the cell takes the glyph whose
//! split, with each part in its mean colour, has the least squared error.
//!
//! - **quadrants** (default, `▘▝▖▗▚▞▙…`): a 2×2 grid;
//! - **symbols**: an 8×8 grid and the Block Elements glyphs —
//!   quadrants and the lower and left eighth bars (`▁…▇`, `▏…▉`) — so a
//!   boundary can sit at any eighth of a cell. (The top and right eighths,
//!   `▔▕`, split a cell as `▇▉` do with colours swapped, so they add
//!   nothing.)
//! - **half-blocks** (`▀`): a 1×2 grid, the upper pixel in front.
//!
//! All are in the Block Elements range that monospace fonts carry, unlike
//! sextants and octants.
//!
//! A character cell is taller than wide — about 1:2, or as the terminal
//! reports its font. The frame is rendered with square pixels, as many rows
//! per cell as that aspect gives, and each glyph pixel averages the frame
//! pixels it covers. Averages are taken in linear light, as the renderer
//! mixes colours.

use super::color::{self, Rgb};
use super::render::Frame;
use crate::util::common::*;
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::Color;
use ratatui::widgets::Widget;
use std::sync::LazyLock;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Glyphs {
    HalfBlocks,
    Quadrants,
    Symbols,
}

impl Glyphs {
    /// Square frame pixels per character cell, (x, y), for cells `aspect`
    /// times as tall as wide: one column per glyph column, and enough rows
    /// that every glyph row gets at least one.
    pub fn pixels_per_cell(self, aspect: f32) -> (usize, usize) {
        let (gw, gh) = self.grid();
        (gw, ((gw as f32 * aspect).round() as usize).max(gh))
    }

    pub fn name(self) -> &'static str {
        match self {
            Glyphs::HalfBlocks => "half-blocks",
            Glyphs::Quadrants => "quadrants",
            Glyphs::Symbols => "symbols",
        }
    }

    /// Glyph pixels per cell, (columns, rows).
    fn grid(self) -> (usize, usize) {
        match self {
            Glyphs::HalfBlocks => (1, 2),
            Glyphs::Quadrants => (2, 2),
            Glyphs::Symbols => (8, 8),
        }
    }

    /// Candidate glyphs with the grid pixels (bit `row * columns + col`)
    /// they paint in the foreground. A blank comes first, so a uniform cell
    /// stays blank.
    fn candidates(self) -> &'static [(char, u64)] {
        static HALF: [(char, u64); 2] = [(' ', 0), ('▀', 0b01)];
        static QUAD: LazyLock<Vec<(char, u64)>> = LazyLock::new(|| quadrant_glyphs(2));
        static SYMBOLS: LazyLock<Vec<(char, u64)>> = LazyLock::new(symbol_glyphs);
        match self {
            Glyphs::HalfBlocks => &HALF,
            Glyphs::Quadrants => &QUAD,
            Glyphs::Symbols => &SYMBOLS,
        }
    }
}

/// Quadrant glyphs by which quarters they fill: bit 0 top-left, 1
/// top-right, 2 bottom-left, 3 bottom-right. Those with bottom-right set
/// repeat another's split with colours swapped, so they are left out.
const QUADRANTS: [char; 8] = [' ', '▘', '▝', '▀', '▖', '▌', '▞', '▛'];

/// Quadrant glyphs on an `n × n` grid.
fn quadrant_glyphs(n: usize) -> Vec<(char, u64)> {
    let half = n / 2;
    let quarter = |q: usize| -> u64 {
        let (qx, qy) = (q % 2, q / 2);
        let mut m = 0u64;
        for y in qy * half..(qy + 1) * half {
            for x in qx * half..(qx + 1) * half {
                m |= 1 << (y * n + x);
            }
        }
        m
    };
    QUADRANTS
        .iter()
        .enumerate()
        .map(|(bits, &ch)| {
            let mask = (0..4)
                .filter(|q| bits >> q & 1 == 1)
                .fold(0, |m, q| m | quarter(q));
            (ch, mask)
        })
        .collect()
}

/// Quadrants plus eighth bars on the 8×8 grid.
fn symbol_glyphs() -> Vec<(char, u64)> {
    let region = |inside: &dyn Fn(usize, usize) -> bool| -> u64 {
        (0..64)
            .filter(|&i| inside(i % 8, i / 8))
            .fold(0, |m, i| m | 1 << i)
    };
    let mut out = quadrant_glyphs(8);
    // Lower k/8: ▁ ▂ ▃ ▄ ▅ ▆ ▇ (U+2581..U+2587).
    for k in 1..8 {
        let ch = char::from_u32(0x2580 + k as u32).expect("block element");
        out.push((ch, region(&|_, y| y >= 8 - k)));
    }
    // Left k/8: ▏ ▎ ▍ ▌ ▋ ▊ ▉ (U+258F down to U+2589).
    for k in 1..8 {
        let ch = char::from_u32(0x2590 - k as u32).expect("block element");
        out.push((ch, region(&|x, _| x < k)));
    }
    out
}

/// A frame fitted to a block of character cells, row by row.
pub struct Cells {
    cols: usize,
    cells: Vec<(char, Rgb, Rgb)>,
}

impl Cells {
    /// Fit `frame` to `cols × rows` cells of `glyphs`, each `ppc` frame
    /// pixels ([`Glyphs::pixels_per_cell`]).
    pub fn fit(
        frame: &Frame,
        glyphs: Glyphs,
        ppc: (usize, usize),
        cols: usize,
        rows: usize,
    ) -> Self {
        let cells = (0..rows * cols)
            .into_par_iter()
            .map(|i| fit(&sample(frame, glyphs, ppc, i % cols, i / cols), glyphs))
            .collect();
        Cells { cols, cells }
    }
}

impl Widget for &Cells {
    fn render(self, area: Rect, buf: &mut Buffer) {
        for (i, &(ch, fg, bg)) in self.cells.iter().enumerate() {
            let (c, r) = (i % self.cols, i / self.cols);
            if c >= area.width as usize || r >= area.height as usize {
                continue;
            }
            let at = Position::new(area.x + c as u16, area.y + r as u16);
            if let Some(cell) = buf.cell_mut(at) {
                cell.set_char(ch).set_fg(rgb(fg)).set_bg(rgb(bg));
            }
        }
    }
}

pub(super) fn rgb(c: Rgb) -> Color {
    Color::Rgb(c[0], c[1], c[2])
}

/// Glyph pixels of cell `(c, r)` in linear light, row-major: each the mean
/// of the frame pixels it covers. Outside the frame reads as background.
fn sample(
    frame: &Frame,
    glyphs: Glyphs,
    (pw, ph): (usize, usize),
    c: usize,
    r: usize,
) -> Vec<[f32; 3]> {
    let (gw, gh) = glyphs.grid();
    let (x0, y0) = (c * pw, r * ph);
    let mut out = Vec::with_capacity(gw * gh);
    for j in 0..gh {
        // Glyph rows split the cell's frame rows as evenly as integers allow.
        let rows = y0 + j * ph / gh..y0 + (j + 1) * ph / gh;
        for i in 0..gw {
            let cols = x0 + i * pw / gw..x0 + (i + 1) * pw / gw;
            let mut sum = [0f32; 3];
            let mut n = 0f32;
            for y in rows.clone() {
                for x in cols.clone() {
                    let p = if x < frame.w && y < frame.h {
                        let o = frame.offset(x, y);
                        [frame.rgba[o], frame.rgba[o + 1], frame.rgba[o + 2]]
                    } else {
                        color::BACKGROUND
                    };
                    for ch in 0..3 {
                        sum[ch] += color::linear(p[ch]);
                    }
                    n += 1.;
                }
            }
            out.push(sum.map(|s| s / n.max(1.)));
        }
    }
    out
}

/// The glyph, foreground and background that best show `px`, a cell's
/// glyph pixels in linear light, row-major.
pub(super) fn fit(px: &[[f32; 3]], glyphs: Glyphs) -> (char, Rgb, Rgb) {
    let n = px.len();
    let mut total = [0f32; 3];
    for p in px {
        for ch in 0..3 {
            total[ch] += p[ch];
        }
    }
    let mean = |sum: [f32; 3], k: usize| sum.map(|s| color::encode_fast(s / k as f32));

    // Squared error of a split, less the constant Σp², is
    // -|S_in|²/n_in - |S_out|²/n_out: maximize the sum of those terms. No
    // candidate covers the whole grid, so the outside is never empty.
    let mut best = (f32::NEG_INFINITY, ' ', 0u64);
    for &(ch, mask) in glyphs.candidates() {
        let mut inside = [0f32; 3];
        let mut bits = mask;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            for c in 0..3 {
                inside[c] += px[i][c];
            }
        }
        let k = mask.count_ones() as usize;
        let outside = [0, 1, 2].map(|c| total[c] - inside[c]);
        let term = |s: [f32; 3], m: usize| {
            if m == 0 {
                0.
            } else {
                s.iter().map(|v| v * v).sum::<f32>() / m as f32
            }
        };
        let score = term(inside, k) + term(outside, n - k);
        // Strictly better only, so ties keep the earlier, simpler glyph.
        if score > best.0 + 1e-6 {
            best = (score, ch, mask);
        }
    }
    let (_, ch, mask) = best;
    let k = mask.count_ones() as usize;
    let inside = (0..n)
        .filter(|&i| mask >> i & 1 == 1)
        .fold([0f32; 3], |acc, i| [0, 1, 2].map(|c| acc[c] + px[i][c]));
    let bg = mean([0, 1, 2].map(|c| total[c] - inside[c]), n - k);
    let fg = if k == 0 { bg } else { mean(inside, k) };
    (ch, fg, bg)
}
