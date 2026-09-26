//! Viewport → RGBA frame.
//!
//! Three modes, picked by how many cells fall on one screen pixel:
//!
//! - **points**, when cells are a few pixels apart: every visible cell is a
//!   disc sized to the typical cell spacing, with edges underneath once they
//!   are long enough on screen to read;
//! - **average**, when cells are about a pixel apart: each pixel shows the
//!   mean colour (in linear light) of the cells that land on it, so a
//!   continuous layer stays smooth instead of showing per-cell noise;
//! - **bins**, when a pixel is wider than a finest grid bin: each pixel shows
//!   the pyramid bin under it, at the finest level whose bins are at least a
//!   pixel wide, so the cost no longer grows with the number of cells.
//!
//! Rows are split into bands drawn in parallel; each band reads only the
//! grid rows it overlaps.

use super::color::{self, Ramp, Rgb, BACKGROUND, NO_COMMUNITY};
use super::data::{Communities, Edges, Geometry, Rect, NO_CLUSTER};
use super::index::{EdgeIndex, Grid, Pyramid, PyramidLevel};
use crate::util::common::*;

/// Rows per parallel band.
const BAND: usize = 16;

/// Edges are drawn only once the typical edge spans this many pixels.
const MIN_EDGE_PX: f32 = 4.;

/// Below this cell spacing in pixels, pixels average their cells.
const MIN_POINT_PX: f32 = 2.5;

/// What colours a cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layer {
    /// The cell's most likely community.
    Argmax,
    /// Community colours mixed by the cell's propensity.
    Soft,
    /// Propensity entropy / ln K.
    Entropy,
    /// One community's propensity.
    Community(usize),
}

impl std::str::FromStr for Layer {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "argmax" => Ok(Layer::Argmax),
            "soft" => Ok(Layer::Soft),
            "entropy" => Ok(Layer::Entropy),
            _ => s
                .strip_prefix(['C', 'c'])
                .and_then(|k| k.parse().ok())
                .map(Layer::Community)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown layer {s:?}: use argmax, soft, entropy or C<k>")
                }),
        }
    }
}

/// Everything a frame is drawn from.
pub struct Scene<'a> {
    pub geom: &'a Geometry,
    pub comm: &'a Communities,
    pub grid: &'a Grid,
    pub pyramid: &'a Pyramid,
    pub edges: Option<(&'a Edges, &'a EdgeIndex)>,
    /// Typical distance between neighbouring cells, world units.
    pub spacing: f32,
}

/// The world window drawn into a `w × h` frame.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub x0: f32,
    pub y0: f32,
    /// World units per pixel.
    pub upp: f32,
    pub w: usize,
    pub h: usize,
}

impl Viewport {
    /// Fit `r` in a `w × h` frame with a small margin, centred.
    pub fn fit(r: Rect, w: usize, h: usize) -> Self {
        let upp = (r.width() / w as f32)
            .max(r.height() / h as f32)
            .max(f32::MIN_POSITIVE)
            * 1.02;
        let cx = 0.5 * (r.x0 + r.x1);
        let cy = 0.5 * (r.y0 + r.y1);
        Viewport {
            x0: cx - 0.5 * w as f32 * upp,
            y0: cy - 0.5 * h as f32 * upp,
            upp,
            w,
            h,
        }
    }

    fn to_px(self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.x0) / self.upp, (y - self.y0) / self.upp)
    }
}

pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
}

impl Frame {
    pub fn write_png(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut enc = png::Encoder::new(file, self.w as u32, self.h as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(png::Compression::Fast);
        enc.write_header()?.write_image_data(&self.rgba)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Style {
    pub layer: Layer,
    pub edges: bool,
}

/// Colours for one layer, shared by the point and bin paths.
struct Paint<'a> {
    layer: Layer,
    k: usize,
    palette: &'a [Rgb],
    /// Palette in linear light, for mixing.
    linear: Vec<[f32; 3]>,
    ramp: Ramp,
    /// sRGB byte → linear light.
    lut: [f32; 256],
}

impl<'a> Paint<'a> {
    fn new(layer: Layer, k: usize, palette: &'a [Rgb]) -> Self {
        let linear = palette.iter().map(|c| c.map(color::decode)).collect();
        let ramp = match layer {
            Layer::Entropy => Ramp::viridis(),
            _ => Ramp::magma(),
        };
        Paint {
            layer,
            k,
            palette,
            linear,
            ramp,
            lut: std::array::from_fn(|v| color::decode(v as u8)),
        }
    }

    fn community(&self, c: u16) -> Rgb {
        if c == NO_CLUSTER {
            NO_COMMUNITY
        } else {
            self.palette[c as usize % self.palette.len()]
        }
    }

    /// Weighted mix of community colours; weights need not sum to one.
    fn mix(&self, weights: impl Iterator<Item = f32>) -> Rgb {
        let mut acc = [0f32; 3];
        let mut total = 0f32;
        for (w, lin) in weights.zip(&self.linear) {
            total += w;
            for c in 0..3 {
                acc[c] += w * lin[c];
            }
        }
        if total <= 0. {
            return NO_COMMUNITY;
        }
        acc.map(|v| color::encode(v / total))
    }

    fn cell(&self, comm: &Communities, i: usize) -> Rgb {
        let k = self.k;
        match self.layer {
            Layer::Argmax => self.community(comm.cluster[i]),
            Layer::Soft => self.mix(comm.prop[i * k..(i + 1) * k].iter().map(|&q| q as f32)),
            Layer::Entropy => match comm.entropy.as_ref() {
                Some(h) => self.ramp.at_u8(h[i]),
                None => NO_COMMUNITY,
            },
            Layer::Community(c) => self
                .ramp
                .at_u8(comm.prop.get(i * k + c).copied().unwrap_or(0)),
        }
    }

    fn bin(&self, level: &PyramidLevel, b: usize) -> Rgb {
        let k = self.k;
        let n = level.count[b] as f32;
        match self.layer {
            Layer::Argmax => self.community(level.top[b]),
            Layer::Soft => self.mix(level.prop[b * k..(b + 1) * k].iter().copied()),
            Layer::Entropy => self.ramp.at(level.entropy[b] / n),
            Layer::Community(c) => self
                .ramp
                .at(level.prop.get(b * k + c).map_or(0., |p| p / n)),
        }
    }
}

pub fn render(scene: &Scene, vp: &Viewport, style: &Style, palette: &[Rgb]) -> Frame {
    let paint = Paint::new(style.layer, scene.comm.k, palette);
    let mut rgba = vec![255u8; vp.w * vp.h * 4];
    for px in rgba.chunks_exact_mut(4) {
        px[..3].copy_from_slice(&BACKGROUND);
    }

    let row_bytes = vp.w * 4;
    let spacing_px = scene.spacing / vp.upp;
    if spacing_px >= MIN_POINT_PX {
        let edges = scene
            .edges
            .filter(|_| style.edges && spacing_px >= MIN_EDGE_PX);
        // Smaller discs when edges show, so the lines between them read.
        let fill = if edges.is_some() { 0.28 } else { 0.45 };
        let radius = (fill * spacing_px).max(0.5);
        let line = (0.06 * spacing_px).clamp(1., 3.);
        rgba.par_chunks_mut(BAND * row_bytes)
            .enumerate()
            .for_each(|(band, buf)| {
                let r0 = band * BAND;
                let r1 = r0 + buf.len() / row_bytes;
                let mut canvas = Canvas {
                    buf,
                    w: vp.w,
                    r0,
                    r1,
                };
                if let Some((edges, index)) = edges {
                    draw_edges(&mut canvas, scene, vp, &paint, edges, index, line);
                }
                draw_points(&mut canvas, scene, vp, &paint, radius);
            });
    } else if vp.upp <= scene.grid.bin {
        // Half the expected cells per pixel counts as fully covered, so
        // sparse pixels at the tissue edge fade instead of speckling.
        let full = 0.5 / (spacing_px * spacing_px);
        rgba.par_chunks_mut(BAND * row_bytes)
            .enumerate()
            .for_each(|(band, buf)| {
                let r0 = band * BAND;
                let r1 = r0 + buf.len() / row_bytes;
                let mut canvas = Canvas {
                    buf,
                    w: vp.w,
                    r0,
                    r1,
                };
                draw_average(&mut canvas, scene, vp, &paint, full);
            });
    } else {
        let level = scene
            .pyramid
            .levels
            .iter()
            .find(|l| l.bin >= vp.upp)
            .unwrap_or_else(|| scene.pyramid.levels.last().expect("non-empty"));
        draw_bins(&mut rgba, scene, vp, &paint, level);
    }

    Frame {
        w: vp.w,
        h: vp.h,
        rgba,
    }
}

/// One band of the frame: rows `r0..r1`.
struct Canvas<'b> {
    buf: &'b mut [u8],
    w: usize,
    r0: usize,
    r1: usize,
}

impl Canvas<'_> {
    fn put(&mut self, x: i64, y: i64, c: Rgb) {
        if x < 0 || x >= self.w as i64 || y < self.r0 as i64 || y >= self.r1 as i64 {
            return;
        }
        let o = ((y as usize - self.r0) * self.w + x as usize) * 4;
        self.buf[o..o + 3].copy_from_slice(&c);
    }

    fn blend(&mut self, x: i64, y: i64, c: Rgb, alpha: f32) {
        if x < 0 || x >= self.w as i64 || y < self.r0 as i64 || y >= self.r1 as i64 {
            return;
        }
        let o = ((y as usize - self.r0) * self.w + x as usize) * 4;
        blend(&mut self.buf[o..o + 3], c, alpha);
    }
}

/// Draw `c` over the RGB pixel `dst` with opacity `alpha`.
fn blend(dst: &mut [u8], c: Rgb, alpha: f32) {
    for (d, &s) in dst.iter_mut().zip(&c) {
        let v = *d as f32;
        *d = (v + alpha * (s as f32 - v)).round() as u8;
    }
}

/// Grid bins overlapping world rows `y0..y1` and columns `x0..x1`.
fn bins_in(grid: &Grid, x0: f32, y0: f32, x1: f32, y1: f32) -> (usize, usize, usize, usize) {
    let (ix0, iy0) = grid.bin_of(x0, y0);
    let (ix1, iy1) = grid.bin_of(x1, y1);
    (ix0, iy0, ix1, iy1)
}

fn draw_points(canvas: &mut Canvas, scene: &Scene, vp: &Viewport, paint: &Paint, radius: f32) {
    let pad = radius * vp.upp;
    let (ix0, iy0, ix1, iy1) = bins_in(
        scene.grid,
        vp.x0 - pad,
        vp.y0 + canvas.r0 as f32 * vp.upp - pad,
        vp.x0 + vp.w as f32 * vp.upp + pad,
        vp.y0 + canvas.r1 as f32 * vp.upp + pad,
    );
    let reach = radius.ceil() as i64;
    let r2 = radius * radius;
    for iy in iy0..=iy1 {
        for ix in ix0..=ix1 {
            for &i in scene.grid.cells(ix, iy) {
                let i = i as usize;
                let (px, py) = vp.to_px(scene.geom.x[i], scene.geom.y[i]);
                if py + radius < canvas.r0 as f32 || py - radius >= canvas.r1 as f32 {
                    continue;
                }
                let c = paint.cell(scene.comm, i);
                let (cx, cy) = (px.floor() as i64, py.floor() as i64);
                for dy in -reach..=reach {
                    for dx in -reach..=reach {
                        let (ox, oy) = ((cx + dx) as f32 + 0.5 - px, (cy + dy) as f32 + 0.5 - py);
                        if ox * ox + oy * oy <= r2 {
                            canvas.put(cx + dx, cy + dy, c);
                        }
                    }
                }
            }
        }
    }
}

fn draw_average(canvas: &mut Canvas, scene: &Scene, vp: &Viewport, paint: &Paint, full: f32) {
    let (ix0, iy0, ix1, iy1) = bins_in(
        scene.grid,
        vp.x0,
        vp.y0 + canvas.r0 as f32 * vp.upp,
        vp.x0 + vp.w as f32 * vp.upp,
        vp.y0 + canvas.r1 as f32 * vp.upp,
    );
    let rows = canvas.r1 - canvas.r0;
    // Linear RGB sums and a count per pixel.
    let mut acc = vec![[0f32; 4]; rows * canvas.w];
    for iy in iy0..=iy1 {
        for ix in ix0..=ix1 {
            for &i in scene.grid.cells(ix, iy) {
                let i = i as usize;
                let (px, py) = vp.to_px(scene.geom.x[i], scene.geom.y[i]);
                let (px, py) = (px.floor(), py.floor());
                if px < 0.
                    || px >= canvas.w as f32
                    || py < canvas.r0 as f32
                    || py >= canvas.r1 as f32
                {
                    continue;
                }
                let a = &mut acc[(py as usize - canvas.r0) * canvas.w + px as usize];
                let c = paint.cell(scene.comm, i);
                for ch in 0..3 {
                    a[ch] += paint.lut[c[ch] as usize];
                }
                a[3] += 1.;
            }
        }
    }
    for (p, a) in acc.iter().enumerate() {
        if a[3] > 0. {
            let c = [0, 1, 2].map(|ch| color::encode(a[ch] / a[3]));
            let (x, y) = (p % canvas.w, canvas.r0 + p / canvas.w);
            canvas.blend(x as i64, y as i64, c, (a[3] / full).min(1.));
        }
    }
}

fn draw_edges(
    canvas: &mut Canvas,
    scene: &Scene,
    vp: &Viewport,
    paint: &Paint,
    edges: &Edges,
    index: &EdgeIndex,
    width: f32,
) {
    let pad = index.max_len;
    let (ix0, iy0, ix1, iy1) = bins_in(
        scene.grid,
        vp.x0 - pad,
        vp.y0 + canvas.r0 as f32 * vp.upp - pad,
        vp.x0 + vp.w as f32 * vp.upp + pad,
        vp.y0 + canvas.r1 as f32 * vp.upp + pad,
    );
    let (x, y) = (&scene.geom.x, &scene.geom.y);
    for iy in iy0..=iy1 {
        for ix in ix0..=ix1 {
            for &e in index.edges(ix, iy) {
                let e = e as usize;
                let (a, b) = (edges.a[e] as usize, edges.b[e] as usize);
                let (ax, ay) = vp.to_px(x[a], y[a]);
                let (bx, by) = vp.to_px(x[b], y[b]);
                if ay.max(by) < canvas.r0 as f32 || ay.min(by) >= canvas.r1 as f32 {
                    continue;
                }
                let c = paint.community(edges.community[e]);
                let steps = (bx - ax).abs().max((by - ay).abs()).ceil().max(1.) as usize;
                // Thicken across the line's minor axis.
                let across: Vec<i64> = (0..width.round() as i64)
                    .map(|o| o - (width as i64) / 2)
                    .collect();
                let steep = (by - ay).abs() > (bx - ax).abs();
                for s in 0..=steps {
                    let t = s as f32 / steps as f32;
                    let (px, py) = (ax + t * (bx - ax), ay + t * (by - ay));
                    let (px, py) = (px.floor() as i64, py.floor() as i64);
                    for &o in &across {
                        let (qx, qy) = if steep { (px + o, py) } else { (px, py + o) };
                        canvas.blend(qx, qy, c, 0.85);
                    }
                }
            }
        }
    }
}

fn draw_bins(rgba: &mut [u8], scene: &Scene, vp: &Viewport, paint: &Paint, level: &PyramidLevel) {
    // Bins at the tissue edge hold few cells; fade them in up to half the
    // mean occupancy so the outline reads as an outline, not a hard block.
    let (filled, total) = level
        .count
        .iter()
        .filter(|&&c| c > 0)
        .fold((0usize, 0u64), |(n, s), &c| (n + 1, s + c as u64));
    let full = 0.5 * total as f32 / filled.max(1) as f32;
    let (ox, oy) = scene.grid.origin;

    rgba.par_chunks_mut(vp.w * 4)
        .enumerate()
        .for_each(|(row, line)| {
            let wy = vp.y0 + (row as f32 + 0.5) * vp.upp;
            let by = ((wy - oy) / level.bin).floor();
            if by < 0. || by >= level.ny as f32 {
                return;
            }
            let by = by as usize;
            for col in 0..vp.w {
                let wx = vp.x0 + (col as f32 + 0.5) * vp.upp;
                let bx = ((wx - ox) / level.bin).floor();
                if bx < 0. || bx >= level.nx as f32 {
                    continue;
                }
                let b = by * level.nx + bx as usize;
                let n = level.count[b];
                if n == 0 {
                    continue;
                }
                let alpha = (n as f32 / full).min(1.);
                let c = paint.bin(level, b);
                blend(&mut line[col * 4..col * 4 + 3], c, alpha);
            }
        });
}
