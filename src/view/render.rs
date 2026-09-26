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

use super::color::{self, Ramp, Rgb, BACKGROUND, DIMMED, NO_COMMUNITY};
use super::data::{Communities, Edges, Geometry, Rect};
use super::index::{EdgeIndex, Grid, Pyramid, PyramidLevel};
use super::scalebar::{self, Units};
use crate::util::common::*;
use crate::util::parquet_io::parse_community_col_name;
use std::ops::RangeInclusive;

/// Rows per parallel band. Tall enough that grid rows shared by two bands
/// (read by both) stay a small part of each band's work.
const BAND: usize = 64;

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
            _ => community_id(s).map(Layer::Community).ok_or_else(|| {
                anyhow::anyhow!("unknown layer {s:?}: use argmax, soft, entropy or C<k>")
            }),
        }
    }
}

/// `C7`, `c7` or `7` → 7, by the same rule that names propensity columns.
pub fn community_id(s: &str) -> Option<usize> {
    let upper = s.strip_prefix('c').map(|rest| format!("C{rest}"));
    parse_community_col_name(upper.as_deref().unwrap_or(s)).and_then(|c| usize::try_from(c).ok())
}

impl Layer {
    /// The colour ramp of a continuous layer (magma for the ones without).
    pub fn ramp(&self) -> &'static Ramp {
        match self {
            Layer::Entropy => color::viridis(),
            _ => color::magma(),
        }
    }

    /// Legend title of a continuous layer.
    pub fn legend_title(&self) -> String {
        match self {
            Layer::Community(c) => format!("C{c} propensity"),
            _ => "entropy / ln K".into(),
        }
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer::Argmax => write!(f, "argmax"),
            Layer::Soft => write!(f, "soft"),
            Layer::Entropy => write!(f, "entropy"),
            Layer::Community(c) => write!(f, "C{c}"),
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
    /// Fit `r` in a `w × h` frame, centred, filling the tighter axis
    /// exactly. Callers wanting a margin pad `r` first.
    pub fn fit(r: Rect, w: usize, h: usize) -> Self {
        let upp = (r.width() / w as f32)
            .max(r.height() / h as f32)
            .max(f32::MIN_POSITIVE);
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

    /// The world window the frame covers.
    pub fn window(&self) -> Rect {
        Rect {
            x0: self.x0,
            y0: self.y0,
            x1: self.x0 + self.w as f32 * self.upp,
            y1: self.y0 + self.h as f32 * self.upp,
        }
    }
}

#[derive(Clone)]
pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
}

impl Frame {
    /// Byte offset of pixel `(x, y)`.
    pub fn offset(&self, x: usize, y: usize) -> usize {
        (y * self.w + x) * 4
    }

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
pub struct Style<'f> {
    pub layer: Layer,
    pub edges: bool,
    /// Communities to show, one flag per community; the rest are dimmed and
    /// their edges hidden. `None` shows every community.
    pub focus: Option<&'f [bool]>,
    /// Burn a scale bar in these units into the frame.
    pub scale_bar: Option<Units>,
}

/// Colours for one layer, shared by the point, average and bin paths.
struct Paint<'a> {
    layer: Layer,
    focus: Option<&'a [bool]>,
    /// One colour per community.
    palette: &'a [Rgb],
    /// Palette in linear light, for mixing.
    linear: Vec<[f32; 3]>,
    ramp: &'static Ramp,
}

impl<'a> Paint<'a> {
    fn new(style: &Style<'a>, palette: &'a [Rgb]) -> Self {
        Paint {
            layer: style.layer,
            focus: style.focus,
            palette,
            linear: palette.iter().map(|c| c.map(color::linear)).collect(),
            ramp: style.layer.ramp(),
        }
    }

    fn k(&self) -> usize {
        self.palette.len()
    }

    fn community(&self, c: u16) -> Rgb {
        self.palette
            .get(c as usize)
            .copied()
            .unwrap_or(NO_COMMUNITY)
    }

    /// Weighted mix of community colours in linear light; `None` when the
    /// weights are all zero.
    fn mix_linear(&self, weights: impl Iterator<Item = f32>) -> Option<[f32; 3]> {
        let mut acc = [0f32; 3];
        let mut total = 0f32;
        for (w, lin) in weights.zip(&self.linear) {
            total += w;
            for (a, &l) in acc.iter_mut().zip(lin) {
                *a += w * l;
            }
        }
        (total > 0.).then(|| acc.map(|v| v / total))
    }

    fn mix(&self, weights: impl Iterator<Item = f32>) -> Rgb {
        self.mix_linear(weights)
            .map_or(NO_COMMUNITY, |c| c.map(color::encode_fast))
    }

    fn in_focus(&self, c: u16) -> bool {
        self.focus
            .is_none_or(|f| f.get(c as usize).copied().unwrap_or(false))
    }

    /// Share of `weights` on focused communities; 1 with no focus.
    fn focused_share(&self, weights: impl Iterator<Item = f32>) -> f32 {
        let Some(focus) = self.focus else { return 1. };
        let (mut on, mut all) = (0f32, 0f32);
        for (w, &f) in weights.zip(focus) {
            all += w;
            if f {
                on += w;
            }
        }
        if all > 0. {
            on / all
        } else {
            0.
        }
    }

    /// Fade `c` toward the dimmed colour by the unfocused share.
    fn dim(c: Rgb, share: f32) -> Rgb {
        if share >= 1. {
            return c;
        }
        [0, 1, 2].map(|ch| {
            let d = DIMMED[ch] as f32;
            (d + share * (c[ch] as f32 - d)).round() as u8
        })
    }

    fn row<'c>(&self, comm: &'c Communities, i: usize) -> &'c [u8] {
        &comm.prop[i * self.k()..(i + 1) * self.k()]
    }

    fn cell(&self, comm: &Communities, i: usize) -> Rgb {
        let c = self.cell_colour(comm, i);
        if self.focus.is_none() {
            return c;
        }
        let share = match self.layer {
            Layer::Soft => self.focused_share(self.row(comm, i).iter().map(|&q| q as f32)),
            _ => f32::from(u8::from(self.in_focus(comm.cluster[i]))),
        };
        Self::dim(c, share)
    }

    /// [`Self::cell`] in linear light, for averaging. The unfocused soft mix
    /// stays linear instead of going to a byte and back.
    fn cell_linear(&self, comm: &Communities, i: usize) -> [f32; 3] {
        if self.layer == Layer::Soft && self.focus.is_none() {
            if let Some(c) = self.mix_linear(self.row(comm, i).iter().map(|&q| q as f32)) {
                return c;
            }
        }
        self.cell(comm, i).map(color::linear)
    }

    fn cell_colour(&self, comm: &Communities, i: usize) -> Rgb {
        match self.layer {
            Layer::Argmax => self.community(comm.cluster[i]),
            Layer::Soft => self.mix(self.row(comm, i).iter().map(|&q| q as f32)),
            Layer::Entropy => match comm.entropy.as_ref() {
                Some(h) => self.ramp.at_u8(h[i]),
                None => NO_COMMUNITY,
            },
            Layer::Community(c) => self
                .ramp
                .at_u8(self.row(comm, i).get(c).copied().unwrap_or(0)),
        }
    }

    fn bin(&self, level: &PyramidLevel, b: usize) -> Rgb {
        let k = self.k();
        let sums = &level.prop[b * k..(b + 1) * k];
        let n = level.count[b] as f32;
        let c = match self.layer {
            Layer::Argmax => self.community(level.top[b]),
            Layer::Soft => self.mix(sums.iter().copied()),
            Layer::Entropy => self.ramp.at(level.entropy[b] / n),
            Layer::Community(c) => self.ramp.at(sums.get(c).map_or(0., |p| p / n)),
        };
        Self::dim(c, self.focused_share(sums.iter().copied()))
    }
}

/// How a viewport is drawn; see the module docs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Points,
    Average,
    /// Pyramid level drawn.
    Bins(usize),
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Points => write!(f, "points"),
            Mode::Average => write!(f, "average"),
            Mode::Bins(l) => write!(f, "bins L{l}"),
        }
    }
}

pub fn mode(scene: &Scene, vp: &Viewport) -> Mode {
    if scene.spacing / vp.upp >= MIN_POINT_PX {
        Mode::Points
    } else if vp.upp <= scene.grid.bin {
        Mode::Average
    } else {
        let levels = &scene.pyramid.levels;
        Mode::Bins(
            levels
                .iter()
                .position(|l| l.bin >= vp.upp)
                .unwrap_or(levels.len() - 1),
        )
    }
}

pub fn render(scene: &Scene, vp: &Viewport, style: &Style, palette: &[Rgb]) -> Frame {
    let paint = Paint::new(style, palette);
    let mut frame = Frame {
        w: vp.w,
        h: vp.h,
        rgba: vec![0u8; vp.w * vp.h * 4],
    };
    let spacing_px = scene.spacing / vp.upp;
    match mode(scene, vp) {
        Mode::Points => {
            let edges = scene
                .edges
                .filter(|_| style.edges && spacing_px >= MIN_EDGE_PX);
            // Smaller discs when edges show, so the lines between them read.
            let fill = if edges.is_some() { 0.28 } else { 0.45 };
            let radius = (fill * spacing_px).max(0.5);
            // Line offsets across its minor axis, for thickness.
            let width = (0.06 * spacing_px).clamp(1., 3.).round() as i64;
            let across: Vec<i64> = (0..width).map(|o| o - width / 2).collect();
            for_bands(
                &mut frame,
                || (),
                |_, canvas| {
                    if let Some((edges, index)) = edges {
                        draw_edges(canvas, scene, vp, &paint, edges, index, &across);
                    }
                    draw_points(canvas, scene, vp, &paint, radius);
                },
            );
        }
        Mode::Average => {
            // Half the expected cells per pixel counts as fully covered, so
            // sparse pixels at the tissue edge fade instead of speckling.
            let full = 0.5 / (spacing_px * spacing_px);
            for_bands(&mut frame, Vec::new, |acc, canvas| {
                draw_average(canvas, scene, vp, &paint, full, acc);
            });
        }
        Mode::Bins(l) => {
            let level = &scene.pyramid.levels[l];
            for_bands(
                &mut frame,
                || (),
                |_, canvas| {
                    draw_bins(canvas, scene, vp, &paint, level);
                },
            );
        }
    }
    if let Some(units) = style.scale_bar {
        scalebar::draw(&mut frame, vp, units);
    }
    frame
}

/// Split the frame into bands of rows, clear each to the background and
/// draw it in parallel. `init` makes per-thread scratch state, reused
/// across the bands that thread draws.
fn for_bands<T>(
    frame: &mut Frame,
    init: impl Fn() -> T + Sync + Send,
    draw: impl Fn(&mut T, &mut Canvas) + Sync + Send,
) {
    let w = frame.w;
    frame
        .rgba
        .par_chunks_mut(BAND * w * 4)
        .enumerate()
        .for_each_init(init, |state, (band, buf)| {
            for px in buf.chunks_exact_mut(4) {
                px.copy_from_slice(&[BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 255]);
            }
            let r0 = band * BAND;
            let r1 = r0 + buf.len() / (w * 4);
            draw(state, &mut Canvas { buf, w, r0, r1 });
        });
}

/// One band of the frame: rows `r0..r1`.
struct Canvas<'b> {
    buf: &'b mut [u8],
    w: usize,
    r0: usize,
    r1: usize,
}

impl Canvas<'_> {
    /// Byte offset of pixel `(x, y)`, if it lies in this band.
    fn at(&self, x: i64, y: i64) -> Option<usize> {
        let inside = x >= 0 && x < self.w as i64 && y >= self.r0 as i64 && y < self.r1 as i64;
        inside.then(|| ((y as usize - self.r0) * self.w + x as usize) * 4)
    }

    fn put(&mut self, x: i64, y: i64, c: Rgb) {
        if let Some(o) = self.at(x, y) {
            self.buf[o..o + 3].copy_from_slice(&c);
        }
    }

    fn blend(&mut self, x: i64, y: i64, c: Rgb, alpha: f32) {
        if let Some(o) = self.at(x, y) {
            blend(&mut self.buf[o..o + 3], c, alpha);
        }
    }

    /// Grid bin columns and rows under this band, widened by `pad` world
    /// units on every side.
    fn bins(
        &self,
        grid: &Grid,
        vp: &Viewport,
        pad: f32,
    ) -> (RangeInclusive<usize>, RangeInclusive<usize>) {
        let (ix0, iy0) = grid.bin_of(vp.x0 - pad, vp.y0 + self.r0 as f32 * vp.upp - pad);
        let (ix1, iy1) = grid.bin_of(
            vp.x0 + vp.w as f32 * vp.upp + pad,
            vp.y0 + self.r1 as f32 * vp.upp + pad,
        );
        (ix0..=ix1, iy0..=iy1)
    }
}

/// Draw `c` over the RGB pixel `dst` with opacity `alpha`.
fn blend(dst: &mut [u8], c: Rgb, alpha: f32) {
    for (d, &s) in dst.iter_mut().zip(&c) {
        let v = *d as f32;
        *d = (v + alpha * (s as f32 - v)).round() as u8;
    }
}

fn draw_points(canvas: &mut Canvas, scene: &Scene, vp: &Viewport, paint: &Paint, radius: f32) {
    let (xs, ys) = canvas.bins(scene.grid, vp, radius * vp.upp);
    let reach = radius.ceil() as i64;
    let r2 = radius * radius;
    for iy in ys {
        for ix in xs.clone() {
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

/// `acc` is per-thread scratch: linear RGB sums and a count per pixel.
fn draw_average(
    canvas: &mut Canvas,
    scene: &Scene,
    vp: &Viewport,
    paint: &Paint,
    full: f32,
    acc: &mut Vec<[f32; 4]>,
) {
    let (w, r0, r1) = (canvas.w, canvas.r0, canvas.r1);
    acc.clear();
    acc.resize((r1 - r0) * w, [0.; 4]);
    let (xs, ys) = canvas.bins(scene.grid, vp, 0.);
    for iy in ys {
        for ix in xs.clone() {
            for &i in scene.grid.cells(ix, iy) {
                let i = i as usize;
                let (px, py) = vp.to_px(scene.geom.x[i], scene.geom.y[i]);
                let (px, py) = (px.floor(), py.floor());
                if px < 0. || px >= w as f32 || py < r0 as f32 || py >= r1 as f32 {
                    continue;
                }
                let a = &mut acc[(py as usize - r0) * w + px as usize];
                let c = paint.cell_linear(scene.comm, i);
                for ch in 0..3 {
                    a[ch] += c[ch];
                }
                a[3] += 1.;
            }
        }
    }
    for (a, px) in acc.iter().zip(canvas.buf.chunks_exact_mut(4)) {
        if a[3] > 0. {
            let c = [0, 1, 2].map(|ch| color::encode_fast(a[ch] / a[3]));
            blend(&mut px[..3], c, (a[3] / full).min(1.));
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
    across: &[i64],
) {
    let (xs, ys) = canvas.bins(scene.grid, vp, index.max_len);
    let (x, y) = (&scene.geom.x, &scene.geom.y);
    for iy in ys {
        for ix in xs.clone() {
            for &e in index.edges(ix, iy) {
                let e = e as usize;
                let (a, b) = (edges.a[e] as usize, edges.b[e] as usize);
                let (ax, ay) = vp.to_px(x[a], y[a]);
                let (bx, by) = vp.to_px(x[b], y[b]);
                if ay.max(by) < canvas.r0 as f32 || ay.min(by) >= canvas.r1 as f32 {
                    continue;
                }
                let community = edges.community[e];
                if !paint.in_focus(community) {
                    continue;
                }
                let c = paint.community(community);
                let steps = (bx - ax).abs().max((by - ay).abs()).ceil().max(1.) as usize;
                let steep = (by - ay).abs() > (bx - ax).abs();
                for s in 0..=steps {
                    let t = s as f32 / steps as f32;
                    let (px, py) = (ax + t * (bx - ax), ay + t * (by - ay));
                    let (px, py) = (px.floor() as i64, py.floor() as i64);
                    for &o in across {
                        let (qx, qy) = if steep { (px + o, py) } else { (px, py + o) };
                        canvas.blend(qx, qy, c, 0.85);
                    }
                }
            }
        }
    }
}

fn draw_bins(
    canvas: &mut Canvas,
    scene: &Scene,
    vp: &Viewport,
    paint: &Paint,
    level: &PyramidLevel,
) {
    let (ox, oy) = scene.grid.origin;
    let w = canvas.w;
    for (r, line) in canvas.buf.chunks_exact_mut(w * 4).enumerate() {
        let wy = vp.y0 + ((canvas.r0 + r) as f32 + 0.5) * vp.upp;
        let by = ((wy - oy) / level.bin).floor();
        if by < 0. || by >= level.ny as f32 {
            continue;
        }
        let by = by as usize;
        // Neighbouring pixels usually share a bin; colour each bin once.
        let mut last: Option<(usize, Rgb)> = None;
        for col in 0..w {
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
            let c = match last {
                Some((lb, c)) if lb == b => c,
                _ => {
                    let c = paint.bin(level, b);
                    last = Some((b, c));
                    c
                }
            };
            // Bins at the tissue edge hold few cells; they fade in up to
            // half the mean occupancy so the outline reads as an outline.
            let alpha = (n as f32 / level.full).min(1.);
            blend(&mut line[col * 4..col * 4 + 3], c, alpha);
        }
    }
}
