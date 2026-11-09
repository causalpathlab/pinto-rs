//! A pair's threshold–response curve: the headline effect (the competent effect) at each
//! exposure threshold, as a percent change, with its pointwise 95%
//! intervals as whiskers and the simultaneous 95% band over the
//! thresholds shaded behind. The x axis is the threshold's quantile among
//! the positive exposures. The table's effect is the average over these
//! thresholds, weighted by their evidence: it is drawn dashed across the
//! plot, and the threshold it leans on most is ringed.
//!
//! Drawn into legume-tui's raster frame as the structure plot is: on screen
//! and in a PNG as pixels, in a PDF with its text kept as text.

use super::color::{to_linear, Rgb, Theme};
use super::pdf::{chart_fits, chart_page, CHART_PAGE};
use super::render::Frame;
use crate::lr_activity::ligand_effect::output::CurveRow;
use legume_tui::export::Page;
use legume_tui::map::colour::Lin;
use legume_tui::map::raster::{self, Canvas, Frame as Raster, Shape};
use legume_tui::map::text::{self, Font};

/// What the plot shows.
pub struct Curve<'a> {
    pub title: String,
    pub points: &'a [CurveRow],
    /// The threshold τ the table's average leans on most, ringed when it
    /// is a point.
    pub top: Option<f64>,
    /// The table's effect (log rate ratio): the average over thresholds.
    pub average: Option<f64>,
}

/// A log rate ratio as a percent change.
pub(super) fn pct(x: f64) -> f64 {
    x.exp_m1() * 100.
}

/// The line's colour, and the band's under it.
fn ink(theme: Theme) -> Rgb {
    match theme {
        Theme::Dark => [110, 170, 255],
        Theme::Light => [25, 85, 190],
    }
}

/// `x` at about three significant digits, signed: `+12%`, `-0.5%`.
fn label(x: f64) -> String {
    if x.abs() >= 10. || x == 0. {
        format!("{x:+.0}%")
    } else {
        format!("{x:+.1}%")
    }
}

/// Three to six round ticks covering `lo..=hi`.
pub fn ticks(lo: f64, hi: f64) -> Vec<f64> {
    if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
        return vec![if lo.is_finite() { lo } else { 0. }];
    }
    let raw = (hi - lo) / 4.;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1., 2., 2.5, 5., 10.]
        .iter()
        .map(|m| m * mag)
        .find(|s| (hi - lo) / s <= 5.)
        .unwrap_or(10. * mag);
    let first = (lo / step).ceil() as i64;
    let last = (hi / step).floor() as i64;
    (first..=last).map(|k| k as f64 * step).collect()
}

/// The y range `c` is drawn over, in percent: its points' and the
/// average's, with 0, padded; `None` when no threshold has a point.
#[cfg(test)]
pub fn y_range(c: &Curve) -> Option<(f64, f64)> {
    layout(c).1
}

/// The points that can be drawn, by quantile, as percent changes; their
/// y range (with 0 and the average), padded.
fn layout(c: &Curve) -> (Vec<CurveRow>, Option<(f64, f64)>) {
    let points = c.points;
    // In percent from here on.
    let mut pts: Vec<CurveRow> = points
        .iter()
        .filter(|p| p.quantile.is_finite() && p.effect.is_finite())
        .map(|p| CurveRow {
            effect: pct(p.effect),
            lo: pct(p.lo),
            hi: pct(p.hi),
            band_lo: pct(p.band_lo),
            band_hi: pct(p.band_hi),
            ..*p
        })
        .collect();
    pts.sort_by(|a, b| a.quantile.total_cmp(&b.quantile));
    let ys = pts
        .iter()
        .flat_map(|p| [p.effect, p.lo, p.hi, p.band_lo, p.band_hi])
        .chain(c.average.map(pct))
        .filter(|v| v.is_finite())
        .chain(std::iter::once(0.));
    let (lo, hi) = ys.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), y| {
        (a.min(y), b.max(y))
    });
    if pts.is_empty() {
        return (pts, None);
    }
    let pad = ((hi - lo) * 0.08).max(0.1);
    (pts, Some((lo - pad, hi + pad)))
}

/// Draw `c` over the whole of `frame` in `font` (titles in `bold`).
pub fn draw(frame: &mut Raster, c: &Curve, theme: Theme, font: Font, bold: Font) {
    let page = theme.page();
    let bg = frame.background();
    let (text_ink, dim) = (to_linear(page.text), to_linear(page.muted));
    let (w, h) = (frame.w as i32, frame.h as i32);
    let lh = font.line_height() as i32;
    text::draw(frame, bold, &c.title, lh / 2, lh / 4, text_ink, bg);
    let (pts, range) = layout(c);
    let Some((y_lo, y_hi)) = range else {
        let msg = "no threshold-response curve: not estimable at any threshold";
        text::draw(frame, font, msg, lh / 2, 2 * lh, text_ink, bg);
        return;
    };
    let ys = ticks(y_lo, y_hi);
    let y_labels: Vec<String> = ys.iter().map(|&y| label(y)).collect();
    let left = y_labels
        .iter()
        .map(|s| font.width(s) as i32)
        .max()
        .unwrap_or(0)
        + lh * 2;
    let (top, bottom, right) = (lh * 2, lh * 5, lh);
    let (x0, x1, y0, y1) = (
        left,
        (w - right).max(left + 2),
        top,
        (h - bottom).max(top + 2),
    );
    let px = |q: f64| x0 as f32 + (q.clamp(0., 1.) as f32) * (x1 - x0) as f32;
    let py = |v: f64| {
        let t = ((v - y_lo) / (y_hi - y_lo)).clamp(0., 1.) as f32;
        y1 as f32 - t * (y1 - y0) as f32
    };
    // Axes and the zero line.
    frame.fill(
        [x0 as usize, y1 as usize, x1 as usize, y1 as usize + 1],
        dim,
    );
    frame.fill(
        [x0 as usize, y0 as usize, x0 as usize + 1, y1 as usize + 1],
        dim,
    );
    // A dashed line across the plot at `y`: dashes `dash` long, gaps as
    // long, `thick` pixels.
    let dashed = |frame: &mut Raster, y: usize, dash: i32, thick: usize, ink: Lin| {
        for x in (x0..x1).step_by(2 * dash as usize) {
            let end = (x + dash).min(x1);
            frame.fill([x as usize, y, end as usize, y + thick], ink);
        }
    };
    dashed(frame, py(0.) as usize, 2, 1, dim);
    for (&y, s) in ys.iter().zip(&y_labels) {
        let at = py(y) as i32;
        frame.fill(
            [
                (x0 - lh / 3) as usize,
                at as usize,
                x0 as usize,
                at as usize + 1,
            ],
            dim,
        );
        let tw = font.width(s) as i32;
        text::draw(frame, font, s, x0 - lh / 2 - tw, at - lh / 2, text_ink, bg);
    }
    for k in 0..=4 {
        let q = k as f64 / 4.;
        let at = px(q) as i32;
        frame.fill(
            [
                at as usize,
                y1 as usize,
                at as usize + 1,
                (y1 + lh / 3) as usize,
            ],
            dim,
        );
        let s = format!("{}", k * 25);
        let tw = font.width(&s) as i32;
        text::draw(frame, font, &s, at - tw / 2, y1 + lh / 2, text_ink, bg);
    }
    let xt = "exposure threshold: percentile of the positive exposures";
    let tw = font.width(xt) as i32;
    text::draw(
        frame,
        font,
        xt,
        (x0 + x1 - tw) / 2,
        y1 + lh * 3 / 2,
        text_ink,
        bg,
    );
    text::draw_vertical(
        frame,
        font,
        "% change",
        (lh / 4, (y0 + y1) / 2 + 2 * lh),
        text_ink,
        bg,
    );

    let line = to_linear(ink(theme));
    // The simultaneous band, column by column between its points.
    let band: Vec<(f32, f32, f32)> = pts
        .iter()
        .filter(|p| p.band_lo.is_finite() && p.band_hi.is_finite())
        .map(|p| (px(p.quantile), py(p.band_hi), py(p.band_lo)))
        .collect();
    fill_band(frame, &band, line, 0.16);
    // Pointwise intervals, the curve and its points.
    let r = (lh as f32 / 5.).max(1.5);
    for p in &pts {
        if p.lo.is_finite() && p.hi.is_finite() {
            let x = px(p.quantile);
            for dx in [-0.5, 0.5] {
                raster::edge(
                    frame,
                    (x + dx, py(p.hi)),
                    (x + dx, py(p.lo)),
                    (0., 0.),
                    line,
                    0.9,
                );
            }
        }
    }
    for w in pts.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let (pa, pb) = (
            (px(a.quantile), py(a.effect)),
            (px(b.quantile), py(b.effect)),
        );
        raster::edge(frame, pa, pb, (0., 0.), line, 1.);
    }
    // The table's average, dashed across the plot.
    if let Some(avg) = c.average.map(pct).filter(|v| v.is_finite()) {
        dashed(frame, py(avg) as usize, (lh / 2).max(4), 2, text_ink);
    }
    for p in &pts {
        let (x, y) = (px(p.quantile), py(p.effect));
        raster::stamp(frame, x, y, r, Shape::Circle, line, 1.);
        let ringed = c
            .top
            .is_some_and(|t| (t - p.tau).abs() <= 1e-6 * t.abs().max(1.));
        if ringed {
            raster::ring(frame, x, y, r + 1.5, text_ink);
        }
    }
    let key = [
        "line: effect at each threshold; whiskers: pointwise 95%; shade: simultaneous 95% band",
        "dashed: the table's effect, averaged over thresholds; ring: the threshold it weighs most",
    ];
    // Both lines or neither: the dashes and the ring need the second.
    if key.iter().all(|line| font.width(line) as i32 <= w - lh) {
        for (k, line) in key.iter().enumerate() {
            let y = h - (2 - k as i32) * lh - lh / 4;
            text::draw(frame, font, line, lh / 2, y, to_linear(page.hint), bg);
        }
    }
}

/// Shade between `(x, top, bottom)` points, interpolated along x.
fn fill_band(frame: &mut Raster, pts: &[(f32, f32, f32)], c: Lin, alpha: f32) {
    let (w, h) = frame.size();
    for seg in pts.windows(2) {
        let ((xa, ta, ba), (xb, tb, bb)) = (seg[0], seg[1]);
        if xb <= xa {
            continue;
        }
        let from = xa.floor().max(0.) as usize;
        let to = (xb.ceil() as usize).min(w);
        for x in from..to {
            let t = ((x as f32 + 0.5 - xa) / (xb - xa)).clamp(0., 1.);
            let (top, bot) = (ta + t * (tb - ta), ba + t * (bb - ba));
            let (y0, y1) = (top.max(0.) as usize, (bot.max(0.) as usize).min(h));
            for y in y0..y1 {
                frame.blend(x, y, c, alpha);
            }
        }
    }
}

/// Width of a curve the viewer saves (`s`) at export scale 1.
pub const STILL_WIDTH: usize = 1200;

/// The plot as a still image `w × h`, its text sized to the image as the
/// screen sizes it to its cells.
pub fn still(c: &Curve, (w, h): (usize, usize), theme: Theme) -> Frame {
    let (w, h) = (w.max(1), h.max(1));
    let cell = (h as f32 / 28.).clamp(10., 64.);
    render(c, (w, h), theme, cell)
}

/// The plot as pixels, `w × h`, its text drawn for lines `cell_px` tall.
pub fn render(c: &Curve, (w, h): (usize, usize), theme: Theme, cell_px: f32) -> Frame {
    let bg = theme.background();
    let mut frame = Raster::blank_exact(w.max(1), h.max(1), bg);
    let (font, bold) = legume_tui::export::fonts(cell_px.max(6.));
    draw(&mut frame, c, theme, font, bold);
    Frame::from_raster(&frame, bg)
}

/// The plot as a PDF page, its text kept as text.
pub fn page(c: &Curve, theme: Theme) -> Result<Page, &'static str> {
    let (w, h) = CHART_PAGE.px(0.6);
    chart_fits(w, h)?;
    let (font, bold) = CHART_PAGE.fonts();
    let (page, ()) = chart_page((w, h), theme, |frame| draw(frame, c, theme, font, bold));
    Ok(page)
}
