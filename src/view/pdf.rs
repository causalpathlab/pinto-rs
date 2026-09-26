//! One-page PDF figure: the map as an embedded image, everything around it
//! as vector text and shapes (title, scale bar, legend, markers).
//!
//! The map stays a raster on purpose: a section is hundreds of thousands of
//! cells, which as vector shapes make a file that viewers crawl through.
//! Text uses Helvetica, one of the fonts every PDF reader carries, so
//! nothing is embedded but the image.

use super::color::{Rgb, Theme};
use super::render::{Frame, Viewport};
use super::scalebar::{self, Units};
use super::thousands;
use pdf_writer::{Content, Filter, Finish, Name, Pdf, Rect, Ref, Str};
use std::io::Write;

pub struct Figure<'a> {
    /// The map, without a burnt-in scale bar.
    pub frame: &'a Frame,
    pub vp: Viewport,
    pub title: String,
    pub subtitle: String,
    pub units: Option<Units>,
    pub legend: Legend,
    pub markers: Vec<MarkerBlock>,
    pub theme: Theme,
}

pub enum Legend {
    Communities(Vec<Entry>),
    /// A continuous layer: its title, colours from 0 up, and the label of
    /// the top of the ramp.
    Ramp {
        title: String,
        stops: Vec<Rgb>,
        top: String,
    },
}

pub struct Entry {
    pub label: String,
    pub count: usize,
    /// The swatch, dimmed already when another community is focused.
    pub colour: Rgb,
    /// False when another community is focused and this one is dimmed.
    pub on: bool,
}

pub struct MarkerBlock {
    pub title: String,
    pub colour: Rgb,
    pub genes: Vec<(String, f32)>,
}

// Page geometry, points.
const MARGIN: f32 = 28.;
const MAP_W: f32 = 460.;
const MAP_H_MAX: f32 = 620.;
const GAP: f32 = 18.;
const COLUMN: f32 = 170.;
const TITLE_H: f32 = 38.;
const LINE: f32 = 11.;

const REGULAR: Name = Name(b"F1");
const BOLD: Name = Name(b"F2");
const MAP: Name = Name(b"Im1");

pub fn write(fig: &Figure, path: &std::path::Path) -> anyhow::Result<()> {
    let (fw, fh) = (fig.frame.w as f32, fig.frame.h as f32);
    let mut map_w = MAP_W;
    let mut map_h = MAP_W * fh / fw;
    if map_h > MAP_H_MAX {
        map_w *= MAP_H_MAX / map_h;
        map_h = MAP_H_MAX;
    }
    let column_h = column_height(fig);
    let page_w = MARGIN + map_w + GAP + COLUMN + MARGIN;
    let page_h = MARGIN + TITLE_H + map_h.max(column_h) + MARGIN;
    let top = page_h - MARGIN;

    let mut c = Content::new();

    // Title.
    text(&mut c, BOLD, 12., (MARGIN, top - 12.), [0.1; 3], &fig.title);
    text(
        &mut c,
        REGULAR,
        8.,
        (MARGIN, top - 25.),
        [0.4; 3],
        &fig.subtitle,
    );

    // Map.
    let (mx, my) = (MARGIN, top - TITLE_H - map_h);
    c.save_state();
    c.transform([map_w, 0., 0., map_h, mx, my]);
    c.x_object(MAP);
    c.restore_state();

    if let Some(units) = fig.units {
        scale_bar(&mut c, fig, units, (mx, my), map_w / fw);
    }

    // Column: legend, then markers.
    let x = mx + map_w + GAP;
    let mut y = top - TITLE_H - 9.;
    match &fig.legend {
        Legend::Communities(entries) => {
            for e in entries {
                let (colour, ink) = (e.colour, if e.on { [0.1; 3] } else { [0.6; 3] });
                swatch(&mut c, (x, y - 1.), colour);
                text(&mut c, REGULAR, 8.5, (x + 13., y), ink, &e.label);
                let n = thousands(e.count);
                let w = width(&n, 8.5);
                text(&mut c, REGULAR, 8.5, (x + COLUMN - w, y), ink, &n);
                y -= LINE;
            }
        }
        Legend::Ramp { title, stops, top } => {
            text(&mut c, BOLD, 8.5, (x, y), [0.1; 3], title);
            y -= LINE + 2.;
            let bar_w = COLUMN - 20.;
            let step = bar_w / stops.len() as f32;
            for (i, s) in stops.iter().enumerate() {
                c.set_fill_rgb(unit(s[0]), unit(s[1]), unit(s[2]));
                // Overlap by a hair so no seams show between steps.
                c.rect(x + i as f32 * step, y - 2., step + 0.3, 9.);
                c.fill_nonzero();
            }
            y -= LINE;
            text(&mut c, REGULAR, 8., (x, y - 1.), [0.3; 3], "0");
            text(
                &mut c,
                REGULAR,
                8.,
                (x + bar_w - width(top, 8.), y - 1.),
                [0.3; 3],
                top,
            );
            y -= LINE;
        }
    }

    for block in &fig.markers {
        y -= 8.;
        swatch(&mut c, (x, y - 1.), block.colour);
        text(&mut c, BOLD, 8.5, (x + 13., y), [0.1; 3], &block.title);
        text(
            &mut c,
            REGULAR,
            7.5,
            (x + COLUMN - width("fold", 7.5), y),
            [0.5; 3],
            "fold",
        );
        y -= LINE;
        for (gene, fold) in &block.genes {
            text(&mut c, REGULAR, 8., (x + 13., y), [0.15; 3], gene);
            let f = format!("×{fold:.1}");
            text(
                &mut c,
                REGULAR,
                8.,
                (x + COLUMN - width(&f, 8.), y),
                [0.3; 3],
                &f,
            );
            y -= 10.;
        }
    }

    // Assemble.
    let (catalog, pages, page, content, image, regular, bold) = (
        Ref::new(1),
        Ref::new(2),
        Ref::new(3),
        Ref::new(4),
        Ref::new(5),
        Ref::new(6),
        Ref::new(7),
    );
    let mut pdf = Pdf::new();
    pdf.catalog(catalog).pages(pages);
    pdf.pages(pages).kids([page]).count(1);
    let mut p = pdf.page(page);
    p.media_box(Rect::new(0., 0., page_w, page_h));
    p.parent(pages);
    p.contents(content);
    {
        let mut res = p.resources();
        res.x_objects().pair(MAP, image);
        res.fonts().pair(REGULAR, regular).pair(BOLD, bold);
    }
    p.finish();

    let rgb: Vec<u8> = fig
        .frame
        .rgba
        .chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(&rgb)?;
    let pixels = z.finish()?;
    let mut img = pdf.image_xobject(image, &pixels);
    img.filter(Filter::FlateDecode);
    img.width(fig.frame.w as i32);
    img.height(fig.frame.h as i32);
    img.color_space().device_rgb();
    img.bits_per_component(8);
    img.finish();

    for (id, name) in [(regular, &b"Helvetica"[..]), (bold, b"Helvetica-Bold")] {
        pdf.type1_font(id)
            .base_font(Name(name))
            .encoding_predefined(Name(b"WinAnsiEncoding"));
    }
    pdf.stream(content, &c.finish());
    std::fs::write(path, pdf.finish())?;
    Ok(())
}

fn column_height(fig: &Figure) -> f32 {
    let legend = match &fig.legend {
        Legend::Communities(e) => e.len() as f32 * LINE,
        Legend::Ramp { .. } => 3. * LINE + 2.,
    };
    let markers: f32 = fig
        .markers
        .iter()
        .map(|b| 8. + LINE + b.genes.len() as f32 * 10.)
        .sum();
    legend + markers + 12.
}

/// The theme's bar colour with a thin outline, bottom left of the map.
fn scale_bar(c: &mut Content, fig: &Figure, units: Units, (mx, my): (f32, f32), pt_per_px: f32) {
    let bar = scalebar::bar_for(fig.vp.w as f32 * fig.vp.upp, units);
    let len = bar.length / fig.vp.upp * pt_per_px;
    let (x, y) = (mx + 8., my + 8.);
    let (fill, edge) = fig.theme.bar();
    let (fill, edge) = (fill.map(unit), edge.map(unit));
    c.set_fill_rgb(fill[0], fill[1], fill[2]);
    c.set_stroke_rgb(edge[0], edge[1], edge[2]);
    c.set_line_width(0.5);
    c.rect(x, y, len, 2.5);
    c.fill_nonzero_and_stroke();
    // A halo in the outline colour under the label.
    for (dx, dy) in [(-0.5, 0.), (0.5, 0.), (0., -0.5), (0., 0.5)] {
        text(c, BOLD, 8., (x + dx, y + 5. + dy), edge, &bar.label);
    }
    text(c, BOLD, 8., (x, y + 5.), fill, &bar.label);
}

fn swatch(c: &mut Content, (x, y): (f32, f32), colour: Rgb) {
    c.set_fill_rgb(unit(colour[0]), unit(colour[1]), unit(colour[2]));
    c.rect(x, y, 9., 7.5);
    c.fill_nonzero();
}

fn text(c: &mut Content, font: Name, size: f32, (x, y): (f32, f32), rgb: [f32; 3], s: &str) {
    c.set_fill_rgb(rgb[0], rgb[1], rgb[2]);
    c.begin_text();
    c.set_font(font, size);
    c.set_text_matrix([1., 0., 0., 1., x, y]);
    c.show(Str(&win_ansi(s)));
    c.end_text();
}

fn unit(v: u8) -> f32 {
    v as f32 / 255.
}

/// Encode for the base fonts' WinAnsi table; unknown characters become '?'.
pub(super) fn win_ansi(s: &str) -> Vec<u8> {
    s.chars()
        .map(|ch| match ch {
            ' '..='~' => ch as u8,
            'µ' => 0xB5,
            '×' => 0xD7,
            '·' => 0xB7,
            '…' => 0x85,
            _ => b'?',
        })
        .collect()
}

/// Helvetica advance width, points, for right-aligning numbers. Digits are
/// exact; other characters use a typical width.
fn width(s: &str, size: f32) -> f32 {
    let units: u32 = s
        .chars()
        .map(|ch| match ch {
            '0'..='9' => 556,
            ',' | '.' | ' ' => 278,
            '×' => 584,
            'f' | 't' => 278,
            'o' | 'd' => 556,
            'l' | 'i' => 222,
            _ => 556,
        })
        .sum();
    units as f32 / 1000. * size
}
