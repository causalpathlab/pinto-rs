use crate::view::color::Theme;
use crate::view::pdf::*;
use crate::view::render::{Frame, Viewport};
use crate::view::scalebar::Units;

#[test]
fn writes_a_pdf_with_the_map_and_the_labels() {
    let frame = Frame {
        w: 40,
        h: 30,
        rgba: vec![128; 40 * 30 * 4],
        background: Theme::Dark.background(),
    };
    let fig = Figure {
        frame: &frame,
        vp: Viewport {
            x0: 0.,
            y0: 0.,
            upp: 10.,
            w: 40,
            h: 30,
        },
        title: "run · final · argmax".into(),
        subtitle: "1,000 cells".into(),
        units: Some(Units::Micron),
        legend: Legend::Communities(vec![Entry {
            label: "C3".into(),
            count: 1234,
            colour: [200, 10, 10],
            on: true,
        }]),
        markers: vec![MarkerBlock {
            title: "C3 markers".into(),
            colour: [200, 10, 10],
            genes: vec![("GENE1".into(), 55.2)],
        }],
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fig.pdf");
    write(&fig, &path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(bytes.starts_with(b"%PDF-"));
    assert!(text.contains("/Helvetica-Bold") && text.contains("/Width 40"));
    // Content streams are uncompressed, so labels appear verbatim.
    assert!(text.contains("(GENE1)") && text.contains("(1,234)"));
    // 400 units wide → a 50 µm bar. pdf-writer hex-encodes strings
    // with non-ASCII bytes: "50 µm" in WinAnsi is 35 30 20 B5 6D.
    assert!(text.contains("<353020B56D>"), "scale bar label");
}

#[test]
fn win_ansi_maps_the_symbols_labels_use() {
    assert_eq!(win_ansi("5 µm ×2.0"), b"5 \xB5m \xD72.0");
    assert_eq!(win_ansi("✓"), b"?");
}

/// A focused PDF of a level with no marker tables (a truth run) is drawn
/// without markers rather than refused.
#[test]
fn a_focused_pdf_of_a_level_without_markers_is_drawn() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, _) = super::lra::lra_run(dir.path());
    let pdf = dir.path().join("a.pdf");
    let png = dir.path().join("a.png");
    let (pdf, png) = (pdf.to_str().unwrap(), png.to_str().unwrap());
    let args = super::view_args(&prefix, &["--pdf", pdf, "--png", png, "--focus", "C0"]);
    crate::view::write_still(&args).unwrap();
    assert!(std::path::Path::new(pdf).is_file() && std::path::Path::new(png).is_file());
}

/// `--png` and `--pdf` make the folders they go in, and a file that
/// cannot be written is named.
#[test]
fn a_still_makes_its_folder_and_names_a_path_it_cannot_write() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, _) = super::lra::lra_run(dir.path());
    let png = dir.path().join("new/sub/a.png");
    let pdf = dir.path().join("new/b.pdf");
    let (png, pdf) = (png.to_str().unwrap(), pdf.to_str().unwrap());
    crate::view::write_still(&super::view_args(&prefix, &["--png", png, "--pdf", pdf])).unwrap();
    assert!(std::path::Path::new(png).is_file() && std::path::Path::new(pdf).is_file());

    // A file where its folder would be.
    std::fs::write(dir.path().join("file"), "").unwrap();
    let blocked = dir.path().join("file/c.png");
    let blocked = blocked.to_str().unwrap();
    for flag in ["--png", "--pdf"] {
        let e = crate::view::write_still(&super::view_args(&prefix, &[flag, blocked])).unwrap_err();
        assert!(format!("{e:#}").contains(blocked), "{flag}: {e:#}");
    }
}
