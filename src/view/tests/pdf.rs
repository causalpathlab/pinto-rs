use crate::view::pdf::*;
use crate::view::render::{Frame, Viewport};
use crate::view::scalebar::Units;

#[test]
fn writes_a_pdf_with_the_map_and_the_labels() {
    let frame = Frame {
        w: 40,
        h: 30,
        rgba: vec![128; 40 * 30 * 4],
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
            genes: vec![("CD3E".into(), 55.2)],
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
    assert!(text.contains("(CD3E)") && text.contains("(1,234)"));
    // 400 units wide → a 50 µm bar. pdf-writer hex-encodes strings
    // with non-ASCII bytes: "50 µm" in WinAnsi is 35 30 20 B5 6D.
    assert!(text.contains("<353020B56D>"), "scale bar label");
}

#[test]
fn win_ansi_maps_the_symbols_labels_use() {
    assert_eq!(win_ansi("5 µm ×2.0"), b"5 \xB5m \xD72.0");
    assert_eq!(win_ansi("✓"), b"?");
}
