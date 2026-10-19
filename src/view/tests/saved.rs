use crate::view::render::Frame;
use crate::view::saved::*;

fn frame(w: usize, h: usize) -> Frame {
    Frame {
        w,
        h,
        rgba: (0..w * h)
            .flat_map(|i| [(i % 256) as u8, 0, 0, 255])
            .collect(),
        background: [0; 3],
    }
}

/// pinto's frame goes in as the thumbnail and comes back as one: the
/// listing itself is legume-tui's.
#[test]
fn a_save_is_listed_with_its_frame_as_thumbnail() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join(".pinto-view");
    let fig = dir.path().join("b.png");
    std::fs::write(&fig, "b").unwrap();
    let mut g = open(&store);
    add(&mut g, &fig, "structure", &frame(100, 50)).unwrap();

    let g = open(&store);
    assert_eq!(g.entries()[0].name(), "b.png");
    assert_eq!(g.entries()[0].what, "structure");
    let t = read_thumb(&g.thumb(&g.entries()[0]).unwrap(), [0; 3]).unwrap();
    assert_eq!((t.w, t.h), (100, 50));
    assert_eq!(&t.rgba[4..8], &[1, 0, 0, 255]);
}

/// The log pinto wrote before its gallery came from legume-tui still
/// lists its saves.
#[test]
fn an_earlier_log_still_lists_its_saves() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join(".pinto-view");
    let fig = dir.path().join("fig.pdf");
    std::fs::write(&fig, "f").unwrap();
    std::fs::create_dir_all(store.join("thumbs")).unwrap();
    frame(4, 3)
        .write_png(&store.join("thumbs").join("1.png"))
        .unwrap();
    let log = serde_json::json!([{
        "path": fig,
        "what": "map · final · argmax",
        "when": 1,
        "thumb": "1.png",
    }]);
    std::fs::write(store.join("saved.json"), log.to_string()).unwrap();

    let g = open(&store);
    assert_eq!(g.entries().len(), 1);
    assert_eq!(g.entries()[0].what, "map · final · argmax");
    let t = read_thumb(&g.thumb(&g.entries()[0]).unwrap(), [0; 3]).unwrap();
    assert_eq!((t.w, t.h), (4, 3));
}
