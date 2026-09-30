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

#[test]
fn saves_are_listed_newest_first_with_a_thumbnail_each() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join(".pinto-view");
    let (a, b) = (dir.path().join("a.pdf"), dir.path().join("b.png"));
    std::fs::write(&a, "a").unwrap();
    std::fs::write(&b, "b").unwrap();
    let mut g = Gallery::open(&store);
    g.add(&a, "map", &frame(480, 360)).unwrap();
    g.add(&b, "structure", &frame(100, 50)).unwrap();
    g.add(&a, "map again", &frame(480, 360)).unwrap();

    let g = Gallery::open(&store);
    let names: Vec<String> = g.entries().iter().map(Entry::name).collect();
    assert_eq!(names, ["a.pdf", "b.png"]);
    assert_eq!(g.entries()[0].what, "map again");
    // Shrunk to the thumbnail width at the picture's aspect; never enlarged.
    let t = read_thumb(&g.thumb(&g.entries()[0]), [0; 3]).unwrap();
    assert_eq!((t.w, t.h), (THUMB_WIDTH, THUMB_WIDTH * 3 / 4));
    let t = read_thumb(&g.thumb(&g.entries()[1]), [0; 3]).unwrap();
    assert_eq!((t.w, t.h), (100, 50));
    // One thumbnail per entry: the replaced one is gone.
    assert_eq!(std::fs::read_dir(store.join("thumbs")).unwrap().count(), 2);

    // A deleted file leaves the list, its thumbnail with it.
    std::fs::remove_file(&b).unwrap();
    let g = Gallery::open(&store);
    assert_eq!(g.entries().len(), 1);
    assert_eq!(std::fs::read_dir(store.join("thumbs")).unwrap().count(), 1);
}

#[test]
fn ages_read_in_the_largest_whole_unit() {
    assert_eq!(ago(100, 130), "just now");
    assert_eq!(ago(0, 7200), "2 h ago");
    assert_eq!(ago(0, 3 * 86_400 + 5), "3 d ago");
}
