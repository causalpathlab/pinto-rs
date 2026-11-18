//! `pinto view` on a small run, driven through a pseudo-terminal: `u`
//! through every layout and back to the tissue, then `U` to the next
//! table, each one made and drawn, nothing drawn over the screen.

#![cfg(all(unix, feature = "view"))]

mod common;

use legume_tui::testing::Private;
use legume_tui::Size;
use std::time::Duration;

const SIZE: Size = Size {
    rows: 40,
    cols: 120,
    width: 1200,
    height: 800,
};

/// The layouts in the order `u` steps through them.
const LAYOUTS: [&str; 7] = [
    "UMAP",
    "tUMAP",
    "PHATE",
    "diffusion map",
    "ForceAtlas2",
    "LLE",
    "stress",
];

/// A smoke run of real layouts on 2,500 cells, about half a minute: not
/// in the default suite. `cargo test --release --test layout_smoke --
/// --ignored --nocapture` runs it and prints each screen.
#[test]
#[ignore]
fn every_layout_is_made_and_drawn_in_the_view() {
    use legume_numeric::matrix::traits::IoOps;
    let dir = tempfile::tempdir().unwrap();
    // past the landmarks, so the layouts that hold every pair use them
    let n = 2500;
    let prefix = common::write_run(dir.path(), n);
    // each cell its own mix, so the layouts have points to spread: the
    // left half mostly C0, the right mostly C1, each varied by its place
    let side = (n as f64).sqrt().ceil() as usize;
    let names: Vec<Box<str>> = (0..n).map(|i| format!("c{i}").into()).collect();
    let prop = nalgebra::DMatrix::<f32>::from_fn(n, 2, |i, j| {
        let (x, y) = ((i % side) as f32, (i / side) as f32);
        let c0 =
            1. / (1. + ((x - side as f32 / 2.) / 3.).exp()) * 0.8 + 0.1 + 0.05 * (y / side as f32);
        if j == 0 {
            c0
        } else {
            1. - c0
        }
    });
    let cols: Vec<Box<str>> = vec!["C0".into(), "C1".into()];
    prop.to_parquet_with_names(
        &format!("{prefix}.propensity.parquet"),
        (Some(&names), Some("cell")),
        Some(&cols),
    )
    .unwrap();
    // a second table of the cells: a cell embedding
    let emb = nalgebra::DMatrix::<f32>::from_fn(n, 3, |i, j| ((i * (j + 3)) % 17) as f32);
    let cols: Vec<Box<str>> = vec!["d0".into(), "d1".into(), "d2".into()];
    let path = format!("{prefix}.cell_embedding.parquet");
    emb.to_parquet_with_names(&path, (Some(&names), Some("cell")), Some(&cols))
        .unwrap();
    let manifest = format!("{prefix}.pinto.json");
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    m["outputs"]["cell_embedding"] = path.into();
    std::fs::write(&manifest, m.to_string()).unwrap();

    let private = Private::new(env!("CARGO_BIN_EXE_pinto")).patience(Duration::from_secs(120));
    let mut term = private.term_sized(&["view", &prefix, "--no-detach"], &[], SIZE);
    term.wait_screen("level");
    for name in LAYOUTS {
        term.type_(b"u");
        term.wait_screen(&format!("{name}: propensity final"));
        // the map repaints after the status line
        std::thread::sleep(Duration::from_millis(500));
        eprintln!("--- {name}\n{}", term.screen());
    }
    term.type_(b"u");
    term.wait_screen("the tissue");
    // U: the next table, then u lays the cells out by it
    term.type_(b"U");
    term.wait_screen("u lays the cells");
    term.type_(b"u");
    term.wait_screen("UMAP: cell embedding");
    eprintln!("--- UMAP of the embedding\n{}", term.screen());
    term.type_(b"q");
    assert_eq!(term.exit_code(), 0);
}
