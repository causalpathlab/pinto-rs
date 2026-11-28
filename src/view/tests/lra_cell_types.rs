//! An lra pair's links by cell type: the counts and shares against the
//! grid summed by hand, the links left to one type pair, the cell types
//! read from a file in place of the table's, and what `view_lra` and
//! `view_map` return.

use super::lra::{lra_run, receivers};
use crate::lr_activity::input::Contacts;
use crate::lr_activity::ligand_effect::link_cell_types::type_pair_named;
use crate::util::metadata::LraOptions;
use crate::view::color::Theme;
use crate::view::data::Edges;
use crate::view::index::{EdgeIndex, Links};
use crate::view::lra::{pair_links, LraView};
use crate::view::Base;

/// The fixture's grid side (as [`lra_run`] builds it).
const SIDE: usize = 10;

/// Column `c`'s type: `left` on the grid's left half, else `right`.
fn type_of(c: usize) -> u32 {
    u32::from(c % SIDE >= SIDE / 2)
}

/// `contacts` with the left and right halves as cell types.
fn with_halves(mut contacts: Contacts) -> Contacts {
    let n = contacts.coords.len();
    let codes = (0..n).map(type_of).collect();
    let names = vec!["left".into(), "right".into(), "unassigned".into()];
    contacts.cell_type = Some((codes, names));
    contacts.cell_type_source = Some("the grid's halves".into());
    contacts
}

/// The map's links on the fixture's contacts.
fn links_on(base: &Base, pairs: &[(usize, usize)]) -> Links {
    let geom = &base.geom;
    let mut edges = Edges::default();
    for &(i, j) in pairs {
        edges.push(geom, &format!("d_{i}"), &format!("d_{j}"), 0);
    }
    let index = EdgeIndex::build(&base.grid, &geom.x, &geom.y, &edges);
    Links {
        edges: edges.into(),
        index: index.into(),
        ends: None,
        kinds: (0, 0),
        pair: None,
    }
}

#[test]
fn a_pairs_links_by_cell_type_add_up_to_its_links_and_responders() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, contacts) = lra_run(dir.path());
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    let pairs = contacts.pairs.clone();
    let rx = receivers(&base, with_halves(contacts));
    let map = rx.frame.pair("L1", "R1", Some(0.0)).unwrap();
    let codes = rx.type_of_row.as_ref().unwrap();
    let names = rx.frame.cell_type_names().unwrap();
    let types = crate::lr_activity::ligand_effect::link_cell_types::links_by_cell_type(
        rx.frame.exposure_contacts(),
        &map,
        codes,
        names,
    );

    // every contact once in each direction
    let links: usize = types.cells.iter().map(|c| c.links).sum();
    assert_eq!(links, 2 * pairs.len());
    // the halves meet along one column boundary: SIDE links each way
    let across = pairs
        .iter()
        .filter(|&&(i, j)| type_of(i) != type_of(j))
        .count();
    assert_eq!(across, SIDE);
    let cell = |s, r| {
        types
            .cells
            .iter()
            .find(|c| (c.sender_type, c.receiver_type) == (s, r))
            .unwrap()
    };
    assert_eq!(cell(0, 1).links, SIDE);
    assert_eq!(cell(1, 0).links, SIDE);

    // a responder's shares sum to 1: a receiver type's sum to its responders
    for t in 0..2u32 {
        let want = (0..map.n())
            .filter(|&r| codes[r] == t && map.exposed[r] && map.competent[r])
            .count();
        let got: f64 = types
            .cells
            .iter()
            .filter(|c| c.receiver_type == t)
            .map(|c| c.share)
            .sum();
        assert!(
            (got - want as f64).abs() < 1e-9,
            "type {t}: {got} against {want}"
        );
    }
    assert!(types.responders > 0);
    assert!(types.cells.windows(2).all(|w| w[0].share >= w[1].share));

    // the links left to left → right: the boundary's alone
    let links = links_on(&base, &pairs);
    let row_of = |c: usize| rx.frame.cell.iter().position(|&x| x == c).unwrap();
    let pair = pair_links(&links, &rx, &map, Some((0, 1)));
    for (e, &(i, j)) in pairs.iter().enumerate() {
        assert_eq!(pair.in_scope[e], type_of(i) != type_of(j), "link {i}-{j}");
        // only the left end's share into the right end counts
        let (from, to) = if type_of(i) == 0 { (i, j) } else { (j, i) };
        if pair.in_scope[e] {
            let want = map.share(row_of(from), row_of(to)) as f32;
            assert_eq!(pair.share[e], want, "link {i}-{j}");
        }
    }
    assert_eq!(pair.counts.0, SIDE);
}

#[test]
fn a_type_pair_is_named_by_its_types() {
    let names: Vec<Box<str>> = vec!["B cell".into(), "T cell".into()];
    assert_eq!(type_pair_named(&names, "b cell > T cell"), Some((0, 1)));
    assert_eq!(type_pair_named(&names, "T cell → B cell"), Some((1, 0)));
    assert_eq!(type_pair_named(&names, "T cell > NK"), None);
    assert_eq!(type_pair_named(&names, "T cell"), None);
    let odd: Vec<Box<str>> = vec!["CD4>CD8".into(), "NK".into()];
    assert_eq!(type_pair_named(&odd, "CD4>CD8 > NK"), Some((0, 1)));
    assert_eq!(type_pair_named(&odd, "NK>CD4>CD8"), Some((1, 0)));
}

#[test]
fn cell_types_from_a_file_stand_in_for_the_tables() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, _) = lra_run(dir.path());
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    let file = dir.path().join("types.tsv");
    let mut text = String::from("cell\tcell_type\n");
    for c in 0..SIDE * SIDE {
        let t = if c / SIDE < SIDE / 2 { "top" } else { "bottom" };
        text.push_str(&format!("d_{c}\t{t}\n"));
    }
    std::fs::write(&file, text).unwrap();
    let mut tables = base.run.tables();
    // read from the run with the genes recorded: no reference needed
    tables[0].options = LraOptions {
        identity_excluded: vec!["L1".into(), "R1".into()],
        ..LraOptions::default()
    };
    let mut v = LraView::new(tables).unwrap().unwrap();
    assert!(v.set_cell_types(Some("no-such.tsv".into())).is_err());
    v.set_cell_types(Some(file.to_string_lossy().into()))
        .unwrap();
    // runs keep the table's own cell types; only the drawing reads the file
    assert_eq!(v.options().cell_types, None);
    let file_name = file.to_string_lossy().to_string();
    assert_eq!(v.drawn_options().cell_types, Some(file_name));
    v.select("L1:R1").unwrap();
    v.on_map(&base, 0).unwrap().unwrap();
    let (_, types) = v.shown_types().unwrap();
    assert_eq!(
        &types.names[..2],
        &["bottom".into(), "top".into()] as &[Box<str>]
    );
    let from = v.cell_types_from().unwrap();
    assert!(
        from.contains("types.tsv") && from.contains("in place of"),
        "{from}"
    );
    let options = v.drawn_options();
    let mut pair = |text: &str| v.type_pair(&base, &options, text);
    assert_eq!(
        pair("top > bottom").unwrap(),
        Some(("top".into(), "bottom".into()))
    );
    assert!(pair("top > left").is_err());
    assert_eq!(pair("").unwrap(), None);
}

#[test]
fn view_lra_lists_links_by_type_and_view_map_draws_their_heatmap() {
    use crate::view::mcp::{LraArgs, MapArgs, Open};
    use legume_mcp::Content;
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let (prefix, contacts) = lra_run(dir.path());
    let mut open = Open::new(&prefix, None).unwrap();
    let lra = |open: &mut Open, a: serde_json::Value| {
        let a: LraArgs = serde_json::from_value(a).unwrap();
        open.lra_with(a, &|_| {})
    };
    let map = |open: &mut Open, a: serde_json::Value| {
        let a: MapArgs = serde_json::from_value(a).unwrap();
        open.map(a)
    };
    let pngs = |content: &[Content]| {
        content
            .iter()
            .filter(|c| matches!(c, Content::Png(_)))
            .count()
    };
    lra(&mut open, json!({})).unwrap();
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    let types_file = dir.path().join("other.tsv");
    std::fs::write(&types_file, "cell\tcell_type\nd_0\tx\n").unwrap();
    let types_file = types_file.to_string_lossy().to_string();

    // without cell types: said so, and no heatmap
    open.lra.as_mut().unwrap().receivers = Some(receivers(&base, contacts));
    let r = lra(&mut open, json!({"pair": "L1:R1"})).unwrap();
    assert!(r["selected"]["map"]["cell_types"]["from"].is_null());
    let e = map(&mut open, json!({"type_heatmap": true}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("no cell types"), "{e}");

    // with them: listed, a type pair kept, the heatmap a second image
    let (_, contacts) = lra_run(dir.path());
    let mut open = Open::new(&prefix, None).unwrap();
    lra(&mut open, json!({})).unwrap();
    open.lra.as_mut().unwrap().receivers = Some(receivers(&base, with_halves(contacts)));
    let r = lra(
        &mut open,
        json!({"pair": "L1:R1", "type_pair": "left > right"}),
    )
    .unwrap();
    let types = &r["selected"]["map"]["cell_types"];
    assert_eq!(types["type_pair"], "left > right");
    let listed = types["links"].as_array().unwrap();
    assert_eq!(listed.len(), types["type_pairs"].as_u64().unwrap() as usize);
    let shares: f64 = listed.iter().map(|c| c["share"].as_f64().unwrap()).sum();
    assert!((shares - types["responders"].as_f64().unwrap()).abs() < 1e-9);
    let e = lra(&mut open, json!({"type_pair": "left > nowhere"}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("left, right"), "{e}");
    assert_eq!(
        pngs(&map(&mut open, json!({"type_heatmap": true})).unwrap()),
        2
    );
    let both = map(&mut open, json!({"type_heatmap": true, "curve": true})).unwrap();
    assert_eq!(pngs(&both), 3);
    // a call that fails keeps the cell types and the type pair it had
    let e = lra(
        &mut open,
        json!({"cell_types": types_file, "pair": "NOPE:R1", "type_pair": ""}),
    );
    assert!(e.is_err());
    let r = lra(&mut open, json!({})).unwrap();
    assert_eq!(
        r["selected"]["map"]["cell_types"]["type_pair"],
        "left > right"
    );
    let r = lra(&mut open, json!({"type_pair": ""})).unwrap();
    assert!(r["selected"]["map"]["cell_types"]["type_pair"].is_null());
}
