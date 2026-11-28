//! An lra pair on the links: a `--link-community` table drawn through its
//! community's links alone, each link's share of its receiver's exposure,
//! and the curve `view_lra` and `view_map` return.

use super::lra::{lra_run, receivers};
use crate::lr_activity::ligand_effect::frame::RunFrame;
use crate::util::metadata::LraOptions;
use crate::view::color::Theme;
use crate::view::data::Edges;
use crate::view::index::{EdgeIndex, Links};
use crate::view::lra::{pair_links, LraView, Receivers};
use crate::view::Base;

/// The fixture's grid side (as [`lra_run`] builds it).
const SIDE: usize = 10;

/// The fixture's contacts: each cell to its right and lower neighbour.
fn grid_pairs() -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for c in 0..SIDE * SIDE {
        if c % SIDE + 1 < SIDE {
            pairs.push((c, c + 1));
        }
        if c / SIDE + 1 < SIDE {
            pairs.push((c, c + SIDE));
        }
    }
    pairs
}

/// Whether cell `c` is in the left two columns: community 1's cells.
fn left(c: usize) -> bool {
    c % SIDE < 2
}

/// The map's links on `pairs`, and one between two cells that touch no
/// more than the corners of the tissue (no contact).
fn links_on(base: &Base, pairs: &[(usize, usize)]) -> Links {
    let geom = &base.geom;
    let mut edges = Edges::default();
    for &(i, j) in pairs.iter().chain(&[(0, SIDE * SIDE - 1)]) {
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
fn a_link_community_table_draws_its_communitys_receivers_through_its_links() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, contacts) = lra_run(dir.path());
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    let pairs = grid_pairs();
    assert_eq!(pairs, contacts.pairs, "the fixture's contacts");
    let community: Vec<i64> = pairs
        .iter()
        .map(|&(i, j)| if left(i) && left(j) { 1 } else { 2 })
        .collect();
    let options = LraOptions {
        link_community: Some(1),
        ..LraOptions::default()
    };
    let frame = RunFrame::without_reference(contacts, &[])
        .unwrap()
        .in_link_community(1, &community)
        .unwrap();
    let rx = Receivers::new(frame, &base, &options);
    let mut tables = base.run.tables();
    tables[0].options = options.clone();
    let mut v = LraView::new(tables).unwrap().unwrap();
    v.receivers = Some(rx);
    v.select("L1:R1").unwrap();
    v.on_map(&base, 0).unwrap().unwrap();
    let on = v.shown().unwrap();
    // only the community's receivers are drawn: the two left columns
    let drawn: usize = on.legend().iter().map(|l| l.2).sum();
    assert_eq!(drawn, 2 * SIDE);

    // each one's exposure is L1 (2 on the three left columns) over the size
    // factor, summed over its community-1 neighbours alone: summed here by
    // hand from the grid, not through the frame's contacts
    let rx = v.receivers.as_ref().unwrap();
    let frame = &rx.frame;
    let row_of = |c: usize| frame.cell.iter().position(|&x| x == c).unwrap();
    for r in 0..frame.n() {
        let c = frame.cell[r];
        let (x, y) = (c % SIDE, c / SIDE);
        let mut neighbours = Vec::new();
        if x == 0 {
            neighbours.push(c + 1);
        }
        if x == 1 {
            neighbours.push(c - 1);
        }
        if left(c) && y > 0 {
            neighbours.push(c - SIDE);
        }
        if left(c) && y + 1 < SIDE {
            neighbours.push(c + SIDE);
        }
        let want: f64 = neighbours
            .iter()
            .map(|&j| 2.0 / frame.size_factor[row_of(j)])
            .sum();
        assert!(
            (on.map.exposure[r] - want).abs() < 1e-9,
            "cell {c}: {} against {want}",
            on.map.exposure[r]
        );
        assert_eq!(on.map.in_scope(r), left(c), "cell {c}");
    }

    // its links: community 1's in scope, every other one out
    let links = links_on(&base, &pairs);
    let pair = pair_links(&links, rx, &on.map);
    for (e, &(i, j)) in pairs.iter().enumerate() {
        assert_eq!(pair.in_scope[e], left(i) && left(j), "link {i}-{j}");
    }
    assert!(!pair.in_scope[pairs.len()], "no contact");
}

#[test]
fn a_responders_links_carry_its_whole_exposure() {
    let dir = tempfile::tempdir().unwrap();
    let (prefix, contacts) = lra_run(dir.path());
    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    let rx = receivers(&base, contacts);
    let map = rx.frame.pair("L1", "R1", Some(0.0)).unwrap();
    let adj = &rx.frame.exposure_contacts().adj;
    let mut responders = 0;
    for r in 0..map.n() {
        let shares: f64 = adj[r].iter().map(|&j| map.share(j as usize, r)).sum();
        if map.exposed[r] && map.competent[r] {
            responders += 1;
            assert!((shares - 1.0).abs() < 1e-9, "row {r}: {shares}");
        } else {
            assert_eq!(shares, 0.0, "row {r} responds to nothing");
        }
    }
    assert!(responders > 0);

    // on the map's links, each the larger of its two directions
    let pairs = grid_pairs();
    let links = links_on(&base, &pairs);
    let pair = pair_links(&links, &rx, &map);
    let row_of = |c: usize| rx.frame.cell.iter().position(|&x| x == c).unwrap();
    for (e, &(i, j)) in pairs.iter().enumerate() {
        let (a, b) = (row_of(i), row_of(j));
        let (ab, ba) = (map.share(a, b), map.share(b, a));
        assert!(pair.in_scope[e]);
        assert_eq!(pair.share[e], ab.max(ba) as f32, "link {i}-{j}");
        if ab != ba {
            assert_eq!(pair.a_sends[e], ab > ba, "link {i}-{j}");
        }
    }
    assert!(pair.counts.1 > 0, "some links carry L1");
}

#[test]
fn view_lra_returns_the_curve_and_view_map_draws_it_under_the_map() {
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
    let e = map(&mut open, json!({"curve": true}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("no lra pair selected"), "{e}");

    let base = Base::load(&prefix, "auto", Theme::Dark).unwrap();
    open.lra.as_mut().unwrap().receivers = Some(receivers(&base, contacts));
    let r = lra(&mut open, json!({"pair": "L1:R1"})).unwrap();
    let curve = r["selected"]["curve"].as_array().unwrap();
    assert_eq!(curve.len(), 4);
    assert!(curve.iter().all(|p| p["tau"].is_number()));
    assert!(
        r["selected"]["map"]["links_in_scope"].is_null(),
        "no link table"
    );
    assert_eq!(pngs(&map(&mut open, json!({})).unwrap()), 1);
    assert_eq!(pngs(&map(&mut open, json!({"curve": true})).unwrap()), 2);
}
