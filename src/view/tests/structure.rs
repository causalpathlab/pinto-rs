use crate::view::color::Theme;
use crate::view::data::{Communities, NO_CLUSTER};
use crate::view::structure::Structure;

/// Five cells over K = 2: shares of community 0 given, the rest is 1's.
fn comm(share0: &[u8]) -> Communities {
    let group: Vec<u16> = share0.iter().map(|&q| u16::from(q < 128)).collect();
    let mut c = Communities::from_groups("t", group, vec!["a".into(), "b".into()]);
    for (i, &q) in share0.iter().enumerate() {
        c.prop[i * 2] = q;
        c.prop[i * 2 + 1] = 255 - q;
    }
    c
}

#[test]
fn cells_sort_by_dominant_community_then_share() {
    // Community 1 dominates more cells overall, so it stacks first.
    let c = comm(&[200, 10, 250, 60, 100]);
    let s = Structure::build(&c, &[0; 5], &["all".into()], &[0]);
    assert_eq!(s.stack, [1, 0]);
    // Community 1's cells by its share (1: 245, 3: 195, 4: 155), then 0's.
    assert_eq!(s.panels[0].1, [1, 3, 4, 2, 0]);
}

#[test]
fn panels_follow_the_given_order() {
    let c = comm(&[200, 10, 250, 60, 100]);
    let panel_of = [1, 0, 1, NO_CLUSTER, 0];
    let names = ["x".to_string(), "y".to_string()];
    let s = Structure::build(&c, &panel_of, &names, &[1, 0]);
    let got: Vec<(&str, usize)> = s
        .panels
        .iter()
        .map(|(n, c)| (n.as_str(), c.len()))
        .collect();
    // The cell with no group goes last, as unassigned.
    assert_eq!(got, [("y", 2), ("x", 2), ("unassigned", 1)]);
}

#[test]
fn bars_stack_from_the_bottom_in_palette_colours() {
    let c = comm(&[255, 255, 0, 0, 0]);
    let s = Structure::build(&c, &[0; 5], &["all".into()], &[0]);
    let palette = [[255, 0, 0], [0, 0, 255]];
    let d = s.render(&c, &palette, (5, 10), None, Theme::Dark);
    let frame = &d.frame;
    assert_eq!((d.panels[0].x0, d.panels[0].x1), (0, 4));
    let px = |x, y| {
        let o = frame.offset(x, y);
        [frame.rgba[o], frame.rgba[o + 1], frame.rgba[o + 2]]
    };
    // Community 1 (three cells) first, all blue; then 0's cells, all red.
    assert_eq!(px(0, 9), [0, 0, 255]);
    assert_eq!(px(0, 0), [0, 0, 255]);
    assert_eq!(px(4, 0), [255, 0, 0]);
    // Each pixel knows its community, for clicks.
    assert_eq!(
        (d.at(0, 9), d.at(4, 0), d.at(5, 0)),
        (Some(1), Some(0), None)
    );
}

#[test]
fn a_focus_dims_the_other_communities() {
    let c = comm(&[255, 255, 0, 0, 0]);
    let s = Structure::build(&c, &[0; 5], &["all".into()], &[0]);
    let palette = [[255, 0, 0], [0, 0, 255]];
    let d = s.render(&c, &palette, (5, 10), Some(&[true, false]), Theme::Dark);
    let o = d.frame.offset(0, 9);
    assert_eq!(&d.frame.rgba[o..o + 3], &Theme::Dark.dimmed());
    let o = d.frame.offset(4, 0);
    assert_eq!(&d.frame.rgba[o..o + 3], &[255, 0, 0]);
    // A dimmed community can still be clicked.
    assert_eq!(d.at(0, 9), Some(1));
}

#[test]
fn cells_no_group_claims_go_last_as_unassigned() {
    let c = comm(&[200, 10, 250, 60, 100]);
    let panel_of = [0, NO_CLUSTER, 0, NO_CLUSTER, 0];
    let s = Structure::build(&c, &panel_of, &["x".to_string()], &[0]);
    let got: Vec<(&str, usize)> = s
        .panels
        .iter()
        .map(|(n, c)| (n.as_str(), c.len()))
        .collect();
    assert_eq!(got, [("x", 3), ("unassigned", 2)]);
}
