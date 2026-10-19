use crate::view::color::Theme;
use crate::view::data::{Communities, NO_CLUSTER};
use crate::view::structure::Structure;
use crate::view::tests::fixture::on_page;

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
    assert_eq!(s.layout().stack, [1, 0]);
    // Community 1's cells by its share (1: 245, 3: 195, 4: 155), then 0's.
    assert_eq!(s.layout().panels[0].rows, [1, 3, 4, 2, 0]);
}

#[test]
fn panels_follow_the_given_order() {
    let c = comm(&[200, 10, 250, 60, 100]);
    let panel_of = [1, 0, 1, NO_CLUSTER, 0];
    let names = ["x".to_string(), "y".to_string()];
    let s = Structure::build(&c, &panel_of, &names, &[1, 0]);
    let got: Vec<(&str, usize)> = s
        .layout()
        .panels
        .iter()
        .map(|p| (p.name.as_str(), p.rows.len()))
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
    assert_eq!((d.panels()[0].x0, d.panels()[0].x1), (0, 4));
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
        .layout()
        .panels
        .iter()
        .map(|p| (p.name.as_str(), p.rows.len()))
        .collect();
    assert_eq!(got, [("x", 3), ("unassigned", 2)]);
}

/// The structure plot as a figure for a PDF: its title, panel names and
/// key are text over the bars, not pixels.
#[test]
fn the_figure_sets_its_title_panels_and_key_as_text() {
    let c = comm(&[200, 10, 250, 60, 100]);
    let panel_of = [1, 0, 1, 0, 0];
    let names = ["x".to_string(), "y".to_string()];
    let s = Structure::build(&c, &panel_of, &names, &[1, 0]);
    let palette = Theme::Light.palette(2);
    let page = s.figure(&c, &palette, "structure · batch", None, Theme::Light);
    let text: Vec<&str> = page.text.iter().map(|t| t.text.as_str()).collect();
    for want in ["structure · batch", "x", "y", "a", "b"] {
        assert!(text.contains(&want), "{want} in {text:?}");
    }
    let pdf = legume_tui::export::pdf_bytes(&[page], 300.).unwrap();
    assert!(pdf.starts_with(b"%PDF"));
}

/// Many communities and long panel names all fit their page: every key
/// entry is there, and no text runs off it.
#[test]
fn a_figure_is_as_tall_as_its_key_and_names() {
    let k = 30;
    let n = 300;
    let group: Vec<u16> = (0..n).map(|i| (i % k) as u16).collect();
    let names: Vec<Box<str>> = (0..k).map(|c| format!("C{c}").into()).collect();
    let mut c = Communities::from_groups("t", group, names);
    for i in 0..n {
        c.prop[i * k + i % k] = 255;
    }
    let panels: Vec<String> = (0..3)
        .map(|p| format!("a-very-long-panel-name-{p}"))
        .collect();
    let panel_of: Vec<u16> = (0..n).map(|i| (i % 3) as u16).collect();
    let s = Structure::build(&c, &panel_of, &panels, &[0, 1, 2]);
    let page = s.figure(
        &c,
        &Theme::Light.palette(k),
        "structure",
        None,
        Theme::Light,
    );
    let text: Vec<&str> = page.text.iter().map(|t| t.text.as_str()).collect();
    for c in 0..k {
        assert!(text.contains(&format!("C{c}").as_str()), "C{c} in the key");
    }
    on_page(&page);
}
