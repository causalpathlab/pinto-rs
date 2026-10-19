use crate::util::common::*;
use crate::view::color::Theme;
use crate::view::data::Communities;
use crate::view::heatmap::Heatmap;
use crate::view::markers::FeatureRates;
use crate::view::tests::fixture::on_page;

fn rates(rows: &[[f32; 3]]) -> FeatureRates {
    let names = (0..rows.len()).map(|i| format!("g{i}").into()).collect();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    FeatureRates {
        names,
        rates: Mat::from_row_slice(rows.len(), 3, &flat),
    }
}

/// Three groups, largest first: 2 (3 cells), 0 (2), 1 (1).
fn groups() -> Communities {
    let names = ["a", "b", "c"].map(Box::<str>::from).to_vec();
    Communities::from_groups("t", vec![2, 2, 2, 0, 0, 1], names)
}

#[test]
fn genes_lead_their_own_group_by_a_margin() {
    let fr = rates(&[
        [9., 1., 8.], // g0: peaks in a but only just ahead of c
        [5., 1., 1.], // g1: peaks in a well ahead
        [1., 1., 7.], // g2: peaks in c
        [3., 3., 1.], // g3: a tie peaks nowhere
        [1., 6., 1.], // g4: peaks in b
    ]);
    let h = Heatmap::build(&fr, &groups(), 1, Box::new(|_| None));
    assert_eq!(h.groups, [2, 0, 1]);
    let genes: Vec<(&str, usize)> = h.genes.iter().map(|(n, _, c)| (n.as_ref(), *c)).collect();
    // By margin g1 leads a, not g0 (which has the highest fold over the mean).
    assert_eq!(genes, [("g2", 2), ("g1", 0), ("g4", 1)]);
    // Every row peaks on the diagonal.
    for (r, row) in h.z.chunks(3).enumerate() {
        let top = (0..3).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        assert_eq!(top, r);
    }
    assert!(!h.observed);
}

#[test]
fn observed_counts_rank_the_models_candidates() {
    let fr = rates(&[
        [9., 1., 1.], // g0: the model's best for a
        [5., 1., 1.], // g1: second for a
        [4., 1., 1.], // g2: third for a, missing from the counts
    ]);
    let asked = std::cell::RefCell::new(Vec::new());
    let h = Heatmap::build(
        &fr,
        &groups(),
        1,
        Box::new(|names: &[&str]| {
            asked
                .borrow_mut()
                .extend(names.iter().map(|n| n.to_string()));
            // Means over the grouping's own groups a, b, c. In the counts g0
            // is flat and g1 leads a.
            Some(vec![Some(vec![1., 1., 1.]), Some(vec![3., 0., 1.]), None])
        }),
    );
    // Every candidate the model proposed was looked up.
    assert_eq!(*asked.borrow(), ["g0", "g1", "g2"]);
    assert!(h.observed);
    let genes: Vec<&str> = h.genes.iter().map(|(n, ..)| n.as_ref()).collect();
    assert_eq!(genes, ["g1"]);
    // Columns in drawing order: c, a, b.
    assert_eq!(h.values, [1., 3., 0.]);
    assert!(h
        .tsv(&groups())
        .starts_with("# mean ln(1+count)\ngene\tmarks\tc\ta\tb\ng1\ta\t"));
}

#[test]
fn without_counts_the_model_chooses() {
    let fr = rates(&[[5., 1., 1.]]);
    let h = Heatmap::build(&fr, &groups(), 1, Box::new(|_| Some(vec![None])));
    assert!(!h.observed);
    assert_eq!(h.genes.len(), 1);
}

/// The heatmap as a figure for a PDF: its genes and groups are text.
#[test]
fn the_figure_sets_its_genes_and_groups_as_text() {
    let fr = rates(&[[5., 1., 1.], [1., 1., 7.], [1., 6., 1.]]);
    let comm = groups();
    let h = Heatmap::build(&fr, &comm, 1, Box::new(|_| None));
    let palette = Theme::Light.palette(3);
    let page = h
        .figure(&comm, &palette, "heatmap · group", Theme::Light)
        .expect("drawn");
    let text: Vec<&str> = page.text.iter().map(|t| t.text.as_str()).collect();
    for want in ["heatmap · group", "g0", "g1", "g2", "a", "b", "c"] {
        assert!(text.contains(&want), "{want} in {text:?}");
    }
    let pdf = legume_tui::export::pdf_bytes(&[page], 300.).unwrap();
    assert!(pdf.starts_with(b"%PDF"));
}

/// Many genes, many groups and long group names: every gene and group is
/// named, and no text runs off the page.
#[test]
fn a_figure_is_as_large_as_its_genes_and_groups() {
    let (genes, k) = (120, 40);
    let names: Vec<Box<str>> = (0..k)
        .map(|c| format!("group-with-a-long-name-{c}").into())
        .collect();
    let comm = Communities::from_groups("t", (0..k as u16).collect(), names);
    let flat: Vec<f32> = (0..genes)
        .flat_map(|g| (0..k).map(move |c| if c == g % k { 9. + (g / k) as f32 } else { 1. }))
        .collect();
    let fr = FeatureRates {
        names: (0..genes).map(|i| format!("g{i}").into()).collect(),
        rates: Mat::from_row_slice(genes, k, &flat),
    };
    let h = Heatmap::build(&fr, &comm, 3, Box::new(|_| None));
    assert_eq!(h.genes.len(), genes);
    let palette = Theme::Light.palette(k);
    let page = h
        .figure(&comm, &palette, "heatmap", Theme::Light)
        .expect("drawn");
    let text: Vec<&str> = page.text.iter().map(|t| t.text.as_str()).collect();
    for g in 0..genes {
        assert!(text.contains(&format!("g{g}").as_str()), "g{g} named");
    }
    let named = text
        .iter()
        .filter(|t| t.starts_with("group-with-a-long"))
        .count();
    assert_eq!(named, k);
    on_page(&page);
}

/// With no genes to show there is no figure to write.
#[test]
fn an_empty_heatmap_has_no_figure() {
    let fr = rates(&[[3., 3., 1.]]);
    let comm = groups();
    let h = Heatmap::build(&fr, &comm, 1, Box::new(|_| None));
    assert!(h.genes.is_empty());
    let palette = Theme::Light.palette(3);
    assert!(h.figure(&comm, &palette, "heatmap", Theme::Light).is_err());
}

/// A few genes under long group names: the names stay on the page.
#[test]
fn long_group_names_fit_a_short_heatmap() {
    let names: Vec<Box<str>> = (0..3)
        .map(|c| format!("a-group-name-{c}-long").into())
        .collect();
    let comm = Communities::from_groups("t", vec![0, 1, 2], names);
    let fr = rates(&[[5., 1., 1.], [1., 1., 7.], [1., 6., 1.]]);
    let h = Heatmap::build(&fr, &comm, 1, Box::new(|_| None));
    let palette = Theme::Light.palette(3);
    let page = h
        .figure(&comm, &palette, "heatmap", Theme::Light)
        .expect("drawn");
    on_page(&page);
}

/// A heatmap too large for a page (every group's top hundred genes) has no
/// figure, rather than one too large to draw or open.
#[test]
fn a_heatmap_too_large_for_a_page_has_no_figure() {
    let (genes, k) = (4000, 40);
    let names: Vec<Box<str>> = (0..k).map(|c| format!("t{c}").into()).collect();
    let comm = Communities::from_groups("t", (0..k as u16).collect(), names);
    let flat: Vec<f32> = (0..genes)
        .flat_map(|g| (0..k).map(move |c| if c == g % k { 9. + g as f32 } else { 1. }))
        .collect();
    let fr = FeatureRates {
        names: (0..genes).map(|i| format!("g{i}").into()).collect(),
        rates: Mat::from_row_slice(genes, k, &flat),
    };
    let h = Heatmap::build(&fr, &comm, 100, Box::new(|_| None));
    assert_eq!(h.genes.len(), genes);
    let palette = Theme::Light.palette(k);
    assert!(h.figure(&comm, &palette, "heatmap", Theme::Light).is_err());
}

/// With one group in view, a gene at level 0 there says nothing: it is
/// left out even when fewer genes than asked for are above it.
#[test]
fn one_group_leaves_out_genes_at_zero() {
    let comm = Communities::from_groups("t", vec![0, 0], vec!["a".into()]);
    let fr = FeatureRates {
        names: (0..3).map(|i| format!("g{i}").into()).collect(),
        rates: Mat::from_row_slice(3, 1, &[2., 0., 1.]),
    };
    let h = Heatmap::build(&fr, &comm, 3, Box::new(|_| None));
    let genes: Vec<&str> = h.genes.iter().map(|(n, _, _)| n.as_ref()).collect();
    assert_eq!(genes, ["g0", "g2"]);
}

/// The same holds for observed counts: a candidate never seen in the one
/// group shown is left out, whatever the model expected of it.
#[test]
fn one_group_leaves_out_genes_never_observed() {
    let comm = Communities::from_groups("t", vec![0, 0], vec!["a".into()]);
    let fr = FeatureRates {
        names: (0..2).map(|i| format!("g{i}").into()).collect(),
        rates: Mat::from_row_slice(2, 1, &[2., 1.]),
    };
    let observe = Box::new(|names: &[&str]| {
        Some(
            names
                .iter()
                .map(|&n| Some(vec![if n == "g0" { 1. } else { 0. }]))
                .collect(),
        )
    });
    let h = Heatmap::build(&fr, &comm, 2, observe);
    assert!(h.observed);
    let genes: Vec<&str> = h.genes.iter().map(|(n, _, _)| n.as_ref()).collect();
    assert_eq!(genes, ["g0"]);
}
