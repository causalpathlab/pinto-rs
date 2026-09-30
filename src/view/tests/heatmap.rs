use crate::util::common::*;
use crate::view::data::Communities;
use crate::view::heatmap::{zscores, Heatmap, CLIP};
use crate::view::markers::FeatureRates;

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

#[test]
fn rows_are_z_scored_clipped_and_flat_rows_are_zero() {
    let z = zscores(&[1., 2., 3., 4., 4., 4., 0., f32::NAN, 2.], 3);
    let s = (2f32 / 3.).sqrt();
    assert!((z[0] + 1. / s).abs() < 1e-5 && z[1].abs() < 1e-6);
    assert_eq!(&z[3..6], &[0., 0., 0.]);
    assert!(z[7].is_nan() && (z[6] + 1.).abs() < 1e-6);
    let wide = zscores(
        &[0.; 99].iter().copied().chain([1000.]).collect::<Vec<_>>(),
        100,
    );
    assert_eq!(wide[99], CLIP);
}
