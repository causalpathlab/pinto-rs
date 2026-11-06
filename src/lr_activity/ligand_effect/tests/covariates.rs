//! The pair columns aimed at segmentation bleed and at a fine niche, the
//! ring bounds, the t reference after Rubin's rules, and the simulator's
//! truth on the measured scale under spillover.

use super::run_sim;
use super::simulate::{base, fixture, opts};
use crate::lr_activity::ligand_effect::averaging::combine;
use crate::lr_activity::ligand_effect::exposure::Contacts;
use crate::lr_activity::ligand_effect::inference::Estimate;
use crate::lr_activity::ligand_effect::response::{niche_ligand, programme_columns};
use crate::lr_activity::ligand_effect::simulate::Scenario;

// * Bleed

/// The bleed column of one weighted programme.
fn neighbour_bleed(
    targets: &[(f64, &[(u32, f32)])],
    bins: &[usize],
    contacts: &Contacts,
    sf: &[f64],
) -> Vec<f64> {
    let genes: Vec<&[(u32, f32)]> = targets.iter().map(|(_, nz)| *nz).collect();
    let w: Vec<f64> = targets.iter().map(|(w, _)| *w).collect();
    programme_columns(&genes, &[w], bins, contacts, sf)
        .1
        .remove(0)
}

/// A path 0 – 1 – 2 and a lone cell 3, bins `[0, 0, 1, 1]`. Bleed reads
/// each neighbour's expected counts (its size factor times its bin's mean
/// count / size factor), sums them over contacts and divides by the
/// receiver's size factor; a cell without contacts gets 0.
#[test]
fn bleed_sums_expected_neighbour_counts_on_the_receivers_scale() {
    let contacts = Contacts::from_pairs(4, &[(0, 1), (1, 2)]);
    let bins = [0, 0, 1, 1];
    let sf = [1.0, 2.0, 1.0, 1.0];
    // Bin means: g1 1.5 and 0.5, g2 1.0 and 0.
    let g1: Vec<(u32, f32)> = vec![(0, 3.0), (2, 1.0)];
    let g2: Vec<(u32, f32)> = vec![(1, 4.0)];
    let b = neighbour_bleed(&[(1.0, &g1), (-1.0, &g2)], &bins, &contacts, &sf);
    let ln = f64::ln;
    // Cell 1: g1 (1·1.5 + 1·0.5) / 2, g2 (1·1.0) / 2.
    assert!((b[1] - (ln(2.0) - ln(1.5)) / 2.0).abs() < 1e-12, "{}", b[1]);
    // Cell 0: g1 2·1.5 / 1, g2 2·1.0 / 1.
    assert!((b[0] - (ln(4.0) - ln(3.0)) / 2.0).abs() < 1e-12, "{}", b[0]);
    assert_eq!(b[3], 0.0);
    // A response a neighbour makes on its own does not move it beyond its
    // bin's mean: the same bin means give the same bleed.
    let g1b: Vec<(u32, f32)> = vec![(1, 6.0), (3, 1.0)];
    let b2 = neighbour_bleed(&[(1.0, &g1b), (-1.0, &g2)], &bins, &contacts, &sf);
    assert!((b2[1] - b[1]).abs() < 1e-12);
}

// * Niche

/// On the four-neighbour grid an interior cell's second-order neighbours
/// are the eight cells two steps away (four straight, four diagonal),
/// never itself or a contact.
#[test]
fn second_neighbours_are_two_steps_away() {
    let f = fixture();
    let t = &f.tissue;
    let side = 60usize;
    let i = 30 * side + 30;
    let s = &t.second[i];
    assert_eq!(s.len(), 8, "{s:?}");
    for &k in s {
        let k = k as usize;
        let (dx, dy) = ((k % side).abs_diff(i % side), (k / side).abs_diff(i / side));
        assert_eq!(dx + dy, 2, "cell {k} is not two steps from {i}");
    }
    // The niche column is log1p of their mean.
    let v: Vec<f64> = (0..t.n()).map(|j| (j % 7) as f64).collect();
    let mean = s.iter().map(|&k| v[k as usize]).sum::<f64>() / 8.0;
    assert!((niche_ligand(t, &v)[i] - mean.ln_1p()).abs() < 1e-12);
}

// * Degrees of freedom

fn est(theta: f64, se: f64, df: f64) -> Estimate {
    Estimate {
        theta,
        se,
        n: 100,
        n_treated: 40,
        mean_h: 0.2,
        ess: 80.0,
        df,
    }
}

/// One draw keeps its df; identical draws lose a little (Barnard–Rubin's
/// observed-data df); draws that disagree lose more, and never exceed the
/// complete-data df.
#[test]
fn rubin_degrees_of_freedom() {
    assert_eq!(combine(&[est(0.1, 0.05, 40.0)]).df, 40.0);
    let same = combine(&[est(0.1, 0.05, 40.0), est(0.1, 0.05, 50.0)]);
    assert!(same.df < 40.0 && same.df > 35.0, "{}", same.df);
    let apart = combine(&[est(0.0, 0.05, 40.0), est(0.2, 0.05, 40.0)]);
    assert!(apart.df < same.df, "{} vs {}", apart.df, same.df);
    assert!(apart.crit() > same.crit());
}

// * Measured-scale truth

/// Bleed adds the neighbours' counts inside `log1p`, which compresses a
/// contrast: the truth under spillover is the measured one, smaller than
/// without bleed on the same draws, and still exactly 0 with nothing
/// planted.
#[test]
fn spillover_truth_is_on_the_measured_scale() {
    let f = fixture();
    let b = base(&f);
    let planted = Scenario {
        effect: 0.5,
        ..Scenario::default()
    };
    let clean = run_sim(&f.tissue, &b, &planted, &opts(), 3).unwrap();
    let bled = run_sim(
        &f.tissue,
        &b,
        &Scenario {
            spillover: 0.3,
            ..planted
        },
        &opts(),
        3,
    )
    .unwrap();
    assert!(clean.truth > 0.0 && bled.truth > 0.0);
    assert!(
        bled.truth < clean.truth,
        "measured truth {:.4} should be below the unbled {:.4}",
        bled.truth,
        clean.truth
    );
    let null = run_sim(
        &f.tissue,
        &b,
        &Scenario {
            spillover: 0.3,
            ..Scenario::default()
        },
        &opts(),
        3,
    )
    .unwrap();
    assert!(null.truth.abs() < 1e-12 && null.truth_weighted.abs() < 1e-12);
}
