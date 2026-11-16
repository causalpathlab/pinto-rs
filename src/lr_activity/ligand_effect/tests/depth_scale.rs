//! The size factor is a cell's total over the common library scale, so a
//! depth-normalized log depends on the cell's own composition and on no
//! other cell: sequencing every cell deeper changes neither it nor what is
//! built from it, and the simulator's rates are counts per that scale.

use super::lattice;
use crate::lr_activity::ligand_effect::dose::Exposure;
use crate::lr_activity::ligand_effect::options::EffectOpts;
use crate::lr_activity::ligand_effect::receivers::{log_normalized, size_factors};
use crate::lr_activity::ligand_effect::response::Response;
use crate::lr_activity::ligand_effect::simulate::{calibrate_producers, target_bin_means};
use crate::lr_activity::ligand_effect::tissue::{Tissue, TissueInputs, TissueOpts};
use crate::util::common::{log_cp10k, Mat, LIBRARY_SCALE};

const SIDE: usize = 24;
const BINS: usize = 4;

/// Cell `i`'s total count before any deepening.
fn total(i: usize) -> f64 {
    300.0 + ((i * 7) % 23) as f64 * 40.0
}

/// A grid of cells touching their four neighbours, four identity bins,
/// each cell's total `deeper` times [`total`].
fn tissue(deeper: f64) -> Tissue {
    let n = SIDE * SIDE;
    let (coords, pairs) = lattice(SIDE);
    let section: Vec<Box<str>> = vec!["s".into(); n];
    let identity_bin: Vec<usize> = (0..n).map(|i| (i * 5 + i / SIDE) % BINS).collect();
    let identity = Mat::from_fn(4, n, |r, c| ((r * 31 + c * 17) % 13) as f32 / 13.0);
    let log1p_depth: Vec<f32> = (0..n).map(|i| (deeper * total(i)).ln_1p() as f32).collect();
    Tissue::build(
        &TissueInputs {
            coords: &coords,
            section: &section,
            pairs: &pairs,
            identity: &identity,
            identity_bin: &identity_bin,
            log1p_depth: &log1p_depth,
            cell_type: None,
            region: None,
        },
        &TissueOpts {
            folds: 3,
            tile: 6.0,
            ..TissueOpts::default()
        },
    )
    .unwrap()
}

/// A ligand in three rows of eleven, counts 1 to 4, `deeper` times over.
fn ligand(n: usize, deeper: f64) -> Vec<f64> {
    (0..n)
        .map(|r| {
            if (r * 13) % 11 < 3 {
                deeper * (1 + r % 4) as f64
            } else {
                0.0
            }
        })
        .collect()
}

/// Two programme genes as non-zero counts by row, `deeper` times over.
fn programme(n: usize, deeper: f64) -> [Vec<(u32, f32)>; 2] {
    let gene = |step: usize, modulus: usize| -> Vec<(u32, f32)> {
        (0..n)
            .filter_map(|r| {
                let x = (r * step) % modulus;
                (x > 0 && x < 5).then_some((r as u32, (deeper * x as f64) as f32))
            })
            .collect()
    };
    [gene(3, 7), gene(5, 9)]
}

fn close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-5 * (1.0 + x.abs()),
            "{what}, row {i}: {x} and {y}"
        );
    }
}

#[test]
fn a_size_factor_is_the_cell_s_total_over_the_common_scale() {
    let scale = f64::from(LIBRARY_SCALE);
    let totals = [1.0, 250.0, scale, 37_500.0];
    let log_depth: Vec<f64> = totals.iter().map(|t: &f64| t.ln_1p()).collect();
    let sf = size_factors(&log_depth);
    for (s, t) in sf.iter().zip(totals) {
        assert!((s - t / scale).abs() <= 1e-12 * t, "total {t}: {s}");
        // A count over it, logged, is the shared normalization of the
        // count in the cell's total.
        for count in [0.0, 1.0, 7.0] {
            let want = log_cp10k(count, t);
            let got = log_normalized(count, *s);
            assert!((got - want).abs() <= 1e-12 * (1.0 + want), "{got} {want}");
        }
    }
    // It is the cell's own: no other cell's depth moves it.
    let among_deeper = size_factors(&[log_depth[1], 9.0, 10.0, 11.0]);
    assert_eq!(among_deeper[0], sf[1]);
    // A cell without counts keeps a size factor a zero count can be
    // divided by.
    assert_eq!(size_factors(&[0.0]), [1.0 / scale]);
}

/// Every cell sequenced eight times deeper, counts and totals alike: each
/// count per library scale is the same, so the response, its bleed, the
/// dose, its threshold and the exposure are. A size factor over the
/// receivers' median depth would pass this too (a total over the median
/// does not move either): what tells the common scale from it is the size
/// factor itself, checked here against the totals and in
/// `a_size_factor_is_the_cell_s_total_over_the_common_scale`.
#[test]
fn sequencing_every_cell_deeper_changes_no_normalized_log_and_no_outcome() {
    let deeper = 8.0;
    let (a, b) = (tissue(1.0), tissue(deeper));
    let n = a.n();
    assert_eq!(n, b.n());
    let scale = f64::from(LIBRARY_SCALE);
    let per_scale: Vec<f64> = a.cell.iter().map(|&c| total(c) / scale).collect();
    close(&a.size_factor, &per_scale, "size factor");
    close(
        &b.size_factor,
        &per_scale.iter().map(|s| s * deeper).collect::<Vec<_>>(),
        "deeper size factor",
    );

    let weights = [1.0, -0.5];
    let response = |t: &Tissue, deeper: f64| {
        let genes = programme(n, deeper);
        let targets: Vec<(f64, &[(u32, f32)])> = weights
            .iter()
            .zip(&genes)
            .map(|(&w, g)| (w, g.as_slice()))
            .collect();
        Response::build(t, &targets)
    };
    let (ra, rb) = (response(&a, 1.0), response(&b, deeper));
    assert!(ra.y.iter().any(|&y| y != 0.0));
    close(&ra.y, &rb.y, "response");
    close(&ra.spill, &rb.spill, "spillover");
    close(&ra.bleed, &rb.bleed, "bleed");

    let opts = EffectOpts::default();
    let xa = Exposure::build(&a, &ligand(n, 1.0), &opts);
    let xb = Exposure::build(&b, &ligand(n, deeper), &opts);
    assert!(xa.t.iter().any(|&t| t) && xa.t.iter().any(|&t| !t));
    close(&xa.dose, &xb.dose, "dose");
    close(&xa.own, &xb.own, "own ligand");
    close(&xa.niche, &xb.niche, "niche ligand");
    close(&[xa.tau], &[xb.tau], "threshold");
    // The same receivers exposed, but for a dose at the threshold, which
    // the depths' f32 rounding may put on either side.
    for (i, (ta, tb)) in xa.t.iter().zip(&xb.t).enumerate() {
        let at_threshold = (xa.dose[i] - xa.tau).abs() <= 1e-5 * (1.0 + xa.tau.abs());
        assert!(ta == tb || at_threshold, "row {i} exposed in one only");
    }
}

/// What the simulator keeps of the real tissue is in counts per library
/// scale: a gene's mean per identity bin, and a producer's rate.
#[test]
fn the_simulator_s_rates_are_counts_per_library_scale() {
    let t = tissue(1.0);
    let n = t.n();
    let scale = f64::from(LIBRARY_SCALE);
    let per_scale = |r: usize, count: f64| count * scale / total(t.cell[r]);
    let mut cells = [0.0; BINS];
    for &b in &t.identity_bin {
        cells[b] += 1.0;
    }

    let [gene, _] = programme(n, 1.0);
    let mut want = [0.0; BINS];
    for &(r, x) in &gene {
        want[t.identity_bin[r as usize]] +=
            per_scale(r as usize, x as f64) / cells[t.identity_bin[r as usize]];
    }
    close(&target_bin_means(&t, &[&gene]), &want, "bin means");

    // A ligand every cell detects: every cell produces, at the bin's mean
    // count per library scale.
    let counts: Vec<f64> = (0..n).map(|r| 2.0 + (r % 3) as f64).collect();
    let mut want = [0.0; BINS];
    for (r, &c) in counts.iter().enumerate() {
        want[t.identity_bin[r]] += per_scale(r, c) / cells[t.identity_bin[r]];
    }
    let cal = calibrate_producers(&t, &counts);
    close(
        &cal.iter().map(|c| c.r).collect::<Vec<_>>(),
        &want,
        "producer rate",
    );
    assert!(cal.iter().all(|c| c.p == 1.0));
}

/// A bin of scattered single counts: mean and detection cannot tell few
/// producers from many weak ones, so the producer rate is the floor, half
/// an expected count at the receivers' median depth. Every cell sequenced
/// deeper by one constant (the counts as they were) moves the median with
/// the rates, so the floor still binds at half a count and the share of
/// producers falls with the depth, as the mean count per depth does.
#[test]
fn the_producer_floor_is_a_count_at_the_receivers_median_depth() {
    let floored = |deeper: f64| {
        let t = tissue(deeper);
        let counts: Vec<f64> = (0..t.n()).map(|r| f64::from(r % 29 == 0)).collect();
        let mut sf = t.size_factor.clone();
        let median = crate::util::common::upper_median(&mut sf).unwrap();
        let cal = calibrate_producers(&t, &counts);
        assert!(cal.iter().all(|c| c.p > 0.0 && c.p < 1.0));
        let at_median: Vec<f64> = cal.iter().map(|c| c.r * median).collect();
        close(&at_median, &[0.5; BINS], "rate at the median depth");
        // Share of producers times their count at the median depth.
        cal.iter().map(|c| c.p * c.r * median).collect::<Vec<f64>>()
    };
    let (a, b) = (floored(1.0), floored(8.0));
    close(&a, &b, "mean count at the median depth");
}
