//! The viewer's pair map ([`pair_map`]) holds what the analysis computed:
//! the same receivers, contacts and size factors as the tissue, each
//! receiver's exposure and exposed flag as the estimate's, competence at the
//! middle of the grid, and the counts and threshold the table's row reports.

use super::estimate_pair;
use crate::lr_activity::ligand_effect::dose::Exposure;
use crate::lr_activity::ligand_effect::estimate::PairData;
use crate::lr_activity::ligand_effect::exposure::competence_grid;
use crate::lr_activity::ligand_effect::options::{EffectOpts, COMPETENCE_GRID};
use crate::lr_activity::ligand_effect::receivers::{
    pair_map, receiver_contacts, size_factors, Class,
};
use crate::lr_activity::ligand_effect::rows::ALL_SECTIONS;
use crate::lr_activity::ligand_effect::tissue::{Tissue, TissueInputs, TissueOpts};
use crate::util::common::Mat;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const SIDE: usize = 24;

struct Small {
    tissue: Tissue,
    coords: Vec<Option<(f32, f32)>>,
    pairs: Vec<(usize, usize)>,
    log1p_depth: Vec<f32>,
    /// Per receiver row.
    ligand: Vec<f64>,
    receptor: Vec<f64>,
    response: Vec<(u32, f32)>,
}

/// A grid of cells touching their four neighbours, some without
/// coordinates and one without contacts (neither is a receiver), depths
/// varying, four identity bins detecting the receptor at different rates,
/// the ligand in a patch.
fn small(seed: u64) -> Small {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = SIDE * SIDE + 1;
    let coords: Vec<Option<(f32, f32)>> = (0..n)
        .map(|i| match i {
            _ if i == n - 1 => Some((1e4, 1e4)),
            _ if i % 37 == 5 => None,
            _ => Some(((i % SIDE) as f32 * 10.0, (i / SIDE) as f32 * 10.0)),
        })
        .collect();
    let mut pairs = Vec::new();
    for i in 0..SIDE * SIDE {
        if i % SIDE + 1 < SIDE {
            pairs.push((i, i + 1));
        }
        if i / SIDE + 1 < SIDE {
            pairs.push((i, i + SIDE));
        }
    }
    let section: Vec<Box<str>> = vec!["s".into(); n];
    let identity_bin: Vec<usize> = (0..n).map(|i| (i * 5 + i / SIDE) % 4).collect();
    let identity = Mat::from_fn(4, n, |r, c| ((r * 31 + c * 17) % 13) as f32 / 13.0);
    let log1p_depth: Vec<f32> = (0..n)
        .map(|_| (200.0 + 800.0 * rng.random::<f32>()).ln_1p())
        .collect();
    let tissue = Tissue::build(
        &TissueInputs {
            coords: &coords,
            section: &section,
            pairs: &pairs,
            identity: &identity,
            identity_bin: &identity_bin,
            log1p_depth: &log1p_depth,
            cell_type: None,
        },
        &TissueOpts {
            folds: 3,
            tile: 6.0,
            ..TissueOpts::default()
        },
    )
    .unwrap();
    let rows = tissue.n();
    let ligand: Vec<f64> = (0..rows)
        .map(|r| {
            let (x, y) = tissue.coords[r];
            let near = x < 120.0 && y < 160.0;
            if near && rng.random::<f64>() < 0.6 {
                1.0 + (rng.random::<f64>() * 4.0).floor()
            } else {
                0.0
            }
        })
        .collect();
    let rate = [0.9, 0.12, 0.07, 0.0];
    let receptor: Vec<f64> = (0..rows)
        .map(|r| (rng.random::<f64>() < rate[tissue.identity_bin[r]]) as u8 as f64)
        .collect();
    let response: Vec<(u32, f32)> = (0..rows as u32)
        .filter_map(|r| {
            let x = (rng.random::<f64>() * 4.0).floor() as f32;
            (x > 0.0).then_some((r, x))
        })
        .collect();
    Small {
        tissue,
        coords,
        pairs,
        log1p_depth,
        ligand,
        receptor,
        response,
    }
}

#[test]
fn the_receivers_contacts_and_size_factors_are_the_tissues() {
    let w = small(1);
    let (cell, row_of, contacts) = receiver_contacts(&w.coords, &w.pairs);
    assert_eq!(cell, w.tissue.cell);
    assert_eq!(contacts.adj, w.tissue.contacts.adj);
    // A cell without coordinates or contacts has no row.
    assert!(cell.len() < w.coords.len());
    for (c, &r) in row_of.iter().enumerate() {
        assert_eq!(r != usize::MAX, cell.contains(&c));
    }
    let log_depth: Vec<f64> = cell.iter().map(|&c| w.log1p_depth[c] as f64).collect();
    assert_eq!(size_factors(&log_depth), w.tissue.size_factor);
}

#[test]
fn a_pair_on_the_map_is_what_the_estimate_saw() {
    let w = small(2);
    let t = &w.tissue;
    let opts = EffectOpts {
        competences: &COMPETENCE_GRID,
        curve: false,
        ..EffectOpts::default()
    };
    let map = pair_map(
        &t.contacts,
        &t.size_factor,
        &t.identity_bin,
        &w.ligand,
        &w.receptor,
        None,
    );
    let ex = Exposure::build(t, &w.ligand, &opts);
    assert_eq!(map.exposure, ex.dose);
    assert_eq!(map.threshold, ex.tau);
    assert_eq!(map.exposed, ex.t);
    let detected: Vec<bool> = w.receptor.iter().map(|&r| r > 0.0).collect();
    let grid = competence_grid(&t.identity_bin, &detected, &COMPETENCE_GRID);
    assert_eq!(map.competence, COMPETENCE_GRID[1]);
    assert_eq!(map.competent, grid[1]);
    let sender: Vec<bool> = w.ligand.iter().map(|&l| l > 0.0).collect();
    assert_eq!(map.sender, sender);
    assert!(map.exposed.iter().any(|&e| e) && map.exposed.iter().any(|&e| !e));
    assert!(map.competent.iter().any(|&c| c) && map.competent.iter().any(|&c| !c));

    // The table's row reports the same threshold and counts.
    let targets: Vec<(f64, &[(u32, f32)])> = vec![(1.0, &w.response[..])];
    let rows = estimate_pair(
        t,
        &PairData {
            ligand: "L",
            receptor: "R",
            ligand_counts: &w.ligand,
            receptor_counts: &w.receptor,
            targets: &targets,
            reweighted: None,
            n_removed: 0,
        },
        &opts,
    )
    .unwrap();
    let pooled = rows.iter().find(|r| &*r.section == ALL_SECTIONS).unwrap();
    assert_eq!(pooled.threshold, map.threshold);
    assert_eq!(pooled.n_exposed, map.exposed.iter().filter(|&&e| e).count());
    assert_eq!(
        pooled.n_competent,
        map.competent.iter().filter(|&&c| c).count()
    );

    // Given the table's threshold, the map takes it.
    let at = pair_map(
        &t.contacts,
        &t.size_factor,
        &t.identity_bin,
        &w.ligand,
        &w.receptor,
        Some(pooled.threshold),
    );
    assert_eq!(at, map);
    let higher = pair_map(
        &t.contacts,
        &t.size_factor,
        &t.identity_bin,
        &w.ligand,
        &w.receptor,
        Some(2.5),
    );
    let want: Vec<bool> = ex.dose.iter().map(|&x| x > 2.5).collect();
    assert_eq!(higher.exposed, want);
    // A threshold that is not a number is chosen as the estimate would.
    let nan = pair_map(
        &t.contacts,
        &t.size_factor,
        &t.identity_bin,
        &w.ligand,
        &w.receptor,
        Some(f64::NAN),
    );
    assert_eq!(nan.threshold, ex.tau);
}

#[test]
fn every_receiver_has_one_class_senders_first() {
    let w = small(3);
    let t = &w.tissue;
    let map = pair_map(
        &t.contacts,
        &t.size_factor,
        &t.identity_bin,
        &w.ligand,
        &w.receptor,
        None,
    );
    let counts = map.counts();
    assert_eq!(counts.iter().sum::<usize>(), t.n());
    assert_eq!(counts[0], map.sender.iter().filter(|&&s| s).count());
    for i in 0..map.n() {
        let c = map.class(i);
        match c {
            Class::Sender => assert!(map.sender[i]),
            Class::ExposedCompetent => assert!(map.exposed[i] && map.competent[i]),
            Class::ExposedIncompetent => assert!(map.exposed[i] && !map.competent[i]),
            Class::UnexposedCompetent => assert!(!map.exposed[i] && map.competent[i]),
            Class::UnexposedIncompetent => assert!(!map.exposed[i] && !map.competent[i]),
        }
    }
}

/// A ligand on nearly every receiver: the estimate splits at the median
/// exposure, and receivers tied at it are unexposed. The table keeps τ as
/// an `f32`; read back below the median, it must still leave them so.
#[test]
fn a_median_threshold_read_back_from_the_table_keeps_the_ties_unexposed() {
    use crate::lr_activity::ligand_effect::receivers::table_threshold;
    let mut found = false;
    for seed in 0..40 {
        let w = small(seed);
        let t = &w.tissue;
        let mut rng = SmallRng::seed_from_u64(seed);
        let ligand: Vec<f64> = (0..t.n())
            .map(|_| 1.0 + (rng.random::<f64>() * 3.0).floor())
            .collect();
        let map = pair_map(
            &t.contacts,
            &t.size_factor,
            &t.identity_bin,
            &ligand,
            &w.receptor,
            None,
        );
        let tau = map.threshold;
        let stored = tau as f32 as f64;
        // Only a τ the f32 rounds down shows the flip.
        if tau <= 0.0 || stored >= tau {
            continue;
        }
        found = true;
        let tied = map.exposure.iter().filter(|&&x| x == tau).count();
        assert!(tied > 0, "τ is the median exposure");
        assert_eq!(table_threshold(&map.exposure, Some(stored)), tau);
        let read = pair_map(
            &t.contacts,
            &t.size_factor,
            &t.identity_bin,
            &ligand,
            &w.receptor,
            Some(stored),
        );
        assert_eq!(read, map, "seed {seed}");
        // A threshold of another table, between exposures, is taken as it is.
        let other = 0.5 * (tau + map.exposure.iter().copied().fold(0.0, f64::max));
        assert_eq!(table_threshold(&map.exposure, Some(other)), other);
    }
    assert!(found, "a seed whose median τ the f32 rounds down");
}
