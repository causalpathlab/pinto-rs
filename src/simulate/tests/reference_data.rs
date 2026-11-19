//! Reference data: states take the reference's groups largest first,
//! each state's rates are its group's, a spot's expected count is the
//! mixture of its states' rates (exactly, with no nuisance), the drawn
//! counts follow those means, the nuisance is a share of the reference's
//! variance between states, and the parameters it replaces are refused.

use super::*;
use crate::simulate::panel::grid;
use crate::simulate::params::SimParams;
use crate::simulate::sim::Sim;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

/// Groups of the fixture: sizes, largest first by index, and the genes each
/// is high in.
const SIZES: [usize; 6] = [140, 120, 110, 100, 90, 80];
const GENES_PER_GROUP: usize = 6;
const N_GENES: usize = SIZES.len() * GENES_PER_GROUP + 4;

/// A reference of `SIZES` cells in six labelled groups, group `k` high
/// (mean 8) in genes `6k..6k+6` and low (0.5) elsewhere, with its labels
/// file; their paths.
fn fixture(dir: &std::path::Path) -> (String, String) {
    let names: Vec<String> = (0..N_GENES).map(|g| format!("G{g}")).collect();
    let rows: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut r = SmallRng::seed_from_u64(7);
    let (mut triplets, mut labels, mut col) = (Vec::new(), String::new(), 0u64);
    for (k, &n) in SIZES.iter().enumerate() {
        for _ in 0..n {
            for g in 0..N_GENES {
                let high = g / GENES_PER_GROUP == k;
                let y: f64 = Poisson::new(if high { 8.0 } else { 0.5 })
                    .unwrap()
                    .sample(&mut r);
                if y > 0.0 {
                    triplets.push((g as u64, col, y as f32));
                }
            }
            labels.push_str(&format!("T{k}\n"));
            col += 1;
        }
    }
    let n_cells = SIZES.iter().sum();
    let data = crate::test_support::write_zarr(dir, "ref", &rows, n_cells, &triplets);
    let lab = dir.join("ref.labels.txt");
    std::fs::write(&lab, labels).unwrap();
    (data.to_string(), lab.to_str().unwrap().to_string())
}

fn params() -> SimParams {
    SimParams {
        grid_side: 16,
        n_domains: 4,
        program_genes: 3,
        ..SimParams::default()
    }
}

fn simulation(dir: &std::path::Path, p: SimParams) -> Sim {
    let (data, labels) = fixture(dir);
    let g = grid(p.grid_side, p.grid_spacing_um);
    Sim::from_reference_data(p, &[g], &data, Some(&labels)).unwrap()
}

#[test]
fn the_states_take_the_largest_groups_and_their_rates() {
    let dir = tempfile::tempdir().unwrap();
    let sim = simulation(dir.path(), params());
    let r = sim.reference_data.as_ref().unwrap();
    let n_states = sim.truth.n_states();
    let names: Vec<&str> = r
        .cell_group_of_state
        .iter()
        .map(|&k| r.cell_groups[k].0.as_str())
        .collect();
    let want: Vec<String> = (0..n_states).map(|k| format!("T{k}")).collect();
    assert_eq!(names, want, "largest first");
    for (c, &k) in r.cell_group_of_state.iter().enumerate() {
        // the group's own genes are high, the others low
        let own = k * GENES_PER_GROUP;
        let name = &r.cell_groups[k].0;
        let idx: usize = name[1..].parse().unwrap();
        assert_eq!(idx * GENES_PER_GROUP, own);
        assert!(
            (r.mean_count_by_state[c][own] - 8.0).abs() < 1.0,
            "{}",
            r.mean_count_by_state[c][own]
        );
        assert!(r.mean_count_by_state[c][N_GENES - 1] < 1.0);
    }
    assert_eq!(sim.genes.names.len(), N_GENES, "the reference's genes");
    // each state's markers are its group's own genes
    for (c, &k) in r.cell_group_of_state.iter().enumerate() {
        let markers: Vec<usize> = (0..N_GENES)
            .filter(|&g| sim.genes.program[c][g] > 0.0)
            .collect();
        assert_eq!(markers.len(), 3);
        assert!(
            markers.iter().all(|&g| g / GENES_PER_GROUP == k),
            "{markers:?}"
        );
    }
}

#[test]
fn a_spots_mean_is_its_mixture_of_rates_with_no_nuisance() {
    let dir = tempfile::tempdir().unwrap();
    let sim = simulation(
        dir.path(),
        SimParams {
            nuisance_off: true,
            ..params()
        },
    );
    let r = sim.reference_data.as_ref().unwrap();
    let on = crate::simulate::expression::active(&sim.p);
    for g in [0, 7, N_GENES - 1] {
        let mu = sim.mean(0, g, on);
        for (i, w) in sim.truth.mixture[0].iter().enumerate() {
            let want: f64 = w
                .iter()
                .zip(&r.mean_count_by_state)
                .map(|(w, m)| w * m[g])
                .sum();
            assert!(
                (mu[i] - want).abs() <= 1e-9 * want.max(1.0),
                "{} {want}",
                mu[i]
            );
        }
    }
}

#[test]
fn the_drawn_counts_follow_their_means() {
    let dir = tempfile::tempdir().unwrap();
    let sim = simulation(dir.path(), params());
    let on = crate::simulate::expression::active(&sim.p);
    let counts = sim.counts(0);
    for g in [0usize, 6, 12] {
        let drawn: f64 = counts
            .iter()
            .filter(|t| t.0 == g as u64)
            .map(|t| f64::from(t.2))
            .sum();
        let mu = sim.mean(0, g, on);
        let want: f64 = mu.iter().sum();
        // a sum of counts each of variance at most a few times its mean
        let sd = (10.0 * want).sqrt();
        assert!(
            (drawn - want).abs() < 5.0 * sd,
            "gene {g}: {drawn} vs {want}"
        );
    }
}

#[test]
fn the_nuisance_is_a_share_of_the_reference_variance_between_states() {
    let dir = tempfile::tempdir().unwrap();
    let p = params();
    let sim = simulation(dir.path(), p.clone());
    let r = sim.reference_data.as_ref().unwrap();
    let share = crate::simulate::sim::spot_share_by_state(&sim.truth);
    let genes = pve_genes(&r.floored_mean_count_by_state, &share, &r.detected);
    assert_eq!(
        genes, r.pve_genes,
        "the genes the fit scaled the nuisance on"
    );
    let t =
        cell_type_log_count_variance(&r.floored_mean_count_by_state, &sim.truth.mixture, &genes);
    assert!(t > 0.0);
    assert!((r.log_count_variance - t / p.pve_type).abs() < 1e-12);
    assert_eq!(sim.p.log_rate_var, r.log_count_variance);
    // with no nuisance the type term is the whole of the realized variance
    let flat = simulation(
        dir.path(),
        SimParams {
            nuisance_off: true,
            ..params()
        },
    );
    let pve = flat.realized_pve_expressed_genes().unwrap();
    assert!((pve[0] - 1.0).abs() < 1e-9, "{pve:?}");
}

/// `T` by hand: two spots, one of each of two states, one gene twice as
/// high in the second, and a flat gene, weighted by their mean rates.
#[test]
fn the_cell_type_variance_is_the_count_weighted_log_mean_variance_over_spots() {
    let mean_count = [vec![1.0, 2.0], vec![1.0, 4.0]];
    let mixture = vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]];
    let genes = pve_genes(&mean_count, &[0.5, 0.5], &[true, true]);
    assert_eq!(
        genes,
        vec![(1, 3.0), (0, 1.0)],
        "by mean count, largest first"
    );
    let t = cell_type_log_count_variance(&mean_count, &mixture, &genes);
    // gene 1: logs ln 2 and ln 4, population variance (ln 2)² / 4, weight 3 of 4
    let want = 3.0 * std::f64::consts::LN_2.powi(2) / 4.0 / 4.0;
    assert!((t - want).abs() < 1e-12, "{t} vs {want}");
}

#[test]
fn the_parameters_a_reference_replaces_are_refused() {
    use crate::simulate::run::{resolve_params, SimulateArgs};
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        a: SimulateArgs,
    }
    let parse = |extra: &[&str]| {
        let mut argv = vec!["x", "--reference-data", "r.zarr", "-o", "o"];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(argv).unwrap().a
    };
    let e = resolve_params(&parse(&["--set", "n_genes=50"]))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("n_genes") && e.contains("--reference-data"),
        "{e}"
    );
    assert!(resolve_params(&parse(&["--set", "pve_donor=0.2"])).is_ok());
}

/// A cell group that never shows a gene puts it at the group's detection
/// limit, not at zero: the between-state variance stays that of counts.
#[test]
fn a_gene_a_group_never_shows_sits_at_its_detection_limit() {
    let floor = detection_limit(0.25, 400);
    assert!((floor - 0.005).abs() < 1e-15, "{floor}");
    // one state shows the gene at 2, the other never: the log means are
    // ln 2 and ln 0.005, not ln 2 and −690
    let floored_mean = vec![vec![2.0], vec![floor]];
    let mixture = vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]];
    let genes = pve_genes(&floored_mean, &[0.5, 0.5], &[true]);
    let t = cell_type_log_count_variance(&floored_mean, &mixture, &genes);
    let want = ((2.0f64.ln() - floor.ln()) / 2.0).powi(2);
    assert!((t - want).abs() < 1e-12, "{t} vs {want}");
}

#[test]
fn k_means_asks_as_many_more_groups_as_were_missing() {
    use crate::simulate::reference_data::more_cell_groups;
    // 17 asked kept 16: ask one more
    assert_eq!(more_cell_groups(17, 16, 17, 30_000), Some(18));
    assert_eq!(more_cell_groups(18, 15, 17, 30_000), Some(20));
    // enough kept
    assert_eq!(more_cell_groups(17, 17, 17, 30_000), None);
    assert_eq!(more_cell_groups(8, 8, 5, 30_000), None);
    // no more than every group could fill with 100 cells
    assert_eq!(more_cell_groups(17, 16, 17, 1_800), Some(18));
    assert_eq!(more_cell_groups(17, 16, 17, 1_799), None);
}
