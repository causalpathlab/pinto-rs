//! `--null-permutations K`: every pair estimated again under `K` permutation
//! draws of its ligand ([`super::super::null_draws`]), each ligand's pairs on
//! one worker sharing a draw's fits, as the real run shares its own.

use super::super::fits::LigandShare;
use super::super::null_draws::{draw_seed, null_rows, permuted_within_strata, NullRow};
use super::{estimate_one, ligand_groups, ligands_at_once, run_bounded, Prepared, Progress};

/// Each pair's null rows over `draws` draws under the run's `seed`, in pair
/// order.
pub(super) fn estimate_null_draws(prep: &Prepared, draws: u32, seed: u64) -> Vec<Vec<NullRow>> {
    let groups = ligand_groups(&prep.pairs);
    let pb = Progress::new(prep.pairs.len() * draws as usize, "null draws of pairs");
    let concurrent = ligands_at_once(prep, groups.len());
    let mut by_pair = run_bounded(&groups, concurrent, |group: &Vec<usize>| {
        let first = &prep.pairs[group[0]];
        let real = prep.ligand_counts(first);
        let mut out: Vec<(usize, Vec<NullRow>)> = group.iter().map(|&k| (k, Vec::new())).collect();
        for draw in 0..draws {
            let seed_of_draw = draw_seed(seed, &first.ligand, draw);
            let lig = permuted_within_strata(&prep.tissue, &real, seed_of_draw);
            let share = LigandShare::default();
            for (k, rows) in out.iter_mut() {
                let effect = estimate_one(prep, &prep.pairs[*k], &lig, &share);
                rows.extend(null_rows(&effect, draw));
                pb.inc();
            }
        }
        out
    });
    by_pair.sort_by_key(|(k, _)| *k);
    pb.finish();
    by_pair.into_iter().map(|(_, rows)| rows).collect()
}
