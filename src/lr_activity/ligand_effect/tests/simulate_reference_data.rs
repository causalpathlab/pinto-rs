//! The expression reference `lra simulate --reference-data` draws program
//! genes with: bins take the best-correlated group, genes are found by
//! name, and a draw keeps the tissue's mean.

use super::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

#[test]
fn a_profile_takes_its_best_correlated_candidate_and_the_first_on_ties() {
    let candidates = vec![
        vec![1.0, 2.0, 3.0],
        vec![3.0, 2.0, 1.0],
        vec![1.0, 2.0, 3.0],
    ];
    assert_eq!(most_correlated_profile(&[0.0, 1.0, 5.0], &candidates), 0);
    assert_eq!(most_correlated_profile(&[9.0, 4.0, 0.0], &candidates), 1);
    // nothing varies: the first
    assert_eq!(most_correlated_profile(&[2.0, 2.0, 2.0], &candidates), 0);
}

/// Three groups of 120 cells, group `k` high (mean 6) in genes `4k..4k+4`
/// and low (0.4) in the rest; the data file's path.
fn fixture(dir: &std::path::Path) -> String {
    let names: Vec<String> = (0..12).map(|g| format!("G{g}")).collect();
    let rows: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut r = SmallRng::seed_from_u64(3);
    let mut triplets = Vec::new();
    for c in 0..360u64 {
        let k = (c / 120) as usize;
        for g in 0..12 {
            let mean = if g / 4 == k { 6.0 } else { 0.4 };
            let y: f64 = Poisson::new(mean).unwrap().sample(&mut r);
            if y > 0.0 {
                triplets.push((g as u64, c, y as f32));
            }
        }
    }
    crate::test_support::write_zarr(dir, "ref", &rows, 360, &triplets).to_string()
}

#[test]
fn each_bin_takes_the_group_its_program_genes_resemble() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    let labels = dir.path().join("labels.txt");
    let text: String = (0..360).map(|c| format!("T{}\n", c / 120)).collect();
    std::fs::write(&labels, text).unwrap();
    // the tissue's program genes: G1, G5, G9 and one the reference lacks
    let genes: Vec<Box<str>> = ["G1", "G5", "G9", "NOPE"].map(Into::into).to_vec();
    // bin 0 is high in G9 (the third group's), bin 1 in G1 (the first's)
    let bin_mean_counts = vec![0.3, 0.2, 5.0, 1.0, 5.5, 0.3, 0.1, 1.0];
    let r =
        ReferenceProgramGenes::fit(&path, labels.to_str(), &genes, &bin_mean_counts, 2).unwrap();
    let name = |b: usize| r.cell_groups[r.cell_group_of_bin[b]].0.clone();
    assert_eq!((name(0).as_str(), name(1).as_str()), ("T2", "T0"));
    assert_eq!(r.gene_position("NOPE"), None);
    assert!(r.gene_position("G5").is_some());
    let provenance = r.provenance();
    assert_eq!(provenance["program_genes_drawn"], 3);
    assert_eq!(provenance["n_cells"], 360);
}

#[test]
fn a_draw_keeps_the_mean_it_is_scaled_to() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(dir.path());
    let genes: Vec<Box<str>> = ["G1", "G5"].map(Into::into).to_vec();
    let r = ReferenceProgramGenes::fit(&path, None, &genes, &[1.0, 1.0], 1).unwrap();
    let j = r.gene_position("G1").unwrap();
    // the mean of the quantile function over a fine grid of u
    let n = 20_000;
    for target in [0.5, 3.0, 12.0] {
        let mean: f64 = (0..n)
            .map(|k| {
                let u = (k as f64 + 0.5) / n as f64;
                r.count_at_quantile(0, j, u, target, |_, m| m)
            })
            .sum::<f64>()
            / n as f64;
        assert!(
            (mean - target).abs() < 0.05 * target + 0.02,
            "{mean} vs {target}"
        );
    }
}
