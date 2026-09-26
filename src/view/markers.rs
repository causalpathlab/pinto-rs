//! Marker features per community, from a level's `feature_community` table.
//!
//! A feature marks community k when its rate there stands above its mean
//! rate over the other communities: fold = μ_gk / mean_{j≠k} μ_gj. The table's
//! rows carry a per-feature Fisher-information weight; it scales every
//! community of a feature alike, so it cancels in the fold.

use crate::util::common::*;
use data_beans::utilities::name_matching::split_id_name;

pub struct FeatureRates {
    pub names: Vec<Box<str>>,
    /// Features × communities.
    pub rates: Mat,
}

pub struct Marker {
    pub name: Box<str>,
    pub fold: f32,
}

/// `ENSG00000116824_CD2` → `CD2`: the symbol half of a composite
/// `{id}_{symbol}` feature name. Only Ensembl ids are dropped, so a name
/// that merely contains the separator (`HLA_DRA`) stays whole.
pub fn symbol(name: &str) -> &str {
    match split_id_name(name) {
        (id, sym) if id.starts_with("ENS") && !sym.is_empty() && id != name => sym,
        _ => name,
    }
}

impl FeatureRates {
    /// The `n` features with the largest fold for community `k`, among those
    /// whose rate in `k` is at least the median rate in `k` (so a fold from
    /// two near-zero rates does not top the list).
    pub fn top(&self, k: usize, n: usize) -> Vec<Marker> {
        let (g, kk) = self.rates.shape();
        if k >= kk || g == 0 {
            return Vec::new();
        }
        let col: Vec<f32> = (0..g).map(|i| self.rates[(i, k)]).collect();
        let mut sorted = col.clone();
        sorted.sort_by(f32::total_cmp);
        let floor = sorted[g / 2];
        let positive: Vec<f32> = sorted.iter().copied().filter(|&v| v > 0.).collect();
        let eps = positive.get(positive.len() / 2).copied().unwrap_or(1.) * 1e-2;

        let mut scored: Vec<Marker> = (0..g)
            .filter(|&i| col[i] >= floor && col[i] > 0.)
            .map(|i| {
                let others = if kk > 1 {
                    (0..kk)
                        .filter(|&j| j != k)
                        .map(|j| self.rates[(i, j)])
                        .sum::<f32>()
                        / (kk - 1) as f32
                } else {
                    0.
                };
                Marker {
                    name: self.names[i].clone(),
                    fold: (col[i] + eps) / (others + eps),
                }
            })
            .collect();
        scored.sort_by(|a, b| b.fold.total_cmp(&a.fold));
        scored.truncate(n);
        scored
    }

    /// `CD2×58.0 CD3E×55.2 …` for the top `n` of community `k`.
    pub fn summary(&self, k: usize, n: usize) -> String {
        self.top(k, n)
            .iter()
            .map(|m| format!("{}×{:.1}", symbol(&m.name), m.fold))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_rank_by_fold_over_other_communities() {
        // g0 is everywhere, g1 marks community 1, g2 marks it more weakly,
        // g3 is specific to 1 but too faint to count. The median rate in
        // community 1 is 6, so g0 (5) falls below the floor too.
        let names: Vec<Box<str>> = ["g0", "g1", "g2", "g3"].map(Box::from).to_vec();
        let rates = Mat::from_row_slice(
            4,
            3,
            &[
                5., 5., 5., //
                1., 9., 1., //
                2., 6., 2., //
                0., 0.01, 0., //
            ],
        );
        let fr = FeatureRates { names, rates };
        let top = fr.top(1, 3);
        let names: Vec<&str> = top.iter().map(|m| m.name.as_ref()).collect();
        assert_eq!(names, ["g1", "g2"]);
        // 9 / 1, softened by the pseudocount (1% of the median rate).
        assert!((8.0..9.0).contains(&top[0].fold));
        assert!(fr.top(7, 3).is_empty());
    }

    #[test]
    fn symbols_drop_the_ensembl_id() {
        assert_eq!(symbol("ENSG00000116824_CD2"), "CD2");
        assert_eq!(symbol("CD45RA"), "CD45RA");
        assert_eq!(symbol("HLA_DRA"), "HLA_DRA");
    }
}
