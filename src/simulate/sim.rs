//! One simulation: the panel, the truth and the expression, drawn once,
//! with the expected and drawn counts and the realized read-outs.

use super::draw::{nb_quantile, phi, Latent};
use super::expression::{
    active, draw_depth, draw_genes, draw_nuisance, scaling, spot_noise, Components, Genes,
    Nuisance, Scaling, Term, Terms,
};
use super::field::mean_sd;
use super::panel::{build_panel, Coords, Section};
use super::params::SimParams;
use super::reference_data::ReferenceData;
use super::rng::{rng, Stream};
use super::truth::{plant, Truth};
use rayon::prelude::*;

pub struct Sim {
    pub p: SimParams,
    pub sections: Vec<Section>,
    pub truth: Truth,
    pub genes: Genes,
    pub nuisance: Nuisance,
    pub scaling: Scaling,
    /// Per section, per spot: the depth multiplier, or under a reference
    /// the depth coordinate the genes follow ([`super::reference_data`]).
    pub depth: Vec<Vec<f64>>,
    /// The donors with no shift ([`SimParams::unconfounded_donor_set`]).
    pub unconfounded_donors: Vec<usize>,
    /// The reference data the counts are drawn from, when there is one.
    pub reference_data: Option<ReferenceData>,
}

impl Sim {
    pub fn new(p: SimParams, geometries: &[Coords]) -> Self {
        let sections = build_panel(&p, geometries);
        let truth = plant(&p, &sections);
        let genes = draw_genes(&p, &truth);
        let nuisance = draw_nuisance(&p, truth.n_states());
        let scaling = scaling(&p, &sections, &truth, &genes);
        let depth = draw_depth(&p, &sections);
        let unconfounded_donors = p.unconfounded_donor_set();
        Self {
            p,
            sections,
            truth,
            genes,
            nuisance,
            scaling,
            depth,
            unconfounded_donors,
            reference_data: None,
        }
    }

    /// A simulation of `p` whose expression is the reference at `path`'s
    /// (grouped by `labels` when given; [`super::reference_data`]): the panel and
    /// the truth as without one, the genes the reference's, each state's
    /// rates its group's, the nuisance a share of the reference's variance
    /// between states, each state's markers its program.
    pub fn from_reference_data(
        mut p: SimParams,
        geometries: &[Coords],
        path: &str,
        labels: Option<&str>,
    ) -> anyhow::Result<Self> {
        let sections = build_panel(&p, geometries);
        let truth = plant(&p, &sections);
        let spot_share_by_state = spot_share_by_state(&truth);
        let reference_data =
            ReferenceData::fit(path, labels, &spot_share_by_state, &truth.mixture, &p)?;
        p.n_genes = reference_data.n_genes();
        p.log_rate_var = reference_data.log_count_variance;
        let markers = reference_data.marker_genes_by_state(p.program_genes);
        let first_state_of_marker: Vec<Option<usize>> = (0..p.n_genes)
            .map(|g| (0..markers.len()).find(|&c| markers[c][g] > 0.0))
            .collect();
        let genes = Genes {
            names: reference_data.gene_names.clone(),
            a: reference_data
                .pooled_mean_count
                .iter()
                .map(|m| m.max(1e-12).ln())
                .collect(),
            dispersion: reference_data.pooled_extra_poisson_variance.clone(),
            program: markers,
            program_of: first_state_of_marker,
            factor: vec![0; p.n_genes],
        };
        let nuisance = draw_nuisance(&p, truth.n_states());
        let scaling = scaling(&p, &sections, &truth, &genes);
        // the depth coordinate: one standard normal per spot
        let depth = (0..sections.len())
            .map(|s| super::expression::normals(rng(p.seed, Stream::Depth, &[s]), sections[s].n()))
            .collect();
        let unconfounded_donors = p.unconfounded_donor_set();
        Ok(Self {
            p,
            sections,
            truth,
            genes,
            nuisance,
            scaling,
            depth,
            unconfounded_donors,
            reference_data: Some(reference_data),
        })
    }

    /// Whether donor `d` carries a shift: the nuisance is on, `d` is not an
    /// unconfounded donor, and a donor-wide or donor-by-cell-type term, or a
    /// section term on one of its sections, has a nonzero scale.
    pub fn confounded(&self, d: usize) -> bool {
        let sc = &self.scaling;
        let free_sections = self.p.unconfounded_section_set();
        let section_shift = sc.section > 0.0
            && (0..self.sections.len())
                .any(|s| self.sections[s].donor == d && !free_sections.contains(&s));
        active(&self.p)[Term::Donor as usize]
            && !self.unconfounded_donors.contains(&d)
            && (sc.donor > 0.0 || sc.inter > 0.0 || section_shift)
    }

    /// The scaled log-rate terms of gene `g` in section `s`.
    pub fn components(&self, s: usize, g: usize) -> Components {
        let (sc, nu, genes) = (&self.scaling, &self.nuisance, &self.genes);
        let d = self.sections[s].donor;
        // the reference's rates are the type term, on the log scale (floored,
        // so a term switched off is 0, not 0 · −∞); its noise is its own
        if let Some(r) = &self.reference_data {
            return Components {
                a: 0.0,
                ty: r.log_mean_count_by_state.iter().map(|m| m[g]).collect(),
                inter: nu.inter[g].iter().map(|x| sc.inter * x[d]).collect(),
                donor: sc.donor * nu.donor[g][d],
                section: sc.section * nu.section[g][s],
                noise: vec![0.0; self.sections[s].n()],
            };
        }
        let (shift, scale) = sc.ty[g];
        Components {
            a: genes.a[g],
            ty: genes
                .program
                .iter()
                .map(|prog| scale * (prog[g] - shift))
                .collect(),
            inter: nu.inter[g].iter().map(|x| sc.inter * x[d]).collect(),
            donor: sc.donor * nu.donor[g][d],
            section: sc.section * nu.section[g][s],
            noise: spot_noise(&self.p, s, g, self.sections[s].n())
                .into_iter()
                .map(|z| sc.noise * z)
                .collect(),
        }
    }

    /// Expected counts of gene `g` over the spots of section `s`, with the
    /// terms in `on` (the run's own: [`active`]).
    pub fn mean(&self, s: usize, g: usize, on: Terms) -> Vec<f64> {
        // under a reference depth enters through the latent, not the mean
        let scaled = self.reference_data.is_none();
        self.components(s, g)
            .log_rate(on, &self.truth.mixture[s])
            .into_iter()
            .zip(&self.depth[s])
            .map(|(l, &d)| if scaled { d * l.exp() } else { l.exp() })
            .collect()
    }

    /// Counts of section `s` as `(gene, spot, count)`.
    pub fn counts(&self, s: usize) -> Vec<(u64, u64, f32)> {
        if let Some(r) = &self.reference_data {
            return self.counts_from_reference_data(r, s);
        }
        let latent = Latent::new(&self.p, s, self.sections[s].n());
        let on = active(&self.p);
        (0..self.p.n_genes)
            .into_par_iter()
            .flat_map_iter(|g| {
                let mu = self.mean(s, g, on);
                let z = latent.gene(g, self.genes.factor[g]);
                let disp = self.genes.dispersion[g];
                (0..mu.len()).filter_map(move |i| {
                    let y = nb_quantile(phi(z[i]), mu[i], disp);
                    (y > 0).then_some((g as u64, i as u64, y as f32))
                })
            })
            .collect()
    }

    /// Realized PVE of each term over program genes, from the drawn log
    /// rates (depth out): the variance over spots with that term alone,
    /// summed over genes, over the variance with every term the run keeps,
    /// summed over genes. Pooled, since a nuisance term's share varies from
    /// gene to gene and matches its target in expectation.
    /// Terms the run drops read 0.
    pub fn realized_pve(&self) -> [f64; 5] {
        let prog: Vec<usize> = (0..self.p.n_genes)
            .filter(|&g| self.genes.program_of[g].is_some())
            .collect();
        let ones = vec![1.0; prog.len()];
        self.realized_pve_over(&prog, &ones)
    }

    /// Under reference data, [`Self::realized_pve`] over the genes and
    /// weights the nuisance was scaled on
    /// ([`super::reference_data::pve_genes`]).
    pub fn realized_pve_expressed_genes(&self) -> Option<[f64; 5]> {
        let r = self.reference_data.as_ref()?;
        let share = spot_share_by_state(&self.truth);
        let (genes, weight): (Vec<usize>, Vec<f64>) =
            super::reference_data::pve_genes(&r.log_mean_count_by_state, &share, &r.detected)
                .into_iter()
                .unzip();
        Some(self.realized_pve_over(&genes, &weight))
    }

    /// Each term's variance alone, summed over `genes` at `weight`, over the
    /// total's.
    fn realized_pve_over(&self, genes: &[usize], weight: &[f64]) -> [f64; 5] {
        let run = active(&self.p);
        // per gene: each term's variance alone, and the total's
        let per_gene: Vec<([f64; 5], f64)> = genes
            .par_iter()
            .zip(weight)
            .map(|(&g, &wg)| {
                let parts: Vec<Components> = (0..self.sections.len())
                    .map(|s| self.components(s, g))
                    .collect();
                let var = |on: Terms| {
                    let v: Vec<f64> = parts
                        .iter()
                        .enumerate()
                        .flat_map(|(s, c)| c.log_rate(on, &self.truth.mixture[s]))
                        .collect();
                    mean_sd(&v).1.powi(2)
                };
                let only = |k: usize| -> Terms { std::array::from_fn(|j| j == k && run[j]) };
                let alone = std::array::from_fn(|k| wg * var(only(k)));
                (alone, wg * var(run))
            })
            .collect();
        // pooled over genes: a term's variance summed, over the total's
        let total: f64 = per_gene.iter().map(|t| t.1).sum();
        std::array::from_fn(|k| {
            if total > 0.0 {
                per_gene.iter().map(|t| t.0[k]).sum::<f64>() / total
            } else {
                0.0
            }
        })
    }

    /// Counts of section `s` drawn from the reference `r`
    /// ([`super::reference_data`]): each spot's dominant state's distribution of
    /// each gene, its rates scaled to the spot's mean, through the
    /// reference's copula and depth coupling.
    fn counts_from_reference_data(&self, r: &ReferenceData, s: usize) -> Vec<(u64, u64, f32)> {
        let n = self.sections[s].n();
        let rank = r.copula_loadings.ncols();
        // the copula's factors, rank per spot
        let copula_factors =
            super::expression::normals(rng(self.p.seed, Stream::Factor, &[s]), n * rank);
        let (zd, dominant) = (&self.depth[s], &self.truth.dominant[s]);
        let on = active(&self.p);
        (0..self.p.n_genes)
            .into_par_iter()
            .filter(|&g| r.detected[g])
            .flat_map_iter(|g| {
                let mu = self.mean(s, g, on);
                let mut unique = rng(self.p.seed, Stream::Unique, &[s, g]);
                let row = r.copula_row_of_gene[g].map(|h| h as usize);
                let copula_factors = &copula_factors;
                (0..n).filter_map(move |i| {
                    let own: f64 =
                        rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut unique);
                    let latent = match row {
                        Some(h) => {
                            let from_factors: f64 = (0..rank)
                                .map(|k| {
                                    f64::from(r.copula_loadings[(h, k)])
                                        * copula_factors[i * rank + k]
                                })
                                .sum();
                            from_factors + f64::from(r.copula_gene_sd[h]) * own
                        }
                        None => own,
                    };
                    let c = dominant[i];
                    let rho = r.depth_coupling_by_state[c][g].clamp(-1.0, 1.0);
                    let u = phi(rho * zd[i] + (1.0 - rho * rho).sqrt() * latent);
                    let state_mean = r.mean_count_by_state[c][g];
                    let y = if state_mean > 0.0 {
                        r.count_distribution_by_state[c][g].inverse_cdf(u, mu[i] / state_mean)
                    } else {
                        // a gene the group never shows: Poisson at its mean
                        nb_quantile(u, mu[i], 0.0)
                    };
                    (y > 0).then_some((g as u64, i as u64, y as f32))
                })
            })
            .collect()
    }
}

/// Each state's share of the spots, by their dominant state.
pub fn spot_share_by_state(truth: &Truth) -> Vec<f64> {
    let mut share = vec![0.0; truth.n_states()];
    let n: usize = truth.dominant.iter().map(Vec::len).sum();
    for &c in truth.dominant.iter().flatten() {
        share[c] += 1.0 / n.max(1) as f64;
    }
    share
}
