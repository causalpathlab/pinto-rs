//! One simulation: the panel, the truth and the expression, drawn once,
//! with the expected and drawn counts and the realised read-outs.

use super::draw::{nb_quantile, phi, Latent};
use super::expression::{
    active, draw_depth, draw_genes, draw_nuisance, scaling, spot_noise, Components, Genes,
    Nuisance, Scaling, Terms,
};
use super::field::mean_sd;
use super::panel::{build_panel, Coords, Section};
use super::params::SimParams;
use super::truth::{plant, Truth};
use rayon::prelude::*;

pub struct Sim {
    pub p: SimParams,
    pub sections: Vec<Section>,
    pub truth: Truth,
    pub genes: Genes,
    pub nuisance: Nuisance,
    pub scaling: Scaling,
    /// Per section, per spot.
    pub depth: Vec<Vec<f64>>,
}

impl Sim {
    pub fn new(p: SimParams, geometries: &[Coords]) -> Self {
        let sections = build_panel(&p, geometries);
        let truth = plant(&p, &sections);
        let genes = draw_genes(&p, &truth);
        let nuisance = draw_nuisance(&p, truth.n_states());
        let scaling = scaling(&p, &sections, &truth, &genes);
        let depth = draw_depth(&p, &sections);
        Self {
            p,
            sections,
            truth,
            genes,
            nuisance,
            scaling,
            depth,
        }
    }

    /// The scaled log-rate terms of gene `g` in section `s`.
    pub fn components(&self, s: usize, g: usize) -> Components {
        let (sc, nu, genes) = (&self.scaling, &self.nuisance, &self.genes);
        let d = self.sections[s].donor;
        let (shift, scale) = sc.ty[g];
        Components {
            a: genes.a[g],
            ty: genes
                .programme
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
        self.components(s, g)
            .log_rate(on, &self.truth.mixture[s])
            .into_iter()
            .zip(&self.depth[s])
            .map(|(l, d)| d * l.exp())
            .collect()
    }

    /// Counts of section `s` as `(gene, spot, count)`.
    pub fn counts(&self, s: usize) -> Vec<(u64, u64, f32)> {
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

    /// Realised PVE of each term over programme genes, from the drawn log
    /// rates (depth out): the variance over spots with that term alone,
    /// summed over genes, over the variance with every term the run keeps,
    /// summed over genes. Pooled, since a nuisance term's share varies from
    /// gene to gene and matches its target in expectation.
    /// Terms the run drops read 0.
    pub fn realised_pve(&self) -> [f64; 5] {
        let run = active(&self.p);
        let prog: Vec<usize> = (0..self.p.n_genes)
            .filter(|&g| self.genes.programme_of[g].is_some())
            .collect();
        // per gene: each term's variance alone, and the total's
        let per_gene: Vec<([f64; 5], f64)> = prog
            .par_iter()
            .map(|&g| {
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
                let alone = std::array::from_fn(|k| var(only(k)));
                (alone, var(run))
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
}
