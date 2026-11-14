//! The planted truth: domains on the tissue, each domain's mix of cell
//! states, and each spot's mixture. Location sets the domain and the domain
//! sets the likely cell states; expression follows from the states
//! (`expression.rs`). Nothing here sees expression.
//!
//! Domains: each shared domain `k` has a smooth field `g_k` per section, and
//! a spot takes `argmax_k g_k + c_k`, with the offsets `c_k` tuned so domain
//! `k` covers its donor's share `π_{d,k} ∝ exp(jit_b e_{d,k})`. Shared
//! domain `k` is dominated by shared state `k`. The donor-specific domain
//! has its own field, of its own length scale, on the owner's sections
//! only, and takes the spots above the quantile that makes it cover
//! `patch_share` of them.

use super::field::rff_field;
use super::panel::Section;
use super::params::{PatchKind, SimParams};
use super::rng::{rng, Stream};
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, StandardNormal};

/// Rounds of the offset update that matches domain shares to their targets.
const OFFSET_ROUNDS: usize = 60;

/// A domain or a state: its label, and the donor that alone has it.
pub struct Info {
    pub label: Box<str>,
    pub owner: Option<usize>,
}

pub struct Truth {
    pub states: Vec<Info>,
    pub domains: Vec<Info>,
    /// `composition[k][c]`: domain `k`'s mix over states.
    pub composition: Vec<Vec<f64>>,
    /// Per section, per spot: domain index.
    pub domain: Vec<Vec<usize>>,
    /// Per section, per spot: mixture over states.
    pub mixture: Vec<Vec<Vec<f64>>>,
    /// Per section, per spot: the state with the largest share.
    pub dominant: Vec<Vec<usize>>,
}

impl Truth {
    pub fn n_states(&self) -> usize {
        self.states.len()
    }
}

/// Domain share targets per donor: equal, spread by `jit_b`.
pub fn domain_targets(p: &SimParams) -> Vec<Vec<f64>> {
    let mut r = rng(p.seed, Stream::DomainShares, &[]);
    (0..p.n_donors)
        .map(|_| {
            let w: Vec<f64> = (0..p.n_domains)
                .map(|_| {
                    let e: f64 = StandardNormal.sample(&mut r);
                    (p.jit_b * e).exp()
                })
                .collect();
            let t: f64 = w.iter().sum();
            w.into_iter().map(|v| v / t).collect()
        })
        .collect()
}

fn argmax(v: impl Iterator<Item = f64>) -> usize {
    v.enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(i, _)| i)
}

/// Argmax of `fields[k] + c_k`, with `c_k` tuned so each domain's share of
/// the spots matches `target`.
pub fn assign_by_offsets(fields: &[Vec<f64>], target: &[f64]) -> Vec<usize> {
    let n = fields[0].len();
    let mut c = vec![0.0; fields.len()];
    let assign = |c: &[f64]| -> Vec<usize> {
        (0..n)
            .map(|i| argmax(fields.iter().zip(c).map(|(f, ck)| f[i] + ck)))
            .collect()
    };
    for _ in 0..OFFSET_ROUNDS {
        let mut share = vec![0.0; fields.len()];
        assign(&c).iter().for_each(|&k| share[k] += 1.0 / n as f64);
        for (ck, (t, sh)) in c.iter_mut().zip(target.iter().zip(&share)) {
            *ck += 0.5 * (t / sh.max(0.5 / n as f64)).ln();
        }
    }
    assign(&c)
}

/// `w ~ Dirichlet(kappa · comp)` over the positive entries of `comp`; the
/// others stay 0.
fn dirichlet(comp: &[f64], kappa: f64, r: &mut SmallRng) -> Vec<f64> {
    let w: Vec<f64> = comp
        .iter()
        .map(|&a| {
            if a > 0.0 {
                Gamma::new(kappa * a, 1.0)
                    .expect("positive shape")
                    .sample(r)
            } else {
                0.0
            }
        })
        .collect();
    let t: f64 = w.iter().sum();
    w.into_iter().map(|v| v / t).collect()
}

/// States, domains and compositions. Shared state `k` dominates shared
/// domain `k`, the rest spread evenly over the other shared states. A new
/// state is appended after the shared ones and dominates the patch.
fn catalogue(p: &SimParams) -> (Vec<Info>, Vec<Info>, Vec<Vec<f64>>) {
    let n_shared = p.n_domains;
    let n_states = n_shared + (p.patch_kind == PatchKind::NewState) as usize;
    let owned = |o: bool| o.then_some(p.patch_owner);
    // names that say what they are: shared ones by kind, donor-specific ones
    // by their owner too (`shared_domain_1`, `D3_specific_domain_1`; states
    // `CT1`, `D3_specific_CT1`)
    let owner = super::panel::donor_label(p.patch_owner);
    let name = |shared: &str, specific: &str, k: usize| -> Box<str> {
        if k < n_shared {
            format!("{shared}{}", k + 1).into()
        } else {
            format!("{owner}_specific_{specific}{}", k - n_shared + 1).into()
        }
    };
    let states = (0..n_states)
        .map(|c| Info {
            label: name("CT", "CT", c),
            owner: owned(c >= n_shared),
        })
        .collect();
    // `on` takes `dominant`; the other shared states share the rest
    let dominated_by = |on: usize| -> Vec<f64> {
        let others = (0..n_shared).filter(|&c| c != on).count() as f64;
        (0..n_states)
            .map(|c| match c {
                c if c == on => p.dominant,
                c if c < n_shared => (1.0 - p.dominant) / others,
                _ => 0.0,
            })
            .collect()
    };
    let mut composition: Vec<Vec<f64>> = (0..n_shared).map(dominated_by).collect();
    match p.patch_kind {
        PatchKind::None => {}
        PatchKind::NewState => composition.push(dominated_by(n_shared)),
        PatchKind::NewMix => {
            let mut comp = vec![0.0; n_states];
            comp[0] = 0.5;
            comp[1] = 0.5;
            composition.push(comp);
        }
    }
    let domains = (0..composition.len())
        .map(|k| Info {
            label: name("shared_domain_", "domain_", k),
            owner: owned(k >= n_shared),
        })
        .collect();
    (states, domains, composition)
}

pub fn plant(p: &SimParams, sections: &[Section]) -> Truth {
    let (states, domains, composition) = catalogue(p);
    let targets = domain_targets(p);
    let patch = p.patch_kind != PatchKind::None;
    let (mut domain, mut mixture, mut dominant) = (Vec::new(), Vec::new(), Vec::new());
    for (s, sec) in sections.iter().enumerate() {
        let fields: Vec<Vec<f64>> = (0..p.n_domains)
            .map(|k| {
                let mut r = rng(p.seed, Stream::DomainField, &[s, k]);
                rff_field(&sec.coords, p.domain_length_um, &mut r)
            })
            .collect();
        let mut z = assign_by_offsets(&fields, &targets[sec.donor]);
        if patch && sec.donor == p.patch_owner {
            let mut r = rng(p.seed, Stream::PatchField, &[s]);
            let h = rff_field(&sec.coords, p.patch_length_um, &mut r);
            let mut sorted = h.clone();
            sorted.sort_by(f64::total_cmp);
            let n_patch = ((p.patch_share * sec.n() as f64).round() as usize).clamp(1, sec.n());
            let thr = sorted[sec.n() - n_patch];
            for (zi, &hi) in z.iter_mut().zip(&h) {
                if hi >= thr {
                    *zi = p.n_domains;
                }
            }
        }
        let mut r = rng(p.seed, Stream::Mixture, &[s]);
        let w: Vec<Vec<f64>> = z
            .iter()
            .map(|&k| dirichlet(&composition[k], p.kappa, &mut r))
            .collect();
        dominant.push(w.iter().map(|wi| argmax(wi.iter().copied())).collect());
        domain.push(z);
        mixture.push(w);
    }
    Truth {
        states,
        domains,
        composition,
        domain,
        mixture,
        dominant,
    }
}
