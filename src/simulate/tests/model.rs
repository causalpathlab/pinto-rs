use crate::simulate::draw::{nb_quantile, phi, Latent};
use crate::simulate::expression::{active, draw_nuisance, rotate_to_cosine};
use crate::simulate::field::{mean_sd, rff_field};
use crate::simulate::panel::grid;
use crate::simulate::params::{PatchKind, SimParams, SCENARIOS};
use crate::simulate::rng::{key, Stream};
use crate::simulate::sim::Sim;
use crate::simulate::truth::assign_by_offsets;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// 3 donors × 2 sections of a 24 × 24 grid, 200 genes.
fn small(p: SimParams) -> Sim {
    sim_of(SimParams {
        grid_side: 24,
        n_genes: 200,
        programme_genes: 20,
        ..p
    })
}

/// A simulation of `p` on its own grid, every setting as given.
fn sim_of(p: SimParams) -> Sim {
    let g = grid(p.grid_side, p.grid_spacing_um);
    Sim::new(p, &[g])
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let ((ma, sa), (mb, sb)) = (mean_sd(a), mean_sd(b));
    let cov = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / a.len() as f64;
    cov / (sa * sb)
}

fn sorted(mut v: Vec<(u64, u64, f32)>) -> Vec<(u64, u64, f32)> {
    v.sort_by_key(|t| (t.0, t.1));
    v
}

////////////////
// parameters //
////////////////

#[test]
fn every_scenario_is_valid() {
    for sc in SCENARIOS {
        SimParams::scenario(sc.name).unwrap().validate().unwrap();
    }
    assert!(SimParams::scenario("nope").is_err());
}

#[test]
fn pve_terms_may_not_exceed_one() {
    let p = SimParams::default()
        .with_overrides(&["pve_donor=0.5".into(), "pve_inter=0.3".into()])
        .unwrap();
    assert!(p.validate().is_err());
}

#[test]
fn overrides_set_fields_and_reject_unknown_keys() {
    let p = SimParams::default()
        .with_overrides(&[
            "pve_donor=0.2".into(),
            "patch_kind=new-mix".into(),
            "clean_donors=[1]".into(),
            "nuisance_off=true".into(),
        ])
        .unwrap();
    assert_eq!(p.pve_donor, 0.2);
    assert_eq!(p.patch_kind, PatchKind::NewMix);
    assert_eq!(p.clean_donors, vec![1]);
    assert!(p.nuisance_off);
    assert!(SimParams::default()
        .with_overrides(&["pve_typo=1".into()])
        .is_err());
}

/// Distinct streams and indices give distinct keys: no tag plus index can
/// land on another stream's key, and swapped indices are different keys.
#[test]
fn stream_keys_do_not_collide() {
    let mut seen = std::collections::HashSet::new();
    let streams = [
        Stream::Programmes,
        Stream::Noise,
        Stream::Depth,
        Stream::Factor,
        Stream::Unique,
    ];
    for st in streams {
        for i in 0..3000 {
            assert!(seen.insert(key(1, st, &[i])));
        }
    }
    let mut pairs = std::collections::HashSet::new();
    for s in 0..60 {
        for g in 0..60 {
            assert!(
                pairs.insert(key(1, Stream::Noise, &[s, g])),
                "section {s}, gene {g}"
            );
        }
    }
}

///////////
// field //
///////////

/// Spots much closer than the length scale move together; spots five
/// length scales apart do not. On a window ten length scales wide, so the
/// standardization does not itself correlate far spots.
#[test]
fn a_field_is_smooth_at_its_length_scale() {
    let side = 60;
    let coords = grid(side, 50.0);
    let (mut near, mut far) = (0.0, 0.0);
    let n_seeds = 10;
    for seed in 0..n_seeds {
        let z = rff_field(&coords, 300.0, &mut SmallRng::seed_from_u64(seed));
        let (m, sd) = mean_sd(&z);
        assert!(m.abs() < 1e-9 && (sd - 1.0).abs() < 1e-9);
        // along each column: neighbours 50 um apart, and spots 1500 um apart
        let pairs = |lag: usize| -> f64 {
            let v: Vec<f64> = (0..side)
                .flat_map(|col| {
                    (0..side - lag).map(move |j| (col * side + j, col * side + j + lag))
                })
                .map(|(a, b)| z[a] * z[b])
                .collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        near += pairs(1);
        far += pairs(30);
    }
    let (near, far) = (near / n_seeds as f64, far / n_seeds as f64);
    assert!(near > 0.9, "near {near}");
    assert!(far.abs() < 0.15, "far {far}");
}

///////////
// truth //
///////////

#[test]
fn offsets_bring_domain_shares_to_their_targets() {
    let coords = grid(30, 100.0);
    let fields: Vec<Vec<f64>> = (0..3)
        .map(|k| rff_field(&coords, 600.0, &mut SmallRng::seed_from_u64(k)))
        .collect();
    let target = [0.5, 0.3, 0.2];
    let z = assign_by_offsets(&fields, &target);
    for (k, &t) in target.iter().enumerate() {
        let share = z.iter().filter(|&&d| d == k).count() as f64 / 900.0;
        assert!((share - t).abs() < 0.02, "domain {k}: {share} vs {t}");
    }
}

/// The donor-specific domain covers its share of the owner's spots and
/// appears in no other donor.
#[test]
fn the_donor_patch_sits_in_its_owner_alone() {
    let sim = small(SimParams::scenario("L2").unwrap());
    let (p, truth) = (&sim.p, &sim.truth);
    let patch = p.n_domains;
    assert_eq!(truth.domains[patch].owner, Some(p.patch_owner));
    for (s, sec) in sim.sections.iter().enumerate() {
        let share = truth.domain[s].iter().filter(|&&k| k == patch).count() as f64 / sec.n() as f64;
        if sec.donor == p.patch_owner {
            assert!(
                (share - p.patch_share).abs() < 0.01,
                "{}: {share}",
                sec.name
            );
        } else {
            assert_eq!(share, 0.0, "{}", sec.name);
        }
    }
}

#[test]
fn a_new_state_appears_only_in_the_donor_patch() {
    let sim = small(SimParams::scenario("L2").unwrap());
    let (p, truth) = (&sim.p, &sim.truth);
    let new_state = p.n_domains;
    assert_eq!(truth.states[new_state].owner, Some(p.patch_owner));
    for s in 0..sim.sections.len() {
        for (i, w) in truth.mixture[s].iter().enumerate() {
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            if truth.domain[s][i] != p.n_domains {
                assert_eq!(w[new_state], 0.0);
            }
        }
    }
}

/// A new mix holds two shared states evenly: no shared domain's composition.
#[test]
fn a_new_mix_is_made_of_shared_states_only() {
    let sim = small(SimParams::scenario("L2-mix").unwrap());
    let (p, truth) = (&sim.p, &sim.truth);
    assert_eq!(truth.n_states(), p.n_domains);
    let comp = &truth.composition[p.n_domains];
    assert_eq!((comp[0], comp[1]), (0.5, 0.5));
    for c in &truth.composition[..p.n_domains] {
        assert!(c.iter().cloned().fold(0.0, f64::max) >= p.dominant);
    }
}

////////////////
// expression //
////////////////

#[test]
fn rotation_gives_the_requested_cosine_at_the_same_norm() {
    let mut r = SmallRng::seed_from_u64(3);
    let mut b = vec![0.0; 300];
    (0..30).for_each(|g| b[g * 7] = 1.0);
    for c in [0.0, 0.3, 0.8, -0.5] {
        let v = rotate_to_cosine(&b, c, 30, &mut r);
        let dot: f64 = v.iter().zip(&b).map(|(x, y)| x * y).sum();
        let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((dot / (nv * nb) - c).abs() < 1e-9, "cosine {c}");
        assert!((nv - nb).abs() < 1e-9);
    }
}

/// For pure spots, each term's share of the drawn log rate's variance is
/// its PVE. (Mixed spots shrink the type term inside the log; the manifest
/// reports what a run realises.)
#[test]
fn realised_pve_matches_the_targets() {
    // enough programme genes that the pooled nuisance shares sit near their
    // expectation: with three donors each gene's donor variance has 2
    // degrees of freedom, so over 200 genes the donor share spreads by about
    // 0.014; the tolerance is 2.5 times that
    let sim = sim_of(SimParams {
        grid_side: 24,
        n_genes: 2000,
        programme_genes: 200,
        dominant: 1.0,
        pve_type: 0.3,
        pve_donor: 0.2,
        pve_inter: 0.1,
        pve_section: 0.05,
        ..SimParams::default()
    });
    let r = sim.realised_pve();
    let want = [0.3, 0.2, 0.1, 0.05, sim.p.pve_noise()];
    for (k, (got, want)) in r.iter().zip(want).enumerate() {
        assert!((got - want).abs() < 0.035, "term {k}: {got} vs {want}");
    }
}

/// With nuisance off, the realised nuisance shares are 0.
#[test]
fn realised_pve_follows_nuisance_off() {
    let sim = small(SimParams {
        pve_donor: 0.2,
        pve_section: 0.1,
        nuisance_off: true,
        ..SimParams::default()
    });
    let r = sim.realised_pve();
    assert!(r[1] < 1e-12 && r[2] < 1e-12 && r[3] < 1e-12, "{r:?}");
}

/// A clean donor's spots get no donor or donor-by-type term: switching all
/// nuisance off changes nothing there when there is no section term.
#[test]
fn a_clean_donor_carries_no_donor_nuisance() {
    let p = SimParams {
        pve_donor: 0.2,
        pve_inter: 0.2,
        clean_donors: vec![0],
        ..SimParams::default()
    };
    let sim = small(p.clone());
    let (on, off) = (
        active(&p),
        active(&SimParams {
            nuisance_off: true,
            ..p
        }),
    );
    for g in 0..sim.p.n_genes {
        let (a, b) = (sim.mean(0, g, on), sim.mean(0, g, off));
        assert!(a
            .iter()
            .zip(&b)
            .all(|(x, y)| (x - y).abs() <= 1e-12 * x.abs().max(1.0)));
    }
    // a donor with nuisance does change
    let last = sim.sections.len() - 1;
    let (a, b) = (sim.mean(last, 0, on), sim.mean(last, 0, off));
    assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6));
}

/// The same seed with nuisance off keeps every other draw: the counts of a
/// clean donor are identical, and elsewhere the counts move with the rate.
#[test]
fn nuisance_off_is_the_exact_counterfactual() {
    let p = SimParams {
        pve_donor: 0.3,
        clean_donors: vec![0],
        ..SimParams::default()
    };
    let (a, b) = (
        small(p.clone()),
        small(SimParams {
            nuisance_off: true,
            ..p
        }),
    );
    assert_eq!(sorted(a.counts(0)), sorted(b.counts(0)));
    let last = a.sections.len() - 1;
    assert_ne!(sorted(a.counts(last)), sorted(b.counts(last)));
}

/// Adding the donor-only state (L1 to L2) leaves every other nuisance
/// draw as it was, so the two rungs are paired.
#[test]
fn adding_a_state_keeps_every_other_nuisance_draw() {
    let p = SimParams::scenario("L1").unwrap();
    let (a, b) = (
        draw_nuisance(&p, p.n_domains),
        draw_nuisance(&p, p.n_domains + 1),
    );
    for g in 0..p.n_genes {
        assert_eq!(a.donor[g], b.donor[g]);
        assert_eq!(a.section[g], b.section[g]);
        assert_eq!(a.inter[g][..], b.inter[g][..p.n_domains]);
    }
}

/// With one donor, no nuisance, no spot noise, no depth spread and pure
/// domains of one state each, a programme gene's rate is one value inside
/// its domain and another outside: SRTsim's fold-change domains.
#[test]
fn one_donor_without_nuisance_reduces_to_fold_change_domains() {
    let sim = small(SimParams {
        n_donors: 1,
        pve_type: 1.0,
        depth_sd: 0.0,
        dominant: 1.0,
        ..SimParams::default()
    });
    let genes = &sim.genes;
    let g = (0..sim.p.n_genes)
        .find(|&g| {
            genes.programme_of[g] == Some(0)
                && (1..sim.p.n_domains).all(|c| genes.programme[c][g] == 0.0)
        })
        .expect("a gene in the first state's programme alone");
    for s in 0..sim.sections.len() {
        let mu = sim.mean(s, g, active(&sim.p));
        let part = |inside: bool| -> Vec<f64> {
            (0..mu.len())
                .filter(|&i| (sim.truth.domain[s][i] == 0) == inside)
                .map(|i| mu[i])
                .collect()
        };
        let (inside, outside) = (part(true), part(false));
        let same = |v: &[f64]| v.iter().all(|x| (x - v[0]).abs() < 1e-9 * v[0]);
        assert!(same(&inside) && same(&outside));
        assert!(inside[0] > outside[0]);
    }
}

//////////
// draw //
//////////

#[test]
fn phi_is_the_normal_cdf() {
    assert!((phi(0.0) - 0.5).abs() < 1e-15);
    assert!((phi(1.959963984540054) - 0.975).abs() < 1e-9);
    assert!((phi(-1.0) + phi(1.0) - 1.0).abs() < 1e-15);
}

/// Quantiles of evenly spaced uniforms reproduce the NB mean and variance,
/// also past where the mass at 0 underflows.
#[test]
fn the_nb_quantile_has_the_right_mean_and_variance() {
    // (mean, dispersion, uniforms, tolerance on the variance)
    let cases = [
        (0.3, 0.0, 20_000, 0.03),
        (5.0, 0.0, 20_000, 0.03),
        (2.0, 0.5, 20_000, 0.03),
        (40.0, 0.2, 20_000, 0.03),
        (800.0, 0.0, 2_000, 0.05),
        (1500.0, 0.0, 2_000, 0.05),
        (1200.0, 0.001, 2_000, 0.05),
    ];
    for (mu, disp, n, tol) in cases {
        let y: Vec<f64> = (0..n)
            .map(|k| nb_quantile((k as f64 + 0.5) / n as f64, mu, disp) as f64)
            .collect();
        let (m, sd) = mean_sd(&y);
        let want_v = mu + disp * mu * mu;
        assert!((m - mu).abs() < 0.01 * mu.max(1.0), "mean {m} vs {mu}");
        assert!(
            (sd * sd - want_v).abs() < tol * want_v,
            "var {} vs {want_v}",
            sd * sd
        );
    }
}

/// Two genes on one factor have latent correlation `copula_strength`; on
/// different factors, none; at strength 0, none.
#[test]
fn the_copula_correlates_genes_through_shared_factors() {
    let n = 20_000;
    let p = SimParams {
        copula_rank: 4,
        copula_strength: 0.4,
        ..SimParams::default()
    };
    let lat = Latent::new(&p, 0, n);
    let (a, b, c) = (lat.gene(0, 1), lat.gene(1, 1), lat.gene(2, 2));
    assert!((corr(&a, &b) - 0.4).abs() < 0.03);
    assert!(corr(&a, &c).abs() < 0.03);
    let lat = Latent::new(
        &SimParams {
            copula_strength: 0.0,
            ..p
        },
        0,
        n,
    );
    assert!(corr(&lat.gene(0, 1), &lat.gene(1, 1)).abs() < 0.03);
}

/// A gene's latent draws do not depend on how many genes there are.
#[test]
fn a_gene_keeps_its_draws_when_the_gene_count_changes() {
    let p = SimParams::default();
    let a = Latent::new(&p, 0, 50).gene(7, 2);
    let more = SimParams { n_genes: 5000, ..p };
    assert_eq!(a, Latent::new(&more, 0, 50).gene(7, 2));
}

/// No gene's nuisance term is far outside its target spread. A gene whose
/// few donor or section draws happen to lie close together must not get a
/// huge scale (and a huge offset): over a thousand genes, every scaled term
/// stays within a few multiples of √(pve · V), with and without a clean donor.
#[test]
fn no_gene_gets_an_outsized_nuisance_term() {
    for name in ["L1", "clean-one"] {
        let sim = sim_of(SimParams {
            grid_side: 12,
            ..SimParams::scenario(name).unwrap()
        });
        let (p, nu, sc) = (&sim.p, &sim.nuisance, &sim.scaling);
        let bound = |pve: f64| 6.0 * (pve * p.log_rate_var).sqrt();
        for g in 0..p.n_genes {
            for d in 0..p.n_donors {
                assert!(
                    (sc.donor * nu.donor[g][d]).abs() <= bound(p.pve_donor),
                    "{name} gene {g} donor {d}"
                );
                for x in &nu.inter[g] {
                    assert!(
                        (sc.inter * x[d]).abs() <= bound(p.pve_inter),
                        "{name} gene {g} inter"
                    );
                }
            }
            for (s, x) in nu.section[g].iter().take(p.n_sections()).enumerate() {
                assert!(
                    (sc.section * x).abs() <= bound(p.pve_section),
                    "{name} gene {g} section {s}"
                );
            }
        }
    }
}

/// A gene's scaled nuisance terms depend on its own draws alone, not on how
/// many other genes there are: a contract, so that no later change scales a
/// nuisance term from other genes' draws.
#[test]
fn a_gene_keeps_its_nuisance_when_the_gene_count_changes() {
    let base = SimParams {
        grid_side: 12,
        ..SimParams::scenario("L1").unwrap()
    };
    let a = sim_of(SimParams {
        n_genes: 200,
        ..base.clone()
    });
    let b = sim_of(SimParams {
        n_genes: 600,
        ..base
    });
    for g in 0..200 {
        let (ca, cb) = (a.components(0, g), b.components(0, g));
        assert_eq!((ca.donor, ca.section), (cb.donor, cb.section), "gene {g}");
        assert_eq!(ca.inter, cb.inter, "gene {g}");
    }
}
