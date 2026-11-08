//! Semi-synthetic benchmark: can the estimator recover a known effect on
//! this tissue, and does the crude contrast get fooled?
//!
//! The tissue stays real: coordinates, contacts, the confounder matrix,
//! identity bins, depth, receptor competence, and the per-bin levels of the
//! real ligand and of every real programme gene. What the truth needs is
//! simulated: who truly makes ligand, and how each programme gene responds.
//!
//! 1. **True and detected ligand.** Each cell has a latent producing state
//!    `P_i ~ Bernoulli(p_i)` and a true rate `λ_i = P_i r_b`; its detected
//!    count is `Poisson(sf_i λ_i)`. Calibration, per identity bin `b`, from
//!    the real ligand: a zero-inflated Poisson fitted by moments, matching
//!    the bin's mean `m_b` of count / size factor and its detection share
//!    `d_b`: `p_b r_b = m_b` and `p_b · mean_i(1 − e^{−sf_i r_b}) = d_b`.
//!    `p_b` is thus the detection-corrected share of producers and `r_b`
//!    the mean among them. A bin detected at least as often as Poisson
//!    with every cell producing gets `p_b = 1`, `r_b = m_b`. The rate is
//!    floored at `R_FLOOR` expected counts at median depth: in a bin of
//!    scattered single counts mean and detection cannot tell few producers
//!    from many weak ones, and without the floor ambient counts would make
//!    every cell a weak producer. A hidden field moves production:
//!    `logit p_i = logit p_b + α f(Z_i)`.
//!
//!    The **true exposure** is the dose version
//!    `T* = 1{Σ_contacts λ_j > τ*}`, with `τ*` from the estimator's
//!    threshold rule applied to the latent dose (so `τ* = 0`, "some contact
//!    produces", unless that leaves an arm thin). Receivers respond to `T*`;
//!    the estimator only sees detected counts, so dropout misclassifies its
//!    exposure and attenuates its estimate (column `misclassified`).
//! 2. **Response from the real programme.** Each programme gene `g`, with
//!    signed weight `w_g` (sign `s_g`), is drawn per receiver as
//!    `Poisson(sf · μ_{b,g} · e^{s_g β f(Z)} · e^{s_g θ_b C T*})`,
//!    with `μ_{b,g}` the real gene's mean count / size factor in the bin and
//!    `C` whether the receiver responds. The response is the signed
//!    weighted programme the estimator uses,
//!    `Σ_g w_g log1p(x_g / sf) / Σ_g |w_g|`. Potential outcomes are drawn
//!    from one shared uniform per receiver and gene, so their difference is
//!    the effect alone.
//! 3. **Hidden confounder** `Z`, standardized, never shown to the estimator
//!    ([`ConfounderKind`]): random Fourier features (the adjustment's own
//!    basis: a best case), random Gaussian bumps at random centres, or a
//!    real gene's expression smoothed over contacts (a field from neither
//!    the adjustment's basis nor its identity, which leaves the pair genes
//!    out). It enters through a shape ([`ConfoundShape`]): `Z` or `Z² − 1`.
//!    It acts alike in every cell type: a field that differs by cell type is
//!    not simulated, so these results do not speak to it (future work).
//!    Composition confounding comes free: `p_b`, `r_b` and `μ_{b,g}` follow identity,
//!    and neighbours' identities are correlated.
//! 4. **Truths**, two of them, on the scale the response lives on, as measured
//!    (with spillover's bleed, below). `truth` is what the estimate targets: the
//!    realized effect `T* (y(1) − y(0))`,
//!    contrasted between the arms the estimator sees under its arm weights at
//!    each threshold, then averaged over thresholds by the estimate's own
//!    weights. Misclassified exposure is thus part of the target, so coverage
//!    tests the estimator alone. `truth_exposed` is the mean effect among the
//!    truly exposed competent receivers, without the estimator; `estimate` far
//!    below it means the arms could not show the effect.
//! 5. **Heterogeneous effect**: `θ_b = θ u_b`, `u_b` uniform on `[0.5, 1.5]`
//!    per identity bin.
//! 6. **Marker**: the ligand is made only by a "marker type", the top
//!    quarter of identity bins by real ligand, which alone responds and
//!    carries a programme baseline raised by [`MARKER_LIFT`]; contacts stay
//!    real, so the type's homotypic contacts make its cells the exposed
//!    ones. The receptor call the estimator uses is the real one plus the
//!    marker type. Exposure then tracks cell type: the crude contrast should
//!    reject almost always, the estimator should find no overlap in the
//!    marker type (not estimable, or near 0 under no effect), and power
//!    under an effect is low.
//!
//! 7. **Confounded receptor** (on in every scenario of the default grid,
//!    `confound_receptor`; off in [`Scenario::default`], which keeps the
//!    real receptor). The receptor is simulated too, so the hidden field
//!    that drives ligand and response also drives the receptor:
//!    `Poisson(sf · ρ_b · e^{β f(Z)} / mean e^{β f(Z)})`, with `ρ_b` the real
//!    receptor's mean count / size factor in the bin and `β` the response's
//!    confounding strength. Without it the co-expression baselines
//!    ([`super::baselines`]) would see a receptor the field never touches
//!    and could not be fooled by it. The receptor does nothing to the
//!    response: competence (who responds, and whom the estimator calls
//!    competent) is read by the estimator's own rule from the simulated
//!    receptor's detection, as `pinto lra` reads it from the real one.
//!
//! Every run also scores the co-expression baselines of
//! [`super::baselines`] on the ligand and receptor the estimator sees.
//!
//! Four departures, each off by default:
//!
//! - **Spillover** κ: segmentation hands each receiver a share κ of every
//!   contact neighbour's programme counts, and of their ligand counts in
//!   its own-ligand covariate. It changes the measurement, not the effect:
//!   the truths are the same contrasts on the measured scale, each
//!   potential count carrying the realized bleed from the neighbours'
//!   observed counts (bleed inside `log1p` compresses a contrast, and an
//!   unbled truth would charge that compression to the estimator). The
//!   neighbour-response and bleed covariates are what should absorb it,
//!   and a fit without them is reported beside.
//! - **Shift**: the estimator sees the detected ligand field displaced by
//!   a few contact lengths (each cell reads the cell nearest its point plus
//!   the offset, in its own section). Receivers still respond to the
//!   undisplaced field, so the seen exposure has no effect (truths 0); a
//!   contact effect should vanish while confounding at a coarser scale
//!   still fools the crude contrast.
//! - **Communities** ([`CommunityLayout`]): synthetic labels of a chosen number,
//!   skew and mixing. The effect then travels only through contacts between
//!   communities, so `T*` reads the dose over those contacts alone, and the
//!   truths follow it. Hidden fields tied to the labels make communities differ
//!   in ligand and baseline ([`ConfounderKind::Community`]) or make the boundary
//!   itself raise both ([`ConfounderKind::Boundary`]).

mod calibrate;
pub mod communities;
mod draw;
mod fields;
mod row;
mod scenario;

pub use calibrate::*;
pub use draw::*;
pub use fields::*;
pub use row::*;
pub use scenario::*;
