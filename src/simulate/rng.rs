//! Every random stream of the simulator, keyed in one place. A stream is
//! named by what it draws plus its indices (section, gene, state, ...), and
//! its seed chains `mix_seed` over the run seed, the stream's tag and each
//! index. Keys never add an index to a tag, so two streams cannot collide,
//! and a draw depends only on its own key: adding a gene, a state or a term
//! leaves every other draw as it was. Exact counterfactuals rest on this.

use legume_numeric::matrix::rand_util::mix_seed;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[derive(Clone, Copy)]
pub enum Stream {
    /// Domain field of (section, domain).
    DomainField,
    /// The donor-specific domain's field of (section).
    PatchField,
    /// Domain share targets.
    DomainShares,
    /// Spot mixtures of (section).
    Mixture,
    GeneBaselines,
    Programmes,
    /// Donor-wide nuisance of (gene).
    Donor,
    /// Donor-by-type nuisance of (gene, state).
    Inter,
    /// Section nuisance of (gene).
    Section,
    /// Spot depths of (section).
    Depth,
    /// Spot noise of (section, gene).
    Noise,
    /// Copula factors of (section).
    Factor,
    /// Copula unique part of (section, gene).
    Unique,
    /// Copula factor of each gene outside every programme.
    Loading,
}

pub fn key(seed: u64, stream: Stream, idx: &[usize]) -> u64 {
    idx.iter().fold(mix_seed(seed, 1 + stream as u64), |k, &i| {
        mix_seed(k, i as u64)
    })
}

pub fn rng(seed: u64, stream: Stream, idx: &[usize]) -> SmallRng {
    SmallRng::seed_from_u64(key(seed, stream, idx))
}
