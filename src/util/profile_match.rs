//! A fitted model's community profiles on another dataset's rows: matching
//! the feature axes by canonical name, fitting a unit's community proportions
//! by EM over the profiles, and the training data's per-feature totals. Shared
//! by `pinto impute` and held-out scoring (`pinto predict` on lc / dsvd), and
//! the totals by cage's `pinto predict` too.

use crate::util::common::*;

/// A model's community profiles on a dataset's own rows: the model features
/// the data carries, each profile conditioned on that window.
pub struct MatchedProfiles {
    /// `[K, G_matched]` per-community feature probabilities, TRANSPOSED so a
    /// feature's K values are one contiguous column, renormalized over the
    /// matched features (f64 for the per-unit EM accumulators).
    pub p_t: nalgebra::DMatrix<f64>,
    /// data row → matched-profile column.
    pub row_to_matched: Vec<Option<usize>>,
}

/// Match `profiles` (`[G_model, K]`, rows named `model_features`) to the rows
/// `data_features` names, and condition each community's profile on the
/// matched window: `p̃_k(g) = λ_gk / Σ_{g∈matched} λ_gk`.
pub fn match_profiles(
    data_features: &[Box<str>],
    profiles: &Mat,
    model_features: &[Box<str>],
    feature_kind: &data_beans::aux::feature_names::FeatureNameKind,
    what: &str,
    n_cells: usize,
) -> anyhow::Result<MatchedProfiles> {
    let k = profiles.ncols();

    // Model feature → profile row. By exact name, and by canonical name too,
    // so a name mode that shortens `ENSG…_SYMBOL` to its symbol matches the
    // model's full names; a canonical name two model features share (two
    // genes, one symbol) is left out rather than given to either.
    let mut canonical: HashMap<Box<str>, Option<usize>> = HashMap::default();
    for (i, g) in model_features.iter().enumerate() {
        canonical
            .entry(feature_kind.canonicalize(g))
            .and_modify(|e| *e = None)
            .or_insert(Some(i));
    }
    let n_ambiguous = canonical.values().filter(|v| v.is_none()).count();
    if n_ambiguous > 0 {
        log::warn!(
            "{what}: {n_ambiguous} canonical feature names are shared by several model features; \
             those match only by their exact names"
        );
    }
    let mut model_idx: HashMap<Box<str>, usize> = canonical
        .into_iter()
        .filter_map(|(k, v)| v.map(|i| (k, i)))
        .collect();
    for (i, g) in model_features.iter().enumerate() {
        model_idx.insert(g.clone(), i);
    }
    let mut matched_rows: Vec<usize> = Vec::new();
    let mut row_to_matched: Vec<Option<usize>> = vec![None; data_features.len()];
    let mut seen: HashMap<usize, usize> = HashMap::default();
    for (row, name) in data_features.iter().enumerate() {
        let hit = model_idx
            .get(name)
            .or_else(|| model_idx.get(&feature_kind.canonicalize(name)));
        if let Some(&g) = hit {
            let slot = *seen.entry(g).or_insert_with(|| {
                matched_rows.push(g);
                matched_rows.len() - 1
            });
            row_to_matched[row] = Some(slot);
        }
    }
    anyhow::ensure!(
        !matched_rows.is_empty(),
        "{what}: none of its {} features match the model's {} — check --feature-name-mode",
        data_features.len(),
        model_features.len()
    );
    info!(
        "{what}: {} of the model's {} features are present across {} cells",
        matched_rows.len(),
        model_features.len(),
        n_cells
    );

    // Stored [K, G_matched] so the EM's per-feature access is one contiguous
    // column.
    let g_m = matched_rows.len();
    let mut p_t = nalgebra::DMatrix::<f64>::zeros(k, g_m);
    for (slot, &g) in matched_rows.iter().enumerate() {
        for c in 0..k {
            p_t[(c, slot)] = f64::from(profiles[(g, c)]).max(0.0);
        }
    }
    for c in 0..k {
        let total: f64 = p_t.row(c).iter().sum();
        if total > 0.0 {
            let mut row = p_t.row_mut(c);
            row /= total;
        }
    }
    Ok(MatchedProfiles {
        p_t,
        row_to_matched,
    })
}

/// One unit's community proportions `π` by EM over the profile mixture,
/// written into `pi`: `π ← Σ_g x_g · r_g` with responsibilities
/// `r_{g,k} ∝ π_k p̃_k(g)`, iterated to convergence. `obs` is the unit's
/// counts as (matched column, count). A unit with no counts gets a zero `π`
/// and `false`.
pub fn em_proportions(
    obs: &[(usize, f64)],
    p_t: &nalgebra::DMatrix<f64>,
    iters: usize,
    pi: &mut [f64],
) -> bool {
    let k = p_t.nrows();
    if obs.iter().map(|&(_, x)| x).sum::<f64>() <= 0.0 {
        pi.fill(0.0);
        return false;
    }
    let mut num = vec![0.0f64; k];
    let mut resp = vec![0.0f64; k];
    pi.fill(1.0 / k as f64);
    for _ in 0..iters.max(1) {
        num.fill(0.0);
        for &(slot, x) in obs {
            // `resp_c = π_c · p̃_c(g)`, computed once for both the
            // denominator and the numerator; the profile column is
            // contiguous, so this is a straight cache-line scan.
            let p_col = p_t.column(slot);
            let mut denom = 0.0f64;
            for c in 0..k {
                resp[c] = pi[c] * p_col[c];
                denom += resp[c];
            }
            if denom <= 0.0 {
                continue; // feature absent from every community
            }
            let f = x / denom;
            for c in 0..k {
                num[c] += resp[c] * f;
            }
        }
        let assigned: f64 = num.iter().sum();
        if assigned <= 0.0 {
            pi.fill(0.0);
            return false;
        }
        let mut delta = 0.0f64;
        for (c, pi_c) in pi.iter_mut().enumerate() {
            let next = num[c] / assigned;
            delta = delta.max((next - *pi_c).abs());
            *pi_c = next;
        }
        if delta < 1e-6 {
            break;
        }
    }
    true
}

/// Per-feature totals from the training half, on this sample's feature axis.
///
/// The training data gets its own `FeatureAxis`, so a channelized training matrix
/// pools the same way this one does before the two are matched by name. A feature
/// the training half never saw keeps a zero total, which `PairDictionary` reads
/// as "not on the partition axis" and drops -- the same treatment a feature with no
/// counts here gets, and the honest one: the model has no abundance for it.
pub fn training_feature_totals(
    files: &[Box<str>],
    common: &crate::util::input::SrtInputArgs,
    feature_kind: &data_beans::aux::feature_names::FeatureNameKind,
    target_features: &[Box<str>],
) -> anyhow::Result<Vec<f64>> {
    // The query's row names were canonicalized on the way in, by the same
    // `--feature-name-mode` this uses. Without applying it here the two sides are
    // spelled differently and nothing matches — which is exactly the failure this
    // hit first, and it surfaced as "shares no expressed feature" rather than as a
    // naming problem.
    let mut loaded = data_beans::sparse_io_vector::SparseIoVec::new();
    if let Some(canon) = feature_kind.clone().into_canonicalizer() {
        loaded = loaded
            .with_row_canonicalizer(move |name| canon(name))
            .expect("with_row_canonicalizer on empty SparseIoVec");
    }
    for f in files {
        let data = crate::util::input::open_sparse(f, common.preload_data)?;
        loaded.push(std::sync::Arc::from(data), None)?;
    }
    let axis = crate::util::feature_axis::FeatureAxis::resolve(&loaded.row_names()?)?;
    let per_row = crate::link_community::profiles::compute_row_totals(&loaded, common.block_size)?;
    let per_feature = axis.pool_totals(&per_row);

    let total_of: HashMap<&str, f64> = axis
        .feature_names()
        .iter()
        .map(std::convert::AsRef::as_ref)
        .zip(per_feature.iter().copied())
        .collect();
    let out: Vec<f64> = target_features
        .iter()
        .map(|g| total_of.get(g.as_ref()).copied().unwrap_or(0.0))
        .collect();

    let matched = out.iter().filter(|&&t| t > 0.0).count();
    anyhow::ensure!(
        matched > 0,
        "--null-from shares no expressed feature with this sample after canonicalizing names \
         with --feature-name-mode; check that it points at the training data for this model"
    );
    info!(
        "Training totals: {matched} of {} features carry counts in the training half ({} cells)",
        target_features.len(),
        loaded.num_columns()
    );
    Ok(out)
}
