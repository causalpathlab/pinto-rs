//! `pinto lra --link-community K`: the contact effect through the contacts
//! of one stage-1 link community.
//!
//! The run's link communities cluster its spatial cell pairs (links); each
//! link has one. For community K:
//!
//! - a receiver's exposure counts only the neighbours it reaches through
//!   community-K links ([`CommunityScope::exposure_contacts`]);
//! - the receivers of community K, those with at least one community-K link,
//!   are the only rows fitted and reported. Every other receiver stays in the
//!   tissue (its counts enter its neighbours' spillover, niche and
//!   confounder means) but is left out of the cross-fitting folds and of
//!   every scope: it is out of scope, never an unexposed control through
//!   links it does not have.
//!
//! Confounders, spillover, bleed, niche and competence keep the full contact
//! graph, so the adjustment is the marginal run's.

use super::contacts::Contacts;

/// `--min-receivers`' default: the fewest receivers of a community to fit it.
pub const MIN_RECEIVERS: usize = 200;

/// What a `--link-community` run reads from the contacts: the community, its
/// links as data columns, and the fewest receivers it needs.
pub struct LinkCommunityInputs {
    pub link_community: i64,
    /// The community's links, as pairs of data columns.
    pub pairs: Vec<(usize, usize)>,
    pub min_receivers: usize,
}

/// One link community on the tissue's rows.
pub struct CommunityScope {
    pub link_community: i64,
    /// The community's links between rows: the graph the exposure dose is
    /// summed over.
    pub exposure_contacts: Contacts,
    /// Whether each row is a receiver of the community.
    pub in_community: Vec<bool>,
    /// The receivers of the community, ascending rows.
    pub receiver_rows: Vec<u32>,
}

/// The links of community `link_community`, of `pairs` (each pair's community
/// in `community_of_pair`).
pub fn pairs_of_link_community(
    pairs: &[(usize, usize)],
    community_of_pair: &[i64],
    link_community: i64,
) -> Vec<(usize, usize)> {
    pairs
        .iter()
        .zip(community_of_pair)
        .filter(|&(_, &c)| c == link_community)
        .map(|(&pair, _)| pair)
        .collect()
}

/// The community on the tissue's `number_of_rows` rows (`row_of` each data
/// column's row, `usize::MAX` for none); an error below `min_receivers`.
pub fn community_scope(
    inputs: &LinkCommunityInputs,
    row_of: &[usize],
    number_of_rows: usize,
) -> anyhow::Result<CommunityScope> {
    let exposure_contacts =
        super::receivers::contacts_between_rows(&inputs.pairs, row_of, number_of_rows);
    let in_community: Vec<bool> = (0..number_of_rows)
        .map(|row| exposure_contacts.degree(row) > 0)
        .collect();
    let receiver_rows: Vec<u32> = (0..number_of_rows as u32)
        .filter(|&row| in_community[row as usize])
        .collect();
    anyhow::ensure!(
        receiver_rows.len() >= inputs.min_receivers,
        "link community {} has {} receivers (cells with a link in it), fewer than \
         --min-receivers {}",
        inputs.link_community,
        receiver_rows.len(),
        inputs.min_receivers
    );
    Ok(CommunityScope {
        link_community: inputs.link_community,
        exposure_contacts,
        in_community,
        receiver_rows,
    })
}

/// One line of `pinto lra link-communities`.
pub struct LinkCommunitySize {
    pub link_community: i64,
    /// Its links (spatial pairs).
    pub links: usize,
    /// Its receivers: the distinct cells of its links, an upper bound on what
    /// `--link-community` fits (it drops cells missing from the data or
    /// without coordinates).
    pub receivers: usize,
}

/// Every link community of `pairs` (cells named any way), with its links and
/// receivers, by community.
pub fn link_community_sizes<Cell: Ord + Clone>(
    pairs: &[(Cell, Cell)],
    community_of_pair: &[i64],
) -> Vec<LinkCommunitySize> {
    let mut cells_of: std::collections::BTreeMap<i64, (usize, Vec<Cell>)> = Default::default();
    for ((i, j), &c) in pairs.iter().zip(community_of_pair) {
        let (links, cells) = cells_of.entry(c).or_default();
        *links += 1;
        cells.extend([i.clone(), j.clone()]);
    }
    cells_of
        .into_iter()
        .map(|(link_community, (links, mut cells))| {
            cells.sort_unstable();
            cells.dedup();
            LinkCommunitySize {
                link_community,
                links,
                receivers: cells.len(),
            }
        })
        .collect()
}

/// Where a run's tables sit: `{out}.lra` for the marginal run,
/// `{out}.lra.link-community-K` for a community's. Every path below is this
/// base plus the same suffix, so the marginal names are lra's as ever.
fn table_base(out: &str, link_community: Option<i64>) -> String {
    match link_community {
        Some(k) => format!("{out}.lra.link-community-{k}"),
        None => format!("{out}.lra"),
    }
}

/// The effect and curve tables of a run with `--only only` (empty for all
/// pairs) and `--link-community` (`None` for the marginal run):
/// `{out}.lra[.link-community-K][.only-TAG].parquet` and its curves.
pub fn table_and_curve_paths(
    out: &str,
    only: &[Box<str>],
    link_community: Option<i64>,
) -> (String, String) {
    let base = table_base(out, link_community);
    let table = if only.is_empty() {
        format!("{base}.parquet")
    } else {
        format!("{base}.only-{}.parquet", super::only::tag(only))
    };
    let curve = super::output::curve_beside(&table);
    (table, curve)
}

/// What a table's `q_scope` column says: corrected over every pair, within
/// the `--only` subset (`n_pairs` of `n_all`), within the link community, or
/// within both.
pub fn q_scope(
    only: &[Box<str>],
    n_pairs: usize,
    n_all: usize,
    link_community: Option<i64>,
) -> String {
    match (link_community, only.is_empty()) {
        (None, true) => super::output::Q_SCOPE_ALL.into(),
        (None, false) => super::only::q_scope(n_pairs, n_all),
        (Some(k), true) => format!("co-expression lfdr fitted within link community {k}"),
        (Some(k), false) => format!(
            "co-expression lfdr fitted within link community {k} and the --only subset ({n_pairs} of {n_all} pairs)"
        ),
    }
}

/// A shard's file: `{out}.lra[.link-community-K].shard-I-of-N.cbor`, so the
/// shards of different communities, or of the marginal run, never collide.
pub fn shard_path(out: &str, shard: super::shard::Shard, link_community: Option<i64>) -> String {
    shard_path_named(out, shard.index, shard.count, link_community)
}

/// [`shard_path`] with the shard and count as given: numbers, or the
/// `{shard}` and `{shards}` placeholders of a cluster spec.
pub fn shard_path_named(
    out: &str,
    index: impl std::fmt::Display,
    count: impl std::fmt::Display,
    link_community: Option<i64>,
) -> String {
    format!(
        "{}.shard-{index}-of-{count}.cbor",
        table_base(out, link_community)
    )
}

/// `pinto lra merge`'s record of the shards it put together, beside the
/// tables it wrote: `{out}.lra[.link-community-K].merge.json`.
pub fn merge_record_path(out: &str, link_community: Option<i64>) -> String {
    format!("{}.merge.json", table_base(out, link_community))
}

/// The link community a lra file's name says it is of: `Some(K)` for
/// `….lra.link-community-K.<rest>`, `None` for any other name (the marginal
/// run's, or not a lra file).
pub fn link_community_in_name(name: &str) -> Option<i64> {
    let (_, rest) = name.rsplit_once(".lra.link-community-")?;
    let digits = rest.split('.').next()?;
    digits.parse().ok()
}

/// The cross-fitting folds with every row outside `community` left out, so
/// only its receivers train and are predicted; an error when a fold keeps
/// none of them.
pub fn restrict_folds_to_community(
    folds: super::crossfit::CrossFit,
    community: &CommunityScope,
) -> anyhow::Result<super::crossfit::CrossFit> {
    let outside: Vec<usize> = (0..community.in_community.len())
        .filter(|&row| !community.in_community[row])
        .collect();
    let restricted = folds.leaving_out(&outside);
    anyhow::ensure!(
        restricted.test.iter().all(|rows| !rows.is_empty()),
        "link community {}: a cross-fitting fold holds none of its {} receivers",
        community.link_community,
        community.receiver_rows.len()
    );
    Ok(restricted)
}

/// `--link-community K`'s community and its links among the run's contacts;
/// `None` for the marginal run.
pub fn community_pairs_of_run(
    inp: &super::RunInputs<'_>,
) -> anyhow::Result<Option<LinkCommunityInputs>> {
    let Some(link_community) = inp.settings.link_community else {
        return Ok(None);
    };
    let community_of_pair = inp.link_community_of_pair.ok_or_else(|| {
        anyhow::anyhow!("--link-community {link_community}: the contacts carry no link communities")
    })?;
    let pairs = pairs_of_link_community(inp.pairs, community_of_pair, link_community);
    anyhow::ensure!(
        !pairs.is_empty(),
        "--link-community {link_community}: no contact of the run is in that community \
         (`pinto lra link-communities` lists them)"
    );
    Ok(Some(LinkCommunityInputs {
        link_community,
        pairs,
        min_receivers: inp.settings.min_receivers,
    }))
}

/// `pinto lra link-communities`: the `--from` run's link communities over its
/// spatial contacts, one per line (community, links, receivers, whether
/// `--min-receivers` lets `--link-community` fit it), in community order, so
/// line I of the body is job I of an array.
pub fn list_link_communities(
    args: &crate::lr_activity::args::LraLinkCommunitiesArgs,
) -> anyhow::Result<()> {
    let run = crate::util::metadata::Run::open(&args.from)?;
    let path = run.required_recorded_output(
        run.meta.outputs.link_community.as_deref(),
        "link_community.parquet",
    )?;
    // its spatial pairs and their communities; a negative id is no community
    let (all_pairs, all_communities, _) = crate::util::parquet_io::read_link_community(&path)?;
    let (pairs, communities): (Vec<_>, Vec<i64>) = all_pairs
        .into_iter()
        .zip(all_communities)
        .filter(|&(_, community)| community >= 0)
        .unzip();
    println!("link_community\tlinks\treceivers\tfitted");
    for size in link_community_sizes(&pairs, &communities) {
        println!(
            "{}\t{}\t{}\t{}",
            size.link_community,
            size.links,
            size.receivers,
            if size.receivers >= args.min_receivers {
                "yes"
            } else {
                "no"
            }
        );
    }
    Ok(())
}

/// The table `pinto lra merge` on `inputs` writes under `out`: the collected
/// table for effect tables; for shards, the table of the run they are of, its
/// link community read from the shard files' names
/// (`….lra.link-community-K.shard-I-of-N.cbor`), so known before they exist.
#[cfg(feature = "view")]
pub fn merge_output_of_inputs(inputs: &[String], out: &str) -> String {
    if super::output::collect::are_tables(inputs) {
        return super::output::collect::by_link_community_path(out);
    }
    let link_community = inputs
        .first()
        .and_then(|shard| link_community_in_name(shard));
    table_and_curve_paths(out, &[], link_community).0
}
