use super::*;

fn part(pairs: &[(&str, i64)]) -> Partition {
    Partition {
        cells: pairs.iter().map(|(c, _)| (*c).into()).collect(),
        labels: pairs.iter().map(|&(_, l)| l).collect(),
    }
}

fn side() -> Side {
    Side {
        input: String::new(),
        propensity: String::new(),
        level: "final".into(),
        n_cells: 0,
        n_communities: 0,
    }
}

/// Cells named c0..c{n-1} with the given labels.
fn labelled(labels: &[i64]) -> Partition {
    Partition {
        cells: (0..labels.len()).map(|i| format!("c{i}").into()).collect(),
        labels: labels.to_vec(),
    }
}

#[test]
fn identical_partitions_score_one_whatever_the_names() {
    let a = labelled(&[0, 0, 1, 1, 2, 2]);
    let b = labelled(&[5, 5, 3, 3, 9, 9]);
    let t = Table::new(&a, &b).unwrap();
    assert!((ari(&t) - 1.0).abs() < 1e-12);
    assert!((nmi(&t) - 1.0).abs() < 1e-12);
    let r = compare(&t, side(), side(), 0.2, String::new());
    assert_eq!(r.matching.len(), 3);
    assert!(r.matching.iter().all(|m| (m.jaccard - 1.0).abs() < 1e-12));
    let pairs: Vec<(&str, &str)> = r
        .matching
        .iter()
        .map(|m| (m.a.as_str(), m.b.as_str()))
        .collect();
    assert_eq!(pairs, vec![("C0", "C5"), ("C1", "C3"), ("C2", "C9")]);
    assert!(r.vanished.is_empty() && r.split.is_empty() && r.merged.is_empty());
    assert!(r.communities_a.iter().all(|f| f.status == "kept"));
}

#[test]
fn ari_matches_a_worked_example() {
    // scikit-learn: adjusted_rand_score([0,0,0,1,1,1], [0,0,1,1,2,2]) = 0.24242...
    // normalized_mutual_info_score(same) = 0.51580374...
    let a = labelled(&[0, 0, 0, 1, 1, 1]);
    let b = labelled(&[0, 0, 1, 1, 2, 2]);
    let t = Table::new(&a, &b).unwrap();
    assert!((ari(&t) - 0.242_424_242).abs() < 1e-6, "{}", ari(&t));
    assert!((nmi(&t) - 0.515_803_743).abs() < 1e-8, "{}", nmi(&t));
}

#[test]
fn split_merge_vanish_and_new_are_named() {
    // A0 (4 cells) splits into B0 and B1; A1 and A2 merge into B2;
    // A3's 5 cells scatter over B3..B7, so at tau 0.3 it has no partner and
    // those B communities each hold only a scrap of it.
    let mut labels_a = vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 2];
    let mut labels_b = vec![0, 0, 1, 1, 2, 2, 2, 2, 2, 2];
    labels_a.extend([3, 3, 3, 3, 3]);
    labels_b.extend([3, 4, 5, 6, 7]);
    // Give B3..B7 cells of their own, from an A community too small to claim them.
    for (k, b) in (3..8).enumerate() {
        for _ in 0..3 {
            labels_a.push(10 + k as i64);
            labels_b.push(b);
        }
    }
    let t = Table::new(&labelled(&labels_a), &labelled(&labels_b)).unwrap();
    let r = compare(&t, side(), side(), 0.3, String::new());
    let status = |c: &str| {
        r.communities_a
            .iter()
            .find(|f| f.community == c)
            .unwrap()
            .status
    };
    assert_eq!(status("C0"), "split");
    assert_eq!(status("C1"), "merged");
    assert_eq!(status("C2"), "merged");
    assert_eq!(status("C3"), "vanished");
    assert_eq!(r.split.len(), 1);
    assert_eq!(r.split[0].into, vec!["C0", "C1"]);
    assert_eq!(r.merged.len(), 1);
    assert_eq!(r.merged[0].b, "C2");
    assert_eq!(r.vanished, vec!["C3"]);
    let b0 = r
        .communities_b
        .iter()
        .find(|f| f.community == "C0")
        .unwrap();
    assert_eq!(b0.status, "split_part");
}

#[test]
fn a_community_the_other_run_lacks_is_new() {
    let a = labelled(&[0, 0, 0, 0]);
    let b = labelled(&[0, 0, 0, 1]);
    let t = Table::new(&a, &b).unwrap();
    // B1 holds 1/4 of A0, below tau 0.3, but all of B1 is A0.
    let r = compare(&t, side(), side(), 0.3, String::new());
    assert!(r.new.is_empty());
    let b1 = r
        .communities_b
        .iter()
        .find(|f| f.community == "C1")
        .unwrap();
    assert_eq!(b1.status, "kept");
    // With B1 made of cells A did not place, it has no partner.
    let a = part(&[("x", 0), ("y", 0), ("z", -1)]);
    let b = part(&[("x", 0), ("y", 1), ("z", 1)]);
    let t = Table::new(&a, &b).unwrap();
    assert_eq!(t.n_only_b, 1);
    assert_eq!(t.total(), 2);
}

#[test]
fn only_shared_cells_count() {
    let a = part(&[("x", 0), ("y", 0), ("z", 1)]);
    let b = part(&[("y", 0), ("z", 1), ("w", 1)]);
    let t = Table::new(&a, &b).unwrap();
    assert_eq!(t.total(), 2);
    assert_eq!((t.n_only_a, t.n_only_b), (1, 1));
}

#[test]
fn runs_with_no_common_cell_are_refused() {
    let a = part(&[("x", 0)]);
    let b = part(&[("y", 0)]);
    assert!(Table::new(&a, &b).is_err());
}

#[test]
fn hungarian_finds_the_cheapest_assignment() {
    let cost = vec![
        vec![4.0, 1.0, 3.0],
        vec![2.0, 0.0, 5.0],
        vec![3.0, 2.0, 2.0],
    ];
    assert_eq!(hungarian(&cost), vec![1, 0, 2]);
    // Rectangular: two rows, three columns.
    let cost = vec![vec![9.0, 1.0, 9.0], vec![9.0, 2.0, 0.5]];
    assert_eq!(hungarian(&cost), vec![1, 2]);
}

#[test]
fn more_a_communities_than_b_leaves_some_unmatched() {
    let a = labelled(&[0, 0, 1, 1, 2, 2]);
    let b = labelled(&[0, 0, 0, 0, 1, 1]);
    let t = Table::new(&a, &b).unwrap();
    let pairs = matching(&t);
    assert_eq!(pairs.len(), 2);
    assert!(pairs.contains(&(2, 1)));
}

#[test]
fn a_level_is_found_by_tag_or_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run");
    let prefix = prefix.to_str().unwrap();
    for f in ["propensity", "draft.propensity", "L2.propensity"] {
        std::fs::write(format!("{prefix}.{f}.parquet"), b"").unwrap();
    }
    let p = propensity_path(prefix, "final").unwrap();
    assert!(p.ends_with("run.propensity.parquet"));
    let p = propensity_path(&format!("{prefix}.pinto.json"), "draft").unwrap();
    assert!(p.ends_with("run.draft.propensity.parquet"));
    assert!(propensity_path(prefix, "L2").is_ok());
    assert!(propensity_path(prefix, "L7").is_err());
    let table = format!("{prefix}.propensity.parquet");
    assert!(propensity_path(&table, "final").is_ok());
    assert!(propensity_path(&table, "draft").is_err());
}
