use crate::view::thousands;

#[test]
fn thousands_groups_digits() {
    assert_eq!(thousands(0), "0");
    assert_eq!(thousands(999), "999");
    assert_eq!(thousands(1000), "1,000");
    assert_eq!(thousands(814243), "814,243");
    assert_eq!(thousands(1234567), "1,234,567");
}

/// A round's clusters are focused by `K3` or by lupin's bare `3`; a
/// level's communities by `C3` or `3`.
#[test]
fn focus_takes_a_rounds_bare_cluster_ids() {
    use super::fixture::{synth_cells, synth_communities};
    use crate::view::data::{Communities, Geometry};
    use crate::view::parse_focus;
    fn on(given: &[&str], comm: &Communities) -> Result<Vec<bool>, String> {
        let given: Vec<Box<str>> = given.iter().map(|&s| s.into()).collect();
        // Named groups here are a round's.
        parse_focus(&given, comm, comm.names.is_some()).map_err(|e| e.to_string())
    }
    let geom = Geometry::from_cells(synth_cells(16, 1));
    let mut comm = synth_communities(&geom, 3);
    assert_eq!(on(&["C2", "0"], &comm), Ok(vec![true, false, true]));

    comm.names = Some(vec!["K4 T".into(), "K7 B".into(), "K9 –".into()]);
    comm.ids = Some(vec!["K4".into(), "K7".into(), "K9".into()]);
    assert_eq!(on(&["K7"], &comm), Ok(vec![false, true, false]));
    assert_eq!(on(&["9", "4"], &comm), Ok(vec![true, false, true]));
    assert!(on(&["2"], &comm).unwrap_err().contains("K2"));
}

/// A level whose communities the manifest names is focused by a name,
/// and still by `C3` or `3`; a group it lacks is named against the level.
#[test]
fn focus_takes_a_named_levels_names_and_community_ids() {
    use super::fixture::{synth_cells, synth_communities};
    use crate::view::data::{Communities, Geometry};
    use crate::view::parse_focus;
    fn on(given: &[&str], comm: &Communities, round: bool) -> Result<Vec<bool>, String> {
        let given: Vec<Box<str>> = given.iter().map(|&s| s.into()).collect();
        parse_focus(&given, comm, round).map_err(|e| e.to_string())
    }
    let geom = Geometry::from_cells(synth_cells(16, 1));
    let mut comm = synth_communities(&geom, 3);
    comm.names = Some(vec!["D1".into(), "D2".into(), "D3".into()]);
    assert_eq!(on(&["D2"], &comm, false), Ok(vec![false, true, false]));
    assert_eq!(on(&["C0", "2"], &comm, false), Ok(vec![true, false, true]));
    let e = on(&["D9"], &comm, false).unwrap_err();
    assert!(e.contains("this level") && !e.contains("round"), "{e}");
    assert!(on(&["C3"], &comm, false).unwrap_err().contains("K=3"));
    // A round's groups are its names only.
    let e = on(&["C0"], &comm, true).unwrap_err();
    assert!(e.contains("this round"), "{e}");
}
