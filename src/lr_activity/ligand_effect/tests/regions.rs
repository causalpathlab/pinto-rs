//! Region labels: their confounder columns and their scopes.

use crate::lr_activity::ligand_effect::confounders::region_levels;
use crate::lr_activity::ligand_effect::tissue::{scope_rows, section_label, Scope};

/// Within each section every region but the most common gets a column, so
/// a region that two sections share gets one in each.
#[test]
fn region_columns_are_per_section_without_the_commonest() {
    let section = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    let region = [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 0];
    assert_eq!(region_levels(&section, &region), [(0, 1), (1, 0), (1, 2)]);
    // A tie goes to the smaller code as the reference.
    assert_eq!(region_levels(&[0, 0], &[3, 1]), [(0, 3)]);
    // One region everywhere: no column.
    assert!(region_levels(&[0, 1], &[4, 4]).is_empty());
}

#[test]
fn region_scopes_take_their_rows() {
    let scopes = [
        Scope::All,
        Scope::Region(1),
        Scope::Section(0),
        Scope::Region(0),
    ];
    let section = [0, 1, 0, 1];
    let region = [1, 1, 0, 2];
    let rows = scope_rows(&scopes, &section, None, Some(&region));
    assert_eq!(rows, [vec![0, 1, 2, 3], vec![0, 1], vec![0, 2], vec![2]]);
}

#[test]
fn a_section_named_like_a_region_row_is_prefixed() {
    assert_eq!(&*section_label("region:x"), "section:region:x");
    assert_eq!(&*section_label("S1"), "S1");
}
