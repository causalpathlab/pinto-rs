use crate::view::thousands;

#[test]
fn thousands_groups_digits() {
    assert_eq!(thousands(0), "0");
    assert_eq!(thousands(999), "999");
    assert_eq!(thousands(1000), "1,000");
    assert_eq!(thousands(814243), "814,243");
    assert_eq!(thousands(1234567), "1,234,567");
}
