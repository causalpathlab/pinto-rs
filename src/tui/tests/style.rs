use super::*;

#[test]
fn cuts_keep_the_end_or_the_start_and_say_so() {
    assert_eq!(short("abcdef", 4), "abc…");
    assert_eq!(short("abc", 4), "abc");
    assert_eq!(tail("abcdef", 4), "…def");
    assert_eq!(tail("abc", 4), "abc");
    assert_eq!(tail("abcdef", 4).chars().count(), 4);
}

#[test]
fn the_cursor_row_sits_in_the_middle_of_its_window_where_it_can() {
    assert_eq!(first_row(0, 10, 100), 0);
    assert_eq!(first_row(50, 10, 100), 45);
    assert_eq!(first_row(99, 10, 100), 90);
    assert_eq!(first_row(3, 10, 5), 0);
}

#[test]
fn wrapping_cuts_the_first_piece_and_the_rest_to_their_widths() {
    assert_eq!(wrap("abcdefg", 3, 2), ["abc", "de", "fg"]);
    assert_eq!(wrap("ab", 3, 2), ["ab"]);
    assert!(wrap("", 3, 2).is_empty());
}
