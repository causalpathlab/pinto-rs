use crate::util::common::default_threads;

#[test]
fn threads_are_capped_unless_the_environment_says() {
    // unset or empty: at most 16, fewer on a smaller machine
    assert_eq!(default_threads(None, 64), Some(16));
    assert_eq!(default_threads(Some(""), 64), Some(16));
    assert_eq!(default_threads(None, 8), Some(8));
    // RAYON_NUM_THREADS set: rayon's own choice stands
    assert_eq!(default_threads(Some("48"), 64), None);
}
