use crate::util::common::{pool_threads, DEFAULT_MAX_THREADS};

#[test]
fn the_pool_is_rayon_s_own_count_capped_at_max_threads() {
    let cap = DEFAULT_MAX_THREADS;
    // unset, empty or not a count: the cores, at most the cap
    assert_eq!(pool_threads(cap, None, 64), cap);
    assert_eq!(pool_threads(cap, Some(""), 64), cap);
    assert_eq!(pool_threads(cap, Some("0"), 64), cap);
    assert_eq!(pool_threads(cap, None, 8), 8);
    // RAYON_NUM_THREADS set: its count, at most the cap
    assert_eq!(pool_threads(cap, Some("4"), 64), 4);
    assert_eq!(pool_threads(cap, Some("48"), 64), cap);
    // a raised cap lets either through
    assert_eq!(pool_threads(64, Some("48"), 8), 48);
    assert_eq!(pool_threads(64, None, 32), 32);
    assert_eq!(pool_threads(1, None, 0), 1);
}
