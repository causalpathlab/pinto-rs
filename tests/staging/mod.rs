//! What `--atomic-outputs` must leave behind: no staging folder, and no
//! record naming one.

use std::path::Path;

/// Every name in `dir` and below that is a staging folder or names one.
pub fn staged_anywhere(dir: &Path) -> Vec<String> {
    let mut found = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.to_string_lossy().contains(".partial-") {
            found.push(p.display().to_string());
        }
        if p.is_dir() {
            found.extend(staged_anywhere(&p));
        } else if p.extension().is_some_and(|x| x == "json")
            && std::fs::read_to_string(&p).unwrap().contains(".partial-")
        {
            found.push(format!("{} names a staged path", p.display()));
        }
    }
    found
}

/// That nothing in `dir` or below is a staging folder or names one.
pub fn assert_none_staged(dir: &Path) {
    let found = staged_anywhere(dir);
    assert!(found.is_empty(), "{found:?}");
}
