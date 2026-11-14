//! Paths as pinto records and shows them.

use std::path::{Path, PathBuf};

/// `p` from the root, against the working directory, so it reads the same
/// from anywhere; `p` as given when that cannot be done.
pub fn absolute(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

/// [`absolute`] as text: how results show a path and manifests record one.
pub fn shown_path(p: &Path) -> String {
    absolute(p).to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relative_path_is_shown_from_the_root_and_an_absolute_one_as_given() {
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(absolute(Path::new("a/b.zarr")), cwd.join("a/b.zarr"));
        assert_eq!(shown_path(Path::new("/x/y.tsv")), "/x/y.tsv");
        assert_eq!(
            shown_path(Path::new("c.tsv")),
            cwd.join("c.tsv").to_string_lossy()
        );
    }
}
