//! An empty ligand-effect reference, for the tests that run lra offline.

use std::path::{Path, PathBuf};

/// The folder a built reference is cached in, under `built`: the build
/// revision and the licence tag, as `src/lr_activity/ligand_effect/
/// reference/mod.rs` names it. A change there is made here too.
const BUILT: &str = "r2-all";

/// A ligand-effect reference cache at `{dir}/reference` with no pairs and
/// no programs, so lra runs offline on the pairs and programs a test
/// gives it. The cache's folder, for `PINTO_REFERENCE_DIR`.
pub fn empty_reference(dir: &Path) -> PathBuf {
    let root = dir.join("reference");
    let built = root.join("built").join(BUILT);
    std::fs::create_dir_all(&built).unwrap();
    std::fs::create_dir_all(root.join("downloads")).unwrap();
    for tag in ["human", "mouse"] {
        std::fs::write(
            built.join(format!("lr_pairs.{tag}.tsv")),
            "ligand\treceptor\n",
        )
        .unwrap();
        std::fs::write(
            built.join(format!("receptor_targets.{tag}.tsv")),
            "receptor\ttarget\tweight\n",
        )
        .unwrap();
    }
    std::fs::write(built.join("reference.json"), "{}").unwrap();
    root
}
