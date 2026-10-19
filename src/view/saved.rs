//! What the viewer has saved: legume-tui's gallery of each export and a
//! small thumbnail of it, kept in `.pinto-view/` in the directory the
//! viewer runs from, so the list lasts from one session to the next. The
//! runs' own directories are not touched.

use super::render::Frame;
use legume_tui::gallery::{self, thumb, Policy};
use std::path::Path;

pub use legume_tui::gallery::{ago, now};

/// Every save remembered in one directory, newest first.
pub type Gallery = gallery::Gallery;

/// Width of a thumbnail on disk, in pixels.
pub const THUMB_WIDTH: usize = gallery::THUMB_WIDTH as usize;

/// The saves logged in `dir`, less those whose file or thumbnail is gone.
/// A thumbnail no save names is deleted once it is a while old (another
/// viewer may be about to log it).
pub fn open(dir: &Path) -> Gallery {
    Gallery::open(dir, Policy::plain())
}

/// The gallery of the directory the viewer runs in: `.pinto-view/`.
pub fn here() -> Gallery {
    open(Path::new(".pinto-view"))
}

/// Remember that `path` was just saved, showing `picture`: at the top,
/// replacing an earlier save of the same file.
pub fn add(g: &mut Gallery, path: &Path, what: &str, picture: &Frame) -> std::io::Result<()> {
    let view = thumb::Picture::new(picture.w as u32, picture.h as u32, &picture.rgba)
        .ok_or_else(|| std::io::Error::other("a frame of the wrong size"))?;
    g.add_view(&[path], what, Some(view), ())
}

/// A thumbnail [`add`] wrote, read back as a frame on `background`.
pub fn read_thumb(path: &Path, background: [u8; 3]) -> Option<Frame> {
    let img = thumb::read_png(path)?;
    Some(Frame {
        w: img.width() as usize,
        h: img.height() as usize,
        rgba: img.into_raw(),
        background,
    })
}
