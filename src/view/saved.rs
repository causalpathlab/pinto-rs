//! What the viewer has saved: a log of each export and a small thumbnail
//! of it, kept in `.pinto-view/` in the directory the viewer runs from, so
//! the list lasts from one session to the next. The runs' own directories
//! are not touched.

use super::color;
use super::render::Frame;
use legume_numeric::matrix::common_io::mkdir;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Width of a thumbnail on disk, in pixels.
pub const THUMB_WIDTH: usize = 240;
/// Saves remembered; older ones go, with their thumbnails.
const KEEP: usize = 200;

/// One saved file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    /// The file, as an absolute path.
    pub path: PathBuf,
    /// What was saved: `map · final · argmax`, `structure · cell types`.
    pub what: String,
    /// When, in seconds since the Unix epoch.
    pub when: u64,
    /// The thumbnail's file name in `thumbs/`.
    thumb: String,
}

impl Entry {
    pub fn name(&self) -> String {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

/// Every save remembered in one directory, newest first.
pub struct Gallery {
    dir: PathBuf,
    entries: Vec<Entry>,
}

impl Gallery {
    /// The saves logged in `dir`, less those whose file or thumbnail is
    /// gone; thumbnails nothing refers to any more are deleted. An
    /// unreadable log starts an empty gallery.
    pub fn open(dir: &Path) -> Self {
        let thumbs = dir.join("thumbs");
        let logged: Vec<Entry> = std::fs::read_to_string(dir.join("saved.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let entries: Vec<Entry> = logged
            .into_iter()
            .filter(|e| e.path.exists() && thumbs.join(&e.thumb).exists())
            .collect();
        if let Ok(listing) = std::fs::read_dir(&thumbs) {
            let kept: std::collections::HashSet<&str> =
                entries.iter().map(|e| e.thumb.as_str()).collect();
            for f in listing.filter_map(Result::ok) {
                if !kept.contains(f.file_name().to_string_lossy().as_ref()) {
                    let _ = std::fs::remove_file(f.path());
                }
            }
        }
        Gallery {
            dir: dir.into(),
            entries,
        }
    }

    /// The gallery of the directory the viewer runs in: `.pinto-view/`.
    pub fn here() -> Self {
        Self::open(Path::new(".pinto-view"))
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn thumb(&self, e: &Entry) -> PathBuf {
        self.dir.join("thumbs").join(&e.thumb)
    }

    /// Remember that `path` was just saved, showing `picture`: at the top,
    /// replacing an earlier save of the same file.
    pub fn add(&mut self, path: &Path, what: &str, picture: &Frame) -> anyhow::Result<()> {
        let path = path.canonicalize()?;
        // Another viewer in this directory may have saved since: build on
        // the log as it is on disk, not on what this one read at start.
        self.entries = Self::open(&self.dir).entries;
        if let Some(i) = self.entries.iter().position(|e| e.path == path) {
            let old = self.entries.remove(i);
            let _ = std::fs::remove_file(self.thumb(&old));
        }
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let thumbs = self.dir.join("thumbs");
        mkdir(&thumbs.to_string_lossy())?;
        let thumb = format!("{}.png", now.as_nanos());
        shrink(picture, THUMB_WIDTH).write_png(&thumbs.join(&thumb))?;
        self.entries.insert(
            0,
            Entry {
                path,
                what: what.into(),
                when: now.as_secs(),
                thumb,
            },
        );
        for old in self.entries.split_off(self.entries.len().min(KEEP)) {
            let _ = std::fs::remove_file(self.thumb(&old));
        }
        std::fs::write(
            self.dir.join("saved.json"),
            serde_json::to_string_pretty(&self.entries)?,
        )?;
        Ok(())
    }
}

/// `frame` shrunk to `width` pixels wide at its aspect, each pixel the
/// mean in linear light of the ones it covers, as the map's average mode
/// mixes colours (never enlarged).
pub fn shrink(frame: &Frame, width: usize) -> Frame {
    let w = width.min(frame.w).max(1);
    let h = ((frame.h * w) as f32 / frame.w.max(1) as f32)
        .round()
        .max(1.) as usize;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        let (y0, y1) = (
            y * frame.h / h,
            ((y + 1) * frame.h / h).max(y * frame.h / h + 1),
        );
        for x in 0..w {
            let (x0, x1) = (
                x * frame.w / w,
                ((x + 1) * frame.w / w).max(x * frame.w / w + 1),
            );
            let mut sum = [0f32; 3];
            let mut n = 0f32;
            for yy in y0..y1.min(frame.h) {
                for xx in x0..x1.min(frame.w) {
                    let o = frame.offset(xx, yy);
                    for (s, &v) in sum.iter_mut().zip(&frame.rgba[o..o + 3]) {
                        *s += color::linear(v);
                    }
                    n += 1.;
                }
            }
            let [r, g, b] = sum.map(|s| color::encode_fast(s / n.max(1.)));
            rgba.extend([r, g, b, 255]);
        }
    }
    Frame {
        w,
        h,
        rgba,
        background: frame.background,
    }
}

/// A thumbnail written by [`Gallery::add`], read back.
pub fn read_thumb(path: &Path, background: [u8; 3]) -> Option<Frame> {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).ok()?));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return None;
    }
    buf.truncate(info.buffer_size());
    Some(Frame {
        w: info.width as usize,
        h: info.height as usize,
        rgba: buf,
        background,
    })
}

/// How long ago `when` was, at `now` (both seconds since the epoch).
pub fn ago(when: u64, now: u64) -> String {
    match now.saturating_sub(when) {
        s if s < 60 => "just now".into(),
        s if s < 3600 => format!("{} min ago", s / 60),
        s if s < 86_400 => format!("{} h ago", s / 3600),
        s => format!("{} d ago", s / 86_400),
    }
}
