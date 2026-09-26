//! Kitty graphics protocol: show a frame as an image placed over a block of
//! terminal cells.
//!
//! Frames alternate between two image ids. The new frame is placed first and
//! the old one deleted after, so the map never flashes empty between frames.
//!
//! Locally the pixels travel through a temporary file that kitty reads and
//! deletes (`t=t`), so a full-screen frame costs a file write, not tens of MB
//! of escape codes. Over ssh the terminal cannot read our files, so pixels are
//! sent inline, zlib-compressed (`o=z`) and base64 encoded in 4 KB chunks.
//! Every command carries `q=2` so kitty never answers into our input stream.
//!
//! Inside tmux, each command is wrapped in tmux's passthrough (`ESC P tmux;
//! … ESC \`, inner escapes doubled), which tmux forwards to the terminal when
//! its `allow-passthrough` option is on. tmux does not know about the image,
//! so this is best effort; the viewer only uses it when asked for kitty
//! explicitly.

use super::render::Frame;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use std::io::Write;

/// Image ids, alternated frame to frame.
pub(super) const IDS: [u32; 2] = [0x7069_0001, 0x7069_0002];

/// Base64 payload bytes per escape sequence (protocol limit is 4096).
const CHUNK: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transport {
    /// Through a temporary file; the terminal must share our filesystem.
    File,
    /// Inline, compressed.
    Direct,
}

impl Transport {
    /// Direct over ssh, file otherwise.
    pub fn detect() -> Self {
        let remote = ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"]
            .iter()
            .any(|v| std::env::var_os(v).is_some());
        if remote {
            Transport::Direct
        } else {
            Transport::File
        }
    }
}

pub struct Kitty {
    transport: Transport,
    /// Wrap commands for tmux's passthrough.
    tmux: bool,
    /// Index into `IDS` of the image on screen.
    shown: Option<usize>,
    /// Bytes written for the last frame.
    pub last_bytes: usize,
}

impl Kitty {
    pub fn new(transport: Transport) -> Self {
        Kitty {
            transport,
            tmux: false,
            shown: None,
            last_bytes: 0,
        }
    }

    pub fn transport(&self) -> Transport {
        self.transport
    }

    /// Wrap commands for tmux's passthrough, or not.
    pub fn through_tmux(mut self, on: bool) -> Self {
        self.tmux = on;
        self
    }

    /// Whether commands go through tmux's passthrough.
    pub fn via_tmux(&self) -> bool {
        self.tmux
    }

    /// Write one graphics command, `ESC _ G {body} ESC \`.
    fn command(&self, out: &mut impl Write, body: &[u8]) -> std::io::Result<()> {
        if self.tmux {
            out.write_all(b"\x1bPtmux;\x1b\x1b_G")?;
            // Escapes inside the passthrough are doubled.
            for &b in body {
                if b == 0x1b {
                    out.write_all(b"\x1b\x1b")?;
                } else {
                    out.write_all(&[b])?;
                }
            }
            out.write_all(b"\x1b\x1b\\\x1b\\")
        } else {
            out.write_all(b"\x1b_G")?;
            out.write_all(body)?;
            out.write_all(b"\x1b\\")
        }
    }

    /// Place `frame` with its top-left at cell `(col, row)`, scaled to
    /// `cols × rows` cells, replacing the previous frame.
    pub fn show(
        &mut self,
        out: &mut impl Write,
        frame: &Frame,
        (col, row): (u16, u16),
        (cols, rows): (u16, u16),
    ) -> anyhow::Result<()> {
        let slot = self.shown.map_or(0, |s| 1 - s);
        let id = IDS[slot];
        // 1-based cursor position; kitty places the image at the cursor.
        write!(out, "\x1b7\x1b[{};{}H", row + 1, col + 1)?;
        let keys = format!(
            "a=T,f=32,s={},v={},i={id},p=1,c={cols},r={rows},C=1,q=2",
            frame.w, frame.h
        );
        self.last_bytes = match self.transport {
            Transport::File => {
                let path = temp_path(slot);
                std::fs::write(&path, &frame.rgba)?;
                let payload = B64.encode(path.to_string_lossy().as_bytes());
                self.command(out, format!("{keys},t=t;{payload}").as_bytes())?;
                frame.rgba.len()
            }
            Transport::Direct => {
                let mut z = flate2::write::ZlibEncoder::new(
                    Vec::with_capacity(frame.rgba.len() / 4),
                    flate2::Compression::fast(),
                );
                z.write_all(&frame.rgba)?;
                let payload = B64.encode(z.finish()?);
                let chunks: Vec<&[u8]> = payload.as_bytes().chunks(CHUNK).collect();
                for (i, chunk) in chunks.iter().enumerate() {
                    let more = u8::from(i + 1 < chunks.len());
                    let head = if i == 0 {
                        format!("{keys},t=d,o=z,m={more};")
                    } else {
                        format!("m={more};")
                    };
                    self.command(out, &[head.as_bytes(), chunk].concat())?;
                }
                payload.len()
            }
        };
        if let Some(old) = self.shown {
            self.command(out, format!("a=d,d=I,i={},q=2", IDS[old]).as_bytes())?;
        }
        // Restore the cursor saved above.
        write!(out, "\x1b8")?;
        out.flush()?;
        self.shown = Some(slot);
        Ok(())
    }

    /// Remove our images from the screen and free them.
    pub fn clear(&mut self, out: &mut impl Write) -> std::io::Result<()> {
        for id in IDS {
            self.command(out, format!("a=d,d=I,i={id},q=2").as_bytes())?;
        }
        self.shown = None;
        out.flush()?;
        // Kitty deletes each file once read; one it never read (a terminal
        // that ignored the command) would otherwise stay behind.
        for slot in 0..IDS.len() {
            let _ = std::fs::remove_file(temp_path(slot));
        }
        Ok(())
    }
}

/// Running inside tmux.
pub fn in_tmux() -> bool {
    std::env::var_os("TMUX").is_some()
}

/// Kitty only deletes files whose name says they are for it.
fn temp_path(slot: usize) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "tty-graphics-protocol-pinto-{}-{slot}.rgba",
        std::process::id()
    ))
}
