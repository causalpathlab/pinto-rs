//! Compute-device selection for subcommands that run a candle model.
//!
//! `cage` and the `stagate` baseline use it. It lives here rather than in
//! either command's arguments so each reaches it without a cross-subcommand
//! import.

use clap::ValueEnum;
use std::fs::{File, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

impl ComputeDevice {
    /// The device. For cuda, this first waits until no other pinto run on
    /// the machine is using that GPU. See [`hold_gpu`].
    pub fn to_device(
        &self,
        device_no: usize,
    ) -> anyhow::Result<legume_numeric::candle::candle_core::Device> {
        use legume_numeric::candle::candle_core::Device;
        Ok(match self {
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => {
                hold_gpu(device_no);
                Device::new_cuda(device_no)?
            }
            ComputeDevice::Metal => Device::new_metal(device_no)?,
        })
    }
}

impl std::fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ComputeDevice::Cpu => "cpu",
            ComputeDevice::Cuda => "cuda",
            ComputeDevice::Metal => "metal",
        })
    }
}

/// One pinto run on each GPU at a time. Runs launched in parallel (by
/// agents, a job queue, or by hand) would otherwise share a card's memory
/// and fail, or take the host down with them.
///
/// The run takes an exclusive lock on `cuda<N>.lock` in `$XDG_RUNTIME_DIR/pinto-gpu`
/// (or in the temp directory) and keeps it until the process exits. The
/// kernel releases the lock however the process ends, crashes included, so a
/// stale file never blocks anyone. `PINTO_GPU_LOCK=0` skips the lock. If the
/// lock file can't be made, the run warns and goes ahead without it.
fn hold_gpu(device_no: usize) {
    if std::env::var_os("PINTO_GPU_LOCK").is_some_and(|v| v == "0") {
        return;
    }
    let dir = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("pinto-gpu");
    if let Err(e) = hold_lock(&dir.join(format!("cuda{device_no}.lock"))) {
        log::warn!("running on GPU {device_no} without its lock: {e}");
    }
}

/// The locks this process holds, kept open until it exits. Taking a path
/// again returns at once: a second open of the same file would wait on this
/// process's own lock.
static HELD: Mutex<Vec<(PathBuf, File)>> = Mutex::new(Vec::new());

/// Waits for the exclusive lock on `path` and keeps it for the life of the
/// process.
fn hold_lock(path: &Path) -> std::io::Result<()> {
    let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
    if held.iter().any(|(p, _)| p == path) {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = File::options()
        .create(true)
        .write(true)
        .truncate(false)
        .open(path)?;
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            log::warn!(
                "waiting for another pinto run to finish with this GPU ({}); \
                 PINTO_GPU_LOCK=0 skips the wait",
                path.display()
            );
            // Wait without holding `HELD`, so a wait on one GPU does not
            // hold up this process taking another.
            drop(held);
            file.lock()?;
            held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
        }
        Err(TryLockError::Error(e)) => return Err(e),
    }
    held.push((path.to_path_buf(), file));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_lock_excludes_others_and_retaking_it_returns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu").join("cuda7.lock");
        hold_lock(&path).unwrap();
        // the same process again: no self-deadlock
        hold_lock(&path).unwrap();
        // anyone else opening the file finds it taken
        let other = File::options().write(true).open(&path).unwrap();
        assert!(matches!(other.try_lock(), Err(TryLockError::WouldBlock)));
    }
}
