use std::{
    os::fd::{AsFd, OwnedFd},
    sync::{Arc, Condvar, Mutex},
};

use rustix::event::{PollFd, PollFlags};

use super::WgpuError;
use crate::backend::renderer::sync::{Fence, Interrupted, SyncPoint};

#[derive(Debug)]
struct FenceState {
    signaled: Mutex<bool>,
    changed: Condvar,
}

#[derive(Debug)]
struct NativeFence {
    fd: OwnedFd,
}

#[derive(Debug, Clone)]
enum FenceKind {
    Native(Arc<NativeFence>),
    Callback {
        state: Arc<FenceState>,
        device: ::wgpu::Device,
    },
}

/// Fence signaled after all WGPU work submitted before its creation has completed.
#[derive(Debug, Clone)]
pub struct WgpuFence {
    kind: FenceKind,
}

impl WgpuFence {
    pub(super) fn after_submission(queue: &::wgpu::Queue, device: &::wgpu::Device) -> SyncPoint {
        match super::vulkan::export_queue_sync_file(device, queue) {
            Ok(Some(fd)) => {
                return WgpuFence {
                    kind: FenceKind::Native(Arc::new(NativeFence { fd })),
                }
                .into();
            }
            Ok(None) => return SyncPoint::signaled(),
            Err(err) => tracing::debug!(?err, "failed to export WGPU native fence"),
        }

        let state = Arc::new(FenceState {
            signaled: Mutex::new(false),
            changed: Condvar::new(),
        });
        let callback_state = state.clone();
        queue.on_submitted_work_done(move || {
            *callback_state.signaled.lock().unwrap() = true;
            callback_state.changed.notify_all();
        });

        WgpuFence {
            kind: FenceKind::Callback {
                state,
                device: device.clone(),
            },
        }
        .into()
    }
}

impl Fence for WgpuFence {
    fn is_signaled(&self) -> bool {
        match &self.kind {
            FenceKind::Native(fence) => poll_native(fence.fd.as_fd(), false).unwrap_or(false),
            FenceKind::Callback { state, device } => {
                if *state.signaled.lock().unwrap() {
                    return true;
                }
                let _ = device.poll(::wgpu::PollType::Poll);
                *state.signaled.lock().unwrap()
            }
        }
    }

    fn wait(&self) -> Result<(), Interrupted> {
        match &self.kind {
            FenceKind::Native(fence) => poll_native(fence.fd.as_fd(), true).map(|_| ()),
            FenceKind::Callback { state, device } => {
                if *state.signaled.lock().unwrap() {
                    return Ok(());
                }
                device
                    .poll(::wgpu::PollType::wait_indefinitely())
                    .map_err(|_| Interrupted)?;

                let mut signaled = state.signaled.lock().unwrap();
                while !*signaled {
                    signaled = state.changed.wait(signaled).map_err(|_| Interrupted)?;
                }
                Ok(())
            }
        }
    }

    fn is_exportable(&self) -> bool {
        matches!(self.kind, FenceKind::Native(_))
    }

    fn export(&self) -> Option<OwnedFd> {
        match &self.kind {
            FenceKind::Native(fence) => fence.fd.try_clone().ok(),
            FenceKind::Callback { .. } => None,
        }
    }
}

fn poll_native(fd: std::os::fd::BorrowedFd<'_>, block: bool) -> Result<bool, Interrupted> {
    let immediate = rustix::time::Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    loop {
        let mut poll_fd = [PollFd::new(&fd, PollFlags::IN)];
        let timeout = (!block).then_some(&immediate);
        match rustix::event::poll(&mut poll_fd, timeout) {
            Ok(0) => return Ok(false),
            Ok(_) => {
                let ready = poll_fd[0].revents();
                return if ready.contains(PollFlags::IN) {
                    Ok(true)
                } else {
                    Err(Interrupted)
                };
            }
            Err(rustix::io::Errno::INTR) => continue,
            Err(_) => return Err(Interrupted),
        }
    }
}

pub(super) fn wait(sync: &SyncPoint) -> Result<(), WgpuError> {
    sync.wait().map_err(|_| WgpuError::SyncInterrupted)
}
