use std::{
    os::fd::OwnedFd,
    sync::{Arc, Condvar, Mutex},
};

use super::WgpuError;
use crate::backend::renderer::sync::{Fence, Interrupted, SyncPoint};

#[derive(Debug)]
struct FenceState {
    signaled: Mutex<bool>,
    changed: Condvar,
}

/// Fence signaled after all WGPU work submitted before its creation has completed.
#[derive(Debug, Clone)]
pub struct WgpuFence {
    state: Arc<FenceState>,
    device: ::wgpu::Device,
}

impl WgpuFence {
    pub(super) fn after_submission(queue: &::wgpu::Queue, device: &::wgpu::Device) -> SyncPoint {
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
            state,
            device: device.clone(),
        }
        .into()
    }
}

impl Fence for WgpuFence {
    fn is_signaled(&self) -> bool {
        if *self.state.signaled.lock().unwrap() {
            return true;
        }
        let _ = self.device.poll(::wgpu::PollType::Poll);
        *self.state.signaled.lock().unwrap()
    }

    fn wait(&self) -> Result<(), Interrupted> {
        if *self.state.signaled.lock().unwrap() {
            return Ok(());
        }
        self.device
            .poll(::wgpu::PollType::wait_indefinitely())
            .map_err(|_| Interrupted)?;

        let mut signaled = self.state.signaled.lock().unwrap();
        while !*signaled {
            signaled = self.state.changed.wait(signaled).map_err(|_| Interrupted)?;
        }
        Ok(())
    }

    fn is_exportable(&self) -> bool {
        false
    }

    fn export(&self) -> Option<OwnedFd> {
        None
    }
}

pub(super) fn wait(sync: &SyncPoint) -> Result<(), WgpuError> {
    sync.wait().map_err(|_| WgpuError::SyncInterrupted)
}
