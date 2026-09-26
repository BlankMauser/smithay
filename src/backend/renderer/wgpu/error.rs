use drm_fourcc::DrmFourcc;
use thiserror::Error;

use crate::backend::SwapBuffersError;

#[cfg(feature = "wayland_frontend")]
use wayland_server::protocol::wl_shm;

/// Error returned by the WGPU renderer.
#[derive(Debug, Error)]
pub enum WgpuError {
    /// Vulkan DMA-BUF interop failed.
    #[cfg(unix)]
    #[error(transparent)]
    Vulkan(#[from] super::vulkan::VulkanError),
    /// The requested pixel format is not supported.
    #[error("Unsupported pixel format: {0:?}")]
    UnsupportedPixelFormat(DrmFourcc),
    /// The requested shared-memory format is not supported.
    #[cfg(feature = "wayland_frontend")]
    #[error("Unsupported wl_shm format: {0:?}")]
    UnsupportedWlPixelFormat(wl_shm::Format),
    /// The supplied buffer is smaller than required by its dimensions and format.
    #[error("Incomplete buffer {actual} < {expected}")]
    IncompleteBuffer {
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length.
        actual: usize,
    },
    /// The requested rectangle lies outside of the texture.
    #[error("Region lies outside of the texture")]
    InvalidRegion,
    /// The texture belongs to a different renderer context.
    #[error("Texture belongs to a different renderer context")]
    ForeignTexture,
    /// The texture cannot be used for the requested operation.
    #[error("Texture does not support the requested operation")]
    UnsupportedTextureUsage,
    /// GPU buffer mapping failed.
    #[error("Buffer mapping failed: {0}")]
    BufferMap(#[from] ::wgpu::BufferAsyncError),
    /// Polling the GPU failed.
    #[error("Polling the device failed: {0}")]
    Poll(#[from] ::wgpu::PollError),
    /// Waiting for an external synchronization point was interrupted.
    #[error("Waiting for a synchronization point was interrupted")]
    SyncInterrupted,
    /// A rendering operation is not supported by this backend.
    #[error("The requested operation is not supported")]
    Unsupported,
    /// The shared-memory buffer could not be accessed.
    #[cfg(feature = "wayland_frontend")]
    #[error("Error accessing the buffer: {0:?}")]
    BufferAccess(#[from] crate::wayland::shm::BufferAccessError),
}

impl From<WgpuError> for SwapBuffersError {
    fn from(value: WgpuError) -> Self {
        match value {
            err @ WgpuError::SyncInterrupted => SwapBuffersError::TemporaryFailure(Box::new(err)),
            err => SwapBuffersError::ContextLost(Box::new(err)),
        }
    }
}
