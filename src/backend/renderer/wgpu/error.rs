use drm_fourcc::DrmFourcc;
use thiserror::Error;

use super::UniformType;
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
    /// The requested WGPU format is not supported.
    #[error("Unsupported WGPU texture format: {0:?}")]
    UnsupportedWgpuFormat(::wgpu::TextureFormat),
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
    /// Custom WGSL shader compilation failed.
    #[error("Shader compilation failed: {0}")]
    Shader(String),
    /// A custom uniform name is empty or duplicated.
    #[error("Invalid custom uniform name: {0}")]
    InvalidUniformName(String),
    /// A custom shader has more uniforms than the fixed payload supports.
    #[error("Too many custom uniforms: {actual} > {maximum}")]
    TooManyUniforms {
        /// Number of supplied declarations.
        actual: usize,
        /// Maximum number of additional declarations.
        maximum: usize,
    },
    /// A draw supplied a uniform not declared by the program.
    #[error("Unknown custom uniform: {0}")]
    UnknownUniform(String),
    /// A draw supplied a uniform with the wrong type.
    #[error("Custom uniform type mismatch: provided {provided:?}, declared {declared:?}")]
    UniformTypeMismatch {
        /// Supplied type.
        provided: UniformType,
        /// Declared type.
        declared: UniformType,
    },
    /// A custom shader was compiled by a different renderer context.
    #[error("Shader program belongs to a different renderer context")]
    ForeignProgram,
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
