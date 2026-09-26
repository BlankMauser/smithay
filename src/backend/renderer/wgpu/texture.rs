use std::{
    fmt,
    sync::{Arc, Mutex},
};

use drm_fourcc::DrmFourcc;

use super::WgpuError;
use crate::{
    backend::renderer::{ContextId, Texture, TextureMapping},
    utils::{Buffer, Size},
};

pub(super) fn format_to_wgpu(format: DrmFourcc) -> Result<::wgpu::TextureFormat, WgpuError> {
    match format {
        DrmFourcc::Argb8888 | DrmFourcc::Xrgb8888 => Ok(::wgpu::TextureFormat::Bgra8Unorm),
        DrmFourcc::Abgr8888 | DrmFourcc::Xbgr8888 => Ok(::wgpu::TextureFormat::Rgba8Unorm),
        DrmFourcc::Abgr2101010 | DrmFourcc::Xbgr2101010 => Ok(::wgpu::TextureFormat::Rgb10a2Unorm),
        _ => Err(WgpuError::UnsupportedPixelFormat(format)),
    }
}

pub(super) fn wgpu_to_format(format: ::wgpu::TextureFormat) -> Result<DrmFourcc, WgpuError> {
    match format {
        ::wgpu::TextureFormat::Bgra8Unorm => Ok(DrmFourcc::Argb8888),
        ::wgpu::TextureFormat::Rgba8Unorm => Ok(DrmFourcc::Abgr8888),
        ::wgpu::TextureFormat::Rgb10a2Unorm => Ok(DrmFourcc::Abgr2101010),
        _ => Err(WgpuError::UnsupportedWgpuFormat(format)),
    }
}

pub(super) fn bytes_per_pixel(format: DrmFourcc) -> Result<usize, WgpuError> {
    format_to_wgpu(format).map(|_| 4)
}

pub(super) fn has_alpha(format: DrmFourcc) -> bool {
    matches!(
        format,
        DrmFourcc::Argb8888 | DrmFourcc::Abgr8888 | DrmFourcc::Abgr2101010
    )
}

pub(crate) struct WgpuTextureInner {
    pub(crate) texture: ::wgpu::Texture,
    pub(crate) view: ::wgpu::TextureView,
    pub(crate) size: Size<i32, Buffer>,
    pub(crate) format: DrmFourcc,
    pub(crate) wgpu_format: ::wgpu::TextureFormat,
    pub(crate) flipped: bool,
    pub(crate) context: ContextId<WgpuTexture>,
    pub(crate) sync: Option<Arc<dyn WgpuTextureSync>>,
    bind_groups: Mutex<[Option<::wgpu::BindGroup>; 4]>,
}

pub(crate) trait WgpuTextureSync: fmt::Debug + Send + Sync {
    fn acquire(
        &self,
        device: &::wgpu::Device,
        queue: &::wgpu::Queue,
        texture: &::wgpu::Texture,
    ) -> Result<(), WgpuError>;

    fn resting_state(&self) -> Option<::wgpu::TextureUses>;

    fn identity(&self) -> Option<(u64, u64)>;

    fn submitted(
        &self,
        device: &::wgpu::Device,
        queue: &::wgpu::Queue,
        submission: ::wgpu::SubmissionIndex,
    ) -> Result<(), WgpuError>;
}

#[cfg(unix)]
impl WgpuTextureSync for super::vulkan::VulkanSync {
    fn acquire(
        &self,
        device: &::wgpu::Device,
        queue: &::wgpu::Queue,
        _texture: &::wgpu::Texture,
    ) -> Result<(), WgpuError> {
        self.acquire(device, queue).map_err(Into::into)
    }

    fn resting_state(&self) -> Option<::wgpu::TextureUses> {
        Some(self.resting_state())
    }

    fn identity(&self) -> Option<(u64, u64)> {
        Some(self.identity())
    }

    fn submitted(
        &self,
        device: &::wgpu::Device,
        queue: &::wgpu::Queue,
        submission: ::wgpu::SubmissionIndex,
    ) -> Result<(), WgpuError> {
        self.release(device, queue, submission).map_err(Into::into)
    }
}

impl fmt::Debug for WgpuTextureInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuTextureInner")
            .field("size", &self.size)
            .field("format", &self.format)
            .field("wgpu_format", &self.wgpu_format)
            .field("flipped", &self.flipped)
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

/// Handle to a texture owned by a [`WgpuRenderer`](super::WgpuRenderer).
#[derive(Debug, Clone)]
pub struct WgpuTexture(pub(crate) Arc<WgpuTextureInner>);

impl WgpuTexture {
    pub(crate) fn from_raw(
        texture: ::wgpu::Texture,
        size: Size<i32, Buffer>,
        format: DrmFourcc,
        wgpu_format: ::wgpu::TextureFormat,
        flipped: bool,
        context: ContextId<WgpuTexture>,
        sync: Option<Arc<dyn WgpuTextureSync>>,
    ) -> Self {
        let view = texture.create_view(&::wgpu::TextureViewDescriptor::default());
        Self(Arc::new(WgpuTextureInner {
            texture,
            view,
            size,
            format,
            wgpu_format,
            flipped,
            context,
            sync,
            bind_groups: Mutex::new(std::array::from_fn(|_| None)),
        }))
    }

    pub(crate) fn raw(&self) -> &::wgpu::Texture {
        &self.0.texture
    }

    pub(crate) fn view(&self) -> &::wgpu::TextureView {
        &self.0.view
    }

    pub(crate) fn context_id(&self) -> &ContextId<WgpuTexture> {
        &self.0.context
    }

    pub(crate) fn wgpu_format(&self) -> ::wgpu::TextureFormat {
        self.0.wgpu_format
    }

    /// Returns whether the texture contents have an inverted y-axis.
    pub fn is_y_inverted(&self) -> bool {
        self.0.flipped
    }

    pub(crate) fn flipped(&self) -> bool {
        self.is_y_inverted()
    }

    pub(crate) fn sync(&self) -> Option<&Arc<dyn WgpuTextureSync>> {
        self.0.sync.as_ref()
    }

    pub(crate) fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || self
                .sync()
                .and_then(|sync| sync.identity())
                .zip(other.sync().and_then(|sync| sync.identity()))
                .is_some_and(|(left, right)| left == right)
    }

    pub(super) fn bind_group(
        &self,
        index: usize,
        device: &::wgpu::Device,
        layout: &::wgpu::BindGroupLayout,
        sampler: &::wgpu::Sampler,
    ) -> ::wgpu::BindGroup {
        let mut bind_groups = self.0.bind_groups.lock().unwrap();
        bind_groups[index]
            .get_or_insert_with(|| {
                device.create_bind_group(&::wgpu::BindGroupDescriptor {
                    label: Some("Smithay WGPU texture bind group"),
                    layout,
                    entries: &[
                        ::wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ::wgpu::BindingResource::TextureView(self.view()),
                        },
                        ::wgpu::BindGroupEntry {
                            binding: 1,
                            resource: ::wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                })
            })
            .clone()
    }
}

impl Texture for WgpuTexture {
    fn width(&self) -> u32 {
        self.0.size.w as u32
    }

    fn height(&self) -> u32 {
        self.0.size.h as u32
    }

    fn size(&self) -> Size<i32, Buffer> {
        self.0.size
    }

    fn format(&self) -> Option<DrmFourcc> {
        Some(self.0.format)
    }
}

/// CPU-visible copy of a WGPU texture.
#[derive(Debug)]
pub struct WgpuMapping {
    pub(super) data: Vec<u8>,
    pub(super) size: Size<i32, Buffer>,
    pub(super) format: DrmFourcc,
}

impl Texture for WgpuMapping {
    fn width(&self) -> u32 {
        self.size.w as u32
    }

    fn height(&self) -> u32 {
        self.size.h as u32
    }

    fn size(&self) -> Size<i32, Buffer> {
        self.size
    }

    fn format(&self) -> Option<DrmFourcc> {
        Some(self.format)
    }
}

impl TextureMapping for WgpuMapping {
    fn flipped(&self) -> bool {
        false
    }
}
