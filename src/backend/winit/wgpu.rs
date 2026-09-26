use std::sync::Arc;

use tracing::{debug, info, instrument};
use winit::window::{Window as WinitWindow, WindowAttributes};

use super::{WinitEventLoop, WinitWindowSetup};
use crate::{
    backend::{
        SwapBuffersError,
        renderer::wgpu::{WgpuError, WgpuRenderer, WgpuTarget, vulkan},
    },
    utils::{Physical, Rectangle, Size},
};

/// Create a WGPU renderer and presentation surface for a winit window.
pub async fn init_wgpu_from_attributes(
    attributes: WindowAttributes,
) -> Result<(WgpuGraphicsBackend, WinitEventLoop), WgpuInitError> {
    let setup = WinitWindowSetup::new(attributes)?;
    let mut instance_descriptor = ::wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = ::wgpu::Backends::VULKAN;
    let instance = ::wgpu::Instance::new(instance_descriptor);
    let surface = instance.create_surface(setup.window.clone())?;
    let adapter = instance
        .request_adapter(&::wgpu::RequestAdapterOptions {
            power_preference: ::wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        })
        .await?;
    let adapter_info = adapter.get_info();
    if adapter_info.device_type == ::wgpu::DeviceType::Cpu {
        return Err(WgpuInitError::SoftwareAdapter(adapter_info.name));
    }
    info!(
        adapter = %adapter_info.name,
        driver = %adapter_info.driver,
        device_type = ?adapter_info.device_type,
        "Using WGPU adapter"
    );

    let capabilities = surface.get_capabilities(&adapter);
    let format = [
        ::wgpu::TextureFormat::Bgra8Unorm,
        ::wgpu::TextureFormat::Rgba8Unorm,
    ]
    .into_iter()
    .find(|format| capabilities.formats.contains(format))
    .ok_or(WgpuInitError::NoSurfaceFormat)?;
    let present_mode = capabilities
        .present_modes
        .iter()
        .copied()
        .find(|mode| *mode == ::wgpu::PresentMode::Fifo)
        .or_else(|| capabilities.present_modes.first().copied())
        .ok_or(WgpuInitError::NoPresentMode)?;
    let alpha_mode = capabilities
        .alpha_modes
        .iter()
        .copied()
        .find(|mode| *mode == ::wgpu::CompositeAlphaMode::Opaque)
        .or_else(|| capabilities.alpha_modes.first().copied())
        .ok_or(WgpuInitError::NoAlphaMode)?;
    let mut usage = ::wgpu::TextureUsages::RENDER_ATTACHMENT | ::wgpu::TextureUsages::COPY_SRC;
    if !capabilities.usages.contains(usage) {
        return Err(WgpuInitError::UnsupportedSurfaceUsage(capabilities.usages));
    }
    if capabilities
        .usages
        .contains(::wgpu::TextureUsages::TEXTURE_BINDING)
    {
        usage |= ::wgpu::TextureUsages::TEXTURE_BINDING;
    }

    let (device, queue) = vulkan::request_device(
        &adapter,
        &::wgpu::DeviceDescriptor {
            label: Some("Smithay winit WGPU device"),
            ..Default::default()
        },
    )?;
    let renderer = WgpuRenderer::new(device, queue)?;
    let backend = WgpuGraphicsBackend {
        renderer,
        surface,
        window: setup.window.clone(),
        configuration: SurfaceConfiguration {
            format,
            present_mode,
            alpha_mode,
            usage,
        },
        configured_size: None,
        acquired: None,
        reconfigure_after_present: false,
        adapter_info,
        span: setup.span.clone(),
    };

    Ok((backend, setup.into_event_loop()))
}

#[derive(Debug, Clone, Copy)]
struct SurfaceConfiguration {
    format: ::wgpu::TextureFormat,
    present_mode: ::wgpu::PresentMode,
    alpha_mode: ::wgpu::CompositeAlphaMode,
    usage: ::wgpu::TextureUsages,
}

/// WGPU renderer and presentation surface created by the winit backend.
pub struct WgpuGraphicsBackend {
    renderer: WgpuRenderer,
    surface: ::wgpu::Surface<'static>,
    window: Arc<dyn WinitWindow>,
    configuration: SurfaceConfiguration,
    configured_size: Option<Size<i32, Physical>>,
    acquired: Option<::wgpu::SurfaceTexture>,
    reconfigure_after_present: bool,
    adapter_info: ::wgpu::AdapterInfo,
    span: tracing::Span,
}

impl std::fmt::Debug for WgpuGraphicsBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuGraphicsBackend")
            .field("renderer", &self.renderer)
            .field("window", &self.window)
            .field("configuration", &self.configuration)
            .field("configured_size", &self.configured_size)
            .field("acquired", &self.acquired.is_some())
            .field("adapter_info", &self.adapter_info)
            .finish_non_exhaustive()
    }
}

impl WgpuGraphicsBackend {
    /// Window size of the underlying window.
    pub fn window_size(&self) -> Size<i32, Physical> {
        let (width, height): (i32, i32) = self.window.surface_size().into();
        (width, height).into()
    }

    /// Scale factor of the underlying window.
    pub fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }

    /// Reference to the underlying window.
    pub fn window(&self) -> &dyn WinitWindow {
        &*self.window
    }

    /// Access the underlying renderer.
    pub fn renderer(&mut self) -> &mut WgpuRenderer {
        &mut self.renderer
    }

    /// Information about the adapter selected for presentation.
    pub fn adapter_info(&self) -> &::wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Bind the current surface image to the renderer.
    ///
    /// Repeated calls before [`Self::submit`] return a target for the same
    /// acquired image.
    #[instrument(level = "trace", parent = &self.span, skip(self))]
    #[profiling::function]
    pub fn bind(&mut self) -> Result<(&mut WgpuRenderer, WgpuTarget<'static>), SwapBuffersError> {
        let window_size = self.window_size();
        if window_size.is_empty() {
            self.acquired = None;
            self.configured_size = None;
            self.reconfigure_after_present = false;
            return Err(WgpuPresentationError::ZeroSized.into());
        }

        if self.configured_size != Some(window_size) {
            self.acquired = None;
            self.configure(window_size);
        }
        if self.acquired.is_none() {
            self.acquired = Some(self.acquire(window_size)?);
        }
        let texture = self
            .acquired
            .as_ref()
            .ok_or(SwapBuffersError::AlreadySwapped)?
            .texture
            .clone();
        let target = self.renderer.bind_wgpu_texture(texture)?;
        Ok((&mut self.renderer, target))
    }

    /// Discard an acquired image without presenting it.
    pub fn unbind(&mut self) {
        self.acquired = None;
        if self.reconfigure_after_present {
            self.configured_size = None;
            self.reconfigure_after_present = false;
        }
    }

    /// Retrieve the age of the current surface image.
    ///
    /// WGPU does not expose retained swapchain contents, so callers must redraw
    /// the complete output.
    pub fn buffer_age(&self) -> Option<usize> {
        Some(0)
    }

    /// Present the previously bound surface image.
    #[instrument(level = "trace", parent = &self.span, skip(self, _damage))]
    #[profiling::function]
    pub fn submit(&mut self, _damage: Option<&[Rectangle<i32, Physical>]>) -> Result<(), SwapBuffersError> {
        let frame = self.acquired.take().ok_or(SwapBuffersError::AlreadySwapped)?;
        self.window.pre_present_notify();
        self.renderer.queue().present(frame);
        if self.reconfigure_after_present {
            self.configured_size = None;
            self.reconfigure_after_present = false;
        }
        Ok(())
    }

    fn configure(&mut self, size: Size<i32, Physical>) {
        debug!(?size, "Configuring WGPU surface");
        self.surface.configure(
            self.renderer.device(),
            &::wgpu::SurfaceConfiguration {
                usage: self.configuration.usage,
                format: self.configuration.format,
                color_space: ::wgpu::SurfaceColorSpace::Auto,
                width: size.w as u32,
                height: size.h as u32,
                desired_maximum_frame_latency: 2,
                present_mode: self.configuration.present_mode,
                alpha_mode: self.configuration.alpha_mode,
                view_formats: Vec::new(),
            },
        );
        self.configured_size = Some(size);
        self.reconfigure_after_present = false;
    }

    fn acquire(&mut self, size: Size<i32, Physical>) -> Result<::wgpu::SurfaceTexture, SwapBuffersError> {
        let retry = match self.surface.get_current_texture() {
            ::wgpu::CurrentSurfaceTexture::Success(frame) => return Ok(frame),
            ::wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.reconfigure_after_present = true;
                return Ok(frame);
            }
            ::wgpu::CurrentSurfaceTexture::Outdated => WgpuPresentationError::Outdated,
            ::wgpu::CurrentSurfaceTexture::Lost => WgpuPresentationError::Lost,
            ::wgpu::CurrentSurfaceTexture::Timeout => return Err(WgpuPresentationError::Timeout.into()),
            ::wgpu::CurrentSurfaceTexture::Occluded => return Err(WgpuPresentationError::Occluded.into()),
            ::wgpu::CurrentSurfaceTexture::Validation => {
                return Err(SwapBuffersError::ContextLost(Box::new(
                    WgpuPresentationError::Validation,
                )));
            }
        };

        debug!(reason = %retry, "Reconfiguring stale WGPU surface");
        self.configure(size);
        match self.surface.get_current_texture() {
            ::wgpu::CurrentSurfaceTexture::Success(frame) => Ok(frame),
            ::wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.reconfigure_after_present = true;
                Ok(frame)
            }
            ::wgpu::CurrentSurfaceTexture::Outdated => Err(WgpuPresentationError::Outdated.into()),
            ::wgpu::CurrentSurfaceTexture::Lost => Err(WgpuPresentationError::Lost.into()),
            ::wgpu::CurrentSurfaceTexture::Timeout => Err(WgpuPresentationError::Timeout.into()),
            ::wgpu::CurrentSurfaceTexture::Occluded => Err(WgpuPresentationError::Occluded.into()),
            ::wgpu::CurrentSurfaceTexture::Validation => Err(SwapBuffersError::ContextLost(Box::new(
                WgpuPresentationError::Validation,
            ))),
        }
    }
}

/// Errors raised while initializing the winit WGPU backend.
#[derive(Debug, thiserror::Error)]
pub enum WgpuInitError {
    /// Winit event loop or window creation failed.
    #[error(transparent)]
    Winit(#[from] super::Error),
    /// Creating a WGPU presentation surface failed.
    #[error("Failed to create WGPU surface: {0}")]
    Surface(#[from] ::wgpu::CreateSurfaceError),
    /// No adapter can present to this surface.
    #[error("Failed to find a WGPU adapter: {0}")]
    Adapter(#[from] ::wgpu::RequestAdapterError),
    /// WGPU selected a software renderer.
    #[error("WGPU selected software adapter {0}")]
    SoftwareAdapter(String),
    /// The surface cannot present the renderer's linear SDR formats.
    #[error("WGPU surface does not support BGRA8 or RGBA8 linear SDR output")]
    NoSurfaceFormat,
    /// The surface reports no presentation modes.
    #[error("WGPU surface reports no presentation modes")]
    NoPresentMode,
    /// The surface reports no alpha modes.
    #[error("WGPU surface reports no alpha modes")]
    NoAlphaMode,
    /// The surface cannot both render and support direct framebuffer readback.
    #[error("WGPU surface lacks render-attachment or copy-source usage: {0:?}")]
    UnsupportedSurfaceUsage(::wgpu::TextureUsages),
    /// Creating a Vulkan device failed.
    #[error(transparent)]
    Device(#[from] vulkan::VulkanError),
    /// Creating the renderer failed.
    #[error(transparent)]
    Renderer(#[from] WgpuError),
}

#[derive(Debug, thiserror::Error)]
enum WgpuPresentationError {
    #[error("Window surface has zero size")]
    ZeroSized,
    #[error("Timed out acquiring a WGPU surface image")]
    Timeout,
    #[error("WGPU surface is occluded")]
    Occluded,
    #[error("WGPU surface is outdated")]
    Outdated,
    #[error("WGPU surface was lost")]
    Lost,
    #[error("WGPU surface validation failed")]
    Validation,
}

impl From<WgpuPresentationError> for SwapBuffersError {
    fn from(error: WgpuPresentationError) -> Self {
        SwapBuffersError::TemporaryFailure(Box::new(error))
    }
}
