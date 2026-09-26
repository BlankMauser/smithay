//! Implementation of the rendering traits using WGPU.
//!
//! Use [`WgpuRenderer::new`] with a WGPU device and its queue for memory-backed rendering.
//! On Linux, [`vulkan::request_device`] enables the extensions needed for DMA-BUF import and
//! rendering into GBM buffers. Only single-plane, explicitly modified RGB buffers are supported.
//!
//! DMA-BUF operations wait for GPU completion before returning buffers to external users.
//! [`WgpuFence`] supports CPU waits but does not export native synchronization files.

use std::{collections::HashMap, mem, ops::Range};

use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};

use crate::{
    backend::{
        allocator::{dmabuf::Dmabuf, format::FormatSet},
        renderer::{
            Bind, Blit, ContextId, DebugFlags, ExportMem, Frame, ImportDma, ImportMem, Offscreen, Renderer,
            RendererSuper, Texture, TextureFilter, sync::SyncPoint,
        },
    },
    utils::{Buffer, Physical, Rectangle, Size, Transform},
};

#[cfg(feature = "wayland_frontend")]
use crate::{
    backend::renderer::{ImportDmaWl, ImportMemWl},
    wayland::shm::{self, shm_format_to_fourcc},
};
#[cfg(feature = "wayland_frontend")]
use wayland_server::protocol::wl_buffer;

#[cfg(all(
    feature = "wayland_frontend",
    feature = "backend_egl",
    feature = "use_system_lib"
))]
use crate::{
    backend::{
        egl::{Error as EglError, display::EGLBufferReader},
        renderer::ImportEgl,
    },
    wayland::compositor::SurfaceData,
};

mod error;
mod frame;
mod sync;
mod texture;
#[cfg(unix)]
/// Vulkan device creation and DMA-BUF interoperability.
pub mod vulkan;

pub use error::WgpuError;
pub use sync::WgpuFence;
pub use texture::{WgpuMapping, WgpuTexture};

const MEM_FORMATS: &[DrmFourcc] = &[
    DrmFourcc::Argb8888,
    DrmFourcc::Xrgb8888,
    DrmFourcc::Abgr8888,
    DrmFourcc::Xbgr8888,
    DrmFourcc::Abgr2101010,
    DrmFourcc::Xbgr2101010,
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Vertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
    color: [f32; 4],
    force_opaque: f32,
}

#[derive(Debug)]
enum DrawKind {
    Replace,
    Solid,
    Texture { texture: WgpuTexture, opaque: bool },
}

#[derive(Debug)]
struct Draw {
    vertices: Range<u32>,
    kind: DrawKind,
}

#[derive(Debug)]
struct Pipelines {
    replace: ::wgpu::RenderPipeline,
    solid: ::wgpu::RenderPipeline,
    texture: ::wgpu::RenderPipeline,
    texture_opaque: ::wgpu::RenderPipeline,
}

/// A framebuffer backed by a WGPU texture.
#[derive(Debug, Clone)]
pub struct WgpuTarget {
    texture: WgpuTexture,
}

impl WgpuTarget {
    /// Returns the texture backing this framebuffer.
    pub fn texture(&self) -> &WgpuTexture {
        &self.texture
    }
}

impl Texture for WgpuTarget {
    fn width(&self) -> u32 {
        self.texture.width()
    }

    fn height(&self) -> u32 {
        self.texture.height()
    }

    fn size(&self) -> Size<i32, Buffer> {
        self.texture.size()
    }

    fn format(&self) -> Option<DrmFourcc> {
        self.texture.format()
    }
}

/// WGPU implementation of Smithay's renderer interfaces.
pub struct WgpuRenderer {
    device: ::wgpu::Device,
    queue: ::wgpu::Queue,
    context_id: ContextId<WgpuTexture>,
    debug_flags: DebugFlags,
    min_filter: TextureFilter,
    mag_filter: TextureFilter,
    texture_layout: ::wgpu::BindGroupLayout,
    samplers: [[::wgpu::Sampler; 2]; 2],
    shader: ::wgpu::ShaderModule,
    pipelines: HashMap<::wgpu::TextureFormat, Pipelines>,
    vertex_buffer: ::wgpu::Buffer,
    vertex_capacity: usize,
    vertices: Vec<Vertex>,
    draws: Vec<Draw>,
    #[cfg(unix)]
    vulkan: Option<vulkan::VulkanInterop>,
}

impl std::fmt::Debug for WgpuRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuRenderer")
            .field("context_id", &self.context_id)
            .field("debug_flags", &self.debug_flags)
            .field("min_filter", &self.min_filter)
            .field("mag_filter", &self.mag_filter)
            .finish_non_exhaustive()
    }
}

impl WgpuRenderer {
    /// Creates a renderer using an existing WGPU device and queue.
    pub fn new(device: ::wgpu::Device, queue: ::wgpu::Queue) -> Result<Self, WgpuError> {
        let texture_layout = device.create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
            label: Some("Smithay WGPU texture layout"),
            entries: &[
                ::wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ::wgpu::ShaderStages::FRAGMENT,
                    ty: ::wgpu::BindingType::Texture {
                        sample_type: ::wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: ::wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                ::wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ::wgpu::ShaderStages::FRAGMENT,
                    ty: ::wgpu::BindingType::Sampler(::wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = |label, min_filter, mag_filter| {
            device.create_sampler(&::wgpu::SamplerDescriptor {
                label: Some(label),
                mag_filter,
                min_filter,
                ..Default::default()
            })
        };
        let samplers = [
            [
                sampler(
                    "Smithay WGPU nearest sampler",
                    ::wgpu::FilterMode::Nearest,
                    ::wgpu::FilterMode::Nearest,
                ),
                sampler(
                    "Smithay WGPU nearest-linear sampler",
                    ::wgpu::FilterMode::Nearest,
                    ::wgpu::FilterMode::Linear,
                ),
            ],
            [
                sampler(
                    "Smithay WGPU linear-nearest sampler",
                    ::wgpu::FilterMode::Linear,
                    ::wgpu::FilterMode::Nearest,
                ),
                sampler(
                    "Smithay WGPU linear sampler",
                    ::wgpu::FilterMode::Linear,
                    ::wgpu::FilterMode::Linear,
                ),
            ],
        ];
        let shader = device.create_shader_module(::wgpu::ShaderModuleDescriptor {
            label: Some("Smithay WGPU renderer shader"),
            source: ::wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });
        let vertex_capacity = 6 * 64;
        let vertex_buffer_size = vertex_capacity * mem::size_of::<Vertex>();
        if vertex_buffer_size as u64 > device.limits().max_buffer_size {
            return Err(WgpuError::InvalidRegion);
        }
        let vertex_buffer = device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("Smithay WGPU vertex buffer"),
            size: vertex_buffer_size as u64,
            usage: ::wgpu::BufferUsages::VERTEX | ::wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        #[cfg(unix)]
        let vulkan = device
            .features()
            .contains(::wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
            .then(|| vulkan::VulkanInterop::new(&device))
            .transpose()?;
        Ok(Self {
            device,
            queue,
            context_id: ContextId::new(),
            debug_flags: DebugFlags::empty(),
            min_filter: TextureFilter::Linear,
            mag_filter: TextureFilter::Linear,
            texture_layout,
            samplers,
            shader,
            pipelines: HashMap::new(),
            vertex_buffer,
            vertex_capacity,
            vertices: Vec::with_capacity(vertex_capacity),
            draws: Vec::with_capacity(64),
            #[cfg(unix)]
            vulkan,
        })
    }

    /// Returns the WGPU device used by this renderer.
    pub fn device(&self) -> &::wgpu::Device {
        &self.device
    }

    /// Returns the WGPU queue used by this renderer.
    pub fn queue(&self) -> &::wgpu::Queue {
        &self.queue
    }

    fn create_pipeline(
        &self,
        format: ::wgpu::TextureFormat,
        fragment: &'static str,
        blend: Option<::wgpu::BlendState>,
        texture_layout: bool,
    ) -> ::wgpu::RenderPipeline {
        let texture_layouts = [Some(&self.texture_layout)];
        let layout = self
            .device
            .create_pipeline_layout(&::wgpu::PipelineLayoutDescriptor {
                label: Some("Smithay WGPU pipeline layout"),
                bind_group_layouts: if texture_layout { &texture_layouts } else { &[] },
                immediate_size: 0,
            });
        self.device
            .create_render_pipeline(&::wgpu::RenderPipelineDescriptor {
                label: Some("Smithay WGPU render pipeline"),
                layout: Some(&layout),
                vertex: ::wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("vertex"),
                    compilation_options: Default::default(),
                    buffers: &[Some(::wgpu::VertexBufferLayout {
                        array_stride: mem::size_of::<Vertex>() as u64,
                        step_mode: ::wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            ::wgpu::VertexAttribute {
                                format: ::wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            ::wgpu::VertexAttribute {
                                format: ::wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                            ::wgpu::VertexAttribute {
                                format: ::wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 2,
                            },
                            ::wgpu::VertexAttribute {
                                format: ::wgpu::VertexFormat::Float32,
                                offset: 32,
                                shader_location: 3,
                            },
                        ],
                    })],
                },
                primitive: ::wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: ::wgpu::MultisampleState::default(),
                fragment: Some(::wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(::wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: ::wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    fn ensure_pipelines(&mut self, format: ::wgpu::TextureFormat) {
        if self.pipelines.contains_key(&format) {
            return;
        }
        let premultiplied = ::wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;
        self.pipelines.insert(
            format,
            Pipelines {
                replace: self.create_pipeline(format, "solid_fragment", None, false),
                solid: self.create_pipeline(format, "solid_fragment", Some(premultiplied), false),
                texture: self.create_pipeline(format, "texture_fragment", Some(premultiplied), true),
                texture_opaque: self.create_pipeline(format, "texture_fragment", None, true),
            },
        );
    }

    fn ensure_vertex_capacity(&mut self, len: usize) -> Result<(), WgpuError> {
        if len <= self.vertex_capacity {
            return Ok(());
        }
        let capacity = len.checked_next_power_of_two().ok_or(WgpuError::InvalidRegion)?;
        let size = capacity
            .checked_mul(mem::size_of::<Vertex>())
            .ok_or(WgpuError::InvalidRegion)? as u64;
        if size > self.device.limits().max_buffer_size {
            return Err(WgpuError::InvalidRegion);
        }
        self.vertex_buffer = self.device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("Smithay WGPU vertex buffer"),
            size,
            usage: ::wgpu::BufferUsages::VERTEX | ::wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.vertex_capacity = capacity;
        Ok(())
    }

    fn submit(
        &mut self,
        target: &WgpuTexture,
        viewport: Size<i32, Physical>,
        vertices: &[Vertex],
        draws: &[Draw],
    ) -> Result<SyncPoint, WgpuError> {
        let mut used = vec![target];
        for texture in draws.iter().filter_map(|draw| match &draw.kind {
            DrawKind::Texture { texture, .. } => Some(texture),
            _ => None,
        }) {
            if !used.iter().any(|used| std::ptr::eq(used.raw(), texture.raw())) {
                used.push(texture);
            }
        }
        if !vertices.is_empty() {
            self.ensure_vertex_capacity(vertices.len())?;
            // Vertex is repr(C) and contains only contiguous f32 fields, with no padding.
            let bytes = unsafe {
                std::slice::from_raw_parts(vertices.as_ptr().cast::<u8>(), mem::size_of_val(vertices))
            };
            self.queue.write_buffer(&self.vertex_buffer, 0, bytes);
        }
        let mut encoder = self
            .device
            .create_command_encoder(&::wgpu::CommandEncoderDescriptor {
                label: Some("Smithay WGPU frame encoder"),
            });
        let pipelines = &self.pipelines[&target.wgpu_format()];
        let index = |filter| match filter {
            TextureFilter::Nearest => 0,
            TextureFilter::Linear => 1,
        };
        let sampler = &self.samplers[index(self.min_filter)][index(self.mag_filter)];
        let bind_groups: Vec<_> = draws
            .iter()
            .map(|draw| match &draw.kind {
                DrawKind::Texture { texture, .. } => {
                    Some(self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
                        label: Some("Smithay WGPU texture bind group"),
                        layout: &self.texture_layout,
                        entries: &[
                            ::wgpu::BindGroupEntry {
                                binding: 0,
                                resource: ::wgpu::BindingResource::TextureView(texture.view()),
                            },
                            ::wgpu::BindGroupEntry {
                                binding: 1,
                                resource: ::wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    }))
                }
                _ => None,
            })
            .collect();
        {
            let mut pass = encoder.begin_render_pass(&::wgpu::RenderPassDescriptor {
                label: Some("Smithay WGPU render pass"),
                color_attachments: &[Some(::wgpu::RenderPassColorAttachment {
                    view: target.view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: ::wgpu::Operations {
                        load: ::wgpu::LoadOp::Load,
                        store: ::wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(0.0, 0.0, viewport.w as f32, viewport.h as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, viewport.w as u32, viewport.h as u32);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            for (draw, bind_group) in draws.iter().zip(&bind_groups) {
                match &draw.kind {
                    DrawKind::Replace => pass.set_pipeline(&pipelines.replace),
                    DrawKind::Solid => pass.set_pipeline(&pipelines.solid),
                    DrawKind::Texture { opaque, .. } => {
                        pass.set_pipeline(if *opaque {
                            &pipelines.texture_opaque
                        } else {
                            &pipelines.texture
                        });
                        pass.set_bind_group(0, bind_group.as_ref().unwrap(), &[]);
                        pass.draw(draw.vertices.clone(), 0..1);
                        continue;
                    }
                }
                pass.draw(draw.vertices.clone(), 0..1);
            }
        }

        let mut resting = Vec::new();
        for texture in &used {
            if let Some(state) = texture.sync().and_then(|sync| sync.resting_state()) {
                resting.push(::wgpu::TextureTransition {
                    texture: texture.raw(),
                    selector: None,
                    state,
                });
            }
        }
        encoder.transition_resources(std::iter::empty(), resting.into_iter());
        for (index, texture) in used.iter().enumerate() {
            if let Some(sync) = texture.sync() {
                if let Err(err) = sync.acquire(&self.device, &self.queue, texture.raw()) {
                    // No frame commands have run. Return the earlier imports in their resting layout.
                    let submission = self.queue.submit([]);
                    for acquired in &used[..index] {
                        if let Some(sync) = acquired.sync() {
                            let _ = sync.submitted(&self.device, &self.queue, submission.clone());
                        }
                    }
                    return Err(err);
                }
            }
        }
        let submission = self.queue.submit([encoder.finish()]);
        let mut release_result = Ok(());
        for texture in &used {
            if let Some(sync) = texture.sync() {
                let result = sync.submitted(&self.device, &self.queue, submission.clone());
                if release_result.is_ok() {
                    release_result = result;
                }
            }
        }
        release_result?;
        Ok(sync::WgpuFence::after_submission(&self.queue, &self.device))
    }

    fn new_texture(
        &self,
        format: DrmFourcc,
        size: Size<i32, Buffer>,
        flipped: bool,
    ) -> Result<WgpuTexture, WgpuError> {
        if size.w <= 0 || size.h <= 0 {
            return Err(WgpuError::InvalidRegion);
        }
        let max_dimension = self.device.limits().max_texture_dimension_2d as i32;
        if size.w > max_dimension || size.h > max_dimension {
            return Err(WgpuError::InvalidRegion);
        }
        let wgpu_format = texture::format_to_wgpu(format)?;
        let texture = self.device.create_texture(&::wgpu::TextureDescriptor {
            label: Some("Smithay WGPU texture"),
            size: ::wgpu::Extent3d {
                width: size.w as u32,
                height: size.h as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: ::wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: ::wgpu::TextureUsages::TEXTURE_BINDING
                | ::wgpu::TextureUsages::RENDER_ATTACHMENT
                | ::wgpu::TextureUsages::COPY_SRC
                | ::wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        Ok(WgpuTexture::from_raw(
            texture,
            size,
            format,
            wgpu_format,
            flipped,
            self.context_id.clone(),
            None,
        ))
    }

    fn read_texture(
        &mut self,
        texture: &WgpuTexture,
        region: Rectangle<i32, Buffer>,
        format: DrmFourcc,
    ) -> Result<WgpuMapping, WgpuError> {
        if texture.context_id() != &self.context_id {
            return Err(WgpuError::ForeignTexture);
        }
        if !texture.raw().usage().contains(::wgpu::TextureUsages::COPY_SRC) {
            return Err(WgpuError::UnsupportedTextureUsage);
        }
        if format != texture.format().unwrap() {
            return Err(WgpuError::UnsupportedPixelFormat(format));
        }
        let right = region
            .loc
            .x
            .checked_add(region.size.w)
            .ok_or(WgpuError::InvalidRegion)?;
        let bottom = region
            .loc
            .y
            .checked_add(region.size.h)
            .ok_or(WgpuError::InvalidRegion)?;
        if region.loc.x < 0
            || region.loc.y < 0
            || region.size.w <= 0
            || region.size.h <= 0
            || right > texture.size().w
            || bottom > texture.size().h
        {
            return Err(WgpuError::InvalidRegion);
        }
        let bytes_per_pixel = texture::bytes_per_pixel(format)?;
        let row_bytes = region.size.w as usize * bytes_per_pixel;
        let padded_row_bytes = row_bytes.div_ceil(::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
            * ::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let buffer_size = padded_row_bytes
            .checked_mul(region.size.h as usize)
            .ok_or(WgpuError::InvalidRegion)?;
        if buffer_size as u64 > self.device.limits().max_buffer_size {
            return Err(WgpuError::InvalidRegion);
        }
        let buffer = self.device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some("Smithay WGPU readback buffer"),
            size: buffer_size as u64,
            usage: ::wgpu::BufferUsages::COPY_DST | ::wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        if let Some(sync) = texture.sync() {
            sync.acquire(&self.device, &self.queue, texture.raw())?;
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            ::wgpu::TexelCopyTextureInfo {
                texture: texture.raw(),
                mip_level: 0,
                origin: ::wgpu::Origin3d {
                    x: region.loc.x as u32,
                    y: region.loc.y as u32,
                    z: 0,
                },
                aspect: ::wgpu::TextureAspect::All,
            },
            ::wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: ::wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes as u32),
                    rows_per_image: Some(region.size.h as u32),
                },
            },
            ::wgpu::Extent3d {
                width: region.size.w as u32,
                height: region.size.h as u32,
                depth_or_array_layers: 1,
            },
        );
        if let Some(state) = texture.sync().and_then(|sync| sync.resting_state()) {
            encoder.transition_resources(
                std::iter::empty(),
                std::iter::once(::wgpu::TextureTransition {
                    texture: texture.raw(),
                    selector: None,
                    state,
                }),
            );
        }
        let submission = self.queue.submit([encoder.finish()]);
        if let Some(sync) = texture.sync() {
            sync.submitted(&self.device, &self.queue, submission)?;
        }
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        buffer.slice(..).map_async(::wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(::wgpu::PollType::wait_indefinitely())?;
        receiver.recv().map_err(|_| WgpuError::Unsupported)??;
        let mapped = buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|_| WgpuError::Unsupported)?;
        let mut data = Vec::with_capacity(row_bytes * region.size.h as usize);
        for row in mapped.chunks_exact(padded_row_bytes).take(region.size.h as usize) {
            data.extend_from_slice(&row[..row_bytes]);
        }
        drop(mapped);
        buffer.unmap();
        Ok(WgpuMapping {
            data,
            size: region.size,
            format,
        })
    }
}

impl RendererSuper for WgpuRenderer {
    type Error = WgpuError;
    type TextureId = WgpuTexture;
    type Framebuffer<'buffer> = WgpuTarget;
    type Frame<'frame, 'buffer>
        = WgpuFrame<'frame>
    where
        'buffer: 'frame;
}

/// In-progress WGPU frame.
#[derive(Debug)]
pub struct WgpuFrame<'frame> {
    renderer: &'frame mut WgpuRenderer,
    target: WgpuTexture,
    output_size: Size<i32, Physical>,
    transform: Transform,
    vertices: Vec<Vertex>,
    draws: Vec<Draw>,
}

impl Renderer for WgpuRenderer {
    fn context_id(&self) -> ContextId<WgpuTexture> {
        self.context_id.clone()
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), WgpuError> {
        self.min_filter = filter;
        Ok(())
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), WgpuError> {
        self.mag_filter = filter;
        Ok(())
    }

    fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.debug_flags = flags;
    }

    fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    fn render<'frame, 'buffer>(
        &'frame mut self,
        framebuffer: &'frame mut WgpuTarget,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<WgpuFrame<'frame>, WgpuError>
    where
        'buffer: 'frame,
    {
        if framebuffer.texture.context_id() != &self.context_id {
            return Err(WgpuError::ForeignTexture);
        }
        if output_size.w <= 0 || output_size.h <= 0 {
            return Err(WgpuError::InvalidRegion);
        }
        let render_size = dst_transform.transform_size(output_size);
        if render_size.w > framebuffer.width() as i32 || render_size.h > framebuffer.height() as i32 {
            return Err(WgpuError::InvalidRegion);
        }
        self.ensure_pipelines(framebuffer.texture.wgpu_format());
        Ok(WgpuFrame {
            target: framebuffer.texture.clone(),
            output_size,
            transform: dst_transform,
            vertices: mem::take(&mut self.vertices),
            draws: mem::take(&mut self.draws),
            renderer: self,
        })
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), WgpuError> {
        sync::wait(sync)
    }
}

impl Bind<WgpuTexture> for WgpuRenderer {
    fn bind<'a>(&mut self, target: &'a mut WgpuTexture) -> Result<WgpuTarget, WgpuError> {
        if target.context_id() != &self.context_id {
            return Err(WgpuError::ForeignTexture);
        }
        if !target
            .raw()
            .usage()
            .contains(::wgpu::TextureUsages::RENDER_ATTACHMENT)
        {
            return Err(WgpuError::UnsupportedTextureUsage);
        }
        Ok(WgpuTarget {
            texture: target.clone(),
        })
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        Some(
            MEM_FORMATS
                .iter()
                .copied()
                .map(|code| DrmFormat {
                    code,
                    modifier: DrmModifier::Linear,
                })
                .collect(),
        )
    }
}

#[cfg(unix)]
impl Bind<Dmabuf> for WgpuRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<WgpuTarget, WgpuError> {
        let interop = self
            .vulkan
            .as_ref()
            .ok_or(vulkan::VulkanError::ExternalMemoryNotEnabled)?;
        let imported = interop.import(
            &self.device,
            target,
            ::wgpu::TextureUsages::RENDER_ATTACHMENT
                | ::wgpu::TextureUsages::TEXTURE_BINDING
                | ::wgpu::TextureUsages::COPY_SRC,
        )?;
        let texture = WgpuTexture::from_raw(
            imported.texture,
            imported.size,
            imported.format,
            imported.wgpu_format,
            imported.flipped,
            self.context_id.clone(),
            Some(imported.sync),
        );
        Ok(WgpuTarget { texture })
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        self.vulkan.as_ref().map(|interop| {
            interop
                .target_formats()
                .iter()
                .filter(|format| interop.sampled_formats().contains(format))
                .copied()
                .collect()
        })
    }
}

#[cfg(unix)]
impl ImportDma for WgpuRenderer {
    fn dmabuf_formats(&self) -> FormatSet {
        self.vulkan
            .as_ref()
            .map(|interop| interop.sampled_formats().clone())
            .unwrap_or_default()
    }

    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, Buffer>]>,
    ) -> Result<WgpuTexture, WgpuError> {
        let interop = self
            .vulkan
            .as_ref()
            .ok_or(vulkan::VulkanError::ExternalMemoryNotEnabled)?;
        let imported = interop.import(
            &self.device,
            dmabuf,
            ::wgpu::TextureUsages::TEXTURE_BINDING | ::wgpu::TextureUsages::COPY_SRC,
        )?;
        Ok(WgpuTexture::from_raw(
            imported.texture,
            imported.size,
            imported.format,
            imported.wgpu_format,
            imported.flipped,
            self.context_id.clone(),
            Some(imported.sync),
        ))
    }
}

#[cfg(all(unix, feature = "wayland_frontend"))]
impl ImportDmaWl for WgpuRenderer {}

impl Offscreen<WgpuTexture> for WgpuRenderer {
    fn create_buffer(
        &mut self,
        format: DrmFourcc,
        size: Size<i32, Buffer>,
    ) -> Result<WgpuTexture, WgpuError> {
        self.new_texture(format, size, false)
    }
}

impl ImportMem for WgpuRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: DrmFourcc,
        size: Size<i32, Buffer>,
        flipped: bool,
    ) -> Result<WgpuTexture, WgpuError> {
        let bytes_per_pixel = texture::bytes_per_pixel(format)?;
        if size.w <= 0 || size.h <= 0 {
            return Err(WgpuError::InvalidRegion);
        }
        let expected = (size.w as usize)
            .checked_mul(size.h as usize)
            .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
            .ok_or(WgpuError::InvalidRegion)?;
        if data.len() < expected {
            return Err(WgpuError::IncompleteBuffer {
                expected,
                actual: data.len(),
            });
        }
        let texture = self.new_texture(format, size, flipped)?;
        self.queue.write_texture(
            ::wgpu::TexelCopyTextureInfo {
                texture: texture.raw(),
                mip_level: 0,
                origin: ::wgpu::Origin3d::ZERO,
                aspect: ::wgpu::TextureAspect::All,
            },
            &data[..expected],
            ::wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.w as u32 * bytes_per_pixel as u32),
                rows_per_image: Some(size.h as u32),
            },
            ::wgpu::Extent3d {
                width: size.w as u32,
                height: size.h as u32,
                depth_or_array_layers: 1,
            },
        );
        Ok(texture)
    }

    fn update_memory(
        &mut self,
        texture: &WgpuTexture,
        data: &[u8],
        region: Rectangle<i32, Buffer>,
    ) -> Result<(), WgpuError> {
        if texture.context_id() != &self.context_id {
            return Err(WgpuError::ForeignTexture);
        }
        if !texture.raw().usage().contains(::wgpu::TextureUsages::COPY_DST) {
            return Err(WgpuError::UnsupportedTextureUsage);
        }
        let size = texture.size();
        let bytes_per_pixel = texture::bytes_per_pixel(texture.format().unwrap())?;
        let expected = size.w as usize * size.h as usize * bytes_per_pixel;
        if data.len() < expected {
            return Err(WgpuError::IncompleteBuffer {
                expected,
                actual: data.len(),
            });
        }
        let right = region
            .loc
            .x
            .checked_add(region.size.w)
            .ok_or(WgpuError::InvalidRegion)?;
        let bottom = region
            .loc
            .y
            .checked_add(region.size.h)
            .ok_or(WgpuError::InvalidRegion)?;
        if region.loc.x < 0
            || region.loc.y < 0
            || region.size.w <= 0
            || region.size.h <= 0
            || right > size.w
            || bottom > size.h
        {
            return Err(WgpuError::InvalidRegion);
        }
        let source_stride = size.w as usize * bytes_per_pixel;
        let offset = region.loc.y as usize * source_stride + region.loc.x as usize * bytes_per_pixel;
        self.queue.write_texture(
            ::wgpu::TexelCopyTextureInfo {
                texture: texture.raw(),
                mip_level: 0,
                origin: ::wgpu::Origin3d {
                    x: region.loc.x as u32,
                    y: region.loc.y as u32,
                    z: 0,
                },
                aspect: ::wgpu::TextureAspect::All,
            },
            data,
            ::wgpu::TexelCopyBufferLayout {
                offset: offset as u64,
                bytes_per_row: Some(source_stride as u32),
                rows_per_image: Some(region.size.h as u32),
            },
            ::wgpu::Extent3d {
                width: region.size.w as u32,
                height: region.size.h as u32,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = DrmFourcc>> {
        Box::new(MEM_FORMATS.iter().copied())
    }
}

#[cfg(feature = "wayland_frontend")]
impl ImportMemWl for WgpuRenderer {
    fn import_shm_buffer(
        &mut self,
        buffer: &wl_buffer::WlBuffer,
        _surface: Option<&crate::wayland::compositor::SurfaceData>,
        _damage: &[Rectangle<i32, Buffer>],
    ) -> Result<WgpuTexture, WgpuError> {
        shm::with_buffer_contents(buffer, |ptr, len, metadata| {
            if metadata.offset < 0 || metadata.width <= 0 || metadata.height <= 0 || metadata.stride <= 0 {
                return Err(WgpuError::InvalidRegion);
            }
            let max_dimension = self.device.limits().max_texture_dimension_2d;
            if metadata.width as u32 > max_dimension || metadata.height as u32 > max_dimension {
                return Err(WgpuError::InvalidRegion);
            }
            let format = shm_format_to_fourcc(metadata.format)
                .ok_or(WgpuError::UnsupportedWlPixelFormat(metadata.format))?;
            let bytes_per_pixel = texture::bytes_per_pixel(format)?;
            let row_bytes = (metadata.width as usize)
                .checked_mul(bytes_per_pixel)
                .ok_or(WgpuError::InvalidRegion)?;
            if row_bytes > metadata.stride as usize {
                return Err(WgpuError::InvalidRegion);
            }
            let data_len = (metadata.stride as usize)
                .checked_mul(metadata.height as usize - 1)
                .and_then(|value| value.checked_add(row_bytes))
                .ok_or(WgpuError::InvalidRegion)?;
            let expected = (metadata.offset as usize)
                .checked_add(data_len)
                .ok_or(WgpuError::InvalidRegion)?;
            if len < expected {
                return Err(WgpuError::IncompleteBuffer {
                    expected,
                    actual: len,
                });
            }
            // The checked range is contained in the mapping borrowed by with_buffer_contents.
            let source = unsafe {
                std::slice::from_raw_parts(ptr.add(metadata.offset as usize), len - metadata.offset as usize)
            };
            let mut data = Vec::with_capacity(row_bytes * metadata.height as usize);
            for row in 0..metadata.height as usize {
                let offset = row * metadata.stride as usize;
                data.extend_from_slice(&source[offset..offset + row_bytes]);
            }
            self.import_memory(&data, format, (metadata.width, metadata.height).into(), false)
        })
        .map_err(WgpuError::BufferAccess)?
    }
}

#[cfg(all(
    feature = "wayland_frontend",
    feature = "backend_egl",
    feature = "use_system_lib"
))]
impl ImportEgl for WgpuRenderer {
    fn bind_wl_display(&mut self, _display: &wayland_server::DisplayHandle) -> Result<(), EglError> {
        Err(EglError::EglExtensionNotSupported(&[
            "wl_drm buffers are not supported by the WGPU renderer",
        ]))
    }

    fn unbind_wl_display(&mut self) {}

    fn egl_reader(&self) -> Option<&EGLBufferReader> {
        None
    }

    fn import_egl_buffer(
        &mut self,
        _buffer: &wl_buffer::WlBuffer,
        _surface: Option<&SurfaceData>,
        _damage: &[Rectangle<i32, Buffer>],
    ) -> Result<WgpuTexture, WgpuError> {
        Err(WgpuError::Unsupported)
    }
}

impl ExportMem for WgpuRenderer {
    type TextureMapping = WgpuMapping;

    fn copy_framebuffer(
        &mut self,
        target: &WgpuTarget,
        region: Rectangle<i32, Buffer>,
        format: DrmFourcc,
    ) -> Result<WgpuMapping, WgpuError> {
        self.read_texture(&target.texture, region, format)
    }

    fn copy_texture(
        &mut self,
        texture: &WgpuTexture,
        region: Rectangle<i32, Buffer>,
        format: DrmFourcc,
    ) -> Result<WgpuMapping, WgpuError> {
        self.read_texture(texture, region, format)
    }

    fn can_read_texture(&mut self, texture: &WgpuTexture) -> Result<bool, WgpuError> {
        Ok(texture.context_id() == &self.context_id
            && texture.raw().usage().contains(::wgpu::TextureUsages::COPY_SRC))
    }

    fn map_texture<'a>(&mut self, mapping: &'a WgpuMapping) -> Result<&'a [u8], WgpuError> {
        Ok(&mapping.data)
    }
}

impl Blit for WgpuRenderer {
    fn blit(
        &mut self,
        from: &WgpuTarget,
        to: &mut WgpuTarget,
        src: Rectangle<i32, Physical>,
        dst: Rectangle<i32, Physical>,
        filter: TextureFilter,
    ) -> Result<SyncPoint, WgpuError> {
        if from.texture.same_storage(&to.texture) {
            return Err(WgpuError::Unsupported);
        }
        let old_min_filter = mem::replace(&mut self.min_filter, filter);
        let old_mag_filter = mem::replace(&mut self.mag_filter, filter);
        let output_size: Size<i32, Physical> = (to.size().w, to.size().h).into();
        let result = (|| {
            let mut frame = self.render(to, output_size, Transform::Normal)?;
            frame.render_texture(
                &from.texture,
                Rectangle::new(
                    (src.loc.x as f64, src.loc.y as f64).into(),
                    (src.size.w as f64, src.size.h as f64).into(),
                ),
                dst,
                &[Rectangle::from_size(dst.size)],
                Transform::Normal,
                1.0,
                true,
            )?;
            frame.finish()
        })();
        self.min_filter = old_min_filter;
        self.mag_filter = old_mag_filter;
        result
    }
}
