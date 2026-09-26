use std::{
    fmt,
    sync::{Arc, Mutex},
};

use ash::{ext, vk};
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
use rustix::event::{PollFd, PollFlags};
use thiserror::Error;
use wgpu::hal::{self, api::Vulkan};

use crate::{
    backend::{
        allocator::{Buffer as _, dmabuf::Dmabuf},
        renderer::FormatSet,
    },
    utils::{Buffer, Size},
};

const EXTERNAL_MEMORY_FEATURE: wgpu::Features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF;
const RESTING_STATE: wgpu::TextureUses = wgpu::TextureUses::COPY_SRC.union(wgpu::TextureUses::RESOURCE);

/// Error returned while setting up Vulkan DMA-BUF interop.
#[derive(Debug, Error)]
pub enum VulkanError {
    /// The adapter does not use Vulkan.
    #[error("The adapter does not use Vulkan")]
    NotVulkan,
    /// The adapter is missing a Vulkan extension required for DMA-BUF interop.
    #[error("The Vulkan adapter is missing {0}")]
    MissingExtension(&'static str),
    /// The device was not opened with DMA-BUF external-memory support.
    #[error("The WGPU device was not opened for Vulkan DMA-BUF import")]
    ExternalMemoryNotEnabled,
    /// The DMA-BUF format or modifier is not supported for the requested usage.
    #[error("Unsupported DMA-BUF format {0:?}")]
    UnsupportedFormat(DrmFormat),
    /// Only single-plane DMA-BUFs can currently be imported.
    #[error("Only single-plane DMA-BUFs are supported")]
    UnsupportedPlaneCount,
    /// The buffer has an invalid implicit modifier.
    #[error("DMA-BUF import requires an explicit or linear modifier")]
    ImplicitModifier,
    /// The buffer dimensions cannot be represented by WGPU.
    #[error("Invalid DMA-BUF size")]
    InvalidSize,
    /// The plane stride cannot hold one row of pixels.
    #[error("Invalid DMA-BUF stride")]
    InvalidStride,
    /// The DMA-BUF belongs to another DRM device.
    #[error("The DMA-BUF belongs to another DRM device")]
    DeviceMismatch,
    /// Opening the Vulkan device failed.
    #[error("Opening the Vulkan device failed: {0}")]
    OpenDevice(#[source] hal::DeviceError),
    /// Registering the raw Vulkan device with WGPU failed.
    #[error("Registering the Vulkan device with WGPU failed: {0}")]
    RegisterDevice(#[source] wgpu::RequestDeviceError),
    /// Importing the DMA-BUF into Vulkan failed.
    #[error("Importing the DMA-BUF failed: {0}")]
    Import(#[source] hal::DeviceError),
    /// Duplicating a DMA-BUF file descriptor failed.
    #[error("Duplicating the DMA-BUF file descriptor failed: {0}")]
    DuplicateFd(#[source] std::io::Error),
    /// Reading the DMA-BUF identity failed.
    #[error("Reading the DMA-BUF identity failed: {0}")]
    InspectFd(#[source] std::io::Error),
    /// The requested limits exceed the adapter's capabilities.
    #[error("Requested device limits exceed the adapter's capabilities")]
    UnsupportedLimits,
    /// Waiting for implicit DMA-BUF synchronization failed.
    #[error("Waiting for DMA-BUF synchronization failed: {0}")]
    DmabufWait(#[source] std::io::Error),
    /// Waiting for WGPU work failed.
    #[error("Waiting for WGPU work failed: {0}")]
    WgpuWait(#[source] wgpu::PollError),
}

/// Request a WGPU Vulkan device capable of importing DMA-BUFs.
///
/// In addition to WGPU's external-memory feature this enables
/// `VK_EXT_queue_family_foreign`, which is needed to preserve the contents of
/// images exchanged with non-Vulkan APIs.
pub fn request_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> Result<(wgpu::Device, wgpu::Queue), VulkanError> {
    if !descriptor.required_limits.check_limits(&adapter.limits()) {
        return Err(VulkanError::UnsupportedLimits);
    }
    let required_features = descriptor.required_features | EXTERNAL_MEMORY_FEATURE;
    if !adapter.features().contains(required_features) {
        return Err(VulkanError::MissingExtension(
            "VK_EXT_external_memory_dma_buf or VK_EXT_image_drm_format_modifier",
        ));
    }

    let descriptor = wgpu::DeviceDescriptor {
        label: descriptor.label,
        required_features,
        required_limits: descriptor.required_limits.clone(),
        experimental_features: descriptor.experimental_features,
        memory_hints: descriptor.memory_hints.clone(),
        trace: descriptor.trace.clone(),
    };

    // SAFETY: The guard remains live while the raw adapter is inspected and no
    // Vulkan handle obtained from it escapes this scope.
    let adapter_guard = unsafe { adapter.as_hal::<Vulkan>() }.ok_or(VulkanError::NotVulkan)?;
    if !adapter_guard
        .physical_device_capabilities()
        .supports_extension(ext::queue_family_foreign::NAME)
    {
        return Err(VulkanError::MissingExtension("VK_EXT_queue_family_foreign"));
    }

    // SAFETY: WGPU validates the feature/limit set above. The callback only
    // adds an extension advertised by this physical device.
    let open_device = unsafe {
        adapter_guard.open_with_callback(
            required_features,
            &descriptor.required_limits,
            &descriptor.memory_hints,
            Some(Box::new(|args| {
                if !args.extensions.contains(&ext::queue_family_foreign::NAME) {
                    args.extensions.push(ext::queue_family_foreign::NAME);
                }
            })),
        )
    }
    .map_err(VulkanError::OpenDevice)?;

    // SAFETY: open_device came from this adapter and has not been registered
    // with another WGPU device.
    unsafe { adapter.create_device_from_hal(open_device, &descriptor) }.map_err(VulkanError::RegisterDevice)
}

/// A DMA-BUF imported into a WGPU texture.
pub(crate) struct ImportedDmabuf {
    pub(crate) texture: wgpu::Texture,
    pub(crate) size: Size<i32, Buffer>,
    pub(crate) format: DrmFourcc,
    pub(crate) wgpu_format: wgpu::TextureFormat,
    pub(crate) flipped: bool,
    pub(crate) sync: Arc<VulkanSync>,
}

impl fmt::Debug for ImportedDmabuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImportedDmabuf")
            .field("size", &self.size)
            .field("format", &self.format)
            .field("wgpu_format", &self.wgpu_format)
            .field("flipped", &self.flipped)
            .finish_non_exhaustive()
    }
}

/// Vulkan state shared by DMA-BUF imports from one WGPU device.
#[derive(Debug)]
pub(crate) struct VulkanInterop {
    sampled_formats: FormatSet,
    target_formats: FormatSet,
    #[cfg(feature = "backend_drm")]
    nodes: [Option<libc::dev_t>; 2],
}

impl VulkanInterop {
    pub(crate) fn new(device: &wgpu::Device) -> Result<Self, VulkanError> {
        if !device.features().contains(EXTERNAL_MEMORY_FEATURE) {
            return Err(VulkanError::ExternalMemoryNotEnabled);
        }

        // SAFETY: The guard is used only to query this device and is not kept
        // across any WGPU calls.
        let device_guard = unsafe { device.as_hal::<Vulkan>() }.ok_or(VulkanError::NotVulkan)?;
        if !device_guard
            .enabled_device_extensions()
            .contains(&ext::queue_family_foreign::NAME)
        {
            return Err(VulkanError::MissingExtension("VK_EXT_queue_family_foreign"));
        }

        let mut sampled_formats = Vec::new();
        let mut target_formats = Vec::new();
        for (fourcc, wgpu_format) in supported_formats() {
            let vk_format = texture_format_as_raw(*wgpu_format);
            for properties in modifier_properties(&device_guard, vk_format) {
                if properties.drm_format_modifier_plane_count != 1 {
                    continue;
                }

                let format = DrmFormat {
                    code: *fourcc,
                    modifier: DrmModifier::from(properties.drm_format_modifier),
                };
                let features = properties.drm_format_modifier_tiling_features;
                let sampling = vk::FormatFeatureFlags::SAMPLED_IMAGE
                    | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
                    | vk::FormatFeatureFlags::TRANSFER_SRC;
                let sampled = features.contains(sampling)
                    && supports_image(
                        &device_guard,
                        vk_format,
                        format.modifier,
                        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
                    );
                let target = features.contains(
                    sampling
                        | vk::FormatFeatureFlags::COLOR_ATTACHMENT
                        | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
                ) && supports_image(
                    &device_guard,
                    vk_format,
                    format.modifier,
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                );
                if sampled {
                    sampled_formats.push(format);
                }
                if target {
                    target_formats.push(format);
                }
            }
        }

        #[cfg(feature = "backend_drm")]
        let nodes = physical_device_nodes(&device_guard);
        Ok(Self {
            sampled_formats: sampled_formats.into_iter().collect(),
            target_formats: target_formats.into_iter().collect(),
            #[cfg(feature = "backend_drm")]
            nodes,
        })
    }

    pub(crate) fn sampled_formats(&self) -> &FormatSet {
        &self.sampled_formats
    }

    pub(crate) fn target_formats(&self) -> &FormatSet {
        &self.target_formats
    }

    pub(crate) fn import(
        &self,
        device: &wgpu::Device,
        dmabuf: &Dmabuf,
        usage: wgpu::TextureUsages,
    ) -> Result<ImportedDmabuf, VulkanError> {
        if dmabuf.num_planes() != 1 {
            return Err(VulkanError::UnsupportedPlaneCount);
        }
        if dmabuf.format().modifier == DrmModifier::Invalid {
            return Err(VulkanError::ImplicitModifier);
        }
        let width = dmabuf.width();
        let height = dmabuf.height();
        let max_dimension = device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max_dimension || height > max_dimension {
            return Err(VulkanError::InvalidSize);
        }
        self.check_node(dmabuf)?;

        let format = dmabuf.format();
        let needs_sampling = usage.contains(wgpu::TextureUsages::TEXTURE_BINDING);
        let needs_target = usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT);
        let supported = match (needs_sampling, needs_target) {
            (true, true) => self.target_formats.contains(&format),
            (true, false) => self.sampled_formats.contains(&format),
            (false, true) => self.target_formats.contains(&format),
            (false, false) => false,
        };
        if !supported {
            return Err(VulkanError::UnsupportedFormat(format));
        }

        let wgpu_format = supported_formats()
            .iter()
            .find_map(|(fourcc, format)| (*fourcc == dmabuf.format().code).then_some(*format))
            .ok_or(VulkanError::UnsupportedFormat(format))?;
        let size = Size::<i32, Buffer>::from((width as i32, height as i32));
        let extent = wgpu::Extent3d {
            width: size.w as u32,
            height: size.h as u32,
            depth_or_array_layers: 1,
        };
        let hal_usage = texture_uses(usage);
        let hal_descriptor = hal::TextureDescriptor {
            label: Some("smithay wgpu dma-buf"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: hal_usage,
            memory_flags: hal::MemoryFlags::empty(),
            view_formats: Vec::new(),
        };
        let descriptor = wgpu::TextureDescriptor {
            label: Some("smithay wgpu dma-buf"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage,
            view_formats: &[],
        };

        let fd = dmabuf
            .handles()
            .next()
            .expect("single-plane DMA-BUF has one file descriptor")
            .try_clone_to_owned()
            .map_err(VulkanError::DuplicateFd)?;
        let stat = rustix::fs::fstat(&fd).map_err(|err| VulkanError::InspectFd(err.into()))?;
        let identity = (stat.st_dev, stat.st_ino);
        let stride = dmabuf.strides().next().unwrap();
        let minimum_stride = width.checked_mul(4).ok_or(VulkanError::InvalidStride)?;
        if stride < minimum_stride {
            return Err(VulkanError::InvalidStride);
        }
        let offset = dmabuf.offsets().next().unwrap() as u64;

        // SAFETY: The guard is scoped to the raw import. All imported handles
        // remain owned by the HAL texture returned below.
        let device_guard = unsafe { device.as_hal::<Vulkan>() }.ok_or(VulkanError::NotVulkan)?;
        // SAFETY: The buffer has one plane, an explicit supported modifier,
        // validated dimensions and stride, and fd is a duplicate owned by HAL.
        let hal_texture = unsafe {
            device_guard.texture_from_dmabuf_fd(
                fd,
                &hal_descriptor,
                u64::from(dmabuf.format().modifier),
                u64::from(stride),
                offset,
            )
        }
        .map_err(VulkanError::Import)?;
        // SAFETY: image is borrowed from hal_texture and is used only while the
        // WGPU texture wrapping hal_texture remains alive.
        let image = unsafe { hal_texture.raw_handle() };
        drop(device_guard);

        // SAFETY: hal_texture belongs to this device, descriptor matches the
        // imported image, whose external layout is RESTING_STATE's Vulkan
        // GENERAL layout. Queue ownership is acquired before its first use.
        let texture =
            unsafe { device.create_texture_from_hal::<Vulkan>(hal_texture, &descriptor, RESTING_STATE) };
        let sync = Arc::new(VulkanSync {
            dmabuf: dmabuf.clone(),
            identity,
            image,
            write: needs_target,
            acquired: Mutex::new(false),
        });

        Ok(ImportedDmabuf {
            texture,
            size,
            format: dmabuf.format().code,
            wgpu_format,
            flipped: dmabuf.y_inverted(),
            sync,
        })
    }

    #[cfg(feature = "backend_drm")]
    fn check_node(&self, dmabuf: &Dmabuf) -> Result<(), VulkanError> {
        let Some(node) = dmabuf.node() else {
            return Ok(());
        };
        self.nodes
            .iter()
            .flatten()
            .any(|device| *device == node.dev_id())
            .then_some(())
            .ok_or(VulkanError::DeviceMismatch)
    }

    #[cfg(not(feature = "backend_drm"))]
    fn check_node(&self, _dmabuf: &Dmabuf) -> Result<(), VulkanError> {
        Ok(())
    }
}

/// Synchronization state for an imported DMA-BUF.
pub(crate) struct VulkanSync {
    dmabuf: Dmabuf,
    identity: (u64, u64),
    image: vk::Image,
    write: bool,
    acquired: Mutex<bool>,
}

impl fmt::Debug for VulkanSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanSync")
            .field("image", &self.image)
            .field("write", &self.write)
            .finish_non_exhaustive()
    }
}

impl VulkanSync {
    pub(crate) fn identity(&self) -> (u64, u64) {
        self.identity
    }

    pub(crate) fn resting_state(&self) -> wgpu::TextureUses {
        RESTING_STATE
    }

    pub(crate) fn acquire(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), VulkanError> {
        let mut acquired = self.acquired.lock().unwrap();
        if *acquired {
            return Ok(());
        }

        wait_dmabuf(&self.dmabuf, self.write)?;
        transfer_ownership(device, queue, self.image, true)?;
        *acquired = true;
        Ok(())
    }

    pub(crate) fn release(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        submission: wgpu::SubmissionIndex,
    ) -> Result<(), VulkanError> {
        let mut acquired = self.acquired.lock().unwrap();
        if !*acquired {
            return Ok(());
        }

        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(VulkanError::WgpuWait)?;
        transfer_ownership(device, queue, self.image, false)?;
        *acquired = false;
        Ok(())
    }
}

fn supported_formats() -> &'static [(DrmFourcc, wgpu::TextureFormat)] {
    &[
        (DrmFourcc::Argb8888, wgpu::TextureFormat::Bgra8Unorm),
        (DrmFourcc::Xrgb8888, wgpu::TextureFormat::Bgra8Unorm),
        (DrmFourcc::Abgr8888, wgpu::TextureFormat::Rgba8Unorm),
        (DrmFourcc::Xbgr8888, wgpu::TextureFormat::Rgba8Unorm),
    ]
}

fn texture_format_as_raw(format: wgpu::TextureFormat) -> vk::Format {
    match format {
        wgpu::TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        wgpu::TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        _ => unreachable!("supported_formats only contains Vulkan formats handled here"),
    }
}

fn texture_uses(usage: wgpu::TextureUsages) -> wgpu::TextureUses {
    let mut uses = RESTING_STATE;
    if usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        uses |= wgpu::TextureUses::RESOURCE;
    }
    if usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        uses |= wgpu::TextureUses::COLOR_TARGET;
    }
    uses
}

fn modifier_properties(
    device: &hal::vulkan::Device,
    format: vk::Format,
) -> Vec<vk::DrmFormatModifierPropertiesEXT> {
    let instance = device.shared_instance().raw_instance();
    let physical_device = device.raw_physical_device();
    let modifier_count = {
        let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut properties = vk::FormatProperties2::default().push_next(&mut modifier_list);
        // SAFETY: properties and its pNext chain are valid for the duration of
        // this Vulkan capability query.
        unsafe { instance.get_physical_device_format_properties2(physical_device, format, &mut properties) };
        modifier_list.drm_format_modifier_count
    };

    let mut modifiers = vec![vk::DrmFormatModifierPropertiesEXT::default(); modifier_count as usize];
    let mut modifier_list =
        vk::DrmFormatModifierPropertiesListEXT::default().drm_format_modifier_properties(&mut modifiers);
    let mut properties = vk::FormatProperties2::default().push_next(&mut modifier_list);
    // SAFETY: modifier_list points at the allocation above and Vulkan writes at
    // most its declared element count.
    unsafe { instance.get_physical_device_format_properties2(physical_device, format, &mut properties) };
    let returned_count = modifier_list.drm_format_modifier_count as usize;
    modifiers.truncate(returned_count);
    modifiers
}

fn supports_image(
    device: &hal::vulkan::Device,
    format: vk::Format,
    modifier: DrmModifier,
    usage: vk::ImageUsageFlags,
) -> bool {
    let mut modifier_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(u64::from(modifier))
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut external_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let image_info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .flags(vk::ImageCreateFlags::empty())
        .push_next(&mut modifier_info)
        .push_next(&mut external_info);
    let mut external_properties = vk::ExternalImageFormatProperties::default();
    let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external_properties);
    // SAFETY: image_info and every structure in its pNext chain remain alive
    // for the capability query; no handles are created.
    let result = unsafe {
        device
            .shared_instance()
            .raw_instance()
            .get_physical_device_image_format_properties2(
                device.raw_physical_device(),
                &image_info,
                &mut properties,
            )
    };
    result.is_ok()
        && external_properties
            .external_memory_properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
}

#[cfg(feature = "backend_drm")]
fn physical_device_nodes(device: &hal::vulkan::Device) -> [Option<libc::dev_t>; 2] {
    let mut drm_properties = vk::PhysicalDeviceDrmPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut drm_properties);
    // SAFETY: properties and drm_properties remain alive for this read-only
    // physical-device query.
    unsafe {
        device
            .shared_instance()
            .raw_instance()
            .get_physical_device_properties2(device.raw_physical_device(), &mut properties)
    };
    [
        (drm_properties.has_primary == vk::TRUE).then(|| {
            libc::makedev(
                drm_properties.primary_major as _,
                drm_properties.primary_minor as _,
            )
        }),
        (drm_properties.has_render == vk::TRUE)
            .then(|| libc::makedev(drm_properties.render_major as _, drm_properties.render_minor as _)),
    ]
}

fn wait_dmabuf(dmabuf: &Dmabuf, write: bool) -> Result<(), VulkanError> {
    let events = if write { PollFlags::OUT } else { PollFlags::IN };
    for handle in dmabuf.handles() {
        loop {
            let mut poll_fd = [PollFd::new(&handle, events)];
            match rustix::event::poll(&mut poll_fd, None) {
                Ok(_) => {
                    let ready = poll_fd[0].revents();
                    if ready.intersects(PollFlags::ERR | PollFlags::HUP | PollFlags::NVAL) {
                        return Err(VulkanError::DmabufWait(std::io::Error::other(format!(
                            "DMA-BUF poll returned {ready:?}"
                        ))));
                    }
                    if ready.contains(events) {
                        break;
                    }
                }
                Err(rustix::io::Errno::INTR) => continue,
                Err(err) => return Err(VulkanError::DmabufWait(err.into())),
            }
        }
    }
    Ok(())
}

fn transfer_ownership(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: vk::Image,
    acquire: bool,
) -> Result<(), VulkanError> {
    // SAFETY: The supplied WGPU device keeps the dispatch table and Vulkan device
    // alive through recording and submission below.
    let device_guard = unsafe { device.as_hal::<Vulkan>() }.ok_or(VulkanError::NotVulkan)?;
    let raw = device_guard.raw_device().clone();
    let family = device_guard.queue_family_index();
    drop(device_guard);

    let (source_family, destination_family) = if acquire {
        (vk::QUEUE_FAMILY_FOREIGN_EXT, family)
    } else {
        (family, vk::QUEUE_FAMILY_FOREIGN_EXT)
    };
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(if acquire {
            vk::AccessFlags::empty()
        } else {
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        })
        .dst_access_mask(if acquire {
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
        } else {
            vk::AccessFlags::empty()
        })
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(source_family)
        .dst_queue_family_index(destination_family)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Smithay WGPU DMA-BUF ownership transfer"),
    });
    // SAFETY: The callback records one barrier into WGPU's active Vulkan
    // command buffer. It neither ends nor destroys the buffer, and encoder is
    // not otherwise used until the callback returns.
    let recorded: Result<(), VulkanError> = unsafe {
        encoder.as_hal_mut::<Vulkan, _, _>(|hal_encoder| {
            let hal_encoder = hal_encoder.ok_or(VulkanError::NotVulkan)?;
            // SAFETY: raw_handle is borrowed for this recording call only.
            let command_buffer = hal_encoder.raw_handle();
            // SAFETY: command_buffer is recording, image remains alive, and
            // the barrier's subresource range covers the imported image.
            raw.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            Ok(())
        })
    };
    recorded?;

    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .map(|_| ())
        .map_err(VulkanError::WgpuWait)
}
