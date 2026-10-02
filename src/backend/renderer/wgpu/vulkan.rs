use std::{
    fmt,
    os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
    sync::{Arc, Mutex},
};

use ash::{ext, khr, vk};
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
use rustix::event::{PollFd, PollFlags};
use rustix::ioctl::{Setter, Updater};
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
    /// Exchanging or waiting for implicit DMA-BUF synchronization failed.
    #[error("DMA-BUF synchronization failed: {0}")]
    DmabufSync(#[source] std::io::Error),
    /// Creating a Vulkan semaphore failed.
    #[error("Creating a Vulkan semaphore failed: {0}")]
    CreateSemaphore(#[source] vk::Result),
    /// Importing a native fence into Vulkan failed.
    #[error("Importing a native fence into Vulkan failed: {0}")]
    ImportSemaphore(#[source] vk::Result),
    /// Exporting a native fence from Vulkan failed.
    #[error("Exporting a native fence from Vulkan failed: {0}")]
    ExportSemaphore(#[source] vk::Result),
    /// Waiting for WGPU work failed.
    #[error("Waiting for WGPU work failed: {0}")]
    WgpuWait(#[source] wgpu::PollError),
}

/// Request a WGPU Vulkan device capable of importing DMA-BUFs.
///
/// In addition to WGPU's external-memory feature this enables
/// `VK_EXT_queue_family_foreign`, which preserves image contents across API
/// ownership transfers, and `VK_KHR_external_semaphore_fd`, which exchanges
/// native fences with implicit DMA-BUF synchronization.
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
    if !adapter_guard
        .physical_device_capabilities()
        .supports_extension(khr::external_semaphore_fd::NAME)
    {
        return Err(VulkanError::MissingExtension("VK_KHR_external_semaphore_fd"));
    }
    if !supports_sync_file(
        adapter_guard.shared_instance().raw_instance(),
        adapter_guard.raw_physical_device(),
    ) {
        return Err(VulkanError::MissingExtension(
            "VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT",
        ));
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
                if !args.extensions.contains(&khr::external_semaphore_fd::NAME) {
                    args.extensions.push(khr::external_semaphore_fd::NAME);
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
        if !device_guard
            .enabled_device_extensions()
            .contains(&khr::external_semaphore_fd::NAME)
        {
            return Err(VulkanError::MissingExtension("VK_KHR_external_semaphore_fd"));
        }
        if !supports_sync_file(
            device_guard.shared_instance().raw_instance(),
            device_guard.raw_physical_device(),
        ) {
            return Err(VulkanError::MissingExtension(
                "VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT",
            ));
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
        let bytes_per_pixel = if wgpu_format == wgpu::TextureFormat::Rgba16Float {
            8
        } else {
            4
        };
        let minimum_stride = width
            .checked_mul(bytes_per_pixel)
            .ok_or(VulkanError::InvalidStride)?;
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
            texture: texture.clone(),
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
    texture: wgpu::Texture,
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

        let command = ownership_transfer(device, self.image, true)?;
        let semaphore = match export_dmabuf_sync_file(&self.dmabuf, self.write) {
            Ok(sync_file) => {
                let (raw, semaphore) = import_sync_file(device, sync_file)?;

                // SAFETY: semaphore belongs to this queue's Vulkan device and
                // remains alive until the submission which consumes its
                // temporary payload has completed. An intervening submission
                // may consume the wait, but the queue's ordering contract still
                // places the ownership transfer after it.
                let queue_guard = unsafe { queue.as_hal::<Vulkan>() }.ok_or_else(|| {
                    // SAFETY: semaphore has not been submitted and has no pending users.
                    unsafe { raw.destroy_semaphore(semaphore, None) };
                    VulkanError::NotVulkan
                })?;
                queue_guard.add_wait_semaphore(semaphore, None, vk::PipelineStageFlags::ALL_COMMANDS);
                drop(queue_guard);
                Some((raw, semaphore))
            }
            Err(VulkanError::DmabufSync(err)) if err.raw_os_error() == Some(libc::ENOTTY) => {
                wait_dmabuf(&self.dmabuf, self.write)?;
                None
            }
            Err(err) => return Err(err),
        };

        queue.submit([command]);
        retain_after_submission(queue, self.texture.clone());
        if let Some((raw, semaphore)) = semaphore {
            destroy_after_submission(queue, raw, semaphore);
        }
        *acquired = true;
        Ok(())
    }

    pub(crate) fn release(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _submission: wgpu::SubmissionIndex,
    ) -> Result<(), VulkanError> {
        let mut acquired = self.acquired.lock().unwrap();
        if !*acquired {
            return Ok(());
        }

        // WGPU serializes queue submissions. This barrier therefore follows
        // the renderer submission even when cloned queues are in use.
        let command = ownership_transfer(device, self.image, false)?;
        let release_submission = queue.submit([command]);
        retain_after_submission(queue, self.texture.clone());
        *acquired = false;

        // Signal on a later submission instead of attaching the semaphore to
        // the release submission. If another thread consumes the staged signal,
        // that submission is still ordered after the ownership transfer.
        let sync_file = match export_queue_sync_file(device, queue) {
            Ok(sync_file) => sync_file,
            Err(err) => {
                wait_for_submission(device, &release_submission)?;
                return Err(err);
            }
        };
        if let Some(sync_file) = sync_file {
            match import_dmabuf_sync_file(&self.dmabuf, self.write, &sync_file) {
                Ok(()) => {}
                Err(VulkanError::DmabufSync(err)) if err.raw_os_error() == Some(libc::ENOTTY) => {
                    if wait_sync_file(&sync_file).is_err() {
                        wait_for_submission(device, &release_submission)?;
                    }
                }
                Err(err) => {
                    if wait_sync_file(&sync_file).is_err() {
                        wait_for_submission(device, &release_submission)?;
                    }
                    return Err(err);
                }
            }
        }
        Ok(())
    }
}

/// Returns a native fence covering every WGPU submission queued before this call.
///
/// `None` denotes an already-signaled fence, as permitted for Vulkan sync FDs.
pub(super) fn export_queue_sync_file(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<Option<OwnedFd>, VulkanError> {
    let (raw, external, semaphore) = create_exportable_semaphore(device)?;

    // SAFETY: semaphore was created by this queue's Vulkan device. A cloned
    // queue may consume it first, but any such submission is ordered after all
    // work submitted before this function was entered.
    let queue_guard = unsafe { queue.as_hal::<Vulkan>() }.ok_or_else(|| {
        // SAFETY: semaphore has not been submitted and has no pending users.
        unsafe { raw.destroy_semaphore(semaphore, None) };
        VulkanError::NotVulkan
    })?;
    queue_guard.add_signal_semaphore(semaphore, None);
    drop(queue_guard);

    queue.submit([]);
    let get_info = vk::SemaphoreGetFdInfoKHR::default()
        .semaphore(semaphore)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    // SAFETY: the semaphore was created exportable as SYNC_FD and has a
    // submitted signal operation pending (or is already signaled).
    let fd = unsafe { external.get_semaphore_fd(&get_info) };
    destroy_after_submission(queue, raw, semaphore);
    let fd = fd.map_err(VulkanError::ExportSemaphore)?;
    if fd == -1 {
        Ok(None)
    } else {
        // SAFETY: Vulkan returned ownership of this valid file descriptor.
        Ok(Some(unsafe { OwnedFd::from_raw_fd(fd) }))
    }
}

fn supported_formats() -> &'static [(DrmFourcc, wgpu::TextureFormat)] {
    &[
        (DrmFourcc::Argb8888, wgpu::TextureFormat::Bgra8Unorm),
        (DrmFourcc::Xrgb8888, wgpu::TextureFormat::Bgra8Unorm),
        (DrmFourcc::Abgr8888, wgpu::TextureFormat::Rgba8Unorm),
        (DrmFourcc::Xbgr8888, wgpu::TextureFormat::Rgba8Unorm),
        (DrmFourcc::Abgr2101010, wgpu::TextureFormat::Rgb10a2Unorm),
        (DrmFourcc::Xbgr2101010, wgpu::TextureFormat::Rgb10a2Unorm),
        (DrmFourcc::Abgr16161616f, wgpu::TextureFormat::Rgba16Float),
        (DrmFourcc::Xbgr16161616f, wgpu::TextureFormat::Rgba16Float),
    ]
}

fn texture_format_as_raw(format: wgpu::TextureFormat) -> vk::Format {
    match format {
        wgpu::TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        wgpu::TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        wgpu::TextureFormat::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        wgpu::TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
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

fn supports_sync_file(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> bool {
    let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    let mut properties = vk::ExternalSemaphoreProperties::default();
    // SAFETY: info and properties remain valid for this read-only capability
    // query and adapter supplies a live physical-device handle.
    unsafe {
        instance.get_physical_device_external_semaphore_properties(physical_device, &info, &mut properties)
    };
    properties.external_semaphore_features.contains(
        vk::ExternalSemaphoreFeatureFlags::IMPORTABLE | vk::ExternalSemaphoreFeatureFlags::EXPORTABLE,
    )
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

fn ownership_transfer(
    device: &wgpu::Device,
    image: vk::Image,
    acquire: bool,
) -> Result<wgpu::CommandBuffer, VulkanError> {
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

    Ok(encoder.finish())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DmabufSyncFile {
    flags: u32,
    fd: i32,
}

const DMA_BUF_SYNC_READ: u32 = 1;
const DMA_BUF_SYNC_WRITE: u32 = 2;
const DMA_BUF_EXPORT_SYNC_FILE: rustix::ioctl::Opcode =
    rustix::ioctl::opcode::read_write::<DmabufSyncFile>(b'b', 2);
const DMA_BUF_IMPORT_SYNC_FILE: rustix::ioctl::Opcode =
    rustix::ioctl::opcode::write::<DmabufSyncFile>(b'b', 3);

fn dmabuf_sync_flags(write: bool) -> u32 {
    if write {
        DMA_BUF_SYNC_WRITE
    } else {
        DMA_BUF_SYNC_READ
    }
}

fn export_dmabuf_sync_file(dmabuf: &Dmabuf, write: bool) -> Result<OwnedFd, VulkanError> {
    let handle = dmabuf
        .handles()
        .next()
        .expect("single-plane DMA-BUF has one file descriptor");
    let mut request = DmabufSyncFile {
        flags: dmabuf_sync_flags(write),
        fd: -1,
    };
    loop {
        // SAFETY: DMA_BUF_EXPORT_SYNC_FILE reads and updates exactly this UAPI
        // structure; handle is a live DMA-BUF file descriptor.
        let result = unsafe {
            rustix::ioctl::ioctl(handle, Updater::<DMA_BUF_EXPORT_SYNC_FILE, _>::new(&mut request))
        };
        match result {
            Ok(()) => break,
            Err(rustix::io::Errno::INTR | rustix::io::Errno::AGAIN) => continue,
            Err(err) => return Err(VulkanError::DmabufSync(err.into())),
        }
    }
    if request.fd < 0 {
        return Err(VulkanError::DmabufSync(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "DMA_BUF_IOCTL_EXPORT_SYNC_FILE returned an invalid file descriptor",
        )));
    }
    // SAFETY: a successful export ioctl returned ownership of this descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(request.fd) })
}

fn import_dmabuf_sync_file(dmabuf: &Dmabuf, write: bool, sync_file: &OwnedFd) -> Result<(), VulkanError> {
    let handle = dmabuf
        .handles()
        .next()
        .expect("single-plane DMA-BUF has one file descriptor");
    let request = DmabufSyncFile {
        flags: dmabuf_sync_flags(write),
        fd: sync_file.as_raw_fd(),
    };
    loop {
        // SAFETY: DMA_BUF_IMPORT_SYNC_FILE reads exactly this UAPI structure;
        // both descriptors remain live for the ioctl call.
        let result =
            unsafe { rustix::ioctl::ioctl(handle, Setter::<DMA_BUF_IMPORT_SYNC_FILE, _>::new(request)) };
        match result {
            Ok(()) => return Ok(()),
            Err(rustix::io::Errno::INTR | rustix::io::Errno::AGAIN) => continue,
            Err(err) => return Err(VulkanError::DmabufSync(err.into())),
        }
    }
}

fn wait_dmabuf(dmabuf: &Dmabuf, write: bool) -> Result<(), VulkanError> {
    let events = if write { PollFlags::OUT } else { PollFlags::IN };
    for handle in dmabuf.handles() {
        wait_fd(handle, events)?;
    }
    Ok(())
}

fn wait_sync_file(sync_file: &OwnedFd) -> Result<(), VulkanError> {
    wait_fd(sync_file, PollFlags::IN)
}

fn wait_fd(fd: impl std::os::fd::AsFd, events: PollFlags) -> Result<(), VulkanError> {
    loop {
        let mut poll_fd = [PollFd::new(&fd, events)];
        match rustix::event::poll(&mut poll_fd, None) {
            Ok(_) => {
                let ready = poll_fd[0].revents();
                if ready.intersects(PollFlags::ERR | PollFlags::HUP | PollFlags::NVAL) {
                    return Err(VulkanError::DmabufSync(std::io::Error::other(format!(
                        "native fence poll returned {ready:?}"
                    ))));
                }
                if ready.contains(events) {
                    return Ok(());
                }
            }
            Err(rustix::io::Errno::INTR) => continue,
            Err(err) => return Err(VulkanError::DmabufSync(err.into())),
        }
    }
}

fn import_sync_file(
    device: &wgpu::Device,
    sync_file: OwnedFd,
) -> Result<(ash::Device, vk::Semaphore), VulkanError> {
    // SAFETY: the guard is used to clone live dispatch handles and is dropped
    // before any WGPU call.
    let device_guard = unsafe { device.as_hal::<Vulkan>() }.ok_or(VulkanError::NotVulkan)?;
    if !device_guard
        .enabled_device_extensions()
        .contains(&khr::external_semaphore_fd::NAME)
    {
        return Err(VulkanError::MissingExtension("VK_KHR_external_semaphore_fd"));
    }
    let raw = device_guard.raw_device().clone();
    let instance = device_guard.shared_instance().raw_instance().clone();
    drop(device_guard);

    // SAFETY: raw is a live Vulkan device and the default create info is valid.
    let semaphore = unsafe { raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
        .map_err(VulkanError::CreateSemaphore)?;
    let external = khr::external_semaphore_fd::Device::new(&instance, &raw);
    let fd = sync_file.into_raw_fd();
    let import_info = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(semaphore)
        .flags(vk::SemaphoreImportFlags::TEMPORARY)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .fd(fd);
    // SAFETY: semaphore is idle and fd owns a Linux sync_file. On success the
    // Vulkan implementation consumes fd; on failure ownership remains here.
    if let Err(err) = unsafe { external.import_semaphore_fd(&import_info) } {
        // SAFETY: Vulkan did not consume fd when import failed.
        drop(unsafe { OwnedFd::from_raw_fd(fd) });
        // SAFETY: semaphore is idle and has never been submitted.
        unsafe { raw.destroy_semaphore(semaphore, None) };
        return Err(VulkanError::ImportSemaphore(err));
    }
    Ok((raw, semaphore))
}

fn create_exportable_semaphore(
    device: &wgpu::Device,
) -> Result<(ash::Device, khr::external_semaphore_fd::Device, vk::Semaphore), VulkanError> {
    // SAFETY: the guard is used to clone live dispatch handles and is dropped
    // before any WGPU call.
    let device_guard = unsafe { device.as_hal::<Vulkan>() }.ok_or(VulkanError::NotVulkan)?;
    if !device_guard
        .enabled_device_extensions()
        .contains(&khr::external_semaphore_fd::NAME)
    {
        return Err(VulkanError::MissingExtension("VK_KHR_external_semaphore_fd"));
    }
    let raw = device_guard.raw_device().clone();
    let instance = device_guard.shared_instance().raw_instance().clone();
    drop(device_guard);

    let mut export_info =
        vk::ExportSemaphoreCreateInfo::default().handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
    // SAFETY: raw is live and the enabled external-semaphore extension supports
    // exporting binary semaphores as sync FDs.
    let semaphore =
        unsafe { raw.create_semaphore(&create_info, None) }.map_err(VulkanError::CreateSemaphore)?;
    let external = khr::external_semaphore_fd::Device::new(&instance, &raw);
    Ok((raw, external, semaphore))
}

fn destroy_after_submission(queue: &wgpu::Queue, raw: ash::Device, semaphore: vk::Semaphore) {
    queue.on_submitted_work_done(move || {
        // SAFETY: the callback runs only after every submission made before it
        // has completed, including the one which consumed this semaphore.
        unsafe { raw.destroy_semaphore(semaphore, None) };
    });
}

fn retain_after_submission(queue: &wgpu::Queue, texture: wgpu::Texture) {
    queue.on_submitted_work_done(move || drop(texture));
}

fn wait_for_submission(device: &wgpu::Device, submission: &wgpu::SubmissionIndex) -> Result<(), VulkanError> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission.clone()),
            timeout: None,
        })
        .map(|_| ())
        .map_err(VulkanError::WgpuWait)
}
