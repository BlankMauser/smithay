//! Connector color state applied with the surface's atomic modeset.

use std::{ops::RangeInclusive, os::fd::AsFd, sync::Arc};

use drm::control::{Device as ControlDevice, connector, crtc, property};

use crate::{
    backend::drm::{
        DrmDeviceFd,
        error::{AccessError, Error},
    },
    utils::DevPath,
};

/// RGB encoding selected by the connector's `Colorspace` property.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// The connector's default RGB encoding.
    #[default]
    Default,
    /// BT.2020 RGB, including HDR10 output.
    Bt2020Rgb,
}

impl ColorSpace {
    const ALL: [Self; 2] = [Self::Default, Self::Bt2020Rgb];

    fn index(self) -> usize {
        match self {
            Self::Default => 0,
            Self::Bt2020Rgb => 1,
        }
    }

    fn name(self) -> &'static [u8] {
        match self {
            Self::Default => b"Default",
            Self::Bt2020Rgb => b"BT2020_RGB",
        }
    }
}

/// EOTF identifiers defined by CTA-861 static HDR metadata type 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HdrEotf {
    /// Traditional SDR gamma.
    TraditionalSdr = 0,
    /// Traditional HDR gamma.
    TraditionalHdr = 1,
    /// SMPTE ST 2084 perceptual quantizer.
    Pq = 2,
    /// Hybrid log-gamma.
    Hlg = 3,
}

/// Static HDR metadata, in physical units rather than kernel quantization units.
///
/// Values are validated when staging a [`ConnectorColorState`]. Unknown content
/// light levels can be zero. This describes the mastering display and content;
/// it does not change framebuffer pixel encoding or discover display support.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdrMetadata {
    /// Transfer function of the signal sent to the display.
    pub eotf: HdrEotf,
    /// CIE 1931 xy coordinates, in red, green, blue order.
    pub display_primaries: [[f64; 2]; 3],
    /// CIE 1931 xy coordinates of the white point.
    pub white_point: [f64; 2],
    /// Minimum mastering luminance in cd/m², encoded in units of 0.0001 cd/m².
    pub min_luminance: f64,
    /// Maximum mastering luminance in cd/m².
    pub max_luminance: f64,
    /// Maximum content light level in cd/m², or zero when unknown.
    pub max_cll: f64,
    /// Maximum frame-average light level in cd/m², or zero when unknown.
    pub max_fall: f64,
}

impl HdrMetadata {
    fn encode(self) -> Result<[u8; 32], &'static str> {
        // struct hdr_output_metadata has a u32 type, a 26-byte type-1 payload
        // and two trailing padding bytes. Write every byte explicitly, without
        // exposing Rust/C padding or depending on the host struct layout.
        let mut bytes = [0; 32];
        bytes[4] = self.eotf as u8;
        let mut offset = 6;
        for point in self.display_primaries.into_iter().chain([self.white_point]) {
            if point.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || point[0] + point[1] > 1.0
                || point[1] == 0.0
            {
                return Err("invalid HDR chromaticity coordinates");
            }
            for value in point {
                let encoded = (value * 50_000.0).round() as u16;
                bytes[offset..offset + 2].copy_from_slice(&encoded.to_ne_bytes());
                offset += 2;
            }
        }
        if self.max_luminance != 0.0 && self.min_luminance > self.max_luminance {
            return Err("minimum HDR luminance exceeds maximum luminance");
        }
        if self.max_cll != 0.0 && self.max_fall > self.max_cll {
            return Err("frame-average HDR light level exceeds content light level");
        }
        for (value, scale) in [
            (self.max_luminance, 1.0),
            (self.min_luminance, 10_000.0),
            (self.max_cll, 1.0),
            (self.max_fall, 1.0),
        ] {
            let scaled = value * scale;
            if !scaled.is_finite() || !(0.0..=u16::MAX as f64).contains(&scaled) {
                return Err("HDR luminance is outside the metadata range");
            }
            bytes[offset..offset + 2].copy_from_slice(&(scaled.round() as u16).to_ne_bytes());
            offset += 2;
        }
        Ok(bytes)
    }
}

/// Color properties to apply together with a connector's next framebuffer.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ConnectorColorState {
    /// Signal color encoding. Enum values are resolved by their kernel names.
    pub colorspace: ColorSpace,
    /// Maximum link bits per component. `None` leaves the current limit alone.
    pub max_bpc: Option<u64>,
    /// Static HDR metadata. `None` clears an existing metadata blob.
    pub hdr_metadata: Option<HdrMetadata>,
}

/// Driver support for connector color properties, independent of EDID support.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ConnectorColorCapabilities {
    /// Whether the connector exposes a writable `HDR_OUTPUT_METADATA` blob.
    pub hdr_metadata: bool,
    /// Supported RGB encodings, resolved from the `Colorspace` enum names.
    pub colorspaces: Vec<ColorSpace>,
    /// Inclusive range of the connector's writable `max bpc` property.
    pub max_bpc: Option<RangeInclusive<u64>>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct ColorValues {
    pub hdr_metadata: Option<u64>,
    pub colorspace: Option<u64>,
    pub max_bpc: Option<u64>,
}

#[derive(Debug)]
struct PropertyBlob {
    device: DrmDeviceFd,
    id: u64,
}

impl Drop for PropertyBlob {
    fn drop(&mut self) {
        let _ = self.device.destroy_property_blob(self.id);
    }
}

/// A one-shot CRTC gamma change, retained until its framebuffer commits.
#[derive(Debug, Clone)]
pub(super) struct GammaLut {
    pub entries: Option<Arc<[[u16; 3]]>>,
    blob: Option<Arc<PropertyBlob>>,
}

impl GammaLut {
    pub fn id(&self) -> u64 {
        self.blob.as_ref().map_or(0, |blob| blob.id)
    }

    pub fn prepare(
        device: &DrmDeviceFd,
        crtc: crtc::Handle,
        entries: Option<&[[u16; 3]]>,
    ) -> Result<Option<Self>, Error> {
        let access = |source| {
            Error::Access(AccessError {
                errmsg: "Error preparing atomic gamma transition",
                dev: device.dev_path(),
                source,
            })
        };
        let mut supported = false;
        let mut size = 0;
        for (handle, value) in device.get_properties(crtc).map_err(access)? {
            let info = device.get_property(handle).map_err(access)?;
            if info.name().to_bytes() == b"GAMMA_LUT"
                && info.mutable()
                && matches!(info.value_type(), property::ValueType::Blob)
            {
                supported = true;
            } else if info.name().to_bytes() == b"GAMMA_LUT_SIZE" {
                size = value;
            }
        }
        if !supported {
            return if entries.is_none() {
                Ok(None)
            } else {
                Err(Error::InvalidGammaLut {
                    crtc,
                    reason: "CRTC has no atomic gamma LUT",
                })
            };
        }
        let blob = if let Some(entries) = entries {
            if !valid_gamma_size(entries.len(), size) {
                return Err(Error::InvalidGammaLut {
                    crtc,
                    reason: "gamma table must match bounded GAMMA_LUT_SIZE",
                });
            }
            let mut bytes = encode_gamma(entries);
            let id = drm_ffi::mode::create_property_blob(device.as_fd(), &mut bytes)
                .map_err(access)?
                .blob_id;
            Some(Arc::new(PropertyBlob {
                device: device.clone(),
                id: u64::from(id),
            }))
        } else {
            None
        };
        Ok(Some(Self {
            entries: entries.map(Arc::from),
            blob,
        }))
    }
}

impl PartialEq for GammaLut {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

fn valid_gamma_size(length: usize, size: u64) -> bool {
    length > 0 && length <= 1 << 20 && length as u64 == size
}

fn encode_gamma(entries: &[[u16; 3]]) -> Vec<u8> {
    let mut bytes = vec![0; entries.len() * 8];
    for (entry, encoded) in entries.iter().zip(bytes.chunks_exact_mut(8)) {
        for (component, output) in entry.iter().zip(encoded.chunks_exact_mut(2)) {
            output.copy_from_slice(&component.to_ne_bytes());
        }
    }
    bytes
}

/// Owned blobs are shared by pending and committed state until both release them.
#[derive(Debug, Clone)]
pub(super) struct ConnectorColor {
    pub state: Option<ConnectorColorState>,
    pub values: ColorValues,
    pub default_colorspace: Option<u64>,
    _blob: Option<Arc<PropertyBlob>>,
}

impl PartialEq for ConnectorColor {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl ConnectorColor {
    pub fn reset_values(&self) -> ColorValues {
        ColorValues {
            hdr_metadata: self.values.hdr_metadata.map(|_| 0),
            colorspace: self.default_colorspace,
            max_bpc: None,
        }
    }
}

pub(super) struct ColorProperties {
    values: ColorValues,
    colorspaces: [Option<u64>; 2],
    max_bpc: Option<RangeInclusive<u64>>,
}

impl ColorProperties {
    pub fn read(device: &(impl ControlDevice + DevPath), conn: connector::Handle) -> Result<Self, Error> {
        let access = |source| {
            Error::Access(AccessError {
                errmsg: "Error reading connector color properties",
                dev: device.dev_path(),
                source,
            })
        };
        let props = device.get_properties(conn).map_err(access)?;
        let mut result = Self {
            values: ColorValues::default(),
            colorspaces: [None; 2],
            max_bpc: None,
        };
        for (handle, value) in props {
            let info = device.get_property(handle).map_err(access)?;
            if !info.mutable() {
                continue;
            }
            match (info.name().to_bytes(), info.value_type()) {
                (b"HDR_OUTPUT_METADATA", property::ValueType::Blob) => {
                    result.values.hdr_metadata = Some(value)
                }
                (b"Colorspace", property::ValueType::Enum(values)) => {
                    for item in values.values().1 {
                        for colorspace in ColorSpace::ALL {
                            if item.name().to_bytes() == colorspace.name() {
                                result.colorspaces[colorspace.index()] = Some(item.value());
                            }
                        }
                    }
                    result.values.colorspace = Some(value);
                }
                (b"max bpc", property::ValueType::UnsignedRange(min, max)) => {
                    result.max_bpc = Some(min..=max);
                    result.values.max_bpc = Some(value);
                }
                _ => {}
            }
        }
        Ok(result)
    }

    pub fn capabilities(&self) -> ConnectorColorCapabilities {
        ConnectorColorCapabilities {
            hdr_metadata: self.values.hdr_metadata.is_some(),
            colorspaces: ColorSpace::ALL
                .into_iter()
                .filter(|value| self.colorspaces[value.index()].is_some())
                .collect(),
            max_bpc: self.max_bpc.clone(),
        }
    }

    pub fn current(&self) -> ConnectorColor {
        ConnectorColor {
            state: None,
            values: self.values,
            default_colorspace: self.colorspaces[ColorSpace::Default.index()],
            _blob: None,
        }
    }

    fn validate(&self, conn: connector::Handle, state: ConnectorColorState) -> Result<(), Error> {
        let missing = |name| Error::UnknownProperty {
            handle: conn.into(),
            name,
        };
        if let Some(metadata) = state.hdr_metadata {
            if self.values.hdr_metadata.is_none() {
                return Err(missing("HDR_OUTPUT_METADATA"));
            }
            metadata.encode().map_err(|reason| Error::InvalidColorState {
                connector: conn,
                reason,
            })?;
        }
        if self.colorspaces[state.colorspace.index()].is_none()
            && (state.colorspace != ColorSpace::Default || self.values.colorspace.is_some())
        {
            return Err(Error::InvalidColorState {
                connector: conn,
                reason: "requested Colorspace enum is unavailable",
            });
        }
        if let Some(bpc) = state.max_bpc {
            let Some(range) = self.max_bpc.as_ref() else {
                return Err(missing("max bpc"));
            };
            if !range.contains(&bpc) {
                return Err(Error::InvalidColorState {
                    connector: conn,
                    reason: "max bpc is outside the connector range",
                });
            }
        }
        Ok(())
    }

    pub fn prepare(
        &self,
        device: &DrmDeviceFd,
        conn: connector::Handle,
        state: ConnectorColorState,
    ) -> Result<ConnectorColor, Error> {
        self.validate(conn, state)?;
        let blob = state
            .hdr_metadata
            .map(|metadata| {
                let bytes = metadata.encode().map_err(|reason| Error::InvalidColorState {
                    connector: conn,
                    reason,
                })?;
                let id = device.create_property_blob(&bytes).map_err(|source| {
                    Error::Access(AccessError {
                        errmsg: "Failed to create HDR metadata blob",
                        dev: device.dev_path(),
                        source,
                    })
                })?;
                Ok::<_, Error>(Arc::new(PropertyBlob {
                    device: device.clone(),
                    id: id.into(),
                }))
            })
            .transpose()?;
        Ok(ConnectorColor {
            state: Some(state),
            values: ColorValues {
                hdr_metadata: self
                    .values
                    .hdr_metadata
                    .map(|_| blob.as_ref().map_or(0, |blob| blob.id)),
                colorspace: self.colorspaces[state.colorspace.index()],
                max_bpc: state.max_bpc,
            },
            default_colorspace: self.colorspaces[ColorSpace::Default.index()],
            _blob: blob,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    fn hdr() -> HdrMetadata {
        HdrMetadata {
            eotf: HdrEotf::Pq,
            display_primaries: [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
            white_point: [0.3127, 0.3290],
            min_luminance: 0.005,
            max_luminance: 1_000.0,
            max_cll: 1_000.0,
            max_fall: 400.0,
        }
    }

    fn properties() -> ColorProperties {
        ColorProperties {
            values: ColorValues {
                hdr_metadata: Some(0),
                colorspace: Some(17),
                max_bpc: Some(8),
            },
            // Values deliberately differ from the kernel's usual enum numbering.
            colorspaces: [Some(17), Some(42)],
            max_bpc: Some(8..=12),
        }
    }

    fn connector() -> connector::Handle {
        NonZeroU32::new(1).unwrap().into()
    }

    #[test]
    fn hdr_metadata_matches_kernel_type_one_layout_and_units() {
        let bytes = hdr().encode().unwrap();
        assert_eq!(u32::from_ne_bytes(bytes[..4].try_into().unwrap()), 0);
        assert_eq!(bytes[4..6], [2, 0]);
        let words = bytes[6..30]
            .chunks_exact(2)
            .map(|word| u16::from_ne_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            words,
            [
                35400, 14600, 8500, 39850, 6550, 2300, 15635, 16450, 1000, 50, 1000, 400
            ]
        );
        assert_eq!(bytes[30..], [0, 0]);
    }

    #[test]
    fn hdr_metadata_rejects_nonfinite_and_unrepresentable_luminance() {
        for value in [f64::NAN, f64::INFINITY, -0.1, 65_536.0] {
            assert!(
                HdrMetadata {
                    max_luminance: value,
                    ..hdr()
                }
                .encode()
                .is_err()
            );
        }
        assert!(
            HdrMetadata {
                min_luminance: 6.554,
                ..hdr()
            }
            .encode()
            .is_err()
        );
        assert!(
            HdrMetadata {
                max_fall: 1_001.0,
                ..hdr()
            }
            .encode()
            .is_err()
        );
        assert!(
            HdrMetadata {
                max_luminance: 0.001,
                ..hdr()
            }
            .encode()
            .is_err()
        );
    }

    #[test]
    fn hdr_metadata_rejects_invalid_primaries() {
        for point in [[f64::NAN, 0.3], [0.7, 0.4], [-0.1, 0.3], [0.3, 0.0]] {
            assert!(
                HdrMetadata {
                    white_point: point,
                    ..hdr()
                }
                .encode()
                .is_err()
            );
        }
    }

    #[test]
    fn hdr_metadata_accepts_unspecified_light_levels_and_rounds_minimum() {
        let bytes = HdrMetadata {
            min_luminance: 0.00016,
            max_cll: 0.0,
            max_fall: 0.0,
            ..hdr()
        }
        .encode()
        .unwrap();
        assert_eq!(u16::from_ne_bytes(bytes[24..26].try_into().unwrap()), 2);
        assert_eq!(bytes[26..30], [0, 0, 0, 0]);
    }

    #[test]
    fn supported_ranges_are_checked_before_staging() {
        let props = properties();
        let conn = connector();
        for bpc in [8, 10, 12] {
            assert!(
                props
                    .validate(
                        conn,
                        ConnectorColorState {
                            max_bpc: Some(bpc),
                            ..Default::default()
                        }
                    )
                    .is_ok()
            );
        }
        for bpc in [0, 7, 13, u64::MAX] {
            assert!(matches!(
                props.validate(
                    conn,
                    ConnectorColorState {
                        max_bpc: Some(bpc),
                        ..Default::default()
                    }
                ),
                Err(Error::InvalidColorState { .. })
            ));
        }
        assert!(
            props
                .validate(
                    conn,
                    ConnectorColorState {
                        colorspace: ColorSpace::Bt2020Rgb,
                        hdr_metadata: Some(hdr()),
                        max_bpc: Some(10)
                    }
                )
                .is_ok()
        );
    }

    #[test]
    fn unsupported_properties_allow_only_sdr_reset() {
        let props = ColorProperties {
            values: ColorValues::default(),
            colorspaces: [None; 2],
            max_bpc: None,
        };
        assert!(
            props
                .validate(connector(), ConnectorColorState::default())
                .is_ok()
        );
        assert!(matches!(
            props.validate(
                connector(),
                ConnectorColorState {
                    hdr_metadata: Some(hdr()),
                    ..Default::default()
                }
            ),
            Err(Error::UnknownProperty {
                name: "HDR_OUTPUT_METADATA",
                ..
            })
        ));
        assert!(
            props
                .validate(
                    connector(),
                    ConnectorColorState {
                        colorspace: ColorSpace::Bt2020Rgb,
                        ..Default::default()
                    }
                )
                .is_err()
        );
        assert!(matches!(
            props.validate(
                connector(),
                ConnectorColorState {
                    max_bpc: Some(10),
                    ..Default::default()
                }
            ),
            Err(Error::UnknownProperty { name: "max bpc", .. })
        ));
    }

    #[test]
    fn reset_uses_discovered_default_and_clears_metadata_without_changing_link_depth() {
        let mut props = properties();
        props.values.hdr_metadata = Some(900);
        props.values.colorspace = Some(42);
        let current = props.current();
        assert_eq!(current.state, None);
        assert_eq!(
            current.reset_values(),
            ColorValues {
                hdr_metadata: Some(0),
                colorspace: Some(17),
                max_bpc: None
            }
        );
        assert_eq!(
            props.capabilities().colorspaces,
            [ColorSpace::Default, ColorSpace::Bt2020Rgb]
        );
    }
    #[test]
    fn gamma_blob_preserves_u16_channels_and_zero_padding() {
        let encoded = encode_gamma(&[[1, 2, 3], [65535, 32768, 0]]);
        let entries: Vec<u16> = encoded
            .chunks_exact(2)
            .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(entries, [1, 2, 3, 0, 65535, 32768, 0, 0]);
        assert!(valid_gamma_size(262145, 262145));
        assert!(!valid_gamma_size(0, 0));
        assert!(!valid_gamma_size(1024, 4096));
        assert!(!valid_gamma_size((1 << 20) + 1, (1 << 20) + 1));
    }
}
