//! Premultiplied colors and fixed input transforms for linear composition.
use std::ops::Mul;

/// A four-component color representing pre-multiplied RGBA color values
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Color32F([f32; 4]);

impl Color32F {
    /// Initialize a new [`Color32F`]
    #[inline]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self([r, g, b, a])
    }
}

impl Color32F {
    /// Transparent color
    pub const TRANSPARENT: Color32F = Color32F::new(0.0, 0.0, 0.0, 0.0);

    /// Solid black color
    pub const BLACK: Color32F = Color32F::new(0f32, 0f32, 0f32, 1f32);
}

impl Color32F {
    /// Red color component
    #[inline]
    pub fn r(&self) -> f32 {
        self.0[0]
    }

    /// Green color component
    #[inline]
    pub fn g(&self) -> f32 {
        self.0[1]
    }

    /// Blue color component
    #[inline]
    pub fn b(&self) -> f32 {
        self.0[2]
    }

    /// Alpha color component
    #[inline]
    pub fn a(&self) -> f32 {
        self.0[3]
    }

    /// Color components
    #[inline]
    pub fn components(self) -> [f32; 4] {
        self.0
    }
}

impl Color32F {
    /// Test if the color represents a opaque color
    #[inline]
    pub fn is_opaque(&self) -> bool {
        self.a() == 1f32
    }
}

impl From<[f32; 4]> for Color32F {
    #[inline]
    fn from(value: [f32; 4]) -> Self {
        Self(value)
    }
}

impl Mul<f32> for Color32F {
    type Output = Color32F;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.r() * rhs, self.g() * rhs, self.b() * rhs, self.a() * rhs)
    }
}

/// Fixed input transform for premultiplied colors, before linear-light blending.
/// Transfer identifiers match color-management-v1; zero is already linear.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorTransform {
    /// Column-major source-linear RGB to working RGB matrix.
    pub matrix: [f32; 9],
    /// Named transfer identifier, or zero for identity.
    pub transfer: u32,
    /// Source black in cd/m².
    pub min_luminance: f32,
    /// Source peak in cd/m² (80 for extended linear scRGB).
    pub max_luminance: f32,
    /// Absolute luminance to working units, including SDR white adjustment.
    pub luminance_scale: f32,
    /// Source reference white in cd/m², for relative SDR appearance matching.
    pub reference_luminance: f32,
    /// Source RGB to Y coefficients for the HLG display OOTF.
    pub luma: [f32; 3],
}

impl ColorTransform {
    /// Identity, including extended negative and above-white values.
    pub const IDENTITY: Self = Self {
        matrix: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
        transfer: 0,
        min_luminance: 0.,
        max_luminance: 1.,
        luminance_scale: 1.,
        reference_luminance: 1.,
        luma: [0.2126, 0.7152, 0.0722],
    };

    /// Apply an output's relative SDR white and saturation to a tagged SDR
    /// source. Extended linear, PQ and HLG sources retain absolute luminance.
    pub fn for_source(self, mut source: Self) -> Self {
        if !matches!(source.transfer, 0 | 5 | 11 | 13) {
            source.luminance_scale *=
                self.luminance_scale * self.reference_luminance * 80.0 / source.reference_luminance.max(1e-6);
            let matrix = source.matrix;
            source.matrix = std::array::from_fn(|index| {
                let row = index % 3;
                let column = index / 3;
                (0..3)
                    .map(|i| self.matrix[i * 3 + row] * matrix[column * 3 + i])
                    .sum()
            });
        }
        source
    }

    /// Five vec4 uniforms, without heap storage or matrix transposition.
    pub fn to_uniforms(self) -> [[f32; 4]; 5] {
        [
            [self.matrix[0], self.matrix[1], self.matrix[2], 0.],
            [self.matrix[3], self.matrix[4], self.matrix[5], 0.],
            [self.matrix[6], self.matrix[7], self.matrix[8], 0.],
            [
                self.transfer as f32,
                self.min_luminance,
                self.max_luminance,
                self.luminance_scale,
            ],
            [self.luma[0], self.luma[1], self.luma[2], 0.],
        ]
    }

    /// Convert premultiplied RGBA, retaining extended range until presentation.
    pub fn apply(self, color: [f32; 4]) -> [f32; 4] {
        if self.transfer == 0 {
            return color;
        }
        let a = color[3];
        if a <= 0. {
            return [0.; 4];
        }
        let mut rgb = [color[0] / a, color[1] / a, color[2] / a];
        if self.transfer == 1 {
            let black = self.min_luminance.powf(1. / 2.4);
            let range = self.max_luminance.powf(1. / 2.4) - black;
            rgb = rgb.map(|v| (v.max(0.) * range + black).powf(2.4));
        } else {
            rgb = rgb.map(|v| decode(self.transfer, v));
            if self.transfer == 13 {
                let y = (rgb[0] * self.luma[0] + rgb[1] * self.luma[1] + rgb[2] * self.luma[2])
                    .max(0.)
                    .powf(0.2);
                rgb = rgb.map(|v| v * y);
            }
            rgb = rgb.map(|v| self.min_luminance + v * (self.max_luminance - self.min_luminance));
        }
        let result: [f32; 3] = std::array::from_fn(|row| {
            (0..3)
                .map(|col| self.matrix[col * 3 + row] * rgb[col])
                .sum::<f32>()
                * self.luminance_scale
                * a
        });
        [result[0], result[1], result[2], a]
    }
}

fn decode(transfer: u32, value: f32) -> f32 {
    let positive = value.max(0.);
    let piece = |v: f32, cutoff: f32, slope: f32, a: f32, gamma: f32| {
        if v <= cutoff {
            v / slope
        } else {
            ((v + a - 1.) / a).powf(gamma)
        }
    };
    match transfer {
        2 => positive.powf(2.2),
        3 => positive.powf(2.8),
        4 => piece(positive, 0.0912, 4., 1.1115, 1. / 0.45),
        5 => value,
        6 => {
            if value <= 0. {
                0.
            } else {
                10_f32.powf(2. * (value - 1.))
            }
        }
        7 => {
            if value <= 0. {
                0.
            } else {
                10_f32.powf(2.5 * (value - 1.))
            }
        }
        8 => value.signum() * piece(value.abs(), 0.081, 4.5, 1.099, 1. / 0.45),
        9 | 14 => piece(positive, 0.04045, 12.92, 1.055, 2.4),
        10 => value.signum() * piece(value.abs(), 0.04045, 12.92, 1.055, 2.4),
        11 => {
            let p = positive.powf(32. / 2523.);
            ((p - 3424. / 4096.).max(0.) / (2413. / 128. - 2392. / 128. * p).max(1e-12)).powf(16384. / 2610.)
        }
        12 => positive.powf(2.6) * 52.37 / 48.,
        13 => {
            if positive <= 0.5 {
                positive * positive / 3.
            } else {
                (((positive - 0.559_910_7) / 0.17883277).exp() + 0.28466892) / 12.
            }
        }
        _ => value,
    }
}
