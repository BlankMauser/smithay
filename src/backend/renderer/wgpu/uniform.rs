use std::borrow::Cow;

/// Value types supported by custom WGPU shader uniforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UniformType {
    /// A single float.
    _1f,
    /// Two floats.
    _2f,
    /// Three floats.
    _3f,
    /// Four floats.
    _4f,
}

/// A custom shader uniform value.
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform<'a> {
    /// Name declared when the shader was compiled.
    pub name: Cow<'a, str>,
    /// Value supplied for this draw.
    pub value: UniformValue,
}

/// A custom shader uniform declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformName<'a> {
    /// Name used to identify the uniform when drawing.
    pub name: Cow<'a, str>,
    /// Type of the uniform.
    pub type_: UniformType,
}

/// Value of a custom shader uniform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UniformValue {
    /// A single float.
    _1f(f32),
    /// Two floats.
    _2f(f32, f32),
    /// Three floats.
    _3f(f32, f32, f32),
    /// Four floats.
    _4f(f32, f32, f32, f32),
}

impl<'a> Uniform<'a> {
    /// Creates a uniform value.
    pub fn new(name: impl Into<Cow<'a, str>>, value: impl Into<UniformValue>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }

    /// Converts this value to an owned value.
    pub fn to_owned(&self) -> Uniform<'static> {
        Uniform {
            name: Cow::Owned(self.name.clone().into_owned()),
            value: self.value,
        }
    }

    /// Converts this value to an owned value.
    pub fn into_owned(self) -> Uniform<'static> {
        Uniform {
            name: Cow::Owned(self.name.into_owned()),
            value: self.value,
        }
    }
}

impl<'a> UniformName<'a> {
    /// Creates a uniform declaration.
    pub fn new(name: impl Into<Cow<'a, str>>, type_: UniformType) -> Self {
        Self {
            name: name.into(),
            type_,
        }
    }

    /// Converts this declaration to an owned declaration.
    pub fn to_owned(&self) -> UniformName<'static> {
        UniformName {
            name: Cow::Owned(self.name.clone().into_owned()),
            type_: self.type_,
        }
    }

    /// Converts this declaration to an owned declaration.
    pub fn into_owned(self) -> UniformName<'static> {
        UniformName {
            name: Cow::Owned(self.name.into_owned()),
            type_: self.type_,
        }
    }
}

impl UniformValue {
    pub(super) fn type_(&self) -> UniformType {
        match self {
            Self::_1f(_) => UniformType::_1f,
            Self::_2f(..) => UniformType::_2f,
            Self::_3f(..) => UniformType::_3f,
            Self::_4f(..) => UniformType::_4f,
        }
    }

    pub(super) fn components(&self) -> [f32; 4] {
        match *self {
            Self::_1f(x) => [x, 0.0, 0.0, 0.0],
            Self::_2f(x, y) => [x, y, 0.0, 0.0],
            Self::_3f(x, y, z) => [x, y, z, 0.0],
            Self::_4f(x, y, z, w) => [x, y, z, w],
        }
    }
}

impl From<f32> for UniformValue {
    fn from(value: f32) -> Self {
        Self::_1f(value)
    }
}

impl From<(f32, f32)> for UniformValue {
    fn from(value: (f32, f32)) -> Self {
        Self::_2f(value.0, value.1)
    }
}

impl From<(f32, f32, f32)> for UniformValue {
    fn from(value: (f32, f32, f32)) -> Self {
        Self::_3f(value.0, value.1, value.2)
    }
}

impl From<(f32, f32, f32, f32)> for UniformValue {
    fn from(value: (f32, f32, f32, f32)) -> Self {
        Self::_4f(value.0, value.1, value.2, value.3)
    }
}

impl From<[f32; 2]> for UniformValue {
    fn from(value: [f32; 2]) -> Self {
        Self::_2f(value[0], value[1])
    }
}

impl From<[f32; 3]> for UniformValue {
    fn from(value: [f32; 3]) -> Self {
        Self::_3f(value[0], value[1], value[2])
    }
}

impl From<[f32; 4]> for UniformValue {
    fn from(value: [f32; 4]) -> Self {
        Self::_4f(value[0], value[1], value[2], value[3])
    }
}
