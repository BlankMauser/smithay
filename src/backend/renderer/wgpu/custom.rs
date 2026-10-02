use std::{
    collections::HashMap,
    future::Future,
    mem,
    pin::pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

use super::{WgpuError, WgpuRenderer, uniform::*};
use crate::backend::renderer::{ColorTransform, ContextId};

pub(super) const UNIFORM_SLOTS: usize = 32;
pub(super) const CUSTOM_UNIFORM_SIZE: u64 = (UNIFORM_SLOTS * mem::size_of::<[f32; 4]>()) as u64;
pub(super) const COLOR_SIZE: u64 = (5 * mem::size_of::<[f32; 4]>()) as u64;
pub(super) const TOTAL_UNIFORM_SLOTS: usize = UNIFORM_SLOTS + 5;
pub(super) const UNIFORM_SIZE: u64 = CUSTOM_UNIFORM_SIZE + COLOR_SIZE;

pub(super) fn set_color_uniforms(data: &mut [f32; TOTAL_UNIFORM_SLOTS * 4], color: Option<ColorTransform>) {
    for (target, value) in data[UNIFORM_SLOTS * 4..]
        .chunks_exact_mut(4)
        .zip(color.unwrap_or(ColorTransform::IDENTITY).to_uniforms())
    {
        target.copy_from_slice(&value);
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct UniformDesc {
    pub(super) slot: usize,
    pub(super) type_: UniformType,
}

#[derive(Debug)]
pub(super) struct CustomProgram {
    module: ::wgpu::ShaderModule,
    context: ContextId<super::WgpuTexture>,
    uniforms: HashMap<String, UniformDesc>,
    pipelines: Mutex<HashMap<(::wgpu::TextureFormat, bool), ::wgpu::RenderPipeline>>,
}

impl CustomProgram {
    pub(super) fn compile(
        renderer: &WgpuRenderer,
        source: &str,
        additional_uniforms: &[UniformName<'_>],
    ) -> Result<Arc<Self>, WgpuError> {
        if additional_uniforms.len() >= UNIFORM_SLOTS {
            return Err(WgpuError::TooManyUniforms {
                actual: additional_uniforms.len(),
                maximum: UNIFORM_SLOTS - 1,
            });
        }
        let mut uniforms = HashMap::with_capacity(additional_uniforms.len());
        for (index, uniform) in additional_uniforms.iter().enumerate() {
            if uniform.name.is_empty() || uniforms.contains_key(uniform.name.as_ref()) {
                return Err(WgpuError::InvalidUniformName(uniform.name.clone().into_owned()));
            }
            uniforms.insert(
                uniform.name.clone().into_owned(),
                UniformDesc {
                    slot: index + 1,
                    type_: uniform.type_,
                },
            );
        }

        let source = format!(
            "{}\n{}\n{}",
            include_str!("color.wgsl"),
            include_str!("shaders.wgsl"),
            source
        );
        let module = color_managed_module(&source)?;
        let scope = renderer.device.push_error_scope(::wgpu::ErrorFilter::Validation);
        let module = renderer
            .device
            .create_shader_module(::wgpu::ShaderModuleDescriptor {
                label: Some("Smithay WGPU custom shader"),
                source: ::wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
            });
        wait_for_error_scope(&renderer.device, scope)?;
        Ok(Arc::new(Self {
            module,
            context: renderer.context_id.clone(),
            uniforms,
            pipelines: Mutex::new(HashMap::new()),
        }))
    }

    pub(super) fn check_context(&self, renderer: &WgpuRenderer) -> Result<(), WgpuError> {
        if self.context != renderer.context_id {
            return Err(WgpuError::ForeignProgram);
        }
        Ok(())
    }

    pub(super) fn pipeline(
        &self,
        renderer: &WgpuRenderer,
        format: ::wgpu::TextureFormat,
        blend: bool,
    ) -> Result<::wgpu::RenderPipeline, WgpuError> {
        self.check_context(renderer)?;
        let mut pipelines = self.pipelines.lock().unwrap();
        if let Some(pipeline) = pipelines.get(&(format, blend)) {
            return Ok(pipeline.clone());
        }

        let layouts = [
            Some(&renderer.texture_layout),
            Some(&renderer.uniform_layout),
            Some(&renderer.texture_layout),
        ];
        let layout = renderer
            .device
            .create_pipeline_layout(&::wgpu::PipelineLayoutDescriptor {
                label: Some("Smithay WGPU custom pipeline layout"),
                bind_group_layouts: &layouts,
                immediate_size: 0,
            });
        let scope = renderer.device.push_error_scope(::wgpu::ErrorFilter::Validation);
        let pipeline = renderer
            .device
            .create_render_pipeline(&::wgpu::RenderPipelineDescriptor {
                label: Some("Smithay WGPU custom pipeline"),
                layout: Some(&layout),
                vertex: ::wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vertex"),
                    compilation_options: Default::default(),
                    buffers: &[Some(super::vertex_buffer_layout())],
                },
                primitive: ::wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: ::wgpu::MultisampleState::default(),
                fragment: Some(::wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("custom_fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(::wgpu::ColorTargetState {
                        format,
                        blend: blend.then_some(::wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: ::wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        wait_for_error_scope(&renderer.device, scope)?;
        pipelines.insert((format, blend), pipeline.clone());
        Ok(pipeline)
    }

    pub(super) fn uniform_data(
        &self,
        size: [f32; 2],
        alpha: f32,
        tint: bool,
        additional_uniforms: &[Uniform<'_>],
        color: Option<ColorTransform>,
    ) -> Result<[f32; TOTAL_UNIFORM_SLOTS * 4], WgpuError> {
        let mut data = [0.0; TOTAL_UNIFORM_SLOTS * 4];
        set_color_uniforms(&mut data, color);
        data[..4].copy_from_slice(&[size[0], size[1], alpha, tint as u32 as f32]);
        for uniform in additional_uniforms {
            let Some(desc) = self.uniforms.get(uniform.name.as_ref()) else {
                return Err(WgpuError::UnknownUniform(uniform.name.clone().into_owned()));
            };
            let provided = uniform.value.type_();
            if provided != desc.type_ {
                return Err(WgpuError::UniformTypeMismatch {
                    provided,
                    declared: desc.type_,
                });
            }
            let start = desc.slot * 4;
            data[start..start + 4].copy_from_slice(&uniform.value.components());
        }
        Ok(data)
    }
}

/// A compiled custom WGPU texture shader.
#[derive(Debug, Clone)]
pub struct WgpuTexProgram(pub(super) Arc<CustomProgram>);

/// A compiled custom WGPU pixel shader.
#[derive(Debug, Clone)]
pub struct WgpuPixelProgram(pub(super) Arc<CustomProgram>);

fn wait_for_error_scope(device: &::wgpu::Device, scope: ::wgpu::ErrorScopeGuard) -> Result<(), WgpuError> {
    let mut future = pin!(scope.pop());
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(error) = future.as_mut().poll(&mut context) {
            return error.map_or(Ok(()), |error| Err(WgpuError::Shader(error.to_string())));
        }
        device.poll(::wgpu::PollType::wait_indefinitely())?;
    }
}

// Transform all exits of the fragment entry point, including early returns, without
// rewriting WGSL text or changing the custom shader's uniform/API contract.
fn color_managed_module(source: &str) -> Result<::wgpu::naga::Module, WgpuError> {
    use ::wgpu::naga::{Expression, ShaderStage, Statement};
    let mut module = ::wgpu::naga::front::wgsl::parse_str(source)
        .map_err(|error| WgpuError::Shader(error.emit_to_string(source)))?;
    let transform = module
        .functions
        .iter()
        .find_map(|(handle, function)| {
            (function.name.as_deref() == Some("smithay_color_transform")).then_some(handle)
        })
        .ok_or_else(|| WgpuError::Shader("Missing color transform".into()))?;
    let entry = module
        .entry_points
        .iter_mut()
        .find(|entry| entry.stage == ShaderStage::Fragment && entry.name == "custom_fragment")
        .ok_or_else(|| WgpuError::Shader("Missing custom_fragment entry point".into()))?;

    fn convert_returns(
        block: &mut ::wgpu::naga::Block,
        expressions: &mut ::wgpu::naga::Arena<Expression>,
        transform: ::wgpu::naga::Handle<::wgpu::naga::Function>,
    ) {
        for (mut statement, span) in mem::take(block).span_into_iter() {
            match &mut statement {
                Statement::Return { value: Some(value) } => {
                    let result = expressions.append(Expression::CallResult(transform), span);
                    block.push(
                        Statement::Call {
                            function: transform,
                            arguments: vec![*value],
                            result: Some(result),
                        },
                        span,
                    );
                    *value = result;
                }
                Statement::Block(child) => convert_returns(child, expressions, transform),
                Statement::If { accept, reject, .. } => {
                    convert_returns(accept, expressions, transform);
                    convert_returns(reject, expressions, transform);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        convert_returns(&mut case.body, expressions, transform);
                    }
                }
                Statement::Loop { body, continuing, .. } => {
                    convert_returns(body, expressions, transform);
                    convert_returns(continuing, expressions, transform);
                }
                _ => {}
            }
            block.push(statement, span);
        }
    }
    convert_returns(
        &mut entry.function.body,
        &mut entry.function.expressions,
        transform,
    );
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_conversion_covers_fragment_early_returns() {
        let source = format!(
            "{}\n{}\n{}",
            include_str!("color.wgsl"),
            include_str!("shaders.wgsl"),
            r#"
@fragment
fn custom_fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    if input.tex_coord.x < 0.0 { discard; }
    if input.tex_coord.x < 0.5 { return vec4<f32>(0.5); }
    switch u32(input.tex_coord.y) {
        case 0u: { return vec4<f32>(1.0); }
        default: { return vec4<f32>(0.0); }
    }
}
"#
        );
        let module = color_managed_module(&source).unwrap();
        ::wgpu::naga::valid::Validator::new(
            ::wgpu::naga::valid::ValidationFlags::all(),
            ::wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let entry = module
            .entry_points
            .iter()
            .find(|entry| entry.name == "custom_fragment")
            .unwrap();
        assert_eq!(
            entry
                .function
                .expressions
                .iter()
                .filter(|(_, expression)| matches!(expression, ::wgpu::naga::Expression::CallResult(_)))
                .count(),
            3
        );
    }
}
