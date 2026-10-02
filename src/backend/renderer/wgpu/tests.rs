use super::*;
use crate::backend::renderer::{Color32F, ColorTransform, Frame};

const FORMAT: DrmFourcc = DrmFourcc::Abgr16161616f;

fn read(renderer: &mut WgpuRenderer, target: &WgpuTexture) -> [f32; 4] {
    let mapping = renderer
        .copy_texture(target, Rectangle::from_size(target.size()), FORMAT)
        .unwrap();
    let bytes = renderer.map_texture(&mapping).unwrap();
    std::array::from_fn(|channel| {
        let bits = u16::from_le_bytes([bytes[channel * 2], bytes[channel * 2 + 1]]);
        let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
        let exponent = (bits >> 10) & 0x1f;
        let mantissa = (bits & 0x3ff) as f32;
        sign * if exponent == 0 {
            mantissa * 2.0_f32.powi(-24)
        } else {
            (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent as i32 - 15)
        }
    })
}

fn assert_pixel(actual: [f32; 4], expected: [f32; 4]) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= expected.abs() * 0.002 + 0.0001,
            "{actual} != {expected}"
        );
    }
}

#[test]
#[ignore = "requires a hardware Vulkan adapter; runs entirely offscreen"]
fn hardware_color_pipeline() {
    let instance = ::wgpu::Instance::new(::wgpu::InstanceDescriptor {
        backends: ::wgpu::Backends::VULKAN,
        ..::wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&::wgpu::RequestAdapterOptions {
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .unwrap();
    let info = adapter.get_info();
    assert_ne!(info.device_type, ::wgpu::DeviceType::Cpu);
    tracing::info!(adapter = %info.name, backend = ?info.backend, "HDR test adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&::wgpu::DeviceDescriptor {
        required_limits: ::wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap();
    let mut renderer = WgpuRenderer::new(device, queue).unwrap();
    let pixel = renderer
        .compile_custom_pixel_shader(
            r#"
struct CustomUniforms { values: array<vec4<f32>, 32>, };
@group(1) @binding(0) var<uniform> custom: CustomUniforms;
@fragment fn custom_fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    if input.tex_coord.x < 0.75 { return custom.values[1]; }
    return vec4<f32>(0.0);
}
"#,
            &[UniformName::new("color", UniformType::_4f)],
        )
        .unwrap();
    let texture_program = renderer
        .compile_custom_texture_shader(
            r#"
@fragment fn custom_fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source_texture, source_sampler, input.tex_coord);
}
"#,
            &[],
        )
        .unwrap();
    let auxiliary_program = renderer
        .compile_custom_texture_shader(
            r#"
@fragment fn custom_fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(auxiliary_texture, auxiliary_sampler, input.tex_coord);
}
"#,
            &[],
        )
        .unwrap();
    let mut target = Offscreen::<WgpuTexture>::create_buffer(&mut renderer, FORMAT, (1, 1).into()).unwrap();
    let source = renderer
        .import_memory(&[128, 64, 32, 128], DrmFourcc::Abgr8888, (1, 1).into(), false)
        .unwrap();
    let source_color = [128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, 128.0 / 255.0];
    let rect = Rectangle::from_size((1, 1).into());
    for transfer in 0..=14 {
        let color = ColorTransform {
            transfer,
            min_luminance: 0.0,
            max_luminance: if transfer == 11 { 10000.0 } else { 80.0 },
            luminance_scale: 1.0 / 80.0,
            matrix: [0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8],
            ..ColorTransform::IDENTITY
        };
        let expected = color.apply(source_color);
        for path in 0..5 {
            renderer.set_color_transform(Some(color));
            let mut fb = renderer.bind(&mut target).unwrap();
            let mut frame = renderer
                .render(&mut fb, (1, 1).into(), Transform::Normal)
                .unwrap();
            assert_eq!(frame.color_transform(), Some(color));
            frame.set_color_transform(None);
            frame.clear(Color32F::TRANSPARENT, &[rect]).unwrap();
            frame.set_color_transform(Some(color));
            match path {
                0 => frame.clear(source_color.into(), &[rect]).unwrap(),
                1 => frame.draw_solid(rect, &[rect], source_color.into()).unwrap(),
                2 => frame
                    .render_pixel_shader_to(
                        &pixel,
                        Rectangle::from_size((1, 1).into()).to_f64(),
                        rect,
                        (1, 1).into(),
                        None,
                        1.0,
                        &[Uniform::new("color", source_color)],
                    )
                    .unwrap(),
                3 => frame
                    .render_texture_from_to(
                        &source,
                        Rectangle::from_size(source.size()).to_f64(),
                        rect,
                        &[rect],
                        &[],
                        Transform::Normal,
                        1.0,
                        None,
                        &[],
                    )
                    .unwrap(),
                _ => frame
                    .render_texture_from_to(
                        &source,
                        Rectangle::from_size(source.size()).to_f64(),
                        rect,
                        &[rect],
                        &[],
                        Transform::Normal,
                        1.0,
                        Some(&texture_program),
                        &[],
                    )
                    .unwrap(),
            }
            let sync = frame.finish().unwrap();
            renderer.wait(&sync).unwrap();
            assert_pixel(read(&mut renderer, &target), expected);
        }
    }
    // Retained linear FP16 values bypass input decoding, including negatives and HDR peaks.
    let extended: Vec<u8> = [0xb800_u16, 0x4000, 0x57d0, 0x3c00]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let linear = renderer
        .import_memory(&extended, FORMAT, (1, 1).into(), false)
        .unwrap();
    linear.set_linear(true);
    let shared = linear.clone();
    assert!(shared.is_linear());
    for auxiliary in [false, true] {
        let mut fb = renderer.bind(&mut target).unwrap();
        let mut frame = renderer
            .render(&mut fb, (1, 1).into(), Transform::Normal)
            .unwrap();
        if auxiliary {
            frame.set_color_transform(None);
        }
        frame
            .render_texture_from_to_with_auxiliary(
                &linear,
                Rectangle::from_size(linear.size()).to_f64(),
                rect,
                &[rect],
                &[rect],
                Transform::Normal,
                1.0,
                auxiliary.then_some(&auxiliary_program),
                &[],
                auxiliary.then_some(&shared),
            )
            .unwrap();
        let sync = frame.finish().unwrap();
        renderer.wait(&sync).unwrap();
        assert_pixel(read(&mut renderer, &target), [-0.5, 2.0, 125.0, 1.0]);
    }
    renderer.set_color_transform(None);
}
