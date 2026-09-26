use smithay::{
    backend::{
        allocator::Fourcc,
        renderer::{
            Bind, Blit, BlitFrame, Color32F, DebugFlags, ExportMem, Frame, ImportMem, Offscreen, Renderer,
            Texture, TextureFilter,
            wgpu::{WgpuRenderer, WgpuTexture},
        },
    },
    utils::{Buffer, Physical, Point, Rectangle, Size, Transform},
};

const WIDTH: i32 = 4;
const HEIGHT: i32 = 4;
const FORMAT: Fourcc = Fourcc::Abgr8888;

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .expect("No Vulkan adapter is available");
    let info = adapter.get_info();
    assert_ne!(
        info.device_type,
        wgpu::DeviceType::Cpu,
        "A CPU Vulkan adapter cannot validate the hardware renderer"
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Smithay WGPU renderer example"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: Default::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create WGPU device");
    let mut renderer = WgpuRenderer::new(device, queue).expect("Failed to create Smithay renderer");

    println!(
        "Vulkan adapter: {} ({:?}, {:?})",
        info.name, info.device_type, info.backend
    );

    test_custom_pixel_first_draw(&mut renderer);
    test_clear_and_damage(&mut renderer);
    test_memory_import_and_update(&mut renderer);
    test_texture_transforms(&mut renderer);
    test_non_square_transforms(&mut renderer);
    test_output_transforms(&mut renderer);
    test_destination_damage(&mut renderer);
    test_small_viewport(&mut renderer);
    test_alpha_and_crop(&mut renderer);
    test_nearest_filter(&mut renderer);
    test_blits(&mut renderer);
    test_invalid_bounds(&mut renderer);

    println!(
        "Passed: custom pixel, clear, damage, viewport, memory, transforms, alpha, crop, filtering, blits, bounds"
    );
}

fn target(renderer: &mut WgpuRenderer, size: (i32, i32)) -> WgpuTexture {
    Offscreen::<WgpuTexture>::create_buffer(renderer, FORMAT, Size::from(size))
        .expect("Failed to create target")
}

macro_rules! render {
    ($renderer:expr, $target:expr, $transform:expr, |$frame:ident| $body:block) => {{
        let __size = Size::<i32, Physical>::from(($target.width() as i32, $target.height() as i32));
        let mut framebuffer = $renderer.bind($target).expect("Failed to bind target");
        let mut __frame = $renderer
            .render(&mut framebuffer, __size, $transform)
            .expect("Failed to begin frame");
        {
            let $frame = &mut __frame;
            $body
        }
        let sync = __frame.finish().expect("Failed to finish frame");
        $renderer.wait(&sync).expect("Failed to wait for frame");
    }};
}

fn test_custom_pixel_first_draw(renderer: &mut WgpuRenderer) {
    let program = renderer
        .compile_custom_pixel_shader(
            r#"
struct CustomUniforms {
    values: array<vec4<f32>, 32>,
};

@group(1) @binding(0)
var<uniform> custom: CustomUniforms;

@fragment
fn custom_fragment(_input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(custom.values[0].z, 0.0, 0.0, custom.values[0].z);
}
"#,
            &[],
        )
        .expect("Failed to compile custom pixel shader");
    let size = Size::<i32, Buffer>::from((2, 2));
    let mut texture = target(renderer, (size.w, size.h));
    render!(renderer, &mut texture, Transform::Normal, |frame| {
        frame
            .render_pixel_shader_to(
                &program,
                Rectangle::from_size(size).to_f64(),
                physical_rect((size.w, size.h)),
                size,
                None,
                1.0,
                &[],
            )
            .expect("Failed to render custom pixel shader as the first draw");
    });
    for pixel in read(renderer, &texture).chunks_exact(4) {
        assert_eq!(pixel, [255, 0, 0, 255]);
    }
}

fn read(renderer: &mut WgpuRenderer, texture: &WgpuTexture) -> Vec<u8> {
    let mapping = renderer
        .copy_texture(
            texture,
            Rectangle::<i32, Buffer>::from_size(texture.size()),
            FORMAT,
        )
        .expect("Failed to read texture");
    renderer
        .map_texture(&mapping)
        .expect("Failed to map texture")
        .to_vec()
}

fn physical_rect(size: (i32, i32)) -> Rectangle<i32, Physical> {
    Rectangle::from_size(Size::from(size))
}

fn pixel<K>(pixels: &[u8], size: Size<i32, K>, x: i32, y: i32) -> [u8; 4] {
    let offset = ((y * size.w + x) * 4) as usize;
    pixels[offset..offset + 4].try_into().unwrap()
}

fn assert_pixel<K>(pixels: &[u8], size: Size<i32, K>, x: i32, y: i32, expected: [u8; 4]) {
    let actual = pixel(pixels, size, x, y);
    assert_eq!(actual, expected, "pixel ({x}, {y})");
}

fn test_clear_and_damage(renderer: &mut WgpuRenderer) {
    let size = Size::<i32, Buffer>::from((WIDTH, HEIGHT));
    let mut texture = target(renderer, (WIDTH, HEIGHT));
    render!(renderer, &mut texture, Transform::Normal, |frame| {
        frame
            .clear(
                Color32F::new(0.0, 0.0, 1.0, 1.0),
                &[physical_rect((WIDTH, HEIGHT))],
            )
            .expect("Failed to clear target");
        frame
            .clear(
                Color32F::new(1.0, 0.0, 0.0, 1.0),
                &[Rectangle::new((1, 1).into(), (2, 2).into())],
            )
            .expect("Failed to clear damaged region");
    });
    let pixels = read(renderer, &texture);
    assert_pixel(&pixels, size, 0, 0, [0, 0, 255, 255]);
    assert_pixel(&pixels, size, 1, 1, [255, 0, 0, 255]);
    assert_pixel(&pixels, size, 2, 2, [255, 0, 0, 255]);
    assert_pixel(&pixels, size, 3, 3, [0, 0, 255, 255]);
}

fn test_memory_import_and_update(renderer: &mut WgpuRenderer) {
    let size = Size::<i32, Buffer>::from((2, 2));
    let initial = [
        255, 0, 0, 255, 0, 255, 0, 255, // red, green
        0, 0, 255, 255, 255, 255, 255, 255, // blue, white
    ];
    let texture = renderer
        .import_memory(&initial, FORMAT, size, false)
        .expect("Failed to import memory");
    assert_eq!(read(renderer, &texture), initial);

    let updated = [
        255, 0, 0, 255, 255, 255, 0, 255, // red, yellow
        0, 0, 255, 255, 255, 255, 255, 255, // blue, white
    ];
    renderer
        .update_memory(&texture, &updated, Rectangle::new((1, 0).into(), (1, 1).into()))
        .expect("Failed to update memory");
    let pixels = read(renderer, &texture);
    assert_pixel(&pixels, size, 0, 0, [255, 0, 0, 255]);
    assert_pixel(&pixels, size, 1, 0, [255, 255, 0, 255]);
}

fn test_texture_transforms(renderer: &mut WgpuRenderer) {
    let size = Size::<i32, Buffer>::from((2, 2));
    let source = [
        255, 0, 0, 255, 0, 255, 0, 255, // red, green
        0, 0, 255, 255, 255, 255, 255, 255, // blue, white
    ];
    let cases = transform_cases();

    let texture = renderer
        .import_memory(&source, FORMAT, size, false)
        .expect("Failed to import transform fixture");
    for (transform, expected) in cases {
        let mut output = target(renderer, (2, 2));
        render!(renderer, &mut output, Transform::Normal, |frame| {
            frame
                .render_texture_from_to(
                    &texture,
                    Rectangle::from_size(size).to_f64(),
                    physical_rect((2, 2)),
                    &[physical_rect((2, 2))],
                    &[],
                    transform,
                    1.0,
                    None,
                    &[],
                )
                .expect("Failed to render transformed texture");
        });
        assert_eq!(
            read(renderer, &output),
            expected,
            "source transform {transform:?}"
        );
    }

    let flipped = renderer
        .import_memory(&source, FORMAT, size, true)
        .expect("Failed to import flipped fixture");
    let mut output = target(renderer, (2, 2));
    render!(renderer, &mut output, Transform::Normal, |frame| {
        frame
            .render_texture_at(
                &flipped,
                Point::from((0, 0)),
                1,
                1.0,
                Transform::Normal,
                &[physical_rect((2, 2))],
                &[],
                1.0,
            )
            .expect("Failed to render flipped texture");
    });
    assert_pixel(&read(renderer, &output), size, 0, 0, [0, 0, 255, 255]);
}

fn test_output_transforms(renderer: &mut WgpuRenderer) {
    for (transform, expected) in transform_cases() {
        let mut output = target(renderer, (2, 2));
        render!(renderer, &mut output, transform, |frame| {
            for (rect, color) in [
                (
                    Rectangle::new((0, 0).into(), (1, 1).into()),
                    Color32F::new(1.0, 0.0, 0.0, 1.0),
                ),
                (
                    Rectangle::new((1, 0).into(), (1, 1).into()),
                    Color32F::new(0.0, 1.0, 0.0, 1.0),
                ),
                (
                    Rectangle::new((0, 1).into(), (1, 1).into()),
                    Color32F::new(0.0, 0.0, 1.0, 1.0),
                ),
                (
                    Rectangle::new((1, 1).into(), (1, 1).into()),
                    Color32F::new(1.0, 1.0, 1.0, 1.0),
                ),
            ] {
                frame
                    .draw_solid(rect, &[Rectangle::from_size(rect.size)], color)
                    .expect("Failed to draw test pixel");
            }
        });
        assert_eq!(
            read(renderer, &output),
            expected,
            "output transform {transform:?}"
        );
    }
}

fn transform_cases() -> [(Transform, [u8; 16]); 8] {
    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];
    [
        (Transform::Normal, pixels4(RED, GREEN, BLUE, WHITE)),
        (Transform::_90, pixels4(BLUE, RED, WHITE, GREEN)),
        (Transform::_180, pixels4(WHITE, BLUE, GREEN, RED)),
        (Transform::_270, pixels4(GREEN, WHITE, RED, BLUE)),
        (Transform::Flipped, pixels4(GREEN, RED, WHITE, BLUE)),
        (Transform::Flipped90, pixels4(RED, BLUE, GREEN, WHITE)),
        (Transform::Flipped180, pixels4(BLUE, WHITE, RED, GREEN)),
        (Transform::Flipped270, pixels4(WHITE, GREEN, BLUE, RED)),
    ]
}

fn pixels4(a: [u8; 4], b: [u8; 4], c: [u8; 4], d: [u8; 4]) -> [u8; 16] {
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], d[0], d[1], d[2], d[3],
    ]
}

fn test_non_square_transforms(renderer: &mut WgpuRenderer) {
    const A: [u8; 4] = [255, 0, 0, 255];
    const B: [u8; 4] = [0, 255, 0, 255];
    const C: [u8; 4] = [0, 0, 255, 255];
    const D: [u8; 4] = [255, 255, 0, 255];
    const E: [u8; 4] = [255, 0, 255, 255];
    const F: [u8; 4] = [0, 255, 255, 255];
    let source_size = Size::<i32, Buffer>::from((3, 2));
    let source = [A, B, C, D, E, F].concat();
    let texture = renderer
        .import_memory(&source, FORMAT, source_size, false)
        .expect("Failed to import non-square transform fixture");
    let cases = [
        (Transform::Normal, (3, 2), pixels6(A, B, C, D, E, F)),
        (Transform::_90, (2, 3), pixels6(D, A, E, B, F, C)),
        (Transform::_180, (3, 2), pixels6(F, E, D, C, B, A)),
        (Transform::_270, (2, 3), pixels6(C, F, B, E, A, D)),
        (Transform::Flipped, (3, 2), pixels6(C, B, A, F, E, D)),
        (Transform::Flipped90, (2, 3), pixels6(A, D, B, E, C, F)),
        (Transform::Flipped180, (3, 2), pixels6(D, E, F, A, B, C)),
        (Transform::Flipped270, (2, 3), pixels6(F, C, E, B, D, A)),
    ];
    for (transform, size, expected) in cases {
        let mut output = target(renderer, size);
        render!(renderer, &mut output, Transform::Normal, |frame| {
            frame
                .render_texture_from_to(
                    &texture,
                    Rectangle::from_size(source_size).to_f64(),
                    physical_rect(size),
                    &[physical_rect(size)],
                    &[],
                    transform,
                    1.0,
                    None,
                    &[],
                )
                .expect("Failed to render non-square transform");
        });
        assert_eq!(
            read(renderer, &output),
            expected,
            "non-square source transform {transform:?}"
        );
    }
}

fn pixels6(a: [u8; 4], b: [u8; 4], c: [u8; 4], d: [u8; 4], e: [u8; 4], f: [u8; 4]) -> [u8; 24] {
    [
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], d[0], d[1], d[2], d[3], e[0],
        e[1], e[2], e[3], f[0], f[1], f[2], f[3],
    ]
}

fn test_destination_damage(renderer: &mut WgpuRenderer) {
    let mut output = target(renderer, (4, 3));
    render!(renderer, &mut output, Transform::Normal, |frame| {
        frame
            .clear(Color32F::TRANSPARENT, &[physical_rect((4, 3))])
            .expect("Failed to clear damage target");
        frame
            .draw_solid(
                Rectangle::new((2, 1).into(), (2, 2).into()),
                &[Rectangle::new((1, 0).into(), (1, 1).into())],
                Color32F::new(1.0, 0.0, 0.0, 1.0),
            )
            .expect("Failed to draw damaged solid");
    });
    let size = Size::<i32, Buffer>::from((4, 3));
    let pixels = read(renderer, &output);
    assert_pixel(&pixels, size, 3, 1, [255, 0, 0, 255]);
    assert_pixel(&pixels, size, 2, 1, [0, 0, 0, 0]);

    let source_size = Size::<i32, Buffer>::from((2, 2));
    let source = [
        255, 0, 0, 255, 0, 255, 0, 255, // red, green
        0, 0, 255, 255, 255, 255, 255, 255, // blue, white
    ];
    let texture = renderer
        .import_memory(&source, FORMAT, source_size, false)
        .expect("Failed to import destination-damage fixture");
    let mut output = target(renderer, (4, 3));
    render!(renderer, &mut output, Transform::Normal, |frame| {
        frame
            .clear(Color32F::TRANSPARENT, &[physical_rect((4, 3))])
            .expect("Failed to clear texture damage target");
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size(source_size).to_f64(),
                Rectangle::new((2, 1).into(), (2, 2).into()),
                &[Rectangle::new((1, 0).into(), (1, 1).into())],
                &[],
                Transform::Normal,
                1.0,
                None,
                &[],
            )
            .expect("Failed to render damaged texture");
    });
    let pixels = read(renderer, &output);
    assert_pixel(&pixels, size, 3, 1, [0, 255, 0, 255]);
    assert_pixel(&pixels, size, 2, 1, [0, 0, 0, 0]);
}

fn test_alpha_and_crop(renderer: &mut WgpuRenderer) {
    let source_size = Size::<i32, Buffer>::from((2, 2));
    let source = [
        128, 0, 0, 128, 0, 255, 0, 255, // half red, green
        0, 0, 255, 255, 255, 255, 255, 255, // blue, white
    ];
    let texture = renderer
        .import_memory(&source, FORMAT, source_size, false)
        .expect("Failed to import alpha fixture");
    let size = Size::<i32, Buffer>::from((2, 2));
    let mut output = target(renderer, (2, 2));
    render!(renderer, &mut output, Transform::Normal, |frame| {
        frame
            .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[physical_rect((2, 2))])
            .expect("Failed to clear alpha target");
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size(source_size).to_f64(),
                physical_rect((2, 2)),
                &[physical_rect((2, 2))],
                &[],
                Transform::Normal,
                1.0,
                None,
                &[],
            )
            .expect("Failed to blend alpha fixture");
    });
    assert_pixel(&read(renderer, &output), size, 0, 0, [128, 0, 127, 255]);

    let crop_size = Size::<i32, Buffer>::from((4, 1));
    let crop_source = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255];
    let texture = renderer
        .import_memory(&crop_source, FORMAT, crop_size, false)
        .expect("Failed to import crop fixture");
    let mut output = target(renderer, (2, 1));
    render!(renderer, &mut output, Transform::Normal, |frame| {
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::new((1.0, 0.0).into(), (2.0, 1.0).into()),
                physical_rect((2, 1)),
                &[physical_rect((2, 1))],
                &[],
                Transform::Normal,
                1.0,
                None,
                &[],
            )
            .expect("Failed to render crop");
    });
    let output_size = Size::<i32, Buffer>::from((2, 1));
    let pixels = read(renderer, &output);
    assert_pixel(&pixels, output_size, 0, 0, [0, 255, 0, 255]);
    assert_pixel(&pixels, output_size, 1, 0, [0, 0, 255, 255]);
}

fn test_nearest_filter(renderer: &mut WgpuRenderer) {
    renderer
        .upscale_filter(TextureFilter::Nearest)
        .expect("Failed to select nearest filtering");
    let size = Size::<i32, Buffer>::from((2, 2));
    let source = [
        255, 0, 0, 255, 0, 255, 0, 255, // red, green
        0, 0, 255, 255, 255, 255, 255, 255, // blue, white
    ];
    let texture = renderer
        .import_memory(&source, FORMAT, size, false)
        .expect("Failed to import nearest fixture");
    let output_size = Size::<i32, Buffer>::from((4, 4));
    let mut output = target(renderer, (4, 4));
    render!(renderer, &mut output, Transform::Normal, |frame| {
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size(size).to_f64(),
                physical_rect((4, 4)),
                &[physical_rect((4, 4))],
                &[],
                Transform::Normal,
                1.0,
                None,
                &[],
            )
            .expect("Failed to scale nearest fixture");
    });
    let pixels = read(renderer, &output);
    assert_pixel(&pixels, output_size, 0, 0, [255, 0, 0, 255]);
    assert_pixel(&pixels, output_size, 1, 1, [255, 0, 0, 255]);
    assert_pixel(&pixels, output_size, 2, 1, [0, 255, 0, 255]);
    assert_pixel(&pixels, output_size, 3, 0, [0, 255, 0, 255]);
    assert_pixel(&pixels, output_size, 0, 3, [0, 0, 255, 255]);
    assert_pixel(&pixels, output_size, 3, 3, [255, 255, 255, 255]);
    renderer
        .upscale_filter(TextureFilter::Linear)
        .expect("Failed to restore linear filtering");

    renderer
        .downscale_filter(TextureFilter::Nearest)
        .expect("Failed to select nearest minification");
    let source = [
        255, 0, 0, 255, 0, 255, 0, 255, // red, green
        0, 0, 255, 255, 255, 255, 255, 255, // blue, white
    ];
    let texture = renderer
        .import_memory(&source, FORMAT, size, false)
        .expect("Failed to import minification fixture");
    let mut output = target(renderer, (1, 1));
    render!(renderer, &mut output, Transform::Normal, |frame| {
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size(size).to_f64(),
                physical_rect((1, 1)),
                &[physical_rect((1, 1))],
                &[],
                Transform::Normal,
                1.0,
                None,
                &[],
            )
            .expect("Failed to minify nearest fixture");
    });
    let sample = pixel(&read(renderer, &output), Size::<i32, Buffer>::from((1, 1)), 0, 0);
    assert!(
        [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 255, 255],
        ]
        .contains(&sample),
        "nearest minification sampled an interpolated color: {sample:?}"
    );
    renderer
        .downscale_filter(TextureFilter::Linear)
        .expect("Failed to restore linear minification");
}

fn test_blits(renderer: &mut WgpuRenderer) {
    let size = Size::<i32, Buffer>::from((2, 2));
    let mut source = target(renderer, (2, 2));
    render!(renderer, &mut source, Transform::Normal, |frame| {
        frame
            .clear(Color32F::TRANSPARENT, &[physical_rect((2, 2))])
            .expect("Failed to clear blit source");
        frame
            .draw_solid(
                physical_rect((2, 2)),
                &[physical_rect((2, 2))],
                Color32F::new(0.5, 0.0, 0.0, 0.5),
            )
            .expect("Failed to draw blit source");
    });

    let mut destination = target(renderer, (2, 2));
    render!(renderer, &mut destination, Transform::Normal, |frame| {
        frame
            .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[physical_rect((2, 2))])
            .expect("Failed to clear blit destination");
    });
    {
        let source_fb = renderer.bind(&mut source).expect("Failed to bind blit source");
        let mut destination_fb = renderer
            .bind(&mut destination)
            .expect("Failed to bind blit destination");
        let sync = renderer
            .blit(
                &source_fb,
                &mut destination_fb,
                physical_rect((2, 2)),
                physical_rect((2, 2)),
                TextureFilter::Linear,
            )
            .expect("Failed to blit framebuffers");
        renderer.wait(&sync).expect("Failed to wait for framebuffer blit");
    }
    assert_pixel(&read(renderer, &destination), size, 0, 0, [128, 0, 0, 128]);

    renderer
        .upscale_filter(TextureFilter::Nearest)
        .expect("Failed to set nearest magnification before blit_from");
    renderer
        .downscale_filter(TextureFilter::Nearest)
        .expect("Failed to set nearest minification before blit_from");
    let mut destination = target(renderer, (2, 2));
    render!(renderer, &mut destination, Transform::Normal, |frame| {
        frame
            .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[physical_rect((2, 2))])
            .expect("Failed to clear blit_from destination");
    });
    {
        let source_fb = renderer
            .bind(&mut source)
            .expect("Failed to bind blit_from source");
        let mut destination_fb = renderer
            .bind(&mut destination)
            .expect("Failed to bind blit_from destination");
        let mut frame = renderer
            .render(
                &mut destination_fb,
                Size::<i32, Physical>::from((2, 2)),
                Transform::Normal,
            )
            .expect("Failed to begin blit_from frame");
        let blit_sync = frame
            .blit_from(
                &source_fb,
                physical_rect((2, 2)),
                physical_rect((2, 2)),
                TextureFilter::Linear,
            )
            .expect("Failed to blit_from frame");
        let frame_sync = frame.finish().expect("Failed to finish blit_from frame");
        renderer.wait(&blit_sync).expect("Failed to wait for blit_from");
        renderer
            .wait(&frame_sync)
            .expect("Failed to wait for blit_from frame");
    }
    assert_pixel(&read(renderer, &destination), size, 0, 0, [128, 0, 0, 128]);

    test_transformed_blit_from(renderer);

    let mut magnified = target(renderer, (4, 4));
    let texture = renderer
        .import_memory(
            &[
                255, 0, 0, 255, 0, 255, 0, 255, // red, green
                0, 0, 255, 255, 255, 255, 255, 255, // blue, white
            ],
            FORMAT,
            size,
            false,
        )
        .expect("Failed to import filter restoration fixture");
    render!(renderer, &mut magnified, Transform::Normal, |frame| {
        frame
            .render_texture_from_to(
                &texture,
                Rectangle::from_size(size).to_f64(),
                physical_rect((4, 4)),
                &[physical_rect((4, 4))],
                &[],
                Transform::Normal,
                1.0,
                None,
                &[],
            )
            .expect("Failed to draw after blit_from");
    });
    assert_pixel(
        &read(renderer, &magnified),
        Size::<i32, Buffer>::from((4, 4)),
        1,
        1,
        [255, 0, 0, 255],
    );

    let mut destination = target(renderer, (2, 2));
    render!(renderer, &mut destination, Transform::Normal, |frame| {
        frame
            .clear(Color32F::new(0.0, 0.0, 1.0, 1.0), &[physical_rect((2, 2))])
            .expect("Failed to clear blit_to destination");
    });
    {
        let mut source_fb = renderer.bind(&mut source).expect("Failed to bind blit_to source");
        let mut destination_fb = renderer
            .bind(&mut destination)
            .expect("Failed to bind blit_to destination");
        let mut frame = renderer
            .render(
                &mut source_fb,
                Size::<i32, Physical>::from((2, 2)),
                Transform::Normal,
            )
            .expect("Failed to begin blit_to frame");
        let blit_sync = frame
            .blit_to(
                &mut destination_fb,
                physical_rect((2, 2)),
                physical_rect((2, 2)),
                TextureFilter::Nearest,
            )
            .expect("Failed to blit_to frame");
        let frame_sync = frame.finish().expect("Failed to finish blit_to frame");
        renderer.wait(&blit_sync).expect("Failed to wait for blit_to");
        renderer
            .wait(&frame_sync)
            .expect("Failed to wait for blit_to frame");
    }
    assert_pixel(&read(renderer, &destination), size, 0, 0, [128, 0, 0, 128]);
}

fn test_transformed_blit_from(renderer: &mut WgpuRenderer) {
    let size = Size::<i32, Buffer>::from((2, 2));
    let source_data = [
        255, 0, 0, 255, 0, 255, 0, 255, // red, green
        0, 0, 255, 255, 255, 255, 255, 255, // blue, white
    ];
    for (flipped, flags) in [
        (false, DebugFlags::empty()),
        (true, DebugFlags::empty()),
        (false, DebugFlags::TINT),
        (true, DebugFlags::TINT),
    ] {
        renderer.set_debug_flags(flags);
        let mut source = renderer
            .import_memory(&source_data, FORMAT, size, flipped)
            .expect("Failed to import transformed blit source");
        let mut destination = target(renderer, (2, 2));
        {
            let source_fb = renderer
                .bind(&mut source)
                .expect("Failed to bind transformed blit source");
            let mut destination_fb = renderer
                .bind(&mut destination)
                .expect("Failed to bind transformed blit destination");
            let mut frame = renderer
                .render(
                    &mut destination_fb,
                    Size::<i32, Physical>::from((2, 2)),
                    Transform::_90,
                )
                .expect("Failed to begin transformed blit frame");
            let blit_sync = frame
                .blit_from(
                    &source_fb,
                    physical_rect((2, 2)),
                    physical_rect((2, 2)),
                    TextureFilter::Nearest,
                )
                .expect("Failed to blit into transformed frame");
            let frame_sync = frame.finish().expect("Failed to finish transformed blit frame");
            renderer
                .wait(&blit_sync)
                .expect("Failed to wait for transformed blit");
            renderer
                .wait(&frame_sync)
                .expect("Failed to wait for transformed blit frame");
        }
        assert_eq!(read(renderer, &destination), source_data);
    }
    renderer.set_debug_flags(DebugFlags::empty());
}

fn test_small_viewport(renderer: &mut WgpuRenderer) {
    let mut output = target(renderer, (4, 4));
    render!(renderer, &mut output, Transform::Normal, |frame| {
        frame
            .clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &[physical_rect((4, 4))])
            .expect("Failed to initialize small-viewport target");
    });
    {
        let mut framebuffer = renderer.bind(&mut output).expect("Failed to bind small viewport");
        let mut frame = renderer
            .render(
                &mut framebuffer,
                Size::<i32, Physical>::from((2, 2)),
                Transform::Normal,
            )
            .expect("Failed to begin small-viewport frame");
        frame
            .draw_solid(
                physical_rect((2, 2)),
                &[physical_rect((2, 2))],
                Color32F::new(0.0, 1.0, 0.0, 1.0),
            )
            .expect("Failed to draw within small viewport");
        let sync = frame.finish().expect("Failed to finish small-viewport frame");
        renderer
            .wait(&sync)
            .expect("Failed to wait for small-viewport frame");
    }
    let pixels = read(renderer, &output);
    let size = Size::<i32, Buffer>::from((4, 4));
    assert_pixel(&pixels, size, 0, 0, [0, 255, 0, 255]);
    assert_pixel(&pixels, size, 1, 1, [0, 255, 0, 255]);
    assert_pixel(&pixels, size, 2, 2, [255, 0, 0, 255]);
    assert_pixel(&pixels, size, 3, 3, [255, 0, 0, 255]);
}

fn test_invalid_bounds(renderer: &mut WgpuRenderer) {
    let size = Size::<i32, Buffer>::from((2, 2));
    let pixels = [0; 16];
    assert!(
        renderer
            .import_memory(&pixels[..12], FORMAT, size, false)
            .is_err()
    );

    let texture = renderer
        .import_memory(&pixels, FORMAT, size, false)
        .expect("Failed to import bounds fixture");
    assert!(
        renderer
            .update_memory(&texture, &pixels, Rectangle::new((1, 1).into(), (2, 1).into()))
            .is_err()
    );
}
