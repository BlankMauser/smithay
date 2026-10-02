struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) force_opaque: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) force_opaque: f32,
};

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.tex_coord = input.tex_coord;
    output.color = input.color;
    output.force_opaque = input.force_opaque;
    return output;
}

@group(0) @binding(0)
var source_texture: texture_2d<f32>;

@group(0) @binding(1)
var source_sampler: sampler;

@fragment
fn texture_fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(source_texture, source_sampler, input.tex_coord);
    if input.force_opaque > 0.5 {
        color.a = 1.0;
    }
    return smithay_color_transform(color * input.color);
}

@fragment
fn solid_fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}

// Optional auxiliary image, for example a flattened monitor color LUT.
@group(2) @binding(0) var auxiliary_texture: texture_2d<f32>;
@group(2) @binding(1) var auxiliary_sampler: sampler;
