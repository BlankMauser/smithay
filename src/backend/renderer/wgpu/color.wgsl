// Input conversion uses absolute luminance, then a column-major gamut transform.
struct SmithayColorUniforms {
    reserved_custom: array<vec4<f32>, 32>,
    matrix: array<vec4<f32>, 3>,
    parameters: vec4<f32>,
    luma: vec4<f32>,
};
@group(1) @binding(1) var<uniform> smithay_color: SmithayColorUniforms;

fn smithay_piece(v: f32, cutoff: f32, slope: f32, a: f32, gamma: f32) -> f32 {
    if v <= cutoff { return v / slope; }
    return pow((v + a - 1.0) / a, gamma);
}

fn smithay_decode(transfer: u32, value: f32) -> f32 {
    let v = max(value, 0.0);
    switch transfer {
        case 2u: { return pow(v, 2.2); }
        case 3u: { return pow(v, 2.8); }
        case 4u: { return smithay_piece(v, 0.0912, 4.0, 1.1115, 1.0 / 0.45); }
        case 5u: { return value; }
        case 6u: {
            if value <= 0.0 { return 0.0; }
            return pow(10.0, 2.0 * (value - 1.0));
        }
        case 7u: {
            if value <= 0.0 { return 0.0; }
            return pow(10.0, 2.5 * (value - 1.0));
        }
        case 8u: { return sign(value) * smithay_piece(abs(value), 0.081, 4.5, 1.099, 1.0 / 0.45); }
        case 9u, 14u: { return smithay_piece(v, 0.04045, 12.92, 1.055, 2.4); }
        case 10u: { return sign(value) * smithay_piece(abs(value), 0.04045, 12.92, 1.055, 2.4); }
        case 11u: {
            let p = pow(v, 32.0 / 2523.0);
            return pow(max(p - 3424.0 / 4096.0, 0.0) / max(2413.0 / 128.0 - 2392.0 / 128.0 * p, 1e-12), 16384.0 / 2610.0);
        }
        case 12u: { return pow(v, 2.6) * 52.37 / 48.0; }
        case 13u: {
            if v <= 0.5 { return v * v / 3.0; }
            return (exp((v - 0.55991073) / 0.17883277) + 0.28466892) / 12.0;
        }
        default: { return value; }
    }
}

fn smithay_color_transform(color: vec4<f32>) -> vec4<f32> {
    let transfer = u32(smithay_color.parameters.x);
    if transfer == 0u { return color; }
    if color.a <= 0.0 { return vec4<f32>(0.0); }
    var rgb = color.rgb / color.a;
    let black = smithay_color.parameters.y;
    let peak = smithay_color.parameters.z;
    if transfer == 1u {
        let base = pow(black, 1.0 / 2.4);
        let range = pow(peak, 1.0 / 2.4) - base;
        rgb = pow(max(rgb, vec3<f32>(0.0)) * range + vec3<f32>(base), vec3<f32>(2.4));
    } else {
        rgb = vec3<f32>(smithay_decode(transfer, rgb.r), smithay_decode(transfer, rgb.g), smithay_decode(transfer, rgb.b));
        if transfer == 13u {
            rgb *= pow(max(dot(rgb, smithay_color.luma.xyz), 0.0), 0.2);
        }
        rgb = vec3<f32>(black) + rgb * (peak - black);
    }
    let matrix = mat3x3<f32>(smithay_color.matrix[0].xyz, smithay_color.matrix[1].xyz, smithay_color.matrix[2].xyz);
    return vec4<f32>(matrix * rgb * smithay_color.parameters.w * color.a, color.a);
}
