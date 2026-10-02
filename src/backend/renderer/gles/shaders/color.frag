// Input conversion runs after coverage/opacity, before framebuffer blending.
// The original shader retains its texture/geometry ABI and premultiplication.
uniform highp vec4 smithay_color[5];
highp float smithay_piece(highp float v, highp float cutoff, highp float slope, highp float a, highp float gamma) {
    return v <= cutoff ? v / slope : pow((v + a - 1.0) / a, gamma);
}
highp float smithay_decode(highp float v, highp float tf) {
    highp float p = max(v, 0.0);
    if (tf == 2.0) return pow(p, 2.2);
    if (tf == 3.0) return pow(p, 2.8);
    if (tf == 4.0) return smithay_piece(p, 0.0912, 4.0, 1.1115, 1.0 / 0.45);
    if (tf == 5.0) return v;
    if (tf == 6.0) return v <= 0.0 ? 0.0 : pow(10.0, 2.0 * (v - 1.0));
    if (tf == 7.0) return v <= 0.0 ? 0.0 : pow(10.0, 2.5 * (v - 1.0));
    if (tf == 8.0) return sign(v) * smithay_piece(abs(v), 0.081, 4.5, 1.099, 1.0 / 0.45);
    if (tf == 9.0 || tf == 14.0) return smithay_piece(p, 0.04045, 12.92, 1.055, 2.4);
    if (tf == 10.0) return sign(v) * smithay_piece(abs(v), 0.04045, 12.92, 1.055, 2.4);
    if (tf == 11.0) {
        p = pow(p, 32.0 / 2523.0);
        return pow(max(p - 3424.0 / 4096.0, 0.0) / max(2413.0 / 128.0 - 2392.0 / 128.0 * p, 1e-12), 16384.0 / 2610.0);
    }
    if (tf == 12.0) return pow(p, 2.6) * 52.37 / 48.0;
    if (tf == 13.0) return p <= 0.5 ? p * p / 3.0 : (exp((p - 0.55991073) / 0.17883277) + 0.28466892) / 12.0;
    return v;
}
void main() {
    smithay_original_main();
    highp vec4 params = smithay_color[3];
    if (params.x == 0.0) return;
    highp float a = gl_FragColor.a;
    if (a <= 0.0) { gl_FragColor = vec4(0.0); return; }
    highp vec3 rgb = gl_FragColor.rgb / a;
    if (params.x == 1.0) {
        highp float black = pow(params.y, 1.0 / 2.4);
        highp float range = pow(params.z, 1.0 / 2.4) - black;
        rgb = pow(max(rgb, vec3(0.0)) * range + black, vec3(2.4));
    } else {
        rgb = vec3(smithay_decode(rgb.r, params.x), smithay_decode(rgb.g, params.x), smithay_decode(rgb.b, params.x));
        if (params.x == 13.0) rgb *= pow(max(dot(rgb, smithay_color[4].xyz), 0.0), 0.2);
        rgb = params.y + rgb * (params.z - params.y);
    }
    rgb = mat3(smithay_color[0].xyz, smithay_color[1].xyz, smithay_color[2].xyz) * rgb;
    gl_FragColor = vec4(rgb * (params.w * a), a);
}
