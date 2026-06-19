#version 450

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vUV;
layout(location = 2) in vec3 vWorld;

layout(set = 0, binding = 0) uniform sampler2D uTex;

// Shares the push-constant block with the vertex stage (viewProj+model = 128 B);
// .a > 0 => draw this solid color (wireframe-over-solid edges). Cross-section clip
// plane (nx,ny,nz,d) at offset 160; a zero normal disables clipping. Color mode at
// offset 176 (0 default / 1 height / 2 position / 3 grayscale).
layout(push_constant) uniform Push {
    layout(offset = 128) vec4 forceColor;
    layout(offset = 148) int  lighting;
    layout(offset = 160) vec4 clipPlane;
    layout(offset = 176) int  colorMode;
} pc;

layout(location = 0) out vec4 outColor;

vec3 jet(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3(1.5 - abs(4.0 * t - 3.0),
                      1.5 - abs(4.0 * t - 2.0),
                      1.5 - abs(4.0 * t - 1.0)), 0.0, 1.0);
}
vec3 colorFor(vec3 c, vec3 w) {
    if (pc.colorMode == 1) return jet(w.y * 0.15 + 0.5);
    if (pc.colorMode == 2) return clamp(w * 0.15 + 0.5, 0.0, 1.0);
    if (pc.colorMode == 3) return vec3(dot(c, vec3(0.299, 0.587, 0.114)));
    return c;
}

void main() {
    if (dot(pc.clipPlane.xyz, pc.clipPlane.xyz) > 0.0 &&
        dot(vWorld, pc.clipPlane.xyz) + pc.clipPlane.w > 0.0) discard;
    if (pc.forceColor.a > 0.0) { outColor = vec4(pc.forceColor.rgb, 1.0); return; }
    vec4 texc = texture(uTex, vUV) * vec4(colorFor(vColor, vWorld), 1.0);  // keep alpha
    vec3 col = texc.rgb;
    if (pc.lighting == 1) {
        // No vertex normals: derive a flat face normal from screen-space derivatives.
        vec3 N = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
        float diff = max(abs(dot(N, normalize(vec3(0.4, 0.7, 0.6)))), 0.0) * 0.7 + 0.3;
        col *= diff;
    }
    outColor = vec4(col, texc.a);
}
