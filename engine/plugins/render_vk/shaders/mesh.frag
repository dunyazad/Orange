#version 450

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vUV;

layout(set = 0, binding = 0) uniform sampler2D uTex;

// Shares the push-constant block with the vertex stage (viewProj+model = 128 B);
// .a > 0 => draw this solid color (wireframe-over-solid edges).
layout(push_constant) uniform Push {
    layout(offset = 128) vec4 forceColor;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    if (pc.forceColor.a > 0.0) { outColor = vec4(pc.forceColor.rgb, 1.0); return; }
    outColor = texture(uTex, vUV) * vec4(vColor, 1.0);
}
