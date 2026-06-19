#version 450

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aUV;

// viewProj+model (128); fragment force-color occupies offset 128; point size at 144.
layout(push_constant) uniform Push {
    mat4 viewProj;  // already includes the Vulkan clip-space correction
    mat4 model;
    layout(offset = 144) float pointSize;
} pc;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vWorld;  // world-space position for cross-section clipping

void main() {
    vColor = aColor;
    vUV = aUV;
    vec4 world = pc.model * vec4(aPos, 1.0);
    vWorld = world.xyz;
    gl_Position = pc.viewProj * world;
    gl_PointSize = pc.pointSize;  // used by the point-cloud pipeline
}
