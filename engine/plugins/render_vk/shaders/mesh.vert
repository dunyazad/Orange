#version 450

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aUV;

// 128 bytes = the guaranteed minimum push-constant size (two mat4).
layout(push_constant) uniform Push {
    mat4 viewProj;  // already includes the Vulkan clip-space correction
    mat4 model;
} pc;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec2 vUV;

void main() {
    vColor = aColor;
    vUV = aUV;
    gl_Position = pc.viewProj * pc.model * vec4(aPos, 1.0);
    gl_PointSize = 3.0;  // used when the point-mode pipeline is bound
}
