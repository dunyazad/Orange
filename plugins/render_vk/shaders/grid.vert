#version 450

// Vertex-less full-screen triangle. Unprojects two points along each pixel's
// view ray so the fragment shader can intersect it with the world y=0 plane.
layout(push_constant) uniform Push {
    mat4 viewProj;     // clip-corrected proj*view (used in the FS for depth)
    mat4 invViewProj;  // its inverse (used here to unproject)
} pc;

layout(location = 0) out vec3 vNear;
layout(location = 1) out vec3 vFar;

vec3 unproject(float x, float y, float z) {
    vec4 p = pc.invViewProj * vec4(x, y, z, 1.0);
    return p.xyz / p.w;
}

void main() {
    vec2 c = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
    vNear = unproject(c.x, c.y, 0.0);  // Vulkan NDC near depth = 0
    vFar  = unproject(c.x, c.y, 1.0);  // Vulkan NDC far depth  = 1
    gl_Position = vec4(c, 0.0, 1.0);
}
