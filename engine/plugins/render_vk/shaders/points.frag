#version 450

// Sphere-imposter fragment shader for point-cloud rendering: rounds each point
// sprite into a disc and shades it like a sphere using gl_PointCoord.

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vUV;
layout(location = 2) in vec3 vWorld;

// Lighting flag at offset 148 (fragment stage): 1 => diffuse shade, 0 => flat.
// Cross-section clip plane (nx,ny,nz,d) at offset 160; zero normal = disabled.
// Color mode at offset 176 (0 default / 1 height / 2 position / 3 grayscale).
layout(push_constant) uniform Push {
    layout(offset = 148) int lighting;
    layout(offset = 160) vec4 clipPlane;
    layout(offset = 176) int colorMode;
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
    vec2 d = gl_PointCoord * 2.0 - 1.0;  // [-1,1] across the sprite
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;               // round, not square
    vec3 N = vec3(d.x, -d.y, sqrt(1.0 - r2));            // sphere normal
    float diff = pc.lighting == 1
        ? max(dot(N, normalize(vec3(0.4, 0.7, 0.6))), 0.0) * 0.75 + 0.25
        : 1.0;                                           // unlit: flat color
    outColor = vec4(colorFor(vColor, vWorld) * diff, 1.0);
}
