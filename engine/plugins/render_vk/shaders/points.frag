#version 450

// Sphere-imposter fragment shader for point-cloud rendering: rounds each point
// sprite into a disc and shades it like a sphere using gl_PointCoord.

layout(location = 0) in vec3 vColor;
layout(location = 1) in vec2 vUV;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;  // [-1,1] across the sprite
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;               // round, not square
    vec3 N = vec3(d.x, -d.y, sqrt(1.0 - r2));            // sphere normal
    float diff = max(dot(N, normalize(vec3(0.4, 0.7, 0.6))), 0.0) * 0.75 + 0.25;
    outColor = vec4(vColor * diff, 1.0);
}
