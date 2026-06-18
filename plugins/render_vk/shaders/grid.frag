#version 450

layout(push_constant) uniform Push {
    mat4 viewProj;
    mat4 invViewProj;
} pc;

layout(location = 0) in vec3 vNear;
layout(location = 1) in vec3 vFar;
layout(location = 0) out vec4 outColor;

float gridFactor(vec2 coord) {
    vec2 d = fwidth(coord);
    vec2 g = abs(fract(coord - 0.5) - 0.5) / d;
    return 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
}

void main() {
    float t = -vNear.y / (vFar.y - vNear.y);
    if (t <= 0.0) discard;                 // y=0 plane is behind the camera
    vec3 world = vNear + t * (vFar - vNear);

    vec4 clip = pc.viewProj * vec4(world, 1.0);
    if (clip.w <= 0.0) discard;
    gl_FragDepth = clip.z / clip.w;        // Vulkan NDC depth already [0,1]

    float fade = 1.0 - smoothstep(22.0, 75.0, length(world.xz));
    if (fade <= 0.0) discard;

    float minor = gridFactor(world.xz);
    float major = gridFactor(world.xz * 0.1);
    vec3  col = mix(vec3(0.33, 0.35, 0.40), vec3(0.60, 0.63, 0.70), major);
    float a   = max(minor * 0.55, major);

    vec2 aw = fwidth(world.xz);
    if (abs(world.x) < aw.x) {              // Z axis (blue)
        col = vec3(0.30, 0.52, 0.95);
        a   = max(a, 1.0 - clamp(abs(world.x) / aw.x, 0.0, 1.0));
    }
    if (abs(world.z) < aw.y) {              // X axis (red)
        col = vec3(0.95, 0.32, 0.36);
        a   = max(a, 1.0 - clamp(abs(world.z) / aw.y, 0.0, 1.0));
    }

    a *= fade;
    if (a <= 0.001) discard;
    outColor = vec4(col, a);
}
