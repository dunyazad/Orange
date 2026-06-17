#pragma once

#include <cstddef>
#include <cstdint>

#include "orange/render/buffer.h"

// Plain data types crossing the engine <-> render-plugin boundary.
// Deliberately POD / trivially-copyable so the boundary stays ABI-friendly.

namespace orange::render {

enum class Backend : uint32_t {
    OpenGL = 1,
    Vulkan = 2,
};

inline const char* to_string(Backend b) {
    switch (b) {
        case Backend::OpenGL: return "OpenGL";
        case Backend::Vulkan: return "Vulkan";
    }
    return "Unknown";
}

// A handle to a GPU mesh owned by the renderer. 0 == invalid.
using MeshHandle = uint32_t;
inline constexpr MeshHandle kInvalidMesh = 0;

// A handle to a GPU texture. 0 == invalid (draws fall back to a 1x1 white tex).
using TextureHandle = uint32_t;
inline constexpr TextureHandle kInvalidTexture = 0;

// 8-bit RGBA texture upload description (tightly packed, row-major, top row first).
struct TextureDesc {
    uint32_t       width  = 0;
    uint32_t       height = 0;
    const uint8_t* pixels = nullptr;  // width*height*4 bytes, RGBA8
};

// A convenience interleaved vertex used by the sandbox. Backends never depend
// on it -- they only see raw buffers + a VertexLayout. layout() returns the
// matching layout so callers don't hand-write offsets.
struct Vertex {
    float position[3];
    float color[3];
    float uv[2];  // omitted in aggregate init => {0,0} (fine for untextured meshes)

    static VertexLayout layout() {
        VertexLayout l;
        l.stride         = sizeof(Vertex);
        l.attributeCount = 3;
        l.attributes[0]  = {0, AttributeFormat::Float3, offsetof(Vertex, position)};
        l.attributes[1]  = {1, AttributeFormat::Float3, offsetof(Vertex, color)};
        l.attributes[2]  = {2, AttributeFormat::Float2, offsetof(Vertex, uv)};
        return l;
    }
};

// A mesh is now a *composition* over buffer objects: a vertex buffer (required),
// an optional index buffer, and the layout describing the vertex bytes. The
// renderer owns the GPU-side binding state (e.g. a GL VAO) but NOT the buffers,
// whose lifetime the caller controls via destroyBuffer().
struct MeshDesc {
    BufferHandle vertexBuffer = kInvalidBuffer;
    BufferHandle indexBuffer  = kInvalidBuffer;  // kInvalidBuffer => non-indexed
    VertexLayout layout{};
    uint32_t     vertexCount  = 0;
    uint32_t     indexCount   = 0;  // used only when indexBuffer is valid
};

// Everything the renderer needs to stand up a swapchain / GL context.
// nativeWindow is an SDL_Window* (the engine always uses SDL3).
struct InitInfo {
    void*    nativeWindow = nullptr;
    uint32_t width        = 0;
    uint32_t height       = 0;
    bool     vsync        = true;
};

// Per-frame, view-wide state. Matrices are column-major (OpenGL convention).
struct FrameContext {
    float    view[16];
    float    proj[16];
    float    clearColor[4] = {0.05f, 0.06f, 0.08f, 1.0f};
    uint32_t width  = 0;
    uint32_t height = 0;
};

// One drawable, submitted between beginFrame()/endFrame().
struct DrawItem {
    MeshHandle    mesh    = kInvalidMesh;
    TextureHandle texture = kInvalidTexture;  // invalid => 1x1 white
    float         model[16];                  // column-major model matrix
};

// Begins an overlay sub-pass within the current frame: restricts subsequent
// draws to a pixel rect (origin top-left) with its own view/proj, optionally
// clearing depth so the overlay (e.g. an axis gizmo) draws on top of the scene.
// Call after the scene's submits, before endFrame().
struct OverlayContext {
    int   x      = 0;
    int   y      = 0;
    int   width  = 0;
    int   height = 0;
    float view[16];
    float proj[16];
    bool  clearDepth = true;
};

} // namespace orange::render
