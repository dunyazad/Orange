#pragma once

#include <cstddef>
#include <cstdint>

// Backend-agnostic GPU buffer abstraction.
//
// A Buffer is the lowest-level GPU resource: a typed, sized block of device
// memory. Meshes, instance data, and uniforms are all built on top of it. The
// engine refers to buffers by opaque handle so the type crosses the plugin C
// ABI cleanly; each backend maps the handle to its native object (GL buffer
// name / VkBuffer + memory).

namespace orange::render {

// Opaque handle to a renderer-owned GPU buffer. 0 == invalid.
using BufferHandle = uint32_t;
inline constexpr BufferHandle kInvalidBuffer = 0;

// What the buffer feeds. Determines the native binding target / usage flags.
enum class BufferType : uint32_t {
    Vertex  = 0,
    Index   = 1,  // 32-bit indices (uint32_t)
    Uniform = 2,
};

// How often the contents change. Hints the driver's memory placement.
enum class BufferUsage : uint32_t {
    Static  = 0,  // upload once, draw many
    Dynamic = 1,  // updated frequently via updateBuffer()
};

struct BufferDesc {
    BufferType  type  = BufferType::Vertex;
    BufferUsage usage = BufferUsage::Static;
    const void* data  = nullptr;  // optional initial contents (may be null)
    size_t      size  = 0;        // size in bytes
};

// --- Vertex layout ---------------------------------------------------------
// Describes how to interpret the bytes of a vertex buffer when drawing. This
// is what lets a mesh be assembled from arbitrary buffers instead of a fixed
// Vertex struct.

enum class AttributeFormat : uint32_t {
    Float1 = 0,
    Float2 = 1,
    Float3 = 2,
    Float4 = 3,
};

inline uint32_t componentCount(AttributeFormat f) {
    switch (f) {
        case AttributeFormat::Float1: return 1;
        case AttributeFormat::Float2: return 2;
        case AttributeFormat::Float3: return 3;
        case AttributeFormat::Float4: return 4;
    }
    return 0;
}

struct VertexAttribute {
    uint32_t        location = 0;                       // shader input location
    AttributeFormat format   = AttributeFormat::Float3;
    uint32_t        offset   = 0;                       // byte offset in vertex
};

struct VertexLayout {
    uint32_t        stride         = 0;  // bytes per vertex
    uint32_t        attributeCount = 0;
    VertexAttribute attributes[8]{};     // small fixed cap keeps this POD
};

} // namespace orange::render
