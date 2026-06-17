#pragma once

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

#include "orange/render/renderer.h"

// Type-safe, RAII buffer object parameterized by its element type.
//
// The renderer's plugin ABI is necessarily handle + byte-size based (you can't
// pass a template across a C ABI / vtable boundary). This host-side template
// wraps that handle API so call sites work in *elements of T* instead of raw
// bytes: size is `count * sizeof(T)`, update() takes a `const T*`, and the GPU
// buffer is released automatically on destruction.
//
//   core::VertexBuffer<Vertex> vbo(renderer, verts, 8);
//   core::IndexBuffer          ibo(renderer, idx,  36);
//   mesh = renderer.createMesh({vbo.handle(), ibo.handle(), Vertex::layout(), ...});

namespace orange::core {

template <typename T, render::BufferType Type>
class TypedBuffer {
public:
    using value_type = T;
    static constexpr render::BufferType kType = Type;

    TypedBuffer() = default;

    TypedBuffer(render::IRenderer& renderer, const T* data, size_t count,
                render::BufferUsage usage = render::BufferUsage::Static)
        : renderer_(&renderer), count_(count) {
        render::BufferDesc desc;
        desc.type  = Type;
        desc.usage = usage;
        desc.data  = data;
        desc.size  = count * sizeof(T);
        handle_ = renderer.createBuffer(desc);
    }

    TypedBuffer(render::IRenderer& renderer, const std::vector<T>& data,
                render::BufferUsage usage = render::BufferUsage::Static)
        : TypedBuffer(renderer, data.data(), data.size(), usage) {}

    TypedBuffer(render::IRenderer& renderer, std::initializer_list<T> data,
                render::BufferUsage usage = render::BufferUsage::Static)
        : TypedBuffer(renderer, data.begin(), data.size(), usage) {}

    ~TypedBuffer() { release(); }

    // Move-only: a buffer owns a unique GPU resource.
    TypedBuffer(TypedBuffer&& other) noexcept { moveFrom(std::move(other)); }
    TypedBuffer& operator=(TypedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            moveFrom(std::move(other));
        }
        return *this;
    }
    TypedBuffer(const TypedBuffer&) = delete;
    TypedBuffer& operator=(const TypedBuffer&) = delete;

    // Overwrite `count` elements starting at element index `firstElement`.
    void update(const T* data, size_t count, size_t firstElement = 0) {
        if (renderer_ && handle_ != render::kInvalidBuffer) {
            renderer_->updateBuffer(handle_, data, count * sizeof(T),
                                    firstElement * sizeof(T));
        }
    }
    void update(const std::vector<T>& data, size_t firstElement = 0) {
        update(data.data(), data.size(), firstElement);
    }

    render::BufferHandle handle()   const { return handle_; }
    size_t               count()    const { return count_; }
    size_t               byteSize() const { return count_ * sizeof(T); }
    bool                 valid()    const { return handle_ != render::kInvalidBuffer; }
    static constexpr render::BufferType type() { return Type; }

private:
    render::IRenderer*   renderer_ = nullptr;
    render::BufferHandle handle_   = render::kInvalidBuffer;
    size_t               count_    = 0;

    void release() {
        if (renderer_ && handle_ != render::kInvalidBuffer) {
            renderer_->destroyBuffer(handle_);
        }
        renderer_ = nullptr;
        handle_   = render::kInvalidBuffer;
        count_    = 0;
    }

    void moveFrom(TypedBuffer&& other) noexcept {
        renderer_ = other.renderer_;
        handle_   = other.handle_;
        count_    = other.count_;
        other.renderer_ = nullptr;
        other.handle_   = render::kInvalidBuffer;
        other.count_    = 0;
    }
};

// Convenience aliases: the buffer kind is fixed by the alias, the element type
// is the template argument.
template <typename T>
using VertexBuffer = TypedBuffer<T, render::BufferType::Vertex>;

template <typename T>
using UniformBuffer = TypedBuffer<T, render::BufferType::Uniform>;

// The renderer expects 32-bit indices, so the element type is fixed here.
using IndexBuffer = TypedBuffer<uint32_t, render::BufferType::Index>;

} // namespace orange::core
