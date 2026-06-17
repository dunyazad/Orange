#include "gl_renderer.h"

#include <SDL3/SDL.h>

#include <cstring>
#include <vector>

#include "gl_loader.h"

namespace orange::gl {

namespace {

const char* kVertexShader = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec2 aUV;
uniform mat4 uViewProj;
uniform mat4 uModel;
out vec3 vColor;
out vec2 vUV;
void main() {
    vColor = aColor;
    vUV = aUV;
    gl_Position = uViewProj * uModel * vec4(aPos, 1.0);
}
)";

const char* kFragmentShader = R"(#version 330 core
in vec3 vColor;
in vec2 vUV;
uniform sampler2D uTex;
out vec4 FragColor;
void main() {
    FragColor = texture(uTex, vUV) * vec4(vColor, 1.0);
}
)";

GLenum toTarget(render::BufferType type) {
    switch (type) {
        case render::BufferType::Vertex:  return GL_ARRAY_BUFFER;
        case render::BufferType::Index:   return GL_ELEMENT_ARRAY_BUFFER;
        case render::BufferType::Uniform: return GL_UNIFORM_BUFFER;
    }
    return GL_ARRAY_BUFFER;
}

GLenum toUsage(render::BufferUsage usage) {
    return usage == render::BufferUsage::Dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW;
}

unsigned int compile(GLenum type, const char* src) {
    unsigned int sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = GL_FALSE;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        SDL_Log("GLRenderer: shader compile error: %s", log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

} // namespace

bool GLRenderer::init(const render::InitInfo& info) {
    window_  = static_cast<SDL_Window*>(info.nativeWindow);
    width_   = info.width;
    height_  = info.height;

    context_ = SDL_GL_CreateContext(window_);
    if (!context_) {
        SDL_Log("GLRenderer: SDL_GL_CreateContext failed: %s", SDL_GetError());
        return false;
    }
    SDL_GL_MakeCurrent(window_, static_cast<SDL_GLContext>(context_));
    SDL_GL_SetSwapInterval(info.vsync ? 1 : 0);

    if (!loadGLFunctions()) {
        SDL_Log("GLRenderer: failed to load GL functions");
        return false;
    }

    if (!buildProgram()) return false;

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 1x1 white fallback texture so untextured draws keep their vertex color.
    const unsigned char white[4] = {255, 255, 255, 255};
    glGenTextures(1, &whiteTex_);
    glBindTexture(GL_TEXTURE_2D, whiteTex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glViewport(0, 0, static_cast<GLsizei>(width_), static_cast<GLsizei>(height_));

    std::memset(viewProj_, 0, sizeof(viewProj_));
    viewProj_[0] = viewProj_[5] = viewProj_[10] = viewProj_[15] = 1.0f;
    return true;
}

bool GLRenderer::buildProgram() {
    unsigned int vs = compile(GL_VERTEX_SHADER, kVertexShader);
    unsigned int fs = compile(GL_FRAGMENT_SHADER, kFragmentShader);
    if (!vs || !fs) return false;

    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);

    GLint ok = GL_FALSE;
    glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
        SDL_Log("GLRenderer: program link error: %s", log);
        return false;
    }

    uViewProj_ = glGetUniformLocation(program_, "uViewProj");
    uModel_    = glGetUniformLocation(program_, "uModel");
    uTex_      = glGetUniformLocation(program_, "uTex");
    glUseProgram(program_);
    glUniform1i(uTex_, 0);  // sampler uses texture unit 0
    return true;
}

render::TextureHandle GLRenderer::createTexture(const render::TextureDesc& desc) {
    if (!desc.pixels || desc.width == 0 || desc.height == 0)
        return render::kInvalidTexture;
    // Atlas data and UVs share the "row 0 = top, v = 0 = top" convention, so the
    // pixels upload as-is (no flip) -- matching the Vulkan backend.
    unsigned int tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(desc.width),
                 static_cast<GLsizei>(desc.height), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 desc.pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    render::TextureHandle handle = nextTexture_++;
    textures_[handle] = tex;
    return handle;
}

void GLRenderer::destroyTexture(render::TextureHandle handle) {
    auto it = textures_.find(handle);
    if (it == textures_.end()) return;
    glDeleteTextures(1, &it->second);
    textures_.erase(it);
}

void GLRenderer::shutdown() {
    for (auto& [handle, mesh] : meshes_) glDeleteVertexArrays(1, &mesh.vao);
    meshes_.clear();
    for (auto& [handle, buf] : buffers_) glDeleteBuffers(1, &buf.id);
    buffers_.clear();
    for (auto& [handle, tex] : textures_) glDeleteTextures(1, &tex);
    textures_.clear();
    if (whiteTex_) { glDeleteTextures(1, &whiteTex_); whiteTex_ = 0; }
    if (program_) {
        glDeleteProgram(program_);
        program_ = 0;
    }
    if (context_) {
        SDL_GL_DestroyContext(static_cast<SDL_GLContext>(context_));
        context_ = nullptr;
    }
}

render::BufferHandle GLRenderer::createBuffer(const render::BufferDesc& desc) {
    if (desc.size == 0) return render::kInvalidBuffer;

    GLBuffer buf;
    buf.target = toTarget(desc.type);
    buf.size   = desc.size;
    glGenBuffers(1, &buf.id);
    glBindBuffer(buf.target, buf.id);
    glBufferData(buf.target, static_cast<GLsizeiptr>(desc.size), desc.data,
                 toUsage(desc.usage));
    glBindBuffer(buf.target, 0);

    render::BufferHandle handle = nextBuffer_++;
    buffers_[handle] = buf;
    return handle;
}

void GLRenderer::updateBuffer(render::BufferHandle handle, const void* data,
                              size_t size, size_t offset) {
    auto it = buffers_.find(handle);
    if (it == buffers_.end() || !data) return;
    const GLBuffer& buf = it->second;
    if (offset + size > buf.size) {
        SDL_Log("GLRenderer: updateBuffer out of range (%zu+%zu > %zu)", offset,
                size, buf.size);
        return;
    }
    glBindBuffer(buf.target, buf.id);
    glBufferSubData(buf.target, static_cast<GLintptr>(offset),
                    static_cast<GLsizeiptr>(size), data);
    glBindBuffer(buf.target, 0);
}

void GLRenderer::destroyBuffer(render::BufferHandle handle) {
    auto it = buffers_.find(handle);
    if (it == buffers_.end()) return;
    glDeleteBuffers(1, &it->second.id);
    buffers_.erase(it);
}

render::MeshHandle GLRenderer::createMesh(const render::MeshDesc& desc) {
    auto vbIt = buffers_.find(desc.vertexBuffer);
    if (vbIt == buffers_.end() || desc.vertexCount == 0) {
        SDL_Log("GLRenderer: createMesh needs a valid vertex buffer");
        return render::kInvalidMesh;
    }

    GLMesh mesh;
    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);

    // Bind the vertex buffer and wire up attributes from the layout.
    glBindBuffer(GL_ARRAY_BUFFER, vbIt->second.id);
    const render::VertexLayout& layout = desc.layout;
    for (uint32_t i = 0; i < layout.attributeCount; ++i) {
        const render::VertexAttribute& a = layout.attributes[i];
        glEnableVertexAttribArray(a.location);
        glVertexAttribPointer(a.location,
                              static_cast<GLint>(render::componentCount(a.format)),
                              GL_FLOAT, GL_FALSE,
                              static_cast<GLsizei>(layout.stride),
                              reinterpret_cast<const void*>(
                                  static_cast<size_t>(a.offset)));
    }

    // Optional index buffer.
    auto ibIt = buffers_.find(desc.indexBuffer);
    if (desc.indexBuffer != render::kInvalidBuffer && ibIt != buffers_.end() &&
        desc.indexCount > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibIt->second.id);
        mesh.indexed = true;
        mesh.count   = static_cast<int>(desc.indexCount);
    } else {
        mesh.indexed = false;
        mesh.count   = static_cast<int>(desc.vertexCount);
    }

    glBindVertexArray(0);

    render::MeshHandle handle = nextMesh_++;
    meshes_[handle] = mesh;
    return handle;
}

void GLRenderer::destroyMesh(render::MeshHandle handle) {
    auto it = meshes_.find(handle);
    if (it == meshes_.end()) return;
    glDeleteVertexArrays(1, &it->second.vao);  // buffers are owned separately
    meshes_.erase(it);
}

void GLRenderer::beginFrame(const render::FrameContext& frame) {
    SDL_GL_MakeCurrent(window_, static_cast<SDL_GLContext>(context_));
    width_  = frame.width;   // remember for overlay y-flip
    height_ = frame.height;
    glViewport(0, 0, static_cast<GLsizei>(frame.width), static_cast<GLsizei>(frame.height));
    glClearColor(frame.clearColor[0], frame.clearColor[1], frame.clearColor[2],
                 frame.clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // viewProj = proj * view (column-major multiply).
    const float* a = frame.proj;
    const float* b = frame.view;
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + r] * b[c * 4 + k];
            viewProj_[c * 4 + r] = s;
        }

    glUseProgram(program_);
    glUniformMatrix4fv(uViewProj_, 1, GL_FALSE, viewProj_);
}

void GLRenderer::submit(const render::DrawItem& item) {
    auto it = meshes_.find(item.mesh);
    if (it == meshes_.end()) return;
    const GLMesh& mesh = it->second;

    unsigned int tex = whiteTex_;
    auto texIt = textures_.find(item.texture);
    if (texIt != textures_.end()) tex = texIt->second;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    glUniformMatrix4fv(uModel_, 1, GL_FALSE, item.model);
    glBindVertexArray(mesh.vao);
    if (mesh.indexed)
        glDrawElements(GL_TRIANGLES, mesh.count, GL_UNSIGNED_INT, nullptr);
    else
        glDrawArrays(GL_TRIANGLES, 0, mesh.count);
}

void GLRenderer::beginOverlay(const render::OverlayContext& ov) {
    // GL viewport/scissor use a bottom-left origin; flip the top-left rect.
    GLint gy = static_cast<GLint>(height_) - ov.y - ov.height;
    glViewport(ov.x, gy, ov.width, ov.height);

    if (ov.clearDepth) {
        glEnable(GL_SCISSOR_TEST);
        glScissor(ov.x, gy, ov.width, ov.height);
        glClear(GL_DEPTH_BUFFER_BIT);  // clears to 1.0 -> overlay draws on top
        glDisable(GL_SCISSOR_TEST);
    }

    // viewProj = proj * view (column-major).
    const float* a = ov.proj;
    const float* b = ov.view;
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + r] * b[c * 4 + k];
            viewProj_[c * 4 + r] = s;
        }
    glUseProgram(program_);
    glUniformMatrix4fv(uViewProj_, 1, GL_FALSE, viewProj_);
}

void GLRenderer::endFrame() {
    glBindVertexArray(0);
    SDL_GL_SwapWindow(window_);
}

void GLRenderer::resize(uint32_t width, uint32_t height) {
    width_  = width;
    height_ = height;
}

bool GLRenderer::readPixels(std::vector<uint8_t>& out, uint32_t& w, uint32_t& h) {
    SDL_GL_MakeCurrent(window_, static_cast<SDL_GLContext>(context_));
    w = width_;
    h = height_;
    if (w == 0 || h == 0) return false;
    out.resize(static_cast<size_t>(w) * h * 4);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_FRONT);  // the presented frame
    glReadPixels(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h), GL_RGBA,
                 GL_UNSIGNED_BYTE, out.data());
    glReadBuffer(GL_BACK);

    // GL returns bottom-up rows; flip to top-down for image files.
    size_t stride = static_cast<size_t>(w) * 4;
    std::vector<uint8_t> tmp(stride);
    for (uint32_t y = 0; y < h / 2; ++y) {
        uint8_t* a = out.data() + static_cast<size_t>(y) * stride;
        uint8_t* b = out.data() + static_cast<size_t>(h - 1 - y) * stride;
        std::memcpy(tmp.data(), a, stride);
        std::memcpy(a, b, stride);
        std::memcpy(b, tmp.data(), stride);
    }
    return true;
}

} // namespace orange::gl
