#pragma once

#include <unordered_map>

#include "orange/render/renderer.h"

struct SDL_Window;

namespace orange::gl {

// A GPU buffer object: a native GL buffer name + its binding target.
struct GLBuffer {
    unsigned int id     = 0;
    unsigned int target = 0;  // GL_ARRAY_BUFFER / GL_ELEMENT_ARRAY_BUFFER / ...
    size_t       size   = 0;
};

// A mesh references buffer objects; it owns only the VAO binding state.
struct GLMesh {
    unsigned int vao     = 0;
    int          count   = 0;  // index count if indexed, else vertex count
    bool         indexed = false;
};

class GLRenderer final : public render::IRenderer {
public:
    bool init(const render::InitInfo& info) override;
    void shutdown() override;

    render::BufferHandle createBuffer(const render::BufferDesc& desc) override;
    void                 updateBuffer(render::BufferHandle buffer, const void* data,
                                      size_t size, size_t offset) override;
    void                 destroyBuffer(render::BufferHandle buffer) override;

    render::TextureHandle createTexture(const render::TextureDesc& desc) override;
    void                  destroyTexture(render::TextureHandle texture) override;

    render::MeshHandle createMesh(const render::MeshDesc& desc) override;
    void               destroyMesh(render::MeshHandle mesh) override;

    void beginFrame(const render::FrameContext& frame) override;
    void submit(const render::DrawItem& item) override;
    void beginOverlay(const render::OverlayContext& overlay) override;
    void endFrame() override;
    void drawGrid(int upAxis) override;

    void resize(uint32_t width, uint32_t height) override;
    void setVsync(bool enabled) override;
    bool readPixels(std::vector<uint8_t>& out, uint32_t& w, uint32_t& h) override;

    render::Backend backend() const override { return render::Backend::OpenGL; }
    const char*     name() const override { return "Orange OpenGL Renderer"; }

private:
    SDL_Window*  window_  = nullptr;
    void*        context_ = nullptr;  // SDL_GLContext
    unsigned int program_ = 0;
    int          uViewProj_ = -1;
    int          uModel_    = -1;
    int          uTex_      = -1;
    uint32_t     width_  = 0;
    uint32_t     height_ = 0;

    // Infinite-grid pass: a vertex-less full-screen shader + an empty VAO.
    unsigned int gridProgram_ = 0;
    unsigned int gridVao_     = 0;
    int          uGridViewProj_    = -1;
    int          uGridInvViewProj_ = -1;
    int          uGridUpAxis_      = -1;
    float        invViewProj_[16];  // inverse(viewProj_), refreshed each beginFrame

    unsigned int whiteTex_ = 0;  // 1x1 white fallback

    render::BufferHandle nextBuffer_ = 1;
    std::unordered_map<render::BufferHandle, GLBuffer> buffers_;

    render::TextureHandle nextTexture_ = 1;
    std::unordered_map<render::TextureHandle, unsigned int> textures_;

    render::MeshHandle nextMesh_ = 1;
    std::unordered_map<render::MeshHandle, GLMesh> meshes_;

    // Cached per-frame view*proj, uploaded once per beginFrame.
    float viewProj_[16];

    bool buildProgram();
    bool buildGridProgram();
};

} // namespace orange::gl
