#pragma once

#include <cstdint>
#include <vector>

#include "orange/render/types.h"

namespace orange::render {

// Abstract renderer interface implemented by each backend plugin.
//
// NOTE on ABI: this is a C++ abstract class with a vtable. It is safe to pass
// across the shared-library boundary because the whole project is built with a
// single toolchain. If you later want to mix compilers/runtimes, replace this
// with a flat C function table in plugin_abi.h -- the rest of the engine talks
// only to this interface, so the blast radius stays small.
class IRenderer {
public:
    virtual ~IRenderer() = default;

    // Lifecycle ------------------------------------------------------------
    virtual bool init(const InitInfo& info) = 0;
    virtual void shutdown() = 0;

    // Buffers --------------------------------------------------------------
    // Low-level GPU memory. createBuffer uploads optional initial data;
    // updateBuffer overwrites a sub-range (offset+size must fit the buffer).
    virtual BufferHandle createBuffer(const BufferDesc& desc) = 0;
    virtual void         updateBuffer(BufferHandle buffer, const void* data,
                                      size_t size, size_t offset = 0) = 0;
    virtual void         destroyBuffer(BufferHandle buffer) = 0;

    // Textures -------------------------------------------------------------
    virtual TextureHandle createTexture(const TextureDesc& desc) = 0;
    virtual void          destroyTexture(TextureHandle texture) = 0;

    // Meshes ---------------------------------------------------------------
    // Assembled from existing buffer objects + a vertex layout. The mesh does
    // not own its buffers; destroy the buffers separately.
    virtual MeshHandle createMesh(const MeshDesc& desc) = 0;
    virtual void       destroyMesh(MeshHandle mesh) = 0;

    // Frame --------------------------------------------------------------- -
    virtual void beginFrame(const FrameContext& frame) = 0;
    virtual void submit(const DrawItem& item) = 0;
    // Switch to an overlay sub-pass (corner viewport, own view/proj, optional
    // depth clear). Subsequent submit() calls draw into it. Call before endFrame.
    virtual void beginOverlay(const OverlayContext& overlay) = 0;
    virtual void endFrame() = 0;

    // Draws an "infinite" ground grid on the world y=0 plane: a full-screen pass
    // that ray-casts each pixel onto the plane and renders anti-aliased grid lines
    // (fwidth-based) with distance fade and correct depth, so the scene occludes
    // it. Uses the current frame's view/proj/invViewProj. `upAxis` (1 = Y up,
    // 2 = Z up) only recolors the in-plane depth axis line (blue Z vs green Y) to
    // match the gizmo's up-axis toggle; the grid plane itself is unchanged. Call
    // between the scene submits and endFrame().
    virtual void drawGrid(int upAxis) = 0;

    // Events ---------------------------------------------------------------
    virtual void resize(uint32_t width, uint32_t height) = 0;

    // Toggles vertical sync at runtime. GL flips the swap interval; Vulkan
    // recreates the swapchain with a FIFO (on) or immediate/mailbox (off)
    // present mode on the next frame.
    virtual void setVsync(bool enabled) = 0;

    // Sets the drawing mode for subsequent mesh submit()s, mirroring Helium's
    // Renderable DrawingMode: 0 = none (skip), 1 = solid, 2 = wireframe,
    // 3 = wireframe-over-solid (shaded fill + edges on top), 4 = point. GL drives
    // glPolygonMode (+ point size, polygon offset); Vulkan binds the matching
    // pipeline (fill / line / point) and double-draws for mode 3. The host scopes
    // this to scene meshes (it resets to solid before the grid and overlays), so
    // it does not affect drawGrid() or overlay passes.
    virtual void setDrawMode(uint32_t mode) = 0;

    // Sets the pixel diameter of point-cloud sprites (the sphere-imposter points).
    // Affects only meshes created with PrimitiveTopology::Points; ignored by
    // triangle meshes. Both backends drive gl_PointSize from this.
    virtual void setPointSize(float pixels) = 0;

    // Capture --------------------------------------------------------------
    // Reads the last presented frame into `out` as top-left-origin RGBA8.
    // Returns false if unsupported. Call after endFrame().
    virtual bool readPixels(std::vector<uint8_t>& out, uint32_t& width,
                            uint32_t& height) = 0;

    // Introspection --------------------------------------------------------
    virtual Backend     backend() const = 0;
    virtual const char* name() const = 0;
};

} // namespace orange::render
