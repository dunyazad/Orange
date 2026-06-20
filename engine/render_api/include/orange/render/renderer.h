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
    // match the gizmo's up-axis toggle; the grid plane itself is unchanged.
    // `cellSize` is the world size of one minor grid cell (major lines every 10);
    // the host derives it from the loaded model's size so the grid scales with it.
    // `cameraPos` is the camera's world position and `viewRadius` how far it can see
    // across the ground; the grid fades around the camera over that radius so it
    // always fills the view at any zoom (not a fixed disc around the origin).
    // Call between the scene submits and endFrame().
    virtual void drawGrid(int upAxis, float cellSize, const float cameraPos[3],
                          float viewRadius) = 0;

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

    // Toggles per-fragment diffuse lighting on scene geometry. Point-cloud sprites
    // shade via their sphere-imposter normal; triangle meshes (which carry no vertex
    // normals) shade via a flat face normal derived from screen-space derivatives of
    // the world position. When off, both draw as their flat (mode-mapped) color. The
    // grid and overlays are never shaded. Both backends carry this as a shader flag
    // (GL uniform / VK fragment push constant). Default: on.
    virtual void setLighting(bool enabled) = 0;

    // Cross-section clipping. When `enabled`, scene-mesh fragments on the positive
    // side of the world-space plane `plane` = (nx, ny, nz, d) -- i.e. where
    // dot(worldPos, (nx,ny,nz)) + d > 0 -- are discarded, revealing the cut
    // interior. Applies to triangle meshes AND point clouds; the grid and overlays
    // are never clipped. `plane` is in render-world space (after the model & up-axis
    // basis). `enabled` = false (or a zero normal) disables clipping. Both backends
    // carry this as a shader uniform / fragment push constant. Default: off.
    virtual void setCrossSection(bool enabled, const float plane[4]) = 0;

    // Selects how scene meshes are colored (Shift+` cycles it). The fragment shader
    // derives the base color from this and the world position: 0 = default (vertex
    // color), 1 = height (world Y -> jet heatmap), 2 = position (world XYZ -> RGB),
    // 3 = grayscale (luminance of the vertex color). Applies to triangle meshes and
    // point clouds; the grid and overlays always use mode 0. Both backends carry it
    // as a shader uniform / fragment push constant. Default: 0.
    virtual void setColorMode(uint32_t mode) = 0;

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
