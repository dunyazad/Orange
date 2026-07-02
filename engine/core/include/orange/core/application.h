#pragma once

#include <functional>
#include <memory>
#include <string>

#include <entt/entt.hpp>

#include "orange/core/input.h"
#include "orange/core/plugin_loader.h"
#include "orange/core/window.h"
#include "orange/render/types.h"

namespace orange::core {

struct AppConfig {
    std::string     title       = "Orange";
    uint32_t        width       = 1280;
    uint32_t        height      = 720;
    render::Backend backend     = render::Backend::OpenGL;
    bool            vsync       = true;

    // Plugin base name per backend ("render_gl" / "render_vk"). Empty => auto.
    std::string     pluginName;
    // Directory to search for plugins. Empty => directory of the executable.
    std::string     pluginDir;
};

// Owns the window, the render plugin, and the ECS world, and drives the loop.
class Application {
public:
    Application() = default;
    ~Application();

    bool init(const AppConfig& config);

    // Run until quit. `onUpdate(world, dt)` runs every frame before rendering;
    // use it for game logic / spawning. May be empty.
    void run(const std::function<void(entt::registry&, float)>& onUpdate = {});

    entt::registry&    world()    { return world_; }
    render::IRenderer* renderer() { return plugin_ ? plugin_->renderer() : nullptr; }
    Window&            window()   { return window_; }
    const Input&       input()    { return input_; }

private:
    AppConfig                     config_;
    Window                        window_;
    std::unique_ptr<RenderPlugin> plugin_;
    entt::registry                world_;
    Input                         input_;
    bool                          running_ = false;
    bool                          capture_ = false;  // 'C' pressed -> screenshot
    float                         pointSize_ = 6.0f; // point-cloud sprite size (+/- keys)
    bool                          lighting_  = true; // point-sprite shading (` toggles)
    uint32_t                      colorMode_ = 0;    // legacy base; per-mesh Renderable.colorMode drives the scene now
    bool                          vsync_     = true; // swap interval (menu toggles it)

    // 3D tooth-segmentation view: a per-tooth-coloured copy of the source mesh,
    // spawned while the source is hidden. Tracked so a re-run cleans up the old one.
    entt::entity         segEntity_ = entt::null;
    entt::entity         segSource_ = entt::null;
    render::MeshHandle   segMesh_   = render::kInvalidMesh;
    render::BufferHandle segVbo_    = render::kInvalidBuffer;

    // Dispatch a MenuBar action (mirrors the keyboard shortcuts); set checkmarks on
    // the menu items from the live state before drawing. Both run each frame.
    void applyMenuAction(int action);
    void syncMenu();
    void recomputeCusps();      // re-run findCusps from cached CuspParams, refresh CuspViz
    bool fitPlaneFromCusps();   // PCA-fit the (non-outlier) cusp tips -> OcclusalPlaneViz

    static std::string defaultPluginName(render::Backend backend);
    static std::string executableDir();
};

} // namespace orange::core
