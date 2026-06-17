#pragma once

#include <memory>
#include <string>

#include "orange/render/plugin_abi.h"

namespace orange::core {

// Owns a loaded render plugin (the shared library + the renderer instance it
// produced) and tears both down in the right order on destruction.
class RenderPlugin {
public:
    RenderPlugin() = default;
    ~RenderPlugin();

    RenderPlugin(const RenderPlugin&) = delete;
    RenderPlugin& operator=(const RenderPlugin&) = delete;

    // Load a plugin by its base name (e.g. "render_gl"). The platform-specific
    // filename (render_gl.dll / librender_gl.so / librender_gl.dylib) is built
    // automatically and resolved relative to `dir`.
    static std::unique_ptr<RenderPlugin> load(const std::string& dir,
                                              const std::string& baseName);

    render::IRenderer*       renderer()       { return renderer_; }
    const render::PluginInfo* info() const    { return info_; }

private:
    void*                     handle_   = nullptr;  // OS library handle
    render::IRenderer*        renderer_ = nullptr;
    const render::PluginInfo* info_     = nullptr;
    OrangeRenderDestroyFn     destroy_  = nullptr;

    void close();
};

// Build the platform-specific shared-library filename for a base name.
std::string pluginFileName(const std::string& baseName);

} // namespace orange::core
