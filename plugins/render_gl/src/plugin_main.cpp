#include "orange/render/plugin_abi.h"

#include "gl_renderer.h"

// The OpenGL plugin's C ABI exports. The host resolves these by name.
extern "C" {

ORANGE_DECLARE_RENDER_PLUGIN()

ORANGE_PLUGIN_EXPORT const orange::render::PluginInfo* orangeRenderPluginInfo(void) {
    static const orange::render::PluginInfo info{
        ORANGE_PLUGIN_ABI_VERSION,
        orange::render::Backend::OpenGL,
        "Orange OpenGL Renderer",
    };
    return &info;
}

ORANGE_PLUGIN_EXPORT orange::render::IRenderer* orangeRenderCreate(void) {
    return new orange::gl::GLRenderer();
}

ORANGE_PLUGIN_EXPORT void orangeRenderDestroy(orange::render::IRenderer* r) {
    delete r;
}

} // extern "C"
