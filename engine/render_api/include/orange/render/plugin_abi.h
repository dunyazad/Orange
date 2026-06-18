#pragma once

#include "orange/render/export.h"
#include "orange/render/renderer.h"
#include "orange/render/types.h"

// ---------------------------------------------------------------------------
// The C ABI a render plugin must export.
//
// The host resolves these three symbols by name (no link-time dependency on
// the plugin), so a plugin is hot-swappable: drop a different shared library
// next to the app and load it.
// ---------------------------------------------------------------------------

#define ORANGE_PLUGIN_ABI_VERSION 8u

// Symbol names the host looks up via the dynamic loader.
#define ORANGE_PLUGIN_SYM_INFO    "orangeRenderPluginInfo"
#define ORANGE_PLUGIN_SYM_CREATE  "orangeRenderCreate"
#define ORANGE_PLUGIN_SYM_DESTROY "orangeRenderDestroy"

namespace orange::render {

struct PluginInfo {
    uint32_t    abiVersion;  // must equal ORANGE_PLUGIN_ABI_VERSION
    Backend     backend;
    const char* name;
};

} // namespace orange::render

extern "C" {

// Describe the plugin without instantiating anything.
typedef const orange::render::PluginInfo* (*OrangeRenderPluginInfoFn)(void);

// Create / destroy the renderer instance.
typedef orange::render::IRenderer* (*OrangeRenderCreateFn)(void);
typedef void (*OrangeRenderDestroyFn)(orange::render::IRenderer*);

// Convenience macro for a plugin to declare all three exports.
#define ORANGE_DECLARE_RENDER_PLUGIN()                                          \
    ORANGE_PLUGIN_EXPORT const orange::render::PluginInfo*                      \
        orangeRenderPluginInfo(void);                                          \
    ORANGE_PLUGIN_EXPORT orange::render::IRenderer* orangeRenderCreate(void);   \
    ORANGE_PLUGIN_EXPORT void orangeRenderDestroy(orange::render::IRenderer*);

} // extern "C"
