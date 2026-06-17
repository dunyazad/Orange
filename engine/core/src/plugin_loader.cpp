#include "orange/core/plugin_loader.h"

#include <SDL3/SDL_log.h>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace orange::core {

namespace {

void* openLibrary(const std::string& path) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* getSymbol(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(
        GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

void closeLibrary(void* handle) {
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

const char* lastError() {
#if defined(_WIN32)
    return "see GetLastError()";
#else
    const char* e = dlerror();
    return e ? e : "unknown";
#endif
}

} // namespace

std::string pluginFileName(const std::string& baseName) {
#if defined(_WIN32)
    return baseName + ".dll";
#elif defined(__APPLE__)
    return "lib" + baseName + ".dylib";
#else
    return "lib" + baseName + ".so";
#endif
}

std::unique_ptr<RenderPlugin> RenderPlugin::load(const std::string& dir,
                                                 const std::string& baseName) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/' && path.back() != '\\')
        path += '/';
    path += pluginFileName(baseName);

    void* handle = openLibrary(path);
    if (!handle) {
        SDL_Log("RenderPlugin: failed to load '%s' (%s)", path.c_str(), lastError());
        return nullptr;
    }

    auto infoFn = reinterpret_cast<OrangeRenderPluginInfoFn>(
        getSymbol(handle, ORANGE_PLUGIN_SYM_INFO));
    auto createFn = reinterpret_cast<OrangeRenderCreateFn>(
        getSymbol(handle, ORANGE_PLUGIN_SYM_CREATE));
    auto destroyFn = reinterpret_cast<OrangeRenderDestroyFn>(
        getSymbol(handle, ORANGE_PLUGIN_SYM_DESTROY));

    if (!infoFn || !createFn || !destroyFn) {
        SDL_Log("RenderPlugin: '%s' is missing required exports", path.c_str());
        closeLibrary(handle);
        return nullptr;
    }

    const render::PluginInfo* info = infoFn();
    if (!info || info->abiVersion != ORANGE_PLUGIN_ABI_VERSION) {
        SDL_Log("RenderPlugin: '%s' ABI mismatch (got %u, expected %u)",
                path.c_str(), info ? info->abiVersion : 0u,
                ORANGE_PLUGIN_ABI_VERSION);
        closeLibrary(handle);
        return nullptr;
    }

    render::IRenderer* renderer = createFn();
    if (!renderer) {
        SDL_Log("RenderPlugin: '%s' create returned null", path.c_str());
        closeLibrary(handle);
        return nullptr;
    }

    auto plugin = std::make_unique<RenderPlugin>();
    plugin->handle_   = handle;
    plugin->renderer_ = renderer;
    plugin->info_     = info;
    plugin->destroy_  = destroyFn;

    SDL_Log("RenderPlugin: loaded '%s' (%s backend)", info->name,
            render::to_string(info->backend));
    return plugin;
}

void RenderPlugin::close() {
    if (renderer_ && destroy_) destroy_(renderer_);
    renderer_ = nullptr;
    destroy_  = nullptr;
    if (handle_) closeLibrary(handle_);
    handle_ = nullptr;
}

RenderPlugin::~RenderPlugin() { close(); }

} // namespace orange::core
