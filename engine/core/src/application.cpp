#include "orange/core/application.h"

#include <SDL3/SDL.h>

#include <vector>

#include "orange/core/draw_mode.h"
#include "orange/core/modes.h"
#include "orange/core/screenshot.h"
#include "orange/ecs/components.h"
#include "orange/ecs/systems.h"

namespace orange::core {

std::string Application::defaultPluginName(render::Backend backend) {
    switch (backend) {
        case render::Backend::OpenGL: return "render_gl";
        case render::Backend::Vulkan: return "render_vk";
    }
    return "render_gl";
}

std::string Application::executableDir() {
    const char* base = SDL_GetBasePath();  // owned by SDL, do not free
    return base ? std::string(base) : std::string();
}

bool Application::init(const AppConfig& config) {
    config_ = config;

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        SDL_Log("Application: SDL_Init failed: %s", SDL_GetError());
        return false;
    }

    const std::string dir =
        config_.pluginDir.empty() ? executableDir() : config_.pluginDir;
    const std::string name =
        config_.pluginName.empty() ? defaultPluginName(config_.backend)
                                   : config_.pluginName;

    plugin_ = RenderPlugin::load(dir, name);
    if (!plugin_) {
        SDL_Log("Application: could not load render plugin '%s'", name.c_str());
        return false;
    }

    // The window must match the backend the plugin actually provides.
    const render::Backend backend = plugin_->info()->backend;
    if (!window_.create(config_.title, config_.width, config_.height, backend)) {
        return false;
    }

    render::InitInfo info;
    info.nativeWindow = window_.handle();
    info.width        = window_.width();
    info.height       = window_.height();
    info.vsync        = config_.vsync;

    if (!plugin_->renderer()->init(info)) {
        SDL_Log("Application: renderer init failed");
        return false;
    }

    SDL_Log("Application: initialized with %s backend",
            render::to_string(backend));
    return true;
}

void Application::run(const std::function<void(entt::registry&, float)>& onUpdate) {
    if (!plugin_ || !plugin_->renderer()) return;
    running_ = true;

    Uint64 last = SDL_GetPerformanceCounter();
    const double freq = static_cast<double>(SDL_GetPerformanceFrequency());

    while (running_) {
        input_.newFrame();  // reset per-frame deltas before draining events

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_EVENT_QUIT:
                    running_ = false;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (e.key.key == SDLK_ESCAPE) running_ = false;
                    if (e.key.scancode == SDL_SCANCODE_C) capture_ = true;  // screenshot
                    if (e.key.scancode == SDL_SCANCODE_TAB) {  // cycle each selected mesh's draw mode
                        auto v = world_.view<ecs::Renderable>();
                        // With a single mesh, Tab applies to it without an explicit
                        // selection (there's nothing else it could mean).
                        size_t count = 0;
                        for (auto ent : v) { (void)ent; ++count; }
                        for (auto ent : v) {
                            auto& r = v.get<ecs::Renderable>(ent);
                            if (!r.selected && count != 1) continue;
                            r.drawMode = static_cast<DrawMode>(
                                (static_cast<uint32_t>(r.drawMode) + 1) %
                                static_cast<uint32_t>(DrawMode::Count));
                        }
                    }
                    if (e.key.scancode == SDL_SCANCODE_M) {                 // cycle processing mode
                        auto& ctx = world_.ctx();
                        if (!ctx.contains<modes::ModeState>()) ctx.emplace<modes::ModeState>();
                        auto& ms = ctx.get<modes::ModeState>();
                        ms.index = (ms.index + 1) % modes::modeCount();
                        ms.generation++;
                        SDL_Log("Application: processing mode = %s", modes::modeName(ms.index));
                    }
                    if (e.key.scancode == SDL_SCANCODE_A && (e.key.mod & SDL_KMOD_CTRL)) {
                        // Ctrl+A: select every Renderable that is visible on screen.
                        auto v = world_.view<ecs::Renderable>();
                        for (auto ent : v)
                            if (ecs::entityVisibleOnScreen(world_, ent, window_.width(),
                                                           window_.height()))
                                v.get<ecs::Renderable>(ent).selected = true;
                    }
                    if (e.key.scancode == SDL_SCANCODE_DELETE) {
                        // Delete the selected meshes (collect first, then destroy).
                        std::vector<entt::entity> dead;
                        auto v = world_.view<ecs::Renderable>();
                        for (auto ent : v)
                            if (v.get<ecs::Renderable>(ent).selected) dead.push_back(ent);
                        for (auto ent : dead) world_.destroy(ent);
                        if (!dead.empty())
                            SDL_Log("Application: deleted %zu mesh(es)", dead.size());
                    }
                    if (e.key.scancode == SDL_SCANCODE_EQUALS ||
                        e.key.scancode == SDL_SCANCODE_KP_PLUS) {  // grow point sprites
                        pointSize_ = pointSize_ + 1.0f > 64.0f ? 64.0f : pointSize_ + 1.0f;
                        plugin_->renderer()->setPointSize(pointSize_);
                        SDL_Log("Application: point size = %.0f", pointSize_);
                    }
                    if (e.key.scancode == SDL_SCANCODE_MINUS ||
                        e.key.scancode == SDL_SCANCODE_KP_MINUS) {  // shrink point sprites
                        pointSize_ = pointSize_ - 1.0f < 1.0f ? 1.0f : pointSize_ - 1.0f;
                        plugin_->renderer()->setPointSize(pointSize_);
                        SDL_Log("Application: point size = %.0f", pointSize_);
                    }
                    if (e.key.scancode == SDL_SCANCODE_H) {
                        // Unhide all: reveal meshes hidden by the None draw mode so
                        // they can be seen and selected again.
                        int n = 0;
                        auto v = world_.view<ecs::Renderable>();
                        for (auto ent : v) {
                            auto& r = v.get<ecs::Renderable>(ent);
                            if (r.drawMode == DrawMode::None) { r.drawMode = DrawMode::Solid; ++n; }
                        }
                        if (n) SDL_Log("Application: revealed %d hidden mesh(es)", n);
                    }
                    break;
                case SDL_EVENT_MOUSE_MOTION: {
                    // SDL reports mouse in logical points; the render viewport
                    // (and gizmo rect) are in pixels. Convert so picking lines up.
                    int lw = 0, lh = 0;
                    SDL_GetWindowSize(window_.handle(), &lw, &lh);
                    float sx = lw > 0 ? static_cast<float>(window_.width()) / lw : 1.0f;
                    float sy = lh > 0 ? static_cast<float>(window_.height()) / lh : 1.0f;
                    input_.mousePosX = e.motion.x * sx;
                    input_.mousePosY = e.motion.y * sy;
                    input_.mouseDeltaX += e.motion.xrel * sx;
                    input_.mouseDeltaY += e.motion.yrel * sy;
                    break;
                }
                case SDL_EVENT_MOUSE_WHEEL:
                    input_.wheel += e.wheel.y;
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                case SDL_EVENT_MOUSE_BUTTON_UP: {
                    bool down = (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
                    if (e.button.button == SDL_BUTTON_LEFT) {
                        input_.buttonLeft = down;
                        if (down) input_.leftClicked = true;
                    }
                    if (e.button.button == SDL_BUTTON_RIGHT)  input_.buttonRight  = down;
                    if (e.button.button == SDL_BUTTON_MIDDLE) input_.buttonMiddle = down;
                    break;
                }
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: {
                    uint32_t w = static_cast<uint32_t>(e.window.data1);
                    uint32_t h = static_cast<uint32_t>(e.window.data2);
                    window_.setSize(w, h);
                    plugin_->renderer()->resize(w, h);
                    break;
                }
                default:
                    break;
            }
        }

        Uint64 now = SDL_GetPerformanceCounter();
        float dt = static_cast<float>((now - last) / freq);
        last = now;

        input_.shift = (SDL_GetModState() & SDL_KMOD_SHIFT) != 0;
        input_.ctrl  = (SDL_GetModState() & SDL_KMOD_CTRL) != 0;

        if (onUpdate) onUpdate(world_, dt);
        ecs::menuBarInputSystem(world_, input_, window_.width(), window_.height());
        ecs::fpsWidgetInputSystem(world_, input_, dt, window_.width(), window_.height());
        ecs::cameraControlsInputSystem(world_, input_, dt, window_.width(), window_.height());
        ecs::axisGizmoInputSystem(world_, input_, dt, window_.width(), window_.height());
        ecs::cameraManipulatorSystem(world_, input_, dt);
        ecs::pickingSystem(world_, input_, window_.width(), window_.height());
        ecs::spinSystem(world_, dt);
        ecs::processingModeSystem(world_);  // emits the active mode's debug geometry
        ecs::renderSystem(world_, *plugin_->renderer(), window_.width(),
                          window_.height());

        if (capture_) {
            capture_ = false;
            saveScreenshot(*plugin_->renderer(), executableDir() + "orange_capture.png");
        }
    }
}

Application::~Application() {
    if (plugin_ && plugin_->renderer()) plugin_->renderer()->shutdown();
    plugin_.reset();
    window_.destroy();
    SDL_Quit();
}

} // namespace orange::core
