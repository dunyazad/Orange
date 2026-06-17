#include "orange/core/application.h"

#include <SDL3/SDL.h>

#include "orange/core/screenshot.h"
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

        if (onUpdate) onUpdate(world_, dt);
        ecs::fpsWidgetInputSystem(world_, input_, dt, window_.width(), window_.height());
        ecs::cameraControlsInputSystem(world_, input_, window_.width(), window_.height());
        ecs::axisGizmoInputSystem(world_, input_, dt, window_.width(), window_.height());
        ecs::cameraManipulatorSystem(world_, input_, dt);
        ecs::spinSystem(world_, dt);
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
