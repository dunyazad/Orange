#include "orange/core/window.h"

#include <SDL3/SDL.h>

namespace orange::core {

bool Window::create(const std::string& title, uint32_t width, uint32_t height,
                    render::Backend backend) {
    SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;

    if (backend == render::Backend::OpenGL) {
        // GL attributes must be set before window creation.
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                            SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        flags |= SDL_WINDOW_OPENGL;
    } else if (backend == render::Backend::Vulkan) {
        flags |= SDL_WINDOW_VULKAN;
    }

    window_ = SDL_CreateWindow(title.c_str(), static_cast<int>(width),
                               static_cast<int>(height), flags);
    if (!window_) {
        SDL_Log("Window: SDL_CreateWindow failed: %s", SDL_GetError());
        return false;
    }

    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    width_  = static_cast<uint32_t>(w);
    height_ = static_cast<uint32_t>(h);
    return true;
}

void Window::destroy() {
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
}

Window::~Window() { destroy(); }

} // namespace orange::core
