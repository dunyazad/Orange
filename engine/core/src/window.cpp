#include "orange/core/window.h"

#include <SDL3/SDL.h>

namespace orange::core {

namespace {

// The display with the greatest pixel area (the "largest monitor"), or 0.
SDL_DisplayID largestDisplay() {
    int count = 0;
    SDL_DisplayID* ids = SDL_GetDisplays(&count);
    if (!ids || count == 0) {
        if (ids) SDL_free(ids);
        return 0;
    }
    SDL_DisplayID best = ids[0];
    long bestArea = -1;
    for (int i = 0; i < count; ++i) {
        SDL_Rect b;
        if (SDL_GetDisplayBounds(ids[i], &b)) {
            long area = static_cast<long>(b.w) * b.h;
            if (area > bestArea) { bestArea = area; best = ids[i]; }
        }
    }
    SDL_free(ids);
    return best;
}

} // namespace

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

    // Start maximized on the largest monitor. The maximized pixel size arrives
    // later as an SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED, which the app loop uses to
    // resize the renderer -- so we deliberately do NOT call SDL_SyncWindow here
    // (pumping events that early delivered a spurious ESC that quit the app).
    if (SDL_DisplayID disp = largestDisplay()) {
        SDL_SetWindowPosition(window_, SDL_WINDOWPOS_CENTERED_DISPLAY(disp),
                              SDL_WINDOWPOS_CENTERED_DISPLAY(disp));
    }
    SDL_MaximizeWindow(window_);

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
