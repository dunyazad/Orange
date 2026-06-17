#pragma once

#include <cstdint>
#include <string>

#include "orange/render/types.h"

struct SDL_Window;

namespace orange::core {

// Thin SDL3 window wrapper. The window must be created with flags that match
// the chosen render backend (GL context vs. Vulkan surface), so the backend is
// passed in at construction time.
class Window {
public:
    Window() = default;
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool create(const std::string& title, uint32_t width, uint32_t height,
                render::Backend backend);
    void destroy();

    SDL_Window* handle() const { return window_; }
    uint32_t    width()  const { return width_; }
    uint32_t    height() const { return height_; }
    void        setSize(uint32_t w, uint32_t h) { width_ = w; height_ = h; }

private:
    SDL_Window* window_ = nullptr;
    uint32_t    width_  = 0;
    uint32_t    height_ = 0;
};

} // namespace orange::core
