#include "orange/core/screenshot.h"

#include <cstdint>
#include <vector>

#include <SDL3/SDL_log.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace orange::core {

bool saveScreenshot(render::IRenderer& renderer, const std::string& path) {
    std::vector<uint8_t> pixels;
    uint32_t w = 0, h = 0;
    if (!renderer.readPixels(pixels, w, h) || w == 0 || h == 0) {
        SDL_Log("Screenshot: readPixels failed");
        return false;
    }
    int ok = stbi_write_png(path.c_str(), static_cast<int>(w), static_cast<int>(h),
                            4, pixels.data(), static_cast<int>(w) * 4);
    if (!ok) {
        SDL_Log("Screenshot: failed to write '%s'", path.c_str());
        return false;
    }
    SDL_Log("Screenshot: saved %ux%u -> %s", w, h, path.c_str());
    return true;
}

} // namespace orange::core
