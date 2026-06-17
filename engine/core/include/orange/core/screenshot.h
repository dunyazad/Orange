#pragma once

#include <string>

#include "orange/render/renderer.h"

namespace orange::core {

// Reads the renderer's last presented frame and writes it to `path` as a PNG.
// Returns false if the backend can't read pixels or the file can't be written.
bool saveScreenshot(render::IRenderer& renderer, const std::string& path);

} // namespace orange::core
