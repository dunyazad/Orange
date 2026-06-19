#pragma once

#include <cstdint>

namespace orange::core {

// Scene drawing mode, cycled by the Tab key. Mirrors Helium's Renderable
// DrawingMode set. Stored in the registry's context (entt ctx) so the input
// handler (Application) sets it and the renderSystem reads it; the values are
// passed straight to IRenderer::setDrawMode(), which each backend renders like
// Helium's DrawImplementation.
enum class DrawMode : uint32_t {
    None               = 0,  // draw nothing
    Solid              = 1,  // filled triangles
    WireFrame          = 2,  // edges only
    WireFrameOverSolid = 3,  // shaded fill with edges drawn on top
    Point              = 4,  // vertices as points
    Count              = 5,
};

inline const char* to_string(DrawMode m) {
    switch (m) {
        case DrawMode::None:               return "none";
        case DrawMode::Solid:              return "solid";
        case DrawMode::WireFrame:          return "wireframe";
        case DrawMode::WireFrameOverSolid: return "wireframe-over-solid";
        case DrawMode::Point:              return "point";
        default:                           return "?";
    }
}

struct DrawModeState {
    DrawMode mode = DrawMode::Solid;
};

} // namespace orange::core
