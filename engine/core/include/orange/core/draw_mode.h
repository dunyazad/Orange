#pragma once

#include <cstdint>

namespace orange::core {

// Per-mesh drawing mode, mirroring Helium's Renderable DrawingMode set. Stored
// on each Renderable; the Tab key cycles it for the current selection and the
// renderSystem passes it to IRenderer::setDrawMode(), which each backend renders
// like Helium's DrawImplementation.
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

} // namespace orange::core
