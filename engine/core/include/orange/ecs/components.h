#pragma once

#include "orange/core/font.h"
#include "orange/core/math.h"
#include "orange/render/types.h"

// ECS components. These are plain structs; behavior lives in systems.
// We use EnTT (entt::registry) as the entity/component store -- there is no
// scene graph: world state is a flat set of entities + components.

namespace orange::ecs {

struct Transform {
    math::Vec3 position{0, 0, 0};
    math::Quat orientation{};       // unit quaternion (no gimbal lock)
    math::Vec3 scale{1, 1, 1};

    math::Mat4 matrix() const {
        return math::translate(position) * math::toMat4(orientation) *
               math::scale(scale);
    }
};

// Makes an entity drawable with a GPU mesh owned by the renderer. This is the
// "actual rendering" component -- attached to an entity (composition), not a
// base class others inherit from. Extend it with material/tint later.
struct Renderable {
    render::MeshHandle mesh    = render::kInvalidMesh;
    bool               visible = true;
};

enum class ProjectionMode { Perspective = 0, Orthographic = 1 };

// A camera. Position/orientation come from the entity's Transform.
struct Camera {
    ProjectionMode mode = ProjectionMode::Perspective;
    float fovYDegrees = 60.0f;   // used in Perspective
    float orthoSize   = 4.0f;    // half-height of the view volume, used in Orthographic
    float zNear       = 0.1f;
    float zFar        = 100.0f;
    bool  primary     = false;
};

// Trackball (arcball) camera controller. The cameraManipulatorSystem reads
// mouse input and writes the owning entity's Transform so it orbits `target`.
// Orientation is a quaternion accumulated incrementally, so there is no gimbal
// lock and the view can tumble freely in any direction:
//   left-drag         -> orbit (rotate about the camera's own axes)
//   wheel             -> zoom (distance)
//   middle/right-drag -> pan (move target)
struct CameraManipulator {
    math::Vec3 target{0, 0, 0};        // pivot the camera orbits around
    float      distance = 6.0f;        // camera distance from target
    math::Quat orientation{};          // camera orientation (identity => +Z)

    float rotateSpeed = 0.006f;
    float zoomSpeed   = 0.6f;
    float panSpeed    = 0.0016f;

    float minDistance = 1.0f;
    float maxDistance = 60.0f;

    // Smooth snap animation (driven by the axis gizmo). While animating, manual
    // orbit is suspended and `orientation` is slerped from animFrom to animTo.
    bool       animating    = false;
    math::Quat animFrom{};
    math::Quat animTo{};
    float      animTime     = 0.0f;
    float      animDuration = 0.45f;
};

// A ViewCube-style axis gizmo drawn in a screen corner. It shows the world
// orientation as seen by the primary camera; clicking a face/edge/corner snaps
// the camera to that view. There is normally a single one in the world.
// What part of the gizmo a hover/click refers to.
enum class GizmoPart { None = 0, Cube = 1, Ring = 2 };

// Face/edge boundary on a cube face: |coord| > kGizmoEdge => thin edge/corner
// border, else => the large face center. Shared by picking, the highlight, and
// the drawn grid lines so all three line up.
inline constexpr float kGizmoEdge = 0.78f;

struct AxisGizmo {
    render::MeshHandle mesh          = render::kInvalidMesh;  // colored cube
    render::MeshHandle labelMesh     = render::kInvalidMesh;  // X/Y/Z text quads
    render::MeshHandle ringMesh      = render::kInvalidMesh;  // 4-sector ring behind cube
    render::MeshHandle highlightMesh = render::kInvalidMesh;  // dynamic patch
    render::BufferHandle highlightVbo = render::kInvalidBuffer;  // dynamic VB
    render::TextureHandle labelTexture = render::kInvalidTexture;  // label atlas

    int sizePx = 150;  // square size in pixels
    int margin = 14;   // distance from the top-right corner

    // Runtime state, updated each frame by axisGizmoInputSystem.
    GizmoPart  hoverPart   = GizmoPart::None;
    math::Vec3 hoverDir{0, 0, 0};  // cube region (face/edge/corner) under cursor
    int        hoverSector = -1;   // ring sector 0..3 (right/top/left/bottom)

    float      flash       = 0.0f; // click feedback timer (seconds remaining)
    GizmoPart  flashPart   = GizmoPart::None;
    math::Vec3 flashDir{0, 0, 0};
    int        flashSector = -1;
};

// A draggable on-screen FPS graph widget (bar history + numeric readout).
struct FpsWidget {
    int  x = 16, y = 16, w = 240, h = 96;  // pixel rect (top-left origin)
    bool visible = true;

    static constexpr int kSamples = 64;
    float history[kSamples] = {};   // recent FPS values (ring buffer)
    int   head      = 0;
    float smoothFps = 60.0f;
    float maxScale  = 120.0f;       // FPS mapped to the top of the graph

    // Drag state.
    bool  dragging = false;
    float dragOffX = 0.0f, dragOffY = 0.0f;

    // GPU resources (dynamic vertex buffer rewritten each frame).
    render::MeshHandle   mesh = render::kInvalidMesh;
    render::BufferHandle vbo  = render::kInvalidBuffer;

    // Shared proportional font for the readout; atlas = font->texture (its white
    // texel backs the opaque panel/bars).
    const core::Font*     font  = nullptr;
    render::TextureHandle atlas = render::kInvalidTexture;
};

// A small control panel (below the gizmo) for the primary camera: a projection
// mode toggle and FOV (or, in ortho, size) +/- buttons with a value readout.
struct CameraControls {
    int w = 184, h = 76;   // panel size (px); positioned top-right under the gizmo
    int x = 0, y = 0;      // computed each frame

    const core::Font*    font  = nullptr;  // shared text font
    render::TextureHandle atlas = render::kInvalidTexture;
    render::MeshHandle    mesh = render::kInvalidMesh;
    render::BufferHandle  vbo  = render::kInvalidBuffer;
};

// Demo behavior: continuous rotation, consumed by SpinSystem.
struct Spin {
    math::Vec3 axisRadiansPerSec{0, 1.0f, 0};
};

} // namespace orange::ecs
