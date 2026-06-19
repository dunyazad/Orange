#pragma once

#include "orange/core/font.h"
#include "orange/core/math.h"
#include "orange/render/types.h"

// ECS components. These are plain structs; behavior lives in systems.
// We use EnTT (entt::registry) as the entity/component store -- there is no
// scene graph: world state is a flat set of entities + components.

namespace orange::ecs {

struct Transform {
    Eigen::Vector3f position   = Eigen::Vector3f::Zero();
    Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();  // unit quaternion (no gimbal lock)
    Eigen::Vector3f scale      = Eigen::Vector3f::Ones();

    Eigen::Matrix4f matrix() const {
        return math::translate(position) * math::toMat4(orientation) *
               math::scale(scale);
    }
};

// Makes an entity drawable with a GPU mesh owned by the renderer. This is the
// "actual rendering" component -- attached to an entity (composition), not a
// base class others inherit from. Extend it with material/tint later.
//
// boundsMin/boundsMax are the mesh's local-space axis-aligned bounds, used by
// the pickingSystem for left-click ray selection. They default to a unit cube
// ([-0.5, 0.5]^3); set them to match the actual mesh for accurate picking.
struct Renderable {
    render::MeshHandle mesh    = render::kInvalidMesh;
    bool               visible = true;

    Eigen::Vector3f boundsMin = Eigen::Vector3f(-0.5f, -0.5f, -0.5f);
    Eigen::Vector3f boundsMax = Eigen::Vector3f( 0.5f,  0.5f,  0.5f);
    bool       selected = false;  // set by pickingSystem on left-click
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
//   right-drag  -> orbit (rotate about the camera's own axes)
//   wheel       -> zoom (distance)
//   middle-drag -> pan (move target)
// (Left-click is reserved for picking; see pickingSystem.)
struct CameraManipulator {
    Eigen::Vector3f target   = Eigen::Vector3f::Zero();  // pivot the camera orbits around
    float      distance = 6.0f;        // camera distance from target
    Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();  // camera orientation (identity => +Z)

    float rotateSpeed = 0.006f;
    float zoomSpeed   = 0.6f;
    float panSpeed    = 0.0016f;

    float minDistance = 1.0f;
    float maxDistance = 60.0f;

    // Smooth snap animation (driven by the axis gizmo). While animating, manual
    // orbit is suspended and `orientation` is slerped from animFrom to animTo.
    bool       animating    = false;
    Eigen::Quaternionf animFrom = Eigen::Quaternionf::Identity();
    Eigen::Quaternionf animTo   = Eigen::Quaternionf::Identity();
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

    // World up-axis toggle, shown as a small button in a corner of the gizmo.
    // false => Y up (grid on y=0), true => Z up (grid on z=0). axisGizmoInputSystem
    // flips it on click and smoothly re-orients the camera; renderSystem passes the
    // axis to IRenderer::drawGrid and draws the button.
    bool zUp = false;
    bool upBtnHover = false;  // cursor over the toggle button (set each frame)

    // GPU resources for the toggle button (dynamic vertex buffer + UI font).
    render::MeshHandle    upBtnMesh = render::kInvalidMesh;
    render::BufferHandle  upBtnVbo  = render::kInvalidBuffer;
    const core::Font*     font      = nullptr;
    render::TextureHandle uiAtlas   = render::kInvalidTexture;

    // Runtime state, updated each frame by axisGizmoInputSystem.
    GizmoPart  hoverPart   = GizmoPart::None;
    Eigen::Vector3f hoverDir = Eigen::Vector3f::Zero();  // cube region (face/edge/corner) under cursor
    int        hoverSector = -1;   // ring sector 0..3 (right/top/left/bottom)

    float      flash       = 0.0f; // click feedback timer (seconds remaining)
    GizmoPart  flashPart   = GizmoPart::None;
    Eigen::Vector3f flashDir = Eigen::Vector3f::Zero();
    int        flashSector = -1;
};

// A draggable on-screen FPS graph widget (bar history + numeric readout).
struct FpsWidget {
    int  x = 16, y = 16, w = 260, h = 132;  // pixel rect (computed each frame)
    bool visible = true;

    // Position is stored as a fraction of the viewport (top-left corner) so the
    // panel keeps its relative spot when the window is resized. fpsWidgetInputSystem
    // derives x/y from these each frame; dragging writes them back.
    float relX = 0.0125f, relY = 0.052f;  // below the menu bar

    static constexpr int kSamples = 64;
    float history[kSamples] = {};   // recent FPS values (ring buffer)
    int   head      = 0;
    float smoothFps = 60.0f;
    float maxScale  = 120.0f;       // FPS mapped to the top of the graph

    // Sampling cadence: rather than push a sample / refresh the readout every
    // frame (jittery at high FPS), accumulate and update once per interval.
    float updateInterval = 0.5f;    // seconds between graph samples + readout
    float accumTime      = 0.0f;
    int   accumFrames    = 0;
    bool  primed         = false;   // history pre-filled on the first update

    // Drag state.
    bool  dragging = false;
    float dragOffX = 0.0f, dragOffY = 0.0f;

    // VSYNC toggle checkbox (top-right of the panel). vsyncDirty is set on click
    // and consumed by renderSystem, which applies it to the renderer.
    bool vsync      = true;
    bool vsyncDirty = false;

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

    // +/- button hold state: time the active button has been held (for key-repeat
    // in Shift-snap mode). Reset on each fresh press.
    float holdTime = 0.0f;

    const core::Font*    font  = nullptr;  // shared text font
    render::TextureHandle atlas = render::kInvalidTexture;
    render::MeshHandle    mesh = render::kInvalidMesh;
    render::BufferHandle  vbo  = render::kInvalidBuffer;
};

// Height (px) of the top menu bar. The axis gizmo and camera-controls panel are
// pushed down by this so they never overlap the bar. Shared by systems.cpp.
inline constexpr int kMenuBarHeight = 28;

// A classic top-of-window menu bar with a single "File" menu whose only item is
// "Open...". menuBarInputSystem handles open/close + clicks and renderSystem
// draws it as an overlay. When "Open..." is chosen, requestOpenFile is raised
// for the app to consume (e.g. show a native file dialog and load a mesh).
// There is normally a single one in the world.
struct MenuBar {
    int  height  = kMenuBarHeight;  // bar height in px
    bool visible = true;

    bool fileOpen  = false;  // is the File dropdown expanded?
    int  hoverItem = -1;     // hovered dropdown item index (-1 = none)

    // Raised by menuBarInputSystem when "Open..." is clicked; the app clears it
    // once it has acted on it (one-shot edge flag, like FpsWidget::vsyncDirty).
    bool requestOpenFile = false;

    // GPU resources (dynamic vertex buffer rewritten each frame).
    render::MeshHandle   mesh = render::kInvalidMesh;
    render::BufferHandle vbo  = render::kInvalidBuffer;

    // Shared proportional font; atlas = font->texture (its white texel backs the
    // opaque bar + dropdown fills).
    const core::Font*     font  = nullptr;
    render::TextureHandle atlas = render::kInvalidTexture;
};

// Demo behavior: continuous rotation, consumed by SpinSystem.
struct Spin {
    Eigen::Vector3f axisRadiansPerSec = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
};

} // namespace orange::ecs
