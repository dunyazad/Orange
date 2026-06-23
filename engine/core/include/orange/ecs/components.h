#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "orange/core/bvh.h"
#include "orange/core/draw_mode.h"
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

    // Per-mesh drawing mode (Helium's set); Tab cycles it for the selection, so
    // each mesh keeps its own. Selected meshes also get a silhouette outline.
    core::DrawMode drawMode = core::DrawMode::Solid;

    // A non-indexed point cloud (drawn as points). Selection shows a bounding-box
    // wireframe instead of the stencil silhouette (which needs a solid surface).
    bool pointCloud = false;
};

// Optional CPU-side triangle soup (local space) for accurate ray picking. When
// present, the pickingSystem ray-tests the actual triangles instead of just the
// Renderable's AABB -- so clicking through a concave/empty region of a mesh (e.g.
// the gaps of a loaded model) correctly misses it and hits whatever is behind.
struct PickGeometry {
    std::vector<Eigen::Vector3f> positions;  // local-space vertex positions
    std::vector<uint32_t>        indices;    // triangle list (3 per face)
};

// Background build of a pick BVH for a huge mesh (millions of triangles), so the
// click that triggers it doesn't freeze the main thread. Held by shared_ptr so the
// worker can be detached. Small meshes skip this and build inline (instant).
struct PickBvhJob {
    std::atomic<bool> done{false};
    geometry::BVH     bvh;
};

// Cached BVH over a PickGeometry's triangles, built on first ray pick so repeated
// picks of a large mesh are O(log n) instead of testing every triangle. Small
// meshes build inline; a huge mesh builds on a worker (`job`) while picks fall back
// to the coarse AABB until it is ready. References the entity's PickGeometry buffers.
struct PickBVH {
    geometry::BVH                bvh;
    bool                         built = false;
    std::shared_ptr<PickBvhJob>  job;    // in-flight background build (null = none)
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

    // Smooth recenter animation (Ctrl+left-click on geometry): the orbit pivot
    // `target` is eased from targetFrom to targetTo, so the camera position
    // (target + offset) glides along with it. Independent of the orientation snap
    // above; cancelled by a manual pan/zoom. `distance` is eased from distFrom to
    // distTo over the same timeline (used by the R reset; held constant by the
    // Ctrl+click recenter). Reuses animDuration for timing.
    bool       targetAnimating = false;
    Eigen::Vector3f targetFrom = Eigen::Vector3f::Zero();
    Eigen::Vector3f targetTo   = Eigen::Vector3f::Zero();
    float      targetAnimTime  = 0.0f;
    float      distFrom = 6.0f, distTo = 6.0f;  // distance eased alongside target

    // Home pose restored by the R key (set when the camera is created). The reset
    // animates orientation + target + distance back to these.
    Eigen::Vector3f homeTarget   = Eigen::Vector3f::Zero();
    float           homeDistance = 6.0f;
    Eigen::Quaternionf homeOrientation = Eigen::Quaternionf::Identity();
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
    // Top-left, just right of the selection toolbar (far-left ~54px strip) and just
    // below the menu bar; the TreeView stacks directly under it (shared left x).
    // (Tuned for a maximized ~4K window; both widgets are draggable + persisted.)
    float relX = 0.0161f, relY = 0.0263f;

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

// A draggable tree-view widget: a scene outliner that lists the world's drawable
// entities grouped into collapsible categories (Meshes / Point Clouds). Built the
// same way as FpsWidget -- a dynamic vertex buffer rewritten each frame into a
// font-atlas overlay quad mesh. treeViewInputSystem hit-tests the title bar (drag),
// the group arrows (expand/collapse), the scroll wheel, and the rows; clicking a
// row selects that entity (syncing Renderable::selected, so the viewport silhouette
// updates too). The rows themselves are rebuilt from the world each frame -- the
// component only holds UI state + GPU resources.
struct TreeView {
    static constexpr int kGroups = 2;  // 0 = Meshes, 1 = Point Clouds

    bool visible = true;
    int  w = 252, h = 340;             // panel size (px)
    int  x = 0, y = 0;                 // computed each frame from relX/relY

    // Position stored as a viewport fraction (top-left), so it tracks resizes.
    // Stacked directly under the FpsWidget in the top-left column (shared left x,
    // ~6px below the FPS panel), clear of the selection toolbar. Tuned for a
    // maximized ~4K window; draggable + persisted thereafter.
    float relX = 0.0161f, relY = 0.0935f;

    bool  expanded[kGroups] = {true, true};  // per-group collapse state
    float scroll = 0.0f;               // vertical scroll offset (px, >= 0)
    int   hover  = -1;                  // hovered row index (-1 none)

    // Drag state (panel moved by its title bar).
    bool  dragging = false;
    float dragOffX = 0.0f, dragOffY = 0.0f;

    // GPU resources (dynamic vertex buffer rewritten each frame).
    render::MeshHandle   mesh = render::kInvalidMesh;
    render::BufferHandle vbo  = render::kInvalidBuffer;

    // Shared proportional font; atlas = font->texture (white texel backs the
    // opaque panel/row fills).
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

// Cross-section ("단면") panel: a slider that sweeps a world-aligned clipping
// plane through the scene so you can see a cut interior. crossSectionInputSystem
// handles the enable toggle / axis cycle / flip / slider drag; renderSystem turns
// (enabled, axis, flip, pos) into a world plane and calls IRenderer::setCrossSection
// before the scene submits. Positioned top-right, under the CameraControls panel.
struct CrossSection {
    bool  enabled = false;
    int   axis    = 0;        // clip axis in render-world space (0=X, 1=Y, 2=Z)
    bool  flip    = false;    // keep the +side instead of the -side of the plane
    float pos     = 0.0f;     // plane position along the axis (world units)
    float minPos  = -3.0f;    // slider range (content fits ~3 world units, centered)
    float maxPos  =  3.0f;

    int  w = 184, h = 96;     // panel size (px)
    int  x = 0, y = 0;        // computed each frame (top-right, under the controls)
    bool dragging = false;    // slider handle is being dragged

    const core::Font*     font  = nullptr;  // shared text font
    render::TextureHandle atlas = render::kInvalidTexture;
    render::MeshHandle     mesh = render::kInvalidMesh;
    render::BufferHandle   vbo  = render::kInvalidBuffer;
};

// --- Selection modes (driven by the left selection toolbar) ----------------
// What a click selects, how the region is gathered, which entity types are
// eligible, and how the result combines with the existing selection.
enum class SelTarget   { Object = 0, Vertex, Edge, Face };  // selection granularity
enum class SelAction   { Single = 0, Box, Lasso, Paint };   // how the region is drawn
enum class SelFilter   { All = 0, Mesh, Point };            // eligible entity types
enum class SelModifier { Replace = 0, Add, Subtract };      // combine with current

// Live selection state, stored in the registry ctx (read by pickingSystem and the
// toolbar). Holds the four mode enums plus the transient drag geometry for the
// Box/Lasso/Paint actions (window-space pixels).
struct SelectionMode {
    SelTarget   target   = SelTarget::Object;
    SelAction   action   = SelAction::Single;
    SelFilter   filter   = SelFilter::All;
    SelModifier modifier = SelModifier::Replace;

    bool  dragging = false;                  // a Box/Lasso/Paint drag is in progress
    float dragX0 = 0, dragY0 = 0;            // box rubber-band start (px)
    float dragX1 = 0, dragY1 = 0;            // box rubber-band current (px)
    std::vector<float> lassoX, lassoY;       // freehand lasso polygon (px)
};

// Per-entity selected sub-elements (when SelTarget is Vertex/Edge/Face). Indices
// refer to the entity's PickGeometry. Empty unless element selection is used.
struct ElementSelection {
    struct Edge { uint32_t a = 0, b = 0; };
    std::vector<uint32_t> vertices;  // selected vertex indices
    std::vector<uint32_t> faces;     // selected triangle indices (PickGeometry tri i/3)
    std::vector<Edge>     edges;     // selected edges (vertex-index pairs)
};

// Per-entity cached GPU mesh of the ElementSelection highlight (vertex cubes /
// edge tubes / face triangles), baked in the entity's local space and drawn each
// frame with model = Mworld * Transform. Rebuilt only when the selection content
// changes (tracked by `sig`, a hash of the index sets) -- the old path rebuilt a
// billboard quad per selected vertex into the per-frame debug-draw mesh and
// re-uploaded it every frame, which tanked the FPS on large (e.g. box) selections.
struct ElementSelCache {
    uint64_t             sig  = 0;   // hash of the selection the mesh was built from
    bool                 built = false;
    render::MeshHandle   mesh = render::kInvalidMesh;
    render::BufferHandle vbo  = render::kInvalidBuffer;
    uint32_t             vertexCount = 0;
};

// Left vertical selection toolbar. One labelled button per mode value, grouped
// (target / action / filter / modifier). selectionToolbarInputSystem hit-tests
// the buttons and writes the chosen value into the ctx SelectionMode; renderSystem
// draws it as an overlay. Normally a single one in the world.
struct ToolbarButton {
    std::string label;   // short caption (e.g. "Obj", "Box", "+")
    int group = 0;       // 0 target, 1 action, 2 filter, 3 modifier
    int value = 0;       // enum value within the group
};
struct SelectionToolbar {
    bool visible = true;
    std::vector<ToolbarButton> buttons;
    int  hover = -1;     // hovered button index (-1 none)

    render::MeshHandle   mesh = render::kInvalidMesh;
    render::BufferHandle vbo  = render::kInvalidBuffer;
    const core::Font*     font  = nullptr;
    render::TextureHandle atlas = render::kInvalidTexture;
};

// View toggles stored in the registry context (registry::ctx), not on an entity.
// `grid` is flipped by the Space key and read by renderSystem to show/hide the
// infinite ground grid. Absent context => defaults (grid on).
struct GridState {
    bool visible = true;
};

// Current point-sprite size in pixels (mirrors Application::pointSize_, set each
// frame). renderSystem reads it so vertex-selection markers match the on-screen
// size of the point sprites. Absent => default.
struct PointSizeState {
    float size = 6.0f;
};

// Spatial-structure overlay selector (ctx), set from the Spatial menu. renderSystem
// builds the chosen structure for each *selected* entity's PickGeometry and draws
// its node/cell boxes as a wireframe overlay. 0 = off, 1 = BVH, 2 = Octree, 3 = KD-tree.
struct SpatialViz {
    int kind = 0;
};

// Background build of a spatial-structure wireframe: a worker thread fills `verts`
// (the structure + its node boxes/spheres, in the entity's local space) from a
// self-contained copy of the points, then sets `done`; the main thread uploads it
// to the GPU. Heavy on a million-point cloud, so it must not run on the render
// thread. Held by shared_ptr so the worker can be detached and outlive the cache.
struct SpatialVizJob {
    std::atomic<bool>           done{false};
    std::atomic<float>          progress{0.0f};  // [0,1]
    std::vector<render::Vertex> verts;
    int                         kind = 0;
    int                         boxCount = 0, drawn = 0, total = 0;
};

// Per-entity cached GPU wireframe of the current SpatialViz kind, plus the
// in-flight background build that produces it. The wireframe is built on a worker
// thread (the structure build + box generation is O(N log N) over every point);
// the main thread only uploads the finished vertices and draws the cached mesh
// each frame with model = Mworld * Transform.
struct SpatialVizCache {
    int kind        = -1;   // kind the current GPU mesh shows (-1 = none)
    int pendingKind = -1;   // kind a background build is currently producing (-1 = none)
    std::shared_ptr<SpatialVizJob> job;  // in-flight build (null = idle)

    render::MeshHandle   mesh = render::kInvalidMesh;
    render::BufferHandle vbo  = render::kInvalidBuffer;
    uint32_t             vertexCount = 0;
    int boxCount = 0, inliers = 0, total = 0;  // diagnostics (shown in the menu bar)
};

// Height (px) of the top menu bar. The axis gizmo and camera-controls panel are
// pushed down by this so they never overlap the bar. Shared by systems.cpp.
inline constexpr int kMenuBarHeight = 46;

// Action ids raised by a menu item click. menuBarInputSystem writes the chosen
// item's `action` into MenuBar::triggered; Application::applyMenuAction consumes
// it (mirroring the keyboard shortcuts), except MenuAction::OpenFile which sets
// requestOpenFile for the app's native file dialog. Keep in sync with the menus
// built by defaultAppMenus() and the dispatch in application.cpp.
enum class MenuAction : int {
    None = 0,
    OpenFile, Screenshot, Quit,
    ToggleGrid, ResetCamera, ToggleUpAxis, ToggleProjection,
    ToggleLighting, ToggleVsync, ToggleCrossSection,
    ColorOriginal, ColorHeight, ColorPosition, ColorGray,
    DrawNone, DrawSolid, DrawWireframe, DrawWireSolid, DrawPoint,
    SelectAll, DeleteSelected, ClearSelection, UnhideAll,
    PointSizeUp, PointSizeDown,
    // Geometry-processing operators. Contiguous so the menu/dispatch can map a
    // mode index i to MenuAction(Mode0 + i). Reserve a generous range so adding
    // modes in modes.cpp needs no new enum here.
    ModeOff,  // deactivate the processing operator (default state)
    Mode0, Mode1, Mode2, Mode3, Mode4, Mode5, Mode6, Mode7,
    Mode8, Mode9, Mode10, Mode11, Mode12, Mode13, Mode14, Mode15,
    // Parametric primitive spawners (Create menu) -> a new Renderable entity.
    CreatePlane, CreateBox, CreateSphere, CreateCylinder, CreateCone,
    CreateTorus, CreateDisk, CreateCapsule, CreateArrow,
    SpatialNone, SpatialBVH, SpatialOctree, SpatialKDTree,
    SpatialGrid, SpatialLoose, SpatialBSP, SpatialRTree, SpatialBall,
};

// One row in a dropdown. kind: Action = clickable command; Check = command that
// also shows a tick when `checked`; Separator = a thin divider (no label/click).
struct MenuItem {
    enum Kind { Action, Check, Separator };
    std::string label;
    std::string shortcut;                 // right-aligned hint, e.g. "Tab" (optional)
    MenuAction  action  = MenuAction::None;
    Kind        kind    = Action;
    bool        checked = false;          // tick state for Check items (synced by app)
};

// One top-level menu (a title in the bar + its dropdown items).
struct Menu {
    std::string           title;
    std::vector<MenuItem> items;
};

// Top-of-window menu bar with any number of menus. menuBarInputSystem handles
// open/close, hover and clicks; renderSystem draws it as an overlay. A clicked
// item raises `triggered` for Application to dispatch. There is normally a single
// MenuBar in the world.
struct MenuBar {
    int  height  = kMenuBarHeight;  // bar height in px
    bool visible = true;

    std::vector<Menu> menus;        // populated by defaultAppMenus()
    int  openMenu  = -1;            // index of the expanded menu (-1 = none)
    int  hoverItem = -1;            // hovered dropdown item index (-1 = none)

    // Action raised by the most recent item click; Application clears it after
    // dispatch. MenuAction::None = nothing pending this frame.
    MenuAction triggered = MenuAction::None;

    // Raised when MenuAction::OpenFile is dispatched; the app clears it once it
    // has shown the native dialog (one-shot edge flag).
    bool requestOpenFile = false;

    // Right-aligned status text drawn in the bar (e.g. "Loading 42%"). The app
    // sets it while a background load runs and clears it when finished. Empty =
    // nothing drawn. ASCII only (the UI font bakes 32..126).
    std::string statusText;

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
