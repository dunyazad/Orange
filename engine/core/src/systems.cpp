#include "orange/ecs/systems.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <execution>
#include <unordered_set>
#include <vector>

#include "orange/core/ball_tree.h"
#include "orange/core/bsp.h"
#include "orange/core/color.h"
#include "orange/core/debug_draw.h"
#include "orange/core/draw_mode.h"
#include "orange/core/geometry.h"
#include "orange/core/kdtree.h"
#include "orange/core/loose_octree.h"
#include "orange/core/octree.h"
#include "orange/core/rtree.h"
#include "orange/core/uniform_grid.h"
#include "orange/core/math.h"
#include "orange/core/modes.h"
#include "orange/ecs/components.h"

namespace orange::ecs {

namespace {
// Persistent GPU resources for immediate-mode debug drawing. Raw handles (like
// the overlay widgets) so the registry's teardown never outlives the renderer.
struct DebugMeshState {
    render::BufferHandle vbo  = render::kInvalidBuffer;
    render::MeshHandle   mesh = render::kInvalidMesh;
    size_t               capacity = 0;  // vertices (multiple of 3)
};

// Uploads everything accumulated via debug::DebugDraw this frame as one dynamic
// non-indexed mesh, then clears the accumulator. Grows the buffer on demand and
// pads the unused tail with degenerate (zero-area) triangles.
void drawDebugGeometry(entt::registry& world, render::IRenderer& renderer,
                       const Eigen::Matrix4f& worldMatrix) {
    auto& dd = debug::DebugDraw::instance();
    const auto& verts = dd.vertices();
    if (verts.empty()) return;

    if (!world.ctx().contains<DebugMeshState>()) world.ctx().emplace<DebugMeshState>();
    auto& st = world.ctx().get<DebugMeshState>();

    if (verts.size() > st.capacity) {
        if (st.mesh != render::kInvalidMesh) renderer.destroyMesh(st.mesh);
        if (st.vbo != render::kInvalidBuffer) renderer.destroyBuffer(st.vbo);
        size_t cap = 3;
        while (cap < verts.size()) cap *= 2;
        cap = ((cap + 2) / 3) * 3;  // keep triangle-aligned

        std::vector<render::Vertex> init(cap);  // zero => degenerate
        render::BufferDesc bd;
        bd.type  = render::BufferType::Vertex;
        bd.usage = render::BufferUsage::Dynamic;
        bd.data  = init.data();
        bd.size  = cap * sizeof(render::Vertex);
        st.vbo = renderer.createBuffer(bd);
        st.capacity = cap;

        render::MeshDesc md;
        md.vertexBuffer = st.vbo;
        md.indexBuffer  = render::kInvalidBuffer;
        md.layout       = render::Vertex::layout();
        md.vertexCount  = static_cast<uint32_t>(cap);
        st.mesh = renderer.createMesh(md);
    }

    std::vector<render::Vertex> buf(st.capacity, render::Vertex{});
    std::copy(verts.begin(), verts.end(), buf.begin());
    renderer.updateBuffer(st.vbo, buf.data(), buf.size() * sizeof(render::Vertex));

    render::DrawItem item;
    item.mesh = st.mesh;
    std::memcpy(item.model, worldMatrix.data(), sizeof(item.model));
    renderer.submit(item);

    dd.clear();
}

// Cached debug geometry for the active processing mode (recomputed only when the
// mode selection changes), stored in the registry ctx.
struct ModeCache {
    uint64_t generation = 0;
    std::vector<render::Vertex> verts;
};
} // namespace

// Runs the active processing mode on the input point cloud (both in the registry
// ctx), caching its debug geometry and re-emitting it each frame so the heavy
// computation only happens when the selected mode changes.
void processingModeSystem(entt::registry& world) {
    if (!world.ctx().contains<modes::ModeInput>()) return;
    const auto& input = world.ctx().get<modes::ModeInput>();

    modes::ModeState state;
    if (world.ctx().contains<modes::ModeState>()) state = world.ctx().get<modes::ModeState>();

    if (!world.ctx().contains<ModeCache>()) world.ctx().emplace<ModeCache>();
    auto& cache = world.ctx().get<ModeCache>();

    if (cache.generation != state.generation) {
        debug::DebugDraw tmp;
        modes::runMode(state.index, input, tmp);
        cache.verts = tmp.vertices();
        cache.generation = state.generation;
    }

    debug::DebugDraw::instance().addRaw(cache.verts);
}

namespace {
float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// --- Axis-gizmo overlay constants (shared by render + picking) -------------
constexpr float kGizmoEyeZ      = 3.0f;  // overlay camera distance along +Z
// The unit cube projects to at most sqrt(3) ~= 1.732 from center, so the ring
// must start beyond that to never be covered by the cube.
constexpr float kRingInner      = 1.9f;  // clickable ring behind the cube
constexpr float kRingOuter      = 2.3f;
// kGizmoEdge (face/edge boundary) lives in components.h so the appOrange grid uses
// the same value; it is in scope here via namespace orange::ecs.
constexpr float kGizmoOrthoHalf = 2.45f; // ortho half-extent (fits the ring + margin)
constexpr float kHalfPi         = 1.57079633f;
constexpr int   kHlQuads        = 12;           // highlight mesh capacity (corner = 3 faces x 4 border strips)
constexpr int   kHlVerts        = kHlQuads * 4; // = 48

// World up-axis basis. Identity for Y up; for Z up, rotateX(-90deg) so a model's
// logical +Z maps to render +Y (vertical). Applied to ALL scene content + the
// gizmo cube so toggling genuinely re-expresses the world's coordinate frame
// (the camera and the horizontal ground stay put -- it is not a view spin).
Eigen::Quaternionf worldUpQuat(bool zUp) {
    return zUp ? math::quatAxisAngle(Eigen::Vector3f(1, 0, 0), -kHalfPi)
               : Eigen::Quaternionf::Identity();
}
Eigen::Matrix4f worldUpMatrix(bool zUp) { return math::toMat4(worldUpQuat(zUp)); }

// The single gizmo's up-axis flag (true => Z up), or false if there is none.
bool worldZUp(entt::registry& world) {
    auto v = world.view<AxisGizmo>();
    for (auto e : v) return v.get<AxisGizmo>(e).zUp;
    return false;
}

struct GizmoRect { int x, y, w, h; };

GizmoRect gizmoRect(const AxisGizmo& g, uint32_t W, uint32_t H) {
    (void)H;
    GizmoRect r;
    r.w = g.sizePx;
    r.h = g.sizePx;
    r.x = static_cast<int>(W) - g.sizePx - g.margin;  // top-right
    r.y = g.margin + kMenuBarHeight;                  // below the menu bar
    if (r.x < 0) r.x = 0;
    return r;
}

// Ray vs. axis-aligned box [-1,1]^3. Returns the entry hit point.
bool intersectUnitBox(Eigen::Vector3f o, Eigen::Vector3f dir, Eigen::Vector3f& hit) {
    float O[3] = {o.x(), o.y(), o.z()};
    float D[3] = {dir.x(), dir.y(), dir.z()};
    float tmin = -1e30f, tmax = 1e30f;
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(D[i]) < 1e-8f) {
            if (O[i] < -1.0f || O[i] > 1.0f) return false;
        } else {
            float t1 = (-1.0f - O[i]) / D[i];
            float t2 = (1.0f - O[i]) / D[i];
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
        }
    }
    if (tmax < tmin || tmax < 0.0f) return false;
    float t = tmin >= 0.0f ? tmin : tmax;
    hit = o + dir * t;
    return true;
}

// Ray vs. axis-aligned box [bmin, bmax]. On hit, returns true and writes the
// entry distance along `dir` (or the exit distance if the origin is inside).
bool intersectAABB(Eigen::Vector3f o, Eigen::Vector3f dir, Eigen::Vector3f bmin, Eigen::Vector3f bmax,
                   float& tHit) {
    float O[3] = {o.x(), o.y(), o.z()};
    float D[3] = {dir.x(), dir.y(), dir.z()};
    float lo[3] = {bmin.x(), bmin.y(), bmin.z()};
    float hi[3] = {bmax.x(), bmax.y(), bmax.z()};
    float tmin = -1e30f, tmax = 1e30f;
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(D[i]) < 1e-8f) {
            if (O[i] < lo[i] || O[i] > hi[i]) return false;
        } else {
            float t1 = (lo[i] - O[i]) / D[i];
            float t2 = (hi[i] - O[i]) / D[i];
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
        }
    }
    if (tmax < tmin || tmax < 0.0f) return false;
    tHit = tmin >= 0.0f ? tmin : tmax;
    return true;
}

// Classify a hit point on the cube into a direction triple in {-1,0,1}^3:
// one nonzero => face, two => edge, three => corner. The triple is the world
// view direction to snap the camera to.
Eigen::Vector3f classifyDir(Eigen::Vector3f hit) {
    const float k = kGizmoEdge;
    auto zone = [&](float c) { return c > k ? 1.0f : (c < -k ? -1.0f : 0.0f); };
    Eigen::Vector3f d(zone(hit.x()), zone(hit.y()), zone(hit.z()));
    if (d.x() == 0 && d.y() == 0 && d.z() == 0) {  // dead-center: snap to dominant axis
        float ax = std::fabs(hit.x()), ay = std::fabs(hit.y()), az = std::fabs(hit.z());
        if (ax >= ay && ax >= az)      d.x() = hit.x() > 0 ? 1.0f : -1.0f;
        else if (ay >= az)             d.y() = hit.y() > 0 ? 1.0f : -1.0f;
        else                           d.z() = hit.z() > 0 ? 1.0f : -1.0f;
    }
    return d;
}

// Ray-pick the gizmo cube under cursor (mx,my). Returns the picked region's
// direction triple, or false if the cursor misses the cube.
bool pickGizmo(const Eigen::Quaternionf& camOrient, const GizmoRect& r, float mx, float my,
               bool zUp, Eigen::Vector3f& outDir) {
    if (mx < r.x || mx > r.x + r.w || my < r.y || my > r.y + r.h) return false;
    float ndcX = (mx - r.x) / r.w * 2.0f - 1.0f;
    float ndcY = 1.0f - (my - r.y) / r.h * 2.0f;
    // Pick ray into the cube's local space. The cube model is
    // conjugate(camera) * Mworld, so undo the camera then the world up-axis basis
    // to land in the cube's (logical-axis) local frame.
    Eigen::Vector3f lo = math::rotate(
        camOrient, Eigen::Vector3f(ndcX * kGizmoOrthoHalf, ndcY * kGizmoOrthoHalf, kGizmoEyeZ));
    Eigen::Vector3f ld = math::rotate(camOrient, Eigen::Vector3f(0, 0, -1));
    Eigen::Quaternionf mwInv = math::conjugate(worldUpQuat(zUp));
    lo = math::rotate(mwInv, lo);
    ld = math::rotate(mwInv, ld);
    Eigen::Vector3f hit;
    if (!intersectUnitBox(lo, ld, hit)) return false;
    outDir = classifyDir(hit);
    return true;
}

// Ring pick: returns sector 0..3 (right/top/left/bottom) if the cursor is over
// the ring annulus behind the cube, else -1.
int pickRing(const GizmoRect& r, float mx, float my) {
    if (mx < r.x || mx > r.x + r.w || my < r.y || my > r.y + r.h) return -1;
    float sx = ((mx - r.x) / r.w * 2.0f - 1.0f) * kGizmoOrthoHalf;
    float sy = (1.0f - (my - r.y) / r.h * 2.0f) * kGizmoOrthoHalf;
    float rad = std::sqrt(sx * sx + sy * sy);
    if (rad < kRingInner || rad > kRingOuter) return -1;
    float deg = std::atan2(sy, sx) * 57.29578f;
    return ((static_cast<int>(std::lround(deg / 90.0f)) % 4) + 4) % 4;
}

// The 90-degree camera rotation a ring sector applies (about local axes).
Eigen::Quaternionf ringDelta(int sector) {
    switch (sector) {
        case 0: return math::quatAxisAngle(Eigen::Vector3f(0, 1, 0), -kHalfPi);  // right
        case 2: return math::quatAxisAngle(Eigen::Vector3f(0, 1, 0),  kHalfPi);  // left
        case 1: return math::quatAxisAngle(Eigen::Vector3f(1, 0, 0),  kHalfPi);  // top
        case 3: return math::quatAxisAngle(Eigen::Vector3f(1, 0, 0), -kHalfPi);  // bottom
    }
    return Eigen::Quaternionf::Identity();
}

void setComp(Eigen::Vector3f& v, int a, float val) { v[a] = val; }
float getComp(const Eigen::Vector3f& v, int a) { return v[a]; }

void emitQuad(render::Vertex* out, int& q, const Eigen::Vector3f c[4], const float col[3]) {
    if (q >= kHlQuads) return;
    for (int i = 0; i < 4; ++i)
        out[q * 4 + i] = {{c[i].x(), c[i].y(), c[i].z()}, {col[0], col[1], col[2]}};
    ++q;
}
void padDegenerate(render::Vertex* out, int q) {
    for (int i = q * 4; i < kHlVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// Highlight that conforms to the cube: filled cells lying flat on the faces.
// With kGizmoEdge = 0.78 the center face cell is large and the edge/corner cells
// are thin borders -- they tile the face with no gaps and connect at shared
// edges, and exactly match the clickable regions (same kGizmoEdge).
void buildCubeHighlight(Eigen::Vector3f d, const float col[3], render::Vertex out[kHlVerts]) {
    const float k = kGizmoEdge, eps = 0.013f;
    int q = 0;
    for (int a = 0; a < 3; ++a) {
        float da = getComp(d, a);
        if (da == 0) continue;
        int axes[2], na = 0;
        for (int x = 0; x < 3; ++x) if (x != a) axes[na++] = x;
        int b = axes[0], c = axes[1];
        float faceCoord = da * (1.0f + eps);
        auto range = [&](float dj, float& lo, float& hi) {
            if (dj == 0)      { lo = -k; hi = k; }
            else if (dj > 0)  { lo = k;  hi = 1; }
            else              { lo = -1; hi = -k; }
        };
        float blo, bhi, clo, chi;
        range(getComp(d, b), blo, bhi);
        range(getComp(d, c), clo, chi);
        auto mk = [&](float bv, float cv) {
            Eigen::Vector3f v;
            setComp(v, a, faceCoord); setComp(v, b, bv); setComp(v, c, cv);
            return v;
        };
        Eigen::Vector3f cor[4] = {mk(blo, clo), mk(bhi, clo), mk(bhi, chi), mk(blo, chi)};
        emitQuad(out, q, cor, col);
    }
    padDegenerate(out, q);
}

// Highlight a ring sector (screen-plane annulus wedge, drawn with identity model).
void buildRingHighlight(int sector, const float col[3], render::Vertex out[kHlVerts]) {
    int q = 0;
    float a0 = sector * kHalfPi - kHalfPi * 0.5f;  // sector center +/- 45deg
    const float zf = 0.02f;                        // in front of the ring base
    for (int s = 0; s < kHlQuads; ++s) {
        float t0 = a0 + kHalfPi * s / kHlQuads;
        float t1 = a0 + kHalfPi * (s + 1) / kHlQuads;
        Eigen::Vector3f cor[4] = {
            Eigen::Vector3f(std::cos(t0) * kRingInner, std::sin(t0) * kRingInner, zf),
            Eigen::Vector3f(std::cos(t0) * kRingOuter, std::sin(t0) * kRingOuter, zf),
            Eigen::Vector3f(std::cos(t1) * kRingOuter, std::sin(t1) * kRingOuter, zf),
            Eigen::Vector3f(std::cos(t1) * kRingInner, std::sin(t1) * kRingInner, zf),
        };
        emitQuad(out, q, cor, col);
    }
    padDegenerate(out, q);
}

// --- Gizmo up-axis toggle button -------------------------------------------
constexpr int kUpBtnQuads = 16;            // dynamic mesh capacity (bg + glyph)
constexpr int kUpBtnVerts = kUpBtnQuads * 4;
constexpr int kUpBtnPx     = 28;           // square button size (px)
constexpr int kUpBtnMargin = 4;            // inset from the gizmo's bottom-left corner

// The toggle button as a screen-pixel rect, tucked into the gizmo's bottom-left
// corner (outside the ring). Shared by the input hit-test and the renderer.
GizmoRect upToggleRect(const GizmoRect& g) {
    GizmoRect b;
    b.w = kUpBtnPx;
    b.h = kUpBtnPx;
    b.x = g.x + kUpBtnMargin;
    b.y = g.y + g.h - kUpBtnPx - kUpBtnMargin;
    return b;
}
// buildUpToggleGeometry is defined after appendText (which it uses), below.

// --- FPS widget geometry ---------------------------------------------------
constexpr int kFpsQuads = 256;            // dynamic mesh capacity
constexpr int kFpsVerts = kFpsQuads * 4;  // = 1024

// VSYNC checkbox, defined in PIXELS so it stays square regardless of the panel's
// aspect ratio. Anchored to the panel's top-right corner.
constexpr float kVsCbPx  = 16.0f;  // square side
constexpr float kVsCbPad = 9.0f;   // inset from the top/right edges

// The checkbox as a screen-pixel rect (top-left origin). Shared by the geometry
// builder (draw) and the input system (hit-test) so they line up.
struct CbRect { float x0, y0, x1, y1; };
CbRect fpsCheckboxRect(const FpsWidget& w) {
    float x1 = w.x + w.w - kVsCbPad, y0 = w.y + kVsCbPad;
    return {x1 - kVsCbPx, y0, x1, y0 + kVsCbPx};
}

// Appends textured glyph quads for `s` (proportional layout) using a baked Font.
// y-up; baseY is the text baseline. `h` is the glyph height in normalized-Y;
// horizontal extents are multiplied by `xScale` (= panelHeightPx/panelWidthPx)
// so glyphs keep their true aspect inside a non-square overlay viewport.
// Writes into out[q..], capped at `cap` quads.
void appendText(render::Vertex* out, int& q, int cap, const core::Font& f,
                const char* s, float penX, float baseY, float h, const float col[3],
                float z, float xScale) {
    for (; *s; ++s) {
        const core::Glyph& g = f.glyph(*s);
        if (g.w > 0 && g.h > 0 && q < cap) {
            float x0 = penX + g.xoff * h * xScale, y1 = baseY - g.yoff * h;
            float x1 = x0 + g.w * h * xScale, y0 = y1 - g.h * h;
            out[q * 4 + 0] = {{x0, y0, z}, {col[0], col[1], col[2]}, {g.u0, g.v1}};
            out[q * 4 + 1] = {{x1, y0, z}, {col[0], col[1], col[2]}, {g.u1, g.v1}};
            out[q * 4 + 2] = {{x1, y1, z}, {col[0], col[1], col[2]}, {g.u1, g.v0}};
            out[q * 4 + 3] = {{x0, y1, z}, {col[0], col[1], col[2]}, {g.u0, g.v0}};
            ++q;
        }
        penX += g.advance * h * xScale;
    }
}

// Builds the up-axis toggle button in normalized [0,1]^2 (y-up): a panel fill
// plus the current up-axis letter ("Y" or "Z") centered. `out` holds kUpBtnVerts.
void buildUpToggleGeometry(const AxisGizmo& g, render::Vertex* out) {
    int q = 0;
    if (!g.font) { for (int i = 0; i < kUpBtnVerts; ++i) out[i] = {{0,0,0},{0,0,0}}; return; }
    const core::Font& f = *g.font;
    const float wu = f.whiteU, wv = f.whiteV;
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float gg, float b,
                     float z) {
        if (q >= kUpBtnQuads) return;
        out[q*4+0] = {{x0,y0,z},{r,gg,b},{wu,wv}};
        out[q*4+1] = {{x1,y0,z},{r,gg,b},{wu,wv}};
        out[q*4+2] = {{x1,y1,z},{r,gg,b},{wu,wv}};
        out[q*4+3] = {{x0,y1,z},{r,gg,b},{wu,wv}};
        ++q;
    };
    // Panel: brighter when hovered; blue accent strip when active (Z up).
    float bg = g.upBtnHover ? 0.30f : 0.18f;
    solid(0.04f, 0.04f, 0.96f, 0.96f, bg, bg + 0.02f, bg + 0.05f, 0.0f);
    if (g.zUp)
        solid(0.04f, 0.04f, 0.96f, 0.12f, 0.30f, 0.62f, 0.95f, 0.1f);

    const char  letter[2] = {g.zUp ? 'Z' : 'Y', '\0'};
    const float col[3]    = {0.93f, 0.95f, 0.97f};
    const float h = 0.5f, xs = 1.0f;                    // button is square
    // Center the glyph's bounding box in the button so it is never clipped.
    const core::Glyph& gl = f.glyph(letter[0]);
    float penX  = 0.5f - (gl.xoff + gl.w * 0.5f) * h * xs;
    float baseY = 0.5f + gl.yoff * h + gl.h * h * 0.5f;  // box vertically centered
    appendText(out, q, kUpBtnQuads, f, letter, penX, baseY, h, col, 0.5f, xs);

    for (int i = q * 4; i < kUpBtnVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// Builds the widget geometry in normalized [0,1]^2 space (y-up). Layout is
// derived from the panel's pixel size so proportions hold at any size. The graph
// is a Windows-Task-Manager-style filled line chart: a faint grid, an opaque
// area under the curve, and a bright polyline along the top edge.
void buildFpsGeometry(const FpsWidget& wgt, render::Vertex* out) {
    int q = 0;
    if (!wgt.font) { for (int i = 0; i < kFpsVerts; ++i) out[i] = {{0,0,0},{0,0,0}}; return; }
    const core::Font& f = *wgt.font;
    const float wu = f.whiteU, wv = f.whiteV;
    const float W = static_cast<float>(wgt.w), H = static_cast<float>(wgt.h);
    auto nx    = [&](float px) { return px / W; };          // pixel x  -> normalized
    auto nyTop = [&](float px) { return 1.0f - px / H; };   // px-from-top -> norm y

    // Axis-aligned quad (white texel) and a free 4-corner quad (for the curve).
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) {
        if (q >= kFpsQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {wu, wv}};
        ++q;
    };
    auto quad4 = [&](float ax, float ay, float bx, float by, float cx, float cy,
                     float dx, float dy, float r, float g, float b, float z) {
        if (q >= kFpsQuads) return;
        out[q * 4 + 0] = {{ax, ay, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 1] = {{bx, by, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 2] = {{cx, cy, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 3] = {{dx, dy, z}, {r, g, b}, {wu, wv}};
        ++q;
    };

    solid(0, 0, 1, 1, 0.07f, 0.08f, 0.10f, 0.0f);  // background panel

    // --- Header: readout (left) + VSYNC label & checkbox (right) -----------
    int val = static_cast<int>(wgt.smoothFps + 0.5f);
    if (val > 9999) val = 9999;
    char s[16]; int len = 0;
    { char tmp[8]; int n = 0, v = val;
      if (v == 0) tmp[n++] = '0';
      while (v > 0) { tmp[n++] = char('0' + v % 10); v /= 10; }
      for (int i = n - 1; i >= 0; --i) s[len++] = tmp[i]; }
    s[len++] = ' '; s[len++] = 'F'; s[len++] = 'P'; s[len++] = 'S'; s[len] = '\0';

    const float xs = H / W;                       // glyph horizontal aspect fix
    float h = 28.0f / H;                          // readout text height (~28px)
    float tw = f.textWidth(s, h) * xs;            // rendered width (aspect-corrected)
    if (tw > 0.56f) { h *= 0.56f / tw; }          // keep it left of the VSYNC area
    const float col[3] = {0.92f, 0.95f, 0.93f};
    appendText(out, q, kFpsQuads, f, s, nx(9.0f), nyTop(27.0f), h, col, 0.6f, xs);

    // Checkbox (square, top-right). Reuse the hit-test rect for exact alignment.
    CbRect cb = fpsCheckboxRect(wgt);
    float bx0 = nx(cb.x0 - wgt.x), bx1 = nx(cb.x1 - wgt.x);
    float by1 = nyTop(cb.y0 - wgt.y), by0 = nyTop(cb.y1 - wgt.y);  // top/bottom
    float ix = nx(2.0f), iy = 2.0f / H;           // 2px border
    solid(bx0, by0, bx1, by1, 0.70f, 0.74f, 0.80f, 0.50f);            // border
    solid(bx0 + ix, by0 + iy, bx1 - ix, by1 - iy, 0.10f, 0.11f, 0.14f, 0.55f);  // well
    if (wgt.vsync)
        solid(bx0 + 2 * ix, by0 + 2 * iy, bx1 - 2 * ix, by1 - 2 * iy,
              0.25f, 0.85f, 0.45f, 0.60f);                            // check

    float hL = 16.0f / H;
    float lw = f.textWidth("VSYNC", hL) * xs;     // aspect-corrected width
    const float lcol[3] = {0.78f, 0.82f, 0.88f};
    appendText(out, q, kFpsQuads, f, "VSYNC", bx0 - nx(6.0f) - lw, nyTop(21.0f), hL,
               lcol, 0.6f, xs);

    // --- Graph area: grid + filled line chart ------------------------------
    const float gx0 = nx(8.0f), gx1 = nx(W - 8.0f);
    const float gy0 = nyTop(H - 8.0f);                       // bottom
    const float gy1 = nyTop(kVsCbPad + kVsCbPx + 8.0f);      // top, below header
    const float tnx = nx(1.0f), tny = 1.0f / H;              // 1px grid line

    const float gcol[3] = {0.15f, 0.17f, 0.21f};             // faint grid
    for (int i = 1; i < 4; ++i) {                            // 3 horizontal lines
        float y = gy0 + (gy1 - gy0) * i / 4.0f;
        solid(gx0, y - tny, gx1, y + tny, gcol[0], gcol[1], gcol[2], 0.1f);
    }
    for (int i = 1; i < 6; ++i) {                            // 5 vertical lines
        float x = gx0 + (gx1 - gx0) * i / 6.0f;
        solid(x - tnx, gy0, x + tnx, gy1, gcol[0], gcol[1], gcol[2], 0.1f);
    }

    // Auto-scale to the recent peak (min 60 so an idle scene stays sensible).
    const int N = FpsWidget::kSamples;
    float peak = 60.0f;
    for (int i = 0; i < N; ++i) peak = (std::max)(peak, wgt.history[i]);
    float scale = peak * 1.18f;
    auto xAt = [&](int i) { return gx0 + (gx1 - gx0) * i / (N - 1); };
    auto yAt = [&](int i) {
        float fps = wgt.history[(wgt.head + i) % N];         // oldest -> newest
        return gy0 + (gy1 - gy0) * clampf(fps / scale, 0.0f, 1.0f);
    };

    const float fill[3] = {0.10f, 0.34f, 0.22f};             // area under curve
    const float line[3] = {0.30f, 0.88f, 0.52f};             // bright top edge
    const float lt = 2.2f / H;                               // line thickness
    for (int i = 0; i < N - 1; ++i) {
        float x0 = xAt(i), x1 = xAt(i + 1), y0 = yAt(i), y1 = yAt(i + 1);
        // Filled trapezoid down to the baseline (opaque -> hides grid below).
        quad4(x0, gy0, x1, gy0, x1, y1, x0, y0, fill[0], fill[1], fill[2], 0.25f);
        // Bright band hugging the top edge = the polyline.
        quad4(x0, y0 - lt, x1, y1 - lt, x1, y1, x0, y0, line[0], line[1], line[2], 0.45f);
    }

    for (int i = q * 4; i < kFpsVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Camera controls panel -------------------------------------------------
constexpr int kCtrlQuads = 64;
constexpr int kCtrlVerts = kCtrlQuads * 4;

struct CRect { float x, y, w, h; };          // screen-pixel rect
struct CtrlRects { CRect mode, minus, plus; };

CtrlRects controlRects(const CameraControls& cc) {
    CtrlRects r;
    float pad = 6.0f;
    r.mode  = {cc.x + pad, cc.y + pad, cc.w - 2 * pad, cc.h * 0.42f};
    float row2 = cc.y + cc.h * 0.50f, bh = cc.h * 0.42f;
    r.minus = {cc.x + pad, row2, 30.0f, bh};
    r.plus  = {cc.x + cc.w - pad - 30.0f, row2, 30.0f, bh};
    return r;
}

// Builds the panel: background + buttons (white texel) + text (font glyphs).
void buildControlsGeometry(const CameraControls& cc, const Camera& cam,
                           render::Vertex* out) {
    const core::Font& f = *cc.font;
    int q = 0;
    const float wu = f.whiteU, wv = f.whiteV;
    auto quad = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                    float u0, float v0, float u1, float v1, float z) {
        if (q >= kCtrlQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {u0, v1}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {u1, v1}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {u1, v0}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {u0, v0}};
        ++q;
    };
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) { quad(x0, y0, x1, y1, r, g, b, wu, wv, wu, wv, z); };
    // Horizontal aspect fix: the panel maps normalized X to cc.w px and Y to
    // cc.h px, so glyph widths must be scaled by cc.h/cc.w to avoid stretching.
    const float xs = static_cast<float>(cc.h) / static_cast<float>(cc.w);
    auto text = [&](const char* s, float penX, float baseY, float h, float r, float g,
                    float b, float z) {
        for (; *s; ++s) {
            const core::Glyph& gl = f.glyph(*s);
            if (gl.w > 0 && gl.h > 0) {
                float x0 = penX + gl.xoff * h * xs, y1 = baseY - gl.yoff * h;
                float x1 = x0 + gl.w * h * xs, y0 = y1 - gl.h * h;
                quad(x0, y0, x1, y1, r, g, b, gl.u0, gl.v0, gl.u1, gl.v1, z);
            }
            penX += gl.advance * h * xs;
        }
    };
    auto textW = [&](const char* s, float h) { return f.textWidth(s, h) * xs; };
    auto toN = [&](const CRect& rr, float& x0, float& y0, float& x1, float& y1) {
        x0 = (rr.x - cc.x) / cc.w;            x1 = (rr.x + rr.w - cc.x) / cc.w;
        y1 = 1.0f - (rr.y - cc.y) / cc.h;     y0 = 1.0f - (rr.y + rr.h - cc.y) / cc.h;
    };

    solid(0, 0, 1, 1, 0.10f, 0.11f, 0.13f, 0.0f);  // panel background

    CtrlRects R = controlRects(cc);
    float x0, y0, x1, y1;

    // Projection mode button.
    toN(R.mode, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.20f, 0.22f, 0.27f, 0.3f);
    const char* modeTxt =
        cam.mode == ProjectionMode::Perspective ? "Perspective" : "Orthographic";
    float h1 = (y1 - y0) * 0.64f;
    text(modeTxt, (x0 + x1) * 0.5f - textW(modeTxt, h1) * 0.5f,
         (y0 + y1) * 0.5f - h1 * 0.35f, h1, 0.92f, 0.95f, 0.95f, 0.6f);

    // Minus / plus buttons, glyphs drawn as crisp bars sized off the button's
    // smaller dimension so they stay symmetric (not stubby).
    const float gWhite[3] = {0.95f, 0.96f, 0.96f};
    toN(R.minus, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.20f, 0.22f, 0.27f, 0.3f);
    float cx = (x0 + x1) * 0.5f, cy = (y0 + y1) * 0.5f;
    float md = (std::min)(x1 - x0, y1 - y0);
    float len = md * 0.34f, th = md * 0.11f;
    solid(cx - len, cy - th, cx + len, cy + th, gWhite[0], gWhite[1], gWhite[2], 0.6f);  // "-"
    float vx0 = x1;  // right edge of minus
    toN(R.plus, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.20f, 0.22f, 0.27f, 0.3f);
    cx = (x0 + x1) * 0.5f; cy = (y0 + y1) * 0.5f;
    md = (std::min)(x1 - x0, y1 - y0); len = md * 0.34f; th = md * 0.11f;
    solid(cx - len, cy - th, cx + len, cy + th, gWhite[0], gWhite[1], gWhite[2], 0.6f);  // "+" h
    solid(cx - th, cy - len, cx + th, cy + len, gWhite[0], gWhite[1], gWhite[2], 0.6f);  // "+" v
    float vx1 = x0;  // left edge of plus

    // Value readout between the buttons.
    char buf[32];
    if (cam.mode == ProjectionMode::Perspective)
        std::snprintf(buf, sizeof(buf), "FOV %d", static_cast<int>(cam.fovYDegrees + 0.5f));
    else
        std::snprintf(buf, sizeof(buf), "Size %.1f", cam.orthoSize);
    float hv = (y1 - y0) * 0.54f;
    text(buf, (vx0 + vx1) * 0.5f - textW(buf, hv) * 0.5f, (y0 + y1) * 0.5f - hv * 0.35f,
         hv, 0.92f, 0.95f, 0.95f, 0.6f);

    for (int i = q * 4; i < kCtrlVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Cross-section panel ---------------------------------------------------
constexpr int kCsQuads = 64;
constexpr int kCsVerts = kCsQuads * 4;

struct CsRects { CRect enable, axis, flip, track; };

// Pixel rects for the panel widgets. The slider handle is derived from pos, so
// only the track (groove) is stored; the input system maps clicks across it.
CsRects crossSectionRects(const CrossSection& cs) {
    CsRects r;
    const float pad = 8.0f;
    r.enable = {cs.x + cs.w - pad - 16.0f, cs.y + 6.0f, 16.0f, 16.0f};
    r.axis   = {cs.x + pad, cs.y + 30.0f, 46.0f, 22.0f};
    r.flip   = {cs.x + cs.w - pad - 52.0f, cs.y + 30.0f, 52.0f, 22.0f};
    r.track  = {cs.x + 12.0f, cs.y + 72.0f, cs.w - 24.0f, 6.0f};
    return r;
}

// Maps the plane position to the handle's center X (px) along the track.
float crossSectionHandleX(const CrossSection& cs, const CRect& track) {
    float span = cs.maxPos - cs.minPos;
    float t    = span > 1e-6f ? (cs.pos - cs.minPos) / span : 0.5f;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return track.x + t * track.w;
}

// Builds the cross-section panel: background, title + enable checkbox, axis and
// flip buttons, a value readout, and the slider (groove + filled + handle).
void buildCrossSectionGeometry(const CrossSection& cs, render::Vertex* out) {
    const core::Font& f = *cs.font;
    int q = 0;
    const float wu = f.whiteU, wv = f.whiteV;
    auto quad = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                    float u0, float v0, float u1, float v1, float z) {
        if (q >= kCsQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {u0, v1}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {u1, v1}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {u1, v0}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {u0, v0}};
        ++q;
    };
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) { quad(x0, y0, x1, y1, r, g, b, wu, wv, wu, wv, z); };
    const float xs = static_cast<float>(cs.h) / static_cast<float>(cs.w);  // aspect fix
    auto text = [&](const char* s, float penX, float baseY, float h, float r, float g,
                    float b, float z) {
        for (; *s; ++s) {
            const core::Glyph& gl = f.glyph(*s);
            if (gl.w > 0 && gl.h > 0) {
                float x0 = penX + gl.xoff * h * xs, y1 = baseY - gl.yoff * h;
                float x1 = x0 + gl.w * h * xs, y0 = y1 - gl.h * h;
                quad(x0, y0, x1, y1, r, g, b, gl.u0, gl.v0, gl.u1, gl.v1, z);
            }
            penX += gl.advance * h * xs;
        }
    };
    auto textW = [&](const char* s, float h) { return f.textWidth(s, h) * xs; };
    // Normalize an absolute pixel rect into the panel's [0,1] (y-up) space.
    auto toN = [&](const CRect& rr, float& x0, float& y0, float& x1, float& y1) {
        x0 = (rr.x - cs.x) / cs.w;            x1 = (rr.x + rr.w - cs.x) / cs.w;
        y1 = 1.0f - (rr.y - cs.y) / cs.h;     y0 = 1.0f - (rr.y + rr.h - cs.y) / cs.h;
    };

    solid(0, 0, 1, 1, 0.10f, 0.11f, 0.13f, 0.0f);  // panel background

    CsRects R = crossSectionRects(cs);
    float x0, y0, x1, y1;

    // Title. Match the camera-controls panel label size above (~20px: that panel's
    // h=76, button 0.42 tall, text 0.64 of the button).
    float th = 20.0f / cs.h;
    text("Section", 10.0f / cs.w, 1.0f - 8.0f / cs.h - th, th, 0.85f, 0.88f, 0.92f, 0.6f);

    // Enable checkbox (green tick when on).
    toN(R.enable, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.70f, 0.74f, 0.80f, 0.5f);             // border
    float ix = 2.0f / cs.w, iy = 2.0f / cs.h;
    solid(x0 + ix, y0 + iy, x1 - ix, y1 - iy, 0.10f, 0.11f, 0.14f, 0.55f);  // well
    if (cs.enabled)
        solid(x0 + 2 * ix, y0 + 2 * iy, x1 - 2 * ix, y1 - 2 * iy, 0.25f, 0.85f, 0.45f, 0.6f);

    // Axis button (X/Y/Z).
    toN(R.axis, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.20f, 0.22f, 0.27f, 0.3f);
    const char axisTxt[2] = {static_cast<char>('X' + cs.axis), '\0'};
    float ah = (y1 - y0) * 0.72f;
    text(axisTxt, (x0 + x1) * 0.5f - textW(axisTxt, ah) * 0.5f,
         (y0 + y1) * 0.5f - ah * 0.35f, ah, 0.92f, 0.95f, 0.95f, 0.6f);

    // Flip button (brighter when keeping the +side).
    toN(R.flip, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, cs.flip ? 0.28f : 0.20f, cs.flip ? 0.34f : 0.22f,
          cs.flip ? 0.44f : 0.27f, 0.3f);
    float fh = (y1 - y0) * 0.72f;
    text("Flip", (x0 + x1) * 0.5f - textW("Flip", fh) * 0.5f,
         (y0 + y1) * 0.5f - fh * 0.35f, fh, 0.90f, 0.93f, 0.96f, 0.6f);

    // Value readout (between the two buttons).
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%.2f", cs.pos);
    float vh = 17.0f / cs.h;  // match the controls value (FOV) size above
    float vMid = (R.axis.x + R.axis.w + R.flip.x) * 0.5f;  // pixel midpoint of the gap
    text(buf, (vMid - cs.x) / cs.w - textW(buf, vh) * 0.5f,
         1.0f - (R.axis.y + R.axis.h * 0.5f - cs.y) / cs.h - vh * 0.35f, vh,
         0.80f, 0.84f, 0.90f, 0.6f);

    // Slider: groove, filled portion up to the handle, then the handle itself.
    float hx = crossSectionHandleX(cs, R.track);
    toN(R.track, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.20f, 0.21f, 0.25f, 0.3f);                   // groove
    float hxn = (hx - cs.x) / cs.w;
    float fillCol = cs.enabled ? 1.0f : 0.4f;
    solid(x0, y0, hxn, y1, 0.30f * fillCol, 0.55f * fillCol, 0.95f * fillCol, 0.35f);  // filled
    // Handle (taller than the groove).
    float hw = 5.0f / cs.w;
    float hy0 = 1.0f - (R.track.y + R.track.h * 0.5f + 10.0f - cs.y) / cs.h;
    float hy1 = 1.0f - (R.track.y + R.track.h * 0.5f - 10.0f - cs.y) / cs.h;
    float hc = cs.enabled ? 0.95f : 0.5f;
    solid(hxn - hw, hy0, hxn + hw, hy1, hc, hc, hc, 0.5f);

    for (int i = q * 4; i < kCsVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Top menu bar ----------------------------------------------------------
constexpr int   kMenuQuads = 1024;          // must match kMenuQ in main.cpp
constexpr int   kMenuVerts = kMenuQuads * 4;
constexpr float kMenuTitlePad = 14.0f;      // l/r padding around a bar title (px)
constexpr float kMenuItemH    = 38.0f;      // dropdown item row height (px)
constexpr float kMenuSepH     = 11.0f;      // separator row height (px)
constexpr float kMenuCheckW   = 24.0f;      // left gutter holding the check tick
constexpr float kMenuPadX     = 14.0f;      // dropdown left/right padding
constexpr float kMenuShortGap = 28.0f;      // gap between label and the shortcut
constexpr float kMenuMinDropW = 190.0f;     // minimum dropdown width (px)
constexpr float kMenuTextH    = 26.0f;      // dropdown/title text px height

// Left edge (px) of menu title `i` and (via out_x1) its right edge. Titles are laid
// out left to right, each `textWidth + 2*pad` wide. Identical maths in the builder
// and the hit-test so they always agree.
float menuTitleRect(const MenuBar& mb, const core::Font& f, float th, int i,
                    float* out_x1) {
    float x = 0.0f;
    for (int k = 0; k < static_cast<int>(mb.menus.size()); ++k) {
        float w = f.textWidth(mb.menus[k].title.c_str(), th) + 2.0f * kMenuTitlePad;
        if (k == i) { if (out_x1) *out_x1 = x + w; return x; }
        x += w;
    }
    if (out_x1) *out_x1 = x;
    return x;
}

// Dropdown panel width for a menu: the widest item (check gutter + label + optional
// shortcut), clamped to a minimum.
float menuDropWidth(const Menu& m, const core::Font& f, float th) {
    float w = kMenuMinDropW;
    for (const auto& it : m.items) {
        if (it.kind == MenuItem::Separator) continue;
        float row = kMenuCheckW + f.textWidth(it.label.c_str(), th) + kMenuPadX;
        if (!it.shortcut.empty())
            row += kMenuShortGap + f.textWidth(it.shortcut.c_str(), th);
        w = (std::max)(w, row);
    }
    return w;
}

// Total dropdown height (sum of row heights; separators are shorter).
float menuDropHeight(const Menu& m) {
    float h = 0.0f;
    for (const auto& it : m.items) h += (it.kind == MenuItem::Separator ? kMenuSepH : kMenuItemH);
    return h;
}

// Top edge (px, relative to the bar bottom) of item `i`, and its height via out_h.
float menuItemTop(const Menu& m, int i, float* out_h) {
    float y = 0.0f;
    for (int k = 0; k < static_cast<int>(m.items.size()); ++k) {
        float h = (m.items[k].kind == MenuItem::Separator ? kMenuSepH : kMenuItemH);
        if (k == i) { if (out_h) *out_h = h; return y; }
        y += h;
    }
    if (out_h) *out_h = 0.0f;
    return y;
}

// Builds the bar in normalized [0,1]^2 (y-up) over an overlay whose pixel size is
// viewportW x overlayH. Working in pixels and converting keeps the text crisp and
// the layout identical to the hit-tests in menuBarInputSystem. overlayH grows to
// include the open dropdown.
void buildMenuGeometry(const MenuBar& mb, render::Vertex* out, uint32_t viewportW,
                       float overlayH) {
    int q = 0;
    if (!mb.font) { for (int i = 0; i < kMenuVerts; ++i) out[i] = {{0,0,0},{0,0,0}}; return; }
    const core::Font& f = *mb.font;
    const float wu = f.whiteU, wv = f.whiteV;
    const float W = static_cast<float>(viewportW);
    const float barH = static_cast<float>(mb.height);
    const float xs = overlayH / W;                       // glyph horizontal aspect fix
    auto nx  = [&](float px) { return px / W; };
    auto nyT = [&](float py) { return 1.0f - py / overlayH; };  // px-from-top -> norm y
    // Solid (white-texel) rect from a top-left pixel rect.
    auto solid = [&](float x0, float yTop, float x1, float yBot, float r, float g,
                     float b, float z) {
        if (q >= kMenuQuads) return;
        float ny0 = nyT(yBot), ny1 = nyT(yTop), nx0 = nx(x0), nx1 = nx(x1);
        out[q * 4 + 0] = {{nx0, ny0, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 1] = {{nx1, ny0, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 2] = {{nx1, ny1, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 3] = {{nx0, ny1, z}, {r, g, b}, {wu, wv}};
        ++q;
    };
    // Text at a pixel pen position; baseY is the baseline measured from the top.
    auto text = [&](const char* s, float penPxX, float basePxY, float hPx,
                    const float col[3], float z) {
        appendText(out, q, kMenuQuads, f, s, nx(penPxX), nyT(basePxY), hPx / overlayH,
                   col, z, xs);
    };

    const float txt[3]  = {0.90f, 0.92f, 0.95f};
    const float dim[3]  = {0.60f, 0.63f, 0.70f};
    const float th = kMenuTextH;

    solid(0, 0, W, barH, 0.16f, 0.17f, 0.20f, 0.0f);     // bar background

    // Top-level titles.
    for (int i = 0; i < static_cast<int>(mb.menus.size()); ++i) {
        float x1, x0 = menuTitleRect(mb, f, th, i, &x1);
        if (i == mb.openMenu) solid(x0, 0, x1, barH, 0.26f, 0.28f, 0.34f, 0.1f);
        text(mb.menus[i].title.c_str(), x0 + kMenuTitlePad,
             (barH + th) * 0.5f - 2.0f, th, txt, 0.5f);
    }

    // Right-aligned status (e.g. "Loading 42%") shown while a background load runs.
    if (!mb.statusText.empty()) {
        const char* st = mb.statusText.c_str();
        const float stw = f.textWidth(st, th);           // pixel width at height th
        const float stCol[3] = {0.55f, 0.85f, 1.0f};
        text(st, W - stw - 14.0f, (barH + th) * 0.5f - 2.0f, th, stCol, 0.5f);
    }

    // Open dropdown.
    if (mb.openMenu >= 0 && mb.openMenu < static_cast<int>(mb.menus.size())) {
        const Menu& m = mb.menus[mb.openMenu];
        float tx1, tx0 = menuTitleRect(mb, f, th, mb.openMenu, &tx1);
        float dw = menuDropWidth(m, f, th);
        float dx0 = clampf(tx0, 0.0f, (std::max)(0.0f, W - dw));  // keep on screen
        float dh = menuDropHeight(m);
        solid(dx0, barH, dx0 + dw, barH + dh, 0.13f, 0.14f, 0.17f, 0.2f);  // panel

        for (int i = 0; i < static_cast<int>(m.items.size()); ++i) {
            const MenuItem& it = m.items[i];
            float ih, iy = barH + menuItemTop(m, i, &ih);
            if (it.kind == MenuItem::Separator) {
                solid(dx0 + kMenuPadX, iy + ih * 0.5f - 1.0f, dx0 + dw - kMenuPadX,
                      iy + ih * 0.5f + 1.0f, 0.30f, 0.32f, 0.38f, 0.25f);
                continue;
            }
            if (i == mb.hoverItem)
                solid(dx0, iy, dx0 + dw, iy + ih, 0.24f, 0.30f, 0.42f, 0.25f);
            float baseY = iy + (ih + th) * 0.5f - 2.0f;
            if (it.kind == MenuItem::Check && it.checked)  // tick in the left gutter
                solid(dx0 + 9.0f, iy + ih * 0.5f - 5.0f, dx0 + 19.0f,
                      iy + ih * 0.5f + 5.0f, 0.55f, 0.85f, 1.0f, 0.3f);
            text(it.label.c_str(), dx0 + kMenuCheckW, baseY, th, txt, 0.5f);
            if (!it.shortcut.empty()) {
                float sw = f.textWidth(it.shortcut.c_str(), th);
                text(it.shortcut.c_str(), dx0 + dw - kMenuPadX - sw, baseY, th, dim, 0.5f);
            }
        }
    }

    for (int i = q * 4; i < kMenuVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}
} // namespace

namespace {
// --- Left selection toolbar geometry ---------------------------------------
constexpr float kTbX       = 8.0f;                 // left margin (px)
constexpr float kTbTop     = kMenuBarHeight + 10.0f;
constexpr float kTbBtn     = 46.0f;                // button size (px, square)
constexpr float kTbGap     = 4.0f;                 // gap between buttons in a group
constexpr float kTbGroupGap= 14.0f;               // gap between groups
constexpr float kTbText    = 22.0f;               // button caption px height
constexpr int   kTbQuads   = 512;
constexpr int   kTbVerts   = kTbQuads * 4;

// Top edge (px) of toolbar button `i`, accounting for the extra gap between groups.
float toolbarButtonTop(const std::vector<ToolbarButton>& b, int i) {
    float y = kTbTop;
    for (int k = 0; k < i; ++k)
        y += kTbBtn + (b[k].group != b[k + 1].group ? kTbGroupGap : kTbGap);
    return y;
}

// Is button `b` the active value for its group given the current mode?
bool toolbarButtonActive(const ToolbarButton& b, const SelectionMode& sm) {
    switch (b.group) {
        case 0: return b.value == static_cast<int>(sm.target);
        case 1: return b.value == static_cast<int>(sm.action);
        case 2: return b.value == static_cast<int>(sm.filter);
        case 3: return b.value == static_cast<int>(sm.modifier);
    }
    return false;
}

void buildToolbarGeometry(const SelectionToolbar& tb, const SelectionMode& sm,
                          render::Vertex* out, uint32_t viewportW, uint32_t viewportH) {
    int q = 0;
    if (!tb.font) { for (int i = 0; i < kTbVerts; ++i) out[i] = {{0,0,0},{0,0,0}}; return; }
    const core::Font& f = *tb.font;
    const float wu = f.whiteU, wv = f.whiteV;
    const float W = static_cast<float>(viewportW), Hh = static_cast<float>(viewportH);
    const float xs = Hh / W;
    auto nx  = [&](float px) { return px / W; };
    auto nyT = [&](float py) { return 1.0f - py / Hh; };
    auto solid = [&](float x0, float yTop, float x1, float yBot, float r, float g,
                     float b, float z) {
        if (q >= kTbQuads) return;
        float ny0 = nyT(yBot), ny1 = nyT(yTop), nx0 = nx(x0), nx1 = nx(x1);
        out[q*4+0] = {{nx0, ny0, z}, {r, g, b}, {wu, wv}};
        out[q*4+1] = {{nx1, ny0, z}, {r, g, b}, {wu, wv}};
        out[q*4+2] = {{nx1, ny1, z}, {r, g, b}, {wu, wv}};
        out[q*4+3] = {{nx0, ny1, z}, {r, g, b}, {wu, wv}};
        ++q;
    };
    auto text = [&](const char* s, float penPxX, float basePxY, float hPx,
                    const float col[3], float z) {
        appendText(out, q, kTbQuads, f, s, nx(penPxX), nyT(basePxY), hPx / Hh, col, z, xs);
    };

    const float on[3]   = {0.20f, 0.55f, 0.95f};  // active button fill
    const float hov[3]  = {0.30f, 0.33f, 0.40f};
    const float off[3]  = {0.17f, 0.18f, 0.22f};
    const float txt[3]  = {0.92f, 0.94f, 0.97f};

    for (int i = 0; i < static_cast<int>(tb.buttons.size()); ++i) {
        const ToolbarButton& b = tb.buttons[i];
        float y0 = toolbarButtonTop(tb.buttons, i), y1 = y0 + kTbBtn;
        float x0 = kTbX, x1 = kTbX + kTbBtn;
        const float* fill = toolbarButtonActive(b, sm) ? on : (i == tb.hover ? hov : off);
        solid(x0, y0, x1, y1, fill[0], fill[1], fill[2], 0.2f);
        float tw = f.textWidth(b.label.c_str(), kTbText);
        text(b.label.c_str(), x0 + (kTbBtn - tw) * 0.5f, y0 + (kTbBtn + kTbText) * 0.5f - 2.0f,
             kTbText, txt, 0.5f);
    }

    // Box rubber-band / lasso overlay drawn while dragging.
    const float band[3] = {0.95f, 0.75f, 0.20f};
    auto seg = [&](float ax, float ay, float bx, float by) {  // thin quad between 2 px points
        float dx = bx - ax, dy = by - ay;
        float len = std::sqrt(dx*dx + dy*dy);
        if (len < 1e-3f) return;
        float npx = -dy / len * 1.2f, npy = dx / len * 1.2f;  // 1.2px half-thickness
        if (q >= kTbQuads) return;
        out[q*4+0] = {{nx(ax+npx), nyT(ay+npy), 0.4f}, {band[0],band[1],band[2]}, {wu,wv}};
        out[q*4+1] = {{nx(bx+npx), nyT(by+npy), 0.4f}, {band[0],band[1],band[2]}, {wu,wv}};
        out[q*4+2] = {{nx(bx-npx), nyT(by-npy), 0.4f}, {band[0],band[1],band[2]}, {wu,wv}};
        out[q*4+3] = {{nx(ax-npx), nyT(ay-npy), 0.4f}, {band[0],band[1],band[2]}, {wu,wv}};
        ++q;
    };
    if (sm.dragging && sm.action == SelAction::Box) {
        float x0 = sm.dragX0, y0 = sm.dragY0, x1 = sm.dragX1, y1 = sm.dragY1;
        seg(x0, y0, x1, y0); seg(x1, y0, x1, y1); seg(x1, y1, x0, y1); seg(x0, y1, x0, y0);
    } else if (sm.dragging && sm.action == SelAction::Lasso) {
        for (size_t i = 1; i < sm.lassoX.size(); ++i)
            seg(sm.lassoX[i-1], sm.lassoY[i-1], sm.lassoX[i], sm.lassoY[i]);
    }

    for (int i = q * 4; i < kTbVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}
} // namespace

void spinSystem(entt::registry& world, float dt) {
    auto view = world.view<Transform, Spin>();
    for (auto entity : view) {
        auto& t = view.get<Transform>(entity);
        auto& s = view.get<Spin>(entity);
        // Integrate angular velocity as a rigid rotation (no Euler drift/lock).
        Eigen::Vector3f w = s.axisRadiansPerSec;
        float speed = std::sqrt(math::dot(w, w));
        if (speed > 1e-8f) {
            Eigen::Quaternionf dq = math::quatAxisAngle(w, speed * dt);
            t.orientation = math::normalize(t.orientation * dq);
        }
    }
}

void cameraManipulatorSystem(entt::registry& world, const core::Input& input,
                             float dt) {
    auto view = world.view<Transform, Camera, CameraManipulator>();
    for (auto entity : view) {
        auto& t = view.get<Transform>(entity);
        auto& m = view.get<CameraManipulator>(entity);

        if (m.animating) {
            // Smooth snap (gizmo click): slerp orientation, suspend orbit.
            m.animTime += dt;
            float tt = m.animDuration > 0.0f ? m.animTime / m.animDuration : 1.0f;
            if (tt >= 1.0f) { tt = 1.0f; m.animating = false; }
            float e = tt * tt * (3.0f - 2.0f * tt);  // smoothstep ease
            m.orientation = math::slerp(m.animFrom, m.animTo, e);
        } else if (!input.captured && input.buttonRight &&
                   (input.mouseDeltaX != 0.0f || input.mouseDeltaY != 0.0f)) {
            // Orbit: right-drag tumbles about the camera's OWN x/y axes and
            // composes on the local side -- a true trackball, no gimbal lock.
            Eigen::Quaternionf dYaw =
                math::quatAxisAngle(Eigen::Vector3f(0, 1, 0), -input.mouseDeltaX * m.rotateSpeed);
            Eigen::Quaternionf dPitch =
                math::quatAxisAngle(Eigen::Vector3f(1, 0, 0), -input.mouseDeltaY * m.rotateSpeed);
            m.orientation = math::normalize(m.orientation * dYaw * dPitch);
        }

        // Zoom: scroll wheel scales distance multiplicatively, so each notch moves
        // a constant fraction -- same feel whether the model is millimetres or
        // kilometres across (a fixed step would crawl on big models, teleport on
        // small ones).
        if (input.wheel != 0.0f) {
            m.targetAnimating = false;  // a manual zoom overrides a reset/recenter glide
            m.distance *= std::pow(0.9f, input.wheel * m.zoomSpeed);
            m.distance  = clampf(m.distance, m.minDistance, m.maxDistance);
        }

        // Pan: middle-drag slides the target across the camera plane.
        // (Right-drag is orbit; left-click is picking.)
        if (input.buttonMiddle) {
            m.targetAnimating = false;  // a manual pan overrides a recenter glide
            Eigen::Vector3f right = math::rotate(m.orientation, Eigen::Vector3f(1, 0, 0));
            Eigen::Vector3f up    = math::rotate(m.orientation, Eigen::Vector3f(0, 1, 0));
            float k = m.panSpeed * m.distance;
            m.target = m.target - right * (input.mouseDeltaX * k) +
                       up * (input.mouseDeltaY * k);
        }

        // Recenter glide (Ctrl+left-click) / R reset: ease the orbit pivot and the
        // distance. The position below follows automatically since it is
        // target+distance based.
        if (m.targetAnimating) {
            m.targetAnimTime += dt;
            float tt = m.animDuration > 0.0f ? m.targetAnimTime / m.animDuration : 1.0f;
            if (tt >= 1.0f) { tt = 1.0f; m.targetAnimating = false; }
            float e = tt * tt * (3.0f - 2.0f * tt);  // smoothstep ease
            m.target   = m.targetFrom + (m.targetTo - m.targetFrom) * e;
            m.distance = m.distFrom + (m.distTo - m.distFrom) * e;
        }

        // Place the camera on the orbit sphere; eye sits along the local +Z.
        Eigen::Vector3f offset = math::rotate(m.orientation, Eigen::Vector3f(0, 0, 1));
        t.position    = m.target + offset * m.distance;
        t.orientation = m.orientation;
    }
}

void menuBarInputSystem(entt::registry& world, core::Input& input,
                        uint32_t viewportW, uint32_t viewportH) {
    (void)viewportH;
    MenuBar* mb = nullptr;
    auto view = world.view<MenuBar>();
    for (auto e : view) { mb = &view.get<MenuBar>(e); break; }
    if (!mb || !mb->visible || !mb->font) return;

    const core::Font& f = *mb->font;
    const float th   = kMenuTextH;
    const float barH = static_cast<float>(mb->height);
    const float W    = static_cast<float>(viewportW);
    float mx = input.mousePosX, my = input.mousePosY;
    const int nMenus = static_cast<int>(mb->menus.size());

    // Which title is the pointer over (-1 = none)?
    int onTitle = -1;
    if (my >= 0.0f && my < barH) {
        for (int i = 0; i < nMenus; ++i) {
            float x1, x0 = menuTitleRect(*mb, f, th, i, &x1);
            if (mx >= x0 && mx < x1) { onTitle = i; break; }
        }
    }
    bool inBar = my >= 0.0f && my < barH;

    // Dropdown geometry of the open menu + which item the pointer is over.
    bool inDrop = false;
    mb->hoverItem = -1;
    if (mb->openMenu >= 0 && mb->openMenu < nMenus) {
        const Menu& m = mb->menus[mb->openMenu];
        float tx1, tx0 = menuTitleRect(*mb, f, th, mb->openMenu, &tx1);
        float dw  = menuDropWidth(m, f, th);
        float dx0 = clampf(tx0, 0.0f, (std::max)(0.0f, W - dw));
        float dh  = menuDropHeight(m);
        if (mx >= dx0 && mx < dx0 + dw && my >= barH && my < barH + dh) {
            inDrop = true;
            float local = my - barH;
            for (int i = 0; i < static_cast<int>(m.items.size()); ++i) {
                float ih, iy = menuItemTop(m, i, &ih);
                if (local >= iy && local < iy + ih) {
                    if (m.items[i].kind != MenuItem::Separator) mb->hoverItem = i;
                    break;
                }
            }
        }
    }

    // Hovering a different title while a menu is open switches to it (classic UX).
    if (mb->openMenu >= 0 && onTitle >= 0 && onTitle != mb->openMenu)
        mb->openMenu = onTitle;

    // The bar (and an open dropdown) own their pixels: suppress orbit/picking.
    if (inBar || inDrop) input.captured = true;

    if (input.leftClicked) {
        if (onTitle >= 0) {
            mb->openMenu   = (mb->openMenu == onTitle) ? -1 : onTitle;  // toggle
            input.captured = true;
        } else if (inDrop) {
            if (mb->hoverItem >= 0) {
                const Menu& m = mb->menus[mb->openMenu];
                mb->triggered = m.items[mb->hoverItem].action;  // dispatched by the app
            }
            mb->openMenu   = -1;
            input.captured = true;
        } else {
            mb->openMenu = -1;                // click elsewhere closes the menu
        }
    }
}

std::vector<Menu> defaultAppMenus() {
    using A = MenuAction;
    using K = MenuItem;
    auto act   = [](const char* l, A a, const char* sc = "") {
        return MenuItem{l, sc, a, K::Action, false};
    };
    auto chk   = [](const char* l, A a, const char* sc = "") {
        return MenuItem{l, sc, a, K::Check, false};
    };
    auto sep   = []() { return MenuItem{"", "", A::None, K::Separator, false}; };

    std::vector<Menu> menus;
    menus.push_back({"File", {
        act("Open...",    A::OpenFile),
        act("Screenshot", A::Screenshot, "C"),
        sep(),
        act("Quit",       A::Quit, "Esc"),
    }});
    menus.push_back({"View", {
        chk("Ground Grid",   A::ToggleGrid, "Space"),
        act("Reset Camera",  A::ResetCamera, "R"),
        sep(),
        act("Z-Up / Y-Up",   A::ToggleUpAxis),
        act("Perspective / Ortho", A::ToggleProjection),
    }});
    menus.push_back({"Render", {
        chk("Lighting",      A::ToggleLighting, "`"),
        chk("VSync",         A::ToggleVsync),
        chk("Cross-Section", A::ToggleCrossSection),
        sep(),
        chk("Color: Original", A::ColorOriginal),
        chk("Color: Height",   A::ColorHeight),
        chk("Color: Position", A::ColorPosition),
        chk("Color: Grayscale",A::ColorGray),
    }});
    menus.push_back({"Draw", {
        act("Hidden (None)",       A::DrawNone),
        act("Solid",               A::DrawSolid),
        act("Wireframe",           A::DrawWireframe),
        act("Wireframe + Solid",   A::DrawWireSolid),
        act("Point",               A::DrawPoint),
        sep(),
        act("Point Size +",        A::PointSizeUp, "+"),
        act("Point Size -",        A::PointSizeDown, "-"),
    }});
    menus.push_back({"Select", {
        act("Select All On-Screen", A::SelectAll, "Ctrl+A"),
        act("Clear Selection",      A::ClearSelection),
        act("Delete Selected",      A::DeleteSelected, "Del"),
        act("Unhide All",           A::UnhideAll, "H"),
    }});
    menus.push_back({"Points", {
        act("Mode: Clustering",  A::Mode0),
        act("Mode: Morphology",  A::Mode1),
        act("Mode: SDF Filter",  A::Mode2),
        act("Mode: Reconstruct", A::Mode3),
    }});
    menus.push_back({"Spatial", {
        chk("Off",           A::SpatialNone),
        chk("BVH",           A::SpatialBVH),
        chk("Octree",        A::SpatialOctree),
        chk("KD-Tree",       A::SpatialKDTree),
        chk("Uniform Grid",  A::SpatialGrid),
        chk("Loose Octree",  A::SpatialLoose),
        chk("BSP",           A::SpatialBSP),
        chk("R-Tree",        A::SpatialRTree),
        chk("Ball Tree",     A::SpatialBall),
    }});
    return menus;
}

std::vector<ToolbarButton> defaultSelectionToolbar() {
    return {
        {"Obj", 0, static_cast<int>(SelTarget::Object)},
        {"Vtx", 0, static_cast<int>(SelTarget::Vertex)},
        {"Edg", 0, static_cast<int>(SelTarget::Edge)},
        {"Fac", 0, static_cast<int>(SelTarget::Face)},
        {"Sgl", 1, static_cast<int>(SelAction::Single)},
        {"Box", 1, static_cast<int>(SelAction::Box)},
        {"Las", 1, static_cast<int>(SelAction::Lasso)},
        {"Pnt", 1, static_cast<int>(SelAction::Paint)},
        {"All", 2, static_cast<int>(SelFilter::All)},
        {"Msh", 2, static_cast<int>(SelFilter::Mesh)},
        {"Pts", 2, static_cast<int>(SelFilter::Point)},
        {"Set", 3, static_cast<int>(SelModifier::Replace)},
        {"+",   3, static_cast<int>(SelModifier::Add)},
        {"-",   3, static_cast<int>(SelModifier::Subtract)},
    };
}

void selectionToolbarInputSystem(entt::registry& world, core::Input& input,
                                 uint32_t viewportW, uint32_t viewportH) {
    (void)viewportW; (void)viewportH;
    SelectionToolbar* tb = nullptr;
    auto view = world.view<SelectionToolbar>();
    for (auto e : view) { tb = &view.get<SelectionToolbar>(e); break; }
    if (!tb || !tb->visible) return;

    auto& ctx = world.ctx();
    if (!ctx.contains<SelectionMode>()) ctx.emplace<SelectionMode>();
    auto& sm = ctx.get<SelectionMode>();

    float mx = input.mousePosX, my = input.mousePosY;
    tb->hover = -1;
    for (int i = 0; i < static_cast<int>(tb->buttons.size()); ++i) {
        float y0 = toolbarButtonTop(tb->buttons, i), y1 = y0 + kTbBtn;
        float x0 = kTbX, x1 = kTbX + kTbBtn;
        if (mx >= x0 && mx < x1 && my >= y0 && my < y1) {
            tb->hover = i;
            input.captured = true;  // bar owns its pixels
            if (input.leftClicked) {
                const ToolbarButton& b = tb->buttons[i];
                switch (b.group) {
                    case 0: sm.target   = static_cast<SelTarget>(b.value);   break;
                    case 1: sm.action   = static_cast<SelAction>(b.value);   break;
                    case 2: sm.filter   = static_cast<SelFilter>(b.value);   break;
                    case 3: sm.modifier = static_cast<SelModifier>(b.value); break;
                }
            }
            break;
        }
    }
}

bool entityVisibleOnScreen(entt::registry& world, entt::entity e, uint32_t viewportW,
                           uint32_t viewportH) {
    if (!world.all_of<Transform, Renderable>(e) || viewportW == 0 || viewportH == 0)
        return false;
    // Not drawn (hidden or None draw mode) counts as not visible.
    const auto& rr = world.get<Renderable>(e);
    if (!rr.visible || rr.mesh == render::kInvalidMesh || rr.drawMode == core::DrawMode::None)
        return false;

    const Transform* camT = nullptr;
    const Camera*    cam  = nullptr;
    auto cams = world.view<Transform, Camera>();
    for (auto ce : cams) {
        if (cams.get<Camera>(ce).primary) {
            camT = &cams.get<Transform>(ce);
            cam  = &cams.get<Camera>(ce);
            break;
        }
    }
    if (!camT || !cam) return false;

    float aspect = static_cast<float>(viewportW) / static_cast<float>(viewportH);
    Eigen::Matrix4f proj =
        cam->mode == ProjectionMode::Orthographic
            ? math::ortho(-aspect * cam->orthoSize, aspect * cam->orthoSize, -cam->orthoSize,
                          cam->orthoSize, cam->zNear, cam->zFar)
            : math::perspective(cam->fovYDegrees * 3.14159265f / 180.0f, aspect, cam->zNear,
                                cam->zFar);
    Eigen::Vector3f fwd = math::rotate(camT->orientation, Eigen::Vector3f(0, 0, -1));
    Eigen::Vector3f up  = math::rotate(camT->orientation, Eigen::Vector3f(0, 1, 0));
    Eigen::Matrix4f view = math::lookAt(camT->position, camT->position + fwd, up);

    // Same transform chain as the renderer: proj * view * Mworld * (T*R*S).
    Eigen::Matrix4f clip = proj * view * worldUpMatrix(worldZUp(world)) *
                           world.get<Transform>(e).matrix();
    const auto& r = world.get<Renderable>(e);
    const Eigen::Vector3f& mn = r.boundsMin;
    const Eigen::Vector3f& mx = r.boundsMax;
    for (int i = 0; i < 8; ++i) {
        Eigen::Vector4f corner((i & 1) ? mx.x() : mn.x(), (i & 2) ? mx.y() : mn.y(),
                               (i & 4) ? mx.z() : mn.z(), 1.0f);
        Eigen::Vector4f c = clip * corner;
        if (c.w() <= 1e-6f) continue;  // behind the camera
        float x = c.x() / c.w(), y = c.y() / c.w(), z = c.z() / c.w();
        if (x >= -1.0f && x <= 1.0f && y >= -1.0f && y <= 1.0f && z >= -1.0f && z <= 1.0f)
            return true;
    }
    return false;
}

// Point-in-polygon test (even-odd rule) for the lasso, in pixel space.
static bool pointInPolygon(const std::vector<float>& xs, const std::vector<float>& ys,
                           float px, float py) {
    bool in = false;
    size_t n = xs.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        if (((ys[i] > py) != (ys[j] > py)) &&
            (px < (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i] + 1e-9f) + xs[i]))
            in = !in;
    }
    return in;
}

void pickingSystem(entt::registry& world, const core::Input& input,
                   uint32_t viewportW, uint32_t viewportH) {
    const Transform* camT = nullptr;
    const Camera*    cam  = nullptr;
    auto cams = world.view<Transform, Camera>();
    for (auto e : cams) {
        if (cams.get<Camera>(e).primary) {
            camT = &cams.get<Transform>(e);
            cam  = &cams.get<Camera>(e);
            break;
        }
    }
    if (!camT || !cam) return;

    float W = static_cast<float>(viewportW), H = static_cast<float>(viewportH);
    if (W <= 0.0f || H <= 0.0f) return;
    float mx = input.mousePosX, my = input.mousePosY;
    float ndcX = 2.0f * mx / W - 1.0f;
    float ndcY = 1.0f - 2.0f * my / H;
    float aspect = W / H;

    Eigen::Vector3f right   = math::rotate(camT->orientation, Eigen::Vector3f(1, 0, 0));
    Eigen::Vector3f up      = math::rotate(camT->orientation, Eigen::Vector3f(0, 1, 0));
    Eigen::Vector3f forward = math::rotate(camT->orientation, Eigen::Vector3f(0, 0, -1));

    Eigen::Vector3f rayO, rayD;
    if (cam->mode == ProjectionMode::Orthographic) {
        float s = cam->orthoSize;
        rayO = camT->position + right * (ndcX * aspect * s) + up * (ndcY * s);
        rayD = forward;
    } else {
        float tanHalf = std::tan(cam->fovYDegrees * 3.14159265f / 180.0f * 0.5f);
        rayO = camT->position;
        rayD = math::normalize(forward + right * (ndcX * aspect * tanHalf) +
                               up * (ndcY * tanHalf));
    }
    Eigen::Vector3f rayOW = rayO, rayDW = rayD;
    Eigen::Quaternionf mwInv = math::conjugate(worldUpQuat(worldZUp(world)));
    rayO = math::rotate(mwInv, rayO);
    rayD = math::rotate(mwInv, rayD);
    const Eigen::Matrix4f Mworld = worldUpMatrix(worldZUp(world));

    // Projection used for screen-space region tests (Box/Lasso/Paint).
    Eigen::Matrix4f proj = cam->mode == ProjectionMode::Orthographic
        ? math::ortho(-aspect * cam->orthoSize, aspect * cam->orthoSize, -cam->orthoSize,
                      cam->orthoSize, cam->zNear, cam->zFar)
        : math::perspective(cam->fovYDegrees * 3.14159265f / 180.0f, aspect, cam->zNear, cam->zFar);
    Eigen::Matrix4f viewM = math::lookAt(camT->position, camT->position + forward, up);
    Eigen::Matrix4f viewProj = proj * viewM * Mworld;

    auto& ctx = world.ctx();
    if (!ctx.contains<SelectionMode>()) ctx.emplace<SelectionMode>();
    auto& sm = ctx.get<SelectionMode>();

    auto drawables = world.view<Transform, Renderable>();
    auto eligible = [&](const Renderable& r) {
        if (!r.visible || r.mesh == render::kInvalidMesh || r.drawMode == core::DrawMode::None)
            return false;
        if (sm.filter == SelFilter::Mesh && r.pointCloud)  return false;
        if (sm.filter == SelFilter::Point && !r.pointCloud) return false;
        return true;
    };
    // Project a local point (of entity with model `m`) to window pixels.
    auto project = [&](const Eigen::Matrix4f& m, const Eigen::Vector3f& p, float& sx,
                       float& sy) -> bool {
        Eigen::Vector4f c = viewProj * m * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
        if (c.w() <= 1e-6f) return false;
        sx = (c.x() / c.w() * 0.5f + 0.5f) * W;
        sy = (1.0f - (c.y() / c.w() * 0.5f + 0.5f)) * H;
        return true;
    };
    auto clearAll = [&]() {
        for (auto e : drawables) drawables.get<Renderable>(e).selected = false;
        auto esv = world.view<ElementSelection>();
        for (auto e : esv) {
            auto& es = esv.get<ElementSelection>(e);
            es.vertices.clear(); es.faces.clear(); es.edges.clear();
        }
    };
    auto toggleV = [&](ElementSelection& es, uint32_t v, bool sub) {
        auto it = std::find(es.vertices.begin(), es.vertices.end(), v);
        if (sub) { if (it != es.vertices.end()) es.vertices.erase(it); }
        else if (it == es.vertices.end()) es.vertices.push_back(v);
    };
    auto toggleF = [&](ElementSelection& es, uint32_t fidx, bool sub) {
        auto it = std::find(es.faces.begin(), es.faces.end(), fidx);
        if (sub) { if (it != es.faces.end()) es.faces.erase(it); }
        else if (it == es.faces.end()) es.faces.push_back(fidx);
    };
    auto toggleE = [&](ElementSelection& es, uint32_t a, uint32_t b, bool sub) {
        uint32_t lo = std::min(a, b), hi = std::max(a, b);
        auto same = [&](const ElementSelection::Edge& e) {
            return std::min(e.a, e.b) == lo && std::max(e.a, e.b) == hi;
        };
        auto it = std::find_if(es.edges.begin(), es.edges.end(), same);
        if (sub) { if (it != es.edges.end()) es.edges.erase(it); }
        else if (it == es.edges.end()) es.edges.push_back({a, b});
    };

    // Keyboard modifiers override the toolbar modifier for this action: Shift adds,
    // Alt subtracts, and (in a region action, where Ctrl isn't the camera recenter)
    // Ctrl subtracts. No modifier key => the toolbar's Set/Add/Subtract.
    SelModifier effMod = sm.modifier;
    if (input.shift)      effMod = SelModifier::Add;
    else if (input.alt)   effMod = SelModifier::Subtract;
    else if (input.ctrl && sm.action != SelAction::Single) effMod = SelModifier::Subtract;

    // ---- Region actions: Box / Lasso / Paint (span multiple frames) ----------
    if (sm.action != SelAction::Single) {
        // Apply the current region to the scene (used live for Paint, on release for
        // Box/Lasso). `inRegion` decides membership in window pixels.
        // Merge `add` into `dst` (uint32 index sets) honoring the modifier without an
        // O(n^2) per-item search: after a Replace, clearAll() left dst empty so the
        // non-subtract path is a plain append.
        auto mergeIdx = [](std::vector<uint32_t>& dst, const std::vector<uint32_t>& add,
                           bool sub) {
            if (add.empty()) return;
            if (sub) {
                std::unordered_set<uint32_t> rm(add.begin(), add.end());
                dst.erase(std::remove_if(dst.begin(), dst.end(),
                                         [&](uint32_t v) { return rm.count(v) > 0; }),
                          dst.end());
            } else if (dst.empty()) {
                dst = add;
            } else {
                std::unordered_set<uint32_t> have(dst.begin(), dst.end());
                for (uint32_t v : add) if (have.insert(v).second) dst.push_back(v);
            }
        };
        auto edgeKey = [](uint32_t a, uint32_t b) {
            uint64_t lo = std::min(a, b), hi = std::max(a, b);
            return (lo << 32) | hi;
        };
        auto mergeEdges = [&](std::vector<ElementSelection::Edge>& dst,
                              const std::vector<ElementSelection::Edge>& add, bool sub) {
            if (add.empty()) return;
            if (sub) {
                std::unordered_set<uint64_t> rm;
                for (auto& e : add) rm.insert(edgeKey(e.a, e.b));
                dst.erase(std::remove_if(dst.begin(), dst.end(),
                              [&](const ElementSelection::Edge& e) { return rm.count(edgeKey(e.a, e.b)) > 0; }),
                          dst.end());
            } else {
                std::unordered_set<uint64_t> have;
                for (auto& e : dst) have.insert(edgeKey(e.a, e.b));
                for (auto& e : add) if (have.insert(edgeKey(e.a, e.b)).second) dst.push_back(e);
            }
        };

        auto applyRegion = [&](int kind /*0 box,1 lasso,2 paint*/) {
            const float paintR = 16.0f;
            float bx0 = std::min(sm.dragX0, sm.dragX1), bx1 = std::max(sm.dragX0, sm.dragX1);
            float by0 = std::min(sm.dragY0, sm.dragY1), by1 = std::max(sm.dragY0, sm.dragY1);
            auto inRegion = [&](float sx, float sy) -> bool {
                if (kind == 0) return sx >= bx0 && sx <= bx1 && sy >= by0 && sy <= by1;
                if (kind == 1) return pointInPolygon(sm.lassoX, sm.lassoY, sx, sy);
                float dx = sx - mx, dy = sy - my; return dx * dx + dy * dy <= paintR * paintR;
            };
            if (effMod == SelModifier::Replace) clearAll();
            bool sub = effMod == SelModifier::Subtract;

            for (auto e : drawables) {
                const auto& t = drawables.get<Transform>(e);
                auto& r = drawables.get<Renderable>(e);
                if (!eligible(r)) continue;
                Eigen::Matrix4f m = t.matrix();
                if (sm.target == SelTarget::Object) {
                    Eigen::Vector3f ctr = (r.boundsMin + r.boundsMax) * 0.5f;
                    float sx, sy;
                    if (project(m, ctr, sx, sy) && inRegion(sx, sy))
                        r.selected = !sub;
                    continue;
                }
                const PickGeometry* pg = world.try_get<PickGeometry>(e);
                if (!pg) continue;
                auto& es = world.get_or_emplace<ElementSelection>(e);

                // Combined clip matrix (proj*view*Mworld*model) as scalars: avoids an
                // Eigen Vector4 temporary per point in the hot loops below.
                Eigen::Matrix4f MM = viewProj * m;
                const float* A = MM.data();
                auto projPx = [&](float x, float y, float z, float& sx, float& sy) -> bool {
                    float cx = A[0]*x + A[4]*y + A[8]*z + A[12];
                    float cy = A[1]*x + A[5]*y + A[9]*z + A[13];
                    float cw = A[3]*x + A[7]*y + A[11]*z + A[15];
                    if (cw <= 1e-6f) return false;
                    sx = (cx / cw * 0.5f + 0.5f) * W;
                    sy = (1.0f - (cy / cw * 0.5f + 0.5f)) * H;
                    return true;
                };

                if (sm.target == SelTarget::Vertex) {
                    // Project + region-test every point in parallel chunks (point
                    // clouds can hold millions), then bulk-merge the matches.
                    const auto& P = pg->positions;
                    const size_t n = P.size();
                    const size_t kChunk = 16384;
                    const size_t chunks = (n + kChunk - 1) / kChunk;
                    std::vector<std::vector<uint32_t>> partial(chunks);
                    std::vector<size_t> ids(chunks);
                    for (size_t c = 0; c < chunks; ++c) ids[c] = c;
                    std::for_each(std::execution::par, ids.begin(), ids.end(), [&](size_t c) {
                        size_t b = c * kChunk, en = std::min(b + kChunk, n);
                        auto& outv = partial[c];
                        for (size_t i = b; i < en; ++i) {
                            float sx, sy;
                            if (projPx(P[i].x(), P[i].y(), P[i].z(), sx, sy) && inRegion(sx, sy))
                                outv.push_back(static_cast<uint32_t>(i));
                        }
                    });
                    std::vector<uint32_t> matched;
                    for (auto& p : partial) matched.insert(matched.end(), p.begin(), p.end());
                    mergeIdx(es.vertices, matched, sub);
                } else if (sm.target == SelTarget::Face) {
                    std::vector<uint32_t> matched;
                    for (size_t i = 0; i + 2 < pg->indices.size(); i += 3) {
                        const auto& a = pg->positions[pg->indices[i]];
                        const auto& b = pg->positions[pg->indices[i + 1]];
                        const auto& c = pg->positions[pg->indices[i + 2]];
                        Eigen::Vector3f ctr = (a + b + c) / 3.0f;
                        float sx, sy;
                        if (projPx(ctr.x(), ctr.y(), ctr.z(), sx, sy) && inRegion(sx, sy))
                            matched.push_back(static_cast<uint32_t>(i / 3));
                    }
                    mergeIdx(es.faces, matched, sub);
                } else {  // Edge
                    std::vector<ElementSelection::Edge> matched;
                    for (size_t i = 0; i + 2 < pg->indices.size(); i += 3) {
                        uint32_t idx[3] = {pg->indices[i], pg->indices[i + 1], pg->indices[i + 2]};
                        for (int k = 0; k < 3; ++k) {
                            Eigen::Vector3f mid =
                                (pg->positions[idx[k]] + pg->positions[idx[(k + 1) % 3]]) * 0.5f;
                            float sx, sy;
                            if (projPx(mid.x(), mid.y(), mid.z(), sx, sy) && inRegion(sx, sy))
                                matched.push_back({idx[k], idx[(k + 1) % 3]});
                        }
                    }
                    mergeEdges(es.edges, matched, sub);
                }
            }
        };

        if (input.leftClicked && !input.captured) {
            sm.dragging = true;
            sm.dragX0 = sm.dragX1 = mx; sm.dragY0 = sm.dragY1 = my;
            sm.lassoX.assign(1, mx); sm.lassoY.assign(1, my);
            if (sm.action == SelAction::Paint) applyRegion(2);
        } else if (sm.dragging && input.buttonLeft) {
            sm.dragX1 = mx; sm.dragY1 = my;
            if (sm.action == SelAction::Lasso) {
                float lx = sm.lassoX.back(), ly = sm.lassoY.back();
                if ((mx - lx) * (mx - lx) + (my - ly) * (my - ly) > 9.0f) {
                    sm.lassoX.push_back(mx); sm.lassoY.push_back(my);
                }
            }
            if (sm.action == SelAction::Paint) applyRegion(2);
        }
        if (sm.dragging && input.leftReleased) {
            if (sm.action == SelAction::Box)   applyRegion(0);
            else if (sm.action == SelAction::Lasso) applyRegion(1);
            sm.dragging = false;
            sm.lassoX.clear(); sm.lassoY.clear();
        }
        return;
    }

    // ---- Single click / Ctrl-recenter ---------------------------------------
    if (!input.leftClicked || input.captured) return;

    // Nearest eligible drawable along the ray (front-most under the cursor).
    entt::entity best = entt::null;
    float bestT = 1e30f;
    for (auto e : drawables) {
        const auto& t = drawables.get<Transform>(e);
        const auto& r = drawables.get<Renderable>(e);
        if (!eligible(r)) continue;

        Eigen::Vector3f lo = math::rotate(math::conjugate(t.orientation), rayO - t.position);
        Eigen::Vector3f ld = math::rotate(math::conjugate(t.orientation), rayD);
        Eigen::Vector3f inv(t.scale.x() != 0 ? 1.0f / t.scale.x() : 0.0f,
                       t.scale.y() != 0 ? 1.0f / t.scale.y() : 0.0f,
                       t.scale.z() != 0 ? 1.0f / t.scale.z() : 0.0f);
        lo = lo.cwiseProduct(inv);
        ld = ld.cwiseProduct(inv);

        if (const PickGeometry* pg = world.try_get<PickGeometry>(e); pg && r.pointCloud) {
            Eigen::Vector3f margin =
                (r.boundsMax - r.boundsMin).cwiseMax(Eigen::Vector3f(1, 1, 1)) * 0.05f;
            float tAabb;
            if (!intersectAABB(lo, ld, r.boundsMin - margin, r.boundsMax + margin, tAabb))
                continue;
            const Eigen::Matrix4f toWorld = Mworld * t.matrix();
            const float* M = toWorld.data();
            const float ox = rayOW.x(), oy = rayOW.y(), oz = rayOW.z();
            const float dx = rayDW.x(), dy = rayDW.y(), dz = rayDW.z();
            const float pixelRadius = 6.0f;
            const bool  ortho = (cam->mode == ProjectionMode::Orthographic);
            const float kAng  = ortho ? 0.0f
                : 2.0f * std::tan(cam->fovYDegrees * 3.14159265f / 180.0f * 0.5f) / H * pixelRadius;
            const float orthoThr = ortho ? (2.0f * cam->orthoSize / H * pixelRadius) : 0.0f;
            const auto& P = pg->positions;
            const size_t n = P.size();
            const size_t kChunk = 16384;
            const size_t chunks = (n + kChunk - 1) / kChunk;
            const float upper = bestT;
            std::vector<float> chunkMin(chunks, upper);
            std::vector<size_t> chunkIds(chunks);
            for (size_t c = 0; c < chunks; ++c) chunkIds[c] = c;
            std::for_each(std::execution::par, chunkIds.begin(), chunkIds.end(), [&](size_t c) {
                const size_t begin = c * kChunk;
                const size_t end = std::min(begin + kChunk, n);
                float localMin = upper;
                for (size_t i = begin; i < end; ++i) {
                    const float x = P[i].x(), y = P[i].y(), z = P[i].z();
                    const float wx = M[0]*x + M[4]*y + M[8]*z + M[12];
                    const float wy = M[1]*x + M[5]*y + M[9]*z + M[13];
                    const float wz = M[2]*x + M[6]*y + M[10]*z + M[14];
                    const float rx = wx - ox, ry = wy - oy, rz = wz - oz;
                    const float tw = rx*dx + ry*dy + rz*dz;
                    if (tw <= 0.0f || tw >= localMin) continue;
                    const float perp2 = (rx*rx + ry*ry + rz*rz) - tw*tw;
                    const float thr = ortho ? orthoThr : kAng * tw;
                    if (perp2 < thr * thr) localMin = tw;
                }
                chunkMin[c] = localMin;
            });
            float entMin = upper;
            for (float v : chunkMin) entMin = std::min(entMin, v);
            if (entMin < bestT) { bestT = entMin; best = e; }
            continue;
        }

        float tLocal = 0.0f;
        bool hit = false;
        if (const PickGeometry* pg = world.try_get<PickGeometry>(e); pg && !pg->indices.empty()) {
            // Accurate triangle pick accelerated by a lazily-built, cached BVH.
            auto& accel = world.get_or_emplace<PickBVH>(e);
            if (!accel.built) { accel.bvh.build(pg->positions, pg->indices); accel.built = true; }
            float tl; int tri;
            if (accel.bvh.nearestHit(geometry::Ray(lo, ld), tl, tri)) { tLocal = tl; hit = true; }
        } else {
            hit = intersectAABB(lo, ld, r.boundsMin, r.boundsMax, tLocal);
        }
        if (!hit) continue;
        Eigen::Vector3f localHit = lo + ld * tLocal;
        Eigen::Vector4f wh =
            Mworld * t.matrix() * Eigen::Vector4f(localHit.x(), localHit.y(), localHit.z(), 1.0f);
        float tWorld = (Eigen::Vector3f(wh.x(), wh.y(), wh.z()) - rayOW).dot(rayDW);
        if (tWorld > 0.0f && tWorld < bestT) { bestT = tWorld; best = e; }
    }

    if (input.ctrl) {
        // Ctrl+left-click recenters the camera on the picked point (selection kept).
        if (best != entt::null) {
            const Eigen::Vector3f hit = rayOW + rayDW * bestT;
            auto manips = world.view<Camera, CameraManipulator>();
            for (auto e : manips) {
                if (!manips.get<Camera>(e).primary) continue;
                auto& m = manips.get<CameraManipulator>(e);
                m.targetFrom = m.target; m.targetTo = hit;
                m.distFrom = m.distTo = m.distance;
                m.targetAnimTime = 0.0f; m.targetAnimating = true;
                break;
            }
        }
        return;
    }

    bool sub = effMod == SelModifier::Subtract;
    if (effMod == SelModifier::Replace) clearAll();

    if (sm.target == SelTarget::Object) {
        if (best != entt::null) drawables.get<Renderable>(best).selected = !sub;
        return;
    }
    // Element single-pick on the front-most entity under the cursor.
    if (best == entt::null) return;
    const PickGeometry* pg = world.try_get<PickGeometry>(best);
    if (!pg || pg->positions.empty()) return;
    const auto& t = drawables.get<Transform>(best);
    auto& es = world.get_or_emplace<ElementSelection>(best);
    Eigen::Matrix4f m = t.matrix();

    if (sm.target == SelTarget::Vertex) {
        uint32_t bestV = 0; float bestD = 1e30f; bool found = false;
        for (uint32_t vi = 0; vi < pg->positions.size(); ++vi) {
            float sx, sy;
            if (!project(m, pg->positions[vi], sx, sy)) continue;
            float d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
            if (d < bestD) { bestD = d; bestV = vi; found = true; }
        }
        if (found && bestD <= 14.0f * 14.0f) toggleV(es, bestV, sub);
    } else {
        // Face/Edge: ray-hit the triangle (local space), then pick that face or its
        // nearest edge to the cursor.
        Eigen::Vector3f lo = math::rotate(math::conjugate(t.orientation), rayO - t.position);
        Eigen::Vector3f ld = math::rotate(math::conjugate(t.orientation), rayD);
        Eigen::Vector3f inv(t.scale.x() != 0 ? 1.0f / t.scale.x() : 0.0f,
                       t.scale.y() != 0 ? 1.0f / t.scale.y() : 0.0f,
                       t.scale.z() != 0 ? 1.0f / t.scale.z() : 0.0f);
        lo = lo.cwiseProduct(inv); ld = ld.cwiseProduct(inv);
        const size_t kNoFace = static_cast<size_t>(-1);
        size_t hitFace = kNoFace;
        auto& accel = world.get_or_emplace<PickBVH>(best);
        if (!accel.built) { accel.bvh.build(pg->positions, pg->indices); accel.built = true; }
        float tl; int tri;
        if (accel.bvh.nearestHit(geometry::Ray(lo, ld), tl, tri))
            hitFace = static_cast<size_t>(tri) * 3;
        if (hitFace == kNoFace) return;
        if (sm.target == SelTarget::Face) {
            toggleF(es, static_cast<uint32_t>(hitFace / 3), sub);
        } else {  // Edge: nearest of the hit triangle's 3 edges to the cursor
            uint32_t idx[3] = {pg->indices[hitFace], pg->indices[hitFace + 1],
                               pg->indices[hitFace + 2]};
            int bestE = 0; float bestD = 1e30f;
            for (int k = 0; k < 3; ++k) {
                Eigen::Vector3f mid = (pg->positions[idx[k]] + pg->positions[idx[(k+1)%3]]) * 0.5f;
                float sx, sy;
                if (!project(m, mid, sx, sy)) continue;
                float d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
                if (d < bestD) { bestD = d; bestE = k; }
            }
            toggleE(es, idx[bestE], idx[(bestE + 1) % 3], sub);
        }
    }
}

void axisGizmoInputSystem(entt::registry& world, core::Input& input,
                          float dt, uint32_t viewportW, uint32_t viewportH) {
    AxisGizmo* gizmo = nullptr;
    auto gz = world.view<AxisGizmo>();
    for (auto e : gz) { gizmo = &gz.get<AxisGizmo>(e); break; }
    if (!gizmo) return;

    if (gizmo->flash > 0.0f) gizmo->flash -= dt;  // decay click feedback

    // Find the primary camera manipulator (its orientation drives the gizmo).
    CameraManipulator* manip = nullptr;
    auto cams = world.view<Camera, CameraManipulator>();
    for (auto e : cams) {
        if (cams.get<Camera>(e).primary) { manip = &cams.get<CameraManipulator>(e); break; }
    }
    if (!manip) return;

    GizmoRect r = gizmoRect(*gizmo, viewportW, viewportH);

    // Up-axis toggle button (bottom-left corner): flip Y/Z up. This re-expresses
    // the whole world (content + gizmo) in renderSystem via the up-axis basis;
    // the camera and the horizontal ground stay put. Takes priority over the
    // cube/ring beneath it.
    GizmoRect b = upToggleRect(r);
    float mx = input.mousePosX, my = input.mousePosY;
    gizmo->upBtnHover = mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h;
    if (gizmo->upBtnHover) {
        gizmo->hoverPart = GizmoPart::None;
        input.captured   = true;  // suppress orbit/picking over the button
        if (input.leftClicked) gizmo->zUp = !gizmo->zUp;
        return;
    }

    // Hover: cube first (it sits in front of the ring), then the ring.
    Eigen::Vector3f dir;
    if (pickGizmo(manip->orientation, r, input.mousePosX, input.mousePosY, gizmo->zUp, dir)) {
        gizmo->hoverPart = GizmoPart::Cube;
        gizmo->hoverDir  = dir;
    } else {
        int sec = pickRing(r, input.mousePosX, input.mousePosY);
        if (sec >= 0) {
            gizmo->hoverPart   = GizmoPart::Ring;
            gizmo->hoverSector = sec;
        } else {
            gizmo->hoverPart = GizmoPart::None;
        }
    }

    // Click: snap the camera and flash the region.
    if (input.leftClicked && gizmo->hoverPart != GizmoPart::None) {
        Eigen::Quaternionf target;
        if (gizmo->hoverPart == GizmoPart::Cube)
            // hoverDir is a logical axis; map it through the up-axis basis to the
            // render direction the camera should look down.
            target = math::quatLookZ(math::rotate(worldUpQuat(gizmo->zUp), gizmo->hoverDir));
        else
            target = math::normalize(manip->orientation * ringDelta(gizmo->hoverSector));

        manip->animFrom  = manip->orientation;
        manip->animTo    = target;
        manip->animTime  = 0.0f;
        manip->animating = true;

        gizmo->flash       = manip->animDuration;
        gizmo->flashPart   = gizmo->hoverPart;
        gizmo->flashDir    = gizmo->hoverDir;
        gizmo->flashSector = gizmo->hoverSector;
    }
}

void fpsWidgetInputSystem(entt::registry& world, core::Input& input, float dt,
                          uint32_t viewportW, uint32_t viewportH) {
    auto view = world.view<FpsWidget>();
    for (auto e : view) {
        auto& wgt = view.get<FpsWidget>(e);

        // Accumulate frames and only push a graph sample / refresh the readout
        // once per updateInterval, so the number and graph advance at a calm,
        // readable rate instead of flickering every frame.
        wgt.accumTime += dt;
        wgt.accumFrames += 1;
        if (wgt.accumTime >= wgt.updateInterval) {
            float fps = wgt.accumFrames / wgt.accumTime;   // avg over the interval
            if (!wgt.primed) {                             // avoid an empty graph at start
                for (int i = 0; i < FpsWidget::kSamples; ++i) wgt.history[i] = fps;
                wgt.primed = true;
            }
            wgt.history[wgt.head] = fps;
            wgt.head = (wgt.head + 1) % FpsWidget::kSamples;
            wgt.smoothFps = fps;
            wgt.accumTime = 0.0f;
            wgt.accumFrames = 0;
        }

        // Place the panel from its viewport-relative anchor, so it tracks window
        // resizes; clamp so it stays fully on-screen.
        float maxX = static_cast<float>(viewportW) - wgt.w;
        float maxY = static_cast<float>(viewportH) - wgt.h;
        wgt.x = static_cast<int>(clampf(wgt.relX * viewportW, 0.0f, (std::max)(0.0f, maxX)));
        wgt.y = static_cast<int>(clampf(wgt.relY * viewportH, 0.0f, (std::max)(0.0f, maxY)));

        float mx = input.mousePosX, my = input.mousePosY;
        bool inside = mx >= wgt.x && mx <= wgt.x + wgt.w && my >= wgt.y &&
                      my <= wgt.y + wgt.h;

        // VSYNC checkbox click takes priority over starting a drag, and consumes
        // the click so it neither drags the panel nor picks an entity.
        CbRect cb = fpsCheckboxRect(wgt);
        bool onCheckbox = mx >= cb.x0 && mx <= cb.x1 && my >= cb.y0 && my <= cb.y1;
        if (input.leftClicked && onCheckbox) {
            wgt.vsync      = !wgt.vsync;
            wgt.vsyncDirty = true;
            input.captured = true;
        } else if (input.leftClicked && inside) {
            wgt.dragging = true;
            wgt.dragOffX = mx - wgt.x;
            wgt.dragOffY = my - wgt.y;
        }
        if (wgt.dragging) {
            if (input.buttonLeft) {
                wgt.x = static_cast<int>(clampf(mx - wgt.dragOffX, 0.0f, (std::max)(0.0f, maxX)));
                wgt.y = static_cast<int>(clampf(my - wgt.dragOffY, 0.0f, (std::max)(0.0f, maxY)));
                // Store the new spot as a viewport fraction (resize-stable).
                wgt.relX = viewportW > 0 ? static_cast<float>(wgt.x) / viewportW : 0.0f;
                wgt.relY = viewportH > 0 ? static_cast<float>(wgt.y) / viewportH : 0.0f;
                input.captured = true;  // suppress camera orbit while dragging
            } else {
                wgt.dragging = false;
            }
        }
    }
}

void cameraControlsInputSystem(entt::registry& world, core::Input& input, float dt,
                               uint32_t viewportW, uint32_t viewportH) {
    CameraControls* cc = nullptr;
    auto view = world.view<CameraControls>();
    for (auto e : view) { cc = &view.get<CameraControls>(e); break; }
    if (!cc) return;

    // Position under the gizmo (top-right). Gizmo: size 150, margin 14, itself
    // pushed down by the menu bar.
    cc->x = static_cast<int>(viewportW) - cc->w - 14;
    cc->y = kMenuBarHeight + 14 + 150 + 10;
    if (cc->x < 0) cc->x = 0;

    Camera* cam = nullptr;
    auto cams = world.view<Camera>();
    for (auto e : cams) { if (cams.get<Camera>(e).primary) { cam = &cams.get<Camera>(e); break; } }
    if (!cam) return;

    float mx = input.mousePosX, my = input.mousePosY;
    auto hit = [&](const CRect& r) {
        return mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h;
    };
    CtrlRects R = controlRects(*cc);
    bool persp = cam->mode == ProjectionMode::Perspective;

    // Projection toggle: a plain single click.
    if (input.leftClicked && hit(R.mode)) {
        cam->mode = persp ? ProjectionMode::Orthographic : ProjectionMode::Perspective;
        input.captured = true;
        cc->holdTime = 0.0f;
        return;
    }

    // +/- buttons: hold to change continuously (small steps). Shift snaps to a
    // coarse grid with key-repeat. Direction = whichever button is under the
    // cursor while the left button is held.
    int dir = 0;
    if (input.buttonLeft) {
        if (hit(R.minus)) dir = -1; else if (hit(R.plus)) dir = +1;
    }
    if (dir == 0) { cc->holdTime = 0.0f; return; }
    input.captured = true;
    bool edge = input.leftClicked;  // first frame of this press

    auto setVal = [&](float v) {
        if (persp) cam->fovYDegrees = clampf(v, 10.0f, 120.0f);
        else       cam->orthoSize   = clampf(v, 0.5f, 30.0f);
    };
    float cur = persp ? cam->fovYDegrees : cam->orthoSize;

    if (input.shift) {
        // Snap to the next multiple of `step` in `dir`, repeating while held.
        float step = persp ? 5.0f : 0.5f;
        bool fire = edge;
        if (edge) {
            cc->holdTime = 0.0f;
        } else {
            float before = cc->holdTime;
            cc->holdTime += dt;
            auto ticks = [](float t) {
                const float delay = 0.35f, rep = 0.11f;  // initial delay, then repeat
                return t < delay ? 0 : 1 + static_cast<int>((t - delay) / rep);
            };
            fire = ticks(cc->holdTime) > ticks(before);
        }
        if (fire) {
            float g = cur / step;
            setVal((dir > 0 ? std::floor(g + 1.0f) : std::ceil(g - 1.0f)) * step);
        }
    } else {
        // Fine continuous change: a quick tap nudges by `kick`, holding ramps at
        // `rate` per second.
        float rate = persp ? 16.0f : 1.6f;   // units / second while held
        float kick = persp ? 1.0f  : 0.1f;   // one tap
        if (edge) { cc->holdTime = 0.0f; setVal(cur + dir * kick); }
        else      { cc->holdTime += dt;   setVal(cur + dir * rate * dt); }
    }
}

void crossSectionInputSystem(entt::registry& world, core::Input& input,
                             uint32_t viewportW, uint32_t viewportH) {
    (void)viewportH;
    CrossSection* cs = nullptr;
    auto view = world.view<CrossSection>();
    for (auto e : view) { cs = &view.get<CrossSection>(e); break; }
    if (!cs) return;

    // Position top-right, directly under the camera-controls panel.
    int baseY = kMenuBarHeight + 14 + 150 + 10 + 76 + 10;
    auto ccv = world.view<CameraControls>();
    for (auto e : ccv) { const auto& cc = ccv.get<CameraControls>(e); baseY = cc.y + cc.h + 10; break; }
    cs->x = static_cast<int>(viewportW) - cs->w - 14;
    cs->y = baseY;
    if (cs->x < 0) cs->x = 0;

    float mx = input.mousePosX, my = input.mousePosY;
    auto  hit = [&](const CRect& r) {
        return mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h;
    };
    CsRects R = crossSectionRects(*cs);
    bool inPanel = mx >= cs->x && mx <= cs->x + cs->w && my >= cs->y && my <= cs->y + cs->h;

    // Clicks on the buttons (single-shot). The slider drag is handled after.
    if (input.leftClicked && hit(R.enable)) {
        cs->enabled = !cs->enabled; input.captured = true; return;
    }
    if (input.leftClicked && hit(R.axis)) {
        cs->axis = (cs->axis + 1) % 3; input.captured = true; return;
    }
    if (input.leftClicked && hit(R.flip)) {
        cs->flip = !cs->flip; input.captured = true; return;
    }

    // Slider: grab anywhere in a band around the groove, then track the cursor.
    CRect band = {R.track.x - 6.0f, cs->y + 58.0f, R.track.w + 12.0f, 30.0f};
    if (input.leftClicked && hit(band)) cs->dragging = true;
    if (!input.buttonLeft) cs->dragging = false;
    if (cs->dragging) {
        float t = R.track.w > 0.0f ? (mx - R.track.x) / R.track.w : 0.0f;
        t = clampf(t, 0.0f, 1.0f);
        cs->pos     = cs->minPos + t * (cs->maxPos - cs->minPos);
        cs->enabled = true;  // moving the section turns it on
        input.captured = true;
    }
    if (inPanel) input.captured = true;
}

void renderSystem(entt::registry& world, render::IRenderer& renderer,
                  uint32_t viewportW, uint32_t viewportH) {
    // --- Find the primary camera ---------------------------------------
    render::FrameContext frame;
    frame.width  = viewportW;
    frame.height = viewportH;

    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf camOrient = Eigen::Quaternionf::Identity();  // captured for the gizmo overlay
    const Camera* primaryCam = nullptr;  // captured for the controls panel
    Eigen::Vector3f camPos = Eigen::Vector3f::Zero();  // camera eye (grid fade center)
    float camDist = 1.0f;                              // orbit distance (grid view radius)

    auto cams = world.view<Transform, Camera>();
    for (auto entity : cams) {
        auto& cam = cams.get<Camera>(entity);
        if (!cam.primary) continue;
        const auto& ct = cams.get<Transform>(entity);
        primaryCam = &cam;
        camPos = ct.position;
        if (const auto* mm = world.try_get<CameraManipulator>(entity)) camDist = mm->distance;

        // Clip planes track the orbit distance so geometry and the (camera-faded)
        // grid never z-clip at any zoom -- a fixed far plane would cut the grid when
        // zoomed out, a fixed near plane would cut the model when zoomed in.
        cam.zNear = (std::max)(0.001f, camDist * 0.005f);
        cam.zFar  = camDist * 50.0f;

        float aspect = viewportH > 0
                           ? static_cast<float>(viewportW) / static_cast<float>(viewportH)
                           : 1.0f;
        if (cam.mode == ProjectionMode::Orthographic) {
            float s = cam.orthoSize;
            proj = math::ortho(-aspect * s, aspect * s, -s, s, cam.zNear, cam.zFar);
        } else {
            proj = math::perspective(cam.fovYDegrees * 3.14159265f / 180.0f, aspect,
                                     cam.zNear, cam.zFar);
        }
        // Derive view from the camera Transform's orientation, so any camera
        // controller (e.g. CameraManipulator) just writes the Transform.
        Eigen::Vector3f forward = math::rotate(ct.orientation, Eigen::Vector3f(0, 0, -1));
        Eigen::Vector3f up      = math::rotate(ct.orientation, Eigen::Vector3f(0, 1, 0));
        view = math::lookAt(ct.position, ct.position + forward, up);
        camOrient = ct.orientation;
        break;
    }

    std::memcpy(frame.view, view.data(), sizeof(frame.view));
    std::memcpy(frame.proj, proj.data(), sizeof(frame.proj));

    // --- Submit all drawables ------------------------------------------
    renderer.beginFrame(frame);

    // World up-axis basis: prepended to every model so the scene's coordinate
    // frame (not the camera) changes when the gizmo toggles Y/Z up.
    const Eigen::Matrix4f Mworld = worldUpMatrix(worldZUp(world));

    // Scene world-space AABB over every visible drawable (corners pushed through
    // Mworld * model). Drives the grid cell size and the cross-section slider range
    // so both scale with whatever model is loaded, not a fixed ~3-unit assumption.
    Eigen::Vector3f sceneMin = Eigen::Vector3f::Zero(), sceneMax = Eigen::Vector3f::Zero();
    bool sceneValid = false;
    {
        auto bv = world.view<Transform, Renderable>();
        for (auto e : bv) {
            const auto& t = bv.get<Transform>(e);
            const auto& r = bv.get<Renderable>(e);
            if (!r.visible || r.mesh == render::kInvalidMesh) continue;
            Eigen::Matrix4f m = Mworld * t.matrix();
            for (int c = 0; c < 8; ++c) {
                Eigen::Vector4f corner((c & 1) ? r.boundsMax.x() : r.boundsMin.x(),
                                       (c & 2) ? r.boundsMax.y() : r.boundsMin.y(),
                                       (c & 4) ? r.boundsMax.z() : r.boundsMin.z(), 1.0f);
                Eigen::Vector3f w = (m * corner).head<3>();
                if (!sceneValid) { sceneMin = sceneMax = w; sceneValid = true; }
                else { sceneMin = sceneMin.cwiseMin(w); sceneMax = sceneMax.cwiseMax(w); }
            }
        }
    }

    // Cross-section: turn (enabled, axis, flip, pos) into a render-world clip plane
    // and hand it to the backend before the scene submits. Keep the half-space with
    // coordinate <= pos (or >= pos when flipped); the opposite side is discarded.
    {
        bool  csEnabled = false;
        float plane[4]  = {0, 0, 0, 0};
        auto  csv = world.view<CrossSection>();
        for (auto e : csv) {
            auto& cs = csv.get<CrossSection>(e);
            int   a  = cs.axis < 0 ? 0 : (cs.axis > 2 ? 2 : cs.axis);
            // Fit the slider range to the model's extent on the chosen axis.
            if (sceneValid) {
                float lo  = sceneMin[a], hi = sceneMax[a];
                float pad = (hi - lo) * 0.02f + 1e-4f;
                cs.minPos = lo - pad;
                cs.maxPos = hi + pad;
                cs.pos    = clampf(cs.pos, cs.minPos, cs.maxPos);
            }
            if (cs.enabled) {
                float s = cs.flip ? -1.0f : 1.0f;
                plane[0] = plane[1] = plane[2] = 0.0f;
                plane[a] = s;                          // normal along the chosen axis
                plane[3] = cs.flip ? cs.pos : -cs.pos; // discard dot(world,n)+d > 0
                csEnabled = true;
            }
            break;
        }
        renderer.setCrossSection(csEnabled, plane);
    }

    // Each mesh carries its own drawing mode (Tab cycles it for the selection).
    // A selected mesh also gets a silhouette outline that reads as a thin border
    // in *any* draw mode: draw the mesh, then a solid stencil footprint, then an
    // enlarged copy drawn only outside that footprint (stencil != 1) -- so only a
    // thin border ring shows. (Reset to solid after the loop so debug geometry,
    // grid and overlays stay filled.)
    const uint32_t kSolid = static_cast<uint32_t>(core::DrawMode::Solid);

    auto drawables = world.view<Transform, Renderable>();
    for (auto entity : drawables) {
        const auto& t = drawables.get<Transform>(entity);
        const auto& r = drawables.get<Renderable>(entity);
        // None draw mode == fully invisible: no visible pass and no silhouette.
        if (!r.visible || r.mesh == render::kInvalidMesh ||
            r.drawMode == core::DrawMode::None)
            continue;

        Eigen::Matrix4f model = Mworld * t.matrix();

        // 1) Visible pass, in this mesh's own drawing mode.
        renderer.setDrawMode(static_cast<uint32_t>(r.drawMode));
        render::DrawItem item;
        item.mesh = r.mesh;
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);

        if (r.selected && r.pointCloud) {
            // A point cloud has no solid surface for a stencil silhouette, so mark
            // selection with a clean bounding-box wireframe instead. Compute it in
            // content space (T*R*S, *not* Mworld) -- drawDebugGeometry applies Mworld
            // itself, so folding it in here would double-apply on the Z-up toggle.
            const Eigen::Matrix4f cm = t.matrix();
            Eigen::Vector3f bmin = Eigen::Vector3f::Constant(1e30f);
            Eigen::Vector3f bmax = Eigen::Vector3f::Constant(-1e30f);
            for (int c = 0; c < 8; ++c) {
                Eigen::Vector4f corner((c & 1) ? r.boundsMax.x() : r.boundsMin.x(),
                                       (c & 2) ? r.boundsMax.y() : r.boundsMin.y(),
                                       (c & 4) ? r.boundsMax.z() : r.boundsMin.z(), 1.0f);
                Eigen::Vector4f w = cm * corner;
                bmin = bmin.cwiseMin(w.head<3>());
                bmax = bmax.cwiseMax(w.head<3>());
            }
            float thick = (bmax - bmin).norm() * 0.0015f;
            debug::DebugDraw::instance().addWireBox(bmin, bmax, {1.0f, 0.55f, 0.1f}, thick);
        } else if (r.selected) {
            // 2) Solid stencil footprint (marks where the mesh is, so the outline
            //    rings the silhouette regardless of the draw mode or mesh shape).
            render::DrawItem mask;
            mask.mesh = r.mesh;
            mask.stencilMask = true;
            std::memcpy(mask.model, model.data(), sizeof(mask.model));
            renderer.submit(mask);

            // 3) Enlarged copy in the outline color, drawn only outside the stencil
            //    footprint -> only the border shows.
            Eigen::Matrix4f hull = Mworld * math::translate(t.position) *
                                   math::toMat4(t.orientation) *
                                   math::scale(t.scale * 1.05f);
            render::DrawItem outline;
            outline.mesh = r.mesh;
            outline.outline = true;
            std::memcpy(outline.model, hull.data(), sizeof(outline.model));
            renderer.submit(outline);
        }
    }
    renderer.setDrawMode(kSolid);  // scene only: rest stays solid

    // --- Element selection highlights (vertex/edge/face) ---------------
    // Emitted into the debug-draw accumulator in content space (T*R*S); the
    // drawDebugGeometry call below applies Mworld, like the point-cloud box.
    {
        auto esv = world.view<Transform, ElementSelection, PickGeometry>();
        auto& dd = debug::DebugDraw::instance();
        const Eigen::Vector3f hl(0.15f, 1.0f, 0.25f);  // bright lime, pops on pink/white

        // Vertex markers are camera-facing quads sized a few times the point-sprite
        // pixel size, so a selected point reads clearly over the dot it covers.
        // Debug geometry is emitted in content space (Mworld applied later), so the
        // camera right/up axes are pulled back through Mworld^-1.
        float px = 6.0f;
        if (world.ctx().contains<PointSizeState>()) px = world.ctx().get<PointSizeState>().size;
        px *= 2.5f;  // markers a bit larger than the points so they stand out
        const bool  ortho = primaryCam && primaryCam->mode == ProjectionMode::Orthographic;
        const float fovY  = primaryCam ? primaryCam->fovYDegrees : 60.0f;
        const float Hpx   = static_cast<float>(viewportH > 0 ? viewportH : 1);
        const float kPersp = 2.0f * std::tan(fovY * 3.14159265f / 180.0f * 0.5f) / Hpx * px;
        const float kOrtho = primaryCam ? 2.0f * primaryCam->orthoSize / Hpx * px : 0.01f;
        const Eigen::Quaternionf wq    = worldUpQuat(worldZUp(world));
        const Eigen::Quaternionf wqInv = math::conjugate(wq);
        const Eigen::Vector3f cr = math::rotate(wqInv, math::rotate(camOrient, Eigen::Vector3f(1, 0, 0)));
        const Eigen::Vector3f cu = math::rotate(wqInv, math::rotate(camOrient, Eigen::Vector3f(0, 1, 0)));

        for (auto e : esv) {
            const auto& t  = esv.get<Transform>(e);
            const auto& es = esv.get<ElementSelection>(e);
            const auto& pg = esv.get<PickGeometry>(e);
            Eigen::Matrix4f m = t.matrix();
            auto wp = [&](const Eigen::Vector3f& p) {
                Eigen::Vector4f w = m * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
                return Eigen::Vector3f(w.x(), w.y(), w.z());
            };
            auto marker = [&](const Eigen::Vector3f& cp) {
                // Half-size in content units that projects to ~px/2 pixels.
                Eigen::Vector3f wpos = math::rotate(wq, cp);  // Mworld is pure rotation
                float dist = (wpos - camPos).norm();
                float half = 0.5f * (ortho ? kOrtho : kPersp * dist);
                dd.addQuad(cp - (cr + cu) * half, cp + (cr - cu) * half,
                           cp + (cr + cu) * half, cp + (cu - cr) * half, hl);
            };
            for (uint32_t vi : es.vertices)
                if (vi < pg.positions.size()) marker(wp(pg.positions[vi]));
            for (const auto& ed : es.edges)
                if (ed.a < pg.positions.size() && ed.b < pg.positions.size())
                    dd.addLine(wp(pg.positions[ed.a]), wp(pg.positions[ed.b]), hl,
                               (ortho ? kOrtho : kPersp * camDist) * 0.3f);
            for (uint32_t fi : es.faces) {
                size_t i = static_cast<size_t>(fi) * 3;
                if (i + 2 < pg.indices.size())
                    dd.addTriangle(wp(pg.positions[pg.indices[i]]),
                                   wp(pg.positions[pg.indices[i + 1]]),
                                   wp(pg.positions[pg.indices[i + 2]]), hl);
            }
        }
    }

    // --- Spatial-structure overlay (BVH / Octree / KD-tree) ------------
    // For each selected entity, build the chosen structure ONCE into a static GPU
    // wireframe mesh (local space, per-level colored) and draw it each frame with
    // model = Mworld * Transform. Rebuilding it per frame (the old debug-draw path)
    // made deep trees < 1 fps; this makes the per-frame cost a single draw call.
    {
        int vizKind = 0;
        if (world.ctx().contains<SpatialViz>()) vizKind = world.ctx().get<SpatialViz>().kind;
        if (vizKind > 0) {
            // BVH/Octree box count is bounded by node count, so go deep; a k-d tree
            // has one node per point (2^depth cells), so cap it shallower.
            const int kDepthTree = 18;   // BVH / Octree
            const int kDepthKD   = 12;   // KD-tree
            auto levelColor = [](int d) {
                Eigen::Vector4f c = color::FromHSV(color::fract(d * 0.16f), 0.85f, 1.0f);
                return Eigen::Vector3f(c.x(), c.y(), c.z());
            };
            auto sv = world.view<Transform, Renderable, PickGeometry>();
            for (auto e : sv) {
                const auto& r = sv.get<Renderable>(e);
                if (!r.selected) continue;
                const auto& pg = sv.get<PickGeometry>(e);
                if (pg.positions.empty()) continue;
                auto& cache = world.get_or_emplace<SpatialVizCache>(e);

                if (cache.kind != vizKind) {  // (re)build the static wireframe mesh
                    if (cache.mesh != render::kInvalidMesh) renderer.destroyMesh(cache.mesh);
                    if (cache.vbo != render::kInvalidBuffer) renderer.destroyBuffer(cache.vbo);
                    cache.mesh = render::kInvalidMesh; cache.vbo = render::kInvalidBuffer;
                    cache.vertexCount = 0; cache.kind = vizKind;

                    std::vector<geometry::AABB> boxes; std::vector<int> levels;
                    // Ball tree visualizes spheres instead of boxes.
                    std::vector<Eigen::Vector3f> sphC; std::vector<float> sphR; std::vector<int> sphL;

                    // Build on ALL points so the whole model is covered. Robust bounds
                    // are used only to size the line thickness and to cull a handful of
                    // oversized cells (outlier points blow a few cells up to span the
                    // scene); they do NOT filter the points fed to the structure.
                    const std::vector<Eigen::Vector3f>& P = pg.positions;
                    geometry::AABB rb = geometry::robustBounds(P, 0.001f);  // only extreme strays
                    float md = (rb.max - rb.min).norm();
                    geometry::AABB full; for (const auto& p : P) full.expand(p);

                    if (vizKind == 1 && !pg.indices.empty()) {
                        geometry::BVH b; b.build(P, pg.indices);
                        b.nodeBoxesDepth(boxes, levels, kDepthTree);
                    } else if (vizKind == 2) {
                        geometry::Octree o; o.build(P, 64);
                        o.nodeBoxesDepth(boxes, levels, kDepthTree);
                    } else if (vizKind == 3) {
                        geometry::KDTree k; k.build(P);
                        k.cellBoxesDepth(full, boxes, levels, kDepthKD);
                    } else if (vizKind == 4) {
                        geometry::UniformGrid g; g.build(P, (md > 1e-6f ? md : 1.0f) / 40.0f);
                        g.occupiedCellBoxes(boxes); levels.assign(boxes.size(), 0);
                    } else if (vizKind == 5) {
                        geometry::LooseOctree lo; lo.build(P, 64);
                        lo.nodeBoxesDepth(boxes, levels, kDepthTree);
                    } else if (vizKind == 6) {
                        geometry::BSP bsp; bsp.build(P, 64);
                        bsp.nodeBoxesDepth(boxes, levels, kDepthTree);
                    } else if (vizKind == 7) {
                        geometry::RTree rt; rt.build(P, 16);
                        rt.nodeBoxesDepth(boxes, levels, kDepthTree);
                    } else if (vizKind == 8) {
                        geometry::BallTree bt; bt.build(P, 32);
                        bt.nodeSpheresDepth(sphC, sphR, sphL, 7);  // spheres get heavy -> shallow
                    }

                    if (!boxes.empty() || !sphC.empty()) {
                        float thick = (md > 1e-6f ? md : 1.0f) * 0.0005f;
                        float maxDiag = (md > 1e-6f ? md : 1.0f) * 1.5f;  // cull only bigger-than-model cells
                        const size_t kCap = 60000;                        // box budget
                        debug::DebugDraw tmp;  // local accumulator (not the frame one)
                        // Cull oversized boxes, then stride-sample to the budget.
                        std::vector<size_t> keep;
                        for (size_t i = 0; i < boxes.size(); ++i)
                            if ((boxes[i].max - boxes[i].min).norm() <= maxDiag) keep.push_back(i);
                        size_t stride = keep.size() > kCap ? (keep.size() + kCap - 1) / kCap : 1;
                        int drawn = 0;
                        for (size_t j = 0; j < keep.size(); j += stride) {
                            size_t i = keep[j];
                            tmp.addWireBox(boxes[i].min, boxes[i].max, levelColor(levels[i]), thick);
                            ++drawn;
                        }
                        std::vector<size_t> ks;
                        for (size_t i = 0; i < sphR.size(); ++i)
                            if (sphR[i] <= maxDiag) ks.push_back(i);
                        size_t ss = ks.size() > kCap ? (ks.size() + kCap - 1) / kCap : 1;
                        for (size_t j = 0; j < ks.size(); j += ss) {
                            size_t i = ks[j];
                            tmp.addSphere(sphC[i], sphR[i], levelColor(sphL[i]), 8);
                            ++drawn;
                        }
                        cache.boxCount = static_cast<int>(boxes.size() + sphC.size());
                        cache.inliers  = drawn;
                        cache.total    = static_cast<int>(P.size());
                        const auto& verts = tmp.vertices();
                        if (!verts.empty()) {
                            render::BufferDesc bd;
                            bd.type = render::BufferType::Vertex;
                            bd.usage = render::BufferUsage::Static;
                            bd.data = verts.data();
                            bd.size = verts.size() * sizeof(render::Vertex);
                            cache.vbo = renderer.createBuffer(bd);
                            render::MeshDesc md2;
                            md2.vertexBuffer = cache.vbo;
                            md2.layout = render::Vertex::layout();
                            md2.vertexCount = static_cast<uint32_t>(verts.size());
                            md2.topology = render::PrimitiveTopology::Triangles;
                            cache.mesh = renderer.createMesh(md2);
                            cache.vertexCount = static_cast<uint32_t>(verts.size());
                        }
                    }
                }

                if (cache.mesh != render::kInvalidMesh) {
                    renderer.setDrawMode(kSolid);
                    const auto& t = sv.get<Transform>(e);
                    Eigen::Matrix4f model = Mworld * t.matrix();
                    render::DrawItem item;
                    item.mesh = cache.mesh;
                    std::memcpy(item.model, model.data(), sizeof(item.model));
                    renderer.submit(item);
                }
            }
        }
    }

    // --- Immediate-mode debug geometry (world space, behind the grid) --
    drawDebugGeometry(world, renderer, Mworld);

    // --- Infinite ground grid (depth-tested against the scene) ---------
    // The grid is always the horizontal ground; the up-axis toggle re-expresses
    // content via Mworld. The axis arg only recolors the in-plane depth line
    // (blue Z in Y-up, green Y in Z-up) to match the gizmo. Skipped when the Space
    // toggle (GridState in ctx) has hidden it.
    {
        const auto* gs = world.ctx().find<GridState>();
        if (!gs || gs->visible) {
            // Cell size scales with the scene: take the largest horizontal (x/z)
            // extent of the world AABB, aim for ~10 minor cells across it, then snap
            // to a power of ten so lines land on round coordinates. 1.0 if empty.
            float cell = 1.0f;
            if (sceneValid) {
                Eigen::Vector3f s = sceneMax - sceneMin;
                float ext = (std::max)(s.x(), s.z());
                if (ext > 1e-6f)
                    cell = std::pow(10.0f, std::round(std::log10(ext / 10.0f)));
            }
            // Fade radius tracks the camera: ~ orbit distance, so the grid keeps
            // filling the view as you zoom out (with a floor so it never vanishes).
            float viewRadius = (std::max)(camDist * 4.0f, cell * 20.0f);
            float cp[3] = {camPos.x(), camPos.y(), camPos.z()};
            renderer.drawGrid(worldZUp(world) ? 2 : 1, cell, cp, viewRadius);
        }
    }

    // --- Axis gizmo overlay (top-right corner) -------------------------
    auto gz = world.view<AxisGizmo>();
    for (auto entity : gz) {
        const auto& gizmo = gz.get<AxisGizmo>(entity);
        if (gizmo.mesh == render::kInvalidMesh) break;

        GizmoRect r = gizmoRect(gizmo, viewportW, viewportH);

        render::OverlayContext ov;
        ov.x = r.x; ov.y = r.y; ov.width = r.w; ov.height = r.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = math::lookAt(Eigen::Vector3f(0, 0, kGizmoEyeZ), Eigen::Vector3f(0, 0, 0),
                                         Eigen::Vector3f(0, 1, 0));
        Eigen::Matrix4f ovProj = math::ortho(-kGizmoOrthoHalf, kGizmoOrthoHalf,
                                        -kGizmoOrthoHalf, kGizmoOrthoHalf, 0.1f, 100.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
        // Cube model = inverse(camera orientation) * world up-axis basis: shows the
        // logical world axes (Y or Z up) as the camera sees them. The ring is
        // screen-aligned (identity model).
        Eigen::Matrix4f cubeModel =
            math::toMat4(math::conjugate(camOrient)) * worldUpMatrix(gizmo.zUp);
        render::DrawItem item;

        // 1) Ring background (behind the cube), screen-aligned.
        if (gizmo.ringMesh != render::kInvalidMesh) {
            item.mesh = gizmo.ringMesh;
            std::memcpy(item.model, identity.data(), sizeof(item.model));
            renderer.submit(item);
        }

        // 2) Cube.
        item.mesh = gizmo.mesh;
        std::memcpy(item.model, cubeModel.data(), sizeof(item.model));
        renderer.submit(item);

        // 3) Highlight (click flash takes priority over hover). Cube regions use
        //    the cube model and conform to the faces; ring sectors are screen-aligned.
        bool clicked = gizmo.flash > 0.0f;
        GizmoPart part = clicked ? gizmo.flashPart : gizmo.hoverPart;
        if (part != GizmoPart::None && gizmo.highlightMesh != render::kInvalidMesh) {
            const float col[3] = {1.0f, clicked ? 0.5f : 0.9f, clicked ? 0.15f : 0.35f};
            render::Vertex hv[kHlVerts];
            if (part == GizmoPart::Cube) {
                buildCubeHighlight(clicked ? gizmo.flashDir : gizmo.hoverDir, col, hv);
                std::memcpy(item.model, cubeModel.data(), sizeof(item.model));
            } else {
                buildRingHighlight(clicked ? gizmo.flashSector : gizmo.hoverSector, col, hv);
                std::memcpy(item.model, identity.data(), sizeof(item.model));
            }
            renderer.updateBuffer(gizmo.highlightVbo, hv, sizeof(hv));
            item.mesh = gizmo.highlightMesh;
            renderer.submit(item);
        }

        // 4) Labels last (textured), so the text stays on top of cube + highlight.
        if (gizmo.labelMesh != render::kInvalidMesh) {
            item.mesh    = gizmo.labelMesh;
            item.texture = gizmo.labelTexture;
            std::memcpy(item.model, cubeModel.data(), sizeof(item.model));
            renderer.submit(item);
            item.texture = render::kInvalidTexture;
        }
        break;
    }

    // --- Gizmo up-axis toggle button overlay (corner of the gizmo) -----
    auto upbtn = world.view<AxisGizmo>();
    for (auto entity : upbtn) {
        auto& g = upbtn.get<AxisGizmo>(entity);
        if (!g.font || g.upBtnMesh == render::kInvalidMesh ||
            g.upBtnVbo == render::kInvalidBuffer)
            break;

        GizmoRect gr = gizmoRect(g, viewportW, viewportH);
        GizmoRect br = upToggleRect(gr);

        render::Vertex verts[kUpBtnVerts];
        buildUpToggleGeometry(g, verts);
        renderer.updateBuffer(g.upBtnVbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = br.x; ov.y = br.y; ov.width = br.w; ov.height = br.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = g.upBtnMesh;
        item.texture = g.uiAtlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- FPS widget overlay (draggable panel) --------------------------
    auto fw = world.view<FpsWidget>();
    for (auto entity : fw) {
        auto& wgt = fw.get<FpsWidget>(entity);

        // Apply a pending VSYNC toggle (set by fpsWidgetInputSystem) to the
        // renderer. Done here because this is the one ECS->renderer bridge.
        if (wgt.vsyncDirty) {
            renderer.setVsync(wgt.vsync);
            wgt.vsyncDirty = false;
        }

        if (!wgt.visible || wgt.mesh == render::kInvalidMesh ||
            wgt.vbo == render::kInvalidBuffer)
            break;

        render::Vertex verts[kFpsVerts];
        buildFpsGeometry(wgt, verts);
        renderer.updateBuffer(wgt.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = wgt.x; ov.y = wgt.y; ov.width = wgt.w; ov.height = wgt.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = wgt.mesh;
        item.texture = wgt.atlas;  // glyph atlas (white cell backs the solid parts)
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- Camera controls panel overlay --------------------------------
    auto ctrls = world.view<CameraControls>();
    for (auto entity : ctrls) {
        const auto& cc = ctrls.get<CameraControls>(entity);
        if (!cc.font || cc.mesh == render::kInvalidMesh || !primaryCam) break;

        render::Vertex verts[kCtrlVerts];
        buildControlsGeometry(cc, *primaryCam, verts);
        renderer.updateBuffer(cc.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = cc.x; ov.y = cc.y; ov.width = cc.w; ov.height = cc.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = cc.mesh;
        item.texture = cc.atlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- Cross-section panel overlay ----------------------------------
    auto csview = world.view<CrossSection>();
    for (auto entity : csview) {
        const auto& cs = csview.get<CrossSection>(entity);
        if (!cs.font || cs.mesh == render::kInvalidMesh) break;

        render::Vertex verts[kCsVerts];
        buildCrossSectionGeometry(cs, verts);
        renderer.updateBuffer(cs.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = cs.x; ov.y = cs.y; ov.width = cs.w; ov.height = cs.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = cs.mesh;
        item.texture = cs.atlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- Left selection toolbar overlay -------------------------------
    auto tbs = world.view<SelectionToolbar>();
    for (auto entity : tbs) {
        auto& tb = tbs.get<SelectionToolbar>(entity);
        if (!tb.visible || !tb.font || tb.mesh == render::kInvalidMesh ||
            tb.vbo == render::kInvalidBuffer)
            break;
        SelectionMode sm;
        if (world.ctx().contains<SelectionMode>()) sm = world.ctx().get<SelectionMode>();

        render::Vertex verts[kTbVerts];
        buildToolbarGeometry(tb, sm, verts, viewportW, viewportH);
        renderer.updateBuffer(tb.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = 0; ov.y = 0;
        ov.width = static_cast<int>(viewportW);
        ov.height = static_cast<int>(viewportH);
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = tb.mesh;
        item.texture = tb.atlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- Top menu bar overlay (full width, on top) ---------------------
    auto mbs = world.view<MenuBar>();
    for (auto entity : mbs) {
        auto& mb = mbs.get<MenuBar>(entity);
        if (!mb.visible || !mb.font || mb.mesh == render::kInvalidMesh ||
            mb.vbo == render::kInvalidBuffer)
            break;

        // Overlay grows downward to include the dropdown when a menu is open.
        float dropH = (mb.openMenu >= 0 && mb.openMenu < static_cast<int>(mb.menus.size()))
                          ? menuDropHeight(mb.menus[mb.openMenu])
                          : 0.0f;
        float overlayH = static_cast<float>(mb.height) + dropH;

        render::Vertex verts[kMenuVerts];
        buildMenuGeometry(mb, verts, viewportW, overlayH);
        renderer.updateBuffer(mb.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = 0; ov.y = 0;
        ov.width = static_cast<int>(viewportW);
        ov.height = static_cast<int>(overlayH);
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = mb.mesh;
        item.texture = mb.atlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    renderer.endFrame();
}

} // namespace orange::ecs
