#include "orange/ecs/systems.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "orange/core/math.h"
#include "orange/ecs/components.h"

namespace orange::ecs {

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

struct GizmoRect { int x, y, w, h; };

GizmoRect gizmoRect(const AxisGizmo& g, uint32_t W, uint32_t H) {
    (void)H;
    GizmoRect r;
    r.w = g.sizePx;
    r.h = g.sizePx;
    r.x = static_cast<int>(W) - g.sizePx - g.margin;  // top-right
    r.y = g.margin;
    if (r.x < 0) r.x = 0;
    return r;
}

// Ray vs. axis-aligned box [-1,1]^3. Returns the entry hit point.
bool intersectUnitBox(math::Vec3 o, math::Vec3 dir, math::Vec3& hit) {
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
bool intersectAABB(math::Vec3 o, math::Vec3 dir, math::Vec3 bmin, math::Vec3 bmax,
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
math::Vec3 classifyDir(math::Vec3 hit) {
    const float k = kGizmoEdge;
    auto zone = [&](float c) { return c > k ? 1.0f : (c < -k ? -1.0f : 0.0f); };
    math::Vec3 d(zone(hit.x()), zone(hit.y()), zone(hit.z()));
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
bool pickGizmo(const math::Quat& camOrient, const GizmoRect& r, float mx, float my,
               math::Vec3& outDir) {
    if (mx < r.x || mx > r.x + r.w || my < r.y || my > r.y + r.h) return false;
    float ndcX = (mx - r.x) / r.w * 2.0f - 1.0f;
    float ndcY = 1.0f - (my - r.y) / r.h * 2.0f;
    // Pick ray built directly in the cube's local space (cube model =
    // conjugate(camera), so its local axes are the world axes).
    math::Vec3 lo = math::rotate(
        camOrient, math::Vec3(ndcX * kGizmoOrthoHalf, ndcY * kGizmoOrthoHalf, kGizmoEyeZ));
    math::Vec3 ld = math::rotate(camOrient, math::Vec3(0, 0, -1));
    math::Vec3 hit;
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
math::Quat ringDelta(int sector) {
    switch (sector) {
        case 0: return math::quatAxisAngle(math::Vec3(0, 1, 0), -kHalfPi);  // right
        case 2: return math::quatAxisAngle(math::Vec3(0, 1, 0),  kHalfPi);  // left
        case 1: return math::quatAxisAngle(math::Vec3(1, 0, 0),  kHalfPi);  // top
        case 3: return math::quatAxisAngle(math::Vec3(1, 0, 0), -kHalfPi);  // bottom
    }
    return math::Quat::Identity();
}

void setComp(math::Vec3& v, int a, float val) { v[a] = val; }
float getComp(const math::Vec3& v, int a) { return v[a]; }

void emitQuad(render::Vertex* out, int& q, const math::Vec3 c[4], const float col[3]) {
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
void buildCubeHighlight(math::Vec3 d, const float col[3], render::Vertex out[kHlVerts]) {
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
            math::Vec3 v;
            setComp(v, a, faceCoord); setComp(v, b, bv); setComp(v, c, cv);
            return v;
        };
        math::Vec3 cor[4] = {mk(blo, clo), mk(bhi, clo), mk(bhi, chi), mk(blo, chi)};
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
        math::Vec3 cor[4] = {
            math::Vec3(std::cos(t0) * kRingInner, std::sin(t0) * kRingInner, zf),
            math::Vec3(std::cos(t0) * kRingOuter, std::sin(t0) * kRingOuter, zf),
            math::Vec3(std::cos(t1) * kRingOuter, std::sin(t1) * kRingOuter, zf),
            math::Vec3(std::cos(t1) * kRingInner, std::sin(t1) * kRingInner, zf),
        };
        emitQuad(out, q, cor, col);
    }
    padDegenerate(out, q);
}

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
} // namespace

void spinSystem(entt::registry& world, float dt) {
    auto view = world.view<Transform, Spin>();
    for (auto entity : view) {
        auto& t = view.get<Transform>(entity);
        auto& s = view.get<Spin>(entity);
        // Integrate angular velocity as a rigid rotation (no Euler drift/lock).
        math::Vec3 w = s.axisRadiansPerSec;
        float speed = std::sqrt(math::dot(w, w));
        if (speed > 1e-8f) {
            math::Quat dq = math::quatAxisAngle(w, speed * dt);
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
            math::Quat dYaw =
                math::quatAxisAngle(math::Vec3(0, 1, 0), -input.mouseDeltaX * m.rotateSpeed);
            math::Quat dPitch =
                math::quatAxisAngle(math::Vec3(1, 0, 0), -input.mouseDeltaY * m.rotateSpeed);
            m.orientation = math::normalize(m.orientation * dYaw * dPitch);
        }

        // Zoom: scroll wheel changes distance.
        if (input.wheel != 0.0f) {
            m.distance -= input.wheel * m.zoomSpeed;
            m.distance  = clampf(m.distance, m.minDistance, m.maxDistance);
        }

        // Pan: middle-drag slides the target across the camera plane.
        // (Right-drag is orbit; left-click is picking.)
        if (input.buttonMiddle) {
            math::Vec3 right = math::rotate(m.orientation, math::Vec3(1, 0, 0));
            math::Vec3 up    = math::rotate(m.orientation, math::Vec3(0, 1, 0));
            float k = m.panSpeed * m.distance;
            m.target = m.target - right * (input.mouseDeltaX * k) +
                       up * (input.mouseDeltaY * k);
        }

        // Place the camera on the orbit sphere; eye sits along the local +Z.
        math::Vec3 offset = math::rotate(m.orientation, math::Vec3(0, 0, 1));
        t.position    = m.target + offset * m.distance;
        t.orientation = m.orientation;
    }
}

void pickingSystem(entt::registry& world, const core::Input& input,
                   uint32_t viewportW, uint32_t viewportH) {
    // Only act on a fresh left-click that no UI widget already consumed.
    if (!input.leftClicked || input.captured) return;

    // Build a world-space pick ray from the primary camera + cursor position.
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
    float ndcX = 2.0f * input.mousePosX / W - 1.0f;
    float ndcY = 1.0f - 2.0f * input.mousePosY / H;  // screen y-down -> ndc y-up
    float aspect = W / H;

    math::Vec3 right   = math::rotate(camT->orientation, math::Vec3(1, 0, 0));
    math::Vec3 up      = math::rotate(camT->orientation, math::Vec3(0, 1, 0));
    math::Vec3 forward = math::rotate(camT->orientation, math::Vec3(0, 0, -1));

    math::Vec3 rayO, rayD;
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

    // Test every drawable; keep the nearest hit along the ray.
    entt::entity best = entt::null;
    float bestT = 1e30f;
    auto drawables = world.view<Transform, Renderable>();
    for (auto e : drawables) {
        const auto& t = drawables.get<Transform>(e);
        const auto& r = drawables.get<Renderable>(e);
        if (!r.visible || r.mesh == render::kInvalidMesh) continue;

        // Transform the ray into the entity's local space: inverse(T*R*S) =
        // S^-1 * R^-1 * T^-1. Scale is applied component-wise.
        math::Vec3 lo = math::rotate(math::conjugate(t.orientation),
                                     rayO - t.position);
        math::Vec3 ld = math::rotate(math::conjugate(t.orientation), rayD);
        math::Vec3 inv(t.scale.x() != 0 ? 1.0f / t.scale.x() : 0.0f,
                       t.scale.y() != 0 ? 1.0f / t.scale.y() : 0.0f,
                       t.scale.z() != 0 ? 1.0f / t.scale.z() : 0.0f);
        lo = lo.cwiseProduct(inv);
        ld = ld.cwiseProduct(inv);

        float tHit;
        if (intersectAABB(lo, ld, r.boundsMin, r.boundsMax, tHit) && tHit < bestT) {
            bestT = tHit;
            best  = e;
        }
    }

    // Single-select: clear previous selection, mark the hit (if any).
    for (auto e : drawables) drawables.get<Renderable>(e).selected = false;
    if (best != entt::null) {
        drawables.get<Renderable>(best).selected = true;
        std::printf("picked entity %u\n", static_cast<unsigned>(best));
        std::fflush(stdout);
    }
}

void axisGizmoInputSystem(entt::registry& world, const core::Input& input,
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

    // Hover: cube first (it sits in front of the ring), then the ring.
    math::Vec3 dir;
    if (pickGizmo(manip->orientation, r, input.mousePosX, input.mousePosY, dir)) {
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
        math::Quat target;
        if (gizmo->hoverPart == GizmoPart::Cube)
            target = math::quatLookZ(gizmo->hoverDir);           // look at that view
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

    // Position under the gizmo (top-right). Gizmo: size 150, margin 14.
    cc->x = static_cast<int>(viewportW) - cc->w - 14;
    cc->y = 14 + 150 + 10;
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

void renderSystem(entt::registry& world, render::IRenderer& renderer,
                  uint32_t viewportW, uint32_t viewportH) {
    // --- Find the primary camera ---------------------------------------
    render::FrameContext frame;
    frame.width  = viewportW;
    frame.height = viewportH;

    math::Mat4 view = math::Mat4::Identity();
    math::Mat4 proj = math::Mat4::Identity();
    math::Quat camOrient = math::Quat::Identity();  // captured for the gizmo overlay
    const Camera* primaryCam = nullptr;  // captured for the controls panel

    auto cams = world.view<Transform, Camera>();
    for (auto entity : cams) {
        const auto& cam = cams.get<Camera>(entity);
        if (!cam.primary) continue;
        const auto& ct = cams.get<Transform>(entity);
        primaryCam = &cam;

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
        math::Vec3 forward = math::rotate(ct.orientation, math::Vec3(0, 0, -1));
        math::Vec3 up      = math::rotate(ct.orientation, math::Vec3(0, 1, 0));
        view = math::lookAt(ct.position, ct.position + forward, up);
        camOrient = ct.orientation;
        break;
    }

    std::memcpy(frame.view, view.data(), sizeof(frame.view));
    std::memcpy(frame.proj, proj.data(), sizeof(frame.proj));

    // --- Submit all drawables ------------------------------------------
    renderer.beginFrame(frame);

    auto drawables = world.view<Transform, Renderable>();
    for (auto entity : drawables) {
        const auto& t = drawables.get<Transform>(entity);
        const auto& r = drawables.get<Renderable>(entity);
        if (!r.visible || r.mesh == render::kInvalidMesh) continue;

        render::DrawItem item;
        item.mesh = r.mesh;
        // Selected entities pop slightly larger as a pick highlight (the render
        // ABI carries no per-item tint, so scale stands in for it).
        math::Mat4 model = r.selected
                               ? math::translate(t.position) *
                                     math::toMat4(t.orientation) *
                                     math::scale(t.scale * 1.15f)
                               : t.matrix();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
    }

    // --- Infinite ground grid (depth-tested against the scene) ---------
    renderer.drawGrid();

    // --- Axis gizmo overlay (top-right corner) -------------------------
    auto gz = world.view<AxisGizmo>();
    for (auto entity : gz) {
        const auto& gizmo = gz.get<AxisGizmo>(entity);
        if (gizmo.mesh == render::kInvalidMesh) break;

        GizmoRect r = gizmoRect(gizmo, viewportW, viewportH);

        render::OverlayContext ov;
        ov.x = r.x; ov.y = r.y; ov.width = r.w; ov.height = r.h;
        ov.clearDepth = true;
        math::Mat4 ovView = math::lookAt(math::Vec3(0, 0, kGizmoEyeZ), math::Vec3(0, 0, 0),
                                         math::Vec3(0, 1, 0));
        math::Mat4 ovProj = math::ortho(-kGizmoOrthoHalf, kGizmoOrthoHalf,
                                        -kGizmoOrthoHalf, kGizmoOrthoHalf, 0.1f, 100.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        math::Mat4 identity = math::Mat4::Identity();
        // Cube model = inverse(camera orientation): shows world axes as the
        // camera sees them. The ring is screen-aligned (identity model).
        math::Mat4 cubeModel = math::toMat4(math::conjugate(camOrient));
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
        math::Mat4 ovView = math::Mat4::Identity();
        math::Mat4 ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = wgt.mesh;
        item.texture = wgt.atlas;  // glyph atlas (white cell backs the solid parts)
        math::Mat4 model = math::Mat4::Identity();
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
        math::Mat4 ovView = math::Mat4::Identity();
        math::Mat4 ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = cc.mesh;
        item.texture = cc.atlas;
        math::Mat4 model = math::Mat4::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    renderer.endFrame();
}

} // namespace orange::ecs
