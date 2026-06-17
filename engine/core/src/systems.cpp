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
// kGizmoEdge (face/edge boundary) lives in components.h so the sandbox grid uses
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
    float O[3] = {o.x, o.y, o.z};
    float D[3] = {dir.x, dir.y, dir.z};
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

// Classify a hit point on the cube into a direction triple in {-1,0,1}^3:
// one nonzero => face, two => edge, three => corner. The triple is the world
// view direction to snap the camera to.
math::Vec3 classifyDir(math::Vec3 hit) {
    const float k = kGizmoEdge;
    auto zone = [&](float c) { return c > k ? 1.0f : (c < -k ? -1.0f : 0.0f); };
    math::Vec3 d{zone(hit.x), zone(hit.y), zone(hit.z)};
    if (d.x == 0 && d.y == 0 && d.z == 0) {  // dead-center: snap to dominant axis
        float ax = std::fabs(hit.x), ay = std::fabs(hit.y), az = std::fabs(hit.z);
        if (ax >= ay && ax >= az)      d.x = hit.x > 0 ? 1.0f : -1.0f;
        else if (ay >= az)             d.y = hit.y > 0 ? 1.0f : -1.0f;
        else                           d.z = hit.z > 0 ? 1.0f : -1.0f;
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
        camOrient, {ndcX * kGizmoOrthoHalf, ndcY * kGizmoOrthoHalf, kGizmoEyeZ});
    math::Vec3 ld = math::rotate(camOrient, {0, 0, -1});
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
        case 0: return math::quatAxisAngle({0, 1, 0}, -kHalfPi);  // right
        case 2: return math::quatAxisAngle({0, 1, 0},  kHalfPi);  // left
        case 1: return math::quatAxisAngle({1, 0, 0},  kHalfPi);  // top
        case 3: return math::quatAxisAngle({1, 0, 0}, -kHalfPi);  // bottom
    }
    return math::Quat::identity();
}

void setComp(math::Vec3& v, int a, float val) {
    if (a == 0) v.x = val; else if (a == 1) v.y = val; else v.z = val;
}
float getComp(const math::Vec3& v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }

void emitQuad(render::Vertex* out, int& q, const math::Vec3 c[4], const float col[3]) {
    if (q >= kHlQuads) return;
    for (int i = 0; i < 4; ++i)
        out[q * 4 + i] = {{c[i].x, c[i].y, c[i].z}, {col[0], col[1], col[2]}};
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
            {std::cos(t0) * kRingInner, std::sin(t0) * kRingInner, zf},
            {std::cos(t0) * kRingOuter, std::sin(t0) * kRingOuter, zf},
            {std::cos(t1) * kRingOuter, std::sin(t1) * kRingOuter, zf},
            {std::cos(t1) * kRingInner, std::sin(t1) * kRingInner, zf},
        };
        emitQuad(out, q, cor, col);
    }
    padDegenerate(out, q);
}

// --- FPS widget geometry ---------------------------------------------------
constexpr int kFpsQuads = 160;            // dynamic mesh capacity
constexpr int kFpsVerts = kFpsQuads * 4;  // = 640

// Appends textured glyph quads for `s` (proportional layout) using a baked Font.
// y-up; baseY is the text baseline. Writes into out[q..], capped at `cap` quads.
void appendText(render::Vertex* out, int& q, int cap, const core::Font& f,
                const char* s, float penX, float baseY, float h, const float col[3],
                float z) {
    for (; *s; ++s) {
        const core::Glyph& g = f.glyph(*s);
        if (g.w > 0 && g.h > 0 && q < cap) {
            float x0 = penX + g.xoff * h, y1 = baseY - g.yoff * h;
            float x1 = x0 + g.w * h, y0 = y1 - g.h * h;
            out[q * 4 + 0] = {{x0, y0, z}, {col[0], col[1], col[2]}, {g.u0, g.v1}};
            out[q * 4 + 1] = {{x1, y0, z}, {col[0], col[1], col[2]}, {g.u1, g.v1}};
            out[q * 4 + 2] = {{x1, y1, z}, {col[0], col[1], col[2]}, {g.u1, g.v0}};
            out[q * 4 + 3] = {{x0, y1, z}, {col[0], col[1], col[2]}, {g.u0, g.v0}};
            ++q;
        }
        penX += g.advance * h;
    }
}

// Builds the widget geometry in normalized [0,1]^2 space. Solid fills use the
// font's white texel; the readout uses proportional Malgun Gothic glyphs. The
// bar graph auto-scales to its recent peak so variation is always visible.
void buildFpsGeometry(const FpsWidget& wgt, render::Vertex* out) {
    int q = 0;
    if (!wgt.font) { for (int i = 0; i < kFpsVerts; ++i) out[i] = {{0,0,0},{0,0,0}}; return; }
    const core::Font& f = *wgt.font;
    const float wu = f.whiteU, wv = f.whiteV;
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) {
        if (q >= kFpsQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {wu, wv}};
        ++q;
    };

    solid(0, 0, 1, 1, 0.08f, 0.09f, 0.11f, 0.0f);  // background panel

    // Auto-scale the graph to its recent peak (min 60 so idle stays sensible).
    const int N = FpsWidget::kSamples;
    float peak = 60.0f;
    for (int i = 0; i < N; ++i) peak = (std::max)(peak, wgt.history[i]);
    float scale = peak * 1.15f;

    const float gx0 = 0.04f, gx1 = 0.96f, gy0 = 0.08f, gy1 = 0.56f;
    float bw = (gx1 - gx0) / N;
    for (int i = 0; i < N; ++i) {
        float fps = wgt.history[(wgt.head + i) % N];  // oldest -> newest
        float hn = clampf(fps / scale, 0.0f, 1.0f);
        float t  = clampf(fps / 60.0f, 0.0f, 1.0f);   // red (low) -> green (high)
        solid(gx0 + i * bw, gy0, gx0 + i * bw + bw * 0.85f, gy0 + (gy1 - gy0) * hn,
              1.0f - t * 0.85f, 0.25f + t * 0.65f, 0.28f, 0.3f);
    }

    // Readout "<fps> FPS".
    int val = static_cast<int>(wgt.smoothFps + 0.5f);
    if (val > 9999) val = 9999;
    char s[16]; int len = 0;
    { char tmp[8]; int n = 0, v = val;
      if (v == 0) tmp[n++] = '0';
      while (v > 0) { tmp[n++] = char('0' + v % 10); v /= 10; }
      for (int i = n - 1; i >= 0; --i) s[len++] = tmp[i]; }
    s[len++] = ' '; s[len++] = 'F'; s[len++] = 'P'; s[len++] = 'S'; s[len] = '\0';

    float h = 0.30f;
    float tw = f.textWidth(s, h);
    if (tw > 0.92f) { h *= 0.92f / tw; tw = 0.92f; }
    const float col[3] = {0.93f, 0.96f, 0.93f};
    appendText(out, q, kFpsQuads, f, s, 0.06f, 0.70f, h, col, 0.6f);

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
    auto text = [&](const char* s, float penX, float baseY, float h, float r, float g,
                    float b, float z) {
        for (; *s; ++s) {
            const core::Glyph& gl = f.glyph(*s);
            if (gl.w > 0 && gl.h > 0) {
                float x0 = penX + gl.xoff * h, y1 = baseY - gl.yoff * h;
                float x1 = x0 + gl.w * h, y0 = y1 - gl.h * h;
                quad(x0, y0, x1, y1, r, g, b, gl.u0, gl.v0, gl.u1, gl.v1, z);
            }
            penX += gl.advance * h;
        }
    };
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
    float h1 = (y1 - y0) * 0.50f;
    text(modeTxt, (x0 + x1) * 0.5f - f.textWidth(modeTxt, h1) * 0.5f,
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
    float hv = (y1 - y0) * 0.42f;
    text(buf, (vx0 + vx1) * 0.5f - f.textWidth(buf, hv) * 0.5f, (y0 + y1) * 0.5f - hv * 0.35f,
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
        } else if (!input.captured && input.buttonLeft &&
                   (input.mouseDeltaX != 0.0f || input.mouseDeltaY != 0.0f)) {
            // Orbit: left-drag tumbles about the camera's OWN x/y axes and
            // composes on the local side -- a true trackball, no gimbal lock.
            math::Quat dYaw =
                math::quatAxisAngle({0, 1, 0}, -input.mouseDeltaX * m.rotateSpeed);
            math::Quat dPitch =
                math::quatAxisAngle({1, 0, 0}, -input.mouseDeltaY * m.rotateSpeed);
            m.orientation = math::normalize(m.orientation * dYaw * dPitch);
        }

        // Zoom: scroll wheel changes distance.
        if (input.wheel != 0.0f) {
            m.distance -= input.wheel * m.zoomSpeed;
            m.distance  = clampf(m.distance, m.minDistance, m.maxDistance);
        }

        // Pan: middle/right-drag slides the target across the camera plane.
        if (input.buttonMiddle || input.buttonRight) {
            math::Vec3 right = math::rotate(m.orientation, {1, 0, 0});
            math::Vec3 up    = math::rotate(m.orientation, {0, 1, 0});
            float k = m.panSpeed * m.distance;
            m.target = m.target - right * (input.mouseDeltaX * k) +
                       up * (input.mouseDeltaY * k);
        }

        // Place the camera on the orbit sphere; eye sits along the local +Z.
        math::Vec3 offset = math::rotate(m.orientation, {0, 0, 1});
        t.position    = m.target + offset * m.distance;
        t.orientation = m.orientation;
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

        // Push current FPS into the ring buffer (+ a smoothed value for the text).
        float fps = dt > 1e-6f ? 1.0f / dt : 0.0f;
        wgt.history[wgt.head] = fps;
        wgt.head = (wgt.head + 1) % FpsWidget::kSamples;
        wgt.smoothFps = wgt.smoothFps * 0.92f + fps * 0.08f;

        // Dragging.
        float mx = input.mousePosX, my = input.mousePosY;
        bool inside = mx >= wgt.x && mx <= wgt.x + wgt.w && my >= wgt.y &&
                      my <= wgt.y + wgt.h;
        if (input.leftClicked && inside) {
            wgt.dragging = true;
            wgt.dragOffX = mx - wgt.x;
            wgt.dragOffY = my - wgt.y;
        }
        if (wgt.dragging) {
            if (input.buttonLeft) {
                wgt.x = static_cast<int>(clampf(mx - wgt.dragOffX, 0.0f,
                                                static_cast<float>(viewportW - wgt.w)));
                wgt.y = static_cast<int>(clampf(my - wgt.dragOffY, 0.0f,
                                                static_cast<float>(viewportH - wgt.h)));
                input.captured = true;  // suppress camera orbit while dragging
            } else {
                wgt.dragging = false;
            }
        }
    }
}

void cameraControlsInputSystem(entt::registry& world, core::Input& input,
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
    if (!cam || !input.leftClicked) return;

    float mx = input.mousePosX, my = input.mousePosY;
    auto hit = [&](const CRect& r) {
        return mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h;
    };
    CtrlRects R = controlRects(*cc);
    bool persp = cam->mode == ProjectionMode::Perspective;
    if (hit(R.mode)) {
        cam->mode = persp ? ProjectionMode::Orthographic : ProjectionMode::Perspective;
        input.captured = true;
    } else if (hit(R.minus)) {
        if (persp) cam->fovYDegrees = clampf(cam->fovYDegrees - 5.0f, 10.0f, 120.0f);
        else       cam->orthoSize   = clampf(cam->orthoSize - 0.5f, 0.5f, 30.0f);
        input.captured = true;
    } else if (hit(R.plus)) {
        if (persp) cam->fovYDegrees = clampf(cam->fovYDegrees + 5.0f, 10.0f, 120.0f);
        else       cam->orthoSize   = clampf(cam->orthoSize + 0.5f, 0.5f, 30.0f);
        input.captured = true;
    }
}

void renderSystem(entt::registry& world, render::IRenderer& renderer,
                  uint32_t viewportW, uint32_t viewportH) {
    // --- Find the primary camera ---------------------------------------
    render::FrameContext frame;
    frame.width  = viewportW;
    frame.height = viewportH;

    math::Mat4 view = math::Mat4::identity();
    math::Mat4 proj = math::Mat4::identity();
    math::Quat camOrient{};       // captured for the gizmo overlay
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
        math::Vec3 forward = math::rotate(ct.orientation, {0, 0, -1});
        math::Vec3 up      = math::rotate(ct.orientation, {0, 1, 0});
        view = math::lookAt(ct.position, ct.position + forward, up);
        camOrient = ct.orientation;
        break;
    }

    std::memcpy(frame.view, view.m, sizeof(frame.view));
    std::memcpy(frame.proj, proj.m, sizeof(frame.proj));

    // --- Submit all drawables ------------------------------------------
    renderer.beginFrame(frame);

    auto drawables = world.view<Transform, Renderable>();
    for (auto entity : drawables) {
        const auto& t = drawables.get<Transform>(entity);
        const auto& r = drawables.get<Renderable>(entity);
        if (!r.visible || r.mesh == render::kInvalidMesh) continue;

        render::DrawItem item;
        item.mesh = r.mesh;
        math::Mat4 model = t.matrix();
        std::memcpy(item.model, model.m, sizeof(item.model));
        renderer.submit(item);
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
        math::Mat4 ovView = math::lookAt({0, 0, kGizmoEyeZ}, {0, 0, 0}, {0, 1, 0});
        math::Mat4 ovProj = math::ortho(-kGizmoOrthoHalf, kGizmoOrthoHalf,
                                        -kGizmoOrthoHalf, kGizmoOrthoHalf, 0.1f, 100.0f);
        std::memcpy(ov.view, ovView.m, sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.m, sizeof(ov.proj));
        renderer.beginOverlay(ov);

        math::Mat4 identity = math::Mat4::identity();
        // Cube model = inverse(camera orientation): shows world axes as the
        // camera sees them. The ring is screen-aligned (identity model).
        math::Mat4 cubeModel = math::toMat4(math::conjugate(camOrient));
        render::DrawItem item;

        // 1) Ring background (behind the cube), screen-aligned.
        if (gizmo.ringMesh != render::kInvalidMesh) {
            item.mesh = gizmo.ringMesh;
            std::memcpy(item.model, identity.m, sizeof(item.model));
            renderer.submit(item);
        }

        // 2) Cube.
        item.mesh = gizmo.mesh;
        std::memcpy(item.model, cubeModel.m, sizeof(item.model));
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
                std::memcpy(item.model, cubeModel.m, sizeof(item.model));
            } else {
                buildRingHighlight(clicked ? gizmo.flashSector : gizmo.hoverSector, col, hv);
                std::memcpy(item.model, identity.m, sizeof(item.model));
            }
            renderer.updateBuffer(gizmo.highlightVbo, hv, sizeof(hv));
            item.mesh = gizmo.highlightMesh;
            renderer.submit(item);
        }

        // 4) Labels last (textured), so the text stays on top of cube + highlight.
        if (gizmo.labelMesh != render::kInvalidMesh) {
            item.mesh    = gizmo.labelMesh;
            item.texture = gizmo.labelTexture;
            std::memcpy(item.model, cubeModel.m, sizeof(item.model));
            renderer.submit(item);
            item.texture = render::kInvalidTexture;
        }
        break;
    }

    // --- FPS widget overlay (draggable panel) --------------------------
    auto fw = world.view<FpsWidget>();
    for (auto entity : fw) {
        const auto& wgt = fw.get<FpsWidget>(entity);
        if (!wgt.visible || wgt.mesh == render::kInvalidMesh ||
            wgt.vbo == render::kInvalidBuffer)
            break;

        render::Vertex verts[kFpsVerts];
        buildFpsGeometry(wgt, verts);
        renderer.updateBuffer(wgt.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = wgt.x; ov.y = wgt.y; ov.width = wgt.w; ov.height = wgt.h;
        ov.clearDepth = true;
        math::Mat4 ovView = math::Mat4::identity();
        math::Mat4 ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.m, sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.m, sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = wgt.mesh;
        item.texture = wgt.atlas;  // glyph atlas (white cell backs the solid parts)
        math::Mat4 model = math::Mat4::identity();
        std::memcpy(item.model, model.m, sizeof(item.model));
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
        math::Mat4 ovView = math::Mat4::identity();
        math::Mat4 ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.m, sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.m, sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = cc.mesh;
        item.texture = cc.atlas;
        math::Mat4 model = math::Mat4::identity();
        std::memcpy(item.model, model.m, sizeof(item.model));
        renderer.submit(item);
        break;
    }

    renderer.endFrame();
}

} // namespace orange::ecs
