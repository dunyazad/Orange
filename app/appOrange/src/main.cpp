// Orange appOrange: loads a render backend plugin at runtime, builds an ECS
// world (no scene graph), and spins a cube.
//
//   appOrange            -> OpenGL backend (default)
//   appOrange --vulkan   -> Vulkan backend (if the render_vk plugin was built)
//   appOrange --gl       -> OpenGL backend (explicit)

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>  // declares SDL_SetMainReady() under SDL_MAIN_HANDLED

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include "mesh_io.h"
#include "orange/core/application.h"
#include "orange/core/crash_handler.h"
#include "orange/core/screenshot.h"
#include "orange/core/ui_layout.h"
#include "orange/core/buffer.h"
#include "orange/core/console.h"
#include "orange/core/math.h"
#include "orange/core/normals.h"
#include "orange/core/poisson_reconstruction.h"
#include "orange/ecs/components.h"
#include "orange/ecs/systems.h"
#include "orange/render/types.h"

using namespace orange;

namespace {

const char* const kLabels[6] = {"X", "-X", "Y", "-Y", "Z", "-Z"};
constexpr int kCellW = 256, kCellH = 256, kCells = 6;  // high-res for crisp text

// Candidate Malgun Gothic locations (Windows). First that loads wins.
const char* const kFontPaths[] = {
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunsl.ttf",
};

std::vector<unsigned char> readFile(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());
}

std::vector<unsigned char> loadFont() {
    for (const char* p : kFontPaths) {
        auto data = readFile(p);
        if (!data.empty()) { SDL_Log("Font: loaded %s", p); return data; }
    }
    SDL_Log("Font: Malgun Gothic not found; text will be blank");
    return {};
}

// Bakes ASCII 32..126 into a packed glyph atlas (Malgun Gothic) and fills a
// core::Font with UVs + metrics. A white texel at (0,0) backs solid UI fills.
bool bakeFont(render::IRenderer& renderer, const std::vector<unsigned char>& ttf,
              float pxH, core::Font& out) {
    if (ttf.empty()) return false;
    const int W = 1024, H = 512;
    std::vector<unsigned char> bmp(static_cast<size_t>(W) * H, 0);

    stbtt_pack_context pc;
    if (!stbtt_PackBegin(&pc, bmp.data(), W, H, 0, 1, nullptr)) return false;
    stbtt_PackSetOversampling(&pc, 2, 2);
    stbtt_packedchar chars[95];
    stbtt_PackFontRange(&pc, ttf.data(), 0, pxH, 32, 95, chars);
    stbtt_PackEnd(&pc);

    bmp[0] = 255;  // white texel for solid fills
    out.whiteU = 0.5f / W;
    out.whiteV = 0.5f / H;
    for (int i = 0; i < 95; ++i) {
        const stbtt_packedchar& p = chars[i];
        core::Glyph& g = out.glyphs[i];
        g.u0 = p.x0 / float(W); g.v0 = p.y0 / float(H);
        g.u1 = p.x1 / float(W); g.v1 = p.y1 / float(H);
        // Quad size must come from xoff2/yoff2 (screen px, oversample-corrected),
        // not x1-x0 (oversampled texels) -- otherwise glyphs render 2x too big.
        g.w = (p.xoff2 - p.xoff) / pxH; g.h = (p.yoff2 - p.yoff) / pxH;
        g.xoff = p.xoff / pxH; g.yoff = p.yoff / pxH;
        g.advance = p.xadvance / pxH;
    }

    std::vector<uint8_t> rgba(static_cast<size_t>(W) * H * 4);
    for (int i = 0; i < W * H; ++i) {
        rgba[i * 4 + 0] = 255; rgba[i * 4 + 1] = 255; rgba[i * 4 + 2] = 255;
        rgba[i * 4 + 3] = bmp[i];
    }
    render::TextureDesc td;
    td.width = W; td.height = H; td.pixels = rgba.data();
    out.texture = renderer.createTexture(td);
    return out.texture != render::kInvalidTexture;
}

// Rasterizes each cell's string (centered, white text) into one RGBA atlas using
// the given TrueType font. Cells are laid out left-to-right, top row first.
bool bakeCellAtlas(const std::vector<unsigned char>& ttf,
                   const std::vector<std::string>& cells, int cellW, int cellH,
                   float pxHeight, std::vector<uint8_t>& px, int& W, int& H) {
    W = cellW * static_cast<int>(cells.size());
    H = cellH;
    px.assign(static_cast<size_t>(W) * H * 4, 0);  // transparent
    if (ttf.empty()) return false;

    std::vector<unsigned char> cov(static_cast<size_t>(W) * H, 0);  // glyph coverage

    stbtt_fontinfo font;
    if (!stbtt_InitFont(&font, ttf.data(), stbtt_GetFontOffsetForIndex(ttf.data(), 0)))
        return false;
    float sc = stbtt_ScaleForPixelHeight(&font, pxHeight);
    int asc, desc, gap;
    stbtt_GetFontVMetrics(&font, &asc, &desc, &gap);

    for (size_t c = 0; c < cells.size(); ++c) {
        const std::string& s = cells[c];
        if (s.empty()) continue;
        // Measure advance width.
        float tw = 0.0f;
        for (char ch : s) {
            int adv, lsb; stbtt_GetCodepointHMetrics(&font, ch, &adv, &lsb);
            tw += adv * sc;
        }
        float penX = c * cellW + (cellW - tw) * 0.5f;
        float baseline = (cellH + (asc + desc) * sc) * 0.5f;  // vertically centered
        for (char ch : s) {
            int adv, lsb; stbtt_GetCodepointHMetrics(&font, ch, &adv, &lsb);
            int x0, y0, x1, y1;
            stbtt_GetCodepointBitmapBox(&font, ch, sc, sc, &x0, &y0, &x1, &y1);
            int gw = x1 - x0, gh = y1 - y0;
            if (gw > 0 && gh > 0) {
                std::vector<unsigned char> bmp(static_cast<size_t>(gw) * gh);
                stbtt_MakeCodepointBitmap(&font, bmp.data(), gw, gh, gw, sc, sc, ch);
                int ox = static_cast<int>(penX + lsb * sc) + x0;
                int oy = static_cast<int>(baseline) + y0;
                for (int j = 0; j < gh; ++j)
                    for (int i = 0; i < gw; ++i) {
                        int ax = ox + i, ay = oy + j;
                        if (ax < 0 || ay < 0 || ax >= W || ay >= H) continue;
                        unsigned char a = bmp[j * gw + i];
                        size_t k = static_cast<size_t>(ay) * W + ax;
                        if (a > cov[k]) cov[k] = a;
                    }
            }
            penX += adv * sc;
        }
    }

    // Composite: white glyph with a dark outline (dilated coverage) so labels
    // stay readable on any face color.
    const int rad = 3;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            unsigned char a = cov[static_cast<size_t>(y) * W + x];  // text
            unsigned char o = a;                                   // dilated
            for (int dy = -rad; dy <= rad; ++dy)
                for (int dx = -rad; dx <= rad; ++dx) {
                    int sx = x + dx, sy = y + dy;
                    if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
                    o = (std::max)(o, cov[static_cast<size_t>(sy) * W + sx]);
                }
            if (!o) continue;
            float af = a / 255.0f;  // 1 = text, 0 = outline-only
            unsigned char c = static_cast<unsigned char>((0.05f + 0.95f * af) * 255.0f);
            size_t k = (static_cast<size_t>(y) * W + x) * 4;
            px[k] = px[k + 1] = px[k + 2] = c;
            px[k + 3] = o;  // dilated coverage as alpha (outline + text)
        }
    return true;
}

// The 6 gizmo labels rendered with Malgun Gothic into the label atlas.
void buildLabelAtlas(const std::vector<unsigned char>& ttf, std::vector<uint8_t>& px,
                     int& W, int& H) {
    std::vector<std::string> cells(kLabels, kLabels + kCells);
    bakeCellAtlas(ttf, cells, kCellW, kCellH, kCellH * 0.66f, px, W, H);
}

// One textured quad per gizmo face, with UVs into the atlas cell for that face.
void buildLabelQuads(std::vector<render::Vertex>& verts, std::vector<uint32_t>& idx) {
    using Eigen::Vector3f;
    struct LabelFace { Vector3f n, rt, up; };
    const LabelFace faces[6] = {
        {Vector3f(1, 0, 0),  Vector3f(0, 0, -1), Vector3f(0, 1, 0)},   // X
        {Vector3f(-1, 0, 0), Vector3f(0, 0, 1),  Vector3f(0, 1, 0)},   // -X
        {Vector3f(0, 1, 0),  Vector3f(1, 0, 0),  Vector3f(0, 0, 1)},   // Y
        {Vector3f(0, -1, 0), Vector3f(1, 0, 0),  Vector3f(0, 0, -1)},  // -Y
        {Vector3f(0, 0, 1),  Vector3f(1, 0, 0),  Vector3f(0, 1, 0)},   // Z
        {Vector3f(0, 0, -1), Vector3f(-1, 0, 0), Vector3f(0, 1, 0)},   // -Z
    };
    const float hw = 0.55f, hh = 0.55f, out = 1.05f;  // square quad (square atlas cell)
    for (int c = 0; c < 6; ++c) {
        const LabelFace& f = faces[c];
        Vector3f C = f.n * out;
        float u0 = static_cast<float>(c) / kCells, u1 = static_cast<float>(c + 1) / kCells;
        // top-left, top-right, bottom-right, bottom-left  (v=0 is the atlas top)
        Vector3f p[4] = {C - f.rt * hw + f.up * hh, C + f.rt * hw + f.up * hh,
                     C + f.rt * hw - f.up * hh, C - f.rt * hw - f.up * hh};
        float uv[4][2] = {{u0, 0}, {u1, 0}, {u1, 1}, {u0, 1}};
        uint32_t base = static_cast<uint32_t>(verts.size());
        for (int i = 0; i < 4; ++i)
            verts.push_back({{p[i].x(), p[i].y(), p[i].z()}, {1, 1, 1}, {uv[i][0], uv[i][1]}});
        idx.insert(idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    }
}

// A 4-sector ring behind the gizmo cube (right/top/left/bottom), screen-aligned.
// Clicking a sector rotates the camera 90 degrees in that direction.
void buildRing(std::vector<render::Vertex>& verts, std::vector<uint32_t>& idx) {
    const float inner = 1.9f, outer = 2.3f, z = -0.05f;  // > cube diagonal (~1.732)
    const float halfPi = 1.57079633f, gap = 0.05f;
    const int   seg = 12;
    const float col[4][3] = {
        {0.32f, 0.22f, 0.22f},  // right
        {0.22f, 0.32f, 0.22f},  // top
        {0.22f, 0.24f, 0.34f},  // left
        {0.32f, 0.30f, 0.20f},  // bottom
    };
    for (int s = 0; s < 4; ++s) {
        float a0 = s * halfPi - halfPi * 0.5f + gap;
        float a1 = s * halfPi + halfPi * 0.5f - gap;
        for (int i = 0; i < seg; ++i) {
            float t0 = a0 + (a1 - a0) * i / seg;
            float t1 = a0 + (a1 - a0) * (i + 1) / seg;
            float pts[4][3] = {{std::cos(t0) * inner, std::sin(t0) * inner, z},
                               {std::cos(t0) * outer, std::sin(t0) * outer, z},
                               {std::cos(t1) * outer, std::sin(t1) * outer, z},
                               {std::cos(t1) * inner, std::sin(t1) * inner, z}};
            uint32_t base = static_cast<uint32_t>(verts.size());
            for (int c = 0; c < 4; ++c)
                verts.push_back({{pts[c][0], pts[c][1], pts[c][2]},
                                 {col[s][0], col[s][1], col[s][2]}});
            idx.insert(idx.end(),
                       {base, base + 1, base + 2, base, base + 2, base + 3});
        }
    }
}

// Thread-safe handoff for the async open-file dialog. SDL invokes the callback
// from a platform thread, so it just parks the chosen path under a mutex; the
// main loop drains it. `busy` guards against opening a second dialog.
struct FileDropResult {
    std::mutex               mtx;
    std::vector<std::string> paths;
    std::atomic<bool>        busy{false};
};

void SDLCALL onFilePicked(void* userdata, const char* const* filelist, int /*filter*/) {
    auto* res = static_cast<FileDropResult*>(userdata);
    if (filelist && filelist[0]) {  // null => error, empty => cancelled
        std::lock_guard<std::mutex> lk(res->mtx);
        res->paths.emplace_back(filelist[0]);
    }
    res->busy.store(false);
}

// A single background mesh-load in flight. The worker thread parses the file
// (the slow part) into CPU arrays and reports progress via `percent`; the main
// thread polls `done`, then does the GPU upload + entity spawn (which must stay
// on the render thread). Only the worker touches the result vectors until it
// sets `done` (release), after which the main thread owns them -- so no lock is
// needed on `verts`/`indices`.
struct LoadJob {
    std::thread                 worker;
    std::atomic<int>            percent{0};      // 0..100, written by the worker
    std::atomic<bool>           done{false};     // worker finished (success or fail)
    std::atomic<bool>           ok{false};       // load succeeded
    std::string                 path;            // source file (for logging/status)
    std::vector<render::Vertex> verts;           // result (valid when done && ok)
    std::vector<uint32_t>       indices;
    bool                        active = false;  // main-thread only: a job exists

    ~LoadJob() { if (worker.joinable()) worker.join(); }
};

// Background Poisson reconstruction: same shape as LoadJob. Triggered by the
// dialog's "Reconstruct" button, it estimates normals + solves on a worker, and
// the main thread turns the resulting triangle soup into a new mesh entity.
struct PoissonJob {
    std::thread                 worker;
    std::atomic<int>            percent{0};
    std::atomic<bool>           done{false};
    std::atomic<bool>           ok{false};
    std::vector<render::Vertex> verts;
    std::vector<uint32_t>       indices;
    entt::entity                source = entt::null;  // the cloud entity to hide on success
    bool                        active = false;

    ~PoissonJob() { if (worker.joinable()) worker.join(); }
};

} // namespace

int main(int argc, char** argv) {
    SDL_SetMainReady();

    // Install first so any later crash prints a symbolized stack trace +
    // orange_crash.txt / orange_crash.dmp instead of vanishing silently.
    core::installCrashHandler();

    // Diagnostic: `appOrange --crashtest` deliberately faults here so you can
    // confirm the crash handler produces a stack trace (with file:line) and the
    // crash files. Harmless otherwise.
    bool shotMode = false;
    bool hangMode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--crashtest") == 0) {
            volatile int* p = nullptr;
            *p = 42;
        }
        // `appOrange --shot` renders a few frames, saves auto_shot.png, and exits
        // -- a deterministic headless screenshot for verifying the UI layout.
        if (std::strcmp(argv[i], "--shot") == 0) shotMode = true;
        // `appOrange --hangtest` deliberately freezes the main thread so you can
        // confirm the watchdog catches the hang (writes orange_hang.txt/.dmp).
        if (std::strcmp(argv[i], "--hangtest") == 0) hangMode = true;
    }
    int shotFrames = 0;
    int hangFrames = 0;

    // Console on the monitor right of the largest; 3D window maximizes on the
    // largest monitor (handled in Window::create).
    core::setupConsoleWindow();

    core::AppConfig config;
    config.title  = "Orange appOrange";
    config.width  = 1280;
    config.height = 720;
    config.backend = render::Backend::OpenGL;
    config.vsync   = true;   // vsync on by default (toggle via the FPS widget)

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--vulkan") == 0)
            config.backend = render::Backend::Vulkan;
        else if (std::strcmp(argv[i], "--gl") == 0)
            config.backend = render::Backend::OpenGL;
    }

    core::Application app;
    if (!app.init(config)) {
        SDL_Log("appOrange: initialization failed");
        return 1;
    }

    // --- Build the ECS world -------------------------------------------
    entt::registry& world = app.world();

    // Camera entity with a trackball controller orbiting the origin.
    // Right-drag orbits, wheel zooms, middle-drag pans, left-click picks.
    {
        auto cam = world.create();
        world.emplace<ecs::Transform>(cam);  // written each frame by the manipulator
        ecs::Camera c;
        c.primary = true;
        world.emplace<ecs::Camera>(cam, c);
        ecs::CameraManipulator manip;
        manip.target      = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        manip.distance    = 7.0f;
        manip.orientation = math::quatAxisAngle(Eigen::Vector3f(1, 0, 0), -0.3f);  // tilt down slightly
        // Home pose restored by the R key.
        manip.homeTarget      = manip.target;
        manip.homeDistance    = manip.distance;
        manip.homeOrientation = manip.orientation;
        world.emplace<ecs::CameraManipulator>(cam, manip);
    }

    // --- Axis gizmo cube: a [-1,1] cube with one solid color per face -------
    std::vector<render::Vertex> gVerts;
    std::vector<uint32_t>       gIdx;
    {
        struct Face { float c[4][3]; float col[3]; };
        const Face faces[] = {
            {{{ 1,-1,-1},{ 1, 1,-1},{ 1, 1, 1},{ 1,-1, 1}}, {0.85f, 0.25f, 0.25f}}, // +X
            {{{-1,-1, 1},{-1, 1, 1},{-1, 1,-1},{-1,-1,-1}}, {0.45f, 0.12f, 0.12f}}, // -X
            {{{-1, 1,-1},{-1, 1, 1},{ 1, 1, 1},{ 1, 1,-1}}, {0.30f, 0.80f, 0.35f}}, // +Y
            {{{-1,-1, 1},{-1,-1,-1},{ 1,-1,-1},{ 1,-1, 1}}, {0.13f, 0.42f, 0.18f}}, // -Y
            {{{-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1}}, {0.27f, 0.47f, 0.92f}}, // +Z
            {{{ 1,-1,-1},{-1,-1,-1},{-1, 1,-1},{ 1, 1,-1}}, {0.14f, 0.22f, 0.50f}}, // -Z
        };
        for (const auto& f : faces) {
            uint32_t base = static_cast<uint32_t>(gVerts.size());
            for (int i = 0; i < 4; ++i)
                gVerts.push_back({{f.c[i][0], f.c[i][1], f.c[i][2]},
                                  {f.col[0], f.col[1], f.col[2]}});
            gIdx.insert(gIdx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }
    }
    core::VertexBuffer<render::Vertex> gizmoVbo(*app.renderer(), gVerts);
    core::IndexBuffer                  gizmoIbo(*app.renderer(), gIdx);

    render::MeshDesc gizmoMesh;
    gizmoMesh.vertexBuffer = gizmoVbo.handle();
    gizmoMesh.indexBuffer  = gizmoIbo.handle();
    gizmoMesh.layout       = render::Vertex::layout();
    gizmoMesh.vertexCount  = static_cast<uint32_t>(gizmoVbo.count());
    gizmoMesh.indexCount   = static_cast<uint32_t>(gizmoIbo.count());

    // Axis labels (X/-X/Y/...): a textured quad per face sampling a text atlas
    // rasterized with Malgun Gothic.
    std::vector<unsigned char> fontTtf = loadFont();
    std::vector<uint8_t> atlasPx;
    int atlasW = 0, atlasH = 0;
    buildLabelAtlas(fontTtf, atlasPx, atlasW, atlasH);
    render::TextureDesc atlasDesc;
    atlasDesc.width  = static_cast<uint32_t>(atlasW);
    atlasDesc.height = static_cast<uint32_t>(atlasH);
    atlasDesc.pixels = atlasPx.data();
    render::TextureHandle labelTex = app.renderer()->createTexture(atlasDesc);

    // Shared proportional UI font (Malgun Gothic) for the FPS readout + controls.
    static core::Font uiFont;
    bakeFont(*app.renderer(), fontTtf, 40.0f, uiFont);

    std::vector<render::Vertex> labelV;
    std::vector<uint32_t>       labelI;
    buildLabelQuads(labelV, labelI);
    core::VertexBuffer<render::Vertex> labelVbo(*app.renderer(), labelV);
    core::IndexBuffer                  labelIbo(*app.renderer(), labelI);
    render::MeshDesc labelMesh;
    labelMesh.vertexBuffer = labelVbo.handle();
    labelMesh.indexBuffer  = labelIbo.handle();
    labelMesh.layout       = render::Vertex::layout();
    labelMesh.vertexCount  = static_cast<uint32_t>(labelVbo.count());
    labelMesh.indexCount   = static_cast<uint32_t>(labelIbo.count());

    // Ring background mesh (static).
    std::vector<render::Vertex> ringV;
    std::vector<uint32_t>       ringI;
    buildRing(ringV, ringI);
    core::VertexBuffer<render::Vertex> ringVbo(*app.renderer(), ringV);
    core::IndexBuffer                  ringIbo(*app.renderer(), ringI);
    render::MeshDesc ringMesh;
    ringMesh.vertexBuffer = ringVbo.handle();
    ringMesh.indexBuffer  = ringIbo.handle();
    ringMesh.layout       = render::Vertex::layout();
    ringMesh.vertexCount  = static_cast<uint32_t>(ringVbo.count());
    ringMesh.indexCount   = static_cast<uint32_t>(ringIbo.count());

    // Highlight patch: a dynamic mesh of up to 12 quads (48 verts), rewritten
    // each frame for the hovered/clicked cube region (thin outline) or ring sector.
    const int kHlQuads = 12, kHlVerts = kHlQuads * 4;
    const std::vector<render::Vertex> hlInit(kHlVerts, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> hlIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kHlQuads); ++q) {
        uint32_t b = q * 4;
        hlIdx.insert(hlIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> hlVbo(*app.renderer(), hlInit,
                                             render::BufferUsage::Dynamic);
    core::IndexBuffer                  hlIbo(*app.renderer(), hlIdx);
    render::MeshDesc hlMeshDesc;
    hlMeshDesc.vertexBuffer = hlVbo.handle();
    hlMeshDesc.indexBuffer  = hlIbo.handle();
    hlMeshDesc.layout       = render::Vertex::layout();
    hlMeshDesc.vertexCount  = static_cast<uint32_t>(kHlVerts);
    hlMeshDesc.indexCount   = static_cast<uint32_t>(kHlQuads * 6);

    // Gizmo up-axis (Y/Z) toggle button: a small dynamic mesh (panel + glyph)
    // drawn in a corner of the gizmo. Capacity must match kUpBtnQuads in systems.cpp.
    const int kUpQ = 16, kUpV = kUpQ * 4;
    const std::vector<render::Vertex> upInit(kUpV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> upIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kUpQ); ++q) {
        uint32_t b = q * 4;
        upIdx.insert(upIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> upBtnVbo(*app.renderer(), upInit,
                                                render::BufferUsage::Dynamic);
    core::IndexBuffer                  upBtnIbo(*app.renderer(), upIdx);
    render::MeshDesc upBtnMeshDesc;
    upBtnMeshDesc.vertexBuffer = upBtnVbo.handle();
    upBtnMeshDesc.indexBuffer  = upBtnIbo.handle();
    upBtnMeshDesc.layout       = render::Vertex::layout();
    upBtnMeshDesc.vertexCount  = static_cast<uint32_t>(kUpV);
    upBtnMeshDesc.indexCount   = static_cast<uint32_t>(kUpQ * 6);

    {
        auto giz = world.create();
        ecs::AxisGizmo gizmo;
        gizmo.mesh          = app.renderer()->createMesh(gizmoMesh);
        gizmo.labelMesh     = app.renderer()->createMesh(labelMesh);
        gizmo.labelTexture  = labelTex;
        gizmo.ringMesh      = app.renderer()->createMesh(ringMesh);
        gizmo.highlightMesh = app.renderer()->createMesh(hlMeshDesc);
        gizmo.highlightVbo  = hlVbo.handle();
        gizmo.upBtnMesh     = app.renderer()->createMesh(upBtnMeshDesc);
        gizmo.upBtnVbo      = upBtnVbo.handle();
        gizmo.font          = &uiFont;            // shared proportional UI font
        gizmo.uiAtlas       = uiFont.texture;
        world.emplace<ecs::AxisGizmo>(giz, gizmo);
    }

    // Draggable FPS widget: dynamic vertex buffer (rewritten each frame) +
    // static quad index pattern.
    const int kFpsQ = 256, kFpsV = kFpsQ * 4;  // must match kFpsQuads in systems.cpp
    const std::vector<render::Vertex> fpsInit(kFpsV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> fpsIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kFpsQ); ++q) {
        uint32_t b = q * 4;
        fpsIdx.insert(fpsIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> fpsVbo(*app.renderer(), fpsInit,
                                              render::BufferUsage::Dynamic);
    core::IndexBuffer                  fpsIbo(*app.renderer(), fpsIdx);
    render::MeshDesc fpsMeshDesc;
    fpsMeshDesc.vertexBuffer = fpsVbo.handle();
    fpsMeshDesc.indexBuffer  = fpsIbo.handle();
    fpsMeshDesc.layout       = render::Vertex::layout();
    fpsMeshDesc.vertexCount  = static_cast<uint32_t>(kFpsV);
    fpsMeshDesc.indexCount   = static_cast<uint32_t>(kFpsQ * 6);
    {
        auto e = world.create();
        ecs::FpsWidget widget;
        widget.mesh  = app.renderer()->createMesh(fpsMeshDesc);
        widget.vbo   = fpsVbo.handle();
        widget.font  = &uiFont;            // shared proportional font
        widget.atlas = uiFont.texture;
        widget.vsync = config.vsync;       // reflect the actual initial state
        world.emplace<ecs::FpsWidget>(e, widget);
    }

    // Draggable tree-view (scene outliner): same dynamic-VB + quad-index pattern
    // as the FPS widget. kTreeQ must match kTreeQuads in systems.cpp.
    const int kTreeQ = 600, kTreeV = kTreeQ * 4;
    const std::vector<render::Vertex> treeInit(kTreeV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> treeIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kTreeQ); ++q) {
        uint32_t b = q * 4;
        treeIdx.insert(treeIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> treeVbo(*app.renderer(), treeInit,
                                               render::BufferUsage::Dynamic);
    core::IndexBuffer treeIbo(*app.renderer(), treeIdx);
    render::MeshDesc treeMeshDesc;
    treeMeshDesc.vertexBuffer = treeVbo.handle();
    treeMeshDesc.indexBuffer  = treeIbo.handle();
    treeMeshDesc.layout       = render::Vertex::layout();
    treeMeshDesc.vertexCount  = static_cast<uint32_t>(kTreeV);
    treeMeshDesc.indexCount   = static_cast<uint32_t>(kTreeQ * 6);
    {
        auto e = world.create();
        ecs::TreeView tv;
        tv.mesh  = app.renderer()->createMesh(treeMeshDesc);
        tv.vbo   = treeVbo.handle();
        tv.font  = &uiFont;
        tv.atlas = uiFont.texture;
        world.emplace<ecs::TreeView>(e, tv);
    }

    // Restore the draggable widgets (FPS + tree) to their last-saved positions;
    // saved again after the run loop so each launch reopens where they were left.
    const char* uiBase = SDL_GetBasePath();  // owned by SDL, do not free
    const std::string uiLayoutPath =
        (uiBase ? std::string(uiBase) : std::string()) + "orange_ui_layout.txt";
    const std::string lastMeshFile =
        (uiBase ? std::string(uiBase) : std::string()) + "orange_last_mesh.txt";
    core::loadWidgetLayout(world, uiLayoutPath);

    // Camera controls panel (projection toggle + FOV/size), under the gizmo.
    const int kCtrlQ = 64, kCtrlV = kCtrlQ * 4;
    const std::vector<render::Vertex> ctrlInit(kCtrlV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> ctrlIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kCtrlQ); ++q) {
        uint32_t b = q * 4;
        ctrlIdx.insert(ctrlIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> ctrlVbo(*app.renderer(), ctrlInit,
                                               render::BufferUsage::Dynamic);
    core::IndexBuffer                  ctrlIbo(*app.renderer(), ctrlIdx);
    render::MeshDesc ctrlMeshDesc;
    ctrlMeshDesc.vertexBuffer = ctrlVbo.handle();
    ctrlMeshDesc.indexBuffer  = ctrlIbo.handle();
    ctrlMeshDesc.layout       = render::Vertex::layout();
    ctrlMeshDesc.vertexCount  = static_cast<uint32_t>(kCtrlV);
    ctrlMeshDesc.indexCount   = static_cast<uint32_t>(kCtrlQ * 6);
    {
        auto e = world.create();
        ecs::CameraControls cc;
        cc.font  = &uiFont;
        cc.atlas = uiFont.texture;
        cc.mesh  = app.renderer()->createMesh(ctrlMeshDesc);
        cc.vbo   = ctrlVbo.handle();
        world.emplace<ecs::CameraControls>(e, cc);
    }

    // Cross-section panel (clip-plane slider), under the camera controls.
    const int kCsQ = 64, kCsV = kCsQ * 4;  // must match kCsQuads in systems.cpp
    const std::vector<render::Vertex> csInit(kCsV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> csIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kCsQ); ++q) {
        uint32_t b = q * 4;
        csIdx.insert(csIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> csVbo(*app.renderer(), csInit,
                                             render::BufferUsage::Dynamic);
    core::IndexBuffer                  csIbo(*app.renderer(), csIdx);
    render::MeshDesc csMeshDesc;
    csMeshDesc.vertexBuffer = csVbo.handle();
    csMeshDesc.indexBuffer  = csIbo.handle();
    csMeshDesc.layout       = render::Vertex::layout();
    csMeshDesc.vertexCount  = static_cast<uint32_t>(kCsV);
    csMeshDesc.indexCount   = static_cast<uint32_t>(kCsQ * 6);
    {
        auto e = world.create();
        ecs::CrossSection cs;
        cs.font  = &uiFont;
        cs.atlas = uiFont.texture;
        cs.mesh  = app.renderer()->createMesh(csMeshDesc);
        cs.vbo   = csVbo.handle();
        world.emplace<ecs::CrossSection>(e, cs);
    }

    // Poisson reconstruction dialog (parameter sliders + Reconstruct button).
    const int kPoissonQ = 256, kPoissonV = kPoissonQ * 4;  // must match kPoissonQuads in systems.cpp
    const std::vector<render::Vertex> pdInit(kPoissonV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> pdIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kPoissonQ); ++q) {
        uint32_t b = q * 4;
        pdIdx.insert(pdIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> pdVbo(*app.renderer(), pdInit,
                                             render::BufferUsage::Dynamic);
    core::IndexBuffer                  pdIbo(*app.renderer(), pdIdx);
    render::MeshDesc pdMeshDesc;
    pdMeshDesc.vertexBuffer = pdVbo.handle();
    pdMeshDesc.indexBuffer  = pdIbo.handle();
    pdMeshDesc.layout       = render::Vertex::layout();
    pdMeshDesc.vertexCount  = static_cast<uint32_t>(kPoissonV);
    pdMeshDesc.indexCount   = static_cast<uint32_t>(kPoissonQ * 6);
    {
        auto e = world.create();
        ecs::PoissonDialog pd;
        pd.font  = &uiFont;
        pd.atlas = uiFont.texture;
        pd.mesh  = app.renderer()->createMesh(pdMeshDesc);
        pd.vbo   = pdVbo.handle();
        world.emplace<ecs::PoissonDialog>(e, pd);
    }

    // Confirm (Yes/No) dialog -- in-app modal used by the "load last mesh" prompt.
    const int kConfirmQ = 192, kConfirmV = kConfirmQ * 4;  // must match kConfirmQuads in systems.cpp
    const std::vector<render::Vertex> cdInit(kConfirmV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> cdIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kConfirmQ); ++q) {
        uint32_t b = q * 4;
        cdIdx.insert(cdIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> cdVbo(*app.renderer(), cdInit,
                                             render::BufferUsage::Dynamic);
    core::IndexBuffer                  cdIbo(*app.renderer(), cdIdx);
    render::MeshDesc cdMeshDesc;
    cdMeshDesc.vertexBuffer = cdVbo.handle();
    cdMeshDesc.indexBuffer  = cdIbo.handle();
    cdMeshDesc.layout       = render::Vertex::layout();
    cdMeshDesc.vertexCount  = static_cast<uint32_t>(kConfirmV);
    cdMeshDesc.indexCount   = static_cast<uint32_t>(kConfirmQ * 6);
    {
        auto e = world.create();
        ecs::ConfirmDialog cd;
        cd.font  = &uiFont;
        cd.atlas = uiFont.texture;
        cd.mesh  = app.renderer()->createMesh(cdMeshDesc);
        cd.vbo   = cdVbo.handle();
        world.emplace<ecs::ConfirmDialog>(e, cd);
    }

    // Occlusal 2D render: a shared unit quad (uv 0..1) for the channel panels.
    const std::vector<render::Vertex> orQuad = {
        {{0, 0, 0}, {1, 1, 1}, {0, 1}}, {{1, 0, 0}, {1, 1, 1}, {1, 1}},
        {{1, 1, 0}, {1, 1, 1}, {1, 0}}, {{0, 1, 0}, {1, 1, 1}, {0, 0}},
    };
    const std::vector<uint32_t> orQuadIdx = {0, 1, 2, 0, 2, 3};
    core::VertexBuffer<render::Vertex> orVbo(*app.renderer(), orQuad);
    core::IndexBuffer                  orIbo(*app.renderer(), orQuadIdx);
    render::MeshDesc orMeshDesc;
    orMeshDesc.vertexBuffer = orVbo.handle();
    orMeshDesc.indexBuffer  = orIbo.handle();
    orMeshDesc.layout       = render::Vertex::layout();
    orMeshDesc.vertexCount  = 4;
    orMeshDesc.indexCount   = 6;
    {
        auto e = world.create();
        ecs::OcclusalRenderViz orv;
        orv.quad = app.renderer()->createMesh(orMeshDesc);
        world.emplace<ecs::OcclusalRenderViz>(e, orv);
    }

    // Top menu bar (multi-menu). Dynamic vertex buffer rewritten each frame.
    const int kMenuQ = 1024, kMenuV = kMenuQ * 4;  // must match kMenuQuads in systems.cpp
    const std::vector<render::Vertex> menuInit(kMenuV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> menuIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kMenuQ); ++q) {
        uint32_t b = q * 4;
        menuIdx.insert(menuIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> menuVbo(*app.renderer(), menuInit,
                                               render::BufferUsage::Dynamic);
    core::IndexBuffer                  menuIbo(*app.renderer(), menuIdx);
    render::MeshDesc menuMeshDesc;
    menuMeshDesc.vertexBuffer = menuVbo.handle();
    menuMeshDesc.indexBuffer  = menuIbo.handle();
    menuMeshDesc.layout       = render::Vertex::layout();
    menuMeshDesc.vertexCount  = static_cast<uint32_t>(kMenuV);
    menuMeshDesc.indexCount   = static_cast<uint32_t>(kMenuQ * 6);
    {
        auto e = world.create();
        ecs::MenuBar mb;
        mb.mesh  = app.renderer()->createMesh(menuMeshDesc);
        mb.vbo   = menuVbo.handle();
        mb.font  = &uiFont;
        mb.atlas = uiFont.texture;
        mb.menus = ecs::defaultAppMenus();
        world.emplace<ecs::MenuBar>(e, mb);
    }

    // Left selection-mode toolbar. Dynamic vertex buffer rewritten each frame.
    const int kTbQ = 512, kTbV = kTbQ * 4;  // must match kTbQuads in systems.cpp
    const std::vector<render::Vertex> tbInit(kTbV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> tbIdx;
    for (uint32_t qi = 0; qi < static_cast<uint32_t>(kTbQ); ++qi) {
        uint32_t b = qi * 4;
        tbIdx.insert(tbIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> tbVbo(*app.renderer(), tbInit,
                                             render::BufferUsage::Dynamic);
    core::IndexBuffer                  tbIbo(*app.renderer(), tbIdx);
    render::MeshDesc tbMeshDesc;
    tbMeshDesc.vertexBuffer = tbVbo.handle();
    tbMeshDesc.indexBuffer  = tbIbo.handle();
    tbMeshDesc.layout       = render::Vertex::layout();
    tbMeshDesc.vertexCount  = static_cast<uint32_t>(kTbV);
    tbMeshDesc.indexCount   = static_cast<uint32_t>(kTbQ * 6);
    {
        auto e = world.create();
        ecs::SelectionToolbar tb;
        tb.mesh    = app.renderer()->createMesh(tbMeshDesc);
        tb.vbo     = tbVbo.handle();
        tb.font    = &uiFont;
        tb.atlas   = uiFont.texture;
        tb.buttons = ecs::defaultSelectionToolbar();
        world.emplace<ecs::SelectionToolbar>(e, tb);
    }
    world.ctx().emplace<ecs::SelectionMode>();

    // --- File > Open... -> load a mesh ---------------------------------
    // Buffers for loaded meshes must outlive the renderer, so they live here
    // (destroyed before `app`) and are kept alive past the loading scope.
    render::IRenderer* renderer = app.renderer();
    std::vector<std::unique_ptr<core::VertexBuffer<render::Vertex>>> loadedVbos;
    std::vector<std::unique_ptr<core::IndexBuffer>>                  loadedIbos;

    FileDropResult fileDrop;
    static const SDL_DialogFileFilter kMeshFilters[] = {
        {"Mesh files (OBJ, STL, PLY)", "obj;stl;ply"},
        {"All files", "*"},
    };

    // Takes the CPU mesh produced by the background loader, uploads GPU buffers
    // at its original coordinates, and spawns a
    // (Transform, Renderable) entity that renderSystem picks up. Runs on the main
    // (render) thread because GPU resource creation is context-affine.
    auto finalizeMesh = [&](const std::string& path, std::vector<render::Vertex>& mv,
                            std::vector<uint32_t>& mi) {
        if (mv.empty()) {
            SDL_Log("Mesh: failed to load '%s'", path.c_str());
            return;
        }
        Eigen::Vector3f mn(mv[0].position[0], mv[0].position[1], mv[0].position[2]);
        Eigen::Vector3f mx = mn;
        for (const auto& v : mv) {
            Eigen::Vector3f p(v.position[0], v.position[1], v.position[2]);
            mn = mn.cwiseMin(p);
            mx = mx.cwiseMax(p);
        }
        const bool isPoints = mi.empty();  // faceless PLY -> point cloud

        loadedVbos.push_back(
            std::make_unique<core::VertexBuffer<render::Vertex>>(*renderer, mv));
        render::MeshDesc md;
        md.vertexBuffer = loadedVbos.back()->handle();
        md.layout       = render::Vertex::layout();
        md.vertexCount  = static_cast<uint32_t>(loadedVbos.back()->count());
        if (isPoints) {
            md.topology = render::PrimitiveTopology::Points;  // sphere-imposter points
        } else {
            loadedIbos.push_back(std::make_unique<core::IndexBuffer>(*renderer, mi));
            md.indexBuffer = loadedIbos.back()->handle();
            md.indexCount  = static_cast<uint32_t>(loadedIbos.back()->count());
        }
        render::MeshHandle mesh = renderer->createMesh(md);

        auto e = world.create();
        ecs::Transform t;
        t.scale = Eigen::Vector3f(1.0f, 1.0f, 1.0f);  // draw at original coordinates
        world.emplace<ecs::Transform>(e, t);
        ecs::Renderable r;
        r.mesh      = mesh;
        r.boundsMin = mn;  // local-space bounds for picking (original coords)
        r.boundsMax = mx;
        if (isPoints) r.pointCloud = true;  // drawn as point sprites; box-wireframe selection
        world.emplace<ecs::Renderable>(e, r);

        // CPU geometry for accurate picking: triangles for real meshes, or the raw
        // point positions for a point cloud (picked by proximity, so empty space
        // inside the AABB doesn't select). indices stays empty for a point cloud.
        ecs::PickGeometry pick;
        pick.positions.reserve(mv.size());
        for (const auto& v : mv)
            pick.positions.emplace_back(v.position[0], v.position[1], v.position[2]);
        if (!isPoints) pick.indices = mi;
        world.emplace<ecs::PickGeometry>(e, std::move(pick));

        // Frame the camera on the just-loaded mesh: orbit pivot -> bounds center,
        // distance -> fit the bounding sphere to the vertical FOV (+ margin). Also
        // updates the home pose so R resets to this framing, and widens the zoom
        // range so big/far models don't clip.
        {
            auto camv = world.view<ecs::Camera, ecs::CameraManipulator>();
            for (auto ce : camv) {
                auto& cam = camv.get<ecs::Camera>(ce);
                if (!cam.primary) continue;
                auto& m      = camv.get<ecs::CameraManipulator>(ce);
                Eigen::Vector3f c = (mn + mx) * 0.5f;
                float radius = (mx - mn).norm() * 0.5f;
                float fovY   = cam.fovYDegrees * 3.14159265f / 180.0f;
                float dist   = radius / std::sin(fovY * 0.5f) * 1.3f;  // margin
                dist         = (std::max)(dist, 0.01f);
                m.minDistance = (std::min)(m.minDistance, dist * 0.05f);
                m.maxDistance = (std::max)(m.maxDistance, dist * 4.0f);
                // Clip planes scaled to the fit so big/far models don't z-clip.
                cam.zNear = (std::max)(0.001f, dist * 0.01f);
                cam.zFar  = (m.maxDistance + radius) * 1.5f;
                m.target = c; m.distance = dist;
                m.targetAnimating = false; m.animating = false;
                m.homeTarget = c; m.homeDistance = dist;
                m.homeOrientation = m.orientation;
                break;
            }
        }

        if (isPoints)
            SDL_Log("Mesh: loaded '%s' (%zu points)", path.c_str(), mv.size());
        else
            SDL_Log("Mesh: loaded '%s' (%zu verts, %zu tris)", path.c_str(), mv.size(),
                    mi.size() / 3);
    };

    LoadJob    loadJob;
    PoissonJob poissonJob;
    float      statusHold = 0.0f;  // seconds to keep the final "Loaded/Failed" message

    auto onUpdate = [&](entt::registry& w, float dt) {
        core::watchdogHeartbeat();  // tell the watchdog the main loop is alive

        // Headless screenshot mode: let the scene settle, capture, then exit.
        if (shotMode && ++shotFrames == 30) {
            core::saveScreenshot(*app.renderer(), "auto_shot.png");
            std::exit(0);
        }
        // Hang test: freeze the main thread so the watchdog can catch it.
        if (hangMode && ++hangFrames == 30) {
            volatile int x = 0;
            for (;;) ++x;  // deliberate infinite loop (never returns)
        }

        ecs::MenuBar* mb = nullptr;
        auto          mbv = w.view<ecs::MenuBar>();
        for (auto e : mbv) { mb = &mbv.get<ecs::MenuBar>(e); break; }

        // Menu raised a request? Open the native dialog (unless one is already up).
        if (mb && mb->requestOpenFile) {
            mb->requestOpenFile = false;
            if (!fileDrop.busy.exchange(true)) {
                SDL_ShowOpenFileDialog(onFilePicked, &fileDrop, app.window().handle(),
                                       kMeshFilters, 2, nullptr, /*allow_many=*/false);
            }
        }
        // In-app confirm dialog answered? A Yes on the "load last mesh" prompt
        // queues that path like a normal open.
        {
            auto cdv = w.view<ecs::ConfirmDialog>();
            for (auto e : cdv) {
                auto& cd = cdv.get<ecs::ConfirmDialog>(e);
                if (cd.answered) {
                    cd.answered = false;
                    if (cd.yes && !cd.payload.empty()) {
                        std::lock_guard<std::mutex> lk(fileDrop.mtx);
                        fileDrop.paths.push_back(cd.payload);
                    }
                }
                break;
            }
        }
        // Drain whatever the dialog callback handed back (it runs on a platform
        // thread). Only one load runs at a time -- a pick while busy is dropped.
        std::string path;
        {
            std::lock_guard<std::mutex> lk(fileDrop.mtx);
            if (!fileDrop.paths.empty()) {
                path = fileDrop.paths.front();
                fileDrop.paths.clear();
            }
        }
        if (!path.empty() && !loadJob.active) {
            loadJob.active = true;
            loadJob.path   = path;
            loadJob.percent.store(0);
            loadJob.done.store(false);
            loadJob.ok.store(false);
            loadJob.verts.clear();
            loadJob.indices.clear();
            // Parse the file off the main thread, reporting progress into `percent`.
            loadJob.worker = std::thread([job = &loadJob] {
                std::vector<render::Vertex> v;
                std::vector<uint32_t>       i;
                bool ok = meshio::loadMeshFile(
                              job->path, v, i,
                              [job](float p) {
                                  job->percent.store(static_cast<int>(p * 100.0f + 0.5f));
                              }) &&
                          !v.empty();
                if (ok) { job->verts = std::move(v); job->indices = std::move(i); }
                job->percent.store(100);
                job->ok.store(ok);
                job->done.store(true);  // publish last; main thread now owns the result
            });
        }

        // Drive status text + finalize a finished load on the main thread.
        if (loadJob.active) {
            if (!loadJob.done.load()) {
                if (mb) mb->statusText = "Loading " + std::to_string(loadJob.percent.load()) + "%";
            } else {
                loadJob.worker.join();
                bool ok = loadJob.ok.load();
                if (ok) {
                    finalizeMesh(loadJob.path, loadJob.verts, loadJob.indices);
                    std::ofstream(lastMeshFile, std::ios::trunc) << loadJob.path;  // remember for next launch
                }
                loadJob.verts.clear();   loadJob.verts.shrink_to_fit();
                loadJob.indices.clear(); loadJob.indices.shrink_to_fit();
                loadJob.active = false;
                statusHold     = ok ? 1.0f : 3.0f;  // linger on the final message
                if (mb) mb->statusText = ok ? "Loaded 100%" : "Load failed";
            }
        } else if (statusHold > 0.0f) {
            statusHold -= dt;
            if (statusHold <= 0.0f && mb) mb->statusText.clear();
        } else {
            // No load in progress: surface any background processing-mode or
            // spatial-structure build progress.
            float pct = 0.0f;
            std::string name;
            if (ecs::processingModeProgress(w, pct, name) || ecs::spatialVizProgress(w, pct, name)) {
                if (mb) mb->statusText = name + " " + std::to_string((int)(pct * 100.0f + 0.5f)) + "%";
            } else if (mb && !mb->statusText.empty()) {
                mb->statusText.clear();  // finished -> clear the line
            }
        }

        // --- Poisson reconstruction (dialog "Reconstruct" button) -------------
        ecs::PoissonDialog* pd = nullptr;
        auto pdv = w.view<ecs::PoissonDialog>();
        for (auto e : pdv) { pd = &pdv.get<ecs::PoissonDialog>(e); break; }

        if (pd && pd->requestRun) {
            pd->requestRun = false;
            if (!poissonJob.active) {
                // Gather the first selected point cloud, in world space.
                std::vector<Eigen::Vector3f> pts;
                entt::entity srcEntity = entt::null;
                auto rv = w.view<ecs::Renderable, ecs::PickGeometry>();
                for (auto e : rv) {
                    if (!rv.get<ecs::Renderable>(e).selected) continue;
                    const auto& pg = rv.get<ecs::PickGeometry>(e);
                    if (pg.positions.empty()) break;
                    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
                    if (w.all_of<ecs::Transform>(e)) M = w.get<ecs::Transform>(e).matrix();
                    pts.reserve(pg.positions.size());
                    for (const auto& p : pg.positions) {
                        Eigen::Vector4f wp = M * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
                        pts.emplace_back(wp.x(), wp.y(), wp.z());
                    }
                    srcEntity = e;
                    break;
                }
                if (pts.size() >= 4) {
                    geometry::PoissonParams pp;
                    pp.depth       = pd->depth;
                    pp.iterations  = pd->iterations;
                    pp.scale       = pd->scale;
                    pp.pointWeight = pd->pointWeight;
                    poissonJob.active = true;
                    poissonJob.done.store(false);
                    poissonJob.ok.store(false);
                    poissonJob.percent.store(0);
                    poissonJob.verts.clear();
                    poissonJob.indices.clear();
                    poissonJob.source = srcEntity;
                    poissonJob.worker = std::thread(
                        [job = &poissonJob, pts = std::move(pts), pp]() mutable {
                            bool ok = false;
                            // A deep depth on a dense grid can exhaust RAM; catch the
                            // bad_alloc so it surfaces as "Poisson failed", not a crash.
                            try {
                                auto nrm = geometry::estimateNormals(
                                    pts, 16, [job](float f) { job->percent.store((int)(f * 30.0f)); });
                                auto tris = geometry::poissonReconstruct(
                                    pts, nrm, pp,
                                    [job](float f) { job->percent.store(30 + (int)(f * 70.0f)); });
                                std::vector<render::Vertex> v;
                                std::vector<uint32_t>       idx;
                                v.reserve(tris.size() * 3);
                                idx.reserve(tris.size() * 3);
                                uint32_t c = 0;
                                for (const auto& t : tris)
                                    for (int k = 0; k < 3; ++k) {
                                        render::Vertex vv{};
                                        vv.position[0] = t.v[k].x(); vv.position[1] = t.v[k].y();
                                        vv.position[2] = t.v[k].z();
                                        vv.color[0] = t.c[k].x(); vv.color[1] = t.c[k].y();
                                        vv.color[2] = t.c[k].z();
                                        v.push_back(vv);
                                        idx.push_back(c++);
                                    }
                                ok = !v.empty();
                                if (ok) { job->verts = std::move(v); job->indices = std::move(idx); }
                            } catch (...) {
                                ok = false;  // out of memory (deep depth) or other failure
                            }
                            job->percent.store(100);
                            job->ok.store(ok);
                            job->done.store(true);
                        });
                } else if (mb) {
                    mb->statusText = "Poisson: select a point cloud first";
                    statusHold     = 2.5f;
                }
            }
        }

        // Drive Poisson status + finalize a finished reconstruction (main thread).
        if (poissonJob.active) {
            if (!poissonJob.done.load()) {
                if (mb)
                    mb->statusText = "Poisson " + std::to_string(poissonJob.percent.load()) + "%";
            } else {
                poissonJob.worker.join();
                bool ok = poissonJob.ok.load();
                if (ok) {
                    finalizeMesh("poisson", poissonJob.verts, poissonJob.indices);
                    // Hide the source point cloud so the new mesh isn't buried under
                    // its sprites. drawMode None (not visible=false) so the H key
                    // (Unhide All) brings it back.
                    if (w.valid(poissonJob.source) && w.all_of<ecs::Renderable>(poissonJob.source))
                        w.get<ecs::Renderable>(poissonJob.source).drawMode = core::DrawMode::None;
                }
                poissonJob.verts.clear();   poissonJob.verts.shrink_to_fit();
                poissonJob.indices.clear(); poissonJob.indices.shrink_to_fit();
                poissonJob.active = false;
                statusHold        = ok ? 1.0f : 3.0f;
                if (mb) mb->statusText = ok ? "Poisson done" : "Poisson failed";
            }
        }
    };

    SDL_Log("appOrange: running. File > Open... loads a mesh (OBJ/STL).");
    SDL_Log("appOrange: click selects (Ctrl+click toggles, Ctrl+A all on-screen, empty clears, "
            "Delete removes); Tab cycles the selection's drawing mode (H reveals None-hidden meshes); "
            "+/- resize point-cloud sprites.");
    SDL_Log("appOrange: ESC or close the window to quit.");

    // Offer to reload the mesh from the previous session via the in-app dialog;
    // a Yes is handled in onUpdate (queues the load like a normal open).
    {
        std::ifstream in(lastMeshFile);
        std::string last;
        std::getline(in, last);
        in.close();
        std::ifstream test(last, std::ios::binary);
        if (!last.empty() && test.good()) {
            test.close();
            auto cdv = world.view<ecs::ConfirmDialog>();
            for (auto e : cdv) {
                auto& cd  = cdv.get<ecs::ConfirmDialog>(e);
                cd.line1   = "Load last mesh?";
                cd.line2   = last;
                cd.payload = last;
                cd.visible = true;
                break;
            }
        }
    }

    // Watch for a main-thread hang (shorter timeout in the hang self-test).
    core::installWatchdog(hangMode ? 2.0 : 10.0);
    app.run(onUpdate);  // onUpdate + spinSystem + renderSystem run each frame
    core::stopWatchdog();

    // Persist the draggable widgets' final positions for next launch.
    core::saveWidgetLayout(world, uiLayoutPath);
    return 0;
}
