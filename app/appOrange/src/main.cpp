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
#include "orange/core/color.h"
#include "orange/core/compare.h"
#include "orange/core/crash_handler.h"
#include "orange/core/modes.h"
#include "orange/core/screenshot.h"
#include "orange/core/ui_layout.h"
#include "orange/core/buffer.h"
#include "orange/core/console.h"
#include "orange/core/math.h"
#include "orange/core/normals.h"
#include "orange/core/poisson_reconstruction.h"
#include "orange/ecs/components.h"
#include "orange/ecs/systems.h"
#include "orange/ecs/undo.h"
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
    std::vector<meshio::V3>     normals;         // file's own normals (may be empty)
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

// App-local component: the file a mesh entity was loaded from (or last saved
// to). File > Save overwrites this path; absent (e.g. Create-menu primitives)
// => Save falls back to the Save As dialog.
struct SourcePath {
    std::string path;
};

// Background 3D Compare (Geomagic-style deviation analysis): the worker
// computes signed distances of the test drawable's points to the reference
// surface; the main thread paints them as a blue-green-red heatmap (undoable).
struct CompareJob {
    std::thread             worker;
    std::atomic<int>        percent{0};
    std::atomic<bool>       done{false};
    std::atomic<bool>       ok{false};
    std::vector<float>      devs;              // signed deviation per test vertex
    geometry::CompareStats  stats;
    entt::entity            test = entt::null; // the entity that gets painted
    bool                    active = false;

    ~CompareJob() { if (worker.joinable()) worker.join(); }
};

// Recolor a drawable in place: upload `verts` into its existing vertex buffer
// and keep the CPU copy in sync. Color-only (same vertex count/topology), works
// for meshes and point clouds alike. Shared by the compare apply and its
// undo/redo closures.
void paintVertices(entt::registry& w, render::IRenderer& r, entt::entity e,
                   const std::vector<render::Vertex>& verts) {
    if (!w.valid(e) || !w.all_of<ecs::VertexSource>(e)) return;
    auto& vs = w.get<ecs::VertexSource>(e);
    if (vs.vertices.size() != verts.size()) return;
    vs.vertices = verts;
    r.updateBuffer(vs.vbo, verts.data(), verts.size() * sizeof(render::Vertex));
}

// Background mesh save: same shape as LoadJob. The main thread snapshots the
// entity's CPU geometry (VertexSource + PickGeometry, transform baked in), the
// worker writes the file, the main thread stamps SourcePath on success.
struct SaveJob {
    std::thread                 worker;
    std::atomic<bool>           done{false};
    std::atomic<bool>           ok{false};
    std::string                 path;
    std::vector<render::Vertex> verts;    // snapshot: worker-owned while active
    std::vector<uint32_t>       indices;
    std::vector<meshio::V3>     normals;  // source normals to write back (may be empty)
    entt::entity                target = entt::null;  // gets SourcePath on success
    bool                        active = false;

    ~SaveJob() { if (worker.joinable()) worker.join(); }
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
    bool pipeMode = false;
    bool legendMode = false;
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
        // `appOrange --shot --pipeline` opens the Pipeline Design dialog before
        // the capture -- headless verification of the node-canvas UI.
        if (std::strcmp(argv[i], "--pipeline") == 0) pipeMode = true;
        // `appOrange --shot --legend` shows the color-bar legend with synthetic
        // mode-legend values (range thumbs included) for headless UI checks.
        if (std::strcmp(argv[i], "--legend") == 0) legendMode = true;
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

    // 3D Compare legend (banded color bar + numeric ticks), under the section panel.
    const int kLgQ = 128, kLgV = kLgQ * 4;  // must match kLegendQuads in systems.cpp
    const std::vector<render::Vertex> lgInit(kLgV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> lgIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kLgQ); ++q) {
        uint32_t b = q * 4;
        lgIdx.insert(lgIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> lgVbo(*app.renderer(), lgInit,
                                             render::BufferUsage::Dynamic);
    core::IndexBuffer                  lgIbo(*app.renderer(), lgIdx);
    render::MeshDesc lgMeshDesc;
    lgMeshDesc.vertexBuffer = lgVbo.handle();
    lgMeshDesc.indexBuffer  = lgIbo.handle();
    lgMeshDesc.layout       = render::Vertex::layout();
    lgMeshDesc.vertexCount  = static_cast<uint32_t>(kLgV);
    lgMeshDesc.indexCount   = static_cast<uint32_t>(kLgQ * 6);
    {
        auto e = world.create();
        ecs::CompareLegend lg;
        lg.font  = &uiFont;
        lg.atlas = uiFont.texture;
        lg.mesh  = app.renderer()->createMesh(lgMeshDesc);
        lg.vbo   = lgVbo.handle();
        world.emplace<ecs::CompareLegend>(e, lg);
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

    // Font Size dialog (View menu): slider + numeric box for the UI font size.
    const int kFontDlgQ = 128, kFontDlgV = kFontDlgQ * 4;  // must match kFontDlgQuads in systems.cpp
    const std::vector<render::Vertex> fdInit(kFontDlgV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> fdIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kFontDlgQ); ++q) {
        uint32_t b = q * 4;
        fdIdx.insert(fdIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> fdVbo(*app.renderer(), fdInit,
                                             render::BufferUsage::Dynamic);
    core::IndexBuffer                  fdIbo(*app.renderer(), fdIdx);
    render::MeshDesc fdMeshDesc;
    fdMeshDesc.vertexBuffer = fdVbo.handle();
    fdMeshDesc.indexBuffer  = fdIbo.handle();
    fdMeshDesc.layout       = render::Vertex::layout();
    fdMeshDesc.vertexCount  = static_cast<uint32_t>(kFontDlgV);
    fdMeshDesc.indexCount   = static_cast<uint32_t>(kFontDlgQ * 6);
    {
        auto e = world.create();
        ecs::FontSizeDialog fd;
        fd.font  = &uiFont;
        fd.atlas = uiFont.texture;
        fd.mesh  = app.renderer()->createMesh(fdMeshDesc);
        fd.vbo   = fdVbo.handle();
        world.emplace<ecs::FontSizeDialog>(e, fd);
    }

    // Mode-parameters dialog (generic sliders for the active geometry mode).
    const int kModeDlgQ = 256, kModeDlgV = kModeDlgQ * 4;  // must match kModeDlgQuads in systems.cpp
    const std::vector<render::Vertex> mdInit(kModeDlgV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> mdIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kModeDlgQ); ++q) {
        uint32_t b = q * 4;
        mdIdx.insert(mdIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> mdVbo(*app.renderer(), mdInit,
                                             render::BufferUsage::Dynamic);
    core::IndexBuffer                  mdIbo(*app.renderer(), mdIdx);
    render::MeshDesc mdMeshDesc;
    mdMeshDesc.vertexBuffer = mdVbo.handle();
    mdMeshDesc.indexBuffer  = mdIbo.handle();
    mdMeshDesc.layout       = render::Vertex::layout();
    mdMeshDesc.vertexCount  = static_cast<uint32_t>(kModeDlgV);
    mdMeshDesc.indexCount   = static_cast<uint32_t>(kModeDlgQ * 6);
    {
        auto e = world.create();
        ecs::ModeParamsDialog md;
        md.font  = &uiFont;
        md.atlas = uiFont.texture;
        md.mesh  = app.renderer()->createMesh(mdMeshDesc);
        md.vbo   = mdVbo.handle();
        world.emplace<ecs::ModeParamsDialog>(e, md);
    }

    // Pipeline Design dialog (modeless, resizable node canvas).
    const int kPipeDlgQ = 2048, kPipeDlgV = kPipeDlgQ * 4;  // must match kPipeDlgQuads in systems.cpp
    const std::vector<render::Vertex> plInit(kPipeDlgV, render::Vertex{{0, 0, 0}, {0, 0, 0}});
    std::vector<uint32_t> plIdx;
    for (uint32_t q = 0; q < static_cast<uint32_t>(kPipeDlgQ); ++q) {
        uint32_t b = q * 4;
        plIdx.insert(plIdx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
    core::VertexBuffer<render::Vertex> plVbo(*app.renderer(), plInit,
                                             render::BufferUsage::Dynamic);
    core::IndexBuffer                  plIbo(*app.renderer(), plIdx);
    render::MeshDesc plMeshDesc;
    plMeshDesc.vertexBuffer = plVbo.handle();
    plMeshDesc.indexBuffer  = plIbo.handle();
    plMeshDesc.layout       = render::Vertex::layout();
    plMeshDesc.vertexCount  = static_cast<uint32_t>(kPipeDlgV);
    plMeshDesc.indexCount   = static_cast<uint32_t>(kPipeDlgQ * 6);
    {
        auto e = world.create();
        ecs::PipelineDialog pl;
        pl.font  = &uiFont;
        pl.atlas = uiFont.texture;
        pl.mesh  = app.renderer()->createMesh(plMeshDesc);
        pl.vbo   = plVbo.handle();
        world.emplace<ecs::PipelineDialog>(e, pl);
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

    // A non-flag CLI argument is a mesh/point-cloud path: queue it like a normal
    // File > Open pick (`appOrange <file> --shot` renders it headless).
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') continue;
        std::lock_guard<std::mutex> lk(fileDrop.mtx);
        fileDrop.paths.emplace_back(argv[i]);
        break;
    }

    // File > Save / Save As. The save dialog reports into its own queue so a
    // picked save path is never mistaken for a load. pendingSave remembers which
    // entity the open dialog is for (validated again when the pick arrives).
    FileDropResult saveDrop;
    entt::entity   pendingSave = entt::null;
    std::string    saveDefaultLoc;  // keeps the default-location string alive for SDL
    static const SDL_DialogFileFilter kSaveFilters[] = {
        {"PLY (binary)", "ply"},
        {"OBJ", "obj"},
        {"STL (binary)", "stl"},
        {"XYZ points", "xyz"},
    };

    // Takes the CPU mesh produced by the background loader, uploads GPU buffers
    // at its original coordinates, and spawns a
    // (Transform, Renderable) entity that renderSystem picks up. Runs on the main
    // (render) thread because GPU resource creation is context-affine.
    auto finalizeMesh = [&](const std::string& path, std::vector<render::Vertex>& mv,
                            std::vector<uint32_t>& mi, std::vector<meshio::V3>* nrm = nullptr) {
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

        // CPU copy of the vertex buffer: lets mask modes (bump detect/remove)
        // recolor or delete points in place, and undo/redo rebuild the entity.
        {
            ecs::VertexSource vsrc;
            vsrc.vbo      = loadedVbos.back()->handle();
            vsrc.vertices = mv;
            world.emplace<ecs::VertexSource>(e, std::move(vsrc));
        }
        world.emplace<SourcePath>(e, SourcePath{path});  // File > Save target
        if (nrm && nrm->size() == mv.size()) {           // keep file normals (save + modes)
            ecs::SourceNormals sn;
            sn.normals.reserve(nrm->size());
            for (const auto& n : *nrm) sn.normals.emplace_back(n.x, n.y, n.z);
            world.emplace<ecs::SourceNormals>(e, std::move(sn));
        }
        ecs::pushSpawnOp(world, e, ("Load " + path).c_str());

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
    SaveJob    saveJob;
    CompareJob compareJob;
    float      statusHold = 0.0f;  // seconds to keep the final "Loaded/Failed" message

    // Snapshot the entity's current CPU geometry (transform baked in, so spawned
    // primitives keep their placement) and write it on a worker thread.
    auto startSave = [&](entt::entity tgt, const std::string& path) {
        if (saveJob.active || !world.all_of<ecs::VertexSource>(tgt)) return;
        saveJob.verts = world.get<ecs::VertexSource>(tgt).vertices;  // copy: worker owns it
        saveJob.indices.clear();
        if (world.all_of<ecs::PickGeometry>(tgt))
            saveJob.indices = world.get<ecs::PickGeometry>(tgt).indices;
        // Source-file normals, if still aligned with the (possibly edited) verts.
        saveJob.normals.clear();
        if (auto* sn = world.try_get<ecs::SourceNormals>(tgt))
            if (sn->normals.size() == saveJob.verts.size()) {
                saveJob.normals.reserve(sn->normals.size());
                for (const auto& n : sn->normals)
                    saveJob.normals.push_back({n.x(), n.y(), n.z()});
            }
        Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
        if (world.all_of<ecs::Transform>(tgt)) M = world.get<ecs::Transform>(tgt).matrix();
        if (!M.isApprox(Eigen::Matrix4f::Identity())) {
            for (auto& v : saveJob.verts) {
                Eigen::Vector4f p =
                    M * Eigen::Vector4f(v.position[0], v.position[1], v.position[2], 1.0f);
                v.position[0] = p.x(); v.position[1] = p.y(); v.position[2] = p.z();
            }
            // Normals transform by the inverse-transpose (handles non-uniform scale).
            Eigen::Matrix3f N = M.block<3, 3>(0, 0).inverse().transpose();
            for (auto& n : saveJob.normals) {
                Eigen::Vector3f r = (N * Eigen::Vector3f(n.x, n.y, n.z)).normalized();
                n = {r.x(), r.y(), r.z()};
            }
        }
        saveJob.path   = path;
        saveJob.target = tgt;
        saveJob.active = true;
        saveJob.done.store(false);
        saveJob.ok.store(false);
        saveJob.worker = std::thread([job = &saveJob] {
            job->ok.store(meshio::saveMeshFile(job->path, job->verts, job->indices,
                                               job->normals.empty() ? nullptr : &job->normals));
            job->done.store(true);  // publish last; main thread reaps + reports
        });
    };

    auto onUpdate = [&](entt::registry& w, float dt) {
        core::watchdogHeartbeat();  // tell the watchdog the main loop is alive

        // Headless screenshot mode: let the scene settle, capture, then exit.
        // The frame counter pauses while a load is running so a CLI-given mesh
        // is actually in the scene when the shot is taken.
        if (shotMode && !loadJob.active && ++shotFrames == 30) {
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

        // --pipeline: pop the Pipeline Design dialog once, via the menu action.
        static bool pipeToggled = false;
        if (pipeMode && !pipeToggled && mb) {
            mb->triggered = ecs::MenuAction::PipelineDialogToggle;
            pipeToggled   = true;
        }
        // --legend: show the legend once with synthetic mode values.
        static bool legendShown = false;
        if (legendMode && !legendShown) {
            auto lgv = w.view<ecs::CompareLegend>();
            for (auto le : lgv) {
                auto& lg    = lgv.get<ecs::CompareLegend>(le);
                lg.title    = "Normal Divergence";
                lg.range    = 0.0523f;
                lg.isSigned = true;
                lg.bands    = 1;
                lg.rms      = -1.0f;
                lg.fromMode = true;
                lg.selMin   = 0.22f;
                lg.selMax   = 0.81f;
                lg.visible  = true;
                break;
            }
            // Also open the Font Size dialog (headless UI check).
            auto fdv = w.view<ecs::FontSizeDialog>();
            for (auto fe : fdv) { fdv.get<ecs::FontSizeDialog>(fe).visible = true; break; }
            // Also open the params dialog so its legend strip + Extract show.
            auto pdv = w.view<ecs::ModeParamsDialog>();
            for (auto pe : pdv) {
                auto& pd     = pdv.get<ecs::ModeParamsDialog>(pe);
                pd.modeIndex = orange::modes::modeIndexByName("Normal Divergence");
                int n        = orange::modes::modeParamCount(pd.modeIndex);
                pd.values.resize(n);
                for (int i = 0; i < n; ++i)
                    pd.values[i] = orange::modes::modeParam(pd.modeIndex, i).defV;
                pd.visible = true;
                break;
            }
            legendShown = true;
        }

        // Menu raised a request? Open the native dialog (unless one is already up).
        if (mb && mb->requestOpenFile) {
            mb->requestOpenFile = false;
            if (!fileDrop.busy.exchange(true)) {
                SDL_ShowOpenFileDialog(onFilePicked, &fileDrop, app.window().handle(),
                                       kMeshFilters, 2, nullptr, /*allow_many=*/false);
            }
        }
        // Save / Save As request (menu or Ctrl+S / Ctrl+Shift+S). Target: first
        // selected entity with CPU geometry, else the scene's only such entity.
        // Save overwrites the SourcePath; Save As (or no source path) asks.
        if (mb && (mb->requestSaveFile || mb->requestSaveFileAs)) {
            const bool saveAs   = mb->requestSaveFileAs;
            mb->requestSaveFile = mb->requestSaveFileAs = false;
            entt::entity tgt  = entt::null;
            int          nSrc = 0;
            auto sv = w.view<ecs::Renderable, ecs::VertexSource>();
            for (auto e : sv) {
                ++nSrc;
                if (tgt == entt::null && sv.get<ecs::Renderable>(e).selected) tgt = e;
            }
            if (tgt == entt::null && nSrc == 1)
                for (auto e : sv) tgt = e;
            if (tgt == entt::null) {
                mb->statusText = "Save: select a mesh first";
                statusHold     = 2.0f;
            } else if (!saveJob.active) {
                auto* sp = w.try_get<SourcePath>(tgt);
                if (!saveAs && sp) {
                    startSave(tgt, sp->path);
                } else if (!saveDrop.busy.exchange(true)) {
                    pendingSave    = tgt;
                    saveDefaultLoc = sp ? sp->path : std::string();
                    SDL_ShowSaveFileDialog(onFilePicked, &saveDrop, app.window().handle(),
                                           kSaveFilters, 4,
                                           saveDefaultLoc.empty() ? nullptr
                                                                  : saveDefaultLoc.c_str());
                }
            }
        }
        // 3D Compare request (Geometry menu): deviation heatmap of the test
        // drawable against the reference, from exactly two selected drawables.
        if (mb && mb->requestCompare) {
            mb->requestCompare = false;
            std::vector<entt::entity> sel;
            auto cv = w.view<ecs::Renderable, ecs::VertexSource, ecs::PickGeometry>();
            for (auto e : cv)
                if (cv.get<ecs::Renderable>(e).selected) sel.push_back(e);
            if (sel.size() != 2) {
                mb->statusText = "Compare: select exactly 2 meshes";
                statusHold     = 2.5f;
            } else if (!compareJob.active) {
                // Reference = the triangle mesh when exactly one of the pair has
                // triangles (scan-vs-CAD); otherwise the first found. The other
                // one is the test and gets the heatmap.
                const bool tri0 = !w.get<ecs::PickGeometry>(sel[0]).indices.empty();
                const bool tri1 = !w.get<ecs::PickGeometry>(sel[1]).indices.empty();
                entt::entity ref = sel[0], test = sel[1];
                if (tri1 && !tri0) std::swap(ref, test);

                auto worldPts = [&](entt::entity e) {
                    const auto&     pg = w.get<ecs::PickGeometry>(e);
                    Eigen::Matrix4f M  = Eigen::Matrix4f::Identity();
                    if (w.all_of<ecs::Transform>(e)) M = w.get<ecs::Transform>(e).matrix();
                    std::vector<Eigen::Vector3f> out(pg.positions.size());
                    for (size_t i = 0; i < pg.positions.size(); ++i) {
                        const auto&     p = pg.positions[i];
                        Eigen::Vector4f h = M * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
                        out[i]            = h.head<3>();
                    }
                    return out;
                };
                std::vector<Eigen::Vector3f> testPts = worldPts(test);
                std::vector<Eigen::Vector3f> refPts  = worldPts(ref);
                std::vector<uint32_t>        refIdx  = w.get<ecs::PickGeometry>(ref).indices;
                // A point-cloud reference signs the deviation with its source
                // normals (rotated to world space) when it has them.
                std::vector<Eigen::Vector3f> refNrm;
                if (refIdx.empty()) {
                    if (auto* sn = w.try_get<ecs::SourceNormals>(ref);
                        sn && sn->normals.size() == refPts.size()) {
                        Eigen::Matrix3f N = Eigen::Matrix3f::Identity();
                        if (w.all_of<ecs::Transform>(ref))
                            N = w.get<ecs::Transform>(ref)
                                    .matrix().block<3, 3>(0, 0).inverse().transpose();
                        refNrm.resize(sn->normals.size());
                        for (size_t i = 0; i < sn->normals.size(); ++i)
                            refNrm[i] = (N * sn->normals[i]).normalized();
                    }
                }
                compareJob.active = true;
                compareJob.test   = test;
                compareJob.percent.store(0);
                compareJob.done.store(false);
                compareJob.ok.store(false);
                SDL_Log("Compare: test %zu pts vs ref %zu pts / %zu tris",
                        testPts.size(), refPts.size(), refIdx.size() / 3);
                compareJob.worker = std::thread(
                    [job = &compareJob, testPts = std::move(testPts), refPts = std::move(refPts),
                     refIdx = std::move(refIdx), refNrm = std::move(refNrm)]() mutable {
                        job->devs = geometry::compareDeviations(
                            testPts, refPts, refIdx, refNrm, job->stats, [job](float f) {
                                job->percent.store(static_cast<int>(f * 100.0f));
                            });
                        job->ok.store(!job->devs.empty());
                        job->done.store(true);  // publish last
                    });
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
            loadJob.normals.clear();
            // Parse the file off the main thread, reporting progress into `percent`.
            loadJob.worker = std::thread([job = &loadJob] {
                std::vector<render::Vertex> v;
                std::vector<uint32_t>       i;
                std::vector<meshio::V3>     n;
                bool ok = meshio::loadMeshFile(
                              job->path, v, i,
                              [job](float p) {
                                  job->percent.store(static_cast<int>(p * 100.0f + 0.5f));
                              },
                              &n) &&
                          !v.empty();
                if (ok) {
                    job->verts   = std::move(v);
                    job->indices = std::move(i);
                    job->normals = std::move(n);
                }
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
                    finalizeMesh(loadJob.path, loadJob.verts, loadJob.indices, &loadJob.normals);
                    std::ofstream(lastMeshFile, std::ios::trunc) << loadJob.path;  // remember for next launch
                }
                loadJob.verts.clear();   loadJob.verts.shrink_to_fit();
                loadJob.indices.clear(); loadJob.indices.shrink_to_fit();
                loadJob.normals.clear(); loadJob.normals.shrink_to_fit();
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
            if (ecs::processingModeProgress(w, pct, name) || ecs::spatialVizProgress(w, pct, name) ||
                ecs::pipelineRunProgress(w, pct, name)) {
                if (mb) mb->statusText = name + " " + std::to_string((int)(pct * 100.0f + 0.5f)) + "%";
            } else if (mb && !mb->statusText.empty()) {
                mb->statusText.clear();  // finished -> clear the line
            }
        }

        // Drain a picked save path (the dialog callback runs on a platform
        // thread) and start the write. Unknown extension -> default to .ply.
        {
            std::string savePath;
            {
                std::lock_guard<std::mutex> lk(saveDrop.mtx);
                if (!saveDrop.paths.empty()) {
                    savePath = saveDrop.paths.front();
                    saveDrop.paths.clear();
                }
            }
            if (!savePath.empty()) {
                auto        sepDot = savePath.find_last_of("./\\");
                std::string ext = (sepDot != std::string::npos && savePath[sepDot] == '.')
                                      ? savePath.substr(sepDot + 1)
                                      : std::string();
                for (auto& c : ext) c = static_cast<char>(std::tolower(c));
                if (ext != "ply" && ext != "obj" && ext != "stl" && ext != "xyz")
                    savePath += ".ply";
                // The entity may have been deleted while the dialog was up.
                if (w.valid(pendingSave) && !saveJob.active) startSave(pendingSave, savePath);
                pendingSave = entt::null;
            }
        }
        // Drive status text + finish a completed save on the main thread.
        if (saveJob.active) {
            if (!saveJob.done.load()) {
                if (mb) mb->statusText = "Saving...";
            } else {
                saveJob.worker.join();
                const bool ok = saveJob.ok.load();
                if (ok) {
                    // Save As stamps the new path so a later plain Save reuses it.
                    if (w.valid(saveJob.target))
                        w.emplace_or_replace<SourcePath>(saveJob.target,
                                                         SourcePath{saveJob.path});
                    SDL_Log("Mesh: saved '%s' (%zu verts, %zu tris)", saveJob.path.c_str(),
                            saveJob.verts.size(), saveJob.indices.size() / 3);
                } else {
                    SDL_Log("Mesh: save failed for '%s'", saveJob.path.c_str());
                }
                saveJob.verts.clear();   saveJob.verts.shrink_to_fit();
                saveJob.indices.clear(); saveJob.indices.shrink_to_fit();
                saveJob.normals.clear(); saveJob.normals.shrink_to_fit();
                saveJob.active = false;
                statusHold     = ok ? 1.5f : 3.0f;
                if (mb) mb->statusText = ok ? "Saved" : "Save failed";
            }
        }
        // Drive status text + finish a completed 3D Compare on the main thread:
        // paint the signed deviations as a heatmap (blue = below the reference,
        // green = on it, red = above; unsigned refs map green->red), undoable.
        if (compareJob.active) {
            if (!compareJob.done.load()) {
                if (mb)
                    mb->statusText =
                        "Comparing " + std::to_string(compareJob.percent.load()) + "%";
            } else {
                compareJob.worker.join();
                bool ok = compareJob.ok.load();
                if (ok && w.valid(compareJob.test) &&
                    w.all_of<ecs::VertexSource>(compareJob.test) &&
                    w.get<ecs::VertexSource>(compareJob.test).vertices.size() ==
                        compareJob.devs.size()) {
                    const auto& vs   = w.get<ecs::VertexSource>(compareJob.test);
                    auto oldV = std::make_shared<std::vector<render::Vertex>>(vs.vertices);
                    auto newV = std::make_shared<std::vector<render::Vertex>>(vs.vertices);
                    const auto& s = compareJob.stats;
                    // Quantized 5-stop jet: discrete bands read as deviation
                    // contours (a smooth ramp hides the magnitudes).
                    constexpr int kBands = 12;
                    for (size_t i = 0; i < newV->size(); ++i) {
                        Eigen::Vector3f c = geometry::compareBandColor(
                            compareJob.devs[i], s.range, s.isSigned, kBands);
                        (*newV)[i].color[0] = c.x();
                        (*newV)[i].color[1] = c.y();
                        (*newV)[i].color[2] = c.z();
                    }
                    // Show the legend so the band colors map to numbers on screen.
                    {
                        auto lgv = w.view<ecs::CompareLegend>();
                        for (auto le : lgv) {
                            auto& lg    = lgv.get<ecs::CompareLegend>(le);
                            lg.title    = "Compare";
                            lg.range    = s.range;
                            lg.isSigned = s.isSigned;
                            lg.rms      = s.rms;
                            lg.bands    = kBands;
                            lg.fromMode = false;
                            lg.visible  = true;
                            break;
                        }
                    }
                    paintVertices(w, *app.renderer(), compareJob.test, *newV);
                    ecs::UndoOp op;
                    op.label = "3D Compare";
                    op.undo  = [e = compareJob.test, oldV](entt::registry& wr,
                                                          render::IRenderer& r) {
                        paintVertices(wr, r, e, *oldV);
                    };
                    op.redo = [e = compareJob.test, newV](entt::registry& wr,
                                                          render::IRenderer& r) {
                        paintVertices(wr, r, e, *newV);
                    };
                    ecs::undoStack(w).push(std::move(op));
                    SDL_Log("Compare: %zu pts  mean %+.4f  RMS %.4f  min %+.4f  max %+.4f  "
                            "color range +-%.4f%s",
                            s.count, s.mean, s.rms, s.minDev, s.maxDev, s.range,
                            s.isSigned ? "" : " (unsigned: reference has no orientation)");
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "Compare: RMS %.3f  max %+.3f/%+.3f  (+-%.3f)",
                                  s.rms, s.maxDev, s.minDev, s.range);
                    if (mb) mb->statusText = buf;
                    statusHold = 8.0f;  // stats linger long enough to read
                } else {
                    ok = false;
                    if (mb) mb->statusText = "Compare failed";
                    statusHold = 3.0f;
                }
                compareJob.devs.clear();
                compareJob.devs.shrink_to_fit();
                compareJob.active = false;
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

    // Offer to reload the data (mesh or point cloud) from the previous session
    // via the in-app dialog; a Yes is handled in onUpdate (queues the load like
    // a normal open). Enter answers Yes, Esc answers No (see application.cpp).
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
                cd.line1   = "Load last data?";
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
