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
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include "orange/core/application.h"
#include "orange/core/buffer.h"
#include "orange/core/console.h"
#include "orange/ecs/components.h"
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
    using math::Vec3;
    struct LabelFace { Vec3 n, rt, up; };
    const LabelFace faces[6] = {
        {{1, 0, 0},  {0, 0, -1}, {0, 1, 0}},   // X
        {{-1, 0, 0}, {0, 0, 1},  {0, 1, 0}},   // -X
        {{0, 1, 0},  {1, 0, 0},  {0, 0, 1}},   // Y
        {{0, -1, 0}, {1, 0, 0},  {0, 0, -1}},  // -Y
        {{0, 0, 1},  {1, 0, 0},  {0, 1, 0}},   // Z
        {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},   // -Z
    };
    const float hw = 0.55f, hh = 0.55f, out = 1.05f;  // square quad (square atlas cell)
    for (int c = 0; c < 6; ++c) {
        const LabelFace& f = faces[c];
        Vec3 C = f.n * out;
        float u0 = static_cast<float>(c) / kCells, u1 = static_cast<float>(c + 1) / kCells;
        // top-left, top-right, bottom-right, bottom-left  (v=0 is the atlas top)
        Vec3 p[4] = {C - f.rt * hw + f.up * hh, C + f.rt * hw + f.up * hh,
                     C + f.rt * hw - f.up * hh, C - f.rt * hw - f.up * hh};
        float uv[4][2] = {{u0, 0}, {u1, 0}, {u1, 1}, {u0, 1}};
        uint32_t base = static_cast<uint32_t>(verts.size());
        for (int i = 0; i < 4; ++i)
            verts.push_back({{p[i].x, p[i].y, p[i].z}, {1, 1, 1}, {uv[i][0], uv[i][1]}});
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

} // namespace

int main(int argc, char** argv) {
    SDL_SetMainReady();

    // Console on the monitor right of the largest; 3D window maximizes on the
    // largest monitor (handled in Window::create).
    core::setupConsoleWindow();

    core::AppConfig config;
    config.title  = "Orange appOrange";
    config.width  = 1280;
    config.height = 720;
    config.backend = render::Backend::OpenGL;
    config.vsync   = false;  // uncapped so the FPS graph shows variation

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
    // Left-drag orbits, wheel zooms, middle/right-drag pans.
    {
        auto cam = world.create();
        world.emplace<ecs::Transform>(cam);  // written each frame by the manipulator
        ecs::Camera c;
        c.primary = true;
        world.emplace<ecs::Camera>(cam, c);
        ecs::CameraManipulator manip;
        manip.target      = {0.0f, 0.0f, 0.0f};
        manip.distance    = 7.0f;
        manip.orientation = math::quatAxisAngle({1, 0, 0}, -0.3f);  // tilt down slightly
        world.emplace<ecs::CameraManipulator>(cam, manip);
    }

    // A unit cube as type-safe buffer objects. Declared *after* `app` so they
    // are destroyed (freeing the GPU buffers) before the renderer shuts down.
    const std::vector<render::Vertex> vertices = {
        {{-0.5f, -0.5f, -0.5f}, {0, 0, 0}},
        {{ 0.5f, -0.5f, -0.5f}, {1, 0, 0}},
        {{ 0.5f,  0.5f, -0.5f}, {1, 1, 0}},
        {{-0.5f,  0.5f, -0.5f}, {0, 1, 0}},
        {{-0.5f, -0.5f,  0.5f}, {0, 0, 1}},
        {{ 0.5f, -0.5f,  0.5f}, {1, 0, 1}},
        {{ 0.5f,  0.5f,  0.5f}, {1, 1, 1}},
        {{-0.5f,  0.5f,  0.5f}, {0, 1, 1}},
    };
    const std::vector<uint32_t> indices = {
        0, 1, 2, 2, 3, 0,  4, 6, 5, 6, 4, 7,  // back, front
        4, 5, 1, 1, 0, 4,  3, 2, 6, 6, 7, 3,  // bottom, top
        4, 0, 3, 3, 7, 4,  1, 5, 6, 6, 2, 1,  // left, right
    };

    core::VertexBuffer<render::Vertex> vbo(*app.renderer(), vertices);
    core::IndexBuffer                  ibo(*app.renderer(), indices);

    render::MeshDesc meshDesc;
    meshDesc.vertexBuffer = vbo.handle();
    meshDesc.indexBuffer  = ibo.handle();
    meshDesc.layout       = render::Vertex::layout();
    meshDesc.vertexCount  = static_cast<uint32_t>(vbo.count());
    meshDesc.indexCount   = static_cast<uint32_t>(ibo.count());
    render::MeshHandle cube = app.renderer()->createMesh(meshDesc);

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

    {
        auto giz = world.create();
        ecs::AxisGizmo gizmo;
        gizmo.mesh          = app.renderer()->createMesh(gizmoMesh);
        gizmo.labelMesh     = app.renderer()->createMesh(labelMesh);
        gizmo.labelTexture  = labelTex;
        gizmo.ringMesh      = app.renderer()->createMesh(ringMesh);
        gizmo.highlightMesh = app.renderer()->createMesh(hlMeshDesc);
        gizmo.highlightVbo  = hlVbo.handle();
        world.emplace<ecs::AxisGizmo>(giz, gizmo);
    }

    // Draggable FPS widget: dynamic vertex buffer (rewritten each frame) +
    // static quad index pattern.
    const int kFpsQ = 160, kFpsV = kFpsQ * 4;
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
        world.emplace<ecs::FpsWidget>(e, widget);
    }

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

    // A few spinning cubes, each sharing the same GPU mesh.
    const float xs[] = {-1.6f, 0.0f, 1.6f};
    for (float x : xs) {
        auto e = world.create();
        ecs::Transform t;
        t.position = {x, 0.0f, 0.0f};
        world.emplace<ecs::Transform>(e, t);
        world.emplace<ecs::Renderable>(e, ecs::Renderable{cube});
        ecs::Spin spin;
        spin.axisRadiansPerSec = {0.4f, 0.9f, 0.0f};
        world.emplace<ecs::Spin>(e, spin);
    }

    SDL_Log("appOrange: running. ESC or close the window to quit.");
    app.run();  // spinSystem + renderSystem run internally each frame
    return 0;
}
