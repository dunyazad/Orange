#include "orange/ecs/systems.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <execution>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

#include <SDL3/SDL.h>  // SDL_Log (background-job notices)

#include "orange/core/ball_tree.h"
#include "orange/core/bsp.h"
#include "orange/core/color.h"
#include "orange/core/compare.h"
#include "orange/core/debug_draw.h"
#include "orange/core/draw_mode.h"
#include "orange/core/ui_layout.h"  // core::uiFontPx (View > Font Size...)
#include "orange/core/geometry.h"
#include "orange/core/kdtree.h"
#include "orange/core/loose_octree.h"
#include "orange/core/octree.h"
#include "orange/core/rtree.h"
#include "orange/core/uniform_grid.h"
#include "orange/core/math.h"
#include "orange/core/modes.h"
#include "orange/ecs/components.h"
#include "orange/ecs/undo.h"

namespace orange::ecs {

namespace {
// Shared GUI text height (px) for every overlay widget -- menu bar, toolbar,
// tree view, and all dialogs render at this one size so the chrome is uniform.
// Runtime-adjustable (View > Font Size...): the value lives in
// core::uiFontPx(); uiScale() converts the legacy 20px-era layout literals.
// The former constexpr constants below are macros so every use site rescales
// live without touching hundreds of lines.
float uiPx() { return core::uiFontPx(); }
float uiScale() { return core::uiFontPx() / 20.0f; }
#define kUiTextPx (uiPx())

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

// Re-emit the latched occlusal-plane result (a plane quad + a normal arrow) into
// the immediate-mode DebugDraw each frame while active. Built in the same world
// frame as the meshes (renderSystem applies Mworld when it uploads the debug
// geometry). Set by Application::applyMenuAction; cleared geometry is re-added
// every frame because DebugDraw is flushed after each upload.
void emitOcclusalPlaneViz(entt::registry& world) {
    if (!world.ctx().contains<OcclusalPlaneViz>()) return;
    const auto& v = world.ctx().get<OcclusalPlaneViz>();
    if (!v.active) return;

    auto& dd = debug::DebugDraw::instance();
    const Eigen::Vector3f n = v.normal.normalized();
    const float s = v.size > 0.0f ? v.size : 1.0f;

    // In-plane orthonormal basis (an arbitrary rotation about the normal).
    Eigen::Vector3f a = (std::abs(n.x()) < 0.9f) ? Eigen::Vector3f::UnitX()
                                                 : Eigen::Vector3f::UnitY();
    Eigen::Vector3f u = n.cross(a).normalized();
    Eigen::Vector3f w = n.cross(u).normalized();

    const float hs = s * 0.6f;  // quad half-extent
    Eigen::Vector3f p0 = v.position - u * hs - w * hs;
    Eigen::Vector3f p1 = v.position + u * hs - w * hs;
    Eigen::Vector3f p2 = v.position + u * hs + w * hs;
    Eigen::Vector3f p3 = v.position - u * hs + w * hs;
    const Eigen::Vector3f fill(0.20f, 0.55f, 0.85f);
    dd.addQuad(p0, p1, p2, p3, fill);  // front
    dd.addQuad(p0, p3, p2, p1, fill);  // back (single-sided quads -> draw both)

    // Plane border for readability.
    const Eigen::Vector3f edge(0.55f, 0.85f, 1.0f);
    const float t = s * 0.004f;
    dd.addLine(p0, p1, edge, t); dd.addLine(p1, p2, edge, t);
    dd.addLine(p2, p3, edge, t); dd.addLine(p3, p0, edge, t);

    // Normal arrow: shaft + cone head.
    const Eigen::Vector3f arrow(1.0f, 0.55f, 0.15f);
    const float len = s * 0.8f;
    Eigen::Vector3f neck = v.position + n * len * 0.8f;
    Eigen::Vector3f tip  = v.position + n * len;
    dd.addLine(v.position, neck, arrow, s * 0.01f);
    const float r = len * 0.06f;
    const int seg = 12;
    for (int i = 0; i < seg; ++i) {
        float a0 = (float)i / seg * 6.2831853f;
        float a1 = (float)(i + 1) / seg * 6.2831853f;
        Eigen::Vector3f c0 = neck + (u * std::cos(a0) + w * std::sin(a0)) * r;
        Eigen::Vector3f c1 = neck + (u * std::cos(a1) + w * std::sin(a1)) * r;
        dd.addTriangle(c0, c1, tip, arrow);
    }
}

// Re-emit the latched cusp-detection result (one sphere marker per cusp) into
// DebugDraw each frame while active. Same world frame as the meshes.
void emitCuspViz(entt::registry& world) {
    if (!world.ctx().contains<CuspViz>()) return;
    const auto& c = world.ctx().get<CuspViz>();
    if (!c.active || c.points.empty()) return;

    auto& dd = debug::DebugDraw::instance();
    const Eigen::Vector3f yellow(1.0f, 0.85f, 0.1f);  // normal cusp
    const Eigen::Vector3f red(0.95f, 0.15f, 0.12f);   // outlier (far from centroid)
    const float r = (c.size > 0.0f ? c.size : 1.0f) * 0.006f;
    for (size_t i = 0; i < c.points.size(); ++i) {
        bool out = i < c.outlier.size() && c.outlier[i];
        dd.addSphere(c.points[i], r, out ? red : yellow, 8);
    }
}

// A background computation of one processing-mode result. The worker thread fills
// `verts` from a self-contained copy of the input (it never touches the registry),
// then sets `done`; the main thread harvests it. `gen`/`sig` tag which request it
// answers, so a stale result (selection/mode changed meanwhile) is discarded.
struct ModeJob {
    std::atomic<bool>           done{false};
    std::atomic<float>          progress{0.0f};  // [0,1], updated by the worker
    std::vector<render::Vertex> verts;            // debug geometry (Draw / Recolor extras)
    std::vector<uint8_t>        mask;              // per-point delete flags (Remove)
    std::vector<Eigen::Vector3f> colors;           // per-point colors (Recolor; x<0 = keep)
    std::vector<Eigen::Vector3f> fitted;           // new world positions (fit action)
    // Count-changing fit (e.g. SDF Filter resamples the cloud): the worker gets
    // the source vertices (srcVerts) to transfer colors from, and returns the
    // full replacement vertex set (fitVerts, world-space positions).
    std::vector<render::Vertex> srcVerts;
    std::vector<render::Vertex> fitVerts;
    bool                        isFit = false;     // this job runs the mode's Fit action
    modes::ApplyKind            applyKind = modes::ApplyKind::Draw;
    uint64_t                    gen = 0, sig = 0;
    int                         index = -1;       // which mode (for the status label)
};

// Cached debug geometry for the active processing mode + the in-flight background
// job that produces it, stored in the registry ctx. The heavy mode computation
// runs on `worker` so the main (render) thread never blocks -- a million-point
// cloud no longer freezes the UI.
struct ModeCache {
    uint64_t generation = 0;
    uint64_t selSig     = 0;   // signature of the selection the displayed result was built from
    std::vector<render::Vertex> verts;
    bool hideSource = false;   // hide the source while the displayed result is a Draw overlay

    std::shared_ptr<ModeJob> job;     // running computation (null if idle)
    std::thread              worker;

    // Source entity hidden while its mode result is displayed (the result
    // redraws the cloud itself; the original underneath would z-fight the
    // recolored points and mask any point-removal result). Restored when the
    // mode goes off or the selection changes.
    entt::entity hiddenSrc = entt::null;

    // Source entity whose vertex buffer currently holds a recolored copy (bump
    // detect). Original colors re-uploaded from VertexSource when the mode
    // ends or the selection/mode changes.
    entt::entity recoloredSrc = entt::null;

    // Latest recolor result kept for the legend's range filter: thumbs on the
    // color bar hide points whose scalar falls outside [selMin, selMax]
    // without re-running the mode.
    std::vector<Eigen::Vector3f> lastColors;
    std::vector<float>           lastScalars;  // empty = mode has no scalar
    float appliedMin = 0.0f, appliedMax = 1.0f;

    ~ModeCache() {
        // Never join (could block for minutes) nor std::terminate at teardown: a
        // detached worker is self-contained and dies with the process.
        if (worker.joinable()) worker.detach();
    }
};

// Legend helpers defined with the legend/dialog UI further down; used by
// processingModeSystem's range filter and Extract.
CompareLegend* activeModeLegend(entt::registry& world);
float legendSelValue(const CompareLegend& lg, float sel);

// Replace a point cloud's vertex set: new GPU buffer + mesh (the old buffer may
// be smaller than a restore), refreshed bounds and pick positions. Shared by the
// bump-remove apply and its undo/redo closures.
void applyCloudVertices(entt::registry& world, render::IRenderer& renderer, entt::entity e,
                        const std::vector<render::Vertex>& verts) {
    if (!world.valid(e) || !world.all_of<VertexSource>(e) || !world.all_of<Renderable>(e) ||
        verts.empty())
        return;
    auto& vs    = world.get<VertexSource>(e);
    vs.vertices = verts;
    render::BufferDesc bd;
    bd.type  = render::BufferType::Vertex;
    bd.usage = render::BufferUsage::Static;
    bd.data  = vs.vertices.data();
    bd.size  = vs.vertices.size() * sizeof(render::Vertex);
    vs.vbo = renderer.createBuffer(bd);

    auto& r = world.get<Renderable>(e);
    renderer.destroyMesh(r.mesh);
    render::MeshDesc md;
    md.vertexBuffer = vs.vbo;
    md.layout       = render::Vertex::layout();
    md.vertexCount  = (uint32_t)vs.vertices.size();
    md.topology     = render::PrimitiveTopology::Points;
    r.mesh = renderer.createMesh(md);

    Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX);
    Eigen::Vector3f mx = Eigen::Vector3f::Constant(-FLT_MAX);
    auto& pg = world.get_or_emplace<PickGeometry>(e);
    pg.positions.clear();
    pg.positions.reserve(vs.vertices.size());
    for (const auto& v : vs.vertices) {
        Eigen::Vector3f p(v.position[0], v.position[1], v.position[2]);
        pg.positions.push_back(p);
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
    }
    r.boundsMin = mn;
    r.boundsMax = mx;
}

// Spawn a real, pickable/editable/saveable triangle-mesh entity from a
// non-indexed triangle soup (sequential indices). Used by the Reconstruct
// apply (dialog Fit) to turn the previewed Draw result into scene content.
entt::entity spawnMeshFromTriangles(entt::registry& world, render::IRenderer& renderer,
                                    const std::vector<render::Vertex>& verts,
                                    const char* label) {
    if (verts.size() < 3) return entt::null;
    render::BufferDesc bd;
    bd.type  = render::BufferType::Vertex;
    bd.usage = render::BufferUsage::Static;
    bd.data  = verts.data();
    bd.size  = verts.size() * sizeof(render::Vertex);
    render::BufferHandle vbo = renderer.createBuffer(bd);

    std::vector<uint32_t> idx(verts.size());
    for (uint32_t i = 0; i < (uint32_t)idx.size(); ++i) idx[i] = i;
    render::BufferDesc ibd;
    ibd.type  = render::BufferType::Index;
    ibd.usage = render::BufferUsage::Static;
    ibd.data  = idx.data();
    ibd.size  = idx.size() * sizeof(uint32_t);

    render::MeshDesc md;
    md.vertexBuffer = vbo;
    md.indexBuffer  = renderer.createBuffer(ibd);
    md.layout       = render::Vertex::layout();
    md.vertexCount  = (uint32_t)verts.size();
    md.indexCount   = (uint32_t)idx.size();

    auto e = world.create();
    world.emplace<Transform>(e, Transform{});  // verts are already world-space
    Renderable r;
    r.mesh = renderer.createMesh(md);
    Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX);
    Eigen::Vector3f mx = Eigen::Vector3f::Constant(-FLT_MAX);
    PickGeometry pick;
    pick.positions.reserve(verts.size());
    for (const auto& v : verts) {
        Eigen::Vector3f p(v.position[0], v.position[1], v.position[2]);
        pick.positions.push_back(p);
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
    }
    pick.indices = idx;
    r.boundsMin  = mn;
    r.boundsMax  = mx;
    world.emplace<Renderable>(e, r);
    world.emplace<PickGeometry>(e, std::move(pick));
    VertexSource vs;
    vs.vbo      = vbo;
    vs.vertices = verts;
    world.emplace<VertexSource>(e, std::move(vs));
    pushSpawnOp(world, e, label);
    return e;
}
} // namespace

// Runs the active processing mode on the *selected* entity's point cloud. The
// heavy computation (kNN, reconstruction, smoothing -- O(N) over up to millions of
// points) runs on a BACKGROUND thread so the main render loop never blocks: the
// system kicks off a worker, keeps drawing the last result, and swaps in the new
// one when the worker finishes. A request superseded by a selection/mode change
// before its worker finishes is discarded. The input is the first selected entity
// with a PickGeometry (its vertices, transformed to world space); a ctx
// modes::ModeInput, if present, overrides it (tests / fixed clouds).
void processingModeSystem(entt::registry& world, render::IRenderer& renderer) {
    auto& ctx = world.ctx();

    modes::ModeState state;
    if (ctx.contains<modes::ModeState>()) state = ctx.get<modes::ModeState>();

    if (!ctx.contains<ModeCache>()) ctx.emplace<ModeCache>();
    auto& cache = ctx.get<ModeCache>();

    // Desired source + a cheap signature (no cloud copy yet).
    const bool active   = state.index >= 0;
    const bool hasFixed = ctx.contains<modes::ModeInput>();
    entt::entity src = entt::null;
    uint64_t selSig = 0;
    if (active) {
        if (hasFixed) {
            selSig = 0xFFFFFFFFFFFFFFFFull;
        } else {
            // The first selected entity -- or, when nothing is selected and
            // exactly one cloud/mesh is loaded, that one (no need to select it).
            entt::entity only = entt::null;
            size_t count = 0;
            auto v = world.view<Renderable, PickGeometry>();
            for (auto e : v) {
                if (v.get<PickGeometry>(e).positions.empty()) continue;
                ++count;
                only = e;
                if (v.get<Renderable>(e).selected) { src = e; break; }
            }
            if (src == entt::null && count == 1) src = only;
            if (src != entt::null)
                selSig = ((uint64_t)(uint32_t)entt::to_integral(src) << 32) ^
                         (uint64_t)world.get<PickGeometry>(src).positions.size();
        }
    }
    // What the displayed result should answer. Fold in the mode index so switching
    // modes on the same selection recomputes.
    const uint64_t want = active ? (state.generation ^ (selSig * 0x9E3779B97F4A7C15ull) ^
                                    ((uint64_t)(state.index + 1) << 8))
                                 : 0;

    // Hide the source cloud while its mode result is displayed (the result
    // redraws the points itself; the original underneath would z-fight the
    // recolored points and mask any point-removal result).
    auto syncHidden = [&](entt::entity wantHidden) {
        if (cache.hiddenSrc == wantHidden) return;
        if (cache.hiddenSrc != entt::null && world.valid(cache.hiddenSrc) &&
            world.all_of<Renderable>(cache.hiddenSrc))
            world.get<Renderable>(cache.hiddenSrc).visible = true;
        if (wantHidden != entt::null) world.get<Renderable>(wantHidden).visible = false;
        cache.hiddenSrc = wantHidden;
    };

    // Re-upload the original vertex colors of a recolored cloud (bump detect)
    // once its result no longer applies.
    auto restoreRecolor = [&]() {
        if (cache.recoloredSrc == entt::null) return;
        if (world.valid(cache.recoloredSrc) && world.all_of<VertexSource>(cache.recoloredSrc)) {
            const auto& vs = world.get<VertexSource>(cache.recoloredSrc);
            renderer.updateBuffer(vs.vbo, vs.vertices.data(),
                                  vs.vertices.size() * sizeof(render::Vertex));
        }
        cache.recoloredSrc = entt::null;
        cache.lastColors.clear();
        cache.lastScalars.clear();
        cache.appliedMin = 0.0f;
        cache.appliedMax = 1.0f;
        // The recolor's legend goes with it (Compare's own legend stays).
        auto lgv = world.view<CompareLegend>();
        for (auto le : lgv) {
            auto& lg = lgv.get<CompareLegend>(le);
            if (lg.fromMode) { lg.visible = false; lg.fromMode = false; }
            break;
        }
    };

    // 1) Harvest a finished background job (non-blocking: only join once done).
    if (cache.job && cache.job->done.load(std::memory_order_acquire)) {
        if (cache.worker.joinable()) cache.worker.join();
        if (cache.job->gen == want && cache.job->sig == selSig) {  // still relevant
            if (cache.job->isFit && src != entt::null && world.all_of<VertexSource>(src) &&
                world.all_of<Renderable>(src) && world.get<Renderable>(src).pointCloud) {
                // Fit action: replace the cloud's positions with the mode's
                // fitted (world-space) positions, mapped back to local space.
                // Undoable; the mode is deactivated afterwards so the moved
                // cloud is not immediately re-processed.
                auto&       vs     = world.get<VertexSource>(src);
                const auto& fitted = cache.job->fitted;
                if (fitted.size() == vs.vertices.size()) {
                    Eigen::Matrix4f Minv = Eigen::Matrix4f::Identity();
                    if (world.all_of<Transform>(src))
                        Minv = world.get<Transform>(src).matrix().inverse();
                    auto oldVerts = std::make_shared<std::vector<render::Vertex>>(vs.vertices);
                    auto newVerts = std::make_shared<std::vector<render::Vertex>>(vs.vertices);
                    size_t moved = 0;
                    for (size_t i = 0; i < fitted.size(); ++i) {
                        Eigen::Vector4f l = Minv * Eigen::Vector4f(fitted[i].x(), fitted[i].y(),
                                                                   fitted[i].z(), 1.0f);
                        auto& p = (*newVerts)[i].position;
                        if (p[0] != l.x() || p[1] != l.y() || p[2] != l.z()) ++moved;
                        p[0] = l.x(); p[1] = l.y(); p[2] = l.z();
                    }
                    applyCloudVertices(world, renderer, src, *newVerts);
                    UndoOp op;
                    op.label = std::string(modes::modeName(state.index)) + " Fit";
                    op.undo  = [e = src, oldVerts](entt::registry& w, render::IRenderer& r) {
                        applyCloudVertices(w, r, e, *oldVerts);
                    };
                    op.redo = [e = src, newVerts](entt::registry& w, render::IRenderer& r) {
                        applyCloudVertices(w, r, e, *newVerts);
                    };
                    undoStack(world).push(std::move(op));
                    SDL_Log("processingModeSystem: %s fit done (%zu points moved)",
                            modes::modeName(state.index), moved);
                } else if (!cache.job->fitVerts.empty()) {
                    // Count-changing fit (SDF Filter resample): swap in the
                    // worker-built replacement set, positions mapped back to
                    // local space. Undoable like the in-place fit.
                    Eigen::Matrix4f Minv = Eigen::Matrix4f::Identity();
                    if (world.all_of<Transform>(src))
                        Minv = world.get<Transform>(src).matrix().inverse();
                    auto oldVerts = std::make_shared<std::vector<render::Vertex>>(vs.vertices);
                    auto newVerts = std::make_shared<std::vector<render::Vertex>>(
                        std::move(cache.job->fitVerts));
                    for (auto& v : *newVerts) {
                        Eigen::Vector4f l = Minv * Eigen::Vector4f(v.position[0], v.position[1],
                                                                   v.position[2], 1.0f);
                        v.position[0] = l.x(); v.position[1] = l.y(); v.position[2] = l.z();
                    }
                    applyCloudVertices(world, renderer, src, *newVerts);
                    UndoOp op;
                    op.label = std::string(modes::modeName(state.index)) + " Fit";
                    op.undo  = [e = src, oldVerts](entt::registry& w, render::IRenderer& r) {
                        applyCloudVertices(w, r, e, *oldVerts);
                    };
                    op.redo = [e = src, newVerts](entt::registry& w, render::IRenderer& r) {
                        applyCloudVertices(w, r, e, *newVerts);
                    };
                    undoStack(world).push(std::move(op));
                    SDL_Log("processingModeSystem: %s fit done (%zu -> %zu points)",
                            modes::modeName(state.index), oldVerts->size(), newVerts->size());
                }
                cache.verts.clear();
                cache.hideSource = false;
                if (ctx.contains<modes::ModeState>()) {
                    auto& ms = ctx.get<modes::ModeState>();
                    ms.index = -1;
                    ms.generation++;
                }
            } else if (cache.job->applyKind == modes::ApplyKind::Recolor && src != entt::null &&
                world.all_of<VertexSource>(src)) {
                // Paint the mode's per-point colors IN the source buffer (a
                // color with x < 0 keeps the original; originals stay in
                // VertexSource for the restore). Extra debug geometry (e.g.
                // cluster boxes) still displays via cache.verts.
                const auto& vs     = world.get<VertexSource>(src);
                const auto& colors = cache.job->colors;
                std::vector<render::Vertex> painted = vs.vertices;
                size_t n = std::min(painted.size(), colors.size());
                for (size_t i = 0; i < n; ++i) {
                    if (colors[i].x() < 0.0f) continue;
                    painted[i].color[0] = colors[i].x();
                    painted[i].color[1] = colors[i].y();
                    painted[i].color[2] = colors[i].z();
                }
                renderer.updateBuffer(vs.vbo, painted.data(),
                                      painted.size() * sizeof(render::Vertex));
                cache.recoloredSrc = src;
                cache.verts        = std::move(cache.job->verts);
                cache.hideSource   = false;
                // Heatmap modes publish a numeric scale: show it as the color-
                // bar legend (hide a previous MODE legend when this mode has
                // none; a 3D-Compare legend is left alone).
                {
                    modes::ModeLegend ml = modes::modeLastLegend();
                    auto lgv = world.view<CompareLegend>();
                    for (auto le : lgv) {
                        auto& lg = lgv.get<CompareLegend>(le);
                        if (ml.valid) {
                            lg.title    = modes::modeName(state.index);
                            lg.range    = ml.range;
                            lg.isSigned = ml.isSigned;
                            lg.bands    = ml.bands;
                            lg.rms      = -1.0f;
                            lg.fromMode = true;
                            lg.visible  = true;
                            lg.selMin   = 0.0f;  // fresh legend: full range
                            lg.selMax   = 1.0f;
                            lg.dragThumb = -1;
                        } else if (lg.fromMode) {
                            lg.visible  = false;
                            lg.fromMode = false;
                        }
                        break;
                    }
                    // Keep the result around for the legend's range filter.
                    cache.lastColors  = colors;
                    cache.lastScalars = ml.valid ? std::move(ml.scalars)
                                                 : std::vector<float>{};
                    cache.appliedMin  = 0.0f;
                    cache.appliedMax  = 1.0f;
                }
                SDL_Log("processingModeSystem: %s done (recolored %zu points)",
                        modes::modeName(state.index), n);
            } else if (cache.job->applyKind == modes::ApplyKind::Remove && src != entt::null &&
                       world.all_of<VertexSource>(src)) {
                // Delete the flagged points: compact the CPU copy and swap in a
                // new buffer/mesh (undoable). Deactivate the mode afterwards so
                // the shrunk cloud isn't reprocessed.
                auto&       vs   = world.get<VertexSource>(src);
                const auto& mask = cache.job->mask;
                std::vector<render::Vertex> kept;
                kept.reserve(vs.vertices.size());
                for (size_t i = 0; i < vs.vertices.size(); ++i)
                    if (i >= mask.size() || !mask[i]) kept.push_back(vs.vertices[i]);
                size_t removed = vs.vertices.size() - kept.size();
                if (removed > 0 && !kept.empty()) {
                    auto oldVerts =
                        std::make_shared<std::vector<render::Vertex>>(vs.vertices);
                    auto newVerts =
                        std::make_shared<std::vector<render::Vertex>>(std::move(kept));
                    applyCloudVertices(world, renderer, src, *newVerts);
                    UndoOp op;
                    op.label = "Bump: Remove";
                    op.undo  = [e = src, oldVerts](entt::registry& w, render::IRenderer& r) {
                        applyCloudVertices(w, r, e, *oldVerts);
                    };
                    op.redo = [e = src, newVerts](entt::registry& w, render::IRenderer& r) {
                        applyCloudVertices(w, r, e, *newVerts);
                    };
                    undoStack(world).push(std::move(op));
                }
                SDL_Log("processingModeSystem: %s done (%zu points removed)",
                        modes::modeName(state.index), removed);
                cache.verts.clear();
                cache.hideSource = false;
                if (ctx.contains<modes::ModeState>()) {
                    auto& ms = ctx.get<modes::ModeState>();
                    ms.index = -1;
                    ms.generation++;
                }
            } else {
                cache.verts      = std::move(cache.job->verts);
                cache.hideSource = !cache.verts.empty();
                SDL_Log("processingModeSystem: %s done (%zu verts)", modes::modeName(state.index),
                        cache.verts.size());
            }
            cache.generation = want;
            cache.selSig     = selSig;
        }
        cache.job.reset();
    }

    // 2) Inactive / nothing selected: show nothing (let any in-flight job finish
    //    and be discarded above on a later frame).
    if (!active || (!hasFixed && src == entt::null)) {
        if (!cache.verts.empty()) { cache.verts.clear(); cache.generation = 0; cache.selSig = 0; }
        cache.hideSource = false;
        syncHidden(entt::null);
        restoreRecolor();
        debug::DebugDraw::instance().addRaw(cache.verts);
        return;
    }

    // The params dialog (if it is editing this mode): current values + a
    // pending Fit request (kept pending while a job is in flight).
    ModeParamsDialog* dlg = nullptr;
    {
        auto dv = world.view<ModeParamsDialog>();
        for (auto de : dv) {
            auto& d = dv.get<ModeParamsDialog>(de);
            if (d.modeIndex == state.index) dlg = &d;
            break;
        }
    }
    // Apply-as-mesh (Reconstruct): the Fit button turns the CURRENT cached
    // triangle preview into a real mesh entity, hides the source cloud (like
    // Poisson's finalize; the tree-view eye unhides it), and deactivates the
    // mode. Kept pending while a recompute is in flight.
    if (dlg && dlg->requestFit && modes::modeAppliesMesh(state.index)) {
        if (!cache.job) {
            dlg->requestFit = false;
            if (!cache.verts.empty() && cache.generation == want && cache.selSig == selSig) {
                std::string  label = std::string(modes::modeName(state.index)) + " Apply";
                entt::entity e =
                    spawnMeshFromTriangles(world, renderer, cache.verts, label.c_str());
                if (e != entt::null) {
                    if (src != entt::null && world.valid(src) && world.all_of<Renderable>(src)) {
                        syncHidden(entt::null);  // release the mode's own hide first
                        world.get<Renderable>(src).visible = false;
                    }
                    SDL_Log("processingModeSystem: %s applied as mesh (%zu tris)",
                            modes::modeName(state.index), cache.verts.size() / 3);
                    cache.verts.clear();
                    cache.hideSource = false;
                    cache.generation = 0;
                    if (ctx.contains<modes::ModeState>()) {
                        auto& ms = ctx.get<modes::ModeState>();
                        ms.index = -1;
                        ms.generation++;
                    }
                    debug::DebugDraw::instance().addRaw(cache.verts);
                    return;
                }
            }
        }
    }

    const bool wantFit = dlg && dlg->requestFit && modes::modeCanFit(state.index);

    // 3) Need a (re)compute (or a Fit) and no worker in flight? Build the input
    //    on the main thread (a cheap copy) and launch the heavy work on a
    //    background thread.
    const bool needCompute = (cache.generation != want || cache.selSig != selSig);
    if ((needCompute || wantFit) && !cache.job) {
        if (wantFit) dlg->requestFit = false;  // consumed by this launch
        // The displayed result is being replaced -- if it was a recolor, put the
        // original colors back first (the new result may be another mode or
        // another entity).
        restoreRecolor();

        // Main thread only copies the raw positions (a fast memcpy) + the model
        // matrix; the worker does the world transform AND the heavy operator.
        modes::ModeInput fixedInput;
        std::vector<Eigen::Vector3f> raw;
        std::vector<Eigen::Vector3f> rawN;  // source-file oriented normals (may be empty)
        std::vector<uint32_t> rawI;  // triangle indices -- oriented-normal fallback
        Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
        if (hasFixed) {
            fixedInput = ctx.get<modes::ModeInput>();
        } else {
            const auto& pg = world.get<PickGeometry>(src);
            raw = pg.positions;  // vector copy
            if (world.all_of<Transform>(src)) M = world.get<Transform>(src).matrix();
            if (auto* sn = world.try_get<SourceNormals>(src))
                if (sn->normals.size() == raw.size()) rawN = sn->normals;
            // No file normals but a real triangle mesh: its winding gives
            // consistently ORIENTED vertex normals (computed on the worker) --
            // far better input for the normal-field modes than the per-point
            // PCA estimate, whose signs are ambiguous.
            if (rawN.empty() && !pg.indices.empty()) rawI = pg.indices;
        }

        // Current parameter values (modes fall back to their defaults).
        std::vector<float> params;
        if (dlg) params = dlg->values;  // sized to the mode's param count on bind

        auto job = std::make_shared<ModeJob>();
        job->gen   = want;
        job->sig   = selSig;
        job->index = state.index;
        // Recolor/Remove/Fit edit the source buffer instead of drawing -- only
        // possible when the source is a real entity with a CPU vertex copy
        // (not a fixed ctx input); Remove/Fit rebuild the mesh as points, so
        // they additionally need a point cloud. Otherwise fall back to drawing.
        const bool canEdit = !hasFixed && src != entt::null && world.all_of<VertexSource>(src);
        const bool isCloud = canEdit && world.get<Renderable>(src).pointCloud;
        modes::ApplyKind k = modes::modeApplyKind(state.index);
        job->applyKind = (k == modes::ApplyKind::Recolor && canEdit) ? k
                       : (k == modes::ApplyKind::Remove && isCloud)  ? k
                                                                     : modes::ApplyKind::Draw;
        job->isFit = wantFit && isCloud;
        if (wantFit && !isCloud)
            SDL_Log("processingModeSystem: Fit skipped -- it edits point clouds only "
                    "(the selected entity is a triangle mesh or not editable)");
        if (job->isFit)  // color source for a count-changing fit result
            job->srcVerts = world.get<VertexSource>(src).vertices;
        cache.job  = job;
        SDL_Log("processingModeSystem: %s running in background on %zu points...",
                modes::modeName(state.index),
                hasFixed ? fixedInput.points.size() : raw.size());
        int  idx = state.index;
        bool useFixed = hasFixed;
        cache.worker = std::thread(
            [job, idx, useFixed, fixed = std::move(fixedInput), raw = std::move(raw),
             rawN = std::move(rawN), rawI = std::move(rawI), M,
             params = std::move(params)]() mutable {
                modes::ModeInput in;
                if (useFixed) {
                    in = std::move(fixed);
                } else {
                    in.points.resize(raw.size());
                    for (size_t i = 0; i < raw.size(); ++i) {
                        Eigen::Vector4f w = M * Eigen::Vector4f(raw[i].x(), raw[i].y(), raw[i].z(), 1.0f);
                        in.points[i]      = Eigen::Vector3f(w.x(), w.y(), w.z());
                    }
                    if (!rawN.empty()) {  // normals rotate by the inverse-transpose
                        Eigen::Matrix3f N = M.block<3, 3>(0, 0).inverse().transpose();
                        in.normals.resize(rawN.size());
                        for (size_t i = 0; i < rawN.size(); ++i)
                            in.normals[i] = (N * rawN[i]).normalized();
                    } else if (!rawI.empty()) {
                        // Area-weighted vertex normals from the triangle winding
                        // (world-space points, so no inverse-transpose needed).
                        in.normals.assign(in.points.size(), Eigen::Vector3f::Zero());
                        for (size_t t = 0; t + 2 < rawI.size(); t += 3) {
                            const Eigen::Vector3f& a = in.points[rawI[t]];
                            Eigen::Vector3f fn = (in.points[rawI[t + 1]] - a)
                                                     .cross(in.points[rawI[t + 2]] - a);
                            in.normals[rawI[t]] += fn;
                            in.normals[rawI[t + 1]] += fn;
                            in.normals[rawI[t + 2]] += fn;
                        }
                        for (auto& n : in.normals) {
                            float l2 = n.squaredNorm();
                            n = l2 > 1e-24f ? Eigen::Vector3f(n / std::sqrt(l2))
                                            : Eigen::Vector3f::UnitY();
                        }
                    }
                }
                in.params = std::move(params);
                auto onProgress = [job](float f) {
                    job->progress.store(f, std::memory_order_relaxed);
                };
                if (job->isFit) {
                    modes::runModeFit(idx, in, job->fitted, onProgress);
                    // Count changed (a resampling fit like SDF Filter): build
                    // the full replacement vertex set here on the worker --
                    // world positions + colors from the nearest source point.
                    if (job->fitted.size() != in.points.size() && !job->fitted.empty() &&
                        !job->srcVerts.empty() && job->srcVerts.size() == in.points.size()) {
                        geometry::KDTree tree;
                        tree.build(in.points);
                        job->fitVerts.resize(job->fitted.size());
                        for (size_t i = 0; i < job->fitted.size(); ++i) {
                            render::Vertex v{};
                            const auto& p = job->fitted[i];
                            v.position[0] = p.x(); v.position[1] = p.y(); v.position[2] = p.z();
                            int j = tree.nearest(p);
                            if (j >= 0) {
                                v.color[0] = job->srcVerts[j].color[0];
                                v.color[1] = job->srcVerts[j].color[1];
                                v.color[2] = job->srcVerts[j].color[2];
                            } else {
                                v.color[0] = v.color[1] = v.color[2] = 0.8f;
                            }
                            job->fitVerts[i] = v;
                        }
                    }
                } else if (job->applyKind == modes::ApplyKind::Remove) {
                    modes::runModeMask(idx, in, job->mask, onProgress);
                } else if (job->applyKind == modes::ApplyKind::Recolor) {
                    debug::DebugDraw extras;
                    modes::runModeColors(idx, in, job->colors, extras, onProgress);
                    job->verts = extras.vertices();
                } else {
                    debug::DebugDraw tmp;
                    modes::runMode(idx, in, tmp, onProgress);
                    job->verts = tmp.vertices();
                }
                job->done.store(true, std::memory_order_release);
            });
    }

    // Hide the source only while a Draw-overlay result of ITS OWN is on screen
    // (Recolor results paint the source itself, so it must stay visible; a
    // stale result from a previous selection must not hide the new source).
    syncHidden(cache.hideSource && !hasFixed && cache.selSig == selSig ? src : entt::null);

    // Legend range filter: when the color-bar thumbs move, repaint the
    // recolored cloud with out-of-range points hidden (NaN positions clip on
    // the GPU; VertexSource keeps the pristine originals for the restore).
    if (cache.recoloredSrc != entt::null && world.valid(cache.recoloredSrc) &&
        world.all_of<VertexSource>(cache.recoloredSrc) && !cache.lastScalars.empty()) {
        float wantMin = 0.0f, wantMax = 1.0f;
        const CompareLegend* lgp = nullptr;
        auto lgv = world.view<CompareLegend>();
        for (auto le : lgv) { lgp = &lgv.get<CompareLegend>(le); break; }
        if (lgp && lgp->visible && lgp->fromMode) {
            wantMin = lgp->selMin;
            wantMax = lgp->selMax;
        }
        if (wantMin != cache.appliedMin || wantMax != cache.appliedMax) {
            // Thumbs parked at the ends mean "unbounded" (the range only spans
            // the 95th percentile; the tails must not clip at full span).
            float lo = -FLT_MAX, hi = FLT_MAX;
            if (lgp && wantMin > 0.001f)
                lo = lgp->isSigned ? -lgp->range + wantMin * 2.0f * lgp->range
                                   : wantMin * lgp->range;
            if (lgp && wantMax < 0.999f)
                hi = lgp->isSigned ? -lgp->range + wantMax * 2.0f * lgp->range
                                   : wantMax * lgp->range;
            const auto& vs = world.get<VertexSource>(cache.recoloredSrc);
            std::vector<render::Vertex> painted = vs.vertices;
            const float nanv = std::nanf("");
            size_t n = std::min({painted.size(), cache.lastScalars.size(),
                                 cache.lastColors.size()});
            for (size_t i = 0; i < n; ++i) {
                if (cache.lastColors[i].x() >= 0.0f) {
                    painted[i].color[0] = cache.lastColors[i].x();
                    painted[i].color[1] = cache.lastColors[i].y();
                    painted[i].color[2] = cache.lastColors[i].z();
                }
                if (cache.lastScalars[i] < lo || cache.lastScalars[i] > hi)
                    painted[i].position[0] = painted[i].position[1] =
                        painted[i].position[2] = nanv;
            }
            renderer.updateBuffer(vs.vbo, painted.data(),
                                  painted.size() * sizeof(render::Vertex));
            cache.appliedMin = wantMin;
            cache.appliedMax = wantMax;
        }
    }

    // "Extract" (params-dialog legend strip): spawn the points inside the
    // legend's [selMin, selMax] range as a NEW point cloud (original colors,
    // same transform as the source; undoable).
    {
        ModeParamsDialog* dlg2 = nullptr;
        auto dv2 = world.view<ModeParamsDialog>();
        for (auto e : dv2) { dlg2 = &dv2.get<ModeParamsDialog>(e); break; }
        CompareLegend* lgp = dlg2 && dlg2->requestExtract ? activeModeLegend(world) : nullptr;
        if (dlg2 && dlg2->requestExtract) dlg2->requestExtract = false;
        if (lgp && cache.recoloredSrc != entt::null && world.valid(cache.recoloredSrc) &&
            world.all_of<VertexSource>(cache.recoloredSrc) && !cache.lastScalars.empty()) {
            float lo = -FLT_MAX, hi = FLT_MAX;  // parked thumbs = unbounded tails
            if (lgp->selMin > 0.001f) lo = legendSelValue(*lgp, lgp->selMin);
            if (lgp->selMax < 0.999f) hi = legendSelValue(*lgp, lgp->selMax);
            const auto& vs = world.get<VertexSource>(cache.recoloredSrc);
            std::vector<render::Vertex> kept;
            size_t n = std::min(vs.vertices.size(), cache.lastScalars.size());
            kept.reserve(n);
            for (size_t i = 0; i < n; ++i)
                if (cache.lastScalars[i] >= lo && cache.lastScalars[i] <= hi)
                    kept.push_back(vs.vertices[i]);
            if (kept.empty()) {
                SDL_Log("processingModeSystem: Extract found no points in range");
            } else {
                render::BufferDesc bd;
                bd.type  = render::BufferType::Vertex;
                bd.usage = render::BufferUsage::Static;
                bd.data  = kept.data();
                bd.size  = kept.size() * sizeof(render::Vertex);
                render::BufferHandle vbo = renderer.createBuffer(bd);
                render::MeshDesc md;
                md.vertexBuffer = vbo;
                md.layout       = render::Vertex::layout();
                md.vertexCount  = (uint32_t)kept.size();
                md.topology     = render::PrimitiveTopology::Points;

                auto e = world.create();
                world.emplace<Transform>(e, world.all_of<Transform>(cache.recoloredSrc)
                                                ? world.get<Transform>(cache.recoloredSrc)
                                                : Transform{});
                Renderable r;
                r.mesh       = renderer.createMesh(md);
                r.pointCloud = true;
                Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX);
                Eigen::Vector3f mx = Eigen::Vector3f::Constant(-FLT_MAX);
                PickGeometry pick;
                pick.positions.reserve(kept.size());
                for (const auto& v : kept) {
                    Eigen::Vector3f p(v.position[0], v.position[1], v.position[2]);
                    pick.positions.push_back(p);
                    mn = mn.cwiseMin(p);
                    mx = mx.cwiseMax(p);
                }
                r.boundsMin = mn;
                r.boundsMax = mx;
                world.emplace<Renderable>(e, r);
                world.emplace<PickGeometry>(e, std::move(pick));
                world.emplace<VertexSource>(e, VertexSource{vbo, std::move(kept)});
                pushSpawnOp(world, e, "Extract");
                SDL_Log("processingModeSystem: Extract -> %u points",
                        (unsigned)world.get<VertexSource>(e).vertices.size());
            }
        }
    }

    debug::DebugDraw::instance().addRaw(cache.verts);
}

bool processingModeProgress(entt::registry& world, float& outProgress, std::string& outName) {
    auto& ctx = world.ctx();
    if (!ctx.contains<ModeCache>()) return false;
    auto& cache = ctx.get<ModeCache>();
    if (!cache.job || cache.job->done.load(std::memory_order_acquire)) return false;
    outProgress = cache.job->progress.load(std::memory_order_relaxed);
    outName     = modes::modeName(cache.job->index);
    return true;
}

// --- Undo / redo (see orange/ecs/undo.h) -------------------------------------

UndoStack& undoStack(entt::registry& world) {
    auto& ctx = world.ctx();
    if (!ctx.contains<UndoStack>()) ctx.emplace<UndoStack>();
    return ctx.get<UndoStack>();
}

EntitySnapshotPtr captureEntity(entt::registry& world, entt::entity e) {
    if (!world.valid(e) || !world.all_of<Renderable>(e) || !world.all_of<VertexSource>(e))
        return nullptr;
    EntitySnapshotPtr snap;
    if (world.all_of<UndoRef>(e) && world.get<UndoRef>(e).snap)
        snap = world.get<UndoRef>(e).snap;
    else {
        snap = std::make_shared<EntitySnapshot>();
        world.emplace_or_replace<UndoRef>(e, UndoRef{snap});
    }
    const auto& r = world.get<Renderable>(e);
    snap->entity     = e;
    snap->vertices   = world.get<VertexSource>(e).vertices;
    snap->indices    = world.all_of<PickGeometry>(e) ? world.get<PickGeometry>(e).indices
                                                     : std::vector<uint32_t>{};
    snap->normals.clear();
    if (auto* sn = world.try_get<SourceNormals>(e))
        if (sn->normals.size() == snap->vertices.size()) snap->normals = sn->normals;
    snap->transform  = world.all_of<Transform>(e) ? world.get<Transform>(e) : Transform{};
    snap->drawMode   = r.drawMode;
    snap->colorMode  = r.colorMode;
    snap->pointCloud = r.pointCloud;
    return snap;
}

void rebuildEntity(entt::registry& world, render::IRenderer& renderer,
                   const EntitySnapshotPtr& snap) {
    if (!snap || snap->vertices.empty()) return;
    if (world.valid(snap->entity)) return;  // already alive (double-redo guard)

    render::BufferDesc bd;
    bd.type  = render::BufferType::Vertex;
    bd.usage = render::BufferUsage::Static;
    bd.data  = snap->vertices.data();
    bd.size  = snap->vertices.size() * sizeof(render::Vertex);
    render::BufferHandle vbo = renderer.createBuffer(bd);

    render::MeshDesc md;
    md.vertexBuffer = vbo;
    md.layout       = render::Vertex::layout();
    md.vertexCount  = (uint32_t)snap->vertices.size();
    if (snap->pointCloud) {
        md.topology = render::PrimitiveTopology::Points;
    } else if (!snap->indices.empty()) {
        render::BufferDesc id;
        id.type  = render::BufferType::Index;
        id.usage = render::BufferUsage::Static;
        id.data  = snap->indices.data();
        id.size  = snap->indices.size() * sizeof(uint32_t);
        md.indexBuffer = renderer.createBuffer(id);
        md.indexCount  = (uint32_t)snap->indices.size();
    }

    auto e = world.create();
    world.emplace<Transform>(e, snap->transform);
    Renderable r;
    r.mesh       = renderer.createMesh(md);
    r.drawMode   = snap->drawMode;
    r.colorMode  = snap->colorMode;
    r.pointCloud = snap->pointCloud;
    Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX);
    Eigen::Vector3f mx = Eigen::Vector3f::Constant(-FLT_MAX);
    PickGeometry pick;
    pick.positions.reserve(snap->vertices.size());
    for (const auto& v : snap->vertices) {
        Eigen::Vector3f p(v.position[0], v.position[1], v.position[2]);
        pick.positions.push_back(p);
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
    }
    if (!snap->pointCloud) pick.indices = snap->indices;
    r.boundsMin = mn;
    r.boundsMax = mx;
    world.emplace<Renderable>(e, r);
    world.emplace<PickGeometry>(e, std::move(pick));
    world.emplace<VertexSource>(e, VertexSource{vbo, snap->vertices});
    if (snap->normals.size() == snap->vertices.size())
        world.emplace<SourceNormals>(e, SourceNormals{snap->normals});
    world.emplace<UndoRef>(e, UndoRef{snap});
    snap->entity = e;
}

void pushSpawnOp(entt::registry& world, entt::entity e, const char* label) {
    auto snap = captureEntity(world, e);
    if (!snap) return;
    UndoOp op;
    op.label = label;
    op.undo  = [snap](entt::registry& w, render::IRenderer&) {
        if (w.valid(snap->entity)) w.destroy(snap->entity);
        snap->entity = entt::null;
    };
    op.redo = [snap](entt::registry& w, render::IRenderer& r) { rebuildEntity(w, r, snap); };
    undoStack(world).push(std::move(op));
}

void pushDeleteOp(entt::registry& world, const std::vector<entt::entity>& dead,
                  const char* label) {
    std::vector<EntitySnapshotPtr> snaps;
    for (auto e : dead) {
        auto s = captureEntity(world, e);
        if (s) snaps.push_back(std::move(s));
    }
    if (snaps.empty()) return;
    UndoOp op;
    op.label = label;
    op.undo  = [snaps](entt::registry& w, render::IRenderer& r) {
        for (const auto& s : snaps) rebuildEntity(w, r, s);
    };
    op.redo = [snaps](entt::registry& w, render::IRenderer&) {
        for (const auto& s : snaps) {
            if (w.valid(s->entity)) w.destroy(s->entity);
            s->entity = entt::null;
        }
    };
    undoStack(world).push(std::move(op));
}

bool undoLast(entt::registry& world, render::IRenderer& renderer) {
    auto& st = undoStack(world);
    if (st.done.empty()) return false;
    UndoOp op = std::move(st.done.back());
    st.done.pop_back();
    if (op.undo) op.undo(world, renderer);
    SDL_Log("undo: %s", op.label.c_str());
    st.undone.push_back(std::move(op));
    return true;
}

bool redoLast(entt::registry& world, render::IRenderer& renderer) {
    auto& st = undoStack(world);
    if (st.undone.empty()) return false;
    UndoOp op = std::move(st.undone.back());
    st.undone.pop_back();
    if (op.redo) op.redo(world, renderer);
    SDL_Log("redo: %s", op.label.c_str());
    st.done.push_back(std::move(op));
    return true;
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
#define kVsCbPx  (0.8f * uiPx())   // square side
#define kVsCbPad (0.45f * uiPx())  // inset from the top/right edges

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

// --- Tree-view (scene outliner) widget -------------------------------------
constexpr int   kTreeQuads  = 600;            // dynamic mesh capacity
constexpr int   kTreeVerts  = kTreeQuads * 4;
#define kTreeTitleH (1.7f * uiPx())           // title bar (drag handle) height, px
#define kTreeRowH   (1.5f * uiPx())           // row height, px
#define kTreeTextPx (uiPx())                  // glyph height, px

// One row of the outliner -- a group header or an entity leaf. Rebuilt from the
// world each frame (shared by the input hit-test and the geometry builder so they
// always agree on row layout).
struct TreeRow {
    bool         isGroup  = false;
    bool         expanded = false;
    bool         selected = false;
    bool         shown    = true;          // Renderable::visible (group: any child)
    bool         hasEye   = true;          // false = no eye icon (empty group)
    int          depth    = 0;
    int          group    = 0;             // group id (for header toggle)
    entt::entity entity   = entt::null;    // leaf entity (null for headers)
    char         label[48] = {0};
};

// Right-edge eye-icon hit zone of every row (panel px from the right edge).
// Clicking it toggles Renderable::visible (a group header eye toggles all its
// children) instead of selecting the row.
constexpr float kTreeEyeZone = 34.0f;

void buildTreeRows(entt::registry& world, const TreeView& tv, std::vector<TreeRow>& rows) {
    static const char* kGroupName[TreeView::kGroups] = {"Meshes", "Point Clouds"};
    auto v = world.view<Renderable>();
    for (int g = 0; g < TreeView::kGroups; ++g) {
        const bool wantCloud = (g == 1);
        int count = 0;
        for (auto e : v)
            if (v.get<Renderable>(e).pointCloud == wantCloud) ++count;

        bool anyShown = false;
        for (auto e : v) {
            const auto& r = v.get<Renderable>(e);
            if (r.pointCloud == wantCloud && r.visible) { anyShown = true; break; }
        }

        TreeRow hr;
        hr.isGroup = true; hr.expanded = tv.expanded[g]; hr.group = g; hr.depth = 0;
        hr.shown   = anyShown;
        hr.hasEye  = count > 0;
        std::snprintf(hr.label, sizeof(hr.label), "%s (%d)", kGroupName[g], count);
        rows.push_back(hr);

        if (!tv.expanded[g]) continue;
        for (auto e : v) {
            const auto& r = v.get<Renderable>(e);
            if (r.pointCloud != wantCloud) continue;
            TreeRow row;
            row.depth = 1; row.selected = r.selected; row.entity = e;
            row.shown = r.visible;
            unsigned int id = (unsigned int)entt::to_integral(e);
            std::snprintf(row.label, sizeof(row.label), "%s %u", wantCloud ? "Cloud" : "Mesh", id);
            rows.push_back(row);
        }
    }
}

float treeMaxScroll(const TreeView& tv, size_t rowCount) {
    float content = (float)rowCount * kTreeRowH;
    float viewH   = (float)tv.h - kTreeTitleH;
    return (std::max)(0.0f, content - viewH);
}

// Builds the panel in normalized [0,1]^2 (y-up); rects are specified in panel
// pixels measured from the top-left. Rows are drawn first, then the title bar on
// top (higher z) so rows scrolled up vanish under it.
void buildTreeGeometry(const TreeView& tv, const std::vector<TreeRow>& rows, render::Vertex* out) {
    int q = 0;
    if (!tv.font) { for (int i = 0; i < kTreeVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}}; return; }
    const core::Font& f = *tv.font;
    const float wu = f.whiteU, wv = f.whiteV;
    const float W = (float)tv.w, H = (float)tv.h;
    const float xs = H / W;                         // glyph horizontal aspect fix
    auto nx    = [&](float px) { return px / W; };
    auto nyTop = [&](float px) { return 1.0f - px / H; };
    // Rect in panel pixels (x0<x1 left/right, t0<t1 top/bottom from the top edge).
    auto rect = [&](float x0, float t0, float x1, float t1, float r, float g, float b, float z) {
        if (q >= kTreeQuads) return;
        float y0 = nyTop(t1), y1 = nyTop(t0);       // bottom, top in y-up
        float X0 = nx(x0), X1 = nx(x1);
        out[q * 4 + 0] = {{X0, y0, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 1] = {{X1, y0, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 2] = {{X1, y1, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 3] = {{X0, y1, z}, {r, g, b}, {wu, wv}};
        ++q;
    };

    rect(0, 0, W, H, 0.07f, 0.08f, 0.10f, 0.0f);    // panel background

    const float lblH  = kTreeTextPx / H;            // leaf glyph height
    const float grpH  = kTreeTextPx / H;            // header glyph height
    const float grpCol[3]  = {0.90f, 0.93f, 0.97f};
    const float leafCol[3] = {0.74f, 0.80f, 0.86f};
    const float arwCol[3]  = {0.62f, 0.70f, 0.80f};

    for (size_t i = 0; i < rows.size(); ++i) {
        float top = kTreeTitleH + (float)i * kTreeRowH - tv.scroll;
        if (top + kTreeRowH <= kTreeTitleH - 0.5f) continue;  // scrolled above (hidden by title)
        if (top >= H) break;                                  // below the panel
        const TreeRow& row = rows[i];

        if ((int)i == tv.hover)
            rect(2.0f, top, W - 2.0f, top + kTreeRowH, 0.16f, 0.18f, 0.22f, 0.2f);
        if (row.selected)
            rect(2.0f, top, W - 2.0f, top + kTreeRowH, 0.15f, 0.40f, 0.24f, 0.25f);

        float indent   = 8.0f + (float)row.depth * 14.0f;
        float baseline = top + kTreeRowH * 0.72f;
        if (row.isGroup) {
            const char arrow[2] = {row.expanded ? '-' : '+', '\0'};
            appendText(out, q, kTreeQuads, f, arrow, nx(indent), nyTop(baseline), grpH, arwCol,
                       0.35f, xs);
            indent += 14.0f;
        }
        const float* col = row.isGroup ? grpCol : leafCol;
        appendText(out, q, kTreeQuads, f, row.label, nx(indent), nyTop(baseline),
                   row.isGroup ? grpH : lblH, col, 0.35f, xs);

        // Visibility eye at the right edge (axis-aligned quads off the white
        // texel -- the font bakes ASCII only, so the icon is drawn, not typed).
        // Open eye = lens (three stacked bands) + dark pupil; hidden = dim bar.
        if (row.hasEye) {
            float ex = W - kTreeEyeZone + 4.0f;           // icon left (22px wide)
            float cy = top + kTreeRowH * 0.5f;            // row center
            if (row.shown) {
                const float er = 0.78f, eg = 0.84f, eb = 0.90f;
                rect(ex,     cy - 3.5f, ex + 22, cy + 3.5f, er, eg, eb, 0.35f);  // mid band
                rect(ex + 4, cy - 7.0f, ex + 18, cy - 3.5f, er, eg, eb, 0.35f);  // upper
                rect(ex + 4, cy + 3.5f, ex + 18, cy + 7.0f, er, eg, eb, 0.35f);  // lower
                rect(ex + 8.5f, cy - 3.5f, ex + 13.5f, cy + 3.5f,                // pupil
                     0.10f, 0.12f, 0.16f, 0.4f);
            } else {
                rect(ex + 2, cy - 2.0f, ex + 20, cy + 2.0f, 0.38f, 0.42f, 0.48f, 0.35f);
            }
        }
    }

    // Title bar on top (covers any row scrolled up under it).
    rect(0, 0, W, kTreeTitleH, 0.11f, 0.12f, 0.15f, 0.6f);
    rect(0, kTreeTitleH - 1.0f, W, kTreeTitleH, 0.03f, 0.03f, 0.04f, 0.62f);  // divider
    const float titleCol[3] = {0.92f, 0.95f, 0.93f};
    appendText(out, q, kTreeQuads, f, "Scene", nx(9.0f), nyTop(kTreeTitleH * 0.70f),
               kTreeTextPx / H, titleCol, 0.75f, xs);

    for (int i = q * 4; i < kTreeVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Camera controls panel -------------------------------------------------
constexpr int kCtrlQuads = 64;
constexpr int kCtrlVerts = kCtrlQuads * 4;

struct CRect { float x, y, w, h; };          // screen-pixel rect
struct CtrlRects { CRect mode, minus, plus; };

CtrlRects controlRects(const CameraControls& cc) {
    CtrlRects r;
    float pad = 6.0f * uiScale(), bw = 30.0f * uiScale();
    r.mode  = {cc.x + pad, cc.y + pad, cc.w - 2 * pad, cc.h * 0.42f};
    float row2 = cc.y + cc.h * 0.50f, bh = cc.h * 0.42f;
    r.minus = {cc.x + pad, row2, bw, bh};
    r.plus  = {cc.x + cc.w - pad - bw, row2, bw, bh};
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
    const float s = uiScale(), pad = 8.0f * s;
    r.enable = {cs.x + cs.w - pad - 16.0f * s, cs.y + 6.0f * s, 16.0f * s, 16.0f * s};
    r.axis   = {cs.x + pad, cs.y + 30.0f * s, 46.0f * s, 22.0f * s};
    r.flip   = {cs.x + cs.w - pad - 52.0f * s, cs.y + 30.0f * s, 52.0f * s, 22.0f * s};
    r.track  = {cs.x + 12.0f * s, cs.y + 72.0f * s, cs.w - 24.0f * s, 6.0f * s};
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
    float th = kUiTextPx / cs.h;
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
    float vh = kUiTextPx / cs.h;  // match the controls value (FOV) size above
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

// --- 3D Compare legend -------------------------------------------------------
constexpr int kLegendQuads = 128;
constexpr int kLegendVerts = kLegendQuads * 4;

// Close-box rect (screen px). Shared by the builder and the input system.
CRect legendCloseRect(const CompareLegend& lg) {
    float u = uiPx();
    return {lg.x + lg.w - 2.2f * u, lg.y + 0.5f * u, 1.6f * u, 1.6f * u};
}
// Color-bar geometry in PANEL pixels, shared by the builder and the thumb
// hit-testing. All track the app font size.
#define kLgBarL    (0.6f * uiPx())
#define kLgBarR    (2.0f * uiPx())
#define kLgBarT    (2.2f * uiPx())
#define kLgBarPadB (2.0f * uiPx())
float legendBarBottom(const CompareLegend& lg) { return lg.h - kLgBarPadB; }
// Thumb center Y in panel px for a normalized selection (0 = bottom).
float legendThumbY(const CompareLegend& lg, float sel) {
    float barB = legendBarBottom(lg);
    return barB - (barB - kLgBarT) * sel;
}
// Selection value in scalar units.
float legendSelValue(const CompareLegend& lg, float sel) {
    return lg.isSigned ? -lg.range + sel * 2.0f * lg.range : sel * lg.range;
}

// Vertical banded color bar (top = +range / red, bottom = -range / blue) with a
// numeric tick beside every 2nd band boundary, plus the RMS readout. The band
// colors come from the same compareBandColor() used to paint the mesh, so the
// legend and the model always agree.
void buildCompareLegendGeometry(const CompareLegend& lg, render::Vertex* out) {
    const core::Font& f = *lg.font;
    int q = 0;
    const float wu = f.whiteU, wv = f.whiteV;
    auto quad = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                    float u0, float v0, float u1, float v1, float z) {
        if (q >= kLegendQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {u0, v1}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {u1, v1}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {u1, v0}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {u0, v0}};
        ++q;
    };
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) { quad(x0, y0, x1, y1, r, g, b, wu, wv, wu, wv, z); };
    const float xs = static_cast<float>(lg.h) / static_cast<float>(lg.w);
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
    auto nx    = [&](float px) { return px / lg.w; };
    auto nyTop = [&](float px) { return 1.0f - px / lg.h; };

    solid(0, 0, 1, 1, 0.10f, 0.11f, 0.13f, 0.0f);  // panel background

    float th = uiPx() / lg.h;  // the app font size IS the legend's size
    {   // shrink long titles (mode names) to the panel width
        float tw   = f.textWidth(lg.title.c_str(), th) * xs;
        float maxW = (lg.w - 3.2f * uiPx()) / lg.w;
        if (tw > maxW) th *= maxW / tw;
    }
    text(lg.title.c_str(), nx(0.4f * uiPx()), nyTop(1.1f * uiPx()), th,
         0.85f, 0.88f, 0.92f, 0.6f);

    // Close box.
    CRect cb = legendCloseRect(lg);
    float cx0 = nx(cb.x - lg.x), cx1 = nx(cb.x - lg.x + cb.w);
    float cy1 = nyTop(cb.y - lg.y), cy0 = nyTop(cb.y - lg.y + cb.h);
    solid(cx0, cy0, cx1, cy1, 0.22f, 0.24f, 0.29f, 0.5f);
    float xh = (cy1 - cy0) * 0.7f;
    text("x", (cx0 + cx1) * 0.5f - f.textWidth("x", xh) * xs * 0.5f,
         (cy0 + cy1) * 0.5f - xh * 0.32f, xh, 0.85f, 0.88f, 0.92f, 0.6f);

    // Color bar: bands stacked bottom (lowest deviation) -> top (highest).
    // bands == 1 means a smooth ramp: draw it as 32 thin strips.
    const float barL = kLgBarL, barR = kLgBarR, barT = kLgBarT, barB = legendBarBottom(lg);
    const int   n    = lg.bands > 1 ? lg.bands : 32;
    for (int b = 0; b < n; ++b) {
        float t0 = (float)b / n, t1 = (float)(b + 1) / n;   // 0 = bottom
        // Band-center deviation -> exact painted color.
        float tc  = (t0 + t1) * 0.5f;
        float dev = legendSelValue(lg, tc);
        Eigen::Vector3f c = geometry::compareBandColor(dev, lg.range, lg.isSigned, lg.bands);
        float py0 = barB - (barB - barT) * t0;  // band bottom (px from panel top)
        float py1 = barB - (barB - barT) * t1;  // band top
        solid(nx(barL), nyTop(py0), nx(barR), nyTop(py1), c.x(), c.y(), c.z(), 0.3f);
    }

    // Tick labels at 0, 1/4, 1/2, 3/4, 1 of the bar (bottom -> top).
    const float u  = uiPx();
    const float lh = 0.75f * u / lg.h;
    char buf[24];
    for (int k = 0; k <= 4; ++k) {
        float t   = (float)k / 4.0f;
        float dev = legendSelValue(lg, t);
        std::snprintf(buf, sizeof(buf), lg.isSigned ? "%+.3g" : "%.3g", dev);
        float py = barB - (barB - barT) * t;  // px from top
        solid(nx(barR), nyTop(py + 2), nx(barR + 0.25f * u), nyTop(py - 2),  // tick mark
              0.62f, 0.66f, 0.72f, 0.35f);
        text(buf, nx(barR + 0.4f * u), nyTop(py + 0.25f * u), lh, 0.80f, 0.84f, 0.90f, 0.6f);
    }

    // Range-selection thumbs (mode legends): dim the excluded bar segments and
    // draw a labeled handle at each end of the kept range.
    if (lg.fromMode) {
        float yMin = legendThumbY(lg, lg.selMin), yMax = legendThumbY(lg, lg.selMax);
        if (lg.selMin > 0.001f)  // below-min mask
            solid(nx(barL), nyTop(barB), nx(barR), nyTop(yMin), 0.13f, 0.14f, 0.17f, 0.42f);
        if (lg.selMax < 0.999f)  // above-max mask
            solid(nx(barL), nyTop(yMax), nx(barR), nyTop(barT), 0.13f, 0.14f, 0.17f, 0.42f);
        auto thumb = [&](float py, float sel, bool isMin) {
            solid(nx(barL - 0.2f * u), nyTop(py + 5), nx(barR + 0.2f * u), nyTop(py - 5),
                  0.92f, 0.94f, 0.98f, 0.55f);  // handle bar
            std::snprintf(buf, sizeof(buf), lg.isSigned ? "%+.3g" : "%.3g",
                          legendSelValue(lg, sel));
            float ty = isMin ? py + 0.85f * u : py - 0.35f * u;  // below min / above max
            text(buf, nx(barL), nyTop(ty), lh, 0.95f, 0.96f, 1.0f, 0.6f);
        };
        thumb(yMin, lg.selMin, true);
        thumb(yMax, lg.selMax, false);
    }

    if (lg.rms >= 0.0f) {  // mode legends have no RMS
        std::snprintf(buf, sizeof(buf), "RMS %.3g", lg.rms);
        text(buf, nx(0.6f * u), nyTop(lg.h - 0.6f * u), lh, 0.72f, 0.78f, 0.85f, 0.6f);
    }

    for (int i = q * 4; i < kLegendVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Poisson reconstruction dialog -----------------------------------------
constexpr int kPoissonQuads = 256;
constexpr int kPoissonVerts = kPoissonQuads * 4;

// Per-slider metadata shared by the builder and the input system.
struct PSliderSpec { const char* label; float vmin, vmax; bool isInt; };
constexpr PSliderSpec kPSliders[4] = {
    {"Depth",      4.0f, 14.0f, true},   // 2^depth grid; dense, so ~9-10 is the RAM ceiling
    {"Iterations", 4.0f, 80.0f, true},   // Gauss-Seidel passes
    {"Scale",      1.0f, 2.0f,  false},  // bounds padding
    {"Pt Weight",  0.0f, 16.0f, false},  // screening strength
};

float poissonSliderValue(const PoissonDialog& d, int i) {
    switch (i) {
        case 0:  return (float)d.depth;
        case 1:  return (float)d.iterations;
        case 2:  return d.scale;
        default: return d.pointWeight;
    }
}
void poissonSetSliderValue(PoissonDialog& d, int i, float v) {
    switch (i) {
        case 0:  d.depth       = (int)std::lround(v); break;
        case 1:  d.iterations  = (int)std::lround(v); break;
        case 2:  d.scale       = v; break;
        default: d.pointWeight = v; break;
    }
}

struct PoissonRects { CRect track[4]; CRect run; CRect close; };
PoissonRects poissonRects(const PoissonDialog& d) {
    PoissonRects r;
    const float pad = 16.0f;
    for (int i = 0; i < 4; ++i) {
        float rowTop = d.y + 42.0f + i * 36.0f;
        r.track[i] = {d.x + pad, rowTop + 22.0f, d.w - 2 * pad, 6.0f};
    }
    r.run   = {d.x + pad, (float)(d.y + d.h) - 38.0f, d.w - 2 * pad, 28.0f};
    r.close = {(float)(d.x + d.w) - 24.0f, (float)d.y + 7.0f, 16.0f, 16.0f};
    return r;
}
float poissonHandleX(const PSliderSpec& s, float val, const CRect& track) {
    float t = s.vmax > s.vmin ? (val - s.vmin) / (s.vmax - s.vmin) : 0.5f;
    return track.x + clampf(t, 0.0f, 1.0f) * track.w;
}

// Builds the Poisson dialog: background, title + close, four labeled sliders
// (groove + filled + handle + value readout) and a "Reconstruct" button.
void buildPoissonDialogGeometry(const PoissonDialog& d, render::Vertex* out) {
    const core::Font& f = *d.font;
    int q = 0;
    const float wu = f.whiteU, wv = f.whiteV;
    auto quad = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                    float u0, float v0, float u1, float v1, float z) {
        if (q >= kPoissonQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {u0, v1}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {u1, v1}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {u1, v0}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {u0, v0}};
        ++q;
    };
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) { quad(x0, y0, x1, y1, r, g, b, wu, wv, wu, wv, z); };
    const float xs = static_cast<float>(d.h) / static_cast<float>(d.w);  // aspect fix
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
        x0 = (rr.x - d.x) / d.w;            x1 = (rr.x + rr.w - d.x) / d.w;
        y1 = 1.0f - (rr.y - d.y) / d.h;     y0 = 1.0f - (rr.y + rr.h - d.y) / d.h;
    };

    solid(0, 0, 1, 1, 0.10f, 0.11f, 0.13f, 0.0f);                       // panel background
    solid(0, 1.0f - 28.0f / d.h, 1, 1, 0.16f, 0.18f, 0.22f, 0.05f);     // title bar

    PoissonRects R = poissonRects(d);
    float x0, y0, x1, y1;

    // Title. Match the cross-section panel's 20px title.
    float th = kUiTextPx / d.h;
    text("Poisson Reconstruction", 12.0f / d.w, 1.0f - 7.0f / d.h - th, th,
         0.88f, 0.90f, 0.94f, 0.6f);

    // Close button (x).
    toN(R.close, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.34f, 0.20f, 0.22f, 0.5f);
    float ch = (y1 - y0) * 0.7f;
    text("x", (x0 + x1) * 0.5f - textW("x", ch) * 0.5f, (y0 + y1) * 0.5f - ch * 0.35f, ch,
         0.92f, 0.86f, 0.86f, 0.6f);

    // Sliders.
    char buf[32];
    for (int i = 0; i < 4; ++i) {
        const PSliderSpec& s = kPSliders[i];
        const CRect& trk = R.track[i];
        float val = poissonSliderValue(d, i);

        // Label (left) at the row top, above the groove. 17px (panel default).
        float lh = kUiTextPx / d.h;
        float labelY = 1.0f - (trk.y - 18.0f - d.y) / d.h;
        text(s.label, (trk.x - d.x) / d.w, labelY, lh, 0.80f, 0.84f, 0.90f, 0.6f);

        // Value (right).
        if (s.isInt) std::snprintf(buf, sizeof(buf), "%d", (int)std::lround(val));
        else         std::snprintf(buf, sizeof(buf), "%.2f", val);
        text(buf, (trk.x + trk.w - d.x) / d.w - textW(buf, lh), labelY, lh,
             0.70f, 0.85f, 1.0f, 0.6f);

        // Groove + filled + handle.
        float hx = poissonHandleX(s, val, trk);
        toN(trk, x0, y0, x1, y1);
        solid(x0, y0, x1, y1, 0.20f, 0.21f, 0.25f, 0.3f);              // groove
        float hxn = (hx - d.x) / d.w;
        solid(x0, y0, hxn, y1, 0.30f, 0.55f, 0.95f, 0.35f);           // filled
        float hw = 5.0f / d.w;
        float hy0 = 1.0f - (trk.y + trk.h * 0.5f + 9.0f - d.y) / d.h;
        float hy1 = 1.0f - (trk.y + trk.h * 0.5f - 9.0f - d.y) / d.h;
        solid(hxn - hw, hy0, hxn + hw, hy1, 0.95f, 0.95f, 0.95f, 0.5f);  // handle
    }

    // Reconstruct button.
    toN(R.run, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.20f, 0.42f, 0.30f, 0.3f);
    float bh = (y1 - y0) * 0.62f;
    text("Reconstruct", (x0 + x1) * 0.5f - textW("Reconstruct", bh) * 0.5f,
         (y0 + y1) * 0.5f - bh * 0.35f, bh, 0.90f, 0.96f, 0.92f, 0.6f);

    for (int i = q * 4; i < kPoissonVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Mode-parameters dialog (generic sliders for the active geometry mode) ----
// Same visual language as the Poisson dialog, but the rows come from the modes
// registry (modes::modeParam) so every parameterised mode shares one panel.
constexpr int kModeDlgQuads = 512;  // 9+ param rows x per-glyph quads

constexpr int kModeDlgVerts = kModeDlgQuads * 4;

struct ModeDlgRects {
    CRect track[12]; CRect minus[12]; CRect plus[12]; CRect apply; CRect fit; CRect close;
    CRect lgBar;    // horizontal legend color bar (when the mode has a legend)
    CRect extract;  // "Extract" button under the bar
};
// Extra dialog height when the legend strip (bar + labels + Extract) shows.
#define kModeDlgLegendH (4.8f * uiPx())
// All row metrics scale with the app font size (uiScale() = 1 at the legacy
// 20px layout the literals were designed for).
ModeDlgRects modeDlgRects(const ModeParamsDialog& d, int nParams, bool hasLegend) {
    ModeDlgRects r;
    const float s = uiScale();
    const float pad = 16.0f * s, btn = 18.0f * s, gap = 6.0f * s;
    // Row pitch 48: label line (~uiPx tall) + groove/handle band. The label
    // baseline sits at rowTop + 18 (see the geometry builder) so glyph tops
    // stay inside the row instead of colliding with the previous row's handle.
    for (int i = 0; i < nParams && i < 12; ++i) {
        float rowTop = d.y + (42.0f + i * 48.0f) * s;
        // Track leaves room for the -/+ nudge buttons on its right.
        float trackW = d.w - 2 * pad - 2 * (btn + gap);
        r.track[i] = {d.x + pad, rowTop + 30.0f * s, trackW, 6.0f * s};
        float by = rowTop + (30.0f + 3.0f) * s - btn * 0.5f;  // centered on the groove
        r.minus[i] = {d.x + pad + trackW + gap, by, btn, btn};
        r.plus[i]  = {d.x + pad + trackW + gap + btn + gap, by, btn, btn};
    }
    if (hasLegend) {
        float lt  = d.y + (42.0f + (nParams < 12 ? nParams : 12) * 48.0f + 4.0f) * s;
        r.lgBar   = {d.x + pad, lt + 24.0f * s, d.w - 2 * pad, 16.0f * s};
        r.extract = {d.x + pad, lt + 62.0f * s, d.w - 2 * pad, 26.0f * s};
    } else {
        r.lgBar = r.extract = {0, 0, 0, 0};
    }
    if ((modes::modeCanFit(d.modeIndex) || modes::modeAppliesMesh(d.modeIndex)) &&
        d.pipeNodeId < 0) {
        r.apply = {d.x + pad, (float)(d.y + d.h) - 74.0f * s, d.w - 2 * pad, 28.0f * s};
        r.fit   = {d.x + pad, (float)(d.y + d.h) - 38.0f * s, d.w - 2 * pad, 28.0f * s};
    } else {
        r.apply = {d.x + pad, (float)(d.y + d.h) - 38.0f * s, d.w - 2 * pad, 28.0f * s};
        r.fit   = {0, 0, 0, 0};
    }
    r.close = {(float)(d.x + d.w) - 24.0f * s, (float)d.y + 7.0f * s, 16.0f * s, 16.0f * s};
    return r;
}

// Section-header checkbox (the header row's enable flag): left-aligned on the
// row, centered on the header text line. Shared by the geometry builder and
// the input hit test.
CRect modeHeaderCheckRect(const CRect& trk, float s) {
    float sz = 16.0f * s;
    return {trk.x, trk.y - 26.0f * s, sz, sz};
}

// One increment of param `s` for the -/+ buttons and slider snapping.
float modeParamStep(const modes::ModeParam& s) {
    if (s.step > 0.0f) return s.step;
    if (s.isInt) return 1.0f;
    return (s.maxV - s.minV) / 100.0f;
}
// Clamp + snap a raw value to the param's range/step grid.
float modeParamSnap(const modes::ModeParam& s, float v) {
    float st = modeParamStep(s);
    if (s.step > 0.0f || s.isInt)
        v = s.minV + std::round((v - s.minV) / st) * st;
    return clampf(v, s.minV, s.maxV);
}

// The mode legend shown INSIDE the params dialog (and driving its range strip):
// the shared CompareLegend component while a mode owns it.
CompareLegend* activeModeLegend(entt::registry& world) {
    auto v = world.view<CompareLegend>();
    for (auto e : v) {
        auto& lg = v.get<CompareLegend>(e);
        return lg.visible && lg.fromMode ? &lg : nullptr;
    }
    return nullptr;
}

void buildModeParamsDialogGeometry(const ModeParamsDialog& d, const CompareLegend* lg,
                                   render::Vertex* out) {
    const core::Font& f = *d.font;
    int q = 0;
    const float wu = f.whiteU, wv = f.whiteV;
    auto quad = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                    float u0, float v0, float u1, float v1, float z) {
        if (q >= kModeDlgQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {u0, v1}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {u1, v1}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {u1, v0}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {u0, v0}};
        ++q;
    };
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) { quad(x0, y0, x1, y1, r, g, b, wu, wv, wu, wv, z); };
    const float xs = static_cast<float>(d.h) / static_cast<float>(d.w);  // aspect fix
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
        x0 = (rr.x - d.x) / d.w;            x1 = (rr.x + rr.w - d.x) / d.w;
        y1 = 1.0f - (rr.y - d.y) / d.h;     y0 = 1.0f - (rr.y + rr.h - d.y) / d.h;
    };

    const float sc = uiScale();
    solid(0, 0, 1, 1, 0.10f, 0.11f, 0.13f, 0.0f);                       // panel background
    solid(0, 1.0f - 28.0f * sc / d.h, 1, 1, 0.16f, 0.18f, 0.22f, 0.05f);  // title bar

    const int nParams = modes::modeParamCount(d.modeIndex);
    ModeDlgRects R = modeDlgRects(d, nParams, lg != nullptr);
    float x0, y0, x1, y1;

    float th = kUiTextPx / d.h;
    text(modes::modeName(d.modeIndex), 12.0f * sc / d.w, 1.0f - 7.0f * sc / d.h - th, th,
         0.88f, 0.90f, 0.94f, 0.6f);

    // Close button (x).
    toN(R.close, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.34f, 0.20f, 0.22f, 0.5f);
    float ch = (y1 - y0) * 0.7f;
    text("x", (x0 + x1) * 0.5f - textW("x", ch) * 0.5f, (y0 + y1) * 0.5f - ch * 0.35f, ch,
         0.92f, 0.86f, 0.86f, 0.6f);

    // Sliders (a "--" name is a section header: centered label between divider
    // lines, no slider row).
    char buf[32];
    for (int i = 0; i < nParams && i < 12; ++i) {
        modes::ModeParam s = modes::modeParam(d.modeIndex, i);
        const CRect& trk = R.track[i];
        float val = d.values[i];

        float lh = kUiTextPx / d.h;
        // Label baseline 12px above the groove: with the 48px row pitch the
        // glyph tops stay below the previous row's handle band.
        float labelY = 1.0f - (trk.y - 12.0f * sc - d.y) / d.h;
        if (modes::modeParamIsHeader(s)) {
            // Header = labeled divider + section-enable checkbox (the row's
            // value slot). Box geometry must match modeHeaderCheckRect.
            const bool on = val != 0.0f;
            // Strip the "--" fencing off the stored name.
            const char* nm = s.name + 2;
            while (*nm == ' ' || *nm == '-') ++nm;
            size_t len = std::strlen(nm);
            while (len && (nm[len - 1] == ' ' || nm[len - 1] == '-')) --len;
            char hbuf[32];
            std::snprintf(hbuf, sizeof(hbuf), "%.*s", (int)(len < 31 ? len : 31), nm);
            float tw = textW(hbuf, lh);
            float cx = (trk.x - d.x) / d.w;
            float cw = (R.plus[i].x + R.plus[i].w - trk.x) / d.w;  // full row width
            float midY = labelY + lh * 0.30f;  // divider through the cap midline
            float lineH = 1.5f * sc / d.h;
            float tx0 = cx + (cw - tw) * 0.5f;
            CRect cb = modeHeaderCheckRect(R.track[i], sc);
            float bx0, by0, bx1, by1;
            toN(cb, bx0, by0, bx1, by1);
            solid(bx0, by0, bx1, by1, 0.70f, 0.74f, 0.80f, 0.3f);  // border
            float inx = 2.0f * sc / d.w, iny = 2.0f * sc / d.h;
            solid(bx0 + inx, by0 + iny, bx1 - inx, by1 - iny, 0.13f, 0.14f, 0.17f, 0.35f);
            if (on)
                solid(bx0 + 2 * inx, by0 + 2 * iny, bx1 - 2 * inx, by1 - 2 * iny,
                      0.30f, 0.75f, 0.42f, 0.4f);  // green tick fill
            float hr = on ? 0.62f : 0.45f, hg = on ? 0.72f : 0.50f, hb = on ? 0.86f : 0.58f;
            solid((cb.x + cb.w + 6.0f * sc - d.x) / d.w, midY, tx0 - 6.0f * sc / d.w,
                  midY + lineH, 0.36f, 0.40f, 0.48f, 0.3f);
            solid(tx0 + tw + 6.0f * sc / d.w, midY, cx + cw, midY + lineH,
                  0.36f, 0.40f, 0.48f, 0.3f);
            text(hbuf, tx0, labelY, lh, hr, hg, hb, 0.6f);
            continue;
        }
        text(s.name, (trk.x - d.x) / d.w, labelY, lh, 0.80f, 0.84f, 0.90f, 0.6f);

        if (s.isInt)                              std::snprintf(buf, sizeof(buf), "%d", (int)std::lround(val));
        else if (s.step > 0.0f && s.step < 0.01f) std::snprintf(buf, sizeof(buf), "%.3f", val);
        else                                      std::snprintf(buf, sizeof(buf), "%.2f", val);
        text(buf, (R.plus[i].x + R.plus[i].w - d.x) / d.w - textW(buf, lh), labelY, lh,
             0.70f, 0.85f, 1.0f, 0.6f);

        float t = s.maxV > s.minV ? (val - s.minV) / (s.maxV - s.minV) : 0.5f;
        float hx = trk.x + clampf(t, 0.0f, 1.0f) * trk.w;
        toN(trk, x0, y0, x1, y1);
        solid(x0, y0, x1, y1, 0.20f, 0.21f, 0.25f, 0.3f);              // groove
        float hxn = (hx - d.x) / d.w;
        solid(x0, y0, hxn, y1, 0.30f, 0.55f, 0.95f, 0.35f);            // filled
        float hw = 5.0f * sc / d.w;
        float hy0 = 1.0f - (trk.y + trk.h * 0.5f + 9.0f * sc - d.y) / d.h;
        float hy1 = 1.0f - (trk.y + trk.h * 0.5f - 9.0f * sc - d.y) / d.h;
        solid(hxn - hw, hy0, hxn + hw, hy1, 0.95f, 0.95f, 0.95f, 0.5f);  // handle

        // -/+ nudge buttons (one step per click).
        const char* signs[2] = {"-", "+"};
        const CRect* brc[2]  = {&R.minus[i], &R.plus[i]};
        for (int b = 0; b < 2; ++b) {
            toN(*brc[b], x0, y0, x1, y1);
            solid(x0, y0, x1, y1, 0.20f, 0.24f, 0.30f, 0.4f);
            float sh = (y1 - y0) * 0.7f;
            text(signs[b], (x0 + x1) * 0.5f - textW(signs[b], sh) * 0.5f,
                 (y0 + y1) * 0.5f - sh * 0.35f, sh, 0.85f, 0.90f, 0.95f, 0.6f);
        }
    }

    // Legend strip: horizontal color bar + range thumbs + Extract button.
    if (lg) {
        const CRect& bar = R.lgBar;
        toN(bar, x0, y0, x1, y1);
        const int strips = lg->bands > 1 ? lg->bands : 32;
        for (int b = 0; b < strips; ++b) {
            float t0 = (float)b / strips, t1 = (float)(b + 1) / strips;  // 0 = left = min
            float dev = legendSelValue(*lg, (t0 + t1) * 0.5f);
            Eigen::Vector3f c =
                geometry::compareBandColor(dev, lg->range, lg->isSigned, lg->bands);
            solid(x0 + (x1 - x0) * t0, y0, x0 + (x1 - x0) * t1, y1, c.x(), c.y(), c.z(),
                  0.3f);
        }
        // Dim the excluded ends + white thumb handles at selMin/selMax.
        float xMin = x0 + (x1 - x0) * lg->selMin, xMax = x0 + (x1 - x0) * lg->selMax;
        if (lg->selMin > 0.001f) solid(x0, y0, xMin, y1, 0.13f, 0.14f, 0.17f, 0.42f);
        if (lg->selMax < 0.999f) solid(xMax, y0, x1, y1, 0.13f, 0.14f, 0.17f, 0.42f);
        float tw2 = 3.0f / d.w, ty0 = y0 - 5.0f / d.h, ty1 = y1 + 5.0f / d.h;
        solid(xMin - tw2, ty0, xMin + tw2, ty1, 0.95f, 0.95f, 0.98f, 0.55f);
        solid(xMax - tw2, ty0, xMax + tw2, ty1, 0.95f, 0.95f, 0.98f, 0.55f);
        // Thumb values above the bar (min left-aligned, max right-aligned).
        float lh2 = kUiTextPx / d.h;
        std::snprintf(buf, sizeof(buf), lg->isSigned ? "%+.3g" : "%.3g",
                      legendSelValue(*lg, lg->selMin));
        text(buf, x0, y1 + 4.0f / d.h, lh2, 0.90f, 0.92f, 0.98f, 0.6f);
        std::snprintf(buf, sizeof(buf), lg->isSigned ? "%+.3g" : "%.3g",
                      legendSelValue(*lg, lg->selMax));
        text(buf, x1 - textW(buf, lh2), y1 + 4.0f / d.h, lh2, 0.90f, 0.92f, 0.98f, 0.6f);
        // Extract button.
        toN(R.extract, x0, y0, x1, y1);
        solid(x0, y0, x1, y1, 0.45f, 0.34f, 0.16f, 0.3f);
        float eh = (y1 - y0) * 0.62f;
        text("Extract", (x0 + x1) * 0.5f - textW("Extract", eh) * 0.5f,
             (y0 + y1) * 0.5f - eh * 0.35f, eh, 0.98f, 0.92f, 0.80f, 0.6f);
    }

    // Apply button.
    toN(R.apply, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.20f, 0.42f, 0.30f, 0.3f);
    float bh = (y1 - y0) * 0.62f;
    text("Apply", (x0 + x1) * 0.5f - textW("Apply", bh) * 0.5f,
         (y0 + y1) * 0.5f - bh * 0.35f, bh, 0.90f, 0.96f, 0.92f, 0.6f);

    // Fit button (modes with a fit action only; hidden in node-editing mode).
    if ((modes::modeCanFit(d.modeIndex) || modes::modeAppliesMesh(d.modeIndex)) &&
        d.pipeNodeId < 0) {
        toN(R.fit, x0, y0, x1, y1);
        solid(x0, y0, x1, y1, 0.22f, 0.34f, 0.52f, 0.3f);
        float fh = (y1 - y0) * 0.62f;
        text("Fit", (x0 + x1) * 0.5f - textW("Fit", fh) * 0.5f,
             (y0 + y1) * 0.5f - fh * 0.35f, fh, 0.90f, 0.94f, 0.98f, 0.6f);
    }

    for (int i = q * 4; i < kModeDlgVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Pipeline Design dialog (Blueprint-style node canvas, modeless) -----------
constexpr int kPipeDlgQuads = 2048;
constexpr int kPipeDlgVerts = kPipeDlgQuads * 4;
#define kPipeText   (uiPx())                  // one app-wide text size
#define kPipeTitleH (2.2f * uiPx())           // title bar
#define kPipeGrip   (0.9f * uiPx())           // resize-grip square (bottom-right)
#define kPipeNodeW  (10.0f * uiPx())          // node box (px)
#define kPipeNodeH  (3.4f * uiPx())
#define kPipePinR   (0.45f * uiPx())          // pin half-size (px)
#define kPipeBtnW   (6.0f * uiPx())           // palette button (px)
#define kPipeBtnH   (1.3f * uiPx())
constexpr float kPipeBtnGap  = 5.0f;

// One palette entry. The list is generated from the modes registry: Source,
// then a stage for every mode with a points->points action, plus a
// "<name> Fit" stage where the mode's Fit action is distinct, then Output.
// Adding a mode in modes.cpp makes it appear on the palette automatically.
struct PipeStageInfo {
    PipeNodeRole role;
    std::string  label;  // compact palette/node caption
    std::string  mode;   // backing mode display name (Stage only)
    bool         fit = false;
};
const std::vector<PipeStageInfo>& pipeStageList() {
    static const std::vector<PipeStageInfo> list = [] {
        // Registry names are descriptive; compact them for 86px buttons.
        auto shortLabel = [](std::string s) {
            auto rep = [&](const char* a, const char* b) {
                size_t p = s.find(a);
                if (p != std::string::npos) s.replace(p, std::strlen(a), b);
            };
            rep("Outlier: ", "");
            rep("Divergence Select", "Div Select");
            rep(" (Dev Gate)", " Dev");
            rep(" (bilateral)", "");
            rep("Surface Dev (MLS)", "MLS");
            rep(" (MLS+PFOR)", "");
            rep(" (Div Gate)", " Div");
            rep("Bump: Remove", "Bump Rm");
            rep("Morphology", "Morph");
            rep("SDF Filter", "SDF");
            return s;
        };
        std::vector<PipeStageInfo> v;
        v.push_back({PipeNodeRole::Source, "Source", "", false});
        for (int i = 0; i < modes::modeCount(); ++i) {
            std::string name = modes::modeName(i);
            if (modes::modeCanTransformPoints(i))
                v.push_back({PipeNodeRole::Stage, shortLabel(name), name, false});
            if (modes::modeFitIsDistinct(i))
                v.push_back({PipeNodeRole::Stage, shortLabel(name) + " Fit", name, true});
        }
        v.push_back({PipeNodeRole::Output, "Output", "", false});
        return v;
    }();
    return list;
}
// Palette caption for a node (stage label, or the raw mode name if the mode
// vanished from the registry after a file load).
std::string pipeNodeLabel(const PipeNode& n) {
    if (n.role == PipeNodeRole::Source) return "Source";
    if (n.role == PipeNodeRole::Output) return "Output";
    for (const auto& s : pipeStageList())
        if (s.role == PipeNodeRole::Stage && s.mode == n.mode && s.fit == n.fit) return s.label;
    return n.mode.empty() ? "?" : n.mode;
}

CRect pipeCloseRect(const PipelineDialog& d) {
    return {(float)(d.x + d.w) - 34.0f, (float)d.y + 10.0f, 24.0f, 24.0f};
}
CRect pipeGripRect(const PipelineDialog& d) {
    return {(float)(d.x + d.w) - kPipeGrip, (float)(d.y + d.h) - kPipeGrip,
            kPipeGrip, kPipeGrip};
}
// Node box in SCREEN pixels (canvas coords + pan + panel origin).
CRect pipeNodeRect(const PipelineDialog& d, const PipeNode& n) {
    return {d.x + 10.0f + d.panX + n.x, d.y + kPipeTitleH + 4.0f + d.panY + n.y,
            kPipeNodeW, kPipeNodeH};
}
// Input (left) / output (right) pin centers.
void pipePinCenters(const CRect& nr, float& inX, float& inY, float& outX, float& outY) {
    inX  = nr.x;             inY  = nr.y + nr.h * 0.5f;
    outX = nr.x + nr.w;      outY = nr.y + nr.h * 0.5f;
}
// Condition (candidate-set) input pin: lower-left, only on candidate-aware stages.
void pipeCondPinCenter(const CRect& nr, float& cx, float& cy) {
    cx = nr.x;
    cy = nr.y + nr.h * 0.85f;
}
bool pipeNodeHasCondPin(const PipeNode& n) {
    return n.role == PipeNodeRole::Stage &&
           modes::modeSupportsCandidates(modes::modeIndexByName(n.mode.c_str()));
}
CRect pipeNodeCloseRect(const CRect& nr) {
    return {nr.x + nr.w - 24.0f, nr.y + 3.0f, 20.0f, 20.0f};
}
// Palette buttons wrap into as many bottom-bar rows as the stage list needs
// at the dialog's current width; Save/Load/Run live in the title bar.
int pipePaletteCols(const PipelineDialog& d) {
    int c = (int)((d.w - 16.0f) / (kPipeBtnW + kPipeBtnGap));
    return c < 1 ? 1 : c;
}
int pipePaletteRows(const PipelineDialog& d) {
    int slots = (int)pipeStageList().size();
    int cols  = pipePaletteCols(d);
    return (slots + cols - 1) / cols;
}
float pipePaletteH(const PipelineDialog& d) {
    return pipePaletteRows(d) * (kPipeBtnH + kPipeBtnGap) + 7.0f;
}
CRect pipePaletteRect(const PipelineDialog& d, int slot) {
    int cols = pipePaletteCols(d);
    int row = slot / cols, col = slot % cols;
    return {d.x + 8.0f + col * (kPipeBtnW + kPipeBtnGap),
            (float)(d.y + d.h) - pipePaletteH(d) + 6.0f + row * (kPipeBtnH + kPipeBtnGap),
            kPipeBtnW, kPipeBtnH};
}
CRect pipeRunRect(const PipelineDialog& d) {
    return {(float)(d.x + d.w) - 34.0f - 5.0f - 76.0f, (float)d.y + 10.0f, 76.0f, 24.0f};
}
CRect pipeSaveRect(const PipelineDialog& d) {
    CRect run = pipeRunRect(d);
    return {run.x - 2 * (62.0f + 5.0f), run.y, 62.0f, run.h};
}
CRect pipeLoadRect(const PipelineDialog& d) {
    CRect run = pipeRunRect(d);
    return {run.x - (62.0f + 5.0f), run.y, 62.0f, run.h};
}
const PipeNode* pipeFindNode(const PipelineDialog& d, int id) {
    for (const auto& n : d.nodes)
        if (n.id == id) return &n;
    return nullptr;
}
PipeNode* pipeFindNodeMut(PipelineDialog& d, int id) {
    for (auto& n : d.nodes)
        if (n.id == id) return &n;
    return nullptr;
}
// Number of editable params of a node's backing mode (0 for Source/Output).
int pipeNodeParamCount(const PipeNode& n) {
    if (n.role != PipeNodeRole::Stage) return 0;
    return modes::modeParamCount(modes::modeIndexByName(n.mode.c_str()));
}
// Seed a fresh node's params from the mode's registered defaults.
void pipeInitNodeParams(PipeNode& n) {
    if (n.role != PipeNodeRole::Stage) return;
    int idx = modes::modeIndexByName(n.mode.c_str());
    int cnt = modes::modeParamCount(idx);
    n.params.resize(cnt);
    for (int i = 0; i < cnt; ++i) n.params[i] = modes::modeParam(idx, i).defV;
}

// Graph persistence: a simple line format next to the exe.
std::string pipeGraphPath() {
    const char* base = SDL_GetBasePath();  // owned by SDL
    return (base ? std::string(base) : std::string()) + "orange_pipeline.txt";
}
void pipeSaveGraph(const PipelineDialog& d) {
    std::ofstream f(pipeGraphPath(), std::ios::trunc);
    if (!f) { SDL_Log("Pipeline: save failed (%s)", pipeGraphPath().c_str()); return; }
    // v2: MLS kind inserted; v3: per-node param count (was fixed 4);
    // v4: role + fit + mode NAME (trailing, may contain spaces) replace the
    // kind int, so stages survive mode reordering/insertion in modes.cpp.
    f << "ver 4\n";
    f << "pan " << d.panX << " " << d.panY << "\n";
    for (const auto& n : d.nodes) {
        f << "node " << n.id << " " << (int)n.role << " " << (n.fit ? 1 : 0) << " "
          << n.x << " " << n.y << " " << n.params.size();
        for (float p : n.params) f << " " << p;
        if (n.role == PipeNodeRole::Stage) f << " " << n.mode;
        f << "\n";
    }
    for (const auto& l : d.links)
        f << "link " << l.from << " " << l.to << " " << l.toPin << "\n";
    SDL_Log("Pipeline: saved %zu node(s), %zu link(s)", d.nodes.size(), d.links.size());
}
void pipeLoadGraph(PipelineDialog& d) {
    std::ifstream f(pipeGraphPath());
    if (!f) { SDL_Log("Pipeline: nothing to load (%s)", pipeGraphPath().c_str()); return; }
    std::vector<PipeNode> nodes;
    std::vector<PipeLink> links;
    float panX = 0, panY = 0;
    int maxId = 0;
    int ver   = 1;  // files without a "ver" line predate the version tag
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "ver") {
            ss >> ver;
        } else if (tag == "pan") {
            ss >> panX >> panY;
        } else if (tag == "node") {
            PipeNode n;
            std::vector<float> vals;
            if (ver >= 4) {
                int role = 0, fit = 0;
                size_t pn = 0;
                ss >> n.id >> role >> fit >> n.x >> n.y >> pn;
                for (float v; vals.size() < pn && (ss >> v);) vals.push_back(v);
                if (role < 0 || role > (int)PipeNodeRole::Output) continue;
                n.role = (PipeNodeRole)role;
                n.fit  = fit != 0;
                if (n.role == PipeNodeRole::Stage) {  // rest of the line = mode name
                    std::string name;
                    std::getline(ss, name);
                    size_t b = name.find_first_not_of(' ');
                    n.mode = b == std::string::npos ? std::string() : name.substr(b);
                }
            } else {
                // v1-v3 stored a kind int indexing the old fixed table.
                int kind = 0;
                ss >> n.id >> kind >> n.x >> n.y;
                if (ver >= 3) {
                    size_t pn = 0;
                    ss >> pn;
                    for (float v; vals.size() < pn && (ss >> v);) vals.push_back(v);
                } else {  // v1/v2 wrote exactly 4 values per node
                    for (float v; vals.size() < 4 && (ss >> v);) vals.push_back(v);
                }
                if (ver < 2 && kind == 7) kind = 8;  // v1 predates the MLS kind
                struct Legacy { PipeNodeRole role; const char* mode; bool fit; };
                static const Legacy kLegacy[9] = {
                    {PipeNodeRole::Source, "", false},
                    {PipeNodeRole::Stage, "Outlier: SOR", false},
                    {PipeNodeRole::Stage, "Outlier: ROR", false},
                    {PipeNodeRole::Stage, "Outlier: PFOR", false},
                    {PipeNodeRole::Stage, "Outlier: PFOR", true},
                    {PipeNodeRole::Stage, "Smooth (bilateral)", false},
                    {PipeNodeRole::Stage, "Bump: Remove", false},
                    {PipeNodeRole::Stage, "Surface Dev (MLS)", false},
                    {PipeNodeRole::Output, "", false},
                };
                if (kind < 0 || kind >= 9) continue;
                n.role = kLegacy[kind].role;
                n.mode = kLegacy[kind].mode;
                n.fit  = kLegacy[kind].fit;
            }
            // Size to the backing mode's CURRENT param count (defaults), then
            // overlay whatever the file had -- old files stay loadable when a
            // mode gains parameters.
            pipeInitNodeParams(n);
            for (size_t i = 0; i < vals.size() && i < n.params.size(); ++i)
                n.params[i] = vals[i];
            maxId = std::max(maxId, n.id);
            nodes.push_back(n);
        } else if (tag == "link") {
            PipeLink l;
            ss >> l.from >> l.to;
            if (!(ss >> l.toPin)) l.toPin = 0;  // pre-v4 links: main pin
            links.push_back(l);
        }
    }
    if (nodes.empty()) { SDL_Log("Pipeline: load found no nodes"); return; }
    d.nodes  = std::move(nodes);
    d.links  = std::move(links);
    d.panX   = panX;
    d.panY   = panY;
    d.nextId = maxId + 1;
    d.linkFrom = -1;
    d.dragNode = -1;
    SDL_Log("Pipeline: loaded %zu node(s), %zu link(s)", d.nodes.size(), d.links.size());
}

void buildPipelineDialogGeometry(const PipelineDialog& d, render::Vertex* out) {
    const core::Font& f = *d.font;
    int q = 0;
    const float wu = f.whiteU, wv = f.whiteV;
    auto quad = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                    float u0, float v0, float u1, float v1, float z) {
        if (q >= kPipeDlgQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {u0, v1}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {u1, v1}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {u1, v0}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {u0, v0}};
        ++q;
    };
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) { quad(x0, y0, x1, y1, r, g, b, wu, wv, wu, wv, z); };
    const float xs = static_cast<float>(d.h) / static_cast<float>(d.w);  // aspect fix
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
        x0 = (rr.x - d.x) / d.w;            x1 = (rr.x + rr.w - d.x) / d.w;
        y1 = 1.0f - (rr.y - d.y) / d.h;     y0 = 1.0f - (rr.y + rr.h - d.y) / d.h;
    };

    // Arbitrary-corner quad (for wires). Corners in normalized panel space,
    // given in the same relative order as quad() so the winding matches.
    auto quad4 = [&](float ax, float ay, float bx, float by, float cx, float cy, float dx2,
                     float dy2, float r, float g, float b, float z) {
        if (q >= kPipeDlgQuads) return;
        out[q * 4 + 0] = {{ax, ay, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 1] = {{bx, by, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 2] = {{cx, cy, z}, {r, g, b}, {wu, wv}};
        out[q * 4 + 3] = {{dx2, dy2, z}, {r, g, b}, {wu, wv}};
        ++q;
    };
    // Line segment between two SCREEN-pixel points, drawn as a thin quad.
    auto segPx = [&](float ax, float ay, float bx, float by, float t, float r, float g,
                     float b, float z) {
        if (bx < ax || (bx == ax && by < ay)) { std::swap(ax, bx); std::swap(ay, by); }
        float ddx = bx - ax, ddy = by - ay;
        float len = std::sqrt(ddx * ddx + ddy * ddy);
        if (len < 1e-3f) return;
        float px = -ddy / len * t * 0.5f, py = ddx / len * t * 0.5f;
        auto nx = [&](float sx) { return (sx - d.x) / d.w; };
        auto ny = [&](float sy) { return 1.0f - (sy - d.y) / d.h; };
        quad4(nx(ax - px), ny(ay - py), nx(bx - px), ny(by - py),
              nx(bx + px), ny(by + py), nx(ax + px), ny(ay + py), r, g, b, z);
    };
    // Blueprint-style wire: cubic bezier with horizontal tangents.
    auto wirePx = [&](float ax, float ay, float bx, float by, float r, float g, float b) {
        float dist = std::max(30.0f, std::abs(bx - ax) * 0.5f);
        float c1x = ax + dist, c1y = ay, c2x = bx - dist, c2y = by;
        float lx = ax, ly = ay;
        const int N = 14;
        for (int s = 1; s <= N; ++s) {
            float t = (float)s / N, u = 1.0f - t;
            float px2 = u * u * u * ax + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * bx;
            float py2 = u * u * u * ay + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * by;
            segPx(lx, ly, px2, py2, 3.0f, r, g, b, 0.2f);
            lx = px2; ly = py2;
        }
    };

    solid(0, 0, 1, 1, 0.10f, 0.11f, 0.13f, 0.0f);                          // panel background
    solid(0, 1.0f - kPipeTitleH / d.h, 1, 1, 0.16f, 0.18f, 0.22f, 0.05f);  // title bar
    // Bottom palette bar (as many rows as the stage list needs at this width).
    solid(0, 0, 1, pipePaletteH(d) / d.h, 0.13f, 0.14f, 0.17f, 0.1f);

    float x0, y0, x1, y1;
    float th = kPipeText / d.h;
    text("Pipeline Design", 12.0f / d.w,
         1.0f - (kPipeTitleH * 0.5f + kPipeText * 0.5f) / d.h, th,
         0.88f, 0.90f, 0.94f, 0.6f);

    // Close button (x).
    toN(pipeCloseRect(d), x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.34f, 0.20f, 0.22f, 0.5f);
    float ch = kPipeText / d.h;
    text("x", (x0 + x1) * 0.5f - textW("x", ch) * 0.5f, (y0 + y1) * 0.5f - ch * 0.35f, ch,
         0.92f, 0.86f, 0.86f, 0.6f);

    // Wires (under the nodes). Condition wires land on the lower cond pin and
    // draw amber so the two inputs read apart at a glance.
    for (const auto& l : d.links) {
        const PipeNode* a = pipeFindNode(d, l.from);
        const PipeNode* b = pipeFindNode(d, l.to);
        if (!a || !b) continue;
        CRect ra = pipeNodeRect(d, *a), rb = pipeNodeRect(d, *b);
        float ix, iy, ox, oy, ix2, iy2, ox2, oy2;
        pipePinCenters(ra, ix, iy, ox, oy);
        pipePinCenters(rb, ix2, iy2, ox2, oy2);
        if (l.toPin == 1) pipeCondPinCenter(rb, ix2, iy2);
        if (l.toPin == 1) wirePx(ox, oy, ix2, iy2, 0.95f, 0.75f, 0.35f);
        else              wirePx(ox, oy, ix2, iy2, 0.55f, 0.75f, 0.95f);
    }
    // Pending wire follows the cursor.
    if (d.linkFrom >= 0) {
        const PipeNode* a = pipeFindNode(d, d.linkFrom);
        if (a) {
            CRect ra = pipeNodeRect(d, *a);
            float ix, iy, ox, oy;
            pipePinCenters(ra, ix, iy, ox, oy);
            wirePx(ox, oy, d.panSX, d.panSY, 0.9f, 0.8f, 0.3f);  // panSX/Y = cursor (see input)
        }
    }

    // Nodes.
    float lh = kPipeText / d.h;
    char pbuf[48];
    for (const auto& n : d.nodes) {
        CRect nr = pipeNodeRect(d, n);
        toN(nr, x0, y0, x1, y1);
        bool hov = d.hoverNode == n.id;
        // Body + header strip.
        solid(x0, y0, x1, y1, hov ? 0.22f : 0.17f, hov ? 0.26f : 0.20f, hov ? 0.34f : 0.27f,
              0.3f);
        bool src = n.role == PipeNodeRole::Source, outk = n.role == PipeNodeRole::Output;
        solid(x0, y1 - 28.0f / d.h, x1, y1,
              src ? 0.18f : (outk ? 0.38f : 0.24f),
              src ? 0.40f : (outk ? 0.24f : 0.33f),
              src ? 0.26f : (outk ? 0.20f : 0.50f), 0.35f);
        // Label (in the header strip).
        std::string lbl = pipeNodeLabel(n);
        float tx = x0 + 8.0f / d.w;
        text(lbl.c_str(), tx, y1 - 25.0f / d.h, lh, 0.92f, 0.94f, 0.98f, 0.6f);
        // Param summary (body line, e.g. "16 / 0.050").
        int pc = pipeNodeParamCount(n);
        if (pc > 0) {
            int idx = modes::modeIndexByName(n.mode.c_str());
            std::string s;
            for (int pi = 0; pi < pc && pi < (int)n.params.size(); ++pi) {
                modes::ModeParam mp = modes::modeParam(idx, pi);
                if (modes::modeParamIsHeader(mp)) continue;  // header slot, no value
                if (mp.isInt) std::snprintf(pbuf, sizeof(pbuf), "%d", (int)std::lround(n.params[pi]));
                else if (mp.step > 0.0f && mp.step < 0.01f)
                    std::snprintf(pbuf, sizeof(pbuf), "%.3f", n.params[pi]);
                else std::snprintf(pbuf, sizeof(pbuf), "%.2f", n.params[pi]);
                if (!s.empty()) s += " / ";
                s += pbuf;
            }
            text(s.c_str(), tx, y0 + 10.0f / d.h, lh, 0.70f, 0.80f, 0.92f, 0.6f);
        }
        // Node close (x).
        CRect cxr = pipeNodeCloseRect(nr);
        float cx0, cy0, cx1, cy1;
        toN(cxr, cx0, cy0, cx1, cy1);
        text("x", cx0 + 3.0f / d.w, cy0 + 1.0f / d.h, lh, 0.90f, 0.60f, 0.60f, 0.6f);
        // Pins.
        float ix, iy, ox, oy;
        pipePinCenters(nr, ix, iy, ox, oy);
        auto pin = [&](float px2, float py2, bool lit) {
            float p0x = (px2 - kPipePinR - d.x) / d.w, p1x = (px2 + kPipePinR - d.x) / d.w;
            float p0y = 1.0f - (py2 + kPipePinR - d.y) / d.h;
            float p1y = 1.0f - (py2 - kPipePinR - d.y) / d.h;
            solid(p0x, p0y, p1x, p1y, lit ? 0.95f : 0.55f, lit ? 0.85f : 0.75f,
                  lit ? 0.30f : 0.95f, 0.5f);
        };
        if (!src) pin(ix, iy, false);
        if (!outk) pin(ox, oy, d.linkFrom == n.id);
        if (pipeNodeHasCondPin(n)) {  // amber condition pin (candidate set in)
            float cx2, cy2;
            pipeCondPinCenter(nr, cx2, cy2);
            float p0x = (cx2 - kPipePinR - d.x) / d.w, p1x = (cx2 + kPipePinR - d.x) / d.w;
            float p0y = 1.0f - (cy2 + kPipePinR - d.y) / d.h;
            float p1y = 1.0f - (cy2 - kPipePinR - d.y) / d.h;
            solid(p0x, p0y, p1x, p1y, 0.95f, 0.75f, 0.35f, 0.5f);
        }
    }

    // Palette buttons + Save/Load + Run.
    auto button = [&](const CRect& br, const char* lbl2, float r, float g, float b) {
        toN(br, x0, y0, x1, y1);
        solid(x0, y0, x1, y1, r, g, b, 0.4f);
        float bh = kPipeText / d.h;  // same size as every other label
        text(lbl2, (x0 + x1) * 0.5f - textW(lbl2, bh) * 0.5f, (y0 + y1) * 0.5f - bh * 0.35f,
             bh, 0.88f, 0.92f, 0.96f, 0.6f);
    };
    const auto& stages = pipeStageList();
    for (int s = 0; s < (int)stages.size(); ++s)
        button(pipePaletteRect(d, s), stages[s].label.c_str(),
               stages[s].role == PipeNodeRole::Stage ? 0.20f : 0.24f,
               stages[s].role == PipeNodeRole::Stage ? 0.24f : 0.30f, 0.30f);
    button(pipeSaveRect(d), "Save", 0.26f, 0.26f, 0.36f);
    button(pipeLoadRect(d), "Load", 0.26f, 0.26f, 0.36f);
    button(pipeRunRect(d), "Run", 0.20f, 0.42f, 0.30f);

    // Resize grip strokes.
    for (int s = 0; s < 3; ++s) {
        float inset = 3.0f + s * 4.0f;
        float gx0 = (d.w - kPipeGrip + inset) / d.w, gy0 = inset / d.h;
        solid(gx0, gy0, gx0 + 2.0f / d.w, gy0 + (kPipeGrip - inset - 2.0f) / d.h,
              0.45f, 0.50f, 0.58f, 0.5f);
    }

    for (int i = q * 4; i < kPipeDlgVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Confirm (Yes/No) dialog -----------------------------------------------
constexpr int kConfirmQuads = 192;
constexpr int kConfirmVerts = kConfirmQuads * 4;

struct ConfirmRects { CRect yes; CRect no; };
ConfirmRects confirmRects(const ConfirmDialog& d) {
    const float s = uiScale();
    const float bw = 92.0f * s, bh = 30.0f * s, pad = 16.0f * s, gap = 10.0f * s;
    float by = (float)(d.y + d.h) - pad - bh;
    ConfirmRects r;
    r.yes = {(float)(d.x + d.w) - pad - bw, by, bw, bh};
    r.no  = {r.yes.x - gap - bw, by, bw, bh};
    return r;
}

void buildConfirmDialogGeometry(const ConfirmDialog& d, render::Vertex* out) {
    const core::Font& f = *d.font;
    int q = 0;
    const float wu = f.whiteU, wv = f.whiteV;
    auto quad = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                    float u0, float v0, float u1, float v1, float z) {
        if (q >= kConfirmQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {u0, v1}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {u1, v1}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {u1, v0}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {u0, v0}};
        ++q;
    };
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) { quad(x0, y0, x1, y1, r, g, b, wu, wv, wu, wv, z); };
    const float xs = (float)d.h / (float)d.w;
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
        x0 = (rr.x - d.x) / d.w;         x1 = (rr.x + rr.w - d.x) / d.w;
        y1 = 1.0f - (rr.y - d.y) / d.h;  y0 = 1.0f - (rr.y + rr.h - d.y) / d.h;
    };

    const float sc = uiScale();
    solid(0, 0, 1, 1, 0.10f, 0.11f, 0.13f, 0.0f);                    // panel
    solid(0, 1.0f - 28.0f * sc / d.h, 1, 1, 0.16f, 0.18f, 0.22f, 0.05f);  // title bar
    float th = kUiTextPx / d.h;
    text(d.title.c_str(), 12.0f * sc / d.w, 1.0f - 7.0f * sc / d.h - th, th,
         0.88f, 0.90f, 0.94f, 0.6f);

    float mh = kUiTextPx / d.h;
    text(d.line1.c_str(), 16.0f * sc / d.w, 1.0f - 58.0f * sc / d.h, mh,
         0.85f, 0.88f, 0.92f, 0.6f);
    if (!d.line2.empty())
        text(d.line2.c_str(), 16.0f * sc / d.w, 1.0f - 84.0f * sc / d.h, mh,
             0.70f, 0.82f, 0.98f, 0.6f);

    ConfirmRects R = confirmRects(d);
    float x0, y0, x1, y1;
    toN(R.no, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.28f, 0.29f, 0.33f, 0.3f);
    float bbh = (y1 - y0) * 0.5f;
    text("No", (x0 + x1) * 0.5f - textW("No", bbh) * 0.5f, (y0 + y1) * 0.5f - bbh * 0.35f,
         bbh, 0.92f, 0.92f, 0.92f, 0.6f);
    toN(R.yes, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.20f, 0.42f, 0.30f, 0.3f);
    text("Yes", (x0 + x1) * 0.5f - textW("Yes", bbh) * 0.5f, (y0 + y1) * 0.5f - bbh * 0.35f,
         bbh, 0.90f, 0.96f, 0.92f, 0.6f);

    for (int i = q * 4; i < kConfirmVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Font Size dialog (View menu) -------------------------------------------
// A slider (12..64 px) plus a numeric input box; both drive core::setUiFontPx,
// which every overlay reads live -- the whole UI rescales as you drag.
constexpr int kFontDlgQuads = 128;
constexpr int kFontDlgVerts = kFontDlgQuads * 4;
constexpr float kFontPxMin = 12.0f, kFontPxMax = 64.0f;

struct FontDlgRects { CRect track; CRect box; CRect close; };
FontDlgRects fontDlgRects(const FontSizeDialog& d) {
    FontDlgRects r;
    const float s = uiScale(), pad = 16.0f * s;
    float rowY = d.y + 52.0f * s;
    float boxW = 64.0f * s;
    r.track = {d.x + pad, rowY + 8.0f * s, d.w - 2 * pad - boxW - 10.0f * s, 6.0f * s};
    r.box   = {(float)(d.x + d.w) - pad - boxW, rowY - 6.0f * s, boxW, 28.0f * s};
    r.close = {(float)(d.x + d.w) - 24.0f * s, (float)d.y + 7.0f * s, 16.0f * s, 16.0f * s};
    return r;
}

void buildFontSizeDialogGeometry(const FontSizeDialog& d, render::Vertex* out) {
    const core::Font& f = *d.font;
    int q = 0;
    const float wu = f.whiteU, wv = f.whiteV;
    auto quad = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                    float u0, float v0, float u1, float v1, float z) {
        if (q >= kFontDlgQuads) return;
        out[q * 4 + 0] = {{x0, y0, z}, {r, g, b}, {u0, v1}};
        out[q * 4 + 1] = {{x1, y0, z}, {r, g, b}, {u1, v1}};
        out[q * 4 + 2] = {{x1, y1, z}, {r, g, b}, {u1, v0}};
        out[q * 4 + 3] = {{x0, y1, z}, {r, g, b}, {u0, v0}};
        ++q;
    };
    auto solid = [&](float x0, float y0, float x1, float y1, float r, float g, float b,
                     float z) { quad(x0, y0, x1, y1, r, g, b, wu, wv, wu, wv, z); };
    const float xs = static_cast<float>(d.h) / static_cast<float>(d.w);
    auto text = [&](const char* s2, float penX, float baseY, float h, float r, float g,
                    float b, float z) {
        for (; *s2; ++s2) {
            const core::Glyph& gl = f.glyph(*s2);
            if (gl.w > 0 && gl.h > 0) {
                float x0 = penX + gl.xoff * h * xs, y1 = baseY - gl.yoff * h;
                float x1 = x0 + gl.w * h * xs, y0 = y1 - gl.h * h;
                quad(x0, y0, x1, y1, r, g, b, gl.u0, gl.v0, gl.u1, gl.v1, z);
            }
            penX += gl.advance * h * xs;
        }
    };
    auto textW = [&](const char* s2, float h) { return f.textWidth(s2, h) * xs; };
    auto toN = [&](const CRect& rr, float& x0, float& y0, float& x1, float& y1) {
        x0 = (rr.x - d.x) / d.w;            x1 = (rr.x + rr.w - d.x) / d.w;
        y1 = 1.0f - (rr.y - d.y) / d.h;     y0 = 1.0f - (rr.y + rr.h - d.y) / d.h;
    };

    const float sc = uiScale();
    solid(0, 0, 1, 1, 0.10f, 0.11f, 0.13f, 0.0f);
    solid(0, 1.0f - 28.0f * sc / d.h, 1, 1, 0.16f, 0.18f, 0.22f, 0.05f);

    FontDlgRects R = fontDlgRects(d);
    float x0, y0, x1, y1;
    float th = kUiTextPx / d.h;
    text("Font Size", 12.0f * sc / d.w, 1.0f - 7.0f * sc / d.h - th, th,
         0.88f, 0.90f, 0.94f, 0.6f);

    toN(R.close, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.34f, 0.20f, 0.22f, 0.5f);
    float ch = (y1 - y0) * 0.7f;
    text("x", (x0 + x1) * 0.5f - textW("x", ch) * 0.5f, (y0 + y1) * 0.5f - ch * 0.35f, ch,
         0.92f, 0.86f, 0.86f, 0.6f);

    // Slider (12..64) with the handle at the current size.
    const float cur = core::uiFontPx();
    float t = (cur - kFontPxMin) / (kFontPxMax - kFontPxMin);
    toN(R.track, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, 0.20f, 0.21f, 0.25f, 0.3f);
    float hxn = x0 + (x1 - x0) * clampf(t, 0.0f, 1.0f);
    solid(x0, y0, hxn, y1, 0.30f, 0.55f, 0.95f, 0.35f);
    float hw = 5.0f * sc / d.w;
    float hy0 = 1.0f - (R.track.y + R.track.h * 0.5f + 9.0f * sc - d.y) / d.h;
    float hy1 = 1.0f - (R.track.y + R.track.h * 0.5f - 9.0f * sc - d.y) / d.h;
    solid(hxn - hw, hy0, hxn + hw, hy1, 0.95f, 0.95f, 0.95f, 0.5f);

    // Number box: shows the edit buffer while focused (with a caret), else the
    // current value. Click to focus, type digits, Enter to apply.
    toN(R.box, x0, y0, x1, y1);
    solid(x0, y0, x1, y1, d.editing ? 0.16f : 0.13f, d.editing ? 0.22f : 0.15f,
          d.editing ? 0.34f : 0.19f, 0.3f);
    char buf[16];
    if (d.editing)
        std::snprintf(buf, sizeof(buf), "%s_", d.editBuf.c_str());
    else
        std::snprintf(buf, sizeof(buf), "%d", (int)std::lround(cur));
    float bh = (y1 - y0) * 0.62f;
    text(buf, (x0 + x1) * 0.5f - textW(buf, bh) * 0.5f, (y0 + y1) * 0.5f - bh * 0.35f, bh,
         0.90f, 0.94f, 1.0f, 0.6f);

    for (int i = q * 4; i < kFontDlgVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}

// --- Top menu bar ----------------------------------------------------------
constexpr int   kMenuQuads = 1024;          // must match kMenuQ in main.cpp
constexpr int   kMenuVerts = kMenuQuads * 4;
#define kMenuTitlePad (0.7f * uiPx())       // l/r padding around a bar title (px)
#define kMenuItemH    (1.9f * uiPx())       // dropdown item row height (px)
#define kMenuSepH     (0.55f * uiPx())      // separator row height (px)
#define kMenuCheckW   (1.2f * uiPx())       // left gutter holding the check tick
#define kMenuPadX     (0.7f * uiPx())       // dropdown left/right padding
#define kMenuShortGap (1.4f * uiPx())       // gap between label and the shortcut
#define kMenuMinDropW (9.5f * uiPx())       // minimum dropdown width (px)
#define kMenuTextH    (uiPx())              // dropdown/title text px height

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

// Panel width for a list of items: the widest item (check gutter + label +
// optional shortcut + submenu arrow), clamped to a minimum.
float menuItemsWidth(const std::vector<MenuItem>& items, const core::Font& f, float th) {
    float w = kMenuMinDropW;
    for (const auto& it : items) {
        if (it.kind == MenuItem::Separator) continue;
        float row = kMenuCheckW + f.textWidth(it.label.c_str(), th) + kMenuPadX;
        if (!it.shortcut.empty())
            row += kMenuShortGap + f.textWidth(it.shortcut.c_str(), th);
        if (!it.submenu.empty())
            row += kMenuShortGap + f.textWidth(">", th);
        w = (std::max)(w, row);
    }
    return w;
}

// Total panel height (sum of row heights; separators are shorter).
float menuItemsHeight(const std::vector<MenuItem>& items) {
    float h = 0.0f;
    for (const auto& it : items) h += (it.kind == MenuItem::Separator ? kMenuSepH : kMenuItemH);
    return h;
}

// Top edge (px, relative to the panel top) of item `i`, and its height via out_h.
float menuItemTopIn(const std::vector<MenuItem>& items, int i, float* out_h) {
    float y = 0.0f;
    for (int k = 0; k < static_cast<int>(items.size()); ++k) {
        float h = (items[k].kind == MenuItem::Separator ? kMenuSepH : kMenuItemH);
        if (k == i) { if (out_h) *out_h = h; return y; }
        y += h;
    }
    if (out_h) *out_h = 0.0f;
    return y;
}

// Menu-level wrappers (operate on a Menu's item list).
float menuDropWidth(const Menu& m, const core::Font& f, float th) {
    return menuItemsWidth(m.items, f, th);
}
float menuDropHeight(const Menu& m) { return menuItemsHeight(m.items); }
float menuItemTop(const Menu& m, int i, float* out_h) {
    return menuItemTopIn(m.items, i, out_h);
}

// If a dropdown item with a submenu is currently open, return its flyout panel
// rect in pixels (fx0/fy0 top-left, fw/fh size) and the child item list; else
// return null. The flyout sits to the right of the dropdown, aligned to the
// parent item's top, and flips to the left if it would run off-screen.
const std::vector<MenuItem>* openFlyoutRect(const MenuBar& mb, const core::Font& f,
                                            float th, float W, float barH, float& fx0,
                                            float& fy0, float& fw, float& fh) {
    if (mb.openMenu < 0 || mb.openMenu >= static_cast<int>(mb.menus.size())) return nullptr;
    const Menu& m = mb.menus[mb.openMenu];
    if (mb.openSub < 0 || mb.openSub >= static_cast<int>(m.items.size())) return nullptr;
    const MenuItem& parent = m.items[mb.openSub];
    if (parent.submenu.empty()) return nullptr;

    float tx1, tx0 = menuTitleRect(mb, f, th, mb.openMenu, &tx1);
    float dw  = menuDropWidth(m, f, th);
    float dx0 = clampf(tx0, 0.0f, (std::max)(0.0f, W - dw));
    float ih, iy = barH + menuItemTop(m, mb.openSub, &ih);

    fw = menuItemsWidth(parent.submenu, f, th);
    fh = menuItemsHeight(parent.submenu);
    fx0 = dx0 + dw;
    if (fx0 + fw > W) fx0 = (std::max)(0.0f, dx0 - fw);  // flip left if off-screen
    fy0 = iy;
    return &parent.submenu;
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

    // Draw one dropdown/flyout panel of items at pixel rect (px0, pyTop) sized
    // (pw x its content height). Shared by the main dropdown and the flyout.
    auto drawPanel = [&](const std::vector<MenuItem>& items, float px0, float pw,
                         float pyTop, int hoverIdx) {
        float ph = menuItemsHeight(items);
        solid(px0, pyTop, px0 + pw, pyTop + ph, 0.13f, 0.14f, 0.17f, 0.2f);  // panel
        for (int i = 0; i < static_cast<int>(items.size()); ++i) {
            const MenuItem& it = items[i];
            float ih, iy = pyTop + menuItemTopIn(items, i, &ih);
            if (it.kind == MenuItem::Separator) {
                solid(px0 + kMenuPadX, iy + ih * 0.5f - 1.0f, px0 + pw - kMenuPadX,
                      iy + ih * 0.5f + 1.0f, 0.30f, 0.32f, 0.38f, 0.25f);
                continue;
            }
            if (i == hoverIdx)
                solid(px0, iy, px0 + pw, iy + ih, 0.24f, 0.30f, 0.42f, 0.25f);
            float baseY = iy + (ih + th) * 0.5f - 2.0f;
            if (it.kind == MenuItem::Check && it.checked)  // tick in the left gutter
                solid(px0 + 9.0f, iy + ih * 0.5f - 5.0f, px0 + 19.0f,
                      iy + ih * 0.5f + 5.0f, 0.55f, 0.85f, 1.0f, 0.3f);
            text(it.label.c_str(), px0 + kMenuCheckW, baseY, th, txt, 0.5f);
            if (!it.submenu.empty()) {                       // submenu arrow
                float aw = f.textWidth(">", th);
                text(">", px0 + pw - kMenuPadX - aw, baseY, th, dim, 0.5f);
            } else if (!it.shortcut.empty()) {
                float sw = f.textWidth(it.shortcut.c_str(), th);
                text(it.shortcut.c_str(), px0 + pw - kMenuPadX - sw, baseY, th, dim, 0.5f);
            }
        }
    };

    // Open dropdown.
    if (mb.openMenu >= 0 && mb.openMenu < static_cast<int>(mb.menus.size())) {
        const Menu& m = mb.menus[mb.openMenu];
        float tx1, tx0 = menuTitleRect(mb, f, th, mb.openMenu, &tx1);
        float dw = menuDropWidth(m, f, th);
        float dx0 = clampf(tx0, 0.0f, (std::max)(0.0f, W - dw));  // keep on screen
        drawPanel(m.items, dx0, dw, barH, mb.hoverItem);

        // Submenu flyout (one nesting level), to the right of the open item.
        float fx0, fy0, fw, fh;
        const std::vector<MenuItem>* sub =
            openFlyoutRect(mb, f, th, W, barH, fx0, fy0, fw, fh);
        if (sub) drawPanel(*sub, fx0, fw, fy0, mb.hoverSub);
    }

    for (int i = q * 4; i < kMenuVerts; ++i) out[i] = {{0, 0, 0}, {0, 0, 0}};
}
} // namespace

namespace {
// --- Left selection toolbar geometry ---------------------------------------
constexpr float kTbX       = 8.0f;                 // left margin (px)
#define kTbTop      ((float)kMenuBarHeight + 10.0f)
#define kTbBtn      (2.3f * uiPx())               // button size (px, square)
constexpr float kTbGap     = 4.0f;                 // gap between buttons in a group
constexpr float kTbGroupGap= 14.0f;               // gap between groups
#define kTbText     (uiPx())                      // button caption px height
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
            float tt = m.targetAnimDuration > 0.0f ? m.targetAnimTime / m.targetAnimDuration : 1.0f;
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
    mb->height = kMenuBarHeight;  // tracks the app font size

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

    // Submenu flyout: hovering a dropdown item with a submenu opens its flyout;
    // hovering a non-submenu item closes any flyout. While the pointer is inside
    // the flyout itself, hoverItem is -1 (it is outside the main panel) so
    // openSub is left untouched and the flyout stays open.
    if (mb->openMenu >= 0 && mb->openMenu < nMenus && mb->hoverItem >= 0) {
        const Menu& m = mb->menus[mb->openMenu];
        mb->openSub = m.items[mb->hoverItem].submenu.empty() ? -1 : mb->hoverItem;
    }
    mb->hoverSub = -1;
    bool inFlyout = false;
    {
        float fx0, fy0, fw, fh;
        const std::vector<MenuItem>* sub =
            openFlyoutRect(*mb, f, th, W, barH, fx0, fy0, fw, fh);
        if (sub && mx >= fx0 && mx < fx0 + fw && my >= fy0 && my < fy0 + fh) {
            inFlyout = true;
            inDrop   = true;  // treat the flyout as part of the dropdown
            float local = my - fy0;
            for (int i = 0; i < static_cast<int>(sub->size()); ++i) {
                float ih, iy = menuItemTopIn(*sub, i, &ih);
                if (local >= iy && local < iy + ih) {
                    if ((*sub)[i].kind != MenuItem::Separator) mb->hoverSub = i;
                    break;
                }
            }
        }
    }

    // Hovering a different title while a menu is open switches to it (classic UX).
    if (mb->openMenu >= 0 && onTitle >= 0 && onTitle != mb->openMenu) {
        mb->openMenu = onTitle;
        mb->openSub = -1; mb->hoverSub = -1;
    }

    // The bar (and an open dropdown/flyout) own their pixels: suppress orbit/picking.
    if (inBar || inDrop) input.captured = true;

    if (input.leftClicked) {
        if (onTitle >= 0) {
            mb->openMenu   = (mb->openMenu == onTitle) ? -1 : onTitle;  // toggle
            mb->openSub = -1; mb->hoverSub = -1;
            input.captured = true;
        } else if (inFlyout) {
            if (mb->hoverSub >= 0) {
                float fx0, fy0, fw, fh;
                const std::vector<MenuItem>* sub =
                    openFlyoutRect(*mb, f, th, W, barH, fx0, fy0, fw, fh);
                if (sub) mb->triggered = (*sub)[mb->hoverSub].action;
            }
            mb->openMenu = -1; mb->openSub = -1; mb->hoverSub = -1;
            input.captured = true;
        } else if (inDrop) {
            if (mb->hoverItem >= 0) {
                const Menu& m = mb->menus[mb->openMenu];
                const MenuItem& it = m.items[mb->hoverItem];
                if (!it.submenu.empty()) {
                    mb->openSub = mb->hoverItem;   // parent: open flyout, don't trigger
                } else {
                    mb->triggered = it.action;     // dispatched by the app
                    mb->openMenu = -1; mb->openSub = -1; mb->hoverSub = -1;
                }
            }
            input.captured = true;
        } else {
            mb->openMenu = -1; mb->openSub = -1; mb->hoverSub = -1;  // elsewhere closes
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
        act("Open...",    A::OpenFile, "Ctrl+O"),
        act("Save",       A::SaveFile, "Ctrl+S"),
        act("Save As...", A::SaveFileAs, "Ctrl+Shift+S"),
        sep(),
        act("Screenshot", A::Screenshot, "C"),
        sep(),
        act("Quit",       A::Quit, "Esc"),
    }});
    menus.push_back({"Edit", {
        act("Undo", A::Undo, "Ctrl+Z"),
        act("Redo", A::Redo, "Ctrl+Y"),
    }});
    menus.push_back({"View", {
        chk("Ground Grid",   A::ToggleGrid, "Space"),
        act("Reset Camera",  A::ResetCamera, "R"),
        sep(),
        act("Z-Up / Y-Up",   A::ToggleUpAxis),
        act("Perspective / Ortho", A::ToggleProjection),
        sep(),
        act("Font Size...",  A::FontSizeDialogToggle),
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
    // Geometry-processing operators, generated from the modes registry and grouped
    // by category (Generate / Analyze / Filter) with separators between groups.
    // Each mode i maps to MenuAction(Mode0 + i); adding a mode in modes.cpp makes
    // it appear here automatically (up to the Mode0..Mode23 action range).
    {
        Menu geo{"Geometry", {}};
        geo.items.push_back(act("Off", A::ModeOff));  // no active operator (default)
        geo.items.push_back(sep());
        modes::ModeCategory prev = modes::modeCategory(0);
        bool first = true;
        for (int i = 0; i < modes::modeCount() && i < 24; ++i) {
            modes::ModeCategory cat = modes::modeCategory(i);
            if (!first && cat != prev) geo.items.push_back(sep());
            first = false;
            prev = cat;
            geo.items.push_back(act(modes::modeName(i), static_cast<A>(static_cast<int>(A::Mode0) + i)));
        }
        geo.items.push_back(sep());
        geo.items.push_back(act("Poisson Reconstruction...", A::PoissonDialogToggle));
        geo.items.push_back(act("3D Compare (2 selected)",   A::Compare3D));
        menus.push_back(std::move(geo));
    }
    {  // Pipelines: multi-step geometry pipelines, with nested sub-steps.
        MenuItem occlusal{"Estimate Occlusal Plane", "", A::None, K::Action, false};
        occlusal.submenu = {
            act("Estimate Occlusal Plane", A::PipelineOcclusalEstimate),  // full pipeline
            sep(),
            act("Whole-mesh PCA",   A::PipelineOcclusalWholePCA),
            act("Find Cusp",        A::PipelineOcclusalFindCusp),
            act("Plane from Cusps", A::PipelineOcclusalPlaneFromCusps),
        };
        Menu pipe{"Pipelines", {}};
        pipe.items.push_back(occlusal);
        pipe.items.push_back(act("Occlusal 2D Render", A::PipelineOcclusal2DRender));
        pipe.items.push_back(act("Segment 2D (Classical)", A::PipelineSegment2DClassical));
        pipe.items.push_back(act("Segment 2D (SAM/AI)",    A::PipelineSegment2DSAM));
        pipe.items.push_back(act("Segment 3D (Teeth)",     A::PipelineSegment3D));
        pipe.items.push_back(sep());
        pipe.items.push_back(act("Pipeline Design...",     A::PipelineDialogToggle));
        menus.push_back(std::move(pipe));
    }
    menus.push_back({"Create", {
        act("Plane",    A::CreatePlane),
        act("Box",      A::CreateBox),
        act("Sphere",   A::CreateSphere),
        act("Cylinder", A::CreateCylinder),
        act("Cone",     A::CreateCone),
        act("Torus",    A::CreateTorus),
        act("Disk",     A::CreateDisk),
        act("Capsule",  A::CreateCapsule),
        act("Arrow",    A::CreateArrow),
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

// Ensure the entity's pick BVH is ready to query. Small meshes build inline
// (instant); a huge mesh builds on a background thread (so the click doesn't
// freeze) and this returns false until it finishes -- callers fall back to the
// coarse AABB meanwhile. Harvests a finished background build on a later call.
static bool ensurePickBVH(entt::registry& world, entt::entity e, const PickGeometry& pg) {
    auto& accel = world.get_or_emplace<PickBVH>(e);
    if (accel.built) return true;
    if (accel.job) {
        if (!accel.job->done.load(std::memory_order_acquire)) return false;  // still building
        // Take ownership of the worker's stable geometry + tree, then bind the BVH
        // to our heap-held copies (whose addresses survive component relocation).
        accel.pos = accel.job->pos;
        accel.idx = accel.job->idx;
        accel.bvh = std::move(accel.job->bvh);
        accel.bvh.rebind(*accel.pos, *accel.idx);
        accel.built = true;
        accel.job.reset();
        return true;
    }
    // Own a stable-address copy of the geometry: the BVH references THIS (heap
    // vectors), never the entity's PickGeometry (which EnTT moves on entity erase).
    auto pos = std::make_shared<std::vector<Eigen::Vector3f>>(pg.positions);
    auto idx = std::make_shared<std::vector<uint32_t>>(pg.indices);

    const size_t kInlineTris = 200000;  // build modest BVHs inline; background huge ones
    if (pg.indices.size() <= kInlineTris * 3) {
        accel.pos = std::move(pos);
        accel.idx = std::move(idx);
        accel.bvh.build(*accel.pos, *accel.idx);
        accel.built = true;
        return true;
    }
    auto job  = std::make_shared<PickBvhJob>();
    job->pos  = std::move(pos);
    job->idx  = std::move(idx);
    accel.job = job;
    std::thread([job]() {
        job->bvh.build(*job->pos, *job->idx);
        job->done.store(true, std::memory_order_release);
    }).detach();
    return false;
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
            // Accurate triangle pick via a cached BVH (built off-thread for huge
            // meshes; coarse AABB until it is ready, so the click never freezes).
            if (ensurePickBVH(world, e, *pg)) {
                float tl; int tri;
                if (world.get<PickBVH>(e).bvh.nearestHit(geometry::Ray(lo, ld), tl, tri)) {
                    tLocal = tl; hit = true;
                }
            } else {
                hit = intersectAABB(lo, ld, r.boundsMin, r.boundsMax, tLocal);
            }
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
        if (!ensurePickBVH(world, best, *pg)) return;  // BVH still building -> skip this pick
        float tl; int tri;
        if (world.get<PickBVH>(best).bvh.nearestHit(geometry::Ray(lo, ld), tl, tri))
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

        // Panel size tracks the app font size. Place from the viewport-relative
        // anchor (resize-stable); clamp on-screen AND clear of the scaled
        // selection toolbar on the far left.
        wgt.w = (int)(260.0f * uiScale());
        wgt.h = (int)(132.0f * uiScale());
        float minX = kTbX + kTbBtn + 8.0f;
        float maxX = static_cast<float>(viewportW) - wgt.w;
        float maxY = static_cast<float>(viewportH) - wgt.h;
        wgt.x = static_cast<int>(clampf(wgt.relX * viewportW,
                                        (std::min)(minX, (std::max)(0.0f, maxX)),
                                        (std::max)(0.0f, maxX)));
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

void treeViewInputSystem(entt::registry& world, core::Input& input, uint32_t viewportW,
                         uint32_t viewportH) {
    auto view = world.view<TreeView>();
    for (auto e : view) {
        auto& tv = view.get<TreeView>(e);
        if (!tv.visible) continue;

        // Panel size tracks the app font size; keep the panel clear of the
        // (also scaled) selection toolbar on the far left.
        tv.w = (int)(252.0f * uiScale());
        tv.h = (int)(340.0f * uiScale());
        float minX = kTbX + kTbBtn + 8.0f;
        float maxX = (float)viewportW - tv.w;
        float maxY = (float)viewportH - tv.h;
        tv.x = (int)clampf(tv.relX * viewportW, (std::min)(minX, (std::max)(0.0f, maxX)),
                           (std::max)(0.0f, maxX));
        tv.y = (int)clampf(tv.relY * viewportH, 0.0f, (std::max)(0.0f, maxY));
        // Keep clear of the FPS widget while not being dragged: the stacked
        // top-left defaults were tuned per font size, so a size change can
        // land the panels on top of each other.
        if (!tv.dragging) {
            auto fv = world.view<FpsWidget>();
            for (auto fe : fv) {
                const auto& fw = fv.get<FpsWidget>(fe);
                bool overlap = tv.x < fw.x + fw.w && tv.x + tv.w > fw.x &&
                               tv.y < fw.y + fw.h && tv.y + tv.h > fw.y;
                if (overlap) tv.y = (int)clampf((float)(fw.y + fw.h + 6), 0.0f,
                                                (std::max)(0.0f, maxY));
                break;
            }
        }

        std::vector<TreeRow> rows;
        buildTreeRows(world, tv, rows);
        float maxScroll = treeMaxScroll(tv, rows.size());
        tv.scroll = clampf(tv.scroll, 0.0f, maxScroll);

        float mx = input.mousePosX, my = input.mousePosY;
        bool inside = mx >= tv.x && mx <= tv.x + tv.w && my >= tv.y && my <= tv.y + tv.h;
        tv.hover = -1;
        if (!inside && !tv.dragging) continue;
        if (inside) input.captured = true;

        // Scroll wheel over the panel scrolls the list (and is consumed so the
        // camera doesn't zoom).
        if (inside && input.wheel != 0.0f) {
            tv.scroll = clampf(tv.scroll - input.wheel * kTreeRowH * 2.0f, 0.0f, maxScroll);
            input.wheel = 0.0f;
        }

        float localY = my - tv.y;            // px from panel top
        bool onTitle = inside && localY < kTreeTitleH;
        int  hoveredRow = -1;
        if (inside && localY >= kTreeTitleH) {
            int idx = (int)((localY - kTreeTitleH + tv.scroll) / kTreeRowH);
            if (idx >= 0 && idx < (int)rows.size()) hoveredRow = idx;
        }
        tv.hover = hoveredRow;

        if (input.leftClicked && onTitle) {
            tv.dragging = true;
            tv.dragOffX = mx - tv.x;
            tv.dragOffY = my - tv.y;
        } else if (input.leftClicked && hoveredRow >= 0) {
            const TreeRow& row   = rows[hoveredRow];
            const bool     onEye = row.hasEye && (mx - tv.x) >= tv.w - kTreeEyeZone &&
                               (mx - tv.x) <= tv.w - 4.0f;
            if (onEye) {
                // Eye click: toggle visibility, never selection. A group eye
                // hides all its children (or shows them when all were hidden).
                if (row.isGroup) {
                    const bool wantCloud = (row.group == 1);
                    const bool newVis    = !row.shown;
                    auto rv = world.view<Renderable>();
                    for (auto o : rv) {
                        auto& rr = rv.get<Renderable>(o);
                        if (rr.pointCloud == wantCloud) rr.visible = newVis;
                    }
                } else if (row.entity != entt::null && world.valid(row.entity)) {
                    if (auto* rr = world.try_get<Renderable>(row.entity))
                        rr->visible = !rr->visible;
                }
            } else if (row.isGroup) {
                tv.expanded[row.group] = !tv.expanded[row.group];
            } else if (row.entity != entt::null && world.valid(row.entity)) {
                bool additive = input.ctrl;  // Ctrl-click adds/toggles, else replace
                if (!additive) {
                    auto rv = world.view<Renderable>();
                    for (auto o : rv) rv.get<Renderable>(o).selected = false;
                }
                if (auto* rr = world.try_get<Renderable>(row.entity))
                    rr->selected = additive ? !rr->selected : true;
            }
            input.captured = true;
        }

        if (tv.dragging) {
            if (input.buttonLeft) {
                tv.x = (int)clampf(mx - tv.dragOffX, 0.0f, (std::max)(0.0f, maxX));
                tv.y = (int)clampf(my - tv.dragOffY, 0.0f, (std::max)(0.0f, maxY));
                tv.relX = viewportW > 0 ? (float)tv.x / viewportW : 0.0f;
                tv.relY = viewportH > 0 ? (float)tv.y / viewportH : 0.0f;
                input.captured = true;
            } else {
                tv.dragging = false;
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

    // Panel size tracks the app font size. Position under the gizmo
    // (top-right): gizmo box 150, margin 14, pushed down by the menu bar.
    cc->w = (int)(184.0f * uiScale());
    cc->h = (int)(76.0f * uiScale());
    cc->x = static_cast<int>(viewportW) - cc->w - 14;
    cc->y = kMenuBarHeight + (int)((14 + 150 + 10) * uiScale());
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

    // Panel size tracks the app font size; position top-right, directly under
    // the camera-controls panel.
    const float s = uiScale();
    cs->w = (int)(184.0f * s);
    cs->h = (int)(96.0f * s);
    int baseY = kMenuBarHeight + (int)((14 + 150 + 10 + 76 + 10) * s);
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

void compareLegendInputSystem(entt::registry& world, core::Input& input,
                              uint32_t viewportW, uint32_t viewportH) {
    (void)viewportH;
    auto view = world.view<CompareLegend>();
    for (auto e : view) {
        auto& lg = view.get<CompareLegend>(e);
        if (!lg.visible) continue;

        // Panel size tracks the app font size (View > Font Size...).
        lg.w = (int)(7.5f * uiPx());
        lg.h = (int)(17.0f * uiPx());
        // Top-right column, directly under the cross-section panel -- until the
        // user drags it somewhere (userPlaced).
        if (!lg.userPlaced) {
            int baseY = kMenuBarHeight + 14 + 150 + 10 + 76 + 10 + 96 + 10;
            auto csv = world.view<CrossSection>();
            for (auto ce : csv) { const auto& cs = csv.get<CrossSection>(ce); baseY = cs.y + cs.h + 10; break; }
            lg.x = static_cast<int>(viewportW) - lg.w - 14;
            lg.y = baseY;
            if (lg.x < 0) lg.x = 0;
        }

        float mx = input.mousePosX, my = input.mousePosY;

        // Selection-thumb drag (mode legends): my -> normalized bar position.
        auto selFromMouse = [&]() {
            float barB = legendBarBottom(lg);
            float t    = (barB - (my - lg.y)) / (barB - kLgBarT);
            return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        };
        if (!input.buttonLeft) { lg.dragThumb = -1; lg.dragging = false; }
        if (lg.dragThumb >= 0) {
            float t = selFromMouse();
            if (lg.dragThumb == 0) lg.selMin = std::min(t, lg.selMax - 0.01f);
            else                   lg.selMax = std::max(t, lg.selMin + 0.01f);
            input.captured = true;
            break;
        }
        if (lg.dragging) {  // panel move
            lg.x = (int)(mx - lg.dragDX);
            lg.y = (int)(my - lg.dragDY);
            int maxX = (int)viewportW - lg.w, maxY = (int)viewportH - lg.h;
            lg.x = lg.x < 0 ? 0 : (lg.x > maxX ? (maxX < 0 ? 0 : maxX) : lg.x);
            lg.y = lg.y < 0 ? 0 : (lg.y > maxY ? (maxY < 0 ? 0 : maxY) : lg.y);
            input.captured = true;
            break;
        }

        bool inPanel = mx >= lg.x && mx <= lg.x + lg.w && my >= lg.y && my <= lg.y + lg.h;
        if (!inPanel) continue;
        if (input.leftClicked) {
            CRect cb = legendCloseRect(lg);
            if (mx >= cb.x && mx <= cb.x + cb.w && my >= cb.y && my <= cb.y + cb.h) {
                lg.visible = false;
                input.captured = true;
                break;
            }
            // Thumb grab (mode legends only): a band around each handle.
            if (lg.fromMode && mx >= lg.x + kLgBarL - 20 && mx <= lg.x + kLgBarR + 20) {
                float yMin = lg.y + legendThumbY(lg, lg.selMin);
                float yMax = lg.y + legendThumbY(lg, lg.selMax);
                if (std::abs(my - yMin) <= 14.0f &&
                    std::abs(my - yMin) <= std::abs(my - yMax))
                    lg.dragThumb = 0;
                else if (std::abs(my - yMax) <= 14.0f)
                    lg.dragThumb = 1;
                if (lg.dragThumb >= 0) {
                    input.captured = true;
                    break;
                }
            }
            // Anywhere else on the panel: start moving it.
            lg.dragging   = true;
            lg.userPlaced = true;
            lg.dragDX     = mx - lg.x;
            lg.dragDY     = my - lg.y;
        }
        input.captured = true;
        break;
    }
}

void fontSizeDialogInputSystem(entt::registry& world, core::Input& input,
                               uint32_t viewportW, uint32_t viewportH) {
    FontSizeDialog* d = nullptr;
    auto view = world.view<FontSizeDialog>();
    for (auto e : view) { d = &view.get<FontSizeDialog>(e); break; }
    if (!d || !d->visible) {
        if (d) { d->dragSlider = false; d->dragging = false; d->editing = false; }
        return;
    }

    // Panel size tracks the very value it edits.
    const float s = uiScale();
    d->w = (int)(280.0f * s);
    d->h = (int)(96.0f * s);
    if (!d->placed) {
        d->x = (static_cast<int>(viewportW) - d->w) / 2;
        d->y = (static_cast<int>(viewportH) - d->h) / 3;
        d->placed = true;
    }

    float mx = input.mousePosX, my = input.mousePosY;
    auto  hit = [&](const CRect& r) {
        return mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h;
    };
    FontDlgRects R = fontDlgRects(*d);
    bool inPanel = mx >= d->x && mx <= d->x + d->w && my >= d->y && my <= d->y + d->h;

    // Number-box editing: digits append, backspace deletes, Enter applies.
    if (d->editing) {
        for (const char* c = input.text; *c; ++c)
            if (*c >= '0' && *c <= '9' && d->editBuf.size() < 3) d->editBuf += *c;
        if (input.backspace && !d->editBuf.empty()) d->editBuf.pop_back();
        if (input.enter) {
            if (!d->editBuf.empty()) core::setUiFontPx((float)std::atoi(d->editBuf.c_str()));
            d->editing = false;
        }
        if (input.leftClicked && !hit(R.box)) {  // click-away commits too
            if (!d->editBuf.empty()) core::setUiFontPx((float)std::atoi(d->editBuf.c_str()));
            d->editing = false;
        }
    }

    if (input.leftClicked && hit(R.close)) {
        d->visible = false; d->dragging = false; d->editing = false;
        input.captured = true;
        return;
    }
    if (input.leftClicked && hit(R.box)) {
        d->editing = true;
        d->editBuf.clear();
        input.captured = true;
        return;
    }

    // Slider.
    if (input.leftClicked) {
        CRect band = {R.track.x - 6.0f, R.track.y - 11.0f * s, R.track.w + 12.0f,
                      R.track.h + 22.0f * s};
        if (hit(band)) d->dragSlider = true;
    }
    if (!input.buttonLeft) d->dragSlider = false;
    if (d->dragSlider) {
        float t = R.track.w > 0 ? (mx - R.track.x) / R.track.w : 0.0f;
        core::setUiFontPx(kFontPxMin + clampf(t, 0.0f, 1.0f) * (kFontPxMax - kFontPxMin));
        input.captured = true;
        return;
    }

    // Title-bar drag.
    CRect titleBar = {(float)d->x, (float)d->y, (float)d->w - 28.0f * s, 28.0f * s};
    if (input.leftClicked && hit(titleBar)) {
        d->dragging = true;
        d->dragDX = mx - d->x;
        d->dragDY = my - d->y;
    }
    if (!input.buttonLeft) d->dragging = false;
    if (d->dragging) {
        d->x = (int)(mx - d->dragDX);
        d->y = (int)(my - d->dragDY);
        input.captured = true;
        return;
    }

    if (inPanel) input.captured = true;
}

void poissonDialogInputSystem(entt::registry& world, core::Input& input,
                              uint32_t viewportW, uint32_t viewportH) {
    PoissonDialog* d = nullptr;
    auto view = world.view<PoissonDialog>();
    for (auto e : view) { d = &view.get<PoissonDialog>(e); break; }
    if (!d || !d->visible) { if (d) { d->dragSlider = -1; d->dragging = false; } return; }

    // Center once on first show; thereafter the title bar moves it.
    if (!d->placed) {
        d->x = (static_cast<int>(viewportW) - d->w) / 2;
        d->y = (static_cast<int>(viewportH) - d->h) / 2;
        d->placed = true;
    }

    float mx = input.mousePosX, my = input.mousePosY;
    auto  hit = [&](const CRect& r) {
        return mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h;
    };
    PoissonRects R = poissonRects(*d);
    bool inPanel = mx >= d->x && mx <= d->x + d->w && my >= d->y && my <= d->y + d->h;

    // Close button.
    if (input.leftClicked && hit(R.close)) {
        d->visible = false; d->dragging = false; input.captured = true; return;
    }

    // Title-bar drag (top 28px, excluding the close button).
    CRect titleBar = {(float)d->x, (float)d->y, (float)d->w - 28.0f, 28.0f};
    if (input.leftClicked && hit(titleBar)) {
        d->dragging = true;
        d->dragDX = mx - d->x;
        d->dragDY = my - d->y;
    }
    if (!input.buttonLeft) d->dragging = false;
    if (d->dragging) {
        d->x = static_cast<int>(mx - d->dragDX);
        d->y = static_cast<int>(my - d->dragDY);
        // Keep the panel on screen.
        int maxX = static_cast<int>(viewportW) - d->w, maxY = static_cast<int>(viewportH) - d->h;
        d->x = d->x < 0 ? 0 : (d->x > maxX ? (maxX < 0 ? 0 : maxX) : d->x);
        d->y = d->y < 0 ? 0 : (d->y > maxY ? (maxY < 0 ? 0 : maxY) : d->y);
        input.captured = true;
        return;
    }

    // Reconstruct: raise the request edge; the app reads the params off the dialog,
    // runs Poisson on a worker, and spawns the result as a new static mesh entity.
    if (input.leftClicked && hit(R.run)) {
        d->requestRun = true;
        input.captured = true;
        return;
    }

    // Slider grab: a click anywhere in a band around a groove starts the drag.
    if (input.leftClicked) {
        for (int i = 0; i < 4; ++i) {
            const CRect& t = R.track[i];
            CRect band = {t.x - 6.0f, t.y - 11.0f, t.w + 12.0f, t.h + 22.0f};
            if (hit(band)) { d->dragSlider = i; break; }
        }
    }
    if (!input.buttonLeft) d->dragSlider = -1;
    if (d->dragSlider >= 0 && d->dragSlider < 4) {
        const PSliderSpec& s = kPSliders[d->dragSlider];
        const CRect& t = R.track[d->dragSlider];
        float u = t.w > 0.0f ? (mx - t.x) / t.w : 0.0f;
        u = clampf(u, 0.0f, 1.0f);
        poissonSetSliderValue(*d, d->dragSlider, s.vmin + u * (s.vmax - s.vmin));
        input.captured = true;
    }

    if (inPanel) input.captured = true;
}

void modeParamsDialogInputSystem(entt::registry& world, core::Input& input,
                                 uint32_t viewportW, uint32_t viewportH) {
    ModeParamsDialog* d = nullptr;
    auto view = world.view<ModeParamsDialog>();
    for (auto e : view) { d = &view.get<ModeParamsDialog>(e); break; }
    if (!d || !d->visible) { if (d) { d->dragSlider = -1; d->dragging = false; } return; }

    const int nParams = modes::modeParamCount(d->modeIndex);
    if (nParams <= 0) { d->visible = false; return; }

    // Place once beside the center (so it doesn't cover the Poisson dialog).
    if (!d->placed) {
        d->x = (static_cast<int>(viewportW) - d->w) / 2 + d->w;
        d->y = (static_cast<int>(viewportH) - d->h) / 2;
        d->placed = true;
    }

    // The legend strip appears once the mode's recolor published a scale
    // (async), and everything tracks the app font size: recompute per frame.
    CompareLegend* lg = d->pipeNodeId < 0 ? activeModeLegend(world) : nullptr;
    const float s = uiScale();
    d->w = (int)(280.0f * s);
    d->h = (int)((42.0f + nParams * 48.0f + 46.0f +
                  ((modes::modeCanFit(d->modeIndex) ||
                    modes::modeAppliesMesh(d->modeIndex)) &&
                           d->pipeNodeId < 0
                       ? 36.0f
                       : 0.0f)) *
                 s) +
           (lg ? (int)kModeDlgLegendH : 0);

    float mx = input.mousePosX, my = input.mousePosY;
    auto  hit = [&](const CRect& r) {
        return mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h;
    };
    ModeDlgRects R = modeDlgRects(*d, nParams, lg != nullptr);
    bool inPanel = mx >= d->x && mx <= d->x + d->w && my >= d->y && my <= d->y + d->h;

    // Legend range thumbs (horizontal): drag maps mouse X to [0,1].
    if (lg) {
        if (!input.buttonLeft) lg->dragThumb = -1;
        auto selFromX = [&]() {
            float t = R.lgBar.w > 0 ? (mx - R.lgBar.x) / R.lgBar.w : 0.0f;
            return clampf(t, 0.0f, 1.0f);
        };
        if (lg->dragThumb >= 0) {
            float t = selFromX();
            if (lg->dragThumb == 0) lg->selMin = std::min(t, lg->selMax - 0.01f);
            else                    lg->selMax = std::max(t, lg->selMin + 0.01f);
            input.captured = true;
            return;
        }
        CRect band = {R.lgBar.x - 8.0f, R.lgBar.y - 10.0f, R.lgBar.w + 16.0f,
                      R.lgBar.h + 20.0f};
        if (input.leftClicked && hit(band)) {
            float t = selFromX();
            lg->dragThumb =
                std::abs(t - lg->selMin) <= std::abs(t - lg->selMax) ? 0 : 1;
            input.captured = true;
            return;
        }
        if (input.leftClicked && hit(R.extract)) {
            d->requestExtract = true;
            input.captured    = true;
            return;
        }
    }

    if (input.leftClicked && hit(R.close)) {
        d->visible = false; d->dragging = false; input.captured = true; return;
    }

    CRect titleBar = {(float)d->x, (float)d->y, (float)d->w - 28.0f * s, 28.0f * s};
    if (input.leftClicked && hit(titleBar)) {
        d->dragging = true;
        d->dragDX = mx - d->x;
        d->dragDY = my - d->y;
    }
    if (!input.buttonLeft) d->dragging = false;
    if (d->dragging) {
        d->x = static_cast<int>(mx - d->dragDX);
        d->y = static_cast<int>(my - d->dragDY);
        int maxX = static_cast<int>(viewportW) - d->w, maxY = static_cast<int>(viewportH) - d->h;
        d->x = d->x < 0 ? 0 : (d->x > maxX ? (maxX < 0 ? 0 : maxX) : d->x);
        d->y = d->y < 0 ? 0 : (d->y > maxY ? (maxY < 0 ? 0 : maxY) : d->y);
        input.captured = true;
        return;
    }

    if (input.leftClicked && hit(R.apply)) {
        d->requestApply = true;
        input.captured = true;
        return;
    }
    if (input.leftClicked &&
        (modes::modeCanFit(d->modeIndex) || modes::modeAppliesMesh(d->modeIndex)) &&
        d->pipeNodeId < 0 && hit(R.fit)) {
        d->requestFit  = true;
        input.captured = true;
        return;
    }

    // Header checkboxes: toggle the section's enable flag.
    if (input.leftClicked) {
        for (int i = 0; i < nParams && i < 12; ++i) {
            if (!modes::modeParamIsHeader(modes::modeParam(d->modeIndex, i))) continue;
            CRect cb = modeHeaderCheckRect(R.track[i], s);
            CRect band = {cb.x - 4.0f, cb.y - 4.0f, cb.w + 8.0f, cb.h + 8.0f};
            if (hit(band)) {
                d->values[i] = d->values[i] != 0.0f ? 0.0f : 1.0f;
                input.captured = true;
                return;
            }
        }
    }

    // -/+ nudge buttons: one step per click. Header rows have no controls.
    if (input.leftClicked) {
        for (int i = 0; i < nParams && i < 12; ++i) {
            modes::ModeParam s = modes::modeParam(d->modeIndex, i);
            if (modes::modeParamIsHeader(s)) continue;
            float st = modeParamStep(s);
            if (hit(R.minus[i])) {
                d->values[i] = modeParamSnap(s, d->values[i] - st);
                input.captured = true;
                return;
            }
            if (hit(R.plus[i])) {
                d->values[i] = modeParamSnap(s, d->values[i] + st);
                input.captured = true;
                return;
            }
        }
    }

    if (input.leftClicked) {
        for (int i = 0; i < nParams && i < 12; ++i) {
            if (modes::modeParamIsHeader(modes::modeParam(d->modeIndex, i))) continue;
            const CRect& t = R.track[i];
            CRect band = {t.x - 6.0f, t.y - 11.0f, t.w + 12.0f, t.h + 22.0f};
            if (hit(band)) { d->dragSlider = i; break; }
        }
    }
    if (!input.buttonLeft) d->dragSlider = -1;
    if (d->dragSlider >= 0 && d->dragSlider < nParams) {
        modes::ModeParam s = modes::modeParam(d->modeIndex, d->dragSlider);
        const CRect& t = R.track[d->dragSlider];
        float u = t.w > 0.0f ? (mx - t.x) / t.w : 0.0f;
        u = clampf(u, 0.0f, 1.0f);
        d->values[d->dragSlider] = modeParamSnap(s, s.minV + u * (s.maxV - s.minV));
        input.captured = true;
    }

    if (inPanel) input.captured = true;
}

void pipelineDialogInputSystem(entt::registry& world, core::Input& input,
                               uint32_t viewportW, uint32_t viewportH) {
    PipelineDialog* d = nullptr;
    auto view = world.view<PipelineDialog>();
    for (auto e : view) { d = &view.get<PipelineDialog>(e); break; }
    if (!d || !d->visible) {
        if (d) {
            d->dragging = false; d->resizing = false; d->panning = false;
            d->dragNode = -1; d->hoverNode = -1;
        }
        return;
    }

    // First show: dock near the right edge + seed a Source -> Output skeleton.
    if (!d->placed) {
        d->x = (int)viewportW - d->w - 40;
        d->y = 80;
        if (d->x < 0) d->x = 0;
        d->placed = true;
    }
    if (d->nodes.empty()) {
        PipeNode s;
        s.id = d->nextId++; s.role = PipeNodeRole::Source; s.x = 16.0f; s.y = 60.0f;
        PipeNode o;
        o.id = d->nextId++; o.role = PipeNodeRole::Output;
        o.x = d->w - kPipeNodeW - 40.0f; o.y = 60.0f;
        d->nodes.push_back(std::move(s));
        d->nodes.push_back(std::move(o));
    }

    // The mode-params dialog, when bound to one of our nodes, writes its
    // slider values back into that node every frame (and detaches if the node
    // is gone).
    ModeParamsDialog* pd = nullptr;
    {
        auto pv = world.view<ModeParamsDialog>();
        for (auto e : pv) { pd = &pv.get<ModeParamsDialog>(e); break; }
    }
    if (pd && pd->pipeNodeId >= 0) {
        PipeNode* n = pipeFindNodeMut(*d, pd->pipeNodeId);
        if (!n) {
            pd->pipeNodeId = -1;
            pd->visible    = false;
        } else {
            int cnt = std::min((int)n->params.size(), (int)pd->values.size());
            for (int i = 0; i < cnt; ++i) n->params[i] = pd->values[i];
        }
    }

    float mx = input.mousePosX, my = input.mousePosY;
    auto  hit = [&](const CRect& r) {
        return mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h;
    };
    bool inPanel = mx >= d->x && mx <= d->x + d->w && my >= d->y && my <= d->y + d->h;

    // The pending-wire endpoint follows the cursor (drawn by the builder).
    if (d->linkFrom >= 0 && !d->panning) { d->panSX = mx; d->panSY = my; }

    if (input.leftClicked && hit(pipeCloseRect(*d))) {
        d->visible = false; d->dragging = false; d->resizing = false; d->linkFrom = -1;
        input.captured = true;
        return;
    }

    // Resize grip (bottom-right).
    if (input.leftClicked && hit(pipeGripRect(*d))) d->resizing = true;
    if (!input.buttonLeft) d->resizing = false;
    if (d->resizing) {
        d->w = (int)mx - d->x;
        d->h = (int)my - d->y;
        if (d->w < d->minW) d->w = d->minW;
        if (d->h < d->minH) d->h = d->minH;
        if (d->w > (int)viewportW) d->w = (int)viewportW;
        if (d->h > (int)viewportH) d->h = (int)viewportH;
        input.captured = true;
        return;
    }

    // Title-bar drag (the Save/Load/Run buttons living in the bar win first).
    CRect titleBar = {(float)d->x, (float)d->y, (float)d->w - 38.0f, kPipeTitleH};
    if (input.leftClicked && hit(titleBar) && !hit(pipeRunRect(*d)) &&
        !hit(pipeSaveRect(*d)) && !hit(pipeLoadRect(*d))) {
        d->dragging = true;
        d->dragDX = mx - d->x;
        d->dragDY = my - d->y;
    }
    if (!input.buttonLeft) d->dragging = false;
    if (d->dragging) {
        d->x = (int)(mx - d->dragDX);
        d->y = (int)(my - d->dragDY);
        int maxX = (int)viewportW - d->w, maxY = (int)viewportH - d->h;
        d->x = d->x < 0 ? 0 : (d->x > maxX ? (maxX < 0 ? 0 : maxX) : d->x);
        d->y = d->y < 0 ? 0 : (d->y > maxY ? (maxY < 0 ? 0 : maxY) : d->y);
        input.captured = true;
        return;
    }

    // Palette buttons (add node at the canvas center) + Save/Load + Run.
    if (input.leftClicked) {
        const auto& stages = pipeStageList();
        for (int s = 0; s < (int)stages.size(); ++s) {
            if (!hit(pipePaletteRect(*d, s))) continue;
            PipeNode n;
            n.id   = d->nextId++;
            n.role = stages[s].role;
            n.mode = stages[s].mode;
            n.fit  = stages[s].fit;
            n.x    = d->w * 0.5f - kPipeNodeW * 0.5f - d->panX;
            n.y    = d->h * 0.4f - d->panY;
            pipeInitNodeParams(n);
            d->nodes.push_back(std::move(n));
            input.captured = true;
            return;
        }
        if (hit(pipeSaveRect(*d))) {
            pipeSaveGraph(*d);
            input.captured = true;
            return;
        }
        if (hit(pipeLoadRect(*d))) {
            pipeLoadGraph(*d);
            if (pd) { pd->pipeNodeId = -1; pd->visible = false; }  // stale binding
            input.captured = true;
            return;
        }
        if (hit(pipeRunRect(*d))) {
            d->requestRun  = true;
            input.captured = true;
            return;
        }
    }

    // Nodes: close button, pins (click output pin -> click input pin wires
    // them; clicking a wired input pin unplugs it), then body drag.
    d->hoverNode = -1;
    if (input.leftClicked) {
        for (size_t i = d->nodes.size(); i-- > 0;) {  // topmost (last drawn) first
            PipeNode& n = d->nodes[i];
            CRect nr = pipeNodeRect(*d, n);
            // Delete node.
            if (hit(pipeNodeCloseRect(nr))) {
                int id = n.id;
                d->links.erase(std::remove_if(d->links.begin(), d->links.end(),
                                              [id](const PipeLink& l) {
                                                  return l.from == id || l.to == id;
                                              }),
                               d->links.end());
                if (d->linkFrom == id) d->linkFrom = -1;
                if (pd && pd->pipeNodeId == id) { pd->pipeNodeId = -1; pd->visible = false; }
                d->nodes.erase(d->nodes.begin() + i);
                input.captured = true;
                return;
            }
            float ix, iy, ox, oy;
            pipePinCenters(nr, ix, iy, ox, oy);
            auto nearPin = [&](float px, float py) {
                return std::abs(mx - px) <= kPipePinR + 3.0f &&
                       std::abs(my - py) <= kPipePinR + 3.0f;
            };
            // Output pin: start a pending wire.
            if (n.role != PipeNodeRole::Output && nearPin(ox, oy)) {
                d->linkFrom    = n.id;
                input.captured = true;
                return;
            }
            // Input pins (main, and the amber condition pin on candidate-aware
            // stages): finish the pending wire, or unplug an existing one.
            int pinHit = -1;
            if (n.role != PipeNodeRole::Source && nearPin(ix, iy)) pinHit = 0;
            if (pinHit < 0 && pipeNodeHasCondPin(n)) {
                float cx2, cy2;
                pipeCondPinCenter(nr, cx2, cy2);
                if (nearPin(cx2, cy2)) pinHit = 1;
            }
            if (pinHit >= 0) {
                int to = n.id;  // one wire per input pin: replace/unplug
                d->links.erase(std::remove_if(d->links.begin(), d->links.end(),
                                              [to, pinHit](const PipeLink& l) {
                                                  return l.to == to && l.toPin == pinHit;
                                              }),
                               d->links.end());
                if (d->linkFrom >= 0 && d->linkFrom != n.id) {
                    d->links.push_back({d->linkFrom, n.id, pinHit});
                    d->linkFrom = -1;
                }
                input.captured = true;
                return;
            }
            // Body: grab for move (also cancels a pending wire) + bind the
            // params dialog to this node when its mode has tunables.
            if (hit(nr)) {
                d->linkFrom = -1;
                d->dragNode = n.id;
                d->nodeDX   = mx - nr.x;
                d->nodeDY   = my - nr.y;
                int cnt = pipeNodeParamCount(n);
                if (pd && cnt > 0) {
                    pd->modeIndex  = modes::modeIndexByName(n.mode.c_str());
                    pd->pipeNodeId = n.id;
                    pd->values.assign(cnt, 0.0f);
                    for (int pi = 0; pi < cnt && pi < (int)n.params.size(); ++pi)
                        pd->values[pi] = n.params[pi];
                    // Height/width recomputed per frame by its input system.
                    pd->h       = (int)((42.0f + cnt * 48.0f + 46.0f) * uiScale());
                    pd->visible = true;
                } else if (pd && pd->pipeNodeId >= 0) {
                    pd->pipeNodeId = -1;  // clicked a param-less node: unbind
                    pd->visible    = false;
                }
                input.captured = true;
                return;
            }
        }
        // Empty canvas click: cancel pending wire / start panning.
        if (inPanel && my > d->y + kPipeTitleH && my < d->y + d->h - pipePaletteH(*d)) {
            if (d->linkFrom >= 0) {
                d->linkFrom = -1;
            } else {
                d->panning = true;
                d->panSX = mx; d->panSY = my;
                d->panOX = d->panX; d->panOY = d->panY;
            }
            input.captured = true;
        }
    }

    if (!input.buttonLeft) { d->dragNode = -1; d->panning = false; }
    if (d->dragNode >= 0) {
        for (auto& n : d->nodes)
            if (n.id == d->dragNode) {
                n.x = mx - d->nodeDX - d->x - 10.0f - d->panX;
                n.y = my - d->nodeDY - d->y - kPipeTitleH - 4.0f - d->panY;
                break;
            }
        input.captured = true;
        return;
    }
    if (d->panning) {
        d->panX = d->panOX + (mx - d->panSX);
        d->panY = d->panOY + (my - d->panSY);
        input.captured = true;
        return;
    }

    // Hover feedback.
    for (size_t i = d->nodes.size(); i-- > 0;) {
        if (hit(pipeNodeRect(*d, d->nodes[i]))) { d->hoverNode = d->nodes[i].id; break; }
    }

    // Modeless: only the panel area eats input; everything else stays live.
    if (inPanel) input.captured = true;
}

void confirmDialogInputSystem(entt::registry& world, core::Input& input,
                              uint32_t viewportW, uint32_t viewportH) {
    ConfirmDialog* d = nullptr;
    auto view = world.view<ConfirmDialog>();
    for (auto e : view) { d = &view.get<ConfirmDialog>(e); break; }
    if (!d || !d->visible) return;

    // Size tracks the app font, widened to fit the message lines (line2 is
    // often a full file path).
    const float s = uiScale();
    float wantW = 460.0f * s;
    if (d->font) {
        float tw = d->font->textWidth(d->line1.c_str(), uiPx());
        if (!d->line2.empty())
            tw = (std::max)(tw, d->font->textWidth(d->line2.c_str(), uiPx()));
        wantW = (std::max)(wantW, tw + 40.0f * s);
    }
    d->w = (int)(std::min)(wantW, (float)viewportW - 20.0f);
    d->h = (int)(150.0f * s);

    if (!d->placed) {
        d->x = ((int)viewportW - d->w) / 2;
        d->y = ((int)viewportH - d->h) / 2;
        d->placed = true;
    }

    float mx = input.mousePosX, my = input.mousePosY;
    bool inPanel = mx >= d->x && mx <= d->x + d->w && my >= d->y && my <= d->y + d->h;
    if (inPanel) input.captured = true;  // soft-modal: eat clicks over the panel

    auto hit = [&](const CRect& r) {
        return mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h;
    };
    ConfirmRects R = confirmRects(*d);
    if (input.leftClicked) {
        if (hit(R.yes)) { d->yes = true;  d->answered = true; d->visible = false; input.captured = true; }
        else if (hit(R.no)) { d->yes = false; d->answered = true; d->visible = false; input.captured = true; }
    }
}

// Build the wireframe vertices for a spatial structure over `P` (and `indices`
// for the triangle BVH). Self-contained (no GPU, no registry) so it runs on a
// worker thread; the caller uploads `outVerts` to the GPU. `progress` reports
// completion in [0,1]. Extracted from renderSystem's SpatialViz pass.
static void buildSpatialVizVerts(const std::vector<Eigen::Vector3f>& P,
                                 const std::vector<uint32_t>& indices, int vizKind,
                                 std::vector<render::Vertex>& outVerts, int& outBoxCount,
                                 int& outDrawn, int& outTotal,
                                 const std::function<void(float)>& progress) {
    auto levelColor = [](int d) {
        Eigen::Vector4f c = color::FromHSV(color::fract(d * 0.16f), 0.85f, 1.0f);
        return Eigen::Vector3f(c.x(), c.y(), c.z());
    };
    const int kDepthTree = 18;  // BVH / Octree
    const int kDepthKD   = 12;  // KD-tree
    outVerts.clear();
    outBoxCount = 0; outDrawn = 0; outTotal = (int)P.size();
    if (P.empty()) return;
    if (progress) progress(0.05f);

    std::vector<geometry::AABB> boxes; std::vector<int> levels;
    std::vector<Eigen::Vector3f> sphC; std::vector<float> sphR; std::vector<int> sphL;
    geometry::AABB rb = geometry::robustBounds(P, 0.001f);
    float md = (rb.max - rb.min).norm();
    geometry::AABB full; for (const auto& p : P) full.expand(p);

    if (vizKind == 1 && !indices.empty()) {
        geometry::BVH b; b.build(P, indices); b.nodeBoxesDepth(boxes, levels, kDepthTree);
    } else if (vizKind == 2) {
        geometry::Octree o; o.build(P, 64); o.nodeBoxesDepth(boxes, levels, kDepthTree);
    } else if (vizKind == 3) {
        geometry::KDTree k; k.build(P); k.cellBoxesDepth(full, boxes, levels, kDepthKD);
    } else if (vizKind == 4) {
        geometry::UniformGrid g; g.build(P, (md > 1e-6f ? md : 1.0f) / 40.0f);
        g.occupiedCellBoxes(boxes); levels.assign(boxes.size(), 0);
    } else if (vizKind == 5) {
        geometry::LooseOctree lo; lo.build(P, 64); lo.nodeBoxesDepth(boxes, levels, kDepthTree);
    } else if (vizKind == 6) {
        geometry::BSP bsp; bsp.build(P, 64); bsp.nodeBoxesDepth(boxes, levels, kDepthTree);
    } else if (vizKind == 7) {
        geometry::RTree rt; rt.build(P, 16); rt.nodeBoxesDepth(boxes, levels, kDepthTree);
    } else if (vizKind == 8) {
        geometry::BallTree bt; bt.build(P, 32); bt.nodeSpheresDepth(sphC, sphR, sphL, 7);
    }
    if (progress) progress(0.5f);
    if (boxes.empty() && sphC.empty()) return;

    float thick   = (md > 1e-6f ? md : 1.0f) * 0.0005f;
    float maxDiag = (md > 1e-6f ? md : 1.0f) * 1.5f;  // cull only bigger-than-model cells
    const size_t kCap = 60000;                         // box budget
    debug::DebugDraw tmp;
    std::vector<size_t> keep;
    for (size_t i = 0; i < boxes.size(); ++i)
        if ((boxes[i].max - boxes[i].min).norm() <= maxDiag) keep.push_back(i);
    size_t stride = keep.size() > kCap ? (keep.size() + kCap - 1) / kCap : 1;
    int drawn = 0;
    for (size_t j = 0; j < keep.size(); j += stride) {
        tmp.addWireBox(boxes[keep[j]].min, boxes[keep[j]].max, levelColor(levels[keep[j]]), thick);
        ++drawn;
        if (progress && (j % 4096 == 0) && !keep.empty())
            progress(0.5f + 0.5f * (float)j / (float)keep.size());
    }
    std::vector<size_t> ks;
    for (size_t i = 0; i < sphR.size(); ++i)
        if (sphR[i] <= maxDiag) ks.push_back(i);
    size_t ss = ks.size() > kCap ? (ks.size() + kCap - 1) / kCap : 1;
    for (size_t j = 0; j < ks.size(); j += ss) {
        tmp.addSphere(sphC[ks[j]], sphR[ks[j]], levelColor(sphL[ks[j]]), 8);
        ++drawn;
    }
    outBoxCount = (int)(boxes.size() + sphC.size());
    outDrawn    = drawn;
    outVerts    = tmp.vertices();
    if (progress) progress(1.0f);
}

bool spatialVizProgress(entt::registry& world, float& outProgress, std::string& outName) {
    auto v = world.view<SpatialVizCache>();
    for (auto e : v) {
        const auto& c = v.get<SpatialVizCache>(e);
        if (c.job && !c.job->done.load(std::memory_order_acquire)) {
            outProgress = c.job->progress.load(std::memory_order_relaxed);
            outName     = "Spatial";
            return true;
        }
    }
    return false;
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

    // Master lighting switch (` / Render menu); per drawable a point cloud is
    // always unlit (flat vertex color), a triangle mesh lit while enabled.
    const auto* ls  = world.ctx().find<LightingState>();
    const bool  lit = !ls || ls->enabled;

    auto drawables = world.view<Transform, Renderable>();
    for (auto entity : drawables) {
        const auto& t = drawables.get<Transform>(entity);
        const auto& r = drawables.get<Renderable>(entity);
        // None draw mode == fully invisible: no visible pass and no silhouette.
        if (!r.visible || r.mesh == render::kInvalidMesh ||
            r.drawMode == core::DrawMode::None)
            continue;

        Eigen::Matrix4f model = Mworld * t.matrix();

        // 1) Visible pass, in this mesh's own drawing + coloring mode.
        renderer.setDrawMode(static_cast<uint32_t>(r.drawMode));
        renderer.setColorMode(r.colorMode);
        renderer.setLighting(lit && !r.pointCloud);
        render::DrawItem item;
        item.mesh = r.mesh;
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        renderer.setColorMode(0);   // selection silhouette/box stays its own color
        renderer.setLighting(false);  // ...and unshaded (flat orange outline)

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
    renderer.setDrawMode(kSolid);   // scene only: rest stays solid
    renderer.setColorMode(0);       // overlays/grid/debug use original color

    // --- Element selection highlights (vertex/edge/face) ---------------
    // Baked ONCE into a static, per-entity GPU mesh (local space) when the
    // selection content changes, then drawn each frame with model = Mworld *
    // Transform -- exactly like the spatial-viz overlay below. The old path
    // rebuilt a camera-facing quad per selected vertex into the per-frame
    // debug-draw mesh and re-uploaded the whole thing every frame, so a large
    // (e.g. box) selection of tens of thousands of points tanked the FPS.
    // Markers are now view-independent little cubes (visible from any angle),
    // sized from the mesh bounds instead of in screen pixels.
    {
        auto esv = world.view<Transform, ElementSelection, PickGeometry, Renderable>();
        const Eigen::Vector3f hl(0.15f, 1.0f, 0.25f);  // bright lime, pops on pink/white

        for (auto e : esv) {
            const auto& es = esv.get<ElementSelection>(e);
            const auto& pg = esv.get<PickGeometry>(e);

            // Cheap content signature: rebuild only when the selection changes.
            uint64_t sig = 1469598103934665603ull;  // FNV-1a over the index sets
            auto mix = [&](uint64_t x) { sig ^= x; sig *= 1099511628211ull; };
            mix(es.vertices.size()); mix(es.faces.size()); mix(es.edges.size());
            for (uint32_t v : es.vertices)        mix(v);
            for (uint32_t f : es.faces)           mix(f);
            for (const auto& ed : es.edges) { mix(ed.a); mix(ed.b); }

            auto& cache = world.get_or_emplace<ElementSelCache>(e);
            if (!cache.built || cache.sig != sig) {
                if (cache.mesh != render::kInvalidMesh) renderer.destroyMesh(cache.mesh);
                if (cache.vbo  != render::kInvalidBuffer) renderer.destroyBuffer(cache.vbo);
                cache.mesh = render::kInvalidMesh; cache.vbo = render::kInvalidBuffer;
                cache.vertexCount = 0; cache.sig = sig; cache.built = true;

                const auto& r = esv.get<Renderable>(e);
                float diag = (r.boundsMax - r.boundsMin).norm();
                if (diag < 1e-6f) diag = 1.0f;
                const float half  = diag * 0.004f;   // vertex-cube half size
                const float thick = diag * 0.0015f;  // edge-tube thickness
                const Eigen::Vector3f h(half, half, half);

                debug::DebugDraw tmp;  // local accumulator (not the per-frame one)
                // Cube markers dominate the vertex count; stride-sample to a budget
                // so a pathological selection can't blow the static mesh up.
                const size_t kCap = 80000;
                size_t stride = es.vertices.size() > kCap
                                    ? (es.vertices.size() + kCap - 1) / kCap : 1;
                for (size_t j = 0; j < es.vertices.size(); j += stride) {
                    uint32_t vi = es.vertices[j];
                    if (vi < pg.positions.size())
                        tmp.addBox(pg.positions[vi] - h, pg.positions[vi] + h, hl);
                }
                for (const auto& ed : es.edges)
                    if (ed.a < pg.positions.size() && ed.b < pg.positions.size())
                        tmp.addLine(pg.positions[ed.a], pg.positions[ed.b], hl, thick);
                for (uint32_t fi : es.faces) {
                    size_t i = static_cast<size_t>(fi) * 3;
                    if (i + 2 < pg.indices.size())
                        tmp.addTriangle(pg.positions[pg.indices[i]],
                                        pg.positions[pg.indices[i + 1]],
                                        pg.positions[pg.indices[i + 2]], hl);
                }

                const auto& verts = tmp.vertices();
                if (!verts.empty()) {
                    render::BufferDesc bd;
                    bd.type  = render::BufferType::Vertex;
                    bd.usage = render::BufferUsage::Static;
                    bd.data  = verts.data();
                    bd.size  = verts.size() * sizeof(render::Vertex);
                    cache.vbo = renderer.createBuffer(bd);
                    render::MeshDesc md;
                    md.vertexBuffer = cache.vbo;
                    md.layout       = render::Vertex::layout();
                    md.vertexCount  = static_cast<uint32_t>(verts.size());
                    md.topology     = render::PrimitiveTopology::Triangles;
                    cache.mesh = renderer.createMesh(md);
                    cache.vertexCount = static_cast<uint32_t>(verts.size());
                }
            }

            if (cache.mesh != render::kInvalidMesh) {
                renderer.setDrawMode(kSolid);
                const auto& t = esv.get<Transform>(e);
                Eigen::Matrix4f model = Mworld * t.matrix();
                render::DrawItem item;
                item.mesh = cache.mesh;
                std::memcpy(item.model, model.data(), sizeof(item.model));
                renderer.submit(item);
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
            auto sv = world.view<Transform, Renderable, PickGeometry>();
            // No selection + exactly one loaded mesh => visualize that one.
            entt::entity fallback = entt::null;
            {
                entt::entity only = entt::null;
                size_t count = 0;
                bool anySelected = false;
                for (auto e : sv) {
                    if (sv.get<PickGeometry>(e).positions.empty()) continue;
                    ++count;
                    only = e;
                    if (sv.get<Renderable>(e).selected) { anySelected = true; break; }
                }
                if (!anySelected && count == 1) fallback = only;
            }
            for (auto e : sv) {
                const auto& r = sv.get<Renderable>(e);
                if (!r.selected && e != fallback) continue;
                const auto& pg = sv.get<PickGeometry>(e);
                if (pg.positions.empty()) continue;
                auto& cache = world.get_or_emplace<SpatialVizCache>(e);

                // Launch a BACKGROUND build when the wanted kind is neither shown nor
                // already being produced, so the render thread never blocks on a
                // million-point structure build. The old mesh keeps drawing until the
                // new one is ready (no flicker).
                if (vizKind != cache.kind && vizKind != cache.pendingKind) {
                    cache.pendingKind = vizKind;
                    auto job  = std::make_shared<SpatialVizJob>();
                    job->kind = vizKind;
                    cache.job = job;
                    std::thread([job, P = pg.positions, idx = pg.indices, vizKind]() {
                        buildSpatialVizVerts(
                            P, idx, vizKind, job->verts, job->boxCount, job->drawn, job->total,
                            [job](float f) { job->progress.store(f, std::memory_order_relaxed); });
                        job->done.store(true, std::memory_order_release);
                    }).detach();
                }

                // Harvest a finished build: upload its vertices to the GPU here on the
                // main (render) thread, then swap it in.
                if (cache.job && cache.job->done.load(std::memory_order_acquire)) {
                    if (cache.mesh != render::kInvalidMesh) renderer.destroyMesh(cache.mesh);
                    if (cache.vbo != render::kInvalidBuffer) renderer.destroyBuffer(cache.vbo);
                    cache.mesh = render::kInvalidMesh; cache.vbo = render::kInvalidBuffer;
                    cache.vertexCount = 0;
                    const auto& verts = cache.job->verts;
                    if (!verts.empty()) {
                        render::BufferDesc bd;
                        bd.type  = render::BufferType::Vertex;
                        bd.usage = render::BufferUsage::Static;
                        bd.data  = verts.data();
                        bd.size  = verts.size() * sizeof(render::Vertex);
                        cache.vbo = renderer.createBuffer(bd);
                        render::MeshDesc md2;
                        md2.vertexBuffer = cache.vbo;
                        md2.layout       = render::Vertex::layout();
                        md2.vertexCount  = static_cast<uint32_t>(verts.size());
                        md2.topology     = render::PrimitiveTopology::Triangles;
                        cache.mesh = renderer.createMesh(md2);
                        cache.vertexCount = static_cast<uint32_t>(verts.size());
                    }
                    cache.kind        = cache.job->kind;
                    cache.boxCount    = cache.job->boxCount;
                    cache.inliers     = cache.job->drawn;
                    cache.total       = cache.job->total;
                    cache.pendingKind = -1;
                    cache.job.reset();
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
    emitOcclusalPlaneViz(world);  // re-add the latched occlusal plane/arrow
    emitCuspViz(world);           // re-add the latched cusp markers
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

    // --- Tree-view (scene outliner) overlay ---------------------------
    auto tvv = world.view<TreeView>();
    for (auto entity : tvv) {
        auto& tv = tvv.get<TreeView>(entity);
        if (!tv.visible || tv.mesh == render::kInvalidMesh || tv.vbo == render::kInvalidBuffer)
            break;

        std::vector<TreeRow> rows;
        buildTreeRows(world, tv, rows);
        render::Vertex verts[kTreeVerts];
        buildTreeGeometry(tv, rows, verts);
        renderer.updateBuffer(tv.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = tv.x; ov.y = tv.y; ov.width = tv.w; ov.height = tv.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = tv.mesh;
        item.texture = tv.atlas;
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

    // --- 3D Compare legend overlay -------------------------------------
    {
        auto lgview = world.view<CompareLegend>();
        for (auto entity : lgview) {
            const auto& lg = lgview.get<CompareLegend>(entity);
            if (!lg.visible || !lg.font || lg.mesh == render::kInvalidMesh) break;

            render::Vertex verts[kLegendVerts];
            buildCompareLegendGeometry(lg, verts);
            renderer.updateBuffer(lg.vbo, verts, sizeof(verts));

            render::OverlayContext ov;
            ov.x = lg.x; ov.y = lg.y; ov.width = lg.w; ov.height = lg.h;
            ov.clearDepth = true;
            Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
            Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
            std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
            std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
            renderer.beginOverlay(ov);

            render::DrawItem item;
            item.mesh    = lg.mesh;
            item.texture = lg.atlas;
            Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
            std::memcpy(item.model, model.data(), sizeof(item.model));
            renderer.submit(item);
            break;
        }
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

    // --- Font Size dialog overlay --------------------------------------
    auto fdview = world.view<FontSizeDialog>();
    for (auto entity : fdview) {
        const auto& fd = fdview.get<FontSizeDialog>(entity);
        if (!fd.visible || !fd.font || fd.mesh == render::kInvalidMesh) break;

        render::Vertex verts[kFontDlgVerts];
        buildFontSizeDialogGeometry(fd, verts);
        renderer.updateBuffer(fd.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = fd.x; ov.y = fd.y; ov.width = fd.w; ov.height = fd.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = fd.mesh;
        item.texture = fd.atlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- Poisson reconstruction dialog overlay ------------------------
    auto pdview = world.view<PoissonDialog>();
    for (auto entity : pdview) {
        const auto& pd = pdview.get<PoissonDialog>(entity);
        if (!pd.visible || !pd.font || pd.mesh == render::kInvalidMesh) break;

        render::Vertex verts[kPoissonVerts];
        buildPoissonDialogGeometry(pd, verts);
        renderer.updateBuffer(pd.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = pd.x; ov.y = pd.y; ov.width = pd.w; ov.height = pd.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = pd.mesh;
        item.texture = pd.atlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- Mode-parameters dialog overlay -------------------------------
    auto mdview = world.view<ModeParamsDialog>();
    for (auto entity : mdview) {
        const auto& md = mdview.get<ModeParamsDialog>(entity);
        if (!md.visible || !md.font || md.mesh == render::kInvalidMesh) break;

        render::Vertex verts[kModeDlgVerts];
        buildModeParamsDialogGeometry(
            md, md.pipeNodeId < 0 ? activeModeLegend(world) : nullptr, verts);
        renderer.updateBuffer(md.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = md.x; ov.y = md.y; ov.width = md.w; ov.height = md.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = md.mesh;
        item.texture = md.atlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- Pipeline Design dialog overlay (modeless) --------------------
    auto plview = world.view<PipelineDialog>();
    for (auto entity : plview) {
        const auto& pl = plview.get<PipelineDialog>(entity);
        if (!pl.visible || !pl.font || pl.mesh == render::kInvalidMesh) break;

        render::Vertex verts[kPipeDlgVerts];
        buildPipelineDialogGeometry(pl, verts);
        renderer.updateBuffer(pl.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = pl.x; ov.y = pl.y; ov.width = pl.w; ov.height = pl.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = pl.mesh;
        item.texture = pl.atlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- Confirm (Yes/No) dialog overlay ------------------------------
    auto cdview = world.view<ConfirmDialog>();
    for (auto entity : cdview) {
        const auto& cd = cdview.get<ConfirmDialog>(entity);
        if (!cd.visible || !cd.font || cd.mesh == render::kInvalidMesh) break;

        render::Vertex verts[kConfirmVerts];
        buildConfirmDialogGeometry(cd, verts);
        renderer.updateBuffer(cd.vbo, verts, sizeof(verts));

        render::OverlayContext ov;
        ov.x = cd.x; ov.y = cd.y; ov.width = cd.w; ov.height = cd.h;
        ov.clearDepth = true;
        Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
        std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
        renderer.beginOverlay(ov);

        render::DrawItem item;
        item.mesh    = cd.mesh;
        item.texture = cd.atlas;
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        std::memcpy(item.model, model.data(), sizeof(item.model));
        renderer.submit(item);
        break;
    }

    // --- Occlusal 2D render: depth / normal / curvature panels ---------
    auto orview = world.view<OcclusalRenderViz>();
    for (auto entity : orview) {
        const auto& orv = orview.get<OcclusalRenderViz>(entity);
        if (!orv.visible || orv.quad == render::kInvalidMesh) break;
        const int sz = 200, pad = 10;
        int y = (int)viewportH - sz - pad;
        for (int c = 0; c < orv.count; ++c) {
            if (orv.tex[c] == render::kInvalidTexture) continue;
            render::OverlayContext ov;
            ov.x = pad + c * (sz + pad); ov.y = y; ov.width = sz; ov.height = sz;
            ov.clearDepth = true;
            Eigen::Matrix4f ovView = Eigen::Matrix4f::Identity();
            Eigen::Matrix4f ovProj = math::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
            std::memcpy(ov.view, ovView.data(), sizeof(ov.view));
            std::memcpy(ov.proj, ovProj.data(), sizeof(ov.proj));
            renderer.beginOverlay(ov);
            render::DrawItem item;
            item.mesh    = orv.quad;
            item.texture = orv.tex[c];
            Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
            std::memcpy(item.model, model.data(), sizeof(item.model));
            renderer.submit(item);
        }
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
        // Grow further if an open submenu flyout extends past the dropdown bottom.
        if (mb.font) {
            float fx0, fy0, fw, fh;
            const float W = static_cast<float>(viewportW);
            if (openFlyoutRect(mb, *mb.font, kMenuTextH, W, static_cast<float>(mb.height),
                               fx0, fy0, fw, fh))
                overlayH = (std::max)(overlayH, fy0 + fh);
        }

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

// --- Pipeline-graph execution (Pipeline Design dialog "Run") ------------------

namespace {
struct PipeRunJob {
    std::atomic<bool>  done{false};
    std::atomic<float> progress{0.0f};
    std::vector<std::vector<Eigen::Vector3f>> results;  // one cloud per Output node
    size_t inCount = 0;
};
struct PipeRunCache {
    std::shared_ptr<PipeRunJob> job;
    std::thread                 worker;
    // The cloud the last run read from. A run SPAWNS its result as a new
    // cloud, so the "exactly one cloud in the scene" fallback stops resolving
    // right after the first run -- with nothing selected, Run #2 would
    // silently do nothing. Falling back to the previous source keeps repeated
    // Runs working on the same input.
    entt::entity lastSource = entt::null;
    // The previous run's spawned outputs: a re-run REPLACES them (undoable).
    // Stacking a new cloud over the visually identical old one made parameter
    // tweaks look like they had no effect.
    std::vector<entt::entity> lastResults;
    ~PipeRunCache() {
        if (worker.joinable()) worker.detach();  // self-contained; dies with the process
    }
};
} // namespace

void pipelineGraphSystem(entt::registry& world, render::IRenderer& renderer) {
    auto& ctx = world.ctx();
    if (!ctx.contains<PipeRunCache>()) ctx.emplace<PipeRunCache>();
    auto& cache = ctx.get<PipeRunCache>();

    // Harvest: spawn each Output's result as a new, selectable point cloud
    // (each undoable).
    if (cache.job && cache.job->done.load(std::memory_order_acquire)) {
        if (cache.worker.joinable()) cache.worker.join();
        // Replace the previous run's outputs (unless the user deleted them
        // already) so a re-run updates the result instead of stacking clouds.
        {
            std::vector<entt::entity> prev;
            for (auto pe : cache.lastResults)
                if (world.valid(pe) && world.all_of<Renderable>(pe)) prev.push_back(pe);
            if (!prev.empty() && !cache.job->results.empty()) {
                pushDeleteOp(world, prev, "Pipeline Re-run");
                for (auto pe : prev) world.destroy(pe);
            }
            cache.lastResults.clear();
        }
        size_t spawned = 0;
        for (const auto& pts : cache.job->results) {
            if (pts.empty()) continue;
            std::vector<render::Vertex> verts(pts.size());
            for (size_t i = 0; i < pts.size(); ++i) {
                verts[i] = {{pts[i].x(), pts[i].y(), pts[i].z()},
                            {0.85f, 0.85f, 0.88f}};
            }
            render::BufferDesc bd;
            bd.type  = render::BufferType::Vertex;
            bd.usage = render::BufferUsage::Static;
            bd.data  = verts.data();
            bd.size  = verts.size() * sizeof(render::Vertex);
            render::BufferHandle vbo = renderer.createBuffer(bd);
            render::MeshDesc md;
            md.vertexBuffer = vbo;
            md.layout       = render::Vertex::layout();
            md.vertexCount  = (uint32_t)verts.size();
            md.topology     = render::PrimitiveTopology::Points;

            auto e = world.create();
            world.emplace<Transform>(e, Transform{});
            Renderable r;
            r.mesh       = renderer.createMesh(md);
            r.pointCloud = true;
            Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX);
            Eigen::Vector3f mx = Eigen::Vector3f::Constant(-FLT_MAX);
            PickGeometry pick;
            pick.positions = pts;
            for (const auto& p : pts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
            r.boundsMin = mn;
            r.boundsMax = mx;
            world.emplace<Renderable>(e, r);
            world.emplace<PickGeometry>(e, std::move(pick));
            world.emplace<VertexSource>(e, VertexSource{vbo, std::move(verts)});
            pushSpawnOp(world, e, "Pipeline Result");
            cache.lastResults.push_back(e);
            SDL_Log("pipelineGraphSystem: output %zu: %zu -> %zu points", spawned,
                    cache.job->inCount, pts.size());
            ++spawned;
        }
        if (spawned == 0) SDL_Log("pipelineGraphSystem: produced no points");
        cache.job.reset();
    }

    // Launch on the Run edge.
    PipelineDialog* d = nullptr;
    auto dv = world.view<PipelineDialog>();
    for (auto e : dv) { d = &dv.get<PipelineDialog>(e); break; }
    if (!d || !d->requestRun) return;
    if (cache.job) return;  // still running; keep the edge pending
    d->requestRun = false;

    // Validate: at least one Source and one Output on the canvas.
    bool hasSource = false, hasOutput = false;
    for (const auto& n : d->nodes) {
        hasSource |= n.role == PipeNodeRole::Source;
        hasOutput |= n.role == PipeNodeRole::Output;
    }
    if (!hasSource || !hasOutput) {
        SDL_Log("pipelineGraphSystem: graph needs a Source and an Output node");
        return;
    }

    // Source cloud: first selected entity with points, else the previous
    // run's source (still alive), else the only cloud in the scene.
    entt::entity src = entt::null, only = entt::null;
    size_t count = 0;
    auto v = world.view<Renderable, PickGeometry>();
    for (auto e : v) {
        if (v.get<PickGeometry>(e).positions.empty()) continue;
        ++count;
        only = e;
        if (v.get<Renderable>(e).selected) { src = e; break; }
    }
    if (src == entt::null && cache.lastSource != entt::null &&
        world.valid(cache.lastSource) && world.all_of<Renderable, PickGeometry>(cache.lastSource) &&
        !world.get<PickGeometry>(cache.lastSource).positions.empty())
        src = cache.lastSource;
    if (src == entt::null && count == 1) src = only;
    if (src == entt::null) {
        SDL_Log("pipelineGraphSystem: no source cloud (select one)");
        return;
    }
    cache.lastSource = src;
    std::vector<Eigen::Vector3f> raw = world.get<PickGeometry>(src).positions;
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    if (world.all_of<Transform>(src)) M = world.get<Transform>(src).matrix();

    auto job = std::make_shared<PipeRunJob>();
    job->inCount = raw.size();
    cache.job    = job;
    SDL_Log("pipelineGraphSystem: running graph (%zu nodes, %zu links) on %zu points...",
            d->nodes.size(), d->links.size(), raw.size());
    // Self-contained copies of the graph for the worker.
    std::vector<PipeNode> nodes = d->nodes;
    std::vector<PipeLink> links = d->links;
    cache.worker = std::thread([job, nodes = std::move(nodes), links = std::move(links),
                                raw = std::move(raw), M]() mutable {
        auto source = std::make_shared<std::vector<Eigen::Vector3f>>(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            Eigen::Vector4f w = M * Eigen::Vector4f(raw[i].x(), raw[i].y(), raw[i].z(), 1.0f);
            (*source)[i] = Eigen::Vector3f(w.x(), w.y(), w.z());
        }

        // Memoized DAG evaluation: each node computes once, so a branching
        // output pin feeds all its consumers without re-running the stage.
        auto findNode = [&](int id) -> const PipeNode* {
            for (const auto& n : nodes)
                if (n.id == id) return &n;
            return nullptr;
        };
        size_t stageTotal = 0;
        for (const auto& n : nodes)
            if (n.role == PipeNodeRole::Stage) ++stageTotal;
        std::atomic<size_t> stageDone{0};
        using Cloud = std::shared_ptr<std::vector<Eigen::Vector3f>>;
        std::unordered_map<int, Cloud> memo;
        std::unordered_set<int> visiting;  // cycle guard
        std::function<Cloud(int)> eval = [&](int id) -> Cloud {
            auto it = memo.find(id);
            if (it != memo.end()) return it->second;
            if (!visiting.insert(id).second) return nullptr;  // cycle
            const PipeNode* n = findNode(id);
            if (!n) return nullptr;
            Cloud result;
            if (n->role == PipeNodeRole::Source) {
                result = source;
            } else {
                const PipeLink* inL = nullptr;
                const PipeLink* condL = nullptr;
                for (const auto& l : links) {
                    if (l.to != id) continue;
                    if (l.toPin == 1) condL = &l;
                    else              inL = &l;
                }
                Cloud up = inL ? eval(inL->from) : nullptr;
                if (up) {
                    if (n->role == PipeNodeRole::Output) {
                        result = up;
                    } else {
                        int idx = modes::modeIndexByName(n->mode.c_str());
                        if (idx >= 0) {
                            modes::ModeInput in;
                            in.points = *up;
                            in.params = n->params;
                            // Condition pin: the upstream cloud on the amber pin
                            // is the CANDIDATE set -- the filter may only touch
                            // main-input points that also appear there (matched
                            // bit-exact; selection stages pass points through
                            // unmodified, so coordinates survive intact).
                            if (condL && modes::modeSupportsCandidates(idx)) {
                                Cloud cond = eval(condL->from);
                                if (cond) {
                                    auto key = [](const Eigen::Vector3f& p) {
                                        uint32_t k[3];
                                        std::memcpy(k, p.data(), sizeof(k));
                                        return ((uint64_t)k[0] << 32 | k[1]) ^
                                               ((uint64_t)k[2] * 0x9E3779B97F4A7C15ull);
                                    };
                                    std::unordered_set<uint64_t> set;
                                    set.reserve(cond->size() * 2);
                                    for (const auto& p : *cond) set.insert(key(p));
                                    in.candidates.resize(in.points.size());
                                    for (size_t pi = 0; pi < in.points.size(); ++pi)
                                        in.candidates[pi] = set.count(key(in.points[pi])) ? 1 : 0;
                                }
                            }
                            auto next = std::make_shared<std::vector<Eigen::Vector3f>>();
                            size_t sdone = stageDone.load(std::memory_order_relaxed);
                            auto prog = [job, sdone, stageTotal](float f) {
                                job->progress.store(
                                    stageTotal ? ((float)sdone + f) / (float)stageTotal : f,
                                    std::memory_order_relaxed);
                            };
                            if (n->fit)
                                modes::runModeFit(idx, in, *next, prog);
                            else
                                modes::runModePoints(idx, in, *next, prog);
                            stageDone.fetch_add(1, std::memory_order_relaxed);
                            result = next;
                        } else {
                            result = up;  // unknown stage: pass through
                        }
                    }
                }
            }
            visiting.erase(id);
            memo[id] = result;
            return result;
        };

        for (const auto& n : nodes) {
            if (n.role != PipeNodeRole::Output) continue;
            Cloud r = eval(n.id);
            if (r && !r->empty()) job->results.push_back(*r);
        }
        job->done.store(true, std::memory_order_release);
    });
}

bool pipelineRunProgress(entt::registry& world, float& outProgress, std::string& outName) {
    auto& ctx = world.ctx();
    if (!ctx.contains<PipeRunCache>()) return false;
    auto& cache = ctx.get<PipeRunCache>();
    if (!cache.job || cache.job->done.load(std::memory_order_acquire)) return false;
    outProgress = cache.job->progress.load(std::memory_order_relaxed);
    outName     = "Pipeline";
    return true;
}


} // namespace orange::ecs
