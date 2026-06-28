#include "orange/core/application.h"

#include <SDL3/SDL.h>

#include <cfloat>
#include <vector>

#include "orange/core/draw_mode.h"
#include "orange/core/modes.h"
#include "orange/core/occlusal_plane.h"
#include "orange/core/occlusal_render.h"
#include "orange/core/primitives.h"
#include "orange/core/sam_segment.h"
#include "orange/core/tooth_segment.h"
#include "orange/core/screenshot.h"
#include "orange/ecs/components.h"
#include "orange/ecs/systems.h"

namespace orange::core {

std::string Application::defaultPluginName(render::Backend backend) {
    switch (backend) {
        case render::Backend::OpenGL: return "render_gl";
        case render::Backend::Vulkan: return "render_vk";
    }
    return "render_gl";
}

std::string Application::executableDir() {
    const char* base = SDL_GetBasePath();  // owned by SDL, do not free
    return base ? std::string(base) : std::string();
}

bool Application::init(const AppConfig& config) {
    config_ = config;

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        SDL_Log("Application: SDL_Init failed: %s", SDL_GetError());
        return false;
    }

    const std::string dir =
        config_.pluginDir.empty() ? executableDir() : config_.pluginDir;
    const std::string name =
        config_.pluginName.empty() ? defaultPluginName(config_.backend)
                                   : config_.pluginName;

    plugin_ = RenderPlugin::load(dir, name);
    if (!plugin_) {
        SDL_Log("Application: could not load render plugin '%s'", name.c_str());
        return false;
    }

    // The window must match the backend the plugin actually provides.
    const render::Backend backend = plugin_->info()->backend;
    if (!window_.create(config_.title, config_.width, config_.height, backend)) {
        return false;
    }

    render::InitInfo info;
    info.nativeWindow = window_.handle();
    info.width        = window_.width();
    info.height       = window_.height();
    info.vsync        = config_.vsync;

    if (!plugin_->renderer()->init(info)) {
        SDL_Log("Application: renderer init failed");
        return false;
    }

    SDL_Log("Application: initialized with %s backend",
            render::to_string(backend));
    return true;
}

void Application::run(const std::function<void(entt::registry&, float)>& onUpdate) {
    if (!plugin_ || !plugin_->renderer()) return;
    running_ = true;

    // Push the initial scene coloring (grayscale by default) so the first frame
    // already reflects it; the renderer otherwise starts in mode 0 (original).
    plugin_->renderer()->setColorMode(colorMode_);
    vsync_ = config_.vsync;

    Uint64 last = SDL_GetPerformanceCounter();
    const double freq = static_cast<double>(SDL_GetPerformanceFrequency());

    while (running_) {
        input_.newFrame();  // reset per-frame deltas before draining events

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_EVENT_QUIT:
                    running_ = false;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (e.key.key == SDLK_ESCAPE) running_ = false;
                    if (e.key.scancode == SDL_SCANCODE_C) capture_ = true;  // screenshot
                    if (e.key.scancode == SDL_SCANCODE_TAB) {  // cycle each selected mesh's draw mode
                        auto v = world_.view<ecs::Renderable>();
                        // With a single mesh, Tab applies to it without an explicit
                        // selection (there's nothing else it could mean).
                        size_t count = 0;
                        for (auto ent : v) { (void)ent; ++count; }
                        for (auto ent : v) {
                            auto& r = v.get<ecs::Renderable>(ent);
                            if (!r.selected && count != 1) continue;
                            r.drawMode = static_cast<DrawMode>(
                                (static_cast<uint32_t>(r.drawMode) + 1) %
                                static_cast<uint32_t>(DrawMode::Count));
                        }
                    }
                    if (e.key.scancode == SDL_SCANCODE_M) {                 // cycle processing mode
                        auto& ctx = world_.ctx();
                        if (!ctx.contains<modes::ModeState>()) ctx.emplace<modes::ModeState>();
                        auto& ms = ctx.get<modes::ModeState>();
                        ms.index = (ms.index + 1) % modes::modeCount();
                        ms.generation++;
                        SDL_Log("Application: processing mode = %s", modes::modeName(ms.index));
                    }
                    if (e.key.scancode == SDL_SCANCODE_O && (e.key.mod & SDL_KMOD_CTRL)) {
                        // Ctrl+O: open the native file dialog (same as File > Open...).
                        auto mv = world_.view<ecs::MenuBar>();
                        for (auto ent : mv) { mv.get<ecs::MenuBar>(ent).requestOpenFile = true; break; }
                    }
                    if (e.key.scancode == SDL_SCANCODE_A && (e.key.mod & SDL_KMOD_CTRL)) {
                        // Ctrl+A: select every Renderable that is visible on screen.
                        auto v = world_.view<ecs::Renderable>();
                        for (auto ent : v)
                            if (ecs::entityVisibleOnScreen(world_, ent, window_.width(),
                                                           window_.height()))
                                v.get<ecs::Renderable>(ent).selected = true;
                    }
                    if (e.key.scancode == SDL_SCANCODE_DELETE) {
                        // Delete the selected meshes (collect first, then destroy).
                        std::vector<entt::entity> dead;
                        auto v = world_.view<ecs::Renderable>();
                        for (auto ent : v)
                            if (v.get<ecs::Renderable>(ent).selected) dead.push_back(ent);
                        for (auto ent : dead) world_.destroy(ent);
                        if (!dead.empty())
                            SDL_Log("Application: deleted %zu mesh(es)", dead.size());
                    }
                    if (e.key.scancode == SDL_SCANCODE_EQUALS ||
                        e.key.scancode == SDL_SCANCODE_KP_PLUS) {  // grow point sprites
                        pointSize_ = pointSize_ + 1.0f > 64.0f ? 64.0f : pointSize_ + 1.0f;
                        plugin_->renderer()->setPointSize(pointSize_);
                        SDL_Log("Application: point size = %.0f", pointSize_);
                    }
                    if (e.key.scancode == SDL_SCANCODE_MINUS ||
                        e.key.scancode == SDL_SCANCODE_KP_MINUS) {  // shrink point sprites
                        pointSize_ = pointSize_ - 1.0f < 1.0f ? 1.0f : pointSize_ - 1.0f;
                        plugin_->renderer()->setPointSize(pointSize_);
                        SDL_Log("Application: point size = %.0f", pointSize_);
                    }
                    if (e.key.scancode == SDL_SCANCODE_GRAVE) {
                        if (e.key.mod & SDL_KMOD_SHIFT) {  // Shift+` cycles the selection's coloring
                            auto v = world_.view<ecs::Renderable>();
                            size_t count = 0; for (auto e2 : v) { (void)e2; ++count; }
                            uint32_t base = 0;
                            for (auto e2 : v) {
                                auto& rr = v.get<ecs::Renderable>(e2);
                                if (rr.selected || count == 1) { base = rr.colorMode; break; }
                            }
                            uint32_t next = (base + 1) % 4;  // default/height/position/gray
                            for (auto e2 : v) {
                                auto& rr = v.get<ecs::Renderable>(e2);
                                if (rr.selected || count == 1) rr.colorMode = next;
                            }
                            static const char* kColorNames[4] = {"default", "height",
                                                                 "position", "grayscale"};
                            SDL_Log("Application: color mode = %s", kColorNames[next]);
                        } else {                           // ` toggles point-sprite lighting
                            lighting_ = !lighting_;
                            plugin_->renderer()->setLighting(lighting_);
                            SDL_Log("Application: lighting = %s", lighting_ ? "on" : "off");
                        }
                    }
                    if (e.key.scancode == SDL_SCANCODE_LEFTBRACKET ||
                        e.key.scancode == SDL_SCANCODE_RIGHTBRACKET) {  // [ / ] tune cusp prominence
                        auto& ctx = world_.ctx();
                        if (ctx.contains<ecs::CuspParams>() && ctx.get<ecs::CuspParams>().ready) {
                            auto& cp = ctx.get<ecs::CuspParams>();
                            cp.heightGate += (e.key.scancode == SDL_SCANCODE_RIGHTBRACKET) ? 0.05f : -0.05f;
                            if (cp.heightGate < 0.0f) cp.heightGate = 0.0f;
                            if (cp.heightGate > 0.9f) cp.heightGate = 0.9f;
                            recomputeCusps();
                            SDL_Log("Application: cusp height gate = %.2f (%zu cusps)",
                                    cp.heightGate, ctx.get<ecs::CuspViz>().points.size());
                        }
                    }
                    if (e.key.scancode == SDL_SCANCODE_SPACE) {  // Space toggles the grid
                        auto& ctx = world_.ctx();
                        if (!ctx.contains<ecs::GridState>()) ctx.emplace<ecs::GridState>();
                        auto& gs = ctx.get<ecs::GridState>();
                        gs.visible = !gs.visible;
                        SDL_Log("Application: grid = %s", gs.visible ? "on" : "off");
                    }
                    if (e.key.scancode == SDL_SCANCODE_R) {  // R resets the camera to its home pose
                        auto v = world_.view<ecs::Camera, ecs::CameraManipulator>();
                        for (auto ent : v) {
                            if (!v.get<ecs::Camera>(ent).primary) continue;
                            auto& m = v.get<ecs::CameraManipulator>(ent);
                            m.animFrom = m.orientation;  m.animTo = m.homeOrientation;
                            m.animTime = 0.0f;           m.animating = true;
                            m.targetFrom = m.target;     m.targetTo = m.homeTarget;
                            m.distFrom = m.distance;     m.distTo = m.homeDistance;
                            m.targetAnimTime = 0.0f;     m.targetAnimating = true;
                            break;
                        }
                        SDL_Log("Application: camera reset");
                    }
                    if (e.key.scancode == SDL_SCANCODE_H) {
                        // Unhide all: reveal meshes hidden by the None draw mode so
                        // they can be seen and selected again.
                        int n = 0;
                        auto v = world_.view<ecs::Renderable>();
                        for (auto ent : v) {
                            auto& r = v.get<ecs::Renderable>(ent);
                            if (r.drawMode == DrawMode::None) { r.drawMode = DrawMode::Solid; ++n; }
                        }
                        if (n) SDL_Log("Application: revealed %d hidden mesh(es)", n);
                    }
                    break;
                case SDL_EVENT_MOUSE_MOTION: {
                    // SDL reports mouse in logical points; the render viewport
                    // (and gizmo rect) are in pixels. Convert so picking lines up.
                    int lw = 0, lh = 0;
                    SDL_GetWindowSize(window_.handle(), &lw, &lh);
                    float sx = lw > 0 ? static_cast<float>(window_.width()) / lw : 1.0f;
                    float sy = lh > 0 ? static_cast<float>(window_.height()) / lh : 1.0f;
                    input_.mousePosX = e.motion.x * sx;
                    input_.mousePosY = e.motion.y * sy;
                    input_.mouseDeltaX += e.motion.xrel * sx;
                    input_.mouseDeltaY += e.motion.yrel * sy;
                    break;
                }
                case SDL_EVENT_MOUSE_WHEEL:
                    input_.wheel += e.wheel.y;
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                case SDL_EVENT_MOUSE_BUTTON_UP: {
                    bool down = (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
                    if (e.button.button == SDL_BUTTON_LEFT) {
                        input_.buttonLeft = down;
                        if (down) input_.leftClicked = true;
                        else      input_.leftReleased = true;
                    }
                    if (e.button.button == SDL_BUTTON_RIGHT)  input_.buttonRight  = down;
                    if (e.button.button == SDL_BUTTON_MIDDLE) input_.buttonMiddle = down;
                    break;
                }
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: {
                    uint32_t w = static_cast<uint32_t>(e.window.data1);
                    uint32_t h = static_cast<uint32_t>(e.window.data2);
                    window_.setSize(w, h);
                    plugin_->renderer()->resize(w, h);
                    break;
                }
                default:
                    break;
            }
        }

        Uint64 now = SDL_GetPerformanceCounter();
        float dt = static_cast<float>((now - last) / freq);
        last = now;

        input_.shift = (SDL_GetModState() & SDL_KMOD_SHIFT) != 0;
        input_.ctrl  = (SDL_GetModState() & SDL_KMOD_CTRL) != 0;
        input_.alt   = (SDL_GetModState() & SDL_KMOD_ALT) != 0;

        if (onUpdate) onUpdate(world_, dt);
        ecs::menuBarInputSystem(world_, input_, window_.width(), window_.height());
        // A menu click raised an action: dispatch it (mirrors the keys), then sync
        // the menu checkmarks to the live state for this frame's draw.
        {
            auto mv = world_.view<ecs::MenuBar>();
            for (auto e : mv) {
                auto& mb = mv.get<ecs::MenuBar>(e);
                if (mb.triggered != ecs::MenuAction::None) {
                    applyMenuAction(static_cast<int>(mb.triggered));
                    mb.triggered = ecs::MenuAction::None;
                }
                break;
            }
        }
        syncMenu();
        {  // expose the point size so renderSystem can size selection markers
            auto& c = world_.ctx();
            if (!c.contains<ecs::PointSizeState>()) c.emplace<ecs::PointSizeState>();
            c.get<ecs::PointSizeState>().size = pointSize_;
        }
        ecs::fpsWidgetInputSystem(world_, input_, dt, window_.width(), window_.height());
        ecs::treeViewInputSystem(world_, input_, window_.width(), window_.height());
        ecs::cameraControlsInputSystem(world_, input_, dt, window_.width(), window_.height());
        ecs::crossSectionInputSystem(world_, input_, window_.width(), window_.height());
        ecs::poissonDialogInputSystem(world_, input_, window_.width(), window_.height());
        ecs::confirmDialogInputSystem(world_, input_, window_.width(), window_.height());
        ecs::axisGizmoInputSystem(world_, input_, dt, window_.width(), window_.height());
        ecs::selectionToolbarInputSystem(world_, input_, window_.width(), window_.height());
        ecs::cameraManipulatorSystem(world_, input_, dt);
        ecs::pickingSystem(world_, input_, window_.width(), window_.height());
        ecs::spinSystem(world_, dt);
        ecs::processingModeSystem(world_);  // emits the active mode's debug geometry
        ecs::renderSystem(world_, *plugin_->renderer(), window_.width(),
                          window_.height());

        if (capture_) {
            capture_ = false;
            saveScreenshot(*plugin_->renderer(), executableDir() + "orange_capture.png");
        }
    }
}

void Application::applyMenuAction(int action) {
    using A = ecs::MenuAction;
    auto* r = plugin_ ? plugin_->renderer() : nullptr;

    // Set every selected mesh's draw mode (or the only mesh, if there's just one).
    auto setDrawModeSel = [&](DrawMode dm) {
        auto v = world_.view<ecs::Renderable>();
        size_t count = 0;
        for (auto ent : v) { (void)ent; ++count; }
        for (auto ent : v) {
            auto& rr = v.get<ecs::Renderable>(ent);
            if (!rr.selected && count != 1) continue;
            rr.drawMode = dm;
        }
    };
    // Apply a scene-coloring mode to the selected meshes (or the only mesh).
    auto setColor = [&](uint32_t mode) {
        colorMode_ = mode;
        auto v = world_.view<ecs::Renderable>();
        size_t count = 0;
        for (auto ent : v) { (void)ent; ++count; }
        for (auto ent : v) {
            auto& rr = v.get<ecs::Renderable>(ent);
            if (!rr.selected && count != 1) continue;
            rr.colorMode = mode;
        }
    };
    auto setMode = [&](int idx) {
        auto& ctx = world_.ctx();
        if (!ctx.contains<modes::ModeState>()) ctx.emplace<modes::ModeState>();
        auto& ms = ctx.get<modes::ModeState>();
        ms.index = idx % modes::modeCount();
        ms.generation++;
    };
    // Upload a primitive triangle soup and spawn a real, pickable entity at the
    // primary camera's pivot, sized to its orbit distance so it is always visible.
    auto spawnPrimitive = [&](const std::vector<geometry::Triangle>& tris) {
        if (tris.empty() || !r) return;
        std::vector<render::Vertex> verts;
        verts.reserve(tris.size() * 3);
        ecs::PickGeometry pick;
        Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX);
        Eigen::Vector3f mx = Eigen::Vector3f::Constant(-FLT_MAX);
        uint32_t idx = 0;
        for (const auto& t : tris)
            for (int k = 0; k < 3; ++k) {
                const Eigen::Vector3f& p = t.v[k];
                render::Vertex v{};
                v.position[0] = p.x(); v.position[1] = p.y(); v.position[2] = p.z();
                v.color[0] = t.c[k].x(); v.color[1] = t.c[k].y(); v.color[2] = t.c[k].z();
                verts.push_back(v);
                pick.positions.push_back(p);
                pick.indices.push_back(idx++);
                mn = mn.cwiseMin(p); mx = mx.cwiseMax(p);
            }

        render::BufferDesc bd;
        bd.type  = render::BufferType::Vertex;
        bd.usage = render::BufferUsage::Static;
        bd.data  = verts.data();
        bd.size  = verts.size() * sizeof(render::Vertex);
        render::BufferHandle vbo = r->createBuffer(bd);

        render::MeshDesc md;
        md.vertexBuffer = vbo;
        md.indexBuffer  = render::kInvalidBuffer;
        md.layout       = render::Vertex::layout();
        md.vertexCount  = static_cast<uint32_t>(verts.size());
        render::MeshHandle mesh = r->createMesh(md);

        Eigen::Vector3f pos = Eigen::Vector3f::Zero();
        float scl = 1.0f;
        auto cv = world_.view<ecs::Camera, ecs::CameraManipulator>();
        for (auto e : cv) {
            if (!cv.get<ecs::Camera>(e).primary) continue;
            auto& m = cv.get<ecs::CameraManipulator>(e);
            pos = m.target;
            scl = m.distance * 0.3f;
            break;
        }

        auto e = world_.create();
        ecs::Transform tr;
        tr.position = pos;
        tr.scale    = Eigen::Vector3f::Constant(scl);
        world_.emplace<ecs::Transform>(e, tr);
        ecs::Renderable rr;
        rr.mesh      = mesh;
        rr.boundsMin = mn;
        rr.boundsMax = mx;
        world_.emplace<ecs::Renderable>(e, rr);
        world_.emplace<ecs::PickGeometry>(e, std::move(pick));
    };
    const Eigen::Vector3f kPrimColor(0.72f, 0.74f, 0.78f);  // neutral (recolored by color mode)

    // Pull the first selected mesh's vertices into world space (for the pipeline
    // steps), along with its triangle index list. Returns false if nothing
    // suitable is selected; reports the world bbox diagonal for marker sizing.
    auto selectedWorldPoints = [&](std::vector<Eigen::Vector3f>& out,
                                   std::vector<uint32_t>& outIdx, float& diag) -> bool {
        entt::entity src = entt::null, only = entt::null;
        size_t count = 0;
        auto v = world_.view<ecs::Renderable, ecs::PickGeometry, ecs::Transform>();
        for (auto ent : v) {
            ++count; only = ent;
            if (v.get<ecs::Renderable>(ent).selected) { src = ent; break; }
        }
        if (src == entt::null && count == 1) src = only;  // single mesh: run without selecting
        if (src == entt::null) return false;
        const auto& pick = world_.get<ecs::PickGeometry>(src);
        const Eigen::Matrix4f M = world_.get<ecs::Transform>(src).matrix();
        out.clear();
        out.reserve(pick.positions.size());
        Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX);
        Eigen::Vector3f mx = Eigen::Vector3f::Constant(-FLT_MAX);
        for (const auto& p : pick.positions) {
            Eigen::Vector4f wp = M * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
            Eigen::Vector3f w(wp.x(), wp.y(), wp.z());
            out.push_back(w);
            mn = mn.cwiseMin(w); mx = mx.cwiseMax(w);
        }
        outIdx = pick.indices;  // triangle list shares the position indexing
        diag = out.empty() ? 0.0f : (mx - mn).norm();
        return !out.empty();
    };

    // Compute the occlusal channels for the current plane + cached mesh, then
    // upload `count` of them (3 = channels, 4 = + segmentation) as the bottom
    // panels. Returns false if there is no plane/mesh yet.
    auto showChannels = [&](int count) -> bool {
        auto& ctx = world_.ctx();
        if (!ctx.contains<ecs::OcclusalPlaneViz>() || !ctx.contains<ecs::CuspParams>() || !r)
            return false;
        auto& pv = ctx.get<ecs::OcclusalPlaneViz>();  // position/normal persist even if hidden
        auto& cp = ctx.get<ecs::CuspParams>();
        if (!cp.ready || cp.points.empty()) return false;

        geometry::OcclusalChannels ch =
            geometry::renderOcclusalChannels(cp.points, pv.position, pv.normal, 256);
        if (!ch.valid) return false;

        auto rv = world_.view<ecs::OcclusalRenderViz>();
        for (auto e : rv) {
            auto& orv = rv.get<ecs::OcclusalRenderViz>(e);
            for (int c = 0; c < 4; ++c)
                if (orv.tex[c] != render::kInvalidTexture) {
                    r->destroyTexture(orv.tex[c]); orv.tex[c] = render::kInvalidTexture;
                }
            auto mk = [&](const std::vector<uint8_t>& px) {
                render::TextureDesc td;
                td.width = ch.width; td.height = ch.height; td.pixels = px.data();
                return r->createTexture(td);
            };
            orv.tex[0] = mk(ch.depth);
            orv.tex[1] = mk(ch.normal);
            orv.tex[2] = mk(ch.curvature);
            if (count >= 4) orv.tex[3] = mk(ch.segment);
            orv.count   = count;
            orv.visible = true;
            return true;
        }
        return false;
    };

    // Any menu action clears the pipeline overlays (occlusal plane / cusp
    // markers); the pipeline steps below re-activate their own when they run.
    {
        auto& c = world_.ctx();
        if (c.contains<ecs::OcclusalPlaneViz>())  c.get<ecs::OcclusalPlaneViz>().active = false;
        if (c.contains<ecs::CuspViz>())           c.get<ecs::CuspViz>().active = false;
        if (c.contains<ecs::OcclusalRenderViz>()) c.get<ecs::OcclusalRenderViz>().visible = false;
    }

    switch (static_cast<A>(action)) {
        case A::OpenFile: {
            auto mv = world_.view<ecs::MenuBar>();
            for (auto e : mv) { mv.get<ecs::MenuBar>(e).requestOpenFile = true; break; }
            break;
        }
        case A::Screenshot: capture_ = true; break;
        case A::Quit:       running_ = false; break;

        case A::ToggleGrid: {
            auto& ctx = world_.ctx();
            if (!ctx.contains<ecs::GridState>()) ctx.emplace<ecs::GridState>();
            auto& gs = ctx.get<ecs::GridState>();
            gs.visible = !gs.visible;
            break;
        }
        case A::ResetCamera: {
            auto v = world_.view<ecs::Camera, ecs::CameraManipulator>();
            for (auto ent : v) {
                if (!v.get<ecs::Camera>(ent).primary) continue;
                auto& m = v.get<ecs::CameraManipulator>(ent);
                m.animFrom = m.orientation;  m.animTo = m.homeOrientation;
                m.animTime = 0.0f;           m.animating = true;
                m.targetFrom = m.target;     m.targetTo = m.homeTarget;
                m.distFrom = m.distance;     m.distTo = m.homeDistance;
                m.targetAnimTime = 0.0f;     m.targetAnimating = true;
                break;
            }
            break;
        }
        case A::ToggleUpAxis: {
            auto v = world_.view<ecs::AxisGizmo>();
            for (auto ent : v) { auto& g = v.get<ecs::AxisGizmo>(ent); g.zUp = !g.zUp; break; }
            break;
        }
        case A::ToggleProjection: {
            auto v = world_.view<ecs::Camera>();
            for (auto ent : v) {
                auto& c = v.get<ecs::Camera>(ent);
                if (!c.primary) continue;
                c.mode = (c.mode == ecs::ProjectionMode::Perspective)
                             ? ecs::ProjectionMode::Orthographic
                             : ecs::ProjectionMode::Perspective;
                break;
            }
            break;
        }
        case A::ToggleLighting: lighting_ = !lighting_; if (r) r->setLighting(lighting_); break;
        case A::ToggleVsync:    vsync_ = !vsync_;       if (r) r->setVsync(vsync_);       break;
        case A::ToggleCrossSection: {
            auto v = world_.view<ecs::CrossSection>();
            for (auto ent : v) { auto& cs = v.get<ecs::CrossSection>(ent); cs.enabled = !cs.enabled; break; }
            break;
        }

        case A::ColorOriginal: setColor(0); break;
        case A::ColorHeight:   setColor(1); break;
        case A::ColorPosition: setColor(2); break;
        case A::ColorGray:     setColor(3); break;

        case A::DrawNone:      setDrawModeSel(DrawMode::None);             break;
        case A::DrawSolid:     setDrawModeSel(DrawMode::Solid);            break;
        case A::DrawWireframe: setDrawModeSel(DrawMode::WireFrame);          break;
        case A::DrawWireSolid: setDrawModeSel(DrawMode::WireFrameOverSolid); break;
        case A::DrawPoint:     setDrawModeSel(DrawMode::Point);            break;

        case A::SelectAll: {
            auto v = world_.view<ecs::Renderable>();
            for (auto ent : v)
                if (ecs::entityVisibleOnScreen(world_, ent, window_.width(), window_.height()))
                    v.get<ecs::Renderable>(ent).selected = true;
            break;
        }
        case A::ClearSelection: {
            auto v = world_.view<ecs::Renderable>();
            for (auto ent : v) v.get<ecs::Renderable>(ent).selected = false;
            auto ev = world_.view<ecs::ElementSelection>();
            for (auto ent : ev) {
                auto& es = ev.get<ecs::ElementSelection>(ent);
                es.vertices.clear(); es.faces.clear(); es.edges.clear();
            }
            break;
        }
        case A::DeleteSelected: {
            std::vector<entt::entity> dead;
            auto v = world_.view<ecs::Renderable>();
            for (auto ent : v) if (v.get<ecs::Renderable>(ent).selected) dead.push_back(ent);
            for (auto ent : dead) world_.destroy(ent);
            break;
        }
        case A::UnhideAll: {
            auto v = world_.view<ecs::Renderable>();
            for (auto ent : v) {
                auto& rr = v.get<ecs::Renderable>(ent);
                if (rr.drawMode == DrawMode::None) rr.drawMode = DrawMode::Solid;
            }
            break;
        }
        case A::PointSizeUp:
            pointSize_ = pointSize_ + 1.0f > 64.0f ? 64.0f : pointSize_ + 1.0f;
            if (r) r->setPointSize(pointSize_);
            break;
        case A::PointSizeDown:
            pointSize_ = pointSize_ - 1.0f < 1.0f ? 1.0f : pointSize_ - 1.0f;
            if (r) r->setPointSize(pointSize_);
            break;

        case A::ModeOff: setMode(-1); break;
        case A::Mode0: case A::Mode1: case A::Mode2:  case A::Mode3:
        case A::Mode4: case A::Mode5: case A::Mode6:  case A::Mode7:
        case A::Mode8: case A::Mode9: case A::Mode10: case A::Mode11:
        case A::Mode12: case A::Mode13: case A::Mode14: case A::Mode15:
            setMode(action - static_cast<int>(A::Mode0));
            break;

        case A::CreatePlane:    spawnPrimitive(geometry::buildPlane(1.0f, kPrimColor)); break;
        case A::CreateBox:      spawnPrimitive(geometry::buildBox(Eigen::Vector3f(1, 1, 1), kPrimColor)); break;
        case A::CreateSphere:   spawnPrimitive(geometry::buildSphere(0.5f, 24, kPrimColor)); break;
        case A::CreateCylinder: spawnPrimitive(geometry::buildCylinder(0.5f, 1.0f, 24, kPrimColor)); break;
        case A::CreateCone:     spawnPrimitive(geometry::buildCone(0.5f, 1.0f, 24, kPrimColor)); break;
        case A::CreateTorus:    spawnPrimitive(geometry::buildTorus(0.5f, 0.18f, 24, 16, kPrimColor)); break;
        case A::CreateDisk:     spawnPrimitive(geometry::buildDisk(0.5f, 24, kPrimColor)); break;
        case A::CreateCapsule:  spawnPrimitive(geometry::buildCapsule(0.3f, 0.6f, 16, kPrimColor)); break;
        case A::CreateArrow:    spawnPrimitive(geometry::buildArrow(1.0f, 0.08f, 16, kPrimColor)); break;

        case A::SpatialNone:   case A::SpatialBVH:    case A::SpatialOctree:
        case A::SpatialKDTree: case A::SpatialGrid:   case A::SpatialLoose:
        case A::SpatialBSP:    case A::SpatialRTree:  case A::SpatialBall: {
            int k = action == static_cast<int>(A::SpatialBVH)    ? 1
                  : action == static_cast<int>(A::SpatialOctree) ? 2
                  : action == static_cast<int>(A::SpatialKDTree) ? 3
                  : action == static_cast<int>(A::SpatialGrid)   ? 4
                  : action == static_cast<int>(A::SpatialLoose)  ? 5
                  : action == static_cast<int>(A::SpatialBSP)    ? 6
                  : action == static_cast<int>(A::SpatialRTree)  ? 7
                  : action == static_cast<int>(A::SpatialBall)   ? 8 : 0;
            auto& ctx = world_.ctx();
            if (!ctx.contains<ecs::SpatialViz>()) ctx.emplace<ecs::SpatialViz>();
            ctx.get<ecs::SpatialViz>().kind = k;
            break;
        }
        case A::PoissonDialogToggle: {
            auto v = world_.view<ecs::PoissonDialog>();
            for (auto ent : v) { auto& pd = v.get<ecs::PoissonDialog>(ent); pd.visible = !pd.visible; break; }
            break;
        }

        case A::PipelineOcclusalWholePCA: {
            // Coarse whole-mesh PCA plane of the selected mesh, latched for viz.
            std::vector<Eigen::Vector3f> pts;
            std::vector<uint32_t> idx;
            float diag = 0.0f;
            if (!selectedWorldPoints(pts, idx, diag)) break;
            geometry::OcclusalPlane plane = geometry::wholeMeshPCA(pts);
            if (!plane.valid) break;

            auto& ctx = world_.ctx();
            if (!ctx.contains<ecs::OcclusalPlaneViz>()) ctx.emplace<ecs::OcclusalPlaneViz>();
            auto& viz = ctx.get<ecs::OcclusalPlaneViz>();
            viz.active   = true;
            viz.position = plane.position;
            viz.normal   = plane.normal;
            viz.size     = diag * 0.5f;
            break;
        }
        case A::PipelineOcclusalFindCusp: {
            // Detect cusp tips on the selected mesh (mesh-graph local maxima).
            // Cache the input so [ / ] can re-tune the prominence threshold live.
            std::vector<Eigen::Vector3f> pts;
            std::vector<uint32_t> idx;
            float diag = 0.0f;
            if (!selectedWorldPoints(pts, idx, diag)) break;

            auto& ctx = world_.ctx();
            if (!ctx.contains<ecs::CuspParams>()) ctx.emplace<ecs::CuspParams>();
            auto& cp = ctx.get<ecs::CuspParams>();
            cp.points  = std::move(pts);
            cp.indices = std::move(idx);
            cp.diag    = diag;
            cp.ready   = true;
            recomputeCusps();  // uses cp.prominence (persists across runs)
            break;
        }
        case A::PipelineOcclusalPlaneFromCusps:
            fitPlaneFromCusps();  // needs cusps from a prior Find Cusp
            break;
        case A::PipelineOcclusalEstimate: {
            // Full pipeline in one click: detect cusps on the selected mesh, then
            // fit the occlusal plane to them. Shows plane + cusp markers together.
            std::vector<Eigen::Vector3f> pts;
            std::vector<uint32_t> idx;
            float diag = 0.0f;
            if (!selectedWorldPoints(pts, idx, diag)) break;

            auto& ctx = world_.ctx();
            if (!ctx.contains<ecs::CuspParams>()) ctx.emplace<ecs::CuspParams>();
            auto& cp = ctx.get<ecs::CuspParams>();
            cp.points  = std::move(pts);
            cp.indices = std::move(idx);
            cp.diag    = diag;
            cp.ready   = true;
            recomputeCusps();     // -> CuspViz (markers + outliers)
            fitPlaneFromCusps();  // -> OcclusalPlaneViz (refined plane)
            break;
        }
        case A::PipelineOcclusal2DRender:
            // Stage 2: depth / normal / curvature panels (needs a plane first).
            showChannels(3);
            break;
        case A::PipelineSegment2DClassical:
            // Stage 3a: + the classical watershed segmentation panel.
            showChannels(4);
            break;
        case A::PipelineSegment2DSAM: {
            // Stage 3b: on-device MobileSAM. Encode the occlusal normal image, run
            // the mask decoder per cusp tip (prompt), merge into per-tooth masks.
            auto setStatus = [&](const std::string& s) {
                auto mv = world_.view<ecs::MenuBar>();
                for (auto e : mv) { mv.get<ecs::MenuBar>(e).statusText = s; break; }
            };
            if (!orange::ml::samAvailable()) { setStatus("SAM: ONNX runtime not built"); break; }

            auto& ctx = world_.ctx();
            if (!ctx.contains<ecs::OcclusalPlaneViz>() || !ctx.contains<ecs::CuspParams>() ||
                !ctx.contains<ecs::CuspViz>() || !r) { setStatus("SAM: run Estimate first"); break; }
            auto& pv  = ctx.get<ecs::OcclusalPlaneViz>();
            auto& cp  = ctx.get<ecs::CuspParams>();
            auto& cvz = ctx.get<ecs::CuspViz>();
            if (!cp.ready || cp.points.empty() || cvz.points.empty()) { setStatus("SAM: run Estimate first"); break; }

            geometry::OcclusalChannels ch =
                geometry::renderOcclusalChannels(cp.points, pv.position, pv.normal, 256);
            if (!ch.valid) break;

            // SAM input image = the normal channel (RGB), prompts = non-outlier cusps.
            std::vector<uint8_t> rgb((size_t)ch.width * ch.height * 3);
            for (size_t i = 0; i < (size_t)ch.width * ch.height; ++i) {
                rgb[i * 3 + 0] = ch.normal[i * 4 + 0];
                rgb[i * 3 + 1] = ch.normal[i * 4 + 1];
                rgb[i * 3 + 2] = ch.normal[i * 4 + 2];
            }
            std::vector<std::pair<float, float>> pts;
            for (size_t j = 0; j < cvz.points.size(); ++j) {
                if (j < cvz.outlier.size() && cvz.outlier[j]) continue;
                Eigen::Vector3f d = cvz.points[j] - ch.origin;
                float px = (d.dot(ch.u) - ch.aOff) / ch.cell;
                float row = (float)(ch.height - 1) - (d.dot(ch.v) - ch.bOff) / ch.cell;
                if (px >= 0 && px < ch.width && row >= 0 && row < ch.height)
                    pts.emplace_back(px, row);
            }

            orange::ml::SamResult sam = orange::ml::samSegment(rgb, ch.width, ch.height, pts);

            auto rv = world_.view<ecs::OcclusalRenderViz>();
            for (auto e : rv) {
                auto& orv = rv.get<ecs::OcclusalRenderViz>(e);
                for (int c = 0; c < 4; ++c)
                    if (orv.tex[c] != render::kInvalidTexture) {
                        r->destroyTexture(orv.tex[c]); orv.tex[c] = render::kInvalidTexture;
                    }
                auto mk = [&](const std::vector<uint8_t>& pxs) {
                    render::TextureDesc td;
                    td.width = ch.width; td.height = ch.height; td.pixels = pxs.data();
                    return r->createTexture(td);
                };
                orv.tex[0] = mk(ch.depth);
                orv.tex[1] = mk(ch.normal);
                orv.tex[2] = mk(ch.curvature);
                orv.tex[3] = mk(sam.valid ? sam.rgba : ch.segment);
                orv.count   = 4;
                orv.visible = true;
                break;
            }
            setStatus(sam.valid ? ("SAM: " + std::to_string(sam.regions) + " teeth")
                                : "SAM: inference failed");
            break;
        }
        case A::PipelineSegment3D: {
            // Stage 5: per-tooth watershed on the mesh itself; spawn a colour-coded
            // copy and hide the source. Runs on the selected (or only) mesh.
            if (!r) break;
            entt::entity src = entt::null, only = entt::null;
            size_t cnt = 0;
            auto v = world_.view<ecs::Renderable, ecs::PickGeometry, ecs::Transform>();
            for (auto ent : v) { ++cnt; only = ent; if (v.get<ecs::Renderable>(ent).selected) { src = ent; break; } }
            if (src == entt::null && cnt == 1) src = only;
            if (src == entt::null) break;

            const auto& pick = world_.get<ecs::PickGeometry>(src);
            const Eigen::Matrix4f M = world_.get<ecs::Transform>(src).matrix();
            std::vector<Eigen::Vector3f> wv;
            wv.reserve(pick.positions.size());
            for (const auto& p : pick.positions) {
                Eigen::Vector4f w = M * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
                wv.emplace_back(w.x(), w.y(), w.z());
            }
            geometry::ToothSegResult seg = geometry::segmentTeeth(wv, pick.indices);
            if (!seg.valid) break;

            // Clean up a previous run + restore the previously-hidden source.
            if (segEntity_ != entt::null && world_.valid(segEntity_)) world_.destroy(segEntity_);
            if (segMesh_ != render::kInvalidMesh) r->destroyMesh(segMesh_);
            if (segVbo_ != render::kInvalidBuffer) r->destroyBuffer(segVbo_);
            if (segSource_ != entt::null && world_.valid(segSource_) && world_.all_of<ecs::Renderable>(segSource_))
                world_.get<ecs::Renderable>(segSource_).visible = true;

            auto toothColor = [](int lbl) -> Eigen::Vector3f {
                if (lbl < 0) return Eigen::Vector3f(0.55f, 0.55f, 0.58f);  // gingiva/base: grey
                uint32_t h = (uint32_t)(lbl + 1) * 2654435761u;
                float hue = (float)(h & 0xFFFF) / 65535.0f * 6.0f;
                float x = 1.0f - std::fabs(std::fmod(hue, 2.0f) - 1.0f);
                Eigen::Vector3f c;
                if (hue < 1) c = {1, x, 0}; else if (hue < 2) c = {x, 1, 0};
                else if (hue < 3) c = {0, 1, x}; else if (hue < 4) c = {0, x, 1};
                else if (hue < 5) c = {x, 0, 1}; else c = {1, 0, x};
                return c * 0.65f + Eigen::Vector3f(0.18f, 0.18f, 0.18f);
            };

            // Colour-coded triangle soup (world coords; spawned at identity).
            std::vector<render::Vertex> verts;
            verts.reserve(pick.indices.size());
            Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX), mx = Eigen::Vector3f::Constant(-FLT_MAX);
            for (uint32_t ii : pick.indices) {
                if (ii >= wv.size()) continue;
                const Eigen::Vector3f& p = wv[ii];
                Eigen::Vector3f col = toothColor(ii < seg.label.size() ? seg.label[ii] : -1);
                render::Vertex vv{};
                vv.position[0] = p.x(); vv.position[1] = p.y(); vv.position[2] = p.z();
                vv.color[0] = col.x(); vv.color[1] = col.y(); vv.color[2] = col.z();
                verts.push_back(vv);
                mn = mn.cwiseMin(p); mx = mx.cwiseMax(p);
            }

            render::BufferDesc bd;
            bd.type = render::BufferType::Vertex; bd.usage = render::BufferUsage::Static;
            bd.data = verts.data(); bd.size = verts.size() * sizeof(render::Vertex);
            segVbo_ = r->createBuffer(bd);
            render::MeshDesc md;
            md.vertexBuffer = segVbo_; md.indexBuffer = render::kInvalidBuffer;
            md.layout = render::Vertex::layout(); md.vertexCount = (uint32_t)verts.size();
            segMesh_ = r->createMesh(md);

            auto e = world_.create();
            world_.emplace<ecs::Transform>(e, ecs::Transform{});
            ecs::Renderable rr;
            rr.mesh = segMesh_; rr.boundsMin = mn; rr.boundsMax = mx;
            rr.colorMode = 0;  // show the per-tooth vertex colours, not a height map
            world_.emplace<ecs::Renderable>(e, rr);

            world_.get<ecs::Renderable>(src).visible = false;  // hide the original
            segEntity_ = e; segSource_ = src;

            auto mvb = world_.view<ecs::MenuBar>();
            for (auto me : mvb) { mvb.get<ecs::MenuBar>(me).statusText =
                "3D segmentation: " + std::to_string(seg.regions) + " teeth"; break; }
            break;
        }
        case A::None:  break;
    }
}

void Application::recomputeCusps() {
    auto& ctx = world_.ctx();
    if (!ctx.contains<ecs::CuspParams>()) return;
    auto& cp = ctx.get<ecs::CuspParams>();
    if (!cp.ready) return;

    geometry::OcclusalPlaneParams params;
    params.cuspHeightGate = cp.heightGate;
    std::vector<Eigen::Vector3f> cusps = geometry::findCusps(cp.points, cp.indices, params);

    // Flag tips whose HEIGHT along the occlusal axis deviates far from the mean
    // tip height (drawn red) -- only the axis projection matters, not distance in
    // the occlusal plane (the arch spreads tips out in-plane on purpose).
    std::vector<uint8_t> outlier(cusps.size(), 0);
    if (cusps.size() >= 3) {
        geometry::OcclusalPlane plane = geometry::wholeMeshPCA(cp.points);
        const Eigen::Vector3f n = plane.normal, o = plane.position;
        std::vector<float> h(cusps.size());
        double sum = 0.0, sum2 = 0.0;
        for (size_t i = 0; i < cusps.size(); ++i) {
            h[i] = (cusps[i] - o).dot(n);
            sum += h[i]; sum2 += (double)h[i] * h[i];
        }
        double mean = sum / cusps.size();
        double sd = std::sqrt(std::max(0.0, sum2 / cusps.size() - mean * mean));
        for (size_t i = 0; i < cusps.size(); ++i)
            outlier[i] = std::abs((double)h[i] - mean) > 2.0 * sd ? 1 : 0;
    }

    if (!ctx.contains<ecs::CuspViz>()) ctx.emplace<ecs::CuspViz>();
    auto& cv = ctx.get<ecs::CuspViz>();
    cv.active  = true;
    cv.points  = std::move(cusps);
    cv.outlier = std::move(outlier);
    cv.size    = cp.diag;
}

bool Application::fitPlaneFromCusps() {
    // Fit the occlusal plane to the (non-outlier) cusp tips -- the design doc's
    // refined estimate. Keeps the cusp markers visible alongside the plane.
    auto& ctx = world_.ctx();
    if (!ctx.contains<ecs::CuspViz>()) return false;
    auto& cv = ctx.get<ecs::CuspViz>();
    if (cv.points.empty()) return false;  // run Find Cusp first

    std::vector<Eigen::Vector3f> tips;
    tips.reserve(cv.points.size());
    for (size_t i = 0; i < cv.points.size(); ++i)
        if (i >= cv.outlier.size() || !cv.outlier[i]) tips.push_back(cv.points[i]);
    if (tips.size() < 3) return false;

    geometry::OcclusalPlane plane = geometry::wholeMeshPCA(tips);
    if (!plane.valid) return false;

    if (!ctx.contains<ecs::OcclusalPlaneViz>()) ctx.emplace<ecs::OcclusalPlaneViz>();
    auto& pv = ctx.get<ecs::OcclusalPlaneViz>();
    pv.active   = true;
    pv.position = plane.position;
    pv.normal   = plane.normal;
    pv.size     = (cv.size > 0.0f ? cv.size : 1.0f) * 0.5f;
    cv.active   = true;  // keep the cusp markers on screen with the plane
    return true;
}

void Application::syncMenu() {
    using A = ecs::MenuAction;
    ecs::MenuBar* mb = nullptr;
    auto mv = world_.view<ecs::MenuBar>();
    for (auto e : mv) { mb = &mv.get<ecs::MenuBar>(e); break; }
    if (!mb) return;

    bool gridOn = true;
    if (world_.ctx().contains<ecs::GridState>()) gridOn = world_.ctx().get<ecs::GridState>().visible;
    bool csOn = false;
    { auto v = world_.view<ecs::CrossSection>();
      for (auto e : v) { csOn = v.get<ecs::CrossSection>(e).enabled; break; } }
    int viz = 0;
    if (world_.ctx().contains<ecs::SpatialViz>()) viz = world_.ctx().get<ecs::SpatialViz>().kind;
    // Color-menu ticks reflect the selected mesh (or the only mesh).
    uint32_t colorSel = colorMode_;
    { auto v = world_.view<ecs::Renderable>();
      size_t count = 0; for (auto e : v) { (void)e; ++count; }
      for (auto e : v) { auto& rr = v.get<ecs::Renderable>(e);
                         if (rr.selected || count == 1) { colorSel = rr.colorMode; break; } } }

    for (auto& menu : mb->menus)
        for (auto& it : menu.items) {
            switch (it.action) {
                case A::ToggleGrid:         it.checked = gridOn;          break;
                case A::ToggleLighting:     it.checked = lighting_;       break;
                case A::ToggleVsync:        it.checked = vsync_;          break;
                case A::ToggleCrossSection: it.checked = csOn;            break;
                case A::ColorOriginal:      it.checked = colorSel == 0;   break;
                case A::ColorHeight:        it.checked = colorSel == 1;   break;
                case A::ColorPosition:      it.checked = colorSel == 2;   break;
                case A::ColorGray:          it.checked = colorSel == 3;   break;
                case A::SpatialNone:        it.checked = viz == 0;        break;
                case A::SpatialBVH:         it.checked = viz == 1;        break;
                case A::SpatialOctree:      it.checked = viz == 2;        break;
                case A::SpatialKDTree:      it.checked = viz == 3;        break;
                case A::SpatialGrid:        it.checked = viz == 4;        break;
                case A::SpatialLoose:       it.checked = viz == 5;        break;
                case A::SpatialBSP:         it.checked = viz == 6;        break;
                case A::SpatialRTree:       it.checked = viz == 7;        break;
                case A::SpatialBall:        it.checked = viz == 8;        break;
                default: break;
            }
        }
}

Application::~Application() {
    if (plugin_ && plugin_->renderer()) plugin_->renderer()->shutdown();
    plugin_.reset();
    window_.destroy();
    SDL_Quit();
}

} // namespace orange::core
