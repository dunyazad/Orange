#include "orange/core/application.h"

#include <SDL3/SDL.h>

#include <cfloat>
#include <vector>

#include "orange/core/draw_mode.h"
#include "orange/core/modes.h"
#include "orange/core/primitives.h"
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
                        if (e.key.mod & SDL_KMOD_SHIFT) {  // Shift+` cycles scene coloring
                            colorMode_ = (colorMode_ + 1) % 4;  // default/height/position/gray
                            plugin_->renderer()->setColorMode(colorMode_);
                            static const char* kColorNames[4] = {"default", "height",
                                                                 "position", "grayscale"};
                            SDL_Log("Application: color mode = %s", kColorNames[colorMode_]);
                        } else {                           // ` toggles point-sprite lighting
                            lighting_ = !lighting_;
                            plugin_->renderer()->setLighting(lighting_);
                            SDL_Log("Application: lighting = %s", lighting_ ? "on" : "off");
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
    auto setColor = [&](uint32_t mode) {
        colorMode_ = mode;
        if (r) r->setColorMode(colorMode_);
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
        case A::None:  break;
    }
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

    for (auto& menu : mb->menus)
        for (auto& it : menu.items) {
            switch (it.action) {
                case A::ToggleGrid:         it.checked = gridOn;          break;
                case A::ToggleLighting:     it.checked = lighting_;       break;
                case A::ToggleVsync:        it.checked = vsync_;          break;
                case A::ToggleCrossSection: it.checked = csOn;            break;
                case A::ColorOriginal:      it.checked = colorMode_ == 0; break;
                case A::ColorHeight:        it.checked = colorMode_ == 1; break;
                case A::ColorPosition:      it.checked = colorMode_ == 2; break;
                case A::ColorGray:          it.checked = colorMode_ == 3; break;
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
