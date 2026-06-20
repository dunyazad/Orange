// Headless behavior tests for the GUI *logic* -- the ECS input systems that back
// the selection toolbar and picking. These run with no window and no renderer:
// the systems operate purely on an entt::registry + a synthetic core::Input, so
// the selection behavior (modes, filters, modifiers, box-select, element picking)
// is fully automatable. Pixel rendering and the font-measured menu bar still need
// the manual checklist in docs/TESTING.md.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <Eigen/Core>
#include <entt/entt.hpp>

#include "orange/core/input.h"
#include "orange/ecs/components.h"
#include "orange/ecs/systems.h"

using namespace orange;
using Eigen::Vector3f;

static int g_total = 0, g_fail = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        ++g_total;                                                                 \
        if (!(cond)) { ++g_fail; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

// Build a scene: a primary camera at (0,0,5) looking down -Z, and a unit quad in
// the z=0 plane with a center vertex (index 4) for element picking. Returns the
// quad entity.
static entt::entity makeScene(entt::registry& w) {
    auto cam = w.create();
    ecs::Transform ct;
    ct.position = Vector3f(0, 0, 5);          // identity orientation => forward -Z
    w.emplace<ecs::Transform>(cam, ct);
    ecs::Camera c; c.primary = true; c.mode = ecs::ProjectionMode::Perspective;
    c.fovYDegrees = 60.0f; c.zNear = 0.1f; c.zFar = 100.0f;
    w.emplace<ecs::Camera>(cam, c);
    ecs::CameraManipulator m; m.distance = 5.0f;
    w.emplace<ecs::CameraManipulator>(cam, m);

    auto q = w.create();
    w.emplace<ecs::Transform>(q, ecs::Transform{});
    ecs::Renderable r;
    r.mesh = 1;  // any non-invalid handle so the entity counts as drawable/pickable
    r.boundsMin = Vector3f(-1, -1, -0.01f);
    r.boundsMax = Vector3f(1, 1, 0.01f);
    r.pointCloud = false;
    w.emplace<ecs::Renderable>(q, r);
    ecs::PickGeometry pg;
    pg.positions = {Vector3f(-1, -1, 0), Vector3f(1, -1, 0), Vector3f(1, 1, 0),
                    Vector3f(-1, 1, 0), Vector3f(0, 0, 0)};  // index 4 = center
    pg.indices = {0, 1, 2, 0, 2, 3};
    w.emplace<ecs::PickGeometry>(q, std::move(pg));
    return q;
}

static core::Input clickAt(float x, float y) {
    core::Input in;
    in.mousePosX = x; in.mousePosY = y;
    in.leftClicked = true; in.buttonLeft = true;
    return in;
}

// --- selection toolbar: clicking a button sets the matching mode -------------
// Layout constants MUST mirror systems.cpp (kTb*). Buttons:
//  0..3 target, 4..7 action, 8..10 filter, 11..13 modifier.
static void test_toolbar() {
    std::printf("[toolbar]\n");
    const float X = 8.0f, BTN = 46.0f, TOP = 46.0f + 10.0f, GAP = 4.0f, GGAP = 14.0f;
    auto buttons = ecs::defaultSelectionToolbar();
    auto centerY = [&](int i) {
        float y = TOP;
        for (int k = 0; k < i; ++k)
            y += BTN + (buttons[k].group != buttons[k + 1].group ? GGAP : GAP);
        return y + BTN * 0.5f;
    };
    const float cx = X + BTN * 0.5f;

    entt::registry w;
    auto e = w.create();
    ecs::SelectionToolbar tb; tb.buttons = buttons;
    w.emplace<ecs::SelectionToolbar>(e, tb);
    w.ctx().emplace<ecs::SelectionMode>();

    auto click = [&](int i) {
        core::Input in = clickAt(cx, centerY(i));
        ecs::selectionToolbarInputSystem(w, in, 200, 800);
    };
    click(1);  // Vtx
    CHECK(w.ctx().get<ecs::SelectionMode>().target == ecs::SelTarget::Vertex);
    click(5);  // Box
    CHECK(w.ctx().get<ecs::SelectionMode>().action == ecs::SelAction::Box);
    click(10); // Pts filter
    CHECK(w.ctx().get<ecs::SelectionMode>().filter == ecs::SelFilter::Point);
    click(12); // + (Add)
    CHECK(w.ctx().get<ecs::SelectionMode>().modifier == ecs::SelModifier::Add);
    // A click that lands on the bar captures the mouse (so the scene won't pick).
    core::Input in = clickAt(cx, centerY(0));
    ecs::selectionToolbarInputSystem(w, in, 200, 800);
    CHECK(in.captured);
}

// --- picking: object single-select / clear ----------------------------------
static void test_pick_object() {
    std::printf("[pick.object]\n");
    entt::registry w;
    auto q = makeScene(w);
    w.ctx().emplace<ecs::SelectionMode>();  // defaults: Object/Single/All/Replace

    core::Input in = clickAt(100, 100);     // center of a 200x200 viewport
    ecs::pickingSystem(w, in, 200, 200);
    CHECK(w.get<ecs::Renderable>(q).selected);

    core::Input miss = clickAt(5, 5);       // ray misses the quad -> clears
    ecs::pickingSystem(w, miss, 200, 200);
    CHECK(!w.get<ecs::Renderable>(q).selected);
}

// --- picking: filter excludes non-matching types ----------------------------
static void test_pick_filter() {
    std::printf("[pick.filter]\n");
    entt::registry w;
    auto q = makeScene(w);
    auto& sm = w.ctx().emplace<ecs::SelectionMode>();
    sm.filter = ecs::SelFilter::Point;      // mesh quad is ineligible

    core::Input in = clickAt(100, 100);
    ecs::pickingSystem(w, in, 200, 200);
    CHECK(!w.get<ecs::Renderable>(q).selected);

    sm.filter = ecs::SelFilter::Mesh;       // now eligible
    core::Input in2 = clickAt(100, 100);
    ecs::pickingSystem(w, in2, 200, 200);
    CHECK(w.get<ecs::Renderable>(q).selected);
}

// --- picking: Add / Subtract modifiers --------------------------------------
static void test_pick_modifier() {
    std::printf("[pick.modifier]\n");
    entt::registry w;
    auto q = makeScene(w);
    auto& sm = w.ctx().emplace<ecs::SelectionMode>();

    core::Input in = clickAt(100, 100);     // Replace: select
    ecs::pickingSystem(w, in, 200, 200);
    CHECK(w.get<ecs::Renderable>(q).selected);

    sm.modifier = ecs::SelModifier::Add;    // empty Add keeps the selection
    core::Input miss = clickAt(5, 5);
    ecs::pickingSystem(w, miss, 200, 200);
    CHECK(w.get<ecs::Renderable>(q).selected);

    sm.modifier = ecs::SelModifier::Subtract;  // subtract the hit
    core::Input in2 = clickAt(100, 100);
    ecs::pickingSystem(w, in2, 200, 200);
    CHECK(!w.get<ecs::Renderable>(q).selected);
}

// --- picking: box-select an object over a drag rectangle --------------------
static void test_pick_box() {
    std::printf("[pick.box]\n");
    entt::registry w;
    auto q = makeScene(w);
    auto& sm = w.ctx().emplace<ecs::SelectionMode>();
    sm.action = ecs::SelAction::Box;

    // Drag from (50,50) to (150,150); the object center projects to (100,100).
    core::Input down; down.mousePosX = 50; down.mousePosY = 50;
    down.leftClicked = true; down.buttonLeft = true;
    ecs::pickingSystem(w, down, 200, 200);

    core::Input drag; drag.mousePosX = 150; drag.mousePosY = 150; drag.buttonLeft = true;
    ecs::pickingSystem(w, drag, 200, 200);

    core::Input up; up.mousePosX = 150; up.mousePosY = 150; up.leftReleased = true;
    ecs::pickingSystem(w, up, 200, 200);

    CHECK(w.get<ecs::Renderable>(q).selected);
}

// --- picking: element (vertex) single-select --------------------------------
static void test_pick_vertex() {
    std::printf("[pick.vertex]\n");
    entt::registry w;
    auto q = makeScene(w);
    auto& sm = w.ctx().emplace<ecs::SelectionMode>();
    sm.target = ecs::SelTarget::Vertex;

    core::Input in = clickAt(100, 100);  // center -> nearest vertex is index 4 (0,0,0)
    ecs::pickingSystem(w, in, 200, 200);
    CHECK(w.all_of<ecs::ElementSelection>(q));
    if (w.all_of<ecs::ElementSelection>(q)) {
        const auto& es = w.get<ecs::ElementSelection>(q);
        CHECK(es.vertices.size() == 1);
        if (!es.vertices.empty()) CHECK(es.vertices[0] == 4u);
    }
}

// --- on-screen visibility test (used by Ctrl+A) -----------------------------
static void test_visibility() {
    std::printf("[visibility]\n");
    entt::registry w;
    auto q = makeScene(w);
    CHECK(ecs::entityVisibleOnScreen(w, q, 200, 200));

    auto off = w.create();
    ecs::Transform t; t.position = Vector3f(1000, 0, 0);
    w.emplace<ecs::Transform>(off, t);
    ecs::Renderable r;
    r.mesh = 1;
    r.boundsMin = Vector3f(-0.5f, -0.5f, -0.5f);
    r.boundsMax = Vector3f(0.5f, 0.5f, 0.5f);
    w.emplace<ecs::Renderable>(off, r);
    CHECK(!ecs::entityVisibleOnScreen(w, off, 200, 200));
}

int main() {
    std::printf("Orange GUI behavior tests\n");
    test_toolbar();
    test_pick_object();
    test_pick_filter();
    test_pick_modifier();
    test_pick_box();
    test_pick_vertex();
    test_visibility();
    std::printf("\n%d checks, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
