// Regression guard for the pick-BVH dangling crash: a cached PickBVH must keep
// working after other PickGeometry entities are destroyed (EnTT swap-and-pop
// relocates the pools). Also smoke-tests the Poisson reconstruction pipeline
// (cloud -> normals -> mesh -> sequential indices -> BVH -> pick) end to end.
#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Core>
#include <entt/entt.hpp>

#include "orange/core/bvh.h"
#include "orange/core/geometry.h"
#include "orange/core/input.h"
#include "orange/core/normals.h"
#include "orange/core/poisson_reconstruction.h"
#include "orange/ecs/components.h"
#include "orange/ecs/systems.h"

using namespace orange;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); } } while (0)

static void addMesh(entt::registry& w, entt::entity e,
                    const std::vector<Eigen::Vector3f>& pos, const std::vector<uint32_t>& idx) {
    w.emplace<ecs::Transform>(e, ecs::Transform{});
    ecs::Renderable r; r.mesh = 1;
    r.boundsMin = Eigen::Vector3f(-1.2f, -1.2f, -1.2f);
    r.boundsMax = Eigen::Vector3f(1.2f, 1.2f, 1.2f);
    w.emplace<ecs::Renderable>(e, r);
    ecs::PickGeometry pg; pg.positions = pos; pg.indices = idx;
    w.emplace<ecs::PickGeometry>(e, std::move(pg));
}

int main() {
    // Sphere cloud -> Poisson mesh.
    std::vector<Eigen::Vector3f> pts;
    const int U = 60, V = 60;
    for (int i = 0; i < U; ++i)
        for (int j = 0; j < V; ++j) {
            float a = (float)i / U * 6.2831853f, b = (float)j / (V - 1) * 3.14159265f;
            pts.emplace_back(std::sin(b) * std::cos(a), std::cos(b), std::sin(b) * std::sin(a));
        }
    auto nrm = geometry::estimateNormals(pts, 16);
    geometry::PoissonParams pp;  // defaults
    auto tris = geometry::poissonReconstruct(pts, nrm, pp);
    std::printf("poisson tris: %zu\n", tris.size());
    CHECK(!tris.empty());

    std::vector<Eigen::Vector3f> positions; std::vector<uint32_t> indices;
    uint32_t c = 0;
    double rsum = 0; size_t rn = 0;
    for (const auto& t : tris)
        for (int k = 0; k < 3; ++k) {
            CHECK(t.v[k].allFinite());
            positions.push_back(t.v[k]); indices.push_back(c++);
            rsum += t.v[k].norm(); ++rn;
        }
    // Unit-sphere input: a correct reconstruction hugs radius ~1. A blobby/unconverged
    // solve instead fills the padded bounding box (mean radius >> 1).
    double meanR = rn ? rsum / rn : 0.0;
    std::printf("mean vertex radius: %.3f (expect ~1.0)\n", meanR);
    CHECK(meanR > 0.75 && meanR < 1.25);

    // Scene: camera + the Poisson mesh, plus extra mesh entities sharing geometry.
    entt::registry w;
    auto cam = w.create();
    ecs::Transform ct; ct.position = Eigen::Vector3f(0, 0, 4);
    w.emplace<ecs::Transform>(cam, ct);
    ecs::Camera cc; cc.primary = true; cc.mode = ecs::ProjectionMode::Perspective;
    cc.fovYDegrees = 60.0f; cc.zNear = 0.1f; cc.zFar = 100.0f;
    w.emplace<ecs::Camera>(cam, cc);
    ecs::CameraManipulator mm; mm.distance = 4.0f;
    w.emplace<ecs::CameraManipulator>(cam, mm);

    auto e = w.create();
    addMesh(w, e, positions, indices);
    std::vector<entt::entity> extra;
    for (int n = 0; n < 12; ++n) { auto x = w.create(); addMesh(w, x, positions, indices); extra.push_back(x); }

    auto click = [&]() {
        core::Input in; in.mousePosX = 400; in.mousePosY = 300;
        in.leftClicked = true; in.buttonLeft = true;
        ecs::pickingSystem(w, in, 800, 600);  // builds + queries the cached BVHs
    };

    // Build every entity's cached BVH (no crash on first build).
    for (int f = 0; f < 3; ++f) click();

    // Destroy half the extra meshes: EnTT relocates the PickGeometry/PickBVH pools.
    // A BVH that borrowed PickGeometry spans would now dangle; the owned-heap copy
    // keeps it valid. The crash this guards against happened HERE on the next pick.
    for (size_t n = 0; n < extra.size(); n += 2) w.destroy(extra[n]);
    for (int f = 0; f < 5; ++f) click();  // must not crash

    std::printf(g_fail ? "FAILED (%d)\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
