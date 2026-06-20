// Unit tests for Orange's CPU-side toolkit (orange_core): geometry primitives,
// Morton keys, color utilities, the sparse grid, the processing modes, and mesh
// IO. These are the headless, deterministic parts of the engine -- the GUI
// (rendering, menu, selection toolbar, picking) is covered by the manual
// checklist in docs/TESTING.md.
//
// No external test framework (deps are network-fetched); a tiny assert harness
// keeps this buildable offline. Run via CTest (see docs/TESTING.md).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "orange/core/color.h"
#include "orange/core/debug_draw.h"
#include "orange/core/geometry.h"
#include "orange/core/modes.h"
#include "orange/core/morton3d.h"
#include "orange/core/serialization.h"
#include "orange/core/sparse_grid.h"

// --- tiny test harness ------------------------------------------------------
static int g_total = 0;
static int g_fail  = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        ++g_total;                                                                 \
        if (!(cond)) { ++g_fail; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                      \
    do {                                                                           \
        ++g_total;                                                                 \
        double da = (a), db = (b);                                                 \
        if (std::fabs(da - db) > (eps)) {                                          \
            ++g_fail;                                                              \
            std::printf("  FAIL %s:%d  |%g - %g| > %g\n", __FILE__, __LINE__, da, db, (double)(eps)); \
        }                                                                          \
    } while (0)

using Eigen::Vector3f;

// --- geometry: Ray / AABB ---------------------------------------------------
static void test_geometry() {
    std::printf("[geometry]\n");
    // Ray down -Z hits a triangle in the z=-2 plane straddling the origin.
    orange::geometry::Ray ray(Vector3f(0, 0, 0), Vector3f(0, 0, -1));
    float t = 0.0f;
    CHECK(ray.intersectTriangle(Vector3f(-1, -1, -2), Vector3f(1, -1, -2),
                                Vector3f(0, 2, -2), t));
    CHECK_NEAR(t, 2.0f, 1e-4);
    // A triangle behind the ray is not hit.
    CHECK(!ray.intersectTriangle(Vector3f(-1, -1, 2), Vector3f(1, -1, 2),
                                 Vector3f(0, 2, 2), t));
    // Sphere in front is hit at the near surface.
    CHECK(ray.intersectSphere(Vector3f(0, 0, -5), 1.0f, t));
    CHECK_NEAR(t, 4.0f, 1e-4);

    orange::geometry::AABB box;
    box.expand(Vector3f(-1, -1, -1));
    box.expand(Vector3f(1, 1, 1));
    CHECK(box.contains(Vector3f(0, 0, 0)));
    CHECK(!box.contains(Vector3f(2, 0, 0)));
    float tn = 0, tf = 0;
    CHECK(box.intersectRay(orange::geometry::Ray(Vector3f(0, 0, 5), Vector3f(0, 0, -1)), tn, tf));
    orange::geometry::AABB other;
    other.expand(Vector3f(0.5f, 0.5f, 0.5f));
    other.expand(Vector3f(3, 3, 3));
    CHECK(box.intersects(other));
    orange::geometry::AABB far;
    far.expand(Vector3f(5, 5, 5));
    far.expand(Vector3f(6, 6, 6));
    CHECK(!box.intersects(far));
}

// --- Morton3D (Z-order keys) ------------------------------------------------
static void test_morton() {
    std::printf("[morton]\n");
    using M = orange::geometry::Morton3D;
    for (uint32_t x : {0u, 1u, 7u, 1023u, 1048575u}) {
        uint32_t dx, dy, dz;
        M::decode(M::encode(x, 0, 0), dx, dy, dz);
        CHECK(dx == x && dy == 0 && dz == 0);
    }
    uint32_t ax, ay, az;
    M::decode(M::encode(13, 42, 7), ax, ay, az);
    CHECK(ax == 13 && ay == 42 && az == 7);
    // Position -> voxel index (floor) about a grid origin.
    auto idx = M::positionToIndex(Vector3f(2.5f, 0.0f, 9.9f), Vector3f(0, 0, 0), 1.0f);
    CHECK(idx.x() == 2 && idx.y() == 0 && idx.z() == 9);
}

// --- color utilities --------------------------------------------------------
static void test_color() {
    std::printf("[color]\n");
    auto red = orange::color::FromHSV(0.0f, 1.0f, 1.0f);
    CHECK_NEAR(red.x(), 1.0f, 1e-3); CHECK_NEAR(red.y(), 0.0f, 1e-3); CHECK_NEAR(red.z(), 0.0f, 1e-3);
    auto gray = orange::color::FromHSV(0.3f, 0.0f, 0.5f);  // zero saturation -> gray = v
    CHECK_NEAR(gray.x(), 0.5f, 1e-3); CHECK_NEAR(gray.y(), 0.5f, 1e-3); CHECK_NEAR(gray.z(), 0.5f, 1e-3);

    auto lo = orange::color::GetHeatMapColor(0.0f, 0.0f, 1.0f);   // blue end
    CHECK_NEAR(lo.z(), 1.0f, 1e-3);
    auto hi = orange::color::GetHeatMapColor(1.0f, 0.0f, 1.0f);   // red end
    CHECK_NEAR(hi.x(), 1.0f, 1e-3);

    auto mid = orange::color::Lerp(orange::color::black(), orange::color::white(), 0.5f);
    CHECK_NEAR(mid.x(), 0.5f, 1e-3);

    auto pal = orange::color::GetContrastingColors(8);
    CHECK(pal.size() == 8);
    // Deterministic golden-ratio index colors (no RNG).
    CHECK((orange::color::RandomFromIndex(5) - orange::color::RandomFromIndex(5)).norm() < 1e-6f);
}

// --- sparse grid (radius / kNN / closest) -----------------------------------
static void test_sparse_grid() {
    std::printf("[sparse_grid]\n");
    std::vector<Vector3f> pts;
    for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 5; ++y)
            for (int z = 0; z < 5; ++z)
                pts.emplace_back((float)x, (float)y, (float)z);  // 125-point lattice, spacing 1

    orange::geometry::SparseGrid grid;
    grid.build(pts, 1.0f);

    std::vector<unsigned int> idx;
    std::vector<float> dist;
    // radius 1.1 around an interior point -> itself + 6 face neighbors = 7.
    grid.pointsWithinRadius(pts, Vector3f(2, 2, 2), 1.1f, idx, dist);
    CHECK(idx.size() == 7);

    float d = 0.0f;
    int c = grid.closestPoint(pts, Vector3f(2.1f, 2.0f, 2.0f), d);
    CHECK(c >= 0);
    CHECK(pts[c].isApprox(Vector3f(2, 2, 2)));

    std::vector<unsigned int> kIdx; std::vector<float> kDist;
    grid.kNearestNeighbors(pts, Vector3f(0, 0, 0), 3, kIdx, kDist);
    CHECK(kIdx.size() == 3);
    CHECK(kDist.front() <= kDist.back());  // sorted nearest-first
}

// --- processing modes -------------------------------------------------------
static void test_modes() {
    std::printf("[modes]\n");
    CHECK(orange::modes::modeCount() >= 4);
    for (int i = 0; i < orange::modes::modeCount(); ++i)
        CHECK(orange::modes::modeName(i) != nullptr);

    // A small solid-ish cloud so the clustering/morphology modes have something
    // to emit. Each mode must run without crashing; at least one emits geometry.
    orange::modes::ModeInput in;
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            for (int z = 0; z < 8; ++z) {
                in.points.emplace_back((float)x, (float)y, (float)z);
                in.normals.emplace_back(0, 1, 0);
            }

    auto& dd = orange::debug::DebugDraw::instance();
    bool anyEmitted = false;
    for (int i = 0; i < orange::modes::modeCount(); ++i) {
        dd.clear();
        orange::modes::runMode(i, in, dd);
        if (!dd.empty()) anyEmitted = true;
    }
    dd.clear();
    CHECK(anyEmitted);
}

// --- mesh IO roundtrip ------------------------------------------------------
static void test_io() {
    std::printf("[io]\n");
    const std::string path = "orange_test_tmp.xyz";
    {
        orange::io::XYZFormat out;
        out.AddPoint(1.0f, 2.0f, 3.0f);
        out.AddPoint(-4.0f, 5.5f, 6.25f);
        CHECK(out.Serialize(path));
    }
    {
        orange::io::XYZFormat in;
        CHECK(in.Deserialize(path));
        const auto& p = in.GetPoints();
        CHECK(p.size() == 2);
        if (p.size() == 2) {
            CHECK(p[0].isApprox(Vector3f(1.0f, 2.0f, 3.0f), 1e-4f));
            CHECK(p[1].isApprox(Vector3f(-4.0f, 5.5f, 6.25f), 1e-4f));
        }
    }
    std::remove(path.c_str());
}

int main() {
    std::printf("Orange unit tests\n");
    test_geometry();
    test_morton();
    test_color();
    test_sparse_grid();
    test_modes();
    test_io();
    std::printf("\n%d checks, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
