// Numeric build check on a real point cloud. Loads a PLY, builds every spatial
// structure, and verifies (vs brute force) that radius queries are correct and the
// structure covers all points -- the same things the on-screen viz relies on.
// Usage: orange_ply_check [path.ply]   (default: D:/Resources/3D/PLY/Compound.ply)

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "orange/core/ball_tree.h"
#include "orange/core/bsp.h"
#include "orange/core/geometry.h"
#include "orange/core/kdtree.h"
#include "orange/core/loose_octree.h"
#include "orange/core/octree.h"
#include "orange/core/rtree.h"
#include "orange/core/serialization.h"
#include "orange/core/uniform_grid.h"

using namespace orange;
using Eigen::Vector3f;
using Clock = std::chrono::high_resolution_clock;
static int g_fail = 0;

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static int bruteCount(const std::vector<Vector3f>& P, const Vector3f& q, float r) {
    int c = 0; float r2 = r * r;
    for (const auto& p : P) if ((p - q).squaredNorm() <= r2) ++c;
    return c;
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "D:/Resources/3D/PLY/Compound.ply";
    std::printf("Loading %s\n", path.c_str());

    orange::io::PLYFormat ply;
    auto l0 = Clock::now();
    if (!ply.Deserialize(path)) { std::printf("FAILED to load\n"); return 2; }
    auto l1 = Clock::now();
    const std::vector<Vector3f>& P = ply.GetPoints();
    std::printf("Loaded %zu points in %.0f ms\n", P.size(), ms(l0, l1));
    if (P.empty()) { std::printf("no points\n"); return 2; }

    geometry::AABB full; for (const auto& p : P) full.expand(p);
    geometry::AABB rb = geometry::robustBounds(P, 0.001f);
    float md = (rb.max - rb.min).norm();
    std::printf("full diag  = %.3f\nrobust diag= %.3f  (ratio %.2f -> outliers %s)\n",
                (full.max - full.min).norm(), md, (full.max - full.min).norm() / md,
                (full.max - full.min).norm() > md * 1.5f ? "PRESENT" : "negligible");

    // Query points: a deterministic spread of existing points.
    std::vector<Vector3f> Q;
    for (int i = 0; i < 25; ++i) Q.push_back(P[(size_t)i * (P.size() / 25)]);
    float r = md * 0.02f;

    auto check = [&](const char* name, auto build, auto query, bool doCover, auto coverBoxes) {
        auto t0 = Clock::now(); build(); auto t1 = Clock::now();
        bool correct = true;
        std::vector<int> out;
        for (const auto& q : Q) {
            query(q, out);
            if ((int)out.size() != bruteCount(P, q, r)) { correct = false; break; }
        }
        // Coverage: a sample of points each lands in some emitted box.
        int boxN = 0; bool covered = true;
        if (doCover) {
            std::vector<geometry::AABB> boxes = coverBoxes();
            boxN = (int)boxes.size();
            for (size_t i = 0; i < P.size(); i += P.size() / 200 + 1) {
                bool in = false;
                for (const auto& b : boxes) if (b.contains(P[i])) { in = true; break; }
                if (!in) { covered = false; break; }
            }
        }
        std::printf("  %-12s build %8.1f ms  query %-7s  %s%s\n", name, ms(t0, t1),
                    correct ? "OK" : "WRONG",
                    doCover ? (covered ? "cover OK " : "cover GAP ") : "",
                    doCover ? ("boxes=" + std::to_string(boxN)).c_str() : "");
        if (!correct || (doCover && !covered)) ++g_fail;
    };

    { geometry::Octree s;
      check("Octree", [&]{ s.build(P, 64); },
            [&](const Vector3f& q, std::vector<int>& o){ s.radiusQuery(q, r, o); },
            true, [&]{ std::vector<geometry::AABB> b; s.leafBoxes(b); return b; }); }
    { geometry::KDTree s;
      check("KDTree", [&]{ s.build(P); },
            [&](const Vector3f& q, std::vector<int>& o){ s.radiusQuery(q, r, o); },
            false, []{ return std::vector<geometry::AABB>{}; }); }
    { geometry::UniformGrid s;
      check("UniformGrid", [&]{ s.build(P, r); },
            [&](const Vector3f& q, std::vector<int>& o){ s.radiusQuery(q, r, o); },
            true, [&]{ std::vector<geometry::AABB> b; s.occupiedCellBoxes(b); return b; }); }
    { geometry::LooseOctree s;
      check("LooseOctree", [&]{ s.build(P, 64); },
            [&](const Vector3f& q, std::vector<int>& o){ s.radiusQuery(q, r, o); },
            false, []{ return std::vector<geometry::AABB>{}; }); }
    { geometry::BSP s;
      check("BSP", [&]{ s.build(P, 64); },
            [&](const Vector3f& q, std::vector<int>& o){ s.radiusQuery(q, r, o); },
            false, []{ return std::vector<geometry::AABB>{}; }); }
    { geometry::RTree s;
      check("RTree", [&]{ s.build(P, 16); },
            [&](const Vector3f& q, std::vector<int>& o){ s.radiusQuery(q, r, o); },
            false, []{ return std::vector<geometry::AABB>{}; }); }
    { geometry::BallTree s;
      check("BallTree", [&]{ s.build(P, 32); },
            [&](const Vector3f& q, std::vector<int>& o){ s.radiusQuery(q, r, o); },
            false, []{ return std::vector<geometry::AABB>{}; }); }

    std::printf("\n%s\n", g_fail == 0 ? "ALL OK" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
