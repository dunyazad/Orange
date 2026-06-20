// L4 evaluation harness: builds each spatial structure on synthetic clouds, then
// measures build time and query time vs brute force, printing a table and a
// speedup. Doubles as a regression guard -- it returns non-zero if a structure is
// incorrect or (at the large size) not faster than brute force.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <Eigen/Core>

#include "orange/core/ball_tree.h"
#include "orange/core/bsp.h"
#include "orange/core/kdtree.h"
#include "orange/core/loose_octree.h"
#include "orange/core/octree.h"
#include "orange/core/rtree.h"
#include "orange/core/uniform_grid.h"

using Eigen::Vector3f;
using Clock = std::chrono::high_resolution_clock;
static int g_fail = 0;

static float frand(uint32_t n) {
    n = (n ^ 61u) ^ (n >> 16); n *= 9u; n = n ^ (n >> 4);
    n *= 0x27d4eb2du; n = n ^ (n >> 15);
    return (n & 0xffffff) / static_cast<float>(0x1000000);
}
static std::vector<Vector3f> cloud(int n) {
    std::vector<Vector3f> p; p.reserve(n);
    for (int i = 0; i < n; ++i)
        p.emplace_back(frand(i * 3) * 100.f, frand(i * 3 + 1) * 100.f, frand(i * 3 + 2) * 100.f);
    return p;
}
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// queries: returns total result count; `qfn(center, out)` runs one query.
template <typename Build, typename Query>
static void run(const char* name, const std::vector<Vector3f>& pts,
                const std::vector<Vector3f>& queries, float radius, bool guard,
                Build build, Query query) {
    auto t0 = Clock::now();
    build();
    auto t1 = Clock::now();
    long long total = 0;
    std::vector<int> out;
    for (const auto& q : queries) { query(q, out); total += (long long)out.size(); }
    auto t2 = Clock::now();

    // brute reference (count only) for correctness + speedup.
    long long bruteTotal = 0;
    auto b0 = Clock::now();
    for (const auto& q : queries) {
        long long c = 0; float r2 = radius * radius;
        for (const auto& p : pts) if ((p - q).squaredNorm() <= r2) ++c;
        bruteTotal += c;
    }
    auto b1 = Clock::now();

    double buildMs = ms(t0, t1), qMs = ms(t1, t2), bMs = ms(b0, b1);
    double speedup = qMs > 1e-6 ? bMs / qMs : 0.0;
    bool ok = (total == bruteTotal);
    std::printf("  %-13s build %8.2f ms   query %8.3f ms   brute %8.3f ms   x%6.1f  %s\n",
                name, buildMs, qMs, bMs, speedup, ok ? "" : "<-- WRONG");
    if (!ok) ++g_fail;
    if (guard && ok && speedup < 1.0) {
        std::printf("       ^ regression: %s slower than brute\n", name);
        ++g_fail;
    }
}

int main() {
    std::printf("Orange spatial-structure benchmark (radius query)\n");
    const float radius = 5.0f;
    for (int n : {20000, 100000}) {
        auto pts = cloud(n);
        std::vector<Vector3f> queries;
        for (int i = 0; i < 200; ++i)
            queries.emplace_back(frand(i * 5) * 100.f, frand(i * 5 + 1) * 100.f,
                                 frand(i * 5 + 2) * 100.f);
        bool guard = (n == 100000);  // expect a real speedup at the large size
        std::printf("\nN = %d points, %zu queries, radius %.1f\n", n, queries.size(), radius);

        { orange::geometry::Octree s;
          run("Octree", pts, queries, radius, guard,
              [&] { s.build(pts, 16); },
              [&](const Vector3f& q, std::vector<int>& o) { s.radiusQuery(q, radius, o); }); }
        { orange::geometry::KDTree s;
          run("KDTree", pts, queries, radius, guard,
              [&] { s.build(pts); },
              [&](const Vector3f& q, std::vector<int>& o) { s.radiusQuery(q, radius, o); }); }
        { orange::geometry::UniformGrid s;
          run("UniformGrid", pts, queries, radius, guard,
              [&] { s.build(pts, radius); },
              [&](const Vector3f& q, std::vector<int>& o) { s.radiusQuery(q, radius, o); }); }
        { orange::geometry::LooseOctree s;
          run("LooseOctree", pts, queries, radius, guard,
              [&] { s.build(pts, 16); },
              [&](const Vector3f& q, std::vector<int>& o) { s.radiusQuery(q, radius, o); }); }
        { orange::geometry::BSP s;
          run("BSP", pts, queries, radius, guard,
              [&] { s.build(pts, 16); },
              [&](const Vector3f& q, std::vector<int>& o) { s.radiusQuery(q, radius, o); }); }
        { orange::geometry::RTree s;
          run("RTree", pts, queries, radius, guard,
              [&] { s.build(pts, 8); },
              [&](const Vector3f& q, std::vector<int>& o) { s.radiusQuery(q, radius, o); }); }
        { orange::geometry::BallTree s;
          run("BallTree", pts, queries, radius, guard,
              [&] { s.build(pts, 16); },
              [&](const Vector3f& q, std::vector<int>& o) { s.radiusQuery(q, radius, o); }); }
    }
    std::printf("\n%s\n", g_fail == 0 ? "OK" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
