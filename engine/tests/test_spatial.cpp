// Tests for the spatial acceleration structures (BVH / Octree / KD-tree): each
// query is checked against a brute-force reference over the same data, so a wrong
// traversal or pruning bug shows up as a mismatch. Headless, deterministic.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>  // MatrixBase::cross is defined here, not in Core

#include "orange/core/ball_tree.h"
#include "orange/core/bsp.h"
#include "orange/core/bvh.h"
#include "orange/core/geometry.h"
#include "orange/core/kdtree.h"
#include "orange/core/loose_octree.h"
#include "orange/core/octree.h"
#include "orange/core/rtree.h"
#include "orange/core/uniform_grid.h"

using Eigen::Vector3f;
static int g_total = 0, g_fail = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        ++g_total;                                                                 \
        if (!(cond)) { ++g_fail; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

// Deterministic pseudo-random in [0,1) from an integer (no RNG dependency).
static float frand(uint32_t n) {
    n = (n ^ 61u) ^ (n >> 16);
    n *= 9u; n = n ^ (n >> 4); n *= 0x27d4eb2du; n = n ^ (n >> 15);
    return (n & 0xffffff) / static_cast<float>(0x1000000);
}

static void test_bvh() {
    std::printf("[bvh]\n");
    // A bumpy grid of triangles in roughly the z=0 plane.
    std::vector<Vector3f> pos;
    std::vector<uint32_t> idx;
    const int N = 12;
    for (int y = 0; y <= N; ++y)
        for (int x = 0; x <= N; ++x)
            pos.emplace_back((float)x, (float)y, frand(y * 131 + x) * 0.5f);
    auto vid = [&](int x, int y) { return (uint32_t)(y * (N + 1) + x); };
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            idx.insert(idx.end(), {vid(x, y), vid(x + 1, y), vid(x + 1, y + 1)});
            idx.insert(idx.end(), {vid(x, y), vid(x + 1, y + 1), vid(x, y + 1)});
        }

    orange::geometry::BVH bvh;
    bvh.build(pos, idx);
    CHECK(!bvh.empty());

    auto brute = [&](const orange::geometry::Ray& r, float& t) {
        float best = 1e30f; bool hit = false;
        for (size_t i = 0; i + 2 < idx.size(); i += 3) {
            float tt;
            if (orange::geometry::Ray(r).intersectTriangle(pos[idx[i]], pos[idx[i + 1]],
                                                           pos[idx[i + 2]], tt) &&
                tt >= 0 && tt < best) { best = tt; hit = true; }
        }
        t = best; return hit;
    };

    int agree = 0, rays = 0;
    for (int s = 0; s < 200; ++s) {
        Vector3f o(frand(s) * N, frand(s * 7 + 1) * N, 5.0f);
        Vector3f d(frand(s * 13) * 0.2f - 0.1f, frand(s * 17) * 0.2f - 0.1f, -1.0f);
        d.normalize();
        orange::geometry::Ray ray(o, d);
        float tb; bool hb = brute(ray, tb);
        float tv; int tri; bool hv = bvh.nearestHit(ray, tv, tri);
        ++rays;
        CHECK(hb == hv);
        if (hb && hv) { CHECK(std::fabs(tb - tv) < 1e-3f); if (std::fabs(tb - tv) < 1e-3f) ++agree; }
        else if (!hb && !hv) ++agree;
    }
    CHECK(agree == rays);
}

static std::vector<Vector3f> cloud() {
    std::vector<Vector3f> p;
    for (int i = 0; i < 600; ++i)
        p.emplace_back(frand(i * 3) * 10.0f, frand(i * 3 + 1) * 10.0f, frand(i * 3 + 2) * 10.0f);
    return p;
}

static void test_octree() {
    std::printf("[octree]\n");
    auto pts = cloud();
    orange::geometry::Octree oc;
    oc.build(pts, 8);
    CHECK(!oc.empty());

    Vector3f q(4.3f, 5.1f, 6.2f);
    float r = 2.0f;
    std::vector<int> got;
    oc.radiusQuery(q, r, got);
    int brute = 0;
    for (auto& p : pts) if ((p - q).squaredNorm() <= r * r) ++brute;
    CHECK((int)got.size() == brute);

    // AABB query vs brute.
    Vector3f mn(2, 2, 2), mx(6, 6, 6);
    std::vector<int> inBox;
    oc.aabbQuery(mn, mx, inBox);
    int bbrute = 0;
    for (auto& p : pts)
        if (p.x() >= mn.x() && p.x() <= mx.x() && p.y() >= mn.y() && p.y() <= mx.y() &&
            p.z() >= mn.z() && p.z() <= mx.z()) ++bbrute;
    CHECK((int)inBox.size() == bbrute);

    // kNN distances match the brute-force k smallest.
    std::vector<int> knn;
    oc.kNearest(q, 5, knn);
    CHECK(knn.size() == 5);
    std::vector<float> bd;
    for (auto& p : pts) bd.push_back((p - q).norm());
    std::sort(bd.begin(), bd.end());
    for (int i = 0; i < 5; ++i)
        CHECK(std::fabs((pts[knn[i]] - q).norm() - bd[i]) < 1e-4f);
}

static void test_kdtree() {
    std::printf("[kdtree]\n");
    auto pts = cloud();
    orange::geometry::KDTree kd;
    kd.build(pts);
    CHECK(!kd.empty());

    Vector3f q(7.7f, 1.3f, 3.9f);
    float d2 = 0; int n = kd.nearest(q, &d2);
    int bn = 0; float bestD2 = 1e30f;
    for (int i = 0; i < (int)pts.size(); ++i) {
        float dd = (pts[i] - q).squaredNorm();
        if (dd < bestD2) { bestD2 = dd; bn = i; }
    }
    CHECK(n == bn);

    std::vector<int> knn;
    kd.kNearest(q, 7, knn);
    CHECK(knn.size() == 7);
    std::vector<float> bd;
    for (auto& p : pts) bd.push_back((p - q).norm());
    std::sort(bd.begin(), bd.end());
    for (int i = 0; i < 7; ++i)
        CHECK(std::fabs((pts[knn[i]] - q).norm() - bd[i]) < 1e-4f);

    std::vector<int> rad;
    float r = 3.0f;
    kd.radiusQuery(q, r, rad);
    int brute = 0;
    for (auto& p : pts) if ((p - q).squaredNorm() <= r * r) ++brute;
    CHECK((int)rad.size() == brute);
}

// Reference radius count for the point structures.
static int bruteRadius(const std::vector<Vector3f>& pts, const Vector3f& q, float r) {
    int n = 0; for (auto& p : pts) if ((p - q).squaredNorm() <= r * r) ++n; return n;
}

static void test_uniform_grid() {
    std::printf("[uniform_grid]\n");
    auto pts = cloud();
    orange::geometry::UniformGrid g; g.build(pts, 1.0f);
    CHECK(!g.empty());
    Vector3f q(4.3f, 5.1f, 6.2f); float r = 2.0f;
    std::vector<int> got; g.radiusQuery(q, r, got);
    CHECK((int)got.size() == bruteRadius(pts, q, r));
    std::vector<orange::geometry::AABB> cells; g.occupiedCellBoxes(cells);
    CHECK(!cells.empty());
}

static void test_loose_octree() {
    std::printf("[loose_octree]\n");
    auto pts = cloud();
    orange::geometry::LooseOctree lo; lo.build(pts, 8);
    CHECK(!lo.empty());
    Vector3f q(2.7f, 8.1f, 3.3f); float r = 2.5f;
    std::vector<int> got; lo.radiusQuery(q, r, got);
    // Loose cells overlap, but the per-point distance filter keeps the count exact.
    CHECK((int)got.size() == bruteRadius(pts, q, r));
}

static void test_bsp() {
    std::printf("[bsp]\n");
    auto pts = cloud();
    orange::geometry::BSP bsp; bsp.build(pts, 8);
    CHECK(!bsp.empty());
    Vector3f q(6.1f, 2.2f, 7.4f); float r = 2.0f;
    std::vector<int> got; bsp.radiusQuery(q, r, got);
    CHECK((int)got.size() == bruteRadius(pts, q, r));
}

static void test_rtree() {
    std::printf("[rtree]\n");
    auto pts = cloud();
    orange::geometry::RTree rt; rt.build(pts, 8);
    CHECK(!rt.empty());
    Vector3f q(5.5f, 5.5f, 5.5f); float r = 2.0f;
    std::vector<int> got; rt.radiusQuery(q, r, got);
    CHECK((int)got.size() == bruteRadius(pts, q, r));
    Vector3f mn(2, 2, 2), mx(6, 6, 6);
    std::vector<int> box; rt.aabbQuery(mn, mx, box);
    int bbrute = 0;
    for (auto& p : pts)
        if (p.x() >= mn.x() && p.x() <= mx.x() && p.y() >= mn.y() && p.y() <= mx.y() &&
            p.z() >= mn.z() && p.z() <= mx.z()) ++bbrute;
    CHECK((int)box.size() == bbrute);
}

static void test_ball_tree() {
    std::printf("[ball_tree]\n");
    auto pts = cloud();
    orange::geometry::BallTree bt; bt.build(pts, 8);
    CHECK(!bt.empty());
    Vector3f q(3.3f, 6.6f, 1.1f); float r = 2.5f;
    std::vector<int> got; bt.radiusQuery(q, r, got);
    CHECK((int)got.size() == bruteRadius(pts, q, r));
    std::vector<int> knn; bt.kNearest(q, 6, knn);
    CHECK(knn.size() == 6);
    std::vector<float> bd; for (auto& p : pts) bd.push_back((p - q).norm());
    std::sort(bd.begin(), bd.end());
    for (int i = 0; i < 6; ++i)
        CHECK(std::fabs((pts[knn[i]] - q).norm() - bd[i]) < 1e-4f);
}

// Structural invariants that turn the visualization bugs we hit by eye into
// headless assertions: full coverage, outlier robustness, enough subdivision.
static void test_properties() {
    std::printf("[properties]\n");
    auto pts = cloud();

    // Coverage: every point lies inside some octree leaf box ("not all drawn").
    orange::geometry::Octree oc; oc.build(pts, 8);
    std::vector<orange::geometry::AABB> leaves; oc.leafBoxes(leaves);
    int covered = 0;
    for (const auto& p : pts)
        for (const auto& b : leaves)
            if (b.contains(p)) { ++covered; break; }
    CHECK(covered == (int)pts.size());

    // Subdivision: a dense cloud yields many nodes at the viz depth ("only part").
    std::vector<orange::geometry::AABB> boxes; std::vector<int> levels;
    oc.nodeBoxesDepth(boxes, levels, 18);
    CHECK(boxes.size() > 50);

    // Outlier robustness: a few far points must not inflate the robust bounds.
    auto withOutliers = pts;
    withOutliers.emplace_back(1000.0f, 0, 0);
    withOutliers.emplace_back(-900.0f, 500.0f, 0);
    withOutliers.emplace_back(0, 0, 2000.0f);
    orange::geometry::AABB rb = orange::geometry::robustBounds(withOutliers, 0.01f);
    // The cloud lives in [0,10]^3; robust bounds should stay near that, not ~2000.
    CHECK((rb.max - rb.min).maxCoeff() < 50.0f);
    orange::geometry::AABB full;
    for (const auto& p : withOutliers) full.expand(p);
    CHECK((full.max - full.min).maxCoeff() > 500.0f);  // raw AABB *is* inflated
}

int main() {
    std::printf("Orange spatial-partitioning tests\n");
    test_bvh();
    test_octree();
    test_kdtree();
    test_uniform_grid();
    test_loose_octree();
    test_bsp();
    test_rtree();
    test_ball_tree();
    test_properties();
    std::printf("\n%d checks, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
