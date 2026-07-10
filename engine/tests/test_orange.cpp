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

#include <Eigen/Geometry>
#include <entt/entt.hpp>

#include "orange/core/ball_tree.h"
#include "orange/core/bsp.h"
#include "orange/core/bvh.h"
#include "orange/core/color.h"
#include "orange/core/kdtree.h"
#include "orange/core/loose_octree.h"
#include "orange/core/octree.h"
#include "orange/core/rtree.h"
#include "orange/core/uniform_grid.h"
#include "orange/core/debug_draw.h"
#include "orange/core/geometry.h"
#include "orange/core/modes.h"
#include "orange/core/morton3d.h"
#include "orange/core/normals.h"
#include "orange/core/point_ops.h"
#include "orange/core/primitives.h"
#include "orange/core/serialization.h"
#include "orange/core/sparse_grid.h"
#include "orange/core/ui_layout.h"
#include "orange/ecs/components.h"

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
    int  progressCalls = 0;
    bool progressInRange = true;
    for (int i = 0; i < orange::modes::modeCount(); ++i) {
        dd.clear();
        orange::modes::runMode(i, in, dd, [&](float f) {
            ++progressCalls;
            if (f < 0.0f || f > 1.0001f) progressInRange = false;
        });
        if (!dd.empty()) anyEmitted = true;
    }
    dd.clear();
    CHECK(anyEmitted);
    CHECK(progressCalls > 0);     // modes report progress through the callback
    CHECK(progressInRange);       // always within [0,1]
}

// --- Normal Divergence: sphere normals fan out => positive (warm) everywhere --
static void test_normal_divergence() {
    std::printf("[normal divergence]\n");
    const int idx = orange::modes::modeIndexByName("Normal Divergence");
    CHECK(idx >= 0);

    orange::modes::ModeInput in;
    for (int a = 0; a < 24; ++a)
        for (int b = 1; b < 12; ++b) {  // poles excluded (duplicate points)
            float phi = 2.0f * 3.14159265f * a / 24.0f, th = 3.14159265f * b / 12.0f;
            Eigen::Vector3f n(std::sin(th) * std::cos(phi), std::cos(th),
                              std::sin(th) * std::sin(phi));
            in.points.push_back(n);   // unit sphere: position == outward normal
            in.normals.push_back(n);
        }
    std::vector<Eigen::Vector3f> colors;
    orange::debug::DebugDraw extras;
    orange::modes::runModeColors(idx, in, colors, extras);
    CHECK(colors.size() == in.points.size());
    // Diverging map: warm (R > B) = positive divergence. A sphere is convex
    // everywhere, so the overwhelming majority must land on the warm side.
    int warm = 0;
    for (const auto& c : colors) warm += c.x() > c.z();
    CHECK(warm > (int)colors.size() * 9 / 10);

    // The mode publishes a signed legend scale; a keep/drop filter does not.
    orange::modes::ModeLegend ml = orange::modes::modeLastLegend();
    CHECK(ml.valid && ml.isSigned && ml.range > 0.0f);
    orange::modes::runModeColors(orange::modes::modeIndexByName("Outlier: QFOR"), in, colors,
                                 extras);
    CHECK(!orange::modes::modeLastLegend().valid);
}

// --- QFOR: quadric-fit outlier filter ----------------------------------------
// A curved (paraboloid) surface with a deterministic ripple and a few injected
// spikes. The quadric fit must absorb the smooth curvature (curved inliers stay
// green) while every spike lands red; Fit must pull the spikes back onto the
// surface and leave inliers in place.
static void test_qfor() {
    std::printf("[qfor]\n");
    const int idx = orange::modes::modeIndexByName("Outlier: QFOR");
    CHECK(idx >= 0);
    CHECK(orange::modes::modeApplyKind(idx) == orange::modes::ApplyKind::Recolor);
    CHECK(orange::modes::modeCanFit(idx));
    CHECK(orange::modes::modeCanTransformPoints(idx));

    orange::modes::ModeInput in;
    std::vector<int> spikes;
    for (int gy = 0; gy < 20; ++gy)
        for (int gx = 0; gx < 20; ++gx) {
            float x = -1.0f + gx * (2.0f / 19.0f);
            float y = -1.0f + gy * (2.0f / 19.0f);
            float z = 0.3f * (x * x + y * y)                       // smooth curvature
                    + 0.004f * std::sin(12.9898f * x + 78.233f * y);  // ripple "noise"
            int i = (int)in.points.size();
            if (gx % 7 == 3 && gy % 9 == 4) { z += 0.4f; spikes.push_back(i); }  // spike
            in.points.emplace_back(x, y, z);
        }
    CHECK(spikes.size() >= 3);

    std::vector<Eigen::Vector3f> colors;
    orange::debug::DebugDraw extras;
    orange::modes::runModeColors(idx, in, colors, extras);
    CHECK(colors.size() == in.points.size());
    const Eigen::Vector3f keepGreen(0.1f, 0.9f, 0.2f);
    auto kept = [&](size_t i) { return (colors[i] - keepGreen).squaredNorm() < 1e-8f; };
    int spikeFlagged = 0, falsePositives = 0;
    std::vector<uint8_t> isSpike(in.points.size(), 0);
    for (int i : spikes) isSpike[i] = 1;
    for (size_t i = 0; i < in.points.size(); ++i) {
        if (isSpike[i]) spikeFlagged += !kept(i);
        else falsePositives += !kept(i);
    }
    CHECK(spikeFlagged == (int)spikes.size());               // every spike caught
    CHECK(falsePositives <= (int)in.points.size() / 20);     // curvature not misread (<5%)

    // Fit: spikes move (back toward the surface), inliers stay put.
    std::vector<Eigen::Vector3f> fitted;
    orange::modes::runModeFit(idx, in, fitted);
    CHECK(fitted.size() == in.points.size());
    bool spikesMoved = true, inliersHeld = true;
    for (size_t i = 0; i < in.points.size(); ++i) {
        float moved = (fitted[i] - in.points[i]).norm();
        if (isSpike[i]) spikesMoved = spikesMoved && moved > 0.2f;
        else if (kept(i)) inliersHeld = inliersHeld && moved == 0.0f;
    }
    CHECK(spikesMoved);
    CHECK(inliersHeld);

    // Candidate mask (the Pipeline condition pin): QFOR may only drop flagged
    // candidates -- the other spikes stay, however large their residual.
    CHECK(orange::modes::modeSupportsCandidates(idx));
    in.candidates.assign(in.points.size(), 0);
    in.candidates[spikes[0]] = 1;  // only the first spike is eligible
    orange::modes::runModeColors(idx, in, colors, extras);
    int droppedMasked = 0;
    for (size_t i = 0; i < colors.size(); ++i) droppedMasked += !kept(i);
    CHECK(!kept(spikes[0]));
    CHECK(droppedMasked == 1);
    in.candidates.clear();

    // Divergence Select: registered, chainable, and NOT itself candidate-aware
    // (it PRODUCES the candidate set).
    const int selIdx = orange::modes::modeIndexByName("Divergence Select");
    CHECK(selIdx >= 0);
    CHECK(orange::modes::modeCanTransformPoints(selIdx));
    CHECK(!orange::modes::modeSupportsCandidates(selIdx));
}

// --- QFOR (Div Gate): divergence-anomaly gate decides WHO gets judged ---------
// A unit sphere (supplied radial normals) with radial spikes. The sphere's
// uniform curvature high-passes to anomaly ~ 0; a spike tip's divergence drops
// BELOW its neighbourhood's (the stretched-out geometry dilutes the normal
// spread), so its anomaly is strongly NEGATIVE. Side=2 (both) gates the spikes
// in and the quadric residual removes them; Side=0 (+ only) gates them out, so
// the very same spikes survive -- proving the gate, not the residual, decides.
static void test_qfor_div_gate() {
    std::printf("[qfor div gate]\n");
    const int idx = orange::modes::modeIndexByName("Outlier: QFOR (Div Gate)");
    CHECK(idx >= 0);
    CHECK(orange::modes::modeCanFit(idx));

    orange::modes::ModeInput in;
    for (int a = 0; a < 24; ++a)
        for (int b = 1; b < 12; ++b) {
            float phi = 2.0f * 3.14159265f * a / 24.0f, th = 3.14159265f * b / 12.0f;
            Eigen::Vector3f n(std::sin(th) * std::cos(phi), std::cos(th),
                              std::sin(th) * std::sin(phi));
            in.points.push_back(n);
            in.normals.push_back(n);
        }
    std::vector<int> spikes = {30, 95, 160, 225};
    for (int i : spikes) in.points[i] *= 1.3f;  // radial protrusion

    // 3-band viz: only RED means removed (amber = gated but kept).
    const Eigen::Vector3f dropRed(0.5f, 0.12f, 0.12f);
    auto dropped = [&](const std::vector<Eigen::Vector3f>& c, size_t i) {
        return (c[i] - dropRed).squaredNorm() < 1e-8f;
    };
    orange::debug::DebugDraw extras;

    // Params: {div on, Div K, Sigma Thresh, Side, Raw Field,
    //          qfor on, K, Alpha, Sparse Skip} -- indices 0/5 are the section
    // headers' enable checkboxes.
    // Side = 2 (both directions), anomaly gate: the spikes' anomaly clears the
    // gate, QFOR's residual removes them; false positives stay rare.
    in.params = {1.0f, 24.0f, 2.0f, 2.0f, 0.0f, 1.0f, 24.0f, 3.0f, 1.0f};
    std::vector<Eigen::Vector3f> colors;
    orange::modes::runModeColors(idx, in, colors, extras);
    CHECK(colors.size() == in.points.size());
    int spikeDrops = 0, fp = 0;
    std::vector<uint8_t> isSpike(in.points.size(), 0);
    for (int i : spikes) isSpike[i] = 1;
    for (size_t i = 0; i < colors.size(); ++i) {
        if (isSpike[i]) spikeDrops += dropped(colors, i);
        else fp += dropped(colors, i);
    }
    CHECK(spikeDrops == (int)spikes.size());
    CHECK(fp <= (int)in.points.size() / 20);

    // Side = 0 (positive-anomaly gate): the spikes' anomaly is negative ->
    // gated OUT -> they survive, residual notwithstanding.
    in.params = {1.0f, 24.0f, 2.0f, 0.0f, 0.0f, 1.0f, 24.0f, 3.0f, 1.0f};
    orange::modes::runModeColors(idx, in, colors, extras);
    int spikeKept = 0;
    for (int i : spikes) spikeKept += !dropped(colors, i);
    CHECK(spikeKept == (int)spikes.size());

    // Raw Field = 1, Sigma Thresh = 0 (strict sign test): the RAW divergence
    // is positive over the whole convex sphere, so Side = 0 gates everything
    // in -- the spikes drop again. Div K (16) deliberately differs from
    // QFOR's K (24): the gate and the quadric fit use independent neighbour
    // counts.
    in.params = {1.0f, 16.0f, 0.0f, 0.0f, 1.0f, 1.0f, 24.0f, 3.0f, 1.0f};
    orange::modes::runModeColors(idx, in, colors, extras);
    int spikeDropsRaw = 0;
    for (int i : spikes) spikeDropsRaw += dropped(colors, i);
    CHECK(spikeDropsRaw == (int)spikes.size());

    // Divergence checkbox OFF: no gate -- plain QFOR over the whole cloud, so
    // the spikes drop even though Side = 0 (anomaly negative) protected them
    // above.
    in.params = {0.0f, 24.0f, 2.0f, 0.0f, 0.0f, 1.0f, 24.0f, 3.0f, 1.0f};
    orange::modes::runModeColors(idx, in, colors, extras);
    int divOffDrops = 0;
    for (int i : spikes) divOffDrops += dropped(colors, i);
    CHECK(divOffDrops == (int)spikes.size());

    // QFOR checkbox OFF: the gate alone is the filter. Raw divergence > 0
    // covers (nearly) the whole convex sphere -- most of the cloud drops,
    // residual never consulted.
    in.params = {1.0f, 24.0f, 0.0f, 0.0f, 1.0f, 0.0f, 24.0f, 3.0f, 1.0f};
    orange::modes::runModeColors(idx, in, colors, extras);
    int gateOnly = 0;
    for (size_t i = 0; i < colors.size(); ++i) gateOnly += dropped(colors, i);
    CHECK(gateOnly >= (int)in.points.size() * 9 / 10);

    // Both checkboxes OFF: nothing drops.
    in.params = {0.0f, 24.0f, 0.0f, 0.0f, 1.0f, 0.0f, 24.0f, 3.0f, 1.0f};
    orange::modes::runModeColors(idx, in, colors, extras);
    int bothOff = 0;
    for (size_t i = 0; i < colors.size(); ++i) bothOff += dropped(colors, i);
    CHECK(bothOff == 0);

    // PFOR (Div Gate): same gate, plane-distance core. Side = 2 drops the
    // spikes; Side = 0 (their anomaly is negative) protects them.
    const int pidx = orange::modes::modeIndexByName("Outlier: PFOR (Div Gate)");
    CHECK(pidx >= 0);
    CHECK(orange::modes::modeSupportsCandidates(pidx));
    in.params = {16.0f, 0.05f, 2.0f, 2.0f, 0.0f};
    orange::modes::runModeColors(pidx, in, colors, extras);
    int pDrops = 0, pFp = 0;
    for (size_t i = 0; i < colors.size(); ++i) {
        if (isSpike[i]) pDrops += dropped(colors, i);
        else pFp += dropped(colors, i);
    }
    CHECK(pDrops == (int)spikes.size());
    // PFOR's plane is contaminated by the spike it shares a neighbourhood
    // with (the reason QFOR exists), so points AROUND a spike may co-drop:
    // allow a looser false-positive band than the quadric variant's 5%.
    CHECK(pFp <= (int)in.points.size() * 3 / 20);
    in.params = {16.0f, 0.05f, 2.0f, 0.0f, 0.0f};
    orange::modes::runModeColors(pidx, in, colors, extras);
    int pKept = 0;
    for (int i : spikes) pKept += !dropped(colors, i);
    CHECK(pKept == (int)spikes.size());
}

// --- QFOR (Dev Gate): normal-deviation angle decides WHO gets judged ----------
// Sphere with two kinds of radial spikes: ones whose normal is TILTED 60 deg
// (deviation >> Dev Deg -> gated -> removed) and ones whose normal stays
// radial (deviation ~ 0 -> untouchable, residual notwithstanding).
static void test_qfor_dev_gate() {
    std::printf("[qfor dev gate]\n");
    const int idx = orange::modes::modeIndexByName("Outlier: QFOR (Dev Gate)");
    CHECK(idx >= 0);
    CHECK(orange::modes::modeSupportsCandidates(idx));
    CHECK(orange::modes::modeCanTransformPoints(idx));

    orange::modes::ModeInput in;
    for (int a = 0; a < 24; ++a)
        for (int b = 1; b < 12; ++b) {
            float phi = 2.0f * 3.14159265f * a / 24.0f, th = 3.14159265f * b / 12.0f;
            Eigen::Vector3f n(std::sin(th) * std::cos(phi), std::cos(th),
                              std::sin(th) * std::sin(phi));
            in.points.push_back(n);
            in.normals.push_back(n);
        }
    std::vector<int> tilted = {30, 95, 160}, clean = {225, 60};
    for (int i : tilted) {
        in.points[i] *= 1.3f;
        Eigen::Vector3f n = in.normals[i];
        Eigen::Vector3f t = n.cross(Eigen::Vector3f::UnitY());
        if (t.squaredNorm() < 1e-6f) t = n.cross(Eigen::Vector3f::UnitX());
        t.normalize();
        in.normals[i] = (0.5f * n + 0.866f * t).normalized();  // ~60 deg off
    }
    for (int i : clean) in.points[i] *= 1.3f;  // big residual, orderly normal

    const Eigen::Vector3f dropRed(0.5f, 0.12f, 0.12f);
    std::vector<Eigen::Vector3f> colors;
    orange::debug::DebugDraw extras;
    in.params = {24.0f, 3.0f, 15.0f, 1.0f};
    orange::modes::runModeColors(idx, in, colors, extras);
    CHECK(colors.size() == in.points.size());
    auto dropped = [&](size_t i) { return (colors[i] - dropRed).squaredNorm() < 1e-8f; };
    int tiltedDrops = 0, cleanKept = 0, totalDrops = 0;
    for (size_t i = 0; i < colors.size(); ++i) totalDrops += dropped(i);
    for (int i : tilted) tiltedDrops += dropped(i);
    for (int i : clean) cleanKept += !dropped(i);
    CHECK(tiltedDrops == (int)tilted.size());
    CHECK(cleanKept == (int)clean.size());
    CHECK(totalDrops <= (int)tilted.size() + (int)in.points.size() / 20);
}

// --- SFOR: algebraic-sphere outlier filter ------------------------------------
// A unit sphere fits ITSELF exactly, so SFOR must catch every radial spike
// with (near) zero false positives, and Fit must pull the spikes back to
// radius ~1.
static void test_sfor() {
    std::printf("[sfor]\n");
    const int idx = orange::modes::modeIndexByName("Outlier: SFOR");
    CHECK(idx >= 0);
    CHECK(orange::modes::modeSupportsCandidates(idx));

    orange::modes::ModeInput in;
    for (int a = 0; a < 24; ++a)
        for (int b = 1; b < 12; ++b) {
            float phi = 2.0f * 3.14159265f * a / 24.0f, th = 3.14159265f * b / 12.0f;
            in.points.emplace_back(std::sin(th) * std::cos(phi), std::cos(th),
                                   std::sin(th) * std::sin(phi));
        }
    std::vector<int> spikes = {40, 120, 200};
    for (int i : spikes) in.points[i] *= 1.25f;

    const Eigen::Vector3f dropRed(0.5f, 0.12f, 0.12f);
    std::vector<Eigen::Vector3f> colors;
    orange::debug::DebugDraw extras;
    orange::modes::runModeColors(idx, in, colors, extras);
    CHECK(colors.size() == in.points.size());
    int spikeDrops = 0, fp = 0;
    std::vector<uint8_t> isSpike(in.points.size(), 0);
    for (int i : spikes) isSpike[i] = 1;
    for (size_t i = 0; i < colors.size(); ++i) {
        bool drop = (colors[i] - dropRed).squaredNorm() < 1e-8f;
        if (isSpike[i]) spikeDrops += drop;
        else fp += drop;
    }
    CHECK(spikeDrops == (int)spikes.size());
    CHECK(fp <= (int)in.points.size() / 50);  // sphere fits itself: ~0 FP

    std::vector<Eigen::Vector3f> fitted;
    orange::modes::runModeFit(idx, in, fitted);
    bool back = true;
    for (int i : spikes) back = back && std::fabs(fitted[i].norm() - 1.0f) < 0.05f;
    CHECK(back);
}

// --- RIMLS: edge-preserving filter ---------------------------------------------
// A sharp roof (two planes meeting at a ridge) with per-face normals. RIMLS
// must remove the injected spikes WITHOUT flagging the crease points -- the
// robust normal weight keeps the far face out of each point's reference
// surface, so the edge is not rounded into a false outlier.
static void test_rimls() {
    std::printf("[rimls]\n");
    const int idx = orange::modes::modeIndexByName("Outlier: RIMLS");
    CHECK(idx >= 0);

    orange::modes::ModeInput in;
    std::vector<int> ridge;
    const float s = 1.0f / std::sqrt(2.0f);
    for (int gy = 0; gy < 24; ++gy)
        for (int gx = 0; gx < 24; ++gx) {
            float x = -1.0f + gx * (2.0f / 23.0f);
            float y = -1.0f + gy * (2.0f / 23.0f);
            float z = -std::fabs(x);  // roof with the ridge along x = 0
            if (std::fabs(x) < 0.05f) ridge.push_back((int)in.points.size());
            in.points.emplace_back(x, y, z);
            in.normals.emplace_back(x < 0.0f ? Eigen::Vector3f(-s, 0, s)
                                             : Eigen::Vector3f(s, 0, s));
        }
    std::vector<int> spikes = {100, 300, 480};
    for (int i : spikes) in.points[i] += in.normals[i] * 0.25f;

    const Eigen::Vector3f dropRed(0.5f, 0.12f, 0.12f);
    std::vector<Eigen::Vector3f> colors;
    orange::debug::DebugDraw extras;
    in.params = {6.0f, 3.0f, 0.5f};  // radius wide enough to see past the spikes
    orange::modes::runModeColors(idx, in, colors, extras);
    CHECK(colors.size() == in.points.size());
    int spikeDrops = 0, ridgeDrops = 0;
    std::vector<uint8_t> isSpike(in.points.size(), 0);
    for (int i : spikes) isSpike[i] = 1;
    for (int i : ridge)
        if (!isSpike[i]) ridgeDrops += (colors[i] - dropRed).squaredNorm() < 1e-8f;
    for (int i : spikes) spikeDrops += (colors[i] - dropRed).squaredNorm() < 1e-8f;
    CHECK(spikeDrops == (int)spikes.size());
    CHECK(ridgeDrops == 0);  // the crease must survive
}

// --- RANSAC Shapes: plane + sphere get separated --------------------------------
static void test_ransac_shapes() {
    std::printf("[ransac shapes]\n");
    const int idx = orange::modes::modeIndexByName("RANSAC Shapes");
    CHECK(idx >= 0);

    orange::modes::ModeInput in;
    size_t nPlane = 0;
    for (int gy = 0; gy < 20; ++gy)
        for (int gx = 0; gx < 20; ++gx) {
            in.points.emplace_back(-1.0f + gx / 9.5f, -1.0f + gy / 9.5f, 0.0f);
            ++nPlane;
        }
    for (int a = 0; a < 18; ++a)
        for (int b = 1; b < 17; ++b) {  // sphere r=0.5 at (3,0,0.5)
            float phi = 2.0f * 3.14159265f * a / 18.0f, th = 3.14159265f * b / 17.0f;
            in.points.emplace_back(3.0f + 0.5f * std::sin(th) * std::cos(phi),
                                   0.5f * std::sin(th) * std::sin(phi),
                                   0.5f + 0.5f * std::cos(th));
        }

    std::vector<Eigen::Vector3f> colors;
    orange::debug::DebugDraw extras;
    in.params = {1.0f, 10.0f, 4.0f};
    orange::modes::runModeColors(idx, in, colors, extras);
    CHECK(colors.size() == in.points.size());
    // Plane points share one assigned color, sphere points another, distinct.
    auto assigned = [&](size_t i) { return colors[i].x() >= 0.0f; };
    size_t pa = 0, sa = 0;
    for (size_t i = 0; i < nPlane; ++i) pa += assigned(i);
    for (size_t i = nPlane; i < colors.size(); ++i) sa += assigned(i);
    CHECK(pa > nPlane * 8 / 10);
    CHECK(sa > (colors.size() - nPlane) * 8 / 10);
    // Majority colors of the two groups must differ.
    Eigen::Vector3f cp = Eigen::Vector3f::Zero(), csx = Eigen::Vector3f::Zero();
    for (size_t i = 0; i < nPlane; ++i)
        if (assigned(i)) { cp = colors[i]; break; }
    for (size_t i = nPlane; i < colors.size(); ++i)
        if (assigned(i)) { csx = colors[i]; break; }
    CHECK((cp - csx).squaredNorm() > 1e-4f);
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

// --- geometry processing: primitives / normals / smoothing / ICP ------------
static void test_processing() {
    std::printf("[processing]\n");

    // Primitives: non-empty, finite, with unit-ish extent around the origin.
    auto box = orange::geometry::buildBox(Eigen::Vector3f(1, 1, 1), Eigen::Vector3f(1, 1, 1));
    CHECK(box.size() == 12);  // 6 faces * 2 tris
    auto sphere = orange::geometry::buildSphere(0.5f, 16, Eigen::Vector3f(1, 1, 1));
    CHECK(!sphere.empty());
    bool finite = true;
    for (const auto& t : sphere)
        for (int k = 0; k < 3; ++k)
            finite = finite && t.v[k].allFinite() && t.n[k].allFinite() &&
                     std::abs(t.n[k].norm() - 1.0f) < 1e-3f;  // sphere normals unit
    CHECK(finite);

    // An asymmetric slab cloud (recoverable rotation) for normals/smooth/ICP.
    std::vector<Eigen::Vector3f> cloud;
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 2; ++z)
                cloud.emplace_back((float)x, (float)y, (float)z);

    auto nrm = orange::geometry::estimateNormals(cloud, 12);
    CHECK(nrm.size() == cloud.size());

    // Smoothing preserves the count and stays finite/bounded.
    auto sm = orange::geometry::smoothPoints(cloud, 4, 0.5f, true, 10);
    CHECK(sm.size() == cloud.size());
    bool smOk = true;
    for (const auto& p : sm) smOk = smOk && p.allFinite();
    CHECK(smOk);

    // ICP recovers a known rigid transform: build src = R*dst + t, then align.
    Eigen::AngleAxisf R(0.15f, Eigen::Vector3f::UnitY());
    Eigen::Vector3f t(0.4f, -0.3f, 0.2f);
    std::vector<Eigen::Vector3f> src(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i) src[i] = R * cloud[i] + t;
    float rmse = 1e9f; int iters = 0;
    Eigen::Matrix4f T = orange::geometry::icpAlign(src, cloud, 60, rmse, iters);
    CHECK(rmse < 0.05f);   // converged to near-zero correspondence error
    CHECK(iters >= 1);
    CHECK(T.allFinite());
}

// --- widget layout persistence ---------------------------------------------
static void test_ui_layout() {
    std::printf("[ui_layout]\n");
    const std::string path = "orange_ui_layout_test.txt";
    {
        entt::registry w;
        auto e1 = w.create();
        orange::ecs::FpsWidget fps;
        fps.relX = 0.123f; fps.relY = 0.456f;
        w.emplace<orange::ecs::FpsWidget>(e1, fps);
        auto e2 = w.create();
        orange::ecs::TreeView tv;
        tv.relX = 0.777f; tv.relY = 0.222f; tv.expanded[0] = false; tv.expanded[1] = true;
        w.emplace<orange::ecs::TreeView>(e2, tv);
        orange::core::saveWidgetLayout(w, path);
    }
    {
        entt::registry w;
        auto e1 = w.create();
        w.emplace<orange::ecs::FpsWidget>(e1);   // defaults
        auto e2 = w.create();
        w.emplace<orange::ecs::TreeView>(e2);
        orange::core::loadWidgetLayout(w, path);
        const auto& fps = w.get<orange::ecs::FpsWidget>(e1);
        CHECK(std::fabs(fps.relX - 0.123f) < 1e-4f);
        CHECK(std::fabs(fps.relY - 0.456f) < 1e-4f);
        const auto& tv = w.get<orange::ecs::TreeView>(e2);
        CHECK(std::fabs(tv.relX - 0.777f) < 1e-4f);
        CHECK(std::fabs(tv.relY - 0.222f) < 1e-4f);
        CHECK(tv.expanded[0] == false);
        CHECK(tv.expanded[1] == true);
    }
    std::remove(path.c_str());
}

// --- BVH rebind (background-build safety) -----------------------------------
// A BVH references (does not own) its source arrays. The async pick-BVH build
// builds against a worker-thread copy, then rebinds to the entity's stable arrays.
// This reproduces that: build against temporaries, rebind, let the temporaries
// die, then query -- it must still return the correct hit (no dangling read).
static void test_bvh_rebind() {
    std::printf("[bvh_rebind]\n");
    std::vector<Eigen::Vector3f> pos = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    std::vector<uint32_t>        idx = {0, 1, 2, 0, 2, 3};
    orange::geometry::BVH bvh;
    {
        std::vector<Eigen::Vector3f> tmpPos = pos;  // worker-local copies
        std::vector<uint32_t>        tmpIdx = idx;
        bvh.build(tmpPos, tmpIdx);
        bvh.rebind(pos, idx);  // re-point at the stable arrays
    }                          // tmpPos/tmpIdx destroyed here
    float t = 0.0f; int tri = -1;
    bool hit = bvh.nearestHit(
        orange::geometry::Ray(Eigen::Vector3f(0.6f, 0.4f, 1.0f), Eigen::Vector3f(0, 0, -1)),
        t, tri);
    CHECK(hit);
    CHECK(std::fabs(t - 1.0f) < 1e-3f);

    // The point-based structures share the same "references input" pattern and now
    // expose rebind() too. Build each against a temporary copy, rebind to the
    // stable array, let the copy die, then query -- must not read freed memory.
    std::vector<Eigen::Vector3f> cloud;
    for (int x = 0; x < 6; ++x)
        for (int y = 0; y < 6; ++y) cloud.emplace_back((float)x, (float)y, 0.0f);
    Eigen::Vector3f q(2.0f, 2.0f, 0.0f);
    int brute = 0;
    for (const auto& p : cloud) if ((p - q).squaredNorm() <= 1.5f * 1.5f) ++brute;

    auto rebindRadius = [&](auto& s, const char* /*name*/) {
        { auto tmp = cloud; s.build(tmp, 8); s.rebind(cloud); }  // tmp dies here
        std::vector<int> out;
        s.radiusQuery(q, 1.5f, out);
        CHECK((int)out.size() == brute);
    };
    orange::geometry::Octree      oc;  rebindRadius(oc, "octree");
    orange::geometry::LooseOctree lo;  rebindRadius(lo, "loose");
    orange::geometry::BSP         bsp; rebindRadius(bsp, "bsp");
    orange::geometry::RTree       rt;  rebindRadius(rt, "rtree");
    orange::geometry::BallTree    bt;  rebindRadius(bt, "ball");

    orange::geometry::UniformGrid g;
    { auto tmp = cloud; g.build(tmp, 1.0f); g.rebind(cloud); }
    { std::vector<int> out; g.radiusQuery(q, 1.5f, out); CHECK((int)out.size() == brute); }

    orange::geometry::KDTree kd;
    { auto tmp = cloud; kd.build(tmp); kd.rebind(cloud); }
    { float dd = 0; int n = kd.nearest(q, &dd); CHECK(n >= 0 && (cloud[n] - q).norm() < 1e-4f); }
}

int main() {
    std::printf("Orange unit tests\n");
    test_geometry();
    test_bvh_rebind();
    test_morton();
    test_color();
    test_sparse_grid();
    test_modes();
    test_normal_divergence();
    test_qfor();
    test_qfor_div_gate();
    test_qfor_dev_gate();
    test_sfor();
    test_rimls();
    test_ransac_shapes();
    test_processing();
    test_ui_layout();
    test_io();
    std::printf("\n%d checks, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
