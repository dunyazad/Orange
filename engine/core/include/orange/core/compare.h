#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>  // .cross()

#include "orange/core/kdtree.h"

// 3D Compare (Geomagic-style deviation analysis): signed distance from every
// point of a TEST cloud/mesh to a REFERENCE surface. Reference is either a
// triangle mesh (exact point-to-triangle distance, sign from the face normal)
// or a point cloud (nearest-point distance, sign from the reference normals
// when supplied, unsigned otherwise). CPU, Eigen-backed, header-only.

namespace orange::geometry {

struct CompareStats {
    float  minDev = 0, maxDev = 0;  // signed extremes
    float  mean = 0, rms = 0;       // of the signed deviations
    float  range = 0;               // suggested symmetric color range (98th pct of |dev|)
    bool   isSigned = true;         // false => reference gave no orientation
    size_t count = 0;
};

// Deviation -> display color. Five-stop jet (blue - cyan - green - yellow -
// red) QUANTIZED into `bands` discrete steps: a smooth two-stop ramp hides
// magnitude, discrete bands read as deviation contours (Geomagic-style).
// Signed: -range = blue, 0 = green, +range = red. Unsigned: 0 = blue -> red.
inline Eigen::Vector3f compareBandColor(float dev, float range, bool isSigned,
                                        int bands = 12) {
    float r = range > 1e-12f ? range : 1e-12f;
    float t = isSigned ? (dev + r) / (2.0f * r) : dev / r;
    t = std::clamp(t, 0.0f, 1.0f);
    if (bands > 1)  // snap to the containing band's center
        t = (std::floor(std::min(t, 0.9999f) * (float)bands) + 0.5f) / (float)bands;
    // 5-stop jet: blue(0) cyan(.25) green(.5) yellow(.75) red(1).
    float s = t * 4.0f;
    if (s < 1.0f) return {0.0f, s, 1.0f};               // blue -> cyan
    if (s < 2.0f) return {0.0f, 1.0f, 2.0f - s};        // cyan -> green
    if (s < 3.0f) return {s - 2.0f, 1.0f, 0.0f};        // green -> yellow
    return {1.0f, 4.0f - s, 0.0f};                      // yellow -> red
}

// Closest point on triangle (a,b,c) to p -- Ericson, Real-Time Collision Detection.
inline Eigen::Vector3f closestPointOnTriangle(const Eigen::Vector3f& p, const Eigen::Vector3f& a,
                                              const Eigen::Vector3f& b, const Eigen::Vector3f& c) {
    Eigen::Vector3f ab = b - a, ac = c - a, ap = p - a;
    float d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) return a;
    Eigen::Vector3f bp = p - b;
    float d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
    Eigen::Vector3f cp = p - c;
    float d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

// Signed deviation per test point. `refIdx` empty => point-cloud reference
// (then `refNormals`, when non-empty and refPos-sized, orients the sign; else
// the result is unsigned distances and stats.isSigned = false). For a triangle
// reference the true nearest triangle is found among the k nearest centroids
// (k widens with none found nearby) -- exact for dense meshes in practice.
// `progress` (optional, 0..1) is safe to drive from the caller's atomic.
inline std::vector<float> compareDeviations(const std::vector<Eigen::Vector3f>& testPts,
                                            const std::vector<Eigen::Vector3f>& refPos,
                                            const std::vector<uint32_t>&        refIdx,
                                            const std::vector<Eigen::Vector3f>& refNormals,
                                            CompareStats&                       stats,
                                            const std::function<void(float)>&   progress = {}) {
    std::vector<float> dev(testPts.size(), 0.0f);
    stats = {};
    if (testPts.empty() || refPos.empty()) return dev;

    const size_t triCount = refIdx.size() / 3;
    const bool   triRef   = triCount > 0;

    // Precompute triangle corners + normals + centroids (KDTree candidates).
    std::vector<Eigen::Vector3f> centroids;
    std::vector<Eigen::Vector3f> triNrm;
    if (triRef) {
        centroids.reserve(triCount);
        triNrm.reserve(triCount);
        for (size_t t = 0; t < triCount; ++t) {
            const Eigen::Vector3f& a = refPos[refIdx[t * 3]];
            const Eigen::Vector3f& b = refPos[refIdx[t * 3 + 1]];
            const Eigen::Vector3f& c = refPos[refIdx[t * 3 + 2]];
            centroids.push_back((a + b + c) / 3.0f);
            Eigen::Vector3f n = (b - a).cross(c - a);
            float len = n.norm();
            triNrm.push_back(len > 1e-20f ? Eigen::Vector3f(n / len) : Eigen::Vector3f(0, 1, 0));
        }
    }
    KDTree tree;
    tree.build(triRef ? centroids : refPos);

    const bool cloudSigned = !triRef && refNormals.size() == refPos.size();
    stats.isSigned         = triRef || cloudSigned;

    std::atomic<size_t> doneCount{0};
    const size_t        n = testPts.size();

    // Plain indexed loop split by hardware threads (matches the modes' style of
    // CPU parallelism without needing <execution> here).
    auto worker = [&](size_t begin, size_t end) {
        std::vector<int> cand;
        for (size_t i = begin; i < end; ++i) {
            const Eigen::Vector3f& p = testPts[i];
            if (triRef) {
                float bestD2 = 3.4e38f;
                float best   = 0.0f;
                // Widen the candidate set until something is found (tiny meshes).
                for (int k = 24; ; k *= 4) {
                    tree.kNearest(p, k, cand);
                    for (int t : cand) {
                        const Eigen::Vector3f& a = refPos[refIdx[(size_t)t * 3]];
                        const Eigen::Vector3f& b = refPos[refIdx[(size_t)t * 3 + 1]];
                        const Eigen::Vector3f& c = refPos[refIdx[(size_t)t * 3 + 2]];
                        Eigen::Vector3f cp = closestPointOnTriangle(p, a, b, c);
                        float d2 = (p - cp).squaredNorm();
                        if (d2 < bestD2) {
                            bestD2 = d2;
                            float d = std::sqrt(d2);
                            best    = ((p - cp).dot(triNrm[t]) < 0.0f) ? -d : d;
                        }
                    }
                    if (bestD2 < 3.4e38f || (size_t)k >= triCount) break;
                }
                dev[i] = best;
            } else {
                float d2 = 0.0f;
                int   j  = tree.nearest(p, &d2);
                float d  = std::sqrt(d2);
                if (j >= 0 && cloudSigned) {
                    dev[i] = ((p - refPos[j]).dot(refNormals[j]) < 0.0f) ? -d : d;
                } else {
                    dev[i] = d;
                }
            }
            size_t c = ++doneCount;
            if (progress && (c & 0xFFF) == 0)
                progress(static_cast<float>(c) / static_cast<float>(n));
        }
    };
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    if (n < 4096 || hw == 1) {
        worker(0, n);
    } else {
        std::vector<std::thread> pool;
        size_t chunk = (n + hw - 1) / hw;
        for (unsigned t = 0; t < hw; ++t) {
            size_t b = t * chunk, e = std::min(n, b + chunk);
            if (b >= e) break;
            pool.emplace_back(worker, b, e);
        }
        for (auto& th : pool) th.join();
    }

    // Stats + suggested symmetric color range (98th percentile of |dev| so a
    // few outliers don't wash the map out).
    double sum = 0, sum2 = 0;
    stats.minDev = dev[0];
    stats.maxDev = dev[0];
    for (float d : dev) {
        sum += d; sum2 += (double)d * d;
        stats.minDev = std::min(stats.minDev, d);
        stats.maxDev = std::max(stats.maxDev, d);
    }
    stats.count = n;
    stats.mean  = static_cast<float>(sum / n);
    stats.rms   = static_cast<float>(std::sqrt(sum2 / n));
    std::vector<float> mag(dev.size());
    for (size_t i = 0; i < dev.size(); ++i) mag[i] = std::fabs(dev[i]);
    size_t p98 = std::min(mag.size() - 1, (size_t)((double)mag.size() * 0.98));
    std::nth_element(mag.begin(), mag.begin() + p98, mag.end());
    stats.range = std::max(mag[p98], 1e-9f);
    if (progress) progress(1.0f);
    return dev;
}

}  // namespace orange::geometry
