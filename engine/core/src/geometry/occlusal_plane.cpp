// Occlusal-plane estimation (pipeline stage 1). See occlusal_plane.h for the
// bootstrap rationale. ASCII comments only (pipeline project rule).

#include "orange/core/occlusal_plane.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <unordered_map>
#include <utility>

#include <Eigen/Eigenvalues>

#include "orange/core/half_edge.h"
#include "orange/core/sparse_grid.h"

namespace orange::geometry {
namespace {

// A right-handed orthonormal frame for the arch: `axis` is the occlusal normal
// (smallest-eigenvalue direction); `u`,`v` span the in-plane (occlusal) plane.
struct Frame {
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f axis     = Eigen::Vector3f::UnitY();
    Eigen::Vector3f u        = Eigen::Vector3f::UnitX();
    Eigen::Vector3f v        = Eigen::Vector3f::UnitZ();
};

// PCA over a subset of points (empty `indices` == all points). Ascending
// eigenvalues: column 0 is the least-variance direction (occlusal normal), and
// columns 2/1 are the two widest in-plane directions. Returns false if there are
// too few points to form a covariance.
bool pcaFrame(const std::vector<Eigen::Vector3f>& pts,
              const std::vector<uint32_t>& indices, Frame& out) {
    const size_t n = indices.empty() ? pts.size() : indices.size();
    if (n < 3) return false;

    auto at = [&](size_t i) -> const Eigen::Vector3f& {
        return indices.empty() ? pts[i] : pts[indices[i]];
    };

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (size_t i = 0; i < n; ++i) centroid += at(i);
    centroid /= (float)n;

    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3f d = at(i) - centroid;
        cov += d * d.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
    if (es.info() != Eigen::Success) return false;

    out.centroid = centroid;
    out.axis = es.eigenvectors().col(0).normalized();  // smallest variance
    out.u    = es.eigenvectors().col(2).normalized();  // widest
    out.v    = es.eigenvectors().col(1).normalized();
    return true;
}

// Extract cusp / incisal-edge candidates for the given (oriented) frame.
//
// A cusp is a true local maximum of height (projection on `axis`) within a small
// 3D radius -- the cusp tip / incisal edge pokes up above everything around it.
// The old "tallest per 2D cell" picked one point per column, which slid onto
// buccal ridges and the gingival rim; real local-maxima NMS sits on the bumps.
//
// Three gates remove the false peaks the user saw:
//   - height gate: drop the lowest band (gingival margin, scan floor),
//   - density gate: drop sparse points (open mesh boundary / trimmed rim),
//   - protrusion: the peak must rise above its neighbourhood centroid.
std::vector<uint32_t> extractCandidates(const std::vector<Eigen::Vector3f>& pts,
                                        const Frame& frame, const SparseGrid& grid,
                                        const OcclusalPlaneParams& params) {
    (void)params;
    const size_t n = pts.size();
    if (n < 8) return {};
    const float diag = (grid.aabb.max - grid.aabb.min).norm();
    if (diag <= 0.0f) return {};

    const float r       = diag * 0.035f;   // NMS / neighbourhood radius (~cusp spacing)
    const float cell    = std::max(r * 0.6f, diag * 0.005f);  // pre-pass cell (< r)
    const int   minNbr  = 8;               // density gate (drop boundary points)
    const float hGateFr = 0.22f;           // drop the lowest 22% of the height band

    // Per-vertex height + in-plane coords; track height range for the gate.
    std::vector<float> H(n);
    float aMin = 1e30f, bMin = 1e30f, hMin = 1e30f, hMax = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3f d = pts[i] - frame.centroid;
        H[i] = d.dot(frame.axis);
        aMin = std::min(aMin, d.dot(frame.u));
        bMin = std::min(bMin, d.dot(frame.v));
        hMin = std::min(hMin, H[i]); hMax = std::max(hMax, H[i]);
    }
    const float hGate = hMin + hGateFr * (hMax - hMin);

    // Pre-pass: tallest vertex per fine cell, among the upper height band. This
    // thins ~1M verts to a few thousand peak candidates so the radius queries
    // below stay cheap.
    std::unordered_map<uint64_t, uint32_t> best;
    std::unordered_map<uint64_t, float>    bestH;
    for (size_t i = 0; i < n; ++i) {
        if (H[i] < hGate) continue;
        Eigen::Vector3f d = pts[i] - frame.centroid;
        int ia = (int)std::floor((d.dot(frame.u) - aMin) / cell);
        int ib = (int)std::floor((d.dot(frame.v) - bMin) / cell);
        uint64_t k = ((uint64_t)(uint32_t)ia << 32) | (uint32_t)ib;
        auto it = bestH.find(k);
        if (it == bestH.end() || H[i] > it->second) { bestH[k] = H[i]; best[k] = (uint32_t)i; }
    }

    // Confirm each cell winner is a true local maximum within radius r, dense
    // (interior, not boundary), and protruding above its neighbourhood.
    std::vector<uint32_t> out;
    out.reserve(best.size());
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    for (const auto& kv : best) {
        uint32_t idx = kv.second;
        grid.pointsWithinRadius(pts, pts[idx], r, nbr, dist);
        if ((int)nbr.size() < minNbr) continue;  // density gate

        bool isMax = true;
        Eigen::Vector3f nc = Eigen::Vector3f::Zero();
        int cnt = 0;
        for (unsigned int j : nbr) {
            if (j == idx) continue;
            if (H[j] > H[idx] + 1e-6f) { isMax = false; break; }  // a taller neighbour
            nc += pts[j]; ++cnt;
        }
        if (!isMax || cnt == 0) continue;
        nc /= (float)cnt;
        if ((pts[idx] - nc).dot(frame.axis) <= r * 0.02f) continue;  // must protrude
        out.push_back(idx);
    }
    return out;
}

// Build the kNN grid and the coarse, orientation-locked occlusal frame: whole-
// mesh PCA, then flip the axis toward whichever side yields more convex peaks
// (the occlusal/biting side). Shared by findCusps and estimateOcclusalPlane.
bool coarseFrame(const std::vector<Eigen::Vector3f>& vertices,
                 const OcclusalPlaneParams& params, SparseGrid& grid, Frame& frame) {
    if (vertices.size() < 8) return false;
    Eigen::Vector3f mn = vertices[0], mx = vertices[0];
    for (const auto& p : vertices) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    float diag = (mx - mn).norm();
    grid.build(vertices, diag > 0.0f ? diag * 0.02f : 1.0f);

    if (!pcaFrame(vertices, {}, frame)) return false;
    Frame flipped = frame;
    flipped.axis = -frame.axis;
    size_t nPlus  = extractCandidates(vertices, frame,   grid, params).size();
    size_t nMinus = extractCandidates(vertices, flipped, grid, params).size();
    if (nMinus > nPlus) frame.axis = -frame.axis;
    return true;
}

} // namespace

OcclusalPlane wholeMeshPCA(const std::vector<Eigen::Vector3f>& vertices) {
    OcclusalPlane result;
    Frame frame;
    if (!pcaFrame(vertices, {}, frame)) return result;
    Eigen::Vector3f normal = frame.axis;
    // No occlusal cue at this step: orient toward the +Y hemisphere just so the
    // arrow has a stable, readable direction.
    if (normal.dot(Eigen::Vector3f::UnitY()) < 0.0f) normal = -normal;
    result.position = frame.centroid;
    result.normal = normal.normalized();
    result.valid = true;
    return result;
}

std::vector<Eigen::Vector3f> findCusps(const std::vector<Eigen::Vector3f>& vertices,
                                       const std::vector<uint32_t>& indices,
                                       const OcclusalPlaneParams& params) {
    std::vector<Eigen::Vector3f> out;
    if (vertices.size() < 8 || indices.size() < 3) return out;

    // Half-edge mesh -> O(1) one-ring traversal. build() welds the triangle soup
    // (STL) so the graph is connected and shares vertices.
    HalfEdgeMesh he;
    if (!he.build(vertices, indices)) return out;
    const uint32_t n = (uint32_t)he.vertexCount();

    // Rough occlusal normal from whole-mesh PCA, oriented to the occlusal (up)
    // side. Height = projection on that normal.
    Frame frame;
    if (!pcaFrame(he.positions(), {}, frame)) return out;
    Eigen::Vector3f axis = frame.axis;
    if (axis.dot(Eigen::Vector3f::UnitY()) < 0.0f) axis = -axis;

    std::vector<float> H(n);
    float hMin = 1e30f, hMax = -1e30f;
    for (uint32_t i = 0; i < n; ++i) {
        H[i] = (he.position(i) - frame.centroid).dot(axis);
        hMin = std::min(hMin, H[i]); hMax = std::max(hMax, H[i]);
    }

    // Cache the one-ring adjacency once (half-edge is the source of truth; this
    // CSR is just a flat copy for the tight smoothing / climbing loops).
    std::vector<uint32_t> off(n + 1, 0);
    {
        std::vector<uint32_t> ring;
        for (uint32_t v = 0; v < n; ++v) { he.oneRing(v, ring); off[v + 1] = off[v] + (uint32_t)ring.size(); }
    }
    std::vector<uint32_t> nbrs(off[n]);
    {
        std::vector<uint32_t> ring;
        for (uint32_t v = 0; v < n; ++v) {
            he.oneRing(v, ring);
            std::copy(ring.begin(), ring.end(), nbrs.begin() + off[v]);
        }
    }

    // Light smoothing of the height field before climbing. Without it the raw
    // scan noise spawns many local maxima per cusp, so each cusp shatters into a
    // cluster of tips; smoothing merges those into one basin per cusp. Climb on
    // the smoothed field Hs, but gate by true height below.
    std::vector<float> Hs = H;
    {
        std::vector<float> tmp(n);
        for (int it = 0; it < std::max(0, params.cuspSmoothIters); ++it) {
            for (uint32_t v = 0; v < n; ++v) {
                uint32_t e0 = off[v], e1 = off[v + 1];
                if (e1 == e0) { tmp[v] = Hs[v]; continue; }
                float s = 0.0f;
                for (uint32_t e = e0; e < e1; ++e) s += Hs[nbrs[e]];
                tmp[v] = 0.5f * Hs[v] + 0.5f * (s / (float)(e1 - e0));
            }
            Hs.swap(tmp);
        }
    }

    // Steepest-ascent hill climbing on Hs. flow[v] = the cusp-tip vertex that v
    // climbs to (-1 = unassigned). Every vertex flows uphill to exactly one local
    // max; the distinct local maxima are cusp tips. A climb that reaches an
    // assigned vertex adopts its tip (memoized), so the mesh is processed once.
    std::vector<int32_t> flow(n, -1);
    std::vector<uint32_t> path;
    for (uint32_t seed = 0; seed < n; ++seed) {
        if (flow[seed] != -1) continue;
        path.clear();
        uint32_t v = seed;
        int32_t tip = -1;
        for (;;) {
            if (flow[v] != -1) { tip = flow[v]; break; }  // reached a known basin
            path.push_back(v);
            uint32_t best = v;
            float bh = Hs[v];
            for (uint32_t e = off[v]; e < off[v + 1]; ++e) {
                uint32_t u = nbrs[e];
                if (Hs[u] > bh) { bh = Hs[u]; best = u; }
            }
            if (best == v) { tip = (int32_t)v; break; }   // no higher neighbour: cusp tip
            v = best;
        }
        for (uint32_t p : path) flow[p] = tip;
    }

    // Collect distinct tips, dropping those below the height gate (floor / gingiva
    // / scan base are far below the occlusal band).
    const float hGate = hMin + std::max(0.0f, params.cuspHeightGate) * (hMax - hMin);
    std::vector<uint8_t> isTip(n, 0);
    for (uint32_t v = 0; v < n; ++v)
        if (flow[v] >= 0) isTip[flow[v]] = 1;
    size_t dropped = 0;
    for (uint32_t v = 0; v < n; ++v) {
        if (!isTip[v]) continue;
        if (H[v] < hGate) { ++dropped; continue; }  // floor/gingiva tip
        out.push_back(he.position(v));
    }

    std::fprintf(stderr, "[findCusps] welded=%u smooth=%d gate=%.2f cusps=%zu (dropped %zu low)\n",
                 n, params.cuspSmoothIters, params.cuspHeightGate, out.size(), dropped);
    return out;
}

OcclusalPlane estimateOcclusalPlane(const std::vector<Eigen::Vector3f>& vertices,
                                    const OcclusalPlaneParams& params) {
    OcclusalPlane result;

    // 1-2. Coarse whole-mesh PCA + orientation lock (axis toward the cusp side).
    SparseGrid grid;
    Frame frame;
    if (!coarseFrame(vertices, params, grid, frame)) return result;
    const Eigen::Vector3f meshCentroid = frame.centroid;

    // 3. Re-PCA on the cusp candidates, repeated to converge. Keep the axis sign
    //    aligned across passes so "top" stays the occlusal side.
    std::vector<uint32_t> cands;
    for (int iter = 0; iter < std::max(1, params.refineIters); ++iter) {
        cands = extractCandidates(vertices, frame, grid, params);
        if (cands.size() < 3) break;
        Frame refined;
        if (!pcaFrame(vertices, cands, refined)) break;
        if (refined.axis.dot(frame.axis) < 0.0f) refined.axis = -refined.axis;
        frame = refined;
    }

    // Final candidate set + their centroid = a point on the plane.
    cands = extractCandidates(vertices, frame, grid, params);
    if (cands.size() < 3) return result;
    Eigen::Vector3f cuspCentroid = Eigen::Vector3f::Zero();
    for (uint32_t i : cands) cuspCentroid += vertices[i];
    cuspCentroid /= (float)cands.size();

    // 4. Sign fix (doc rule): the cusp centroid lies on the + side of the plane
    //    relative to the whole-mesh centroid, since cusps stick up off the bulk.
    Eigen::Vector3f normal = frame.axis;
    if ((cuspCentroid - meshCentroid).dot(normal) < 0.0f) normal = -normal;

    result.position = cuspCentroid;
    result.normal = normal.normalized();
    result.valid = true;
    return result;
}

OcclusalPlane estimateOcclusalPlane(const std::vector<Triangle>& mesh,
                                    const OcclusalPlaneParams& params) {
    // Dedup the soup's corners onto a quantized grid so dense regions don't bias
    // the PCA/centroid by triangle density.
    std::vector<Eigen::Vector3f> verts;
    if (mesh.empty()) return {};

    Eigen::Vector3f mn = mesh[0].v[0], mx = mesh[0].v[0];
    for (const auto& t : mesh)
        for (int i = 0; i < 3; ++i) { mn = mn.cwiseMin(t.v[i]); mx = mx.cwiseMax(t.v[i]); }
    float quant = (mx - mn).norm() * 1e-4f;
    if (quant <= 0.0f) quant = 1e-6f;

    std::unordered_map<uint64_t, uint8_t> seen;
    seen.reserve(mesh.size() * 3);
    verts.reserve(mesh.size() * 3);
    auto cellKey = [&](const Eigen::Vector3f& p) -> uint64_t {
        int64_t x = (int64_t)std::llround(p.x() / quant);
        int64_t y = (int64_t)std::llround(p.y() / quant);
        int64_t z = (int64_t)std::llround(p.z() / quant);
        uint64_t h = (uint64_t)(x * 73856093) ^ (uint64_t)(y * 19349663) ^ (uint64_t)(z * 83492791);
        return h;
    };
    for (const auto& t : mesh)
        for (int i = 0; i < 3; ++i)
            if (seen.emplace(cellKey(t.v[i]), 1).second) verts.push_back(t.v[i]);

    return estimateOcclusalPlane(verts, params);
}

} // namespace orange::geometry
