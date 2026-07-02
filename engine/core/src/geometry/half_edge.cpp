// Half-edge (DCEL) mesh construction + one-ring traversal. See half_edge.h.
// ASCII comments only (pipeline project rule).

#include "orange/core/half_edge.h"

#include <cmath>
#include <unordered_map>

namespace orange::geometry {

bool HalfEdgeMesh::build(const std::vector<Eigen::Vector3f>& positions,
                         const std::vector<uint32_t>& indices, float weldEps) {
    vertices_.clear();
    halfEdges_.clear();
    vertHalf_.clear();
    remap_.clear();
    if (positions.size() < 3 || indices.size() < 3) return false;

    // Bounding box -> auto weld epsilon.
    Eigen::Vector3f mn = positions[0], mx = positions[0];
    for (const auto& p : positions) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    const float diag = (mx - mn).norm();
    const float eps = weldEps > 0.0f ? weldEps : (diag > 0.0f ? diag * 1e-5f : 1e-6f);
    const float inv = 1.0f / (eps > 0.0f ? eps : 1e-6f);

    // Weld coincident positions onto a quantized grid (handles triangle soup).
    auto keyOf = [&](const Eigen::Vector3f& p) -> uint64_t {
        int64_t x = (int64_t)std::llround(p.x() * inv);
        int64_t y = (int64_t)std::llround(p.y() * inv);
        int64_t z = (int64_t)std::llround(p.z() * inv);
        return (uint64_t)(x * 73856093) ^ (uint64_t)(y * 19349663) ^ (uint64_t)(z * 83492791);
    };
    std::unordered_map<uint64_t, uint32_t> weld;
    weld.reserve(positions.size());
    remap_.resize(positions.size());
    vertices_.reserve(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        uint64_t k = keyOf(positions[i]);
        auto it = weld.find(k);
        if (it == weld.end()) {
            uint32_t id = (uint32_t)vertices_.size();
            weld.emplace(k, id);
            vertices_.push_back(positions[i]);
            remap_[i] = id;
        } else {
            remap_[i] = it->second;
        }
    }
    vertHalf_.assign(vertices_.size(), -1);

    // One half-edge per (directed) triangle edge. Twin lookup keys an ordered
    // (from,to) vertex pair to its half-edge so the opposite (to,from) finds it.
    std::unordered_map<uint64_t, uint32_t> edgeMap;
    edgeMap.reserve(indices.size());
    auto edgeKey = [](uint32_t from, uint32_t to) -> uint64_t {
        return ((uint64_t)from << 32) | (uint64_t)to;
    };

    const uint32_t nv = (uint32_t)vertices_.size();
    halfEdges_.reserve(indices.size());
    uint32_t face = 0;
    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        if (indices[t] >= remap_.size() || indices[t + 1] >= remap_.size() ||
            indices[t + 2] >= remap_.size())
            continue;
        uint32_t v[3] = {remap_[indices[t]], remap_[indices[t + 1]], remap_[indices[t + 2]]};
        if (v[0] == v[1] || v[1] == v[2] || v[0] == v[2]) continue;  // degenerate
        if (v[0] >= nv || v[1] >= nv || v[2] >= nv) continue;

        uint32_t base = (uint32_t)halfEdges_.size();
        for (int k = 0; k < 3; ++k) {
            HalfEdge he;
            he.origin = v[k];
            he.next = base + (uint32_t)((k + 1) % 3);
            he.prev = base + (uint32_t)((k + 2) % 3);
            he.twin = -1;
            he.face = face;
            halfEdges_.push_back(he);
            if (vertHalf_[v[k]] < 0) vertHalf_[v[k]] = (int32_t)(base + k);
            edgeMap[edgeKey(v[k], v[(k + 1) % 3])] = base + k;
        }
        ++face;
    }

    // Resolve twins: each half-edge (from->to) pairs with (to->from) if present.
    for (uint32_t h = 0; h < halfEdges_.size(); ++h) {
        uint32_t from = halfEdges_[h].origin;
        uint32_t to = halfEdges_[halfEdges_[h].next].origin;
        auto it = edgeMap.find(edgeKey(to, from));
        if (it != edgeMap.end()) halfEdges_[h].twin = (int32_t)it->second;
    }

    // Prefer a boundary half-edge as each vertex's representative, so the one-ring
    // walk starts at the open end of a boundary fan and covers it fully.
    for (uint32_t h = 0; h < halfEdges_.size(); ++h) {
        if (halfEdges_[h].twin < 0) {
            uint32_t from = halfEdges_[h].origin;
            vertHalf_[from] = (int32_t)h;
        }
    }
    return true;
}

bool HalfEdgeMesh::isBoundary(uint32_t v) const {
    if (v >= vertHalf_.size() || vertHalf_[v] < 0) return false;
    int32_t start = vertHalf_[v];
    int32_t h = start;
    int guard = 0;
    do {
        if (halfEdges_[h].twin < 0) return true;
        h = halfEdges_[(uint32_t)halfEdges_[h].twin].next;
        if (++guard > 100000) break;
    } while (h != start && h >= 0);
    return false;
}

void HalfEdgeMesh::oneRing(uint32_t v, std::vector<uint32_t>& out) const {
    out.clear();
    if (v >= vertHalf_.size() || vertHalf_[v] < 0) return;
    const int32_t start = vertHalf_[v];

    // CCW fan: each outgoing half-edge h contributes its destination vertex; step
    // to the next outgoing edge via twin->next. Stops at a boundary or full loop.
    int32_t h = start;
    int guard = 0;
    bool boundary = false;
    do {
        out.push_back(halfEdges_[(uint32_t)halfEdges_[h].next].origin);  // dest(h)
        int32_t tw = halfEdges_[h].twin;
        if (tw < 0) { boundary = true; break; }
        h = (int32_t)halfEdges_[(uint32_t)tw].next;
        if (++guard > 100000) break;
    } while (h != start);

    if (!boundary) return;

    // Open fan: walk the other direction (CW) from start to gather the rest.
    int32_t g = start;
    guard = 0;
    while (true) {
        int32_t p = (int32_t)halfEdges_[(uint32_t)g].prev;
        int32_t tw = halfEdges_[(uint32_t)p].twin;
        if (tw < 0) break;
        g = tw;
        out.push_back(halfEdges_[(uint32_t)halfEdges_[(uint32_t)g].next].origin);  // dest(g)
        if (++guard > 100000) break;
    }
}

std::vector<uint32_t> HalfEdgeMesh::oneRing(uint32_t v) const {
    std::vector<uint32_t> out;
    oneRing(v, out);
    return out;
}

} // namespace orange::geometry
