#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "orange/core/geometry.h"

// Loose octree: an octree whose node bounds are enlarged (here 2x about the
// centre) so neighbours overlap. Trades tighter packing for cheaper, jitter-free
// insertion/queries near cell borders (popular for moving objects). The tight
// cells partition space; the loose cells are what queries test against.
// CPU, Eigen-backed, header-only.

namespace orange::geometry {

class LooseOctree {
public:
    void build(const std::vector<Eigen::Vector3f>& points, int leafCapacity = 16,
               int maxDepth = 16, float looseness = 2.0f) {
        pts_ = &points;
        nodes_.clear();
        cap_ = std::max(1, leafCapacity);
        maxDepth_ = maxDepth;
        loose_ = looseness;
        if (points.empty()) return;
        AABB root;
        for (const auto& p : points) root.expand(p);
        Eigen::Vector3f pad = (root.max - root.min) * 1e-4f + Eigen::Vector3f::Constant(1e-6f);
        root.min -= pad; root.max += pad;
        std::vector<int> all(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i) all[i] = i;
        nodes_.push_back({});
        buildNode(0, root, all, 0);
    }

    // `points` is REFERENCED, not copied -- keep it alive while querying. If you
    // built against a temporary (e.g. on a worker thread), rebind() to a stable
    // array of identical contents before querying. O(1).
    void rebind(const std::vector<Eigen::Vector3f>& points) { pts_ = &points; }

    bool empty() const { return nodes_.empty(); }

    void radiusQuery(const Eigen::Vector3f& center, float radius, std::vector<int>& out) const {
        out.clear();
        if (nodes_.empty()) return;
        recurse(0, center, radius * radius, out);
    }

    // Loose (enlarged) box of each node at depth <= maxDepth, with its depth.
    void nodeBoxesDepth(std::vector<AABB>& boxes, std::vector<int>& depth,
                        int maxDepth = 1 << 30) const {
        boxes.clear(); depth.clear();
        if (nodes_.empty()) return;
        std::vector<std::pair<int, int>> st; st.push_back({0, 0});
        while (!st.empty()) {
            auto cur = st.back(); st.pop_back();
            const Node& n = nodes_[cur.first];
            boxes.push_back(loosen(n.box)); depth.push_back(cur.second);
            if (!n.leaf && cur.second < maxDepth)
                for (int o = 0; o < 8; ++o)
                    if (n.child[o] >= 0) st.push_back({n.child[o], cur.second + 1});
        }
    }

private:
    struct Node {
        AABB box; int child[8]; std::vector<int> pts; bool leaf = true;
        Node() { for (int i = 0; i < 8; ++i) child[i] = -1; }
    };

    AABB loosen(const AABB& b) const {
        Eigen::Vector3f c = (b.min + b.max) * 0.5f;
        Eigen::Vector3f h = (b.max - b.min) * 0.5f * loose_;
        AABB r; r.min = c - h; r.max = c + h; return r;
    }

    void buildNode(int idx, const AABB& box, std::vector<int>& items, int depth) {
        nodes_[idx].box = box;
        if (static_cast<int>(items.size()) <= cap_ || depth >= maxDepth_) {
            nodes_[idx].leaf = true; nodes_[idx].pts = std::move(items); return;
        }
        nodes_[idx].leaf = false;
        Eigen::Vector3f c = (box.min + box.max) * 0.5f;
        std::vector<int> buckets[8];
        for (int i : items) {
            const Eigen::Vector3f& p = (*pts_)[i];
            int o = (p.x() > c.x() ? 1 : 0) | (p.y() > c.y() ? 2 : 0) | (p.z() > c.z() ? 4 : 0);
            buckets[o].push_back(i);
        }
        for (int o = 0; o < 8; ++o) {
            if (buckets[o].empty()) continue;
            AABB cb;
            cb.min = Eigen::Vector3f((o & 1) ? c.x() : box.min.x(), (o & 2) ? c.y() : box.min.y(),
                                     (o & 4) ? c.z() : box.min.z());
            cb.max = Eigen::Vector3f((o & 1) ? box.max.x() : c.x(), (o & 2) ? box.max.y() : c.y(),
                                     (o & 4) ? box.max.z() : c.z());
            int ci = static_cast<int>(nodes_.size());
            nodes_[idx].child[o] = ci;
            nodes_.push_back({});
            buildNode(ci, cb, buckets[o], depth + 1);
        }
    }

    static float dist2ToBox(const AABB& b, const Eigen::Vector3f& p) {
        Eigen::Vector3f q = p.cwiseMax(b.min).cwiseMin(b.max);
        return (q - p).squaredNorm();
    }

    void recurse(int idx, const Eigen::Vector3f& center, float r2, std::vector<int>& out) const {
        const Node& n = nodes_[idx];
        if (dist2ToBox(loosen(n.box), center) > r2) return;
        if (n.leaf) {
            for (int i : n.pts)
                if (((*pts_)[i] - center).squaredNorm() <= r2) out.push_back(i);
            return;
        }
        for (int o = 0; o < 8; ++o)
            if (n.child[o] >= 0) recurse(n.child[o], center, r2, out);
    }

    const std::vector<Eigen::Vector3f>* pts_ = nullptr;
    std::vector<Node> nodes_;
    int cap_ = 16; int maxDepth_ = 16; float loose_ = 2.0f;
};

} // namespace orange::geometry
