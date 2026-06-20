#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

#include <Eigen/Core>

#include "orange/core/geometry.h"

// Point octree: 8-way recursive subdivision of an AABB, points stored in leaves.
// Good for region queries (AABB / radius), kNN with node pruning, and frustum/box
// culling of large point clouds. CPU, Eigen-backed, header-only.

namespace orange::geometry {

class Octree {
public:
    void build(const std::vector<Eigen::Vector3f>& points, int leafCapacity = 16,
               int maxDepth = 16) {
        pts_ = &points;
        nodes_.clear();
        cap_ = std::max(1, leafCapacity);
        maxDepth_ = maxDepth;
        if (points.empty()) return;

        AABB root;
        for (const auto& p : points) root.expand(p);
        // Pad so points on the max face still fall inside.
        Eigen::Vector3f pad = (root.max - root.min) * 1e-4f + Eigen::Vector3f::Constant(1e-6f);
        root.min -= pad; root.max += pad;

        std::vector<int> all(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i) all[i] = i;
        nodes_.push_back({});
        buildNode(0, root, all, 0);
    }

    bool empty() const { return nodes_.empty(); }

    // AABB of every node (for visualization).
    void nodeBoxes(std::vector<AABB>& out) const {
        out.clear(); out.reserve(nodes_.size());
        for (const auto& n : nodes_) out.push_back(n.box);
    }

    // AABB of leaf nodes only (cleaner visualization).
    void leafBoxes(std::vector<AABB>& out) const {
        out.clear();
        for (const auto& n : nodes_) if (n.leaf) out.push_back(n.box);
    }

    // Every node's AABB plus its tree depth (root = 0) for per-level coloring.
    // Recursion stops at maxDepth (nodes at that depth are emitted but not split).
    void nodeBoxesDepth(std::vector<AABB>& boxes, std::vector<int>& depth,
                        int maxDepth = 1 << 30) const {
        boxes.clear(); depth.clear();
        if (nodes_.empty()) return;
        std::vector<std::pair<int, int>> st;  // (node, depth)
        st.push_back({0, 0});
        while (!st.empty()) {
            auto cur = st.back(); st.pop_back();
            const Node& n = nodes_[cur.first];
            boxes.push_back(n.box); depth.push_back(cur.second);
            if (!n.leaf && cur.second < maxDepth)
                for (int o = 0; o < 8; ++o)
                    if (n.child[o] >= 0) st.push_back({n.child[o], cur.second + 1});
        }
    }

    // Indices of points within `radius` of `center`.
    void radiusQuery(const Eigen::Vector3f& center, float radius,
                     std::vector<int>& out) const {
        out.clear();
        if (nodes_.empty()) return;
        float r2 = radius * radius;
        recurseRadius(0, center, radius, r2, out);
    }

    // Indices of points inside the axis-aligned box [mn, mx].
    void aabbQuery(const Eigen::Vector3f& mn, const Eigen::Vector3f& mx,
                   std::vector<int>& out) const {
        out.clear();
        if (nodes_.empty()) return;
        AABB q; q.min = mn; q.max = mx;
        recurseAABB(0, q, out);
    }

    // k nearest point indices to `query`, nearest first.
    void kNearest(const Eigen::Vector3f& query, int k, std::vector<int>& out) const {
        out.clear();
        if (nodes_.empty() || k <= 0) return;
        // Max-heap of (dist2, idx); keep the k smallest.
        std::priority_queue<std::pair<float, int>> heap;
        recurseKnn(0, query, k, heap);
        std::vector<std::pair<float, int>> tmp;
        tmp.reserve(heap.size());
        while (!heap.empty()) { tmp.push_back(heap.top()); heap.pop(); }
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& e : tmp) out.push_back(e.second);
    }

private:
    struct Node {
        AABB box;
        int  child[8];        // child node indices, -1 if none
        std::vector<int> pts; // point indices (leaf only)
        bool leaf = true;
        Node() { for (int i = 0; i < 8; ++i) child[i] = -1; }
    };

    void buildNode(int idx, const AABB& box, std::vector<int>& items, int depth) {
        nodes_[idx].box = box;
        if (static_cast<int>(items.size()) <= cap_ || depth >= maxDepth_) {
            nodes_[idx].leaf = true;
            nodes_[idx].pts = std::move(items);
            return;
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

    // Squared distance from a point to an AABB (0 if inside).
    static float dist2ToBox(const AABB& b, const Eigen::Vector3f& p) {
        Eigen::Vector3f q = p.cwiseMax(b.min).cwiseMin(b.max);
        return (q - p).squaredNorm();
    }

    void recurseRadius(int idx, const Eigen::Vector3f& center, float radius, float r2,
                       std::vector<int>& out) const {
        const Node& n = nodes_[idx];
        if (dist2ToBox(n.box, center) > r2) return;
        if (n.leaf) {
            for (int i : n.pts)
                if (((*pts_)[i] - center).squaredNorm() <= r2) out.push_back(i);
            return;
        }
        for (int o = 0; o < 8; ++o)
            if (n.child[o] >= 0) recurseRadius(n.child[o], center, radius, r2, out);
    }

    void recurseAABB(int idx, const AABB& q, std::vector<int>& out) const {
        const Node& n = nodes_[idx];
        if (!n.box.intersects(q)) return;
        if (n.leaf) {
            for (int i : n.pts)
                if (q.contains((*pts_)[i])) out.push_back(i);
            return;
        }
        for (int o = 0; o < 8; ++o)
            if (n.child[o] >= 0) recurseAABB(n.child[o], q, out);
    }

    void recurseKnn(int idx, const Eigen::Vector3f& query, int k,
                    std::priority_queue<std::pair<float, int>>& heap) const {
        const Node& n = nodes_[idx];
        if (static_cast<int>(heap.size()) >= k && dist2ToBox(n.box, query) > heap.top().first)
            return;
        if (n.leaf) {
            for (int i : n.pts) {
                float d2 = ((*pts_)[i] - query).squaredNorm();
                if (static_cast<int>(heap.size()) < k) heap.push({d2, i});
                else if (d2 < heap.top().first) { heap.pop(); heap.push({d2, i}); }
            }
            return;
        }
        // Visit children nearest-first for better pruning.
        std::pair<float, int> order[8];
        int m = 0;
        for (int o = 0; o < 8; ++o)
            if (n.child[o] >= 0) order[m++] = {dist2ToBox(nodes_[n.child[o]].box, query), n.child[o]};
        std::sort(order, order + m, [](const auto& a, const auto& b) { return a.first < b.first; });
        for (int j = 0; j < m; ++j) {
            if (static_cast<int>(heap.size()) >= k && order[j].first > heap.top().first) break;
            recurseKnn(order[j].second, query, k, heap);
        }
    }

    const std::vector<Eigen::Vector3f>* pts_ = nullptr;
    std::vector<Node> nodes_;
    int cap_ = 16;
    int maxDepth_ = 16;
};

} // namespace orange::geometry
