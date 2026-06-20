#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "orange/core/geometry.h"

// Axis-aligned BSP tree over points: at each node, split the *space* in half along
// the longest axis of the node's box (spatial median), partitioning the points.
// Unlike the k-d tree (splits at a point on the alternating axis), this splits at
// the geometric midpoint of the dominant axis -> balanced cells, not balanced
// counts. CPU, Eigen-backed, header-only.

namespace orange::geometry {

class BSP {
public:
    void build(const std::vector<Eigen::Vector3f>& points, int leafCapacity = 16,
               int maxDepth = 32) {
        pts_ = &points;
        nodes_.clear();
        order_.resize(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i) order_[i] = i;
        cap_ = std::max(1, leafCapacity);
        maxDepth_ = maxDepth;
        if (points.empty()) return;
        AABB root;
        for (const auto& p : points) root.expand(p);
        nodes_.push_back({});
        buildNode(0, 0, static_cast<int>(points.size()), root, 0);
    }

    bool empty() const { return nodes_.empty(); }

    void radiusQuery(const Eigen::Vector3f& center, float radius, std::vector<int>& out) const {
        out.clear();
        if (nodes_.empty()) return;
        recurse(0, center, radius * radius, out);
    }

    // Node region box + depth (root = 0) up to maxDepth, for per-level coloring.
    void nodeBoxesDepth(std::vector<AABB>& boxes, std::vector<int>& depth,
                        int maxDepth = 1 << 30) const {
        boxes.clear(); depth.clear();
        if (nodes_.empty()) return;
        std::vector<std::pair<int, int>> st; st.push_back({0, 0});
        while (!st.empty()) {
            auto cur = st.back(); st.pop_back();
            const Node& n = nodes_[cur.first];
            boxes.push_back(n.box); depth.push_back(cur.second);
            if (!n.leaf() && cur.second < maxDepth) {
                st.push_back({n.left, cur.second + 1});
                st.push_back({n.right, cur.second + 1});
            }
        }
    }

private:
    struct Node {
        AABB box;
        int  axis = 0; float split = 0.0f;
        int  left = -1, right = -1;
        int  start = 0, count = 0;
        bool leaf() const { return left < 0; }
    };

    void buildNode(int idx, int begin, int end, const AABB& box, int depth) {
        nodes_[idx].box = box;
        int count = end - begin;
        if (count <= cap_ || depth >= maxDepth_) {
            nodes_[idx].left = -1; nodes_[idx].start = begin; nodes_[idx].count = count;
            return;
        }
        Eigen::Vector3f ext = box.max - box.min;
        int axis = (ext.x() >= ext.y() && ext.x() >= ext.z()) ? 0 : (ext.y() >= ext.z() ? 1 : 2);
        float split = (box.min[axis] + box.max[axis]) * 0.5f;
        int mid = static_cast<int>(std::partition(order_.begin() + begin, order_.begin() + end,
                      [&](int i) { return (*pts_)[i][axis] < split; }) - order_.begin());
        if (mid == begin || mid == end) {  // degenerate (all on one side) -> leaf
            nodes_[idx].left = -1; nodes_[idx].start = begin; nodes_[idx].count = count;
            return;
        }
        AABB lb = box, rb = box;
        lb.max[axis] = split; rb.min[axis] = split;
        nodes_[idx].axis = axis; nodes_[idx].split = split;
        int l = static_cast<int>(nodes_.size()); nodes_.push_back({});
        buildNode(l, begin, mid, lb, depth + 1);
        int r = static_cast<int>(nodes_.size()); nodes_.push_back({});
        buildNode(r, mid, end, rb, depth + 1);
        nodes_[idx].left = l; nodes_[idx].right = r;
    }

    static float dist2ToBox(const AABB& b, const Eigen::Vector3f& p) {
        Eigen::Vector3f q = p.cwiseMax(b.min).cwiseMin(b.max);
        return (q - p).squaredNorm();
    }

    void recurse(int idx, const Eigen::Vector3f& center, float r2, std::vector<int>& out) const {
        const Node& n = nodes_[idx];
        if (dist2ToBox(n.box, center) > r2) return;
        if (n.leaf()) {
            for (int i = 0; i < n.count; ++i) {
                int p = order_[n.start + i];
                if (((*pts_)[p] - center).squaredNorm() <= r2) out.push_back(p);
            }
            return;
        }
        recurse(n.left, center, r2, out);
        recurse(n.right, center, r2, out);
    }

    const std::vector<Eigen::Vector3f>* pts_ = nullptr;
    std::vector<Node> nodes_;
    std::vector<int>  order_;
    int cap_ = 16; int maxDepth_ = 32;
};

} // namespace orange::geometry
