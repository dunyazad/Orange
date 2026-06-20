#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "orange/core/geometry.h"

// Bounding-Volume Hierarchy over a triangle soup, for O(log n) ray picking
// instead of testing every triangle. Binary AABB tree, median split on the
// longest centroid axis; array-stored nodes, stackless-friendly traversal.
// CPU, Eigen-backed, header-only. Lives in orange::geometry beside Ray/AABB.

namespace orange::geometry {

class BVH {
public:
    struct Node {
        AABB box;
        int  left = -1;   // child node index; <0 => leaf
        int  right = -1;  // child node index
        int  start = 0;   // first triangle (into order_) for a leaf
        int  count = 0;   // triangle count for a leaf
        bool leaf() const { return left < 0; }
    };

    // Build over `indices` (triangle list, 3 per face) referencing `positions`.
    // Both spans are referenced, not copied -- keep them alive while querying.
    void build(const std::vector<Eigen::Vector3f>& positions,
               const std::vector<uint32_t>& indices) {
        pos_ = &positions;
        idx_ = &indices;
        nodes_.clear();
        order_.clear();
        const int triCount = static_cast<int>(indices.size() / 3);
        if (triCount == 0) return;

        order_.resize(triCount);
        centroid_.resize(triCount);
        triBox_.resize(triCount);
        for (int t = 0; t < triCount; ++t) {
            order_[t] = t;
            const Eigen::Vector3f& a = positions[indices[t * 3 + 0]];
            const Eigen::Vector3f& b = positions[indices[t * 3 + 1]];
            const Eigen::Vector3f& c = positions[indices[t * 3 + 2]];
            AABB box; box.expand(a); box.expand(b); box.expand(c);
            triBox_[t] = box;
            centroid_[t] = (a + b + c) / 3.0f;
        }
        nodes_.reserve(triCount * 2);
        buildRange(0, triCount);
        centroid_.clear();  // only needed during build
        triBox_.clear();
    }

    bool empty() const { return nodes_.empty(); }

    // AABB of every node (for visualization).
    void nodeBoxes(std::vector<AABB>& out) const {
        out.clear(); out.reserve(nodes_.size());
        for (const auto& n : nodes_) out.push_back(n.box);
    }

    // AABB of leaf nodes only (cleaner visualization than every internal node).
    void leafBoxes(std::vector<AABB>& out) const {
        out.clear();
        for (const auto& n : nodes_) if (n.leaf()) out.push_back(n.box);
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
            if (!n.leaf() && cur.second < maxDepth) {
                st.push_back({n.left, cur.second + 1});
                st.push_back({n.right, cur.second + 1});
            }
        }
    }

    // Nearest forward triangle hit. Returns its parametric t in tOut and the
    // triangle index (into indices/3) in triOut. False if the ray misses.
    bool nearestHit(const Ray& ray, float& tOut, int& triOut) const {
        if (nodes_.empty()) return false;
        float best = 1e30f;
        int   bestTri = -1;
        int   stack[64];
        int   sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const Node& n = nodes_[stack[--sp]];
            float tn, tf;
            if (!n.box.intersectRay(ray, tn, tf) || tn > best) continue;
            if (n.leaf()) {
                for (int i = 0; i < n.count; ++i) {
                    int t = order_[n.start + i];
                    const Eigen::Vector3f& a = (*pos_)[(*idx_)[t * 3 + 0]];
                    const Eigen::Vector3f& b = (*pos_)[(*idx_)[t * 3 + 1]];
                    const Eigen::Vector3f& c = (*pos_)[(*idx_)[t * 3 + 2]];
                    float tt;
                    if (ray.intersectTriangle(a, b, c, tt) && tt >= 0.0f && tt < best) {
                        best = tt; bestTri = t;
                    }
                }
            } else if (sp + 2 <= 64) {
                stack[sp++] = n.left;
                stack[sp++] = n.right;
            }
        }
        if (bestTri < 0) return false;
        tOut = best; triOut = bestTri;
        return true;
    }

private:
    static constexpr int kLeafSize = 8;

    int buildRange(int begin, int end) {
        int nodeIdx = static_cast<int>(nodes_.size());
        nodes_.push_back({});
        AABB box;
        for (int i = begin; i < end; ++i) box.expand(triBox_[order_[i]]);
        int count = end - begin;
        if (count <= kLeafSize) {
            Node leaf; leaf.box = box; leaf.left = -1; leaf.right = -1;
            leaf.start = begin; leaf.count = count;
            nodes_[nodeIdx] = leaf;
            return nodeIdx;
        }
        // Split on the longest axis of the centroid bounds at the median.
        AABB cb;
        for (int i = begin; i < end; ++i) cb.expand(centroid_[order_[i]]);
        Eigen::Vector3f ext = cb.max - cb.min;
        int axis = (ext.x() >= ext.y() && ext.x() >= ext.z()) ? 0 : (ext.y() >= ext.z() ? 1 : 2);
        int mid = (begin + end) / 2;
        std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
                         [&](int a, int b) { return centroid_[a][axis] < centroid_[b][axis]; });
        int l = buildRange(begin, mid);
        int r = buildRange(mid, end);
        Node node; node.box = box; node.left = l; node.right = r;
        nodes_[nodeIdx] = node;
        return nodeIdx;
    }

    const std::vector<Eigen::Vector3f>* pos_ = nullptr;
    const std::vector<uint32_t>*        idx_ = nullptr;
    std::vector<Node> nodes_;
    std::vector<int>  order_;     // triangle indices, partitioned per node
    std::vector<Eigen::Vector3f> centroid_;  // build scratch
    std::vector<AABB>            triBox_;     // build scratch
};

} // namespace orange::geometry
