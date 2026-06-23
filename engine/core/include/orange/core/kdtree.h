#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

#include <Eigen/Core>

#include "orange/core/geometry.h"

// 3D k-d tree over a point set: median-split on the alternating axis. Strong for
// nearest / kNN / radius queries on non-uniform clouds (where a uniform hash grid
// degrades). CPU, Eigen-backed, header-only.

namespace orange::geometry {

class KDTree {
public:
    void build(const std::vector<Eigen::Vector3f>& points) {
        pts_ = &points;
        nodes_.clear();
        idx_.resize(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i) idx_[i] = i;
        root_ = points.empty() ? -1 : buildRange(0, static_cast<int>(points.size()), 0);
    }

    // `points` is REFERENCED, not copied -- keep it alive while querying. If you
    // built against a temporary (e.g. on a worker thread), rebind() to a stable
    // array of identical contents before querying. O(1).
    void rebind(const std::vector<Eigen::Vector3f>& points) { pts_ = &points; }

    bool empty() const { return root_ < 0; }

    // Leaf-cell AABBs: recursively split `root` by each node's plane (for
    // visualization). `root` should be the cloud's bounding box.
    void cellBoxes(const geometry::AABB& root, std::vector<geometry::AABB>& out) const {
        out.clear();
        if (root_ >= 0) recurseCells(root_, root, out);
    }

    // Every node's region box plus its depth (root = 0) for per-level coloring.
    // Recursion stops at maxDepth (a k-d tree has one node per point, so an
    // unbounded walk is impractical to visualize).
    void cellBoxesDepth(const geometry::AABB& root, std::vector<geometry::AABB>& boxes,
                        std::vector<int>& depth, int maxDepth = 1 << 30) const {
        boxes.clear(); depth.clear();
        if (root_ >= 0) recurseCellsDepth(root_, root, 0, maxDepth, boxes, depth);
    }

    // Nearest point index to `query` (-1 if the tree is empty).
    int nearest(const Eigen::Vector3f& query, float* outDist2 = nullptr) const {
        int best = -1; float bestD2 = 1e30f;
        if (root_ >= 0) recurseNearest(root_, query, best, bestD2);
        if (outDist2) *outDist2 = bestD2;
        return best;
    }

    // k nearest point indices, nearest first.
    void kNearest(const Eigen::Vector3f& query, int k, std::vector<int>& out) const {
        out.clear();
        if (root_ < 0 || k <= 0) return;
        std::priority_queue<std::pair<float, int>> heap;  // max-heap of k smallest
        recurseKnn(root_, query, k, heap);
        std::vector<std::pair<float, int>> tmp;
        while (!heap.empty()) { tmp.push_back(heap.top()); heap.pop(); }
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& e : tmp) out.push_back(e.second);
    }

    // Indices of points within `radius` of `query`.
    void radiusQuery(const Eigen::Vector3f& query, float radius, std::vector<int>& out) const {
        out.clear();
        if (root_ >= 0) recurseRadius(root_, query, radius * radius, out);
    }

private:
    struct Node {
        int point;        // index into the point set
        int axis;         // split axis 0/1/2
        int left = -1, right = -1;
    };

    int buildRange(int begin, int end, int depth) {
        if (begin >= end) return -1;
        int axis = depth % 3;
        int mid = (begin + end) / 2;
        std::nth_element(idx_.begin() + begin, idx_.begin() + mid, idx_.begin() + end,
                         [&](int a, int b) { return (*pts_)[a][axis] < (*pts_)[b][axis]; });
        int nodeIdx = static_cast<int>(nodes_.size());
        nodes_.push_back({idx_[mid], axis, -1, -1});
        int l = buildRange(begin, mid, depth + 1);
        int r = buildRange(mid + 1, end, depth + 1);
        nodes_[nodeIdx].left = l;
        nodes_[nodeIdx].right = r;
        return nodeIdx;
    }

    void recurseNearest(int ni, const Eigen::Vector3f& q, int& best, float& bestD2) const {
        const Node& n = nodes_[ni];
        const Eigen::Vector3f& p = (*pts_)[n.point];
        float d2 = (p - q).squaredNorm();
        if (d2 < bestD2) { bestD2 = d2; best = n.point; }
        float diff = q[n.axis] - p[n.axis];
        int near = diff < 0 ? n.left : n.right;
        int far  = diff < 0 ? n.right : n.left;
        if (near >= 0) recurseNearest(near, q, best, bestD2);
        if (far >= 0 && diff * diff < bestD2) recurseNearest(far, q, best, bestD2);
    }

    void recurseKnn(int ni, const Eigen::Vector3f& q, int k,
                    std::priority_queue<std::pair<float, int>>& heap) const {
        const Node& n = nodes_[ni];
        const Eigen::Vector3f& p = (*pts_)[n.point];
        float d2 = (p - q).squaredNorm();
        if (static_cast<int>(heap.size()) < k) heap.push({d2, n.point});
        else if (d2 < heap.top().first) { heap.pop(); heap.push({d2, n.point}); }
        float diff = q[n.axis] - p[n.axis];
        int near = diff < 0 ? n.left : n.right;
        int far  = diff < 0 ? n.right : n.left;
        if (near >= 0) recurseKnn(near, q, k, heap);
        if (far >= 0 &&
            (static_cast<int>(heap.size()) < k || diff * diff < heap.top().first))
            recurseKnn(far, q, k, heap);
    }

    void recurseCells(int ni, const geometry::AABB& box,
                      std::vector<geometry::AABB>& out) const {
        const Node& n = nodes_[ni];
        float s = (*pts_)[n.point][n.axis];
        geometry::AABB lb = box, rb = box;
        lb.max[n.axis] = s;
        rb.min[n.axis] = s;
        if (n.left >= 0) recurseCells(n.left, lb, out); else out.push_back(lb);
        if (n.right >= 0) recurseCells(n.right, rb, out); else out.push_back(rb);
    }

    void recurseCellsDepth(int ni, const geometry::AABB& box, int d, int maxDepth,
                           std::vector<geometry::AABB>& boxes, std::vector<int>& depth) const {
        const Node& n = nodes_[ni];
        boxes.push_back(box); depth.push_back(d);
        if (d >= maxDepth) return;
        float s = (*pts_)[n.point][n.axis];
        geometry::AABB lb = box, rb = box;
        lb.max[n.axis] = s;
        rb.min[n.axis] = s;
        if (n.left >= 0) recurseCellsDepth(n.left, lb, d + 1, maxDepth, boxes, depth);
        else { boxes.push_back(lb); depth.push_back(d + 1); }
        if (n.right >= 0) recurseCellsDepth(n.right, rb, d + 1, maxDepth, boxes, depth);
        else { boxes.push_back(rb); depth.push_back(d + 1); }
    }

    void recurseRadius(int ni, const Eigen::Vector3f& q, float r2, std::vector<int>& out) const {
        const Node& n = nodes_[ni];
        const Eigen::Vector3f& p = (*pts_)[n.point];
        if ((p - q).squaredNorm() <= r2) out.push_back(n.point);
        float diff = q[n.axis] - p[n.axis];
        int near = diff < 0 ? n.left : n.right;
        int far  = diff < 0 ? n.right : n.left;
        if (near >= 0) recurseRadius(near, q, r2, out);
        if (far >= 0 && diff * diff <= r2) recurseRadius(far, q, r2, out);
    }

    const std::vector<Eigen::Vector3f>* pts_ = nullptr;
    std::vector<Node> nodes_;
    std::vector<int>  idx_;
    int root_ = -1;
};

} // namespace orange::geometry
