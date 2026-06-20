#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

#include <Eigen/Core>

#include "orange/core/geometry.h"

// Ball tree: recursive bounding *spheres*. At each node, split the points by the
// axis of largest spread at the median, recurse. Sphere bounds suit radius / kNN
// queries on clustered or non-axis-aligned data better than AABB trees.
// CPU, Eigen-backed, header-only.

namespace orange::geometry {

class BallTree {
public:
    void build(const std::vector<Eigen::Vector3f>& points, int leafCapacity = 16) {
        pts_ = &points;
        nodes_.clear();
        order_.resize(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i) order_[i] = i;
        cap_ = std::max(1, leafCapacity);
        root_ = points.empty() ? -1 : buildNode(0, static_cast<int>(points.size()));
    }

    bool empty() const { return root_ < 0; }

    void radiusQuery(const Eigen::Vector3f& q, float radius, std::vector<int>& out) const {
        out.clear();
        if (root_ >= 0) recurseRadius(root_, q, radius, radius * radius, out);
    }

    void kNearest(const Eigen::Vector3f& q, int k, std::vector<int>& out) const {
        out.clear();
        if (root_ < 0 || k <= 0) return;
        std::priority_queue<std::pair<float, int>> heap;  // max-heap of k smallest
        recurseKnn(root_, q, k, heap);
        std::vector<std::pair<float, int>> tmp;
        while (!heap.empty()) { tmp.push_back(heap.top()); heap.pop(); }
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& e : tmp) out.push_back(e.second);
    }

    // Node bounding spheres (center,radius) + depth, up to maxDepth.
    void nodeSpheresDepth(std::vector<Eigen::Vector3f>& centers, std::vector<float>& radii,
                          std::vector<int>& depth, int maxDepth = 1 << 30) const {
        centers.clear(); radii.clear(); depth.clear();
        if (root_ < 0) return;
        std::vector<std::pair<int, int>> st; st.push_back({root_, 0});
        while (!st.empty()) {
            auto cur = st.back(); st.pop_back();
            const Node& n = nodes_[cur.first];
            centers.push_back(n.center); radii.push_back(n.radius); depth.push_back(cur.second);
            if (n.left >= 0 && cur.second < maxDepth) {
                st.push_back({n.left, cur.second + 1});
                st.push_back({n.right, cur.second + 1});
            }
        }
    }

private:
    struct Node {
        Eigen::Vector3f center;
        float radius = 0.0f;
        int left = -1, right = -1;
        int start = 0, count = 0;  // leaf range into order_
    };

    int buildNode(int begin, int end) {
        int idx = static_cast<int>(nodes_.size());
        nodes_.push_back({});
        // Bounding sphere: centroid + max distance.
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        for (int i = begin; i < end; ++i) c += (*pts_)[order_[i]];
        c /= float(end - begin);
        float r2 = 0.0f;
        for (int i = begin; i < end; ++i)
            r2 = std::max(r2, ((*pts_)[order_[i]] - c).squaredNorm());
        nodes_[idx].center = c;
        nodes_[idx].radius = std::sqrt(r2);

        if (end - begin <= cap_) {
            nodes_[idx].left = -1; nodes_[idx].start = begin; nodes_[idx].count = end - begin;
            return idx;
        }
        // Split on the axis of largest spread at the median.
        Eigen::Vector3f mn = (*pts_)[order_[begin]], mx = mn;
        for (int i = begin; i < end; ++i) {
            mn = mn.cwiseMin((*pts_)[order_[i]]); mx = mx.cwiseMax((*pts_)[order_[i]]);
        }
        Eigen::Vector3f ext = mx - mn;
        int axis = (ext.x() >= ext.y() && ext.x() >= ext.z()) ? 0 : (ext.y() >= ext.z() ? 1 : 2);
        int mid = (begin + end) / 2;
        std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
                         [&](int a, int b) { return (*pts_)[a][axis] < (*pts_)[b][axis]; });
        int l = buildNode(begin, mid);
        int r = buildNode(mid, end);
        nodes_[idx].left = l; nodes_[idx].right = r;
        return idx;
    }

    void recurseRadius(int ni, const Eigen::Vector3f& q, float radius, float r2,
                       std::vector<int>& out) const {
        const Node& n = nodes_[ni];
        if ((q - n.center).norm() - n.radius > radius) return;  // sphere outside query
        if (n.left < 0) {
            for (int i = 0; i < n.count; ++i) {
                int p = order_[n.start + i];
                if (((*pts_)[p] - q).squaredNorm() <= r2) out.push_back(p);
            }
            return;
        }
        recurseRadius(n.left, q, radius, r2, out);
        recurseRadius(n.right, q, radius, r2, out);
    }

    void recurseKnn(int ni, const Eigen::Vector3f& q, int k,
                    std::priority_queue<std::pair<float, int>>& heap) const {
        const Node& n = nodes_[ni];
        if (static_cast<int>(heap.size()) >= k) {
            float dmin = std::max(0.0f, (q - n.center).norm() - n.radius);
            if (dmin * dmin > heap.top().first) return;
        }
        if (n.left < 0) {
            for (int i = 0; i < n.count; ++i) {
                int p = order_[n.start + i];
                float d2 = ((*pts_)[p] - q).squaredNorm();
                if (static_cast<int>(heap.size()) < k) heap.push({d2, p});
                else if (d2 < heap.top().first) { heap.pop(); heap.push({d2, p}); }
            }
            return;
        }
        float dl = (q - nodes_[n.left].center).norm() - nodes_[n.left].radius;
        float dr = (q - nodes_[n.right].center).norm() - nodes_[n.right].radius;
        if (dl < dr) { recurseKnn(n.left, q, k, heap); recurseKnn(n.right, q, k, heap); }
        else { recurseKnn(n.right, q, k, heap); recurseKnn(n.left, q, k, heap); }
    }

    const std::vector<Eigen::Vector3f>* pts_ = nullptr;
    std::vector<Node> nodes_;
    std::vector<int>  order_;
    int cap_ = 16;
    int root_ = -1;
};

} // namespace orange::geometry
