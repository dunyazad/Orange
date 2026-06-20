#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "orange/core/geometry.h"

// Packed R-tree (Sort-Tile-Recursive bulk load) over points. Nodes hold up to M
// children and a minimum bounding rectangle (MBR); good as a static index for
// AABB / radius queries when you want balanced fan-out rather than binary splits.
// CPU, Eigen-backed, header-only.

namespace orange::geometry {

class RTree {
public:
    void build(const std::vector<Eigen::Vector3f>& points, int maxEntries = 8) {
        pts_ = &points;
        nodes_.clear();
        root_ = -1;
        M_ = std::max(2, maxEntries);
        if (points.empty()) return;

        std::vector<int> items(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i) items[i] = i;
        std::vector<int> level = packLeaves(items);
        while (level.size() > 1) level = packInternal(level);
        root_ = level.empty() ? -1 : level[0];
    }

    bool empty() const { return root_ < 0; }

    void radiusQuery(const Eigen::Vector3f& center, float radius, std::vector<int>& out) const {
        out.clear();
        if (root_ >= 0) recurse(root_, center, radius * radius, out);
    }

    void aabbQuery(const Eigen::Vector3f& mn, const Eigen::Vector3f& mx,
                   std::vector<int>& out) const {
        out.clear();
        if (root_ < 0) return;
        AABB q; q.min = mn; q.max = mx;
        recurseAABB(root_, q, out);
    }

    // Node MBR + depth (root = 0) up to maxDepth, for per-level coloring.
    void nodeBoxesDepth(std::vector<AABB>& boxes, std::vector<int>& depth,
                        int maxDepth = 1 << 30) const {
        boxes.clear(); depth.clear();
        if (root_ < 0) return;
        std::vector<std::pair<int, int>> st; st.push_back({root_, 0});
        while (!st.empty()) {
            auto cur = st.back(); st.pop_back();
            const Node& n = nodes_[cur.first];
            boxes.push_back(n.mbr); depth.push_back(cur.second);
            if (!n.leaf && cur.second < maxDepth)
                for (int c : n.children) st.push_back({c, cur.second + 1});
        }
    }

private:
    struct Node {
        AABB mbr;
        std::vector<int> children;  // point indices (leaf) or node indices (internal)
        bool leaf = true;
    };

    Eigen::Vector3f pointCenter(int i) const { return (*pts_)[i]; }
    Eigen::Vector3f nodeCenter(int n) const {
        return (nodes_[n].mbr.min + nodes_[n].mbr.max) * 0.5f;
    }

    // Sort-Tile-Recursive grouping of `ids` into chunks of <= M_, by `center`.
    template <typename CenterFn>
    std::vector<std::vector<int>> strGroups(std::vector<int> ids, CenterFn center) const {
        int n = static_cast<int>(ids.size());
        int leaves = (n + M_ - 1) / M_;
        int slices = std::max(1, static_cast<int>(std::ceil(std::sqrt((double)leaves))));
        std::sort(ids.begin(), ids.end(),
                  [&](int a, int b) { return center(a).x() < center(b).x(); });
        std::vector<std::vector<int>> groups;
        int perSlice = slices * M_;
        for (int s = 0; s < n; s += perSlice) {
            int se = std::min(s + perSlice, n);
            std::sort(ids.begin() + s, ids.begin() + se,
                      [&](int a, int b) { return center(a).y() < center(b).y(); });
            for (int g = s; g < se; g += M_) {
                int ge = std::min(g + M_, se);
                groups.emplace_back(ids.begin() + g, ids.begin() + ge);
            }
        }
        return groups;
    }

    std::vector<int> packLeaves(const std::vector<int>& pointIds) {
        auto groups = strGroups(pointIds, [&](int i) { return pointCenter(i); });
        std::vector<int> out;
        for (auto& g : groups) {
            Node nd; nd.leaf = true; nd.children = g;
            for (int i : g) nd.mbr.expand((*pts_)[i]);
            out.push_back(static_cast<int>(nodes_.size()));
            nodes_.push_back(std::move(nd));
        }
        return out;
    }

    std::vector<int> packInternal(const std::vector<int>& childNodes) {
        auto groups = strGroups(childNodes, [&](int n) { return nodeCenter(n); });
        std::vector<int> out;
        for (auto& g : groups) {
            Node nd; nd.leaf = false; nd.children = g;
            for (int c : g) nd.mbr.expand(nodes_[c].mbr);
            out.push_back(static_cast<int>(nodes_.size()));
            nodes_.push_back(std::move(nd));
        }
        return out;
    }

    static float dist2ToBox(const AABB& b, const Eigen::Vector3f& p) {
        Eigen::Vector3f q = p.cwiseMax(b.min).cwiseMin(b.max);
        return (q - p).squaredNorm();
    }

    void recurse(int ni, const Eigen::Vector3f& center, float r2, std::vector<int>& out) const {
        const Node& n = nodes_[ni];
        if (dist2ToBox(n.mbr, center) > r2) return;
        if (n.leaf) {
            for (int i : n.children)
                if (((*pts_)[i] - center).squaredNorm() <= r2) out.push_back(i);
        } else {
            for (int c : n.children) recurse(c, center, r2, out);
        }
    }

    void recurseAABB(int ni, const AABB& q, std::vector<int>& out) const {
        const Node& n = nodes_[ni];
        if (!n.mbr.intersects(q)) return;
        if (n.leaf) {
            for (int i : n.children)
                if (q.contains((*pts_)[i])) out.push_back(i);
        } else {
            for (int c : n.children) recurseAABB(c, q, out);
        }
    }

    const std::vector<Eigen::Vector3f>* pts_ = nullptr;
    std::vector<Node> nodes_;
    int root_ = -1;
    int M_ = 8;
};

} // namespace orange::geometry
