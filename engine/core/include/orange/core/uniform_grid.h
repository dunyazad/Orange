#pragma once

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "orange/core/geometry.h"

// Uniform grid: space diced into equal cells, points bucketed by a hashed integer
// cell key. The simplest accelerator -- O(1) cell lookup, best when point density
// is roughly uniform. CPU, Eigen-backed, header-only.

namespace orange::geometry {

class UniformGrid {
public:
    void build(const std::vector<Eigen::Vector3f>& points, float cellSize) {
        pts_ = &points;
        cells_.clear();
        cell_ = cellSize > 1e-9f ? cellSize : 1.0f;
        origin_ = Eigen::Vector3f::Zero();
        if (points.empty()) return;
        AABB b; for (const auto& p : points) b.expand(p);
        origin_ = b.min;
        for (int i = 0; i < static_cast<int>(points.size()); ++i)
            cells_[key(cellOf(points[i]))].push_back(i);
    }

    bool empty() const { return cells_.empty(); }

    // Indices within `radius` of `center` (scans the overlapped cell range).
    void radiusQuery(const Eigen::Vector3f& center, float radius, std::vector<int>& out) const {
        out.clear();
        if (cells_.empty()) return;
        float r2 = radius * radius;
        Eigen::Vector3i lo = cellOf(center - Eigen::Vector3f::Constant(radius));
        Eigen::Vector3i hi = cellOf(center + Eigen::Vector3f::Constant(radius));
        for (int x = lo.x(); x <= hi.x(); ++x)
            for (int y = lo.y(); y <= hi.y(); ++y)
                for (int z = lo.z(); z <= hi.z(); ++z) {
                    auto it = cells_.find(key({x, y, z}));
                    if (it == cells_.end()) continue;
                    for (int i : it->second)
                        if (((*pts_)[i] - center).squaredNorm() <= r2) out.push_back(i);
                }
    }

    // AABB of each occupied cell (for visualization).
    void occupiedCellBoxes(std::vector<AABB>& out) const {
        out.clear(); out.reserve(cells_.size());
        for (const auto& kv : cells_) {
            Eigen::Vector3i c = unkey(kv.first);
            AABB b;
            b.min = origin_ + c.cast<float>() * cell_;
            b.max = b.min + Eigen::Vector3f::Constant(cell_);
            out.push_back(b);
        }
    }

    float cellSize() const { return cell_; }

private:
    Eigen::Vector3i cellOf(const Eigen::Vector3f& p) const {
        Eigen::Vector3f q = (p - origin_) / cell_;
        return Eigen::Vector3i(static_cast<int>(std::floor(q.x())),
                               static_cast<int>(std::floor(q.y())),
                               static_cast<int>(std::floor(q.z())));
    }
    static int64_t key(const Eigen::Vector3i& c) {
        // Pack 3x21 bits (offset to keep negatives non-negative).
        int64_t x = (c.x() + (1 << 20)) & 0x1fffff;
        int64_t y = (c.y() + (1 << 20)) & 0x1fffff;
        int64_t z = (c.z() + (1 << 20)) & 0x1fffff;
        return x | (y << 21) | (z << 42);
    }
    static Eigen::Vector3i unkey(int64_t k) {
        int x = static_cast<int>(k & 0x1fffff) - (1 << 20);
        int y = static_cast<int>((k >> 21) & 0x1fffff) - (1 << 20);
        int z = static_cast<int>((k >> 42) & 0x1fffff) - (1 << 20);
        return Eigen::Vector3i(x, y, z);
    }

    const std::vector<Eigen::Vector3f>* pts_ = nullptr;
    std::unordered_map<int64_t, std::vector<int>> cells_;
    Eigen::Vector3f origin_ = Eigen::Vector3f::Zero();
    float cell_ = 1.0f;
};

} // namespace orange::geometry
