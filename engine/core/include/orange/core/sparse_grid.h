#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <robin_hood/robin_hood.h>

#include "orange/core/geometry.h"  // AABB

// Hash-based sparse uniform grid over a point set, for nearest-neighbour and
// radius queries. Points are bucketed into cells of `cellSize`; each cell holds
// the head of an intrusive linked list (nextPoint) into the point array.
//
// Ported from Elements/Helium (SparseGrid), decoupled from Helium's PointCloud:
// Build() now takes a plain std::vector<Eigen::Vector3f>. CPU-only.

namespace orange::geometry {

class SparseGrid {
public:
    robin_hood::unordered_flat_map<uint64_t, int> voxelPointListHead;
    std::vector<int> nextPoint;

    AABB aabb;
    float cellSize = 0.1f;

    // 21-bit-per-axis packed cell key.
    uint64_t key(int x, int y, int z) const {
        const uint64_t mask = 0x1FFFFF;
        return (((uint64_t)x & mask) << 42) | (((uint64_t)y & mask) << 21) | ((uint64_t)z & mask);
    }

    Eigen::Vector3i index(const Eigen::Vector3f& position) const {
        return Eigen::Vector3i(
            (int)std::floor((position.x() - aabb.min.x()) / cellSize),
            (int)std::floor((position.y() - aabb.min.y()) / cellSize),
            (int)std::floor((position.z() - aabb.min.z()) / cellSize));
    }

    // Bucket all points; aabb is derived from the points (padded by 10% of a cell).
    void build(const std::vector<Eigen::Vector3f>& points, float cellSize);

    int closestPoint(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& queryPos,
                     float& outDist) const;

    void kNearestNeighbors(const std::vector<Eigen::Vector3f>& points,
                           const Eigen::Vector3f& queryPos, int k,
                           std::vector<unsigned int>& outIndices,
                           std::vector<float>& outDistances) const;

    void pointsWithinRadius(const std::vector<Eigen::Vector3f>& points,
                            const Eigen::Vector3f& queryPos, float radius,
                            std::vector<unsigned int>& outIndices,
                            std::vector<float>& outDistances) const;
};

} // namespace orange::geometry
