#include "orange/core/sparse_grid.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <queue>

// Ported from Elements/Helium SparseGrid.cpp. Build() builds the cell -> point
// linked lists; the queries walk shells of cells outward from the query cell and
// stop once the best candidate is provably closer than any unsearched cell.

namespace orange::geometry {

void SparseGrid::build(const std::vector<Eigen::Vector3f>& points, float cellSizeIn) {
    voxelPointListHead.clear();
    nextPoint.clear();
    if (points.empty()) return;

    cellSize = cellSizeIn;

    aabb = AABB{};
    for (const auto& p : points) aabb.expand(p);
    aabb.min -= Eigen::Vector3f::Constant(cellSize * 0.1f);
    aabb.max += Eigen::Vector3f::Constant(cellSize * 0.1f);

    voxelPointListHead.reserve(points.size());
    nextPoint.assign(points.size(), -1);

    for (int i = 0; i < (int)points.size(); ++i) {
        Eigen::Vector3i g = index(points[i]);
        uint64_t k = key(g.x(), g.y(), g.z());
        auto it = voxelPointListHead.find(k);
        if (it != voxelPointListHead.end()) {
            nextPoint[i] = it->second;
            it->second = i;
        } else {
            voxelPointListHead[k] = i;
        }
    }
}

int SparseGrid::closestPoint(const std::vector<Eigen::Vector3f>& points,
                             const Eigen::Vector3f& queryPos, float& outDist) const {
    outDist = FLT_MAX;
    if (points.empty()) return -1;

    Eigen::Vector3i start = index(queryPos);
    float minDistSq = FLT_MAX;
    int closestIdx = -1;
    const int maxSearchRadius = 100;

    for (int searchRadius = 0; searchRadius < maxSearchRadius; ++searchRadius) {
        for (int dz = -searchRadius; dz <= searchRadius; ++dz) {
            for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                    if (searchRadius > 0 && std::abs(dx) != searchRadius &&
                        std::abs(dy) != searchRadius && std::abs(dz) != searchRadius)
                        continue;

                    auto it = voxelPointListHead.find(
                        key(start.x() + dx, start.y() + dy, start.z() + dz));
                    if (it == voxelPointListHead.end()) continue;
                    for (int curr = it->second; curr != -1; curr = nextPoint[curr]) {
                        float sqDist = (queryPos - points[curr]).squaredNorm();
                        if (sqDist < minDistSq) {
                            minDistSq = sqDist;
                            closestIdx = curr;
                        }
                    }
                }
            }
        }

        if (closestIdx != -1) {
            float minX = aabb.min.x() + (start.x() - searchRadius) * cellSize;
            float maxX = aabb.min.x() + (start.x() + searchRadius + 1) * cellSize;
            float minY = aabb.min.y() + (start.y() - searchRadius) * cellSize;
            float maxY = aabb.min.y() + (start.y() + searchRadius + 1) * cellSize;
            float minZ = aabb.min.z() + (start.z() - searchRadius) * cellSize;
            float maxZ = aabb.min.z() + (start.z() + searchRadius + 1) * cellSize;
            float dX = std::min(std::abs(queryPos.x() - minX), std::abs(queryPos.x() - maxX));
            float dY = std::min(std::abs(queryPos.y() - minY), std::abs(queryPos.y() - maxY));
            float dZ = std::min(std::abs(queryPos.z() - minZ), std::abs(queryPos.z() - maxZ));
            float minDistToBoundary = std::min({dX, dY, dZ});
            if (minDistSq < minDistToBoundary * minDistToBoundary) break;
        }
    }

    if (closestIdx != -1) outDist = std::sqrt(minDistSq);
    return closestIdx;
}

void SparseGrid::kNearestNeighbors(const std::vector<Eigen::Vector3f>& points,
                                   const Eigen::Vector3f& queryPos, int k,
                                   std::vector<unsigned int>& outIndices,
                                   std::vector<float>& outDistances) const {
    outIndices.clear();
    outDistances.clear();
    if (points.empty() || k <= 0) return;

    std::priority_queue<std::pair<float, int>> pq;  // max-heap by distance
    Eigen::Vector3i start = index(queryPos);
    const int maxSearchRadius = 100;

    for (int searchRadius = 0; searchRadius < maxSearchRadius; ++searchRadius) {
        for (int dz = -searchRadius; dz <= searchRadius; ++dz) {
            for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                    if (searchRadius > 0 && std::abs(dx) != searchRadius &&
                        std::abs(dy) != searchRadius && std::abs(dz) != searchRadius)
                        continue;

                    auto it = voxelPointListHead.find(
                        key(start.x() + dx, start.y() + dy, start.z() + dz));
                    if (it == voxelPointListHead.end()) continue;
                    for (int curr = it->second; curr != -1; curr = nextPoint[curr]) {
                        float sqDist = (queryPos - points[curr]).squaredNorm();
                        if (pq.size() < (size_t)k) {
                            pq.push({sqDist, curr});
                        } else if (sqDist < pq.top().first) {
                            pq.pop();
                            pq.push({sqDist, curr});
                        }
                    }
                }
            }
        }

        if (pq.size() == (size_t)k) {
            float minX = aabb.min.x() + (start.x() - searchRadius) * cellSize;
            float maxX = aabb.min.x() + (start.x() + searchRadius + 1) * cellSize;
            float minY = aabb.min.y() + (start.y() - searchRadius) * cellSize;
            float maxY = aabb.min.y() + (start.y() + searchRadius + 1) * cellSize;
            float minZ = aabb.min.z() + (start.z() - searchRadius) * cellSize;
            float maxZ = aabb.min.z() + (start.z() + searchRadius + 1) * cellSize;
            float dX = std::min(std::abs(queryPos.x() - minX), std::abs(queryPos.x() - maxX));
            float dY = std::min(std::abs(queryPos.y() - minY), std::abs(queryPos.y() - maxY));
            float dZ = std::min(std::abs(queryPos.z() - minZ), std::abs(queryPos.z() - maxZ));
            float minDistToBoundary = std::min({dX, dY, dZ});
            if (minDistToBoundary * minDistToBoundary > pq.top().first) break;
        }
    }

    size_t count = pq.size();
    outIndices.resize(count);
    outDistances.resize(count);
    for (int i = (int)count - 1; i >= 0; --i) {
        outIndices[i] = pq.top().second;
        outDistances[i] = std::sqrt(pq.top().first);
        pq.pop();
    }
}

void SparseGrid::pointsWithinRadius(const std::vector<Eigen::Vector3f>& points,
                                    const Eigen::Vector3f& queryPos, float radius,
                                    std::vector<unsigned int>& outIndices,
                                    std::vector<float>& outDistances) const {
    outIndices.clear();
    outDistances.clear();
    if (points.empty() || radius <= 0.0f) return;

    float radiusSq = radius * radius;
    int range = (int)std::ceil(radius / cellSize);
    Eigen::Vector3i center = index(queryPos);

    for (int dz = -range; dz <= range; ++dz) {
        for (int dy = -range; dy <= range; ++dy) {
            for (int dx = -range; dx <= range; ++dx) {
                auto it = voxelPointListHead.find(
                    key(center.x() + dx, center.y() + dy, center.z() + dz));
                if (it == voxelPointListHead.end()) continue;
                for (int curr = it->second; curr != -1; curr = nextPoint[curr]) {
                    float sqDist = (queryPos - points[curr]).squaredNorm();
                    if (sqDist <= radiusSq) {
                        outIndices.push_back((unsigned int)curr);
                        outDistances.push_back(std::sqrt(sqDist));
                    }
                }
            }
        }
    }
}

} // namespace orange::geometry
