#include "orange/core/sparse_data_block.h"

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>

#include "orange/core/morton3d.h"

// Ported from Elements/Helium SparseDataBlock.cpp (fromPointsData == FromPointsData).
// Two parallel passes: (1) allocate every block a point can write into, including
// neighbours within the truncation margin; (2) splat each point's weighted SDF
// into the 3x3x3 voxel neighbourhood it covers.

namespace orange::geometry {

namespace {
constexpr int kAxis = VoxelConfig::kVoxelsPerBlockAxis;
}

SparseDataBlockVoxel* SparseDataBlock::voxelByIndex(int gx, int gy, int gz) {
    int bx = (int)std::floor((float)gx / kAxis);
    int by = (int)std::floor((float)gy / kAxis);
    int bz = (int)std::floor((float)gz / kAxis);

    Eigen::Vector3f blockMin = gridOrigin + Eigen::Vector3f((float)bx * blockSizePerAxis,
                                                            (float)by * blockSizePerAxis,
                                                            (float)bz * blockSizePerAxis);
    auto key = Morton3D::encodeFromPosition(blockMin + Eigen::Vector3f::Constant(voxelSize * 0.1f),
                                            gridOrigin, blockSizePerAxis);
    auto it = dataBlocks.find(key);
    if (it == dataBlocks.end()) return nullptr;

    int lx = gx % kAxis; if (lx < 0) lx += kAxis;
    int ly = gy % kAxis; if (ly < 0) ly += kAxis;
    int lz = gz % kAxis; if (lz < 0) lz += kAxis;
    return &it->second->voxels[lz * kAxis * kAxis + ly * kAxis + lx];
}

void SparseDataBlock::fromPointsData(const std::vector<Eigen::Vector3f>& points,
                                     const std::vector<Eigen::Vector3f>& normals,
                                     const std::vector<Eigen::Vector3f>& colors,
                                     const Eigen::Vector3f& aabbMin) {
    blockSizePerAxis = voxelSize * kAxis;

    gridOrigin.x() = std::floor(aabbMin.x() / blockSizePerAxis) * blockSizePerAxis;
    gridOrigin.y() = std::floor(aabbMin.y() / blockSizePerAxis) * blockSizePerAxis;
    gridOrigin.z() = std::floor(aabbMin.z() / blockSizePerAxis) * blockSizePerAxis;

    float truncDist = std::max(voxelSize * 4.0f, 0.15f);

    size_t numberOfPoints = points.size();
    std::vector<size_t> indices(numberOfPoints);
    std::iota(indices.begin(), indices.end(), 0);

    // --- Pass 1: allocate the blocks every point touches (incl. spill-over). ---
    {
        using KeyPair = std::pair<BlockKey, Eigen::Vector3f>;
        std::vector<KeyPair> keysToAllocate;
        keysToAllocate.reserve(numberOfPoints * 2);
        std::mutex vecMutex;

        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            std::vector<KeyPair> localKeys;
            localKeys.reserve(8);

            Eigen::Vector3f p = points[i];
            Eigen::Vector3f vecFromOrigin = p - gridOrigin;
            int centerGx = (int)std::floor(vecFromOrigin.x() / voxelSize);
            int centerGy = (int)std::floor(vecFromOrigin.y() / voxelSize);
            int centerGz = (int)std::floor(vecFromOrigin.z() / voxelSize);

            int bx = centerGx / kAxis, by = centerGy / kAxis, bz = centerGz / kAxis;

            Eigen::Vector3f blockMin = gridOrigin + Eigen::Vector3f((float)bx * blockSizePerAxis,
                                                                    (float)by * blockSizePerAxis,
                                                                    (float)bz * blockSizePerAxis);
            auto key = Morton3D::encodeFromPosition(
                blockMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin, blockSizePerAxis);
            localKeys.push_back({key, blockMin});

            Eigen::Vector3f localP = p - blockMin;
            float margin = truncDist + voxelSize * 0.5f;
            bool nxN = localP.x() < margin, nxP = localP.x() > blockSizePerAxis - margin;
            bool nyN = localP.y() < margin, nyP = localP.y() > blockSizePerAxis - margin;
            bool nzN = localP.z() < margin, nzP = localP.z() > blockSizePerAxis - margin;

            if (nxN || nxP || nyN || nyP || nzN || nzP) {
                int dxMin = nxN ? -1 : 0, dxMax = nxP ? 1 : 0;
                int dyMin = nyN ? -1 : 0, dyMax = nyP ? 1 : 0;
                int dzMin = nzN ? -1 : 0, dzMax = nzP ? 1 : 0;
                for (int dz = dzMin; dz <= dzMax; ++dz)
                    for (int dy = dyMin; dy <= dyMax; ++dy)
                        for (int dx = dxMin; dx <= dxMax; ++dx) {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            Eigen::Vector3f nbMin =
                                gridOrigin + Eigen::Vector3f((float)(bx + dx) * blockSizePerAxis,
                                                             (float)(by + dy) * blockSizePerAxis,
                                                             (float)(bz + dz) * blockSizePerAxis);
                            auto nKey = Morton3D::encodeFromPosition(
                                nbMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin,
                                blockSizePerAxis);
                            localKeys.push_back({nKey, nbMin});
                        }
            }

            if (!localKeys.empty()) {
                std::lock_guard<std::mutex> lock(vecMutex);
                keysToAllocate.insert(keysToAllocate.end(), localKeys.begin(), localKeys.end());
            }
        });

        std::sort(std::execution::par, keysToAllocate.begin(), keysToAllocate.end(),
                  [](const KeyPair& a, const KeyPair& b) { return a.first < b.first; });
        auto last = std::unique(std::execution::par, keysToAllocate.begin(), keysToAllocate.end(),
                                [](const KeyPair& a, const KeyPair& b) { return a.first == b.first; });
        keysToAllocate.erase(last, keysToAllocate.end());

        for (const auto& kv : keysToAllocate) {
            if (dataBlocks.find(kv.first) == dataBlocks.end()) {
                auto block = std::make_unique<DataBlock>();
                block->initialize();
                block->blockMin = kv.second;
                dataBlocks[kv.first] = std::move(block);
            }
        }
    }

    // --- Pass 2: splat each point's weighted SDF into nearby voxels. ---
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
        Eigen::Vector3f p = points[i];
        Eigen::Vector3f n = normals.empty() ? Eigen::Vector3f(0, 1, 0) : normals[i];
        Eigen::Vector3f c = colors.empty() ? Eigen::Vector3f(1, 1, 1) : colors[i];
        if (c.x() > 1.0f) c /= 255.0f;

        Eigen::Vector3f vecFromOrigin = p - gridOrigin;
        int centerGx = (int)std::floor(vecFromOrigin.x() / voxelSize);
        int centerGy = (int)std::floor(vecFromOrigin.y() / voxelSize);
        int centerGz = (int)std::floor(vecFromOrigin.z() / voxelSize);

        BlockKey lastKey = (BlockKey)-1;
        DataBlock* cachedBlock = nullptr;

        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int gx = centerGx + dx, gy = centerGy + dy, gz = centerGz + dz;

                    Eigen::Vector3f voxelCenter =
                        gridOrigin + Eigen::Vector3f((gx + 0.5f) * voxelSize, (gy + 0.5f) * voxelSize,
                                                     (gz + 0.5f) * voxelSize);
                    float dist = (p - voxelCenter).norm();
                    if (dist > truncDist) continue;

                    int bx = (int)std::floor((float)gx / kAxis);
                    int by = (int)std::floor((float)gy / kAxis);
                    int bz = (int)std::floor((float)gz / kAxis);
                    Eigen::Vector3f blockMin =
                        gridOrigin + Eigen::Vector3f(bx * blockSizePerAxis, by * blockSizePerAxis,
                                                     bz * blockSizePerAxis);
                    auto key = Morton3D::encodeFromPosition(
                        blockMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin,
                        blockSizePerAxis);

                    DataBlock* targetBlock = nullptr;
                    if (key == lastKey && cachedBlock) {
                        targetBlock = cachedBlock;
                    } else {
                        auto it = dataBlocks.find(key);
                        if (it != dataBlocks.end()) {
                            targetBlock = it->second.get();
                            lastKey = key;
                            cachedBlock = targetBlock;
                        }
                    }
                    if (!targetBlock) continue;

                    int lx = gx % kAxis; if (lx < 0) lx += kAxis;
                    int ly = gy % kAxis; if (ly < 0) ly += kAxis;
                    int lz = gz % kAxis; if (lz < 0) lz += kAxis;

                    float weight = 1.0f - (dist / truncDist);
                    float sdf = std::clamp((voxelCenter - p).dot(n), -truncDist, truncDist);

                    std::lock_guard<std::mutex> lock(targetBlock->blockMutex);
                    SparseDataBlockVoxel& voxel =
                        targetBlock->voxels[lz * kAxis * kAxis + ly * kAxis + lx];

                    if (voxel.weight <= 0.0001f) {
                        voxel.signedDistance = sdf;
                        voxel.color = c;
                        voxel.normal = n;
                        voxel.weight = weight;
                        voxel.valid = true;
                        voxel.divergence = 0.0f;
                    } else {
                        float newW = voxel.weight + weight;
                        Eigen::Vector3f currentDir = voxel.normal.normalized();
                        float dotVal = std::clamp(currentDir.dot(n), -1.0f, 1.0f);
                        float newDiv = 1.0f - dotVal;
                        voxel.divergence = (voxel.divergence * voxel.weight + newDiv * weight) / newW;
                        voxel.signedDistance = (voxel.signedDistance * voxel.weight + sdf * weight) / newW;
                        voxel.color = (voxel.color * voxel.weight + c * weight) / newW;
                        voxel.normal = (voxel.normal * voxel.weight + n * weight) / newW;
                        voxel.weight = newW;
                    }
                }
    });
}

} // namespace orange::geometry
