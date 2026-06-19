#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

// Hierarchical sparse voxel grid storing a truncated signed-distance field (TSDF)
// fused from an oriented point cloud. Space is divided into 8x8x8 voxel blocks
// keyed by Morton code; only blocks near the surface are allocated. Each voxel
// accumulates a weighted SDF, normal, color and a normal-divergence measure.
//
// Ported from Elements/Helium (SparseDataBlock), decoupled from PointCloud: use
// fromPointsData(positions, normals, colors, aabbMin). CPU-only, parallel build.

namespace orange::geometry {

// Voxel grid configuration (was SpatialPartitioningConfiguration).
struct VoxelConfig {
    static constexpr int kVoxelsPerBlockAxis = 8;
    static constexpr int kVoxelsPerBlock =
        kVoxelsPerBlockAxis * kVoxelsPerBlockAxis * kVoxelsPerBlockAxis;  // 512
    static constexpr float kDefaultVoxelSize = 0.3f;
};

class SparseDataBlockVoxel {
public:
    bool valid = false;
    float signedDistance = 0.0f;
    float weight = 0.0f;
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    Eigen::Vector3f color = Eigen::Vector3f::Zero();
    float divergence = 0.0f;
};

struct DataBlock {
    Eigen::Vector3f blockMin = Eigen::Vector3f::Zero();
    SparseDataBlockVoxel voxels[VoxelConfig::kVoxelsPerBlock];
    std::mutex blockMutex;

    void initialize() {
        for (int i = 0; i < VoxelConfig::kVoxelsPerBlock; ++i)
            voxels[i] = SparseDataBlockVoxel();
    }
};

class SparseDataBlock {
public:
    using BlockKey = uint64_t;

    float voxelSize = VoxelConfig::kDefaultVoxelSize;
    Eigen::Vector3f gridOrigin = Eigen::Vector3f::Zero();
    std::unordered_map<BlockKey, std::unique_ptr<DataBlock>> dataBlocks;

    float blockSizePerAxis = voxelSize * VoxelConfig::kVoxelsPerBlockAxis;

    // Global voxel index (gx,gy,gz) -> voxel pointer, or nullptr if not allocated.
    SparseDataBlockVoxel* voxelByIndex(int gx, int gy, int gz);

    // Fuse an oriented point cloud into the TSDF. normals/colors may be empty
    // (defaults: +Y normal, white color); colors in 0..255 are auto-normalized.
    void fromPointsData(const std::vector<Eigen::Vector3f>& points,
                        const std::vector<Eigen::Vector3f>& normals,
                        const std::vector<Eigen::Vector3f>& colors,
                        const Eigen::Vector3f& aabbMin);
};

} // namespace orange::geometry
