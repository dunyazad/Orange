#pragma once

#include <vector>

#include <Eigen/Core>

#include "orange/core/sparse_data_block.h"

// Surface extraction from a TSDF (SparseDataBlock) -> a triangle soup. This is a
// dual-contouring / surface-nets style extractor: one vertex per surface cell
// (averaged edge crossings), quads stitched between adjacent crossing voxels.
//
// Ported from Elements/Helium (PointCloudGenerateMesh), with the Helium runtime
// (HeliumCore lookup, VisualDebugging output) stripped so it is a pure
// mesh-out function. CPU-only, parallel.

namespace orange::geometry {

// A colored, per-vertex-normal triangle.
struct Triangle {
    Eigen::Vector3f v[3];  // positions
    Eigen::Vector3f n[3];  // normals
    Eigen::Vector3f c[3];  // colors (0..1)
};

// Extract the iso-surface (default isoLevel 0) of the TSDF as triangles.
std::vector<Triangle> generateMesh(SparseDataBlock& block, float isoLevel = 0.0f);

// Convenience "points in -> mesh out": fuse an oriented point cloud into a TSDF
// at the given voxel size, then extract the surface. normals/colors may be empty.
std::vector<Triangle> pointsToMesh(const std::vector<Eigen::Vector3f>& points,
                                   const std::vector<Eigen::Vector3f>& normals,
                                   const std::vector<Eigen::Vector3f>& colors,
                                   float voxelSize = VoxelConfig::kDefaultVoxelSize);

} // namespace orange::geometry

#include <string>

namespace orange::io {
// Load a point cloud from a PLY/XYZ/OBJ/OFF file and surface-reconstruct it into
// a triangle mesh via the TSDF voxel pipeline. Defined in src/io/serialization.cpp.
std::vector<geometry::Triangle> reconstructMeshFromFile(
    const std::string& path,
    float voxelSize = geometry::VoxelConfig::kDefaultVoxelSize);
} // namespace orange::io
