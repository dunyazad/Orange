// Forces the header-only ports to be compiled (and surfaces any errors) inside
// the engine, and provides a small bridge from the serialization formats to the
// geometry toolkit: load an oriented point cloud and reconstruct a surface mesh.

#include "orange/core/serialization.h"

#include <cctype>
#include <memory>

#include "orange/core/color.h"
#include "orange/core/geometry.h"
#include "orange/core/marching_cubes_tables.h"
#include "orange/core/mesh_generation.h"
#include "orange/core/morton3d.h"

namespace orange::io {

// Read a point cloud from a PLY/XYZ/OBJ file (positions only; normals optional)
// and surface-reconstruct it into a triangle mesh via the TSDF voxel pipeline.
// Returns an empty mesh if the file can't be read or has no points.
std::vector<geometry::Triangle> reconstructMeshFromFile(const std::string& path, float voxelSize) {
    std::unique_ptr<HSerializable> fmt;
    auto dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : path.substr(dot + 1);
    for (auto& c : ext) c = (char)std::tolower(c);

    if (ext == "ply")      fmt = std::make_unique<PLYFormat>();
    else if (ext == "xyz") fmt = std::make_unique<XYZFormat>();
    else if (ext == "obj") fmt = std::make_unique<OBJFormat>();
    else if (ext == "off") fmt = std::make_unique<OFFFormat>();
    else return {};

    if (!fmt->Deserialize(path)) return {};

    const std::vector<Eigen::Vector3f>& points = fmt->GetPoints();
    if (points.empty()) return {};

    std::vector<Eigen::Vector3f> noNormals, noColors;  // pipeline fills defaults
    return geometry::pointsToMesh(points, noNormals, noColors, voxelSize);
}

} // namespace orange::io
