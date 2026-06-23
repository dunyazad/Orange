#pragma once

#include <vector>

#include <Eigen/Core>

#include "orange/core/mesh_generation.h"  // geometry::Triangle

// Parametric primitive mesh builders -- the CPU port of Elements/Helium's
// GeometryBuilder, decoupled from Helium's Renderable: each returns a plain
// triangle soup (geometry::Triangle, with per-vertex positions/normals/colors)
// in local space centred on the origin. The app uploads the soup to the renderer
// to spawn a real, pickable entity. CPU-only, no dependencies beyond Eigen.
//
// Conventions: shapes are built around the origin; `up` is +Y. Segment counts
// control tessellation. Colors are flat (0..1 RGB) unless noted.

namespace orange::geometry {

std::vector<Triangle> buildPlane(float size, const Eigen::Vector3f& color);
std::vector<Triangle> buildBox(const Eigen::Vector3f& size, const Eigen::Vector3f& color);
std::vector<Triangle> buildSphere(float radius, int segments, const Eigen::Vector3f& color);
std::vector<Triangle> buildCylinder(float radius, float height, int segments,
                                    const Eigen::Vector3f& color);
std::vector<Triangle> buildCone(float radius, float height, int segments,
                                const Eigen::Vector3f& color);
std::vector<Triangle> buildTorus(float majorRadius, float minorRadius, int segMajor,
                                 int segMinor, const Eigen::Vector3f& color);
std::vector<Triangle> buildDisk(float radius, int segments, const Eigen::Vector3f& color);
std::vector<Triangle> buildCapsule(float radius, float cylinderHeight, int segments,
                                   const Eigen::Vector3f& color);
std::vector<Triangle> buildArrow(float length, float radius, int segments,
                                 const Eigen::Vector3f& color);

} // namespace orange::geometry
