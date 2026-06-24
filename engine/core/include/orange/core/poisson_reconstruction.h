#pragma once

#include <functional>
#include <vector>

#include <Eigen/Core>

#include "orange/core/mesh_generation.h"  // geometry::Triangle

// Screened Poisson surface reconstruction: oriented point cloud -> watertight
// triangle mesh. Unlike pointsToMesh() (a local TSDF + dual contouring, which
// only fills space near the samples), this solves a global Poisson problem for
// an indicator function whose gradient matches the splatted point normals, then
// extracts its iso-surface with marching cubes. The result bridges gaps and
// closes the surface the way Kazhdan's Poisson reconstruction does, at the cost
// of a (uniform-grid) linear solve.
//
// This is a uniform-grid take on the algorithm (not the octree-adaptive one):
//   1. splat the oriented normals into a vector field V on a 2^depth grid,
//   2. solve (Laplacian - lambda*W) chi = div V  (screened, Gauss-Seidel),
//   3. marching-cubes the indicator chi at the average iso-value at the samples.
// CPU-only, Eigen-backed, no external deps.

namespace orange::geometry {

// Tunables exposed by the appOrange Poisson dialog. Defaults give a smooth,
// closed surface on a clean cloud; raise depth for detail, pointWeight to pull
// the surface tighter onto the samples.
struct PoissonParams {
    int   depth       = 6;     // grid resolution = 2^depth samples/axis (clamped 3..14)
    int   iterations  = 24;    // multigrid smoothing sweeps per level (clamped 1..200)
    float scale       = 1.2f;  // cubic bounds padding vs. the cloud extent (>= 1)
    float pointWeight  = 4.0f;  // screening strength: 0 = pure Poisson, higher = tighter
};

// NOTE: this is a DENSE uniform-grid solver, so memory is O((2^depth)^3). depth 9
// (512^3) ~ 2.7 GB; each step deeper is 8x. Past ~depth 10 the grid exhausts RAM:
// the allocation throws std::bad_alloc -- appOrange's worker catches it and reports
// "Poisson failed" (no crash). Guard the call if you need that behavior. True
// very-high depth (12-14) is only practical with an adaptive octree solver.

// Reconstruct a triangle mesh from points + per-point unit normals (same length;
// estimate them with geometry::estimateNormals if the cloud has none). Returns an
// empty mesh on degenerate input. `progress` (optional) reports [0,1].
std::vector<Triangle> poissonReconstruct(
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<Eigen::Vector3f>& normals,
    const PoissonParams& params,
    const std::function<void(float)>& progress = {});

} // namespace orange::geometry
