#pragma once

#include <functional>
#include <vector>

#include <Eigen/Core>

// Point-cloud normal estimation. For each point, fit a local plane to its k
// nearest neighbours (PCA on the neighbourhood covariance); the eigenvector of
// the smallest eigenvalue is the surface normal. Normals are oriented to point
// consistently outward (away from the cloud centroid) -- a cheap heuristic that
// is correct for convex-ish shapes and good enough for the processing operators
// (reconstruction sign, normal-deviation analysis) that consume them.
//
// Ported from Elements/Helium's PCA-based normal estimation, decoupled from
// CUDA/Helium's PointCloud and built on Orange's SparseGrid kNN. CPU-only.

namespace orange::geometry {

// Estimate one unit normal per point. `k` is the neighbourhood size used for the
// plane fit (clamped to the cloud size). Returns a vector the same length as
// `points`; degenerate neighbourhoods fall back to +Y. `progress` (optional)
// reports completion in [0,1] while the per-point fit runs.
std::vector<Eigen::Vector3f> estimateNormals(const std::vector<Eigen::Vector3f>& points,
                                             int k = 16,
                                             const std::function<void(float)>& progress = {});

} // namespace orange::geometry
