#pragma once

#include <functional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

// Point-cloud transform operators ported (and CPU-reimplemented) from the
// Elements/Helium GPU pipeline, which had no CPU path:
//   - smoothPoints: Laplacian / bilateral (edge-preserving) smoothing
//   - icpAlign:     point-to-point Iterative Closest Point registration
// Both are CPU-only, Eigen-backed, and built on Orange's SparseGrid for the
// neighbour / closest-point queries (the role Helium's GPU hash grid played).

namespace orange::geometry {

// Smooth a point cloud. Each iteration moves every point a fraction `lambda`
// (0..1) toward a weighted average of its k nearest neighbours. When
// `edgePreserving` is true the average is bilaterally weighted by neighbour
// distance AND agreement with the local surface normal, so creases survive.
std::vector<Eigen::Vector3f> smoothPoints(const std::vector<Eigen::Vector3f>& points,
                                          int iterations = 5, float lambda = 0.5f,
                                          bool edgePreserving = true, int k = 12,
                                          const std::function<void(float)>& progress = {});

// Align `src` onto `dst` with point-to-point ICP: repeatedly match each source
// point to its closest target point, solve the optimal rigid transform (SVD of
// the cross-covariance), and compose. Returns the 4x4 transform mapping src into
// dst's frame; `outRmse` receives the final RMS correspondence distance and
// `outIters` the iterations actually run (early-out on convergence).
Eigen::Matrix4f icpAlign(const std::vector<Eigen::Vector3f>& src,
                         const std::vector<Eigen::Vector3f>& dst, int maxIterations,
                         float& outRmse, int& outIters,
                         const std::function<void(float)>& progress = {});

} // namespace orange::geometry
