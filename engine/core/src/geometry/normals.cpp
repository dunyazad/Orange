#include "orange/core/normals.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <execution>
#include <numeric>

#include <Eigen/Eigenvalues>

#include "orange/core/sparse_grid.h"

namespace orange::geometry {

std::vector<Eigen::Vector3f> estimateNormals(const std::vector<Eigen::Vector3f>& points, int k,
                                             const std::function<void(float)>& progress) {
    std::vector<Eigen::Vector3f> normals(points.size(), Eigen::Vector3f::UnitY());
    if (points.size() < 3) return normals;
    k = std::max(3, std::min(k, (int)points.size() - 1));

    // Cell size ~ the mean point spacing so each query cell holds a few points.
    Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX);
    Eigen::Vector3f mx = Eigen::Vector3f::Constant(-FLT_MAX);
    for (const auto& p : points) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    Eigen::Vector3f centroid = 0.5f * (mn + mx);
    float diag = (mx - mn).norm();
    float cell = diag > 0.0f ? diag * 0.02f : 1.0f;

    SparseGrid grid;
    grid.build(points, cell);

    // Per-point fit is independent (the grid query is const/read-only), so run it
    // in parallel. Scratch buffers are declared inside the lambda => per-thread.
    std::vector<size_t> idx(points.size());
    std::iota(idx.begin(), idx.end(), (size_t)0);
    const size_t step = points.size() / 100 + 1;  // report ~1% granularity
    std::atomic<size_t> done{0};
    std::for_each(std::execution::par, idx.begin(), idx.end(), [&](size_t i) {
        std::vector<unsigned int> nbr;
        std::vector<float> dist;
        grid.kNearestNeighbors(points, points[i], k + 1, nbr, dist);  // [0] is self
        if (nbr.size() >= 3) {
            Eigen::Vector3f mean = Eigen::Vector3f::Zero();
            for (unsigned int j : nbr) mean += points[j];
            mean /= (float)nbr.size();

            Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
            for (unsigned int j : nbr) {
                Eigen::Vector3f d = points[j] - mean;
                cov += d * d.transpose();
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
            if (es.info() == Eigen::Success) {
                Eigen::Vector3f n = es.eigenvectors().col(0);  // smallest eigenvalue
                if (n.squaredNorm() >= 1e-12f) {
                    n.normalize();
                    if (n.dot(points[i] - centroid) < 0.0f) n = -n;  // orient outward
                    normals[i] = n;
                }
            }
        }
        if (progress) {
            size_t c = done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (c % step == 0) progress((float)c / (float)points.size());
        }
    });
    return normals;
}

} // namespace orange::geometry
