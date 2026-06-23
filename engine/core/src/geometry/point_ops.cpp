#include "orange/core/point_ops.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include <Eigen/SVD>

#include "orange/core/normals.h"
#include "orange/core/sparse_grid.h"

namespace orange::geometry {

namespace {
float boundsDiag(const std::vector<Eigen::Vector3f>& pts) {
    Eigen::Vector3f mn = Eigen::Vector3f::Constant(FLT_MAX);
    Eigen::Vector3f mx = Eigen::Vector3f::Constant(-FLT_MAX);
    for (const auto& p : pts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    return (mx - mn).norm();
}
} // namespace

std::vector<Eigen::Vector3f> smoothPoints(const std::vector<Eigen::Vector3f>& points,
                                          int iterations, float lambda, bool edgePreserving,
                                          int k, const std::function<void(float)>& progress) {
    std::vector<Eigen::Vector3f> pts = points;
    if (pts.size() < 4) return pts;
    k = std::max(3, std::min(k, (int)pts.size() - 1));

    float diag = boundsDiag(pts);
    float cell = diag > 0.0f ? diag * 0.02f : 1.0f;
    float sigmaD = cell * 2.0f;              // spatial falloff
    float sigmaN = diag * 0.01f;             // off-plane (edge) falloff
    float invD2 = 1.0f / (2.0f * sigmaD * sigmaD);
    float invN2 = 1.0f / (2.0f * sigmaN * sigmaN);

    // Progress split: normal estimation gets the first slice, the smoothing
    // iterations the rest.
    const float nrmEnd = edgePreserving ? 0.3f : 0.0f;
    std::vector<Eigen::Vector3f> normals;
    if (edgePreserving)
        normals = estimateNormals(pts, k, [&](float f) { if (progress) progress(f * nrmEnd); });

    const size_t step = pts.size() / 100 + 1;
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    for (int it = 0; it < iterations; ++it) {
        // Rebuild the grid each iteration since points move.
        SparseGrid grid;
        grid.build(pts, cell);
        std::vector<Eigen::Vector3f> next = pts;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (progress && (i % step == 0)) {
                float frac = ((float)it + (float)i / (float)pts.size()) / (float)iterations;
                progress(nrmEnd + (1.0f - nrmEnd) * frac);
            }
            grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
            Eigen::Vector3f acc = Eigen::Vector3f::Zero();
            float wsum = 0.0f;
            for (size_t t = 0; t < nbr.size(); ++t) {
                unsigned int j = nbr[t];
                if (j == (unsigned int)i) continue;
                float w = std::exp(-dist[t] * invD2);  // dist[t] is squared
                if (edgePreserving) {
                    float planeDist = (pts[j] - pts[i]).dot(normals[i]);
                    w *= std::exp(-planeDist * planeDist * invN2);
                }
                acc += w * pts[j];
                wsum += w;
            }
            if (wsum > 1e-12f) {
                Eigen::Vector3f target = acc / wsum;
                next[i] = pts[i] + lambda * (target - pts[i]);
            }
        }
        pts.swap(next);
    }
    if (progress) progress(1.0f);
    return pts;
}

Eigen::Matrix4f icpAlign(const std::vector<Eigen::Vector3f>& src,
                         const std::vector<Eigen::Vector3f>& dst, int maxIterations,
                         float& outRmse, int& outIters,
                         const std::function<void(float)>& progress) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    outRmse = FLT_MAX;
    outIters = 0;
    if (src.size() < 3 || dst.size() < 3) return T;

    SparseGrid grid;
    grid.build(dst, std::max(boundsDiag(dst) * 0.02f, 1e-4f));

    std::vector<Eigen::Vector3f> cur = src;  // src under the running transform
    float prevRmse = FLT_MAX;
    for (int iter = 0; iter < maxIterations; ++iter) {
        if (progress) progress((float)iter / (float)maxIterations);
        // Closest-point correspondences cur[i] -> dst[match].
        Eigen::Vector3f meanS = Eigen::Vector3f::Zero(), meanD = Eigen::Vector3f::Zero();
        std::vector<Eigen::Vector3f> corrD(cur.size());
        double sse = 0.0;
        for (size_t i = 0; i < cur.size(); ++i) {
            float d2 = 0.0f;
            int m = grid.closestPoint(dst, cur[i], d2);
            if (m < 0) m = 0;
            corrD[i] = dst[m];
            meanS += cur[i];
            meanD += dst[m];
            sse += d2;
        }
        meanS /= (float)cur.size();
        meanD /= (float)cur.size();
        float rmse = (float)std::sqrt(sse / cur.size());
        outRmse = rmse;
        outIters = iter + 1;

        // Optimal rotation: SVD of the cross-covariance H.
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        for (size_t i = 0; i < cur.size(); ++i)
            H += (cur[i] - meanS) * (corrD[i] - meanD).transpose();
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU(), V = svd.matrixV();
        Eigen::Matrix3f R = V * U.transpose();
        if (R.determinant() < 0.0f) {  // reflection fix
            V.col(2) *= -1.0f;
            R = V * U.transpose();
        }
        Eigen::Vector3f t = meanD - R * meanS;

        Eigen::Matrix4f step = Eigen::Matrix4f::Identity();
        step.block<3, 3>(0, 0) = R;
        step.block<3, 1>(0, 3) = t;
        T = step * T;
        for (size_t i = 0; i < cur.size(); ++i) cur[i] = R * cur[i] + t;

        if (std::abs(prevRmse - rmse) < 1e-6f) break;  // converged
        prevRmse = rmse;
    }
    if (progress) progress(1.0f);
    return T;
}

} // namespace orange::geometry
