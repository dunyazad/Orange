// 3D per-tooth watershed segmentation on the mesh graph. See tooth_segment.h.
// ASCII comments only (pipeline project rule).

#include "orange/core/tooth_segment.h"

#include <algorithm>
#include <cmath>
#include <functional>

#include <Eigen/Eigenvalues>

#include "orange/core/half_edge.h"
#include "orange/core/sparse_grid.h"

namespace orange::geometry {

ToothSegResult segmentTeeth(const std::vector<Eigen::Vector3f>& vertices,
                            const std::vector<uint32_t>& indices,
                            const ToothSegParams& params) {
    ToothSegResult result;
    if (vertices.size() < 8 || indices.size() < 3) return result;

    HalfEdgeMesh he;
    if (!he.build(vertices, indices)) return result;
    const uint32_t n = (uint32_t)he.vertexCount();

    // Occlusal axis from PCA, oriented up.
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (uint32_t i = 0; i < n; ++i) centroid += he.position(i);
    centroid /= (float)n;
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (uint32_t i = 0; i < n; ++i) {
        Eigen::Vector3f d = he.position(i) - centroid;
        cov += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
    if (es.info() != Eigen::Success) return result;
    Eigen::Vector3f axis = es.eigenvectors().col(0).normalized();
    if (axis.dot(Eigen::Vector3f::UnitY()) < 0.0f) axis = -axis;

    std::vector<float> H(n);
    float hMin = 1e30f, hMax = -1e30f;
    for (uint32_t i = 0; i < n; ++i) {
        H[i] = (he.position(i) - centroid).dot(axis);
        hMin = std::min(hMin, H[i]); hMax = std::max(hMax, H[i]);
    }

    // CSR 1-ring adjacency (half-edge is the source of truth).
    std::vector<uint32_t> off(n + 1, 0);
    {
        std::vector<uint32_t> ring;
        for (uint32_t v = 0; v < n; ++v) { he.oneRing(v, ring); off[v + 1] = off[v] + (uint32_t)ring.size(); }
    }
    std::vector<uint32_t> nbrs(off[n]);
    {
        std::vector<uint32_t> ring;
        for (uint32_t v = 0; v < n; ++v) {
            he.oneRing(v, ring);
            std::copy(ring.begin(), ring.end(), nbrs.begin() + off[v]);
        }
    }

    auto smooth = [&](std::vector<float> f, int iters) {
        std::vector<float> tmp(n);
        for (int it = 0; it < iters; ++it) {
            for (uint32_t v = 0; v < n; ++v) {
                uint32_t e0 = off[v], e1 = off[v + 1];
                if (e1 == e0) { tmp[v] = f[v]; continue; }
                float s = f[v]; int c = 1;
                for (uint32_t e = e0; e < e1; ++e) { s += f[nbrs[e]]; ++c; }
                tmp[v] = s / (float)c;
            }
            f.swap(tmp);
        }
        return f;
    };

    // Per-vertex concavity via kNN PCA: concave grooves (gingival margin,
    // interproximal valleys) get a high positive value and become region
    // boundaries, so a basin stops at the tooth crown instead of flowing on into
    // the gum or a touching neighbour.
    std::vector<float> conc(n, 0.0f);
    float bdiag = 0.0f;
    {
        const std::vector<Eigen::Vector3f>& P = he.positions();
        Eigen::Vector3f a = P[0], b = P[0];
        for (const auto& p : P) { a = a.cwiseMin(p); b = b.cwiseMax(p); }
        bdiag = (b - a).norm();
        SparseGrid grid;
        grid.build(P, bdiag > 0.0f ? bdiag * 0.01f : 1.0f);
        std::vector<unsigned int> nb;
        std::vector<float> ds;
        for (uint32_t v = 0; v < n; ++v) {
            grid.kNearestNeighbors(P, P[v], 13, nb, ds);
            if (nb.size() < 4) continue;
            Eigen::Vector3f c = Eigen::Vector3f::Zero();
            for (unsigned int j : nb) c += P[j];
            c /= (float)nb.size();
            Eigen::Matrix3f cov2 = Eigen::Matrix3f::Zero();
            for (unsigned int j : nb) { Eigen::Vector3f d = P[j] - c; cov2 += d * d.transpose(); }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> e2(cov2);
            if (e2.info() != Eigen::Success) continue;
            Eigen::Vector3f nrm = e2.eigenvectors().col(0).normalized();
            if (nrm.dot(axis) < 0.0f) nrm = -nrm;       // orient outward (occlusal side)
            conc[v] = (c - P[v]).dot(nrm);              // > 0 == concave valley/groove
        }
    }
    conc = smooth(conc, 3);
    const float concBound = params.curvThresh * bdiag;

    // High-pass: residual = light-smoothed - broadly-smoothed base. Removes arch
    // curvature so each tooth's local bump (and the deep valley between teeth)
    // dominates the watershed.
    std::vector<float> den  = smooth(H, std::max(0, params.smoothIters));
    std::vector<float> base = smooth(H, std::max(1, params.baseIters));
    std::vector<float> R(n);
    float rMin = 1e30f, rMax = -1e30f;
    const float hGate = hMin + std::max(0.0f, params.heightGate) * (hMax - hMin);
    for (uint32_t v = 0; v < n; ++v) {
        R[v] = den[v] - base[v];
        if (H[v] >= hGate) { rMin = std::min(rMin, R[v]); rMax = std::max(rMax, R[v]); }
    }
    const float promThresh = (rMax - rMin) * params.prominence;

    // Persistence watershed over the teeth band (H >= gate), descending R.
    std::vector<uint32_t> order;
    order.reserve(n);
    for (uint32_t v = 0; v < n; ++v)
        if (H[v] >= hGate && conc[v] < concBound) order.push_back(v);  // not gum, not a groove
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return R[a] > R[b]; });

    std::vector<int> parent(n, -1), peak(n, 0);
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    std::vector<int> roots;
    for (uint32_t v : order) {
        parent[v] = v; peak[v] = v;
        roots.clear();
        for (uint32_t e = off[v]; e < off[v + 1]; ++e) {
            uint32_t u = nbrs[e];
            if (parent[u] == -1) continue;  // not yet processed (lower or gated out)
            int root = find((int)u);
            if (std::find(roots.begin(), roots.end(), root) == roots.end()) roots.push_back(root);
        }
        if (roots.empty()) continue;  // local maximum: a fresh basin (tooth top)
        int keep = roots[0];
        for (size_t i = 1; i < roots.size(); ++i)
            if (R[peak[roots[i]]] > R[peak[keep]]) keep = roots[i];
        for (int root : roots) {
            if (root == keep) continue;
            // Merge only shallow basins (intra-tooth grooves); a prominent
            // neighbour across a deep interproximal valley stays its own tooth.
            if (R[peak[root]] - R[v] < promThresh) parent[root] = keep;
        }
        parent[v] = keep;
    }

    // Compact root -> tooth label.
    std::vector<int> rootLabel(n, -1);
    int next = 0;
    std::vector<int> labelW(n, -1);
    for (uint32_t v = 0; v < n; ++v) {
        if (parent[v] == -1) continue;  // gingiva/base
        int root = find((int)v);
        if (rootLabel[root] < 0) rootLabel[root] = next++;
        labelW[v] = rootLabel[root];
    }

    // Map welded labels back to the original vertex order.
    result.label.assign(vertices.size(), -1);
    for (uint32_t i = 0; i < (uint32_t)vertices.size(); ++i)
        result.label[i] = labelW[he.welded(i)];
    result.regions = next;
    result.valid = true;
    return result;
}

} // namespace orange::geometry
