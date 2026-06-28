// Multi-channel occlusal orthographic render. See occlusal_render.h.
// ASCII comments only (pipeline project rule).

#include "orange/core/occlusal_render.h"

#include <algorithm>
#include <cmath>
#include <functional>

#include <Eigen/Eigenvalues>

#include "orange/core/color.h"
#include "orange/core/sparse_grid.h"

namespace orange::geometry {
namespace {

inline uint8_t toByte(float v) {
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return (uint8_t)(v * 255.0f + 0.5f);
}

} // namespace

OcclusalChannels renderOcclusalChannels(const std::vector<Eigen::Vector3f>& verts,
                                        const Eigen::Vector3f& planePos,
                                        const Eigen::Vector3f& planeNormal, int res,
                                        float segProminence) {
    OcclusalChannels ch;
    const size_t n = verts.size();
    if (n < 8 || res < 8) return ch;

    // Occlusal frame: nz is the view (up) axis, (u,v) span the image plane.
    Eigen::Vector3f nz = planeNormal.normalized();
    Eigen::Vector3f a = (std::abs(nz.x()) < 0.9f) ? Eigen::Vector3f::UnitX()
                                                  : Eigen::Vector3f::UnitY();
    Eigen::Vector3f u = nz.cross(a).normalized();
    Eigen::Vector3f v = nz.cross(u).normalized();

    // Project + in-plane / height bounds.
    std::vector<float> A(n), B(n), H(n);
    float aMin = 1e30f, aMax = -1e30f, bMin = 1e30f, bMax = -1e30f, hMin = 1e30f, hMax = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3f d = verts[i] - planePos;
        A[i] = d.dot(u); B[i] = d.dot(v); H[i] = d.dot(nz);
        aMin = std::min(aMin, A[i]); aMax = std::max(aMax, A[i]);
        bMin = std::min(bMin, B[i]); bMax = std::max(bMax, B[i]);
        hMin = std::min(hMin, H[i]); hMax = std::max(hMax, H[i]);
    }
    float extent = std::max(aMax - aMin, bMax - bMin);
    if (extent <= 0.0f) return ch;
    float cell = extent / (float)res;
    // Center the (possibly non-square) arch in the square image.
    float aOff = aMin - (extent - (aMax - aMin)) * 0.5f;
    float bOff = bMin - (extent - (bMax - bMin)) * 0.5f;

    // Per-vertex normal + curvature via kNN PCA.
    float diag = (hMax - hMin) + extent;  // crude scale for the grid cell
    SparseGrid grid;
    grid.build(verts, extent * 0.02f > 0.0f ? extent * 0.02f : 1.0f);
    (void)diag;
    std::vector<Eigen::Vector3f> nrm(n, nz);
    std::vector<float> curv(n, 0.0f);
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    const int k = 12;
    for (size_t i = 0; i < n; ++i) {
        grid.kNearestNeighbors(verts, verts[i], k + 1, nbr, dist);
        if (nbr.size() < 4) continue;
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        for (unsigned int j : nbr) c += verts[j];
        c /= (float)nbr.size();
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (unsigned int j : nbr) {
            Eigen::Vector3f dd = verts[j] - c;
            cov += dd * dd.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
        if (es.info() != Eigen::Success) continue;
        Eigen::Vector3f ev = es.eigenvalues();  // ascending
        Eigen::Vector3f normal = es.eigenvectors().col(0).normalized();
        if (normal.dot(nz) < 0.0f) normal = -normal;  // orient toward the camera
        nrm[i] = normal;
        float sum = ev[0] + ev[1] + ev[2];
        curv[i] = sum > 1e-12f ? ev[0] / sum : 0.0f;
    }

    // Splat: per cell keep the topmost (max height) vertex.
    const int px = res * res;
    std::vector<int> top(px, -1);
    std::vector<float> topH(px, -1e30f);
    for (size_t i = 0; i < n; ++i) {
        int ia = (int)((A[i] - aOff) / cell);
        int ib = (int)((B[i] - bOff) / cell);
        if (ia < 0 || ia >= res || ib < 0 || ib >= res) continue;
        int row = res - 1 - ib;  // top row first; +B is up
        int idx = row * res + ia;
        if (H[i] > topH[idx]) { topH[idx] = H[i]; top[idx] = (int)i; }
    }

    // Curvature range over filled cells (p5..p95) so a few spikes don't wash out.
    std::vector<float> cv;
    cv.reserve(px);
    for (int p = 0; p < px; ++p) if (top[p] >= 0) cv.push_back(curv[top[p]]);
    float cLo = 0.0f, cHi = 1.0f;
    if (!cv.empty()) {
        std::sort(cv.begin(), cv.end());
        cLo = cv[(size_t)(cv.size() * 0.05f)];
        cHi = cv[(size_t)(cv.size() * 0.95f)];
        if (cHi <= cLo) cHi = cLo + 1e-6f;
    }

    ch.width = res; ch.height = res;
    ch.depth.assign(px * 4, 0);
    ch.normal.assign(px * 4, 0);
    ch.curvature.assign(px * 4, 0);
    for (int p = 0; p < px; ++p) {
        int i = top[p];
        if (i < 0) continue;  // empty -> transparent
        Eigen::Vector4f hc = color::GetHeatMapColor(H[i], hMin, hMax);
        ch.depth[p * 4 + 0] = toByte(hc.x());
        ch.depth[p * 4 + 1] = toByte(hc.y());
        ch.depth[p * 4 + 2] = toByte(hc.z());
        ch.depth[p * 4 + 3] = 255;
        ch.normal[p * 4 + 0] = toByte(nrm[i].x() * 0.5f + 0.5f);
        ch.normal[p * 4 + 1] = toByte(nrm[i].y() * 0.5f + 0.5f);
        ch.normal[p * 4 + 2] = toByte(nrm[i].z() * 0.5f + 0.5f);
        ch.normal[p * 4 + 3] = 255;
        Eigen::Vector4f cc = color::GetHeatMapColor(curv[i], cLo, cHi);
        ch.curvature[p * 4 + 0] = toByte(cc.x());
        ch.curvature[p * 4 + 1] = toByte(cc.y());
        ch.curvature[p * 4 + 2] = toByte(cc.z());
        ch.curvature[p * 4 + 3] = 255;
    }
    // --- Classical 2D segmentation: watershed on the depth image -------------
    // Each tooth is a depth bump separated from its neighbours by a deep
    // interproximal valley. Smooth the per-cell height, then run topological
    // persistence (hill-climb basins, merge those whose saddle is shallow) so
    // intra-tooth grooves merge but teeth stay split at the deep valleys.
    ch.segment.assign(px * 4, 0);
    {
        std::vector<float> raw(px, -1e30f);
        for (int p = 0; p < px; ++p) if (top[p] >= 0) raw[p] = H[top[p]];

        // Hole-fill: the vertex splat leaves gaps that fragment the image and
        // break segmentation. Fill an empty cell only when it has >= 3 filled
        // 8-neighbours, a few bounded passes -- thin cracks close but the big
        // interior void (tongue space) stays open.
        {
            const int dr8[8] = {-1, -1, -1, 0, 0, 1, 1, 1}, dc8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
            for (int it = 0; it < 6; ++it) {
                std::vector<float> add(px, -1e30f);
                bool any = false;
                for (int r = 0; r < res; ++r)
                    for (int c = 0; c < res; ++c) {
                        int p = r * res + c;
                        if (raw[p] > -1e29f) continue;
                        float s = 0.0f; int cnt = 0;
                        for (int k = 0; k < 8; ++k) {
                            int rr = r + dr8[k], cc = c + dc8[k];
                            if (rr < 0 || rr >= res || cc < 0 || cc >= res) continue;
                            float h = raw[rr * res + cc];
                            if (h > -1e29f) { s += h; ++cnt; }
                        }
                        if (cnt >= 3) { add[p] = s / (float)cnt; any = true; }
                    }
                for (int p = 0; p < px; ++p) if (add[p] > -1e29f) raw[p] = add[p];
                if (!any) break;
            }
        }

        // Smooth over filled 4-neighbours, N iterations.
        auto smooth = [&](std::vector<float> f, int iters) {
            std::vector<float> tmp = f;
            const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
            for (int it = 0; it < iters; ++it) {
                for (int r = 0; r < res; ++r)
                    for (int c = 0; c < res; ++c) {
                        int p = r * res + c;
                        if (f[p] <= -1e29f) continue;
                        float s = f[p]; int cnt = 1;
                        for (int k = 0; k < 4; ++k) {
                            int rr = r + dr[k], cc = c + dc[k];
                            if (rr < 0 || rr >= res || cc < 0 || cc >= res) continue;
                            float h = f[rr * res + cc];
                            if (h > -1e29f) { s += h; ++cnt; }
                        }
                        tmp[p] = s / (float)cnt;
                    }
                f.swap(tmp);
            }
            return f;
        };

        // High-pass: subtract a broadly-smoothed base so the arch curvature and
        // base walls drop out and each tooth's local bump (and the deep valley
        // between teeth) dominate. Watershed then separates teeth instead of
        // merging the whole arch into one basin.
        std::vector<float> den  = smooth(raw, 3);
        std::vector<float> base = smooth(raw, 40);
        std::vector<float> Hg(px, -1e30f);
        float rMin = 1e30f, rMax = -1e30f;
        for (int p = 0; p < px; ++p) {
            if (raw[p] <= -1e29f) continue;
            Hg[p] = den[p] - base[p];
            rMin = std::min(rMin, Hg[p]); rMax = std::max(rMax, Hg[p]);
        }

        const float promThresh = (rMax - rMin) * segProminence;  // > intra-tooth groove
        std::vector<int> order;
        order.reserve(px);
        for (int p = 0; p < px; ++p) if (Hg[p] > -1e29f) order.push_back(p);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return Hg[a] > Hg[b]; });

        std::vector<int> parent(px, -1), peak(px, 0);
        std::function<int(int)> find = [&](int x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
        for (int p : order) {
            parent[p] = p; peak[p] = p;
            int r = p / res, c = p % res;
            std::vector<int> roots;
            for (int k = 0; k < 4; ++k) {
                int rr = r + dr[k], cc = c + dc[k];
                if (rr < 0 || rr >= res || cc < 0 || cc >= res) continue;
                int q = rr * res + cc;
                if (parent[q] == -1) continue;
                int root = find(q);
                if (std::find(roots.begin(), roots.end(), root) == roots.end()) roots.push_back(root);
            }
            if (roots.empty()) continue;  // new basin (local max)
            int keep = roots[0];
            for (size_t i = 1; i < roots.size(); ++i)
                if (Hg[peak[roots[i]]] > Hg[peak[keep]]) keep = roots[i];
            for (int root : roots) {
                if (root == keep) continue;
                // Merge only shallow basins (intra-tooth grooves); a prominent
                // neighbour across a deep interproximal valley stays its own label.
                if (Hg[peak[root]] - Hg[p] < promThresh) parent[root] = keep;
            }
            parent[p] = keep;  // the ridge cell itself joins the taller side
        }

        auto labelColor = [&](int lbl) -> Eigen::Vector3f {
            uint32_t h = (uint32_t)lbl * 2654435761u;
            float hue = (float)(h & 0xFFFF) / 65535.0f * 6.0f;
            float x = 1.0f - std::abs(std::fmod(hue, 2.0f) - 1.0f);
            Eigen::Vector3f c;
            if (hue < 1) c = {1, x, 0}; else if (hue < 2) c = {x, 1, 0};
            else if (hue < 3) c = {0, 1, x}; else if (hue < 4) c = {0, x, 1};
            else if (hue < 5) c = {x, 0, 1}; else c = {1, 0, x};
            return c * 0.55f + Eigen::Vector3f(0.25f, 0.25f, 0.25f);
        };
        for (int p = 0; p < px; ++p) {
            if (parent[p] == -1) continue;
            int root = find(p);
            Eigen::Vector3f col = labelColor(peak[root]);
            ch.segment[p * 4 + 0] = toByte(col.x());
            ch.segment[p * 4 + 1] = toByte(col.y());
            ch.segment[p * 4 + 2] = toByte(col.z());
            ch.segment[p * 4 + 3] = 255;
        }
    }

    ch.origin = planePos; ch.u = u; ch.v = v;
    ch.aOff = aOff; ch.bOff = bOff; ch.cell = cell;
    ch.valid = true;
    return ch;
}

} // namespace orange::geometry
