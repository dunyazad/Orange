#include "orange/core/modes.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Eigenvalues>

#include "orange/core/color.h"
#include "orange/core/mesh_generation.h"
#include "orange/core/normals.h"
#include "orange/core/point_ops.h"
#include "orange/core/sparse_grid.h"

// Mode implementations. Each is a pure function (input points -> debug geometry)
// ported from the Hydrogen apps, decoupled from CUDA/Helium and rescaled to the
// input's bounding box so they work at any data scale.

namespace orange::modes {
namespace {

Eigen::Vector3f boundsExtent(const std::vector<Eigen::Vector3f>& pts, Eigen::Vector3f& mn,
                             Eigen::Vector3f& mx) {
    mn = Eigen::Vector3f::Constant(FLT_MAX);
    mx = Eigen::Vector3f::Constant(-FLT_MAX);
    for (const auto& p : pts) {
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
    }
    return mx - mn;
}

// A distinct color per integer index (golden-ratio hue spread).
Eigen::Vector3f indexColor(int i) { return color::RandomFromIndex((size_t)i).head<3>(); }

// Throttled progress report (fraction in [0,1]).
inline void report(const ProgressFn& p, float f) { if (p) p(f); }

// --- Mode 0: Euclidean clustering (SparseGrid radius + union-find) -----------
void runClustering(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.empty()) return;

    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(pts, mn, mx).norm();
    float radius = diag * 0.02f;
    if (radius <= 0.0f) return;

    geometry::SparseGrid grid;
    grid.build(pts, radius);

    std::vector<int> parent(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) parent[i] = (int)i;
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](int a, int b) { a = find(a); b = find(b); if (a != b) parent[a] = b; };

    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    const size_t step = pts.size() / 100 + 1;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i % step == 0) report(progress, (float)i / (float)pts.size());
        grid.pointsWithinRadius(pts, pts[i], radius, nbr, dist);
        for (unsigned int j : nbr)
            if (j > i) unite((int)i, (int)j);
    }

    // Relabel roots to dense cluster ids.
    std::unordered_map<int, int> rootToCluster;
    std::vector<int> label(pts.size());
    int nClusters = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        int r = find((int)i);
        auto it = rootToCluster.find(r);
        int c = it == rootToCluster.end() ? (rootToCluster[r] = nClusters++) : it->second;
        label[i] = c;
    }

    std::vector<Eigen::Vector3f> cmin(nClusters, Eigen::Vector3f::Constant(FLT_MAX));
    std::vector<Eigen::Vector3f> cmax(nClusters, Eigen::Vector3f::Constant(-FLT_MAX));
    float psize = diag * 0.004f;
    for (size_t i = 0; i < pts.size(); ++i) {
        int c = label[i];
        cmin[c] = cmin[c].cwiseMin(pts[i]);
        cmax[c] = cmax[c].cwiseMax(pts[i]);
        out.addPoint(pts[i], indexColor(c), psize);
    }
    for (int c = 0; c < nClusters; ++c)
        out.addWireBox(cmin[c], cmax[c], indexColor(c), diag * 0.0015f);
}

// --- Mode 1: Voxel morphology (erode + largest connected component) ----------
// Ported from Hydrogen AppMorphology (MorphEngine).
struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VoxelHash {
    size_t operator()(const VoxelKey& k) const {
        return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^
               (std::hash<int>()(k.z) << 1);
    }
};
using VoxelSet = std::unordered_set<VoxelKey, VoxelHash>;

VoxelKey voxelOf(const Eigen::Vector3f& p, float inv) {
    return {(int)std::floor(p.x() * inv), (int)std::floor(p.y() * inv),
            (int)std::floor(p.z() * inv)};
}

void runMorphology(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.empty()) return;

    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(pts, mn, mx).norm();
    float voxelSize = diag * 0.04f;
    if (voxelSize <= 0.0f) return;
    int erodeIter = 2;
    float inv = 1.0f / voxelSize;

    VoxelSet voxels;
    for (const auto& p : pts) voxels.insert(voxelOf(p, inv));

    static const int dx[6] = {-1, 1, 0, 0, 0, 0};
    static const int dy[6] = {0, 0, -1, 1, 0, 0};
    static const int dz[6] = {0, 0, 0, 0, -1, 1};

    // Erosion: keep only voxels whose 6 face-neighbors are all occupied.
    for (int it = 0; it < erodeIter && !voxels.empty(); ++it) {
        VoxelSet next;
        for (const auto& k : voxels) {
            bool interior = true;
            for (int n = 0; n < 6; ++n)
                if (!voxels.count({k.x + dx[n], k.y + dy[n], k.z + dz[n]})) { interior = false; break; }
            if (interior) next.insert(k);
        }
        voxels.swap(next);
    }

    // Keep the largest connected component (BFS over face-connectivity).
    VoxelSet visited, largest;
    for (const auto& start : voxels) {
        if (visited.count(start)) continue;
        VoxelSet comp;
        std::queue<VoxelKey> q;
        q.push(start);
        visited.insert(start);
        comp.insert(start);
        while (!q.empty()) {
            VoxelKey c = q.front();
            q.pop();
            for (int n = 0; n < 6; ++n) {
                VoxelKey nb{c.x + dx[n], c.y + dy[n], c.z + dz[n]};
                if (voxels.count(nb) && !visited.count(nb)) {
                    visited.insert(nb);
                    comp.insert(nb);
                    q.push(nb);
                }
            }
        }
        if (comp.size() > largest.size()) largest.swap(comp);
    }

    // Restore original points near the surviving core voxels (faded red removed,
    // green kept) to show the filtering result.
    float psize = diag * 0.004f;
    int expansion = erodeIter;
    const size_t step = pts.size() / 100 + 1;
    for (size_t pi = 0; pi < pts.size(); ++pi) {
        if (pi % step == 0) report(progress, (float)pi / (float)pts.size());
        const auto& p = pts[pi];
        VoxelKey k = voxelOf(p, inv);
        bool keep = false;
        for (int z = -expansion; z <= expansion && !keep; ++z)
            for (int y = -expansion; y <= expansion && !keep; ++y)
                for (int x = -expansion; x <= expansion && !keep; ++x)
                    if (largest.count({k.x + x, k.y + y, k.z + z})) keep = true;
        out.addPoint(p, keep ? Eigen::Vector3f(0.1f, 0.9f, 0.2f)
                             : Eigen::Vector3f(0.5f, 0.12f, 0.12f),
                     psize);
    }
}

// --- Mode 2: SDF denoise (UDF splat + box blur + isosurface resample) --------
// Ported from Hydrogen AppSDFFiltering (SDFEngine).
void runSdfFilter(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.empty()) return;

    Eigen::Vector3f mn, mx;
    Eigen::Vector3f ext = boundsExtent(pts, mn, mx);
    float diag = ext.norm();

    // Size the grid to the point density (nearest-neighbour spacing), not the
    // bounding box -- outliers inflate the box but not the local spacing, which
    // is what determines whether the iso-surface shell has occupied voxels.
    geometry::SparseGrid sg;
    sg.build(pts, diag * 0.03f);
    float spacing = 0.0f;
    int spacingN = 0;
    std::vector<unsigned int> nbr;
    std::vector<float> nbrDist;
    size_t step = pts.size() > 400 ? pts.size() / 400 : 1;
    for (size_t i = 0; i < pts.size(); i += step) {
        sg.kNearestNeighbors(pts, pts[i], 2, nbr, nbrDist);
        if (nbrDist.size() >= 2) { spacing += nbrDist[1]; spacingN++; }  // [0] is self
    }
    spacing = spacingN ? spacing / spacingN : diag * 0.01f;
    float voxelSize = std::max(spacing, ext.maxCoeff() / 256.0f);  // cap grid resolution
    if (voxelSize <= 0.0f) return;
    const int smoothIter = 2;
    const float pad = voxelSize * 5.0f;
    const float kFar = 100.0f;

    Eigen::Vector3f minBound = mn - Eigen::Vector3f::Constant(pad);
    Eigen::Vector3f maxBound = mx + Eigen::Vector3f::Constant(pad);
    Eigen::Vector3i dim = ((maxBound - minBound) / voxelSize).cast<int>() + Eigen::Vector3i::Ones();
    auto idxOf = [&](int x, int y, int z) {
        return (size_t)z * dim.y() * dim.x() + (size_t)y * dim.x() + x;
    };
    auto gridToWorld = [&](int x, int y, int z) -> Eigen::Vector3f {
        return minBound + Eigen::Vector3f((float)x, (float)y, (float)z) * voxelSize;
    };
    auto valid = [&](int x, int y, int z) {
        return x >= 0 && x < dim.x() && y >= 0 && y < dim.y() && z >= 0 && z < dim.z();
    };

    std::vector<float> data((size_t)dim.x() * dim.y() * dim.z(), kFar);

    // Splat: each point writes its distance into nearby voxels (keep the min).
    // A wider range builds a graded distance band that survives the box blur.
    const int range = 3;
    const size_t splatStep = pts.size() / 100 + 1;
    for (size_t pi = 0; pi < pts.size(); ++pi) {
        if (pi % splatStep == 0) report(progress, 0.5f * (float)pi / (float)pts.size());
        const auto& p = pts[pi];
        Eigen::Vector3i c = ((p - minBound) / voxelSize).cast<int>();
        for (int z = -range; z <= range; ++z)
            for (int y = -range; y <= range; ++y)
                for (int x = -range; x <= range; ++x) {
                    int gx = c.x() + x, gy = c.y() + y, gz = c.z() + z;
                    if (!valid(gx, gy, gz)) continue;
                    float d = (gridToWorld(gx, gy, gz) - p).norm();
                    size_t i = idxOf(gx, gy, gz);
                    if (d < data[i]) data[i] = d;
                }
    }

    // Box blur near the surface (smooths away thin spikes/noise).
    report(progress, 0.5f);
    std::vector<float> tmp = data;
    for (int it = 0; it < smoothIter; ++it) {
        for (int z = 1; z < dim.z() - 1; ++z)
            for (int y = 1; y < dim.y() - 1; ++y)
                for (int x = 1; x < dim.x() - 1; ++x) {
                    size_t i = idxOf(x, y, z);
                    if (data[i] > voxelSize * 3.0f) continue;
                    float sum = 0.0f;
                    int cnt = 0;
                    for (int kz = -1; kz <= 1; ++kz)
                        for (int ky = -1; ky <= 1; ++ky)
                            for (int kx = -1; kx <= 1; ++kx) {
                                float v = data[idxOf(x + kx, y + ky, z + kz)];
                                if (v < 50.0f) { sum += v; cnt++; }
                            }
                    if (cnt > 0) tmp[i] = sum / (float)cnt;
                }
        data = tmp;
    }

    // Resample the iso-surface as points (color by SDF gradient = normal).
    float iso = voxelSize * 1.5f;
    float psize = voxelSize * 0.4f;
    for (int z = 1; z < dim.z() - 1; ++z) {
        report(progress, 0.7f + 0.3f * (float)z / (float)dim.z());
        for (int y = 1; y < dim.y() - 1; ++y)
            for (int x = 1; x < dim.x() - 1; ++x) {
                if (data[idxOf(x, y, z)] >= iso) continue;
                Eigen::Vector3f n(data[idxOf(x + 1, y, z)] - data[idxOf(x - 1, y, z)],
                                  data[idxOf(x, y + 1, z)] - data[idxOf(x, y - 1, z)],
                                  data[idxOf(x, y, z + 1)] - data[idxOf(x, y, z - 1)]);
                n = n.squaredNorm() > 1e-6f ? n.normalized() : Eigen::Vector3f::UnitY();
                out.addPoint(gridToWorld(x, y, z), n * 0.5f + Eigen::Vector3f::Constant(0.5f), psize);
            }
    }
    report(progress, 1.0f);
}

// --- Mode 3: Surface reconstruction (TSDF + dual contouring) -----------------
void runReconstruct(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.empty()) return;

    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(pts, mn, mx).norm();
    float voxelSize = diag * 0.03f;

    // Normal estimation is the first ~60%; the (parallel) meshing the rest.
    std::vector<Eigen::Vector3f> nrm =
        in.normals.size() == pts.size()
            ? in.normals
            : geometry::estimateNormals(pts, 16, [&](float f) { report(progress, f * 0.6f); });
    report(progress, 0.6f);
    std::vector<Eigen::Vector3f> noColors;
    auto tris = geometry::pointsToMesh(pts, nrm, noColors, voxelSize);
    for (const auto& t : tris)
        out.addTriangle(t.v[0], t.v[1], t.v[2], t.c[0]);
    report(progress, 1.0f);
}

// --- Shared helpers for the kNN/PCA operators --------------------------------
// Build a SparseGrid over the points sized to ~2% of the bbox diagonal (a few
// points per cell), returning the diagonal for scale-relative thresholds.
float buildGrid(const std::vector<Eigen::Vector3f>& pts, geometry::SparseGrid& grid,
                Eigen::Vector3f& mn, Eigen::Vector3f& mx) {
    float diag = boundsExtent(pts, mn, mx).norm();
    grid.build(pts, diag > 0.0f ? diag * 0.02f : 1.0f);
    return diag;
}

// PCA of a neighbourhood: ascending eigenvalues into `eval`, columns of `evec`
// the matching eigenvectors. Returns false on a degenerate neighbourhood.
bool neighbourhoodPCA(const std::vector<Eigen::Vector3f>& pts,
                      const std::vector<unsigned int>& nbr, Eigen::Vector3f& centroid,
                      Eigen::Vector3f& eval, Eigen::Matrix3f& evec) {
    if (nbr.size() < 3) return false;
    centroid = Eigen::Vector3f::Zero();
    for (unsigned int j : nbr) centroid += pts[j];
    centroid /= (float)nbr.size();
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (unsigned int j : nbr) {
        Eigen::Vector3f d = pts[j] - centroid;
        cov += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
    if (es.info() != Eigen::Success) return false;
    eval = es.eigenvalues();        // ascending
    evec = es.eigenvectors();
    return true;
}

// Draw a keep/drop result: kept points green, removed points faded red.
void drawKeepMask(const std::vector<Eigen::Vector3f>& pts, const std::vector<uint8_t>& keep,
                  float psize, debug::DebugDraw& out) {
    const Eigen::Vector3f green(0.1f, 0.9f, 0.2f), red(0.5f, 0.12f, 0.12f);
    for (size_t i = 0; i < pts.size(); ++i)
        out.addPoint(pts[i], keep[i] ? green : red, psize);
}

// Draw a per-point scalar field as a heatmap (auto-ranged over [p5, p95] so a
// few extreme values don't wash out the gradient).
void drawScalarField(const std::vector<Eigen::Vector3f>& pts, std::vector<float> scalar,
                     float psize, debug::DebugDraw& out) {
    if (pts.empty()) return;
    std::vector<float> sorted = scalar;
    std::sort(sorted.begin(), sorted.end());
    float lo = sorted[(size_t)(sorted.size() * 0.05f)];
    float hi = sorted[(size_t)(sorted.size() * 0.95f)];
    if (hi <= lo) hi = lo + 1e-6f;
    for (size_t i = 0; i < pts.size(); ++i)
        out.addPoint(pts[i], color::GetHeatMapColor(scalar[i], lo, hi).head<3>(), psize);
}

// --- Filter: Statistical Outlier Removal (SOR) -------------------------------
// Ported from Helium PointCloudSOR. Mark points whose mean distance to their k
// neighbours is more than mu + alpha*sigma over the whole cloud.
void runSOR(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    float diag = buildGrid(pts, grid, mn, mx);
    const int k = 16;
    const float alpha = 1.0f;

    std::vector<float> meanDist(pts.size(), 0.0f);
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    double sum = 0.0, sum2 = 0.0;
    const size_t step = pts.size() / 100 + 1;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i % step == 0) report(progress, (float)i / (float)pts.size());
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        float m = 0.0f; int n = 0;
        for (size_t t = 1; t < dist.size(); ++t) { m += std::sqrt(dist[t]); ++n; }  // [0] self
        m = n ? m / n : 0.0f;
        meanDist[i] = m;
        sum += m; sum2 += (double)m * m;
    }
    double mu = sum / pts.size();
    double var = std::max(0.0, sum2 / pts.size() - mu * mu);
    double thresh = mu + alpha * std::sqrt(var);

    std::vector<uint8_t> keep(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) keep[i] = meanDist[i] <= (float)thresh;
    drawKeepMask(pts, keep, diag * 0.004f, out);
}

// --- Filter: Radius Outlier Removal (ROR) ------------------------------------
// Ported from Helium PointCloudROR. Drop points with fewer than minN neighbours
// inside a radius tied to the cloud scale.
void runROR(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    float diag = buildGrid(pts, grid, mn, mx);
    const float radius = diag * 0.025f;
    const int minN = 5;

    std::vector<uint8_t> keep(pts.size());
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    const size_t step = pts.size() / 100 + 1;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i % step == 0) report(progress, (float)i / (float)pts.size());
        grid.pointsWithinRadius(pts, pts[i], radius, nbr, dist);
        keep[i] = (int)nbr.size() - 1 >= minN;  // exclude self
    }
    drawKeepMask(pts, keep, diag * 0.004f, out);
}

// --- Filter: Plane-Fitting Outlier Removal (PFOR) ----------------------------
// Ported from Helium PointCloudPFOR. Fit a local plane (PCA) per point; drop
// points lying farther than beta * neighbourhood-extent off that plane.
void runPFOR(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    float diag = buildGrid(pts, grid, mn, mx);
    const int k = 16;
    const float beta = 0.6f;

    std::vector<uint8_t> keep(pts.size(), 1);
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
    const size_t step = pts.size() / 100 + 1;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i % step == 0) report(progress, (float)i / (float)pts.size());
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        if (!neighbourhoodPCA(pts, nbr, c, eval, evec)) continue;
        Eigen::Vector3f n = evec.col(0);                 // plane normal
        float planeDist = std::abs((pts[i] - c).dot(n));
        float extent = std::sqrt(std::max(eval.y(), 1e-12f));  // in-plane spread
        keep[i] = planeDist <= beta * extent;
    }
    drawKeepMask(pts, keep, diag * 0.004f, out);
}

// --- Analyze: Kernel Density Estimation (KDE) --------------------------------
// Ported from Helium PointCloudKDE. Gaussian-kernel local density, shown as a
// heatmap (dense = warm).
void runKDE(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    float diag = buildGrid(pts, grid, mn, mx);
    const float h = diag * 0.03f;                 // bandwidth
    const float radius = h * 3.0f;                // truncate the kernel
    const float invH2 = 1.0f / (h * h);

    std::vector<float> density(pts.size(), 0.0f);
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    const size_t step = pts.size() / 100 + 1;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i % step == 0) report(progress, (float)i / (float)pts.size());
        grid.pointsWithinRadius(pts, pts[i], radius, nbr, dist);
        float d = 0.0f;
        for (float d2 : dist) d += std::exp(-d2 * invH2);  // d2 is squared distance
        density[i] = d;
    }
    drawScalarField(pts, std::move(density), diag * 0.004f, out);
}

// --- Analyze: Surface curvature ----------------------------------------------
// Ported from Helium PointCloudCurvatureAnalysis. Surface variation
// lambda0/(lambda0+lambda1+lambda2) from the neighbourhood PCA, as a heatmap.
void runCurvature(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    float diag = buildGrid(pts, grid, mn, mx);
    const int k = 16;

    std::vector<float> curv(pts.size(), 0.0f);
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
    const size_t step = pts.size() / 100 + 1;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i % step == 0) report(progress, (float)i / (float)pts.size());
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        if (!neighbourhoodPCA(pts, nbr, c, eval, evec)) continue;
        float s = eval.x() + eval.y() + eval.z();
        curv[i] = s > 1e-12f ? eval.x() / s : 0.0f;
    }
    drawScalarField(pts, std::move(curv), diag * 0.004f, out);
}

// --- Analyze: Normal deviation -----------------------------------------------
// Ported from Helium PointCloudNormalDeviation. Per-point angle between its
// normal and the mean normal of its neighbourhood (high = creases/noise).
void runNormalDeviation(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    std::vector<Eigen::Vector3f> nrm =
        in.normals.size() == pts.size()
            ? in.normals
            : geometry::estimateNormals(pts, 16, [&](float f) { report(progress, f * 0.5f); });

    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    float diag = buildGrid(pts, grid, mn, mx);
    const int k = 16;

    std::vector<float> devv(pts.size(), 0.0f);
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    const size_t step = pts.size() / 100 + 1;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (i % step == 0) report(progress, 0.5f + 0.5f * (float)i / (float)pts.size());
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        Eigen::Vector3f avg = Eigen::Vector3f::Zero();
        for (unsigned int j : nbr) avg += nrm[j];
        if (avg.squaredNorm() < 1e-12f) continue;
        avg.normalize();
        float c = std::max(-1.0f, std::min(1.0f, nrm[i].dot(avg)));
        devv[i] = std::acos(c);  // radians
    }
    drawScalarField(pts, std::move(devv), diag * 0.004f, out);
}

// --- Transform: Edge-preserving smoothing ------------------------------------
// CPU reimplementation of Helium's GPU edge-preserving smoothing (it had no CPU
// path). Bilateral Laplacian via point_ops::smoothPoints; draws the smoothed
// cloud colored by per-point displacement (still = blue, moved = warm).
void runSmooth(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(pts, mn, mx).norm();

    std::vector<Eigen::Vector3f> sm = geometry::smoothPoints(
        pts, 5, 0.5f, true, 12, [&](float f) { report(progress, f); });
    float psize = diag * 0.004f;
    float maxDisp = diag * 0.02f;
    for (size_t i = 0; i < pts.size(); ++i) {
        float disp = (sm[i] - pts[i]).norm();
        out.addPoint(sm[i], color::GetHeatMapColor(disp, 0.0f, maxDisp).head<3>(), psize);
    }
}

// --- Transform: ICP registration (self-demo) ---------------------------------
// CPU reimplementation of Helium's GPU GlobalRegistration. The selected cloud is
// the target; a known rigid perturbation makes a source copy, then icpAlign
// recovers the transform. Draws target (blue) and the ICP-aligned source (green)
// -- they should coincide when ICP converges.
void runICP(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& dst = in.points;
    if (dst.size() < 8) return;
    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(dst, mn, mx).norm();
    Eigen::Vector3f center = 0.5f * (mn + mx);

    // Perturb: rotate 18 deg about Y + translate, to make the source cloud.
    Eigen::AngleAxisf rot(0.31416f, Eigen::Vector3f::UnitY());
    Eigen::Vector3f trans = Eigen::Vector3f(0.08f, 0.04f, -0.06f) * diag;
    std::vector<Eigen::Vector3f> src(dst.size());
    for (size_t i = 0; i < dst.size(); ++i) src[i] = rot * (dst[i] - center) + center + trans;

    float rmse = 0.0f; int iters = 0;
    Eigen::Matrix4f T =
        geometry::icpAlign(src, dst, 40, rmse, iters, [&](float f) { report(progress, f); });

    float psize = diag * 0.004f;
    const Eigen::Vector3f blue(0.25f, 0.5f, 1.0f), green(0.1f, 0.9f, 0.2f);
    for (const auto& p : dst) out.addPoint(p, blue, psize);
    for (const auto& p : src) {
        Eigen::Vector4f a = T * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
        out.addPoint(Eigen::Vector3f(a.x(), a.y(), a.z()), green, psize);
    }
}

struct ModeEntry {
    const char*  name;
    void       (*fn)(const ModeInput&, debug::DebugDraw&, const ProgressFn&);
    ModeCategory category;
};
const ModeEntry kModes[] = {
    {"Reconstruct",      runReconstruct,     ModeCategory::Generate},
    {"SDF Filter",       runSdfFilter,       ModeCategory::Generate},
    {"Clustering",       runClustering,      ModeCategory::Analyze},
    {"Curvature",        runCurvature,       ModeCategory::Analyze},
    {"Normal Deviation", runNormalDeviation, ModeCategory::Analyze},
    {"Density (KDE)",    runKDE,             ModeCategory::Analyze},
    {"Outlier: SOR",     runSOR,             ModeCategory::Filter},
    {"Outlier: ROR",     runROR,             ModeCategory::Filter},
    {"Outlier: PFOR",    runPFOR,            ModeCategory::Filter},
    {"Morphology",       runMorphology,      ModeCategory::Filter},
    {"Smooth (bilateral)", runSmooth,        ModeCategory::Transform},
    {"ICP Register",     runICP,             ModeCategory::Transform},
};
constexpr int kModeCount = (int)(sizeof(kModes) / sizeof(kModes[0]));

} // namespace

int modeCount() { return kModeCount; }

const char* modeName(int index) {
    if (index < 0 || index >= kModeCount) return "?";
    return kModes[index].name;
}

ModeCategory modeCategory(int index) {
    if (index < 0 || index >= kModeCount) return ModeCategory::Generate;
    return kModes[index].category;
}

const char* modeCategoryName(ModeCategory c) {
    switch (c) {
        case ModeCategory::Filter:    return "Filter";
        case ModeCategory::Analyze:   return "Analyze";
        case ModeCategory::Generate:  return "Generate";
        case ModeCategory::Transform: return "Transform";
    }
    return "?";
}

void runMode(int index, const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    if (index < 0 || index >= kModeCount) return;
    kModes[index].fn(in, out, progress);
}

} // namespace orange::modes
