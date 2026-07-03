#include "orange/core/modes.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <execution>
#include <functional>
#include <numeric>
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

// Read tunable `i` from the input (falls back to the mode's registered default).
float P(const ModeInput& in, size_t i, float def) {
    return i < in.params.size() ? in.params[i] : def;
}

// Sentinel color: keep the point's original color (see runModeColors).
const Eigen::Vector3f kKeepColor(-1.0f, -1.0f, -1.0f);

// Throttled progress report (fraction in [0,1]).
inline void report(const ProgressFn& p, float f) { if (p) p(f); }

// Run fn(i) for i in [0,n) with std::execution::par, reporting progress from
// lo..hi through the (thread-safe) sink. fn must only write its own output
// slot; SparseGrid queries are const and safe to share across threads. Use
// thread_local scratch vectors inside fn for the kNN/radius outputs.
template <typename F>
void parallelFor(size_t n, const ProgressFn& progress, float lo, float hi, F&& fn) {
    if (n == 0) return;
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    std::atomic<size_t> done{0};
    const size_t step = n / 100 + 1;
    std::for_each(std::execution::par, idx.begin(), idx.end(), [&](uint32_t i) {
        fn((size_t)i);
        size_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (progress && d % step == 0) progress(lo + (hi - lo) * (float)d / (float)n);
    });
}

// --- Mode 0: Euclidean clustering (SparseGrid radius + union-find) -----------
void clusteringColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                      debug::DebugDraw& extras, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.empty()) return;

    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(pts, mn, mx).norm();
    float radius = diag * P(in, 0, 2.0f) * 0.01f;
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

    // Parallel radius queries (the expensive part) gathered chunk by chunk, so
    // the cached neighbour lists stay bounded; the union-find pass over each
    // gathered chunk is serial but cheap.
    const size_t chunk = 65536;
    std::vector<std::vector<unsigned int>> gathered(std::min(chunk, pts.size()));
    for (size_t base = 0; base < pts.size(); base += chunk) {
        const size_t cnt = std::min(chunk, pts.size() - base);
        parallelFor(cnt, {}, 0.0f, 0.0f, [&](size_t t) {
            thread_local std::vector<unsigned int> nbr;
            thread_local std::vector<float> dist;
            const size_t i = base + t;
            grid.pointsWithinRadius(pts, pts[i], radius, nbr, dist);
            auto& out = gathered[t];
            out.clear();
            for (unsigned int j : nbr)
                if (j > i) out.push_back(j);
        });
        for (size_t t = 0; t < cnt; ++t)
            for (unsigned int j : gathered[t]) unite((int)(base + t), (int)j);
        report(progress, (float)(base + cnt) / (float)pts.size());
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
    colors.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        int c = label[i];
        cmin[c] = cmin[c].cwiseMin(pts[i]);
        cmax[c] = cmax[c].cwiseMax(pts[i]);
        colors[i] = indexColor(c);
    }
    for (int c = 0; c < nClusters; ++c)
        extras.addWireBox(cmin[c], cmax[c], indexColor(c), diag * 0.0015f);
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

void morphologyColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                      debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.empty()) return;

    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(pts, mn, mx).norm();
    float voxelSize = diag * P(in, 0, 4.0f) * 0.01f;
    if (voxelSize <= 0.0f) return;
    int erodeIter = (int)std::lround(P(in, 1, 2.0f));
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

    // Classify original points near the surviving core voxels (red = removed,
    // green = kept). The (2e+1)^3 hash probes per point dominate -- parallel.
    int expansion = erodeIter;
    colors.assign(pts.size(), Eigen::Vector3f(0.5f, 0.12f, 0.12f));
    parallelFor(pts.size(), progress, 0.0f, 1.0f, [&](size_t pi) {
        VoxelKey k = voxelOf(pts[pi], inv);
        for (int z = -expansion; z <= expansion; ++z)
            for (int y = -expansion; y <= expansion; ++y)
                for (int x = -expansion; x <= expansion; ++x)
                    if (largest.count({k.x + x, k.y + y, k.z + z})) {
                        colors[pi] = Eigen::Vector3f(0.1f, 0.9f, 0.2f);
                        return;
                    }
    });
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
    const int smoothIter = (int)std::lround(P(in, 0, 2.0f));
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
    // Stays serial: concurrent min-writes to overlapping voxels would race
    // (no std::atomic_ref in C++17); blur + resample below are parallel.
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

    // Box blur near the surface (smooths away thin spikes/noise). Parallel per
    // z-slice: reads `data`, writes disjoint rows of `tmp`.
    report(progress, 0.5f);
    std::vector<float> tmp = data;
    const size_t nz = dim.z() > 2 ? (size_t)dim.z() - 2 : 0;
    for (int it = 0; it < smoothIter; ++it) {
        parallelFor(nz, {}, 0.0f, 0.0f, [&](size_t zi) {
            const int z = (int)zi + 1;
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
        });
        data = tmp;
        report(progress, 0.5f + 0.2f * (float)(it + 1) / (float)smoothIter);
    }

    // Resample the iso-surface as points (color by SDF gradient = normal).
    // Parallel per z-slice into per-slice buckets, then a serial emit
    // (DebugDraw is not thread-safe).
    float iso = voxelSize * 1.5f;
    float psize = voxelSize * 0.4f;
    std::vector<std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>> rows(nz);
    parallelFor(nz, progress, 0.7f, 1.0f, [&](size_t zi) {
        const int z = (int)zi + 1;
        auto& row = rows[zi];
        for (int y = 1; y < dim.y() - 1; ++y)
            for (int x = 1; x < dim.x() - 1; ++x) {
                if (data[idxOf(x, y, z)] >= iso) continue;
                Eigen::Vector3f n(data[idxOf(x + 1, y, z)] - data[idxOf(x - 1, y, z)],
                                  data[idxOf(x, y + 1, z)] - data[idxOf(x, y - 1, z)],
                                  data[idxOf(x, y, z + 1)] - data[idxOf(x, y, z - 1)]);
                n = n.squaredNorm() > 1e-6f ? n.normalized() : Eigen::Vector3f::UnitY();
                row.emplace_back(gridToWorld(x, y, z),
                                 n * 0.5f + Eigen::Vector3f::Constant(0.5f));
            }
    });
    for (const auto& row : rows)
        for (const auto& pc : row) out.addPoint(pc.first, pc.second, psize);
    report(progress, 1.0f);
}

// --- Mode 3: Surface reconstruction (TSDF + dual contouring) -----------------
void runReconstruct(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    const auto& pts = in.points;
    if (pts.empty()) return;

    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(pts, mn, mx).norm();
    float voxelSize = diag * P(in, 0, 3.0f) * 0.01f;

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

// Keep/drop result as per-point colors: kept green, removed faded red.
const Eigen::Vector3f kKeepGreen(0.1f, 0.9f, 0.2f);
const Eigen::Vector3f kDropRed(0.5f, 0.12f, 0.12f);
void keepToColors(const std::vector<uint8_t>& keep, std::vector<Eigen::Vector3f>& colors) {
    colors.resize(keep.size());
    for (size_t i = 0; i < keep.size(); ++i) colors[i] = keep[i] ? kKeepGreen : kDropRed;
}

// Per-point scalar field as heatmap colors (auto-ranged over [p5, p95] so a
// few extreme values don't wash out the gradient).
void scalarToColors(const std::vector<float>& scalar, std::vector<Eigen::Vector3f>& colors) {
    if (scalar.empty()) return;
    std::vector<float> sorted = scalar;
    std::sort(sorted.begin(), sorted.end());
    float lo = sorted[(size_t)(sorted.size() * 0.05f)];
    float hi = sorted[(size_t)(sorted.size() * 0.95f)];
    if (hi <= lo) hi = lo + 1e-6f;
    colors.resize(scalar.size());
    for (size_t i = 0; i < scalar.size(); ++i)
        colors[i] = color::GetHeatMapColor(scalar[i], lo, hi).head<3>();
}

// --- Filter: Statistical Outlier Removal (SOR) -------------------------------
// Ported from Helium PointCloudSOR. Mark points whose mean distance to their k
// neighbours is more than mu + alpha*sigma over the whole cloud.
void sorColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
               debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);
    const int k = (int)std::lround(P(in, 0, 16.0f));
    const float alpha = P(in, 1, 1.0f);

    std::vector<float> meanDist(pts.size(), 0.0f);
    parallelFor(pts.size(), progress, 0.0f, 0.95f, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        float m = 0.0f; int n = 0;
        for (size_t t = 1; t < dist.size(); ++t) { m += std::sqrt(dist[t]); ++n; }  // [0] self
        meanDist[i] = n ? m / n : 0.0f;
    });
    double sum = 0.0, sum2 = 0.0;
    for (float m : meanDist) { sum += m; sum2 += (double)m * m; }
    double mu = sum / pts.size();
    double var = std::max(0.0, sum2 / pts.size() - mu * mu);
    double thresh = mu + alpha * std::sqrt(var);

    std::vector<uint8_t> keep(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) keep[i] = meanDist[i] <= (float)thresh;
    keepToColors(keep, colors);
}

// --- Filter: Radius Outlier Removal (ROR) ------------------------------------
// Ported from Helium PointCloudROR. Drop points with fewer than minN neighbours
// inside a radius tied to the cloud scale.
void rorColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
               debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    float diag = buildGrid(pts, grid, mn, mx);
    const float radius = diag * P(in, 0, 2.5f) * 0.01f;
    const int minN = (int)std::lround(P(in, 1, 5.0f));

    std::vector<uint8_t> keep(pts.size());
    parallelFor(pts.size(), progress, 0.0f, 1.0f, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        grid.pointsWithinRadius(pts, pts[i], radius, nbr, dist);
        keep[i] = (int)nbr.size() - 1 >= minN;  // exclude self
    });
    keepToColors(keep, colors);
}

// --- Filter: Plane-Fitting Outlier Removal (PFOR) ----------------------------
// Ported from Helium PointCloudPFOR. Fit a local plane (PCA) per point; drop
// points lying farther than beta * neighbourhood-extent off that plane.
void pforColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);
    const int k = (int)std::lround(P(in, 0, 16.0f));
    const float beta = P(in, 1, 0.05f);

    std::vector<uint8_t> keep(pts.size(), 1);
    parallelFor(pts.size(), progress, 0.0f, 1.0f, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        if (!neighbourhoodPCA(pts, nbr, c, eval, evec)) return;
        Eigen::Vector3f n = evec.col(0);                 // plane normal
        float planeDist = std::abs((pts[i] - c).dot(n));
        float extent = std::sqrt(std::max(eval.y(), 1e-12f));  // in-plane spread
        keep[i] = planeDist <= beta * extent;
    });
    keepToColors(keep, colors);
}

// PFOR "Fit": instead of dropping the outliers, project them onto their local
// fitted plane -- pulls stray points back onto the surface. Inliers stay put.
void pforFit(const ModeInput& in, std::vector<Eigen::Vector3f>& fitted,
             const ProgressFn& progress) {
    const auto& pts = in.points;
    fitted = pts;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);
    const int k = (int)std::lround(P(in, 0, 16.0f));
    const float beta = P(in, 1, 0.05f);

    parallelFor(pts.size(), progress, 0.0f, 1.0f, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        if (!neighbourhoodPCA(pts, nbr, c, eval, evec)) return;
        Eigen::Vector3f n = evec.col(0);
        float d = (pts[i] - c).dot(n);
        float extent = std::sqrt(std::max(eval.y(), 1e-12f));
        if (std::abs(d) > beta * extent) fitted[i] = pts[i] - n * d;  // project outlier
    });
}

// --- Filter: Bump detection / removal (trimmed-plane residual) ----------------
// Separate noise-like bump blobs sticking out of an otherwise smooth surface.
// Per point: gather a bump-sized radius neighbourhood, fit a PCA plane, then
// REFIT on the closer half of the neighbours -- the trim excludes the bump
// itself from the reference surface (a plain smoothing residual fails here:
// the bump drags the local average toward itself and the residual dies). The
// point's distance to the trimmed plane is its bump height. Threshold robustly
// (median + k*MAD over the whole cloud), then grow the seeds through a lower
// hysteresis threshold over radius neighbours so whole blobs are marked, not
// just their tips. SOR/ROR can't catch these: bump interiors have normal
// density, so only a surface-residual test separates them.
std::vector<uint8_t> computeBumpMask(const ModeInput& in, float diag,
                                     const ProgressFn& progress) {
    const auto& pts = in.points;
    geometry::SparseGrid grid;
    const float radius = diag * P(in, 0, 3.0f) * 0.01f;  // reference scale (≈ 2x bump size)
    grid.build(pts, radius);

    std::vector<float> res(pts.size(), 0.0f);
    parallelFor(pts.size(), progress, 0.0f, 0.85f, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr, sub, trimmedNbr;
        thread_local std::vector<float> dist, planeDist, pd;
        Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
        grid.pointsWithinRadius(pts, pts[i], radius, nbr, dist);
        if (nbr.size() < 12) return;  // sparse fringe: no reliable reference plane
        sub.clear();                  // cap the fit cost on dense clouds
        size_t stride = nbr.size() / 64 + 1;
        for (size_t t = 0; t < nbr.size(); t += stride) sub.push_back(nbr[t]);
        if (!neighbourhoodPCA(pts, sub, c, eval, evec)) return;
        Eigen::Vector3f n = evec.col(0);
        planeDist.resize(sub.size());
        for (size_t t = 0; t < sub.size(); ++t)
            planeDist[t] = std::abs((pts[sub[t]] - c).dot(n));
        pd = planeDist;
        size_t half = pd.size() / 2;
        std::nth_element(pd.begin(), pd.begin() + half, pd.end());
        float medPd = pd[half];
        trimmedNbr.clear();
        for (size_t t = 0; t < sub.size(); ++t)
            if (planeDist[t] <= medPd) trimmedNbr.push_back(sub[t]);
        if (neighbourhoodPCA(pts, trimmedNbr, c, eval, evec)) n = evec.col(0);
        res[i] = std::abs((pts[i] - c).dot(n));
    });

    // LOCAL reference residual, not a global threshold: on a scan whose surface
    // complexity varies (smooth patches next to rough detailed areas) a single
    // global median+MAD is dominated by the rough regions -- bumps on smooth
    // patches fall below it while ordinary rough-region points poke above it
    // (detects the wrong places). Instead each point's residual is scored
    // against the typical residual of its own surroundings: median residual per
    // coarse voxel, then per point the median over the 3x3x3 voxel
    // neighbourhood. A noise bump on a smooth patch scores res/ref >> 1; points
    // in an evenly rough region score ~1 however big their absolute residual.
    const float invVox = 1.0f / (radius * 3.0f);
    std::unordered_map<VoxelKey, std::vector<float>, VoxelHash> voxRes;
    for (size_t i = 0; i < pts.size(); ++i)
        voxRes[voxelOf(pts[i], invVox)].push_back(res[i]);
    std::unordered_map<VoxelKey, float, VoxelHash> voxMed;
    for (auto& kv : voxRes) {
        auto& v = kv.second;
        size_t h = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + h, v.end());
        voxMed[kv.first] = v[h];
    }
    const float refFloor = diag * 1e-5f;
    std::vector<float> score(pts.size(), 0.0f);
    parallelFor(pts.size(), {}, 0.0f, 0.0f, [&](size_t i) {
        thread_local std::vector<float> nbrMed;
        VoxelKey k = voxelOf(pts[i], invVox);
        nbrMed.clear();
        for (int z = -1; z <= 1; ++z)
            for (int y = -1; y <= 1; ++y)
                for (int x = -1; x <= 1; ++x) {
                    auto it = voxMed.find({k.x + x, k.y + y, k.z + z});
                    if (it != voxMed.end()) nbrMed.push_back(it->second);
                }
        size_t h = nbrMed.size() / 2;
        std::nth_element(nbrMed.begin(), nbrMed.begin() + h, nbrMed.end());
        float ref = std::max(nbrMed.empty() ? refFloor : nbrMed[h], refFloor);
        score[i] = res[i] / ref;
    });

    // Hysteresis region growing: seeds score high, expand through neighbours
    // scoring above the lower bar. (Serial: BFS order dependence.)
    const float hiScore = P(in, 1, 5.0f), loScore = P(in, 2, 3.0f);
    const float growRadius = diag * 0.02f;
    std::vector<unsigned int> nbr;
    std::vector<float> dist;
    std::vector<uint8_t> bump(pts.size(), 0);
    std::queue<size_t> q;
    for (size_t i = 0; i < pts.size(); ++i)
        if (score[i] > hiScore) { bump[i] = 1; q.push(i); }
    while (!q.empty()) {
        size_t i = q.front();
        q.pop();
        grid.pointsWithinRadius(pts, pts[i], growRadius, nbr, dist);
        for (unsigned int j : nbr)
            if (!bump[j] && score[j] > loScore) { bump[j] = 1; q.push(j); }
    }

    // Only NOISE bumps, not real geometry: legitimate features (cusps, ridges)
    // also fit a plane badly at this scale, but they flag as LARGE connected
    // regions while noise bumps are small isolated blobs. BFS the flagged
    // points into components and unflag any component wider than a cap.
    const float maxBlobExtent = diag * P(in, 3, 5.0f) * 0.01f;
    std::vector<uint8_t> visited(pts.size(), 0);
    std::vector<size_t> comp;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (!bump[i] || visited[i]) continue;
        comp.clear();
        std::queue<size_t> bfs;
        bfs.push(i);
        visited[i] = 1;
        Eigen::Vector3f cmn = pts[i], cmx = pts[i];
        while (!bfs.empty()) {
            size_t a = bfs.front();
            bfs.pop();
            comp.push_back(a);
            cmn = cmn.cwiseMin(pts[a]);
            cmx = cmx.cwiseMax(pts[a]);
            grid.pointsWithinRadius(pts, pts[a], growRadius, nbr, dist);
            for (unsigned int j : nbr)
                if (bump[j] && !visited[j]) { visited[j] = 1; bfs.push(j); }
        }
        // Too wide = real geometry; too few points = lone speckle, not a bump.
        if ((cmx - cmn).norm() > maxBlobExtent || comp.size() < 8)
            for (size_t a : comp) bump[a] = 0;
    }
    report(progress, 1.0f);
    return bump;
}

// Detect visualization: flagged points red, everything else keeps its color.
void bumpDetectColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                      debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(pts, mn, mx).norm();
    std::vector<uint8_t> bump = computeBumpMask(in, diag, progress);

    const Eigen::Vector3f red(1.0f, 0.05f, 0.05f);
    colors.assign(pts.size(), kKeepColor);
    for (size_t i = 0; i < pts.size(); ++i)
        if (bump[i]) colors[i] = red;
}

// --- Analyze: Kernel Density Estimation (KDE) --------------------------------
// Ported from Helium PointCloudKDE. Gaussian-kernel local density, shown as a
// heatmap (dense = warm).
void kdeColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
               debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    float diag = buildGrid(pts, grid, mn, mx);
    const float h = diag * P(in, 0, 3.0f) * 0.01f;  // bandwidth
    const float radius = h * 3.0f;                // truncate the kernel
    const float invH2 = 1.0f / (h * h);

    std::vector<float> density(pts.size(), 0.0f);
    parallelFor(pts.size(), progress, 0.0f, 1.0f, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        grid.pointsWithinRadius(pts, pts[i], radius, nbr, dist);
        float d = 0.0f;
        for (float d2 : dist) d += std::exp(-d2 * invH2);  // d2 is squared distance
        density[i] = d;
    });
    scalarToColors(density, colors);
}

// --- Analyze: Surface curvature ----------------------------------------------
// Ported from Helium PointCloudCurvatureAnalysis. Surface variation
// lambda0/(lambda0+lambda1+lambda2) from the neighbourhood PCA, as a heatmap.
void curvatureColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                     debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);
    const int k = (int)std::lround(P(in, 0, 16.0f));

    std::vector<float> curv(pts.size(), 0.0f);
    parallelFor(pts.size(), progress, 0.0f, 1.0f, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        if (!neighbourhoodPCA(pts, nbr, c, eval, evec)) return;
        float s = eval.x() + eval.y() + eval.z();
        curv[i] = s > 1e-12f ? eval.x() / s : 0.0f;
    });
    scalarToColors(curv, colors);
}

// --- Analyze: Normal deviation -----------------------------------------------
// Ported from Helium PointCloudNormalDeviation. Per-point angle between its
// normal and the mean normal of its neighbourhood (high = creases/noise).
void normalDeviationColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                           debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    const int k = (int)std::lround(P(in, 0, 16.0f));
    std::vector<Eigen::Vector3f> nrm =
        in.normals.size() == pts.size()
            ? in.normals
            : geometry::estimateNormals(pts, k, [&](float f) { report(progress, f * 0.5f); });

    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);

    std::vector<float> devv(pts.size(), 0.0f);
    parallelFor(pts.size(), progress, 0.5f, 1.0f, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        Eigen::Vector3f avg = Eigen::Vector3f::Zero();
        for (unsigned int j : nbr) avg += nrm[j];
        if (avg.squaredNorm() < 1e-12f) return;
        avg.normalize();
        float c = std::max(-1.0f, std::min(1.0f, nrm[i].dot(avg)));
        devv[i] = std::acos(c);  // radians
    });
    scalarToColors(devv, colors);
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
        pts, (int)std::lround(P(in, 0, 5.0f)), P(in, 1, 0.5f), true, 12,
        [&](float f) { report(progress, f); });
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
    Eigen::Matrix4f T = geometry::icpAlign(src, dst, (int)std::lround(P(in, 0, 40.0f)), rmse,
                                           iters, [&](float f) { report(progress, f); });

    float psize = diag * 0.004f;
    const Eigen::Vector3f blue(0.25f, 0.5f, 1.0f), green(0.1f, 0.9f, 0.2f);
    for (const auto& p : dst) out.addPoint(p, blue, psize);
    for (const auto& p : src) {
        Eigen::Vector4f a = T * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
        out.addPoint(Eigen::Vector3f(a.x(), a.y(), a.z()), green, psize);
    }
}

using DrawFn  = void (*)(const ModeInput&, debug::DebugDraw&, const ProgressFn&);
using ColorFn = void (*)(const ModeInput&, std::vector<Eigen::Vector3f>&, debug::DebugDraw&,
                         const ProgressFn&);
using FitFn   = void (*)(const ModeInput&, std::vector<Eigen::Vector3f>&, const ProgressFn&);
using PointsFn = FitFn;  // same shape: points in -> points out

// Points-stage wrapper for the keep/drop filters: run the colors fn, keep the
// points it marked kKeepGreen.
void pointsFromKeepColors(ColorFn cf, const ModeInput& in,
                          std::vector<Eigen::Vector3f>& outPts, const ProgressFn& progress) {
    std::vector<Eigen::Vector3f> colors;
    debug::DebugDraw extras;
    cf(in, colors, extras, progress);
    outPts.clear();
    size_t n = std::min(colors.size(), in.points.size());
    outPts.reserve(n);
    for (size_t i = 0; i < n; ++i)
        if ((colors[i] - kKeepGreen).squaredNorm() < 1e-8f) outPts.push_back(in.points[i]);
}
void sorPoints(const ModeInput& in, std::vector<Eigen::Vector3f>& o, const ProgressFn& p) {
    pointsFromKeepColors(sorColors, in, o, p);
}
void rorPoints(const ModeInput& in, std::vector<Eigen::Vector3f>& o, const ProgressFn& p) {
    pointsFromKeepColors(rorColors, in, o, p);
}
void pforPoints(const ModeInput& in, std::vector<Eigen::Vector3f>& o, const ProgressFn& p) {
    pointsFromKeepColors(pforColors, in, o, p);
}
void morphologyPoints(const ModeInput& in, std::vector<Eigen::Vector3f>& o, const ProgressFn& p) {
    pointsFromKeepColors(morphologyColors, in, o, p);
}
void smoothStagePoints(const ModeInput& in, std::vector<Eigen::Vector3f>& o, const ProgressFn& p) {
    o = geometry::smoothPoints(in.points, (int)std::lround(P(in, 0, 5.0f)), P(in, 1, 0.5f),
                               true, 12, p);
}
void bumpRemovePoints(const ModeInput& in, std::vector<Eigen::Vector3f>& o, const ProgressFn& p) {
    o = in.points;
    if (in.points.size() < 8) return;
    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(in.points, mn, mx).norm();
    std::vector<uint8_t> bump = computeBumpMask(in, diag, p);
    o.clear();
    for (size_t i = 0; i < in.points.size(); ++i)
        if (!bump[i]) o.push_back(in.points[i]);
}
struct ModeEntry {
    const char*  name;
    DrawFn       fn;        // Draw modes (null for Recolor/Remove)
    ModeCategory category;
    ApplyKind    kind      = ApplyKind::Draw;
    ColorFn      colorFn   = nullptr;  // Recolor modes (also the Remove fallback viz)
    int          numParams = 0;
    ModeParam    params[4] = {};
    FitFn        fitFn     = nullptr;  // optional "Fit" action (dialog button)
    PointsFn     pointsFn  = nullptr;  // optional points->points pipeline stage
};
const ModeParam kBumpParams[4] = {
    {"Fit Radius %", 1.0f, 8.0f, 3.0f, false},
    {"Hi Score", 2.0f, 12.0f, 5.0f, false},
    {"Lo Score", 1.0f, 8.0f, 3.0f, false},
    {"Max Blob %", 1.0f, 20.0f, 5.0f, false},
};
const ModeEntry kModes[] = {
    {"Reconstruct", runReconstruct, ModeCategory::Generate, ApplyKind::Draw, nullptr, 1,
     {{"Voxel %", 0.5f, 10.0f, 3.0f, false}}},
    {"SDF Filter", runSdfFilter, ModeCategory::Generate, ApplyKind::Draw, nullptr, 1,
     {{"Blur Iters", 0.0f, 5.0f, 2.0f, true}}},
    {"Clustering", nullptr, ModeCategory::Analyze, ApplyKind::Recolor, clusteringColors, 1,
     {{"Radius %", 0.5f, 10.0f, 2.0f, false}}},
    {"Curvature", nullptr, ModeCategory::Analyze, ApplyKind::Recolor, curvatureColors, 1,
     {{"K Neighbors", 6.0f, 64.0f, 16.0f, true}}},
    {"Normal Deviation", nullptr, ModeCategory::Analyze, ApplyKind::Recolor,
     normalDeviationColors, 1, {{"K Neighbors", 6.0f, 64.0f, 16.0f, true}}},
    {"Density (KDE)", nullptr, ModeCategory::Analyze, ApplyKind::Recolor, kdeColors, 1,
     {{"Bandwidth %", 0.5f, 10.0f, 3.0f, false}}},
    {"Outlier: SOR", nullptr, ModeCategory::Filter, ApplyKind::Recolor, sorColors, 2,
     {{"K Neighbors", 6.0f, 64.0f, 16.0f, true}, {"Alpha", 0.25f, 4.0f, 1.0f, false}},
     nullptr, sorPoints},
    {"Outlier: ROR", nullptr, ModeCategory::Filter, ApplyKind::Recolor, rorColors, 2,
     {{"Radius %", 0.5f, 10.0f, 2.5f, false}, {"Min Nbrs", 1.0f, 32.0f, 5.0f, true}},
     nullptr, rorPoints},
    {"Outlier: PFOR", nullptr, ModeCategory::Filter, ApplyKind::Recolor, pforColors, 2,
     {{"K Neighbors", 4.0f, 256.0f, 16.0f, true},
      {"Beta", 0.0f, 0.1f, 0.05f, false, 0.001f}},
     pforFit, pforPoints},
    {"Morphology", nullptr, ModeCategory::Filter, ApplyKind::Recolor, morphologyColors, 2,
     {{"Voxel %", 1.0f, 10.0f, 4.0f, false}, {"Erode Iters", 1.0f, 4.0f, 2.0f, true}},
     nullptr, morphologyPoints},
    {"Bump: Detect", nullptr, ModeCategory::Filter, ApplyKind::Recolor, bumpDetectColors, 4,
     {kBumpParams[0], kBumpParams[1], kBumpParams[2], kBumpParams[3]}},
    {"Bump: Remove", nullptr, ModeCategory::Filter, ApplyKind::Remove, bumpDetectColors, 4,
     {kBumpParams[0], kBumpParams[1], kBumpParams[2], kBumpParams[3]}, nullptr,
     bumpRemovePoints},
    {"Smooth (bilateral)", runSmooth, ModeCategory::Transform, ApplyKind::Draw, nullptr, 2,
     {{"Iterations", 1.0f, 20.0f, 5.0f, true}, {"Lambda", 0.1f, 1.0f, 0.5f, false}},
     nullptr, smoothStagePoints},
    {"ICP Register", runICP, ModeCategory::Transform, ApplyKind::Draw, nullptr, 1,
     {{"Iterations", 5.0f, 100.0f, 40.0f, true}}},
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
    const ModeEntry& m = kModes[index];
    if (m.colorFn) {
        // Fallback drawing path (fixed ctx inputs / tests): compute the colors,
        // then draw them as debug points.
        std::vector<Eigen::Vector3f> colors;
        m.colorFn(in, colors, out, progress);
        Eigen::Vector3f mn, mx;
        float diag  = boundsExtent(in.points, mn, mx).norm();
        float psize = diag * 0.004f;
        const Eigen::Vector3f gray(0.75f, 0.75f, 0.75f);
        size_t n = std::min(colors.size(), in.points.size());
        for (size_t i = 0; i < n; ++i)
            out.addPoint(in.points[i], colors[i].x() < 0.0f ? gray : colors[i], psize);
    } else if (m.fn) {
        m.fn(in, out, progress);
    }
}

ApplyKind modeApplyKind(int index) {
    if (index < 0 || index >= kModeCount) return ApplyKind::Draw;
    return kModes[index].kind;
}

int modeParamCount(int index) {
    if (index < 0 || index >= kModeCount) return 0;
    return kModes[index].numParams;
}

ModeParam modeParam(int index, int p) {
    if (index < 0 || index >= kModeCount || p < 0 || p >= kModes[index].numParams)
        return {"?", 0.0f, 1.0f, 0.0f, false};
    return kModes[index].params[p];
}

void runModeColors(int index, const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                   debug::DebugDraw& extras, const ProgressFn& progress) {
    colors.clear();
    if (index < 0 || index >= kModeCount || !kModes[index].colorFn) return;
    kModes[index].colorFn(in, colors, extras, progress);
}

void runModeMask(int index, const ModeInput& in, std::vector<uint8_t>& mask,
                 const ProgressFn& progress) {
    mask.assign(in.points.size(), 0);
    if (index < 0 || index >= kModeCount || kModes[index].kind != ApplyKind::Remove) return;
    if (in.points.size() < 8) return;
    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(in.points, mn, mx).norm();
    mask = computeBumpMask(in, diag, progress);
}

bool modeCanFit(int index) {
    if (index < 0 || index >= kModeCount) return false;
    return kModes[index].fitFn != nullptr;
}

bool modeCanTransformPoints(int index) {
    if (index < 0 || index >= kModeCount) return false;
    return kModes[index].pointsFn != nullptr;
}

void runModePoints(int index, const ModeInput& in, std::vector<Eigen::Vector3f>& outPoints,
                   const ProgressFn& progress) {
    outPoints = in.points;
    if (index < 0 || index >= kModeCount || !kModes[index].pointsFn) return;
    kModes[index].pointsFn(in, outPoints, progress);
}

int modeIndexByName(const char* name) {
    for (int i = 0; i < kModeCount; ++i)
        if (std::strcmp(kModes[i].name, name) == 0) return i;
    return -1;
}

void runModeFit(int index, const ModeInput& in, std::vector<Eigen::Vector3f>& fitted,
                const ProgressFn& progress) {
    fitted = in.points;
    if (index < 0 || index >= kModeCount || !kModes[index].fitFn) return;
    kModes[index].fitFn(in, fitted, progress);
}

} // namespace orange::modes
