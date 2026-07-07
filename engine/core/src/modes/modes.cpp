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
#include "orange/core/compare.h"  // compareBandColor (diverging signed map)
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
// Ported from Hydrogen AppSDFFiltering (SDFEngine). The core resampler is
// shared: the Draw visualization colors the samples by their gradient normal,
// and the Fit action replaces the source cloud with them (so SDF Filter can
// actually APPLY its result, like the other operators, not just preview it).
void sdfResample(const ModeInput& in, std::vector<Eigen::Vector3f>& outPts,
                 std::vector<Eigen::Vector3f>& outNrm, float& outPointSize,
                 const ProgressFn& progress) {
    outPts.clear();
    outNrm.clear();
    outPointSize = 0.0f;
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

    // Resample the iso-surface as points (with the SDF gradient as normal).
    // Parallel per z-slice into per-slice buckets, then a serial collect.
    float iso = voxelSize * 1.5f;
    outPointSize = voxelSize * 0.4f;
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
                row.emplace_back(gridToWorld(x, y, z), n);
            }
    });
    for (const auto& row : rows)
        for (const auto& pn : row) { outPts.push_back(pn.first); outNrm.push_back(pn.second); }
    report(progress, 1.0f);
}

void runSdfFilter(const ModeInput& in, debug::DebugDraw& out, const ProgressFn& progress) {
    std::vector<Eigen::Vector3f> ps, ns;
    float psize = 0.0f;
    sdfResample(in, ps, ns, psize, progress);
    for (size_t i = 0; i < ps.size(); ++i)
        out.addPoint(ps[i], ns[i] * 0.5f + Eigen::Vector3f::Constant(0.5f), psize);
}

// SDF Filter "Fit": replace the cloud with the denoised resample. The count
// changes, so the host swaps the whole vertex set (colors transferred from the
// nearest original points) instead of moving points in place.
void sdfFitPoints(const ModeInput& in, std::vector<Eigen::Vector3f>& fitted,
                  const ProgressFn& progress) {
    std::vector<Eigen::Vector3f> ns;
    float psize = 0.0f;
    sdfResample(in, fitted, ns, psize, progress);
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

// Raw normal-field divergence per point: the average radial rate of change of
// the normal toward its kNN neighbours (positive = normals fan out = convex,
// negative = fan in = concave). Shared by the Normal Divergence analyze mode
// and the divergence-gated QFOR filter. Uses the input's normals when present,
// else estimates them (that estimation spans the first half of [p0,p1]).
std::vector<float> normalDivergenceField(const ModeInput& in, int k,
                                         const ProgressFn& progress, float p0, float p1) {
    const auto& pts = in.points;
    std::vector<float> divg(pts.size(), 0.0f);
    if (pts.size() < 8) return divg;
    const bool haveN = in.normals.size() == pts.size();
    const std::vector<Eigen::Vector3f> nrm =
        haveN ? in.normals
              : geometry::estimateNormals(pts, k, [&](float f) {
                    report(progress, p0 + f * 0.5f * (p1 - p0));
                });
    const float pm = haveN ? p0 : p0 + 0.5f * (p1 - p0);

    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);
    parallelFor(pts.size(), progress, pm, p1, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        float s = 0.0f; int n = 0;
        for (size_t t = 1; t < nbr.size(); ++t) {  // [0] is the point itself
            Eigen::Vector3f d = pts[nbr[t]] - pts[i];
            float d2 = d.squaredNorm();
            if (d2 < 1e-20f) continue;
            Eigen::Vector3f nj = nrm[nbr[t]];
            if (nj.dot(nrm[i]) < 0.0f) nj = -nj;  // orientation-flip guard
            s += (nj - nrm[i]).dot(d) / d2;
            ++n;
        }
        divg[i] = n ? s / (float)n : 0.0f;
    });
    return divg;
}

// 95th-percentile magnitude of a scalar field -- the normalization scale the
// divergence colors saturate at and the gated QFOR's Div Thresh is relative to.
float fieldRange95(const std::vector<float>& v) {
    std::vector<float> mag(v.size());
    for (size_t i = 0; i < v.size(); ++i) mag[i] = std::fabs(v[i]);
    size_t p95 = std::min(mag.size() - 1, (size_t)((double)mag.size() * 0.95));
    std::nth_element(mag.begin(), mag.begin() + p95, mag.end());
    return std::max(mag[p95], 1e-9f);
}

// --- Filter: Quadric-Fitting Outlier Removal (QFOR) --------------------------
// PFOR's plane test reads smooth curvature as off-surface distance (on a sphere
// of radius R a valid point sits ~r^2/2R off its neighbourhood plane, so tight
// thresholds flag curved-but-clean regions). QFOR fits a leave-one-out QUADRIC
// w = q(u,v) in the point's local PCA frame instead: smooth curvature is
// absorbed by the quadric coefficients, so the residual left over is pure
// off-surface displacement. On planar data it degenerates to PFOR's answer.
// Leave-one-out matters: the point under test is EXCLUDED from its own fit
// (frame and quadric come from the neighbours only) so a spike cannot drag its
// reference surface toward itself.
//
// Returns the SIGNED residual along the local normal per point (0 where the
// neighbourhood is degenerate) plus that normal, for the projection in Fit.
// Thresholding happens globally in the callers: sigma = 1.4826 * median(|res|)
// (a robust MAD scale -- mean/stddev would be dragged by the very outliers
// we're hunting), outlier when |res| > alpha * sigma. Params: 0 = K, 1 = Alpha.
std::vector<float> qforResiduals(const ModeInput& in, std::vector<Eigen::Vector3f>& normals,
                                 const ProgressFn& progress, float p0, float p1) {
    const auto& pts = in.points;
    std::vector<float> res(pts.size(), 0.0f);
    normals.assign(pts.size(), Eigen::Vector3f::UnitY());
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);
    const int k = (int)std::lround(P(in, 0, 24.0f));

    parallelFor(pts.size(), progress, p0, p1, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr, oth;
        thread_local std::vector<float> dist;
        Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        // Sparse fringe: fewer than half the requested neighbours found -> the
        // reference surface would be under-sampled, skip (point stays an
        // inlier). Absolute floor of 9 keeps the 6-unknown fit overdetermined
        // even at small K.
        if ((int)nbr.size() - 1 < std::max(9, k / 2)) return;
        oth.assign(nbr.begin() + 1, nbr.end());  // [0] is the query point itself
        if (!neighbourhoodPCA(pts, oth, c, eval, evec)) return;
        const Eigen::Vector3f n  = evec.col(0);
        const Eigen::Vector3f t1 = evec.col(2), t2 = evec.col(1);
        // u,v scaled by the in-plane spread so the 6x6 system stays conditioned
        // regardless of the cloud's absolute scale.
        const float s = 1.0f / std::max(std::sqrt(std::max(eval.y() / (float)oth.size(), 0.0f)),
                                        1e-12f);
        // Local frame coordinates of the neighbours (and the query point).
        thread_local std::vector<Eigen::Vector3f> loc;
        loc.resize(oth.size());
        for (size_t t = 0; t < oth.size(); ++t) {
            Eigen::Vector3f d = pts[oth[t]] - c;
            loc[t] = {d.dot(t1) * s, d.dot(t2) * s, d.dot(n)};
        }
        const Eigen::Vector3f di = pts[i] - c;
        const float ui = di.dot(t1) * s, vi = di.dot(t2) * s, wi = di.dot(n);

        // Two-pass robust LS (same trick as mlsCompute): the first fit's
        // high-residual neighbours -- other spikes sitting in this window --
        // are down-weighted for the refit, so one outlier can't drag another
        // outlier's reference surface toward itself.
        thread_local std::vector<float> rob, rres;
        rob.assign(oth.size(), 1.0f);
        float d = wi;  // degenerate fit: fall back to the plane residual
        for (int pass = 0; pass < 2; ++pass) {
            Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
            for (size_t t = 0; t < oth.size(); ++t) {
                float u = loc[t].x(), v = loc[t].y();
                Eigen::Matrix<float, 6, 1> phi;
                phi << u * u, u * v, v * v, u, v, 1.0f;
                A += rob[t] * (phi * phi.transpose());
                b += rob[t] * (phi * loc[t].z());
            }
            Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt(A);
            if (ldlt.info() != Eigen::Success) break;
            Eigen::Matrix<float, 6, 1> q = ldlt.solve(b);
            d = wi - (q[0] * ui * ui + q[1] * ui * vi + q[2] * vi * vi +
                      q[3] * ui + q[4] * vi + q[5]);
            if (pass == 1) break;
            rres.resize(oth.size());
            for (size_t t = 0; t < oth.size(); ++t) {
                float u = loc[t].x(), v = loc[t].y();
                rres[t] = std::fabs(loc[t].z() -
                                    (q[0] * u * u + q[1] * u * v + q[2] * v * v +
                                     q[3] * u + q[4] * v + q[5]));
            }
            thread_local std::vector<float> tmp;
            tmp = rres;
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
            // Floor the scale at a fraction of the in-plane spread (1/s): on an
            // exactly-fittable surface the MAD is ~0 and an unfloored sigma
            // would zero every weight, feeding the refit a singular system.
            float sig    = std::max(1.4826f * tmp[tmp.size() / 2], 1e-4f / s);
            float inv2s2 = 1.0f / (2.0f * (3.0f * sig) * (3.0f * sig));
            for (size_t t = 0; t < oth.size(); ++t)
                rob[t] = std::exp(-rres[t] * rres[t] * inv2s2);
        }
        res[i]     = d;
        normals[i] = n;
    });
    return res;
}

// Robust global outlier threshold over the residuals: alpha * (1.4826 * MAD).
float qforThreshold(const ModeInput& in, const std::vector<float>& res) {
    std::vector<float> mag(res.size());
    for (size_t i = 0; i < res.size(); ++i) mag[i] = std::fabs(res[i]);
    size_t h = mag.size() / 2;
    std::nth_element(mag.begin(), mag.begin() + h, mag.end());
    float sigma = 1.4826f * mag[h];
    Eigen::Vector3f mn, mx;
    sigma = std::max(sigma, boundsExtent(in.points, mn, mx).norm() * 1e-7f);  // exact-plane guard
    return P(in, 1, 3.0f) * sigma;
}

void qforColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 16) return;
    std::vector<Eigen::Vector3f> nrm;
    std::vector<float> res = qforResiduals(in, nrm, progress, 0.0f, 0.98f);
    const float thresh = qforThreshold(in, res);
    std::vector<uint8_t> keep(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) keep[i] = std::fabs(res[i]) <= thresh;
    keepToColors(keep, colors);
}

// QFOR "Fit": project the outliers onto their local quadric (along the fitted
// normal by the residual) instead of dropping them. Inliers stay put.
void qforFit(const ModeInput& in, std::vector<Eigen::Vector3f>& fitted,
             const ProgressFn& progress) {
    const auto& pts = in.points;
    fitted = pts;
    if (pts.size() < 16) return;
    std::vector<Eigen::Vector3f> nrm;
    std::vector<float> res = qforResiduals(in, nrm, progress, 0.0f, 0.98f);
    const float thresh = qforThreshold(in, res);
    for (size_t i = 0; i < pts.size(); ++i)
        if (std::fabs(res[i]) > thresh) fitted[i] = pts[i] - nrm[i] * res[i];
}

// --- Filter: QFOR gated by normal divergence ----------------------------------
// Run QFOR only on points whose normal-field divergence crosses a threshold:
// "consider a point an outlier candidate only where the normal field says
// something is happening" (spike tips fan normals OUT, pits fan them IN).
// Points outside the gate are always kept, however far off their quadric they
// sit -- this protects deliberate sharp detail whose residual is high but
// whose normal field is orderly. The divergence is normalized by its 95th
// percentile magnitude (the same scale the Normal Divergence mode's colors
// saturate at), so Div Thresh is unit-free. Side picks the gate direction:
// 0 = div > +t (convex, spikes), 1 = div < -t (concave, pits), 2 = |div| > t
// (both). The QFOR core reads params 0/1 (K, Alpha) at the same indices as the
// plain QFOR mode. Params: 0 = K, 1 = Alpha, 2 = Div Thresh, 3 = Side.
void qforDivFlags(const ModeInput& in, std::vector<uint8_t>& drop,
                  std::vector<Eigen::Vector3f>& nrmOut, std::vector<float>& resOut,
                  const ProgressFn& progress) {
    const auto& pts = in.points;
    drop.assign(pts.size(), 0);
    resOut.clear();
    if (pts.size() < 16) return;
    const int k = (int)std::lround(P(in, 0, 24.0f));
    const std::vector<float> divg = normalDivergenceField(in, k, progress, 0.0f, 0.4f);
    const float t    = P(in, 2, 0.5f) * fieldRange95(divg);
    const int   side = (int)std::lround(P(in, 3, 2.0f));

    resOut = qforResiduals(in, nrmOut, progress, 0.4f, 0.98f);
    const float thresh = qforThreshold(in, resOut);
    for (size_t i = 0; i < pts.size(); ++i) {
        bool gated = side == 0   ? divg[i] > t
                     : side == 1 ? divg[i] < -t
                                 : std::fabs(divg[i]) > t;
        drop[i] = gated && std::fabs(resOut[i]) > thresh;
    }
}

void qforDivColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                   debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    std::vector<uint8_t> drop;
    std::vector<Eigen::Vector3f> nrm;
    std::vector<float> res;
    qforDivFlags(in, drop, nrm, res, progress);
    std::vector<uint8_t> keep(drop.size());
    for (size_t i = 0; i < drop.size(); ++i) keep[i] = !drop[i];
    keepToColors(keep, colors);
}

void qforDivFit(const ModeInput& in, std::vector<Eigen::Vector3f>& fitted,
                const ProgressFn& progress) {
    fitted = in.points;
    std::vector<uint8_t> drop;
    std::vector<Eigen::Vector3f> nrm;
    std::vector<float> res;
    qforDivFlags(in, drop, nrm, res, progress);
    for (size_t i = 0; i < drop.size(); ++i)
        if (drop[i]) fitted[i] = in.points[i] - nrm[i] * res[i];
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

// --- Analyze: Surface deviation vs the MLS "energy" surface -------------------
// Each point's neighbourhood contributes a Gaussian-weighted (energy) vote to a
// local plane fit -- order-1 moving least squares. The point is then colored by
// its SIGNED distance to that estimated surface: where its energy inflow says
// the surface runs vs where the point actually sits. Above the surface (along
// the oriented normal) = warm, below = cool, on it = green; the diverging map
// is symmetric around zero (95th percentile of |deviation|). Fit projects every
// point onto its MLS plane (one MLS smoothing/projection step).
// Order-2 robust MLS: a weighted PLANE fit alone reads any smooth convexity as
// positive deviation (the plane cuts the dome's chord), washing real
// protrusions out. So: fit a weighted QUADRIC in the plane's local frame --
// smooth curvature (cusps, domes, grooves) is absorbed by the quadric -- and
// robustly DOWN-WEIGHT high-residual neighbours over two passes so a spike
// cannot drag the reference surface toward itself. What remains as deviation
// is exactly the stuff sticking out of (or dented into) the smooth base.
void mlsCompute(const ModeInput& in, std::vector<float>& dev,
                std::vector<Eigen::Vector3f>& proj, const ProgressFn& progress, float pEnd) {
    const auto& pts = in.points;
    dev.assign(pts.size(), 0.0f);
    proj = pts;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    float diag = buildGrid(pts, grid, mn, mx);
    const float h = std::max(diag * P(in, 0, 2.0f) * 0.01f, 1e-9f);  // kernel bandwidth

    const std::vector<Eigen::Vector3f> nrm =
        in.normals.size() == pts.size()
            ? in.normals
            : geometry::estimateNormals(pts, 16, [&](float f) { report(progress, f * 0.4f); });
    const float p0      = in.normals.size() == pts.size() ? 0.0f : 0.4f;
    const float inv2h2  = 1.0f / (2.0f * h * h);

    parallelFor(pts.size(), progress, p0, pEnd, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;   // squared distances
        thread_local std::vector<float> wgt;    // spatial (energy) weight
        thread_local std::vector<float> rob;    // robust multiplier
        thread_local std::vector<Eigen::Vector3f> loc;  // neighbour in (u,v,w) frame
        thread_local std::vector<float> res;    // per-neighbour quadric residual
        grid.pointsWithinRadius(pts, pts[i], h * 2.5f, nbr, dist);
        if (nbr.size() < 8) return;
        // A dense scan can put thousands of points in the kernel radius, but a
        // 6-coefficient quadric only needs a few hundred well-spread samples:
        // uniform-stride subsample. This is the difference between ~28s and a
        // few seconds on a 450k scan, at no visible quality cost.
        constexpr size_t kMaxNbr = 256;
        if (nbr.size() > kMaxNbr) {
            size_t stride = nbr.size() / kMaxNbr + 1;
            size_t w      = 0;
            for (size_t t = 0; t < nbr.size(); t += stride) {
                nbr[w]  = nbr[t];
                dist[w] = dist[t];
                ++w;
            }
            nbr.resize(w);
            dist.resize(w);
        }

        // Local frame from the energy-weighted plane (PCA).
        wgt.resize(nbr.size());
        float W = 0.0f;
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        for (size_t t = 0; t < nbr.size(); ++t) {
            float w = std::exp(-dist[t] * inv2h2);  // energy inflow from neighbour t
            wgt[t]  = w;
            W += w;
            c += w * pts[nbr[t]];
        }
        if (W < 1e-12f) return;
        c /= W;
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (size_t t = 0; t < nbr.size(); ++t) {
            Eigen::Vector3f d = pts[nbr[t]] - c;
            cov += wgt[t] * (d * d.transpose());
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
        if (es.info() != Eigen::Success) return;
        Eigen::Vector3f n = es.eigenvectors().col(0);
        if (n.dot(nrm[i]) < 0.0f) n = -n;                  // orient with the point's normal
        const Eigen::Vector3f t1 = es.eigenvectors().col(2);
        const Eigen::Vector3f t2 = es.eigenvectors().col(1);

        loc.resize(nbr.size());
        for (size_t t = 0; t < nbr.size(); ++t) {
            Eigen::Vector3f d = pts[nbr[t]] - c;
            loc[t] = {d.dot(t1) / h, d.dot(t2) / h, d.dot(n)};  // u,v scaled for conditioning
        }
        Eigen::Vector3f di = pts[i] - c;
        const float ui = di.dot(t1) / h, vi = di.dot(t2) / h, wi = di.dot(n);

        // Robust weighted quadric w(u,v): two fit passes, down-weighting
        // neighbours that sit far off the surface (a spike stops voting).
        rob.assign(nbr.size(), 1.0f);
        res.resize(nbr.size());
        Eigen::Matrix<float, 6, 1> coef = Eigen::Matrix<float, 6, 1>::Zero();
        bool  solved = false;
        float sigma  = h * 0.01f;  // local residual scale (refined below)
        for (int pass = 0; pass < 2; ++pass) {
            Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
            for (size_t t = 0; t < nbr.size(); ++t) {
                float u = loc[t].x(), v = loc[t].y();
                Eigen::Matrix<float, 6, 1> phi;
                phi << u * u, u * v, v * v, u, v, 1.0f;
                float w = wgt[t] * rob[t];
                A += w * (phi * phi.transpose());
                b += w * (phi * loc[t].z());
            }
            Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt(A);
            if (ldlt.info() != Eigen::Success) break;
            coef   = ldlt.solve(b);
            solved = true;
            if (pass == 1) break;
            // Residual scale (MAD) -> redescending weights for the refit.
            for (size_t t = 0; t < nbr.size(); ++t) {
                float u = loc[t].x(), v = loc[t].y();
                res[t] = std::fabs(loc[t].z() -
                                   (coef[0] * u * u + coef[1] * u * v + coef[2] * v * v +
                                    coef[3] * u + coef[4] * v + coef[5]));
            }
            thread_local std::vector<float> tmp;
            tmp = res;
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
            sigma        = std::max(1.4826f * tmp[tmp.size() / 2], h * 1e-4f);
            float inv2s2 = 1.0f / (2.0f * (3.0f * sigma) * (3.0f * sigma));
            for (size_t t = 0; t < nbr.size(); ++t)
                rob[t] = std::exp(-res[t] * res[t] * inv2s2);
        }

        float d;
        if (solved) {
            float wHat = coef[0] * ui * ui + coef[1] * ui * vi + coef[2] * vi * vi +
                         coef[3] * ui + coef[4] * vi + coef[5];
            d = wi - wHat;  // off the smooth quadric base, signed along the normal
        } else {
            d = wi;         // degenerate: fall back to the plane residual
        }
        dev[i]  = d;               // raw signed residual; normalized globally below
        proj[i] = pts[i] - n * d;  // projection always uses the raw distance
    });

    // Normalize to statistical SIGNIFICANCE with a GLOBAL robust scale (MAD of
    // the raw residuals). A per-neighbourhood sigma looked right on synthetic
    // data but real scans inflate it wherever the quadric fits poorly (steep
    // walls, interproximal gaps) -- exactly where spikes live -- washing them
    // out. One global scale keeps "red" meaning "sticks out far beyond the
    // scan's typical residual".
    {
        std::vector<float> mag(dev.size());
        for (size_t i = 0; i < dev.size(); ++i) mag[i] = std::fabs(dev[i]);
        std::nth_element(mag.begin(), mag.begin() + mag.size() / 2, mag.end());
        float gSigma = std::max(1.4826f * mag[mag.size() / 2], h * 1e-4f);
        for (auto& d : dev) d /= 3.0f * gSigma;
    }
    report(progress, pEnd);
}

void mlsDeviationColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                        debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    std::vector<float>           dev;  // significance: residual / (3 * local sigma)
    std::vector<Eigen::Vector3f> proj;
    mlsCompute(in, dev, proj, progress, 0.95f);
    if (dev.empty()) return;
    // Fixed significance range: +-1 = 3 sigma (noise stays green), saturation
    // at +-3 = 9 sigma (unmistakably sticking out => full red / full blue).
    colors.resize(dev.size());
    for (size_t i = 0; i < dev.size(); ++i)
        colors[i] = geometry::compareBandColor(dev[i], 3.0f, /*isSigned=*/true, /*bands=*/1);
    report(progress, 1.0f);
}

void mlsFit(const ModeInput& in, std::vector<Eigen::Vector3f>& fitted,
            const ProgressFn& progress) {
    std::vector<float> dev;
    mlsCompute(in, dev, fitted, progress, 1.0f);
}

// --- Filter: Protrusion detect (MLS AND PFOR) ---------------------------------
// "Find what sticks out": a point is a protrusion only when BOTH independent
// tests agree -- its MLS significance (how far ABOVE the energy-weighted
// quadric base, in global-noise units) exceeds the threshold AND PFOR's
// plane-distance test calls it an off-surface outlier. The intersection kills
// each test's false positives (MLS alone warms up rough-but-valid texture,
// PFOR alone fires on smooth curvature). Flagged points paint red, everything
// else keeps its original color; Fit projects ONLY the flagged points onto
// their MLS base surface (the rest of the cloud is untouched).
// Params: 0 = MLS Radius % (mlsCompute reads it), 1 = MLS significance
// threshold, 2 = PFOR K, 3 = PFOR Beta.
void protrusionFlags(const ModeInput& in, std::vector<uint8_t>& flag,
                     std::vector<Eigen::Vector3f>& proj, const ProgressFn& progress) {
    const auto& pts = in.points;
    flag.assign(pts.size(), 0);
    proj = pts;
    if (pts.size() < 8) return;
    std::vector<float> dev;
    mlsCompute(in, dev, proj, progress, 0.7f);
    const float sigThresh = P(in, 1, 1.0f);  // 1.0 = 3x the scan's residual MAD

    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);
    const int   k    = (int)std::lround(P(in, 2, 16.0f));
    const float beta = P(in, 3, 0.05f);
    std::vector<uint8_t> pfor(pts.size(), 0);
    parallelFor(pts.size(), progress, 0.7f, 1.0f, [&](size_t i) {
        if (dev[i] <= sigThresh) return;  // MLS says smooth: PFOR test not needed
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        if (!neighbourhoodPCA(pts, nbr, c, eval, evec)) return;
        Eigen::Vector3f n = evec.col(0);
        float planeDist = std::abs((pts[i] - c).dot(n));
        float extent    = std::sqrt(std::max(eval.y(), 1e-12f));
        pfor[i] = planeDist > beta * extent;  // PFOR's OUTLIER criterion
    });
    for (size_t i = 0; i < pts.size(); ++i)
        flag[i] = dev[i] > sigThresh && pfor[i];

    // Grow the seeds through the MLS-flagged region so WHOLE blobs flag, not
    // just their bases: a protruding cluster's interior defeats PFOR's plane
    // test (its neighbours are the cluster itself), so the strict intersection
    // only fires where cluster and base mix. Same hysteresis-growth trick as
    // the bump detector. Growth radius scales with the local point spacing.
    {
        float spacing = 0.0f;
        int   sn      = 0;
        std::vector<unsigned int> nbr;
        std::vector<float> dist;
        size_t step = pts.size() > 400 ? pts.size() / 400 : 1;
        for (size_t i = 0; i < pts.size(); i += step) {
            grid.kNearestNeighbors(pts, pts[i], 2, nbr, dist);
            if (dist.size() >= 2) { spacing += std::sqrt(dist[1]); ++sn; }
        }
        spacing = sn ? spacing / (float)sn : 0.0f;
        if (spacing > 0.0f) {
            float growR = spacing * 2.5f;
            std::vector<size_t> stack;
            for (size_t i = 0; i < pts.size(); ++i)
                if (flag[i]) stack.push_back(i);
            while (!stack.empty()) {
                size_t i = stack.back();
                stack.pop_back();
                grid.pointsWithinRadius(pts, pts[i], growR, nbr, dist);
                for (unsigned int j : nbr)
                    if (!flag[j] && dev[j] > sigThresh) {
                        flag[j] = 1;
                        stack.push_back(j);
                    }
            }
        }
    }
}

void protrusionColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                      debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    std::vector<uint8_t>         flag;
    std::vector<Eigen::Vector3f> proj;
    protrusionFlags(in, flag, proj, progress);
    // x < 0 keeps that point's original color; only the intersection turns red.
    colors.assign(in.points.size(), Eigen::Vector3f(-1.0f, -1.0f, -1.0f));
    for (size_t i = 0; i < flag.size(); ++i)
        if (flag[i]) colors[i] = Eigen::Vector3f(1.0f, 0.08f, 0.05f);
}

void protrusionFit(const ModeInput& in, std::vector<Eigen::Vector3f>& fitted,
                   const ProgressFn& progress) {
    std::vector<uint8_t> flag;
    std::vector<Eigen::Vector3f> proj;
    protrusionFlags(in, flag, proj, progress);
    fitted = in.points;
    for (size_t i = 0; i < flag.size(); ++i)
        if (flag[i]) fitted[i] = proj[i];
}

// --- Analyze: Surface curvature ----------------------------------------------
// SIGNED surface curvature (ported from Helium PointCloudCurvatureAnalysis,
// extended with a sign): magnitude = surface variation lambda0/(sum lambda)
// from the neighbourhood PCA; sign = the side of the local surface the
// neighbourhood centroid falls on along the ORIENTED normal. Convex (bumps,
// cusps, ridges) is positive -> warm colors; concave (pits, grooves, fissures)
// is negative -> cool colors; flat = green. Uses the source file's normals
// when present (ModeInput::normals), else estimates + orients heuristically.
// The diverging map is symmetric around zero (95th percentile of |curvature|).
void curvatureColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                     debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);
    const int k = (int)std::lround(P(in, 0, 16.0f));

    const std::vector<Eigen::Vector3f> nrm =
        in.normals.size() == pts.size()
            ? in.normals
            : geometry::estimateNormals(pts, k, [&](float f) { report(progress, f * 0.4f); });

    std::vector<float> curv(pts.size(), 0.0f);
    parallelFor(pts.size(), progress, in.normals.size() == pts.size() ? 0.0f : 0.4f, 1.0f,
                [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        if (!neighbourhoodPCA(pts, nbr, c, eval, evec)) return;
        float s = eval.x() + eval.y() + eval.z();
        float m = s > 1e-12f ? eval.x() / s : 0.0f;
        // Neighbour centroid BELOW the surface (against the outward normal) =>
        // the point sits on a bump => convex, positive. Centroid above => pit.
        curv[i] = (c - pts[i]).dot(nrm[i]) < 0.0f ? m : -m;
    });

    // Diverging colors, symmetric around 0 so the sign is legible: blue =
    // concave, green = flat, red = convex (95th pct of |curv| as the range).
    std::vector<float> mag(curv.size());
    for (size_t i = 0; i < curv.size(); ++i) mag[i] = std::fabs(curv[i]);
    size_t p95 = std::min(mag.size() - 1, (size_t)((double)mag.size() * 0.95));
    std::nth_element(mag.begin(), mag.begin() + p95, mag.end());
    float range = std::max(mag[p95], 1e-9f);
    colors.resize(curv.size());
    for (size_t i = 0; i < curv.size(); ++i)
        colors[i] = geometry::compareBandColor(curv[i], range, /*isSigned=*/true, /*bands=*/1);
}

// Curvature "Fit": flatten the high-curvature points -- any point whose
// curvature magnitude exceeds the threshold is projected onto its local PCA
// plane (spikes and pits both; the sign doesn't matter for the projection).
// The threshold is RELATIVE to the same 95th-percentile range the mode's
// colors saturate at, so what looks strongly red/blue is what moves: 1.0 =
// only fully saturated points, 0.5 (default) = anything past half range.
// An absolute surface-variation cutoff would shift meaning with K (a large
// neighbourhood dilutes the variation) -- relative stays intuitive at any K.
void curvatureFit(const ModeInput& in, std::vector<Eigen::Vector3f>& fitted,
                  const ProgressFn& progress) {
    const auto& pts = in.points;
    fitted = pts;
    if (pts.size() < 8) return;
    geometry::SparseGrid grid;
    Eigen::Vector3f mn, mx;
    buildGrid(pts, grid, mn, mx);
    const int   k   = (int)std::lround(P(in, 0, 16.0f));
    const float rel = P(in, 1, 0.5f);

    // Pass 1: curvature magnitude + the plane projection candidate per point.
    std::vector<float>           curv(pts.size(), 0.0f);
    std::vector<Eigen::Vector3f> proj = pts;
    parallelFor(pts.size(), progress, 0.0f, 0.95f, [&](size_t i) {
        thread_local std::vector<unsigned int> nbr;
        thread_local std::vector<float> dist;
        Eigen::Vector3f c, eval; Eigen::Matrix3f evec;
        grid.kNearestNeighbors(pts, pts[i], k + 1, nbr, dist);
        if (!neighbourhoodPCA(pts, nbr, c, eval, evec)) return;
        float s = eval.x() + eval.y() + eval.z();
        curv[i] = s > 1e-12f ? eval.x() / s : 0.0f;
        Eigen::Vector3f n = evec.col(0);
        proj[i] = pts[i] - n * (pts[i] - c).dot(n);
    });

    // Pass 2: threshold against the color range (95th percentile of magnitude).
    std::vector<float> mag = curv;
    size_t p95 = std::min(mag.size() - 1, (size_t)((double)mag.size() * 0.95));
    std::nth_element(mag.begin(), mag.begin() + p95, mag.end());
    const float thr = rel * std::max(mag[p95], 1e-9f);
    for (size_t i = 0; i < pts.size(); ++i)
        if (curv[i] > thr) fitted[i] = proj[i];
    report(progress, 1.0f);
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

// --- Analyze: Normal divergence ------------------------------------------------
// SIGNED divergence of the normal vector field, estimated per point as the
// average radial rate of change of the normal toward its neighbours:
//   div ~ mean_j [ (n_j - n_i) . (p_j - p_i) / |p_j - p_i|^2 ]
// Where normals fan OUT (a bump apex, spike tip, convex ridge) the divergence
// is positive -> warm; where they fan IN (pits, grooves, concave fillets) it is
// negative -> cool; parallel normals (flat / cylinder axis direction) ~ 0 ->
// green. Complements Curvature (magnitude-based surface variation) with a
// field-based view that highlights sources/sinks of the normal field. Sign
// consistency: a neighbour whose normal points against ours (orientation flip
// in the input) is negated before differencing.
void normalDivergenceColors(const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                            debug::DebugDraw& extras, const ProgressFn& progress) {
    (void)extras;
    const auto& pts = in.points;
    if (pts.size() < 8) return;
    const int k = (int)std::lround(P(in, 0, 16.0f));
    const std::vector<float> divg = normalDivergenceField(in, k, progress, 0.0f, 1.0f);

    // Diverging colors symmetric around 0 (95th pct of |div| as the range):
    // blue = converging normals (pits), green = parallel, red = diverging.
    const float range = fieldRange95(divg);
    colors.resize(divg.size());
    for (size_t i = 0; i < divg.size(); ++i)
        colors[i] = geometry::compareBandColor(divg[i], range, /*isSigned=*/true, /*bands=*/1);
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
void qforPoints(const ModeInput& in, std::vector<Eigen::Vector3f>& o, const ProgressFn& p) {
    pointsFromKeepColors(qforColors, in, o, p);
}
void qforDivGatePoints(const ModeInput& in, std::vector<Eigen::Vector3f>& o, const ProgressFn& p) {
    pointsFromKeepColors(qforDivColors, in, o, p);
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
    // Draw mode whose triangle result can be APPLIED as a real mesh entity
    // (dialog Fit button; the host spawns the cached triangles). Reconstruct.
    bool         meshApply = false;
};
const ModeParam kBumpParams[4] = {
    {"Fit Radius %", 1.0f, 8.0f, 3.0f, false},
    {"Hi Score", 2.0f, 12.0f, 5.0f, false},
    {"Lo Score", 1.0f, 8.0f, 3.0f, false},
    {"Max Blob %", 1.0f, 20.0f, 5.0f, false},
};
const ModeEntry kModes[] = {
    {"Reconstruct", runReconstruct, ModeCategory::Generate, ApplyKind::Draw, nullptr, 1,
     {{"Voxel %", 0.5f, 10.0f, 3.0f, false}}, nullptr, nullptr, /*meshApply=*/true},
    {"SDF Filter", runSdfFilter, ModeCategory::Generate, ApplyKind::Draw, nullptr, 1,
     {{"Blur Iters", 0.0f, 5.0f, 2.0f, true}}, sdfFitPoints, sdfFitPoints},
    {"Clustering", nullptr, ModeCategory::Analyze, ApplyKind::Recolor, clusteringColors, 1,
     {{"Radius %", 0.5f, 10.0f, 2.0f, false}}},
    {"Curvature", nullptr, ModeCategory::Analyze, ApplyKind::Recolor, curvatureColors, 2,
     {{"K Neighbors", 6.0f, 64.0f, 16.0f, true},
      {"Fit Thresh", 0.05f, 2.0f, 0.5f, false, 0.05f}},
     curvatureFit},
    {"Normal Deviation", nullptr, ModeCategory::Analyze, ApplyKind::Recolor,
     normalDeviationColors, 1, {{"K Neighbors", 6.0f, 64.0f, 16.0f, true}}},
    {"Normal Divergence", nullptr, ModeCategory::Analyze, ApplyKind::Recolor,
     normalDivergenceColors, 1, {{"K Neighbors", 6.0f, 64.0f, 16.0f, true}}},
    {"Density (KDE)", nullptr, ModeCategory::Analyze, ApplyKind::Recolor, kdeColors, 1,
     {{"Bandwidth %", 0.5f, 10.0f, 3.0f, false}}},
    {"Surface Dev (MLS)", nullptr, ModeCategory::Analyze, ApplyKind::Recolor,
     mlsDeviationColors, 1, {{"Radius %", 0.5f, 10.0f, 2.0f, false}}, mlsFit, mlsFit},
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
    {"Outlier: QFOR", nullptr, ModeCategory::Filter, ApplyKind::Recolor, qforColors, 2,
     {{"K Neighbors", 10.0f, 256.0f, 24.0f, true},
      {"Alpha", 1.0f, 10.0f, 3.0f, false, 0.25f}},
     qforFit, qforPoints},
    {"Outlier: QFOR (Div Gate)", nullptr, ModeCategory::Filter, ApplyKind::Recolor,
     qforDivColors, 4,
     {{"K Neighbors", 10.0f, 256.0f, 24.0f, true},
      {"Alpha", 1.0f, 10.0f, 3.0f, false, 0.25f},
      {"Div Thresh", 0.05f, 1.0f, 0.5f, false, 0.05f},
      {"Side +|-|Both", 0.0f, 2.0f, 2.0f, true}},
     qforDivFit, qforDivGatePoints},
    {"Morphology", nullptr, ModeCategory::Filter, ApplyKind::Recolor, morphologyColors, 2,
     {{"Voxel %", 1.0f, 10.0f, 4.0f, false}, {"Erode Iters", 1.0f, 4.0f, 2.0f, true}},
     nullptr, morphologyPoints},
    {"Protrusion (MLS+PFOR)", nullptr, ModeCategory::Filter, ApplyKind::Recolor,
     protrusionColors, 4,
     {{"Radius %", 0.5f, 10.0f, 2.0f, false},
      {"Sig Thresh", 0.3f, 5.0f, 1.0f, false, 0.1f},
      {"K Neighbors", 4.0f, 64.0f, 16.0f, true},
      {"Beta", 0.0f, 0.1f, 0.05f, false, 0.001f}},
     protrusionFit, protrusionFit},
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

bool modeAppliesMesh(int index) {
    if (index < 0 || index >= kModeCount) return false;
    return kModes[index].meshApply;
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
