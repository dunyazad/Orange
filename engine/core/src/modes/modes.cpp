#include "orange/core/modes.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "orange/core/color.h"
#include "orange/core/mesh_generation.h"
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

// --- Mode 0: Euclidean clustering (SparseGrid radius + union-find) -----------
void runClustering(const ModeInput& in, debug::DebugDraw& out) {
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
    for (size_t i = 0; i < pts.size(); ++i) {
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

void runMorphology(const ModeInput& in, debug::DebugDraw& out) {
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
    for (const auto& p : pts) {
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
void runSdfFilter(const ModeInput& in, debug::DebugDraw& out) {
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
    for (const auto& p : pts) {
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
    for (int z = 1; z < dim.z() - 1; ++z)
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

// --- Mode 3: Surface reconstruction (TSDF + dual contouring) -----------------
void runReconstruct(const ModeInput& in, debug::DebugDraw& out) {
    const auto& pts = in.points;
    if (pts.empty()) return;

    Eigen::Vector3f mn, mx;
    float diag = boundsExtent(pts, mn, mx).norm();
    float voxelSize = diag * 0.03f;

    std::vector<Eigen::Vector3f> noColors;
    auto tris = geometry::pointsToMesh(pts, in.normals, noColors, voxelSize);
    for (const auto& t : tris)
        out.addTriangle(t.v[0], t.v[1], t.v[2], t.c[0]);
}

struct ModeEntry {
    const char* name;
    void (*fn)(const ModeInput&, debug::DebugDraw&);
};
const ModeEntry kModes[] = {
    {"Clustering", runClustering},
    {"Morphology", runMorphology},
    {"SDF Filter", runSdfFilter},
    {"Reconstruct", runReconstruct},
};
constexpr int kModeCount = (int)(sizeof(kModes) / sizeof(kModes[0]));

} // namespace

int modeCount() { return kModeCount; }

const char* modeName(int index) {
    if (index < 0 || index >= kModeCount) return "?";
    return kModes[index].name;
}

void runMode(int index, const ModeInput& in, debug::DebugDraw& out) {
    if (index < 0 || index >= kModeCount) return;
    kModes[index].fn(in, out);
}

} // namespace orange::modes
