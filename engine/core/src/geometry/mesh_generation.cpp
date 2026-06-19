#include "orange/core/mesh_generation.h"

#include <cfloat>
#include <cmath>
#include <execution>
#include <mutex>
#include <unordered_map>

// Ported from Elements/Helium PointCloudGenerateMesh.cpp (Process), minus the
// HeliumCore/VisualDebugging plumbing and the hole-edge debug pass. Two parallel
// passes over the allocated blocks: compute one surface vertex per crossing cell,
// then emit quads between adjacent crossing voxels along +X/+Y/+Z.

namespace orange::geometry {

namespace {
constexpr int kAxis = VoxelConfig::kVoxelsPerBlockAxis;

struct GridKey {
    int x = 0, y = 0, z = 0;
    bool operator==(const GridKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct GridKeyHash {
    size_t operator()(const GridKey& k) const {
        return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^
               (std::hash<int>()(k.z) << 1);
    }
};
struct SNVertex {
    Eigen::Vector3f pos;
    Eigen::Vector3f normal;
    Eigen::Vector3f color;
};
}  // namespace

std::vector<Triangle> generateMesh(SparseDataBlock& sdb, float isoLevel) {
    std::vector<Triangle> triangles;

    std::vector<DataBlock*> blocks;
    blocks.reserve(sdb.dataBlocks.size());
    for (auto& pair : sdb.dataBlocks) blocks.push_back(pair.second.get());

    size_t estTris = blocks.size() * 96;
    triangles.reserve(estTris);

    std::unordered_map<GridKey, SNVertex, GridKeyHash> snVertices;
    snVertices.reserve(size_t(estTris * 0.6f));
    snVertices.max_load_factor(0.7f);

    std::mutex vertexMutex;
    std::mutex triMutex;

    const Eigen::Vector3i corners[8] = {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1},
                                        {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}};
    const int edgePairs[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                                  {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

    // Pass 1: one surface vertex per crossing cell (averaged edge intersections).
    std::for_each(std::execution::par, blocks.begin(), blocks.end(), [&](DataBlock* block) {
        std::vector<std::pair<GridKey, SNVertex>> localVerts;
        localVerts.reserve(64);

        Eigen::Vector3f diff = block->blockMin - sdb.gridOrigin;
        int startGx = (int)(diff.x() / sdb.voxelSize + 0.5f);
        int startGy = (int)(diff.y() / sdb.voxelSize + 0.5f);
        int startGz = (int)(diff.z() / sdb.voxelSize + 0.5f);

        for (int z = 0; z < kAxis; ++z)
            for (int y = 0; y < kAxis; ++y)
                for (int x = 0; x < kAxis; ++x) {
                    int gx = startGx + x, gy = startGy + y, gz = startGz + z;

                    float dists[8];
                    Eigen::Vector3f colors[8], normals[8];
                    int insideCount = 0;
                    bool allValid = true;

                    for (int i = 0; i < 8; ++i) {
                        const auto* v = sdb.voxelByIndex(gx + corners[i].x(), gy + corners[i].y(),
                                                         gz + corners[i].z());
                        if (!v || !v->valid) {
                            allValid = false;
                            break;
                        }
                        dists[i] = v->signedDistance;
                        colors[i] = v->color;
                        normals[i] = v->normal;
                        if (dists[i] < isoLevel) insideCount++;
                    }

                    if (!allValid || insideCount == 0 || insideCount == 8) continue;

                    Eigen::Vector3f avgPos(0, 0, 0), avgColor(0, 0, 0), avgNormal(0, 0, 0);
                    int intersections = 0;
                    for (int e = 0; e < 12; ++e) {
                        int i1 = edgePairs[e][0], i2 = edgePairs[e][1];
                        if ((dists[i1] < isoLevel) != (dists[i2] < isoLevel)) {
                            float t = (isoLevel - dists[i1]) / (dists[i2] - dists[i1]);
                            Eigen::Vector3f p1 =
                                sdb.gridOrigin + Eigen::Vector3f((float)gx + corners[i1].x(),
                                                                 (float)gy + corners[i1].y(),
                                                                 (float)gz + corners[i1].z()) *
                                                     sdb.voxelSize;
                            Eigen::Vector3f p2 =
                                sdb.gridOrigin + Eigen::Vector3f((float)gx + corners[i2].x(),
                                                                 (float)gy + corners[i2].y(),
                                                                 (float)gz + corners[i2].z()) *
                                                     sdb.voxelSize;
                            avgPos += p1 * (1.0f - t) + p2 * t;
                            avgColor += colors[i1] * (1.0f - t) + colors[i2] * t;
                            avgNormal += normals[i1] * (1.0f - t) + normals[i2] * t;
                            intersections++;
                        }
                    }

                    if (intersections > 0) {
                        SNVertex v;
                        v.pos = avgPos / (float)intersections;
                        v.color = avgColor / (float)intersections;
                        v.normal = avgNormal.normalized();
                        localVerts.push_back({{gx, gy, gz}, v});
                    }
                }

        if (!localVerts.empty()) {
            std::lock_guard<std::mutex> lock(vertexMutex);
            for (const auto& kv : localVerts) snVertices[kv.first] = kv.second;
        }
    });

    // Pass 2: stitch quads between adjacent crossing voxels (faces of the dual).
    std::for_each(std::execution::par, blocks.begin(), blocks.end(), [&](DataBlock* block) {
        std::vector<Triangle> localTris;
        localTris.reserve(128);

        Eigen::Vector3f diff = block->blockMin - sdb.gridOrigin;
        int startGx = (int)(diff.x() / sdb.voxelSize + 0.5f);
        int startGy = (int)(diff.y() / sdb.voxelSize + 0.5f);
        int startGz = (int)(diff.z() / sdb.voxelSize + 0.5f);

        auto addQuad = [&](const GridKey& k1, const GridKey& k2, const GridKey& k3,
                           const GridKey& k4, bool flip) {
            if (!(snVertices.count(k1) && snVertices.count(k2) && snVertices.count(k3) &&
                  snVertices.count(k4)))
                return;
            const auto& v0 = flip ? snVertices[k4] : snVertices[k1];
            const auto& v1 = flip ? snVertices[k3] : snVertices[k2];
            const auto& v2 = flip ? snVertices[k2] : snVertices[k3];
            const auto& v3 = flip ? snVertices[k1] : snVertices[k4];

            Triangle t1, t2;
            t1.v[0] = v0.pos; t1.v[1] = v1.pos; t1.v[2] = v2.pos;
            t1.c[0] = v0.color; t1.c[1] = v1.color; t1.c[2] = v2.color;
            t1.n[0] = v0.normal; t1.n[1] = v1.normal; t1.n[2] = v2.normal;
            t2.v[0] = v0.pos; t2.v[1] = v2.pos; t2.v[2] = v3.pos;
            t2.c[0] = v0.color; t2.c[1] = v2.color; t2.c[2] = v3.color;
            t2.n[0] = v0.normal; t2.n[1] = v2.normal; t2.n[2] = v3.normal;
            localTris.push_back(t1);
            localTris.push_back(t2);
        };

        for (int z = 0; z < kAxis; ++z)
            for (int y = 0; y < kAxis; ++y)
                for (int x = 0; x < kAxis; ++x) {
                    int gx = startGx + x, gy = startGy + y, gz = startGz + z;

                    const auto* vCurr = sdb.voxelByIndex(gx, gy, gz);
                    if (!vCurr || !vCurr->valid) continue;
                    bool bCurr = vCurr->signedDistance < isoLevel;

                    const auto* vX = sdb.voxelByIndex(gx + 1, gy, gz);
                    if (vX && vX->valid && (bCurr != (vX->signedDistance < isoLevel)))
                        addQuad({gx, gy - 1, gz - 1}, {gx, gy, gz - 1}, {gx, gy, gz},
                                {gx, gy - 1, gz}, !bCurr);

                    const auto* vY = sdb.voxelByIndex(gx, gy + 1, gz);
                    if (vY && vY->valid && (bCurr != (vY->signedDistance < isoLevel)))
                        addQuad({gx - 1, gy, gz - 1}, {gx, gy, gz - 1}, {gx, gy, gz},
                                {gx - 1, gy, gz}, bCurr);

                    const auto* vZ = sdb.voxelByIndex(gx, gy, gz + 1);
                    if (vZ && vZ->valid && (bCurr != (vZ->signedDistance < isoLevel)))
                        addQuad({gx - 1, gy - 1, gz}, {gx, gy - 1, gz}, {gx, gy, gz},
                                {gx - 1, gy, gz}, !bCurr);
                }

        if (!localTris.empty()) {
            std::lock_guard<std::mutex> lock(triMutex);
            triangles.insert(triangles.end(), localTris.begin(), localTris.end());
        }
    });

    return triangles;
}

std::vector<Triangle> pointsToMesh(const std::vector<Eigen::Vector3f>& points,
                                   const std::vector<Eigen::Vector3f>& normals,
                                   const std::vector<Eigen::Vector3f>& colors, float voxelSize) {
    if (points.empty()) return {};

    Eigen::Vector3f aabbMin = Eigen::Vector3f::Constant(FLT_MAX);
    for (const auto& p : points) aabbMin = aabbMin.cwiseMin(p);

    SparseDataBlock sdb;
    sdb.voxelSize = voxelSize;
    sdb.fromPointsData(points, normals, colors, aabbMin);
    return generateMesh(sdb, 0.0f);
}

} // namespace orange::geometry
