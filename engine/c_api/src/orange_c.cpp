// Implementation of the C ABI wrapper (orange/c/orange.h).
//
// Rules for everything in this file:
//   * no exception may cross the boundary -- every entry point is wrapped in
//     guard(), which catches and records the message for orangeLastError();
//   * no C++ type appears in a signature; handles are heap structs cast to the
//     opaque pointer types the header declares;
//   * the caller never owns memory returned by pointer -- borrowed views live
//     as long as their handle.

#include "orange/c/orange.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "orange/core/kdtree.h"
#include "orange/core/mesh_generation.h"
#include "orange/core/normals.h"
#include "orange/core/point_ops.h"
#include "orange/core/primitives.h"
#include "orange/core/serialization.h"

#ifndef ORANGE_VERSION_MAJOR
#  define ORANGE_VERSION_MAJOR 0
#  define ORANGE_VERSION_MINOR 0
#  define ORANGE_VERSION_PATCH 0
#endif

namespace {

// --- error channel ---------------------------------------------------------
std::string& lastErrorSlot() {
    static thread_local std::string slot;
    return slot;
}

void setError(const char* msg) { lastErrorSlot() = msg ? msg : ""; }
void clearError() { lastErrorSlot().clear(); }

// Runs `fn`, turning any escaping exception into `onFail` + a recorded message.
template <typename Fn, typename R>
R guard(Fn&& fn, R onFail) {
    try {
        clearError();
        return fn();
    } catch (const std::exception& e) {
        setError(e.what());
    } catch (...) {
        setError("unknown exception");
    }
    return onFail;
}

// --- handle types ----------------------------------------------------------
struct Vec3ArrayImpl {
    std::vector<Eigen::Vector3f> v;
};

struct MeshImpl {
    // Deinterleaved from geometry::Triangle so the C side gets three plain
    // float arrays (9 floats per triangle) instead of a C++ struct layout.
    std::vector<float> positions, normals, colors;
    int32_t triangles = 0;
};

struct KdTreeImpl {
    std::vector<Eigen::Vector3f>  points;  // owned copy: KDTree only references
    orange::geometry::KDTree      tree;
};

Vec3ArrayImpl* asArray(OrangeVec3Array h) { return reinterpret_cast<Vec3ArrayImpl*>(h); }
MeshImpl*      asMesh(OrangeMesh h)       { return reinterpret_cast<MeshImpl*>(h); }
KdTreeImpl*    asTree(OrangeKdTree h)     { return reinterpret_cast<KdTreeImpl*>(h); }

OrangeVec3Array wrap(std::vector<Eigen::Vector3f>&& v) {
    auto* impl = new Vec3ArrayImpl{std::move(v)};
    return reinterpret_cast<OrangeVec3Array>(impl);
}

// Empty input is a failure, not a silent empty handle -- callers otherwise chain
// operations on nothing and only notice at the end.
const std::vector<Eigen::Vector3f>* pointsOf(OrangeVec3Array h, const char* what) {
    if (!h) { setError(what); return nullptr; }
    return &asArray(h)->v;
}

OrangeMesh wrapMesh(const std::vector<orange::geometry::Triangle>& tris) {
    auto impl = std::unique_ptr<MeshImpl>(new MeshImpl());
    impl->triangles = static_cast<int32_t>(tris.size());
    impl->positions.resize(tris.size() * 9);
    impl->normals.resize(tris.size() * 9);
    impl->colors.resize(tris.size() * 9);
    for (size_t t = 0; t < tris.size(); ++t) {
        for (int i = 0; i < 3; ++i) {
            const size_t o = t * 9 + static_cast<size_t>(i) * 3;
            std::memcpy(&impl->positions[o], tris[t].v[i].data(), sizeof(float) * 3);
            std::memcpy(&impl->normals[o],   tris[t].n[i].data(), sizeof(float) * 3);
            std::memcpy(&impl->colors[o],    tris[t].c[i].data(), sizeof(float) * 3);
        }
    }
    return reinterpret_cast<OrangeMesh>(impl.release());
}

Eigen::Vector3f colorOr(const float* color3, float fallback = 1.0f) {
    if (!color3) return Eigen::Vector3f(fallback, fallback, fallback);
    return Eigen::Vector3f(color3[0], color3[1], color3[2]);
}

// Shared by every primitive entry point.
OrangeMesh buildGuard(const std::function<std::vector<orange::geometry::Triangle>()>& build) {
    return guard([&]() -> OrangeMesh {
        const std::vector<orange::geometry::Triangle> tris = build();
        if (tris.empty()) { setError("primitive builder produced no triangles"); return nullptr; }
        return wrapMesh(tris);
    }, static_cast<OrangeMesh>(nullptr));
}

std::string lowerExtension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? std::string() : path.substr(dot + 1);
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext;
}

}  // namespace

// ---------------------------------------------------------------------------
// library info
// ---------------------------------------------------------------------------
extern "C" uint32_t ORANGE_C_CALL orangeVersion(void) {
    return (static_cast<uint32_t>(ORANGE_VERSION_MAJOR) << 16) |
           (static_cast<uint32_t>(ORANGE_VERSION_MINOR) << 8) |
            static_cast<uint32_t>(ORANGE_VERSION_PATCH);
}

extern "C" uint32_t ORANGE_C_CALL orangeCApiVersion(void) { return ORANGE_C_API_VERSION; }

extern "C" const char* ORANGE_C_CALL orangeLastError(void) { return lastErrorSlot().c_str(); }

// ---------------------------------------------------------------------------
// float3 arrays
// ---------------------------------------------------------------------------
extern "C" OrangeVec3Array ORANGE_C_CALL orangeVec3ArrayCreate(const float* xyz, int32_t count) {
    return guard([&]() -> OrangeVec3Array {
        if (count < 0 || (count > 0 && !xyz)) { setError("invalid xyz/count"); return nullptr; }
        std::vector<Eigen::Vector3f> v(static_cast<size_t>(count));
        for (int32_t i = 0; i < count; ++i)
            v[static_cast<size_t>(i)] = Eigen::Vector3f(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]);
        return wrap(std::move(v));
    }, static_cast<OrangeVec3Array>(nullptr));
}

extern "C" int32_t ORANGE_C_CALL orangeVec3ArrayCount(OrangeVec3Array arr) {
    if (!arr) return ORANGE_ERROR_INVALID;
    return static_cast<int32_t>(asArray(arr)->v.size());
}

extern "C" const float* ORANGE_C_CALL orangeVec3ArrayData(OrangeVec3Array arr) {
    if (!arr || asArray(arr)->v.empty()) return nullptr;
    // Eigen::Vector3f is 3 tightly packed floats (the build sets
    // EIGEN_MAX_ALIGN_BYTES=0), so the vector IS an xyz float array.
    static_assert(sizeof(Eigen::Vector3f) == sizeof(float) * 3, "Vector3f must be packed");
    return asArray(arr)->v.front().data();
}

extern "C" int32_t ORANGE_C_CALL orangeVec3ArrayCopy(OrangeVec3Array arr, float* dst,
                                                     int32_t dstFloats) {
    if (!arr || !dst) return ORANGE_ERROR_INVALID;
    const int32_t need = static_cast<int32_t>(asArray(arr)->v.size()) * 3;
    if (dstFloats < need) return ORANGE_ERROR_TOO_SMALL;
    if (need > 0) std::memcpy(dst, asArray(arr)->v.front().data(), sizeof(float) * need);
    return need;
}

extern "C" void ORANGE_C_CALL orangeVec3ArrayDestroy(OrangeVec3Array arr) {
    delete asArray(arr);
}

// ---------------------------------------------------------------------------
// point-cloud IO
// ---------------------------------------------------------------------------
extern "C" OrangeVec3Array ORANGE_C_CALL orangePointsLoadFile(const char* path) {
    return guard([&]() -> OrangeVec3Array {
        if (!path) { setError("path is NULL"); return nullptr; }
        const std::string ext = lowerExtension(path);
        std::unique_ptr<orange::io::HSerializable> fmt;
        if      (ext == "ply") fmt.reset(new orange::io::PLYFormat());
        else if (ext == "xyz") fmt.reset(new orange::io::XYZFormat());
        else if (ext == "obj") fmt.reset(new orange::io::OBJFormat());
        else if (ext == "off") fmt.reset(new orange::io::OFFFormat());
        else { setError("unsupported extension (use ply/xyz/obj/off)"); return nullptr; }

        if (!fmt->Deserialize(path)) { setError("failed to read file"); return nullptr; }
        std::vector<Eigen::Vector3f> pts = fmt->GetPoints();
        if (pts.empty()) { setError("file contains no points"); return nullptr; }
        return wrap(std::move(pts));
    }, static_cast<OrangeVec3Array>(nullptr));
}

// ---------------------------------------------------------------------------
// point-cloud operators
// ---------------------------------------------------------------------------
extern "C" OrangeVec3Array ORANGE_C_CALL orangeEstimateNormals(OrangeVec3Array points, int32_t k) {
    return guard([&]() -> OrangeVec3Array {
        const std::vector<Eigen::Vector3f>* pts = pointsOf(points, "points handle is NULL");
        if (!pts) return nullptr;
        if (pts->empty()) { setError("points are empty"); return nullptr; }
        return wrap(orange::geometry::estimateNormals(*pts, k > 0 ? k : 16));
    }, static_cast<OrangeVec3Array>(nullptr));
}

extern "C" OrangeVec3Array ORANGE_C_CALL orangeSmoothPoints(OrangeVec3Array points,
                                                            int32_t iterations, float lambda,
                                                            int32_t edgePreserving, int32_t k) {
    return guard([&]() -> OrangeVec3Array {
        const std::vector<Eigen::Vector3f>* pts = pointsOf(points, "points handle is NULL");
        if (!pts) return nullptr;
        if (pts->empty()) { setError("points are empty"); return nullptr; }
        return wrap(orange::geometry::smoothPoints(*pts, iterations > 0 ? iterations : 5, lambda,
                                                   edgePreserving != 0, k > 0 ? k : 12));
    }, static_cast<OrangeVec3Array>(nullptr));
}

extern "C" OrangeStatus ORANGE_C_CALL orangeIcpAlign(OrangeVec3Array src, OrangeVec3Array dst,
                                                     int32_t maxIterations, float* outMatrix16,
                                                     float* outRmse, int32_t* outIters) {
    return guard([&]() -> OrangeStatus {
        const std::vector<Eigen::Vector3f>* s = pointsOf(src, "src handle is NULL");
        const std::vector<Eigen::Vector3f>* d = pointsOf(dst, "dst handle is NULL");
        if (!s || !d || !outMatrix16) { setError("invalid argument"); return ORANGE_ERROR_INVALID; }
        if (s->empty() || d->empty()) { setError("point set is empty"); return ORANGE_ERROR_EMPTY; }

        float rmse = 0.0f;
        int   iters = 0;
        const Eigen::Matrix4f m = orange::geometry::icpAlign(
            *s, *d, maxIterations > 0 ? maxIterations : 20, rmse, iters);
        // Eigen is column-major, which is the layout the header promises.
        std::memcpy(outMatrix16, m.data(), sizeof(float) * 16);
        if (outRmse)  *outRmse = rmse;
        if (outIters) *outIters = iters;
        return ORANGE_OK;
    }, ORANGE_ERROR);
}

// ---------------------------------------------------------------------------
// surface reconstruction
// ---------------------------------------------------------------------------
extern "C" OrangeMesh ORANGE_C_CALL orangePointsToMesh(OrangeVec3Array points,
                                                       OrangeVec3Array normals,
                                                       OrangeVec3Array colors, float voxelSize) {
    return guard([&]() -> OrangeMesh {
        const std::vector<Eigen::Vector3f>* pts = pointsOf(points, "points handle is NULL");
        if (!pts) return nullptr;
        if (pts->empty()) { setError("points are empty"); return nullptr; }

        static const std::vector<Eigen::Vector3f> kNone;
        const std::vector<Eigen::Vector3f>& nrm = normals ? asArray(normals)->v : kNone;
        const std::vector<Eigen::Vector3f>& col = colors  ? asArray(colors)->v  : kNone;
        const float vs = voxelSize > 0.0f
                             ? voxelSize
                             : orange::geometry::VoxelConfig::kDefaultVoxelSize;

        const std::vector<orange::geometry::Triangle> tris =
            orange::geometry::pointsToMesh(*pts, nrm, col, vs);
        if (tris.empty()) { setError("reconstruction produced no triangles"); return nullptr; }
        return wrapMesh(tris);
    }, static_cast<OrangeMesh>(nullptr));
}

extern "C" OrangeMesh ORANGE_C_CALL orangeReconstructFromFile(const char* path, float voxelSize) {
    return guard([&]() -> OrangeMesh {
        if (!path) { setError("path is NULL"); return nullptr; }
        const float vs = voxelSize > 0.0f
                             ? voxelSize
                             : orange::geometry::VoxelConfig::kDefaultVoxelSize;
        const std::vector<orange::geometry::Triangle> tris =
            orange::io::reconstructMeshFromFile(path, vs);
        if (tris.empty()) { setError("no mesh produced (unreadable file or empty cloud)"); return nullptr; }
        return wrapMesh(tris);
    }, static_cast<OrangeMesh>(nullptr));
}

// ---------------------------------------------------------------------------
// parametric primitives
// ---------------------------------------------------------------------------
extern "C" OrangeMesh ORANGE_C_CALL orangeBuildPlane(float size, const float* color3) {
    return buildGuard([&] { return orange::geometry::buildPlane(size, colorOr(color3)); });
}

extern "C" OrangeMesh ORANGE_C_CALL orangeBuildBox(const float* size3, const float* color3) {
    return buildGuard([&] {
        const Eigen::Vector3f s = size3 ? Eigen::Vector3f(size3[0], size3[1], size3[2])
                                        : Eigen::Vector3f(1.0f, 1.0f, 1.0f);
        return orange::geometry::buildBox(s, colorOr(color3));
    });
}

extern "C" OrangeMesh ORANGE_C_CALL orangeBuildSphere(float radius, int32_t segments,
                                                       const float* color3) {
    return buildGuard([&] {
        return orange::geometry::buildSphere(radius, segments > 2 ? segments : 32, colorOr(color3));
    });
}

extern "C" OrangeMesh ORANGE_C_CALL orangeBuildCylinder(float radius, float height,
                                                         int32_t segments, const float* color3) {
    return buildGuard([&] {
        return orange::geometry::buildCylinder(radius, height, segments > 2 ? segments : 32,
                                               colorOr(color3));
    });
}

extern "C" OrangeMesh ORANGE_C_CALL orangeBuildCone(float radius, float height, int32_t segments,
                                                     const float* color3) {
    return buildGuard([&] {
        return orange::geometry::buildCone(radius, height, segments > 2 ? segments : 32,
                                           colorOr(color3));
    });
}

extern "C" OrangeMesh ORANGE_C_CALL orangeBuildTorus(float majorRadius, float minorRadius,
                                                      int32_t segMajor, int32_t segMinor,
                                                      const float* color3) {
    return buildGuard([&] {
        return orange::geometry::buildTorus(majorRadius, minorRadius, segMajor > 2 ? segMajor : 32,
                                            segMinor > 2 ? segMinor : 16, colorOr(color3));
    });
}

extern "C" OrangeMesh ORANGE_C_CALL orangeBuildDisk(float radius, int32_t segments,
                                                     const float* color3) {
    return buildGuard([&] {
        return orange::geometry::buildDisk(radius, segments > 2 ? segments : 32, colorOr(color3));
    });
}

extern "C" OrangeMesh ORANGE_C_CALL orangeBuildCapsule(float radius, float cylinderHeight,
                                                        int32_t segments, const float* color3) {
    return buildGuard([&] {
        return orange::geometry::buildCapsule(radius, cylinderHeight, segments > 2 ? segments : 32,
                                              colorOr(color3));
    });
}

extern "C" OrangeMesh ORANGE_C_CALL orangeBuildArrow(float length, float radius, int32_t segments,
                                                      const float* color3) {
    return buildGuard([&] {
        return orange::geometry::buildArrow(length, radius, segments > 2 ? segments : 16,
                                            colorOr(color3));
    });
}

// ---------------------------------------------------------------------------
// mesh access
// ---------------------------------------------------------------------------
extern "C" int32_t ORANGE_C_CALL orangeMeshTriangleCount(OrangeMesh mesh) {
    if (!mesh) return ORANGE_ERROR_INVALID;
    return asMesh(mesh)->triangles;
}

extern "C" const float* ORANGE_C_CALL orangeMeshPositions(OrangeMesh mesh) {
    if (!mesh || asMesh(mesh)->positions.empty()) return nullptr;
    return asMesh(mesh)->positions.data();
}

extern "C" const float* ORANGE_C_CALL orangeMeshNormals(OrangeMesh mesh) {
    if (!mesh || asMesh(mesh)->normals.empty()) return nullptr;
    return asMesh(mesh)->normals.data();
}

extern "C" const float* ORANGE_C_CALL orangeMeshColors(OrangeMesh mesh) {
    if (!mesh || asMesh(mesh)->colors.empty()) return nullptr;
    return asMesh(mesh)->colors.data();
}

extern "C" OrangeStatus ORANGE_C_CALL orangeMeshBounds(OrangeMesh mesh, float* outMin3,
                                                       float* outMax3) {
    if (!mesh || !outMin3 || !outMax3) return ORANGE_ERROR_INVALID;
    const std::vector<float>& p = asMesh(mesh)->positions;
    if (p.empty()) return ORANGE_ERROR_EMPTY;

    Eigen::Vector3f lo(p[0], p[1], p[2]), hi = lo;
    for (size_t i = 3; i + 2 < p.size(); i += 3) {
        lo = lo.cwiseMin(Eigen::Vector3f(p[i], p[i + 1], p[i + 2]));
        hi = hi.cwiseMax(Eigen::Vector3f(p[i], p[i + 1], p[i + 2]));
    }
    std::memcpy(outMin3, lo.data(), sizeof(float) * 3);
    std::memcpy(outMax3, hi.data(), sizeof(float) * 3);
    return ORANGE_OK;
}

extern "C" void ORANGE_C_CALL orangeMeshDestroy(OrangeMesh mesh) { delete asMesh(mesh); }

// ---------------------------------------------------------------------------
// KD-tree
// ---------------------------------------------------------------------------
extern "C" OrangeKdTree ORANGE_C_CALL orangeKdTreeCreate(OrangeVec3Array points) {
    return guard([&]() -> OrangeKdTree {
        const std::vector<Eigen::Vector3f>* pts = pointsOf(points, "points handle is NULL");
        if (!pts) return nullptr;
        if (pts->empty()) { setError("points are empty"); return nullptr; }

        auto impl = std::unique_ptr<KdTreeImpl>(new KdTreeImpl());
        impl->points = *pts;              // own the storage: KDTree only references it
        impl->tree.build(impl->points);
        return reinterpret_cast<OrangeKdTree>(impl.release());
    }, static_cast<OrangeKdTree>(nullptr));
}

extern "C" int32_t ORANGE_C_CALL orangeKdTreeNearest(OrangeKdTree tree, const float* query3,
                                                     float* outDistance) {
    return guard([&]() -> int32_t {
        if (!tree || !query3) return ORANGE_ERROR_INVALID;
        float d2 = 0.0f;
        const int idx = asTree(tree)->tree.nearest(
            Eigen::Vector3f(query3[0], query3[1], query3[2]), &d2);
        if (idx < 0) return ORANGE_ERROR_EMPTY;
        if (outDistance) *outDistance = std::sqrt(d2);
        return static_cast<int32_t>(idx);
    }, static_cast<int32_t>(ORANGE_ERROR));
}

extern "C" int32_t ORANGE_C_CALL orangeKdTreeKNearest(OrangeKdTree tree, const float* query3,
                                                      int32_t k, int32_t* outIndices,
                                                      int32_t capacity) {
    return guard([&]() -> int32_t {
        if (!tree || !query3 || k <= 0 || (capacity > 0 && !outIndices))
            return ORANGE_ERROR_INVALID;
        std::vector<int> hits;
        asTree(tree)->tree.kNearest(Eigen::Vector3f(query3[0], query3[1], query3[2]),
                                    static_cast<int>(k), hits);
        const int32_t n = std::min<int32_t>(static_cast<int32_t>(hits.size()), capacity);
        for (int32_t i = 0; i < n; ++i) outIndices[i] = static_cast<int32_t>(hits[i]);
        return n;
    }, static_cast<int32_t>(ORANGE_ERROR));
}

extern "C" int32_t ORANGE_C_CALL orangeKdTreeRadius(OrangeKdTree tree, const float* query3,
                                                    float radius, int32_t* outIndices,
                                                    int32_t capacity) {
    return guard([&]() -> int32_t {
        if (!tree || !query3 || radius < 0.0f || (capacity > 0 && !outIndices))
            return ORANGE_ERROR_INVALID;
        std::vector<int> hits;
        asTree(tree)->tree.radiusQuery(Eigen::Vector3f(query3[0], query3[1], query3[2]), radius,
                                       hits);
        // capacity 0 => count-only probe (outIndices may be NULL).
        if (capacity <= 0) return static_cast<int32_t>(hits.size());
        const int32_t n = std::min<int32_t>(static_cast<int32_t>(hits.size()), capacity);
        for (int32_t i = 0; i < n; ++i) outIndices[i] = static_cast<int32_t>(hits[i]);
        return n;
    }, static_cast<int32_t>(ORANGE_ERROR));
}

extern "C" void ORANGE_C_CALL orangeKdTreeDestroy(OrangeKdTree tree) { delete asTree(tree); }
