/* ===========================================================================
 * orange.h -- the stable C ABI over Orange's CPU geometry/IO toolkit.
 *
 * Why this exists: orange::core is a C++ library whose interface leaks STL and
 * Eigen types, so a consumer must be built with the *same* compiler and runtime.
 * This wrapper exposes the same functionality through plain C -- opaque handles,
 * primitive types, cdecl -- so orange_c.dll can be consumed from any toolchain
 * (older/newer MSVC, MinGW, clang) and from any language with an FFI (C#, Python
 * ctypes, Rust, Go, ...). It is the same boundary idea as the render plugins.
 *
 * Conventions
 *   - Every function is cdecl and never throws; failures return NULL / a
 *     negative OrangeStatus, and orangeLastError() has a message (thread-local).
 *   - Handles are opaque pointers; each has a matching *Destroy. Destroying NULL
 *     is a no-op. The library owns all memory behind a handle -- never free() it.
 *   - Positions/normals/colors are tightly packed float triples (xyz, xyz, ...).
 *     Mesh arrays are 9 floats per triangle (3 vertices x xyz).
 *   - Colors are 0..1 RGB.
 *   - Handles are not thread-safe individually; distinct handles are independent.
 *
 * Versioning: ORANGE_C_API_VERSION is bumped when this header changes in a way
 * that is not backward compatible. Check it at load time against
 * orangeCApiVersion().
 * ===========================================================================*/
#ifndef ORANGE_C_ORANGE_H
#define ORANGE_C_ORANGE_H

#include <stdint.h>

#define ORANGE_C_API_VERSION 1

#if defined(_WIN32)
#  if defined(ORANGE_C_BUILD)
#    define ORANGE_C_API __declspec(dllexport)
#  elif defined(ORANGE_C_STATIC)
#    define ORANGE_C_API
#  else
#    define ORANGE_C_API __declspec(dllimport)
#  endif
#  define ORANGE_C_CALL __cdecl
#else
#  if defined(ORANGE_C_BUILD)
#    define ORANGE_C_API __attribute__((visibility("default")))
#  else
#    define ORANGE_C_API
#  endif
#  define ORANGE_C_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- status codes ---------------------------------------------------------*/
typedef enum OrangeStatus {
    ORANGE_OK               =  0,
    ORANGE_ERROR            = -1,  /* unspecified failure; see orangeLastError */
    ORANGE_ERROR_INVALID    = -2,  /* NULL handle or out-of-range argument */
    ORANGE_ERROR_IO         = -3,  /* file missing / unreadable / unsupported */
    ORANGE_ERROR_EMPTY      = -4,  /* the operation produced no data */
    ORANGE_ERROR_TOO_SMALL  = -5   /* caller buffer too small (see return docs) */
} OrangeStatus;

/* --- opaque handles -------------------------------------------------------*/
typedef struct OrangeVec3ArrayImpl* OrangeVec3Array;  /* points / normals / colors */
typedef struct OrangeMeshImpl*      OrangeMesh;       /* triangle soup */
typedef struct OrangeKdTreeImpl*    OrangeKdTree;     /* point queries */

/* --- library info ---------------------------------------------------------*/
/* Engine version packed as (major << 16) | (minor << 8) | patch. */
ORANGE_C_API uint32_t    ORANGE_C_CALL orangeVersion(void);
/* Value of ORANGE_C_API_VERSION this binary was built with. */
ORANGE_C_API uint32_t    ORANGE_C_CALL orangeCApiVersion(void);
/* Message for the last failure on the calling thread ("" if none). Valid until
 * the next failing call on that thread. */
ORANGE_C_API const char* ORANGE_C_CALL orangeLastError(void);

/* --- float3 arrays --------------------------------------------------------*/
/* Copies `count` xyz triples out of `xyz` (3*count floats). */
ORANGE_C_API OrangeVec3Array ORANGE_C_CALL orangeVec3ArrayCreate(const float* xyz, int32_t count);
ORANGE_C_API int32_t         ORANGE_C_CALL orangeVec3ArrayCount(OrangeVec3Array arr);
/* Borrowed pointer to 3*count floats; valid until the handle is destroyed. */
ORANGE_C_API const float*    ORANGE_C_CALL orangeVec3ArrayData(OrangeVec3Array arr);
/* Copies into caller memory instead. Returns floats written, or a negative
 * OrangeStatus (ORANGE_ERROR_TOO_SMALL if dstFloats < 3*count). */
ORANGE_C_API int32_t         ORANGE_C_CALL orangeVec3ArrayCopy(OrangeVec3Array arr,
                                                               float* dst, int32_t dstFloats);
ORANGE_C_API void            ORANGE_C_CALL orangeVec3ArrayDestroy(OrangeVec3Array arr);

/* --- point-cloud IO -------------------------------------------------------*/
/* Load the point positions of a .ply / .xyz / .obj / .off file. */
ORANGE_C_API OrangeVec3Array ORANGE_C_CALL orangePointsLoadFile(const char* path);

/* --- point-cloud operators ------------------------------------------------*/
/* PCA normal estimation over the k nearest neighbours (k <= 0 => 16). */
ORANGE_C_API OrangeVec3Array ORANGE_C_CALL orangeEstimateNormals(OrangeVec3Array points, int32_t k);
/* Laplacian / bilateral smoothing. edgePreserving != 0 keeps creases. */
ORANGE_C_API OrangeVec3Array ORANGE_C_CALL orangeSmoothPoints(OrangeVec3Array points,
                                                             int32_t iterations, float lambda,
                                                             int32_t edgePreserving, int32_t k);
/* Point-to-point ICP of `src` onto `dst`. `outMatrix16` receives the 4x4 rigid
 * transform in column-major (OpenGL) order; outRmse / outIters may be NULL. */
ORANGE_C_API OrangeStatus ORANGE_C_CALL orangeIcpAlign(OrangeVec3Array src, OrangeVec3Array dst,
                                                       int32_t maxIterations, float* outMatrix16,
                                                       float* outRmse, int32_t* outIters);

/* --- surface reconstruction ----------------------------------------------*/
/* points -> TSDF -> dual-contoured surface. `normals`/`colors` may be NULL, and
 * voxelSize <= 0 selects the engine default. */
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangePointsToMesh(OrangeVec3Array points,
                                                        OrangeVec3Array normals,
                                                        OrangeVec3Array colors,
                                                        float voxelSize);
/* Load a point cloud file and reconstruct it in one call. */
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeReconstructFromFile(const char* path, float voxelSize);

/* --- parametric primitives (color3 = float[3], may be NULL for white) -----*/
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeBuildPlane(float size, const float* color3);
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeBuildBox(const float* size3, const float* color3);
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeBuildSphere(float radius, int32_t segments,
                                                        const float* color3);
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeBuildCylinder(float radius, float height,
                                                          int32_t segments, const float* color3);
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeBuildCone(float radius, float height,
                                                      int32_t segments, const float* color3);
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeBuildTorus(float majorRadius, float minorRadius,
                                                       int32_t segMajor, int32_t segMinor,
                                                       const float* color3);
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeBuildDisk(float radius, int32_t segments,
                                                      const float* color3);
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeBuildCapsule(float radius, float cylinderHeight,
                                                         int32_t segments, const float* color3);
ORANGE_C_API OrangeMesh ORANGE_C_CALL orangeBuildArrow(float length, float radius,
                                                       int32_t segments, const float* color3);

/* --- mesh access ----------------------------------------------------------*/
ORANGE_C_API int32_t      ORANGE_C_CALL orangeMeshTriangleCount(OrangeMesh mesh);
/* Borrowed pointers, 9 floats per triangle (v0.xyz, v1.xyz, v2.xyz). */
ORANGE_C_API const float* ORANGE_C_CALL orangeMeshPositions(OrangeMesh mesh);
ORANGE_C_API const float* ORANGE_C_CALL orangeMeshNormals(OrangeMesh mesh);
ORANGE_C_API const float* ORANGE_C_CALL orangeMeshColors(OrangeMesh mesh);
/* World-space bounds; outMin3 / outMax3 receive xyz. */
ORANGE_C_API OrangeStatus ORANGE_C_CALL orangeMeshBounds(OrangeMesh mesh,
                                                         float* outMin3, float* outMax3);
ORANGE_C_API void         ORANGE_C_CALL orangeMeshDestroy(OrangeMesh mesh);

/* --- KD-tree point queries -----------------------------------------------*/
/* Copies the points, so `points` may be destroyed afterwards. */
ORANGE_C_API OrangeKdTree ORANGE_C_CALL orangeKdTreeCreate(OrangeVec3Array points);
/* Index of the nearest point, or a negative OrangeStatus. outDistance may be NULL. */
ORANGE_C_API int32_t      ORANGE_C_CALL orangeKdTreeNearest(OrangeKdTree tree, const float* query3,
                                                            float* outDistance);
/* Writes up to `capacity` indices (nearest first) and returns how many were
 * written, or a negative OrangeStatus. */
ORANGE_C_API int32_t      ORANGE_C_CALL orangeKdTreeKNearest(OrangeKdTree tree, const float* query3,
                                                             int32_t k, int32_t* outIndices,
                                                             int32_t capacity);
/* Same, for every point within `radius`. Returns the number written; the total
 * found may exceed `capacity` -- pass capacity 0 with outIndices NULL to count. */
ORANGE_C_API int32_t      ORANGE_C_CALL orangeKdTreeRadius(OrangeKdTree tree, const float* query3,
                                                           float radius, int32_t* outIndices,
                                                           int32_t capacity);
ORANGE_C_API void         ORANGE_C_CALL orangeKdTreeDestroy(OrangeKdTree tree);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ORANGE_C_ORANGE_H */
