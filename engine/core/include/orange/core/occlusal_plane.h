#pragma once

// Stage 1 of the tooth-segmentation pipeline: estimate the occlusal plane of a
// single-arch intraoral scan mesh via a PCA bootstrap.
//
// A single PCA over the whole mesh is tilted by non-occlusal verts (palate,
// extended gingiva, base trim). So we bootstrap, as the design doc prescribes:
//   1. coarse whole-mesh PCA -> rough occlusal axis (smallest-eigenvalue evec),
//   2. extract cusp / incisal-edge candidates (grid NMS top-of-column + local
//      convexity along the axis),
//   3. re-run PCA on the candidates only -> refined plane,
//   4. repeat 2-3 to converge, then fix the normal's sign so it points toward
//      the occlusal (biting) side.
//
// Output is a plane: a position (point on the plane) and a unit normal.
// ASCII comments only (pipeline project rule).

#include <vector>

#include <Eigen/Core>

#include "orange/core/mesh_generation.h"  // Triangle

namespace orange::geometry {

// The estimated occlusal plane.
struct OcclusalPlane {
    Eigen::Vector3f position = Eigen::Vector3f::Zero();   // a point on the plane (cusp centroid)
    Eigen::Vector3f normal   = Eigen::Vector3f::UnitY();  // unit normal, toward the occlusal side
    bool valid = false;                                   // false if the mesh was too small/degenerate
};

// Tunables for the bootstrap. Defaults work for typical IOS single-arch scans.
struct OcclusalPlaneParams {
    int pcaNeighbors = 16;  // kNN size for the local-convexity test
    int refineIters  = 2;   // number of re-PCA refinement passes
    int gridCells    = 24;  // NMS grid resolution along the arch's widest axis
    // findCusps: a peak is kept only if it rises >= cuspProminence * bbox-diagonal
    // above its grooves, and the height field is pre-smoothed cuspSmoothIters times.
    // Lower prominence / fewer smooth iters => more (smaller) cusps detected.
    int   cuspSmoothIters = 6;     // light smoothing before hill climbing (merges noise sub-peaks)
    float cuspHeightGate  = 0.30f; // drop tips below this fraction of the height band (floor/gingiva)
};

// Step 1 only: a single coarse PCA over ALL vertices (no cusp refinement).
// position = centroid, normal = least-variance axis. Without cusp cues the sign
// is just a display convention (oriented to the +Y hemisphere). Use this to
// inspect the raw whole-mesh fit before the bootstrap refines it.
OcclusalPlane wholeMeshPCA(const std::vector<Eigen::Vector3f>& vertices);

// Step 2: detect cusp / incisal-edge tips by walking the MESH GRAPH (not a
// spatial grid). Whole-mesh PCA fixes the occlusal axis; a cusp tip is a local
// maximum of height (projection on that axis) on the vertex-adjacency graph
// built from `indices`. Topological persistence keeps only peaks that rise at
// least a few mm above their surrounding grooves (drops noise, gingival margin,
// scan floor). Returns the cusp vertex positions (same space as `vertices`).
// `indices` is the triangle list (3 per face) sharing `vertices`' indexing.
std::vector<Eigen::Vector3f> findCusps(const std::vector<Eigen::Vector3f>& vertices,
                                       const std::vector<uint32_t>& indices,
                                       const OcclusalPlaneParams& params = {});

// Estimate the occlusal plane from a raw vertex cloud (the mesh's positions).
OcclusalPlane estimateOcclusalPlane(const std::vector<Eigen::Vector3f>& vertices,
                                    const OcclusalPlaneParams& params = {});

// Convenience overload: dedup a triangle soup's corners, then estimate.
OcclusalPlane estimateOcclusalPlane(const std::vector<Triangle>& mesh,
                                    const OcclusalPlaneParams& params = {});

} // namespace orange::geometry
