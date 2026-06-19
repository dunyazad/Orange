#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "orange/core/debug_draw.h"

// Selectable point-cloud processing "modes" — the Orange equivalent of the
// Hydrogen apps. Each mode takes an input point cloud and emits a visualization
// (colored points, boxes, a reconstructed mesh) into a DebugDraw. The host
// cycles the active mode with a key; processingModeSystem runs the active mode
// when it changes and re-emits the cached result each frame.
//
// The algorithms (Euclidean clustering, voxel morphology, SDF denoise, TSDF
// surface reconstruction) are ported from Elements/Helium Hydrogen apps,
// decoupled from CUDA/Helium and built on Orange's geometry toolkit.

namespace orange::modes {

// Input data a mode runs on (populated by the app into the registry ctx).
struct ModeInput {
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;  // optional (used by reconstruction)
};

// Active-mode selector, stored in the registry ctx. The host bumps `generation`
// whenever it changes `index` so the system knows to recompute.
struct ModeState {
    int      index      = 0;
    uint64_t generation = 1;
};

// The registered modes (stable order; index into this list).
int         modeCount();
const char* modeName(int index);

// Run mode `index` on `in`, emitting its visualization into `out`.
void runMode(int index, const ModeInput& in, debug::DebugDraw& out);

} // namespace orange::modes
