#pragma once

#include <cstdint>
#include <functional>
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
// whenever it changes `index` so the system knows to recompute. `index < 0` means
// NO mode is active -- the default, so merely selecting an entity does not run a
// processing operator (it must be turned on from the Geometry menu / M key).
struct ModeState {
    int      index      = -1;
    uint64_t generation = 1;
};

// Operator family, used to group the modes in the menu. Filters mark points
// keep/drop; Analyze maps a per-point scalar to a heatmap; Generate emits new
// geometry (a reconstructed mesh / resampled cloud).
enum class ModeCategory { Filter = 0, Analyze, Generate, Transform };

// The registered modes (stable order; index into this list).
int          modeCount();
const char*  modeName(int index);
ModeCategory modeCategory(int index);
const char*  modeCategoryName(ModeCategory c);

// Progress callback: a long-running mode reports its completion fraction in
// [0,1] as it works (throttled, not per-point). May be empty (ignored). Called
// from the worker thread, so the sink must be thread-safe.
using ProgressFn = std::function<void(float)>;

// Run mode `index` on `in`, emitting its visualization into `out`, reporting
// progress through `progress`. Modes that need normals estimate them from the
// points when `in.normals` is empty.
void runMode(int index, const ModeInput& in, debug::DebugDraw& out,
             const ProgressFn& progress = {});

} // namespace orange::modes
